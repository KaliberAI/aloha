#!/usr/bin/env python3
"""
Tray finder script.

Workflow
--------
1. Script launches. Motors torque on. Controller transitions to ``SLEEP`` and
   the arm eases into the sleep pose. It holds there indefinitely.
2. A remote voice-LLM (running on a separate machine) calls
   ``POST /find_tray {"tray_id": <int>}`` as a tool. On the FIRST call:

   - ``target_id`` is recorded.
   - The state machine transitions ``SLEEP -> OPENING`` (which eases the arm
     to the gripper-down ``TRAY_FIND_DEFAULT_POSE`` over ~4 s).
   - When OPENING completes, the controller auto-transitions to
     ``TRAY_FIND``. Sweeping starts cleanly from default_pose because
     ``TrayFindState.paused`` was set to False up-front.
3. The robot sweeps the waist (with wrist_rotate counter-rotating to keep the
   gripper orientation fixed in world frame). Whenever the AprilTag node
   publishes a detection containing ``target_id``, ``TrayFindState.paused``
   is flipped True and the arm freezes in place. ``phase`` is preserved.
4. A subsequent ``POST /find_tray`` with a NEW ``tray_id`` simply updates the
   target and unpauses. The sweep resumes from the same phase. There is no
   SLEEP/OPENING re-entry on subsequent calls.
5. ``POST /sleep`` returns the robot to the sleep pose at any time, clears
   ``target_id``, and re-arms the SLEEP -> OPENING transition for the next
   ``find_tray`` call.

LLM tool reference (lives on the OTHER machine)
-----------------------------------------------
GPT-style tool schema::

    {
      "name": "find_tray",
      "description": "Sweep the robot to look for a tray with the given AprilTag id. "
                     "Robot freezes when detected. First call also wakes the arm.",
      "parameters": {
        "type": "object",
        "properties": {
          "tray_id": {"type": "integer"}
        },
        "required": ["tray_id"]
      }
    }

    {
      "name": "send_robot_to_sleep",
      "description": "Send the robot back to its sleep/rest pose. Clears any active "
                     "tray search target.",
      "parameters": {"type": "object", "properties": {}}
    }

LLM-side function bodies (Python)::

    import requests
    requests.post(f"http://{ROBOT_HOST}:8080/find_tray",
                  json={"tray_id": tray_id}, timeout=2)
    requests.post(f"http://{ROBOT_HOST}:8080/sleep", timeout=2)

HTTP API
--------
``POST /find_tray``  body ``{"tray_id": <int>}``
    Set the target tag id and start (or resume) sweeping.

``POST /sleep``  no body required
    Stop sweeping, clear the target id, and ease the arm into the sleep pose.

``GET  /status``
    Returns ``{"target_id": <int|null>, "state": <str>, "paused": <bool>,
    "phase": <float>}`` for debugging from a curl on the LLM machine.
"""

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from aloha.robot_utils import (
    torque_on,
    load_yaml_file,
    move_grippers,
    FOLLOWER_GRIPPER_JOINT_OPEN,
)
from aloha.robot_state_machine import (
    RobotStateMachine,
    RobotMotionState,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand

try:
    from apriltag_ros.msg import AprilTagDetectionArray
except ImportError as e:  # pragma: no cover - import-time error path
    raise ImportError(
        "Could not import apriltag_ros.msg.AprilTagDetectionArray. Make sure "
        "the apriltag_ros package is installed and sourced before running "
        "tray_finder.py."
    ) from e


APRILTAG_TOPIC = '/apriltag_ros_continuous_detector_node/tag_detections'


def setup_robots(
    robots: Dict[str, InterbotixManipulatorXS],
    dt: float,
) -> None:
    """
    Power up the follower robots without moving the arms.

    This is a stripped-down ``opening_ceremony``: it reboots the gripper
    motors, sets operating modes, torques the arm and gripper on, and opens
    the grippers - but it does NOT issue an arm move. The first arm motion
    is owned by ``SleepState`` once the controller enters SLEEP, so the arm
    eases from wherever it currently is into the sleep pose under blended
    state-machine control.
    """
    follower_bots = {name: bot for name, bot in robots.items() if 'follower' in name}

    for _, follower_bot in follower_bots.items():
        follower_bot.core.robot_reboot_motors('single', 'gripper', True)
        follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        follower_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'current_based_position'
        )

        torque_on(follower_bot)

        move_grippers(
            [follower_bot],
            [FOLLOWER_GRIPPER_JOINT_OPEN],
            moving_time=0.5,
            dt=dt,
        )


class TrayFinderController:
    """
    Top-level controller for tray_finder.py.

    Drives the state machine, owns the AprilTag subscriber, and runs an
    embedded HTTP server thread that the remote voice-LLM uses as its tool
    transport.
    """

    def __init__(
        self,
        node,
        robots: Dict[str, InterbotixManipulatorXS],
        dt: float,
        http_host: str,
        http_port: int,
        debug_mode: bool = False,
    ):
        self.node = node
        self.robots = robots
        self.dt = dt
        self.debug_mode = debug_mode

        self.state_machine = RobotStateMachine()

        # Target id and a coarse lock that protects everything the HTTP
        # thread might touch (target_id, state transitions, the
        # TrayFindState pause flag).
        self.target_id: Optional[int] = None
        self._lock = threading.Lock()

        # Keep a direct handle to TrayFindState so we don't repeatedly
        # look it up through the state machine.
        self._tray_state = self.state_machine.states[RobotMotionState.TRAY_FIND]

        # Subscribers
        self.tag_sub = self.node.create_subscription(
            AprilTagDetectionArray,
            APRILTAG_TOPIC,
            self.tag_callback,
            10,
        )

        # HTTP server thread
        self._http_thread = _HTTPServerThread(
            controller=self,
            host=http_host,
            port=http_port,
            logger=self.node._logger,
        )
        self._http_thread.start()

        # Terminal settings for optional key input
        self.old_terminal_settings = None
        if self.debug_mode:
            self._setup_terminal()
            self._print_debug_help()

        self.node._logger.info('=' * 60)
        self.node._logger.info('[TRAY_FINDER] Controller initialized.')
        self.node._logger.info(
            f'[TRAY_FINDER] HTTP server listening on http://{http_host}:{http_port}'
        )
        self.node._logger.info(
            f'[TRAY_FINDER] AprilTag topic: {APRILTAG_TOPIC}'
        )
        self.node._logger.info('=' * 60)

    # ------------------------------------------------------------------
    # Public actions invoked by the HTTP server (and tests)
    # ------------------------------------------------------------------

    def set_target(self, tray_id: int) -> None:
        """Set/replace the target tray id and start (or resume) the sweep."""
        with self._lock:
            self.target_id = tray_id
            self._tray_state.set_paused(False)

            current = self.state_machine.get_state()
            if current == RobotMotionState.SLEEP:
                # First call after launch (or after /sleep). Kick off the
                # gentle climb to the gripper-down pose. OPENING -> TRAY_FIND
                # auto-transitions in the main loop once moving_time elapses.
                self.state_machine.transition_to(
                    RobotMotionState.OPENING, time.time()
                )
                self.node._logger.info(
                    f'[TRAY_FINDER] target={tray_id}; SLEEP -> OPENING'
                )
            else:
                self.node._logger.info(
                    f'[TRAY_FINDER] target={tray_id}; resume from {current.value}'
                )

    def go_to_sleep(self) -> None:
        """Stop sweeping, clear the target, and ease back to the sleep pose."""
        with self._lock:
            self.target_id = None
            self._tray_state.set_paused(True)
            self.state_machine.transition_to(
                RobotMotionState.SLEEP, time.time()
            )
            self.node._logger.info('[TRAY_FINDER] /sleep -> returning to SLEEP pose')

    def get_status(self) -> dict:
        """Snapshot of internal state for the GET /status endpoint."""
        with self._lock:
            return {
                'target_id': self.target_id,
                'state': self.state_machine.get_state().value,
                'paused': self._tray_state.is_paused(),
                'phase': self._tray_state.phase,
            }

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def tag_callback(self, msg: AprilTagDetectionArray) -> None:
        """Pause the sweep when the current target tag id appears."""
        with self._lock:
            target = self.target_id
            if target is None:
                return

            for det in msg.detections:
                # det.id is a list (apriltag_ros supports tag bundles).
                if target in det.id:
                    if not self._tray_state.is_paused():
                        self._tray_state.set_paused(True)
                        self.node._logger.info(
                            f'[TRAY_FINDER] Tray {target} detected - freezing sweep'
                        )
                    return

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _update_state_transitions(self, current_time: float) -> None:
        """Single-rule transition table: OPENING -> TRAY_FIND on completion."""
        current = self.state_machine.get_state()
        if current != RobotMotionState.OPENING:
            return

        opening = self.state_machine.states[RobotMotionState.OPENING]
        if opening.start_time is None:
            return
        if (current_time - opening.start_time) >= opening.moving_time:
            self.state_machine.transition_to(
                RobotMotionState.TRAY_FIND, current_time
            )

    def _execute_motion(self, current_time: float) -> None:
        """Tick the state machine and publish commands to all followers."""
        self.state_machine.update(current_time, self.dt)

        for name, bot in self.robots.items():
            if 'follower' not in name:
                continue

            joint_names = bot.arm.group_info.joint_names
            n = min(6, len(joint_names))
            current_joints = list(bot.arm.core.joint_states.position[:n])

            joint_cmd, gripper_cmd = self.state_machine.generate_commands(
                name, current_joints
            )

            msg = JointGroupCommand()
            msg.name = 'arm'
            msg.cmd = joint_cmd[:n]
            bot.arm.core.pub_group.publish(msg)

            if gripper_cmd is not None:
                bot.gripper.core.pub_single.publish(
                    JointSingleCommand(name='gripper', cmd=gripper_cmd)
                )

    def run(self) -> None:
        """Main control loop. Starts in SLEEP."""
        with self._lock:
            self.state_machine.transition_to(
                RobotMotionState.SLEEP, time.time()
            )
        self.node._logger.info('[TRAY_FINDER] Holding SLEEP pose. '
                               'Awaiting first POST /find_tray ...')

        try:
            while rclpy.ok():
                current_time = time.time()
                rclpy.spin_once(self.node, timeout_sec=0.0)

                if self.debug_mode:
                    self._handle_debug_keys()

                with self._lock:
                    self._update_state_transitions(current_time)
                    self._execute_motion(current_time)

                dt_duration = Duration(seconds=0, nanoseconds=self.dt * S_TO_NS)
                get_interbotix_global_node().get_clock().sleep_for(dt_duration)
        finally:
            if self.debug_mode:
                self._restore_terminal()
            self._http_thread.shutdown()

    # ------------------------------------------------------------------
    # Debug-mode terminal helpers (minimal: 'i' status, 'h' help)
    # ------------------------------------------------------------------

    def _setup_terminal(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, AttributeError):
            self.node._logger.warn('Could not set terminal mode')
            self.old_terminal_settings = None

    def _restore_terminal(self) -> None:
        if self.old_terminal_settings is not None and sys.stdin.isatty():
            try:
                termios.tcsetattr(
                    sys.stdin, termios.TCSADRAIN, self.old_terminal_settings
                )
            except (termios.error, AttributeError):
                pass

    def _get_key(self) -> Optional[str]:
        if not sys.stdin.isatty():
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                return sys.stdin.read(1)
        except (IOError, OSError, ValueError):
            pass
        return None

    def _print_debug_help(self) -> None:
        self.node._logger.info('[DEBUG] Tray finder debug keys:')
        self.node._logger.info('[DEBUG]   i/I - print status')
        self.node._logger.info('[DEBUG]   h/H - print this help')

    def _handle_debug_keys(self) -> None:
        key = self._get_key()
        if not key:
            return
        if key in ('i', 'I'):
            status = self.get_status()
            self.node._logger.info(f'[DEBUG] status={status}')
        elif key in ('h', 'H'):
            self._print_debug_help()


# ----------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------


class _HTTPServerThread(threading.Thread):
    """Background thread running the embedded HTTP API."""

    def __init__(self, controller: TrayFinderController, host: str, port: int, logger):
        super().__init__(daemon=True, name='TrayFinderHTTP')
        self._controller = controller
        self._host = host
        self._port = port
        self._logger = logger

        handler_cls = _make_request_handler(controller, logger)
        self._server = ThreadingHTTPServer((host, port), handler_cls)

    def run(self) -> None:
        self._logger.info(
            f'[TRAY_FINDER_HTTP] serve_forever on {self._host}:{self._port}'
        )
        try:
            self._server.serve_forever(poll_interval=0.5)
        finally:
            self._server.server_close()

    def shutdown(self) -> None:
        try:
            self._server.shutdown()
        except Exception:  # pragma: no cover - shutdown best-effort
            pass


def _make_request_handler(controller: TrayFinderController, logger):
    """Build a BaseHTTPRequestHandler subclass closed over the controller."""

    class _Handler(BaseHTTPRequestHandler):

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # Route stdlib HTTP server access logs through ROS logger.
            logger.info(f'[TRAY_FINDER_HTTP] {self.address_string()} - ' + (format % args))

        # ----- helpers -------------------------------------------------

        def _send_json(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json_body(self) -> Optional[dict]:
            length = int(self.headers.get('Content-Length') or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                obj = json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {'ok': False, 'error': 'invalid json body'})
                return None
            if not isinstance(obj, dict):
                self._send_json(400, {'ok': False, 'error': 'body must be a json object'})
                return None
            return obj

        # ----- routes --------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == '/find_tray':
                body = self._read_json_body()
                if body is None:
                    return
                tray_id = body.get('tray_id')
                if not isinstance(tray_id, int) or isinstance(tray_id, bool):
                    self._send_json(400, {
                        'ok': False,
                        'error': 'tray_id must be an integer',
                    })
                    return
                controller.set_target(tray_id)
                self._send_json(200, {'ok': True, 'target_id': tray_id})
                return

            if self.path == '/sleep':
                # Body is optional; we don't read it to keep this endpoint
                # callable with a parameter-less tool.
                controller.go_to_sleep()
                self._send_json(200, {'ok': True})
                return

            self._send_json(404, {'ok': False, 'error': 'unknown path'})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == '/status':
                self._send_json(200, controller.get_status())
                return
            self._send_json(404, {'ok': False, 'error': 'unknown path'})

    return _Handler


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def main(args: dict) -> None:
    node = create_interbotix_global_node('aloha_tray_finder')

    robot_base = args.get('robot', '')
    base_path = Path(__file__).resolve().parent.parent / 'config'
    config = load_yaml_file('robot', robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)

    node._logger.info(f'Control frequency: {1/dt:.1f} Hz (dt={dt:.4f}s)')

    robots = {}
    for follower in config.get('follower_arms', []):
        robots[follower['name']] = InterbotixManipulatorXS(
            robot_model=follower['model'],
            robot_name=follower['name'],
            node=node,
            iterative_update_fk=False,
        )

    robot_startup(node)
    setup_robots(robots, dt)
    node._logger.info(f'Powered up {len(robots)} robot(s); first move owned by SLEEP state.')

    controller = TrayFinderController(
        node=node,
        robots=robots,
        dt=dt,
        http_host=args.get('http_host', '0.0.0.0'),
        http_port=int(args.get('http_port', 8080)),
        debug_mode=args.get('debug', False),
    )

    try:
        controller.run()
    finally:
        robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HTTP-driven tray finder using TRAY_FIND state machine.'
    )
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Robot configuration: aloha_solo, aloha_stationary, or aloha_mobile',
    )
    parser.add_argument(
        '--http_host',
        default='0.0.0.0',
        help='Host/IP to bind the HTTP server to (default: 0.0.0.0).',
    )
    parser.add_argument(
        '--http_port',
        type=int,
        default=8080,
        help='Port for the HTTP server (default: 8080).',
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode: minimal key handler ("i" status, "h" help).',
    )

    main(vars(parser.parse_args()))
