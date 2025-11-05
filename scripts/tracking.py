#!/usr/bin/env python3

import argparse
from aloha.robot_utils import (
    enable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
    load_yaml_file,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    START_ARM_POSE,
)

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand
from pathlib import Path
import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from typing import Dict
import time
import math
from std_msgs.msg import Bool, Float64


def demo_gripper_wave(robots, dt, period=4.0):
    """
    Smoothly oscillate follower grippers open/close by directly publishing commands.
    """
    t = time.time() % period
    phase = math.sin(2 * math.pi * t / period) * 0.5 + 0.5  # normalize 0–1
    target_grip = FOLLOWER_GRIPPER_JOINT_CLOSE + \
        phase * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)

    for name, bot in robots.items():
        if 'follower' not in name:
            continue
        cmd = JointSingleCommand(name='gripper', cmd=target_grip)
        bot.gripper.core.pub_single.publish(cmd)

def demo_base_rotation(robots, dt, period=10.0, amplitude=0.5, continuous=True):
    """
    Smoothly rotates only the base ('waist') joint of follower arms.
    """
    t = time.time()

    for name, bot in robots.items():
        if 'follower' not in name:
            continue

        waist_joint_name = bot.arm.group_info.joint_names[0]  # usually 'waist'

        if continuous:
            # Continuous rotation
            waist_angle = (2 * math.pi * (t / period)) % (2 * math.pi)
        else:
            # Oscillation mode
            waist_angle = amplitude * math.sin(2 * math.pi * t / period)

        # Build a single-joint command
        waist_cmd = JointSingleCommand()
        waist_cmd.name = waist_joint_name
        waist_cmd.cmd = waist_angle

        # Publish directly to that joint's controller
        bot.arm.core.pub_single.publish(waist_cmd)





def demo_multilink_wave_with_patient_found(
    robots, dt, patient_found_state,
    period=8.0, amplitude=0.5, continuous=False, t=None,
    gripper_period=4.0
):
    """
    Natural multi-link sine motion (waist, shoulder, elbow) that pauses/resumes
    smoothly with C¹-continuous blending. When patient is found, arm motion pauses
    (gripper can keep waving). On resume, no jerks on ANY joint.
    """
    now = t if t is not None else time.time()
    omega = 2.0 * math.pi / max(1e-6, period)

    # ---- Persistent state ----
    s = patient_found_state
    s.setdefault('start_offsets', {})       # per-arm base pose
    s.setdefault('last_q_cmd', {})          # per-arm last command for EMA smoothing
    s.setdefault('phase', 0.0)              # global motion phase (advances only when not paused)
    s.setdefault('last_time', now)
    s.setdefault('is_paused', False)
    s.setdefault('resume_blend_t0', None)   # resume start time
    s.setdefault('resume_blend_T', 1.5)     # blend duration (sec), can tune
    s.setdefault('resume_start_pose', {})   # per-arm pose captured at resume
    s.setdefault('gripper_offset', 0.0)
    s.setdefault('last_gripper_time', now)

    # ---- helpers ----
    def raised_cosine(u):
        # u in [0,1] -> 0..1 with zero slope at both ends
        u = max(0.0, min(1.0, u))
        return 0.5 - 0.5 * math.cos(math.pi * u)

    def get_blend():
        if s['resume_blend_t0'] is None:
            return 1.0
        u = (now - s['resume_blend_t0']) / max(1e-6, s['resume_blend_T'])
        b = raised_cosine(u)
        if u >= 1.0:
            s['resume_blend_t0'] = None
            return 1.0
        return b

    # -------------- PAUSE MODE (patient found) --------------
    if s.get('patient_found', False):
        if not s['is_paused']:
            # Entering pause: mark paused; phase stops advancing automatically (we don't add dt)
            s['is_paused'] = True
            
            # Gripper movement only happens the first time patient is found
            for name, follower_bot in robots.items():
                if 'follower' not in name:
                    continue
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_CLOSE],
                    moving_time=0.3,
                    dt=dt,
                )
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_OPEN],
                    moving_time=0.3,
                    dt=dt,
                )
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_CLOSE],
                    moving_time=0.3,
                    dt=dt,
                )
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_OPEN],
                    moving_time=0.3,
                    dt=dt,
                )
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_CLOSE],
                    moving_time=0.3,
                    dt=dt,
                )
                move_grippers(
                    [follower_bot],
                    [FOLLOWER_GRIPPER_JOINT_OPEN],
                    moving_time=0.3,
                    dt=dt,
                )
        

        # --- Gripper: keep waving while paused (time-based so it looks continuous) ---
        dtg = now - s['last_gripper_time']
        s['last_gripper_time'] = now
        s['gripper_offset'] += max(0.0, dtg)
        g_phase = 2 * math.pi * s['gripper_offset'] / max(1e-6, gripper_period)
        phase01 = 0.5 * (math.sin(g_phase) + 1.0)
        target_grip = FOLLOWER_GRIPPER_JOINT_CLOSE + \
            phase01 * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)

        # --- Optional waist tracking while paused (smooth, rate-limited) ---
        x_dist = s.get('x_distance', 0.0)
        DEADZONE = 50.0
        SPEED_RAD_S = 0.09   # rad/s
        MAX_ABS_ANGLE = 0.7  # rad (~28.6°)
        dt_waist = max(1e-3, now - s.get('last_waist_update_time', now))
        s['last_waist_update_time'] = now

        for name, bot in robots.items():
            if 'follower' not in name:
                continue

            # init per-arm stuff if missing
            joint_names = bot.arm.group_info.joint_names
            n = min(6, len(joint_names))
            if name not in s['start_offsets']:
                s['start_offsets'][name] = list(bot.arm.core.joint_states.position[:n])

            # waist control while paused
            current_waist = float(bot.arm.core.joint_states.position[0])
            if not s.get('pf_latched', False):
                s['pf_latched'] = True
                s['waist_center'] = current_waist
                s['waist_angle_offset'] = 0.0

            if x_dist > DEADZONE:
                direction = -1.0
            elif x_dist < -DEADZONE:
                direction = +1.0
            else:
                direction = 0.0

            offset = s.get('waist_angle_offset', 0.0)
            if not ((offset >= MAX_ABS_ANGLE and direction > 0) or
                    (offset <= -MAX_ABS_ANGLE and direction < 0)):
                offset += direction * SPEED_RAD_S * dt_waist
            s['waist_angle_offset'] = offset
            waist_target = s['waist_center'] + offset

            # Publish waist + gripper only while paused
            waist_cmd = JointSingleCommand()
            waist_cmd.name = bot.arm.group_info.joint_names[0]
            waist_cmd.cmd = waist_target
            bot.arm.core.pub_single.publish(waist_cmd)

            bot.gripper.core.pub_single.publish(
                JointSingleCommand(name='gripper', cmd=target_grip)
            )

        # hold arms (no arm group wave while paused)
        s['last_time'] = now
        return

    # -------------- RESUME PATH (patient no longer found) --------------
    # If we were paused before, start a smooth blend from current pose to the wave
    if s['is_paused']:
        s['is_paused'] = False
        s['pf_latched'] = False
        s['resume_blend_t0'] = now  # start blend window
        # Capture resume start pose for each follower (so we blend from *here*)
        for name, bot in robots.items():
            if 'follower' not in name:
                continue
            joint_names = bot.arm.group_info.joint_names
            n = min(6, len(joint_names))
            s['resume_start_pose'][name] = list(bot.arm.core.joint_states.position[:n])

    # Advance phase ONLY by active runtime (dt since last call)
    dt_run = max(0.0, now - s['last_time'])
    s['last_time'] = now
    s['phase'] += omega * dt_run

    # -------------- GENERATE CONTINUOUS WAVE --------------
    blend = get_blend()  # 0→1 with zero slope at ends
    for name, bot in robots.items():
        if 'follower' not in name:
            continue

        joint_names = bot.arm.group_info.joint_names
        n = min(6, len(joint_names))

        # Seed start_offsets once
        if name not in s['start_offsets']:
            s['start_offsets'][name] = list(bot.arm.core.joint_states.position[:n])
        base = s['start_offsets'][name][:n]

        # Resume start pose for blend-from; default to current if missing
        resume_from = s['resume_start_pose'].get(name, list(bot.arm.core.joint_states.position[:n]))

        # Wave deltas (feel free to tweak coupling and phase lags)
        d0 = amplitude * math.sin(s['phase'])            # waist
        d1 = 0.1 * amplitude * math.sin(s['phase'])      # shoulder
        d2 = 0.1 * amplitude * math.sin(s['phase'] + 0.8 * math.pi)  # elbow

        wave = list(base)
        if n > 0: wave[0] = base[0] + d0
        if n > 1: wave[1] = base[1] + d1
        if n > 2: wave[2] = base[2] + d2

        # C¹-smooth blend from resume_from → wave
        # q = (1-blend)*resume_from + blend*wave
        q_cmd = [(1.0 - blend) * rf + blend * w for rf, w in zip(resume_from, wave)]

        # Optional gentle EMA to kill sensor/pub jitter (small gain so it doesn't lag)
        last_q = s['last_q_cmd'].get(name, q_cmd)
        alpha = 0.2  # 0.0=no filter, 1.0=follow q_cmd immediately; keep modest
        smoothed_q = [(1 - alpha) * l + alpha * q for l, q in zip(last_q, q_cmd)]
        s['last_q_cmd'][name] = smoothed_q

        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = smoothed_q
        bot.arm.core.pub_group.publish(msg)


def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS],
                     dt: float,
                     ) -> None:
    """
    Move all leader-follower pairs of robots to a starting pose for demonstration.

    :param robots: Dictionary containing robot instances categorized as 'leader' or 'follower'
    :param dt: Time interval (in seconds) for each movement step
    """

    follower_bots = {name: bot for name,
                     bot in robots.items() if 'follower' in name}

    # Initialize each leader-follower pair
    for follower_name, follower_bot in follower_bots.items():
        # Reboot gripper motors and set operating modes
        follower_bot.core.robot_reboot_motors('single', 'gripper', True)
        follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        follower_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'current_based_position')

        # Enable torque for follower
        torque_on(follower_bot)

        # Move arms to starting position
        # Default: [0.0, -0.96, 1.16, 0.0, -0.3, 0.0] or START_ARM_POSE[:6]
        start_arm_qpos = [0.0, -0.5, 0.5, 0.0, -0.3, 0.0]
        move_arms(
            bot_list=[follower_bot],
            dt=dt,
            target_pose_list=[start_arm_qpos],
            moving_time=4.0,
        )

        # Move grippers to starting position
        move_grippers(
            [follower_bot],
            [FOLLOWER_GRIPPER_JOINT_OPEN],
            moving_time=0.5,
            dt=dt,
        )


def main(args: dict) -> None:
    """
    Main teleoperation setup function.

    :param args: Dictionary containing parsed arguments including gravity compensation
                 and robot configuration.
    """
    gravity_compensation = args.get('gravity_compensation', False)
    node = create_interbotix_global_node('aloha')
    
    
    patient_found_state = {
    'patient_found': False,
    'paused_time_offset': 0.0,
    'last_pause_time': None,
    'last_waist_angle': 0.0,
    'x_distance': 0.0,               # <-- NEW: stores latest distance reading
    'waist_angle_offset': 0.0,       # <-- NEW: keeps smoothed waist offset
    }
    
    # Subscribe to patient_found topic
    def patient_found_callback(msg):
        
        patient_found_state['patient_found'] = msg.data
    
    patient_found_sub = node.create_subscription(
        Bool,
        'patient_found',
        patient_found_callback,
        10
    )

    def patient_x_distance_callback(msg):
        patient_found_state['x_distance'] = msg.data

    distance_sub = node.create_subscription(
        Float64,
        'patient_x_distance',
        patient_x_distance_callback,
        10
    )

    # Load robot configuration
    robot_base = args.get('robot', '')

    # Base path of the config directory using absolute path
    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)
    print(f"dt: {dt}")

    # Initialize dictionary for robot instances
    robots = {}

    # Create follower arms from configuration
    for follower in config.get('follower_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=follower['model'],
            robot_name=follower['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[follower['name']] = robot_instance

    # Startup and initialize robot sequence
    robot_startup(node)
    opening_ceremony(robots, dt)
    node._logger.info('Opening ceremony completed')

    # Motion state to handle smooth start
    motion_state = {
        'start_time': None,
        'initialized': False,
    }

    node._logger.info('Starting base rotation demo...')

    # Define gripper command objects for each follower
    gripper_commands = {
        follower_name: JointSingleCommand(name='gripper') for follower_name in robots if 'follower' in follower_name
    }



    # Main loop
    counter = 0
    while rclpy.ok():
        # Process callbacks to update patient_found_state
        rclpy.spin_once(node, timeout_sec=0.1)

        
        if not motion_state['initialized']:
            motion_state['start_time'] = time.time()
            motion_state['initialized'] = True


        elapsed_t = time.time() - motion_state['start_time']

       
        demo_multilink_wave_with_patient_found(
            robots, dt, patient_found_state,
            period=16.0, amplitude=0.8, continuous=False, t=elapsed_t
            )
        

        # Sleep for the DT duration
        DT_DURATION = Duration(seconds=0, nanoseconds=dt * S_TO_NS)
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gravity_compensation',
        action='store_true',
        help='If set, gravity compensation will be enabled for the leader robots when teleop starts.',
    )
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Specify the robot configuration to use: aloha_solo, aloha_stationary, or aloha_mobile.'
    )
    main(vars(parser.parse_args()))
