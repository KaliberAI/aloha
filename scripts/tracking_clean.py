#!/usr/bin/env python3
"""
Modular robot tracking script with state-based motion control.

This script provides a clean separation between state management and
motion generation, making it easy to add new states and behaviors.
"""

import argparse
import sys
import select
import termios
import tty
import time
from pathlib import Path
from typing import Dict

import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from std_msgs.msg import Bool, Float64, Int32

from aloha.robot_utils import (
    torque_on,
    load_yaml_file,
    move_arms,
    move_grippers,
    FOLLOWER_GRIPPER_JOINT_OPEN,
)
from aloha.robot_state_machine import (
    RobotStateMachine,
    RobotMotionState,
    STATE_KEY_MAP,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand


class RobotTrackingController:
    """
    Main controller for robot tracking with state-based motion.
    
    Handles state transitions based on patient detection and
    delegates motion generation to individual states.
    """
    
    # State ID to RobotMotionState mapping for ROS control mode
    STATE_ID_MAP = {
        0: RobotMotionState.SLEEP,
        1: RobotMotionState.OPENING,
        2: RobotMotionState.WAVING,
        3: RobotMotionState.NOD,
        4: RobotMotionState.RESUMING,
        5: RobotMotionState.TRACK,
        6: RobotMotionState.KNEE,
    }
    
    def __init__(
        self,
        node,
        robots: Dict[str, InterbotixManipulatorXS],
        dt: float,
        debug_mode: bool = False,
        ros_control_mode: bool = False,
    ):
        """
        Initialize tracking controller.
        
        :param node: ROS node
        :param robots: Dictionary of robot instances
        :param dt: Time step
        :param debug_mode: Enable debug mode with key control
        :param ros_control_mode: Enable ROS control mode with /robot_state topic
        """
        self.node = node
        self.robots = robots
        self.dt = dt
        self.debug_mode = debug_mode
        self.ros_control_mode = ros_control_mode
        
        # State machine
        self.state_machine = RobotStateMachine()
        
        # Patient detection state
        self.patient_found = False
        
        # ROS control state
        self.ros_state_id = None  # Current state ID from ROS topic
        
        # Terminal settings for key input (only needed for debug mode)
        self.old_terminal_settings = None
        if self.debug_mode:
            self.setup_terminal()
        
        # Setup ROS subscribers
        self.setup_subscribers()
        
        # Print debug info
        if self.debug_mode:
            self.print_debug_help()
        elif self.ros_control_mode:
            self.node._logger.info('=' * 60)
            self.node._logger.info('[ROS_CONTROL] ROS control mode enabled!')
            self.node._logger.info('[ROS_CONTROL] Subscribing to /robot_state topic')
            self.node._logger.info('[ROS_CONTROL] State mapping:')
            self.node._logger.info('[ROS_CONTROL]   0 - SLEEP')
            self.node._logger.info('[ROS_CONTROL]   1 - OPENING')
            self.node._logger.info('[ROS_CONTROL]   2 - WAVING')
            self.node._logger.info('[ROS_CONTROL]   3 - NOD')
            self.node._logger.info('[ROS_CONTROL]   4 - RESUMING')
            self.node._logger.info('[ROS_CONTROL]   5 - TRACK')
            self.node._logger.info('[ROS_CONTROL]   6 - KNEE')
            self.node._logger.info('=' * 60)
    
    def setup_subscribers(self):
        """Setup ROS subscribers for patient detection and ROS control."""
        self.patient_found_sub = self.node.create_subscription(
            Bool, 'patient_found', self.patient_found_callback, 10
        )
        
        self.distance_sub = self.node.create_subscription(
            Float64, 'patient_x_distance', self.patient_x_distance_callback, 10
        )
        
        # ROS control subscriber
        if self.ros_control_mode:
            self.robot_state_sub = self.node.create_subscription(
                Int32, 'robot_state', self.robot_state_callback, 10
            )
    
    def patient_found_callback(self, msg):
        """Handle patient_found topic."""
        self.patient_found = msg.data
    
    def patient_x_distance_callback(self, msg):
        """Handle patient_x_distance topic."""
        self.state_machine.set_x_distance(msg.data)
    
    def robot_state_callback(self, msg):
        """Handle robot_state topic for ROS control mode."""
        state_id = msg.data
        if state_id in self.STATE_ID_MAP:
            self.ros_state_id = state_id
            self.node._logger.info(
                f'[ROS_CONTROL] Received state ID: {state_id} -> {self.STATE_ID_MAP[state_id].value}'
            )
        else:
            self.node._logger.warn(
                f'[ROS_CONTROL] Invalid state ID: {state_id}. Valid range: 0-6'
            )
    
    def setup_terminal(self):
        """Setup terminal for key input."""
        if not sys.stdin.isatty():
            return
        
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, AttributeError):
            self.node._logger.warn('Could not set terminal mode')
            self.old_terminal_settings = None
    
    def restore_terminal(self):
        """Restore terminal settings."""
        if self.old_terminal_settings is not None and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            except (termios.error, AttributeError):
                pass
    
    def get_key_press(self) -> str:
        """Get key press (non-blocking). Returns character or None."""
        if not sys.stdin.isatty():
            return None
        
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                return sys.stdin.read(1)
        except (IOError, OSError, ValueError):
            pass
        
        return None
    
    def print_debug_help(self):
        """Print debug mode help."""
        self.node._logger.info('=' * 60)
        self.node._logger.info('[DEBUG] Debug mode enabled!')
        self.node._logger.info('[DEBUG] Key Controls:')
        for key, state in STATE_KEY_MAP.items():
            if key.isupper():
                continue  # Skip uppercase duplicates
            self.node._logger.info(f'[DEBUG]   {key}/{key.upper()} - {state.value.upper()} state')
        self.node._logger.info('[DEBUG]   i/I - Print state info')
        self.node._logger.info('[DEBUG]   h/H - Print help')
        self.node._logger.info('=' * 60)
    
    def handle_debug_keys(self, current_time: float):
        """Handle debug key presses for manual state control."""
        key = self.get_key_press()
        if not key:
            return
        
        # Log key press
        self.node._logger.info(f'[DEBUG] Key pressed: {repr(key)}')
        
        # Check state transition keys
        if key in STATE_KEY_MAP:
            new_state = STATE_KEY_MAP[key]
            self.state_machine.transition_to(new_state, current_time)
            return
        
        # Special keys
        if key in ('i', 'I'):
            # Print state info
            state = self.state_machine.get_state()
            self.node._logger.info(f'[DEBUG] Current State: {state.value}')
            
            # Print state-specific info
            if state == RobotMotionState.WAVING:
                wave_state = self.state_machine.states[state]
                self.node._logger.info(f'[DEBUG] Wave Phase: {wave_state.phase:.3f}')
            elif state == RobotMotionState.NOD:
                nod_state = self.state_machine.states[state]
                self.node._logger.info(f'[DEBUG] Nod Phase: {nod_state.phase:.3f}')
            elif state == RobotMotionState.TRACK:
                track_state = self.state_machine.states[state]
                self.node._logger.info(f'[DEBUG] Patient X Distance: {self.state_machine.x_distance:.1f}')
        
        elif key in ('h', 'H'):
            # Print help
            self.print_debug_help()
        
        else:
            self.node._logger.info(f'[DEBUG] Unknown key: {repr(key)}. Press "h" for help.')
    
    def update_state_transitions(self, current_time: float):
        """
        Handle automatic state transitions based on patient detection.
        
        :param current_time: Current timestamp
        """
        current_state = self.state_machine.get_state()
        
        # WAVING -> TRACK (patient found)
        if current_state == RobotMotionState.WAVING:
            if self.patient_found:
                self.state_machine.transition_to(RobotMotionState.TRACK, current_time)
        
        # TRACK -> RESUMING (patient lost)
        elif current_state == RobotMotionState.TRACK:
            if not self.patient_found:
                self.state_machine.transition_to(RobotMotionState.RESUMING, current_time)
        
        # RESUMING -> WAVING (blend complete)
        elif current_state == RobotMotionState.RESUMING:
            resuming_state = self.state_machine.states[RobotMotionState.RESUMING]
            if resuming_state.get_blend_factor(current_time) >= 0.99:
                self.state_machine.transition_to(RobotMotionState.WAVING, current_time)
        
        # OPENING -> WAVING (movement complete)
        elif current_state == RobotMotionState.OPENING:
            opening_state = self.state_machine.states[RobotMotionState.OPENING]
            if opening_state.start_time is not None:
                elapsed = current_time - opening_state.start_time
                if elapsed >= opening_state.moving_time:
                    self.state_machine.transition_to(RobotMotionState.WAVING, current_time)
    
    def execute_motion(self, current_time: float):
        """
        Execute motion for all robots based on current state.
        
        :param current_time: Current timestamp
        """
        # Update state logic
        self.state_machine.update(current_time, self.dt)
        
        # Generate and publish commands for each follower
        for name, bot in self.robots.items():
            if 'follower' not in name:
                continue
            
            # Get current joint positions
            joint_names = bot.arm.group_info.joint_names
            n = min(6, len(joint_names))
            current_joints = list(bot.arm.core.joint_states.position[:n])
            
            # Generate commands from state machine
            joint_cmd, gripper_cmd = self.state_machine.generate_commands(
                name, current_joints
            )
            
            # Publish joint commands
            msg = JointGroupCommand()
            msg.name = 'arm'
            msg.cmd = joint_cmd[:n]
            bot.arm.core.pub_group.publish(msg)
            
            # Publish gripper command if provided
            if gripper_cmd is not None:
                bot.gripper.core.pub_single.publish(
                    JointSingleCommand(name='gripper', cmd=gripper_cmd)
                )
    
    def run(self):
        """Main control loop."""
        # Start in OPENING state
        current_time = time.time()
        self.state_machine.transition_to(RobotMotionState.OPENING, current_time)
        
        try:
            while rclpy.ok():
                current_time = time.time()
                
                # Process ROS callbacks first
                rclpy.spin_once(self.node, timeout_sec=0.0)
                
                # Handle state transitions based on mode
                if self.ros_control_mode:
                    # ROS control mode: transition based on /robot_state topic
                    if self.ros_state_id is not None:
                        target_state = self.STATE_ID_MAP[self.ros_state_id]
                        current_state = self.state_machine.get_state()
                        if target_state != current_state:
                            self.state_machine.transition_to(target_state, current_time)
                elif self.debug_mode:
                    # Debug mode: handle key presses
                    self.handle_debug_keys(current_time)
                else:
                    # Normal mode: automatic state transitions
                    self.update_state_transitions(current_time)
                
                # Execute motion
                self.execute_motion(current_time)
                
                # Sleep for dt
                DT_DURATION = Duration(seconds=0, nanoseconds=self.dt * S_TO_NS)
                get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
        
        finally:
            if self.debug_mode:
                self.restore_terminal()


def opening_ceremony(
    robots: Dict[str, InterbotixManipulatorXS],
    dt: float,
) -> None:
    """
    Move all follower robots to a starting pose for demonstration.
    Based on opening_ceremony from tracking.py.

    :param robots: Dictionary containing robot instances
    :param dt: Time interval (in seconds) for each movement step
    """
    follower_bots = {name: bot for name, bot in robots.items() if 'follower' in name}

    # Initialize each follower
    for follower_name, follower_bot in follower_bots.items():
        # Reboot gripper motors and set operating modes
        follower_bot.core.robot_reboot_motors('single', 'gripper', True)
        follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        follower_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'current_based_position'
        )

        # Enable torque for follower
        torque_on(follower_bot)

        # Move arms to starting position
        start_arm_qpos = [0.0, -1.05, 0.42, 0, 1.05, 0.0]
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
    Main function for modular tracking script.
    
    :param args: Parsed command-line arguments
    """
    # Create ROS node
    node = create_interbotix_global_node('aloha_tracking_v2')
    
    # Load configuration
    robot_base = args.get('robot', '')
    base_path = Path(__file__).resolve().parent.parent / "config"
    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)
    
    node._logger.info(f"Control frequency: {1/dt:.1f} Hz (dt={dt:.4f}s)")
    
    # Create robot instances
    robots = {}
    for follower in config.get('follower_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=follower['model'],
            robot_name=follower['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[follower['name']] = robot_instance
    
    # Initialize robots using opening_ceremony
    robot_startup(node)
    opening_ceremony(robots, dt)
    node._logger.info(f'Initialized {len(robots)} robot(s)')
    node._logger.info('Opening ceremony completed')
    
    # Create and run controller
    controller = RobotTrackingController(
        node=node,
        robots=robots,
        dt=dt,
        debug_mode=args.get('debug', False),
        ros_control_mode=args.get('ros_control', False),
    )
    
    controller.run()
    
    # Shutdown
    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modular robot tracking with state-based motion control'
    )
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Robot configuration: aloha_solo, aloha_stationary, or aloha_mobile'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode: use key presses to control state transitions'
    )
    parser.add_argument(
        '--ros_control',
        action='store_true',
        help='Enable ROS control mode: subscribe to /robot_state topic for state transitions'
    )
    
    main(vars(parser.parse_args()))