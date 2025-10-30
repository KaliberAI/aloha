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
    # TODO: create subscriptions here

    # Load robot configuration
    robot_base = args.get('robot', '')

    # Base path of the config directory using absolute path
    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)

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

    # Define gripper command objects for each follower
    gripper_commands = {
        follower_name: JointSingleCommand(name='gripper') for follower_name in robots if 'follower' in follower_name
    }


    base_joint_positions

    # Main loop
    while rclpy.ok():

        # TODO: do the demo

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
