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

def demo_viper_breathing(robots, dt, period=8.0,
                         base_amplitude=0.35,
                         phase_offset=0.6,
                         decay_factor=0.55):
    """
    Produces a breathing / viper-like coordinated wave across all arm joints.
    Only affects the 6 arm joints (waist → wrist_rotate), leaves gripper untouched.
    """
    t = time.time()
    omega = 2 * math.pi / period

    for name, bot in robots.items():
        if 'follower' not in name:
            continue

        # Expected joint order: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper, ...]
        joint_names = bot.arm.group_info.joint_names
        n = min(6, len(joint_names))  # first six are arm joints

        amplitudes = [base_amplitude * (decay_factor ** i) for i in range(n)]
        q_cmd = []

        for i in range(n):
            # alternate phase direction for a natural breathing feel
            phi = i * phase_offset
            q = amplitudes[i] * math.sin(omega * t + phi)
            q_cmd.append(q)

        # pad with zeros for extra joints (gripper/fingers)
        while len(q_cmd) < len(joint_names):
            q_cmd.append(0.0)

        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = q_cmd
        bot.arm.core.pub_group.publish(msg)

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

from interbotix_xs_msgs.msg import JointSingleCommand

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



    # Main loop
    while rclpy.ok():

        # TODO: do the demo
        # demo_base_rotation(robots, dt, period=8.0, amplitude=0.6, continuous=False)

        demo_base_rotation(robots, dt, period=8.0, amplitude=0.6, continuous=False)
        
        # demo_gripper_wave(robots, dt, period=5.0)

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
