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
from typing import Dict, List, Mapping
import time
import math
import sys
import select
import termios
import tty
from std_msgs.msg import Bool, Float64



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
        start_arm_qpos = [0.0, -1.05, 0.42, 0, 1.05, 0.0]

        move_arms(
            bot_list=[follower_bot],
            dt=dt,
            target_pose_list=[start_arm_qpos],
            moving_time=4.0,
        )


# Hard-coded global poses for generic motions
START_POSE: List[float] = [0.0, -1.05, 0.42, 0.0, 1.05, 0.0]
SLEEP_POSE: List[float] = [0.0, -1.85, 1.6057, 0.0, 0.8203, 0.0]

# Dictionary of tools/instruments and their associated poses
# Each instrument entry contains:
#   - "pick": robot joint pose to pick the instrument
#   - "place_camera": pose to bring the instrument under the camera (optional)
#   - "place_tray": pose to place the instrument on the tray
INSTRUMENT_POSES: Dict[str, Dict[str, List[float]]] = {
    # Current poses are for forceps
    "forceps": {
        "pick": [0.0, 0.069, 0.576, 0.0, 1.134, 0.0],
        # For now, use the same pose as tray placement; adjust as needed
        "place_camera": [1.309, -0.0174, 0.576, 0.0, 1.361, 0.0],
        "place_tray": [1.309, -0.0174, 0.576, 0.0, 1.361, 1.0],
    },
    "cylinder": {
        "pick": [0.0, -0.174, 0.872, 0.0, 0.837, 1.570],
        "place_camera": [1.309, -0.0174, 0.576, 0.0, 1.361, 0.0],
        "place_tray": [1.309, -0.0174, 0.576, 0.0, 1.361, 1.0],
    }
}


def perform_pick_and_place(
    robots: Mapping[str, InterbotixManipulatorXS],
    dt: float,
    instrument_name: str,
    instrument_poses: Mapping[str, Mapping[str, List[float]]] = INSTRUMENT_POSES,
    place_under_camera: bool = False,
) -> None:
    """
    Perform a full pick-and-place operation for a given instrument.

    Sequence:
      1. Open gripper.
      2. Move to instrument pick pose and close gripper.
      3. Return to START_POSE.
      4. Optionally move to `place_camera` pose.
      5. Move to `place_tray` pose and open gripper.
      6. Close gripper and go to SLEEP_POSE.

    :param robots: Mapping of robot names to InterbotixManipulatorXS instances.
    :param dt: Time step used by `move_arms` / `move_grippers`.
    :param instrument_name: Key into `instrument_poses` for which instrument to use.
    :param instrument_poses: Dictionary of instrument poses.
    :param place_under_camera: If True, move to `place_camera` before `place_tray`.
    """
    if instrument_name not in instrument_poses:
        raise ValueError(f"Unknown instrument '{instrument_name}' in instrument_poses.")

    poses_for_instrument = instrument_poses[instrument_name]
    pick_pose = poses_for_instrument["pick"]
    place_tray_pose = poses_for_instrument["place_tray"]
    place_camera_pose = poses_for_instrument.get("place_camera")

    # Ensure we have a camera pose if the user requested it
    if place_under_camera and place_camera_pose is None:
        raise ValueError(
            f"Instrument '{instrument_name}' does not define a 'place_camera' pose "
            "but place_under_camera=True was requested."
        )

    # Step 1: open grippers
    move_grippers(
        robots.values(),
        [FOLLOWER_GRIPPER_JOINT_OPEN],
        moving_time=0.5,
        dt=dt,
    )
    time.sleep(2.0)

    # Step 2: move to pick pose
    move_arms(
        bot_list=robots.values(),
        dt=dt,
        target_pose_list=[pick_pose],
        moving_time=5.0,
    )
    time.sleep(1.0)

    # Close to grasp the instrument
    move_grippers(
        robots.values(),
        [FOLLOWER_GRIPPER_JOINT_CLOSE],
        moving_time=0.5,
        dt=dt,
    )
    time.sleep(1.0)

    # Step 3: return to start pose
    move_arms(
        bot_list=robots.values(),
        dt=dt,
        target_pose_list=[START_POSE],
        moving_time=5.0,
    )
    time.sleep(1.0)

    # Step 4 (optional): move to camera pose
    if place_under_camera and place_camera_pose is not None:
        move_arms(
            bot_list=robots.values(),
            dt=dt,
            target_pose_list=[place_camera_pose],
            moving_time=5.0,
        )
        time.sleep(1.0)

    # Step 5: move to tray pose and open grippers
    move_arms(
        bot_list=robots.values(),
        dt=dt,
        target_pose_list=[place_tray_pose],
        moving_time=5.0,
    )
    time.sleep(1.0)

    move_grippers(
        robots.values(),
        [FOLLOWER_GRIPPER_JOINT_OPEN],
        moving_time=0.5,
        dt=dt,
    )
    time.sleep(1.0)

    # Optionally close gripper again before going to sleep
    move_grippers(
        robots.values(),
        [FOLLOWER_GRIPPER_JOINT_CLOSE],
        moving_time=0.5,
        dt=dt,
    )
    time.sleep(1.0)

    # Step 6: move to sleep pose
    move_arms(
        bot_list=robots.values(),
        dt=dt,
        target_pose_list=[SLEEP_POSE],
        moving_time=5.0,
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
    
    

    # Load robot configuration
    robot_base = args.get('robot', '')

    # Base path of the config directory using absolute path
    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)
    print(f"dt: {dt}")

    # Initialize dictionary for robot instances
    robots: Dict[str, InterbotixManipulatorXS] = {}

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

    node._logger.info('Starting pick and place demo...')

    # Run pick-and-place for the current default instrument (forceps)
    perform_pick_and_place(
        robots=robots,
        dt=dt,
        instrument_name="cylinder",
        place_under_camera=args.get("place_under_camera", False),
    )

    time.sleep(1.0)

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

    parser.add_argument(
        '--place_under_camera',
        action='store_true',
        help='If set, place instrument under camera before placing on tray.',
    )

    main(vars(parser.parse_args()))
