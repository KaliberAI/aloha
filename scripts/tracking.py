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
from std_msgs.msg import Bool, Float32

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

        # q_cmd should only contain n (6) commands for the arm group
        # No padding needed - the arm group only expects 6 joints

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
    smoothly and starts phase-aligned from current joint positions.
    When patient is found, arm stops moving but gripper continues to wave.
    """

    now = t if t is not None else time.time()

    # --- Initialize persistent state fields ---
    patient_found_state.setdefault('paused_time_offset', 0.0)
    patient_found_state.setdefault('last_pause_time', None)
    patient_found_state.setdefault('start_offsets', {})
    patient_found_state.setdefault('last_q_cmd', {})

    # ============================================================
    # 1. Handle pause mode (patient found) → gripper continues waving
    # ============================================================
    if patient_found_state.get('patient_found', False):
        if patient_found_state['last_pause_time'] is None:
            patient_found_state['last_pause_time'] = now

        # Initialize gripper wave parameters if not set
        patient_found_state.setdefault('gripper_offset', 0.0)
        patient_found_state.setdefault('last_gripper_time', now)

        # Increment gripper phase using elapsed time
        dt_gripper = now - patient_found_state['last_gripper_time']
        patient_found_state['last_gripper_time'] = now
        patient_found_state['gripper_offset'] += dt_gripper

        gripper_phase = 2 * math.pi * patient_found_state['gripper_offset'] / gripper_period
        phase = math.sin(gripper_phase) * 0.5 + 0.5  # normalize to 0–1

        target_grip = FOLLOWER_GRIPPER_JOINT_CLOSE + \
                      phase * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)

        # Publish to all follower grippers
        for name, bot in robots.items():
            if 'follower' not in name:
                continue
            cmd = JointSingleCommand(name='gripper', cmd=target_grip)
            bot.gripper.core.pub_single.publish(cmd)

        return

    # ============================================================
    # 2. If we just resumed from pause → fix offsets & apply fade-in
    # ============================================================
    if patient_found_state['last_pause_time'] is not None:

        
        paused_duration = now - patient_found_state['last_pause_time']
        patient_found_state['paused_time_offset'] += paused_duration
        patient_found_state['last_pause_time'] = None
        

        
        for name, bot in robots.items():
            if 'follower' not in name:
                continue
            # Only get the first 6 arm joints, not gripper/finger joints
            joint_names = bot.arm.group_info.joint_names
            n_arm = min(6, len(joint_names))
            current_positions = bot.arm.core.joint_states.position[:n_arm]
            patient_found_state['start_offsets'][name] = list(current_positions)

        # Mark just resumed for smooth fade-in
        patient_found_state['just_resumed'] = True
        patient_found_state['resume_time'] = now

        
    # ============================================================
    # 3. Generate continuous motion
    # ============================================================
    adjusted_time = now - patient_found_state['paused_time_offset']
    
    omega = 2 * math.pi / period

    # Fade in when first starting or right after resume
    fade_duration = 2.0
    if patient_found_state.get('just_resumed', False):
        fade_scale = min(1.0, (now - patient_found_state['resume_time']) / fade_duration)
        if fade_scale >= 1.0:
            patient_found_state['just_resumed'] = False
    else:
        fade_scale = min(1.0, adjusted_time / fade_duration)

    # ============================================================
    # 4. Apply sine motion to each follower arm
    # ============================================================
    for name, bot in robots.items():
        if 'follower' not in name:
            continue

        joint_names = bot.arm.group_info.joint_names
        n = min(6, len(joint_names))

        # Initialize start offsets if not yet stored
        if name not in patient_found_state['start_offsets']:
            current_positions = bot.arm.core.joint_states.position[:n]
            patient_found_state['start_offsets'][name] = list(current_positions)

        start_offsets = patient_found_state['start_offsets'][name]
        
        # Ensure start_offsets only contains arm joints (first 6)
        if len(start_offsets) > n:
            start_offsets = start_offsets[:n]
            patient_found_state['start_offsets'][name] = start_offsets

        # --- Natural wave pattern ---
        q_cmd = list(start_offsets)
        q_cmd[0] += amplitude * math.sin(omega * adjusted_time)         # waist
        if n > 1:
            q_cmd[1] += 0.1 * amplitude * math.sin(omega * adjusted_time)  # shoulder
        if n > 2:
            q_cmd[2] += 0.1 * amplitude * math.sin(omega * adjusted_time + 0.8 * math.pi)  # elbow

        # Apply fade-in scaling
        for i in range(len(q_cmd)):
            q_cmd[i] = start_offsets[i] + (q_cmd[i] - start_offsets[i]) * fade_scale

        # Smooth interpolation with previous command
        if name not in patient_found_state['last_q_cmd']:
            patient_found_state['last_q_cmd'][name] = q_cmd
        last_q = patient_found_state['last_q_cmd'][name]
        # Ensure last_q also only has n elements
        if len(last_q) != len(q_cmd):
            last_q = list(q_cmd)  # Reset if mismatch
        smoothed_q = [0.8 * l + 0.2 * q for l, q in zip(last_q, q_cmd)]
        patient_found_state['last_q_cmd'][name] = smoothed_q

        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = smoothed_q
        bot.arm.core.pub_group.publish(msg)



def demo_base_rotation_time_adjusted(robots, dt, t, period=10.0, amplitude=0.5, continuous=True):
    for name, bot in robots.items():
        if 'follower' not in name:
            continue

        waist_joint_name = bot.arm.group_info.joint_names[0]

        if continuous:
            waist_angle = (2 * math.pi * (t / period)) % (2 * math.pi)
        else:
            waist_angle = amplitude * math.sin(2 * math.pi * t / period)

        waist_cmd = JointSingleCommand()
        waist_cmd.name = waist_joint_name
        waist_cmd.cmd = waist_angle
        bot.arm.core.pub_single.publish(waist_cmd)
# def demo_base_rotation_with_patient_found(robots, dt, patient_found_state, period=10.0, amplitude=0.5, continuous=True):
#     """
#     Smoothly rotates only the base ('waist') joint of follower arms.
#     Stops rotation when patient_found is True, resumes when False.
    
#     :param robots: Dictionary of robot instances
#     :param dt: Time step
#     :param patient_found_state: Dictionary with 'patient_found' boolean flag
#     :param period: Rotation period in seconds
#     :param amplitude: Amplitude for oscillation mode
#     :param continuous: Whether to use continuous rotation or oscillation
#     """
#     # Only rotate if patient is not found
#     if patient_found_state.get('patient_found', False):
#         return  # Stop rotation when patient is found
    
#     # Resume normal rotation when patient is not found
#     demo_base_rotation(robots, dt, period, amplitude, continuous)

# def demo_base_rotation_with_patient_found(robots, dt, patient_found_state,
#                                           period=10.0, amplitude=0.5, continuous=True):
#     """
#     Smoothly rotates only the base ('waist') joint of follower arms.
#     Stops rotation when patient_found is True, resumes seamlessly when False.
#     """
#     now = time.time()

#     # If patient is found, record when pause started (only once)
#     if patient_found_state.get('patient_found', False):
#         if patient_found_state['last_pause_time'] is None:
#             patient_found_state['last_pause_time'] = now
#         return  # Stop sending new commands (rotation freezes)

#     # If we were previously paused, update the offset
#     if patient_found_state['last_pause_time'] is not None:
#         paused_duration = now - patient_found_state['last_pause_time']
#         patient_found_state['paused_time_offset'] += paused_duration
#         patient_found_state['last_pause_time'] = None  # reset after resuming

#     # Compute adjusted time for rotation (frozen during pause)
    
#     adjusted_time = now - patient_found_state['paused_time_offset']

#     # Use adjusted time instead of raw time
#     demo_base_rotation_time_adjusted(robots, dt, adjusted_time, period, amplitude, continuous)

def demo_base_rotation_with_patient_found(robots, dt, patient_found_state,
                                          period=10.0, amplitude=0.5, continuous=True, t=None):
    """
    Smooth base rotation that pauses when patient_found is True and resumes seamlessly.
    Now with time alignment and fade-in smoothing.
    """
    now = t if t is not None else time.time()

    # Pause logic
    if patient_found_state.get('patient_found', False):
        if patient_found_state['last_pause_time'] is None:
            patient_found_state['last_pause_time'] = now
        return
    if patient_found_state['last_pause_time'] is not None:
        paused_duration = now - patient_found_state['last_pause_time']
        patient_found_state['paused_time_offset'] += paused_duration
        patient_found_state['last_pause_time'] = None

    adjusted_time = now - patient_found_state['paused_time_offset']

    # --- SMOOTH START FADE-IN ---
    fade_duration = 2.0
    fade_scale = min(1.0, adjusted_time / fade_duration)

    for name, bot in robots.items():
        if 'follower' not in name:
            continue
        waist_joint_name = bot.arm.group_info.joint_names[0]

        if continuous:
            waist_angle = (2 * math.pi * (adjusted_time / period)) % (2 * math.pi)
        else:
            waist_angle = amplitude * math.sin(2 * math.pi * adjusted_time / period)

        # Apply fade-in scaling
        waist_angle *= fade_scale

        # Optional smoothing
        last_angle = patient_found_state.get('last_waist_angle', 0.0)
        smoothed_angle = 0.8 * last_angle + 0.2 * waist_angle
        patient_found_state['last_waist_angle'] = smoothed_angle

        waist_cmd = JointSingleCommand()
        waist_cmd.name = waist_joint_name
        waist_cmd.cmd = smoothed_angle
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
    
    # Shared state for patient_found topic
    patient_found_state = {'patient_found': False, 'paused_time_offset': 0.0, 'last_pause_time': None, 'last_waist_angle': 0.0}
    
    # Subscribe to patient_found topic
    def patient_found_callback(msg):
        
        patient_found_state['patient_found'] = msg.data
    
    patient_found_sub = node.create_subscription(
        Bool,
        'patient_found',
        patient_found_callback,
        10
    )

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

        
        # demo_base_rotation(robots, dt, period=8.0, amplitude=0.6, continuous=False)
        if not motion_state['initialized']:
            motion_state['start_time'] = time.time()
            motion_state['initialized'] = True


        elapsed_t = time.time() - motion_state['start_time']


        # Demo with patient_found check: stops when patient_found is True, resumes when False
        # demo_base_rotation_with_patient_found(
        #     robots, dt, patient_found_state, 
        #     period=8.0, amplitude=0.6, continuous=False, t = elapsed_t
        # )

       
        demo_multilink_wave_with_patient_found(
            robots, dt, patient_found_state,
            period=16.0, amplitude=0.8, continuous=False, t=elapsed_t
            )
        
       
        
        # demo_viper_breathing(robots, dt, period=8.0,
        #                  base_amplitude=0.35,
        #                  phase_offset=0.6,
        #                  decay_factor=0.55)
        
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
