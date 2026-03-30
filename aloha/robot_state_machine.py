"""
Modular state machine for robot motion control with state-specific behaviors.

Each state encapsulates its own motion generation logic, parameters, and
transition handling. This allows easy addition of new states without modifying
existing code.
"""

from enum import Enum
from typing import Dict, Callable, Optional, List, TYPE_CHECKING
import time
import math
from aloha.robot_utils import START_ARM_POSE
if TYPE_CHECKING:
    from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class RobotMotionState(Enum):
    """Enumeration of robot motion states."""
    IDLE = "idle"
    OPENING = "opening"
    WAVING = "waving"
    PAUSED = "paused"
    TRACK = "track"
    NOD = "nod"
    SLEEP = "sleep"


# Key mappings for debug mode
STATE_KEY_MAP = {
    'w': RobotMotionState.WAVING,
    'W': RobotMotionState.WAVING,
    's': RobotMotionState.SLEEP,
    'S': RobotMotionState.SLEEP,
    't': RobotMotionState.TRACK,
    'T': RobotMotionState.TRACK,
    'n': RobotMotionState.NOD,
    'N': RobotMotionState.NOD,
    'o': RobotMotionState.OPENING,
    'O': RobotMotionState.OPENING,
}


class BaseMotionState:
    """
    Base class for state-specific motion behaviors.
    
    Each state subclass implements its own motion generation logic,
    making the system highly modular and extensible.
    """
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        """
        Initialize base motion state.
        
        :param state_machine: Reference to parent state machine
        """
        self.sm = state_machine
        self.ema_alpha = 0.2  # Default EMA smoothing
    
    def on_enter(self, current_time: float):
        """Called when entering this state."""
        pass
    
    def on_exit(self, current_time: float):
        """Called when exiting this state."""
        pass
    
    def update(self, current_time: float, dt: float):
        """
        Update state logic (called every frame).
        
        :param current_time: Current timestamp
        :param dt: Time step
        """
        pass
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """
        Generate joint commands for this state.
        
        :param robot_name: Name of the robot
        :param current_joints: Current joint positions
        :return: Target joint positions
        """
        return current_joints
    
    def generate_gripper_command(self) -> Optional[float]:
        """
        Generate gripper command for this state.
        
        :return: Gripper position or None to keep current
        """
        return None
    
    def smooth_command(self, last_cmd: List[float], new_cmd: List[float]) -> List[float]:
        """Apply EMA smoothing to joint commands."""
        return [
            (1 - self.ema_alpha) * l + self.ema_alpha * n
            for l, n in zip(last_cmd, new_cmd)
        ]


class OpeningState(BaseMotionState):
    """Opening ceremony state - moves to starting pose."""
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        super().__init__(state_machine)
        # self.target_pose = [0.0, -1.05, 0.42, 0.0, 1.05, 0.0]
        self.target_pose = START_ARM_POSE[:6]
        self.moving_time = 4.0
        self.start_time = None
        self.start_poses = {}
    
    def on_enter(self, current_time: float):
        """Capture starting poses when entering."""
        self.start_time = current_time
        self.start_poses = {}
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """Interpolate from current pose to target pose."""
        if self.start_time is None:
            return current_joints
        
        # Capture initial pose
        if robot_name not in self.start_poses:
            self.start_poses[robot_name] = list(current_joints[:6])
        
        # Calculate interpolation factor
        elapsed = time.time() - self.start_time
        t = min(1.0, elapsed / self.moving_time)
        
        # Smooth interpolation (ease-in-out)
        t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)
        
        # Interpolate
        start = self.start_poses[robot_name]
        target = self.target_pose
        result = [
            s + t_smooth * (tgt - s)
            for s, tgt in zip(start[:6], target)
        ]
        
        return result


class WavingState(BaseMotionState):
    """Waving state - sinusoidal wave motion relative to current pose."""
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        super().__init__(state_machine)
        self.period = 16.0
        self.amplitude = 0.8
        self.omega = 2.0 * math.pi / self.period
        self.phase = 0.0
        self.last_update_time = None
        self.base_poses = {}  # Base pose for each robot (captured on entry)
        self.blend_start_time = {}  # Start time for blend-in per robot
        self.blend_duration = 1  # Blend-in duration in seconds
        self.entry_poses = {}  # Pose when entering state (for smooth transition)
        self.initialized = False
    
    def on_enter(self, current_time: float):
        """Reset entry state when entering waving, but keep phase continuous."""
        self.entry_poses = {}
        self.blend_start_time = {}
        # Don't reset base_poses - keep them for continuity
        # Don't reset phase - keep it continuous
        # Reset last_update_time so phase only accumulates while state is active
        self.last_update_time = current_time
        self.initialized = True
    
    def on_exit(self, current_time: float):
        """Called when exiting this state - phase is preserved."""
        # Phase is stored in self.phase, so it persists across state transitions
        # last_update_time will be reset on next entry
        pass
    
    def update(self, current_time: float, dt: float):
        """Update wave phase only while this state is active."""
        if self.last_update_time is None:
            self.last_update_time = current_time
            return
        
        # Only accumulate phase based on dt (time step), not absolute time difference
        # This ensures phase only advances while state is active
        self.phase += self.omega * dt
        self.last_update_time = current_time
    
    def get_blend_factor(self, robot_name: str, current_time: float) -> float:
        """Calculate blend factor for smooth entry (0 to 1)."""
        if robot_name not in self.blend_start_time:
            return 1.0  # No blending needed if not started
        
        elapsed = current_time - self.blend_start_time[robot_name]
        u = elapsed / max(1e-6, self.blend_duration)
        u = max(0.0, min(1.0, u))
        
        # Smooth blend-in (ease-in-out with zero slope at ends)
        return 0.5 - 0.5 * math.cos(math.pi * u)
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """Generate wave motion relative to base pose with smooth entry."""
        # Capture entry pose and base pose on first call after entering
        if robot_name not in self.entry_poses:
            current_pose = list(current_joints[:6])
            self.entry_poses[robot_name] = current_pose
            
            # If base pose doesn't exist, use current pose as base
            # Otherwise, keep existing base pose for continuity
            if robot_name not in self.base_poses:
                self.base_poses[robot_name] = current_pose
            
            self.blend_start_time[robot_name] = time.time()
        
        base = self.base_poses[robot_name]
        entry_pose = self.entry_poses[robot_name]
        
        # Wave deltas with phase lags
        d0 = self.amplitude * math.sin(self.phase)  # waist
        d1 = 0.1 * self.amplitude * math.sin(self.phase)  # shoulder
        d2 = 0.1 * self.amplitude * math.sin(self.phase + 0.8 * math.pi)  # elbow
        
        # Target wave pose
        wave = list(base[:6])
        wave[0] = base[0] + d0
        wave[1] = base[1] + d1
        wave[2] = base[2] + d2
        
        # Smooth blend from entry pose to wave motion
        blend = self.get_blend_factor(robot_name, time.time())
        result = [
            (1.0 - blend) * entry + blend * w
            for entry, w in zip(entry_pose[:6], wave[:6])
        ]
        
        return result


class TrackState(BaseMotionState):
    """Track state - frozen pose with waist tracking patient."""
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        super().__init__(state_machine)
        self.frozen_poses = {}  # Frozen pose when tracking started
        self.waist_centers = {}
        self.waist_offsets = {}
        self.last_waist_update = time.time()
        
        # Waist tracking parameters
        self.waist_deadzone = 50.0
        self.waist_speed_rad_s = 0.05
        self.waist_max_angle = 0.7
        
        # Gripper wave parameters
        self.gripper_period = 4.0
        self.gripper_offset = 0.0
        self.last_gripper_update = time.time()
    
    def on_enter(self, current_time: float):
        """Capture frozen poses and initialize waist tracking."""
        self.frozen_poses = {}
        self.waist_centers = {}
        self.waist_offsets = {}
        self.last_waist_update = current_time
        self.last_gripper_update = current_time
    
    def update(self, current_time: float, dt: float):
        """Update gripper offset for continuous motion."""
        dt_gripper = max(0.0, current_time - self.last_gripper_update)
        self.gripper_offset += dt_gripper
        self.last_gripper_update = current_time
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """Generate tracking pose with waist offset."""
        # Capture frozen pose on first call
        if robot_name not in self.frozen_poses:
            self.frozen_poses[robot_name] = list(current_joints[:6])
            self.waist_centers[robot_name] = current_joints[0]
            self.waist_offsets[robot_name] = 0.0
        
        # Calculate waist tracking direction
        x_dist = self.sm.x_distance
        if x_dist > self.waist_deadzone:
            direction = -1.0
        elif x_dist < -self.waist_deadzone:
            direction = 1.0
        else:
            direction = 0.0
        
        # Update waist offset
        dt_waist = max(1e-3, time.time() - self.last_waist_update)
        self.last_waist_update = time.time()
        
        offset = self.waist_offsets[robot_name]
        if not ((offset >= self.waist_max_angle and direction > 0) or
                (offset <= -self.waist_max_angle and direction < 0)):
            offset += direction * self.waist_speed_rad_s * dt_waist
        self.waist_offsets[robot_name] = offset
        
        # Generate pose: frozen pose with waist tracking
        frozen = self.frozen_poses[robot_name]
        result = list(frozen)
        result[0] = self.waist_centers[robot_name] + offset
        
        return result
    
    def generate_gripper_command(self) -> float:
        """Generate continuous gripper wave."""
        from aloha.robot_utils import FOLLOWER_GRIPPER_JOINT_CLOSE, FOLLOWER_GRIPPER_JOINT_OPEN
        
        g_phase = 2 * math.pi * self.gripper_offset / max(1e-6, self.gripper_period)
        phase01 = 0.5 * (math.sin(g_phase) + 1.0)
        return FOLLOWER_GRIPPER_JOINT_CLOSE + \
            phase01 * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)


class NodState(BaseMotionState):
    """Nod state - slow elbow up/down motion."""
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        super().__init__(state_machine)
        self.period = 3.0  # 3 seconds per nod
        self.amplitude = 0.3  # 0.3 rad amplitude
        self.omega = 2.0 * math.pi / self.period
        self.phase = 0.0
        self.last_update_time = time.time()
        self.base_poses = {}
    
    def on_enter(self, current_time: float):
        """Capture base poses when entering nod."""
        self.base_poses = {}
        self.phase = 0.0
        self.last_update_time = current_time
    
    def update(self, current_time: float, dt: float):
        """Update nod phase only while this state is active."""
        # Only accumulate phase based on dt (time step), not absolute time difference
        self.phase += self.omega * dt
        self.last_update_time = current_time
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """Generate nod motion - only elbow moves."""
        # Capture base pose on first call
        if robot_name not in self.base_poses:
            self.base_poses[robot_name] = list(current_joints[:6])
        
        base = self.base_poses[robot_name]
        result = list(base)
        
        # Only elbow (joint 2) moves with sine wave
        elbow_offset = self.amplitude * math.sin(self.phase)
        result[2] = base[2] + elbow_offset
        
        return result


class SleepState(BaseMotionState):
    """Sleep state - moves to sleep position."""
    
    def __init__(self, state_machine: 'RobotStateMachine'):
        super().__init__(state_machine)
        self.target_pose = [0.0, -1.85, 1.6057, 0.0, 0.8203, 0.0]
        self.moving_time = 5.0
        self.start_time = None
        self.start_poses = {}
    
    def on_enter(self, current_time: float):
        """Capture starting poses when entering."""
        self.start_time = current_time
        self.start_poses = {}
    
    def generate_joint_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> List[float]:
        """Interpolate from current pose to sleep pose."""
        if self.start_time is None:
            return current_joints
        
        # Capture initial pose
        if robot_name not in self.start_poses:
            self.start_poses[robot_name] = list(current_joints[:6])
        
        # Calculate interpolation factor
        elapsed = time.time() - self.start_time
        t = min(1.0, elapsed / self.moving_time)
        
        # Smooth interpolation (ease-in-out)
        t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)
        
        # Interpolate
        start = self.start_poses[robot_name]
        target = self.target_pose
        result = [
            s + t_smooth * (tgt - s)
            for s, tgt in zip(start[:6], target)
        ]
        
        return result


class RobotStateMachine:
    """
    State machine for managing robot motion with modular state behaviors.
    
    Each state handles its own motion generation, making the system
    highly extensible and maintainable.
    """
    
    def __init__(self):
        """Initialize the state machine."""
        self.current_state = RobotMotionState.IDLE
        
        # Initialize all state handlers
        self.states: Dict[RobotMotionState, BaseMotionState] = {
            RobotMotionState.OPENING: OpeningState(self),
            RobotMotionState.WAVING: WavingState(self),
            RobotMotionState.TRACK: TrackState(self),
            RobotMotionState.NOD: NodState(self),
            RobotMotionState.SLEEP: SleepState(self),
        }
        
        # Per-robot smoothing state
        self.last_joint_commands: Dict[str, List[float]] = {}
        
        # Patient tracking state
        self.x_distance: float = 0.0
    
    def transition_to(self, new_state: RobotMotionState, current_time: float):
        """
        Transition to a new state.
        
        :param new_state: Target state
        :param current_time: Current timestamp
        """
        if new_state == self.current_state:
            return
        
        # Call exit handler for current state
        if self.current_state in self.states:
            self.states[self.current_state].on_exit(current_time)
        
        # Transition
        old_state = self.current_state
        self.current_state = new_state
        
        # Call enter handler for new state
        if new_state in self.states:
            self.states[new_state].on_enter(current_time)
        
        print(f"State transition: {old_state.value} -> {new_state.value}")
    
    def update(self, current_time: float, dt: float):
        """
        Update current state logic.
        
        :param current_time: Current timestamp
        :param dt: Time step
        """
        if self.current_state in self.states:
            self.states[self.current_state].update(current_time, dt)
    
    def generate_commands(
        self,
        robot_name: str,
        current_joints: List[float],
    ) -> tuple[List[float], Optional[float]]:
        """
        Generate joint and gripper commands for current state.
        
        :param robot_name: Name of the robot
        :param current_joints: Current joint positions
        :return: (joint_commands, gripper_command)
        """
        if self.current_state not in self.states:
            return current_joints[:6], None
        
        state_handler = self.states[self.current_state]
        
        # Generate raw commands
        joint_cmd = state_handler.generate_joint_commands(robot_name, current_joints)
        gripper_cmd = state_handler.generate_gripper_command()
        
        # Apply smoothing
        if robot_name in self.last_joint_commands:
            joint_cmd = state_handler.smooth_command(
                self.last_joint_commands[robot_name],
                joint_cmd
            )
        
        self.last_joint_commands[robot_name] = joint_cmd
        
        return joint_cmd, gripper_cmd
    
    def set_x_distance(self, x_distance: float):
        """Update patient x-distance reading."""
        self.x_distance = x_distance
    
    def get_state(self) -> RobotMotionState:
        """Get current state."""
        return self.current_state