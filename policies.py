import math
import numpy as np
from pid import PID
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


class StackPolicy(object):
    """

    
    Policy for the Block Stacking task.
    Phase 1: Hover over Cube A.
    Phase 2: Lower and Grasp Cube A.
    Phase 3: Lift and Move above Cube B.
    Phase 4: Stack on Cube B and Release.


    """
    def __init__(self, obs):
        """
        Args:
            obs (dict): Includes 'cubeA_pos' and 'cubeB_pos'.
        """
        # Initialize PID controller and waypoints here

        self.phase = 1
        self.dt = 0.01

        self.hover_height = 0.10
        self.threshold = .02

        # Initialize PID with dummy target (will be set every phase)
        self.pid = PID(
            kp=1,
            ki=0,
            kd=.2,
            target=np.zeros(3)
        )

        cubeA = obs["cubeA_pos"]
        cubeB = obs["cubeB_pos"]

        hover_A = cubeA + np.array([0.0, 0.0, self.hover_height])
        grasp_A = cubeA + np.array([0.0, 0.0, -0.02])
        lift_A  = cubeA + np.array([0.0, 0.0, self.hover_height])
        hover_B = cubeB + np.array([0.0, 0.0, self.hover_height])
        place_B = cubeB
        final = cubeB + np.array([0.0, 0.0, self.hover_height * 4])

        self.waypoints = {
            1: hover_A,
            2: grasp_A,
            3: lift_A,
            4: hover_B,
            5: place_B,
            6: final
        }

            
    def get_action(self, obs):
        """
        Returns:
            np.ndarray: 7D action (Delta-X, Delta-Y, Delta-Z, Axis-Angle [3], Gripper).
        """
        eef_pos = obs["robot0_eef_pos"]
        target = self.waypoints.get(self.phase)


        if self.phase == 1:
            print("Phase 1: Hovering above Cube A")
            gripper = -1

        elif self.phase == 2:
            print("Phase 2: Descending to grasp Cube A")
            if np.linalg.norm(target - eef_pos) < self.threshold:
                gripper = 1.0
            else:
                gripper = -1

        elif self.phase == 3:
            print("Phase 3: Lifting Cube A")
            gripper = 1.0

        elif self.phase == 4:
            print("Phase 4: Hovering above Cube B")
            self.threshold = .03
            gripper = 1.0

        elif self.phase == 5:
            self.threshold = .07
            print("Phase 5: Placing Cube A on Cube B")            
            if np.linalg.norm(target - eef_pos) < self.threshold:
                gripper = -1.0
            else:
                gripper = 1.0   

        elif self.phase == 6:
            self.threshold = .001
            gripper = -1.0
            print("Task Completed!")
        else:
            print("Invalid Phase")
            return np.zeros(7)  # No action if phase is invalid

        self.pid.target = target
        delta_pos = self.pid.update(eef_pos, self.dt)

        action = np.zeros(7)
        action[0:3] = delta_pos
        action[3:6] = 0.0
        action[6]   = gripper

        if np.linalg.norm(target - eef_pos) < self.threshold:
            self.phase += 1
            self.pid.reset()


        return action

import numpy as np
from scipy.spatial.transform import Rotation as R


class NutAssemblyPolicy:
    # -------------------------------
    # FSM states
    # -------------------------------
    APPROACH = 0
    HOVER = 1
    PRELOAD = 2
    RISE = 3      # Lift straight up from nut's original position
    LIFT = 4      # Move toward peg
    INSERT = 5
    SEAT = 6
    RELEASE = 7
    HOME = 8

    def __init__(self, obs):
        self.dt = 0.01
        self.max_speed = 0.12

        self.pid = PID(kp=2.0, ki=0.0, kd=0.2, target=np.zeros(3))

        self.state = self.APPROACH
        self.delay = 0
        self.rise_pos = None
        self.threshold = 0.02

        self.home_pos = obs["robot0_eef_pos"].copy()
        self.home_quat = obs.get("robot0_eef_quat", None).copy()

        self.sequence = [
            {
                "name": "RoundNut",
                "peg_pos": np.array([0.23, -0.105, 0.85]),
            },
            None,  # HOME
            {
                "name": "SquareNut",
                "peg_pos": np.array([0.23, 0.105, 0.85]),
            },
        ]

        self.sequence_idx = 0

    def handle_position(self, nut_pos, nut_quat):
        rot = R.from_quat(nut_quat)
        return nut_pos + rot.apply([0.065, 0.0, 0.0])

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", None)

        action = np.zeros(7)

        cfg = self.sequence[self.sequence_idx]

        if cfg is None:
            target = self.home_pos
            gripper = -1.0

            curr_rot = R.from_quat(eef_quat)
            home_rot = R.from_quat(self.home_quat)
            rot_err = home_rot * curr_rot.inv()
            euler_err = rot_err.as_euler("xyz")
            euler_err = ((euler_err + np.pi) % (2 * np.pi)) - np.pi

            action[3] = 0.2 * euler_err[0]  # roll
            action[4] = 0.2 * euler_err[1]  # pitch
            action[5] = 0.2 * euler_err[2]  # yaw

            orientation_err = np.linalg.norm(euler_err)
            if np.linalg.norm(target - eef_pos) < self.threshold and orientation_err < 0.05 and self.delay > 100:
                self.sequence_idx += 1
                self.state = self.APPROACH
                self.delay = 0

        else:
            nut_pos = obs[f"{cfg['name']}_pos"]
            nut_quat = obs[f"{cfg['name']}_quat"]
            handle = self.handle_position(nut_pos, nut_quat)

            if self.state == self.APPROACH:
                target = handle + np.array([0, 0, 0.15])
                gripper = -1.0

            elif self.state == self.HOVER:
                target = handle 
                gripper = -1.0

            elif self.state == self.PRELOAD:
                target = handle - np.array([0, 0, .03])
                gripper = 1.0

            elif self.state == self.RISE:
                offset = nut_pos - eef_pos
                target = self.rise_pos + np.array([0, 0, 0.20]) - offset

                if np.linalg.norm(offset) > 0.1:
                    self.state = self.APPROACH
                    self.delay = 0

                gripper = 1.0

            elif self.state == self.LIFT:
                offset = nut_pos - eef_pos

                target = cfg["peg_pos"] + np.array([0, 0, 0.20]) - offset
                gripper = 1.0

            elif self.state == self.INSERT:
                offset = nut_pos - eef_pos
                target = cfg["peg_pos"] - offset
                gripper = 1.0

            elif self.state == self.SEAT:
                offset = nut_pos - eef_pos
                target = cfg["peg_pos"] + np.array([0, 0, -.08]) - offset
                gripper = 1.0

            elif self.state == self.RELEASE:
                offset = nut_pos - eef_pos
                target = cfg["peg_pos"] + np.array([0, 0, 0.20]) - offset
                gripper = -1.0

            if self.state == self.APPROACH:
                rel_key = f"{cfg['name']}_to_robot0_eef_quat"
                rel_quat = obs.get(rel_key, None)

                if rel_quat is not None and np.sum(rel_quat) != 0:
                    yaw_err = R.from_quat(rel_quat).as_euler("xyz")[2]
                    yaw_err = ((yaw_err + np.pi) % (2 * np.pi)) - np.pi
                    action[5] = 0.2 * yaw_err

            elif self.state == self.LIFT:
                # Align to home orientation for peg insertion
                curr_yaw = R.from_quat(eef_quat).as_euler("xyz")[2]
                home_yaw = R.from_quat(self.home_quat).as_euler("xyz")[2]
                yaw_err = ((home_yaw - curr_yaw + np.pi) % (2 * np.pi)) - np.pi
                action[5] = 0.2 * yaw_err

            dist = np.linalg.norm(target - eef_pos)
            

            if self.state == self.APPROACH and dist < self.threshold and self.delay > 200:
                self.delay = 0
                self.state = self.HOVER

            elif self.state == self.HOVER and dist < self.threshold and self.delay > 200:
                self.delay = 0
                self.state = self.PRELOAD

            elif self.state == self.PRELOAD and self.delay > 100:
                self.delay = 0
                self.rise_pos = nut_pos.copy()
                self.state = self.RISE

            elif self.state == self.RISE and dist < self.threshold:
                self.delay = 0
                self.state = self.LIFT

            elif self.state == self.LIFT and dist < self.threshold:
                self.delay = 0
                self.state = self.INSERT

            elif self.state == self.INSERT and dist < self.threshold:
                self.delay = 0
                self.state = self.SEAT

            elif self.state == self.SEAT and self.delay > 100:
                self.delay = 0
                self.state = self.RELEASE

            elif self.state == self.RELEASE and self.delay > 30:
                self.delay = 0
                self.sequence_idx += 1
                self.state = self.HOME
                self.pid.reset(self.home_pos)

        self.pid.target = target
        vel = self.pid.update(eef_pos, self.dt)
        action[:3] = np.clip(vel, -self.max_speed, self.max_speed)


        action[-1] = gripper
        self.delay += 1
        return action


class DoorPolicy:

    APPROACH_ABOVE = 1
    ALIGN_HANDLE = 2
    PRESS_DOWN = 3
    PULL_HANDLE = 4
    OPEN_DOOR = 5

    def __init__(self, obs):
        self.phase = self.APPROACH_ABOVE
        self.dt = 0.01
        self.delay = 0

        handle_pos = obs["handle_pos"]
        handle_hover_pos = handle_pos + np.array([0.0, 0.0, 0.1])

        self.threshold = 0.03

        self.pid = PID(
            kp=1.0,
            ki=0,
            kd=0.1,
            target=handle_hover_pos,
        )

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        handle_pos = obs["handle_pos"]

        handle_hover_pos = handle_pos + np.array([0.0, 0.0, 0.1])
        error = self.pid.get_error()

        action = np.zeros(7)

        if self.phase == self.APPROACH_ABOVE:
            if error < self.threshold:
                self._transition_to_align()

        elif self.phase == self.ALIGN_HANDLE:
            if error < self.threshold:
                self._transition_to_press(handle_pos)

        elif self.phase == self.PRESS_DOWN:
            if error < self.threshold:
                self._transition_to_pull(handle_pos)

        elif self.phase == self.PULL_HANDLE:
            if error < self.threshold or self.delay > 100:
                self.phase = self.OPEN_DOOR
            else:
                self.delay += 1
        elif self.phase == self.OPEN_DOOR:
            action[:3] = np.array([0.0, 1.0, 0.0])
            action[-1] = 1.0
            return action

        delta_pos = self.pid.update(eef_pos, self.dt)
        action[:3] = delta_pos[:3]

        if self.phase >= self.PULL_HANDLE:
            action[-1] = 1.0  
        else:
            action[-1] = -1.0

        return action
    
    def _transition_to_align(self):
        self.phase = self.ALIGN_HANDLE

    def _transition_to_press(self, handle_pos):
        self.phase = self.PRESS_DOWN
        self.pid.reset(handle_pos + np.array([0.0, 0.0, -0.05]))

    def _transition_to_pull(self, handle_pos):
        self.phase = self.PULL_HANDLE
        self.pid.reset(handle_pos + np.array([0.0, 0.0, -0.5]))
