import numpy as np
from pid import PID
from scipy.spatial.transform import Rotation as R


class StackPolicy(object):
    def __init__(self, obs):
        self.phase = 1
        self.dt = 0.01
        self.max_speed = 0.08

        self.lift_height = 0.20
        self.wait_count = 0

        init_target = obs["cubeA_pos"].copy()
        init_target[2] += 0.1
        self.pid = PID(kp=2.5, ki=0.01, kd=0.1, target=init_target)

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        cubeA_pos = obs["cubeA_pos"]
        cubeB_pos = obs["cubeB_pos"]
        gripper = -1.0
        target = self.pid.target

        if self.phase == 1:
            target = cubeA_pos + np.array([0.0, 0.0, 0.15])
            gripper = -1.0

            if self.pid.get_error() < 0.02:
                self.phase = 2
                print("New phase: 2")
                self.pid.reset(cubeA_pos)

        elif self.phase == 2:
            target = cubeA_pos
            gripper = -1.0
            if self.pid.get_error() < 0.015:
                self.wait_count += 1
                if self.wait_count > 20:
                    self.phase = 3
                    print("New phase: 3")
                    self.wait_count = 0

        elif self.phase == 3:
            target = cubeA_pos + np.array([0.0, 0.0, -0.15])
            gripper = 1.0

            self.wait_count += 1
            if self.wait_count > 20:
                self.phase = 4
                print("New phase: 4")
                self.wait_count = 0
                self.pid.reset(cubeB_pos + np.array([0.0, 0.0, self.lift_height]))

        elif self.phase == 4:
            target = cubeB_pos + np.array([0.0, 0.0, 0.06])
            if self.pid.get_error() < 0.02:
                gripper = -1.0
            else:
                gripper = 1.0

        self.pid.target = target
        action_v = self.pid.update(eef_pos, self.dt)
        action_v = np.clip(action_v, -self.max_speed, self.max_speed)

        action = np.concatenate([action_v, [0.0, 0.0, 0.0, gripper]])
        return action


class NutAssemblyPolicy(object):
    def __init__(self, obs):
        self.dt = 0.01
        self.max_speed = 0.12
        self.max_rot_speed = 1.0

        self.pid = PID(kp=3, ki=0.0, kd=0.2, target=np.zeros(3))
        self.rot_pid = PID(kp=2, ki=0.0, kd=0.1, target=np.zeros(3))

        self.phase = 0
        self.wait_count = 0
        self.force_grip = False
        self.rot_flipped = False

        self.home_pos = obs["robot0_eef_pos"].copy()
        self.home_quat = obs.get("robot0_eef_quat", None)

    def compute_handle_pos(self, nut_pos, nut_quat):
        rot = R.from_quat(nut_quat)
        offset = rot.apply([0.065, 0.0, 0.0])
        return nut_pos + offset

    def calculate_nut_target_rotation(self, nut_quat):
        nut_rot = R.from_quat(nut_quat)

        handle_dir = nut_rot.apply([1.0, 0.0, 0.0])

        camera_forward = np.array([0.0, -1.0, 0.0])

        flipped = False
        base_rot = nut_rot
        if np.dot(handle_dir, camera_forward) < 0.0:
            flip_rot = R.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0]))
            base_rot = flip_rot * nut_rot
            flipped = True

        downward_rot = R.from_euler("xyz", [np.pi, 0.0, 0.0])
        target_rot = base_rot * downward_rot

        if not flipped:
            tilt = R.from_euler("y", np.deg2rad(15.0))
            target_rot = tilt * target_rot

        return target_rot, flipped

    def calculate_nut_target_position(self, nut_pos, nut_quat):
        rot = R.from_quat(nut_quat)
        offset = rot.apply([0.065, 0.0, 0.0])
        return nut_pos + offset

    def calculate_peg_target_position(self, peg_pos, flipped):
        shift = -0.065 if not flipped else 0.065
        return peg_pos + np.array([shift, 0.0, 0.0])

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", None)
        action = np.zeros(7)

        if self.phase == 7:  # home
            local_phase = 0
        elif self.phase < 8:  # round nut
            local_phase = self.phase % 7
        else:  # square nut
            local_phase = (self.phase - 8) % 7

        if self.phase == 7:
            target = self.home_pos
            gripper = -1.0
        elif self.phase < 8:
            nut_pos = obs["RoundNut_pos"]
            nut_quat = obs["RoundNut_quat"]
            peg_pos = np.array([0.23, -0.105, 0.85])
            handle_pos = self.calculate_nut_target_position(nut_pos, nut_quat)
            target_rot, self.rot_flipped = self.calculate_nut_target_rotation(nut_quat)

            if local_phase == 0:
                target = handle_pos + np.array([0, 0, 0.15])
                gripper = -1.0

            elif local_phase == 1:
                target = handle_pos
                gripper = -1.0

            elif local_phase == 2:
                preload_z = 0.04
                target = handle_pos - np.array([0, 0, preload_z])
                gripper = 1.0
                self.force_grip = True
                self.wait_count += 1

            elif local_phase == 3:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] + 0.20
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0

            elif local_phase == 4:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] + 0.00
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0

            elif local_phase == 5:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] - 0.07
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0
                self.wait_count += 1

            elif local_phase == 6:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] + 0.20
                target = desired_nut_pos - eef_to_nut
                gripper = -1.0
                self.wait_count += 1
        else:
            nut_pos = obs["SquareNut_pos"]
            nut_quat = obs["SquareNut_quat"]
            peg_pos = np.array([0.23, 0.105, 0.85])
            handle_pos = self.calculate_nut_target_position(nut_pos, nut_quat)
            target_rot, self.rot_flipped = self.calculate_nut_target_rotation(nut_quat)

            if local_phase == 0:
                target = handle_pos + np.array([0, 0, 0.15])
                gripper = -1.0

            elif local_phase == 1:
                target = handle_pos
                gripper = -1.0

            elif local_phase == 2:
                preload_z = 0.05
                target = handle_pos - np.array([0, 0, preload_z])
                gripper = 1.0
                self.force_grip = True
                self.wait_count += 1

            elif local_phase == 3:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] + 0.20
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0

            elif local_phase == 4:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] + 0.00
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0

            elif local_phase == 5:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                seat_dz = -0.09
                desired_nut_pos[2] = peg_pos[2] + seat_dz
                target = desired_nut_pos - eef_to_nut
                gripper = 1.0
                self.wait_count += 1

            elif local_phase == 6:
                eef_to_nut = nut_pos - eef_pos
                desired_nut_pos = peg_pos.copy()
                desired_nut_pos[2] = peg_pos[2] - 0.07
                target = desired_nut_pos - eef_to_nut
                gripper = -1.0
                self.wait_count += 1

        self.pid.target = target
        vel = self.pid.update(eef_pos, self.dt)
        vel = np.clip(vel, -self.max_speed, self.max_speed)
        action[:3] = vel

        if self.phase == 7:
            if eef_quat is not None and self.home_quat is not None:
                curr_yaw = R.from_quat(eef_quat).as_euler("xyz")[2]
                home_yaw = R.from_quat(self.home_quat).as_euler("xyz")[2]
                yaw_err = home_yaw - curr_yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                action[3] = 0.0
                action[4] = 0.0
                action[5] = 0.2 * yaw_err
            else:
                action[3] = 0.0
                action[4] = 0.0
                action[5] = 0.0
        elif self.phase < 8:
            if local_phase < 4:
                rel_key = "RoundNut_to_robot0_eef_quat"
                rel_quat = obs.get(rel_key, None)
                if rel_quat is not None and np.sum(rel_quat) != 0:
                    current_yaw = R.from_quat(rel_quat).as_euler("xyz")[2]
                    action[3] = 0.0
                    action[4] = 0.0
                    action[5] = 0.2 * current_yaw
            else:
                action[3] = 0.0
                action[4] = 0.0
                action[5] = 0.0
        else:
            if local_phase < 3:
                rel_key = "SquareNut_to_robot0_eef_quat"
                rel_quat = obs.get(rel_key, None)
                if rel_quat is not None and np.sum(rel_quat) != 0:
                    current_yaw = R.from_quat(rel_quat).as_euler("xyz")[2]
                    action[3] = 0.0
                    action[4] = 0.0
                    action[5] = 0.2 * current_yaw
            elif local_phase < 4:
                if eef_quat is not None and self.home_quat is not None:
                    curr_yaw = R.from_quat(eef_quat).as_euler("xyz")[2]
                    home_yaw = R.from_quat(self.home_quat).as_euler("xyz")[2]
                    yaw_err = home_yaw - curr_yaw
                    yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                    if yaw_err > np.pi / 2:
                        yaw_err -= np.pi
                    elif yaw_err < -np.pi / 2:
                        yaw_err += np.pi
                    action[3] = 0.0
                    action[4] = 0.0
                    action[5] = 0.2 * yaw_err
            else:
                action[3] = 0.0
                action[4] = 0.0
                action[5] = 0.0

        action[-1] = gripper

        dist = np.linalg.norm(target - eef_pos)

        if self.phase == 7:
            if dist < 0.02:
                self.phase += 1
        elif self.phase < 8:
            if local_phase == 1:
                if dist < 0.01:
                    self.phase += 1

            elif local_phase == 2:
                if self.wait_count > 80:
                    self.phase += 1
                    self.wait_count = 0

            elif local_phase == 5:
                if self.wait_count > 30:
                    self.phase += 1
                    self.wait_count = 0

            elif local_phase == 6:
                if self.wait_count > 30:
                    self.phase += 1
                    self.wait_count = 0
                    self.force_grip = False

            else:
                if dist < 0.02:
                    self.phase += 1

            if self.force_grip and local_phase < 6:
                action[-1] = 1.0
        else:
            if local_phase == 1:
                if dist < 0.01:
                    self.phase += 1

            elif local_phase == 2:
                if self.wait_count > 110:
                    self.phase += 1
                    self.wait_count = 0

            elif local_phase == 5:
                if self.wait_count > 30:
                    self.phase += 1
                    self.wait_count = 0

            elif local_phase == 6:
                if self.wait_count > 30:
                    self.phase += 1
                    self.wait_count = 0
                    self.force_grip = False

            else:
                if dist < 0.02:
                    self.phase += 1

            if self.force_grip and local_phase < 6:
                action[-1] = 1.0

        return action


class DoorPolicy(object):
    def __init__(self, obs):
        self.phase = 1
        self.dt = 0.01
        self.wait_count = 0

        handle_pos = obs["handle_pos"]
        self.pid = PID(
            kp=2.0,
            ki=0.05,
            kd=0.2,
            target=handle_pos + np.array([0.0, 0.0, 0.1]),
        )

    def get_action(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        handle_pos = obs["handle_pos"]

        print("phase:", self.phase, "| error:", self.pid.get_error())
        self.wait_count += 1

        if self.phase == 1 and self.pid.get_error() < 0.02:
            self.phase = 2

        elif (
            self.phase == 2
            and self.pid.get_error() < 0.03
            and self.wait_count > 300
        ):
            self.phase = 3
            self.pid.reset(handle_pos + np.array([0.0, 0.0, -0.05]))

        elif (
            self.phase == 3
            and self.pid.get_error() < 0.03
            and self.wait_count > 500
        ):
            self.phase = 4
            self.pid = PID(
                kp=0.5,
                ki=0.01,
                kd=0.1,
                target=handle_pos + np.array([0.0, 0.0, -0.6]),
            )

        elif self.phase == 4 and self.wait_count > 900:
            self.phase = 5

        action = np.zeros(7)
        if self.phase == 5:
            print("Opening door")
            action[:3] = np.array([0.0, 0.3, 0.0])
            action[-1] = 1.0
        else:
            control = self.pid.update(eef_pos, self.dt)
            action[:3] = control[:3]
            if self.phase >= 4:
                action[-1] = 1.0
            else:
                action[-1] = -1.0

        print("Final action:", action)
        return action
