import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Wrapper, spaces

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig
from crisp_drl.envs.sam3_pe import PoseEstimationHelper


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(axis, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    first = _normalize(np.cross(axis, reference))
    second = _normalize(np.cross(axis, first))
    print(f"Orthogonal basis for axis {axis}:\n  first: {first}\n  second: {second}")
    return first, second


def _rotation_matrix_from_euler_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler_xyz, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _euler_xyz_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    sin_pitch = -float(rotation_matrix[2, 0])
    if abs(abs(sin_pitch) - 1.0) < 1e-8:
        pitch = np.pi / 2.0 if sin_pitch > 0.0 else -np.pi / 2.0
        roll = 0.0
        if sin_pitch > 0.0:
            yaw = float(np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1]))
        else:
            yaw = float(np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1]))
        return np.array([roll, pitch, yaw])

    pitch = float(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))
    roll = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    yaw = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    return np.array([roll, pitch, yaw])


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3)

    axis = rotvec / theta
    kx, ky, kz = axis
    k = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    identity = np.eye(3)
    return identity + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _rotvec_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(rotation_matrix) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-8:
        return np.zeros(3)

    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]
    )
    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-8:
        return np.zeros(3)

    axis = axis / denom
    return axis * theta


def rotate_by_30_y(vec: np.ndarray) -> np.ndarray:
    angle_rad = np.deg2rad(30.0)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ]
    )
    return rotation @ vec


def _barcode_rgbd_from_obs(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (
        obs["observation.images.wrist_camera"],
        obs["observation.images.wrist_depth_camera"],
    )


def _estimate_barcode_poses(
    pose_helper: PoseEstimationHelper,
    obs: dict[str, np.ndarray],
    label: str,
) -> list[np.ndarray]:
    poses = pose_helper.estimate_barcodes(
        _barcode_rgbd_from_obs(obs), obs["observation.state.cartesian"]
    )
    if not poses:
        raise RuntimeError(f"No {label} barcode poses detected")
    return poses


def _barcode_pose_position(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose[:3, 3], dtype=float)


def _offset_pose_world(pose: np.ndarray, offset: np.ndarray, local: bool) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    if local:
        return (
            _barcode_pose_position(pose)
            + np.asarray(pose[:3, :3], dtype=float) @ offset
        )
    return _barcode_pose_position(pose) + offset


def _print_estimation_summary(
    label: str,
    estimated_position: np.ndarray,
    ground_truth_position: np.ndarray,
) -> None:
    estimated_position = np.asarray(estimated_position, dtype=float)
    ground_truth_position = np.asarray(ground_truth_position, dtype=float)
    offset = estimated_position - ground_truth_position
    print(
        f"{label} estimated position: {estimated_position}, "
        f"ground truth position: {ground_truth_position}, "
        f"offset: {offset}"
    )


class InsertionWrapperBox(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: ShelfBoxConfig,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(-0.002, 0.002),
        goal_position_randomisation_plane_range=(-0.0028, 0.0028),
        goal_position_randomisation_plane_x_range=None,
        goal_position_randomisation_plane_y_range=None,
        safety_box_radius=0.004,
        safety_box_step_size=0.0004,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        approach_distance=0.02,
        is_eval=False,
        use_pose_estimation=False,
    ):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position
        self.first_home_config = env_config.custom_first_home_position
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.goal_position_randomisation_plane_x_range = (
            goal_position_randomisation_plane_range
            if goal_position_randomisation_plane_x_range is None
            else goal_position_randomisation_plane_x_range
        )
        self.goal_position_randomisation_plane_y_range = (
            goal_position_randomisation_plane_range
            if goal_position_randomisation_plane_y_range is None
            else goal_position_randomisation_plane_y_range
        )
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.approach_distance = approach_distance
        self.is_eval = is_eval
        self.use_pose_estimation = use_pose_estimation

        self.insertion_axis = _normalize(np.array([-0.5, 0.0, -0.7071]))
        self.plane_axis_1, self.plane_axis_2 = _orthogonal_basis(self.insertion_axis)

        self.z_force_target = self.env_config.insertion_forcetorque
        self.z_force_k = self.env_config.ft_controller_k
        self.axis_force_clip = 0.003
        self.i_term_clip = 0.001
        self.goal_axis_termination_threshold = 0.008

    def _sample_grasp_delta(self) -> np.ndarray:
        return np.array(
            [
                np.random.uniform(*self.grasp_randomisation_x_range),
                0.0,
                np.random.uniform(*self.grasp_randomisation_z_range),
            ]
        )

    def _sample_goal_plane_offset(self) -> np.ndarray:
        offset = np.zeros(2)
        while np.linalg.norm(offset) < 0.001:
            offset = np.array(
                [
                    np.random.uniform(*self.goal_position_randomisation_plane_x_range),
                    np.random.uniform(*self.goal_position_randomisation_plane_y_range),
                ]
            )
        return offset[0] * self.plane_axis_1 + offset[1] * self.plane_axis_2

    def _project_to_plane(self, delta: np.ndarray) -> np.ndarray:
        plane_coords = np.array(
            [
                float(np.dot(delta, self.plane_axis_1)),
                float(np.dot(delta, self.plane_axis_2)),
            ]
        )
        return plane_coords[0] * self.plane_axis_1 + plane_coords[1] * self.plane_axis_2

    def _plane_error_to_action(self, obs, target_position: np.ndarray) -> np.ndarray:
        if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) >= 0.001:
            return np.zeros(3)

        error = target_position - obs["observation.state.cartesian"][:3]
        plane_error = self._project_to_plane(error)
        plane_coords = np.array(
            [
                float(np.dot(plane_error, self.plane_axis_1)),
                float(np.dot(plane_error, self.plane_axis_2)),
            ]
        )
        plane_coords = np.clip(plane_coords, -self.i_term_clip, self.i_term_clip)
        return plane_coords[0] * self.plane_axis_1 + plane_coords[1] * self.plane_axis_2

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
        is_rotated=False,
    ):
        relative_pose = (
            np.zeros(3)
            if relative_pose_euler is None
            else np.asarray(relative_pose_euler, dtype=float)
        )
        target = np.concatenate((np.asarray(position, dtype=float), relative_pose))
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        while (
            np.any(
                np.abs(
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                > 0.002
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))

        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                i_term_clip = (
                    self.i_term_clip
                    if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
                    else (
                        (
                            0.002
                            - np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                        )
                        / 0.002
                        + 1
                    )
                    * self.i_term_clip
                )
                if np.any(np.abs(controller_error) > i_term_clip):
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -i_term_clip,
                                    i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self.env.step(np.zeros(6))

        return obs

    def go_delta(
        self,
        current_obs,
        delta,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
    ):
        relative_pose = (
            np.zeros(3)
            if relative_pose_euler is None
            else np.asarray(relative_pose_euler, dtype=float)
        )
        delta = np.asarray(delta, dtype=float)
        target = delta + current_obs["observation.state.target"][:3]
        obs, *_ = self.env.step(np.concatenate((delta, relative_pose)))

        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))

        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                if np.any(np.abs(controller_error) > self.i_term_clip):
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -self.i_term_clip,
                                    self.i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self.env.step(np.zeros(6))

        return obs

    def z_force_controller_daxis(self, obs):
        z_force_error = (
            self.z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
        )
        axis_impedance_error = float(
            np.dot(
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3],
                self.insertion_axis,
            )
        )
        if z_force_error > 0 and axis_impedance_error < self.axis_force_clip:
            return min(
                z_force_error / self.z_force_k,
                self.axis_force_clip - axis_impedance_error,
            ), z_force_error

        if z_force_error < 0 and axis_impedance_error > -self.axis_force_clip:
            return max(
                z_force_error / self.z_force_k,
                -self.axis_force_clip - axis_impedance_error,
            ), z_force_error

        return 0.0, z_force_error

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            self.obs, *_ = self.env.step(np.zeros(6))
            self.obs = self.go_delta(
                self.obs,
                self.env_config.relative_motion_after_rl_train[:3],
                self.env_config.relative_motion_after_rl_train[3:],
            )
            for pose, res in self.env_config.waypoints_after_rl_train:
                self.obs = self.go_to_waypoint(
                    self.obs,
                    np.asarray(pose[:3], dtype=float),
                    pose[3:],
                    distance_err=res,
                )
            self.obs = self.go_to_waypoint(
                self.obs,
                np.asarray(self.env_config.dropoff_point, dtype=float),
            )
            self.env.unwrapped.gripper.set_target(0.8)  # type: ignore
            time.sleep(2.0)
        else:
            print("homing first time...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.first_home_config
            )
            self.first_reset = False

        print("homing...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        self.target_grasp_position = np.copy(self.grasp_position_ground_truth)
        self.target_grasp_position += self._sample_grasp_delta()

        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.target_grasp_position,
            distance_err=0.0002,
            is_via=False,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(self.env_config.gripper_grasp_position)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )

        print("Picking up...")
        for motion in self.env_config.relative_motions_after_grasp:
            self.obs = self.go_delta(
                self.obs,
                np.asarray(motion[:3], dtype=float),
                motion[3:].tolist(),
                is_via=False,
            )

        self.goal_plane_offset = self._sample_goal_plane_offset()
        self.goal_position = (
            self.goal_position_ground_truth
            + self.reset_grasp_delta
            + self.goal_plane_offset
        )
        self.start_position = (
            self.goal_position - self.insertion_axis * self.approach_distance
        )

        print("Moving to start position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.start_position,
            distance_err=0.0002,
            is_via=False,
        )

        print("Establishing contact...")
        self.obs, *_ = self.env.step(np.zeros(6))
        self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
        axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)
        while abs(axis_force_error) > 0.1:
            plane_delta = self._plane_error_to_action(self.obs, self.start_position)
            action_delta = plane_delta + -axis_step * self.insertion_axis
            self.obs, *_ = self.env.step(np.concatenate((action_delta, np.zeros(3))))
            axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.grasped.delta"] = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"] = self.reset_grasp_delta
        reset_info["reset.goal_position.offset"] = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )

        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        axis_step, _ = self.z_force_controller_daxis(self.obs)
        action_delta = (
            action[0] * self.plane_axis_1 + action[1] * self.plane_axis_2
        ) + -axis_step * self.insertion_axis

        current_pos = self.obs["observation.state.cartesian"][:3]
        goal_error = self.goal_position - current_pos
        plane_error = self._project_to_plane(goal_error)
        if np.linalg.norm(plane_error) > self.safety_box_radius:
            plane_coords = np.array(
                [
                    float(np.dot(plane_error, self.plane_axis_1)),
                    float(np.dot(plane_error, self.plane_axis_2)),
                ]
            )
            norm = float(np.linalg.norm(plane_coords))
            correction = plane_coords * self.safety_box_step_size / norm
            action_delta = (
                correction[0] * self.plane_axis_1
                + correction[1] * self.plane_axis_2
                + -axis_step * self.insertion_axis
            )

        self.obs, reward, terminated, truncated, info = self.env.step(
            np.concatenate((action_delta, np.zeros(3)))
        )
        self.obs = self.add_perfect_action_to_obs(self.obs)

        goal_axis_error = float(
            abs(
                np.dot(
                    self.goal_position - self.obs["observation.state.cartesian"][:3],
                    self.insertion_axis,
                )
            )
        )
        # if goal_axis_error > self.goal_axis_termination_threshold:
        #     terminated = True

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            print("[InsertionWrapperBox] Step limit reached, truncating episode.")
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + self.reset_grasp_delta
        ) - obs["observation.state.cartesian"][:3]
        return obs


class InsertionWrapperBoxRotZ(InsertionWrapperBox):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: ShelfBoxConfig,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(-0.002, 0.002),
        goal_position_randomisation_plane_range=(-0.0028, 0.0028),
        safety_box_radius=0.004,
        safety_box_step_size=0.0004,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        approach_distance=0.02,
        initial_rotation_randomisation_range=np.deg2rad(30.0),
        minimal_start_goal_angle=np.deg2rad(6.0),
        is_eval=False,
        use_pose_estimation=False,
    ):
        super().__init__(
            env,
            alg_config=alg_config,
            env_config=env_config,
            grasp_randomisation_x_range=grasp_randomisation_x_range,
            grasp_randomisation_z_range=grasp_randomisation_z_range,
            goal_position_randomisation_plane_range=goal_position_randomisation_plane_range,
            safety_box_radius=safety_box_radius,
            safety_box_step_size=safety_box_step_size,
            step_limit=step_limit,
            minimal_start_goal_distance=minimal_start_goal_distance,
            approach_distance=approach_distance,
            is_eval=is_eval,
            use_pose_estimation=use_pose_estimation,
        )
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))
        self.initial_rotation_randomisation_range = initial_rotation_randomisation_range
        self.minimal_start_goal_angle = minimal_start_goal_angle
        self.goal_rotation_z = 0.0
        self.start_rotation_z = 0.0
        self.current_rotation_z = 0.0

    def _compose_tcp_z_rotation(
        self, current_euler_xyz: np.ndarray, angle: float
    ) -> np.ndarray:
        current_rotation = _rotation_matrix_from_euler_xyz(current_euler_xyz)
        tcp_rotation = _rotation_matrix_from_rotvec(np.array([0.0, 0.0, angle]))
        target_rotation = current_rotation @ tcp_rotation
        return _euler_xyz_from_rotation_matrix(target_rotation)

    def _apply_tcp_z_rotation(self, obs, angle: float):
        if abs(angle) < 1e-8:
            return obs

        current_euler_xyz = np.asarray(
            obs["observation.state.cartesian"][3:], dtype=float
        )
        target_euler_xyz = self._compose_tcp_z_rotation(current_euler_xyz, angle)
        obs, *_ = self.env.step(
            np.concatenate((np.zeros(3), target_euler_xyz - current_euler_xyz))
        )
        while (
            np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
            or np.linalg.norm(obs["observation.velocity.angular"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            self.obs, *_ = self.env.step(np.zeros(6))
            self.obs = self.go_delta(
                self.obs,
                self.env_config.relative_motion_after_rl_train[:3],
                self.env_config.relative_motion_after_rl_train[3:],
            )
            for pose, res in self.env_config.waypoints_after_rl_train:
                self.obs = self.go_to_waypoint(
                    self.obs,
                    np.asarray(pose[:3], dtype=float),
                    pose[3:],
                    distance_err=res,
                )
            self.obs = self.go_to_waypoint(
                self.obs,
                np.asarray(self.env_config.dropoff_point, dtype=float),
            )
            self.env.unwrapped.gripper.set_target(0.8)  # type: ignore
            time.sleep(2.0)
        else:
            print("homing first time...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.first_home_config
            )
            self.first_reset = False

        print("homing...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        self.target_grasp_position = np.copy(self.grasp_position_ground_truth)
        self.target_grasp_position += self._sample_grasp_delta()

        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.target_grasp_position,
            distance_err=0.0002,
            is_via=False,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(self.env_config.gripper_grasp_position)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )

        print("Picking up...")
        for motion in self.env_config.relative_motions_after_grasp:
            self.obs = self.go_delta(
                self.obs,
                np.asarray(motion[:3], dtype=float),
                motion[3:].tolist(),
                is_via=False,
            )

        self.goal_plane_offset = self._sample_goal_plane_offset()
        self.goal_position = (
            self.goal_position_ground_truth
            + self.reset_grasp_delta
            + self.goal_plane_offset
        )

        self.goal_rotation_z = 0.0
        self.start_rotation_z = 0.0
        while abs(self.start_rotation_z) < self.minimal_start_goal_angle:
            self.start_rotation_z = np.random.uniform(
                -self.initial_rotation_randomisation_range,
                self.initial_rotation_randomisation_range,
            )
        if self.is_eval:
            self.start_rotation_z = 0.0
        self.current_rotation_z = self.start_rotation_z

        print("Applying initial tcp-z rotation...")
        self.obs = self._apply_tcp_z_rotation(self.obs, self.start_rotation_z)

        self.start_position = (
            self.goal_position - self.insertion_axis * self.approach_distance
        )

        print("Moving to start position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.start_position,
            distance_err=0.0002,
            is_via=False,
        )

        print("Establishing contact...")
        self.obs, *_ = self.env.step(np.zeros(6))
        self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
        axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)
        while abs(axis_force_error) > 0.1:
            plane_delta = self._plane_error_to_action(self.obs, self.start_position)
            action_delta = plane_delta + -axis_step * self.insertion_axis
            self.obs, *_ = self.env.step(np.concatenate((action_delta, np.zeros(3))))
            axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.grasped.delta"] = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"] = self.reset_grasp_delta
        reset_info["reset.goal_position.offset"] = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        reset_info["reset.rotation.initial"] = self.start_rotation_z
        reset_info["reset.rotation.goal"] = self.goal_rotation_z

        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.obs["observation.perfect_rotation"] = np.array(
            [self.goal_rotation_z - self.current_rotation_z]
        )
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        axis_step, _ = self.z_force_controller_daxis(self.obs)

        translation_delta = (
            action[0] * self.plane_axis_1 + action[1] * self.plane_axis_2
        )
        current_euler_xyz = np.asarray(
            self.obs["observation.state.cartesian"][3:], dtype=float
        )
        target_euler_xyz = self._compose_tcp_z_rotation(current_euler_xyz, action[2])
        rotation_delta = target_euler_xyz - current_euler_xyz
        action_delta = translation_delta + -axis_step * self.insertion_axis

        current_pos = self.obs["observation.state.cartesian"][:3]
        goal_error = self.goal_position - current_pos
        plane_error = self._project_to_plane(goal_error)
        if np.linalg.norm(plane_error) > self.safety_box_radius:
            plane_coords = np.array(
                [
                    float(np.dot(plane_error, self.plane_axis_1)),
                    float(np.dot(plane_error, self.plane_axis_2)),
                ]
            )
            norm = float(np.linalg.norm(plane_coords))
            correction = plane_coords * self.safety_box_step_size / norm
            action_delta = (
                correction[0] * self.plane_axis_1
                + correction[1] * self.plane_axis_2
                + -axis_step * self.insertion_axis
            )

        self.obs, reward, terminated, truncated, info = self.env.step(
            np.concatenate((action_delta, rotation_delta))
        )
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.current_rotation_z += float(action[2])
        self.obs["observation.perfect_rotation"] = np.array(
            [self.goal_rotation_z - self.current_rotation_z]
        )

        goal_axis_error = float(
            abs(
                np.dot(
                    self.goal_position - self.obs["observation.state.cartesian"][:3],
                    self.insertion_axis,
                )
            )
        )
        if goal_axis_error > self.goal_axis_termination_threshold:
            terminated = True

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info
