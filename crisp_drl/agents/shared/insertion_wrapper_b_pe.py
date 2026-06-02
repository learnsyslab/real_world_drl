import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Wrapper, spaces

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig
from crisp_drl.agents.shared.insertion_wrapper_b import InsertionWrapperBox
from crisp_drl.envs.sam3_pe import PoseEstimationHelper


def r_x(phi: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi), np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_y(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), 0.0, np.sin(phi)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi), 0.0, np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_z(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def euler_xyz_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


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


def _ee_pose_to_world_transform(
    ee_pose: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    ee_pose_arr = np.asarray(ee_pose, dtype=np.float32)
    if ee_pose_arr.shape == (4, 4):
        return ee_pose_arr[:3, :3], ee_pose_arr[:3, 3]

    if ee_pose_arr.ndim != 1 or ee_pose_arr.size < 6:
        raise ValueError(
            "ee_pose must be a 6D cartesian state or a 4x4 transform matrix"
        )

    w_T_t = ee_pose_arr[:3]
    roll, pitch, yaw = ee_pose_arr[3:6]
    w_R_t = euler_xyz_to_rot_matrix(roll, pitch, yaw)
    return w_R_t, w_T_t


def estimate_barcode_poses(
    pose_helper: PoseEstimationHelper, obs: dict[str, np.ndarray], env
) -> list[np.ndarray]:
    for _ in range(5):
        try:
            return pose_helper.estimate_barcodes(
                (
                    obs["observation.images.wrist_camera"],
                    obs["observation.images.wrist_depth_camera"],
                ),
                obs["observation.state.cartesian"],
            )
        except RuntimeError as e:
            print(f"Barcode estimation failed with error: {e}. Retrying...")
            time.sleep(0.5)
            obs, *_ = env.step(
                [0.005, 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # small x-motion to change viewpoint

    raise RuntimeError("Barcode estimation failed after 5 attempts")


def estimate_barcode_poses_tcp(
    pose_helper: PoseEstimationHelper, obs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    return pose_helper.estimate_barcodes_tcp(
        (
            obs["observation.images.wrist_camera"],
            obs["observation.images.wrist_depth_camera"],
        )
    )


def barcode_offset_position(
    pose_world: np.ndarray,
    offset_local_m: np.ndarray,
    local: bool,
) -> np.ndarray:
    if local:
        return pose_world[:3, 3] + pose_world[:3, :3] @ offset_local_m
    return pose_world[:3, 3] + offset_local_m


def object_frame_z_rotation_mod_180_rad(pose_world: np.ndarray) -> float:
    yaw_rad = float(np.arctan2(pose_world[1, 0], pose_world[0, 0]))
    return wrap_angle_to_half_turn(yaw_rad)


def wrap_angle_to_half_turn(angle_rad: float) -> float:
    return float((angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def move_to_target(
    env,
    current_obs: dict[str, np.ndarray],
    target_xyz: np.ndarray,
    target_euler_rad: np.ndarray | None = None,
    target_z_rotation_rad: float | None = None,
    tolerance: float = 0.001,
    orientation_tolerance_rad: float = 0.01,
    settle_steps: int = 3,
    max_settle_iterations: int = 250,
) -> dict[str, np.ndarray]:
    action_dim = 6
    action = np.zeros(action_dim, dtype=np.float32)
    current_state = current_obs["observation.state.cartesian"]
    action[:3] = target_xyz - current_state[:3]

    if target_euler_rad is not None and action_dim > 5:
        action[3:6] = (
            np.asarray(target_euler_rad, dtype=np.float32) - current_state[3:6]
        )
    elif target_z_rotation_rad is not None and action_dim > 5:
        action[5] = wrap_angle_to_half_turn(
            target_z_rotation_rad - float(current_state[5])
        )

    obs, *_ = env.step(action)
    last_i_term = False

    for _ in range(max_settle_iterations):
        current_xyz = obs["observation.state.cartesian"][:3]
        current_euler = obs["observation.state.cartesian"][3:6]
        delta = target_xyz - current_xyz
        position_error = float(np.linalg.norm(delta))
        velocity_error = float(
            np.linalg.norm(obs["observation.velocity.cartesian"][:3])
        )
        orientation_error = 0.0

        if target_euler_rad is not None:
            orientation_error = float(
                np.linalg.norm(
                    np.asarray(target_euler_rad, dtype=np.float32) - current_euler
                )
            )
        elif target_z_rotation_rad is not None:
            orientation_error = float(
                abs(
                    wrap_angle_to_half_turn(
                        target_z_rotation_rad - float(current_euler[2])
                    )
                )
            )

        if (
            position_error <= tolerance
            and velocity_error <= tolerance / 2
            and orientation_error <= orientation_tolerance_rad
        ):
            break

        if (
            velocity_error <= tolerance / 2
            and (
                position_error > tolerance
                or orientation_error > orientation_tolerance_rad
            )
            and not last_i_term
        ):
            action = np.zeros(action_dim, dtype=np.float32)
            action[:3] = delta
            if target_euler_rad is not None and action_dim > 5:
                action[3:6] = (
                    np.asarray(target_euler_rad, dtype=np.float32) - current_euler
                )
            elif target_z_rotation_rad is not None and action_dim > 5:
                action[5] = wrap_angle_to_half_turn(
                    target_z_rotation_rad - float(current_euler[2])
                )
            obs, *_ = env.step(action)
            last_i_term = True
        else:
            obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))
            last_i_term = False

    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    return obs


def descend_along_axis_until_force(
    env,
    obs: dict[str, np.ndarray],
    axis: np.ndarray,
    target_force_delta: float = 1.0,
    step_m: float = 0.001,
    max_contact_force: float = 3.0,
    correction_gain: float = 0.5,
    max_correction_step: float = 0.0005,
    max_iterations: int = 2000,
) -> tuple[dict[str, np.ndarray], bool]:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / float(np.linalg.norm(axis))
    start_pos = obs["observation.state.cartesian"][:3].copy()
    initial_z = float(obs["observation.state.sensors_bota_ft_sensor"][2])
    print(
        f"Initial z force: {initial_z:.3f} N, target delta: {target_force_delta:.3f} N"
    )
    traveled = 0.0
    action_dim = 6

    for _ in range(max_iterations):
        desired_traveled = traveled + step_m
        desired_pos = start_pos + axis * desired_traveled

        current_pos = obs["observation.state.cartesian"][:3]
        error = desired_pos - current_pos

        # correction perpendicular to axis
        error_along = np.dot(error, axis) * axis
        perp_error = error - error_along
        if np.linalg.norm(perp_error) > 0.0:
            corr_step = correction_gain * perp_error
            corr_norm = float(np.linalg.norm(corr_step))
            if corr_norm > max_correction_step:
                corr_step = corr_step / corr_norm * max_correction_step
        else:
            corr_step = np.zeros(3, dtype=np.float32)

        step_vec = axis * step_m + corr_step
        action = np.zeros(action_dim, dtype=np.float32)
        action[:3] = step_vec
        obs, *_ = env.step(action)
        traveled = float(
            np.linalg.norm((obs["observation.state.cartesian"][:3] - start_pos))
        )

        current_z = float(obs["observation.state.sensors_bota_ft_sensor"][2])
        # print(
        #     f"Initial z force: {initial_z:.3f} N, current z force: {current_z:.3f} N, target delta: {target_force_delta:.3f} N, current delta: {current_z - initial_z:.3f} N, traveled: {traveled:.3f} m"
        # )
        if abs(current_z - initial_z) >= abs(target_force_delta):
            return obs, True

        # safety: if measured z force magnitude exceeds max_contact_force, abort
        if abs(current_z) > max_contact_force:
            print(
                f"Aborting descent: z-force {current_z:.3f} N exceeds safety {max_contact_force} N"
            )
            return obs, False

    print("Descent did not reach target force within max iterations")
    return obs, False


class InsertionWrapperBoxPE(InsertionWrapperBox):
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
        step_limit=45,
        minimal_start_goal_distance=0.003,
        approach_distance=0.029,
        is_eval=True,
        use_pose_estimation_for_grasp=True,
        use_pose_estimation_for_goal=True,
        do_pushing_motion_in_eval=True,
    ):
        print(
            "Initializing InsertionWrapperBoxPE with pose estimation for grasp and goal"
        )
        print(f"step_limit: {step_limit}")
        super().__init__(
            env,
            alg_config=alg_config,
            env_config=env_config,
            grasp_randomisation_x_range=grasp_randomisation_x_range,
            grasp_randomisation_z_range=grasp_randomisation_z_range,
            goal_position_randomisation_plane_range=goal_position_randomisation_plane_range,
            goal_position_randomisation_plane_x_range=goal_position_randomisation_plane_x_range,
            goal_position_randomisation_plane_y_range=goal_position_randomisation_plane_y_range,
            safety_box_radius=safety_box_radius,
            safety_box_step_size=safety_box_step_size,
            step_limit=step_limit,
            minimal_start_goal_distance=minimal_start_goal_distance,
            approach_distance=approach_distance,
            is_eval=is_eval,
            use_pose_estimation=use_pose_estimation_for_grasp
            or use_pose_estimation_for_goal,
        )
        self.home_config = env_config.custom_home_position_pe
        self.use_pose_estimation_for_grasp = use_pose_estimation_for_grasp
        self.use_pose_estimation_for_goal = use_pose_estimation_for_goal
        self.pose_estimation_helper_blue = PoseEstimationHelper(
            full_mask_prompt="blue cardboard box",
            filter_depth_outliers=True,
            width_m=0.026,
            height_m=0.0375,
        )
        self.pose_estimation_helper_yellow = PoseEstimationHelper(
            full_mask_prompt="yellow cardboard box",
            filter_depth_outliers=True,
            width_m=0.03,
            height_m=0.04,
            processor=self.pose_estimation_helper_blue.processor,
        )
        self.goal_position_from_barcode_offset_world = np.asarray(
            env_config.goal_position_from_barcode_offset_world, dtype=float
        )
        self.grasp_position_offset_from_barcode_local = np.asarray(
            env_config.grasp_position_offset_from_barcode_local, dtype=float
        )
        self.do_pushing_motion_in_eval = do_pushing_motion_in_eval
        self.blue_initial_offset_m = np.asarray([0.0, 0.0, 0.12], dtype=float)
        self.blue_above_offset_m = np.asarray([0.0, 0.0, 0.065], dtype=float)
        # self.blue_above_offset_m = np.asarray([0.08, 0.0, 0.2], dtype=float)
        self.yellow_initial_offset_m = np.asarray([0.05, 0.0, 0.1], dtype=float)

    def _sample_grasp_target(self) -> np.ndarray:
        target = np.copy(self.grasp_position_ground_truth)
        target += self._sample_grasp_delta()
        return target

    def _sample_goal_target(self) -> np.ndarray:
        self.goal_plane_offset = self._sample_goal_plane_offset()
        target = np.copy(self.goal_position_ground_truth)
        target += self.goal_plane_offset
        return target

    def _estimate_goal_from_blue(
        self, obs
    ) -> tuple[Any, np.ndarray, dict[str, np.ndarray]]:
        print("[Blue] [Estimating] initial estimate...")
        blue_initial_poses = _estimate_barcode_poses(
            self.pose_estimation_helper_blue, obs, "blue"
        )
        blue_initial_pose = blue_initial_poses[0]
        blue_above_target = _offset_pose_world(
            blue_initial_pose,
            self.blue_initial_offset_m,
            local=False,
        )
        print(f"Moving to blue above-target pose: {blue_above_target}")
        obs = self.go_to_waypoint(
            obs,
            blue_above_target,
            distance_err=0.0002,
            is_via=False,
        )

        blue_above_poses = _estimate_barcode_poses(
            self.pose_estimation_helper_blue, obs, "blue"
        )
        blue_above_pose = blue_above_poses[0]
        goal_position = _offset_pose_world(
            blue_above_pose,
            self.goal_position_from_barcode_offset_world,
            local=False,
        )
        _print_estimation_summary(
            "Goal",
            goal_position,
            self.goal_position_ground_truth,
        )
        return (
            obs,
            goal_position,
            {
                "reset.pose_estimation.blue_initial": blue_initial_pose,
                "reset.pose_estimation.blue_above": blue_above_pose,
            },
        )

    def _estimate_grasp_from_yellow(
        self, obs
    ) -> tuple[Any, np.ndarray, dict[str, np.ndarray]]:
        yellow_initial_poses = _estimate_barcode_poses(
            self.pose_estimation_helper_yellow, obs, "yellow"
        )
        yellow_initial_pose = yellow_initial_poses[0]
        yellow_above_target = _offset_pose_world(
            yellow_initial_pose,
            self.yellow_initial_offset_m,
            local=True,
        )
        print(f"Moving to yellow above-target pose: {yellow_above_target}")
        obs = self.go_to_waypoint(
            obs,
            yellow_above_target,
            distance_err=0.0002,
            is_via=False,
        )

        yellow_above_poses = _estimate_barcode_poses(
            self.pose_estimation_helper_yellow, obs, "yellow"
        )
        yellow_above_pose = yellow_above_poses[0]
        target_grasp_position = _offset_pose_world(
            yellow_above_pose,
            self.grasp_position_offset_from_barcode_local,
            local=True,
        )
        _print_estimation_summary(
            "Grasp",
            target_grasp_position,
            self.grasp_position_ground_truth,
        )
        print(f"Moving to yellow grasp target: {target_grasp_position}")
        obs = self.go_to_waypoint(
            obs,
            target_grasp_position,
            distance_err=0.0002,
            is_via=False,
        )

        yellow_final_poses = _estimate_barcode_poses(
            self.pose_estimation_helper_yellow, obs, "yellow"
        )
        yellow_final_pose = yellow_final_poses[0]
        return (
            obs,
            target_grasp_position,
            {
                "reset.pose_estimation.yellow_initial": yellow_initial_pose,
                "reset.pose_estimation.yellow_above": yellow_above_pose,
                "reset.pose_estimation.yellow_final": yellow_final_pose,
            },
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            self.obs, *_ = self.env.step(np.zeros(6))
            if self.do_pushing_motion_in_eval and self.is_eval:
                self.env.unwrapped.gripper.set_target(1.0)  # type: ignore
                time.sleep(2.0)
                # go back 65mm along insertion axis, set gripper to 0.4, rotate by 30° arond insertion axis,
                # push until goal position + 60mm along insertion axis is reached or contact force exceeds 3N, then go to start position
                self.obs = self.go_delta(
                    self.obs,
                    rotate_by_30_y(np.array([0.037, 0.0, 0.065], dtype=float)),  # 0.025
                )
                self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
                time.sleep(2.0)
                # self.obs = self.go_delta(
                #     self.obs,
                #     np.zeros(3, dtype=float),
                #     _euler_xyz_from_rotation_matrix(
                #         _rotation_matrix_from_rotvec(
                #             np.array([1.0 / np.sqrt(3), 0.0, np.sqrt(2.0 / 3.0)])
                #             * np.deg2rad(40.0)
                #         )
                #     ),
                # )
                print("  Descending along axis to establish contact force delta...")
                self.obs, descended = descend_along_axis_until_force(
                    self.env,
                    self.obs,
                    axis=self.insertion_axis,
                    target_force_delta=4.1,
                    step_m=0.0005,
                    max_contact_force=20.0,
                    correction_gain=0.8,
                    max_correction_step=0.0007,
                    max_iterations=2000,
                )
            else:
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
            self.env.save_obs_and_reset_ft_measurement_wrapper()  # type: ignore
        else:
            print("homing first time...")
            self.env.unwrapped.home(home_config=self.first_home_config)  # type: ignore
            self.first_reset = False

        if not self.is_eval:
            print("homing...")
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        else:
            print("homing eval  ...")
            self.env.unwrapped.home(home_config=self.env_config.custom_home_position_pe)  # type: ignore
        if self.do_pushing_motion_in_eval and self.is_eval:
            time.sleep(4.0)  # wait for env reset

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self.env.tare_ft_measurement_wrapper(self.obs)  # type: ignore

        if self.use_pose_estimation_for_goal:
            # Initial blue estimate and target position for next estimate
            blue_initial_pose = estimate_barcode_poses(
                self.pose_estimation_helper_blue, self.obs, self.env
            )[0]
            print("[Blue] [Estimated] initial:", blue_initial_pose[:3, 3])
            blue_above_target = barcode_offset_position(
                blue_initial_pose, self.blue_initial_offset_m, local=False
            )

        if self.use_pose_estimation_for_grasp:
            # Initial yellow estimate and target pose for next estimate
            yellow_initial_pose = estimate_barcode_poses(
                self.pose_estimation_helper_yellow, self.obs, self.env
            )[0]
            yellow_above_target = barcode_offset_position(
                yellow_initial_pose, self.yellow_initial_offset_m, local=True
            )
            yellow_above_rotation_rad = object_frame_z_rotation_mod_180_rad(
                yellow_initial_pose
            )

        if self.use_pose_estimation_for_goal:
            # Go to above pose for blue and estimate final target pose
            print(f"[Blue] [Move] target: above {blue_above_target[:3]}")  # pyright: ignore[reportPossiblyUnboundVariable]
            self.obs = move_to_target(self.env, self.obs, blue_above_target)  # pyright: ignore[reportPossiblyUnboundVariable]
            blue_above_pose = estimate_barcode_poses(
                self.pose_estimation_helper_blue, self.obs, self.env
            )[0]
            print(f"[Blue] [Estimated] final: {blue_above_pose[:3, 3]}")
            if True:
                blue_final_target = barcode_offset_position(
                    blue_above_pose,
                    self.blue_above_offset_m,
                    local=False,
                )

                # Go to final pose and do final estimation
                print(f"[Blue] [Move] target: final {blue_final_target[:3]}")
                self.obs = move_to_target(self.env, self.obs, blue_final_target)
                blue_estimated_pose = estimate_barcode_poses(
                    self.pose_estimation_helper_blue, self.obs, self.env
                )[0]
                print(f"[Blue] [Estimated] goal: {blue_estimated_pose[:3, 3]}")
                ######## Estimated goal pose without grasp offset (world frame)
                estimated_goal_pose_perfect_grasp = (
                    blue_estimated_pose[:3, 3]
                    + self.env_config.goal_position_from_barcode_offset_world
                )
            else:
                estimated_goal_pose_perfect_grasp = (
                    blue_above_pose[:3, 3]
                    + self.env_config.goal_position_from_barcode_offset_world
                )
        else:
            estimated_goal_pose_perfect_grasp = self.goal_position_ground_truth.copy()

        if self.use_pose_estimation_for_grasp:
            # Go to yellow above pose and estimate grasp pose
            print(
                f"[Yellow] [Move] target: above {yellow_above_target[:3]} with rotation {yellow_above_rotation_rad:.2f} rad"  # pyright: ignore[reportPossiblyUnboundVariable]
            )
            self.obs = move_to_target(
                self.env,
                self.obs,
                yellow_above_target,  # pyright: ignore[reportPossiblyUnboundVariable]
                target_z_rotation_rad=yellow_above_rotation_rad,  # pyright: ignore[reportPossiblyUnboundVariable]
            )
            yellow_above_pose = estimate_barcode_poses(
                self.pose_estimation_helper_yellow, self.obs, self.env
            )[0]
            print(f"[Yellow] [Estimated] above: {yellow_above_pose[:3, 3]}")
            yellow_grasp_target = barcode_offset_position(
                yellow_above_pose,
                self.env_config.grasp_position_offset_from_barcode_local,
                local=True,
            )
            yellow_grasp_rotation_rad = object_frame_z_rotation_mod_180_rad(
                yellow_above_pose
            )

            # Move to first estimate of the grasp pose and re-estimate
            print(
                f"[Yellow] [Move] target: grasp {yellow_grasp_target[:3]} with rotation {yellow_grasp_rotation_rad:.2f} rad"
            )
            self.obs = move_to_target(
                self.env,
                self.obs,
                yellow_grasp_target,
                target_z_rotation_rad=yellow_grasp_rotation_rad,
            )
            yellow_final_pose = estimate_barcode_poses(
                self.pose_estimation_helper_yellow, self.obs, self.env
            )[0]
            print(f"[Yellow] [Estimated] final: {yellow_final_pose[:3, 3]}")
            yellow_grasp_target = barcode_offset_position(
                yellow_final_pose,
                self.env_config.grasp_position_offset_from_barcode_local,
                local=True,
            )
            yellow_grasp_rotation_rad = object_frame_z_rotation_mod_180_rad(
                yellow_final_pose
            )

            print(
                f"[Yellow] [Move] target: final {yellow_grasp_target[:3]} with rotation {yellow_grasp_rotation_rad:.2f} rad"
            )
            # Move to the final estimate of the grasp pose
            self.obs = move_to_target(
                self.env,
                self.obs,
                yellow_grasp_target,
                target_z_rotation_rad=yellow_grasp_rotation_rad,
            )
        else:
            # randomize ground truth and go there
            target_grasp_position = self._sample_grasp_target()
            self.obs = move_to_target(
                self.env,
                self.obs,
                target_grasp_position,
            )

        print("Grasping...")
        self.env.unwrapped.gripper.set_target(self.env_config.gripper_grasp_position)  # type: ignore
        time.sleep(2.0)

        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )
        if not self.use_pose_estimation_for_grasp:
            self.reset_grasp_delta = (
                self.actual_grasp_position - self.grasp_position_ground_truth
            )
        else:
            self.reset_grasp_delta = (
                self.actual_grasp_position - yellow_grasp_target  # pyright: ignore[reportPossiblyUnboundVariable]
            )

        lift_target = self.obs["observation.state.cartesian"][:3] + np.array(
            [0.0, 0.0, 0.1], dtype=np.float32
        )
        print(f"  Moving up 10 cm: {lift_target}")
        self.obs = move_to_target(self.env, self.obs, lift_target)

        if self.use_pose_estimation_for_grasp:
            # Undo z-rotation, estimate grasp in tcp frame, compare to GT grasp in tcp frame, adjust goal position
            current_state = self.obs["observation.state.cartesian"]
            current_xyz = current_state[:3].copy()
            desired_z = (
                float(current_state[5]) - yellow_grasp_rotation_rad  # pyright: ignore[reportPossiblyUnboundVariable]
            )
            self.obs = move_to_target(
                self.env, self.obs, current_xyz, target_z_rotation_rad=desired_z
            )
            yellow_grasped_pose_tcp = estimate_barcode_poses_tcp(
                self.pose_estimation_helper_yellow, self.obs
            )[0]
            grasp_offset_tcp = (
                yellow_grasped_pose_tcp[:3, 3]
                - self.env_config.demo_yellow_grasped_pose_tcp
            )
            grasp_offset_tcp[1] = 0.0  # gripper centers object in y-direction
            w_R_t, w_T_t = _ee_pose_to_world_transform(
                self.obs["observation.state.cartesian"]
            )
            grasp_offset_world = w_R_t @ grasp_offset_tcp
            grasp_offset_insertion_plane = rotate_by_30_y(grasp_offset_world)

            self.obs = self.go_delta(
                self.obs,
                np.zeros(3, dtype=float),
                np.array([0.0, np.deg2rad(30), 0.0], dtype=float),
                is_via=False,
            )
        else:
            for motion in self.env_config.relative_motions_after_grasp[1:]:
                self.obs = self.go_delta(
                    self.obs,
                    np.asarray(motion[:3], dtype=float),
                    motion[3:].tolist(),
                    is_via=False,
                )
            grasp_offset_insertion_plane = rotate_by_30_y(self.reset_grasp_delta)

        self.estimated_goal_pose = (
            estimated_goal_pose_perfect_grasp + grasp_offset_insertion_plane
        )

        # Sequence:
        # 1) Estimate blue and yellow coarse
        # 2) Move to blue above, estimate blue (optionally go closer and estimate final)
        # 3) Move to yellow above (including rotation), estimate yellow, go to final offset (incl rot), estimate,
        #       go to grasp pose (=final), grasp, move up, establish z-rotation,
        #       estimate grasp delta in world frame, compute world-coordinates rotated goal-position offset
        # 4) Rotate y-axis, go to goal pos (above), go in contact carefully, reset done

        self.start_position = (
            self.estimated_goal_pose - self.insertion_axis * self.approach_distance
        )

        print("Moving to start position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.start_position,
            distance_err=0.0002,
            is_via=False,
        )

        self.obs, *_ = self.env.step(np.zeros(6))
        self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
        self.obs, *_ = self.env.step(np.zeros(6))
        print("  Descending along axis to establish contact force delta...")
        self.obs, descended = descend_along_axis_until_force(
            self.env,
            self.obs,
            axis=self.insertion_axis,
            target_force_delta=self.z_force_target,
            step_m=0.0005,
            max_contact_force=20.0,
            correction_gain=0.8,
            max_correction_step=0.0007,
            max_iterations=2000,
        )
        # print("Establishing contact...")
        # self.obs, *_ = self.env.step(np.zeros(6))
        # self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
        # axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)
        # while abs(axis_force_error) > 0.03:
        #     plane_delta = self._plane_error_to_action(self.obs, self.start_position)
        #     action_delta = plane_delta + -axis_step * self.insertion_axis
        #     self.obs, *_ = self.env.step(np.concatenate((action_delta, np.zeros(3))))
        #     axis_step, axis_force_error = self.z_force_controller_daxis(self.obs)

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.grasped.delta"] = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"] = self.reset_grasp_delta

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
        goal_error = self.estimated_goal_pose - current_pos
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

        # goal_axis_error = float(
        #     abs(
        #         np.dot(
        #             self.estimated_goal_pose - self.obs["observation.state.cartesian"][:3],
        #             self.insertion_axis,
        #         )
        #     )
        # )
        # if goal_axis_error > self.goal_axis_termination_threshold:
        #     terminated = True

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            print("[InsertionWrapperBoxPE] Step limit reached, truncating episode.")
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + rotate_by_30_y(self.reset_grasp_delta)
        ) - obs["observation.state.cartesian"][:3]
        return obs
