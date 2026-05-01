import time
from typing import Any, Dict, Optional
from gymnasium import Wrapper, spaces
import numpy as np

from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.envs.pose_estimation_helper import (
    PoseEstimationHelper,
    euler_to_rot_matrix,
)
from crisp_drl.agents.shared.algorithm_config import Config

import contextlib


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def rot_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to XYZ roll-pitch-yaw Euler angles."""
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


class InsertionWrapperSiemens(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: SiemensConfig,
        grasp_randomisation_x_range=(-0.001, 0.001),
        grasp_randomisation_z_range=(-0.001, 0.001),
        safety_box_radius=0.002,
        safety_box_step_size=0.0004,
        step_limit=150,
        minimal_start_goal_distance=0.0015,
        is_eval=False,
        use_pose_estimation=False,
        use_ft_controller: bool = True,
        use_6dof_grasp: bool = False,
    ):
        super().__init__(env)
        self.use_6dof_grasp = use_6dof_grasp
        self.action_space = spaces.Box(
            -np.inf, np.inf, (5,) if self.use_6dof_grasp else (2,)
        )
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.grasp_orientation_ground_truth = np.array(
            env_config.grasp_orientation_ground_truth_euler
        )
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        self.use_pose_estimation = use_pose_estimation
        self.use_ft_controller = use_ft_controller
        self.pose_estimation_helper = (
            PoseEstimationHelper(
                assumed_orientation=alg_config.pose_estimation_assumed_orientation,
                lock_orientation=not self.use_6dof_grasp,
            )
            if use_pose_estimation
            else None
        )
        self.pose_estimation_position_euler = np.array(
            alg_config.demo_goal_pose_estimation_euler
        )
        print("[InsertionWrapper] [__init__] Eval mode:", is_eval)

        self.ft_wrench_target = self.env_config.insertion_forcetorque
        self.x_force_k = self.env_config.ft_controller_k
        self.x_force_clip = 0.003
        self.i_term_clip = 0.0009

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
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        target = np.concatenate((position, relative_pose))
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse
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
            # print(
            #     f"Waiting... (v={np.linalg.norm(obs['observation.velocity.cartesian'][:3])}, err={target[:3] - obs['observation.state.cartesian'][:3]}, controller err={obs['observation.state.target'][:3] - obs['observation.state.cartesian'][:3]})"
            # )
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                # print(err, controller_error)
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
                # print("waiting (fine)...")

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
        target = delta + current_obs["observation.state.target"][:3]
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        obs, *_ = self.env.step(
            np.concatenate((delta, relative_pose))
        )

        # coarse
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
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

    def yz_i_controller_dyz(self, obs, target):
        err = target[1:3] - obs["observation.state.cartesian"][1:3]
        controller_error = (
            obs["observation.state.target"][1:3]
            - obs["observation.state.cartesian"][1:3]
        )
        if np.any(np.abs(controller_error) > self.i_term_clip):
            return np.zeros(2)
        else:
            return (
                np.clip(
                    err + controller_error,
                    -self.i_term_clip,
                    self.i_term_clip,
                )
                - controller_error
            )

    def x_torque_controller_dx(self, obs):
        y_torque_sensed = np.sum(
            obs["observation.state.sensors_bota_ft_sensor"][3:5]
        ) / np.sqrt(2)
        y_torque_error = self.ft_wrench_target - y_torque_sensed
        x_force_error = y_torque_error / self.env_config.ft_controller_lever_arm
        x_impedance_error = (
            obs["observation.state.target"][0] - obs["observation.state.cartesian"][0]
        )
        if x_force_error > 0 and x_impedance_error < self.x_force_clip:
            dx = -min(
                x_force_error / self.x_force_k, self.x_force_clip - x_impedance_error
            )

        elif x_force_error < 0 and x_impedance_error > -self.x_force_clip:
            dx = -max(
                x_force_error / self.x_force_k, -self.x_force_clip - x_impedance_error
            )
        else:
            dx = 0.0
        # print(
        #     f"[X torque controller] target: {self.ft_wrench_target}, sensed: {y_torque_sensed:.3f}, force_error: {x_force_error:.3f}, impedance_error: {x_impedance_error:.5f}, dx: {dx:.5f}"
        # )
        return dx, x_force_error

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            # lift up
            self.obs, *_ = self.env.step(np.zeros(6))  # wait one step
            self.obs = self.go_delta(
                self.obs,
                self.env_config.relative_motion_after_rl_train[:3],
                self.env_config.relative_motion_after_rl_train[3:],
            )
            # waypoints back
            for pose, res in self.env_config.waypoints_after_rl_train:
                self.obs = self.go_to_waypoint(
                    self.obs,
                    pose[:3],
                    pose[3:],
                    distance_err=res,
                )
            # dropoff location
            self.obs = self.go_to_waypoint(
                self.obs,
                self.env_config.dropoff_point,
            )

        else:
            print("homing first time...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.env_config.custom_first_home_position
            )

            self.first_reset = False
        print("homing...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        self.target_grasp_position = np.copy(self.grasp_position_ground_truth)

        grasp_randomisation_x = np.random.uniform(
            self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
        )
        grasp_randomisation_z = np.random.uniform(
            self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
        )

        self.target_grasp_position[0] += grasp_randomisation_x
        self.target_grasp_position[2] += grasp_randomisation_z

        # print("Executing before-grasp motion")
        # self.obs, *_ = self.env.step(self.env_config.relative_motion_before_grasp)

        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.target_grasp_position,
            self.grasp_orientation_ground_truth if self.use_6dof_grasp else None,
            distance_err=0.0002,
            is_via=False,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(1.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_delta(
            self.obs,
            self.env_config.relative_motion_after_grasp[:3],
            self.grasp_orientation_ground_truth if self.use_6dof_grasp else None,
        )

        # compute goal position
        if self.use_pose_estimation and self.pose_estimation_helper is not None:
            # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position
            raise RuntimeError("Not yet implemented")
            print("Moving to pose estimation position...")
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=self.pose_estimation_position_euler[:3],
                fine_resolution=0.0005,
            )
            self.obs, *_ = self.env.step(np.zeros(6))
            self.actual_estimation_position = np.copy(
                self.obs["observation.state.cartesian"]
            )
            pose_estimation_joint_state = np.copy(self.obs["observation.state.joints"])
            print("Estimating pose...")
            lavender_pose, purple_pose = (
                self.pose_estimation_helper.estimate_two_lego_bricks_absolute(
                    self.obs["observation.images.wrist_camera"],
                    self.obs["observation.images.wrist_depth_camera"],
                    self.actual_estimation_position,
                )
            )

            self.goal_position = (
                self.actual_estimation_position[:3]
                + purple_pose[:3, 3]
                - lavender_pose[:3, 3]
            )

        else:
            self.goal_position = np.copy(self.goal_position_ground_truth)
            self.estimated_grasp_delta = np.copy(self.reset_grasp_delta)
            self.estimated_grasp_delta[0] += np.random.uniform(
                *self.grasp_randomisation_x_range
            )
            self.estimated_grasp_delta[2] += np.random.uniform(
                *self.grasp_randomisation_z_range
            )

            self.goal_position += self.estimated_grasp_delta

        if not self.is_eval:
            corrected_goal_gt = self.goal_position_ground_truth + self.reset_grasp_delta
            self.start_position = corrected_goal_gt.copy()
            while np.linalg.norm(
                self.start_position[1:3] - corrected_goal_gt[1:3]
            ) < self.minimal_start_goal_distance or np.any(
                np.clip(
                    self.start_position[1:3] - corrected_goal_gt[1:3],
                    [-0.004, -0.001],
                    [0.004, 0.003],
                )
                != self.start_position[1:3] - corrected_goal_gt[1:3]
            ):
                self.start_position[1:3] = self.goal_position[1:3] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            with printoptions(precision=4):
                print(
                    f"goal true (corr) pos: {self.goal_position_ground_truth + self.reset_grasp_delta}, goal est pos {self.goal_position}, start pos {self.start_position}"
                )
        else:
            self.start_position = self.goal_position.copy()

        for pose, res in self.env_config.waypoints_after_grasp:
            self.obs = self.go_to_waypoint(
                self.obs,
                pose[:3] + self.estimated_grasp_delta,
                pose[3:],
                distance_err=res,
            )

        # lower down slowly until z-force is established in steps of 3mm
        if self.use_ft_controller:
            print("Establishing contact...")
            self.obs, *_ = self.env.step(np.zeros(6))
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            # [s.reset() for s in self.env.unwrapped.sensors]  # pyright: ignore[reportAttributeAccessIssue] # tare ft sensor
            x_step, x_force_error = self.x_torque_controller_dx(self.obs)
            while abs(x_force_error) > 0.1:  # wait until some contact
                # delta_yz = self.yz_i_controller_dyz(self.obs, self.start_position[1:3])
                delta_yz = (
                    np.clip(
                        self.start_position[1:3]
                        - self.obs["observation.state.cartesian"][1:3],
                        -self.i_term_clip,
                        self.i_term_clip,
                    )
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) < 0.001
                    else np.zeros(2)
                )
                self.obs, *_ = self.env.step(
                    np.array([x_step, delta_yz[0], delta_yz[1], 0, 0, 0])
                )
                x_step, x_force_error = self.x_torque_controller_dx(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position

        reset_info["reset.grasped.delta"] = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        if self.use_pose_estimation:
            reset_info["reset.pose_estimation.joint_state"] = (
                pose_estimation_joint_state  # pyright: ignore[reportUndefinedVariable]
            )
            reset_info["reset.pose_estimation.cartesian"] = (
                self.actual_estimation_position
            )  # pyright: ignore[reportPossiblyUnboundVariable]
        with printoptions(precision=4):
            print(
                f"goal true (corr) pos: {self.goal_position_ground_truth + self.reset_grasp_delta}, goal est pos {self.goal_position}, start pos {self.start_position}, reached {self.obs['observation.state.cartesian'][:3]}"
            )
        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
        if self.use_6dof_grasp:
            if action.shape[0] < 5:
                raise ValueError(
                    f"Expected 5D action in 6DoF mode, got shape {action.shape}"
                )
            action = np.array(
                [x_action, action[0], action[1], action[2], action[3], action[4]]
            )
        else:
            action = np.array([x_action, action[0], action[1], 0, 0, 0])

        # apply safety box
        current_pos_yz = self.obs["observation.state.cartesian"][1:3]
        delta_yz = current_pos_yz - self.goal_position[1:3]
        delta_yz_clipped = np.clip(
            delta_yz, -self.safety_box_radius, self.safety_box_radius
        )
        if np.any(delta_yz_clipped != delta_yz):
            correcting_action = delta_yz_clipped - delta_yz
            action[1:3] = correcting_action

        # print(f"[Step] received {action_input}, performing {action}")
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + self.reset_grasp_delta
        ) - obs["observation.state.cartesian"][:3]
        return obs


class InsertionWrapperSiemensPE(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: SiemensConfig,
        safety_box_radius=0.002,
        safety_box_step_size=0.0004,
        step_limit=150,
        use_ft_controller: bool = True,
        use_6dof_grasp: bool = False,
        pose_viz_dir: Optional[str] = None,
    ):
        super().__init__(env)
        self.use_ft_controller = use_ft_controller
        self.use_6dof_grasp = use_6dof_grasp
        self.pose_viz_dir = pose_viz_dir
        self._pe_episode_idx = -1
        self._pose_overlay_renderer = None
        if pose_viz_dir is not None:
            from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer
            self._pose_overlay_renderer = PoseOverlayRenderer(
                camera_info_json_path="camera_parameters/realsense_d405_single.json",
            )
            print(f"[InsertionWrapperSiemensPE] pose viz dir = {pose_viz_dir}")
        self.action_space = spaces.Box(
            -np.inf, np.inf, (5,) if self.use_6dof_grasp else (2,)
        )
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position_pe
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.grasp_orientation_ground_truth = np.array(
            env_config.grasp_orientation_ground_truth_euler
        )
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=np.array([])
        )
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.n_steps = 0
        self.pose_estimation_position_euler = np.array(
            alg_config.demo_goal_pose_estimation_euler
        )

        self.ft_wrench_target = self.env_config.insertion_forcetorque
        self.x_force_k = self.env_config.ft_controller_k
        self.z_force_k = self.env_config.ft_controller_k
        self.x_force_clip = 0.003
        self.z_force_clip = 0.003
        self.i_term_clip = 0.0012

        self.o_T_o_tcpgrasp = self.env_config.demo_w_D_w_o[:3, :3].T @ (
            self.grasp_position_ground_truth - self.env_config.demo_w_D_w_o[:3, 3]
        )
        self.pose_estimation_settle_steps = 1
        self.alignment_clip_angle_rad = np.deg2rad(25.0)
        self.alignment_pitch_bias = np.deg2rad(-3.0)
        print(
            "[InsertionWrapperSiemensPE] use_6dof_grasp=",
            self.use_6dof_grasp,
            "action_dim=",
            self.action_space.shape[0],
        )

    def compute_alignment_rpy(self, world_D_world_obj: np.ndarray) -> np.ndarray:
        world_R_demoobj = self.env_config.demo_w_D_w_o[:3, :3]
        world_R_estiobj = world_D_world_obj[:3, :3]
        est_R_demo = world_R_estiobj @ world_R_demoobj.T

        # In non-6DoF mode, take yaw only (open-loop yaw alignment).
        if not self.use_6dof_grasp:
            yaw_only = np.array(
                [0.0, 0.0, rot_matrix_to_euler_xyz(est_R_demo)[2]]
            )
            yaw_only[2] = float(
                np.clip(
                    yaw_only[2],
                    -self.alignment_clip_angle_rad,
                    self.alignment_clip_angle_rad,
                ) 
            )
            return yaw_only

        # 6DoF: apply hardcoded pitch bias in the demo->estimated relative frame,
        # then clip the *total* rotation magnitude (axis-angle) so the bound is
        # meaningful under ZYX coupling and applies uniformly to all components.
        from scipy.spatial.transform import Rotation as _R

        bias_R = euler_to_rot_matrix(0.0, self.alignment_pitch_bias, 0.0)
        biased = est_R_demo @ bias_R
        rotvec = _R.from_matrix(biased).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.alignment_clip_angle_rad and angle > 0.0:
            rotvec = rotvec * (self.alignment_clip_angle_rad / angle)
            biased = _R.from_rotvec(rotvec).as_matrix()
        return rot_matrix_to_euler_xyz(biased)

    def estimate_world_pose_once(self, return_details: bool = False):
        for _ in range(self.pose_estimation_settle_steps):
            self.obs, *_ = self.env.step(np.zeros(6))
        return self.pose_estimation_helper.estimate_siemens_world_frame_coarse(
            self.obs["observation.images.wrist_camera"],
            self.obs["observation.images.wrist_depth_camera"],
            self.obs["observation.state.cartesian"],
            return_details=return_details,
        )

    def validate_world_pose_transform(
        self, world_D_world_obj: np.ndarray, tag: str
    ) -> dict[str, float | bool]:
        rot = world_D_world_obj[:3, :3]
        finite_ok = bool(np.isfinite(world_D_world_obj).all())
        orth_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
        det_r = float(np.linalg.det(rot))
        valid = finite_ok and abs(det_r - 1.0) < 1e-2 and orth_err < 1e-2
        print(
            f"[InsertionWrapperSiemensPE] {tag} transform valid={valid} "
            f"det={det_r:.6f} orth_err={orth_err:.6e}"
        )
        return {"valid": valid, "det_r": det_r, "orth_err": orth_err}

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
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        target = np.concatenate((position, relative_pose))
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse
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
            # print(
            #     f"Waiting... (v={np.linalg.norm(obs['observation.velocity.cartesian'][:3])}, err={target[:3] - obs['observation.state.cartesian'][:3]}, controller err={obs['observation.state.target'][:3] - obs['observation.state.cartesian'][:3]})"
            # )
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                # print(err, controller_error)
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
                # print("waiting (fine)...")

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
        target = delta + current_obs["observation.state.target"][:3]
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        obs, *_ = self.env.step(
            np.concatenate((delta, relative_pose))
        )

        # coarse
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
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

    def yz_i_controller_dyz(self, obs, target):
        err = target[1:3] - obs["observation.state.cartesian"][1:3]
        controller_error = (
            obs["observation.state.target"][1:3]
            - obs["observation.state.cartesian"][1:3]
        )
        if np.any(np.abs(controller_error) > self.i_term_clip):
            return np.zeros(2)
        else:
            return (
                np.clip(
                    err + controller_error,
                    -self.i_term_clip,
                    self.i_term_clip,
                )
                - controller_error
            )

    def x_torque_controller_dx(self, obs):
        y_torque_sensed = np.sum(
            obs["observation.state.sensors_bota_ft_sensor"][3:5]
        ) / np.sqrt(2)
        y_torque_error = self.ft_wrench_target - y_torque_sensed
        x_force_error = y_torque_error / self.env_config.ft_controller_lever_arm
        x_impedance_error = (
            obs["observation.state.target"][0] - obs["observation.state.cartesian"][0]
        )
        if x_force_error > 0 and x_impedance_error < self.x_force_clip:
            dx = -min(
                x_force_error / self.x_force_k, self.x_force_clip - x_impedance_error
            )

        elif x_force_error < 0 and x_impedance_error > -self.x_force_clip:
            dx = -max(
                x_force_error / self.x_force_k, -self.x_force_clip - x_impedance_error
            )
        else:
            dx = 0.0
        # print(
        #     f"[X torque controller] target: {self.ft_wrench_target}, sensed: {y_torque_sensed:.3f}, force_error: {x_force_error:.3f}, impedance_error: {x_impedance_error:.5f}, dx: {dx:.5f}"
        # )
        return dx, x_force_error

    def z_force_controller_dz(self, obs):
        z_force_target = -6
        z_force_error = (
            z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
        )
        z_impedance_error = (
            obs["observation.state.target"][2] - obs["observation.state.cartesian"][2]
        )
        if z_force_error > 0 and z_impedance_error < self.z_force_clip:
            return min(
                z_force_error / self.z_force_k, self.z_force_clip - z_impedance_error
            ), z_force_error

        elif z_force_error < 0 and z_impedance_error > -self.z_force_clip:
            return max(
                z_force_error / self.z_force_k, -self.z_force_clip - z_impedance_error
            ), z_force_error
        else:
            return 0.0, z_force_error

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            # open gripper
            self.env.unwrapped.gripper.set_target(1.0)  # type: ignore
            time.sleep(2.0)
            self.obs, *_ = self.env.step(np.zeros(6))  # wait one step
            # move back and up
            self.obs = self.go_delta(
                self.obs,
                [-0.04, 0.0, 0.02],
            )
            # close gripper
            self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
            time.sleep(2.0)

            if self.use_ft_controller:
                # push with constant force (5N)
                self.obs, *_ = self.env.step(np.zeros(6))
                while self.obs["observation.state.sensors_bota_ft_sensor"][2] > -6:
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) > 0.0015:
                        self.obs, *_ = self.env.step(np.zeros(6))
                    else:
                        dz, _ = self.z_force_controller_dz(self.obs)
                        self.obs, *_ = self.env.step(
                            np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0])
                        )
                self.env.step(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]))
                time.sleep(2.0)
            else:
                print("Skipping constant-force push (FT controller disabled).")

        print("homing...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        time.sleep(3.0)

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self.first_reset = False
        self._pe_episode_idx += 1

        # estimate (coarse)
        _viz_enabled = self._pose_overlay_renderer is not None
        if _viz_enabled:
            world_D_world_obj, _coarse_details = self.estimate_world_pose_once(
                return_details=True
            )
            _coarse_rgb = np.copy(self.obs["observation.images.wrist_camera"])
            _coarse_depth = np.copy(self.obs["observation.images.wrist_depth_camera"])
            _coarse_tcp_cart = np.copy(self.obs["observation.state.cartesian"])
            _coarse_pose_cam = _coarse_details["pose_cam"]
            _coarse_mask = _coarse_details["mask"]
            _coarse_world = np.copy(world_D_world_obj)
        else:
            world_D_world_obj = self.estimate_world_pose_once()
        pose_check_coarse = self.validate_world_pose_transform(
            world_D_world_obj, "coarse"
        )
        est_obj_pos = world_D_world_obj[:3, 3]
        est_obj_euler = rot_matrix_to_euler_xyz(world_D_world_obj[:3, :3])
        print("Estimated object pose (world frame):")
        print(f"  position xyz [m] = {est_obj_pos}")
        print(f"  orientation rpy [rad] = {est_obj_euler}")
        print(f"  orientation rpy [deg] = {np.rad2deg(est_obj_euler)}")
        print("Estimated transform matrix:", world_D_world_obj)
        world_T_demoobj_estiobj = (
            world_D_world_obj[:3, 3] - self.env_config.demo_w_D_w_o[:3, 3]
        )
        print(
            "Delta translation: ",
            world_T_demoobj_estiobj,
        )

        # Apply 6DoF orientation alignment at pre-grasp hover so the wrist
        # camera approaches refined PE from a demo-aligned viewpoint and the
        # final descent delta is small. 25 deg axis-angle clip in
        # compute_alignment_rpy caps the blast radius if coarse PE is bad.
        # 3DoF PE mode keeps tool at home (no rotation), matching the safe
        # legacy behavior.
        coarse_alignment_rpy = self.compute_alignment_rpy(world_D_world_obj)
        if self.use_6dof_grasp:
            coarse_rel_euler = coarse_alignment_rpy
            coarse_R_applied = euler_to_rot_matrix(*coarse_rel_euler)
            print(
                "coarse alignment rpy [deg] (applied at hover):",
                np.rad2deg(coarse_rel_euler),
            )
        else:
            coarse_rel_euler = None
            coarse_R_applied = np.eye(3)
            print(
                "coarse alignment rpy [deg] (3DoF PE, not applied):",
                np.rad2deg(coarse_alignment_rpy),
            )
        # Reassigned to the TOTAL applied (coarse + delta) after descent.
        # At viz-dump time below, this reflects what has been applied so far
        # (coarse only).
        applied_alignment_rpy = (
            np.asarray(coarse_rel_euler, dtype=np.float64)
            if coarse_rel_euler is not None
            else np.zeros(3)
        )

        # Pre-grasp hover: use the rotated grasp offset so a yawed object
        # still ends up centered under the wrist camera for refined PE.
        coarse_grasp_world = (
            world_D_world_obj[:3, 3]
            + world_D_world_obj[:3, :3] @ self.o_T_o_tcpgrasp
        )
        coarse_hover_world = (
            coarse_grasp_world
            + world_D_world_obj[:3, :3] @ np.array([0.0, 0.0, 0.02])
        )
        self.obs = self.go_to_waypoint(
            self.obs,
            coarse_hover_world,
            relative_pose_euler=coarse_rel_euler,
            distance_err=0.0002,
            is_via=False,
        )

        # estimate (refined)
        if _viz_enabled:
            world_D_world_obj, _refined_details = self.estimate_world_pose_once(
                return_details=True
            )
            _refined_rgb = np.copy(self.obs["observation.images.wrist_camera"])
            _refined_depth = np.copy(self.obs["observation.images.wrist_depth_camera"])
            _refined_tcp_cart = np.copy(self.obs["observation.state.cartesian"])
            _refined_pose_cam = _refined_details["pose_cam"]
            _refined_mask = _refined_details["mask"]
            _refined_world = np.copy(world_D_world_obj)
        else:
            world_D_world_obj = self.estimate_world_pose_once()
        pose_check_refined = self.validate_world_pose_transform(
            world_D_world_obj, "refined"
        )
        est_obj_pos_refined = world_D_world_obj[:3, 3]
        est_obj_euler_refined = rot_matrix_to_euler_xyz(world_D_world_obj[:3, :3])
        print("Refined estimated object pose (world frame):")
        print(f"  position xyz [m] = {est_obj_pos_refined}")
        print(f"  orientation rpy [rad] = {est_obj_euler_refined}")
        print(f"  orientation rpy [deg] = {np.rad2deg(est_obj_euler_refined)}")

        # Grasp position: use the full est_R so the object-frame grasp offset
        # (o_T_o_tcpgrasp) is placed into the world correctly regardless of
        # object roll/pitch/yaw.
        est_R = world_D_world_obj[:3, :3]
        w_T_w_tcpgrasp = world_D_world_obj[:3, 3] + est_R @ self.o_T_o_tcpgrasp

        # Grasp orientation: preserve "gripper-in-object" relative to demo,
        # target total = est_R_demo @ R_home (see compute_alignment_rpy).
        # The coarse alignment was already applied at the pre-grasp hover,
        # so the refined stage commands only the DELTA from coarse->refined.
        # go_to_waypoint pre-multiplies in world frame:
        #   R_new = R_delta @ (R_coarse @ R_home) == R_refined @ R_home
        #   => R_delta = R_refined @ R_coarse^T
        if self.use_6dof_grasp:
            refined_rel_euler_full = self.compute_alignment_rpy(world_D_world_obj)
            refined_R = euler_to_rot_matrix(*refined_rel_euler_full)
            delta_R = refined_R @ coarse_R_applied.T
            target_rel_euler = rot_matrix_to_euler_xyz(delta_R)
            print(
                "full refined alignment rpy [deg]:",
                np.rad2deg(refined_rel_euler_full),
                " delta applied on descent rpy [deg]:",
                np.rad2deg(target_rel_euler),
            )
        else:
            target_rel_euler = None
            refined_rel_euler_full = np.zeros(3)
            print("3DoF PE: descent stays top-down (no gripper rotation)")
        est_Z = est_R[:, 2]
        tilt_cos = float(np.clip(est_Z[2], -1.0, 1.0))
        print(
            "Estimated grasp position (full est_R):",
            w_T_w_tcpgrasp,
            " tilt [deg]:",
            np.rad2deg(np.arccos(tilt_cos)),
        )

        refined_alignment_rpy = refined_rel_euler_full
        print("refined alignment rpy [deg] (total):", np.rad2deg(refined_alignment_rpy))

        if _viz_enabled:
            try:
                from crisp_drl.envs.pose_visualizer import dump_raw_npz

                paths = self._pose_overlay_renderer.save_pair(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=_coarse_rgb,
                    mask=_coarse_mask,
                    pose_cam_coarse=_coarse_pose_cam,
                    pose_cam_refined=_refined_pose_cam,
                    rgb_refined=_refined_rgb,
                    mask_refined=_refined_mask,
                )
                raw_path = dump_raw_npz(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb_coarse=_coarse_rgb,
                    rgb_refined=_refined_rgb,
                    depth_coarse=_coarse_depth,
                    depth_refined=_refined_depth,
                    mask_coarse=_coarse_mask,
                    mask_refined=_refined_mask,
                    pose_cam_coarse=_coarse_pose_cam,
                    pose_cam_refined=_refined_pose_cam,
                    world_pose_coarse=_coarse_world,
                    world_pose_refined=_refined_world,
                    tcp_cartesian_coarse=_coarse_tcp_cart,
                    tcp_cartesian_refined=_refined_tcp_cart,
                    applied_alignment_rpy=np.asarray(applied_alignment_rpy),
                    refined_alignment_rpy=np.asarray(refined_alignment_rpy),
                    pose_check_coarse_det_r=np.array([pose_check_coarse["det_r"]], dtype=np.float32),
                    pose_check_coarse_orth_err=np.array([pose_check_coarse["orth_err"]], dtype=np.float32),
                    pose_check_refined_det_r=np.array([pose_check_refined["det_r"]], dtype=np.float32),
                    pose_check_refined_orth_err=np.array([pose_check_refined["orth_err"]], dtype=np.float32),
                )
                print(f"[pose viz] saved {paths} and {raw_path}")
            except Exception as e:
                print(f"[pose viz] failed to save artifacts: {e}")

        # Final descent: apply target_rel_euler (= coarse->refined DELTA) at
        # the refined hover so gripper-in-object matches the demo. Coarse
        # portion was already commanded before refined PE. Rotation delta
        # is applied ONCE at the hover step and then persists on the robot
        # target_pose; the descent call MUST pass None to avoid doubling
        # the rotation (and breaking the undo at end of this method).
        print("Moving to pre-grasp hover...")
        # Keep the same hover height, but express it as a single object-frame
        # offset so the nonzero lateral grasp offset is included explicitly.
        hover_xyz = world_D_world_obj[:3, 3] + est_R @ (
            self.o_T_o_tcpgrasp + np.array([0.0, 0.0, 0.015])
        )
        self.obs = self.go_to_waypoint(
            self.obs,
            hover_xyz,
            relative_pose_euler=target_rel_euler,
            distance_err=0.0005,
            is_via=False,
        )
        print("Descending to grasp...")
        self.obs = self.go_to_waypoint(
            self.obs,
            w_T_w_tcpgrasp,
            relative_pose_euler=None,
            distance_err=0.0002,
            is_via=False,
        )
        # Track the TOTAL applied orientation (coarse @ hover + delta @ descent
        # == refined_rel_euler_full) so the post-pickup "undo" inverses the
        # orientation we actually commanded. 3DoF PE mode didn't rotate ->
        # zeros -> undo is a no-op.
        applied_alignment_rpy = np.asarray(refined_rel_euler_full, dtype=np.float64)

        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        print("Estimating pose at closed gripper...")
        t_D_t_o_at_closed, _grasp_details = (
            self.pose_estimation_helper.estimate_siemens_tcp_frame_coarse(
                self.obs["observation.images.wrist_camera"],
                self.obs["observation.images.wrist_depth_camera"],
                return_details=True,
            )
        )
        print(
            "closed-gripper tcp-frame object translation [m]:",
            t_D_t_o_at_closed[:3, 3],
        )
        if _viz_enabled:
            try:
                grasp_overlay_png = self._pose_overlay_renderer.save_single(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=self.obs["observation.images.wrist_camera"],
                    pose_cam_obj=_grasp_details["pose_cam"],
                    mask=_grasp_details["mask"],
                    suffix="grasp_closed",
                    label=f"ep{self._pe_episode_idx} grasp closed",
                )
                print(f"[pose viz] saved grasp closed overlay: {grasp_overlay_png}")
            except Exception as e:
                print(f"[pose viz] failed to save grasp closed overlay: {e}")

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_delta(
            self.obs,
            self.env_config.relative_motion_after_grasp_pe[:3],
            self.env_config.relative_motion_after_grasp_pe[3:],
            distance_err=0.001,
            is_via=False,
        )

        # estimate grasped delta
        t_D_t_o = self.pose_estimation_helper.estimate_siemens_tcp_frame_coarse(
            self.obs["observation.images.wrist_camera"],
            self.obs["observation.images.wrist_depth_camera"],
        )
        if _viz_enabled:
            try:
                grasped_png = self._pose_overlay_renderer.save_rgb(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=self.obs["observation.images.wrist_camera"],
                    suffix="grasped",
                )
                print(f"[pose viz] saved grasped image: {grasped_png}")
            except Exception as e:
                print(f"[pose viz] failed to save grasped image: {e}")
        self.estimated_grasp_delta = (
            t_D_t_o[:3, 3] - self.env_config.demo_t_D_t_o[:3, 3]
        )
        print(f"grasp delta (estimate): {self.estimated_grasp_delta}")

        # compute goal position
        # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position

        self.goal_position = self.goal_position_ground_truth.copy()
        self.goal_position[0] += self.estimated_grasp_delta[0]
        self.goal_position[2] += self.estimated_grasp_delta[2]

        self.start_position = self.goal_position.copy()

        applied_alignment_rot = euler_to_rot_matrix(
            applied_alignment_rpy[0],
            applied_alignment_rpy[1],
            applied_alignment_rpy[2],
        )
        undo_alignment_rpy = rot_matrix_to_euler_xyz(applied_alignment_rot.T)
        self.obs, *_ = self.env.step(
            [
                0.0,
                0.0,
                0.0,
                undo_alignment_rpy[0],
                undo_alignment_rpy[1],
                undo_alignment_rpy[2],
            ]
        )
        for pose, res in self.env_config.waypoints_after_grasp:
            self.obs = self.go_to_waypoint(
                self.obs,
                pose[:3] + self.estimated_grasp_delta,
                pose[3:],
                distance_err=res,
            )

        # lower down slowly until z-force is established in steps of 3mm
        if self.use_ft_controller:
            print("Establishing contact...")
            self.obs, *_ = self.env.step(np.zeros(6))
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            # [s.reset() for s in self.env.unwrapped.sensors]  # pyright: ignore[reportAttributeAccessIssue] # tare ft sensor
            x_step, x_force_error = self.x_torque_controller_dx(self.obs)
            while abs(x_force_error) > 0.1:  # wait until some contact
                # delta_yz = self.yz_i_controller_dyz(self.obs, self.start_position[1:3])
                delta_yz = (
                    np.clip(
                        self.start_position[1:3]
                        - self.obs["observation.state.cartesian"][1:3],
                        -self.i_term_clip,
                        self.i_term_clip,
                    )
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) < 0.001
                    else np.zeros(2)
                )
                self.obs, *_ = self.env.step(
                    np.array([x_step, delta_yz[0], delta_yz[1], 0, 0, 0])
                )
                x_step, x_force_error = self.x_torque_controller_dx(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.pose_estimation.object_pose_world.position"] = (
            est_obj_pos_refined
        )
        reset_info["reset.pose_estimation.object_pose_world.euler_rpy"] = (
            est_obj_euler_refined
        )
        reset_info["reset.pose_estimation.object_pose_world.matrix"] = world_D_world_obj
        reset_info["reset.pose_estimation.object_pose_world.valid"] = np.array(
            [pose_check_refined["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.det_r"] = np.array(
            [pose_check_refined["det_r"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.orth_err"] = np.array(
            [pose_check_refined["orth_err"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.coarse_valid"] = np.array(
            [pose_check_coarse["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.alignment_rpy_applied"] = applied_alignment_rpy
        reset_info["reset.pose_estimation.object_pose_tcp_at_closed_gripper.position"] = (
            t_D_t_o_at_closed[:3, 3]
        )
        reset_info["reset.pose_estimation.object_pose_tcp_at_closed_gripper.matrix"] = (
            t_D_t_o_at_closed
        )

        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta

        print("Reset complete.")
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
        if self.use_6dof_grasp:
            if action.shape[0] < 5:
                raise ValueError(
                    f"Expected 5D action in 6DoF mode, got shape {action.shape}"
                )
            action = np.array(
                [x_action, action[0], action[1], action[2], action[3], action[4]]
            )
        else:
            action = np.array([x_action, action[0], action[1], 0, 0, 0])

        # apply safety box
        current_pos_yz = self.obs["observation.state.cartesian"][1:3]
        delta_yz = current_pos_yz - self.goal_position[1:3]
        delta_yz_clipped = np.clip(
            delta_yz, -self.safety_box_radius, self.safety_box_radius
        )
        if np.any(delta_yz_clipped != delta_yz):
            correcting_action = delta_yz_clipped - delta_yz
            action[1:3] = correcting_action

        # print(f"[Step] received {action_input}, performing {action}")
        self.obs, reward, terminated, truncated, info = self.env.step(action)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info
