"""3DoF+RotZ LEGO insertion wrapper with FoundationPose-driven grasp and goal.

Combines:
- InsertionWrapper3DoFRotZ action interface: [dx, dy, drz], FT controls z.
- InsertionWrapperLegoPE PE-driven grasp (lavender brick) and goal estimation.

Grasp pose:  from refined FoundationPose of lavender brick.
Goal XY:     from PE of target brick at diagonal hover.
goal_rotation_z: world-frame yaw of target brick (from PE).

Target brick is configurable (default "yellow") and must be supported by the
SAM3 segmenter and PoseEstimator/Tracker (see sam3_detector.py, pose_estimator.py).
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Wrapper, spaces
from scipy.spatial.transform import Rotation

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import LegoConfig
from crisp_drl.agents.shared.insertion_wrapper_s import rot_matrix_to_euler_xyz
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper, euler_to_rot_matrix
from crisp_gym.envs.env_wrapper import LastObservationWrapper


class InsertionWrapper3DoFRotZPE(Wrapper):
    """Real-world 3DoF+RotZ LEGO wrapper with PE-driven grasp and goal.

    Agent action: [dx, dy, drz].  Wrapper expands to 6D [dx, dy, dz_ft, 0, 0, drz]
    for the underlying NoGripperActionWrapper.

    PE schedule (per reset):
      1. Wide PE of both bricks from overview pose.
      2. Coarse orientation alignment + hover above lavender.
      3. Refined PE of lavender → precise grasp pose.
      4. Descend, close gripper, lift, undo orientation.
      5. Diagonal hover above target brick.
      6. PE of both bricks → compute XY goal and goal_rotation_z.
      7. Move to start position + yaw, establish FT contact.
    """

    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: LegoConfig,
        grasp_color: str = "lavender",
        target_color: str = "yellow",
        safety_box_radius: float = 0.003,
        safety_box_step_size: float = 0.0005,
        safety_box_angular_radius: float = np.deg2rad(3),
        safety_box_angular_step_size: float = np.deg2rad(0.5),
        goal_orientation_randomisation_angle: float = np.deg2rad(3),
        step_limit: int = 150,
        minimal_start_goal_distance: float = 0.003,
        minimal_start_goal_angle: float = np.deg2rad(2),
        is_eval: bool = False,
        use_ft_controller: bool = True,
        pose_viz_dir: Optional[str] = None,
        use_tracker: bool = False,
        workspace_box: tuple = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7)),
    ):
        super().__init__(env)
        self.alg_config = alg_config
        self.env_config = env_config
        self.grasp_color = grasp_color
        self.target_color = target_color

        # env_config fields
        self.home_config = env_config.custom_home_position
        self.grasp_position_ground_truth = np.array(env_config.grasp_position_ground_truth)
        self.goal_position_ground_truth = np.array(env_config.goal_position_ground_truth)
        self.wide_pe_pose_euler = np.array(env_config.demo_goal_pose_estimation_euler)
        self.diagonal_hover_offset = np.array(env_config.diagonal_hover_above_purple_offset)
        self.place_z_offset = float(env_config.place_z_offset)
        self.after_grasp_lift_height_pe = float(env_config.after_grasp_lift_height_pe)
        self.alignment_clip_angle_rad = float(env_config.alignment_clip_angle_rad)
        self.alignment_pitch_bias = float(getattr(env_config, "alignment_pitch_bias", 0.0))

        # grasp offset in object frame (same formula as InsertionWrapperLegoPE)
        lav_pose = np.array(env_config.demo_grasped_pose_lavender)
        lav_R = lav_pose[:3, :3]
        lav_t = lav_pose[:3, 3]
        self.o_T_o_tcpgrasp_lavender = lav_R.T @ (
            self.grasp_position_ground_truth - lav_t
        )

        # action/safety params
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.safety_box_angular_radius = safety_box_angular_radius
        self.safety_box_angular_step_size = safety_box_angular_step_size
        self.goal_orientation_randomisation_angle = goal_orientation_randomisation_angle
        self.step_limit = step_limit
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.minimal_start_goal_angle = minimal_start_goal_angle
        self.is_eval = is_eval
        self.use_ft_controller = use_ft_controller
        self.workspace_box = workspace_box

        # FT constants (same as InsertionWrapper3DoFRotZ)
        self.z_force_target = -0.7
        self.z_force_k = 2500
        self.z_force_clip = 0.003
        self.i_term_clip = 0.001 * 2

        # reset / motion constants
        self.reset_lift_height = 0.020
        self.after_grasp_lift_height = 0.016
        self.delta_z_push_reset = 0.003
        self.delta_z_push_reset_step_size = 0.0008
        self.delta_z_push_reset_careful_threshold_distance = 0.003
        self.delta_z_push_reset_careful_threshold_velocity = 0.003
        self.min_coarse_hover_z = 0.055
        self.coarse_hover_min_above_current = 0.005

        # state
        self.first_reset = True
        self.n_since_last_home = 0
        self.n_steps = 0
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))
        self.goal_position = self.goal_position_ground_truth.copy()
        self.goal_rotation_z = 0.0
        self.start_rotation_z = 0.0
        self.actual_grasp_position = self.grasp_position_ground_truth.copy()
        self.obs: dict = {}

        # PE helpers
        brick_size = getattr(env_config, "brick_size", "2x4")
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=np.array([]),
            lock_orientation=False,
            brick_size=brick_size,
            use_tracker=use_tracker,
        )
        self.pose_estimation_settle_steps = 1

        # viz
        self.pose_viz_dir = pose_viz_dir
        self._pe_episode_idx = -1
        self._pose_overlay_renderer = None
        if pose_viz_dir is not None:
            from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer
            yellow_mesh_candidates = (
                f"/workspaces/isaac_ros-dev/lego_assets/lego_{brick_size}_{target_color}_up.obj",
                os.path.expanduser(
                    f"~/workspaces/isaac_ros-dev/lego_assets/lego_{brick_size}_{target_color}_up.obj"
                ),
            )
            mesh_path = next((p for p in yellow_mesh_candidates if os.path.exists(p)), None)
            self._pose_overlay_renderer = PoseOverlayRenderer(
                camera_info_json_path="camera_parameters/realsense_d405_single.json",
                mesh_path=mesh_path,
                use_default_mesh_fallback=False,
            )
            print(f"[InsertionWrapper3DoFRotZPE] pose viz dir = {pose_viz_dir}")

        print(
            f"[InsertionWrapper3DoFRotZPE] __init__: eval={is_eval} "
            f"grasp_color={grasp_color} target_color={target_color} "
            f"use_tracker={use_tracker}"
        )

    # ------------------------------------------------------------------ #
    # Low-level movement helpers (mirroring InsertionWrapper3DoFRotZ)      #
    # ------------------------------------------------------------------ #

    def _step_zeros(self):
        return self.env.step(np.zeros(6))

    def _step_translation(self, dxyz):
        action6 = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0])
        return self.env.step(action6)

    def _step_yaw(self, drz):
        action6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, drz])
        return self.env.step(action6)

    def go_to_cartesian(
        self,
        current_obs,
        target_cartesian=None,
        delta=None,
        fine_resolution=None,
        max_settle_iter: int = 300,
    ):
        assert target_cartesian is not None or delta is not None
        if target_cartesian is None:
            target_cartesian = current_obs["observation.state.cartesian"][:3] + delta
        obs, *_ = self._step_translation(
            target_cartesian - current_obs["observation.state.cartesian"][:3]
        )
        n = 0
        while (
            np.linalg.norm(target_cartesian - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ) and n < max_settle_iter:
            obs, *_ = self._step_zeros()
            n += 1
        if n >= max_settle_iter:
            print(
                f"[InsertionWrapper3DoFRotZPE] go_to_cartesian: max_settle_iter hit; "
                f"residual={np.linalg.norm(target_cartesian - obs['observation.state.cartesian'][:3])*1e3:.2f} mm"
            )
        if fine_resolution is not None:
            err = target_cartesian - obs["observation.state.cartesian"][:3]
            n = 0
            while np.linalg.norm(err) > fine_resolution and n < max_settle_iter:
                obs, *_ = self._step_translation(
                    np.clip(err, -self.i_term_clip, self.i_term_clip)
                )
                err = target_cartesian - obs["observation.state.cartesian"][:3]
                n += 1
            n = 0
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.0005
                and n < max_settle_iter
            ):
                obs, *_ = self._step_zeros()
                n += 1
        return obs

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err: float = 0.002,
        velocity_err: float = 0.0005,
        is_via: bool = True,
        label: str = "waypoint",
        max_settle_iter: int = 300,
    ):
        """Move to position (and optionally apply relative_pose_euler delta)."""
        self._assert_in_workspace(position, label)
        relative = [0.0, 0.0, 0.0] if relative_pose_euler is None else list(relative_pose_euler)
        target = np.concatenate((position, relative))
        print(
            f"[InsertionWrapper3DoFRotZPE] go_to_waypoint({label}) "
            f"target_xyz={target[:3]} rel_rpy={target[3:]} "
            f"current_xyz={current_obs['observation.state.cartesian'][:3]}"
        )
        obs, *_ = self.env.step(
            target - np.concatenate((current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0]))
        )
        n = 0
        while (
            np.any(
                np.abs(obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]) > 0.002
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ) and n < max_settle_iter:
            obs, *_ = self._step_zeros()
            n += 1
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            ctrl_err = obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]
            n = 0
            while np.linalg.norm(err) > distance_err and n < max_settle_iter:
                i_clip = (
                    self.i_term_clip
                    if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
                    else ((0.002 - np.linalg.norm(obs["observation.velocity.cartesian"][:3])) / 0.002 + 1)
                    * self.i_term_clip
                )
                if np.any(np.abs(ctrl_err) > i_clip):
                    obs, *_ = self._step_zeros()
                else:
                    obs, *_ = self.env.step(
                        np.concatenate(
                            (np.clip(err + ctrl_err, -i_clip, i_clip) - ctrl_err, np.zeros(3))
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                ctrl_err = obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]
                n += 1
            n = 0
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
                and n < max_settle_iter
            ):
                obs, *_ = self._step_zeros()
                n += 1
        return obs

    def _yaw_error_to(self, current_rotvec, target_rz):
        goal_rotvec = np.array([0.0, 0.0, float(target_rz)], dtype=np.float64)
        rot_err = LastObservationWrapper._relative_rotation_error(
            goal_rotvec, np.asarray(current_rotvec, dtype=np.float64)
        )
        return float(rot_err[2])

    def go_to_rotation_z(
        self,
        current_obs,
        target_rz,
        tol=np.deg2rad(0.3),
        max_drive_iter=200,
        max_settle_iter=30,
    ):
        obs = current_obs
        rz_err = self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz)
        n = 0
        while abs(rz_err) > tol and n < max_drive_iter:
            drz = np.sign(rz_err) * min(self.safety_box_angular_step_size, abs(rz_err))
            obs, *_ = self._step_yaw(drz)
            rz_err = self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz)
            n += 1
        if n >= max_drive_iter:
            print(
                f"[InsertionWrapper3DoFRotZPE] go_to_rotation_z: max_drive_iter hit; "
                f"residual rz_err={np.rad2deg(rz_err):.3f} deg"
            )
        n_settle = 0
        while (
            np.linalg.norm(obs["observation.velocity.angular"]) > np.deg2rad(1.0)
            and n_settle < max_settle_iter
        ):
            obs, *_ = self._step_zeros()
            n_settle += 1
        return obs

    def go_delta(
        self,
        current_obs,
        delta_xyz,
        relative_pose_euler=None,
        distance_err: float = 0.002,
        velocity_err: float = 0.0005,
        is_via: bool = True,
    ):
        target = np.array(delta_xyz) + current_obs["observation.state.target"][:3]
        relative = [0.0, 0.0, 0.0] if relative_pose_euler is None else list(relative_pose_euler)
        obs, *_ = self.env.step(np.concatenate((delta_xyz, relative)))
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self._step_zeros()
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            ctrl_err = obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]
            while np.linalg.norm(err) > distance_err:
                i_clip = (
                    self.i_term_clip
                    if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
                    else ((0.002 - np.linalg.norm(obs["observation.velocity.cartesian"][:3])) / 0.002 + 1)
                    * self.i_term_clip
                )
                if np.any(np.abs(ctrl_err) > i_clip):
                    obs, *_ = self._step_zeros()
                else:
                    obs, *_ = self.env.step(
                        np.concatenate(
                            (np.clip(err + ctrl_err, -i_clip, i_clip) - ctrl_err, np.zeros(3))
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                ctrl_err = obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]
            while np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err:
                obs, *_ = self._step_zeros()
        return obs

    def z_force_controller_dz(self, obs):
        z_force_error = (
            self.z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
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

    # ------------------------------------------------------------------ #
    # Workspace guard                                                       #
    # ------------------------------------------------------------------ #

    def _assert_in_workspace(self, position, label: str):
        for v, (lo, hi) in zip(position, self.workspace_box):
            if not (lo <= v <= hi):
                raise ValueError(
                    f"[InsertionWrapper3DoFRotZPE] {label} target {position} outside "
                    f"workspace_box {self.workspace_box}"
                )

    # ------------------------------------------------------------------ #
    # PE helpers                                                           #
    # ------------------------------------------------------------------ #

    def _settle_for_pe(self):
        for _ in range(self.pose_estimation_settle_steps):
            self.obs, *_ = self._step_zeros()

    def _estimate_lego_world_both(self):
        """Estimate world-frame poses of both bricks (grasp_color + target_color)."""
        self._settle_for_pe()
        depth_f32 = (
            self.obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
        )
        rgb = self.obs["observation.images.wrist_camera"]
        # brightest mask → target (yellow), darkest → grasp (lavender)
        masks = self.pose_estimation_helper.segmenter.segment_lego(
            rgb, colors=(self.target_color, self.grasp_color)
        )
        for c in (self.grasp_color, self.target_color):
            if c not in masks:
                raise RuntimeError(
                    f"segment_lego missing color {c!r}; got keys={list(masks)}"
                )
        cam_poses = {
            c: self.pose_estimation_helper.pose_estimator.estimate_lego(
                rgb, depth_f32, masks[c], c
            )
            for c in (self.grasp_color, self.target_color)
        }
        if self.pose_estimation_helper.use_tracker:
            cam_poses = {
                c: self.pose_estimation_helper.pose_tracker.track_lego(
                    rgb, depth_f32, cam_poses[c], c
                )
                for c in cam_poses
            }
        tcp_cart = self.obs["observation.state.cartesian"]
        world_grasp = self.pose_estimation_helper._compute_pose_in_world_frame(
            cam_poses[self.grasp_color], tcp_cart
        )
        world_target = self.pose_estimation_helper._compute_pose_in_world_frame(
            cam_poses[self.target_color], tcp_cart
        )
        details = {
            "rgb": np.copy(rgb),
            "depth": np.copy(self.obs["observation.images.wrist_depth_camera"]),
            "tcp_cart": np.copy(tcp_cart),
            f"mask_{self.grasp_color}": masks[self.grasp_color],
            f"mask_{self.target_color}": masks[self.target_color],
            f"pose_cam_{self.grasp_color}": cam_poses[self.grasp_color],
            f"pose_cam_{self.target_color}": cam_poses[self.target_color],
        }
        return world_grasp, world_target, details

    def _estimate_lego_world_single(self, brick: str):
        """Estimate world-frame pose for one brick."""
        self._settle_for_pe()
        depth_f32 = (
            self.obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
        )
        rgb = self.obs["observation.images.wrist_camera"]
        masks = self.pose_estimation_helper.segmenter.segment_lego(
            rgb, colors=(self.target_color, self.grasp_color)
        )
        if brick not in masks:
            raise RuntimeError(
                f"segment_lego did not return mask for {brick!r}; got keys={list(masks)}"
            )
        pose_cam = self.pose_estimation_helper.pose_estimator.estimate_lego(
            rgb, depth_f32, masks[brick], brick
        )
        if self.pose_estimation_helper.use_tracker:
            pose_cam = self.pose_estimation_helper.pose_tracker.track_lego(
                rgb, depth_f32, pose_cam, brick
            )
        tcp_cart = self.obs["observation.state.cartesian"]
        world_pose = self.pose_estimation_helper._compute_pose_in_world_frame(
            pose_cam, tcp_cart
        )
        details = {
            "rgb": np.copy(rgb),
            "depth": np.copy(self.obs["observation.images.wrist_depth_camera"]),
            "tcp_cart": np.copy(tcp_cart),
            "mask": masks[brick],
            "pose_cam": pose_cam,
        }
        return world_pose, details

    def validate_world_pose_transform(self, world_D_world_obj: np.ndarray, tag: str):
        rot = world_D_world_obj[:3, :3]
        finite_ok = bool(np.isfinite(world_D_world_obj).all())
        orth_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
        det_r = float(np.linalg.det(rot))
        valid = finite_ok and abs(det_r - 1.0) < 1e-2 and orth_err < 1e-2
        print(
            f"[InsertionWrapper3DoFRotZPE] {tag} transform valid={valid} "
            f"det={det_r:.6f} orth_err={orth_err:.6e}"
        )
        return {"valid": valid, "det_r": det_r, "orth_err": orth_err}

    def compute_alignment_rpy(self, world_D_world_lav: np.ndarray) -> np.ndarray:
        """Demo-relative orientation for the gripper at the lavender grasp."""
        world_R_demo = np.array(self.env_config.demo_grasped_pose_lavender)[:3, :3]
        world_R_est = world_D_world_lav[:3, :3]
        est_R_demo = world_R_est @ world_R_demo.T

        bias_R = euler_to_rot_matrix(0.0, self.alignment_pitch_bias, 0.0)
        biased = est_R_demo @ bias_R
        rotvec = Rotation.from_matrix(biased).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.alignment_clip_angle_rad and angle > 0.0:
            rotvec = rotvec * (self.alignment_clip_angle_rad / angle)
            biased = Rotation.from_rotvec(rotvec).as_matrix()
        return rot_matrix_to_euler_xyz(biased)

    # ------------------------------------------------------------------ #
    # Viz helpers (best-effort)                                            #
    # ------------------------------------------------------------------ #

    def _viz_save_single(self, suffix: str, det: dict, label: str):
        if self._pose_overlay_renderer is None:
            return
        try:
            png = self._pose_overlay_renderer.save_single(
                out_dir=self.pose_viz_dir,
                episode_idx=self._pe_episode_idx,
                rgb=det["rgb"],
                pose_cam_obj=det["pose_cam"],
                mask=det["mask"],
                suffix=suffix,
                label=label,
            )
            print(f"[pose viz] saved {suffix}: {png}")
        except Exception as e:
            print(f"[pose viz] {suffix} save_single failed: {e}")

    def _viz_save_pair(self, suffix: str, det_a: dict, det_b: dict):
        if self._pose_overlay_renderer is None:
            return
        try:
            paths = self._pose_overlay_renderer.save_pair(
                out_dir=self.pose_viz_dir,
                episode_idx=self._pe_episode_idx,
                rgb=det_a["rgb"],
                mask=det_a.get(f"mask_{self.grasp_color}", det_a.get("mask")),
                pose_cam_coarse=det_a.get(f"pose_cam_{self.grasp_color}", det_a.get("pose_cam")),
                pose_cam_refined=det_b.get(f"pose_cam_{self.target_color}", det_b.get("pose_cam")),
                rgb_refined=det_b["rgb"],
                mask_refined=det_b.get(f"mask_{self.target_color}", det_b.get("mask")),
                names=(self.grasp_color, self.target_color),
            )
            print(f"[pose viz] saved {suffix}: {paths}")
        except Exception as e:
            print(f"[pose viz] {suffix} save_pair failed: {e}")

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.first_reset:
            # Activate Cartesian controller before any step.
            self.obs, _ = self.env.reset()
            self.obs, *_ = self._step_zeros()
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs = self.go_to_cartesian(
                self.obs, delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z])
            )
            self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
            print("Opening gripper...")
            time.sleep(2.0)
        else:
            # 1) lift (gripper may still be in contact at episode end)
            self.obs, *_ = self._step_zeros()
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs = self.go_to_cartesian(
                self.obs, delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z])
            )

            # 2) rotate yaw back to 0 in air
            self.obs = self.go_to_rotation_z(self.obs, target_rz=0.0)

            # 3) go back over grasp stand
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=np.array([
                    self.actual_grasp_position[0],
                    self.actual_grasp_position[1],
                    self.obs["observation.state.cartesian"][2] - self.reset_lift_height,
                ]),
            )

            # 4) push down to re-seat brick
            while (
                abs(
                    self.obs["observation.state.cartesian"][2]
                    - self.obs["observation.state.target"][2]
                )
                < self.delta_z_push_reset
            ):
                delta_xy = (
                    self.actual_grasp_position[0:2]
                    - self.obs["observation.state.cartesian"][0:2]
                )
                delta_z_step = (
                    -self.delta_z_push_reset_step_size
                    if (
                        self.obs["observation.velocity.cartesian"][2]
                        > -self.delta_z_push_reset_careful_threshold_velocity
                        or abs(
                            self.actual_grasp_position[2]
                            - self.obs["observation.state.cartesian"][2]
                        )
                        > self.delta_z_push_reset_careful_threshold_distance
                    )
                    else 0.0
                )
                self.obs, *_ = self._step_translation(
                    np.array([delta_xy[0], delta_xy[1], delta_z_step])
                )

            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs, *_ = self._step_translation(np.array([0.0, 0.0, delta_z * 0.8]))
            self.env.unwrapped.gripper.home()  # type: ignore
            time.sleep(0.5)
            self.n_since_last_home += 1

        if self.n_since_last_home >= 4 or self.first_reset:
            print(
                f"n_since_last_home={self.n_since_last_home}, "
                f"first_reset={self.first_reset}, homing..."
            )
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
            self.n_since_last_home = 0
            self.first_reset = False

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self._pe_episode_idx += 1
        time.sleep(1.0)

        # ---- wide PE -------------------------------------------------- #
        print("Moving to wide PE pose...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=self.wide_pe_pose_euler[:3],
            fine_resolution=0.0005,
        )
        print("Wide PE: estimating both bricks...")
        wide_grasp, wide_target, wide_det = self._estimate_lego_world_both()
        check_wide_grasp = self.validate_world_pose_transform(wide_grasp, f"wide_{self.grasp_color}")
        check_wide_target = self.validate_world_pose_transform(wide_target, f"wide_{self.target_color}")
        print(f"wide PE {self.grasp_color} pos = {wide_grasp[:3, 3]}")
        print(f"wide PE {self.target_color} pos = {wide_target[:3, 3]}")
        self._viz_save_pair("wide", wide_det, wide_det)

        # ---- coarse hover above lavender -------------------------------- #
        coarse_alignment_rpy = self.compute_alignment_rpy(wide_grasp)
        coarse_R_applied = euler_to_rot_matrix(*coarse_alignment_rpy)
        print(f"coarse alignment rpy [deg] = {np.rad2deg(coarse_alignment_rpy)}")

        coarse_grasp_world = (
            wide_grasp[:3, 3] + wide_grasp[:3, :3] @ self.o_T_o_tcpgrasp_lavender
        )
        coarse_hover_world = (
            coarse_grasp_world + wide_grasp[:3, :3] @ np.array([0.0, 0.0, 0.025])
        )
        current_xyz = self.obs["observation.state.cartesian"][:3]
        safe_hover_z = max(
            coarse_hover_world[2],
            self.min_coarse_hover_z,
            current_xyz[2] + self.coarse_hover_min_above_current,
        )
        if safe_hover_z != coarse_hover_world[2]:
            print(
                f"[InsertionWrapper3DoFRotZPE] WARNING: clamping coarse hover z from "
                f"{coarse_hover_world[2]:.6f} to {safe_hover_z:.6f}"
            )
            coarse_hover_world[2] = safe_hover_z

        print(f"Moving to coarse hover = {coarse_hover_world}")
        self.obs = self.go_to_waypoint(
            self.obs,
            coarse_hover_world,
            relative_pose_euler=coarse_alignment_rpy,
            distance_err=0.0005,
            is_via=False,
            label="coarse_hover",
        )

        # ---- refined PE (lavender only) --------------------------------- #
        print("Refined PE: estimating lavender...")
        refined_grasp, refined_det = self._estimate_lego_world_single(self.grasp_color)
        check_refined = self.validate_world_pose_transform(refined_grasp, f"refined_{self.grasp_color}")
        est_R = refined_grasp[:3, :3]
        w_T_w_tcpgrasp = refined_grasp[:3, 3] + est_R @ self.o_T_o_tcpgrasp_lavender

        refined_alignment_rpy = self.compute_alignment_rpy(refined_grasp)
        refined_R = euler_to_rot_matrix(*refined_alignment_rpy)
        delta_R = refined_R @ coarse_R_applied.T
        delta_alignment_rpy = rot_matrix_to_euler_xyz(delta_R)
        print(
            f"refined alignment rpy [deg] = {np.rad2deg(refined_alignment_rpy)}, "
            f"delta rpy [deg] = {np.rad2deg(delta_alignment_rpy)}"
        )
        self._viz_save_single("refined_grasp", refined_det, f"refined {self.grasp_color}")

        # ---- pre-grasp hover + descent ---------------------------------- #
        hover_xyz = refined_grasp[:3, 3] + est_R @ (
            self.o_T_o_tcpgrasp_lavender + np.array([0.0, 0.0, 0.015])
        )
        print(f"Moving to pre-grasp hover = {hover_xyz}")
        self.obs = self.go_to_waypoint(
            self.obs,
            hover_xyz,
            relative_pose_euler=delta_alignment_rpy,
            distance_err=0.0005,
            is_via=False,
            label="pre_grasp_hover",
        )
        print(f"Descending to grasp = {w_T_w_tcpgrasp}")
        self.obs = self.go_to_waypoint(
            self.obs,
            w_T_w_tcpgrasp,
            relative_pose_euler=None,
            distance_err=0.0002,
            is_via=False,
            label="descent_to_grasp",
        )
        applied_alignment_rpy = np.asarray(refined_alignment_rpy, dtype=np.float64)

        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self._step_zeros()
        self.actual_grasp_position = np.copy(self.obs["observation.state.cartesian"][:3])

        # ---- lift + undo orientation ------------------------------------ #
        print("Lifting after grasp...")
        self.obs = self.go_delta(
            self.obs,
            [0.0, 0.0, self.after_grasp_lift_height_pe],
            distance_err=0.001,
            is_via=False,
        )
        applied_alignment_rot = euler_to_rot_matrix(*applied_alignment_rpy)
        undo_alignment_rpy = rot_matrix_to_euler_xyz(applied_alignment_rot.T)
        self.obs, *_ = self.env.step(
            np.array([0.0, 0.0, 0.0, undo_alignment_rpy[0], undo_alignment_rpy[1], undo_alignment_rpy[2]])
        )

        # ---- diagonal hover above target -------------------------------- #
        diag_hover = wide_target[:3, 3] + self.diagonal_hover_offset
        print(f"Moving to diagonal hover above {self.target_color} = {diag_hover}")
        self.obs = self.go_to_waypoint(
            self.obs,
            diag_hover,
            relative_pose_euler=None,
            distance_err=0.0005,
            is_via=False,
            label="diagonal_hover_target",
        )

        # ---- PE both bricks at diagonal hover --------------------------- #
        print(f"PE both bricks at diagonal hover ({self.grasp_color} + {self.target_color})...")
        place_grasp, place_target, place_det = self._estimate_lego_world_both()
        self.validate_world_pose_transform(place_grasp, f"place_{self.grasp_color}")
        self.validate_world_pose_transform(place_target, f"place_{self.target_color}")
        self._viz_save_pair("place", place_det, place_det)

        # ---- compute goal XY and goal_rotation_z ------------------------ #
        ee_now = self.obs["observation.state.cartesian"][:3].copy()
        delta_world = (
            place_target[:3, 3]
            - place_grasp[:3, 3]
            + np.array([0.0, 0.0, self.place_z_offset])
        )
        self.goal_position = ee_now + delta_world
        estimated_grasp_delta = place_grasp[:3, 3] - self.actual_grasp_position

        # goal yaw = world-frame yaw of target brick
        self.goal_rotation_z = float(
            Rotation.from_matrix(place_target[:3, :3]).as_euler("xyz")[2]
        )
        print(
            f"goal_position = {self.goal_position}, "
            f"goal_rotation_z = {np.rad2deg(self.goal_rotation_z):.2f} deg"
        )

        # ---- sample start position and yaw ------------------------------ #
        if not self.is_eval:
            self.start_position = self.goal_position.copy()
            while (
                np.linalg.norm(self.start_position[:2] - self.goal_position[:2])
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            self.start_rotation_z = self.goal_rotation_z
            while (
                abs(self.start_rotation_z - self.goal_rotation_z)
                < self.minimal_start_goal_angle
            ):
                self.start_rotation_z = self.goal_rotation_z + np.random.uniform(
                    -self.goal_orientation_randomisation_angle,
                    self.goal_orientation_randomisation_angle,
                )
        else:
            self.start_position = self.goal_position.copy()
            self.start_rotation_z = self.goal_rotation_z

        # ---- move to start XY ------------------------------------------ #
        print("Moving to start position...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=np.array([
                self.start_position[0],
                self.start_position[1],
                self.obs["observation.state.cartesian"][2],
            ]),
            fine_resolution=0.0005,
        )

        # ---- rotate to start yaw --------------------------------------- #
        print(f"Rotating yaw to {np.rad2deg(self.start_rotation_z):.2f} deg...")
        self.obs = self.go_to_rotation_z(self.obs, target_rz=self.start_rotation_z)

        # ---- establish FT contact --------------------------------------- #
        self.n_steps = 0
        self.obs, *_ = self._step_zeros()
        if self.use_ft_controller:
            print("Establishing contact...")
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            z_step, z_force_error = self.z_force_controller_dz(self.obs)
            while abs(z_force_error) > 0.1:
                delta_xy = (
                    self.start_position[0:2] - self.obs["observation.state.cartesian"][0:2]
                )
                self.obs, *_ = self._step_translation(
                    np.array([delta_xy[0], delta_xy[1], z_step])
                )
                z_step, z_force_error = self.z_force_controller_dz(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        # ---- build reset_info ------------------------------------------ #
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.grasped.delta_estimated"] = estimated_grasp_delta
        reset_info["reset.goal_position"] = self.goal_position.copy()
        reset_info["reset.goal_orientation.rotation_z"] = self.goal_rotation_z
        reset_info["reset.start_orientation.rotation_z"] = self.start_rotation_z
        reset_info[f"reset.pose_estimation.wide.{self.grasp_color}.matrix"] = wide_grasp
        reset_info[f"reset.pose_estimation.wide.{self.target_color}.matrix"] = wide_target
        reset_info[f"reset.pose_estimation.wide.{self.grasp_color}.valid"] = np.array(
            [check_wide_grasp["valid"]], dtype=np.float32
        )
        reset_info[f"reset.pose_estimation.wide.{self.target_color}.valid"] = np.array(
            [check_wide_target["valid"]], dtype=np.float32
        )
        reset_info[f"reset.pose_estimation.refined.{self.grasp_color}.matrix"] = refined_grasp
        reset_info[f"reset.pose_estimation.refined.{self.grasp_color}.valid"] = np.array(
            [check_refined["valid"]], dtype=np.float32
        )
        reset_info[f"reset.pose_estimation.place.{self.grasp_color}.matrix"] = place_grasp
        reset_info[f"reset.pose_estimation.place.{self.target_color}.matrix"] = place_target
        reset_info["reset.pose_estimation.alignment_rpy_applied"] = applied_alignment_rpy

        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        z_action = (
            self.z_force_controller_dz(self.obs)[0] if self.use_ft_controller else 0.0
        )
        action6 = np.array([action[0], action[1], z_action, 0.0, 0.0, action[2]])

        # XY safety box
        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action6[:2] = delta_xy * self.safety_box_step_size / norm_xy

        # angular safety box (SO3-robust yaw error)
        rz_err = self._yaw_error_to(
            self.obs["observation.state.cartesian"][3:6], self.goal_rotation_z
        )
        if abs(rz_err) > self.safety_box_angular_radius:
            action6[5] = np.sign(rz_err) * self.safety_box_angular_step_size

        self.obs, reward, terminated, truncated, info = self.env.step(action6)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # snap_push (for SuccessClassificationWrapper on E_SUCCESS_CLS)        #
    # ------------------------------------------------------------------ #

    def snap_push(
        self,
        push_distance: float = 0.0030,
        pause_before: float = 1.0,
        pause_after: float = 2.0,
        reinforce: bool = False,
        reinforce_lift: float = 0.0150,
        reinforce_push: float = 0.0200,
        reinforce_post_lift: float = 0.0050,
    ) -> None:
        """Push down push_distance m to seat the brick after success.

        If ``reinforce=True``, executes a re-seat cycle afterward.
        """
        print(f"[InsertionWrapper3DoFRotZPE] snap_push: pausing {pause_before}s...")
        t0 = time.time()
        while time.time() - t0 < pause_before:
            self.obs, *_ = self._step_zeros()

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: pushing down {push_distance*1e3:.1f} mm...")
        self.obs = self.go_to_cartesian(
            self.obs,
            delta=np.array([0.0, 0.0, -push_distance]),
            fine_resolution=push_distance * 0.3,
        )

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: holding {pause_after}s...")
        t0 = time.time()
        while time.time() - t0 < pause_after:
            self.obs, *_ = self._step_zeros()
        print("[InsertionWrapper3DoFRotZPE] snap_push: done.")

        if not reinforce:
            return

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.80)  # type: ignore
        time.sleep(1.0)

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — lifting {reinforce_lift*1e3:.1f} mm...")
        lift1_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, reinforce_lift])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = lift1_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — closing gripper...")
        self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
        time.sleep(1.4)

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — pressing down {reinforce_push*1e3:.1f} mm...")
        press_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, -reinforce_push])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = press_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — lifting {reinforce_post_lift*1e3:.1f} mm...")
        lift2_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, reinforce_post_lift])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = lift2_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
        time.sleep(1.0)

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — descending to goal position...")
        grasp_target = self.goal_position.copy()
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = grasp_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — re-grasping lego...")
        self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
        time.sleep(1.0)
        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce done.")

    # ------------------------------------------------------------------ #
    # Observation helpers                                                  #
    # ------------------------------------------------------------------ #

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            obs["observation.state.cartesian"][:3] - self.goal_position
        )
        obs["observation.perfect_rotation"] = (
            obs["observation.state.cartesian"][3:]
            - np.array([0.0, 0.0, self.goal_rotation_z])
        )
        return obs
