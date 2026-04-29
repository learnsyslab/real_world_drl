"""6DoF PE-driven LEGO insertion wrapper.

Mirrors `InsertionWrapperSiemensPE` but adapted to the two-brick LEGO task:
both grasp pose AND placement pose come from pose estimation every reset.

Schedule (per reset):
  1. Wide PE of both bricks from an overview pose.
  2. Move above lavender, refined PE of lavender only.
  3. (Optional) one more PE at the descent hover for tightest grasp.
  4. Descend, close gripper, lift.
  5. Move to a diagonal hover above purple.
  6. PE both bricks (grasped lavender + target purple).
  7. goal_position = current EE + (purple - lavender_grasped) + place_z_offset.

Step contract matches the existing LEGO `InsertionWrapper` (2D action,
z controlled by FT, safety box around goal_position[:2]).
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Wrapper, spaces

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import LegoConfig
from crisp_drl.agents.shared.insertion_wrapper_s import rot_matrix_to_euler_xyz
from crisp_drl.envs.pose_estimation_helper import (
    PoseEstimationHelper,
    euler_to_rot_matrix,
)


class InsertionWrapperLegoPE(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: LegoConfig,
        safety_box_radius: float = 0.004,
        safety_box_step_size: float = 0.0005,
        step_limit: int = 150,
        use_ft_controller: bool = True,
        pose_viz_dir: Optional[str] = None,
        extra_pre_grasp_pe: bool = False,
        dry_run_skip_grasp: bool = False,
        workspace_box: tuple = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7)),
        use_tracker: bool = True,
    ):
        super().__init__(env)
        self.alg_config = alg_config
        self.env_config = env_config
        self.use_ft_controller = use_ft_controller
        self.pose_viz_dir = pose_viz_dir
        self.extra_pre_grasp_pe = extra_pre_grasp_pe
        self.dry_run_skip_grasp = dry_run_skip_grasp
        self.workspace_box = workspace_box
        self.use_tracker = use_tracker

        self._pe_episode_idx = -1
        self._pose_overlay_renderer = None
        brick_size = getattr(env_config, "brick_size", "2x2")
        if pose_viz_dir is not None:
            from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer

            mesh_filename = f"lego_{brick_size}_lavender_up.obj"
            lego_mesh_candidates = (
                f"/workspaces/isaac_ros-dev/lego_assets/{mesh_filename}",
                os.path.expanduser(
                    f"~/workspaces/isaac_ros-dev/lego_assets/{mesh_filename}"
                ),
            )
            lego_mesh_path = next(
                (p for p in lego_mesh_candidates if os.path.exists(p)), None
            )

            self._pose_overlay_renderer = PoseOverlayRenderer(
                camera_info_json_path="camera_parameters/realsense_d405_single.json",
                mesh_path=lego_mesh_path,
                use_default_mesh_fallback=False,
            )
            print(f"[InsertionWrapperLegoPE] pose viz dir = {pose_viz_dir}")

        # 2D RL action (xy) — z is FT-controlled like the existing LEGO wrapper.
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))

        # PE: lock_orientation=False so we get the actual 6DoF rotation back.
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=np.array([]),
            lock_orientation=False,
            brick_size=brick_size,
        )

        self.home_config = env_config.custom_home_position
        self.grasp_position_ground_truth = np.array(env_config.grasp_position_ground_truth)
        self.goal_position_ground_truth = np.array(env_config.goal_position_ground_truth)
        # Reuse existing LEGO wide PE pose (the same one the legacy
        # InsertionWrapper uses for its single PE step).
        self.wide_pe_pose_euler = np.array(env_config.demo_goal_pose_estimation_euler)
        self.diagonal_hover_above_purple_offset = np.array(
            env_config.diagonal_hover_above_purple_offset
        )
        self.demo_w_D_w_lavender = np.array(env_config.demo_grasped_pose_lavender)
        self.place_z_offset = float(env_config.place_z_offset)
        self.after_grasp_lift_height_pe = float(env_config.after_grasp_lift_height_pe)
        self.alignment_clip_angle_rad = float(env_config.alignment_clip_angle_rad)
        self.alignment_pitch_bias = float(env_config.alignment_pitch_bias)

        # Object-frame TCP grasp offset (analog of Siemens o_T_o_tcpgrasp).
        # demo: grasp_position_ground_truth - lavender_demo_t (rotated into
        # lavender's frame), so we can apply it to a freshly-estimated R.
        lav_R = self.demo_w_D_w_lavender[:3, :3]
        lav_t = self.demo_w_D_w_lavender[:3, 3]
        self.o_T_o_tcpgrasp_lavender = lav_R.T @ (
            self.grasp_position_ground_truth - lav_t
        )

        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.first_reset = True
        self.n_steps = 0
        self.i_term_clip = 0.0012
        self.pose_estimation_settle_steps = 1

        # FT (z-axis) — same constants as LEGO `InsertionWrapper`.
        self.z_force_target = -0.7
        self.z_force_k = 2500
        self.z_force_clip = 0.003
        self.reset_lift_height = 0.01
        self.after_grasp_lift_height = 0.016
        # Safety guard for PE-derived hover targets. Coarse hover should not
        # command a sudden downward move from the current camera pose.
        self.min_coarse_hover_z = 0.055
        self.coarse_hover_min_above_current = 0.005

        print(
            "[InsertionWrapperLegoPE] action_dim=2 use_ft_controller=",
            self.use_ft_controller,
            "pose_viz_dir=",
            pose_viz_dir,
        )

    # ---- helpers (6D action stack expected) ----

    def _assert_in_workspace(self, position, label: str):
        for v, (lo, hi) in zip(position, self.workspace_box):
            if not (lo <= v <= hi):
                raise ValueError(
                    f"[InsertionWrapperLegoPE] {label} target {position} outside "
                    f"workspace_box {self.workspace_box}"
                )

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err: float = 0.002,
        velocity_err: float = 0.0005,
        is_via: bool = True,
        label: str = "waypoint",
    ):
        self._assert_in_workspace(position, label)
        relative = [0.0, 0.0, 0.0] if relative_pose_euler is None else list(relative_pose_euler)
        target = np.concatenate((position, relative))
        print(
            f"[InsertionWrapperLegoPE] go_to_waypoint({label}) start "
            f"target_xyz={target[:3]} rel_rpy={target[3:]} "
            f"current_xyz={current_obs['observation.state.cartesian'][:3]}"
        )
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )
        coarse_wait_steps = 0
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
            coarse_wait_steps += 1
            if coarse_wait_steps % 100 == 0:
                cart = obs["observation.state.cartesian"][:3]
                tgt = obs["observation.state.target"][:3]
                vel = np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                print(
                    f"[InsertionWrapperLegoPE] go_to_waypoint({label}) coarse_wait "
                    f"steps={coarse_wait_steps} |target-cart|={np.linalg.norm(tgt - cart):.6f} "
                    f"vel={vel:.6f} cart={cart} target={tgt}"
                )
            obs, *_ = self.env.step(np.zeros(6))
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            fine_wait_steps = 0
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
                fine_wait_steps += 1
                if fine_wait_steps % 100 == 0:
                    print(
                        f"[InsertionWrapperLegoPE] go_to_waypoint({label}) fine_wait "
                        f"steps={fine_wait_steps} |err|={np.linalg.norm(err):.6f} "
                        f"|ctrl_err|={np.linalg.norm(controller_error):.6f} "
                        f"i_term_clip={i_term_clip:.6f} "
                        f"vel={np.linalg.norm(obs['observation.velocity.cartesian'][:3]):.6f}"
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
            settle_wait_steps = 0
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                settle_wait_steps += 1
                if settle_wait_steps % 100 == 0:
                    print(
                        f"[InsertionWrapperLegoPE] go_to_waypoint({label}) settle_wait "
                        f"steps={settle_wait_steps} vel={np.linalg.norm(obs['observation.velocity.cartesian'][:3]):.6f}"
                    )
                obs, *_ = self.env.step(np.zeros(6))
        print(
            f"[InsertionWrapperLegoPE] go_to_waypoint({label}) done "
            f"final_xyz={obs['observation.state.cartesian'][:3]} "
            f"final_target_xyz={obs['observation.state.target'][:3]} "
            f"vel={np.linalg.norm(obs['observation.velocity.cartesian'][:3]):.6f}"
        )
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

    def validate_world_pose_transform(
        self, world_D_world_obj: np.ndarray, tag: str
    ) -> dict[str, float | bool]:
        rot = world_D_world_obj[:3, :3]
        finite_ok = bool(np.isfinite(world_D_world_obj).all())
        orth_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
        det_r = float(np.linalg.det(rot))
        valid = finite_ok and abs(det_r - 1.0) < 1e-2 and orth_err < 1e-2
        print(
            f"[InsertionWrapperLegoPE] {tag} transform valid={valid} "
            f"det={det_r:.6f} orth_err={orth_err:.6e}"
        )
        return {"valid": valid, "det_r": det_r, "orth_err": orth_err}

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
        if z_force_error < 0 and z_impedance_error > -self.z_force_clip:
            return max(
                z_force_error / self.z_force_k, -self.z_force_clip - z_impedance_error
            ), z_force_error
        return 0.0, z_force_error

    # ---- PE wrappers ----

    def _settle_for_pe(self):
        for _ in range(self.pose_estimation_settle_steps):
            self.obs, *_ = self.env.step(np.zeros(6))

    def _estimate_lego_world_both(self):
        """Returns (lavender_world_4x4, purple_world_4x4, details)."""
        self._settle_for_pe()
        depth_f32 = (
            self.obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
        )
        rgb = self.obs["observation.images.wrist_camera"]
        masks = self.pose_estimation_helper.segmenter.segment_lego(rgb)
        if "lavender" not in masks or "purple" not in masks:
            raise RuntimeError(
                f"segment_lego missing brick(s); got keys={list(masks)}"
            )
        cam_poses = {
            k: self.pose_estimation_helper.pose_estimator.estimate_lego(
                rgb, depth_f32, masks[k], k
            )
            for k in ("lavender", "purple")
        }
        if self.use_tracker:
            cam_poses = {
                k: self.pose_estimation_helper.pose_tracker.track_lego(
                    rgb, depth_f32, cam_poses[k], k
                )
                for k in cam_poses
            }
        tcp_cart = self.obs["observation.state.cartesian"]
        world_lav = self.pose_estimation_helper._compute_pose_in_world_frame(
            cam_poses["lavender"], tcp_cart
        )
        world_pur = self.pose_estimation_helper._compute_pose_in_world_frame(
            cam_poses["purple"], tcp_cart
        )
        details = {
            "rgb": np.copy(rgb),
            "depth": np.copy(self.obs["observation.images.wrist_depth_camera"]),
            "tcp_cart": np.copy(tcp_cart),
            "mask_lavender": masks["lavender"],
            "mask_purple": masks["purple"],
            "pose_cam_lavender": cam_poses["lavender"],
            "pose_cam_purple": cam_poses["purple"],
        }
        return world_lav, world_pur, details

    def _estimate_lego_world_single(self, brick: str):
        """Returns (world_4x4, details) for one of the bricks."""
        self._settle_for_pe()
        depth_f32 = (
            self.obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
        )
        rgb = self.obs["observation.images.wrist_camera"]
        masks = self.pose_estimation_helper.segmenter.segment_lego(rgb)
        if brick not in masks:
            raise RuntimeError(
                f"segment_lego did not return mask for {brick!r}; got keys={list(masks)}"
            )
        pose_cam = self.pose_estimation_helper.pose_estimator.estimate_lego(
            rgb, depth_f32, masks[brick], brick
        )
        if self.use_tracker:
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

    # ---- 6DoF orientation alignment for the lavender grasp ----

    def compute_alignment_rpy(self, world_D_world_lav: np.ndarray) -> np.ndarray:
        """Demo-relative orientation for the gripper at the grasp."""
        from scipy.spatial.transform import Rotation as _R

        world_R_demo = self.demo_w_D_w_lavender[:3, :3]
        world_R_est = world_D_world_lav[:3, :3]
        est_R_demo = world_R_est @ world_R_demo.T

        bias_R = euler_to_rot_matrix(0.0, self.alignment_pitch_bias, 0.0)
        biased = est_R_demo @ bias_R
        rotvec = _R.from_matrix(biased).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.alignment_clip_angle_rad and angle > 0.0:
            rotvec = rotvec * (self.alignment_clip_angle_rad / angle)
            biased = _R.from_rotvec(rotvec).as_matrix()
        return rot_matrix_to_euler_xyz(biased)

    # ---- viz dump helpers (best-effort) ----

    def _viz_save_pair(self, suffix: str, det_a: dict, det_b: dict, label: str):
        if self._pose_overlay_renderer is None:
            return
        try:
            paths = self._pose_overlay_renderer.save_pair(
                out_dir=self.pose_viz_dir,
                episode_idx=self._pe_episode_idx,
                rgb=det_a["rgb"],
                mask=det_a["mask"],
                pose_cam_coarse=det_a["pose_cam"],
                pose_cam_refined=det_b["pose_cam"],
                rgb_refined=det_b["rgb"],
                mask_refined=det_b["mask"],
                names=("lavender", "purple"),
            )
            print(f"[pose viz] saved {suffix}: {paths}")
        except Exception as e:
            print(f"[pose viz] {suffix} save_pair failed: {e}")

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

    # ---- reset ----

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            # Open gripper, drop brick, retreat back+up, re-close gripper.
            self.env.unwrapped.gripper.set_target(1.0)  # type: ignore
            time.sleep(2.0)
            self.obs, *_ = self.env.step(np.zeros(6))
            self.obs = self.go_delta(self.obs, [-0.04, 0.0, 0.04])
            self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
            time.sleep(1.0)

        print("homing...")
        self.env.unwrapped.gripper.set_target(1.0)  # type: ignore
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        time.sleep(2.0)

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self.first_reset = False
        self._pe_episode_idx += 1

        # ----- 1. wide PE of both bricks (overview pose) -----
        print("Moving to wide PE pose...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.wide_pe_pose_euler[:3],
            relative_pose_euler=None,
            distance_err=0.0005,
            is_via=False,
            label="wide_pe_pose",
        )
        wide_lav, wide_pur, wide_det = self._estimate_lego_world_both()
        check_wide_lav = self.validate_world_pose_transform(wide_lav, "wide_lavender")
        check_wide_pur = self.validate_world_pose_transform(wide_pur, "wide_purple")
        print(f"wide PE lavender pos = {wide_lav[:3, 3]}")
        print(f"wide PE purple   pos = {wide_pur[:3, 3]}")
        if self._pose_overlay_renderer is not None:
            try:
                paths = self._pose_overlay_renderer.save_pair(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=wide_det["rgb"],
                    mask=wide_det["mask_lavender"],
                    pose_cam_coarse=wide_det["pose_cam_lavender"],
                    pose_cam_refined=wide_det["pose_cam_purple"],
                    rgb_refined=wide_det["rgb"],
                    mask_refined=wide_det["mask_purple"],
                    names=("lavender", "purple"),
                )
                print(f"[pose viz] wide saved: {paths}")
            except Exception as e:
                print(f"[pose viz] wide save_pair failed: {e}")

        # ----- 2. coarse hover above lavender (apply 6DoF orientation) -----
        coarse_alignment_rpy = self.compute_alignment_rpy(wide_lav)
        coarse_R_applied = euler_to_rot_matrix(*coarse_alignment_rpy)
        print(f"coarse alignment rpy [deg] = {np.rad2deg(coarse_alignment_rpy)}")

        coarse_grasp_world = (
            wide_lav[:3, 3] + wide_lav[:3, :3] @ self.o_T_o_tcpgrasp_lavender
        )
        coarse_hover_world = (
            coarse_grasp_world + wide_lav[:3, :3] @ np.array([0.0, 0.0, 0.025])
        )
        current_xyz = self.obs["observation.state.cartesian"][:3]
        safe_hover_z = max(
            coarse_hover_world[2],
            self.min_coarse_hover_z,
            current_xyz[2] + self.coarse_hover_min_above_current,
        )
        if safe_hover_z != coarse_hover_world[2]:
            print(
                "[InsertionWrapperLegoPE] WARNING: clamping coarse hover z from"
                f" {coarse_hover_world[2]:.6f} to {safe_hover_z:.6f}"
                f" (current z={current_xyz[2]:.6f})"
            )
            coarse_hover_world[2] = safe_hover_z
        print(f"Moving to coarse hover above lavender = {coarse_hover_world}")
        self.obs = self.go_to_waypoint(
            self.obs,
            coarse_hover_world,
            relative_pose_euler=coarse_alignment_rpy,
            distance_err=0.0005,
            is_via=False,
            label="coarse_hover_lavender",
        )

        # ----- 3. refined PE (lavender only) -----
        refined_lav, refined_lav_det = self._estimate_lego_world_single("lavender")
        check_refined = self.validate_world_pose_transform(refined_lav, "refined_lavender")
        est_R = refined_lav[:3, :3]
        w_T_w_tcpgrasp = refined_lav[:3, 3] + est_R @ self.o_T_o_tcpgrasp_lavender

        # delta orientation (coarse -> refined) — applied at the descent hover
        refined_alignment_rpy = self.compute_alignment_rpy(refined_lav)
        refined_R = euler_to_rot_matrix(*refined_alignment_rpy)
        delta_R = refined_R @ coarse_R_applied.T
        delta_alignment_rpy = rot_matrix_to_euler_xyz(delta_R)
        print(
            f"refined alignment rpy [deg] = {np.rad2deg(refined_alignment_rpy)}, "
            f"delta-on-descent rpy [deg] = {np.rad2deg(delta_alignment_rpy)}"
        )
        self._viz_save_single("refined_lavender", refined_lav_det, "refined lavender")

        # pre-grasp hover at ~15 mm above grasp, with delta orientation applied
        hover_xyz = refined_lav[:3, 3] + est_R @ (
            self.o_T_o_tcpgrasp_lavender + np.array([0.0, 0.0, 0.015])
        )
        print(f"Moving to pre-grasp hover = {hover_xyz}")
        self.obs = self.go_to_waypoint(
            self.obs,
            hover_xyz,
            relative_pose_euler=delta_alignment_rpy,
            distance_err=0.0005,
            is_via=False,
            label="pre_grasp_hover_lavender",
        )

        # ----- 4. (optional) extra PE at descent hover -----
        if self.extra_pre_grasp_pe:
            try:
                final_lav, final_lav_det = self._estimate_lego_world_single("lavender")
                self.validate_world_pose_transform(final_lav, "final_lavender")
                est_R = final_lav[:3, :3]
                w_T_w_tcpgrasp = (
                    final_lav[:3, 3] + est_R @ self.o_T_o_tcpgrasp_lavender
                )
                self._viz_save_single("final_lavender", final_lav_det, "final lavender")
            except Exception as e:
                print(f"[InsertionWrapperLegoPE] extra pre-grasp PE failed: {e}")

        # ----- 5. descent + close gripper -----
        print(f"Descending to grasp = {w_T_w_tcpgrasp}")
        self.obs = self.go_to_waypoint(
            self.obs,
            w_T_w_tcpgrasp,
            relative_pose_euler=None,
            distance_err=0.0002,
            is_via=False,
            label="descent_to_grasp_lavender",
        )
        applied_alignment_rpy = np.asarray(refined_alignment_rpy, dtype=np.float64)

        if self.dry_run_skip_grasp:
            print("[dry_run_skip_grasp] skipping gripper close.")
        else:
            print("Grasping...")
            self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
            time.sleep(2.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        # ----- 6. lift, then move to diagonal hover above purple -----
        print("Lifting after grasp...")
        self.obs = self.go_delta(
            self.obs,
            [0.0, 0.0, self.after_grasp_lift_height_pe],
            None,
            distance_err=0.001,
            is_via=False,
        )

        # Undo the applied orientation so the wrist camera looks down for the
        # post-grasp PE — same trick as Siemens.
        applied_alignment_rot = euler_to_rot_matrix(*applied_alignment_rpy)
        undo_alignment_rpy = rot_matrix_to_euler_xyz(applied_alignment_rot.T)
        self.obs, *_ = self.env.step(
            np.array(
                [0.0, 0.0, 0.0, undo_alignment_rpy[0], undo_alignment_rpy[1], undo_alignment_rpy[2]]
            )
        )

        diag_hover = wide_pur[:3, 3] + self.diagonal_hover_above_purple_offset
        print(f"Moving to diagonal hover above purple = {diag_hover}")
        self.obs = self.go_to_waypoint(
            self.obs,
            diag_hover,
            relative_pose_euler=None,
            distance_err=0.0005,
            is_via=False,
        )

        # ----- 7. close-up PE of both bricks -----
        place_lav, place_pur, place_det = self._estimate_lego_world_both()
        self.validate_world_pose_transform(place_lav, "place_lavender_grasped")
        self.validate_world_pose_transform(place_pur, "place_purple")
        if self._pose_overlay_renderer is not None:
            try:
                paths = self._pose_overlay_renderer.save_pair(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=place_det["rgb"],
                    mask=place_det["mask_lavender"],
                    pose_cam_coarse=place_det["pose_cam_lavender"],
                    pose_cam_refined=place_det["pose_cam_purple"],
                    rgb_refined=place_det["rgb"],
                    mask_refined=place_det["mask_purple"],
                    names=("lavender", "purple"),
                )
                print(f"[pose viz] place saved: {paths}")
            except Exception as e:
                print(f"[pose viz] place save_pair failed: {e}")

        # ----- 8. compute insertion goal -----
        # The grasped lavender is rigidly attached to the EE. So the EE motion
        # required to place lavender on top of purple equals
        #   delta_world = (purple_world_t + place_z_offset) - lavender_grasped_world_t
        # which we then add to the current EE position.
        ee_now = self.obs["observation.state.cartesian"][:3].copy()
        delta_world = (
            place_pur[:3, 3]
            - place_lav[:3, 3]
            + np.array([0.0, 0.0, self.place_z_offset])
        )
        self.goal_position = ee_now + delta_world
        self.start_position = self.goal_position.copy()
        self.estimated_grasp_delta = (
            place_lav[:3, 3] - self.actual_grasp_position
        )
        print(f"goal_position (insertion target) = {self.goal_position}")
        print(f"estimated_grasp_delta = {self.estimated_grasp_delta}")

        # ----- 9. (optional) FT contact establishment at the start position -----
        if self.use_ft_controller:
            print("Establishing contact...")
            self.obs, *_ = self.env.step(np.zeros(6))
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            z_step, z_force_error = self.z_force_controller_dz(self.obs)
            while abs(z_force_error) > 0.1:
                delta_xy = (
                    self.start_position[:2]
                    - self.obs["observation.state.cartesian"][:2]
                )
                self.obs, *_ = self.env.step(
                    np.array([delta_xy[0], delta_xy[1], z_step, 0.0, 0.0, 0.0])
                )
                z_step, z_force_error = self.z_force_controller_dz(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta
        reset_info["reset.pose_estimation.wide.lavender.matrix"] = wide_lav
        reset_info["reset.pose_estimation.wide.purple.matrix"] = wide_pur
        reset_info["reset.pose_estimation.wide.lavender.valid"] = np.array(
            [check_wide_lav["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.wide.purple.valid"] = np.array(
            [check_wide_pur["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.refined.lavender.matrix"] = refined_lav
        reset_info["reset.pose_estimation.refined.lavender.valid"] = np.array(
            [check_refined["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.place.lavender.matrix"] = place_lav
        reset_info["reset.pose_estimation.place.purple.matrix"] = place_pur
        reset_info["reset.pose_estimation.alignment_rpy_applied"] = applied_alignment_rpy
        reset_info["reset.goal_position"] = self.goal_position.copy()

        print("Reset complete.")
        return self.obs, reset_info

    # ---- step ----

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        z_action = (
            self.z_force_controller_dz(self.obs)[0] if self.use_ft_controller else 0.0
        )
        action_xyz = np.array([action[0], action[1], z_action])

        # safety box around goal_position[:2]
        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action_xyz[:2] = delta_xy * self.safety_box_step_size / norm_xy

        full_action = np.concatenate((action_xyz, np.zeros(3)))
        self.obs, reward, terminated, truncated, info = self.env.step(full_action)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info
