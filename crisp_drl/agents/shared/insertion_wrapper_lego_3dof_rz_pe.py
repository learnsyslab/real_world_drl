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
from crisp_drl.agents.shared.insertion_env_config import LegoConfig, LegoConfig2x4
from crisp_drl.agents.shared.insertion_wrapper_s import rot_matrix_to_euler_xyz
from crisp_drl.envs.pose_estimation_helper import (
    PoseEstimationHelper,
    TCP_R_CAM,
    TCP_T_TCP_CAM,
    euler_to_rot_matrix,
)
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
        env_config: LegoConfig2x4,
        grasp_color: str = "lavender",
        target_color: str = "yellow",
        safety_box_radius: float = 0.003,
        safety_box_step_size: float = 0.0005,
        safety_box_angular_radius: float = np.deg2rad(3),
        safety_box_angular_step_size: float = np.deg2rad(3.0),
        goal_orientation_randomisation_angle: float = np.deg2rad(3),
        step_limit: int = 150,
        minimal_start_goal_distance: float = 0.003,
        minimal_start_goal_angle: float = np.deg2rad(2),
        is_eval: bool = False,
        use_ft_controller: bool = True,
        pose_viz_dir: Optional[str] = None,
        use_tracker: bool = False,
        workspace_box: tuple = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7)),
        pe_3dof: bool = False,
        use_gt_target_goal: bool = False,
        grasp_z_offset: float = 0.0,
        soft_rotation_k_rot: float = 150.0,
        soft_rotation_d_rot: float = 14.0,
        dynamic_rotation_stiffness: bool = True,
        pe_hand_z: bool = False,
        grasp_z_offset_pe: float = 0.0,
    ):
        super().__init__(env)
        self.alg_config = alg_config
        self.env_config = env_config
        self.grasp_color = grasp_color
        self.target_color = target_color

        # Pose targets sourced from alg_config (Config) — matches
        # InsertionWrapper3DoFRotZ and scripts/collect_data_real_3dof_rz.py so
        # the trained policy sees the same physical reference at eval as during
        # data collection. PE-specific calibration stays from env_config.
        self.home_config = alg_config.custom_home_position
        self.grasp_position_ground_truth = np.array(alg_config.grasp_position_ground_truth)
        self.goal_position_ground_truth = np.array(alg_config.goal_position_ground_truth)
        self.wide_pe_pose_euler = np.array(env_config.demo_goal_pose_estimation_euler)
        self.diagonal_hover_offset = np.array(env_config.diagonal_hover_above_purple_offset)
        self.place_z_offset = float(env_config.place_z_offset)
        self.place_xy_correction = np.array(getattr(env_config, "place_xy_correction", [0.0, 0.0]))
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
        # Additive Z offset on the final grasp height (3DOF branch). Positive
        # → gripper closes higher above the table. Useful at eval to give the
        # gripper some clearance for a soft re-grasp.
        self.grasp_z_offset = float(grasp_z_offset)
        # --hand mode: source grasp Z from PE estimate (refined / final) instead
        # of the fixed grasp_position_ground_truth[2] baseline. Flat-laying
        # brick assumption preserved by `_project_pose_to_3dof`. Active only
        # when pe_3dof=True.
        self.pe_hand_z = bool(pe_hand_z)
        self.grasp_z_offset_pe = float(grasp_z_offset_pe)
        if self.pe_hand_z:
            print(
                f"[InsertionWrapper3DoFRotZPE] --hand mode ON: grasp Z driven "
                f"by PE + offset {self.grasp_z_offset_pe:.5f} m"
            )

        # FT constants (same as InsertionWrapper3DoFRotZ)
        self.z_force_target = -0.7
        self.z_force_k = 2500
        self.z_force_clip = 0.003
        self.i_term_clip = 0.001 * 2

        # Per-episode FT log captured by _logged_env_step (one sample per
        # internal env.step), spanning the ENTIRE episode — reset (grasp,
        # PE, contact) + policy steps + snap_push. Eval reads via
        # pop_episode_ft_log() after termination. Cleared on every reset()
        # so an aborted episode doesn't leak into the next one.
        # ``_episode_ft_log_t`` holds per-sample timestamps in seconds since
        # ``_ft_log_t0`` (set at reset start). ``_episode_ft_log_phase``
        # holds a per-sample phase tag string ("reset"|"policy"|"snap_push")
        # for the plot to mark phase boundaries. ``_current_ft_phase`` is
        # mutated by the wrapper at known phase transitions.
        self._episode_ft_log: list = []
        self._episode_ft_log_t: list = []
        self._episode_ft_log_phase: list = []
        self._ft_log_t0: float = 0.0
        self._current_ft_phase: str = "reset"
        self._ft_logging_active: bool = False

        # reset / motion constants
        self.reset_lift_height = 0.030
        self.after_grasp_lift_height = 0.016
        # ``delta_z_push_reset`` is now repurposed as the SAFETY CAP on max
        # Z descent during the reseat push (was: impedance error threshold
        # of the position-based push). The actual contact force is
        # controlled by ``reset_reseat_force_n`` via z_force_controller_dz.
        self.delta_z_push_reset = 0.003
        # Legacy constants — no longer used by the force-controlled reseat
        # but kept for backward-compat (other callers reference them).
        self.delta_z_push_reset_step_size = 0.0008
        self.delta_z_push_reset_careful_threshold_distance = 0.003
        self.delta_z_push_reset_careful_threshold_velocity = 0.003
        # Force target for inter-episode reseat push (-N = press down).
        # ~3 N is plenty to seat a brick; the prior position-based push
        # peaked at K_z × 3 mm ≈ 7.5 N, then jumped further when the
        # gripper.home() squeezed the brick on calibration. With explicit
        # force control + a safety cap the peak is bounded near the target.
        self.reset_reseat_force_n = -3.0
        self.reset_reseat_timeout_s = 2.0
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

        self.pe_3dof = pe_3dof
        self.use_gt_target_goal = use_gt_target_goal

        # Dynamic Cartesian rotational stiffness — soft during reset (free-space
        # motions look smoother), restored to controller default before RL step()
        # so the policy sees its original tracking behavior. Original values
        # are queried lazily on the first reset (via ParametersClient).
        self.dynamic_rotation_stiffness = bool(dynamic_rotation_stiffness)
        self.soft_rotation_k_rot = float(soft_rotation_k_rot)
        self.soft_rotation_d_rot = float(soft_rotation_d_rot)
        self._rotation_stiffness_original: Optional[tuple[float, float]] = None
        self._in_soft_rotation_mode = False

        print(
            f"[InsertionWrapper3DoFRotZPE] __init__: eval={is_eval} "
            f"grasp_color={grasp_color} target_color={target_color} "
            f"use_tracker={use_tracker} pe_3dof={pe_3dof} "
            f"dynamic_rotation_stiffness={dynamic_rotation_stiffness} "
            f"soft(k={soft_rotation_k_rot}, d={soft_rotation_d_rot})"
        )

    # ------------------------------------------------------------------ #
    # Cartesian impedance stiffness control (free-space soft / RL hard)    #
    # ------------------------------------------------------------------ #

    def _parameters_client(self):
        try:
            return self.env.unwrapped.robot.cartesian_controller_parameters_client
        except AttributeError:
            return None

    def _set_rotation_stiffness(self, k_rot: float, d_rot: float, label: str = ""):
        client = self._parameters_client()
        if client is None:
            return False
        try:
            params = [
                ("task.k_rot_x", float(k_rot)),
                ("task.k_rot_y", float(k_rot)),
                ("task.k_rot_z", float(k_rot)),
                ("task.d_rot_x", float(d_rot)),
                ("task.d_rot_y", float(d_rot)),
                ("task.d_rot_z", float(d_rot)),
            ]
            client.set_parameters(params)
            print(
                f"[InsertionWrapper3DoFRotZPE] rotation stiffness {label}: "
                f"k_rot={k_rot}, d_rot={d_rot}"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[InsertionWrapper3DoFRotZPE] WARNING: set rotation stiffness "
                f"{label} failed ({exc})"
            )
            return False

    def _capture_rotation_stiffness_original(self) -> bool:
        """Query the controller for current k_rot/d_rot and cache as 'original'.

        Returns True if values were captured (or already cached).
        """
        if self._rotation_stiffness_original is not None:
            return True
        client = self._parameters_client()
        if client is None:
            return False
        try:
            client.wait_until_ready()
            vals = client.get_parameters(["task.k_rot_x", "task.d_rot_x"])
            if vals[0] is None or vals[1] is None:
                return False
            self._rotation_stiffness_original = (float(vals[0]), float(vals[1]))
            print(
                f"[InsertionWrapper3DoFRotZPE] cached original rotation stiffness: "
                f"k_rot={self._rotation_stiffness_original[0]}, "
                f"d_rot={self._rotation_stiffness_original[1]}"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[InsertionWrapper3DoFRotZPE] WARNING: could not query rotation "
                f"stiffness ({exc})"
            )
            return False

    def _enter_soft_rotation_mode(self):
        if not self.dynamic_rotation_stiffness:
            return
        if not self._capture_rotation_stiffness_original():
            return
        if self._set_rotation_stiffness(
            self.soft_rotation_k_rot, self.soft_rotation_d_rot, label="(soft, free-space)"
        ):
            self._in_soft_rotation_mode = True

    def _exit_soft_rotation_mode(self):
        if not self._in_soft_rotation_mode:
            return
        if self._rotation_stiffness_original is None:
            return
        orig_k, orig_d = self._rotation_stiffness_original
        if self._set_rotation_stiffness(orig_k, orig_d, label="(original, RL)"):
            self._in_soft_rotation_mode = False

    # ------------------------------------------------------------------ #
    # Episode-wide FT logging                                              #
    # ------------------------------------------------------------------ #

    def _logged_env_step(self, action):
        """Drop-in replacement for ``self.env.step(action)`` that also
        appends the resulting FT reading + timestamp + phase tag to the
        per-episode logs when logging is active. Returns the same 5-tuple
        as the inner step."""
        out = self.env.step(action)
        if self._ft_logging_active:
            try:
                ft = out[0].get("observation.state.sensors_bota_ft_sensor")
                if ft is not None:
                    self._episode_ft_log.append(
                        np.asarray(ft, dtype=np.float64).copy()
                    )
                    self._episode_ft_log_t.append(
                        time.perf_counter() - self._ft_log_t0
                    )
                    self._episode_ft_log_phase.append(self._current_ft_phase)
            except Exception:
                # Don't let logging failures break motion.
                pass
        return out

    def pop_episode_ft_log(self) -> list:
        """Return the accumulated per-episode FT log and clear all related
        logs (timestamps + phases). Caller (eval) owns the returned list.
        Backward-compat: returns the bare list of 6-arrays."""
        ft, _, _ = self.pop_episode_ft_log_with_t()
        return ft

    def pop_episode_ft_log_with_t(self) -> tuple[list, list, list]:
        """Return (ft_log, timestamps_log, phase_log) and clear them.
        ``ft_log`` is a list of 6-element ndarrays, ``timestamps_log`` is
        a list of floats (seconds since reset start), ``phase_log`` is a
        list of strings ("reset"|"policy"|"snap_push")."""
        ft = self._episode_ft_log
        t = self._episode_ft_log_t
        ph = self._episode_ft_log_phase
        self._episode_ft_log = []
        self._episode_ft_log_t = []
        self._episode_ft_log_phase = []
        return ft, t, ph

    # ------------------------------------------------------------------ #
    # Low-level movement helpers (mirroring InsertionWrapper3DoFRotZ)      #
    # ------------------------------------------------------------------ #

    def _step_zeros(self):
        return self._logged_env_step(np.zeros(6))

    def _step_translation(self, dxyz):
        action6 = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0])
        return self._logged_env_step(action6)

    def _step_yaw(self, drz):
        action6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, drz])
        return self._logged_env_step(action6)

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
        obs, *_ = self._logged_env_step(
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
                    obs, *_ = self._logged_env_step(
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

    def _yaw_error_to(self, current_rot_obs, target_rz):
        # obs[3:6] is Euler XYZ in radians; index [2] = yaw (z)
        current_rz = float(current_rot_obs[2])
        err = target_rz - current_rz
        return float(((err + np.pi) % (2 * np.pi)) - np.pi)

    def _current_rz(self) -> float:
        """Return current gripper Z-rotation (Euler z) in radians (world frame)."""
        return float(self.obs["observation.state.cartesian"][5])

    @staticmethod
    def _nearest_symmetric_yaw(brick_yaw: float, current_rz: float) -> float:
        """Return brick_yaw or brick_yaw+π — whichever is closest to current_rz.

        For 180°-symmetric objects (2x4 LEGO) this caps the required rotation at 90°.
        """
        def _wrap(a: float) -> float:
            return ((a + np.pi) % (2 * np.pi)) - np.pi

        yaw0 = _wrap(brick_yaw)
        yaw1 = _wrap(brick_yaw + np.pi)
        if abs(_wrap(yaw0 - current_rz)) <= abs(_wrap(yaw1 - current_rz)):
            return yaw0
        return yaw1

    def go_to_rotation_xyz(
        self,
        current_obs,
        target_rx=0.0,
        target_ry=0.0,
        target_rz=0.0,
        tol=np.deg2rad(0.3),
        max_step=np.deg2rad(0.4),
        accel_step=None,
        decel_gain=0.5,
        max_drive_iter=300,
        max_settle_iter=30,
    ):
        """Drive all three Euler angles to target values (default 0°).

        Per-axis smooth ramp (mirrors `go_to_rotation_z`): each cycle the
        per-axis step magnitude is the minimum of:
          - max_step:                cruise cap
          - prev + accel_step:       acceleration cap (ramp 0 → max_step)
          - max(decel_gain*|err|,    proportional braking near target,
                accel_step):        floored so we don't stall before tol
          - |err|:                   never overshoot
        Uses 180°-symmetric nearest target per axis so e.g. roll=-179° →
        target=+180° (1° move) rather than target=0° (179° move).
        """
        if accel_step is None:
            accel_step = max_step / 8.0
        obs = current_obs

        def _wrap(a):
            return ((a + np.pi) % (2 * np.pi)) - np.pi

        def _sym_target(target, current):
            t0 = _wrap(target)
            t1 = _wrap(target + np.pi)
            return t0 if abs(_wrap(t0 - current)) <= abs(_wrap(t1 - current)) else t1

        def _effective_targets(o):
            cur = np.array(o["observation.state.cartesian"][3:6], dtype=np.float64)
            raw = np.array([target_rx, target_ry, target_rz])
            return np.array([_sym_target(raw[i], cur[i]) for i in range(3)])

        def _rot_err(o, eff_targets):
            cur = np.array(o["observation.state.cartesian"][3:6], dtype=np.float64)
            return np.array([_wrap(eff_targets[i] - cur[i]) for i in range(3)])

        eff = _effective_targets(obs)
        err = _rot_err(obs, eff)
        print(
            f"[go_to_rotation_xyz] effective_targets={np.round(np.rad2deg(eff), 2)}° "
            f"initial_err={np.round(np.rad2deg(err), 2)}° "
            f"max_step={np.rad2deg(max_step):.2f}°/cyc "
            f"accel={np.rad2deg(accel_step):.2f}°/cyc² "
            f"raw_obs[3:6]={np.round(np.rad2deg(obs['observation.state.cartesian'][3:6]), 2)}"
        )
        n = 0
        prev_mag = np.zeros(3)
        while np.any(np.abs(err) > tol) and n < max_drive_iter:
            drpy = np.zeros(3)
            for i in range(3):
                if abs(err[i]) > tol:
                    mag = min(
                        max_step,
                        prev_mag[i] + accel_step,
                        max(decel_gain * abs(err[i]), accel_step),
                        abs(err[i]),
                    )
                    drpy[i] = np.sign(err[i]) * mag
                    prev_mag[i] = mag
                else:
                    prev_mag[i] = 0.0
            obs, *_ = self._logged_env_step(np.concatenate((np.zeros(3), drpy)))
            err = _rot_err(obs, eff)
            n += 1
        print(
            f"[go_to_rotation_xyz] done: n={n} final_err={np.round(np.rad2deg(err), 2)}° "
            f"raw_obs[3:6]={np.round(np.rad2deg(obs['observation.state.cartesian'][3:6]), 2)}"
        )
        if n >= max_drive_iter:
            print(f"[go_to_rotation_xyz] WARNING: max_drive_iter hit; residual={np.round(np.rad2deg(err), 3)}°")
        n_settle = 0
        while (
            np.linalg.norm(obs["observation.velocity.angular"]) > np.deg2rad(1.0)
            and n_settle < max_settle_iter
        ):
            obs, *_ = self._step_zeros()
            n_settle += 1
        return obs

    def go_to_rotation_z(
        self,
        current_obs,
        target_rz,
        tol=np.deg2rad(0.3),
        max_drive_iter=400,
        max_settle_iter=30,
        max_step=None,
        accel_step=None,
        decel_gain=0.5,
    ):
        """Smooth yaw move with slew-rate-limited accel + proportional decel.

        Per-cycle step magnitude is the minimum of:
          - max_step:                cruise cap (default = safety_box_angular_step_size)
          - prev + accel_step:       acceleration cap (ramp 0 → max_step)
          - max(decel_gain*|err|,    proportional braking near target,
                accel_step):        floored so we don't stall before tol
          - |err|:                   never overshoot
        """
        if max_step is None:
            max_step = self.safety_box_angular_step_size
        if accel_step is None:
            accel_step = max_step / 4.0   # ramp 0 → max_step over 4 cycles
        obs = current_obs
        raw_rot = obs["observation.state.cartesian"][3:6]
        rz_err = self._yaw_error_to(raw_rot, target_rz)
        print(
            f"[go_to_rotation_z] target={np.rad2deg(target_rz):.2f}° "
            f"initial_err={np.rad2deg(rz_err):.2f}° "
            f"max_step={np.rad2deg(max_step):.2f}°/cyc "
            f"accel={np.rad2deg(accel_step):.2f}°/cyc² "
            f"raw_obs[3:6]={np.round(np.rad2deg(raw_rot), 2)}"
        )
        n = 0
        prev_drz_mag = 0.0
        while abs(rz_err) > tol and n < max_drive_iter:
            drz_mag = min(
                max_step,
                prev_drz_mag + accel_step,
                max(decel_gain * abs(rz_err), accel_step),
                abs(rz_err),
            )
            drz = np.sign(rz_err) * drz_mag
            obs, *_ = self._step_yaw(drz)
            rz_err = self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz)
            prev_drz_mag = drz_mag
            n += 1
        raw_rot_after = obs["observation.state.cartesian"][3:6]
        print(
            f"[go_to_rotation_z] done: n_drive={n} final_err={np.rad2deg(rz_err):.2f}° "
            f"raw_obs[3:6]={np.round(np.rad2deg(raw_rot_after), 2)}"
        )
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

    def go_to_cartesian_rotation_z(
        self,
        current_obs,
        target_cartesian=None,
        target_rz=None,
        pos_tol=0.002,
        rz_tol=np.deg2rad(0.3),
        fine_resolution=None,
        max_drive_iter=400,
        max_settle_iter=300,
        max_step_xy=None,
        max_step_rz=None,
        accel_step_rz=None,
        decel_gain_rz=0.5,
        label="go_to_cart_rotz",
    ):
        """Concurrent XY-Z translation + yaw rotation, time-balanced.

        XY: chunked per cycle, advancing the controller target by
            `max_step_xy` per axis (default auto-balanced to yaw duration).
            Uses controller-target-relative error so no overshoot.
        Yaw: same accel / cruise / proportional-decel ramp as go_to_rotation_z.

        Both axes progress in lock-step each cycle, so XY and yaw finish
        together visually instead of XY sprinting ahead while yaw ramps.

        Pass target_cartesian=None to hold XY, or target_rz=None to hold yaw.
        """
        obs = current_obs
        do_xy = target_cartesian is not None
        do_rz = target_rz is not None
        if not do_xy and not do_rz:
            return obs
        if max_step_rz is None:
            max_step_rz = self.safety_box_angular_step_size
        if accel_step_rz is None:
            accel_step_rz = max_step_rz / 4.0
        if do_xy:
            target_cartesian = np.asarray(target_cartesian, dtype=np.float64)

        cur_xyz = obs["observation.state.cartesian"][:3]
        cur_rpy = obs["observation.state.cartesian"][3:6]
        pos_err_vec = (target_cartesian - cur_xyz) if do_xy else np.zeros(3)
        rz_err = self._yaw_error_to(cur_rpy, target_rz) if do_rz else 0.0

        # Auto-balance XY chunk size to yaw drive duration so both axes
        # finish at roughly the same cycle. When only XY (no yaw), default
        # to ~5 mm/cyc — small enough to look smooth, large enough to be
        # quick. Floor at 1 mm/cyc to avoid pathologically long moves.
        if max_step_xy is None and do_xy:
            pos_err_norm = float(np.linalg.norm(pos_err_vec))
            if do_rz and abs(rz_err) > rz_tol:
                est_yaw_cycles = max(1.0, abs(rz_err) / max_step_rz)
                max_step_xy = float(np.clip(
                    pos_err_norm / est_yaw_cycles, 0.001, 0.010
                ))
            else:
                max_step_xy = 0.005
        if max_step_xy is None:
            max_step_xy = 0.005

        tgt_xyz_str = np.round(target_cartesian, 4).tolist() if do_xy else "hold"
        tgt_rz_str = f"{np.rad2deg(target_rz):.2f}°" if do_rz else "hold"
        print(
            f"[{label}] start: target_xyz={tgt_xyz_str} target_rz={tgt_rz_str} "
            f"initial pos_err={np.round(pos_err_vec*1e3, 2)} mm "
            f"rz_err={np.rad2deg(rz_err):.2f}° "
            f"max_step_xy={max_step_xy*1e3:.2f} mm/cyc "
            f"max_step_rz={np.rad2deg(max_step_rz):.2f}°/cyc"
        )

        # Drive loop: chunked XY (controller-target relative) + ramped yaw.
        # env uses relative actions: action[:3] is added to controller
        # target_pose.position each step. Driving against the controller
        # target (rather than measured pose) prevents accumulation past goal
        # regardless of how fast the robot tracks, and clipping dxyz to
        # max_step_xy paces XY so it visibly co-progresses with yaw ramp.
        n_drive = 0
        prev_drz_mag = 0.0
        while (
            ((do_xy and (np.linalg.norm(pos_err_vec) > pos_tol
                          or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001))
             or (do_rz and abs(rz_err) > rz_tol))
            and n_drive < max_drive_iter
        ):
            if do_xy:
                ctrl_target_xy = obs["observation.state.target"][:3]
                pos_err_ctrl = target_cartesian - ctrl_target_xy
                dxyz = np.clip(pos_err_ctrl, -max_step_xy, max_step_xy)
            else:
                dxyz = np.zeros(3)
            if do_rz and abs(rz_err) > rz_tol:
                drz_mag = min(
                    max_step_rz,
                    prev_drz_mag + accel_step_rz,
                    max(decel_gain_rz * abs(rz_err), accel_step_rz),
                    abs(rz_err),
                )
                drz = np.sign(rz_err) * drz_mag
                prev_drz_mag = drz_mag
            else:
                drz = 0.0
                prev_drz_mag = 0.0
            action6 = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, drz])
            obs, *_ = self._logged_env_step(action6)
            cur_xyz = obs["observation.state.cartesian"][:3]
            cur_rpy = obs["observation.state.cartesian"][3:6]
            if do_xy:
                pos_err_vec = target_cartesian - cur_xyz
            if do_rz:
                rz_err = self._yaw_error_to(cur_rpy, target_rz)
            n_drive += 1

        if n_drive >= max_drive_iter:
            print(
                f"[{label}] max_drive_iter hit; "
                f"pos_residual={np.linalg.norm(pos_err_vec)*1e3:.2f} mm "
                f"rz_residual={np.rad2deg(rz_err):.3f}°"
            )

        n_settle = 0
        while (
            (np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
             or np.linalg.norm(obs["observation.velocity.angular"]) > np.deg2rad(1.0))
            and n_settle < max_settle_iter
        ):
            obs, *_ = self._step_zeros()
            n_settle += 1

        if fine_resolution is not None and do_xy:
            err = target_cartesian - obs["observation.state.cartesian"][:3]
            n_fine = 0
            while np.linalg.norm(err) > fine_resolution and n_fine < max_settle_iter:
                obs, *_ = self._step_translation(
                    np.clip(err, -self.i_term_clip, self.i_term_clip)
                )
                err = target_cartesian - obs["observation.state.cartesian"][:3]
                n_fine += 1
            n_fine_settle = 0
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.0005
                and n_fine_settle < max_settle_iter
            ):
                obs, *_ = self._step_zeros()
                n_fine_settle += 1

        final_pos_err_mm = (
            np.linalg.norm(target_cartesian - obs["observation.state.cartesian"][:3]) * 1e3
            if do_xy else 0.0
        )
        final_rz_err_deg = (
            np.rad2deg(self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz))
            if do_rz else 0.0
        )
        print(
            f"[{label}] done: n_drive={n_drive} "
            f"final_pos_err={final_pos_err_mm:.2f} mm "
            f"final_rz_err={final_rz_err_deg:.2f}°"
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
        obs, *_ = self._logged_env_step(np.concatenate((delta_xyz, relative)))
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
                    obs, *_ = self._logged_env_step(
                        np.concatenate(
                            (np.clip(err + ctrl_err, -i_clip, i_clip) - ctrl_err, np.zeros(3))
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                ctrl_err = obs["observation.state.target"][:3] - obs["observation.state.cartesian"][:3]
            while np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err:
                obs, *_ = self._step_zeros()
        return obs

    def z_force_controller_dz(self, obs, z_force_target=None):
        """Returns (dz, z_force_error). Accepts an optional ``z_force_target``
        override (negative = EE pressing down on the surface, sensor reads
        the reaction). Defaults to ``self.z_force_target`` for backward
        compatibility with the contact-establishment loop. snap_push's
        reinforce press passes a larger magnitude for a firmer seat."""
        if z_force_target is None:
            z_force_target = self.z_force_target
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
        if self.pe_3dof:
            world_grasp = self._project_pose_to_3dof(world_grasp)
            world_target = self._project_pose_to_3dof(world_target)
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
        if self.pe_3dof:
            world_pose = self._project_pose_to_3dof(world_pose)
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

    def _project_pose_to_3dof(self, world_D_world_obj: np.ndarray) -> np.ndarray:
        """Flat-lego constraint: zero roll/pitch, keep only yaw + XYZ.

        Analogous to Siemens lock_orientation=True, but instead of locking to a
        fixed assumed_orientation, we preserve the yaw extracted from the estimated
        world-frame rotation and rebuild the rotation as pure Rz(yaw).  This
        ensures downstream code sees a consistent SE(3) pose where the object Z
        axis is aligned with the world Z axis.
        """
        yaw = float(Rotation.from_matrix(world_D_world_obj[:3, :3]).as_euler("xyz")[2])
        result = world_D_world_obj.copy()
        result[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        return result

    def _pose_world_to_cam(
        self, world_pose: np.ndarray, tcp_cart: np.ndarray
    ) -> np.ndarray:
        """Inverse of PoseEstimationHelper._compute_pose_in_world_frame.

        Maps a 4x4 world-frame object pose back into the wrist-cam frame
        using the same TCP→cam extrinsics. Used so the PE visualization
        overlay can render the 3DoF-projected pose instead of the raw
        cam-frame pose returned by FoundationPose.
        """
        world_R_tcp = euler_to_rot_matrix(*tcp_cart[3:6])
        world_R_cam = world_R_tcp @ TCP_R_CAM
        world_T_world_cam = tcp_cart[:3] + world_R_tcp @ TCP_T_TCP_CAM
        cam_R_world = world_R_cam.T
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_R_world @ world_pose[:3, :3]
        cam_pose[:3, 3] = cam_R_world @ (world_pose[:3, 3] - world_T_world_cam)
        return cam_pose

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

    def validate_brick_detection(self, details: dict, brick_color: str, expected_xy=None, is_single=False) -> dict:
        """Verify correct brick was detected during PE.

        Args:
            is_single: True if from _estimate_lego_world_single() (uses "mask" key)
                       False if from _estimate_lego_world_both() (uses "mask_{color}" key)

        Returns dict with:
          - valid: bool, whether detection looks correct
          - mask_area: number of pixels in mask
          - detection_reason: str, explanation
        """
        # Determine correct mask key based on source
        mask_key = "mask" if is_single else f"mask_{brick_color}"
        pose_key = "pose_cam" if is_single else f"pose_cam_{brick_color}"

        if mask_key not in details:
            return {"valid": False, "mask_area": 0, "detection_reason": f"No mask for {brick_color}"}

        mask = details[mask_key]
        mask_area = float(np.sum(mask > 0)) if mask is not None else 0

        result = {
            "valid": True,
            "mask_area": mask_area,
            "detection_reason": f"✓ {brick_color} mask detected with {mask_area:.0f} pixels",
        }

        # Check mask is reasonable size (not tiny noise, not entire image)
        total_pixels = mask.shape[0] * mask.shape[1] if mask.ndim == 2 else mask.shape[1] * mask.shape[2]
        pixel_pct = (mask_area / total_pixels) * 100

        if mask_area < 100:
            result["valid"] = False
            result["detection_reason"] = f"⚠️  {brick_color} mask too small ({mask_area:.0f} pixels, {pixel_pct:.1f}%)"
        elif pixel_pct > 50:
            result["valid"] = False
            result["detection_reason"] = f"⚠️  {brick_color} mask too large ({pixel_pct:.1f}%, likely wrong object)"
        else:
            result["detection_reason"] += f" ({pixel_pct:.1f}% of image)"

        # Note: Position checking should be done in verify_brick_identity_by_position() which
        # compares world-frame positions correctly. Skipping it here because pose_cam is
        # camera frame while expected_xy is world frame (mixing coordinate frames is meaningless).

        return result

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Brick identity verification (position-based)                        #
    # ------------------------------------------------------------------ #

    def verify_brick_identity_by_position(
        self, detected_pose_world, grasp_pos_world, target_pos_world, brick_name: str
    ) -> tuple[bool, str]:
        """Verify detected brick is actually the expected brick by comparing positions.

        Args:
            detected_pose_world: Detected brick pose in world frame (4x4)
            grasp_pos_world: Expected grasp (lavender) position in world frame (4x4)
            target_pos_world: Expected target (yellow) position in world frame (4x4)
            brick_name: The brick we expected to detect ("lavender" or "yellow")

        Returns:
            (is_correct_brick: bool, reason: str)
        """
        detected_xy = detected_pose_world[:2, 3]
        grasp_xy = grasp_pos_world[:2, 3]
        target_xy = target_pos_world[:2, 3]
        dist_to_grasp = np.linalg.norm(detected_xy - grasp_xy)
        dist_to_target = np.linalg.norm(detected_xy - target_xy)

        if brick_name == self.grasp_color:
            # Expected to detect lavender
            is_correct = dist_to_grasp < dist_to_target
            if is_correct:
                return True, f"✓ Detected position {dist_to_grasp*1000:.1f}mm from {brick_name} (correct)"
            else:
                return (
                    False,
                    f"⚠️  Detected position {dist_to_target*1000:.1f}mm from {self.target_color} "
                    f"(expected {brick_name} at {dist_to_grasp*1000:.1f}mm away) - WRONG BRICK DETECTED!",
                )
        else:
            # Expected to detect yellow/target
            is_correct = dist_to_target < dist_to_grasp
            if is_correct:
                return True, f"✓ Detected position {dist_to_target*1000:.1f}mm from {brick_name} (correct)"
            else:
                return (
                    False,
                    f"⚠️  Detected position {dist_to_grasp*1000:.1f}mm from {self.grasp_color} "
                    f"(expected {brick_name} at {dist_to_target*1000:.1f}mm away) - WRONG BRICK DETECTED!",
                )

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
                stage=suffix,
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
        """Run _reset_body with best-effort stiffness restoration on exit.

        Soft rotational stiffness is toggled inside _reset_body to cover only
        the no-load phases (gripper open, brick not held). The cleanup before
        env.reset() and everything after grasp closure run at original
        stiffness so the wrist has enough torque to overcome static friction.
        """
        # Start a fresh per-episode FT log. Everything from here until the
        # next reset (or pop_episode_ft_log call) is captured: grasp, PE,
        # contact establishment, policy steps, snap_push. Timestamp clock
        # starts at the top of reset so policy/snap_push are offset from
        # cycle-start.
        self._episode_ft_log = []
        self._episode_ft_log_t = []
        self._episode_ft_log_phase = []
        self._ft_log_t0 = time.perf_counter()
        self._current_ft_phase = "reset"
        self._ft_logging_active = True
        try:
            return self._reset_body(seed=seed, options=options)
        finally:
            # Belt-and-braces: if soft mode is somehow still active (e.g. an
            # exception escaped between enter and exit), restore originals so
            # the RL step() never runs at soft stiffness.
            self._exit_soft_rotation_mode()

    def _reset_body(
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
            self._current_ft_phase = "reset_lift"
            #self.obs = self.go_to_cartesian(
            #    self.obs, delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z])
            #)
            #self._current_ft_phase = "reset_release_open"
            self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
            #print("Opening gripper...")
            #time.sleep(2.0)
        else:
            # 1) lift (gripper may still be in contact at episode end)
            self.obs, *_ = self._step_zeros()
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            # Sequential lift → yaw=0: combined motion at this site fights the
            # lego-on-lego contact (brick still wedged after insertion) and the
            # yaw axis never converges. Lifting first frees the gripper, then
            # the yaw move runs in clear air.
            self._current_ft_phase = "reset_lift"
            self.obs = self.go_to_cartesian(
                self.obs,
                delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z]),
            )
            self.obs = self.go_to_rotation_z(self.obs, target_rz=0.0)

            # 3) go back over grasp stand
            # --hand: anchor the hover Z to the recorded grasp pose
            # (actual_grasp_position[2] + reset_lift_height), NOT to the
            # previous-episode end Z. Under OOD setups (taller stand / shim)
            # the policy can leave the gripper at a Z far from the grasp
            # stand; anchoring keeps the put-back consistent across episodes.
            # Default (no --hand): keep the legacy current-Z-minus-lift formula.
        #     if self.pe_hand_z:
        #         back_target = np.array([
        #             self.actual_grasp_position[0],
        #             self.actual_grasp_position[1],
        #             self.actual_grasp_position[2] + self.reset_lift_height,
        #         ])
        #         print(
        #             f"3DOF[--hand]: put-back hover anchored to actual grasp Z = "
        #             f"{self.actual_grasp_position[2]:.5f} + {self.reset_lift_height:.5f} "
        #             f"= {back_target[2]:.5f}"
        #         )
        #     else:
        #         back_target = np.array([
        #             self.actual_grasp_position[0],
        #             self.actual_grasp_position[1],
        #             self.obs["observation.state.cartesian"][2] - self.reset_lift_height,
        #         ])
        #     ### COMMENTED OUT FOR FULL # TASK PIPELINE
        #     self._current_ft_phase = "reset_putback_hover"
        #     self.obs = self.go_to_cartesian(
        #         self.obs,
        #         target_cartesian=back_target,
        #     )

        #     # 4) push down to re-seat brick — FORCE-CONTROLLED.
        #     # Targets ``reset_reseat_force_n`` on FT-Z (negative = press
        #     # down). Stops when (a) target force reached, (b) safety cap
        #     # ``delta_z_push_reset`` of descent hit, or (c) timeout. XY
        #     # correction toward ``actual_grasp_position`` runs every step
        #     # so the brick lands centred on its stand. Velocity gating
        #     # mirrors the contact-establishment loop to avoid windup.
        #     self._current_ft_phase = "reset_reseat_push"
        #     reseat_start_z = float(self.obs["observation.state.cartesian"][2])
        #     reseat_z_floor = reseat_start_z - self.delta_z_push_reset
        #     if self.use_ft_controller:
        #         try:
        #             self.env.tare_ft_sensor(self.obs)  # type: ignore
        #         except Exception as e:
        #             print(f"  [reseat] tare_ft_sensor failed ({e}); continuing.")
        #         print(
        #             f"Reseat push -Z to {self.reset_reseat_force_n:.1f} N "
        #             f"(max descent {self.delta_z_push_reset*1e3:.1f} mm, "
        #             f"timeout {self.reset_reseat_timeout_s:.1f} s)..."
        #         )
        #         t0 = time.time()
        #         while True:
        #             fz = float(self.obs["observation.state.sensors_bota_ft_sensor"][2])
        #             cur_z = float(self.obs["observation.state.cartesian"][2])
        #             if fz <= self.reset_reseat_force_n:
        #                 print(
        #                     f"  [reseat] reached target: fz={fz:.2f} N "
        #                     f"after {(time.time()-t0):.2f}s"
        #                 )
        #                 break
        #             if cur_z <= reseat_z_floor:
        #                 print(
        #                     f"  [reseat] hit max descent: descended "
        #                     f"{(reseat_start_z - cur_z)*1e3:.2f} mm, fz={fz:.2f} N"
        #                 )
        #                 break
        #             if time.time() - t0 > self.reset_reseat_timeout_s:
        #                 print(
        #                     f"  [reseat] timeout: fz={fz:.2f} N, descended "
        #                     f"{(reseat_start_z - cur_z)*1e3:.2f} mm"
        #                 )
        #                 break
        #             delta_xy = (
        #                 self.actual_grasp_position[0:2]
        #                 - self.obs["observation.state.cartesian"][0:2]
        #             )
        #             if np.linalg.norm(
        #                 self.obs["observation.velocity.cartesian"]
        #             ) > 0.0015:
        #                 # Hold XY but settle Z (zero Z action when moving fast).
        #                 self.obs, *_ = self._step_translation(
        #                     np.array([delta_xy[0], delta_xy[1], 0.0])
        #                 )
        #             else:
        #                 dz, _ = self.z_force_controller_dz(
        #                     self.obs, z_force_target=self.reset_reseat_force_n
        #                 )
        #                 self.obs, *_ = self._step_translation(
        #                     np.array([delta_xy[0], delta_xy[1], dz])
        #                 )
        #     else:
        #         # FT controller disabled — fall back to the legacy
        #         # impedance-based descent so behaviour is preserved.
        #         print(
        #             "Reseat push: FT controller disabled, using "
        #             "legacy impedance-based descent."
        #         )
        #         while (
        #             abs(
        #                 self.obs["observation.state.cartesian"][2]
        #                 - self.obs["observation.state.target"][2]
        #             )
        #             < self.delta_z_push_reset
        #         ):
        #             delta_xy = (
        #                 self.actual_grasp_position[0:2]
        #                 - self.obs["observation.state.cartesian"][0:2]
        #             )
        #             delta_z_step = (
        #                 -self.delta_z_push_reset_step_size
        #                 if (
        #                     self.obs["observation.velocity.cartesian"][2]
        #                     > -self.delta_z_push_reset_careful_threshold_velocity
        #                     or abs(
        #                         self.actual_grasp_position[2]
        #                         - self.obs["observation.state.cartesian"][2]
        #                     )
        #                     > self.delta_z_push_reset_careful_threshold_distance
        #                 )
        #                 else 0.0
        #             )
        #             self.obs, *_ = self._step_translation(
        #                 np.array([delta_xy[0], delta_xy[1], delta_z_step])
        #             )

        #     delta_z = abs(
        #         self.obs["observation.state.cartesian"][2]
        #         - self.obs["observation.state.target"][2]
        #     )
        #     self._current_ft_phase = "reset_release_open"
        #     self.obs, *_ = self._step_translation(np.array([0.0, 0.0, delta_z * 0.8]))
        #     # Gripper.home() runs a close→open calibration cycle that can
        #     # squeeze the brick and produce a transient FT spike. Tag a
        #     # dedicated phase and capture one env step AFTER the home +
        #     # sleep so the spike actually gets sampled and the plot draws
        #     # a vertical line exactly at the gripper-home event.
        #     self._current_ft_phase = "reset_gripper_home"
        #     self.env.unwrapped.gripper.home()  # type: ignore
        #     time.sleep(0.5)
        #     self.obs, *_ = self._step_zeros()
        #     self.n_since_last_home += 1

        # if self.n_since_last_home >= 4 or self.first_reset:
        #     if self.pe_3dof:
        #         print(
        #             f"n_since_last_home={self.n_since_last_home}, "
        #             f"first_reset={self.first_reset}, skipping home (3DOF mode, going to PE pose instead)."
        #         )
        #     else:
        #         print(
        #             f"n_since_last_home={self.n_since_last_home}, "
        #             f"first_reset={self.first_reset}, homing..."
        #         )
        #         self._current_ft_phase = "reset_home_joint"
        #         self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        #     self.n_since_last_home = 0
        #     self.first_reset = False

        # if options is not None and options.get("last_reset", False):
        #     print("Last reset, not going to start position.")
        #     return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self._pe_episode_idx += 1
        time.sleep(1.0)

        # Soft rotation stiffness for the no-load PE / setup motions only.
        # Inter-episode cleanup above (lift + push-back) runs at the controller's
        # original stiffness because the gripper is still holding a brick that
        # may be wedged — soft k_rot would lack torque to rotate. Everything
        # from here until grasp closure is gripper-open / free-space, where
        # softer stiffness gives smoother visible motion.
        self._enter_soft_rotation_mode()

        # ---- zero orientation before wide PE ----------------------------- #
        print("Zeroing orientation (symmetric) before wide PE...")
        self._current_ft_phase = "reset_zero_orient"
        self.obs = self.go_to_rotation_xyz(self.obs, target_rx=0.0, target_ry=0.0, target_rz=0.0)

        # ---- wide PE -------------------------------------------------- #
        # When --hand is on, raise the wide PE vantage by 10 cm so the wrist
        # camera has clearance for OOD setups where the brick may sit on a
        # taller stand / shim. The PE Z is then read directly from the depth
        # estimate, so the absolute height of the vantage isn't critical.
        wide_pe_target = self.wide_pe_pose_euler[:3].copy()
        if self.pe_hand_z:
            #wide_pe_target[2] += 0.10
            print(
                f"Moving to wide PE pose (--hand: raised by 10 cm) = {wide_pe_target}"
            )
        else:
            print(f"Moving to wide PE pose = {wide_pe_target}")
        self._current_ft_phase = "approach_pe_wide"
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=wide_pe_target,
            fine_resolution=0.0005,
            max_settle_iter=50,
        )
        print("Wide PE: estimating both bricks...")
        self._current_ft_phase = "pe_wide"
        wide_grasp, wide_target, wide_det = self._estimate_lego_world_both()
        check_wide_grasp = self.validate_world_pose_transform(wide_grasp, f"wide_{self.grasp_color}")
        check_wide_target = self.validate_world_pose_transform(wide_target, f"wide_{self.target_color}")
        print(f"wide PE {self.grasp_color} pos = {wide_grasp[:3, 3]}")
        print(f"wide PE {self.target_color} pos = {wide_target[:3, 3]}")

        # ---- Verify correct bricks detected in wide PE ---- #
        wide_grasp_check = self.validate_brick_detection(wide_det, self.grasp_color)
        wide_target_check = self.validate_brick_detection(wide_det, self.target_color)
        print(f"[BRICK VERIFY] Wide PE {self.grasp_color}: {wide_grasp_check['detection_reason']}")
        print(f"[BRICK VERIFY] Wide PE {self.target_color}: {wide_target_check['detection_reason']}")

        # Cross-check: bricks should be ~5-10cm apart
        brick_distance = np.linalg.norm(wide_target[:3, 3] - wide_grasp[:3, 3])
        print(f"[BRICK VERIFY] Distance between {self.grasp_color} and {self.target_color}: {brick_distance*100:.1f} cm")
        if brick_distance < 0.02 or brick_distance > 0.15:
            print(f"[BRICK VERIFY] ⚠️  WARNING: Brick distance {brick_distance*100:.1f} cm seems unusual!")

        # Attempt to save visualization (Gymnasium may block access to private methods)
        try:
            self._viz_save_pair("wide", wide_det, wide_det)
        except (AttributeError, TypeError):
            pass  # Visualization not available or blocked by Gymnasium wrapper

        if self.pe_3dof:
            # ---- 3DOF path: coarse X-Y hover (no orientation change) ---- #
            grasp_xy = wide_grasp[:3, 3][:2]
            current_xyz = self.obs["observation.state.cartesian"][:3]
            hover_z = max(current_xyz[2], self.min_coarse_hover_z)
            hover_3dof = np.array([grasp_xy[0], grasp_xy[1], hover_z])
            print(f"3DOF: moving to hover above grasp XY = {hover_3dof} + yaw → 0° (concurrent)")
            self._current_ft_phase = "approach_pe_refined"
            self.obs = self.go_to_cartesian_rotation_z(
                self.obs,
                target_cartesian=hover_3dof,
                target_rz=0.0,
                max_settle_iter=50,
                max_step_rz=np.deg2rad(1.0),
                label="hover3dof+yaw0",
            )

            # ---- 3DOF: close-up PE for refined X-Y + yaw ---------------- #
            print("3DOF: close-up PE for refined X-Y + yaw...")
            self._current_ft_phase = "pe_refined"
            refined_grasp, refined_det = self._estimate_lego_world_single(self.grasp_color)
            check_refined = self.validate_world_pose_transform(refined_grasp, f"refined_{self.grasp_color}")

            # ---- Verify correct brick detected in refined PE ---- #
            refined_grasp_check = self.validate_brick_detection(
                refined_det, self.grasp_color, expected_xy=wide_grasp[:3, 3][:2], is_single=True
            )
            print(f"[BRICK VERIFY] Refined PE {self.grasp_color}: {refined_grasp_check['detection_reason']}")

            # ---- Verify by position: is detected brick actually lavender or was wrong brick detected? ---- #
            identity_correct, identity_reason = self.verify_brick_identity_by_position(
                refined_grasp, wide_grasp, wide_target, self.grasp_color
            )
            print(f"[BRICK IDENTITY] {identity_reason}")
            if not identity_correct:
                print(f"[BRICK IDENTITY] 🔴 WRONG BRICK DETECTED!")
                print(f"[BRICK IDENTITY]    Root cause: SAM3 brightness ranking may have flipped at close-up range")
                print(f"[BRICK IDENTITY]    Likely reason: Lighting angle changed, yellow appears brighter at this range")
                print(f"[BRICK IDENTITY] ⚠️  This causes incorrect yaw extraction and XY position!")

            if not refined_grasp_check["valid"]:
                print(f"[BRICK VERIFY] ⚠️  WARNING: {self.grasp_color} detection may be incorrect!")
                print(f"[BRICK VERIFY]    Expected near wide PE position: {wide_grasp[:3, 3][:2]}")
                print(f"[BRICK VERIFY]    Detected position: {refined_grasp[:3, 3][:2]}")

            # Cross-check: position should not jump too far between wide and refined PE
            refined_wide_delta = np.linalg.norm(refined_grasp[:3, 3][:2] - wide_grasp[:3, 3][:2])
            if refined_wide_delta > 0.06:  # > 6cm between wide and refined
                print(f"[BRICK VERIFY] ⚠️  Position delta between wide and refined PE: {refined_wide_delta*100:.1f} cm")
                print(f"[BRICK VERIFY]    This correlates with wrong brick detection!")

            # Extract yaw using multiple methods to diagnose
            # SAM3's X-axis = brick short axis → subtract 90° to get long-axis yaw
            R = refined_grasp[:3, :3]
            raw_yaw = np.arctan2(refined_grasp[1, 0], refined_grasp[0, 0]) - np.pi / 2
            yaw_from_R = np.arctan2(R[1, 0], R[0, 0]) - np.pi / 2
            scipy_yaw = Rotation.from_matrix(R).as_euler('xyz')[2] - np.pi / 2

            print(f"[3DOF YAW DEBUG] Multiple extraction methods (corrected: SAM3 X=short axis, -90°):")
            print(f"  Method 1 - arctan2(T[1,0], T[0,0])-90°: {np.rad2deg(raw_yaw):8.2f}°")
            print(f"  Method 2 - arctan2(R[1,0], R[0,0])-90°: {np.rad2deg(yaw_from_R):8.2f}°")
            print(f"  Method 3 - scipy as_euler('xyz')[2]-90°: {np.rad2deg(scipy_yaw):8.2f}°")

            refined_xy = refined_grasp[:3, 3][:2]

            # Safety: cap refined XY to stay within reasonable range of wide PE position
            # (refined PE observes from hover above the brick, so should not deviate too much)
            max_refined_drift = 0.05  # Maximum 5cm drift from wide PE
            wide_xy = wide_grasp[:3, 3][:2]
            drift = np.linalg.norm(refined_xy - wide_xy)
            if drift > max_refined_drift:
                print(f"[3DOF SAFETY] ⚠️  Refined PE drifted {drift*100:.1f}cm from wide PE (max: {max_refined_drift*100:.1f}cm)")
                print(f"[3DOF SAFETY]    Wide XY: {wide_xy}, Refined XY: {refined_xy}")
                print(f"[3DOF SAFETY]    Using wide PE XY instead for safety")
                refined_xy = wide_xy

            print(f"3DOF: refined X-Y = {refined_xy}, raw brick yaw = {np.rad2deg(raw_yaw):.2f} deg")

            # ---- USE WIDE PE YAW INSTEAD OF REFINED (since refined may detect wrong brick) ---- #
            # Due to SAM3 brightness ranking flipping at close-up, refined PE often detects yellow instead of lavender
            # Solution: Use the yaw from wide PE which detected the correct brick
            wide_grasp_yaw = np.arctan2(wide_grasp[1, 0], wide_grasp[0, 0]) - np.pi / 2
            if not identity_correct:
                print(f"[3DOF YAW FIX] Wrong brick detected in refined PE!")
                print(f"[3DOF YAW FIX] Using yaw from wide PE ({np.rad2deg(wide_grasp_yaw):.2f}°) instead of refined PE ({np.rad2deg(raw_yaw):.2f}°)")
                raw_yaw = wide_grasp_yaw
            else:
                print(f"[3DOF YAW] Correct brick detected. Refined PE yaw: {np.rad2deg(raw_yaw):.2f}°")

            # Attempt to save visualization (Gymnasium may block access to private methods)
            try:
                self._viz_save_single("refined_lavender", refined_det, f"refined {self.grasp_color}")
            except (AttributeError, TypeError):
                pass  # Visualization not available or blocked by Gymnasium wrapper

            # ---- 3DOF DEBUG: Compare yaw sources and check rotation matrix ---- #
            wide_grasp_yaw = np.arctan2(wide_grasp[1, 0], wide_grasp[0, 0]) - np.pi / 2
            print(f"[3DOF DEBUG] wide_grasp yaw = {np.rad2deg(wide_grasp_yaw):.2f} deg")
            print(f"[3DOF DEBUG] refined_grasp yaw = {np.rad2deg(raw_yaw):.2f} deg")
            yaw_differential = np.rad2deg(np.abs(wide_grasp_yaw - raw_yaw))
            print(f"[3DOF DEBUG] ⚠️  Yaw differential: {yaw_differential:.2f}° (expect low if same brick)")
            if abs(yaw_differential - 180.0) < 5.0:
                print(f"[3DOF DEBUG] ⚠️  WARNING: yaws differ by ~180° - possible brick ambiguity or wrong segmentation!")
            print(f"[3DOF DEBUG] Rotation matrix R (refined_grasp):")
            print(R)
            euler_check = Rotation.from_matrix(R).as_euler('xyz')
            print(f"[3DOF DEBUG] Full Euler (XYZ): roll={np.rad2deg(euler_check[0]):.2f}°, "
                  f"pitch={np.rad2deg(euler_check[1]):.2f}°, yaw={np.rad2deg(euler_check[2]):.2f}°")
            if abs(euler_check[0]) > 0.1 or abs(euler_check[1]) > 0.1:
                print(f"[3DOF DEBUG] ⚠️  WARNING: Significant roll/pitch detected! "
                      f"Brick may not be flat or PE orientation is unexpected.")
            print(f"[3DOF DEBUG] Expected ~0° roll/pitch for flat-lying brick on table.")

            # ---- 3DOF: align gripper yaw (180° symmetry: pick nearest) --- #
            # Note: gripper is NOW at 0° due to initialization above
            cur_rz = self._current_rz()
            grasp_yaw = self._nearest_symmetric_yaw(raw_yaw, cur_rz)
            delta_rz = grasp_yaw - cur_rz
            print(
                f"3DOF: aligning gripper: current={np.rad2deg(cur_rz):.2f} deg, "
                f"raw={np.rad2deg(raw_yaw):.2f} deg, "
                f"target={np.rad2deg(grasp_yaw):.2f} deg, "
                f"delta={np.rad2deg(delta_rz):.2f} deg"
            )
            if abs(delta_rz) > np.deg2rad(90):
                print(f"[3DOF DEBUG] ⚠️  WARNING: Rotation > 90° needed ({np.rad2deg(delta_rz):.2f}°)!")
                print(f"[3DOF DEBUG]   This may indicate PE detected wrong brick or yaw is ambiguous.")
                print(f"[3DOF DEBUG]   Attempting to use 180° symmetric alternative...")
                alt_grasp_yaw = grasp_yaw + np.pi if grasp_yaw < 0 else grasp_yaw - np.pi
                alt_delta_rz = alt_grasp_yaw - cur_rz
                print(f"[3DOF DEBUG]   Alternative: target={np.rad2deg(alt_grasp_yaw):.2f}°, delta={np.rad2deg(alt_delta_rz):.2f}°")
                if abs(alt_delta_rz) < abs(delta_rz):
                    print(f"[3DOF DEBUG]   Using alternative (smaller rotation).")
                    grasp_yaw = alt_grasp_yaw
                    delta_rz = alt_delta_rz
            # ---- 3DOF: hover above refined X-Y, then descend ------------ #
            # Pre-compute hover/grasp targets from grasp_yaw (target value, not
            # measured rz) so we can drive yaw alignment + XY hover concurrently
            # via go_to_cartesian_rotation_z. grasp_xy depends on grasp_yaw target
            # via the TCP-offset rotation, so it is known before the move.
            if self.pe_hand_z:
                grasp_z = float(refined_grasp[2, 3]) + self.grasp_z_offset_pe
                print(
                    f"3DOF[--hand]: grasp_z from refined PE = "
                    f"{float(refined_grasp[2, 3]):.5f} + offset "
                    f"{self.grasp_z_offset_pe:.5f} → {grasp_z:.5f}"
                )
            else:
                grasp_z = self.grasp_position_ground_truth[2] + self.grasp_z_offset
            # Apply calibrated TCP offset (object frame → world frame via grasp yaw).
            # o_T_o_tcpgrasp_lavender encodes how far from the brick centroid the TCP
            # was in the demo; rotating it by grasp_yaw gives the world-frame correction.
            c, s = np.cos(grasp_yaw), np.sin(grasp_yaw)
            Rz_grasp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            offset_world = Rz_grasp @ self.o_T_o_tcpgrasp_lavender
            grasp_xy = refined_xy + offset_world[:2]
            print(
                f"3DOF: grasp XY offset: raw_centroid={refined_xy}, "
                f"offset_world_xy={offset_world[:2]*1000} mm, corrected={grasp_xy}"
            )
            hover_xyz = np.array([grasp_xy[0], grasp_xy[1], grasp_z + 0.015])
            w_T_w_tcpgrasp = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            print(
                f"3DOF: concurrent yaw align ({np.rad2deg(grasp_yaw):.2f}°) + "
                f"pre-grasp hover = {hover_xyz}"
            )
            self._current_ft_phase = "approach_grasp"
            self.obs = self.go_to_cartesian_rotation_z(
                self.obs,
                target_cartesian=hover_xyz,
                target_rz=grasp_yaw,
                fine_resolution=0.0005,
                max_settle_iter=50,
                max_step_rz=np.deg2rad(0.5),
                accel_step_rz=np.deg2rad(0.5) / 8.0,
                label="grasp_yaw_align+hover",
            )
            print(f"3DOF: descending to grasp = {w_T_w_tcpgrasp}")
            self._current_ft_phase = "descend_grasp"
            self.obs = self.go_to_cartesian(self.obs, target_cartesian=w_T_w_tcpgrasp, fine_resolution=0.0002)

            # ---- 3DOF: final PE refinement right before gripper closure ---- #
            # Lift a few mm so the wrist camera has a clear view of the brick,
            # rerun PE, refine TCP xy + yaw, then re-descend. Caps protect
            # against PE jitter; if PE returns garbage we keep the existing pose.
            final_pe_lift = 0.000                 # m above grasp_z
            final_pe_max_xy_delta = 0.025         # cap correction at 5 mm
            final_pe_max_yaw_delta = np.deg2rad(25)
            print("3DOF: final PE refinement before grasp closure...")
            self._current_ft_phase = "pe_final"
            pe_view_xyz = np.array([
                self.obs["observation.state.cartesian"][0],
                self.obs["observation.state.cartesian"][1],
                grasp_z + final_pe_lift,
            ])
            self.obs = self.go_to_cartesian(
                self.obs, target_cartesian=pe_view_xyz,
                fine_resolution=0.0005, max_settle_iter=30,
            )
            try:
                final_grasp, final_det = self._estimate_lego_world_single(self.grasp_color)
                # Force flat-lego 3DoF projection (Z up, pure Rz(yaw)) on the
                # estimated world pose. _estimate_lego_world_single already does
                # this when pe_3dof=True; calling here is idempotent and makes
                # the assumption explicit at the consumer site.
                final_grasp = self._project_pose_to_3dof(final_grasp)
                # Mirror the projection into pose_cam so the viz overlay shows
                # the same flat-projected orientation as the world pose used
                # downstream (otherwise the overlay would render the raw,
                # tilted FoundationPose result).
                final_det["pose_cam"] = self._pose_world_to_cam(
                    final_grasp, final_det["tcp_cart"]
                )
                # Save visualization regardless of validation — most useful when PE jitters.
                try:
                    self._viz_save_single(
                        "final_lavender", final_det, f"final {self.grasp_color}"
                    )
                except (AttributeError, TypeError):
                    pass  # blocked by Gymnasium wrapper
                final_check = self.validate_world_pose_transform(
                    final_grasp, f"final_pe_{self.grasp_color}"
                )
                final_brick_check = self.validate_brick_detection(
                    final_det, self.grasp_color,
                    expected_xy=self.obs["observation.state.cartesian"][:2],
                    is_single=True,
                )
                print(
                    f"[BRICK VERIFY] Final PE {self.grasp_color}: "
                    f"{final_brick_check['detection_reason']}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"3DOF: final PE failed ({exc}) — skipping refinement.")
                final_check = {"valid": False}
                final_brick_check = {"valid": False}

            if final_check.get("valid") and final_brick_check.get("valid"):
                final_xy_centroid = final_grasp[:3, 3][:2]
                final_raw_yaw = (
                    np.arctan2(final_grasp[1, 0], final_grasp[0, 0]) - np.pi / 2
                )
                cur_rz_now = self._current_rz()
                final_yaw = self._nearest_symmetric_yaw(final_raw_yaw, cur_rz_now)
                cf, sf = np.cos(final_yaw), np.sin(final_yaw)
                Rz_final = np.array([[cf, -sf, 0], [sf, cf, 0], [0, 0, 1]])
                offset_world_final = Rz_final @ self.o_T_o_tcpgrasp_lavender
                final_grasp_xy = final_xy_centroid + offset_world_final[:2]
                cur_xy = self.obs["observation.state.cartesian"][:2]
                delta_xy = final_grasp_xy - cur_xy
                delta_yaw = final_yaw - cur_rz_now
                apply_xy = np.linalg.norm(delta_xy) <= final_pe_max_xy_delta
                apply_yaw = abs(delta_yaw) <= final_pe_max_yaw_delta
                print(
                    f"3DOF: final PE delta_xy={delta_xy*1000} mm "
                    f"(apply={apply_xy}), "
                    f"delta_yaw={np.rad2deg(delta_yaw):.2f} deg "
                    f"(apply={apply_yaw})"
                )
                # Concurrent XY + yaw correction at current hover Z, then
                # descend to grasp_z. Combined motion is smoother and faster
                # than the prior XY-then-yaw split.
                target_xy = final_grasp_xy if apply_xy else cur_xy
                current_z = self.obs["observation.state.cartesian"][2]
                if apply_xy or apply_yaw:
                    xy_hover_xyz = (
                        np.array([target_xy[0], target_xy[1], current_z])
                        if apply_xy else None
                    )
                    refine_target_rz = final_yaw if apply_yaw else None
                    print(
                        f"3DOF: refine concurrent XY={xy_hover_xyz} "
                        f"yaw={'%.2f°' % np.rad2deg(final_yaw) if apply_yaw else 'hold'}"
                    )
                    self.obs = self.go_to_cartesian_rotation_z(
                        self.obs,
                        target_cartesian=xy_hover_xyz,
                        target_rz=refine_target_rz,
                        fine_resolution=0.0005 if apply_xy else None,
                        max_settle_iter=30,
                        max_step_rz=np.deg2rad(0.5),
                        label="pe_refine_xy+yaw",
                    )
                    if apply_yaw:
                        grasp_yaw = final_yaw  # update for applied_alignment_rpy below
                if self.pe_hand_z:
                    refined_z = float(final_grasp[2, 3]) + self.grasp_z_offset_pe
                    print(
                        f"3DOF[--hand]: refined descent Z from final PE = "
                        f"{float(final_grasp[2, 3]):.5f} + offset "
                        f"{self.grasp_z_offset_pe:.5f} → {refined_z:.5f}"
                    )
                else:
                    refined_z = self.grasp_position_ground_truth[2] + self.grasp_z_offset
                refined_grasp_xyz = np.array([
                    target_xy[0], target_xy[1], refined_z,
                ])
                print(f"3DOF: descending to refined grasp = {refined_grasp_xyz}")
                self._current_ft_phase = "descend_grasp_refined"
                self.obs = self.go_to_cartesian(
                    self.obs, target_cartesian=refined_grasp_xyz,
                    fine_resolution=0.0002,
                )
            else:
                print("3DOF: final PE invalid — re-descending to original target.")
                self._current_ft_phase = "descend_grasp_refined"
                self.obs = self.go_to_cartesian(
                    self.obs, target_cartesian=w_T_w_tcpgrasp,
                    fine_resolution=0.0002,
                )

            applied_alignment_rpy = np.array([0.0, 0.0, grasp_yaw])

        else:
            # ---- coarse hover above lavender -------------------------------- #
            coarse_alignment_rpy = self.compute_alignment_rpy(wide_grasp)
            # For 3DoF flows we only want to align yaw (Z). Zero roll/pitch so the
            # robot's Z axis remains parallel to the original (no roll/pitch change).
            coarse_alignment_yaw = coarse_alignment_rpy[2]
            coarse_alignment_rpy = np.array([0.0, 0.0, coarse_alignment_yaw])
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

            # ---- Verify correct brick detected in refined PE ---- #
            refined_grasp_check = self.validate_brick_detection(
                refined_det, self.grasp_color, expected_xy=wide_grasp[:3, 3][:2], is_single=True
            )
            print(f"[BRICK VERIFY] Refined PE {self.grasp_color}: {refined_grasp_check['detection_reason']}")
            if not refined_grasp_check["valid"]:
                print(f"[BRICK VERIFY] ⚠️  WARNING: {self.grasp_color} detection may be incorrect!")
                print(f"[BRICK VERIFY]    Expected near wide PE position: {wide_grasp[:3, 3][:2]}")
                print(f"[BRICK VERIFY]    Detected position: {refined_grasp[:3, 3][:2]}")

            # Cross-check: position should not jump too far between wide and refined PE
            refined_wide_delta = np.linalg.norm(refined_grasp[:3, 3][:2] - wide_grasp[:3, 3][:2])
            if refined_wide_delta > 0.06:  # > 6cm between wide and refined
                print(f"[BRICK VERIFY] ⚠️  Position delta between wide and refined PE: {refined_wide_delta*100:.1f} cm")
                print(f"[BRICK VERIFY]    This may indicate PE instability or different brick detected.")

            est_R = refined_grasp[:3, :3]
            w_T_w_tcpgrasp = refined_grasp[:3, 3] + est_R @ self.o_T_o_tcpgrasp_lavender

            refined_alignment_rpy = self.compute_alignment_rpy(refined_grasp)
            # Keep only yaw for refined alignment as well (prevent roll/pitch changes)
            refined_alignment_yaw = refined_alignment_rpy[2]
            refined_alignment_rpy = np.array([0.0, 0.0, refined_alignment_yaw])
            refined_R = euler_to_rot_matrix(*refined_alignment_rpy)
            delta_R = refined_R @ coarse_R_applied.T
            delta_alignment_rpy = rot_matrix_to_euler_xyz(delta_R)
            print(
                f"refined alignment rpy [deg] = {np.rad2deg(refined_alignment_rpy)}, "
                f"delta rpy [deg] = {np.rad2deg(delta_alignment_rpy)}"
            )
            # Attempt to save visualization (Gymnasium may block access to private methods)
            try:
                self._viz_save_single("refined_lavender", refined_det, f"refined {self.grasp_color}")
            except (AttributeError, TypeError):
                pass  # Visualization not available or blocked by Gymnasium wrapper

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

        # Restore original rotation stiffness BEFORE grasp closure. From this
        # point on, the gripper holds the lavender brick: post-grasp motions
        # (lift, diagonal hover, approach to start, FT contact) all run at the
        # controller's default k/d so the wrist has full torque against the
        # gripper load and the RL step() inherits unchanged dynamics.
        self._exit_soft_rotation_mode()

        # ---- grasp (shared) --------------------------------------------- #
        print("Grasping...")
        self._current_ft_phase = "grasp"
        time.sleep(1.0)
        self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
        time.sleep(3.0)
        self.obs, *_ = self._step_zeros()
        self.actual_grasp_position = np.copy(self.obs["observation.state.cartesian"][:3])

        # ---- lift + undo orientation + go to diagonal hover -------------- #
        diag_hover = wide_target[:3, 3] + self.diagonal_hover_offset
        print("Lifting after grasp...")
        self._current_ft_phase = "post_grasp_lift"
        if self.pe_3dof:
            if self.pe_hand_z:
                # --hand: under OOD setups the diagonal trajectory from a low
                # grasp Z to a high diag_hover Z can drag the brick laterally
                # before clearing the stand. Lift 30 mm in Z + 50 mm in +X
                # FIRST (with concurrent yaw → 0) so the brick swings clear of
                # the stand, then traverse to diag_hover.
                pre_lift_z = 0.030
                pre_lift_x = 0.050
                lift_target = self.actual_grasp_position + np.array(
                    [pre_lift_x, 0.0, pre_lift_z]
                )
                print(
                    f"3DOF[--hand]: lift +X={pre_lift_x*1e3:.0f} mm, "
                    f"+Z={pre_lift_z*1e3:.0f} mm + yaw→0 → {lift_target}"
                )
                self.obs = self.go_to_cartesian_rotation_z(
                    self.obs,
                    target_cartesian=lift_target,
                    target_rz=0.0,
                    max_step_rz=np.deg2rad(0.5),
                    label="post_grasp_lift_xz+yaw0",
                    fine_resolution=0.0005,
                )
                print(f"Then traversing to diagonal hover = {diag_hover}")
                self.obs = self.go_to_cartesian(
                    self.obs,
                    target_cartesian=diag_hover,
                    fine_resolution=0.0005,
                    max_settle_iter=50,
                )
            else:
                # Fuse post-grasp lift + yaw→0 + translation to diagonal hover
                # into ONE concurrent motion. The combined helper chunks XYZ and
                # ramps yaw per cycle, so Z rises in lock-step with the lateral
                # traverse (no intermediate stop at the lift waypoint). diag_hover
                # is above grasp_z, so Z increases throughout the move — the
                # brick clears the stand without an explicit lift phase.
                print(f"Concurrent post-grasp lift + yaw→0 + diagonal hover = {diag_hover}")
                self.obs = self.go_to_cartesian_rotation_z(
                    self.obs,
                    target_cartesian=diag_hover,
                    target_rz=0.0,
                    max_step_rz=np.deg2rad(0.5),
                    label="post_grasp_to_diag_hover+yaw0",
                    fine_resolution=0.0005,
                )
        else:
            # Non-pe_3dof: sequential lift then full-Euler undo (combined
            # helper does not support roll/pitch).
            self.obs = self.go_delta(
                self.obs,
                [0.0, 0.0, self.after_grasp_lift_height_pe],
                distance_err=0.001,
                is_via=False,
            )
            applied_alignment_rot = euler_to_rot_matrix(*applied_alignment_rpy)
            undo_alignment_rpy = rot_matrix_to_euler_xyz(applied_alignment_rot.T)
            self.obs, *_ = self._logged_env_step(
                np.array([0.0, 0.0, 0.0, undo_alignment_rpy[0], undo_alignment_rpy[1], undo_alignment_rpy[2]])
            )
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
        self._current_ft_phase = "pe_place"
        place_grasp, place_target, place_det = self._estimate_lego_world_both()
        self.validate_world_pose_transform(place_grasp, f"place_{self.grasp_color}")
        self.validate_world_pose_transform(place_target, f"place_{self.target_color}")

        # ---- Verify correct bricks detected in place PE ---- #
        place_grasp_check = self.validate_brick_detection(place_det, self.grasp_color)
        place_target_check = self.validate_brick_detection(place_det, self.target_color)
        print(f"[BRICK VERIFY] Place PE {self.grasp_color}: {place_grasp_check['detection_reason']}")
        print(f"[BRICK VERIFY] Place PE {self.target_color}: {place_target_check['detection_reason']}")

        # Attempt to save visualization (Gymnasium may block access to private methods)
        try:
            self._viz_save_pair("place", place_det, place_det)
        except (AttributeError, TypeError):
            pass  # Visualization not available or blocked by Gymnasium wrapper

        # ---- compute goal XY and goal_rotation_z ------------------------ #
        ee_now = self.obs["observation.state.cartesian"][:3].copy()
        place_grasp_xy = place_grasp[:3, 3][:2]
        place_target_xy = place_target[:3, 3][:2]
        print(f"[PLACEMENT DEBUG] Place PE positions:")
        print(f"  Grasp ({self.grasp_color}): {place_grasp_xy}")
        print(f"  Target ({self.target_color}): {place_target_xy}")
        print(f"  Current EE XY: {ee_now[:2]}")

        # ---- compute goal yaw first (needed to rotate tcp_to_lav below) --- #
        # goal yaw = physically-corrected yaw of target brick (SAM3 X-axis = short axis → -90°).
        # Pick nearest 180°-symmetric equivalent from current gripper yaw (near 0° here)
        # to minimise pre-RL rotation. Gripper will physically rotate to this angle before RL.
        raw_goal_yaw_uncorrected = np.arctan2(place_target[1, 0], place_target[0, 0])
        raw_goal_yaw_corrected = raw_goal_yaw_uncorrected - np.pi / 2
        current_rz_now = self._current_rz()
        self.goal_rotation_z = self._nearest_symmetric_yaw(raw_goal_yaw_corrected, current_rz_now)

        if self.use_gt_target_goal:
            self.goal_position = self.goal_position_ground_truth.copy()
            self.goal_position[2] = ee_now[2] + self.place_z_offset
            self.goal_rotation_z = 0.0
            # Compute PE-estimated goal for comparison (mirrors pe_3dof path)
            _tcp_to_lav = place_grasp_xy - ee_now[:2]
            _delta_rz = raw_goal_yaw_corrected - current_rz_now
            _R2 = np.array([[np.cos(_delta_rz), -np.sin(_delta_rz)],
                             [np.sin(_delta_rz),  np.cos(_delta_rz)]])
            _pe_goal_xy = place_target_xy - _R2 @ _tcp_to_lav + self.place_xy_correction
            _delta_xy = (self.goal_position[:2] - _pe_goal_xy) * 1000
            print(f"[PLACEMENT DEBUG] GT goal: XY={self.goal_position[:2]*1000} mm, rz=0.0 deg (GT override)")
            print(f"[PLACEMENT DEBUG] PE goal: XY={_pe_goal_xy*1000} mm, rz={np.rad2deg(raw_goal_yaw_corrected):.2f} deg")
            print(f"[PLACEMENT DEBUG] GT vs PE delta: X={_delta_xy[0]:.2f} mm  Y={_delta_xy[1]:.2f} mm")
        elif self.pe_3dof:
            # 3DOF: Goal XY places the lavender centroid over the yellow centroid at insertion.
            # tcp_to_lav is measured now (current_rz), but the gripper rotates to goal_rz before
            # RL starts. The brick rotates with the gripper, so rotate tcp_to_lav by
            # (goal_rz - current_rz) to get the correct TCP position at insertion time.
            tcp_to_lav = place_grasp_xy - ee_now[:2]
            delta_rz = self.goal_rotation_z - current_rz_now
            cos_d, sin_d = np.cos(delta_rz), np.sin(delta_rz)
            R2 = np.array([[cos_d, -sin_d], [sin_d, cos_d]])
            tcp_to_lav_at_goal = R2 @ tcp_to_lav
            goal_xy = place_target_xy - tcp_to_lav_at_goal + self.place_xy_correction
            self.goal_position = np.array([
                goal_xy[0],
                goal_xy[1],
                ee_now[2] + self.place_z_offset,
            ])
            _delta_xy = (self.goal_position[:2] - self.goal_position_ground_truth[:2]) * 1000
            print(f"[PLACEMENT DEBUG] PE goal: XY={self.goal_position[:2]*1000} mm, rz={np.rad2deg(self.goal_rotation_z):.2f} deg")
            print(f"[PLACEMENT DEBUG] GT goal: XY={self.goal_position_ground_truth[:2]*1000} mm, rz=0.0 deg")
            print(f"[PLACEMENT DEBUG] GT vs PE delta: X={_delta_xy[0]:.2f} mm  Y={_delta_xy[1]:.2f} mm")
            print(
                f"[PLACEMENT DEBUG]   tcp_to_lav now={tcp_to_lav*1000} mm"
                f"  at_goal={tcp_to_lav_at_goal*1000} mm"
                f"  delta_rz={np.rad2deg(delta_rz):.2f} deg"
                f"  xy_correction={self.place_xy_correction*1000} mm"
            )
        else:
            delta_world = (
                place_target[:3, 3]
                - place_grasp[:3, 3]
                + np.array([0.0, 0.0, self.place_z_offset])
            )
            self.goal_position = ee_now + delta_world
        estimated_grasp_delta = place_grasp[:3, 3] - self.actual_grasp_position

        print(
            f"goal_position = {self.goal_position}, "
            f"goal_rotation_z: raw_sam3={np.rad2deg(raw_goal_yaw_uncorrected):.2f} deg "
            f"corrected={np.rad2deg(raw_goal_yaw_corrected):.2f} deg "
            f"-> used={np.rad2deg(self.goal_rotation_z):.2f} deg "
            f"(corrected, nearest symmetric from {np.rad2deg(current_rz_now):.2f} deg)"
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
            self.start_rotation_z = self.goal_rotation_z  # pre-align gripper to yellow's physical yaw

        # ---- move to start XY + yaw concurrently ----------------------- #
        print(
            f"Moving to start XY={self.start_position[:2]} "
            f"+ yaw={np.rad2deg(self.start_rotation_z):.2f}° (concurrent)..."
        )
        self._current_ft_phase = "approach_goal"
        self.obs = self.go_to_cartesian_rotation_z(
            self.obs,
            target_cartesian=np.array([
                self.start_position[0],
                self.start_position[1],
                self.obs["observation.state.cartesian"][2],
            ]),
            target_rz=self.start_rotation_z,
            fine_resolution=0.0005,
            max_settle_iter=50,
            max_step_rz=np.deg2rad(1.0),
            label="approach_start_xy+yaw",
        )

        # ---- establish FT contact --------------------------------------- #
        self.n_steps = 0
        self.obs, *_ = self._step_zeros()
        if self.use_ft_controller:
            print("Establishing contact...")
            self._current_ft_phase = "contact"
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

        ee_at_rl_start = self.obs["observation.state.cartesian"][:3]
        delta_to_goal = (self.goal_position[:2] - ee_at_rl_start[:2]) * 1000
        print(
            f"[RL START] ee={ee_at_rl_start[:2]*1000} mm  "
            f"goal={self.goal_position[:2]*1000} mm  "
            f"delta={delta_to_goal} mm"
        )

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
        reset_info["reset.pe_3dof"] = self.pe_3dof

        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        # Reset prologue is over; subsequent _logged_env_step calls (from
        # the outer RL loop via step()) are the policy phase.
        self._current_ft_phase = "policy"
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

        self.obs, reward, terminated, truncated, info = self._logged_env_step(action6)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        if (truncated or terminated) and self.n_steps + 1 < self.step_limit:
            print(
                f"[InsertionWrapper3DoFRotZPE] WARNING: base env returned "
                f"{'truncated' if truncated else 'terminated'}=True on step "
                f"{self.n_steps + 1} (limit={self.step_limit})"
            )
        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # snap_push (for SuccessClassificationWrapper on E_SUCCESS_CLS)        #
    # ------------------------------------------------------------------ #

    def _force_press_ramp_and_hold(
        self,
        tag: str,
        force_target_n: float,
        max_descent_m: float,
        ramp_timeout_s: float,
        hold_s: float,
    ) -> None:
        """Reusable force-controlled press: phase-1 ramp to ``force_target_n``
        on FT-Z, phase-2 hold at force. Bounded by ``max_descent_m`` from
        the press starting Z (safety cap) and ``ramp_timeout_s`` for the
        ramp. Hold is skipped if the safety cap was hit in phase 1. Tares
        the FT sensor at entry so the target is relative to no-contact
        baseline. Velocity-gated to avoid windup while the arm settles.
        Tag is used in prints to identify the caller."""
        press_start_z = float(self.obs["observation.state.cartesian"][2])
        press_z_floor = press_start_z - max_descent_m
        try:
            self.env.tare_ft_sensor(self.obs)  # type: ignore
        except Exception as e:
            print(f"  [{tag} press] tare_ft_sensor failed ({e}); continuing.")

        def _press_step():
            if np.linalg.norm(
                self.obs["observation.velocity.cartesian"]
            ) > 0.0015:
                self.obs, *_ = self._step_zeros()
            else:
                dz, _ = self.z_force_controller_dz(
                    self.obs, z_force_target=force_target_n
                )
                self.obs, *_ = self._step_translation(np.array([0.0, 0.0, dz]))

        ramp_t0 = time.time()
        reached = False
        safety_hit = False
        while True:
            fz = float(self.obs["observation.state.sensors_bota_ft_sensor"][2])
            cur_z = float(self.obs["observation.state.cartesian"][2])
            if fz <= force_target_n:
                print(
                    f"  [{tag} press] phase 1 reached target: fz={fz:.2f} N "
                    f"after {(time.time()-ramp_t0):.2f}s"
                )
                reached = True
                break
            if cur_z <= press_z_floor:
                print(
                    f"  [{tag} press] phase 1 hit max descent: descended "
                    f"{(press_start_z - cur_z)*1e3:.2f} mm, fz={fz:.2f} N"
                )
                safety_hit = True
                break
            if time.time() - ramp_t0 > ramp_timeout_s:
                print(
                    f"  [{tag} press] phase 1 ramp timeout: fz={fz:.2f} N, "
                    f"descended {(press_start_z - cur_z)*1e3:.2f} mm"
                )
                break
            _press_step()

        if hold_s > 0.0 and not safety_hit:
            if not reached:
                print(
                    f"  [{tag} press] phase 2 holding (note: target force "
                    f"not reached in phase 1)..."
                )
            else:
                print(
                    f"  [{tag} press] phase 2 holding {hold_s:.1f}s at "
                    f"{force_target_n:.1f} N..."
                )
            hold_t0 = time.time()
            while time.time() - hold_t0 < hold_s:
                cur_z = float(self.obs["observation.state.cartesian"][2])
                if cur_z <= press_z_floor:
                    fz = float(
                        self.obs["observation.state.sensors_bota_ft_sensor"][2]
                    )
                    print(
                        f"  [{tag} press] phase 2 hit max descent during "
                        f"hold: descended {(press_start_z - cur_z)*1e3:.2f} mm, "
                        f"fz={fz:.2f} N"
                    )
                    break
                _press_step()
            fz_end = float(self.obs["observation.state.sensors_bota_ft_sensor"][2])
            cur_z_end = float(self.obs["observation.state.cartesian"][2])
            print(
                f"  [{tag} press] hold done: fz={fz_end:.2f} N, total "
                f"descent={(press_start_z - cur_z_end)*1e3:.2f} mm"
            )

    def snap_push(
        self,
        push_distance: float = 0.0030,
        pause_before: float = 1.0,
        pause_after: float = 1.0,
        push_force_n: float = -3.0,
        push_ramp_timeout_s: float = 1.2,
        push_hold_s: float = 0.5,
        reinforce: bool = False,
        reinforce_lift: float = 0.0130,
        reinforce_push: float = 0.0100,
        reinforce_post_lift: float = 0.0100,
        reinforce_push_offset_x: float = -0.003,
        reinforce_press_force_n: float = -4.5,
        reinforce_press_ramp_timeout_s: float = 1.0,
        reinforce_press_hold_s: float = 1.5,
    ) -> None:
        """Seat the brick after the success classifier fires.

        Phase 1 (always): force-controlled press to ``push_force_n`` on
        FT-Z, ramped over up to ``push_ramp_timeout_s`` with a hard descent
        cap of ``push_distance`` (which is now interpreted as a safety cap,
        not a position target). Then holds at force for ``push_hold_s`` so
        the brick seats rather than just touching and retreating. Falls
        back to the original position-target descent when
        ``use_ft_controller=False``.

        Phase 2 (``reinforce=True``): open / lift / close / force-press /
        lift / open / descend / re-grasp. Press is force-controlled with
        ``reinforce_press_force_n`` — see the reinforce branch for details.
        """
        # Tag all FT samples taken during snap_push so the plot can mark
        # the policy→snap_push boundary.
        self._current_ft_phase = "snap_push"
        print(f"[InsertionWrapper3DoFRotZPE] snap_push: pausing {pause_before}s...")
        t0 = time.time()
        while time.time() - t0 < pause_before:
            self.obs, *_ = self._step_zeros()

        # Phase 1 press: force-controlled ramp + hold (same pattern as the
        # reinforce press below). Falls back to the original position-based
        # descent when the FT controller is disabled.
        if not self.use_ft_controller:
            print(
                f"[InsertionWrapper3DoFRotZPE] snap_push: pushing down "
                f"{push_distance*1e3:.1f} mm (position-based; FT disabled)..."
            )
            self.obs = self.go_to_cartesian(
                self.obs,
                delta=np.array([0.0, 0.0, -push_distance]),
                fine_resolution=push_distance * 0.3,
            )
        else:
            print(
                f"[InsertionWrapper3DoFRotZPE] snap_push: pushing -Z to "
                f"{push_force_n:.1f} N (max descent {push_distance*1e3:.1f} mm, "
                f"ramp timeout {push_ramp_timeout_s:.1f} s, "
                f"hold {push_hold_s:.1f} s)..."
            )
            self._force_press_ramp_and_hold(
                tag="phase1",
                force_target_n=push_force_n,
                max_descent_m=push_distance,
                ramp_timeout_s=push_ramp_timeout_s,
                hold_s=push_hold_s,
            )

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: holding {pause_after}s...")
        t0 = time.time()
        while time.time() - t0 < pause_after:
            self.obs, *_ = self._step_zeros()
        print("[InsertionWrapper3DoFRotZPE] snap_push: done.")

        if not reinforce:
            return

        # Anchor reinforce sub-targets on the PE-estimated goal brick XY
        # (TCP-corrected, see goal_position assignment at reset) instead of the
        # policy-end EE XY. This recenters the closing/pressing/regrasp over
        # the actual placed brick, so the reinforce push lands on-center even
        # when the policy terminated a few mm off-axis.
        snap_pos = self.obs["observation.state.cartesian"][:3].copy()
        goal_xy_pe = np.asarray(self.goal_position[:2], dtype=np.float64).copy()
        regrasp_target = np.array(
            [float(goal_xy_pe[0]), float(goal_xy_pe[1]), float(snap_pos[2])]
        )
        xy_shift_mm = (goal_xy_pe - snap_pos[:2]) * 1e3
        print(
            f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — anchoring to PE "
            f"goal XY=[{goal_xy_pe[0]*1e3:.1f}, {goal_xy_pe[1]*1e3:.1f}] mm "
            f"(shift from snap XY = [{xy_shift_mm[0]:+.2f}, {xy_shift_mm[1]:+.2f}] mm), "
            f"snap Z={snap_pos[2]*1e3:.1f} mm"
        )

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.80)  # type: ignore
        time.sleep(1.0)

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — lifting {reinforce_lift*1e3:.1f} mm...")
        lift1_target = regrasp_target + np.array([reinforce_push_offset_x, 0.0, reinforce_lift]) # -0.003 m offset in X for better push
        t0 = time.time()
        while time.time() - t0 < 1.2:
            err = lift1_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — closing gripper...")
        self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
        time.sleep(1.8)

        # Force-controlled press (replaces the old position-based loop).
        if not self.use_ft_controller:
            print(
                "[InsertionWrapper3DoFRotZPE] snap_push: reinforce — FT "
                "controller disabled, skipping press."
            )
        else:
            print(
                f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — "
                f"pressing -Z to {reinforce_press_force_n:.1f} N "
                f"(max descent {reinforce_push*1e3:.1f} mm, "
                f"ramp timeout {reinforce_press_ramp_timeout_s:.1f} s, "
                f"hold {reinforce_press_hold_s:.1f} s)..."
            )
            self._force_press_ramp_and_hold(
                tag="reinforce",
                force_target_n=reinforce_press_force_n,
                max_descent_m=reinforce_push,
                ramp_timeout_s=reinforce_press_ramp_timeout_s,
                hold_s=reinforce_press_hold_s,
            )

        print(f"[InsertionWrapper3DoFRotZPE] snap_push: reinforce — lifting {reinforce_post_lift*1e3:.1f} mm...")
        lift2_target = regrasp_target + np.array([reinforce_push_offset_x, 0.0, reinforce_post_lift - (reinforce_push - reinforce_lift)])
        t0 = time.time()
        while time.time() - t0 < 1.4:
            err = lift2_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        # print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — opening gripper...")
        # self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
        # time.sleep(1.4)

        # print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — descending to re-grasp position...")
        # t0 = time.time()
        # while time.time() - t0 < 1.5:
        #     err = regrasp_target - self.obs["observation.state.cartesian"][:3]
        #     self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        # print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce — re-grasping lego...")
        # self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
        # time.sleep(2.0)
        # print("[InsertionWrapper3DoFRotZPE] snap_push: reinforce done.")

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
