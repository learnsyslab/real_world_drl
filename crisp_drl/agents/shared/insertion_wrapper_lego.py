"""Real-robot wrapper for the LEGO brick insertion task.

Mirrors InsertionWrapperSiemens structure exactly:
  - reset() handles all motion planning: homing, free-space approach (Poly7Planner),
    and contact establishment (Z force controller)
  - step() is purely RL (XY search) + Z force controller

LEGO task axes (same as sim):
  Z (index 2) = insertion — presses the brick DOWN into the fixed socket
  X (index 0) = search axis 1  (RL action[0])
  Y (index 1) = search axis 2  (RL action[1])

Free-space motion uses Poly7Planner + TrajectoryFollower (7th-order polynomial,
zero jerk at endpoints) instead of the Siemens I-controller.

Usage:
    from crisp_drl.agents.shared.insertion_wrapper_lego import InsertionWrapperRealLEGO
    from crisp_drl.agents.shared.insertion_env_config import LegoConfig
    env = InsertionWrapperRealLEGO(inner_env, alg_config=cfg, env_config=LegoConfig(), ...)
"""

import contextlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Wrapper, spaces

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import LegoConfig
from crisp_drl.motion_planning.trajectory_planner import (
    Poly7Planner,
    TrajectoryFollower,
)


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class InsertionWrapperRealLEGO(Wrapper):
    """Real-robot wrapper for LEGO insertion — mirrors InsertionWrapperSiemens.

    Free-space motion (reset phases ②③) uses Poly7Planner + TrajectoryFollower
    instead of the Siemens I-controller. All other phases match Siemens exactly.

    Parameters
    ----------
    env                          : inner wrapped env (SensorTareWrapper or below)
    alg_config                   : SAC Config
    env_config                   : LegoConfig (goal, grasp, home positions)
    grasp_randomisation_x_range  : (lo, hi) [m] — noise on grasp X offset
    grasp_randomisation_z_range  : (lo, hi) [m] — noise on grasp Z offset
    safety_box_radius            : max XY drift from goal allowed during RL [m]
    safety_box_step_size         : safety box correction step size [m]
    ft_force_target              : target Fz during insertion [N] (negative = pressing down)
    z_force_k                    : Fz error → dz gain [N / (m/step)]
    z_force_clip                 : max impedance error before force ctrl is disabled [m]
    contact_force_threshold      : |Fz error| below this = contact established [N]
    step_limit                   : max RL steps per episode
    minimal_start_goal_distance  : min XY distance from start to goal [m]
    approach_distance            : go_to_waypoint target tolerance [m]
    contact_max_steps            : safety bound for contact loop
    is_eval                      : tighter randomisation (no start offset)
    waypoints_before_insertion   : list of (position_3d, distance_err) tuples —
                                   free-space waypoints executed in reset() before contact
    poly7_a_limit                : Poly7Planner acceleration limit [m/s²]
    poly7_env_dt                 : Poly7Planner env step duration [s]
    """

    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: LegoConfig,
        grasp_randomisation_x_range: Tuple[float, float] = (-0.00175, 0.00175),
        grasp_randomisation_z_range: Tuple[float, float] = (-0.001, 0.001),
        safety_box_radius: float = 0.003,
        safety_box_step_size: float = 0.0004,
        ft_force_target: float = -3.0,          # [N] — pressing force target
        z_force_k: float = 3333.0,              # same as Siemens ft_controller_k
        z_force_clip: float = 0.003,            # same as Siemens x_force_clip [m]
        contact_force_threshold: float = 0.5,   # [N] — contact established threshold
        step_limit: int = 150,
        minimal_start_goal_distance: float = 0.002,
        approach_distance: float = 0.002,
        contact_max_steps: int = 500,
        is_eval: bool = False,
        waypoints_before_insertion: Optional[List[Tuple[np.ndarray, float]]] = None,
        poly7_a_limit: float = 1.0,             # [m/s²] — conservative for real robot
        poly7_env_dt: float = 0.066,            # [s] — 15 Hz control frequency
    ):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.alg_config = alg_config
        self.env_config = env_config

        self.grasp_position_ground_truth = np.copy(env_config.grasp_position_ground_truth)
        self.goal_position_ground_truth  = np.copy(env_config.goal_position_ground_truth)
        self.home_config                 = env_config.custom_home_position

        self.grasp_randomisation_x_range   = grasp_randomisation_x_range
        self.grasp_randomisation_z_range   = grasp_randomisation_z_range
        self.safety_box_radius             = safety_box_radius
        self.safety_box_step_size          = safety_box_step_size
        self.ft_force_target               = ft_force_target
        self.z_force_k                     = z_force_k
        self.z_force_clip                  = z_force_clip
        self.contact_force_threshold       = contact_force_threshold
        self.step_limit                    = step_limit
        self.minimal_start_goal_distance   = minimal_start_goal_distance
        self.approach_distance             = approach_distance
        self.contact_max_steps             = contact_max_steps
        self.is_eval                       = is_eval
        self.waypoints_before_insertion    = waypoints_before_insertion or []
        self.i_term_clip                   = 0.0009   # matches Siemens

        self.reset_grasp_delta   = np.zeros(3)
        self.goal_position       = np.zeros(3)
        self.start_position      = np.zeros(3)
        self.n_steps             = 0
        self.first_reset         = True

        # Free-space motion planning: Poly7Planner + TrajectoryFollower
        self._poly7    = Poly7Planner(a_limit=poly7_a_limit, env_dt=poly7_env_dt)
        self._follower = TrajectoryFollower(
            max_step=0.001,            # 1 mm/step max — same as sim
            max_iter_per_waypoint=200,
            verbose=True,
        )

        print("[InsertionWrapperRealLEGO] Eval mode:", is_eval)
        if waypoints_before_insertion:
            print(f"[InsertionWrapperRealLEGO] {len(waypoints_before_insertion)} free-space waypoints")

    # ------------------------------------------------------------------
    # Free-space motion (Poly7Planner — replaces Siemens go_to_waypoint)
    # ------------------------------------------------------------------

    def go_to_waypoint(
        self,
        obs: dict,
        position: np.ndarray,
        distance_err: float = 0.002,
        is_via: bool = True,
    ) -> dict:
        """Move to position using 7th-order polynomial trajectory.

        Replaces the Siemens I-controller go_to_waypoint() with Poly7Planner +
        TrajectoryFollower. Same interface: returns updated obs dict.

        Parameters
        ----------
        obs          : current observation dict
        position     : 3D target [m]
        distance_err : arrival tolerance [m]. Used for fine settling if not is_via.
        is_via       : if True, use looser tolerance (transit point); if False, settle
                       precisely to distance_err (terminal point).
        """
        target = np.asarray(position, dtype=float)
        current_xyz = obs["observation.state.cartesian"][:3]

        tolerance = 0.003 if is_via else distance_err
        waypoints = self._poly7.plan(current_xyz, target)
        # Update terminal waypoint tolerance
        if waypoints:
            waypoints[-1].tolerance = tolerance

        obs, result = self._follower.follow_sampled(self.env, obs, waypoints)

        if not result.reached_all:
            print(
                f"[InsertionWrapperRealLEGO] WARNING: go_to_waypoint did not reach "
                f"target (final dist={result.final_distance*1000:.1f}mm)"
            )
        return obs

    # ------------------------------------------------------------------
    # Z force controller (adapted from Siemens x_torque_controller_dx)
    # ------------------------------------------------------------------

    def z_force_controller_dz(self, obs: dict):
        """Z insertion force controller — adapted from Siemens x_torque_controller_dx().

        Uses Fz (index 2) directly instead of Y-torque via lever arm.
        Returns (dz, fz_error): dz = delta to apply [m], fz_error = signed force error [N].
        Negative dz = press down; positive dz = back off.
        """
        fz_sensed = obs["observation.state.sensors_bota_ft_sensor"][2]
        fz_error  = self.ft_force_target - fz_sensed   # negative target → neg error when pressing

        z_impedance_error = (
            obs["observation.state.target"][2] - obs["observation.state.cartesian"][2]
        )

        if fz_error > 0 and z_impedance_error < self.z_force_clip:
            dz = -min(fz_error / self.z_force_k, self.z_force_clip - z_impedance_error)
        elif fz_error < 0 and z_impedance_error > -self.z_force_clip:
            dz = -max(fz_error / self.z_force_k, -self.z_force_clip - z_impedance_error)
        else:
            dz = 0.0

        return dz, fz_error

    # ------------------------------------------------------------------
    # reset — phases ① ② ③ ④  (mirrors InsertionWrapperSiemens)
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        # ── Phase ①: Home robot ──────────────────────────────────────────────
        if self.first_reset:
            print("[InsertionWrapperRealLEGO] Homing (first reset)...")
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
            self.first_reset = False
        else:
            print("[InsertionWrapperRealLEGO] Homing...")
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        if options is not None and options.get("last_reset", False):
            print("[InsertionWrapperRealLEGO] Last reset — skip to home.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        # ── Phase ②: Grasp randomisation ────────────────────────────────────
        grasp_dx = np.random.uniform(*self.grasp_randomisation_x_range)
        grasp_dz = np.random.uniform(*self.grasp_randomisation_z_range)
        self.reset_grasp_delta = np.array([grasp_dx, 0.0, grasp_dz])

        self.estimated_grasp_delta = self.reset_grasp_delta.copy()
        self.estimated_grasp_delta[0] += np.random.uniform(*self.grasp_randomisation_x_range)
        self.estimated_grasp_delta[2] += np.random.uniform(*self.grasp_randomisation_z_range)

        self.goal_position = np.copy(self.goal_position_ground_truth)
        self.goal_position += self.estimated_grasp_delta

        if not self.is_eval:
            corrected_goal_gt = self.goal_position_ground_truth + self.reset_grasp_delta
            self.start_position = corrected_goal_gt.copy()
            while (
                np.linalg.norm(self.start_position[:2] - corrected_goal_gt[:2])
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
        else:
            self.start_position = self.goal_position.copy()

        with _printoptions(precision=4):
            print(
                f"[InsertionWrapperRealLEGO] goal_gt={self.goal_position_ground_truth}, "
                f"goal_est={self.goal_position}, start={self.start_position}"
            )

        # ── Phase ③: Free-space waypoints (Poly7Planner) ────────────────────
        # Follow pre-insertion waypoints (e.g. transit → above socket).
        # Each entry: (position_3d, distance_err, is_via).
        for waypoint_position, dist_err in self.waypoints_before_insertion:
            waypoint_position = np.asarray(waypoint_position, dtype=float)
            waypoint_position = waypoint_position + self.estimated_grasp_delta
            print(f"[InsertionWrapperRealLEGO] → waypoint {waypoint_position.round(4)}")
            self.obs = self.go_to_waypoint(
                self.obs, waypoint_position, distance_err=dist_err, is_via=True
            )

        # Final approach to start position (terminal — settle precisely)
        print(f"[InsertionWrapperRealLEGO] → approach start {self.start_position.round(4)}")
        self.obs = self.go_to_waypoint(
            self.obs, self.start_position, distance_err=self.approach_distance, is_via=False
        )

        # ── Phase ④: Contact establishment (Z force controller) ─────────────
        # Mirrors Siemens contact loop: drive Z until |Fz_error| < threshold.
        print("[InsertionWrapperRealLEGO] Establishing contact...")
        self.obs, *_ = self.env.step(np.zeros(6))  # wait one step before tare
        self.env.tare_ft_sensor(self.obs)  # type: ignore  (SensorTareWrapper)

        dz, fz_error = self.z_force_controller_dz(self.obs)
        for _ in range(self.contact_max_steps):
            if abs(fz_error) <= self.contact_force_threshold:
                break
            # Hold XY at start_position during contact establishment (Siemens pattern)
            delta_xy = (
                np.clip(
                    self.start_position[:2] - self.obs["observation.state.cartesian"][:2],
                    -self.i_term_clip,
                    self.i_term_clip,
                )
                if np.linalg.norm(self.obs["observation.velocity.cartesian"]) < 0.001
                else np.zeros(2)
            )
            self.obs, *_ = self.env.step(
                np.array([delta_xy[0], delta_xy[1], dz, 0.0, 0.0, 0.0])
            )
            dz, fz_error = self.z_force_controller_dz(self.obs)
        print("[InsertionWrapperRealLEGO] Contact established.")

        # Re-initialise lower wrappers (resets prev_action/error in LastObservationWrapper)
        self.n_steps = 0
        self.obs, reset_info = self.env.reset()

        reset_info["reset.grasped.delta"]            = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"]  = self.estimated_grasp_delta
        reset_info["reset.goal_position.offset"]     = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )

        self.obs = self.add_perfect_action_to_obs(self.obs)
        print("[InsertionWrapperRealLEGO] Reset complete.")
        return self.obs, reset_info

    # ------------------------------------------------------------------
    # step — RL (XY) + Z force controller  (mirrors Siemens step)
    # ------------------------------------------------------------------

    def step(self, action):
        """RL step: action[0], action[1] control XY search.
        Z is driven by z_force_controller_dz() — never overridden by RL.
        Safety box clips XY exactly like Siemens clips YZ.
        """
        dz, _ = self.z_force_controller_dz(self.obs)
        full_action = np.array([action[0], action[1], dz, 0.0, 0.0, 0.0])

        # Safety box on XY — verbatim from InsertionWrapperSiemens (remapped YZ → XY)
        current_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy   = current_xy - self.goal_position[:2]
        delta_xy_clipped = np.clip(delta_xy, -self.safety_box_radius, self.safety_box_radius)
        if np.any(delta_xy_clipped != delta_xy):
            full_action[:2] = delta_xy_clipped - delta_xy

        self.obs, reward, terminated, truncated, info = self.env.step(full_action)
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def add_perfect_action_to_obs(self, obs: dict) -> dict:
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + self.reset_grasp_delta
        ) - obs["observation.state.cartesian"][:3]
        return obs
