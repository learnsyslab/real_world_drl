"""Self-contained sim stack — no torch, no crisp_gym, no ROS.

Contains everything needed to run the LEGO insertion task in MuJoCo:
  - ActionTimeStampWrapper  (replaces crisp_gym version)
  - LastObservationWrapper  (replaces crisp_gym version)
  - InsertionWrapperSim     (copied from env_wrappers.py, no torch dependency)
  - CustomTerminationWrapper
  - custom_sim_termination  (termination function)
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
from gymnasium import Wrapper, spaces

from crisp_drl.motion_planning.free_space_planner import (
    CartesianWaypoint,
    FreeSpaceMotionPlanner,
)


class ActionTimeStampWrapper(Wrapper):
    """Records the last action in the observation dict.

    crisp_gym's version also timestamps the action; here we only track
    what InsertionWrapperSim and the planners actually need.
    """

    def __init__(self, env):
        super().__init__(env)
        self._last_action = np.zeros(self.action_space.shape)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = np.zeros(self.action_space.shape)
        return obs, info

    def step(self, action):
        self._last_action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


class LastObservationWrapper(Wrapper):
    """Adds derived observation keys used by InsertionWrapperSim.

    Keys added every step:
        observation.error.cartesian          (3,) = target[:3] - cartesian[:3]
        observation.previous.action          (N,) = action from previous step
        observation.previous.error.cartesian (3,) = error from previous step
        observation.velocity.cartesian       (3,) = error - previous_error
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_error = np.zeros(3)
        self._prev_action = np.zeros(self.action_space.shape)

    def _augment(self, obs: dict, action=None) -> dict:
        cartesian = obs["observation.state.cartesian"][:3]
        target    = obs["observation.state.target"][:3]
        error     = target - cartesian

        obs["observation.error.cartesian"]          = error.astype(np.float32)
        obs["observation.previous.error.cartesian"] = self._prev_error.astype(np.float32)
        obs["observation.velocity.cartesian"]       = (error - self._prev_error).astype(np.float32)

        if action is not None:
            obs["observation.previous.action"] = np.asarray(action, dtype=np.float32)
        else:
            obs["observation.previous.action"] = self._prev_action.astype(np.float32)

        self._prev_error  = error.copy()
        if action is not None:
            self._prev_action = np.asarray(action, dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_error  = np.zeros(3)
        self._prev_action = np.zeros(self.action_space.shape)
        obs = self._augment(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment(obs, action)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# InsertionWrapperSim (copied from env_wrappers.py — no torch dependency)
# ---------------------------------------------------------------------------

def _random_point_in_ellipse(x_range, y_range):
    rho = np.random.random()
    phi = np.random.random() * 2 * np.pi
    x = np.sqrt(rho) * np.cos(phi) * (x_range[1] - x_range[0]) / 2 + (x_range[1] + x_range[0]) / 2
    y = np.sqrt(rho) * np.sin(phi) * (y_range[1] - y_range[0]) / 2 + (y_range[1] + y_range[0]) / 2
    return np.array([x, y])


_APPROACH_MAX_STEP = 0.001  # metres per step during GoToGoal approach in reset()


class InsertionWrapperSim(Wrapper):
    def __init__(
        self,
        env,
        config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        grasp_randomisation_mode="box",
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        z_step_size=0.00025,
        max_z_error_deviation=0.001,
        target_z_error=0.0025,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        is_eval=False,
        approach_distance=None,
    ):
        super().__init__(env)
        self.config = config
        self.goal_position_ground_truth = np.array([0.6, 0.0, 0.0])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode = grasp_randomisation_mode
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.z_step_size = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error = target_z_error
        self.step_limit = step_limit
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        self.approach_distance = approach_distance
        print("[InsertionWrapperSim] [__init__] Eval mode:", is_eval)
        if approach_distance is not None:
            print(f"[InsertionWrapperSim] [__init__] Approach distance: {approach_distance*1000:.1f} mm")

    def reset(self, *, seed=None, options=None):
        opts = dict(options) if options else {}
        forced_start_xy = opts.pop("forced_start_xy", None)
        forced_goal_xy  = opts.pop("forced_goal_xy",  None)

        if self.grasp_randomisation_mode == "box":
            grasp_randomisation_x = np.random.uniform(*self.grasp_randomisation_x_range)
            grasp_randomisation_z = np.random.uniform(*self.grasp_randomisation_z_range)
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_randomisation_x, grasp_randomisation_z = _random_point_in_ellipse(
                self.grasp_randomisation_x_range, self.grasp_randomisation_z_range
            )

        self.grasp_position = np.array([grasp_randomisation_x, 0.0, grasp_randomisation_z])
        self.goal_position = np.copy(self.goal_position_ground_truth)

        if forced_goal_xy is not None:
            # Zero X grasp offset so the forced goal IS the true insertion point.
            self.grasp_position[0] = 0.0
            self.goal_position[:2] = np.asarray(forced_goal_xy, dtype=float)
            self.goal_position[2] += self.grasp_position[2]
        else:
            goal_position_randomisation_xy = np.zeros(2)
            while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
                goal_position_randomisation_xy = np.random.uniform(
                    self.goal_position_randomisation_xy_range[0],
                    self.goal_position_randomisation_xy_range[1],
                    size=2,
                )
            self.goal_position[:2] += goal_position_randomisation_xy
            self.goal_position[0] += self.grasp_position[0]
            self.goal_position[2] += self.grasp_position[2]

        if forced_start_xy is not None:
            self.start_position = self.goal_position.copy()
            self.start_position[:2] = np.asarray(forced_start_xy, dtype=float)
        elif not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            while (
                np.linalg.norm(self.start_position[:2] - self.goal_position_ground_truth[:2])
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
        else:
            self.start_position = self.goal_position.copy()

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={"start_position": self.start_position, "grasp_position": self.grasp_position},
        )
        reset_info["reset.grasped.delta"] = self.grasp_position
        reset_info["reset.goal_position.offset"] = self.goal_position - (
            self.goal_position_ground_truth + self.grasp_position
        )
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0

        # Coarse approach — mirrors go_to_waypoint() in InsertionWrapperSiemens.
        # Runs GoToGoal steps inside reset() so the RL policy always starts
        # within approach_distance of the goal, regardless of start_position.
        if self.approach_distance is not None:
            self.obs = self._approach_goal()

        return self.obs, reset_info

    def _approach_goal(self):
        """GoToGoal loop executed during reset() — does not count toward n_steps.

        Moves the robot straight toward goal_xy in steps of _APPROACH_MAX_STEP
        until within approach_distance. No Z pressing during approach.
        Safety bound of 10 000 steps prevents infinite loops.
        """
        goal_xy = self.goal_position[:2]
        for _ in range(10_000):
            current_xy = self.obs["observation.state.cartesian"][:2]
            dist = np.linalg.norm(current_xy - goal_xy)
            if dist <= self.approach_distance:
                break
            delta = goal_xy - current_xy
            action_2d = np.clip(delta, -_APPROACH_MAX_STEP, _APPROACH_MAX_STEP)
            action_6d = np.array([action_2d[0], action_2d[1], 0.0, 0.0, 0.0, 0.0])
            self.obs, _, _, _, _ = self.env.step(action_6d)
            self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs

    def step(self, action):
        action = np.array([action[0], action[1], 0.0, 0.0, 0.0, 0.0])

        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)

        # Only press in Z once XY-aligned; during approach hold Z steady.
        if norm_xy <= self.safety_box_radius:
            z_error = self.obs["observation.error.cartesian"][2]
            if z_error > -self.target_z_error + self.max_z_error_deviation:
                action[2] -= self.z_step_size
            elif z_error < -self.target_z_error - self.max_z_error_deviation:
                action[2] += self.z_step_size

        if norm_xy > self.safety_box_radius:
            action[:2] = delta_xy * self.safety_box_step_size / norm_xy

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True
        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.grasp_position
        )
        return obs


# ---------------------------------------------------------------------------
# CustomTerminationWrapper
# ---------------------------------------------------------------------------

def _append_or_insert(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


class CustomTerminationWrapper(Wrapper):
    def __init__(self, env, termination_fn):
        super().__init__(env)
        self.termination_fn = termination_fn

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        termination = self.termination_fn(observation)
        if termination is not None:
            _append_or_insert(info, "custom_events", (time.time(), termination))
            terminated = True
        return observation, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Termination functions (copied from make_env to avoid its crisp_gym imports)
# ---------------------------------------------------------------------------

def custom_sim_termination(obs):
    if obs["observation.state.target"][2] < 0.125:
        print("E_FAIL (Z)")
        return "E_FAIL"

    fixed_box_pos = np.array([0.6, 0.0, 0.1198])
    moving_box_pos = obs["observation.state.moving_brick"]
    delta = np.abs(moving_box_pos - fixed_box_pos)
    err = np.abs(obs["observation.error.cartesian"])
    if delta[2] < 14e-3 and err[2] > 0.8e-3:
        if delta[0] < 1e-3 and delta[1] < 1e-3:
            return "E_SUCCESS"
        else:
            print("E_FAIL (Stuck)")
            return "E_FAIL"


# ---------------------------------------------------------------------------
# InsertionWrapperSimLEGO
# ---------------------------------------------------------------------------

class InsertionWrapperSimLEGO(Wrapper):
    """Simulation wrapper for the LEGO brick insertion task.

    Mirrors InsertionWrapperSiemens structure exactly:
      - reset() handles motion planning: go_to_waypoint (XY approach) + contact establishment (Z)
      - step() is purely RL (XY search) + Z impedance controller

    LEGO task axes:
      Z (index 2) = insertion — presses the moving brick DOWN into the fixed socket
      X (index 0) = search axis 1  (RL action[0])
      Y (index 1) = search axis 2  (RL action[1])

    Success condition (via CustomTerminationWrapper + custom_sim_termination):
      moving_brick within 1 mm XY of [0.6, 0.0, 0.1198] AND |Z error| > 0.8 mm
    """

    def __init__(
        self,
        env,
        config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        grasp_randomisation_mode="box",
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_step_size=0.0004,      # matches Siemens default
        z_step_size=0.00025,              # Z pressing increment per step [m]
        max_z_error_deviation=0.001,      # deadband around target Z error [m]
        target_z_error=0.0025,            # desired pressing depth below target [m]
        step_limit=150,
        minimal_start_goal_distance=0.003,
        approach_distance=0.003,          # go_to_waypoint stops when within this [m]
        contact_max_steps=500,            # safety bound for contact establishment loop
        is_eval=False,
        waypoints_before_insertion: Optional[List[CartesianWaypoint]] = None,
        use_ruckig: bool = False,
        use_poly7:  bool = False,
    ):
        super().__init__(env)
        self.config = config
        self.goal_position_ground_truth  = np.array([0.6, 0.0, 0.0])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode    = grasp_randomisation_mode
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.safety_box_radius     = safety_box_radius
        self.safety_box_step_size  = safety_box_step_size
        self.z_step_size           = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error        = target_z_error
        self.step_limit            = step_limit
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.approach_distance     = approach_distance
        self.contact_max_steps     = contact_max_steps
        self.is_eval               = is_eval
        self.waypoints_before_insertion = waypoints_before_insertion
        self._free_space_planner   = FreeSpaceMotionPlanner()
        self.action_space          = spaces.Box(-np.inf, np.inf, (2,))
        self.n_steps               = 0
        self._ft_offset            = np.zeros(6)
        # set in reset():
        self.grasp_position = np.zeros(3)
        self.goal_position  = np.zeros(3)
        self.start_position = np.zeros(3)
        self.home_xyz       = np.zeros(3)  # TCP at keyframe home (set in reset)
        # Ruckig XY approach + free-space (optional)
        self._use_ruckig = use_ruckig
        if use_ruckig:
            from crisp_drl.motion_planning.trajectory_planner import RuckigFollower
            from crisp_drl.motion_planning.free_space_planner import RuckigFreeSpacePlanner
            self._ruckig = RuckigFollower(dt=0.066, verbose=False)
            self._free_space_planner = RuckigFreeSpacePlanner(dt=0.066, verbose=True)
        # Poly7 free-space (optional) — 7th-order polynomial, zero jerk at endpoints
        if use_poly7:
            from crisp_drl.motion_planning.free_space_planner import Poly7FreeSpacePlanner
            from crisp_drl.motion_planning.trajectory_planner import Poly7Planner, TrajectoryFollower
            self._free_space_planner = Poly7FreeSpacePlanner(a_limit=2.0, verbose=True)
            self._poly7_planner  = Poly7Planner(a_limit=2.0)
            self._poly7_follower = TrajectoryFollower(max_step=0.001, verbose=False)
        print("[InsertionWrapperSimLEGO] Eval mode:", is_eval)
        print(f"[InsertionWrapperSimLEGO] approach_distance: {approach_distance*1000:.1f} mm")
        print(f"[InsertionWrapperSimLEGO] use_ruckig: {use_ruckig}  use_poly7: {use_poly7}")
        if waypoints_before_insertion:
            print(f"[InsertionWrapperSimLEGO] Free-space waypoints: {len(waypoints_before_insertion)}")

    # ------------------------------------------------------------------
    # Motion planning helpers (used in reset only)
    # ------------------------------------------------------------------

    def go_to_waypoint(self, obs, target_xy, distance_err=0.002):
        """GoToGoal approach to target_xy. Does not count toward n_steps.

        Sim equivalent of go_to_waypoint() in InsertionWrapperSiemens.
        Coarse phase only — no fine settling needed in MuJoCo.
        Z is held fixed throughout (no pressing during approach).

        When use_ruckig=True, delegates to RuckigFollower.follow_xy() for
        jerk-limited online trajectory generation instead of the constant-step loop.
        """
        if self._use_ruckig:
            obs = self._ruckig.follow_xy(
                self.env, obs, target_xy, distance_err=distance_err
            )
            obs = self.add_perfect_action_to_obs(obs)
            return obs

        target_xy = np.asarray(target_xy, dtype=float)
        for _ in range(10_000):
            current_xy = obs["observation.state.cartesian"][:2]
            delta = target_xy - current_xy
            if np.linalg.norm(delta) <= distance_err:
                break
            step_xy = np.clip(delta, -_APPROACH_MAX_STEP, _APPROACH_MAX_STEP)
            action_6d = np.zeros(6)
            action_6d[:2] = step_xy   # X, Y only — Z index 2 left at 0
            obs, _, _, _, _ = self.env.step(action_6d)
            obs = self.add_perfect_action_to_obs(obs)
        return obs

    # ------------------------------------------------------------------
    # Z insertion controller (used in reset phase ⑧ and step)
    # ------------------------------------------------------------------

    def z_impedance_controller_dz(self, obs):
        """Z insertion depth controller — sim equivalent of x_torque_controller_dx().

        Maintains the brick at target_z_error below the Cartesian impedance target,
        creating a controlled pressing force on the socket face.
        Returns the Z delta [m] to apply this step.
        """
        z_error = obs["observation.error.cartesian"][2]  # target_z - current_z
        if z_error > -self.target_z_error + self.max_z_error_deviation:
            return -self.z_step_size   # too high → press down
        elif z_error < -self.target_z_error - self.max_z_error_deviation:
            return self.z_step_size    # too low → back off
        return 0.0

    # ------------------------------------------------------------------
    # reset — phases ⑥ ⑦ ⑧  (mirrors InsertionWrapperSiemens)
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        opts = dict(options) if options else {}
        forced_start_xy = opts.pop("forced_start_xy", None)
        forced_goal_xy  = opts.pop("forced_goal_xy",  None)

        # ── Phase ⑥: sample grasp randomisation, goal position, start position ──
        if self.grasp_randomisation_mode == "box":
            grasp_x = np.random.uniform(*self.grasp_randomisation_x_range)
            grasp_z = np.random.uniform(*self.grasp_randomisation_z_range)
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_x, grasp_z = _random_point_in_ellipse(
                self.grasp_randomisation_x_range, self.grasp_randomisation_z_range
            )

        self.grasp_position = np.array([grasp_x, 0.0, grasp_z])
        self.goal_position  = np.copy(self.goal_position_ground_truth)

        if forced_goal_xy is not None:
            # Zero X grasp offset so the forced goal IS the true insertion point
            self.grasp_position[0] = 0.0
            self.goal_position[:2] = np.asarray(forced_goal_xy, dtype=float)
            self.goal_position[2] += self.grasp_position[2]
        else:
            goal_xy_noise = np.zeros(2)
            while np.linalg.norm(goal_xy_noise) < 0.001:
                goal_xy_noise = np.random.uniform(
                    self.goal_position_randomisation_xy_range[0],
                    self.goal_position_randomisation_xy_range[1],
                    size=2,
                )
            self.goal_position[:2] += goal_xy_noise
            self.goal_position[0]  += self.grasp_position[0]
            self.goal_position[2]  += self.grasp_position[2]

        if forced_start_xy is not None:
            self.start_position = self.goal_position.copy()
            self.start_position[:2] = np.asarray(forced_start_xy, dtype=float)
        elif not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                ) < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
        else:
            self.start_position = self.goal_position.copy()

        # When free-space waypoints are active, keep initial_dxy=[0,0] so MuJoCo
        # resets to the true keyframe home joints without any XY teleport.
        # initial_dxy = start_position[:2] - [0.6, 0.0], so passing [0.6, 0.0, ...]
        # gives initial_dxy=[0,0] → robot stays at keyframe-1 home.
        # Without waypoints, teleport to the randomised start_position as before.
        reset_start = (
            np.array([0.6, 0.0, 0.0])
            if self.waypoints_before_insertion
            else self.start_position
        )

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={"start_position": reset_start, "grasp_position": self.grasp_position},
        )
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0
        # Store home TCP so callers can drive back here after insertion.
        self.home_xyz = self.obs["observation.state.cartesian"][:3].copy()
        tcp_xyz = self.home_xyz
        print(f"[InsertionWrapperSimLEGO] TCP after reset: xyz={tcp_xyz.round(4)}")

        # ── Phase ⓪: free-space approach (Siemens phases ①-⑤ equivalent) ──
        # Follows waypoints_before_insertion from home/transport position to near socket.
        # Skipped when waypoints_before_insertion is None (insertion-only mode).
        if self.waypoints_before_insertion:
            print("[InsertionWrapperSimLEGO] Free-space approach...")
            self.obs = self._free_space_planner.execute(
                self.env, self.obs, self.waypoints_before_insertion
            )

        # ── Phase ⑦: go_to_waypoint — approach goal XY ──
        # Mirrors Siemens waypoints_after_grasp sequence.
        # RL policy always starts within approach_distance of the goal.
        print("[InsertionWrapperSimLEGO] Approaching goal XY...")
        self.obs = self.go_to_waypoint(
            self.obs, self.goal_position[:2], distance_err=self.approach_distance
        )

        # ── Phase ⑧: tare F/T sensor + establish Z contact ──
        # Mirrors Siemens tare_ft_sensor() + X contact loop.
        self._ft_offset = self.obs["observation.state.sensors_bota_ft_sensor"].copy()
        print("[InsertionWrapperSimLEGO] Establishing contact...")
        for _ in range(self.contact_max_steps):
            dz = self.z_impedance_controller_dz(self.obs)
            if dz == 0.0:
                break   # target pressing depth reached
            action_6d = np.zeros(6)
            action_6d[2] = dz   # Z only — XY held at goal by not moving
            self.obs, _, _, _, _ = self.env.step(action_6d)
            self.obs = self.add_perfect_action_to_obs(self.obs)

        reset_info["reset.grasped.delta"]        = self.grasp_position
        reset_info["reset.goal_position.offset"] = self.goal_position - (
            self.goal_position_ground_truth + self.grasp_position
        )
        return self.obs, reset_info

    # ------------------------------------------------------------------
    # step — RL (XY) + Z impedance controller  (mirrors Siemens step)
    # ------------------------------------------------------------------

    def step(self, action):   # action: [dx_rl, dy_rl]
        """RL step: action[0], action[1] control XY search.
        Z is driven exclusively by z_impedance_controller_dz().

        Safety box clips XY exactly like Siemens clips YZ:
        if the robot drifts outside safety_box_radius, override with correcting action.
        """
        dz = self.z_impedance_controller_dz(self.obs)

        full_action = np.zeros(6)
        full_action[0] = action[0]   # X: RL search
        full_action[1] = action[1]   # Y: RL search
        full_action[2] = dz          # Z: impedance controller — never overridden by RL

        # Safety box on XY — Siemens logic verbatim, remapped from YZ to XY
        current_xy   = self.obs["observation.state.cartesian"][:2]
        delta_xy     = current_xy - self.goal_position[:2]   # drift from goal centre
        delta_clipped = np.clip(delta_xy, -self.safety_box_radius, self.safety_box_radius)
        if np.any(delta_clipped != delta_xy):
            full_action[:2] = delta_clipped - delta_xy       # push back into box

        self.obs, reward, terminated, truncated, info = self.env.step(full_action)
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True
        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.grasp_position
        )
        return obs


# ---------------------------------------------------------------------------
# Sim env factory (ROS-free — importable without real robot dependencies)
# ---------------------------------------------------------------------------

def create_simulated_env_lego(
    mujid_config: dict,
    sac_config=None,
    is_eval: bool = False,
    pe_accuracy: float = 0.0015,
):
    """Sim env factory for the LEGO insertion task.

    Uses InsertionWrapperSimLEGO (Siemens-style: reset=approach+contact, step=RL+Z ctrl).
    No cameras, no ObservationFormatterWrapper — raw obs dict for planner-based testing
    and lightweight training without CUDA.

    Stack:
        MujidEnv → ActionTimeStampWrapper → LastObservationWrapper
          → InsertionWrapperSimLEGO → CustomTerminationWrapper

    Parameters
    ----------
    mujid_config  : MujidEnv config dict (initial_keyframe, live_view, etc.)
    sac_config    : algorithm Config object (episode_length, etc.); uses defaults if None
    is_eval       : tighter randomisation when True
    pe_accuracy   : pose estimation accuracy [m] — sets randomisation ranges
    """
    import mujid.env.env as mujid_env
    from crisp_drl.agents.shared.algorithm_config import Config

    if sac_config is None:
        sac_config = Config()

    mujid_config = dict(mujid_config)
    mujid_config["n_cameras"] = 0   # no cameras — no CUDA needed

    env = mujid_env.MujidEnv(config=mujid_config)
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapperSimLEGO(
        env,
        config=sac_config,
        grasp_randomisation_z_range=(
            (-pe_accuracy / 3 + 0.001, pe_accuracy / 3 + 0.001)
            if is_eval
            else (-pe_accuracy / 3 + 0.001 - 0.00025, pe_accuracy / 3 + 0.001 + 0.00025)
        ),
        grasp_randomisation_x_range=(
            (-pe_accuracy, pe_accuracy)
            if is_eval
            else (-pe_accuracy - 0.00025, pe_accuracy + 0.00025)
        ),
        safety_box_radius=2 * pe_accuracy + 0.001 if is_eval else 2 * pe_accuracy,
        goal_position_randomisation_xy_range=(
            -2 * pe_accuracy * 0.9,
            2 * pe_accuracy * 0.9,
        ),
        minimal_start_goal_distance=2 * pe_accuracy,
        step_limit=sac_config.episode_length if not is_eval else 2 * sac_config.episode_length,
        is_eval=is_eval,
        grasp_randomisation_mode="box",
        approach_distance=2 * pe_accuracy,
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination)
    return env
