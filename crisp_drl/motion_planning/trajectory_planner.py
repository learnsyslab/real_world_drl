"""Trajectory planning and following for Cartesian robot motion.

Step 1 of the motion planning framework: given two 3D Cartesian points,
generate a trajectory (list of Waypoints) and follow it using the sim's
Cartesian impedance controller.

Design principles:
  - TrajectoryPlanner  : pure geometry, zero env dependency.
                         Upgrade plan() to RRT/spline without touching the follower.
  - TrajectoryFollower : env interaction only. Works with raw MujidEnv —
                         no wrappers required. Portable to real robot by
                         subclassing and overriding _step().
  - Both classes are intentionally minimal for Step 1. Later steps will add
    obstacle avoidance, rotation control, and state-machine integration.

Usage:
    from crisp_drl.motion_planning.trajectory_planner import (
        TrajectoryPlanner, TrajectoryFollower
    )
    planner  = TrajectoryPlanner()
    follower = TrajectoryFollower()

    waypoints        = planner.plan(start_xyz, goal_xyz)
    obs, result      = follower.follow(env, obs, waypoints)

    print(f"Reached: {result.reached_all}, final dist: {result.final_distance*1000:.1f}mm")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """A single 3D Cartesian target with an arrival tolerance.

    Attributes
    ----------
    position  : (3,) array, world-frame XYZ in metres.
    tolerance : Euclidean distance [m] below which the waypoint is considered
                reached. Default 2 mm — works well for the impedance controller
                at 1 mm/step resolution.
    """
    position:  np.ndarray
    tolerance: float = 0.002   # metres

    def __post_init__(self):
        # Ensure position is always a float64 numpy array, regardless of input type
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Waypoint6D:
    """A single 6D Cartesian target: 3D position + 3D orientation.

    Orientation is stored as a unit quaternion [x, y, z, w] internally, which
    is the most numerically stable representation for SLERP interpolation.

    Use the ``from_axis_angle`` classmethod to construct directly from
    ``obs["observation.state.cartesian"]`` (which encodes orientation as
    axis-angle).

    Attributes
    ----------
    position    : (3,) array, world-frame XYZ in metres.
    orientation : (4,) unit quaternion [x, y, z, w].
    tolerance   : Position arrival threshold [m].  Default 2 mm.
    """
    position:    np.ndarray
    orientation: np.ndarray
    tolerance:   float = 0.002

    def __post_init__(self):
        self.position    = np.asarray(self.position,    dtype=float)
        self.orientation = np.asarray(self.orientation, dtype=float)
        n = np.linalg.norm(self.orientation)
        if n > 1e-9:
            self.orientation = self.orientation / n

    @classmethod
    def from_axis_angle(
        cls,
        position,
        axis_angle,
        tolerance: float = 0.002,
    ) -> "Waypoint6D":
        """Construct from a (3,) axis-angle vector (e.g. from obs["observation.state.cartesian"][3:6]).

        Parameters
        ----------
        position   : (3,) XYZ [m].
        axis_angle : (3,) axis-angle [rad] — same encoding as MuJoCo obs.
        tolerance  : Position arrival threshold [m].
        """
        quat = Rotation.from_rotvec(np.asarray(axis_angle, dtype=float)).as_quat()
        return cls(position=position, orientation=quat, tolerance=tolerance)


@dataclass
class FollowResult:
    """Structured result returned by TrajectoryFollower.follow().

    Attributes
    ----------
    reached_all    : True if every waypoint was reached within its tolerance.
    steps_taken    : Total number of env.step() calls made.
    final_position : TCP XYZ [m] after the last step.
    final_distance : Euclidean distance [m] from the last waypoint position.
    timed_out_at   : Index of the first waypoint that hit max_iter_per_waypoint,
                     or None if all waypoints were reached.
    """
    reached_all:    bool
    steps_taken:    int
    final_position: np.ndarray
    final_distance: float
    timed_out_at:   Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory planner  (pure geometry — no env import)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """Generates a list of Waypoints between two 3D Cartesian points.

    Current strategy: linear interpolation (straight line from start to goal).
    To upgrade to spline or RRT: replace plan() only — the follower is unchanged.

    Parameters
    ----------
    step_size          : Approximate distance between consecutive waypoints [m].
                         Smaller = smoother path but more env.step() calls.
                         Default 1 mm matches the follower's default max_step.
    waypoint_tolerance : Arrival threshold assigned to every generated waypoint [m].
                         The last waypoint uses the same tolerance — for tighter
                         final accuracy, the caller can override it after planning.
    """

    def __init__(
        self,
        step_size:          float = 0.001,   # 1 mm
        waypoint_tolerance: float = 0.002,   # 2 mm
    ):
        self.step_size          = step_size
        self.waypoint_tolerance = waypoint_tolerance

    def plan(
        self,
        start_xyz,
        goal_xyz,
        coarse_step_size: float = None,
        fine_zone: float = 0.02,
    ) -> List[Waypoint]:
        """Generate a straight-line trajectory from start to goal.

        Parameters
        ----------
        start_xyz        : array-like (3,), starting TCP position [m].
        goal_xyz         : array-like (3,), target TCP position [m].
        coarse_step_size : If given, the middle of the trajectory uses this
                           (larger) step size. Only the first and last fine_zone
                           metres use the fine step_size. Set to None for
                           uniform fine spacing throughout (default).
        fine_zone        : Length [m] of the fine-step region at each end of the
                           trajectory. Ignored when coarse_step_size is None.

        Returns
        -------
        List of Waypoint objects. Always ends exactly at goal_xyz.
        If the distance is smaller than step_size, a single waypoint is returned.

        Example
        -------
        >>> planner = TrajectoryPlanner(step_size=0.001)
        >>> wps = planner.plan([0.6, 0.0, 0.20], [0.6, 0.05, 0.17])
        >>> len(wps)   # ≈ 58 waypoints for ~58 mm distance
        58
        >>> wps2 = planner.plan([0.6, 0.0, 0.20], [0.6, 0.05, 0.17],
        ...                     coarse_step_size=0.01, fine_zone=0.02)
        >>> len(wps2)  # ≈ 6 waypoints (20 fine + 18 coarse + 20 fine → grouped)
        6
        """
        start = np.asarray(start_xyz, dtype=float)
        goal  = np.asarray(goal_xyz,  dtype=float)
        dist  = np.linalg.norm(goal - start)

        # Edge case: already at goal (or within one step)
        if dist < self.step_size:
            return [Waypoint(position=goal.copy(), tolerance=self.waypoint_tolerance)]

        # ── Uniform spacing (original behaviour) ─────────────────────────────
        if coarse_step_size is None or dist <= 2 * fine_zone:
            n_segments = max(1, int(dist / self.step_size))
            waypoints = []
            for t in np.linspace(0.0, 1.0, n_segments + 1)[1:]:
                pt = start + t * (goal - start)
                waypoints.append(
                    Waypoint(position=pt.copy(), tolerance=self.waypoint_tolerance)
                )
            return waypoints

        # ── Adaptive spacing: fine → coarse → fine ───────────────────────────
        direction = (goal - start) / dist
        waypoints = []

        def _add_segment(d_start: float, d_end: float, step: float):
            """Append waypoints between d_start and d_end (arc-length offsets)."""
            n = max(1, int((d_end - d_start) / step))
            for d in np.linspace(d_start, d_end, n + 1)[1:]:
                waypoints.append(
                    Waypoint(
                        position=(start + d * direction).copy(),
                        tolerance=self.waypoint_tolerance,
                    )
                )

        fine_z = min(fine_zone, dist / 2)   # clamp so zones don't overlap
        _add_segment(0.0,            fine_z,        self.step_size)      # near start
        _add_segment(fine_z,         dist - fine_z, coarse_step_size)    # middle
        _add_segment(dist - fine_z,  dist,          self.step_size)      # near end

        # Guarantee the final waypoint is exactly at goal (floating-point safety)
        waypoints[-1].position = goal.copy()

        return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory follower  (env interaction only)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryFollower:
    """Follows a list of Waypoints by sending delta actions to env.step().

    Requires only obs["observation.state.cartesian"][:3] (TCP XYZ), so it works
    directly with a raw MujidEnv — no wrappers needed.

    Motion strategy
    ───────────────
    follow() treats the entire waypoint list as a single arc-length-parameterised
    path and drives along it with a **raised-cosine velocity profile**:

        speed ∝ sin(π · t)    (t = 0 → 1 over the full trajectory)

    This gives smooth ease-in / ease-out with no jerk at waypoint boundaries.
    Each env.step() sends a direction-normalised delta (not per-axis clipped),
    so the robot tracks the straight-line path accurately even for diagonal moves.

    After the timed arc-length pass, a short P-controller correction loop drives
    to the final waypoint within its tolerance (handles impedance lag).

    go_to() is kept for single-target use (e.g. hold corrections). It also uses
    direction-normalised steps.

    Real-robot portability: subclass and override _step() to call the robot
    controller instead of env.step().

    Parameters
    ----------
    max_step              : Maximum Cartesian step magnitude per env.step() [m].
                            3 mm is the safe limit for the impedance controller
                            (clips errors to ±3 mm per control cycle).
    max_iter_per_waypoint : Safety cap for the final correction loop [steps].
    verbose               : Print one summary line per follow() call.
    """

    def __init__(
        self,
        max_step:              float = 0.001,   # 1 mm
        max_iter_per_waypoint: int   = 5_000,
        verbose:               bool  = True,
    ):
        self.max_step              = max_step
        self.max_iter_per_waypoint = max_iter_per_waypoint
        self.verbose               = verbose

    # ── low-level: single env step ───────────────────────────────────────────

    def _step(self, env, obs, action: np.ndarray):
        """Send one action to the environment. Override for real-robot use."""
        obs, _, _, _, _ = env.step(action)
        return obs

    # ── geometry helper ──────────────────────────────────────────────────────

    def _interpolate_path(
        self,
        points: list,
        arc_lengths: list,
        s: float,
    ) -> np.ndarray:
        """Return the position on a polyline at arc-length s.

        Parameters
        ----------
        points      : List of (3,) arrays — the polyline vertices.
        arc_lengths : Cumulative arc lengths at each vertex (same length as points).
        s           : Target arc-length value in [0, total_arc].
        """
        for i in range(len(arc_lengths) - 1):
            if s <= arc_lengths[i + 1]:
                seg_len = arc_lengths[i + 1] - arc_lengths[i]
                if seg_len < 1e-9:
                    return points[i + 1].copy()
                t = (s - arc_lengths[i]) / seg_len
                return points[i] + t * (points[i + 1] - points[i])
        return points[-1].copy()

    # ── mid-level: move to a single target ───────────────────────────────────

    def go_to(
        self,
        env,
        obs: dict,
        target_xyz,
        tolerance: float = 0.002,
    ) -> Tuple[dict, bool, int]:
        """P-controller toward a single 3D target (direction-normalised step).

        Parameters
        ----------
        env        : The MuJoCo environment (or any env with step() → obs dict).
        obs        : Current observation dict.
        target_xyz : (3,) target TCP position in world frame [m].
        tolerance  : Stop when Euclidean distance ≤ this value [m].

        Returns
        -------
        obs      : Updated observation after the last step.
        reached  : True if tolerance was achieved.
        n_steps  : Number of env.step() calls made.
        """
        target  = np.asarray(target_xyz, dtype=float)
        n_steps = 0

        for _ in range(self.max_iter_per_waypoint):
            current = obs["observation.state.cartesian"][:3]
            delta   = target - current
            dist    = np.linalg.norm(delta)

            if dist <= tolerance:
                return obs, True, n_steps

            # Direction-normalised step: moves in a straight line, never overshoots
            step = delta / dist * min(dist, self.max_step)

            action     = np.zeros(6)
            action[:3] = step
            obs = self._step(env, obs, action)
            n_steps += 1

        return obs, False, n_steps

    # ── high-level: follow a waypoint list ───────────────────────────────────

    def follow(
        self,
        env,
        obs: dict,
        waypoints: List[Waypoint],
    ) -> Tuple[dict, FollowResult]:
        """Follow a waypoint list with smooth arc-length interpolation.

        The entire path is treated as a single arc-length-parameterised polyline.
        A raised-cosine velocity profile drives speed from 0 → max → 0 over the
        trajectory, producing smooth ease-in / ease-out with no jerk at waypoint
        boundaries.  After the interpolation pass, a short correction loop
        closes any remaining gap to the final waypoint.

        Parameters
        ----------
        env       : The MuJoCo environment.
        obs       : Current observation dict.
        waypoints : Ordered list of Waypoint targets.

        Returns
        -------
        obs    : Updated observation after the final step.
        result : FollowResult with success flag, step count, final position, etc.
        """
        if not waypoints:
            pos = obs["observation.state.cartesian"][:3].copy()
            return obs, FollowResult(
                reached_all=True, steps_taken=0,
                final_position=pos, final_distance=0.0,
            )

        # ── Build arc-length parameterised polyline ───────────────────────────
        start   = obs["observation.state.cartesian"][:3].copy()
        points  = [start] + [wp.position.copy() for wp in waypoints]
        arc_lengths = [0.0]
        for i in range(1, len(points)):
            arc_lengths.append(
                arc_lengths[-1] + np.linalg.norm(points[i] - points[i - 1])
            )
        total_arc = arc_lengths[-1]

        total_steps = 0

        if total_arc >= self.max_step:
            # ── Raised-cosine interpolation pass ─────────────────────────────
            # speed(t) ∝ sin(π·t) — zero at both ends, maximum at midpoint.
            # Integrate to get position: pos(t) = 0.5·(1 − cos(π·t)) · total_arc
            n_steps = max(1, int(total_arc / self.max_step))

            if self.verbose:
                print(
                    f"  [Follower] smooth arc={total_arc*1000:.1f}mm  "
                    f"waypoints={len(waypoints)}  steps={n_steps}"
                )

            for i in range(n_steps):
                t_linear  = (i + 1) / n_steps               # 0 → 1 (exclusive of 0)
                t_smooth  = 0.5 * (1.0 - np.cos(np.pi * t_linear))  # raised cosine
                target_s  = t_smooth * total_arc

                target_pos = self._interpolate_path(points, arc_lengths, target_s)
                current    = obs["observation.state.cartesian"][:3]
                delta      = target_pos - current
                dist       = np.linalg.norm(delta)

                if dist > 1e-6:
                    step = delta / dist * min(dist, self.max_step)
                else:
                    step = np.zeros(3)

                action     = np.zeros(6)
                action[:3] = step
                obs = self._step(env, obs, action)
                total_steps += 1

        # ── Final correction: P-controller to last waypoint ──────────────────
        # Closes any residual gap from impedance lag or floating-point error.
        last_wp = waypoints[-1]
        obs, reached, n_corr = self.go_to(
            env, obs, last_wp.position, last_wp.tolerance
        )
        total_steps += n_corr

        if not reached:
            print(
                f"  [Follower] WARNING: final correction timed out "
                f"(target={last_wp.position.round(4)})"
            )

        final_pos  = obs["observation.state.cartesian"][:3].copy()
        final_dist = float(np.linalg.norm(last_wp.position - final_pos))

        return obs, FollowResult(
            reached_all    = reached,
            steps_taken    = total_steps,
            final_position = final_pos,
            final_distance = final_dist,
            timed_out_at   = None if reached else 0,
        )

    def follow_sampled(
        self,
        env,
        obs: dict,
        waypoints: List[Waypoint],
    ) -> Tuple[dict, FollowResult]:
        """Follow pre-timed waypoints: one env.step() per waypoint.

        Unlike follow(), this does NOT re-parameterise with a raised-cosine profile.
        The velocity profile is encoded in the waypoint spacing (e.g. from Poly7Planner).
        A short P-controller correction at the end closes any residual gap.
        """
        if not waypoints:
            pos = obs["observation.state.cartesian"][:3].copy()
            return obs, FollowResult(
                reached_all=True, steps_taken=0,
                final_position=pos, final_distance=0.0,
            )

        total_steps = 0

        if self.verbose:
            total_arc = sum(
                np.linalg.norm(waypoints[i].position - waypoints[i - 1].position)
                for i in range(1, len(waypoints))
            )
            print(
                f"  [Follower/sampled] {len(waypoints)} waypoints  "
                f"arc={total_arc * 1000:.1f}mm"
            )

        for wp in waypoints:
            current = obs["observation.state.cartesian"][:3]
            delta   = wp.position - current
            dist    = np.linalg.norm(delta)
            step    = delta / dist * min(dist, self.max_step) if dist > 1e-6 else np.zeros(3)
            action  = np.zeros(6)
            action[:3] = step
            obs = self._step(env, obs, action)
            total_steps += 1

        # Final P-controller correction to reach last waypoint within tolerance
        last_wp = waypoints[-1]
        obs, reached, n_corr = self.go_to(env, obs, last_wp.position, last_wp.tolerance)
        total_steps += n_corr

        if not reached:
            print(
                f"  [Follower/sampled] WARNING: final correction timed out "
                f"(target={last_wp.position.round(4)})"
            )

        final_pos  = obs["observation.state.cartesian"][:3].copy()
        final_dist = float(np.linalg.norm(last_wp.position - final_pos))

        return obs, FollowResult(
            reached_all    = reached,
            steps_taken    = total_steps,
            final_position = final_pos,
            final_distance = final_dist,
            timed_out_at   = None if reached else 0,
        )

    def follow_sampled_6d(
        self,
        env,
        obs: dict,
        waypoints: List[Waypoint6D],
        orientation_tolerance: float = 0.01,   # rad (~0.6°)
    ) -> Tuple[dict, FollowResult]:
        """Follow pre-timed 6D waypoints: one env.step() per waypoint.

        Like follow_sampled() but also drives orientation.  At each step:
          * Position delta: P-controller step toward waypoint XYZ (bounded by max_step)
          * Rotation delta: full residual from current orientation to waypoint orientation
            encoded as axis-angle in action[3:6]

        The rotation delta per step is small because the Poly7 profile spreads the total
        rotation across N steps — the peak per-step rotation is O(angle/N).

        A short 6D correction loop at the end closes residual position AND orientation
        error (impedance lag).

        Parameters
        ----------
        env                   : Gymnasium environment.
        obs                   : Current observation dict.
        waypoints             : List[Waypoint6D] from Poly7Planner6D.plan().
        orientation_tolerance : Stop correcting when orientation error ≤ this [rad].
        """
        if not waypoints:
            pos = obs["observation.state.cartesian"][:3].copy()
            return obs, FollowResult(
                reached_all=True, steps_taken=0,
                final_position=pos, final_distance=0.0,
            )

        total_steps = 0

        if self.verbose:
            total_arc = sum(
                np.linalg.norm(waypoints[i].position - waypoints[i - 1].position)
                for i in range(1, len(waypoints))
            )
            last_q  = waypoints[-1].orientation
            first_q = waypoints[0].orientation
            total_angle = 2.0 * float(np.arccos(min(abs(float(np.dot(first_q, last_q))), 1.0)))
            print(
                f"  [Follower/sampled_6d] {len(waypoints)} waypoints  "
                f"arc={total_arc * 1000:.1f}mm  rot={np.degrees(total_angle):.1f}°"
            )

        for wp in waypoints:
            cart = obs["observation.state.cartesian"]
            current_pos = cart[:3]
            current_aa  = cart[3:6]

            # Position delta — bounded step toward waypoint
            delta_pos = wp.position - current_pos
            dist      = float(np.linalg.norm(delta_pos))
            step_pos  = delta_pos / dist * min(dist, self.max_step) if dist > 1e-6 else np.zeros(3)

            # Rotation delta — full residual to waypoint orientation
            R_current = Rotation.from_rotvec(current_aa)
            R_target  = Rotation.from_quat(wp.orientation)
            R_delta   = R_current.inv() * R_target
            step_rot  = R_delta.as_rotvec()   # axis-angle [rad]

            action = np.zeros(6)
            action[:3] = step_pos
            action[3:6] = step_rot
            obs = self._step(env, obs, action)
            total_steps += 1

        # ── Final 6D correction loop ──────────────────────────────────────────
        last_wp = waypoints[-1]
        reached_pos = False
        reached_rot = False
        for _ in range(self.max_iter_per_waypoint):
            cart = obs["observation.state.cartesian"]
            current_pos = cart[:3]
            current_aa  = cart[3:6]

            delta_pos = last_wp.position - current_pos
            dist      = float(np.linalg.norm(delta_pos))
            R_delta   = Rotation.from_rotvec(current_aa).inv() * Rotation.from_quat(last_wp.orientation)
            angle_err = float(np.linalg.norm(R_delta.as_rotvec()))

            reached_pos = dist   <= last_wp.tolerance
            reached_rot = angle_err <= orientation_tolerance
            if reached_pos and reached_rot:
                break

            step_pos = delta_pos / dist * min(dist, self.max_step) if dist > 1e-6 else np.zeros(3)
            action = np.zeros(6)
            action[:3]  = step_pos
            action[3:6] = R_delta.as_rotvec()
            obs = self._step(env, obs, action)
            total_steps += 1

        reached = reached_pos and reached_rot
        if not reached:
            print(
                f"  [Follower/sampled_6d] WARNING: final correction timed out "
                f"(pos_err={dist*1000:.1f}mm, rot_err={np.degrees(angle_err):.2f}°)"
            )

        final_pos  = obs["observation.state.cartesian"][:3].copy()
        final_dist = float(np.linalg.norm(last_wp.position - final_pos))

        return obs, FollowResult(
            reached_all    = reached,
            steps_taken    = total_steps,
            final_position = final_pos,
            final_distance = final_dist,
            timed_out_at   = None if reached else 0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7th-order polynomial planner
# ─────────────────────────────────────────────────────────────────────────────

class Poly7Planner:
    """7th-order polynomial trajectory planner.

    Enforces zero position, velocity, acceleration **and jerk** at both endpoints:

        p(0) = p0,  p(T) = pf
        v(0) = v(T) = 0
        a(0) = a(T) = 0
        j(0) = j(T) = 0

    The normalised position profile (independent of distance and duration):

        s(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷,   τ ∈ [0, 1]

    is evaluated at N+1 equally-spaced τ values to yield waypoints.  N is chosen
    so that one waypoint ≈ one env.step() (dt = 66 ms at 15 Hz), matching the
    natural rhythm of the impedance controller.

    All three axes share the same T = T_min (bound by the axis with the largest
    displacement), giving a straight-line 3D path.

    Use with TrajectoryFollower.follow_sampled() — that method sends exactly one
    env.step() per waypoint, preserving the poly's velocity profile.

    Parameters
    ----------
    a_limit            : Cartesian acceleration limit [m/s²].  Used to compute the
                         minimum feasible trajectory duration T_min.
    env_dt             : Duration of one env.step() in seconds.  Default 0.066 s
                         (198 MuJoCo sub-steps at 3 kHz = 66 ms).
    waypoint_tolerance : Arrival threshold for every generated waypoint [m].
    """

    # Peak value of |s''(τ)| over τ ∈ [0,1], computed analytically.
    # s''(τ) = 420τ² − 1680τ³ + 2100τ⁴ − 840τ⁵  →  max ≈ 10.9805 at τ ≈ 0.2163
    _K: float = 10.9805

    def __init__(
        self,
        a_limit:            float = 2.0,    # m/s²
        env_dt:             float = 0.066,  # s per env.step()
        waypoint_tolerance: float = 0.002,  # m
    ):
        self.a_limit            = a_limit
        self.env_dt             = env_dt
        self.waypoint_tolerance = waypoint_tolerance

    def t_min(self, dist: float) -> float:
        """Minimum feasible duration [s] for a straight-line move of given distance."""
        return float(np.sqrt(self._K * dist / self.a_limit))

    def plan(
        self,
        start_xyz,
        goal_xyz,
    ) -> List[Waypoint]:
        """Generate a 7th-order polynomial trajectory from start to goal.

        Parameters
        ----------
        start_xyz : array-like (3,), starting TCP position [m].
        goal_xyz  : array-like (3,), target TCP position [m].

        Returns
        -------
        List of Waypoint objects sampled at equal τ-steps.  Always ends exactly
        at goal_xyz.  Intended to be followed with TrajectoryFollower.follow_sampled().
        """
        start = np.asarray(start_xyz, dtype=float)
        goal  = np.asarray(goal_xyz,  dtype=float)
        dist  = np.linalg.norm(goal - start)

        if dist < 1e-6:
            return [Waypoint(position=goal.copy(), tolerance=self.waypoint_tolerance)]

        T         = self.t_min(dist)
        N         = max(10, int(np.ceil(T / self.env_dt)))   # env steps
        tau       = np.linspace(0.0, 1.0, N + 1)[1:]         # skip τ=0 (= start)
        s         = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7  # normalised position

        direction = (goal - start) / dist
        waypoints = [
            Waypoint(
                position=(start + si * dist * direction).copy(),
                tolerance=self.waypoint_tolerance,
            )
            for si in s
        ]
        waypoints[-1].position = goal.copy()   # floating-point safety
        return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# 7th-order polynomial planner — 6D (position + orientation)
# ─────────────────────────────────────────────────────────────────────────────

class Poly7Planner6D:
    """7th-order polynomial trajectory planner for 6D poses.

    Extends Poly7Planner to handle orientation as well as position.  The same
    normalised profile drives **both** axes simultaneously:

        s(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷,   τ ∈ [0, 1]

    * **Position**: straight-line interpolation  ``p(τ) = p_start + s(τ)·Δp``
    * **Orientation**: SLERP from ``q_start`` to ``q_goal`` at parameter ``s(τ)``

    The minimum feasible duration T accounts for both constraints::

        T = max( sqrt(K · dist_xyz / a_limit),
                 sqrt(K · angle_rad / alpha_limit) )

    so the slower axis sets the pace and neither limit is violated.

    Orientation inputs can be given as:

    * **(4,) unit quaternion** ``[x, y, z, w]``
    * **(3,) axis-angle** ``[rx, ry, rz]`` — same encoding as
      ``obs["observation.state.cartesian"][3:6]``

    ``None`` defaults to the identity rotation (pure translation — equivalent to
    Poly7Planner but returns Waypoint6D objects).

    Use with ``TrajectoryFollower.follow_sampled_6d()`` — that method reads the
    quaternion from each waypoint, computes the incremental rotation delta, and
    fills ``action[3:6]``.

    Parameters
    ----------
    a_limit            : Cartesian acceleration limit [m/s²].
    alpha_limit        : Angular acceleration limit [rad/s²].
    env_dt             : Duration of one env.step() in seconds (default 66 ms).
    waypoint_tolerance : Position arrival threshold for each waypoint [m].
    """

    _K: float = 10.9805   # peak |s''(τ)| — shared with Poly7Planner

    def __init__(
        self,
        a_limit:            float = 2.0,    # m/s²
        alpha_limit:        float = 1.0,    # rad/s²
        env_dt:             float = 0.066,  # s per env.step()
        waypoint_tolerance: float = 0.002,  # m
    ):
        self.a_limit            = a_limit
        self.alpha_limit        = alpha_limit
        self.env_dt             = env_dt
        self.waypoint_tolerance = waypoint_tolerance

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_quat(rot) -> np.ndarray:
        """Accept (3,) axis-angle or (4,) quaternion; always return (4,) quat [x,y,z,w]."""
        rot = np.asarray(rot, dtype=float)
        if rot.shape == (3,):
            return Rotation.from_rotvec(rot).as_quat()
        if rot.shape == (4,):
            q = rot / np.linalg.norm(rot)
            return q
        raise ValueError(f"Expected shape (3,) or (4,) for orientation, got {rot.shape}")

    @staticmethod
    def _geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
        """Geodesic angle [rad] between two unit quaternions on SO(3)."""
        # Ensure shortest path (q and -q represent the same rotation)
        dot = float(np.dot(q1, q2))
        if dot < 0:
            q2 = -q2
            dot = -dot
        dot = min(dot, 1.0)   # numerical safety for arccos
        return 2.0 * float(np.arccos(dot))

    def t_min(self, dist: float, angle: float) -> float:
        """Minimum feasible duration [s] for a move of given distance and angle."""
        t_pos = float(np.sqrt(self._K * dist  / self.a_limit))  if dist  > 1e-9 else 0.0
        t_rot = float(np.sqrt(self._K * angle / self.alpha_limit)) if angle > 1e-9 else 0.0
        return max(t_pos, t_rot)

    # ── main API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        start_xyz,
        goal_xyz,
        start_orientation=None,
        goal_orientation=None,
    ) -> List[Waypoint6D]:
        """Generate a 6D polynomial trajectory from start to goal.

        Parameters
        ----------
        start_xyz         : (3,) starting TCP position [m].
        goal_xyz          : (3,) target TCP position [m].
        start_orientation : (3,) axis-angle or (4,) quaternion, or None → identity.
        goal_orientation  : (3,) axis-angle or (4,) quaternion, or None → identity.

        Returns
        -------
        List of Waypoint6D sampled at equal τ-steps.  Always ends exactly at
        goal_xyz / goal_orientation.  Use with TrajectoryFollower.follow_sampled_6d().
        """
        start = np.asarray(start_xyz, dtype=float)
        goal  = np.asarray(goal_xyz,  dtype=float)
        dist  = float(np.linalg.norm(goal - start))

        _identity = Rotation.identity().as_quat()
        q_start = self._to_quat(start_orientation) if start_orientation is not None else _identity.copy()
        q_goal  = self._to_quat(goal_orientation)  if goal_orientation  is not None else _identity.copy()

        # Ensure shortest-path SLERP
        if np.dot(q_start, q_goal) < 0:
            q_goal = -q_goal

        angle = self._geodesic_angle(q_start, q_goal)

        # Degenerate: no motion at all
        if dist < 1e-6 and angle < 1e-6:
            return [Waypoint6D(
                position=goal.copy(),
                orientation=q_goal.copy(),
                tolerance=self.waypoint_tolerance,
            )]

        T = self.t_min(dist, angle)
        N = max(10, int(np.ceil(T / self.env_dt)))   # number of env steps

        tau = np.linspace(0.0, 1.0, N + 1)[1:]       # skip τ=0 (= start)
        s   = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7   # normalised position

        # Build SLERP interpolator over s ∈ [0, 1]
        key_rots  = Rotation.from_quat(np.stack([q_start, q_goal]))
        slerp_fn  = Slerp([0.0, 1.0], key_rots)
        orientations = slerp_fn(s)   # Rotation array of length N

        direction = (goal - start) / dist if dist > 1e-6 else np.zeros(3)
        waypoints = [
            Waypoint6D(
                position=(start + s[i] * dist * direction).copy(),
                orientation=orientations[i].as_quat().copy(),
                tolerance=self.waypoint_tolerance,
            )
            for i in range(N)
        ]
        # Floating-point safety: pin last waypoint to exact goal
        waypoints[-1].position    = goal.copy()
        waypoints[-1].orientation = q_goal.copy()
        return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# RuckigFollower — online jerk-limited trajectory follower
# ─────────────────────────────────────────────────────────────────────────────

class RuckigFollower:
    """Online jerk-limited trajectory follower using Ruckig.

    Generates one trajectory step per control cycle from the actual robot
    state.  Goal can be updated between any two steps — no replanning needed.

    Four follow modes
    -----------------
    follow_xy(env, obs, target_xy, ...)
        XY-only approach; Z held at current value.
        Drop-in replacement for InsertionWrapperSimLEGO.go_to_waypoint().
    follow_3d(env, obs, goal_xyz, ...)
        Full 3D position move; orientation unchanged.
    follow_6d(env, obs, goal_xyz, goal_aa, ...)
        6D position + orientation move.
    follow_streaming(env, obs, goal_fn, ...)
        Streaming 6D goal — goal_fn() called every step.
    """

    def __init__(
        self,
        dt:           float = 0.066,   # s — must match env control cycle
        max_vel:      float = 0.05,    # m/s
        max_acc:      float = 0.5,     # m/s²
        max_jerk:     float = 5.0,     # m/s³
        max_ang_vel:  float = 0.3,     # rad/s
        max_ang_acc:  float = 3.0,     # rad/s²
        max_ang_jerk: float = 30.0,    # rad/s³
        verbose:      bool  = False,
    ):
        from ruckig import Ruckig, InputParameter, OutputParameter

        self.verbose = verbose

        # 3-DOF instance (XYZ position only)
        self._otg3 = Ruckig(3, dt)
        self._inp3 = InputParameter(3)
        self._out3 = OutputParameter(3)
        self._inp3.max_velocity     = [max_vel]  * 3
        self._inp3.max_acceleration = [max_acc]  * 3
        self._inp3.max_jerk         = [max_jerk] * 3

        # 6-DOF instance (XYZ + axis-angle)
        self._otg6 = Ruckig(6, dt)
        self._inp6 = InputParameter(6)
        self._out6 = OutputParameter(6)
        self._inp6.max_velocity     = [max_vel]  * 3 + [max_ang_vel]  * 3
        self._inp6.max_acceleration = [max_acc]  * 3 + [max_ang_acc]  * 3
        self._inp6.max_jerk         = [max_jerk] * 3 + [max_ang_jerk] * 3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state_3d(self, obs, goal_xyz):
        pos = obs["observation.state.cartesian"][:3]
        self._inp3.current_position     = pos.tolist()
        self._inp3.current_velocity     = [0.0, 0.0, 0.0]
        self._inp3.current_acceleration = [0.0, 0.0, 0.0]
        self._inp3.target_position      = list(goal_xyz)
        self._inp3.target_velocity      = [0.0, 0.0, 0.0]
        self._inp3.target_acceleration  = [0.0, 0.0, 0.0]

    def _init_state_6d(self, obs, goal_xyz, goal_aa):
        cart = obs["observation.state.cartesian"]
        self._inp6.current_position     = cart.tolist()
        self._inp6.current_velocity     = [0.0] * 6
        self._inp6.current_acceleration = [0.0] * 6
        self._inp6.target_position      = list(goal_xyz) + list(goal_aa)
        self._inp6.target_velocity      = [0.0] * 6
        self._inp6.target_acceleration  = [0.0] * 6

    # ------------------------------------------------------------------
    # follow_xy  —  XY approach, Z fixed (replaces go_to_waypoint)
    # ------------------------------------------------------------------

    def follow_xy(self, env, obs, target_xy, distance_err=0.002, max_steps=10_000):
        """Move to target_xy while holding Z at its current value."""
        target_xy = np.asarray(target_xy, dtype=float)
        z_fixed   = float(obs["observation.state.cartesian"][2])
        goal_xyz  = np.array([target_xy[0], target_xy[1], z_fixed])
        return self.follow_3d(env, obs, goal_xyz, tol=distance_err, max_steps=max_steps)

    # ------------------------------------------------------------------
    # follow_3d
    # ------------------------------------------------------------------

    def follow_3d(self, env, obs, goal_xyz, tol=0.002, max_steps=2000):
        """Online 3D position move; orientation action left at zero (unchanged)."""
        from ruckig import Result

        goal_xyz = np.asarray(goal_xyz, dtype=float)
        self._init_state_3d(obs, goal_xyz)

        for step_i in range(max_steps):
            res = self._otg3.update(self._inp3, self._out3)

            actual_pos = obs["observation.state.cartesian"][:3]
            delta_pos  = np.array(self._out3.new_position) - actual_pos

            action       = np.zeros(6)
            action[:3]   = delta_pos
            obs, *_      = env.step(action)

            # Hybrid feedback: Ruckig velocity model + actual position
            self._out3.pass_to_input(self._inp3)
            self._inp3.current_position = obs["observation.state.cartesian"][:3].tolist()

            if self.verbose and step_i % 50 == 0:
                err = np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz)
                print(f"  [RuckigFollower.follow_3d] step={step_i}  err={err*1000:.2f}mm")

            if res == Result.Finished:
                if np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz) <= tol:
                    break

        return obs

    # ------------------------------------------------------------------
    # follow_6d
    # ------------------------------------------------------------------

    def follow_6d(
        self, env, obs, goal_xyz, goal_aa,
        tol=0.002, orientation_tol=0.01, max_steps=2000,
    ):
        """Online 6D move (position + orientation).

        Orientation delta computed via SO(3) arithmetic — NOT additive axis-angle.
        """
        from ruckig import Result

        goal_xyz = np.asarray(goal_xyz, dtype=float)
        goal_aa  = np.asarray(goal_aa,  dtype=float)
        self._init_state_6d(obs, goal_xyz, goal_aa)

        R_goal = Rotation.from_rotvec(goal_aa)

        for step_i in range(max_steps):
            res = self._otg6.update(self._inp6, self._out6)

            actual    = obs["observation.state.cartesian"]
            actual_pos = actual[:3]
            actual_aa  = actual[3:6]

            # Position delta
            target_pos = np.array(self._out6.new_position[:3])
            delta_pos  = target_pos - actual_pos

            # Orientation delta — SO(3), not additive
            target_aa = np.array(self._out6.new_position[3:6])
            R_current = Rotation.from_rotvec(actual_aa)
            R_target  = Rotation.from_rotvec(target_aa)
            delta_rot = (R_current.inv() * R_target).as_rotvec()

            action       = np.zeros(6)
            action[:3]   = delta_pos
            action[3:6]  = delta_rot
            obs, *_      = env.step(action)

            # Hybrid feedback
            self._out6.pass_to_input(self._inp6)
            self._inp6.current_position = obs["observation.state.cartesian"].tolist()

            if self.verbose and step_i % 50 == 0:
                pos_err = np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz)
                R_delta = Rotation.from_rotvec(obs["observation.state.cartesian"][3:6]).inv() * R_goal
                rot_err = np.linalg.norm(R_delta.as_rotvec())
                print(f"  [RuckigFollower.follow_6d] step={step_i}  "
                      f"pos_err={pos_err*1000:.2f}mm  rot_err={np.degrees(rot_err):.2f}°")

            if res == Result.Finished:
                pos_err = np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz)
                R_delta = Rotation.from_rotvec(obs["observation.state.cartesian"][3:6]).inv() * R_goal
                rot_err = float(np.linalg.norm(R_delta.as_rotvec()))
                if pos_err <= tol and rot_err <= orientation_tol:
                    break

        return obs

    # ------------------------------------------------------------------
    # follow_streaming  —  live goal updates mid-motion
    # ------------------------------------------------------------------

    def follow_streaming(self, env, obs, goal_fn, max_steps=2000, tol=0.002):
        """Follow a streaming 6D goal.

        goal_fn() → (xyz (3,), aa (3,)) is called every step.
        The goal can change between steps — Ruckig adapts online.
        """
        from ruckig import Result

        initialised = False
        for step_i in range(max_steps):
            goal_xyz, goal_aa = goal_fn()
            goal_xyz = np.asarray(goal_xyz, dtype=float)
            goal_aa  = np.asarray(goal_aa,  dtype=float)

            if not initialised:
                self._init_state_6d(obs, goal_xyz, goal_aa)
                initialised = True
            else:
                # Live goal update — Ruckig re-plans each cycle from current state
                self._inp6.target_position = list(goal_xyz) + list(goal_aa)

            res = self._otg6.update(self._inp6, self._out6)

            actual    = obs["observation.state.cartesian"]
            delta_pos = np.array(self._out6.new_position[:3]) - actual[:3]
            target_aa = np.array(self._out6.new_position[3:6])
            R_current = Rotation.from_rotvec(actual[3:6])
            R_target  = Rotation.from_rotvec(target_aa)
            delta_rot = (R_current.inv() * R_target).as_rotvec()

            action       = np.zeros(6)
            action[:3]   = delta_pos
            action[3:6]  = delta_rot
            obs, *_      = env.step(action)

            self._out6.pass_to_input(self._inp6)
            self._inp6.current_position = obs["observation.state.cartesian"].tolist()

            if self.verbose and step_i % 20 == 0:
                err = np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz)
                print(f"  [streaming] step={step_i}  goal={goal_xyz.round(4)}  "
                      f"pos_err={err*1000:.2f}mm")

            if res == Result.Finished:
                if np.linalg.norm(obs["observation.state.cartesian"][:3] - goal_xyz) <= tol:
                    break

        return obs
