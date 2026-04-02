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

    The controller strategy is a P-controller with component-wise step clipping:
      delta = target - current
      action[:3] = clip(delta, -max_step, +max_step)  per axis
      action[3:] = 0  (rotation unchanged)
    This is the same strategy used in FreeSpaceMotionPlanner and InsertionWrapperSimLEGO.

    Real-robot portability: subclass and override _step() to call the robot
    controller instead of env.step().

    Parameters
    ----------
    max_step              : Maximum Cartesian delta per action per axis [m].
                            1 mm is safe for the CartesianImpedanceController
                            (error clipping is ±3 mm, so 1 mm stays well inside).
    max_iter_per_waypoint : Safety cap on env.step() calls per waypoint.
                            At 1 mm/step and 5000 steps = 5 m maximum travel,
                            which is far more than any reachable workspace distance.
    verbose               : Print per-waypoint progress.
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

    # ── low-level: single-step action ────────────────────────────────────────

    def _step(self, env, obs, action: np.ndarray):
        """Send one action to the environment. Override for real-robot use."""
        obs, _, _, _, _ = env.step(action)
        return obs

    # ── mid-level: move to a single target ───────────────────────────────────

    def go_to(
        self,
        env,
        obs: dict,
        target_xyz,
        tolerance: float = 0.002,
    ) -> Tuple[dict, bool, int]:
        """Move toward a single 3D target until within tolerance or timed out.

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
            # Read current TCP position (ground truth from MuJoCo)
            current = obs["observation.state.cartesian"][:3]
            delta   = target - current
            dist    = np.linalg.norm(delta)

            # Check arrival
            if dist <= tolerance:
                return obs, True, n_steps

            # Clip each axis independently to max_step
            # (component-wise clip, not norm-based, matches existing sim code)
            step = np.clip(delta, -self.max_step, self.max_step)

            # Build 6D action: XYZ delta + zero rotation (hold orientation fixed)
            action        = np.zeros(6)
            action[:3]    = step   # X, Y, Z position delta [m]
            # action[3:6] = 0      # rotation axis-angle delta — zero = no rotation change

            obs = self._step(env, obs, action)
            n_steps += 1

        # Timed out without reaching tolerance
        return obs, False, n_steps

    # ── high-level: follow a waypoint list ───────────────────────────────────

    def follow(
        self,
        env,
        obs: dict,
        waypoints: List[Waypoint],
    ) -> Tuple[dict, FollowResult]:
        """Follow a list of Waypoints in order.

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
        total_steps  = 0
        timed_out_at = None

        for i, wp in enumerate(waypoints):
            # Print progress before attempting the waypoint
            if self.verbose:
                current  = obs["observation.state.cartesian"][:3]
                dist_now = np.linalg.norm(wp.position - current) * 1000
                print(
                    f"  [Follower] wp {i+1:3d}/{len(waypoints)}: "
                    f"target={wp.position.round(4)}  "
                    f"dist_to_wp={dist_now:.1f}mm"
                )

            obs, reached, n_steps = self.go_to(
                env, obs, wp.position, wp.tolerance
            )
            total_steps += n_steps

            if not reached:
                # Record the first timeout; continue to next waypoint rather
                # than aborting — partial progress is still useful for debugging.
                if timed_out_at is None:
                    timed_out_at = i
                print(
                    f"  [Follower] WARNING: timed out at waypoint {i} "
                    f"(target={wp.position.round(4)})"
                )

        # Final position and distance to the last waypoint
        final_pos  = obs["observation.state.cartesian"][:3].copy()
        final_dist = (
            float(np.linalg.norm(waypoints[-1].position - final_pos))
            if waypoints else 0.0
        )

        return obs, FollowResult(
            reached_all    = (timed_out_at is None),
            steps_taken    = total_steps,
            final_position = final_pos,
            final_distance = final_dist,
            timed_out_at   = timed_out_at,
        )
