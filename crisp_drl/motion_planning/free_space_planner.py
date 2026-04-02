"""Free-space Cartesian motion planner.

Provides a simple, upgradeable trajectory execution layer for pre-insertion
free-space motions. Designed to mirror the waypoint-following pattern used
in InsertionWrapperSiemens, but as a standalone class that can be reused
across sim and real-robot wrappers.

Current execution strategy: P-controller step loop (Option A).
To upgrade to linear interpolation (Option B), replace go_to_waypoint_3d()
without changing any caller code.

Usage (sim):
    planner = FreeSpaceMotionPlanner()
    obs = planner.execute(env, obs, [
        CartesianWaypoint([0.6, 0.0, 0.15], distance_err=0.005),
        CartesianWaypoint([0.6, 0.0, 0.122], distance_err=0.002),
    ])

Real-robot migration:
    Subclass FreeSpaceMotionPlanner and override go_to_waypoint_3d() to call
    the robot's Cartesian motion controller instead of env.step().
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


_DEFAULT_MAX_STEP = 0.001   # m — matches InsertionWrapperSimLEGO _APPROACH_MAX_STEP


@dataclass
class CartesianWaypoint:
    """A single 3D Cartesian target with approach tolerance.

    Parameters
    ----------
    position_xyz : array-like (3,)
        Target TCP position in world frame [x, y, z] metres.
    distance_err : float
        Approach is considered reached when Euclidean distance ≤ this value [m].
    """
    position_xyz: np.ndarray
    distance_err: float = 0.002

    def __post_init__(self):
        self.position_xyz = np.asarray(self.position_xyz, dtype=float)


class FreeSpaceMotionPlanner:
    """Plans and executes free-space Cartesian trajectories.

    Executes a sequence of 3D Cartesian waypoints using a P-controller step
    loop. Each step sends a clipped delta action to the environment; the
    Cartesian impedance controller handles the low-level tracking.

    This is the sim-compatible implementation (Option A). To upgrade to linear
    interpolation (Option B) without changing any caller code, override
    go_to_waypoint_3d().

    Parameters
    ----------
    max_step : float
        Maximum Cartesian delta per action step [m].
        Default 0.001 m (1 mm) matches the insertion wrapper convention.
    max_iter_per_waypoint : int
        Safety limit on step loop iterations per waypoint.
    verbose : bool
        Print waypoint progress.
    """

    def __init__(
        self,
        max_step: float = _DEFAULT_MAX_STEP,
        max_iter_per_waypoint: int = 10_000,
        verbose: bool = True,
    ):
        self.max_step = max_step
        self.max_iter_per_waypoint = max_iter_per_waypoint
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def go_to_waypoint_3d(self, env, obs, target_xyz, distance_err: float = 0.002):
        """P-controller step loop toward a 3D Cartesian target.

        Sends [dx, dy, dz, 0, 0, 0] actions each step. No rotation change,
        no Z locking — full 3D approach. Does not count toward episode steps.

        Parameters
        ----------
        env : gymnasium.Env
            The unwrapped or inner env whose step() accepts 6D actions.
        obs : dict
            Current observation (must contain 'observation.state.cartesian').
        target_xyz : array-like (3,)
            3D Cartesian target [x, y, z] in metres.
        distance_err : float
            Stop when within this Euclidean distance of target [m].

        Returns
        -------
        obs : dict
            Observation after reaching (or timing out toward) the waypoint.
        """
        target = np.asarray(target_xyz, dtype=float)
        for i in range(self.max_iter_per_waypoint):
            current = obs["observation.state.cartesian"][:3]
            delta = target - current
            dist = np.linalg.norm(delta)
            if dist <= distance_err:
                break
            step = np.clip(delta, -self.max_step, self.max_step)
            action = np.zeros(6)
            action[:3] = step
            obs, _, _, _, _ = env.step(action)
        else:
            print(
                f"[FreeSpacePlanner] WARNING: max_iter reached for waypoint {target} "
                f"(final dist={np.linalg.norm(obs['observation.state.cartesian'][:3] - target)*1000:.1f}mm)"
            )
        return obs

    def execute(self, env, obs, waypoints: List[CartesianWaypoint]):
        """Follow a list of CartesianWaypoint in order.

        Parameters
        ----------
        env : gymnasium.Env
            Inner env whose step() accepts 6D actions.
        obs : dict
            Current observation.
        waypoints : list of CartesianWaypoint
            Sequence of 3D targets to reach, in order.

        Returns
        -------
        obs : dict
            Observation after the last waypoint is reached.
        """
        for i, wp in enumerate(waypoints):
            if self.verbose:
                print(
                    f"[FreeSpacePlanner] Waypoint {i+1}/{len(waypoints)}: "
                    f"{wp.position_xyz} (err<{wp.distance_err*1000:.1f}mm)"
                )
            obs = self.go_to_waypoint_3d(env, obs, wp.position_xyz, wp.distance_err)
        return obs


# ------------------------------------------------------------------
# Option B: linear interpolation (drop-in upgrade)
# ------------------------------------------------------------------

class LinearInterpolationPlanner(FreeSpaceMotionPlanner):
    """Upgrade of FreeSpaceMotionPlanner using dense linear interpolation.

    Instead of a P-controller reactive loop, generates a straight-line
    sequence of intermediate setpoints between start and target, then
    follows them with a fine P-controller at each point.

    Produces smoother motion than the pure P-controller. Replaces
    go_to_waypoint_3d() — all other code (execute(), callers) unchanged.
    """

    def go_to_waypoint_3d(self, env, obs, target_xyz, distance_err: float = 0.002):
        target = np.asarray(target_xyz, dtype=float)
        start = obs["observation.state.cartesian"][:3].copy()
        dist = np.linalg.norm(target - start)
        if dist <= distance_err:
            return obs
        # Number of intermediate points along the straight line
        n_steps = max(1, int(dist / self.max_step))
        for t in np.linspace(0, 1, n_steps + 1)[1:]:  # skip t=0 (start)
            intermediate = start + t * (target - start)
            # Fine P-controller to reach each intermediate point
            for _ in range(500):
                current = obs["observation.state.cartesian"][:3]
                delta = intermediate - current
                if np.linalg.norm(delta) <= self.max_step * 0.5:
                    break
                step = np.clip(delta, -self.max_step, self.max_step)
                action = np.zeros(6)
                action[:3] = step
                obs, _, _, _, _ = env.step(action)
        # Final fine approach to exact target
        for _ in range(self.max_iter_per_waypoint):
            current = obs["observation.state.cartesian"][:3]
            delta = target - current
            if np.linalg.norm(delta) <= distance_err:
                break
            step = np.clip(delta, -self.max_step, self.max_step)
            action = np.zeros(6)
            action[:3] = step
            obs, _, _, _, _ = env.step(action)
        return obs
