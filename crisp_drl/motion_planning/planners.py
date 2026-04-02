"""Motion planners for the LEGO insertion task.

All planners share the same interface:
    planner.reset(goal_position)  -- called once per episode
    planner.plan(obs)             -- called every step, returns 2D action [dx, dy]

"Known goal" variants: goal_position is passed directly from InsertionWrapperSim.
Actions are in metres; the wrapper clips them via the safety box internally,
but planners should also stay within reasonable step sizes (~1mm max).
"""

import numpy as np


MAX_STEP = 0.001  # metres, matches NoRotationNoGripperNoZActionClippedWrapperSim clip


class BasePlanner:
    """Interface all planners must implement."""

    def reset(self, goal_position: np.ndarray) -> None:
        """Called at the start of each episode with the 3D goal position."""
        pass

    def plan(self, obs: dict) -> np.ndarray:
        """Return a 2D action [dx, dy] in metres."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Oracle planner
# ---------------------------------------------------------------------------

class OraclePlanner(BasePlanner):
    """Uses observation.perfect_action directly.

    This is the theoretical upper bound — it knows the exact vector to the
    goal every step.  Note: perfect_action = current - goal (points away from
    goal), so we negate it.
    """

    def plan(self, obs: dict) -> np.ndarray:
        # perfect_action = cartesian[:3] - (goal_gt + grasp_position)
        # → negate to get direction toward goal, clip to max step
        delta = -obs["observation.perfect_action"][:2]
        return np.clip(delta, -MAX_STEP, MAX_STEP)


# ---------------------------------------------------------------------------
# Go-to-goal planner (P controller)
# ---------------------------------------------------------------------------

class GoToGoalPlanner(BasePlanner):
    """Proportional controller: move straight toward the known goal.

    With a perfect goal estimate this should achieve near-100% success and
    serves as the simplest known-goal baseline.
    """

    def __init__(self, gain: float = 1.0, max_step: float = MAX_STEP):
        self.gain = gain
        self.max_step = max_step
        self._goal_xy = None

    def reset(self, goal_position: np.ndarray) -> None:
        self._goal_xy = goal_position[:2].copy()

    def plan(self, obs: dict) -> np.ndarray:
        current_xy = obs["observation.state.cartesian"][:2]
        delta = self._goal_xy - current_xy
        return np.clip(self.gain * delta, -self.max_step, self.max_step)


# ---------------------------------------------------------------------------
# Spiral search planner
# ---------------------------------------------------------------------------

class SpiralPlanner(BasePlanner):
    """Archimedean spiral search centred on the known goal.

    Traces outward from the goal centre in a spiral, then switches to a
    P-controller once it detects insertion contact (Z error exceeds threshold).

    Useful for validating that the search infrastructure works before moving
    to unknown-goal scenarios where the centre is uncertain.

    Parameters
    ----------
    max_radius : float
        Maximum search radius [m].  Should match the env's safety_box_radius.
    pitch : float
        Radial advance per full revolution [m].
    angular_step : float
        Angle increment per step [rad].
    contact_z_threshold : float
        |z_error| above this value is treated as insertion contact,
        at which point the planner switches to a fine P-controller.
    max_step : float
        Maximum action magnitude [m].
    """

    def __init__(
        self,
        max_radius: float = 0.003,
        pitch: float = 0.0006,
        angular_step: float = np.deg2rad(18),
        contact_z_threshold: float = 0.0008,
        max_step: float = MAX_STEP,
    ):
        self.max_radius = max_radius
        self.pitch = pitch
        self.angular_step = angular_step
        self.contact_z_threshold = contact_z_threshold
        self.max_step = max_step

        self._goal_xy = None
        self._angle = 0.0
        self._in_contact = False

    def reset(self, goal_position: np.ndarray) -> None:
        self._goal_xy = goal_position[:2].copy()
        self._angle = 0.0
        self._in_contact = False

    def plan(self, obs: dict) -> np.ndarray:
        z_error = obs["observation.error.cartesian"][2]
        current_xy = obs["observation.state.cartesian"][:2]

        # once insertion contact is detected, switch to fine P-controller
        if abs(z_error) > self.contact_z_threshold:
            self._in_contact = True

        if self._in_contact:
            delta = self._goal_xy - current_xy
            return np.clip(delta, -self.max_step, self.max_step)

        # Archimedean spiral: r = (pitch / 2π) * θ
        r = (self.pitch / (2 * np.pi)) * self._angle
        r = min(r, self.max_radius)

        target_xy = self._goal_xy + r * np.array([
            np.cos(self._angle),
            np.sin(self._angle),
        ])
        self._angle += self.angular_step

        delta = target_xy - current_xy
        return np.clip(delta, -self.max_step, self.max_step)


# ---------------------------------------------------------------------------
# Hybrid planner (coarse GoToGoal → fine RL)
# ---------------------------------------------------------------------------

class HybridPlanner(BasePlanner):
    """GoToGoal for coarse approach, switches to `rl_planner` when close to goal.

    Mirrors the go_to_waypoint → fine-control pattern used in InsertionWrapperSiemens:
    deterministic motion planning covers the approach, RL handles the precision search.

    Parameters
    ----------
    rl_planner : BasePlanner or None
        Planner used in the fine phase once within switch_distance.
        If None, falls back to GoToGoalPlanner (useful for testing switching
        logic without a trained policy).
    switch_distance : float
        XY distance [m] at which to hand off from GoToGoal to rl_planner.
    """

    def __init__(self, rl_planner: "BasePlanner | None" = None, switch_distance: float = 0.004):
        self._coarse = GoToGoalPlanner()
        self._fine   = rl_planner if rl_planner is not None else GoToGoalPlanner()
        self.switch_distance = switch_distance
        self._goal_xy = None
        self._in_rl_mode = False

    def reset(self, goal_position: np.ndarray) -> None:
        self._goal_xy = goal_position[:2].copy()
        self._coarse.reset(goal_position)
        self._fine.reset(goal_position)
        self._in_rl_mode = False

    def plan(self, obs: dict) -> np.ndarray:
        if not self._in_rl_mode:
            dist = np.linalg.norm(obs["observation.state.cartesian"][:2] - self._goal_xy)
            if dist <= self.switch_distance:
                self._in_rl_mode = True
        if self._in_rl_mode:
            return self._fine.plan(obs)
        return self._coarse.plan(obs)
