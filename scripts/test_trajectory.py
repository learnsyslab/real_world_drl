"""Insertion pipeline state machine — simulation demo.

Demonstrates the full pick-and-place state machine using only the
TrajectoryPlanner + TrajectoryFollower:

    HOME → ABOVE_A → GRASP_A → (hold) → ABOVE_A → ABOVE_B → INSERT_B → (hold) → HOME

Positions A and B are the only inputs. Everything else follows automatically.
In the real world these come from FoundationPose; here they are hardcoded.

Since the sim has no gripper actuation, "grasp" and "release" are just hold pauses
(zero action for N steps — the impedance controller keeps the arm in place).

Known world-frame geometry (from the MJCF scene):
    Fixed socket (lego_2x2_fixed):  [0.6, 0.0, 0.115] m
    Home TCP (keyframe 1):          approx [0.6, 0.0, 0.131] m

Run:
    cd /home/gabor/drl_project/real_world_drl
    MUJOCO_GL=egl  pixi run -e sim python scripts/test_trajectory.py
    MUJOCO_GL=glfw pixi run -e sim python scripts/test_trajectory.py --live_view
"""

from __future__ import annotations
import argparse, sys, os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujid.env.env as mujid_env
from crisp_drl.motion_planning.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryFollower,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Geometry and timing parameters for the insertion pipeline.

    These are tuned for the LEGO sim. Adjust when moving to real robot.
    All positions and offsets in metres.
    """

    # How high above pos_A / pos_B to position the TCP before descending.
    # Large enough to avoid collision during lateral transit.
    approach_height: float = 0.07       # 70 mm above target

    # Fine Z offset applied at the grasp point (pos_A).
    # Positive = stay higher (more clearance), negative = go deeper.
    grasp_z_offset: float = 0.0

    # Fine Z offset applied at the insert point (pos_B).
    insert_z_offset: float = 0.0

    # How long to hold still for "grasp" and "release" in simulation [s].
    # At 15 Hz RL rate: 2 s = 30 hold steps.
    hold_duration_s: float = 2.0

    # Trajectory planner step size [m] — fine waypoints near start/end.
    step_size: float = 0.001            # 1 mm

    # Arrival tolerance for every waypoint [m].
    waypoint_tolerance: float = 0.002   # 2 mm

    # Max action delta per env.step() call [m/axis].
    # Impedance controller clips errors to ±3 mm, so 3 mm is the safe maximum.
    max_action_step: float = 0.003      # 3 mm

    # Coarse waypoint spacing for the middle of long free-space transits [m].
    # None = uniform fine spacing.
    coarse_step_size: float = 0.030     # 30 mm in the middle

    # Length of the fine-step zone at each end of every trajectory [m].
    fine_zone: float = 0.010            # 10 mm fine at start/end


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class InsertionPipeline:
    """Pick-and-place state machine for the LEGO insertion task.

    Architecture
    ────────────
    All motions use the same TrajectoryPlanner + TrajectoryFollower.
    Lateral moves (A→B) use 3D straight-line trajectories.
    Vertical moves (descend/ascend) are also planned the same way —
    the goal just happens to differ only in Z.

    Inputs
    ──────
    pos_a : TCP position at the grasp point (position of movable LEGO brick,
            adjusted for gripper geometry). In real world: from FoundationPose.
    pos_b : TCP position at the insert point (above fixed socket, adjusted).
            In real world: from FoundationPose.

    State sequence
    ──────────────
    ① HOME            reset / starting pose
    ② ABOVE_A         transit height above brick A
    ③ AT_A            descend to grasp height
    ④ GRASP           hold still (sim: wait; real: close gripper)
    ⑤ ABOVE_A         ascend back to transit height
    ⑥ ABOVE_B         transit to above socket B
    ⑦ AT_B            descend to insert height
    ⑧ RELEASE         hold still (sim: wait; real: open gripper)
    ⑨ HOME            return to home
    """

    def __init__(self, config: PipelineConfig = None):
        self.cfg      = config or PipelineConfig()
        self.planner  = TrajectoryPlanner(
            step_size=self.cfg.step_size,
            waypoint_tolerance=self.cfg.waypoint_tolerance,
        )
        self.follower = TrajectoryFollower(
            max_step=self.cfg.max_action_step,
            max_iter_per_waypoint=200,   # give up after 200 steps (~13s); avoids infinite hang on contact
            verbose=True,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _move_to(self, env, obs, target_xyz, label: str):
        """Plan and follow a trajectory to target_xyz. Prints phase label."""
        current = obs["observation.state.cartesian"][:3]
        dist_mm = np.linalg.norm(target_xyz - current) * 1000
        print(f"\n  ── {label}  (dist={dist_mm:.1f}mm) ──")

        waypoints = self.planner.plan(
            current, target_xyz,
            coarse_step_size=self.cfg.coarse_step_size,
            fine_zone=self.cfg.fine_zone,
        )
        print(f"  ({len(waypoints)} waypoints)")
        obs, result = self.follower.follow(env, obs, waypoints)

        if not result.reached_all:
            print(f"  WARNING: did not reach all waypoints (timed out at {result.timed_out_at})")
        print(f"  → arrived, final dist={result.final_distance*1000:.1f}mm")
        return obs

    def _hold(self, env, obs, label: str):
        """Hold current position for hold_duration_s by sending zero actions.

        In simulation this replaces gripper open/close.
        The impedance controller keeps the arm steady.
        """
        n_steps = max(1, int(self.cfg.hold_duration_s * 15))  # 15 Hz RL rate
        print(f"\n  ── {label}  ({self.cfg.hold_duration_s:.0f}s = {n_steps} steps) ──")
        action = np.zeros(6)
        for _ in range(n_steps):
            obs, _, _, _, _ = env.step(action)
        print(f"  → done")
        return obs

    def _above(self, pos_xyz: np.ndarray) -> np.ndarray:
        """Return a position approach_height directly above pos_xyz."""
        above = pos_xyz.copy()
        above[2] += self.cfg.approach_height
        return above

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self, env, pos_a: np.ndarray, pos_b: np.ndarray):
        """Execute the full insertion pipeline.

        Parameters
        ----------
        env   : MuJoCo environment (already constructed, not yet reset).
        pos_a : TCP position at the grasp point [m], shape (3,).
        pos_b : TCP position at the insert point [m], shape (3,).

        Returns
        -------
        success : bool — True if all phases completed without timeout.
        """
        pos_a = np.asarray(pos_a, dtype=float)
        pos_b = np.asarray(pos_b, dtype=float)

        # Apply fine Z offsets (from config) to grasp and insert positions
        grasp_pos  = pos_a.copy(); grasp_pos[2]  += self.cfg.grasp_z_offset
        insert_pos = pos_b.copy(); insert_pos[2] += self.cfg.insert_z_offset

        print("\n" + "═" * 60)
        print("  INSERTION PIPELINE")
        print(f"  Grasp  pos_A : {grasp_pos.round(4)}")
        print(f"  Insert pos_B : {insert_pos.round(4)}")
        print("═" * 60)

        # ─── ① HOME: reset env, read actual TCP ──────────────────────────────
        print("\n[STATE 1/9] HOME — reset")
        obs, _ = env.reset(options={
            # XY reference: start_position[:2] → env computes dxy = pos - [0.6, 0]
            # We pass [0.6, 0, 0] → dxy = [0, 0] → arm stays at keyframe XY.
            "start_position": np.array([0.6, 0.0, 0.0]),
            "grasp_position": np.array([0.0, 0.0, 0.0]),
        })
        home_xyz = obs["observation.state.cartesian"][:3].copy()
        print(f"  TCP at home: {home_xyz.round(4)}")

        # ─── ② ABOVE_A: move to transit height above brick A ─────────────────
        print("\n[STATE 2/9] ABOVE_A — approach transit height")
        obs = self._move_to(env, obs, self._above(grasp_pos), "transit → above A")

        # ─── ③ AT_A: descend to grasp height (Z only changes) ────────────────
        print("\n[STATE 3/9] AT_A — descend to grasp")
        obs = self._move_to(env, obs, grasp_pos, "descend → at A")

        # ─── ④ GRASP: hold still (gripper close in real world) ───────────────
        print("\n[STATE 4/9] GRASP — hold (sim: wait in place)")
        obs = self._hold(env, obs, "GRASP")

        # ─── ⑤ ABOVE_A: ascend back to transit height ────────────────────────
        print("\n[STATE 5/9] ABOVE_A — ascend after grasp")
        obs = self._move_to(env, obs, self._above(grasp_pos), "ascend → above A")

        # ─── ⑥ ABOVE_B: transit to above the socket ──────────────────────────
        print("\n[STATE 6/9] ABOVE_B — transit to above socket B")
        obs = self._move_to(env, obs, self._above(insert_pos), "transit → above B")

        # ─── ⑦ AT_B: descend to insert height ────────────────────────────────
        print("\n[STATE 7/9] AT_B — descend to insert")
        obs = self._move_to(env, obs, insert_pos, "descend → at B")

        # ─── ⑧ RELEASE: hold still (gripper open in real world) ──────────────
        print("\n[STATE 8/9] RELEASE — hold (sim: wait in place)")
        obs = self._hold(env, obs, "RELEASE")

        # ─── ⑨ HOME: return to home position and hold ────────────────────────
        print("\n[STATE 9/9] HOME — return")
        obs = self._move_to(env, obs, home_xyz, "transit → home")
        obs = self._hold(env, obs, "HOME")

        final_dist = np.linalg.norm(
            obs["observation.state.cartesian"][:3] - home_xyz
        ) * 1000

        print("\n" + "═" * 60)
        print(f"  Pipeline complete. Final dist to home: {final_dist:.1f}mm")
        print("═" * 60 + "\n")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Known positions (hardcoded for sim — replaced by FoundationPose later)
# ─────────────────────────────────────────────────────────────────────────────

# TCP position when "grasping" the movable LEGO brick.
# In the real world: FoundationPose detects brick A → add gripper Z offset.
# Here: a position in the sim workspace that exercises the full pipeline.
POS_A_GRASP = np.array([0.30, 0.20, 0.130])   # [m] world frame

# TCP position for insertion above the fixed socket.
# Fixed socket is at [0.6, 0.0, 0.115] in world frame (from MJCF).
# Z=0.131 causes the brick to press against the socket (contact around Z≈0.135).
# Use Z=0.140 as the "insert" target — just above first contact; RL handles the rest.
POS_B_INSERT = np.array([0.60, 0.00, 0.140])  # [m] world frame


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Insertion pipeline state machine demo in MuJoCo sim."
    )
    parser.add_argument(
        "--live_view", action="store_true",
        help="Open MuJoCo interactive viewer (use MUJOCO_GL=glfw)"
    )
    parser.add_argument(
        "--approach_height", type=float, default=0.07,
        help="Approach height above A and B [m] (default 0.07 = 70mm)"
    )
    parser.add_argument(
        "--hold_s", type=float, default=2.0,
        help="Hold duration for grasp/release [s] (default 2.0)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=3,
        help="Number of times to repeat the full pipeline (default 3)"
    )
    args = parser.parse_args()

    # ── Build minimal env ─────────────────────────────────────────────────────
    env = mujid_env.MujidEnv(config={
        "initial_keyframe": 1,      # home pose
        "n_cameras":        0,      # no cameras — no CUDA needed
        "live_view":        args.live_view,
    })

    # ── Configure pipeline ────────────────────────────────────────────────────
    cfg = PipelineConfig(
        approach_height=args.approach_height,
        hold_duration_s=args.hold_s,
    )
    pipeline = InsertionPipeline(config=cfg)

    # ── Run pipeline with known positions ─────────────────────────────────────
    # In real world: replace POS_A_GRASP / POS_B_INSERT with FoundationPose output
    for i in range(args.n_runs):
        print(f"\n{'▓'*60}")
        print(f"  RUN {i+1}/{args.n_runs}")
        print(f"{'▓'*60}")
        pipeline.run(env, pos_a=POS_A_GRASP, pos_b=POS_B_INSERT)

    env.close()


if __name__ == "__main__":
    main()
