"""Benchmark and compare trajectory planners in simulation.

Runs a fixed sequence of point-to-point moves (matching the insertion pipeline)
with each planner and reports per-move metrics.  Optionally saves matplotlib
plots of the recorded velocity and acceleration profiles.

Planners compared
-----------------
raised_cosine : TrajectoryPlanner (adaptive waypoint spacing) +
                TrajectoryFollower.follow() (raised-cosine velocity profile)
poly7         : Poly7Planner (7th-order polynomial, zero jerk at endpoints) +
                TrajectoryFollower.follow_sampled() (one step per waypoint)

Run
---
    cd /home/gabor/drl_project/real_world_drl
    MUJOCO_GL=egl  pixi run -e sim python scripts/benchmark_planners.py --planner all
    MUJOCO_GL=glfw pixi run -e sim python scripts/benchmark_planners.py --planner all --live_view --plot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujid.env.env as mujid_env
from crisp_drl.motion_planning.trajectory_planner import (
    Poly7Planner,
    TrajectoryFollower,
    TrajectoryPlanner,
)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry (matches test_trajectory.py)
# ─────────────────────────────────────────────────────────────────────────────

POS_A_GRASP  = np.array([0.30, 0.20, 0.130])   # grasp point
POS_B_INSERT = np.array([0.60, 0.00, 0.140])   # insert point
APPROACH_H   = 0.07                             # transit height above A/B [m]

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoveResult:
    label:         str
    dist_mm:       float
    t_min_s:       float        # theoretical T_min from Poly7 (informational)
    steps:         int
    wall_ms:       float
    final_err_mm:  float
    n_waypoints:   int
    tcp_positions: np.ndarray   # (steps, 3) — recorded at each env.step()


@dataclass
class PlannerResult:
    name:    str
    moves:   List[MoveResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Env helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(live_view: bool) -> mujid_env.MujidEnv:
    return mujid_env.MujidEnv(config={
        "initial_keyframe": 1,
        "n_cameras": 0,
        "live_view": live_view,
    })


def reset_env(env) -> dict:
    obs, _ = env.reset(options={
        "start_position": np.array([0.6, 0.0, 0.0]),
        "grasp_position": np.array([0.0, 0.0, 0.0]),
    })
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Recording follower wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RecordingFollower(TrajectoryFollower):
    """Wraps TrajectoryFollower to record TCP positions at every env.step()."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._record: List[np.ndarray] = []

    def _step(self, env, obs, action):
        obs = super()._step(env, obs, action)
        self._record.append(obs["observation.state.cartesian"][:3].copy())
        return obs

    def reset_record(self):
        self._record = []

    def get_positions(self) -> np.ndarray:
        return np.array(self._record) if self._record else np.empty((0, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Move sequence definition
# ─────────────────────────────────────────────────────────────────────────────

def build_move_sequence(home_xyz: np.ndarray):
    above_a = POS_A_GRASP.copy();  above_a[2] = POS_A_GRASP[2] + APPROACH_H
    above_b = POS_B_INSERT.copy(); above_b[2] = POS_B_INSERT[2] + APPROACH_H
    return [
        ("HOME→ABOVE_A",  home_xyz,    above_a),
        ("ABOVE_A→AT_A",  above_a,     POS_A_GRASP),
        ("AT_A→ABOVE_A",  POS_A_GRASP, above_a),
        ("ABOVE_A→ABOVE_B", above_a,   above_b),
        ("ABOVE_B→AT_B",  above_b,     POS_B_INSERT),
        ("AT_B→HOME",     POS_B_INSERT, home_xyz),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Run one planner through all moves
# ─────────────────────────────────────────────────────────────────────────────

def run_planner(
    name: str,
    env,
    moves,
    follower: RecordingFollower,
    plan_fn,
    follow_fn,
    poly7: Poly7Planner,
) -> PlannerResult:
    result = PlannerResult(name=name)
    obs = reset_env(env)

    for label, start, goal in moves:

        dist = np.linalg.norm(goal - start)
        t_min = poly7.t_min(dist)

        waypoints = plan_fn(start, goal)
        follower.reset_record()

        t0 = time.perf_counter()
        obs, fr = follow_fn(env, obs, waypoints)
        wall_ms = (time.perf_counter() - t0) * 1000

        tcp = follower.get_positions()

        result.moves.append(MoveResult(
            label        = label,
            dist_mm      = dist * 1000,
            t_min_s      = t_min,
            steps        = fr.steps_taken,
            wall_ms      = wall_ms,
            final_err_mm = fr.final_distance * 1000,
            n_waypoints  = len(waypoints),
            tcp_positions= tcp,
        ))

    return result


def _go_to_silent(follower, env, obs, target):
    """Move to target quietly (verbose off), return (obs, reached, steps)."""
    old_v = follower.verbose
    follower.verbose = False
    obs, reached, n = follower.go_to(env, obs, target, tolerance=0.003)
    follower.verbose = old_v
    return obs, reached, n


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: List[PlannerResult]):
    # Collect all move labels in order
    labels = [m.label for m in results[0].moves]

    header = f"{'Move':<22}  {'dist':>6}  {'T_min':>6}  " + "  ".join(
        f"{'steps':>5} {'wall_ms':>7} {'err_mm':>7} {'n_wp':>5}"
        for _ in results
    )
    subhdr = " " * 22 + "  " + " " * 6 + "  " + " " * 6 + "  " + "  ".join(
        f"[{r.name:^29}]" for r in results
    )

    print("\n" + "═" * len(header))
    print(subhdr)
    print(f"{'Move':<22}  {'dist':>6}  {'T_min':>6}  " + "  ".join(
        f"{'steps':>5} {'ms':>7} {'err':>7} {'n_wp':>5}" for _ in results
    ))
    print("─" * len(header))

    for i, label in enumerate(labels):
        moves = [r.moves[i] for r in results]
        dist_mm = moves[0].dist_mm
        t_min   = moves[0].t_min_s
        row = f"{label:<22}  {dist_mm:>5.0f}mm  {t_min:>5.2f}s  "
        row += "  ".join(
            f"{m.steps:>5} {m.wall_ms:>7.0f} {m.final_err_mm:>6.2f}mm {m.n_waypoints:>5}"
            for m in moves
        )
        print(row)

    print("═" * len(header))

    # Totals
    print(f"\n{'TOTAL':<22}  {'':>6}  {'':>6}  " + "  ".join(
        f"{sum(m.steps for m in r.moves):>5} {sum(m.wall_ms for m in r.moves):>7.0f} {'':>7} {'':>5}"
        for r in results
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: List[PlannerResult], save_path: str = "benchmark_results.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[plot] matplotlib not available — skipping plots")
        return

    n_moves   = len(results[0].moves)
    n_metrics = 3   # position along path, velocity, acceleration
    colors    = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig = plt.figure(figsize=(5 * n_moves, 3 * n_metrics), constrained_layout=True)
    fig.suptitle("Planner benchmark — velocity & acceleration profiles", fontsize=13)

    gs = gridspec.GridSpec(n_metrics, n_moves, figure=fig)

    metric_labels = ["Arc position [mm]", "Speed [mm/step]", "Acceleration [mm/step²]"]

    for col, move_label in enumerate(m.label for m in results[0].moves):
        for row in range(n_metrics):
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                ax.set_title(move_label, fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_labels[row], fontsize=8)
            ax.set_xlabel("step", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.4)

            for ri, r in enumerate(results):
                tcp = r.moves[col].tcp_positions
                if len(tcp) < 2:
                    continue

                # Arc length along recorded path
                deltas = np.diff(tcp, axis=0)
                step_dists = np.linalg.norm(deltas, axis=1) * 1000   # mm
                arc = np.concatenate([[0], np.cumsum(step_dists)])

                if row == 0:
                    ax.plot(arc, label=r.name, color=colors[ri], linewidth=1.4)
                    ax.set_ylabel("Arc length [mm]", fontsize=8)
                elif row == 1:
                    ax.plot(step_dists, label=r.name, color=colors[ri], linewidth=1.4)
                elif row == 2:
                    accel = np.diff(step_dists)
                    ax.plot(accel, label=r.name, color=colors[ri], linewidth=1.4)

            if col == n_moves - 1 and row == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.savefig(save_path, dpi=120)
    print(f"\n[plot] saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare trajectory planners in MuJoCo simulation."
    )
    parser.add_argument(
        "--planner", choices=["raised_cosine", "poly7", "all"], default="all",
        help="Which planner(s) to benchmark (default: all)",
    )
    parser.add_argument("--live_view", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--plot", action="store_true",
                        help="Save velocity/acceleration plots to benchmark_results.png")
    parser.add_argument(
        "--a_limit", type=float, default=2.0,
        help="Acceleration limit for Poly7Planner [m/s²] (default 2.0)",
    )
    args = parser.parse_args()

    # ── shared objects ────────────────────────────────────────────────────────
    follower = RecordingFollower(
        max_step=0.003,
        max_iter_per_waypoint=200,
        verbose=True,
    )
    poly7 = Poly7Planner(a_limit=args.a_limit, env_dt=0.066)

    rc_planner = TrajectoryPlanner(
        step_size=0.001,
        waypoint_tolerance=0.002,
    )

    planner_configs = {
        "raised_cosine": (
            lambda s, g: rc_planner.plan(s, g, coarse_step_size=0.030, fine_zone=0.010),
            follower.follow,
        ),
        "poly7": (
            poly7.plan,
            follower.follow_sampled,
        ),
    }

    active = (
        list(planner_configs.keys()) if args.planner == "all" else [args.planner]
    )

    # ── create env once, share across planners ────────────────────────────────
    env = make_env(args.live_view)
    obs = reset_env(env)
    home_xyz = obs["observation.state.cartesian"][:3].copy()
    print(f"\nHome TCP: {home_xyz.round(4)}")

    moves = build_move_sequence(home_xyz)

    # ── run each planner ──────────────────────────────────────────────────────
    results: List[PlannerResult] = []
    for name in active:
        plan_fn, follow_fn = planner_configs[name]
        print(f"\n{'▓'*60}")
        print(f"  PLANNER: {name.upper()}")
        print(f"{'▓'*60}")
        r = run_planner(name, env, moves, follower, plan_fn, follow_fn, poly7)
        results.append(r)

    env.close()

    # ── report ────────────────────────────────────────────────────────────────
    if results:
        print_table(results)

    if args.plot and results:
        plot_results(results)


if __name__ == "__main__":
    main()
