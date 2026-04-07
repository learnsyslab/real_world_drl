"""Test harness for Poly7Planner6D on the raw MuJoCo environment.

Runs N trials: reset to start pose → (optionally move to a random start) →
plan 6D trajectory to goal → follow it → report position and orientation error.

No insertion wrapper needed — tests pure free-space 6D motion.

Usage
-----
    cd /home/gabor/drl_project/real_world_drl

    # Fixed offset, 5 trials (default):
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py

    # Random start + random goal, 20 trials:
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py --randomize --n_trials 20

    # Custom fixed offset:
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py \\
        --goal_offset_xyz 0.02 0.01 0.03 --goal_rot_offset_deg 0 0 30

    # Compare 3D vs 6D (random poses):
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py --randomize --mode 3d
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py --randomize --mode 6d

    # Watch in the viewer:
    MUJOCO_GL=egl pixi run -e sim python scripts/test_poly7_6d.py --randomize --live_view
"""

import argparse
import sys
import os

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujid.env.env as mujid_env
from crisp_drl.motion_planning.trajectory_planner import (
    Poly7Planner,
    Poly7Planner6D,
    TrajectoryFollower,
)


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(live_view: bool = False) -> mujid_env.MujidEnv:
    """Raw MujidEnv — no wrappers, no cameras."""
    return mujid_env.MujidEnv({
        "initial_keyframe": 2,   # closetoinsert — arm near socket, realistic workspace
        "live_view": live_view,
        "n_cameras": 0,
    })


# ---------------------------------------------------------------------------
# Random pose sampling
# ---------------------------------------------------------------------------

def _random_aa(max_angle_deg: float) -> np.ndarray:
    """Random axis-angle with magnitude uniformly sampled in [0, max_angle_deg]."""
    axis  = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, np.radians(max_angle_deg))
    return axis * angle


def sample_random_offset(
    max_pos_m:   float = 0.04,
    max_rot_deg: float = 30.0,
) -> tuple:
    """Return a random (pos_offset (3,), rot_aa_offset (3,)) within limits."""
    pos_offset = np.random.uniform(-max_pos_m, max_pos_m, size=3)
    rot_offset = _random_aa(max_rot_deg)
    return pos_offset, rot_offset


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    env,
    planner_3d: Poly7Planner,
    planner_6d: Poly7Planner6D,
    follower:   TrajectoryFollower,
    goal_offset_xyz,
    goal_rot_offset_aa,
    mode:       str   = "6d",
    start_offset_xyz  = None,
    start_rot_offset_aa = None,
    verbose:    bool  = False,
) -> dict:
    """Reset, optionally move to a different start, plan to goal, report errors.

    Parameters
    ----------
    goal_offset_xyz       : (3,) position offset from start [m].
    goal_rot_offset_aa    : (3,) orientation offset as axis-angle [rad] on top of start.
    mode                  : "3d" → Poly7Planner (position only)
                            "6d" → Poly7Planner6D (position + orientation)
    start_offset_xyz      : If given, move robot to (keyframe_pos + offset) before
                            planning — creates a genuinely different start pose.
    start_rot_offset_aa   : Matching rotation offset for the start move.
    """
    obs, _ = env.reset()

    # ── Optional: move to a randomised start pose ─────────────────────────────
    if start_offset_xyz is not None:
        kf_xyz = obs["observation.state.cartesian"][:3].copy()
        kf_aa  = obs["observation.state.cartesian"][3:6].copy()
        rand_start_xyz = kf_xyz + np.asarray(start_offset_xyz)
        R_s_offset = Rotation.from_rotvec(start_rot_offset_aa)
        rand_start_aa = (R_s_offset * Rotation.from_rotvec(kf_aa)).as_rotvec()

        if verbose:
            print(f"  Moving to random start ...")
        # Use 6D planner to reach the random start (this motion is not measured)
        wps_to_start = planner_6d.plan(kf_xyz, rand_start_xyz, kf_aa, rand_start_aa)
        obs, _ = follower.follow_sampled_6d(env, obs, wps_to_start)

    # ── Record actual start pose ───────────────────────────────────────────────
    start_xyz = obs["observation.state.cartesian"][:3].copy()
    start_aa  = obs["observation.state.cartesian"][3:6].copy()

    # ── Compute goal = start + offset ─────────────────────────────────────────
    goal_xyz    = start_xyz + np.asarray(goal_offset_xyz)
    R_g_offset  = Rotation.from_rotvec(goal_rot_offset_aa)
    goal_rot_aa = (R_g_offset * Rotation.from_rotvec(start_aa)).as_rotvec()

    pos_dist_mm  = float(np.linalg.norm(np.asarray(goal_offset_xyz))) * 1000
    rot_dist_deg = float(np.degrees(np.linalg.norm(goal_rot_offset_aa)))

    if verbose:
        print(f"  Start pos : {start_xyz.round(4)}")
        print(f"  Start rot : {np.degrees(start_aa).round(2)}°")
        print(f"  Goal  pos : {goal_xyz.round(4)}  (dist: {pos_dist_mm:.1f} mm)")
        print(f"  Goal  rot : {np.degrees(goal_rot_aa).round(2)}°  "
              f"(rot: {rot_dist_deg:.1f}°)")

    # ── Plan and follow ────────────────────────────────────────────────────────
    if mode == "3d":
        waypoints = planner_3d.plan(start_xyz, goal_xyz)
        obs, result = follower.follow_sampled(env, obs, waypoints)
    else:
        waypoints = planner_6d.plan(start_xyz, goal_xyz, start_aa, goal_rot_aa)
        obs, result = follower.follow_sampled_6d(env, obs, waypoints)

    # ── Measure errors ─────────────────────────────────────────────────────────
    final_xyz = obs["observation.state.cartesian"][:3]
    final_aa  = obs["observation.state.cartesian"][3:6]

    pos_err_mm = float(np.linalg.norm(final_xyz - goal_xyz)) * 1000

    R_delta    = Rotation.from_rotvec(final_aa).inv() * Rotation.from_rotvec(goal_rot_aa)
    rot_err_deg = float(np.degrees(np.linalg.norm(R_delta.as_rotvec())))

    if verbose:
        print(f"  Final pos : {final_xyz.round(4)}  (err={pos_err_mm:.2f} mm)")
        print(f"  Final rot : {np.degrees(final_aa).round(2)}°  (err={rot_err_deg:.2f}°)")
        print(f"  Steps     : {result.steps_taken}  waypoints={len(waypoints)}  "
              f"reached={result.reached_all}")

    return {
        "reached":       result.reached_all,
        "steps":         result.steps_taken,
        "pos_err_mm":    pos_err_mm,
        "rot_err_deg":   rot_err_deg,
        "n_waypoints":   len(waypoints),
        "pos_dist_mm":   pos_dist_mm,
        "rot_dist_deg":  rot_dist_deg,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(results: list, mode: str):
    n = len(results)
    n_reached = sum(r["reached"] for r in results)

    pos_errs   = [r["pos_err_mm"]   for r in results]
    rot_errs   = [r["rot_err_deg"]  for r in results]
    steps      = [r["steps"]        for r in results]
    n_wps      = [r["n_waypoints"]  for r in results]
    pos_dists  = [r["pos_dist_mm"]  for r in results]
    rot_dists  = [r["rot_dist_deg"] for r in results]

    print(f"\n{'='*60}")
    print(f"Mode             : Poly7Planner{'6D' if mode == '6d' else ' (3D)'}")
    print(f"Trials           : {n}")
    print(f"Reached          : {n_reached}/{n}  ({n_reached/n:.0%})")
    print(f"Distance (pos)   : mean={np.mean(pos_dists):.1f}mm  "
          f"min={np.min(pos_dists):.1f}  max={np.max(pos_dists):.1f}mm")
    if mode == "6d":
        print(f"Distance (rot)   : mean={np.mean(rot_dists):.1f}°  "
              f"min={np.min(rot_dists):.1f}  max={np.max(rot_dists):.1f}°")
    print(f"Position error   : mean={np.mean(pos_errs):.2f}mm  "
          f"max={np.max(pos_errs):.2f}mm  std={np.std(pos_errs):.2f}mm")
    if mode == "6d":
        print(f"Orientation error: mean={np.mean(rot_errs):.2f}°  "
              f"max={np.max(rot_errs):.2f}°  std={np.std(rot_errs):.2f}°")
    print(f"Steps            : mean={np.mean(steps):.0f}  "
          f"min={np.min(steps)}  max={np.max(steps)}")
    print(f"Waypoints        : mean={np.mean(n_wps):.0f}  "
          f"min={np.min(n_wps)}  max={np.max(n_wps)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",  type=int,  default=5)
    parser.add_argument("--live_view", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument(
        "--mode", choices=["3d", "6d"], default="6d",
        help="3d=Poly7Planner (position only), 6d=Poly7Planner6D (pos+rot)"
    )

    # ── Fixed-offset mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--goal_offset_xyz", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
        default=[0.02, 0.01, 0.03],
        help="Goal position offset from start [m] (fixed mode). Default: 20/10/30 mm"
    )
    parser.add_argument(
        "--goal_rot_offset_deg", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
        default=[5.0, 2.5, 10.0],
        help="Goal orientation offset [deg] as axis-angle (fixed mode). "
             "Default: [5, 2.5, 10]° — 11.5° rotation around a 3D axis, "
             "small enough for the arm to track at keyframe 2 without displacement"
    )

    # ── Random mode ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--randomize", action="store_true",
        help="Sample a new random start AND goal for every trial"
    )
    parser.add_argument(
        "--start_range_mm", type=float, default=30.0,
        help="[randomize] Max start offset from keyframe [mm]. Default: 30"
    )
    parser.add_argument(
        "--goal_range_mm", type=float, default=40.0,
        help="[randomize] Max goal offset from start [mm]. Default: 40"
    )
    parser.add_argument(
        "--rot_range_deg", type=float, default=30.0,
        help="[randomize] Max rotation offset [deg] for both start and goal. Default: 30"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # ── Planner params ────────────────────────────────────────────────────────
    parser.add_argument("--a_limit",     type=float, default=2.0)
    parser.add_argument("--alpha_limit", type=float, default=1.0)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Mode             : {'Poly7Planner6D' if args.mode == '6d' else 'Poly7Planner (3D)'}")
    print(f"Trials           : {args.n_trials}")
    if args.randomize:
        print(f"Pose sampling    : RANDOM")
        print(f"  start_range    : ±{args.start_range_mm:.0f} mm / ±{args.rot_range_deg:.0f}°")
        print(f"  goal_range     : ±{args.goal_range_mm:.0f} mm / ±{args.rot_range_deg:.0f}°")
    else:
        print(f"Goal pos offset  : {np.array(args.goal_offset_xyz)*1000} mm")
        print(f"Goal rot offset  : {args.goal_rot_offset_deg}°")
    print(f"a_limit          : {args.a_limit} m/s²")
    if args.mode == "6d":
        print(f"alpha_limit      : {args.alpha_limit} rad/s²")
    if args.seed is not None:
        print(f"Seed             : {args.seed}")
    print()

    env       = make_env(live_view=args.live_view)
    planner3d = Poly7Planner(a_limit=args.a_limit)
    planner6d = Poly7Planner6D(a_limit=args.a_limit, alpha_limit=args.alpha_limit)
    follower  = TrajectoryFollower(max_step=0.001, verbose=args.verbose)

    results = []
    for i in range(args.n_trials):
        if args.verbose or args.n_trials <= 5:
            print(f"\n── Trial {i+1}/{args.n_trials} ──")

        if args.randomize:
            s_off_xyz, s_off_aa = sample_random_offset(
                args.start_range_mm / 1000, args.rot_range_deg
            )
            g_off_xyz, g_off_aa = sample_random_offset(
                args.goal_range_mm / 1000, args.rot_range_deg
            )
        else:
            s_off_xyz, s_off_aa = None, None
            g_off_xyz = np.array(args.goal_offset_xyz)
            g_off_aa  = np.radians(np.array(args.goal_rot_offset_deg))

        r = run_trial(
            env, planner3d, planner6d, follower,
            g_off_xyz, g_off_aa,
            mode=args.mode,
            start_offset_xyz=s_off_xyz,
            start_rot_offset_aa=s_off_aa,
            verbose=args.verbose or args.n_trials <= 5,
        )
        results.append(r)

        if not (args.verbose or args.n_trials <= 5):
            status = "✓" if r["reached"] else "✗"
            print(
                f"  trial {i+1:3d}  {status}  "
                f"dist={r['pos_dist_mm']:.0f}mm/{r['rot_dist_deg']:.0f}°  "
                f"err={r['pos_err_mm']:.2f}mm/{r['rot_err_deg']:.2f}°  "
                f"steps={r['steps']}"
            )

    print_stats(results, args.mode)


if __name__ == "__main__":
    main()
