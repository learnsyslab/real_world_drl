"""Test harness for Poly7Planner6D on the raw MuJoCo environment.

Three modes:
  3d        Free-space 3D position move (Poly7Planner)
  6d        Free-space 6D position+orientation move (Poly7Planner6D)
  pipeline  Full 10-phase LEGO insertion cycle driven by Poly7Planner

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

    # Full Poly7-driven pipeline (10 phases):
    MUJOCO_GL=glfw pixi run -e sim python scripts/test_poly7_6d.py --mode pipeline --n_trials 3 --live_view
    MUJOCO_GL=egl  pixi run -e sim python scripts/test_poly7_6d.py --mode pipeline --n_trials 20 --seed 42
"""

import argparse
import sys
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))   # for test_motion_planner import

import mujid.env.env as mujid_env
from crisp_drl.motion_planning.free_space_planner import CartesianWaypoint
from crisp_drl.motion_planning.planners import GoToGoalPlanner
from crisp_drl.motion_planning.trajectory_planner import (
    Poly7Planner,
    Poly7Planner6D,
    TrajectoryFollower,
)
from test_motion_planner import make_planner_env


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

# ---------------------------------------------------------------------------
# Mode: pipeline — full 10-phase LEGO insertion cycle via Poly7Planner
# ---------------------------------------------------------------------------

def mode_pipeline(args):
    """Full 10-phase LEGO insertion pipeline driven by Poly7Planner.

    Phases ①-⑥ (free-space) use Poly7FreeSpacePlanner via InsertionWrapperSimLEGO.
    Phase ⑦ (XY align) uses the P-controller go_to_waypoint().
    Phase ⑧ (Z contact) uses the impedance controller loop.
    Phase ⑨ (RL insertion) uses GoToGoalPlanner.
    Phase ⑩ (return home) uses Poly7Planner directly, no teleport.
    """
    grasp_xy = np.asarray(args.grasp_xy, dtype=float)

    pipeline_waypoints = [
        CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.28], distance_err=0.005),  # ② above A
        CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.16], distance_err=0.003),  # ③ grasp
        CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.28], distance_err=0.005),  # ④ lift
        CartesianWaypoint([0.60, 0.00, 0.28],               distance_err=0.005),  # ⑤ transit
        CartesianWaypoint([0.60, 0.00, 0.17],               distance_err=0.003),  # ⑥ above socket
    ]

    env, wrapper = make_planner_env(
        live_view=args.live_view,
        wrapper="lego",
        use_poly7=True,
        approach_distance=0.003,
        lego_waypoints=pipeline_waypoints,
    )

    print(f"\nPipeline waypoints (Poly7Planner):")
    print(f"  ① home: keyframe-1 (env.reset())")
    labels = ["② above A", "③ grasp", "④ lift", "⑤ transit", "⑥ above socket"]
    for i, wp in enumerate(pipeline_waypoints):
        print(f"  {labels[i]}: {wp.position_xyz.round(4)}  (tol={wp.distance_err*1000:.1f}mm)")
    print(f"  ⑦ XY align → goal  (go_to_waypoint P-ctrl, tol={wrapper.approach_distance*1000:.1f}mm)")
    print(f"  ⑧ Z contact  (impedance ctrl)")
    print(f"  ⑨ RL insertion  (GoToGoalPlanner, {wrapper.step_limit} steps max)")
    print(f"  ⑩ Return home  (Poly7Planner, reverse path)")
    print()

    results = []
    planner = GoToGoalPlanner()

    for i in range(args.n_trials):
        print(f"── Trial {i+1}/{args.n_trials} ──────────────────────")
        t0 = time.time()
        obs, _ = env.reset()          # phases ①-⑧
        t_reset = time.time() - t0
        print(f"  Reset (phases ①-⑧): {t_reset:.1f}s")

        # Phase ⑨: GoToGoal insertion
        true_goal = wrapper.goal_position_ground_truth + wrapper.grasp_position
        planner.reset(true_goal)
        t_rl = time.time()
        done = False
        while not done:
            action = planner.plan(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        t_rl = time.time() - t_rl

        events = info.get("custom_events", [])
        last_reason = events[-1][1] if events else None
        success = last_reason == "E_SUCCESS"
        status = "SUCCESS" if success else f"FAIL ({last_reason or '?'})"
        print(f"  RL (phase ⑨): {t_rl:.1f}s  {wrapper.n_steps} steps  → {status}")

        # Phase ⑩: return home via Poly7 (no teleport)
        t_home = time.time()
        return_waypoints_xyz = [
            np.array([0.60, 0.00, 0.17]),
            np.array([0.60, 0.00, 0.28]),
            np.array([grasp_xy[0], grasp_xy[1], 0.28]),
            wrapper.home_xyz,
        ]
        for wp_xyz in return_waypoints_xyz:
            wps = wrapper._poly7_planner.plan(
                obs["observation.state.cartesian"][:3], wp_xyz
            )
            obs, _ = wrapper._poly7_follower.follow_sampled(wrapper.env, obs, wps)
        t_home = time.time() - t_home
        home_err_mm = float(np.linalg.norm(
            obs["observation.state.cartesian"][:3] - wrapper.home_xyz
        )) * 1000
        print(f"  Return home (⑩): {t_home:.1f}s  home_err={home_err_mm:.1f}mm")

        results.append({
            "success":    success,
            "reset_time": t_reset,
            "rl_time":    t_rl,
            "home_time":  t_home,
            "rl_steps":   wrapper.n_steps,
        })

    n = len(results)
    sr = sum(r["success"] for r in results) / n
    print(f"\n{'='*60}")
    print(f"Mode: pipeline / Poly7Planner  ({n} trials)")
    print(f"Success rate  : {sr:.0%}  ({sum(r['success'] for r in results)}/{n})")
    print(f"Reset time    : mean={np.mean([r['reset_time'] for r in results]):.1f}s  (phases ①-⑧)")
    print(f"RL steps      : mean={np.mean([r['rl_steps']   for r in results]):.0f}  "
          f"time={np.mean([r['rl_time'] for r in results]):.1f}s  (phase ⑨)")
    print(f"Return home   : mean={np.mean([r['home_time']  for r in results]):.1f}s  (phase ⑩)")
    print(f"Total/episode : mean={np.mean([r['reset_time']+r['rl_time']+r['home_time'] for r in results]):.1f}s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",  type=int,  default=5)
    parser.add_argument("--live_view", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument(
        "--mode", choices=["3d", "6d", "pipeline"], default="6d",
        help="3d=Poly7Planner (position only), 6d=Poly7Planner6D (pos+rot), "
             "pipeline=full 10-phase LEGO insertion cycle"
    )
    parser.add_argument(
        "--grasp_xy", nargs=2, type=float, metavar=("X", "Y"), default=[0.50, -0.15],
        help="Simulated LEGO pickup XY position [m] (pipeline mode). Default: 0.50 -0.15"
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

    if args.mode == "pipeline":
        mode_pipeline(args)
        return

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
