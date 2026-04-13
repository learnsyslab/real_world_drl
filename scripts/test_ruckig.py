"""Test harness for RuckigFollower on the raw MuJoCo environment.

Five modes
----------
A  3d           Free-space 3D position move (optionally compare vs Poly7Planner)
B  6d           Free-space 6D position+orientation move (optionally compare vs Poly7Planner6D)
C  xy           XY approach only — mirrors InsertionWrapperSimLEGO.go_to_waypoint()
D  streaming    Mid-motion goal shift demo (two-phase streaming)
E  pipeline     Full 9-phase LEGO insertion: home→above A→grasp→lift→transit→socket→XY→Z→RL
F  pipeline_6d  Same as pipeline but uses Ruckig6DFreeSpacePlanner (position + orientation)

Usage
-----
    cd /home/gabor/drl_project/real_world_drl

    # 1. Confirm ruckig installed
    pixi run -e sim python -c "import ruckig; print(ruckig.__version__)"

    # 2. Free-space 3D, compare vs Poly7
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode 3d --compare --n_trials 10 --verbose

    # 3. Free-space 6D
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode 6d --compare --n_trials 10

    # 4. XY approach (mirrors go_to_waypoint)
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode xy --n_trials 10

    # 5. Streaming mid-motion goal shift
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode streaming --live_view

    # 6. Full 9-phase LEGO insertion pipeline (3D free-space)
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode pipeline --n_trials 5 --live_view
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode pipeline --n_trials 20 --seed 42

    # 7. Full pipeline with 6D (position + orientation) free-space waypoints
    MUJOCO_GL=egl pixi run -e sim python scripts/test_ruckig.py --mode pipeline_6d --n_trials 5
"""

import argparse
import sys
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujid.env.env as mujid_env
from crisp_drl.motion_planning.free_space_planner import CartesianWaypoint
from crisp_drl.motion_planning.planners import GoToGoalPlanner
from crisp_drl.motion_planning.trajectory_planner import (
    Poly7Planner,
    Poly7Planner6D,
    TrajectoryFollower,
    RuckigFollower,
)
sys.path.insert(0, os.path.dirname(__file__))   # so 'test_motion_planner' is importable
from test_motion_planner import make_planner_env


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(live_view: bool = False) -> mujid_env.MujidEnv:
    return mujid_env.MujidEnv({
        "initial_keyframe": 2,
        "live_view": live_view,
        "n_cameras": 0,
    })


# ---------------------------------------------------------------------------
# Random offset helpers
# ---------------------------------------------------------------------------

def _random_aa(max_angle_deg: float) -> np.ndarray:
    axis  = np.random.randn(3); axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, np.radians(max_angle_deg))
    return axis * angle


def sample_offset(max_pos_m: float = 0.03, max_rot_deg: float = 20.0):
    return (
        np.random.uniform(-max_pos_m, max_pos_m, size=3),
        _random_aa(max_rot_deg),
    )


# ---------------------------------------------------------------------------
# Mode A/B — free-space 3D and 6D
# ---------------------------------------------------------------------------

def run_one_trial_ruckig(env, ruckig, mode, start_xyz, start_aa, goal_xyz, goal_aa):
    obs, _ = env.reset()
    # Move to start via Poly7 (not measured)
    p6d = Poly7Planner6D()
    f   = TrajectoryFollower(max_step=0.001)
    kf_xyz = obs["observation.state.cartesian"][:3].copy()
    kf_aa  = obs["observation.state.cartesian"][3:6].copy()
    wps = p6d.plan(kf_xyz, start_xyz, kf_aa, start_aa)
    obs, _ = f.follow_sampled_6d(env, obs, wps)

    if mode == "3d":
        obs = ruckig.follow_3d(env, obs, goal_xyz, tol=0.001)
    else:
        obs = ruckig.follow_6d(env, obs, goal_xyz, goal_aa, tol=0.001, orientation_tol=0.01)

    final_xyz = obs["observation.state.cartesian"][:3]
    final_aa  = obs["observation.state.cartesian"][3:6]
    pos_err   = float(np.linalg.norm(final_xyz - goal_xyz)) * 1000
    R_delta   = Rotation.from_rotvec(final_aa).inv() * Rotation.from_rotvec(goal_aa)
    rot_err   = float(np.degrees(np.linalg.norm(R_delta.as_rotvec())))
    return pos_err, rot_err


def run_one_trial_poly7(env, mode, start_xyz, start_aa, goal_xyz, goal_aa):
    obs, _ = env.reset()
    p3d = Poly7Planner()
    p6d = Poly7Planner6D()
    f   = TrajectoryFollower(max_step=0.001)
    kf_xyz = obs["observation.state.cartesian"][:3].copy()
    kf_aa  = obs["observation.state.cartesian"][3:6].copy()
    wps = p6d.plan(kf_xyz, start_xyz, kf_aa, start_aa)
    obs, _ = f.follow_sampled_6d(env, obs, wps)

    if mode == "3d":
        wps2 = p3d.plan(start_xyz, goal_xyz)
        obs, _ = f.follow_sampled(env, obs, wps2)
    else:
        wps2 = p6d.plan(start_xyz, goal_xyz, start_aa, goal_aa)
        obs, _ = f.follow_sampled_6d(env, obs, wps2)

    final_xyz = obs["observation.state.cartesian"][:3]
    final_aa  = obs["observation.state.cartesian"][3:6]
    pos_err   = float(np.linalg.norm(final_xyz - goal_xyz)) * 1000
    R_delta   = Rotation.from_rotvec(final_aa).inv() * Rotation.from_rotvec(goal_aa)
    rot_err   = float(np.degrees(np.linalg.norm(R_delta.as_rotvec())))
    return pos_err, rot_err


def mode_free_space(args):
    env    = make_env(live_view=args.live_view)
    ruckig = RuckigFollower(verbose=args.verbose)

    ruckig_pos_errs, ruckig_rot_errs = [], []
    poly7_pos_errs,  poly7_rot_errs  = [], []

    for i in range(args.n_trials):
        s_off, s_aa = sample_offset(args.start_range_mm / 1000, args.rot_range_deg)
        g_off, g_aa = sample_offset(args.goal_range_mm  / 1000, args.rot_range_deg)

        obs, _ = env.reset()
        kf_xyz = obs["observation.state.cartesian"][:3].copy()
        kf_aa  = obs["observation.state.cartesian"][3:6].copy()
        start_xyz = kf_xyz + s_off
        start_aa  = (Rotation.from_rotvec(s_aa) * Rotation.from_rotvec(kf_aa)).as_rotvec()
        goal_xyz  = start_xyz + g_off
        goal_aa   = (Rotation.from_rotvec(g_aa) * Rotation.from_rotvec(start_aa)).as_rotvec()

        rp, rr = run_one_trial_ruckig(env, ruckig, args.mode, start_xyz, start_aa, goal_xyz, goal_aa)
        ruckig_pos_errs.append(rp); ruckig_rot_errs.append(rr)

        if args.compare:
            pp, pr = run_one_trial_poly7(env, args.mode, start_xyz, start_aa, goal_xyz, goal_aa)
            poly7_pos_errs.append(pp); poly7_rot_errs.append(pr)
            print(f"  trial {i+1:3d}  Ruckig: pos={rp:.2f}mm rot={rr:.2f}°  |  "
                  f"Poly7:  pos={pp:.2f}mm rot={pr:.2f}°")
        else:
            print(f"  trial {i+1:3d}  pos_err={rp:.2f}mm  rot_err={rr:.2f}°")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode.upper()}  ({args.n_trials} trials)")
    print(f"Ruckig  pos_err: mean={np.mean(ruckig_pos_errs):.2f}mm  "
          f"max={np.max(ruckig_pos_errs):.2f}mm")
    if args.mode == "6d":
        print(f"Ruckig  rot_err: mean={np.mean(ruckig_rot_errs):.2f}°  "
              f"max={np.max(ruckig_rot_errs):.2f}°")
    if args.compare:
        print(f"Poly7   pos_err: mean={np.mean(poly7_pos_errs):.2f}mm  "
              f"max={np.max(poly7_pos_errs):.2f}mm")
        if args.mode == "6d":
            print(f"Poly7   rot_err: mean={np.mean(poly7_rot_errs):.2f}°  "
                  f"max={np.max(poly7_rot_errs):.2f}°")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Mode C — XY approach (go_to_waypoint replacement)
# ---------------------------------------------------------------------------

def mode_xy(args):
    env    = make_env(live_view=args.live_view)
    ruckig = RuckigFollower(verbose=args.verbose)

    errs = []
    for i in range(args.n_trials):
        obs, _ = env.reset()
        start_xy = obs["observation.state.cartesian"][:2].copy()
        # Random XY offset
        offset   = np.random.uniform(-args.goal_range_mm / 1000,
                                      args.goal_range_mm / 1000, size=2)
        target_xy = start_xy + offset
        dist_mm   = float(np.linalg.norm(offset)) * 1000

        obs = ruckig.follow_xy(env, obs, target_xy, distance_err=0.001)

        final_xy = obs["observation.state.cartesian"][:2]
        err_mm   = float(np.linalg.norm(final_xy - target_xy)) * 1000
        errs.append(err_mm)
        print(f"  trial {i+1:3d}  dist={dist_mm:.1f}mm  xy_err={err_mm:.2f}mm")

    print(f"\n{'='*60}")
    print(f"Mode: XY approach  ({args.n_trials} trials)")
    print(f"XY error: mean={np.mean(errs):.2f}mm  max={np.max(errs):.2f}mm  "
          f"std={np.std(errs):.2f}mm")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Mode D — streaming mid-motion goal shift
# ---------------------------------------------------------------------------

def mode_streaming(args):
    env    = make_env(live_view=args.live_view)
    ruckig = RuckigFollower(verbose=True)

    obs, _ = env.reset()
    start_xyz = obs["observation.state.cartesian"][:3].copy()
    start_aa  = obs["observation.state.cartesian"][3:6].copy()

    # Two goals: A = 30mm forward, B = 30mm sideways
    goal_A = start_xyz + np.array([0.03,  0.00, 0.0])
    goal_B = start_xyz + np.array([0.00,  0.03, 0.0])

    shift_at_step = 10  # switch goal after this many steps

    print(f"Start : {start_xyz.round(4)}")
    print(f"Goal A: {goal_A.round(4)}  (active for first {shift_at_step} steps)")
    print(f"Goal B: {goal_B.round(4)}  (switch at step {shift_at_step})")
    print()

    step_counter = [0]
    def goal_fn():
        g = goal_A if step_counter[0] < shift_at_step else goal_B
        step_counter[0] += 1
        return g, start_aa  # orientation unchanged

    obs = ruckig.follow_streaming(env, obs, goal_fn, max_steps=500, tol=0.001)

    final_xyz = obs["observation.state.cartesian"][:3]
    err_A = float(np.linalg.norm(final_xyz - goal_A)) * 1000
    err_B = float(np.linalg.norm(final_xyz - goal_B)) * 1000
    print(f"\nFinal pos: {final_xyz.round(4)}")
    print(f"  dist to goal_A: {err_A:.2f}mm")
    print(f"  dist to goal_B: {err_B:.2f}mm  (expected to be close)")
    print(f"  total steps: {step_counter[0]}")


# ---------------------------------------------------------------------------
# Mode E — full LEGO insertion pipeline (9 phases, all Ruckig)
# ---------------------------------------------------------------------------

def mode_pipeline(args):
    """Run the complete 10-phase LEGO insertion pipeline.

    Phases ①-⑥ (free-space motion) are driven by RuckigFreeSpacePlanner (3D) or
    Ruckig6DFreeSpacePlanner (6D, --use_6d) via waypoints_before_insertion.
    Phase ⑦ (XY alignment) uses RuckigFollower.follow_xy.
    Phase ⑧ (Z contact) uses the impedance controller loop.
    Phase ⑨ (RL insertion) uses GoToGoalPlanner as a deterministic baseline.
    Phase ⑩ (return home) uses follow_3d or follow_6d (--use_6d).

    Simulated LEGO pickup position (A): configurable via --grasp_xy.
    When --use_6d: orientation is read from home pose after first reset and
    propagated to all waypoints (constant orientation throughout).
    """
    use_6d   = args.use_6d
    grasp_xy = np.asarray(args.grasp_xy, dtype=float)

    # Build initial 3D waypoints (orientation_aa=None); if --use_6d, these will
    # be replaced with 6D versions after the first reset reads home_aa.
    def _make_waypoints(orientation_aa=None):
        return [
            CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.28], distance_err=0.005, orientation_aa=orientation_aa),  # ② above A
            CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.16], distance_err=0.003, orientation_aa=orientation_aa),  # ③ grasp height
            CartesianWaypoint([grasp_xy[0], grasp_xy[1], 0.28], distance_err=0.005, orientation_aa=orientation_aa),  # ④ lift
            CartesianWaypoint([0.60, 0.00, 0.28],               distance_err=0.005, orientation_aa=orientation_aa),  # ⑤ transit
            CartesianWaypoint([0.60, 0.00, 0.17],               distance_err=0.003, orientation_aa=orientation_aa),  # ⑥ above socket
        ]

    # Build env — use_ruckig_6d activates Ruckig6DFreeSpacePlanner
    env, wrapper = make_planner_env(
        live_view=args.live_view,
        wrapper="lego",
        use_ruckig=(not use_6d),
        use_ruckig_6d=use_6d,
        approach_distance=0.003,
        lego_waypoints=_make_waypoints(),   # 3D for now; patched below if --use_6d
    )

    # One dry reset to read home_aa (needed only for --use_6d waypoint construction)
    obs, _ = env.reset()
    home_aa = wrapper.home_aa.copy()

    if use_6d:
        # Patch waypoints with home orientation so Ruckig6DFreeSpacePlanner
        # tracks both position and orientation on every waypoint.
        wrapper.waypoints_before_insertion = _make_waypoints(orientation_aa=home_aa)

    mode_label = "Ruckig 6D (pos+ori)" if use_6d else "Ruckig 3D (pos only)"
    return_label = "Ruckig.follow_3d"  # always 3D — see phase ⑩ comment

    print(f"\nPipeline mode: {mode_label}")
    print(f"  ① home: keyframe-1 (env.reset() — true joint config)")
    labels = ["② above A", "③ grasp", "④ lift", "⑤ transit", "⑥ above socket"]
    for i, wp in enumerate(wrapper.waypoints_before_insertion):
        ori_str = f"  ori={np.degrees(wp.orientation_aa).round(1)}°" if wp.orientation_aa is not None else ""
        print(f"  {labels[i]}: {wp.position_xyz.round(4)}  (tol={wp.distance_err*1000:.1f}mm){ori_str}")
    print(f"  ⑦ XY align → goal  (Ruckig.follow_xy, tol={wrapper.approach_distance*1000:.1f}mm)")
    print(f"  ⑧ Z contact  (impedance ctrl)")
    print(f"  ⑨ RL insertion  (GoToGoalPlanner, {wrapper.step_limit} steps max)")
    print(f"  ⑩ Return home  ({return_label}, reverse path)")
    print()

    results = []
    planner = GoToGoalPlanner()

    for i in range(args.n_trials):
        print(f"── Trial {i+1}/{args.n_trials} ──────────────────────")
        t0 = time.time()
        # From trial 2 onward, reset() picks up the (possibly patched) waypoints.
        # Trial 1 already ran above as the dry reset; reuse its obs.
        if i > 0:
            obs, _ = env.reset()
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

        # Phase ⑩: return to home — reverse path, no teleport
        t_home = time.time()
        return_wps = [
            (np.array([0.60, 0.00, 0.17]), home_aa),
            (np.array([0.60, 0.00, 0.28]), home_aa),
            (np.array([grasp_xy[0], grasp_xy[1], 0.28]), home_aa),
            (wrapper.home_xyz, wrapper.home_aa),
        ]
        for wp_xyz, _ in return_wps:
            # Always use follow_3d for return: orientation target was never changed
            # during insertion (action[3:6]=0 throughout), so the impedance target
            # already holds the correct orientation. follow_6d would compute a
            # non-zero delta_rot from any tiny contact-induced orientation drift and
            # apply it as a Cartesian orientation command while the TCP is still at
            # the socket — causing wrist rotation before the brick is extracted.
            obs = wrapper._ruckig.follow_3d(wrapper.env, obs, wp_xyz, tol=0.005)
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

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(results)
    sr = sum(r["success"] for r in results) / n
    avg_reset = np.mean([r["reset_time"] for r in results])
    avg_rl    = np.mean([r["rl_time"]    for r in results])
    avg_steps = np.mean([r["rl_steps"]   for r in results])
    avg_home  = np.mean([r["home_time"]  for r in results])

    print(f"\n{'='*60}")
    print(f"Mode: pipeline / {mode_label}  ({n} trials)")
    print(f"Success rate  : {sr:.0%}  ({sum(r['success'] for r in results)}/{n})")
    print(f"Reset time    : mean={avg_reset:.1f}s  (phases ①-⑧)")
    print(f"RL steps      : mean={avg_steps:.0f}  time={avg_rl:.1f}s  (phase ⑨)")
    print(f"Return home   : mean={avg_home:.1f}s  (phase ⑩)")
    print(f"Total/episode : mean={avg_reset+avg_rl+avg_home:.1f}s")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["3d", "6d", "xy", "streaming", "pipeline", "pipeline_6d"], default="3d",
    )
    parser.add_argument("--n_trials",       type=int,   default=5)
    parser.add_argument("--compare",        action="store_true",
                        help="Side-by-side vs Poly7Planner (modes 3d/6d only)")
    parser.add_argument("--live_view",      action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--start_range_mm", type=float, default=20.0)
    parser.add_argument("--goal_range_mm",  type=float, default=30.0)
    parser.add_argument("--rot_range_deg",  type=float, default=20.0)
    parser.add_argument(
        "--grasp_xy", nargs=2, type=float, metavar=("X", "Y"), default=[0.50, -0.15],
        help="Simulated LEGO brick pickup XY position [m] (pipeline mode). Default: 0.50 -0.15"
    )
    parser.add_argument(
        "--use_6d", action="store_true",
        help="Pipeline mode: use Ruckig6DFreeSpacePlanner so waypoints track both "
             "position and orientation (follow_6d). Home orientation is propagated "
             "to all waypoints. Default: 3D position only."
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Mode   : {args.mode}")
    if args.mode in ("3d", "6d"):
        print(f"Trials : {args.n_trials}  compare={args.compare}")
        print(f"Ranges : pos ±{args.goal_range_mm:.0f}mm  rot ±{args.rot_range_deg:.0f}°")
    elif args.mode in ("pipeline", "pipeline_6d"):
        print(f"Trials   : {args.n_trials}")
        print(f"Grasp XY : {args.grasp_xy}")
        print(f"6D mode  : {args.mode == 'pipeline_6d' or args.use_6d}")
    print()

    if args.mode in ("3d", "6d"):
        mode_free_space(args)
    elif args.mode == "xy":
        mode_xy(args)
    elif args.mode == "streaming":
        mode_streaming(args)
    elif args.mode in ("pipeline", "pipeline_6d"):
        if args.mode == "pipeline_6d":
            args.use_6d = True
        mode_pipeline(args)


if __name__ == "__main__":
    main()
