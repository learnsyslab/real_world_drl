"""Test harness for motion planners on the LEGO insertion task.

Builds a minimal env stack (no vision, no CUDA needed) and runs a chosen
planner for N episodes, reporting success rate and step statistics.

Usage
-----
    cd /home/gabor/drl_project/real_world_drl
    pixi shell -e jazzy
    python scripts/test_motion_planner.py                   # default: GoToGoal
    python scripts/test_motion_planner.py --planner oracle
    python scripts/test_motion_planner.py --planner spiral
    python scripts/test_motion_planner.py --planner go_to_goal --n_episodes 200
    python scripts/test_motion_planner.py --live_view       # open MuJoCo viewer
"""

import argparse
import numpy as np
import sys
import os

# allow running from repo root or scripts/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.motion_planning.free_space_planner import CartesianWaypoint
from crisp_drl.motion_planning.planners import GoToGoalPlanner, HybridPlanner, OraclePlanner, SpiralPlanner
from crisp_drl.motion_planning.sim_wrappers import (
    ActionTimeStampWrapper,
    CustomTerminationWrapper,
    InsertionWrapperSim,
    InsertionWrapperSimLEGO,
    LastObservationWrapper,
    custom_sim_termination,
)
import mujid.env.env as mujid_env


# ---------------------------------------------------------------------------
# Minimal env factory (no DINO, no ObsFormatter — raw obs dict)
# ---------------------------------------------------------------------------

def make_planner_env(
    live_view: bool = False,
    is_eval: bool = False,
    pe_accuracy: float = 0.0015,
    sac_config: Config = None,
    approach_distance: float = None,
    wrapper: str = "sim",
    use_ruckig: bool = False,
    use_poly7:  bool = False,
    lego_waypoints=None,
) -> tuple:
    """Return (env, insertion_wrapper).

    The insertion_wrapper reference lets planners read goal_position after reset.
    Stack: MujidEnv → ActionTimestamp → LastObs → InsertionWrapper* → CustomTermination

    wrapper:
        "sim"  → InsertionWrapperSim  (original; approach optional via approach_distance)
        "lego" → InsertionWrapperSimLEGO (Siemens-style; approach always runs in reset)
    """
    if sac_config is None:
        sac_config = Config()

    # keyframe 1 = home (arm in neutral pose, far from socket)
    # keyframe 2 = closetoinsert (arm already near socket — insertion-only mode)
    mujid_config = {
        "initial_keyframe": 1 if wrapper == "lego" else 2,
        "live_view": live_view,
        "n_cameras": 0,         # no cameras: faster reset, no CUDA needed
    }

    env = mujid_env.MujidEnv(config=mujid_config)
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)

    if wrapper == "lego":
        # Free-space waypoints: home → transit → approach height above socket.
        # Stop at Z=0.17 (above socket contact height) — phase ⑧ (Z contact loop)
        # handles the remaining descent and contact establishment.
        # NOTE: do NOT add a waypoint near Z≈0.12 — the LEGO brick collides with
        # the socket surface before the TCP can reach that Z, causing the planner
        # to loop forever. Let phase ⑧ handle all Z motion near contact.
        if lego_waypoints is None:
            lego_waypoints = [
                CartesianWaypoint([0.6, 0.0, 0.25], distance_err=0.005),  # transit height
                CartesianWaypoint([0.6, 0.0, 0.17], distance_err=0.003),  # above socket
            ]

        insertion_wrapper = InsertionWrapperSimLEGO(
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
            approach_distance=approach_distance if approach_distance is not None else 2 * pe_accuracy,
            waypoints_before_insertion=lego_waypoints,
            use_ruckig=use_ruckig,
            use_poly7=use_poly7,
        )
    else:
        insertion_wrapper = InsertionWrapperSim(
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
            approach_distance=approach_distance,
        )

    env = CustomTerminationWrapper(insertion_wrapper, termination_fn=custom_sim_termination)
    return env, insertion_wrapper


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    env,
    insertion_wrapper,
    planner,
    n_episodes: int,
    verbose: bool = False,
    fixed_start_xy=None,
    fixed_goal_xy=None,
):
    results = []

    reset_options = {}
    if fixed_start_xy is not None:
        reset_options["forced_start_xy"] = fixed_start_xy
    if fixed_goal_xy is not None:
        reset_options["forced_goal_xy"] = fixed_goal_xy

    for ep in range(n_episodes):
        obs, info = env.reset(options=reset_options if reset_options else None)

        # True goal: ground_truth + grasp offset (compensates for randomised grasp).
        # If the user forced a goal XY, use that directly.
        if fixed_goal_xy is not None:
            true_goal = np.array([fixed_goal_xy[0], fixed_goal_xy[1],
                                  insertion_wrapper.goal_position[2]])
        else:
            true_goal = (
                insertion_wrapper.goal_position_ground_truth
                + insertion_wrapper.grasp_position
            )
        planner.reset(true_goal)

        success = False
        n_steps = 0
        start_dist = np.linalg.norm(
            obs["observation.state.cartesian"][:2] - true_goal[:2]
        )

        for _ in range(insertion_wrapper.step_limit):
            action = planner.plan(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1

            if terminated:
                events = info.get("custom_events", [])
                success = any(e[1] == "E_SUCCESS" for e in events)
                break
            if truncated:
                break

        results.append({
            "success": success,
            "steps": n_steps,
            "start_dist_mm": start_dist * 1000,
        })

        if verbose or (ep + 1) % 10 == 0:
            recent = results[-min(10, len(results)):]
            recent_sr = sum(r["success"] for r in recent) / len(recent)
            print(
                f"  ep {ep+1:3d}/{n_episodes}  "
                f"{'✓' if success else '✗'}  "
                f"steps={n_steps:3d}  "
                f"start_dist={start_dist*1000:.1f}mm  "
                f"recent_SR={recent_sr:.0%}"
            )

    return results


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(results: list, planner_name: str):
    n = len(results)
    successes = [r for r in results if r["success"]]
    sr = len(successes) / n

    print(f"\n{'='*50}")
    print(f"Planner : {planner_name}")
    print(f"Episodes: {n}")
    print(f"Success : {len(successes)}/{n}  ({sr:.1%})")

    if successes:
        steps = [r["steps"] for r in successes]
        print(f"Steps (success): mean={np.mean(steps):.1f}  "
              f"min={np.min(steps)}  max={np.max(steps)}  std={np.std(steps):.1f}")

    all_steps = [r["steps"] for r in results]
    print(f"Steps (all)    : mean={np.mean(all_steps):.1f}  "
          f"min={np.min(all_steps)}  max={np.max(all_steps)}")

    dists = [r["start_dist_mm"] for r in results]
    print(f"Start dist [mm]: mean={np.mean(dists):.2f}  "
          f"min={np.min(dists):.2f}  max={np.max(dists):.2f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner", choices=["oracle", "go_to_goal", "spiral", "hybrid"], default="go_to_goal"
    )
    parser.add_argument(
        "--wrapper", choices=["sim", "lego"], default="sim",
        help="sim=InsertionWrapperSim (original), lego=InsertionWrapperSimLEGO (Siemens-style)"
    )
    parser.add_argument(
        "--use_ruckig", action="store_true",
        help="Use RuckigFollower for jerk-limited XY approach in InsertionWrapperSimLEGO reset()"
    )
    parser.add_argument(
        "--use_poly7", action="store_true",
        help="Use Poly7Planner for 7th-order polynomial free-space approach in InsertionWrapperSimLEGO reset()"
    )
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--pe_accuracy", type=float, default=0.0015,
                        help="Pose estimation accuracy [m], controls randomisation range")
    parser.add_argument("--live_view", action="store_true",
                        help="Open MuJoCo interactive viewer")
    parser.add_argument("--eval_mode", action="store_true",
                        help="Use eval randomisation (tighter ranges)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="SAC checkpoint dir for RL fine-phase (hybrid planner only)"
    )
    parser.add_argument(
        "--switch_distance", type=float, default=0.004,
        help="XY distance [m] at which hybrid planner switches from GoToGoal to RL"
    )
    parser.add_argument(
        "--approach_distance", type=float, default=None,
        help="If set, InsertionWrapperSim runs GoToGoal inside reset() until within "
             "this distance [m] of the goal — RL policy always starts close"
    )
    parser.add_argument(
        "--start_xy", nargs=2, type=float, metavar=("X", "Y"), default=None,
        help="Fixed start position [m], e.g. --start_xy 0.605 0.003"
    )
    parser.add_argument(
        "--goal_xy", nargs=2, type=float, metavar=("X", "Y"), default=None,
        help="Fixed goal position [m], e.g. --goal_xy 0.600 0.000"
    )
    args = parser.parse_args()

    # build planner
    if args.planner == "oracle":
        planner = OraclePlanner()
    elif args.planner == "go_to_goal":
        planner = GoToGoalPlanner()
    elif args.planner == "spiral":
        planner = SpiralPlanner(max_radius=2 * args.pe_accuracy + 0.001)
    elif args.planner == "hybrid":
        rl_planner = None
        if args.checkpoint:
            from crisp_drl.motion_planning.rl_policy import RLPolicyPlanner
            rl_planner = RLPolicyPlanner(args.checkpoint)
            print(f"RL checkpoint: {args.checkpoint}")
        else:
            print("No --checkpoint given; hybrid fine-phase will use GoToGoalPlanner.")
        planner = HybridPlanner(rl_planner=rl_planner, switch_distance=args.switch_distance)
        print(f"switch_distance: {args.switch_distance*1000:.1f} mm")

    print(f"Planner : {args.planner}")
    print(f"Wrapper : {args.wrapper}")
    print(f"Episodes: {args.n_episodes}")
    print(f"pe_accuracy: {args.pe_accuracy*1000:.1f} mm")
    print(f"eval_mode  : {args.eval_mode}")
    print()

    env, insertion_wrapper = make_planner_env(
        live_view=args.live_view,
        is_eval=args.eval_mode,
        pe_accuracy=args.pe_accuracy,
        approach_distance=args.approach_distance,
        wrapper=args.wrapper,
        use_ruckig=args.use_ruckig,
        use_poly7=args.use_poly7,
    )

    if args.start_xy or args.goal_xy:
        print(f"start_xy   : {args.start_xy}")
        print(f"goal_xy    : {args.goal_xy}")

    results = run_episodes(
        env, insertion_wrapper, planner, args.n_episodes,
        verbose=args.verbose,
        fixed_start_xy=args.start_xy,
        fixed_goal_xy=args.goal_xy,
    )
    print_stats(results, args.planner)


if __name__ == "__main__":
    main()
