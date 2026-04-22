"""Smoke-test for the simple EE-space motion planner.

Two modes:

    # Pure-math, no ROS, no hardware — verifies planning math + result shape.
    python scripts/test_motion_planner.py --math_only

    # Standalone Robot client (no gym env, no cameras, no policy).
    # Moves the robot through 3 small hops (~2 cm) starting from the
    # current EE pose. Safest hardware test.
    python scripts/test_motion_planner.py --standalone --max_linear_vel 0.03

    # Full env path matching run_sac.py (siemens + PE + no-ft). Exercises
    # the wrapper chain; same hops as --standalone.
    python scripts/test_motion_planner.py --full_env --use_pose_estimation \\
        --no_ft_sensor --max_linear_vel 0.03
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_drl.envs.motion_planner import (
    PlanAndExecuteResult,
    plan_and_execute,
    plan_and_execute_position,
    plan_and_execute_ruckig,
)
from crisp_py.utils.geometry import Pose


DEFAULT_WORKSPACE_BOX = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7))
# Each hop = (position offset, unit rpy about tool-frame axes).
# rpy is scaled by --rot_delta_deg at runtime; yaw-only here so roll/pitch
# don't crash the tool into table/camera. Yaw sum = 0 → net no orientation drift.
HOPS = [
    (np.array([0.02, 0.0, 0.0]),   np.array([0.0, 0.0,  1.0])),
    (np.array([-0.02, 0.02, 0.0]), np.array([0.0, 0.0, -2.0])),
    (np.array([0.0, -0.02, 0.0]),  np.array([0.0, 0.0,  1.0])),
]


def _print_result(tag: str, res: PlanAndExecuteResult) -> None:
    print(
        f"[{tag}] reached={res.reached} aborted={res.aborted} reason={res.reason} "
        f"elapsed={res.elapsed:.3f}s pos_err={res.final_pose_error_m*1e3:.2f}mm "
        f"rot_err={np.degrees(res.final_orient_error_rad):.2f}deg"
    )


class _FakeEnv:
    """Gym-env shim exposing .unwrapped.robot for the planner."""

    class _U:
        pass

    def __init__(self, robot) -> None:
        self.unwrapped = _FakeEnv._U()
        self.unwrapped.robot = robot


def _make_fake_env():
    class _FakeConfig:
        publish_frequency = 50.0

    class _FakeRobot:
        def __init__(self) -> None:
            self.config = _FakeConfig()
            self._pose = Pose(
                np.array([0.5, 0.0, 0.3]), Rotation.from_euler("xyz", [np.pi, 0, 0])
            )

        @property
        def end_effector_pose(self) -> Pose:
            return self._pose

        def set_target(self, position=None, pose=None):
            if pose is not None:
                self._pose = pose.copy()

    robot = _FakeRobot()
    return _FakeEnv(robot), robot


def _run_math_quintic(args: argparse.Namespace) -> int:
    env, robot = _make_fake_env()
    start = robot.end_effector_pose
    target = Pose(start.position + np.array([0.05, 0.02, -0.01]), start.orientation)

    res = plan_and_execute(
        env,
        target,
        max_linear_vel=args.max_linear_vel,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        dry_run=True,
    )
    _print_result("quintic dry_run", res)
    assert res.waypoints is not None and len(res.waypoints) >= 2
    assert np.allclose(res.waypoints[0].position, start.position, atol=1e-9)
    assert np.allclose(res.waypoints[-1].position, target.position, atol=1e-9)

    res2 = plan_and_execute(
        env,
        target,
        max_linear_vel=args.max_linear_vel,
        workspace_box=DEFAULT_WORKSPACE_BOX,
    )
    _print_result("quintic execute (fake robot)", res2)
    measured = robot.end_effector_pose
    assert np.allclose(measured.position, target.position, atol=1e-6), (
        f"fake robot did not track target: {measured.position} vs {target.position}"
    )

    bad_box = ((10.0, 11.0), (-1.0, 1.0), (-1.0, 1.0))
    res_bad = plan_and_execute(env, target, workspace_box=bad_box)
    _print_result("quintic outside-box (expect abort)", res_bad)
    assert res_bad.aborted and "outside_box" in res_bad.reason

    print("[math_only quintic] all assertions passed.")
    return 0


def _run_math_ruckig(args: argparse.Namespace) -> int:
    env, robot = _make_fake_env()
    start = robot.end_effector_pose
    target = Pose(start.position + np.array([0.05, 0.02, -0.01]), start.orientation)

    # Dry-run: trajectory duration populated, waypoints present, final pos matches target.
    res = plan_and_execute_ruckig(
        env,
        target,
        max_linear_vel=args.max_linear_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        dry_run=True,
    )
    _print_result("ruckig dry_run", res)
    assert res.trajectory_duration is not None and res.trajectory_duration > 0.0
    assert res.waypoints is not None and len(res.waypoints) >= 2
    assert np.allclose(res.waypoints[0].position, start.position, atol=1e-3)
    assert np.linalg.norm(res.waypoints[-1].position - target.position) < 5e-3

    # Full execute: fake robot snaps to set_target; final position must match target.
    res2 = plan_and_execute_ruckig(
        env,
        target,
        max_linear_vel=args.max_linear_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        workspace_box=DEFAULT_WORKSPACE_BOX,
    )
    _print_result("ruckig execute (fake robot)", res2)
    measured = robot.end_effector_pose
    assert np.linalg.norm(measured.position - target.position) < 1e-3, (
        f"fake robot did not track target: {measured.position} vs {target.position}"
    )
    assert res2.final_velocity is not None

    # Chained segments: call twice, feeding final_velocity into the next call's
    # initial_velocity. On a colinear continuation, Ruckig should accelerate
    # PAST v=0 into the second segment instead of decelerating to a full stop.
    env2, robot2 = _make_fake_env()
    start2 = robot2.end_effector_pose
    mid = Pose(start2.position + np.array([0.05, 0.0, 0.0]), start2.orientation)
    end = Pose(start2.position + np.array([0.10, 0.0, 0.0]), start2.orientation)

    through_vel = np.array([args.max_linear_vel, 0.0, 0.0])
    resA = plan_and_execute_ruckig(
        env2,
        mid,
        max_linear_vel=args.max_linear_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        hold_after=False,
        target_velocity=through_vel,
    )
    _print_result("ruckig chained seg 1", resA)

    resB = plan_and_execute_ruckig(
        env2,
        end,
        max_linear_vel=args.max_linear_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        initial_velocity=resA.final_velocity,
        initial_acceleration=resA.final_acceleration,
        hold_after=False,
    )
    _print_result("ruckig chained seg 2", resB)

    # Sanity: segment 2 should be strictly shorter in time than a v=0 start
    # of identical geometry if chaining actually carried momentum. We cannot
    # assert sign because on a fake robot there is no controller lag, but
    # Ruckig's internal duration should be smaller.
    resB_cold = plan_and_execute_ruckig(
        _make_fake_env()[0],
        Pose(start2.position + np.array([0.05, 0.0, 0.0]), start2.orientation),
        max_linear_vel=args.max_linear_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        hold_after=False,
    )
    print(
        f"[ruckig chaining] hot-start T={resB.trajectory_duration:.3f}s "
        f"vs cold-start T={resB_cold.trajectory_duration:.3f}s"
    )
    assert resB.trajectory_duration < resB_cold.trajectory_duration, (
        "hot-start chained segment should complete faster than v=0 cold start"
    )

    # Workspace-box safety.
    bad_box = ((10.0, 11.0), (-1.0, 1.0), (-1.0, 1.0))
    res_bad = plan_and_execute_ruckig(env, target, workspace_box=bad_box)
    _print_result("ruckig outside-box (expect abort)", res_bad)
    assert res_bad.aborted and "outside_box" in res_bad.reason

    print("[math_only ruckig] all assertions passed.")
    return 0


def run_math_only(args: argparse.Namespace) -> int:
    """Exercise the planner math without any ROS or robot client."""
    if args.backend in ("quintic", "both"):
        rc = _run_math_quintic(args)
        if rc != 0:
            return rc
    if args.backend in ("ruckig", "both"):
        rc = _run_math_ruckig(args)
        if rc != 0:
            return rc
    return 0


def _build_full_env(args: argparse.Namespace):
    import rclpy

    from crisp_drl.agents.shared.algorithm_config import Config
    from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
    from crisp_drl.envs import make_env

    if not rclpy.ok():
        rclpy.init()

    cfg = Config()
    if args.task == "lego":
        return make_env.create_real_env_v4(cfg, args=args)
    return make_env.create_real_env_s1_pe(
        alg_config=cfg, env_config=SiemensConfig(), args=args
    )


def _build_standalone_env(cartesian_param_config: str | None = None):
    import os

    import rclpy
    from crisp_py.robot.robot import Robot
    from crisp_py.robot.robot_config import FrankaConfig

    if not rclpy.ok():
        rclpy.init()
    robot_config = FrankaConfig(publish_frequency=50.0, time_to_home=1.0)
    robot = Robot(robot_config=robot_config, name="motion_planner_test")
    robot.wait_until_ready(timeout=15.0)

    if cartesian_param_config and os.path.exists(cartesian_param_config):
        print(f"Loading cartesian controller params from {cartesian_param_config}")
        robot.cartesian_controller_parameters_client.load_param_config(
            file_path=cartesian_param_config
        )

    print("Switching to cartesian_impedance_controller...")
    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.reset_targets()
    robot.wait_until_ready(timeout=5.0)
    print("Cartesian controller active.")
    return _FakeEnv(robot)


def run_hardware(args: argparse.Namespace) -> int:
    if args.full_env:
        env = _build_full_env(args)
    else:
        env = _build_standalone_env(cartesian_param_config=args.cartesian_param_config)

    robot = env.unwrapped.robot
    start = robot.end_effector_pose
    print(f"Starting EE position: {start.position}")

    common = dict(
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        workspace_box=DEFAULT_WORKSPACE_BOX,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        timeout=args.timeout,
    )
    ruckig_extra = dict(
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
    )

    def _go(target, chain_from=None):
        if args.backend == "ruckig":
            kw = {**common, **ruckig_extra}
            if chain_from is not None:
                kw["initial_velocity"] = chain_from.final_velocity
                kw["initial_acceleration"] = chain_from.final_acceleration
                kw["hold_after"] = False
            return plan_and_execute_ruckig(env, target, **kw)
        return plan_and_execute(env, target, **common)

    ok = True
    last_res = None
    for i, (offset, rpy_unit) in enumerate(HOPS[: args.n_hops]):
        rpy_rad = np.radians(rpy_unit * args.rot_delta_deg)
        delta_rot = Rotation.from_euler("xyz", rpy_rad)
        target_rot = start.orientation * delta_rot  # body-frame rotation
        target = Pose(start.position + offset, target_rot)
        print(
            f"\n--- hop {i+1}/{args.n_hops} offset={offset} "
            f"rpy_deg={rpy_unit * args.rot_delta_deg} backend={args.backend} ---"
        )
        res = _go(target, chain_from=last_res if args.chained else None)
        _print_result(f"hop {i+1}", res)
        if not res.reached and not args.chained:
            ok = False
            if args.abort_on_fail:
                print("[test] aborting further hops (reached=False).")
                break
        last_res = res if args.chained else None
        if not args.chained:
            time.sleep(0.5)

    target_back = Pose(start.position.copy(), start.orientation)
    print("\n--- return to start ---")
    res = _go(target_back, chain_from=last_res if args.chained else None)
    _print_result("return", res)
    ok = ok and res.reached

    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--math_only",
        action="store_true",
        help="Pure-math smoke test with a fake robot. No ROS, no hardware.",
    )
    mode.add_argument(
        "--standalone",
        action="store_true",
        help="Build only a crisp_py Robot client (no gym env). Default hardware mode.",
    )
    mode.add_argument(
        "--full_env",
        action="store_true",
        help="Build the full gym env stack (like run_sac.py), then stream poses.",
    )

    p.add_argument(
        "--backend",
        type=str,
        choices=["quintic", "ruckig", "both"],
        default="ruckig",
        help="Which planner to exercise. 'both' runs both in math_only mode.",
    )
    p.add_argument(
        "--chained",
        action="store_true",
        help="Ruckig only: feed previous hop's final velocity into the next "
        "call (no v=0 pitstop between hops).",
    )
    p.add_argument("--max_linear_vel", type=float, default=0.03)
    p.add_argument("--max_linear_acc", type=float, default=0.5)
    p.add_argument("--max_linear_jerk", type=float, default=5.0)
    p.add_argument("--max_angular_vel", type=float, default=0.3)
    p.add_argument(
        "--pos_tol",
        type=float,
        default=0.0035,
        help="Position tolerance in m. Cartesian impedance with error_clip=0.003 m "
        "cannot converge tighter than ~3 mm; default allows for that + noise.",
    )
    p.add_argument(
        "--rot_tol",
        type=float,
        default=0.05,
        help="Orientation tolerance in rad. ~2.9 deg; impedance steady-state ~1 deg.",
    )
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--n_hops", type=int, default=len(HOPS))
    p.add_argument(
        "--rot_delta_deg",
        type=float,
        default=10.0,
        help="Per-hop yaw amplitude (tool-z) in deg. HOPS yaw pattern is scaled by "
        "this. Use 0.0 for position-only hops.",
    )
    p.add_argument("--abort_on_fail", action="store_true")

    p.add_argument(
        "--cartesian_param_config",
        type=str,
        default="/home/gabor/repos/crisp_configs_realWorldDRL/control/clipped_cartesian_impedance.yaml",
        help="Cartesian impedance controller yaml loaded before switching. "
        "Pass '' to skip loading (uses whatever is currently on the controller).",
    )
    p.add_argument("--task", type=str, choices=["siemens", "lego"], default="siemens")
    p.add_argument("--use_pose_estimation", action="store_true")
    p.add_argument("--no_ft_sensor", action="store_true")
    p.add_argument("--eval", action="store_true", default=True)

    args = p.parse_args()

    if args.math_only:
        return run_math_only(args)

    if not args.standalone and not args.full_env:
        args.standalone = True

    try:
        return run_hardware(args)
    finally:
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
