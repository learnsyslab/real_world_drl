"""Single-brick FoundationPose + MotionPlanner sanity test for 2x4 LEGO.

Runs WITHOUT the trained 2x2 RL policy. Drives:
    env.reset() (home) ->
    plan_and_execute(overview pose from LegoConfig.demo_goal_pose_estimation_euler) ->
    capture rgb + depth ->
    Sam3Detector.segment_lego + PoseEstimator.estimate_lego(brick_size=2x4) +
        PoseTracker.track_lego ->
    compute world pose of lavender brick ->
    plan_and_execute(top-down hover 5 cm above brick center) ->
    save pose-overlay PNGs.

No grasp. No purple brick. No SuccessClassificationWrapper. No DINO/formatter.

Usage:
    pixi run -e jazzy python scripts/test_pe_motion_planner_lego_2x4.py \\
        --max_linear_vel 0.01 --pose_viz_dir pe_viz/lego_2x4

Prereqs:
    - franka stack running,
    - wrist RGB-D publishing,
    - Isaac ROS FoundationPose / SAM3 alive,
    - BOTA disconnected (this script always uses the no-ft env path),
    - lego_2x4_lavender_up.obj exists (run scripts/prepare_lego_2x4_meshes.py first).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_py.utils.geometry import Pose

WORKSPACE_BOX = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7))


def _euler_pose(xyzrpy: np.ndarray) -> Pose:
    pos = np.asarray(xyzrpy[:3], dtype=np.float64)
    rot = Rotation.from_euler("xyz", np.asarray(xyzrpy[3:], dtype=np.float64))
    return Pose(pos, rot)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_pe_mp_2x4")

    p = argparse.ArgumentParser()
    p.add_argument("--brick", type=str, choices=["lavender", "purple"], default="lavender")
    p.add_argument("--brick_size", type=str, choices=["2x2", "2x4"], default="2x4")
    p.add_argument("--hover_above_m", type=float, default=0.05)
    p.add_argument("--max_linear_vel", type=float, default=0.01)
    p.add_argument("--max_angular_vel", type=float, default=0.3)
    p.add_argument("--max_linear_acc", type=float, default=0.5)
    p.add_argument("--max_linear_jerk", type=float, default=5.0)
    p.add_argument("--pos_tol", type=float, default=0.0035)
    p.add_argument("--rot_tol", type=float, default=0.05)
    p.add_argument("--mp_timeout", type=float, default=20.0)
    p.add_argument("--pose_viz_dir", type=str, default="pe_viz/lego_2x4")
    p.add_argument("--mp_backend", type=str, choices=["quintic", "ruckig", "spline"], default="ruckig")
    p.add_argument("--dry_run", action="store_true", help="Skip the second MP (no approach motion).")
    args = p.parse_args()

    import rclpy
    from crisp_drl.agents.shared.env_wrappers import (
        ActionTimeStampWrapper,
        LastObservationWrapper,
        MotionPlannerWrapper,
        NoGripperActionWrapper,
        ZeroFTInjectorWrapper,
    )
    from crisp_drl.agents.shared.insertion_env_config import LegoConfig
    from crisp_drl.envs.make_env import make_env
    from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
    from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer

    if not rclpy.ok():
        rclpy.init()

    env_config = LegoConfig()
    env_config.brick_size = args.brick_size

    # --- Build a minimal env: base -> ActionTimeStamp -> NoGripperAction ->
    #     LastObservation -> ZeroFTInjector -> MotionPlannerWrapper.
    # Skip InsertionWrapperLegoPE / DINO / formatter / classifier entirely.
    env = make_env("my_env_v4_no_ft")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    env = ZeroFTInjectorWrapper(env)
    env = MotionPlannerWrapper(env, backend=args.mp_backend)

    helper = PoseEstimationHelper(
        assumed_orientation=np.array([]),
        lock_orientation=False,
        brick_size=args.brick_size,
    )

    overlay_renderer = None
    if args.pose_viz_dir:
        os.makedirs(args.pose_viz_dir, exist_ok=True)
        mesh_path = (
            f"/workspaces/isaac_ros-dev/lego_assets/lego_{args.brick_size}_{args.brick}_up.obj"
        )
        if not os.path.exists(mesh_path):
            mesh_path = os.path.expanduser(
                f"~/workspaces/isaac_ros-dev/lego_assets/lego_{args.brick_size}_{args.brick}_up.obj"
            )
        overlay_renderer = PoseOverlayRenderer(
            camera_info_json_path="camera_parameters/realsense_d405_single.json",
            mesh_path=mesh_path,
            use_default_mesh_fallback=False,
        )

    mp_kwargs = dict(
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        max_linear_acc=args.max_linear_acc,
        max_linear_jerk=args.max_linear_jerk,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        timeout=args.mp_timeout,
        workspace_box=WORKSPACE_BOX,
    )

    summary: dict = {}
    try:
        # 1. Reset to home.
        env.reset()
        logger.info("Home reset complete.")

        # 2. MP to overview pose (where current 2x2 pipeline does its wide PE).
        overview_pose = _euler_pose(env_config.demo_goal_pose_estimation_euler)
        logger.info("Plan+execute -> overview xyz=%s", overview_pose.position)
        res1 = env.plan_and_execute(overview_pose, **mp_kwargs)
        logger.info(
            "Overview reached=%s elapsed=%.2fs final_err=%.2fmm",
            res1.reached,
            res1.elapsed,
            res1.final_pose_error_m * 1e3,
        )
        if not res1.reached:
            logger.error("Overview MP did not reach. Aborting.")
            return 2

        # 3. Settle 1 step to refresh obs.
        zero6 = np.zeros(6, dtype=np.float32)
        obs, *_ = env.step(zero6)

        rgb = obs["observation.images.wrist_camera"]
        depth_raw = obs["observation.images.wrist_depth_camera"]
        depth_f32 = depth_raw.astype(np.float32) / 1000.0  # mm -> m
        tcp_cart = obs["observation.state.cartesian"]

        logger.info("Captured rgb shape=%s depth shape=%s", rgb.shape, depth_raw.shape)

        # 4. Segment + estimate + track.
        masks = helper.segmenter.segment_lego(rgb)
        if args.brick not in masks:
            logger.error(
                "Sam3 returned no mask for %r; got keys=%s", args.brick, list(masks)
            )
            return 3

        t0 = time.time()
        pose_cam_coarse = helper.pose_estimator.estimate_lego(
            rgb, depth_f32, masks[args.brick], args.brick
        )
        t1 = time.time()
        logger.info("FP estimate (%.2fs):\n%s", t1 - t0, pose_cam_coarse)

        pose_cam_refined = helper.pose_tracker.track_lego(
            rgb, depth_f32, pose_cam_coarse, args.brick, passes=4
        )
        t2 = time.time()
        logger.info("FP tracker (%.2fs):\n%s", t2 - t1, pose_cam_refined)

        pose_world = helper._compute_pose_in_world_frame(pose_cam_refined, tcp_cart)
        brick_pos_world = pose_world[:3, 3]
        logger.info("Brick world pos = %s", brick_pos_world)

        # 5. Save overlay.
        if overlay_renderer is not None:
            try:
                paths = overlay_renderer.save_pair(
                    out_dir=args.pose_viz_dir,
                    episode_idx=0,
                    rgb=rgb,
                    mask=masks[args.brick],
                    pose_cam_coarse=pose_cam_coarse,
                    pose_cam_refined=pose_cam_refined,
                )
                logger.info("Saved overlays: %s", paths)
            except Exception as e:
                logger.warning("PoseOverlayRenderer.save_pair failed: %s", e)

        # 6. Sanity-check: brick must be inside workspace.
        if not all(lo <= v <= hi for v, (lo, hi) in zip(brick_pos_world, WORKSPACE_BOX)):
            logger.error(
                "Estimated brick pos %s outside workspace box %s — refusing to plan.",
                brick_pos_world,
                WORKSPACE_BOX,
            )
            return 4

        # 7. MP to top-down hover above brick.
        if args.dry_run:
            logger.info("--dry_run: skipping approach MP.")
        else:
            robot = env.unwrapped.robot
            current_orientation = robot.end_effector_pose.orientation  # already top-down
            hover_xyz = brick_pos_world + np.array([0.0, 0.0, args.hover_above_m])
            target = Pose(hover_xyz, current_orientation)
            logger.info("Plan+execute -> hover xyz=%s", hover_xyz)
            res2 = env.plan_and_execute(target, **mp_kwargs)
            logger.info(
                "Hover reached=%s elapsed=%.2fs final_err=%.2fmm",
                res2.reached,
                res2.elapsed,
                res2.final_pose_error_m * 1e3,
            )
            summary["hover_reached"] = res2.reached
            summary["hover_err_m"] = res2.final_pose_error_m

        summary["brick_pos_world"] = brick_pos_world.tolist()
        summary["pose_world"] = pose_world.tolist()
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
