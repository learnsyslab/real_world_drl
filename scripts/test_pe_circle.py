#!/usr/bin/env python3
"""PE quality test: circular sweep over lavender + yellow bricks at two heights.

Motion plan:
  1. Env init + home, lift to coarse_z.
  2. Sweep a circle (coarse_radius) around coarse_center at coarse_z.
     At n_samples equally-spaced angles: run PE for both bricks, render overlays.
     Save per-sample PNGs + one 2×3 composite per color.
  3. Move to refined_center at refined_z (closer to bricks).
  4. Same sweep + PE at refined height.
     Save per-sample PNGs + one 2×3 composite per color.

Gripper stays in its initial orientation throughout — no alignment steps.

Usage:
  pixi run -e jazzy python scripts/test_pe_circle.py \\
      --out_dir pe_viz/lego_2x4_pe_test \\
      --grasp_color lavender --target_color yellow \\
      --n_samples 6 --coarse_radius 0.025 --refined_radius 0.015
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import cv2
import numpy as np
import rclpy

from crisp_drl.agents.shared.insertion_env_config import LegoConfig2x4
from crisp_drl.agents.shared.insertion_wrapper import SensorTareWrapper, install_stop_handler
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer
from crisp_gym.envs.manipulator_env import make_env as _make_gym_env
from crisp_drl.agents.shared.env_wrappers import (
    ActionTimeStampWrapper,
    LastObservationWrapper,
    NoGripperActionWrapper,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

I_TERM_CLIP = 0.001  # max per-step correction (m) for fine position loop


# ── env movement helpers ─────────────────────────────────────────────────────


def step_zeros(env):
    obs, *_ = env.step(np.zeros(6))
    return obs


def step_translation(env, dxyz):
    action6 = np.zeros(6)
    action6[:3] = np.asarray(dxyz, dtype=np.float64)
    obs, *_ = env.step(action6)
    return obs


def go_to_cartesian(
    env,
    obs,
    target_xyz,
    fine_resolution: float | None = None,
    max_iter: int = 300,
) -> dict:
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    obs = step_translation(env, target_xyz - obs["observation.state.cartesian"][:3])
    for _ in range(max_iter):
        err = np.linalg.norm(target_xyz - obs["observation.state.cartesian"][:3])
        vel = np.linalg.norm(obs["observation.velocity.cartesian"][:3])
        if err < 0.002 and vel < 0.001:
            break
        obs = step_zeros(env)
    if fine_resolution is not None:
        err_vec = target_xyz - obs["observation.state.cartesian"][:3]
        for _ in range(max_iter):
            if np.linalg.norm(err_vec) <= fine_resolution:
                break
            obs = step_translation(env, np.clip(err_vec, -I_TERM_CLIP, I_TERM_CLIP))
            err_vec = target_xyz - obs["observation.state.cartesian"][:3]
        for _ in range(max_iter):
            if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) <= 0.0005:
                break
            obs = step_zeros(env)
    return obs


# ── PE helper ────────────────────────────────────────────────────────────────


def do_pe(
    pe_helper: PoseEstimationHelper,
    env,
    obs,
    colors: list[str],
    object_type: str = "lego",
):
    """Settle x2 then run SAM3 + FoundationPose.

    For object_type="lego": segments two LEGO bricks by color.
    For object_type="abus": segments the ABUS key (single object, key="abus").

    Returns (obs, results) where results[key] = {pose_cam, world_pose, mask} or None.
    """
    obs = step_zeros(env)
    obs = step_zeros(env)

    rgb = obs["observation.images.wrist_camera"]
    depth_f32 = obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
    tcp_cart = obs["observation.state.cartesian"]

    results: dict = {"rgb": rgb.copy(), "tcp_cart": tcp_cart.copy()}

    if object_type == "abus":
        try:
            mask = pe_helper.segmenter.segment_abus(rgb)
        except Exception as exc:
            log.error("segment_abus failed: %s", exc)
            results["abus"] = None
            return obs, results
        try:
            pose_cam = pe_helper.pose_estimator.estimate_abus(rgb, depth_f32, mask)
            world_pose = pe_helper._compute_pose_in_world_frame(pose_cam, tcp_cart)
            results["abus"] = {"pose_cam": pose_cam, "world_pose": world_pose, "mask": mask}
            log.info("abus world xyz = [%.4f, %.4f, %.4f]", *world_pose[:3, 3])
        except Exception as exc:
            log.error("PE failed for abus: %s", exc)
            results["abus"] = None
        return obs, results

    # lego path
    try:
        masks = pe_helper.segmenter.segment_lego(rgb, colors=tuple(colors))
    except Exception as exc:
        log.error("segment_lego failed: %s", exc)
        return obs, results

    for color in colors:
        if color not in masks:
            log.warning("SAM3 did not return mask for %r (got %s)", color, list(masks))
            results[color] = None
            continue
        try:
            pose_cam = pe_helper.pose_estimator.estimate_lego(rgb, depth_f32, masks[color], color)
            world_pose = pe_helper._compute_pose_in_world_frame(pose_cam, tcp_cart)
            results[color] = {
                "pose_cam": pose_cam,
                "world_pose": world_pose,
                "mask": masks[color],
            }
            log.info("%s world xyz = [%.4f, %.4f, %.4f]", color, *world_pose[:3, 3])
        except Exception as exc:
            log.error("PE failed for %r: %s", color, exc)
            results[color] = None

    return obs, results


# ── visualization helpers ────────────────────────────────────────────────────


def render_result(
    renderer: PoseOverlayRenderer,
    results: dict,
    color: str,
    label: str,
    silhouette_color: tuple[int, int, int],
) -> np.ndarray:
    """Return a BGR overlay image; returns a black frame if result is None."""
    r = results.get(color)
    rgb = results.get("rgb")
    if r is None or rgb is None:
        h, w = (480, 640) if rgb is None else (rgb.shape[0], rgb.shape[1])
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame, f"{color}: NOT DETECTED", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        return frame
    return renderer.render(
        rgb,
        r["pose_cam"],
        mask=r.get("mask"),
        label=label,
        silhouette_color=silhouette_color,
    )


def make_composite(images_bgr: list, nrows: int = 2, ncols: int = 3, thumb_hw=(240, 320)) -> np.ndarray:
    """Stack list of BGR images into a nrows×ncols grid."""
    h, w = thumb_hw
    padded = []
    for img in images_bgr:
        padded.append(cv2.resize(img, (w, h)) if img is not None else np.zeros((h, w, 3), dtype=np.uint8))
    while len(padded) < nrows * ncols:
        padded.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = [np.concatenate(padded[r * ncols:(r + 1) * ncols], axis=1) for r in range(nrows)]
    return np.concatenate(rows, axis=0)


# ── circle sweep ─────────────────────────────────────────────────────────────


_SIL_COLORS = {
    "lavender": (200, 0, 200),
    "purple": (180, 0, 180),
    "yellow": (0, 180, 255),
    "abus": (0, 220, 100),
}
_DEFAULT_SIL = (180, 180, 180)


def circle_sweep(
    env,
    obs,
    pe_helper: PoseEstimationHelper,
    renderers: dict[str, PoseOverlayRenderer],
    center_xy: np.ndarray,
    z: float,
    radius: float,
    n_samples: int,
    colors: list[str],
    object_type: str,
    out_dir: str,
    sweep_label: str,
) -> dict:
    """Execute one circular sweep + PE at n_samples positions.

    Returns obs (after sweep) and saves individual PNGs + 2×n composite.
    """
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    overlays: dict[str, list] = {c: [] for c in colors}

    os.makedirs(out_dir, exist_ok=True)

    for i, angle in enumerate(angles):
        target_xyz = np.array([
            center_xy[0] + radius * np.cos(angle),
            center_xy[1] + radius * np.sin(angle),
            z,
        ])
        angle_deg = np.rad2deg(angle)
        log.info("[%s] sample %d/%d  angle=%.0f°  target_xyz=%s",
                 sweep_label, i + 1, n_samples, angle_deg, target_xyz)

        obs = go_to_cartesian(env, obs, target_xyz, fine_resolution=0.0005)
        obs, results = do_pe(pe_helper, env, obs, colors, object_type=object_type)
        actual_xyz = obs["observation.state.cartesian"][:3]
        log.info("  EE actual xyz = %s", actual_xyz)

        for color in colors:
            renderer = renderers[color]
            sil_col = _SIL_COLORS.get(color, _DEFAULT_SIL)
            label = f"{sweep_label}_{color} {angle_deg:.0f}°"
            bgr = render_result(renderer, results, color, label, sil_col)
            overlays[color].append(bgr)
            fname = f"{sweep_label}_{color}_angle{int(angle_deg)}.png"
            cv2.imwrite(os.path.join(out_dir, fname), bgr)

    ncols = min(3, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    for color in colors:
        comp = make_composite(overlays[color], nrows=nrows, ncols=ncols)
        fname = f"{sweep_label}_{color}_composite.png"
        cv2.imwrite(os.path.join(out_dir, fname), comp)
        log.info("Saved composite: %s", os.path.join(out_dir, fname))

    return obs


# ── env factory ──────────────────────────────────────────────────────────────


def build_env():
    """Minimal env stack for Cartesian control + camera observations."""
    env = _make_gym_env("my_env_v4")
    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    env = SensorTareWrapper(
        env,
        sensor_key="observation.state.sensors_bota_ft_sensor",
        sensor_data_shape=(6,),
    )
    return env


# ── mesh path helpers ─────────────────────────────────────────────────────────


def _find_mesh(brick_size: str, color: str) -> str | None:
    candidates = [
        f"/workspaces/isaac_ros-dev/lego_assets/lego_{brick_size}_{color}_up.obj",
        os.path.expanduser(
            f"~/workspaces/isaac_ros-dev/lego_assets/lego_{brick_size}_{color}_up.obj"
        ),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _find_abus_mesh() -> str | None:
    candidates = [
        "/workspaces/isaac_ros-dev/abus/abus_key_centered.obj",
        os.path.expanduser("~/workspaces/isaac_ros-dev/abus/abus_key_centered.obj"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="PE circle-sweep test (no grasp).")
    ap.add_argument("--out_dir", default="pe_viz/lego_2x4_pe_test")
    ap.add_argument("--grasp_color", default="lavender")
    ap.add_argument("--target_color", default="yellow")
    ap.add_argument("--n_samples", type=int, default=6,
                    help="PE samples per circle (default 6, composited as 2×3)")
    ap.add_argument("--coarse_radius", type=float, default=0.025,
                    help="Circle radius at coarse (overview) height [m]")
    ap.add_argument("--refined_radius", type=float, default=0.015,
                    help="Circle radius at refined (close-up) height [m]")
    ap.add_argument("--coarse_z", type=float, default=None,
                    help="EE z for coarse circle [m]. Default: from LegoConfig2x4.")
    ap.add_argument("--refined_z", type=float, default=None,
                    help="EE z for refined circle [m]. Default: grasp_z + 0.030.")
    ap.add_argument("--brick_size", default="2x4", choices=["2x2", "2x4"])
    ap.add_argument("--object_type", default="lego", choices=["lego", "abus"],
                    help="Object type: 'lego' (dual-brick, default) or 'abus' (single key).")
    args = ap.parse_args()

    cfg = LegoConfig2x4()
    wide_pe_euler = np.array(cfg.demo_goal_pose_estimation_euler)
    grasp_gt = np.array(cfg.grasp_position_ground_truth)
    goal_gt = np.array(cfg.goal_position_ground_truth)

    coarse_center_xy = wide_pe_euler[:2]
    coarse_z = args.coarse_z if args.coarse_z is not None else float(wide_pe_euler[2])

    refined_center_xy = (grasp_gt[:2] + goal_gt[:2]) / 2.0
    refined_z = args.refined_z if args.refined_z is not None else float(grasp_gt[2] + 0.030)

    log.info("object_type=%s", args.object_type)
    log.info("Coarse sweep: center_xy=%s  z=%.4f  r=%.3f",
             coarse_center_xy, coarse_z, args.coarse_radius)
    log.info("Refined sweep: center_xy=%s  z=%.4f  r=%.3f",
             refined_center_xy, refined_z, args.refined_radius)

    rclpy.init()
    env = build_env()
    install_stop_handler()

    pe_helper = PoseEstimationHelper(
        assumed_orientation=np.array([]),
        lock_orientation=False,
        brick_size=args.brick_size,
        use_tracker=False,
    )

    cam_info_path = "camera_parameters/realsense_d405_single.json"
    if args.object_type == "abus":
        colors = ["abus"]
        abus_mesh = _find_abus_mesh()
        if abus_mesh is None:
            raise FileNotFoundError("ABUS mesh not found at /workspaces/isaac_ros-dev/abus/abus_key_centered.obj")
        renderers = {
            "abus": PoseOverlayRenderer(
                camera_info_json_path=cam_info_path,
                mesh_path=abus_mesh,
                use_default_mesh_fallback=False,
            )
        }
    else:
        colors = [args.grasp_color, args.target_color]
        renderers = {
            args.grasp_color: PoseOverlayRenderer(
                camera_info_json_path=cam_info_path,
                mesh_path=_find_mesh(args.brick_size, args.grasp_color),
                use_default_mesh_fallback=False,
            ),
            args.target_color: PoseOverlayRenderer(
                camera_info_json_path=cam_info_path,
                mesh_path=_find_mesh(args.brick_size, args.target_color),
                use_default_mesh_fallback=False,
            ),
        }

    try:
        log.info("Resetting env...")
        obs, _ = env.reset()
        obs = step_zeros(env)

        # Lift to coarse z before starting circles
        current_xyz = obs["observation.state.cartesian"][:3].copy()
        lift_target = np.array([current_xyz[0], current_xyz[1], coarse_z])
        log.info("Lifting to coarse z = %.4f ...", coarse_z)
        obs = go_to_cartesian(env, obs, lift_target, fine_resolution=0.0005)
        time.sleep(0.5)

        # ── coarse circle sweep ──────────────────────────────────────────── #
        obs = circle_sweep(
            env=env, obs=obs,
            pe_helper=pe_helper,
            renderers=renderers,
            center_xy=coarse_center_xy,
            z=coarse_z,
            radius=args.coarse_radius,
            n_samples=args.n_samples,
            colors=colors,
            object_type=args.object_type,
            out_dir=args.out_dir,
            sweep_label="coarse",
        )

        # ── move to refined center, then refined circle sweep ────────────── #
        log.info("Moving to refined center z=%.4f ...", refined_z)
        refined_center_xyz = np.array([refined_center_xy[0], refined_center_xy[1], refined_z])
        obs = go_to_cartesian(env, obs, refined_center_xyz, fine_resolution=0.0005)
        time.sleep(0.5)

        obs = circle_sweep(
            env=env, obs=obs,
            pe_helper=pe_helper,
            renderers=renderers,
            center_xy=refined_center_xy,
            z=refined_z,
            radius=args.refined_radius,
            n_samples=args.n_samples,
            colors=colors,
            object_type=args.object_type,
            out_dir=args.out_dir,
            sweep_label="refined",
        )

        log.info("Done. Results saved to: %s", args.out_dir)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
