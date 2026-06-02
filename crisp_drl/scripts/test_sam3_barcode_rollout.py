#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crisp_drl.agents.shared.env_wrappers import (
    ActionTimeStampWrapper,
    LastObservationWrapper,
)
from crisp_drl.envs.make_env import make_env
from crisp_drl.envs.sam3_pe import PoseEstimationHelper


def action_dim_from_env(env) -> int:
    action_shape = getattr(env.action_space, "shape", None)
    if not action_shape:
        return 3
    return int(action_shape[0])


def project_pose_frame(
    pose_camera: np.ndarray,
    camera_matrix: np.ndarray,
    axis_length_m: float,
) -> np.ndarray:
    origin = pose_camera[:3, 3]
    axes = pose_camera[:3, :3]
    points_3d = np.array(
        [
            origin,
            origin + axes[:, 0] * axis_length_m,
            origin + axes[:, 1] * axis_length_m,
            origin + axes[:, 2] * axis_length_m,
        ],
        dtype=np.float32,
    )
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    projected = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    valid = z > 1e-6
    projected[valid, 0] = fx * (x[valid] / z[valid]) + cx
    projected[valid, 1] = fy * (y[valid] / z[valid]) + cy
    return projected


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def wrap_angle_to_half_turn(angle_rad: float) -> float:
    return float((angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def object_frame_z_rotation_mod_180_rad(pose_world: np.ndarray) -> float:
    yaw_rad = float(np.arctan2(pose_world[1, 0], pose_world[0, 0]))
    return wrap_angle_to_half_turn(yaw_rad)


def save_debug_plot(
    image: np.ndarray,
    depth_image: np.ndarray,
    pose_helper: PoseEstimationHelper,
    output_path: Path,
    stage: str,
) -> None:
    depth_image = np.squeeze(depth_image)
    masks = pose_helper.last_masks or []
    camera_poses = pose_helper.last_camera_poses or []
    scores = pose_helper.last_mask_scores or []

    if len(masks) != len(camera_poses):
        raise RuntimeError("Mask and pose counts do not match for debug plotting")

    num_masks = len(masks)
    num_plots = 2 * num_masks + 3
    num_cols = 2 if num_plots > 1 else 1
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6 * num_cols, 5 * num_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    axis_length_m = 0.5 * min(pose_helper.width_m, pose_helper.height_m)
    mask_union = np.zeros_like(depth_image, dtype=bool)
    for mask in masks:
        mask_union |= mask > 0
    full_mask = getattr(pose_helper, "last_full_mask", None)
    if full_mask is not None:
        mask_union |= full_mask > 0

    valid_depth = depth_image[mask_union & np.isfinite(depth_image) & (depth_image > 0)]
    if valid_depth.size > 0:
        depth_vmin = float(np.min(valid_depth))
        depth_vmax = float(np.max(valid_depth))
    else:
        depth_vmin = 0.0
        depth_vmax = 1.0

    def draw_pose(ax, pose_camera: np.ndarray, color_suffix: str) -> None:
        projected = project_pose_frame(
            pose_camera,
            pose_helper.camera_matrix,
            axis_length_m,
        )
        origin_px = projected[0]
        x_px = projected[1]
        y_px = projected[2]
        z_px = projected[3]
        ax.plot(
            [origin_px[0], x_px[0]],
            [origin_px[1], x_px[1]],
            color="red",
            linewidth=2,
        )
        ax.plot(
            [origin_px[0], y_px[0]],
            [origin_px[1], y_px[1]],
            color="green",
            linewidth=2,
        )
        ax.plot(
            [origin_px[0], z_px[0]],
            [origin_px[1], z_px[1]],
            color="blue",
            linewidth=2,
        )
        ax.scatter([origin_px[0]], [origin_px[1]], c="white", s=20, edgecolors="black")

    def draw_masked_depth(ax, mask: np.ndarray) -> None:
        masked_depth = np.ma.array(depth_image, mask=~mask)
        ax.imshow(masked_depth, cmap="viridis", vmin=depth_vmin, vmax=depth_vmax)

    def mask_depth_range(mask: np.ndarray) -> str:
        valid_mask_depth = depth_image[
            mask & np.isfinite(depth_image) & (depth_image > 0)
        ]
        if valid_mask_depth.size == 0:
            return "depth=n/a"
        return (
            f"depth=[{float(np.min(valid_mask_depth)):.3f}, "
            f"{float(np.max(valid_mask_depth)):.3f}] m"
        )

    for i, ax in enumerate(flat_axes):
        if i < num_masks:
            mask = masks[i] > 0
            masked_background = np.full_like(image, 32)
            masked_image = np.where(mask[..., None], image, masked_background)
            ax.imshow(masked_image)
            draw_pose(ax, camera_poses[i], f"mask_{i}")
            depth_range_text = mask_depth_range(mask)
            if i < len(scores):
                ax.set_title(f"mask {i + 1} score={scores[i]:.3f} {depth_range_text}")
            else:
                ax.set_title(f"mask {i + 1} {depth_range_text}")
            ax.axis("off")
            continue

        if i < 2 * num_masks:
            mask_idx = i - num_masks
            mask = masks[mask_idx] > 0
            draw_masked_depth(ax, mask)
            draw_pose(ax, camera_poses[mask_idx], f"mask_depth_{mask_idx}")
            depth_range_text = mask_depth_range(mask)
            if mask_idx < len(scores):
                ax.set_title(
                    f"mask {mask_idx + 1} depth score={scores[mask_idx]:.3f} {depth_range_text}"
                )
            else:
                ax.set_title(f"mask {mask_idx + 1} depth {depth_range_text}")
            ax.axis("off")
            continue

        # plot the full mask (from the full-mask prompt)
        if i == 2 * num_masks:
            full_mask = getattr(pose_helper, "last_full_mask", None)
            if full_mask is not None:
                mask = full_mask > 0
                masked_background = np.full_like(image, 32)
                masked_image = np.where(mask[..., None], image, masked_background)
                ax.imshow(masked_image)
                depth_range_text = mask_depth_range(mask)
                for pose_idx, pose_camera in enumerate(camera_poses):
                    draw_pose(ax, pose_camera, "full_mask")
                    projected = project_pose_frame(
                        pose_camera,
                        pose_helper.camera_matrix,
                        axis_length_m,
                    )
                    origin_px = projected[0]
                    ax.text(
                        float(origin_px[0]),
                        float(origin_px[1]),
                        f"{pose_idx + 1}",
                        color="white",
                        fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.5, pad=1),
                    )
                ax.set_title(f"full mask {depth_range_text}")
            else:
                ax.imshow(image)
                ax.set_title("full mask (none)")
            ax.axis("off")
            continue

        if i == 2 * num_masks + 1:
            full_mask = getattr(pose_helper, "last_full_mask", None)
            if full_mask is not None:
                mask = full_mask > 0
                draw_masked_depth(ax, mask)
                depth_range_text = mask_depth_range(mask)
                for pose_idx, pose_camera in enumerate(camera_poses):
                    draw_pose(ax, pose_camera, "full_mask_depth")
                    projected = project_pose_frame(
                        pose_camera,
                        pose_helper.camera_matrix,
                        axis_length_m,
                    )
                    origin_px = projected[0]
                    ax.text(
                        float(origin_px[0]),
                        float(origin_px[1]),
                        f"{pose_idx + 1}",
                        color="white",
                        fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.5, pad=1),
                    )
                ax.set_title(f"full mask depth {depth_range_text}")
            else:
                ax.imshow(depth_image, cmap="viridis", vmin=depth_vmin, vmax=depth_vmax)
                ax.set_title("full mask depth (none)")
            ax.axis("off")
            continue

        # full image with all poses
        if i == 2 * num_masks + 2:
            ax.imshow(image)
            for pose_idx, pose_camera in enumerate(camera_poses):
                projected = project_pose_frame(
                    pose_camera,
                    pose_helper.camera_matrix,
                    axis_length_m,
                )
                origin_px = projected[0]
                x_px = projected[1]
                y_px = projected[2]
                z_px = projected[3]
                ax.plot(
                    [origin_px[0], x_px[0]],
                    [origin_px[1], x_px[1]],
                    color="red",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.plot(
                    [origin_px[0], y_px[0]],
                    [origin_px[1], y_px[1]],
                    color="green",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.plot(
                    [origin_px[0], z_px[0]],
                    [origin_px[1], z_px[1]],
                    color="blue",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.text(
                    float(origin_px[0]),
                    float(origin_px[1]),
                    f"{pose_idx + 1}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1),
                )
            ax.set_title("full image")
            ax.axis("off")
            continue

        ax.axis("off")

    fig.suptitle(f"Barcode debug plot - {stage}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def estimate_barcode_poses(
    pose_helper: PoseEstimationHelper, obs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    poses = pose_helper.estimate_barcodes(
        (
            obs["observation.images.wrist_camera"],
            obs["observation.images.wrist_depth_camera"],
        ),
        obs["observation.state.cartesian"],
    )
    print("Received the following barcode pose estimates (world frame):")
    for i, pose in enumerate(poses):
        print(f"  Barcode {i + 1}: {pose[:3, 3]}")
    return poses


def save_stage_debug_plot(
    pose_helper: PoseEstimationHelper,
    obs: dict[str, np.ndarray],
    stage: str,
    debug_dir: Path,
    cycle_idx: int,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"cycle_{cycle_idx + 1:02d}_{stage}.png"
    save_debug_plot(
        obs["observation.images.wrist_camera"],
        obs["observation.images.wrist_depth_camera"],
        pose_helper,
        output_path,
        stage,
    )
    print(f"  Saved debug plot to {output_path}")


def move_to_target(
    env,
    current_obs: dict[str, np.ndarray],
    target_xyz: np.ndarray,
    target_z_rotation_rad: float | None = None,
    tolerance: float = 0.001,
    settle_steps: int = 3,
    max_settle_iterations: int = 250,
) -> dict[str, np.ndarray]:
    action_dim = action_dim_from_env(env)
    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = target_xyz - current_obs["observation.state.cartesian"][:3]
    current_z_rotation_rad = float(current_obs["observation.state.cartesian"][5])
    if target_z_rotation_rad is not None and action_dim > 5:
        action[5] = wrap_angle_to_half_turn(
            target_z_rotation_rad - current_z_rotation_rad
        )
    obs, *_ = env.step(action)
    last_i_term = False

    for _ in range(max_settle_iterations):
        current_xyz = obs["observation.state.cartesian"][:3]
        current_z_rotation_rad = float(obs["observation.state.cartesian"][5])
        delta = target_xyz - current_xyz
        position_error = float(np.linalg.norm(delta))
        velocity_error = float(
            np.linalg.norm(obs["observation.velocity.cartesian"][:3])
        )

        if position_error <= tolerance and velocity_error <= tolerance / 2:
            break

        if (
            velocity_error <= tolerance / 2
            and position_error > tolerance
            and not last_i_term
        ):
            action = np.zeros(action_dim, dtype=np.float32)
            action[:3] = delta
            obs, *_ = env.step(action)
            last_i_term = True
        else:
            obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))
            last_i_term = False

    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    return obs


def barcode_offset_position(
    pose_world: np.ndarray,
    offset_local_m: np.ndarray,
    local: bool,
) -> np.ndarray:
    if local:
        return pose_world[:3, 3] + pose_world[:3, :3] @ offset_local_m
    return pose_world[:3, 3] + offset_local_m


def barcode_offset_pose(
    pose_world: np.ndarray,
    offset_m: np.ndarray,
    local: bool,
) -> np.ndarray:
    grasp_pose = pose_world.astype(np.float32).copy()
    grasp_pose[:3, 3] = barcode_offset_position(pose_world, offset_m, local)
    return grasp_pose


def write_pose_line(
    handle,
    cycle_idx: int,
    stage: str,
    poses_world: list[np.ndarray],
) -> None:
    record = {
        "cycle": cycle_idx,
        "stage": stage,
        "poses_world": [pose.astype(np.float32).tolist() for pose in poses_world],
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def write_grasp_pose_line(
    handle,
    cycle_idx: int,
    stage: str,
    grasp_pose_world: np.ndarray,
    offset_m: np.ndarray,
    local: bool,
) -> None:
    record = {
        "cycle": cycle_idx,
        "stage": stage,
        "offset_m": np.asarray(offset_m, dtype=np.float32).tolist(),
        "offset_frame": "local" if local else "global",
        "grasp_pose_world": grasp_pose_world.astype(np.float32).tolist(),
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


# 26 x 37.5 mm for blue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat SAM3 barcode pose estimation while moving to offsets around the detected barcode."
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name used to derive rollout output paths",
    )
    parser.add_argument(
        "--full-mask-prompt",
        type=str,
        default="yellow cardboard box",
        help=("Full mask prompt for SAM3"),
    )
    parser.add_argument(
        "--width-m",
        type=float,
        default=0.03,
        help="Barcode width in meters",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=0.04,
        help="Barcode height in meters",
    )
    parser.add_argument(
        "--above-target-offset-m",
        nargs=3,
        type=float,
        default=(0.0, 0.004, 0.112),
        metavar=("X", "Y", "Z"),
        help="Offset from the initial pose to the above-barcode target, in meters",
    )
    parser.add_argument(
        "--final-offset-m",
        nargs=3,
        type=float,
        default=(0.0, 0.004, 0.065),
        metavar=("X", "Y", "Z"),
        help="Offset from the above-barcode pose to the final target, in meters",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Apply barcode offsets in the barcode frame instead of the world frame. "
            "When enabled, the first local offset is also randomized by up to +/-30 deg "
            "around z."
        ),
    )
    parser.add_argument(
        "--filter-depth-outliers",
        action="store_true",
        help="Remove depth values outside +/-25% of the median inside each mask",
    )
    args = parser.parse_args()

    env_name = "my_env_v4"
    output_path = Path("rollout_data") / "tests" / f"{args.exp_name}.txt"
    debug_plot_dir = Path("rollout_data") / "tests" / f"{args.exp_name}_debug"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = 10

    above_target_offset_m = np.asarray(args.above_target_offset_m, dtype=np.float32)
    final_offset_m = np.asarray(args.final_offset_m, dtype=np.float32)
    local_first_offset_rotation_rad = 0.0

    env = make_env(env_name)
    print(f"Env created: {env_name}")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)

    pose_helper = PoseEstimationHelper(
        centroid_method="mean_pixel_depth",
        full_mask_prompt=args.full_mask_prompt,
        width_m=args.width_m,
        height_m=args.height_m,
        filter_depth_outliers=args.filter_depth_outliers,
    )

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for cycle_idx in range(episodes):
                print(
                    f"\n[Cycle {cycle_idx + 1}/{episodes}] Homing and opening gripper..."
                )
                base_env: Any = env.unwrapped
                base_env.home(
                    # home_config=[
                    #     -0.16125268,
                    #     -0.19791204,
                    #     -0.32023647,
                    #     -2.4182775,
                    #     -0.08884472,
                    #     2.2305312,
                    #     0.36234158,
                    # ],
                    home_config=[
                        -1.6601315e-01,
                        7.1060816e-03,
                        -2.8422508e-01,
                        -2.0143490e00,
                        1.0296288e-03,
                        2.0213308e00,
                        3.2737154e-01,
                    ]
                )
                base_env.gripper.open()
                obs, *_ = env.reset()
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))
                if args.local:
                    local_first_offset_rotation_rad = float(
                        np.deg2rad(np.random.uniform(-30.0, 30.0))
                    )
                    print(
                        "Local mode enabled: first above-target offset rotated by "
                        f"{np.degrees(local_first_offset_rotation_rad):+.2f} deg around z"
                    )
                    env.step(
                        np.array(
                            [
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                local_first_offset_rotation_rad,
                                0.0,
                            ],
                            dtype=np.float32,
                        )
                    )

                print("  Estimating initial barcode poses...")
                initial_poses = estimate_barcode_poses(pose_helper, obs)
                if not initial_poses:
                    raise RuntimeError("No barcode poses detected in initial estimate")
                write_pose_line(handle, cycle_idx, "initial", initial_poses)
                save_stage_debug_plot(
                    pose_helper, obs, "a_initial", debug_plot_dir, cycle_idx
                )

                initial_pose = initial_poses[0]

                # above_20cm_target = barcode_offset_pose(
                #     initial_pose, np.array([0.0, 0.0, 0.2], dtype=np.float32)
                # )
                # print(f"  Moving to 20 cm above barcode: {above_20cm_target}")
                # obs = move_to_target(env, obs, above_20cm_target)

                # print("  Estimating pose 20 cm above barcode...")
                # above_20cm_poses = estimate_barcode_poses(pose_helper, obs)
                # if not above_20cm_poses:
                #     raise RuntimeError("No barcode poses detected 20 cm above barcode")
                # write_pose_line(handle, cycle_idx, "above_20cm", above_20cm_poses)
                # save_stage_debug_plot(
                #     pose_helper, obs, "above_20cm", debug_plot_dir, cycle_idx
                # )

                above_target = barcode_offset_position(
                    initial_pose,
                    above_target_offset_m,
                    local=args.local,
                )
                print(f"  Moving to 10 cm above barcode: {above_target}")
                above_target_z_rotation_rad = None
                if args.local:
                    above_target_z_rotation_rad = object_frame_z_rotation_mod_180_rad(
                        initial_pose
                    )
                obs = move_to_target(
                    env,
                    obs,
                    above_target,
                    target_z_rotation_rad=above_target_z_rotation_rad,
                )

                print("  Estimating pose above barcode...")
                above_poses = estimate_barcode_poses(pose_helper, obs)
                if not above_poses:
                    raise RuntimeError("No barcode poses detected above barcode")
                write_pose_line(handle, cycle_idx, "above_1", above_poses)
                save_stage_debug_plot(
                    pose_helper, obs, "b_above_1", debug_plot_dir, cycle_idx
                )

                above_pose = above_poses[0]

                # above_target_2 = barcode_offset_position(above_pose, above_target_offset_m)
                # print(f"  Moving to 10 cm above barcode (2): {above_target_2}")
                # obs = move_to_target(env, obs, above_target_2)

                # print("  Estimating pose above barcode...")
                # above_poses_2 = estimate_barcode_poses(pose_helper, obs)
                # if not above_poses_2:
                #     raise RuntimeError("No barcode poses detected above barcode")
                # write_pose_line(handle, cycle_idx, "above_2", above_poses_2)
                # save_stage_debug_plot(
                #     pose_helper, obs, "b_above_2", debug_plot_dir, cycle_idx
                # )

                # above_pose_2 = above_poses_2[0]

                below_left_target = barcode_offset_position(
                    above_pose,
                    final_offset_m,
                    local=args.local,
                )
                below_left_target_z_rotation_rad = None
                if args.local:
                    below_left_target_z_rotation_rad = (
                        object_frame_z_rotation_mod_180_rad(above_pose)
                    )
                print(
                    f"  Moving to 5 cm below barcode and 3 cm in -x: {below_left_target}"
                )
                obs = move_to_target(
                    env,
                    obs,
                    below_left_target,
                    target_z_rotation_rad=below_left_target_z_rotation_rad,
                )

                print("  Estimating final barcode poses...")
                final_poses = estimate_barcode_poses(pose_helper, obs)
                if not final_poses:
                    raise RuntimeError("No barcode poses detected in final estimate")
                write_pose_line(handle, cycle_idx, "final_1", final_poses)
                save_stage_debug_plot(
                    pose_helper, obs, "c_final_1", debug_plot_dir, cycle_idx
                )
                # final_pose = final_poses[0]

                final_pose_2 = final_poses[0]

                below_left_target_2 = barcode_offset_position(
                    final_pose_2, final_offset_m, local=args.local
                )
                print(
                    f"  Moving to 5 cm below barcode and 3 cm in -x: {below_left_target_2}"
                )
                obs = move_to_target(env, obs, below_left_target_2)

                print("  Estimating final barcode poses...")
                final_poses_2 = estimate_barcode_poses(pose_helper, obs)
                if not final_poses_2:
                    raise RuntimeError(
                        "No barcode poses detected in final estimate (2)"
                    )
                write_pose_line(handle, cycle_idx, "final_2", final_poses_2)
                save_stage_debug_plot(
                    pose_helper, obs, "c_final_2", debug_plot_dir, cycle_idx
                )

                grasp_pose = barcode_offset_pose(
                    final_pose_2,
                    final_offset_m,
                    local=args.local,
                )
                print(f"  Estimated grasp pose (world frame): {grasp_pose[:3, 3]}")
                write_grasp_pose_line(
                    handle,
                    cycle_idx,
                    "grasp_1",
                    grasp_pose,
                    final_offset_m,
                    local=args.local,
                )

                print(f"  Cycle {cycle_idx + 1} complete.")
                time.sleep(0.5)

        print(f"Saved barcode pose estimates to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
