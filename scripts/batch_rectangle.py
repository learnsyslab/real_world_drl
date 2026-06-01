#!/usr/bin/env python3
"""Run the full rectangle pose pipeline over a batch of image pairs.

This script combines the behavior of:
  - mask_rectangle.py
  - estimate_rectangle_pose_pnp.py
  - transform_cam_to_world.py

It expects image pairs inside:
    test_images/{exp_name}/demo_img_color_{idx}_{exp_name}.tiff
    test_images/{exp_name}/demo_img_depth_{idx}_{exp_name}.tiff

It reads the corresponding TCP poses from:
    test_images/{exp_name}/{exp_name}.jsonl

For each index, it:
  1. segments the object with SAM3,
  2. estimates a rectangle pose in the camera frame,
  3. transforms that pose into the world frame,
  4. stores the estimated world pose,
  5. collects mask and masked-depth debug visualizations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_skip_indices(skip_str: str) -> set:
    """Parse a skip-indices string like '1,3,5-7' into a set of ints."""
    if skip_str is None:
        return set()
    skip_set = set()
    parts = [p.strip() for p in skip_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except ValueError:
                continue
            if b_i >= a_i:
                skip_set.update(range(a_i, b_i + 1))
        else:
            try:
                skip_set.add(int(part))
            except ValueError:
                continue
    return skip_set


def r_x(phi: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi), np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_y(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), 0.0, np.sin(phi)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi), 0.0, np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_z(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def euler_xyz_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


def get_t_R_c_and_t_T_t_c() -> Tuple[np.ndarray, np.ndarray]:
    """Return the camera extrinsics mounted on the TCP."""
    c_R_t = np.eye(3, dtype=np.float32)
    # t_T_t_c = np.array([0.0824748 - 0.0013, 0.0, -0.1034 + 0.0095955], dtype=np.float32)
    t_T_t_c = np.array(
        ##[0.0824748 - 0.0013, 0.0, 0.1034 - 0.0095955 - 0.035 + 0.04]
        [0.0824748 - 0.0013, 0.0 - 0.0047, -0.1034 + 0.0095955],
        dtype=np.float32,
    )  ### added offset in Y direction

    c_R_t = r_y(np.deg2rad(25.0)) @ c_R_t
    c_R_t = r_z(np.deg2rad(-90.0)) @ c_R_t

    c_T_c_pinhole = np.array([-0.009, 0.0, -0.0037], dtype=np.float32)
    t_T_t_c = t_T_t_c + c_R_t.T @ c_T_c_pinhole

    t_R_c = c_R_t.T
    return t_R_c, t_T_t_c


def to_world_pose(
    c_D_c_o: np.ndarray,
    w_R_t: np.ndarray,
    w_T_w_t: np.ndarray,
    t_R_c: np.ndarray,
    t_T_t_c: np.ndarray,
) -> np.ndarray:
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]

    w_R_o = w_R_t @ t_R_c @ c_R_o
    w_T_w_o = w_T_w_t + w_R_t @ t_T_t_c + w_R_t @ t_R_c @ c_T_c_o

    w_D_w_o = np.eye(4, dtype=np.float32)
    w_D_w_o[:3, :3] = w_R_o
    w_D_w_o[:3, 3] = w_T_w_o
    return w_D_w_o


def parse_poses_file(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Parse a pose file where each line contains x y z roll pitch yaw."""
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Poses file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            line_clean = (
                line.replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .replace("\u00a0", " ")
            )
            nums: List[float] = []
            for token in line_clean.split():
                try:
                    nums.append(float(token))
                except Exception:
                    continue
            if len(nums) < 6:
                continue

            w_T_w_t = np.array(nums[0:3], dtype=np.float32)
            w_rpy = np.array(nums[3:6], dtype=np.float32)
            poses[idx] = (w_T_w_t, w_rpy)
            idx += 1

    return poses


def load_depth_image(depth_path: Path) -> np.ndarray:
    if depth_path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(depth_path))
    return np.asarray(Image.open(depth_path))


def depth_to_meters(depth_map: np.ndarray) -> np.ndarray:
    depth_m = depth_map.astype(np.float32)
    if np.nanmax(depth_m) > 10.0:
        depth_m = depth_m / 1000.0
    return depth_m


def load_camera_matrix(camera_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with camera_json_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    if "k" not in params or len(params["k"]) != 9:
        raise ValueError(
            f"Camera JSON at {camera_json_path} must contain a flattened 3x3 'k' matrix"
        )

    camera_matrix = np.asarray(params["k"], dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.asarray(params.get("d", []), dtype=np.float32).reshape(-1, 1)
    if dist_coeffs.size == 0:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_rectangle_corners(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contour found in mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx_corners) != 4:
        raise ValueError(
            f"Could not approximate exactly 4 corners. Found {len(approx_corners)}."
        )

    img_pts = approx_corners.reshape(4, 2).astype(np.float32)
    return order_points(img_pts)


def build_object_points(width_m: float, height_m: float) -> np.ndarray:
    return np.array(
        [
            [-width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, height_m / 2.0, 0.0],
            [-width_m / 2.0, height_m / 2.0, 0.0],
        ],
        dtype=np.float32,
    )


def median_mask_depth(depth_map: np.ndarray, mask: np.ndarray) -> float:
    valid_depths = depth_map[mask > 0]
    valid_depths = valid_depths[valid_depths > 0]
    if valid_depths.size == 0:
        raise ValueError("No valid depth values found inside the mask.")
    return float(np.median(valid_depths).astype(np.float32))


def masked_points_3d(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if depth_map_m.shape != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match depth shape {depth_map_m.shape}"
        )

    valid = mask & (depth_map_m > 0)
    pixel_y, pixel_x = np.nonzero(valid)
    if pixel_x.size == 0:
        raise ValueError("No valid masked pixels with positive depth were found")

    depth_values = depth_map_m[pixel_y, pixel_x]
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (pixel_x.astype(np.float32) - cx) * depth_values / fx
    y = (pixel_y.astype(np.float32) - cy) * depth_values / fy
    z = depth_values
    return np.stack([x, y, z], axis=1)


def centroid_from_mask(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    points_3d = masked_points_3d(depth_map_m, mask, camera_matrix)
    return np.mean(points_3d, axis=0)


def centroid_from_mean_mask_pixel_and_depth(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    valid = mask & (depth_map_m > 0)
    pixel_y, pixel_x = np.nonzero(valid)
    if pixel_x.size == 0:
        raise ValueError("No valid masked pixels with positive depth were found")

    mean_x = float(np.mean(pixel_x.astype(np.float32)))
    mean_y = float(np.mean(pixel_y.astype(np.float32)))
    mean_depth = float(np.mean(depth_map_m[pixel_y, pixel_x].astype(np.float32)))

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (mean_x - cx) * mean_depth / fx
    y = (mean_y - cy) * mean_depth / fy
    z = mean_depth
    return np.array([x, y, z], dtype=np.float32)


def compute_centroid(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "point_cloud_mean":
        return centroid_from_mask(depth_map_m, mask, camera_matrix)
    if method == "mean_pixel_depth":
        return centroid_from_mean_mask_pixel_and_depth(depth_map_m, mask, camera_matrix)
    raise ValueError(f"Unknown centroid method: {method}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def project_points_3d(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def build_coordinate_frame(
    best: dict, full_object_centroid_camera_m: np.ndarray
) -> dict:
    origin = best["tvec"].reshape(3).astype(np.float32)
    rotation = best["R"].astype(np.float32)
    short_axis_index = 0 if best["width_m"] <= best["height_m"] else 1

    z_axis = normalize_vector(rotation[:, 2])
    if z_axis[2] > 0:
        z_axis = -z_axis

    x_axis = rotation[:, short_axis_index]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize_vector(x_axis)

    to_full_centroid = full_object_centroid_camera_m - origin
    to_full_centroid = to_full_centroid - np.dot(to_full_centroid, z_axis) * z_axis
    if np.dot(to_full_centroid, x_axis) < 0:
        x_axis = -x_axis

    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))

    return {
        "origin_camera_m": origin,
        "x_axis_camera": x_axis,
        "y_axis_camera": y_axis,
        "z_axis_camera": z_axis,
        "rotation_camera": np.column_stack([x_axis, y_axis, z_axis]),
    }


def solve_rectangle_pose(
    img_pts: np.ndarray,
    width_m: float,
    height_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    observed_z_m: float,
) -> dict:
    obj_pts = build_object_points(width_m, height_m)

    success, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success or len(rvecs) == 0:
        raise ValueError("solvePnPGeneric failed to find a pose solution.")

    best = None
    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(projected - img_pts, axis=1)))
        predicted_z = float(tvec[2, 0])
        z_error = abs(predicted_z - observed_z_m)
        score = reproj_error + z_error

        candidate = {
            "solution_index": idx,
            "rvec": rvec,
            "tvec": tvec,
            "R": cv2.Rodrigues(rvec)[0],
            "reprojection_error_px": reproj_error,
            "predicted_z_m": predicted_z,
            "observed_z_m": observed_z_m,
            "z_error_m": z_error,
            "score": score,
            "solver_reproj_error": float(reproj_errors[idx][0])
            if reproj_errors is not None and len(reproj_errors) > idx
            else None,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No valid pose candidate found.")

    return best


def save_estimated_poses(path: str, poses_world: Dict[int, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for idx, w_pose in sorted(poses_world.items()):
        rows.append(np.concatenate(([float(idx)], w_pose.reshape(-1))))
    data = np.asarray(rows, dtype=np.float32)
    header = "idx " + " ".join(f"m{i}" for i in range(16))
    np.savetxt(path, data, fmt="%.8f", header=header)


def save_mask_figure(
    path: str,
    items: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> None:
    if not items:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    ordered_items = sorted(items.items())
    num_items = len(ordered_items)
    num_cols = min(3, num_items)
    num_rows = int(np.ceil(num_items / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 5 * num_rows),
        squeeze=False,
    )

    light_red = np.array([255, 230, 230], dtype=np.uint8)
    flat_axes = axes.ravel()
    for ax in flat_axes[num_items:]:
        ax.axis("off")

    for ax, (idx, (image, mask)) in zip(flat_axes, ordered_items):
        mask_bool = mask > 0
        masked_background = np.full_like(image, light_red)
        masked_image = np.where(mask_bool[..., None], image, masked_background)

        ax.imshow(masked_image)
        ax.set_title(f"{idx}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_depth_figure(
    path: str,
    items: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> None:
    if not items:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    all_valid_depths = []
    for depth, mask in items.values():
        mask_bool = mask > 0
        depth_masked = depth.copy()
        depth_masked[~mask_bool | (depth_masked == 0)] = np.nan
        valid = depth_masked[~np.isnan(depth_masked)]
        if len(valid) > 0:
            all_valid_depths.extend(valid)

    if not all_valid_depths:
        print(f"Warning: No valid depth values found in {path}")
        return

    vmin = float(np.min(all_valid_depths))
    vmax = float(min(np.max(all_valid_depths), 0.20))

    ordered_items = sorted(items.items())
    num_items = len(ordered_items)
    num_cols = min(3, num_items)
    num_rows = int(np.ceil(num_items / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 5 * num_rows),
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for ax in flat_axes[num_items:]:
        ax.axis("off")

    for ax, (idx, (depth, mask)) in zip(flat_axes, ordered_items):
        mask_bool = mask > 0
        depth_masked = depth.copy()
        depth_masked[~mask_bool | (depth_masked == 0)] = np.nan

        valid_mask = ~np.isnan(depth_masked)
        if not np.any(valid_mask):
            ax.set_title(f"{idx}")
            ax.axis("off")
            continue

        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        depth_cropped = depth_masked[rmin : rmax + 1, cmin : cmax + 1]

        im = ax.imshow(depth_cropped, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{idx}")
        ax.axis("off")
        if idx == ordered_items[0][0]:
            plt.colorbar(im, ax=ax, label="depth (m)")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_mask(
    output: dict, use_top_three_confidence_filter: bool = True
) -> np.ndarray:
    masks = output.get("masks", [])
    scores = output.get("scores", None)
    if len(masks) == 0:
        raise ValueError("No masks found")

    if scores is not None:
        scores_np = scores.cpu().numpy()
        ranked_indices = np.argsort(scores_np)[::-1]
        if use_top_three_confidence_filter:
            ranked_indices = ranked_indices[:3]
            best_score = scores_np[ranked_indices[0]]
            candidate_indices = [
                idx for idx in ranked_indices if best_score - scores_np[idx] < 0.3
            ]
        else:
            candidate_indices = [int(ranked_indices[0])]
    else:
        candidate_indices = list(range(min(3, len(masks))))
        if not use_top_three_confidence_filter:
            candidate_indices = candidate_indices[:1]

    best_mask_idx = max(
        candidate_indices,
        key=lambda idx: (
            np.asarray(masks[idx].cpu().numpy())
            .reshape(masks[0].shape[-2], masks[0].shape[-1])
            .sum()
        ),
    )

    return (
        masks[best_mask_idx]
        .cpu()
        .numpy()
        .reshape(masks[0].shape[-2], masks[0].shape[-1])
        * 255
    )


def run_sam3_mask(
    processor: Sam3Processor,
    image: np.ndarray,
    prompt: str,
    use_top_three_confidence_filter: bool,
) -> np.ndarray:
    inference_state = processor.set_image(Image.fromarray(image))
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return select_mask(
        output, use_top_three_confidence_filter=use_top_three_confidence_filter
    )


def find_existing_indices(
    exp_dir: Path,
    exp_name: str,
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    skip_set: set | None = None,
) -> Iterable[int]:
    if skip_set is None:
        skip_set = set()
    for idx in sorted(poses.keys()):
        if idx in skip_set:
            print(f"Skipping {idx}: requested by --skip-indices")
            continue
        color_path = exp_dir / f"demo_img_color_{idx}_{exp_name}.tiff"
        depth_path = exp_dir / f"demo_img_depth_{idx}_{exp_name}.tiff"
        if color_path.exists() and depth_path.exists():
            yield idx
        else:
            print(f"Skipping {idx}: images not found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM3 masking, rectangle pose estimation, and world-frame transform over a batch"
    )
    parser.add_argument("--exp-name", default="sugar", help="Experiment name suffix")
    parser.add_argument(
        "--sam3-prompt",
        default="white rectangle with barcode",
        help="Text prompt used for the primary mask",
    )
    parser.add_argument(
        "--full-mask-prompt",
        default="yellow white cardboard box with label",
        help="Optional broader prompt for the orientation mask; defaults to --sam3-prompt",
    )
    parser.add_argument(
        "--poses-file",
        default=None,
        help="TCP pose file in JSONL form; defaults to test_images/{exp_name}/{exp_name}.jsonl",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Path to the camera intrinsics JSON; defaults to test_images/{exp_name}/realsense_d405_single.json",
    )
    parser.add_argument(
        "--width-m",
        type=float,
        default=0.03,
        help="Rectangle width in meters",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=0.04,
        help="Rectangle height in meters",
    )
    parser.add_argument(
        "--bpe-path",
        default=None,
        help="Path to the SAM3 BPE vocabulary file",
    )
    parser.add_argument(
        "--output-estimates",
        default=None,
        help="Output path for estimated world poses; defaults to test_images/{exp_name}_estimates.txt",
    )
    parser.add_argument(
        "--mask-figure",
        default=None,
        help="Output path for the mask debug figure",
    )
    parser.add_argument(
        "--depth-figure",
        default=None,
        help="Output path for the masked depth debug figure",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.01,
        help="SAM3 confidence threshold",
    )
    parser.add_argument(
        "--skip-indices",
        default=None,
        help="Comma-separated indices or ranges to skip, e.g. '1,3,5-7'",
    )
    parser.add_argument(
        "--centroid-method",
        choices=["point_cloud_mean", "mean_pixel_depth"],
        default="point_cloud_mean",
        help=(
            "Method for centroid selection used to orient the PnP pose; "
            "'mean_pixel_depth' uses mean mask pixel coordinates and mean depth"
        ),
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    exp_dir = Path("test_images") / exp_name
    poses_file = args.poses_file or str(exp_dir / f"{exp_name}.jsonl")
    estimated_poses_file = args.output_estimates or str(
        exp_dir / f"{exp_name}_estimates.txt"
    )
    mask_figure_file = args.mask_figure or str(exp_dir / f"masks_{exp_name}.png")
    depth_figure_file = args.depth_figure or str(
        exp_dir / f"masked_depth_{exp_name}.png"
    )
    full_mask_prompt = args.full_mask_prompt or args.sam3_prompt
    bpe_path = (
        Path(args.bpe_path)
        if args.bpe_path is not None
        else Path(__file__).resolve().parents[1]
        / "sam3"
        / "sam3"
        / "assets"
        / "bpe_simple_vocab_16e6.txt.gz"
    )

    camera_path = (
        Path(args.camera)
        if args.camera is not None
        else Path("camera_parameters") / "realsense_d405_single.json"
    )
    if not camera_path.exists():
        raise FileNotFoundError(f"Camera JSON not found: {camera_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"SAM3 BPE file not found: {bpe_path}")

    poses = parse_poses_file(poses_file)
    if not poses:
        print("No poses parsed from", poses_file)
        return

    camera_matrix, dist_coeffs = load_camera_matrix(camera_path)
    t_R_c, t_T_t_c = get_t_R_c_and_t_T_t_c()

    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    print("Building SAM3 processor...")
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    estimated_world: Dict[int, np.ndarray] = {}
    mask_visualizations: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    depth_visualizations: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    skip_set = parse_skip_indices(args.skip_indices)
    for idx in find_existing_indices(exp_dir, exp_name, poses, skip_set=skip_set):
        w_T_w_t, w_rpy = poses[idx]
        color_path = exp_dir / f"demo_img_color_{idx}_{exp_name}.tiff"
        depth_path = exp_dir / f"demo_img_depth_{idx}_{exp_name}.tiff"

        print(f"\n[{idx}] Loading {color_path} ...")
        image = np.array(Image.open(color_path).convert("RGB"))
        depth_raw = load_depth_image(depth_path)
        depth_m = depth_to_meters(depth_raw)

        try:
            mask = run_sam3_mask(
                processor,
                image,
                args.sam3_prompt,
                use_top_three_confidence_filter=True,
            )
            full_mask = run_sam3_mask(
                processor,
                image,
                full_mask_prompt,
                use_top_three_confidence_filter=False,
            )
        except Exception as exc:
            print(f"  Masking failed for {idx}: {exc}")
            continue

        mask_visualizations[idx] = (image, mask)
        depth_visualizations[idx] = (depth_m, mask)

        try:
            img_pts = extract_rectangle_corners(mask > 0)
            observed_z_m = median_mask_depth(depth_m, mask > 0)
            full_object_centroid_camera_m = compute_centroid(
                depth_m,
                full_mask > 0,
                camera_matrix,
                args.centroid_method,
            )

            candidates = []
            for width_m, height_m, label in (
                (args.width_m, args.height_m, "width x height"),
                (args.height_m, args.width_m, "height x width"),
            ):
                candidate = solve_rectangle_pose(
                    img_pts=img_pts,
                    width_m=width_m,
                    height_m=height_m,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    observed_z_m=observed_z_m,
                )
                candidate["label"] = label
                candidate["width_m"] = width_m
                candidate["height_m"] = height_m
                candidates.append(candidate)

            best = min(candidates, key=lambda item: item["score"])
            coordinate_frame = build_coordinate_frame(
                best, full_object_centroid_camera_m
            )
            origin_camera_m = coordinate_frame["origin_camera_m"]
            x_axis_camera = coordinate_frame["x_axis_camera"]
            y_axis_camera = coordinate_frame["y_axis_camera"]
            z_axis_camera = coordinate_frame["z_axis_camera"]
            coordinate_rotation_camera = coordinate_frame["rotation_camera"]

            pose_cam = np.eye(4, dtype=np.float32)
            pose_cam[:3, :3] = coordinate_rotation_camera
            pose_cam[:3, 3] = origin_camera_m

            w_R_t = euler_xyz_to_rot_matrix(*w_rpy)
            w_D_w_o = to_world_pose(pose_cam, w_R_t, w_T_w_t, t_R_c, t_T_t_c)
            estimated_world[idx] = w_D_w_o

            x_axis_length_m = 0.5 * min(best["width_m"], best["height_m"])
            y_axis_length_m = 0.5 * max(best["width_m"], best["height_m"])
            z_axis_length_m = 0.5 * min(best["width_m"], best["height_m"])
            axes_projected = project_points_3d(
                np.array(
                    [
                        origin_camera_m,
                        origin_camera_m + x_axis_camera * x_axis_length_m,
                        origin_camera_m + y_axis_camera * y_axis_length_m,
                        origin_camera_m + z_axis_camera * z_axis_length_m,
                    ],
                    dtype=np.float32,
                ),
                camera_matrix,
                dist_coeffs,
            )
            origin_px, x_axis_px, y_axis_px, z_axis_px = axes_projected

            cv2.polylines(
                image,
                [img_pts.astype(np.int32)],
                isClosed=True,
                color=(255, 255, 0),
                thickness=2,
            )
            cv2.circle(image, tuple(origin_px.astype(int)), 4, (255, 255, 255), -1)
            cv2.line(
                image,
                tuple(origin_px.astype(int)),
                tuple(x_axis_px.astype(int)),
                (255, 80, 80),
                2,
            )
            cv2.line(
                image,
                tuple(origin_px.astype(int)),
                tuple(y_axis_px.astype(int)),
                (80, 255, 80),
                2,
            )
            cv2.line(
                image,
                tuple(origin_px.astype(int)),
                tuple(z_axis_px.astype(int)),
                (80, 160, 255),
                2,
            )

            print(f"  Estimated world translation: {w_D_w_o[:3, 3]}")
            print(f"  Selected score: {best['score']:.6f}")
            print(f"  Selected orientation: {best['label']}")
        except Exception as exc:
            print(f"  Pose estimation failed for {idx}: {exc}")
            continue

    if not estimated_world:
        print("No world poses estimated")
        return

    save_estimated_poses(estimated_poses_file, estimated_world)
    save_mask_figure(mask_figure_file, mask_visualizations)
    save_depth_figure(depth_figure_file, depth_visualizations)

    print(f"Saved {len(estimated_world)} estimated poses to {estimated_poses_file}")
    print(f"Saved mask figure to {mask_figure_file}")
    print(f"Saved depth figure to {depth_figure_file}")


if __name__ == "__main__":
    main()
