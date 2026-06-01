import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.sam3_detector import Sam3Detector

exp_name = "sugar"
sam3_prompt = "yellow white cardboard box with label"  # "black cover with circular grille"  # "small grey plastic rectangle with gear"
MESH_FILE_NAME = "ycb_sugar.obj"

poses_file = f"rollout_data/demos/{exp_name}.jsonl"
# poses_file = f"test_images/{exp_name}_poses.txt"
estimated_poses_file = f"test_images/{exp_name}_estimates.txt"
mask_figure_file = f"test_images/masks_{exp_name}.png"
depth_figure_file = f"test_images/masked_depth_{exp_name}.png"
# Path to mesh used for ABUS key
MESH_PATH_CONTAINER = (
    f"/workspaces/isaac_ros-dev/foundation_pose_meshes/{MESH_FILE_NAME}"
)


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
    """
    Returns camera extrinsics mounted on the TCP:
      - t_R_c: rotation from camera frame c into TCP frame t
      - t_T_t_c: translation from t to c in frame t
    """
    c_R_t = np.eye(3, dtype=np.float32)
    t_T_t_c = np.array([0.0824748 - 0.0013, 0.0, -0.1034 + 0.0095955], dtype=np.float32)

    c_R_t = r_y(np.deg2rad(25.0)) @ c_R_t
    c_R_t = r_z(np.deg2rad(-90.0)) @ c_R_t

    # Camera center -> pinhole offset, expressed in camera frame
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
    """
    Parse a poses file with lines like:
    Parse a poses file where each line contains 6 numbers (x y z roll pitch yaw).
    Lines may be bracketed and comma-separated, for example:
      [ 0.547..., 0.0688, 0.1468, -3.1136, 0.00166, 0.4497]
    Returns dict index -> (w_T_w_t (3,), w_rpy (3,)) where index is the
    zero-based line order (0,1,2,...).
    """
    poses = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Poses file not found: {path}")
    with open(path, "r") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Normalize spacing, remove brackets and commas
            line_clean = line.replace("[", " ").replace("]", " ")
            line_clean = line_clean.replace(",", " ")
            # Replace non-breaking spaces if present
            line_clean = line_clean.replace("\u00a0", " ")
            parts = line_clean.split()
            # Extract numeric tokens
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except Exception:
                    continue
            if len(nums) < 6:
                # skip malformed or incomplete lines
                continue
            w_T_w_t = np.array(nums[0:3], dtype=np.float32)
            w_rpy = np.array(nums[3:6], dtype=np.float32)
            poses[idx] = (w_T_w_t, w_rpy)
            idx += 1
    return poses


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
    """Save one subplot per image with masked pixels over a light red background."""
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
    """Save one subplot per image with depth values inside mask using viridis colormap.

    Filters out 0-values (no depth) and shows only masked regions without background.
    All subplots use the same color scale.
    """
    if not items:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Compute global min/max across all masked depth values
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

    vmin = np.min(all_valid_depths)
    vmax = min(np.max(all_valid_depths), 0.20)

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
        # Create masked depth array, keeping only non-zero values inside mask
        depth_masked = depth.copy()
        depth_masked[~mask_bool | (depth_masked == 0)] = np.nan

        # Find bounding box of non-NaN values and crop
        valid_mask = ~np.isnan(depth_masked)
        if not np.any(valid_mask):
            ax.set_title(f"{idx}")
            ax.axis("off")
            continue

        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        depth_cropped = depth_masked[rmin : rmax + 1, cmin : cmax + 1]

        # Plot with viridis, NaN values will not be displayed
        im = ax.imshow(depth_cropped, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{idx}")
        ax.axis("off")
        if idx == ordered_items[0][0]:  # Add colorbar only to the first subplot
            plt.colorbar(im, ax=ax, label="depth (m)")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    poses = parse_poses_file(poses_file)
    if not poses:
        print("No poses parsed from", poses_file)
        return

    segmenter = Sam3Detector()
    estimator = PoseEstimator(
        camera_info_json_path="camera_parameters/realsense_d405_single.json"
    )

    t_R_c, t_T_t_c = get_t_R_c_and_t_T_t_c()

    estimated_world = {}
    mask_visualizations = {}
    depth_visualizations = {}

    for idx, (w_T_w_t, w_rpy) in sorted(poses.items()):
        color_path = f"test_images/demo_img_color_{idx}_{exp_name}.tiff"
        depth_path = f"test_images/demo_img_depth_{idx}_{exp_name}.tiff"
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"Skipping {idx}: images not found")
            continue

        print(f"\n[{idx}] Loading {color_path} ...")
        image = np.array(Image.open(color_path).convert("RGB"))
        depth_raw = tifffile.imread(depth_path)
        depth_m = depth_raw.astype(np.float32) / 1000.0

        # Segment with SAM3 using prompt "key"
        inference_state = segmenter.processor.set_image(Image.fromarray(image))
        output = segmenter.processor.set_text_prompt(
            state=inference_state, prompt=sam3_prompt
        )
        masks = output.get("masks", [])
        scores = output.get("scores", None)
        if len(masks) == 0:
            print(f"  No masks found for {idx}, skipping")
            continue
        if scores is not None:
            scores_np = scores.cpu().numpy()
            mask_idx = int(np.argsort(scores_np)[-1])
        else:
            mask_idx = 0

        mask = (
            masks[mask_idx]
            .cpu()
            .numpy()
            .reshape(masks[0].shape[-2], masks[0].shape[-1])
            * 255
        )

        mask_visualizations[idx] = (image, mask)
        depth_visualizations[idx] = (depth_m, mask)

        # Set the ABUS mesh path in foundationpose and run estimation
        try:
            estimator._set_mesh_path(MESH_PATH_CONTAINER)
            pose_cam = estimator._estimate(image, depth_m, mask)
        except Exception as e:
            print(f"  Pose estimation failed for {idx}: {e}")
            continue

        # Convert to world frame using TCP pose from poses file
        w_R_t = euler_xyz_to_rot_matrix(*w_rpy)
        w_D_w_o = to_world_pose(pose_cam, w_R_t, w_T_w_t, t_R_c, t_T_t_c)
        estimated_world[idx] = w_D_w_o

        print(f"  Estimated world translation: {w_D_w_o[:3, 3]}")

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
