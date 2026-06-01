import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.sam3_detector import Sam3Detector

exp_name = "sugar_grasped"
sam3_prompt = "yellow white cardboard box with label"  # "black cover with circular grille"  # "small grey plastic rectangle with gear"
MESH_FILE_NAME = "ycb_sugar_centered.obj"

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


def to_tcp_pose(
    c_D_c_o: np.ndarray,
    t_R_c: np.ndarray,
    t_T_t_c: np.ndarray,
) -> np.ndarray:
    """Convert pose from camera frame to TCP frame."""
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]

    t_R_o = t_R_c @ c_R_o
    t_T_t_o = t_T_t_c + t_R_c @ c_T_c_o

    t_D_t_o = np.eye(4, dtype=np.float32)
    t_D_t_o[:3, :3] = t_R_o
    t_D_t_o[:3, 3] = t_T_t_o
    return t_D_t_o


def get_image_indices(base_path: str, exp_name: str) -> list:
    """
    Get list of image indices by scanning for image files.
    """
    indices = []
    if not os.path.exists("test_images"):
        raise FileNotFoundError("Directory 'test_images' not found")
    for fname in os.listdir("test_images"):
        if f"{exp_name}" in fname:
            try:
                idx_str = fname.split("_")[3]
                indices.append(int(idx_str))
                # print(f"Found image for index {idx_str}")
            except (IndexError, ValueError):
                # print(f"Warning: could not parse index from filename '{fname}'")
                continue
    return sorted(set(indices))


def save_estimated_poses(path: str, poses_tcp: Dict[int, np.ndarray]) -> None:
    """Save estimated TCP frame poses to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for idx, t_pose in sorted(poses_tcp.items()):
        rows.append(np.concatenate(([float(idx)], t_pose.reshape(-1))))
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
    image_indices = get_image_indices("test_images", exp_name)
    if not image_indices:
        print(f"No images found for {exp_name}")
        return

    segmenter = Sam3Detector()
    estimator = PoseEstimator(
        camera_info_json_path="camera_parameters/realsense_d405_single.json"
    )

    t_R_c, t_T_t_c = get_t_R_c_and_t_T_t_c()

    estimated_tcp = {}
    mask_visualizations = {}
    depth_visualizations = {}

    for idx in image_indices:
        color_path = f"test_images/demo_img_color_{idx}_{exp_name}.tiff"
        depth_path = f"test_images/demo_img_depth_{idx}_{exp_name}.tiff"
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"Skipping {idx}: images not found")
            continue

        print(f"\n[{idx}] Loading {color_path} ...")
        image = np.array(Image.open(color_path).convert("RGB"))
        depth_raw = tifffile.imread(depth_path)
        depth_m = depth_raw.astype(np.float32) / 1000.0

        # Segment with SAM3 using prompt
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

        # Set the mesh path in foundationpose and run estimation
        try:
            estimator._set_mesh_path(MESH_PATH_CONTAINER)
            pose_cam = estimator._estimate(image, depth_m, mask)
        except Exception as e:
            print(f"  Pose estimation failed for {idx}: {e}")
            continue

        # Convert to TCP frame
        t_D_t_o = to_tcp_pose(pose_cam, t_R_c, t_T_t_c)
        estimated_tcp[idx] = t_D_t_o

        print(f"  Estimated TCP translation: {t_D_t_o[:3, 3]}")

    if not estimated_tcp:
        print("No TCP poses estimated")
        return

    save_estimated_poses(estimated_poses_file, estimated_tcp)
    save_mask_figure(mask_figure_file, mask_visualizations)
    save_depth_figure(depth_figure_file, depth_visualizations)
    print(f"Saved {len(estimated_tcp)} estimated TCP poses to {estimated_poses_file}")
    print(f"Saved mask figure to {mask_figure_file}")
    print(f"Saved depth figure to {depth_figure_file}")


if __name__ == "__main__":
    main()
