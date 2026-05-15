import os
import time
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import tifffile
import pyvista as pv

from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.pose_tracker import PoseTracker
from crisp_drl.envs.sam3_detector import Sam3Detector

exp_name = "sd_card_test_pe_g"
poses_file = "test_images/sd_card_poses.txt"
MESH_FILE_NAME = "SD_Card_centered.obj"
# Path to mesh used for ABUS key
MESH_PATH_CONTAINER = (
    f"/workspaces/isaac_ros-dev/foundation_pose_meshes/{MESH_FILE_NAME}"
)
MESH_PATH_LOCAL = f"~/repos/foundation_pose_meshes/{MESH_FILE_NAME}"


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


def show_3d(poses_world: Dict[int, np.ndarray], mesh_path: str):
    plotter = pv.Plotter()
    base_mesh = pv.read(mesh_path)
    base_mesh.points = base_mesh.points.copy()

    n = len(poses_world)
    cmap = pv.LookupTable("coolwarm", n_values=max(n, 2))

    for i, (idx, w_pose) in enumerate(sorted(poses_world.items())):
        R = w_pose[:3, :3]
        t = w_pose[:3, 3]
        mesh_obj = base_mesh.copy(deep=True)
        mesh_obj.points = (R @ mesh_obj.points.T).T + t

        # Map index to a distinct colour and normalize to 0-1
        color = cmap.map_value(i / max(n - 1, 1))[:3]
        color_f = [c for c in color]

        plotter.add_mesh(mesh_obj, opacity=0.7, color=color_f, label=f"img {idx}")

        # axes at pose (slightly darker versions of the mesh colour)
        arrow_x = pv.Arrow(start=t, direction=R[:, 0], scale=0.01)
        plotter.add_mesh(arrow_x, color="red")
        arrow_y = pv.Arrow(start=t, direction=R[:, 1], scale=0.01)
        plotter.add_mesh(arrow_y, color="green")
        arrow_z = pv.Arrow(start=t, direction=R[:, 2], scale=0.01)
        plotter.add_mesh(arrow_z, color="blue")

    plotter.add_axes()
    plotter.add_legend()
    plotter.show()


def main():
    poses = parse_poses_file(poses_file)
    if not poses:
        print("No poses parsed from", poses_file)
        return

    segmenter = Sam3Detector()
    estimator = PoseEstimator(
        camera_info_json_path="camera_parameters/realsense_d405_single.json"
    )
    tracker = PoseTracker(
        camera_info_json_path="camera_parameters/realsense_d405_single.json"
    )

    t_R_c, t_T_t_c = get_t_R_c_and_t_T_t_c()

    estimated_world = {}
    tracked_world = {}

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
            state=inference_state, prompt="SD card"
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
        # Run tracking refinement with 4 passes if tracker is available
        print(f"  Running tracking refinement with 4 passes...")
        refined_pose_cam = tracker._track(image, depth_m, pose_cam, passes=16)
        refined_w_D_w_o = to_world_pose(
            refined_pose_cam, w_R_t, w_T_w_t, t_R_c, t_T_t_c
        )
        tracked_world[idx] = refined_w_D_w_o
        print(f"  Tracked world translation: {refined_w_D_w_o[:3, 3]}")

    if not estimated_world:
        print("No world poses estimated")
        return

    # Show first 3D plot with estimated poses
    print("\n[Visualization 1] Showing estimated poses in 3D...")
    show_3d(estimated_world, MESH_PATH_LOCAL)

    print("\n[Visualization 2] Showing tracked poses in 3D...")
    show_3d(tracked_world, MESH_PATH_LOCAL)


if __name__ == "__main__":
    main()
