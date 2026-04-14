import glob
import os
import time

import numpy as np
from PIL import Image
import tifffile
from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.pose_tracker import PoseTracker
from crisp_drl.envs.sam3_detector import Sam3Detector
import pyvista as pv

estimation_position_euler = [
    0.5187528,
    -0.03277275,
    0.07176751,
    3.1414583,
    -0.00552173,
    0.00340042,
]  # x, y, z, roll, pitch, yaw
mesh_scale = 1.0


def r_x(phi):
    """Rotation matrix around X axis."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ]
    )


def r_y(phi):
    """Rotation matrix around Y axis."""
    return np.array(
        [
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ]
    )


def r_z(phi):
    """Rotation matrix around Z axis."""
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )


def euler_to_rot_matrix(roll, pitch, yaw):
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


def cam_in_tcp():
    cam_R_tcp = np.eye(3)
    tcp_T_tcp_cam = np.zeros(3)
    # offset to cam center
    tcp_T_tcp_cam += np.array([0.0824748 - 0.0013, 0.0, -0.1034 + 0.0095955])
    # rotate around Y by 25 degrees
    Ry = r_y(np.deg2rad(25))
    cam_R_tcp = Ry @ cam_R_tcp
    Rz = r_z(np.deg2rad(-90))
    cam_R_tcp = Rz @ cam_R_tcp
    # offset cam center to pinhole
    tcp_T_tcp_cam += cam_R_tcp.T @ np.array([-0.009, 0.0, -0.0037])
    tcp_R_cam = np.linalg.inv(cam_R_tcp)
    return tcp_R_cam, tcp_T_tcp_cam


TCP_R_CAM, TCP_T_TCP_CAM = cam_in_tcp()


def show_3d(poses: dict[int, np.ndarray]):
    """
    Visualize all estimated Siemens lid poses in 3D using PyVista.

    Args:
        poses: Dictionary with image index as key and 4x4 cam-frame pose as value.
    """
    plotter = pv.Plotter()

    obj_path = "/home/linusschwarz/workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj"
    base_mesh = pv.read(obj_path)
    base_mesh.points = base_mesh.points * mesh_scale

    def plot_axes(R, t, scale=0.01):
        arrow_x = pv.Arrow(start=t, direction=R[:, 0], scale=scale * mesh_scale)
        plotter.add_mesh(arrow_x, color="red")
        arrow_y = pv.Arrow(start=t, direction=R[:, 1], scale=scale * mesh_scale)
        plotter.add_mesh(arrow_y, color="green")
        arrow_z = pv.Arrow(start=t, direction=R[:, 2], scale=scale * mesh_scale)
        plotter.add_mesh(arrow_z, color="blue")

    # Plot TCP position
    sphere = pv.Sphere(radius=0.001 * mesh_scale)
    plotter.add_mesh(sphere, color="black")

    # Use a colormap so each image index gets a distinct colour
    n = len(poses)
    cmap = pv.LookupTable("coolwarm", n_values=max(n, 2))

    for idx, (img_idx, pose_cam) in enumerate(sorted(poses.items())):
        cam_T_cam_obj = pose_cam[:3, 3]
        cam_R_obj = pose_cam[:3, :3]

        # Convert to world (TCP) frame
        tcp_T_tcp_obj = TCP_T_TCP_CAM + TCP_R_CAM @ cam_T_cam_obj
        tcp_R_obj = TCP_R_CAM @ cam_R_obj

        obj_pose_tcp = np.eye(4)
        obj_pose_tcp[:3, :3] = tcp_R_obj
        obj_pose_tcp[:3, 3] = tcp_T_tcp_obj

        mesh_obj = base_mesh.copy(deep=True)
        R = obj_pose_tcp[:3, :3]
        t = obj_pose_tcp[:3, 3]
        mesh_obj.points = (R @ mesh_obj.points.T).T + t

        color = cmap.map_value(idx / max(n - 1, 1))[:3]  # RGB 0-255
        color_f = [c / 255.0 for c in color]

        plotter.add_mesh(mesh_obj, opacity=0.5, color=color_f, label=f"img {img_idx}")  # pyright: ignore[reportArgumentType]
        plot_axes(tcp_R_obj, tcp_T_tcp_obj)

    # Plot world origin
    origin_sphere = pv.Sphere(radius=0.005 * mesh_scale, center=[0, 0, 0])
    plotter.add_mesh(origin_sphere, color="black")

    plotter.add_axes()  # pyright: ignore[reportCallIssue]
    plotter.add_legend()  # pyright: ignore[reportCallIssue]
    plotter.show()


def main():
    print("=" * 60)
    print("Running Siemens Lid Coarse Pose Estimation")
    print("=" * 60)

    # Discover all matching image pairs
    color_pattern = "test_images/demo_img_color_*_siemens.tiff"
    color_files = sorted(glob.glob(color_pattern))
    if not color_files:
        print("No siemens color images found matching pattern:", color_pattern)
        return

    # Extract indices from filenames
    indices: list[int] = []
    for f in color_files:
        base = os.path.basename(f)
        # demo_img_color_{i}_siemens.tiff  – skip any with extra parts (cropped, resized)
        parts = base.replace(".tiff", "").split("_")
        # Expected: demo img color <index> siemens
        if len(parts) == 5 and parts[3].isdigit():
            indices.append(int(parts[3]))
    indices.sort()
    print(f"Found {len(indices)} image pairs (indices {indices[0]}–{indices[-1]})")

    # Initialize estimator and segmenter
    estimator = PoseEstimator(
        camera_info_json_path="camera_parameters/realsense_d405_single.json"
    )
    segmenter = Sam3Detector()

    poses_coarse: dict[int, np.ndarray] = {}

    for i in indices:
        color_path = f"test_images/demo_img_color_{i}_siemens.tiff"
        depth_path = f"test_images/demo_img_depth_{i}_siemens.tiff"

        if not os.path.exists(depth_path):
            print(f"  Skipping index {i}: depth image not found")
            continue

        print(f"\n[{i}] Loading {color_path} ...")
        image = np.array(Image.open(color_path).convert("RGB"))
        depth_raw = tifffile.imread(depth_path)
        depth_f32 = depth_raw.astype(np.float32) / 1000.0  # mm -> m

        t0 = time.time()

        # Segment
        mask = segmenter.segment_siemens(image)

        # Coarse pose estimation (no tracking / refinement)
        pose = estimator.estimate_siemens(image, depth_f32, mask)
        poses_coarse[i] = pose

        dt = time.time() - t0
        print(f"  Pose estimated in {dt:.2f}s")
        print(f"  Translation (cam): {pose[:3, 3]}")
        # poses_fine[i] = pose

    print(f"\nSuccessfully estimated {len(poses_coarse)} / {len(indices)} poses")

    def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
        """Compute the mean rotation in SO(3) via SVD projection of the arithmetic mean."""
        R_sum = np.zeros((3, 3))
        for R in rotations:
            R_sum += R
        R_avg = R_sum / len(rotations)
        U, _, Vt = np.linalg.svd(R_avg)
        # ensure proper rotation (det = +1)
        D = np.diag([1, 1, np.linalg.det(U @ Vt)])
        return U @ D @ Vt

    def rotation_angle_around_z(R_rel: np.ndarray) -> float:
        """Extract the rotation angle around the z-axis from a relative rotation matrix.

        Uses atan2 of the (1,0) and (0,0) elements of R_rel, which gives
        the z-component of the rotation when the off-z components are small.
        """
        return float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))

    def print_pose_stats(poses: dict[int, np.ndarray], label: str):
        positions_tcp_T_obj = []
        rotations_tcp_R_obj = []
        for cam_D_cam_obj in poses.values():
            cam_T_cam_obj = cam_D_cam_obj[:3, 3]
            tcp_T_obj = TCP_T_TCP_CAM + TCP_R_CAM @ cam_T_cam_obj
            tcp_R_obj = TCP_R_CAM @ cam_D_cam_obj[:3, :3]
            positions_tcp_T_obj.append(tcp_T_obj)
            rotations_tcp_R_obj.append(tcp_R_obj)
        positions_tcp_T_obj = np.array(positions_tcp_T_obj)  # (N, 3)
        mean_pos_tcp_T_obj = positions_tcp_T_obj.mean(axis=0)
        deltas = np.linalg.norm(positions_tcp_T_obj - mean_pos_tcp_T_obj, axis=1)
        deltas_full = positions_tcp_T_obj - mean_pos_tcp_T_obj

        # Mean orientation in SO(3)
        tcp_R_obj_mean = mean_rotation(rotations_tcp_R_obj)

        # Relative z-rotation of each sample w.r.t. mean
        z_angles_deg = []
        for R in rotations_tcp_R_obj:
            R_rel = tcp_R_obj_mean.T @ R  # relative rotation
            z_angles_deg.append(np.degrees(rotation_angle_around_z(R_rel)))
        z_angles = np.array(z_angles_deg)

        print(f"\n--- {label} ---")
        print(f"  Mean position (TCP): {mean_pos_tcp_T_obj}")
        print(f"  Mean delta L2:       {deltas.mean():.6f} m")
        print(f"  Min dx:              {deltas_full[:, 0].min():.6f} m")
        print(f"  Min dy:              {deltas_full[:, 1].min():.6f} m")
        print(f"  Min dz:              {deltas_full[:, 2].min():.6f} m")
        print(f"  Max dx:              {deltas_full[:, 0].max():.6f} m")
        print(f"  Max dy:              {deltas_full[:, 1].max():.6f} m")
        print(f"  Max dz:              {deltas_full[:, 2].max():.6f} m")
        print(f"  Std  delta L2:       {deltas.std():.6f} m")
        print(f"  Mean orientation (TCP):")
        for row in tcp_R_obj_mean:
            print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
        print(f"  Relative z-rotation to mean (deg):")
        print(f"    Mean: {z_angles.mean():+.4f}°")
        print(f"    Std:  {z_angles.std():.4f}°")
        print(f"    Min:  {z_angles.min():+.4f}°")
        print(f"    Max:  {z_angles.max():+.4f}°")

    print_pose_stats(poses_coarse, "Coarse estimates")

    show_3d(poses_coarse)


if __name__ == "__main__":
    main()
