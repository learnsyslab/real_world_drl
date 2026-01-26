import time
import numpy as np
import cv2
from PIL import Image
import tifffile
from crisp_drl.agents.shared.config import Config
from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.pose_tracker import PoseTracker
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


r_z_negative_pi_2 = np.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]
)


def euler_to_rot_matrix(roll, pitch, yaw):
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


def cam_in_tcp():
    cam_R_tcp = np.eye(3)
    tcp_T_tcp_cam = np.zeros(3)
    # offset to cam center
    tcp_T_tcp_cam += np.array(
        ##[0.0824748 - 0.0013, 0.0, 0.1034 - 0.0095955 - 0.035 + 0.04]
        [0.0824748 - 0.0013, 0.0, -0.1034 + 0.0095955]
    )  # ? -35mm-z for camera adapter? + ? for aloha gripper instead of franka
    # aloha tcp is 1.3mm in front (x) of gripper center
    # rotate around Y by 155 degrees
    ##Ry = r_y(np.deg2rad(155))
    Ry = r_y(np.deg2rad(25))
    cam_R_tcp = Ry @ cam_R_tcp
    ##Rz = r_z(np.deg2rad(90))
    Rz = r_z(np.deg2rad(-90))
    cam_R_tcp = Rz @ cam_R_tcp
    # # offset cam center to pinhole
    tcp_T_tcp_cam += cam_R_tcp.T @ np.array([-0.009, 0.0, -0.0037])
    # # camera has flipped x-axis
    # cam_R_tcp[0, :] *= -1
    tcp_R_cam = np.linalg.inv(cam_R_tcp)
    return tcp_R_cam, tcp_T_tcp_cam


TCP_R_CAM, TCP_T_TCP_CAM = cam_in_tcp()


def show_3d(block_poses):
    """
    Visualize the estimated block poses in 3D using PyVista.

    Args:
        block_poses: Dictionary with block names as keys and pose matrices (4x4) as values
    """
    plotter = pv.Plotter()

    # Color mapping for blocks
    color_map = {
        "lavender": "purple",
        "purple": "blue",
    }

    world_T_world_tcp = estimation_position_euler[:3]
    roll, pitch, yaw = estimation_position_euler[3:]
    world_R_tcp = euler_to_rot_matrix(roll, pitch, yaw)

    obj_path = "/home/linusschwarz/workspaces/isaac_ros-dev/lego_assets/lego_2x2_lavender_up.obj"
    mesh_scale = 1.0  # scale the loaded mesh if it's too large/small
    base_mesh = pv.read(obj_path)
    base_mesh.points = base_mesh.points * mesh_scale

    def plot_axes(R, t):
        arrow_x = pv.Arrow(start=t, direction=R[:, 0], scale=0.01 * mesh_scale)
        plotter.add_mesh(arrow_x, color="red")
        arrow_y = pv.Arrow(start=t, direction=R[:, 1], scale=0.01 * mesh_scale)
        plotter.add_mesh(arrow_y, color="green")
        arrow_z = pv.Arrow(start=t, direction=R[:, 2], scale=0.01 * mesh_scale)
        plotter.add_mesh(arrow_z, color="blue")

    def make_mesh(cam_T_cam_obj, cam_R_obj, color, axes=False):
        # compute world_T_world_obj
        world_T_world_obj = (
            world_T_world_tcp
            + world_R_tcp @ TCP_T_TCP_CAM
            + world_R_tcp @ TCP_R_CAM @ cam_T_cam_obj
        )
        world_R_obj = world_R_tcp @ TCP_R_CAM @ cam_R_obj

        obj_pose_world = np.eye(4)
        obj_pose_world[:3, :3] = world_R_obj
        obj_pose_world[:3, 3] = world_T_world_obj

        mesh_obj = base_mesh.copy(deep=True)
        R = obj_pose_world[:3, :3]
        t = obj_pose_world[:3, 3]
        mesh_obj.points = (R @ mesh_obj.points.T).T + t

        plotter.add_mesh(mesh_obj, opacity=0.6, color=color)
        if axes:
            plot_axes(world_R_obj, world_T_world_obj)
        return world_R_obj, world_T_world_obj

    # also plot small spheres for tcp positions and 3 arrows for tcp frame
    sphere = pv.Sphere(radius=0.001 * mesh_scale, center=world_T_world_tcp)
    plotter.add_mesh(sphere, color="black")

    # Plot each block
    for color_name, pose_matrix in block_poses.items():
        cam_T_cam_obj = pose_matrix[:3, 3]
        cam_R_obj = pose_matrix[:3, :3]
        _, world_T_world_grp = make_mesh(
            cam_T_cam_obj,
            cam_R_obj,
            "red" if color_name == "lavender" else "purple",
            axes=True,
        )

    # Plot world origin
    origin_sphere = pv.Sphere(radius=0.005 * mesh_scale, center=[0, 0, 0])
    plotter.add_mesh(origin_sphere, color="black")

    # Add world axes
    plotter.add_axes()  # pyright: ignore[reportCallIssue]
    plotter.show()


def main():
    print("=" * 60)
    print("Running Pose Estimation")
    print("=" * 60)

    sac_config = Config()

    try:
        # Load images
        image = Image.open("test_images/demo_img_color.tiff").convert("RGB")
        depth_img = tifffile.imread("test_images/demo_img_depth.tiff")
        print(
            f"Depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}, min: {np.min(depth_img)}, max: {np.max(depth_img)}"
        )

        mask_lavender = cv2.imread(
            "test_images/mask_lavender.png", cv2.IMREAD_GRAYSCALE
        )
        mask_purple = cv2.imread("test_images/mask_purple.png", cv2.IMREAD_GRAYSCALE)

        if mask_lavender is None or mask_purple is None:
            print(
                "Error: Could not load masks. Run test_sam3.py first to generate masks."
            )
            return

        masks_dict = {"lavender": mask_lavender, "purple": mask_purple}

        # Convert depth to float32 (assuming it's in millimeters)
        depth_f32 = depth_img.astype(np.float32) / 1000.0  # Convert to meters

        # Initialize pose estimator
        estimator = PoseEstimator(
            camera_info_json_path="camera_parameters/realsense_d405_single.json"
        )

        # Convert PIL image to numpy array
        image_np = np.array(image)

        tracker = PoseTracker(
            camera_info_json_path="camera_parameters/realsense_d405_single.json"
        )

        # Estimate pose for each object
        block_poses = {}
        for color, mask in masks_dict.items():
            print(f"\nEstimating pose for {color} brick...")
            t_est_start = time.time()

            pose_matrix = estimator.estimate(image_np, depth_f32, mask, color)
            t_est_end = time.time()

            print(
                f"Pose estimation for {color} completed in {t_est_end - t_est_start:.2f} seconds."
            )
            print(f"Pose matrix for {color}:")
            print(pose_matrix)
            print()

            pose_matrix[:3, :3] = sac_config.pose_estimation_assumed_orientation

            final_pose = tracker.track(
                image_np,
                depth_f32,
                initial_pose=pose_matrix,
                passes=8,
                randomization=0.0,
                timeout=10.0,
                color=color,
            )
            print(f"Refined pose matrix for {color}:")
            print(final_pose.tolist())

            block_poses[color] = final_pose
        show_3d(block_poses)

    except Exception as e:
        print(f"Error during pose estimation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
