import os
import time

import numpy as np
from PIL import Image
import tifffile

from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.sam3_detector import Sam3Detector


# -----------------------
# User constants
# -----------------------
rgb_image_path = "test_images/demo_img_color_0_abus.tiff"
depth_image_path = "test_images/demo_img_depth_0_abus.tiff"
camera_info_json_path = "camera_parameters/realsense_d405_single.json"

# End-effector (TCP) pose in world frame
w_T_w_t = np.array([0.5605625, 0.28685653, 0.15395471], dtype=np.float32)
w_t_roll_pitch_yaw = np.array([3.135323, -0.06168376, -0.00784094], dtype=np.float32)

# 0.56657547  0.27647668  0.07798295 -> this tcp pose
# ground truth grasp: 0.56637836, 0.2764249, 0.07775394


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


def get_t_R_c_and_t_T_t_c() -> tuple[np.ndarray, np.ndarray]:
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
    """
    Convert object pose from camera frame to world frame.

    Inputs use:
      - x_R_a: rotation from frame a into frame x
      - y_T_m_n: translation from m to n expressed in frame y
    """
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]

    w_R_o = w_R_t @ t_R_c @ c_R_o
    w_T_w_o = w_T_w_t + w_R_t @ t_T_t_c + w_R_t @ t_R_c @ c_T_c_o

    w_D_w_o = np.eye(4, dtype=np.float32)
    w_D_w_o[:3, :3] = w_R_o
    w_D_w_o[:3, 3] = w_T_w_o
    return w_D_w_o


def main() -> None:
    if not os.path.exists(rgb_image_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_image_path}")
    if not os.path.exists(depth_image_path):
        raise FileNotFoundError(f"Depth image not found: {depth_image_path}")

    print("=" * 60)
    print("Siemens single-image world-frame pose estimation")
    print("=" * 60)

    image_rgb = np.array(Image.open(rgb_image_path).convert("RGB"))
    depth_mm = tifffile.imread(depth_image_path)
    depth_m = depth_mm.astype(np.float32) / 1000.0

    segmenter = Sam3Detector()
    estimator = PoseEstimator(camera_info_json_path=camera_info_json_path)

    t_start = time.time()
    mask = segmenter.segment_siemens(image_rgb)
    c_D_c_o = estimator.estimate_siemens(image_rgb, depth_m, mask)
    dt = time.time() - t_start

    w_R_t = euler_xyz_to_rot_matrix(*w_t_roll_pitch_yaw)
    t_R_c, t_T_t_c = get_t_R_c_and_t_T_t_c()
    w_D_w_o = to_world_pose(c_D_c_o, w_R_t, w_T_w_t, t_R_c, t_T_t_c)

    # Compute TCP frame quantities
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]
    t_R_o = t_R_c @ c_R_o
    t_T_t_o = t_T_t_c + t_R_c @ c_T_c_o
    t_D_t_o = np.eye(4)
    t_D_t_o[:3, :3] = t_R_o
    t_D_t_o[:3, 3] = t_T_t_o

    print(f"Pose estimated in {dt:.2f}s")
    print("c_D_c_o (camera frame):")
    print(c_D_c_o)
    print("t_D_t_o (TCP frame):")
    print(t_D_t_o)
    print("w_D_w_o (world frame):")
    print(w_D_w_o)
    print(f"w_T_w_o: {w_D_w_o[:3, 3]}")


if __name__ == "__main__":
    main()
