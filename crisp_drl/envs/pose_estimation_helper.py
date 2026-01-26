import numpy as np
from crisp_drl.envs.sam3_detector import Sam3Detector
from crisp_drl.envs.pose_estimator import PoseEstimator
from crisp_drl.envs.pose_tracker import PoseTracker


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


def _cam_in_tcp():
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


def euler_to_rot_matrix(roll, pitch, yaw):
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


TCP_R_CAM, TCP_T_TCP_CAM = _cam_in_tcp()


class PoseEstimationHelper:
    def __init__(self, assumed_orientation: np.ndarray):
        self.segmenter = Sam3Detector()
        self.pose_estimator = PoseEstimator(
            "camera_parameters/realsense_d405_single.json"
        )
        self.pose_tracker = PoseTracker("camera_parameters/realsense_d405_single.json")
        self.assumed_orientation = assumed_orientation

    def _compute_translation_in_world_frame(
        self,
        pose_from: np.ndarray,
        pose_to: np.ndarray,
        estimation_position_euler: list[float] | np.ndarray,
    ) -> np.ndarray:
        """Compute translation from pose_from to pose_to in world frame."""
        world_T_world_tcp = estimation_position_euler[:3]
        roll, pitch, yaw = estimation_position_euler[3:]
        world_R_tcp = euler_to_rot_matrix(roll, pitch, yaw)

        def world_T_world_obj(cam_T_cam_obj):
            return (
                world_T_world_tcp
                + world_R_tcp @ TCP_T_TCP_CAM
                + world_R_tcp @ TCP_R_CAM @ cam_T_cam_obj
            )

        cam_T_cam_from = pose_from[:3, 3]
        cam_T_cam_to = pose_to[:3, 3]

        world_T_world_from = world_T_world_obj(cam_T_cam_from)
        world_T_world_to = world_T_world_obj(cam_T_cam_to)

        return world_T_world_to - world_T_world_from

    def _compute_pose_in_world_frame(
        self,
        pose_in_cam_frame: np.ndarray,
        estimation_position_euler: list[float] | np.ndarray,
    ) -> np.ndarray:
        """Compute translation from pose_from to pose_to in world frame."""
        world_T_world_tcp = estimation_position_euler[:3]
        roll, pitch, yaw = estimation_position_euler[3:]
        world_R_tcp = euler_to_rot_matrix(roll, pitch, yaw)

        cam_T_cam_obj = pose_in_cam_frame[:3, 3]
        cam_R_obj = pose_in_cam_frame[:3, :3]

        world_T_world_obj = (
            world_T_world_tcp
            + world_R_tcp @ TCP_T_TCP_CAM
            + world_R_tcp @ TCP_R_CAM @ cam_T_cam_obj
        )
        world_R_obj = world_R_tcp @ TCP_R_CAM @ cam_R_obj

        world_pose_matrix = np.eye(4)
        world_pose_matrix[:3, :3] = world_R_obj
        world_pose_matrix[:3, 3] = world_T_world_obj

        return world_pose_matrix

    def _estimate(self, image: np.ndarray, depth: np.ndarray):
        depth_f32 = depth.astype(np.float32) / 1000.0  # Convert to meters
        masks = self.segmenter.segment(image)
        poses = {
            key: self.pose_estimator.estimate(image, depth_f32, masks[key], key)
            for key in masks
        }
        for key in poses:
            poses[key][:3, :3] = self.assumed_orientation
        return {
            key: self.pose_tracker.track(image, depth_f32, poses[key], key)
            for key in poses
        }

    def estimate_two_lego_bricks_relative(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        estimation_position_euler: list[float] | np.ndarray,
    ) -> np.ndarray:
        refined_poses = self._estimate(image, depth)
        return self._compute_translation_in_world_frame(
            refined_poses["lavender"],
            refined_poses["purple"],
            estimation_position_euler,
        )

    def estimate_two_lego_bricks_absolute(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        estimation_position_euler: list[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        refined_poses = self._estimate(image, depth)
        return self._compute_pose_in_world_frame(
            refined_poses["lavender"],
            estimation_position_euler,
        ), self._compute_pose_in_world_frame(
            refined_poses["purple"],
            estimation_position_euler,
        )
