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
    def __init__(self, assumed_orientation: np.ndarray, lock_orientation: bool = True):
        self.segmenter = Sam3Detector()
        self.pose_estimator = PoseEstimator(
            "camera_parameters/realsense_d405_single.json"
        )
        self.pose_tracker = PoseTracker("camera_parameters/realsense_d405_single.json")
        self.assumed_orientation = assumed_orientation
        self.lock_orientation = lock_orientation

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

        world_D_world_obj = np.eye(4)
        world_D_world_obj[:3, :3] = world_R_obj
        world_D_world_obj[:3, 3] = world_T_world_obj

        return world_D_world_obj

    def _compute_pose_in_tcp_frame(
        self,
        pose_in_cam_frame: np.ndarray,
    ) -> np.ndarray:
        """Compute translation from pose_from to pose_to in world frame."""

        cam_T_cam_obj = pose_in_cam_frame[:3, 3]
        cam_R_obj = pose_in_cam_frame[:3, :3]

        tcp_T_tcp_obj = TCP_T_TCP_CAM + TCP_R_CAM @ cam_T_cam_obj
        tcp_R_obj = TCP_R_CAM @ cam_R_obj

        tcp_D_tcp_obj = np.eye(4)
        tcp_D_tcp_obj[:3, :3] = tcp_R_obj
        tcp_D_tcp_obj[:3, 3] = tcp_T_tcp_obj

        return tcp_D_tcp_obj

    def _estimate_lego(self, image: np.ndarray, depth: np.ndarray):
        depth_f32 = depth.astype(np.float32) / 1000.0  # Convert to meters
        masks = self.segmenter.segment_lego(image)
        poses = {
            key: self.pose_estimator.estimate_lego(image, depth_f32, masks[key], key)
            for key in masks
        }
        if self.lock_orientation and self.assumed_orientation.size > 0:
            for key in poses:
                poses[key][:3, :3] = self.assumed_orientation
        return {
            key: self.pose_tracker.track_lego(image, depth_f32, poses[key], key)
            for key in poses
        }

    def estimate_two_lego_bricks_relative(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        estimation_position_euler: list[float] | np.ndarray,
    ) -> np.ndarray:
        refined_poses = self._estimate_lego(image, depth)
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
        refined_poses = self._estimate_lego(image, depth)
        return self._compute_pose_in_world_frame(
            refined_poses["lavender"],
            estimation_position_euler,
        ), self._compute_pose_in_world_frame(
            refined_poses["purple"],
            estimation_position_euler,
        )

    def estimate_siemens_ee_frame(
        self, image: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        depth_f32 = depth.astype(np.float32) / 1000.0  # Convert to meters
        mask = self.segmenter.segment_siemens(image)
        pose = self.pose_estimator.estimate_siemens(image, depth_f32, mask)
        if self.lock_orientation and self.assumed_orientation.size > 0:
            pose[:3, :3] = self.assumed_orientation
        return self.pose_tracker.track_siemens(image, depth_f32, pose)

    def estimate_siemens_tcp_frame_coarse(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        return_details: bool = False,
    ):
        depth_f32 = depth.astype(np.float32) / 1000.0  # Convert to meters
        mask = self.segmenter.segment_siemens(image)
        pose_cam = self.pose_estimator.estimate_siemens(image, depth_f32, mask)
        pose_tcp = self._compute_pose_in_tcp_frame(pose_cam)
        if return_details:
            return pose_tcp, {"pose_cam": pose_cam, "mask": mask}
        return pose_tcp

    def estimate_siemens_world_frame_coarse(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        estimation_position_euler: np.ndarray,
        return_details: bool = False,
    ):
        depth_f32 = depth.astype(np.float32) / 1000.0  # Convert to meters
        mask = self.segmenter.segment_siemens(image)
        pose_cam = self.pose_estimator.estimate_siemens(image, depth_f32, mask)
        pose_world = self._compute_pose_in_world_frame(
            pose_cam, estimation_position_euler
        )
        if return_details:
            return pose_world, {"pose_cam": pose_cam, "mask": mask}
        return pose_world
