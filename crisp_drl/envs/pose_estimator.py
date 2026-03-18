import numpy as np
import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import Image, CameraInfo
from isaac_ros_tensor_list_interfaces.msg import TensorList, Tensor
from cv_bridge import CvBridge
import json
import os
from threading import Event


class PoseEstimator(Node):
    """Wrapper around FoundationPose for 6DoF pose estimation."""

    def __init__(
        self, camera_info_json_path: str = "", node_name: str = "pose_estimator"
    ):
        """
        Initialize the pose estimator.

        Args:
            camera_info_json_path: Path to camera info JSON file. If empty, tries default location.
            node_name: ROS2 node name.
        """
        # Initialize ROS2
        if not rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
            rclpy.init()

        super().__init__(node_name)
        self.bridge = CvBridge()

        # Load camera parameters
        if camera_info_json_path == "":
            # Try to find default camera info
            camera_info_json_path = (
                "/workspaces/isaac_ros-dev/lego_assets/camera_info.json"
            )

        if not os.path.exists(camera_info_json_path):
            raise FileNotFoundError(
                f"Camera info JSON not found: {camera_info_json_path}"
            )

        with open(camera_info_json_path) as f:
            self.camera_parameters = json.load(f)

        # Publishers for images
        self.rgb_pub = self.create_publisher(Image, "/rgb/image_rect_color", 10)
        self.depth_pub = self.create_publisher(
            Image, "/depth_registered/image_rect", 10
        )
        self.mask_pub = self.create_publisher(Image, "/segmentation", 10)
        self.caminfo_pub = self.create_publisher(CameraInfo, "/rgb/camera_info", 10)

        # Subscriber for pose output
        self.pose_sub = self.create_subscription(
            TensorList, "/pose_estimation/pose_matrix_output", self._pose_callback, 10
        )

        self.last_pose = None
        self.pose_ready = Event()

    def _pose_callback(self, msg: TensorList):
        """Handle pose output from FoundationPose."""
        try:
            # Extract pose tensor
            target: Tensor = next(filter(lambda t: t.name == "poses", msg.tensors))
            data_list = target.data
            b = bytes(data_list)
            arr = np.frombuffer(b, dtype=np.float32)

            # Reshape to 4x4 and transpose
            pose_mat = arr.reshape((4, 4)).T
            self.last_pose = pose_mat

        except Exception as e:
            self.get_logger().error(f"Error processing pose: {e}")

    def _set_mesh_file_lego(self, color: str):
        """Set the mesh file path parameter based on color using parameter client."""
        mesh_path_map = {
            "lavender": "/workspaces/isaac_ros-dev/lego_assets/lego_2x2_lavender_up.obj",
            "purple": "/workspaces/isaac_ros-dev/lego_assets/lego_2x2_purple_up.obj",
        }

        if color not in mesh_path_map:
            raise ValueError(f"Unknown color: {color}. Must be 'lavender' or 'purple'")

        mesh_path = mesh_path_map[color]
        self._set_mesh_path(mesh_path)

    def _set_mesh_file_siemens(self):
        mesh_path = "/workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj"
        self._set_mesh_path(mesh_path)

    def _set_mesh_path(self, mesh_path):
        # Create parameter client for the foundationpose node
        param_client = AsyncParameterClient(self, "/foundationpose")

        try:
            # Set the parameter
            future = param_client.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "mesh_file_path",
                        rclpy.Parameter.Type.STRING,  # pyright: ignore[reportPrivateImportUsage]
                        mesh_path,
                    )
                ]
            )

            # Spin briefly to process the set request
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)  # pyright: ignore[reportPrivateImportUsage]

            if future.done():
                result = future.result()
                if result is None:
                    raise RuntimeError("Parameter set returned None")
                # SetParametersResult has a 'results' list attribute
                if hasattr(result, "results") and len(result.results) > 0:
                    param_result = result.results[0]
                    if param_result.successful:
                        self.get_logger().info(f"Set mesh file to: {mesh_path}")
                    else:
                        raise RuntimeError(
                            f"Failed to set parameter: {param_result.reason}"
                        )
                else:
                    # Fallback: assume success if no results attribute
                    self.get_logger().info(f"Set mesh file to: {mesh_path}")
            else:
                raise TimeoutError("Parameter set request timed out")

        except Exception as e:
            self.get_logger().error(f"Failed to set mesh file: {e}")
            raise

    def estimate_lego(
        self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray, color: str
    ) -> np.ndarray:
        """
        Estimate 6DoF pose for an object.

        Args:
            image: RGB image as numpy array (HxWx3, uint8)
            depth: Depth image as numpy array (HxW, float32 in meters)
            mask: Segmentation mask (HxW, binary or from SAM3)
            color: Object color ("lavender" or "purple") to select correct mesh

        Returns:
            4x4 pose matrix as numpy array
        """
        # Set the mesh file based on color
        self._set_mesh_file_lego(color)
        return self._estimate(image, depth, mask)

    def estimate_siemens(
        self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate 6DoF pose for an object.

        Args:
            image: RGB image as numpy array (HxWx3, uint8)
            depth: Depth image as numpy array (HxW, float32 in meters)
            mask: Segmentation mask (HxW, binary or from SAM3)
            color: Object color ("lavender" or "purple") to select correct mesh

        Returns:
            4x4 pose matrix as numpy array
        """
        self._set_mesh_file_siemens()
        return self._estimate(image, depth, mask)

    def _estimate(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray):
        # Convert to ROS messages
        time_now_msg = self.get_clock().now().to_msg()

        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        except Exception:
            rgb_msg = self.bridge.cv2_to_imgmsg(image[:, :, ::-1], encoding="rgb8")

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        mask = mask.astype(np.uint8)
        print(
            "Mask unique values:",
            np.unique(mask),
            "dtype:",
            mask.dtype,
            "shape:",
            mask.shape,
        )
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        assert mask.sum() > 0, "Mask is empty!"

        print(
            "depth shape:",
            depth.shape,
            "dtype:",
            depth.dtype,
            "min:",
            np.min(depth),
            "max:",
            np.max(depth),
        )

        for m in (rgb_msg, depth_msg, mask_msg):
            m.header.stamp = time_now_msg
            m.header.frame_id = "camera"

        # Create and publish camera info
        cam_info = CameraInfo()
        cam_info.header.stamp = time_now_msg
        cam_info.header.frame_id = "camera"
        cam_info.width = image.shape[1]
        cam_info.height = image.shape[0]
        cam_info.k = self.camera_parameters["k"]
        cam_info.d = self.camera_parameters["d"]
        cam_info.r = self.camera_parameters["r"]
        cam_info.p = self.camera_parameters["p"]
        cam_info.distortion_model = self.camera_parameters["distortion_model"]

        # Reset pose ready event and publish
        self.last_pose = None

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.mask_pub.publish(mask_msg)
        self.caminfo_pub.publish(cam_info)

        # Wait for pose response with timeout
        self.get_logger().info("Waiting for pose estimation...")
        timeout = 10
        frequency = 0.1  # Check every 0.1 seconds
        for _ in range(int(timeout / frequency)):
            # Spin once to process callbacks
            rclpy.spin_once(self, timeout_sec=frequency)
            if self.last_pose is not None:
                self.get_logger().info("Pose received successfully")
                last_pose_copy = self.last_pose.copy()
                return last_pose_copy
        else:
            raise TimeoutError("Pose estimation did not respond within 10 seconds")
