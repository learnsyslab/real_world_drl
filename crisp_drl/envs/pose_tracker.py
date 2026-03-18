import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.parameter
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import Image, CameraInfo
from isaac_ros_tensor_list_interfaces.msg import TensorList, Tensor, TensorShape
from cv_bridge import CvBridge
import json
import os
from threading import Event


class PoseTracker(Node):
    """Wrapper around FoundationPose tracking for iterative 6DoF refinement."""

    def __init__(
        self,
        camera_info_json_path: str = "",
        node_name: str = "pose_tracker",
        frame_id: str = "camera",
    ):
        """
        Initialize the pose tracker.

        Args:
            camera_info_json_path: Path to camera info JSON file. If empty, tries default location.
            node_name: ROS2 node name.
            frame_id: Frame id to stamp on published messages.
        """
        if not rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
            rclpy.init()

        super().__init__(node_name)
        self.bridge = CvBridge()
        self.frame_id = frame_id

        # Load camera parameters
        if camera_info_json_path == "":
            camera_info_json_path = (
                "/workspaces/isaac_ros-dev/lego_assets/camera_info.json"
            )

        if not os.path.exists(camera_info_json_path):
            raise FileNotFoundError(
                f"Camera info JSON not found: {camera_info_json_path}"
            )

        with open(camera_info_json_path) as f:
            self.camera_parameters = json.load(f)

        # Publishers for tracking topics
        self.rgb_pub = self.create_publisher(Image, "/tracking/image", 10)
        self.depth_pub = self.create_publisher(Image, "/tracking/depth_image", 10)
        self.pose_input_pub = self.create_publisher(
            TensorList, "/tracking/pose_input", 10
        )
        self.caminfo_pub = self.create_publisher(
            CameraInfo, "/tracking/camera_info", 10
        )

        # Subscriber for pose output
        self.pose_sub = self.create_subscription(
            TensorList, "/tracking/pose_matrix_output", self._pose_callback, 10
        )

        self.last_pose = None
        self.pose_ready = Event()
        self.expected_stamp = None

    def _pose_callback(self, msg: TensorList):
        """Handle tracking pose output and filter by expected stamp."""
        try:
            stamp = msg.header.stamp
        except Exception:
            stamp = None

        if self.expected_stamp is not None and stamp is not None:
            if int(stamp.sec) != int(self.expected_stamp["sec"]):
                return
            if int(stamp.nanosec) != int(self.expected_stamp["nanosec"]):
                return

        try:
            target: Tensor = next(filter(lambda t: t.name == "poses", msg.tensors))
            data_list = target.data
            b = bytes(data_list)
            arr = np.frombuffer(b, dtype=np.float32)
            pose_mat = arr.reshape((4, 4)).T
            self.last_pose = pose_mat
            self.pose_ready.set()
        except Exception as e:
            self.get_logger().error(f"Error processing tracking pose: {e}")

    def _build_pose_input(self, pose_mat: np.ndarray) -> TensorList:
        """Convert a 4x4 pose matrix to TensorList message expected by tracking."""
        pose_input_tensor_shape = TensorShape()
        pose_input_tensor_shape.rank = 3
        pose_input_tensor_shape.dims = [1, 4, 4]

        pose_input_tensor = Tensor()
        pose_input_tensor.name = "poses"
        pose_input_tensor.shape = pose_input_tensor_shape
        pose_input_tensor.data_type = 9  # FLOAT32
        pose_input_tensor.strides = [64, 16, 4]

        pose_array = np.array(pose_mat, dtype=np.float32).T.flatten()
        pose_input_tensor.data = list(pose_array.tobytes())

        pose_input_msg = TensorList()
        pose_input_msg.tensors = [pose_input_tensor]
        return pose_input_msg

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
        """Set the mesh file path parameter using parameter client."""
        self._set_mesh_path(
            "/workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj"
        )

    def _set_mesh_path(self, mesh_path):
        # Create parameter client for the foundationpose tracking node
        param_client = AsyncParameterClient(self, "/foundationpose_tracking_node")

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

    def track_lego(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        initial_pose: np.ndarray,
        color: str,
        passes: int = 4,
        randomization: float = 0.0,
        timeout: float = 10.0,
    ) -> np.ndarray:
        self._set_mesh_file_lego(color)
        return self._track(image, depth, initial_pose, passes, randomization, timeout)

    def track_siemens(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        initial_pose: np.ndarray,
        passes: int = 4,
        randomization: float = 0.0,
        timeout: float = 10.0,
    ) -> np.ndarray:
        self._set_mesh_file_siemens()
        return self._track(image, depth, initial_pose, passes, randomization, timeout)

    def _track(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        initial_pose: np.ndarray,
        passes: int = 4,
        randomization: float = 0.0,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """
        Perform iterative pose tracking.

        Args:
            image: RGB image as numpy array (HxWx3, uint8).
            depth: Depth image as numpy array (HxW, float32 in meters).
            initial_pose: Initial 4x4 pose matrix.
            passes: Number of refinement passes; each pass feeds the previous output.
            randomization: Uniform translation noise (meters) added on first pass.
            timeout: Seconds to wait for a pose per pass.

        Returns:
            The final 4x4 pose matrix after all passes.
        """
        if passes < 1:
            raise ValueError("passes must be at least 1")

        if initial_pose.shape != (4, 4):
            raise ValueError("initial_pose must be a 4x4 matrix")

        current_pose = np.array(initial_pose, dtype=np.float32)
        if randomization > 0.0:
            offsets = np.random.uniform(-randomization, randomization, size=(3,))
            try:
                current_pose = current_pose.copy()
                current_pose[0:3, 3] = current_pose[0:3, 3] + offsets
            except Exception:
                pass

        for _ in range(passes):
            self.last_pose = None
            self.pose_ready.clear()

            # Prepare ROS messages
            time_now_msg = self.get_clock().now().to_msg()
            self.expected_stamp = {
                "sec": int(time_now_msg.sec),
                "nanosec": int(time_now_msg.nanosec),
            }

            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            except Exception:
                rgb_msg = self.bridge.cv2_to_imgmsg(image[:, :, ::-1], encoding="rgb8")

            depth_msg = self.bridge.cv2_to_imgmsg(
                depth.astype(np.float32), encoding="32FC1"
            )
            pose_input_msg = self._build_pose_input(current_pose)

            for m in (rgb_msg, depth_msg, pose_input_msg):
                m.header.stamp = time_now_msg
                m.header.frame_id = self.frame_id

            cam_info = CameraInfo()
            cam_info.header.stamp = time_now_msg
            cam_info.header.frame_id = self.frame_id
            cam_info.width = image.shape[1]
            cam_info.height = image.shape[0]
            cam_info.k = self.camera_parameters["k"]
            cam_info.d = self.camera_parameters["d"]
            cam_info.r = self.camera_parameters["r"]
            cam_info.p = self.camera_parameters["p"]
            cam_info.distortion_model = self.camera_parameters["distortion_model"]

            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            self.pose_input_pub.publish(pose_input_msg)
            self.caminfo_pub.publish(cam_info)

            # Wait for response with timeout
            elapsed = 0.0
            period = 0.05
            while elapsed < timeout and not self.pose_ready.is_set():
                rclpy.spin_once(self, timeout_sec=period)
                elapsed += period

            if not self.pose_ready.is_set() or self.last_pose is None:
                raise TimeoutError("Tracking did not respond within timeout")

            current_pose = self.last_pose.copy()

        return np.array(current_pose, dtype=np.float32)


def main():
    tracker = PoseTracker()
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    tracker.destroy_node()
    if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
        rclpy.shutdown()


if __name__ == "__main__":
    main()
