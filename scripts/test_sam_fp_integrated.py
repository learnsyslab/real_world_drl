import imageio
import tifffile
import time

from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.agents.shared.config import Config

config = Config()

estimation_position_euler = [
    0.5187528,
    -0.03277275,
    0.07176751,
    3.1414583,
    -0.00552173,
    0.00340042,
]  # x, y, z, roll, pitch, yaw

image = imageio.imread("test_images/demo_img_color.tiff")
print("Image shape:", image.shape, "dtype:", image.dtype)
depth_img = tifffile.imread("test_images/demo_img_depth.tiff")

t0 = time.time()
pose_helper = PoseEstimationHelper(
    assumed_orientation=config.pose_estimation_assumed_orientation
)
t1 = time.time()
print(f"Loaded pose estimation helper in {t1 - t0:.2f} seconds.")
lavender_pose, purple_pose = pose_helper.estimate_two_lego_bricks_absolute(
    image, depth_img, estimation_position_euler
)
t2 = time.time()
print(f"Estimated absolute poses in {t2 - t1:.2f} seconds.")
print("Estimated lavender pose (world frame):", lavender_pose)
print("Estimated purple pose (world frame):", purple_pose)
translation = purple_pose[:3, 3] - lavender_pose[:3, 3]
print("Translation from lavender to purple (world frame):", translation)
