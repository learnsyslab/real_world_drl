"""Loads a lerobot dataset into a replay buffer and saves it to a pickle file."""

import argparse
import os
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from crisp_gym.manipulator_env_config import FrankaEnvConfig
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_gym.manipulator_env import ManipulatorCartesianEnv

from rlmb.data.buffers_cleanrl import ReplayBuffer
from rlmb.data.utils import load_buffer_from_lerobot_dataset
from rlmb.agents.rlpd.config import RLPD_Config


def main():
    parser = argparse.ArgumentParser(description="Loads a lerobot dataset into a replay buffer and saves it to a pickle file.")
    parser.add_argument("lerobot_id", type=str, help="The LeRobot dataset ID (e.g. danielsanjosepro/clean-up-table).")
    parser.add_argument("rb_save_path", type=str, help="Save location for the replay buffer pickle file.")
    parser.add_argument("--rb_file_name", type=str, default=None, help="File name for the replay buffer pickle file. If not set, uses the lerobot_id as file name.")
    args = parser.parse_args()

    ds = LeRobotDataset(args.lerobot_id)
    rb_file_name = args.lerobot_id.split("/")[-1] if args.rb_file_name is None else args.rb_file_name
    os.makedirs(args.rb_save_path, exist_ok=True)
    ds_length = len(ds)

    rlpd_config = RLPD_Config()
    gripper_config = GripperConfig(min_value=0.0, max_value=1.0)    
    primary_config = CameraConfig(
        camera_name="primary",
        resolution=(256, 256), 
        camera_color_image_topic="/camera/camera/color/image_rect_raw",
        camera_color_info_topic="/camera/camera/color/camera_info"
        )
    wrist_config = CameraConfig(
        camera_name="wrist",
        resolution=(256, 256), 
        camera_color_image_topic="/camera/camera/color/image_rect_raw",
        camera_color_info_topic="/camera/camera/color/camera_info"
        )
    manipulator_env_config = FrankaEnvConfig(max_episode_steps=100, control_frequency=rlpd_config.control_frequency, gripper_config=gripper_config, camera_configs=[primary_config, wrist_config])
    env = ManipulatorCartesianEnv(config = manipulator_env_config)
    device = "cuda" if rlpd_config.cuda and torch.cuda.is_available() else "cpu"
    rb = ReplayBuffer(ds_length, 
                      env.observation_space, 
                      [None for _ in env.cameras], 
                      env.action_space, 
                      device=device,
                      handle_timeout_termination=False)
    
    print("Converting dataset to replay buffer...")
    load_buffer_from_lerobot_dataset(ds, rb, ds_length)
    if not os.path.exists(args.rb_save_path):
        os.makedirs(args.rb_save_path)
    rb.save_buffer(os.path.join(args.rb_save_path, rb_file_name + ".pkl"))
    env.close()
    print("Done.")


if __name__ == "__main__":
   main()