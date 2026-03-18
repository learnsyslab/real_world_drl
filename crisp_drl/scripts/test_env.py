import time
import cv2
import tifffile
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.agents.shared.insertion_wrapper_s import printoptions
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
from crisp_drl.envs.make_env import create_real_env_s1
import numpy as np
from pynput import keyboard
import imageio


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


alg_config = Config()
env_config = SiemensConfig()
args = Dict2Class({"eval": False, "use_pose_estimation": False})
env = create_real_env_s1(alg_config=alg_config, env_config=env_config, args=args)

try:
    while True:
        obs, reset_info = env.reset()
        shifted_goal_pos = (
            env_config.goal_position_ground_truth + reset_info["reset.grasped.delta"]
        )
        done = False
        while not done:
            current_pos = obs["observation.state.cartesian"][:3]
            perfect_action_yz = shifted_goal_pos[1:3] - current_pos[1:3]
            perfect_action_yz_norm = np.linalg.norm(perfect_action_yz)
            perfect_action_yz = (
                min(1.0, 0.00025 / perfect_action_yz_norm) * perfect_action_yz
            )
            obs, _, terminated, truncated, info = env.step(perfect_action_yz)
            done = terminated or truncated
        with printoptions(precision=4):
            print(f"Ended at {obs['observation.state.cartesian']}")


except KeyboardInterrupt:
    env.close()

# if velocity < 1mm/s and z-force < threshold -> decrease z
# if z-force > threshold -> increase z
