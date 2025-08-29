import gymnasium as gym
import torch
import numpy as np

from rlmb.agents.sac.networks_cleanrl import Actor

from crisp_gym.manipulator_env_config import NoCamFrankaEnvConfig
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from rlmb.data.utils import crisp_obs_to_tensor
from rlmb.agents.sac.config import SAC_Config

manipulator_env_config = NoCamFrankaEnvConfig(max_episode_steps=50, control_frequency=2)
env = ManipulatorCartesianEnv(config = manipulator_env_config)


actor = Actor(env.observation_space, env.action_space, SAC_Config())
actor.load_state_dict(torch.load("checkpoints/FrankaCartesianEnv__config__20250828-151155/actor_state_dict.pth"))

env.home()
obs, info = env.reset()
for i in range(100):
    print(f"Episode {i}")
    terminated = False
    truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            obs_input = crisp_obs_to_tensor(obs)
            action, _, _ = actor.get_action(obs_input)
        obs, reward, terminated, truncated, info = env.step(action.view(-1).detach().cpu().numpy())

        if terminated or truncated:
            env.home()
            obs, info = env.reset()