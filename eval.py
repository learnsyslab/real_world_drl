import gymnasium as gym
import torch
import numpy as np

from rlmb.agents.sac.networks_cleanrl import Actor



import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

def make_env():
    return gym.make("Pendulum-v1", render_mode="human")

# Create vector env
envs = SyncVectorEnv([make_env for _ in range(1)])

actor = Actor(envs)
actor.load_state_dict(torch.load("model_weights.pth"))

obs, info = envs.reset()
for _ in range(1000):
    with torch.no_grad():
        action, _, _ = actor.get_action(torch.Tensor(obs))
    obs, reward, terminated, truncated, info = envs.step(action)

    # Render each underlying environment
    for e in envs.envs:
        e.render()