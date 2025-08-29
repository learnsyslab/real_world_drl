from gymnasium import RewardWrapper
from gymnasium.wrappers import TimeLimit
import numpy as np

class MaximizeHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot

    def reward(self, reward, action):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        #reward = 10 * action[2] - 0.1 * np.linalg.norm(action)
        reward = z - np.linalg.norm(action)

        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, action), terminated, truncated, info
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()