from gymnasium import RewardWrapper
from gymnasium.wrappers import TimeLimit

class MaximizeHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot

    def reward(self, reward):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        reward = z * 0.1 - 0.15

        return reward
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()