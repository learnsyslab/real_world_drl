from gymnasium import RewardWrapper, ActionWrapper
from gymnasium.wrappers import TimeLimit
import numpy as np
from crisp_gym.manipulator_env import ManipulatorCartesianEnv

class MaximizeHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper

    def reward(self, reward, action):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        #reward = 10 * action[2] - 0.1 * np.linalg.norm(action)
        reward = 10 * z - np.linalg.norm(action)

        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, action), terminated, truncated, info
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()


class SparseHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper

    def reward(self, reward, action):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        #if z > 0.9:
        #    reward = 100.0
        #else:
        #    reward = - 0.1 * np.linalg.norm(action)
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation.pop("observation.state.joint")
        observation.pop("task")
        observation.pop("observation.state.target")
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        if z > 0.9:
            terminated = True
        return observation, self.reward(reward, action), terminated, truncated, info
    
    def close(self):
        self.env.close()

    def reset(self, seed = None):
        obs, info = self.env.reset(seed=seed)
        obs.pop("observation.state.joint")
        obs.pop("task")
        obs.pop("observation.state.target")
        return obs, info

    def home(self):
        self.env.home()


class SafetyBoundingBoxWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper

        self.x_min = 0.25
        self.x_max = 0.6
        self.y_min = -0.15
        self.y_max = 0.15
        self.z_min = 0.18
        self.z_max = 0.5
    
    def action(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        eef_pose = self.robot.end_effector_pose
        current_pos = np.array(eef_pose.position)

        # Predict next position based on action (assuming action is a displacement)
        next_pos = current_pos + action[:3]  # Extract position part of the action

        # Check if next position violates bounds
        clipped_pos = np.clip(next_pos, 
                              [self.x_min, self.y_min, self.z_min],
                              [self.x_max, self.y_max, self.z_max])

        # If position would be clipped, adjust the action accordingly
        if not np.array_equal(next_pos, clipped_pos):
            delta = clipped_pos - current_pos
            action = np.concatenate([delta, action[3:]])  # Keep orientation part unchanged
            
        return action
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()