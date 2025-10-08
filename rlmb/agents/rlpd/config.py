# Copyright notice
#
# This file contains code adapted from cleanrl
# (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
# licensed under the MIT License.
#
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from dataclasses import dataclass
from typing import Tuple
import os
import gymnasium as gym


@dataclass
class RLPD_Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_name: str = "FrankaCartesianEnv"
    # env_name: str = "Pendulum-v1"
    """the environment id of the task"""
    use_cameras: bool = True
    """if true, use camera image observations"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    episode_length: int = 100
    """the maximum length of an episode"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(1)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    num_critics: int = 10
    """number of Q networks to sample for calculating the target value"""
    critic_subset_size: int = 2
    """number of Q networks to use for calculating the target value"""
    update_policy_after: int = 1
    """number of time steps after which the policy is updated"""
    utd_ratio: float = 1.0
    """the ratio of policy updates to environment steps taken"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    max_action: float = 0.05 if env_name not in list(gym.envs.registry.keys()) else None
    """the maximum norm of an action value the policy can output"""
    control_frequency: int = 10
    """the frequency at which the control commands are sent to the robot"""

    ## sparse rewards settings ##
    success_reward: float = 100.0
    """the reward given for task success"""
    failure_reward: float = -10.0
    """the reward given for task failure"""

