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


from dataclasses import dataclass, field
import os
import numpy as np


@dataclass
class Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3
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
    total_timesteps: int = 199_000
    """total timesteps of the experiments"""
    episode_length: int = 150
    """the maximum length of an episode"""
    buffer_size: int = 200_000
    """the replay memory buffer size"""
    gamma: float = 0.97
    """the discount factor gamma"""
    n_step_return: int = 1
    """the number of steps to look ahead for multi-step returns"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 512
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    weight_decay: float = 0.0
    """weight decay for Q network and policy optimizers"""
    num_critics: int = 10
    """number of Q networks to sample for calculating the target value"""
    critic_subset_size: int = 2
    """number of Q networks to use for calculating the target value"""
    update_policy_after: int = 1
    """number of time steps after which the policy is updated"""
    utd_ratio: float = 20.0
    """the ratio of policy updates to environment steps taken"""
    alpha: float = 0.001  # 0.001
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    max_action: np.ndarray = field(default_factory=lambda: np.array([0.00025, 0.00025]))
    """the maximum norm of an action value the policy can output"""
    control_frequency: int = 15
    """the frequency at which the control commands are sent to the robot"""

    ## sparse rewards settings ##
    success_reward: float = 100.0
    """the reward given for task success"""
    failure_reward: float = -10.0
    """the reward given for task failure"""

    n_cameras: int = 1
    """the number of cameras to use for observations"""

    vision_head_input_dim: int = 384  # dinov2 512  # resnet
    """the input dimension of the vision head"""
    vision_head_hidden_dim: int = 128
    """the hidden dimension of the vision head"""
    vision_head_output_dim: int = 16
    """the output dimension of the vision head"""
    actor_nonvision_input_dim: int = 8
    """the input dimension of the non-vision part of the actor network"""
    actor_output_dim: int = 2
    """the output dimension of the actor network"""
    actor_q_hidden_dim: int = 128
    """the hidden dimension of the actor and Q-function networks"""

    actor_std: float = 0.03
    """the fixed standard deviation for the actor's action distribution"""

    shared_encoder_gradient: bool = False
    """if true, the gradients from the actor and critic will be backpropagated through the shared encoder"""
    pre_train_perfect: bool = False
    """if true, pre-train the Q-fn with perfect actions before training the RL agent"""

    custom_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.03049143,
                0.4469818,
                -0.02475526,
                -2.3372357,
                0.01456,
                2.7885091,
                0.71526223,
            ]
        )
    )
    """custom home position for the robot end-effector"""
    goal_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.54262590, -0.030810941, 0.051836114])
    )
    """the ground truth goal position in the real world"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.5106526, -0.03026352, 0.04147444])
    )
    """the ground truth grasp position in the real world"""
