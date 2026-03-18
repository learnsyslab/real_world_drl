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
    pre_train_save_interval: int = 1
    """epochs between saving pre-training checkpoints"""
    pre_train_save_interval_pre_1: float = 0.1
    """epochs between saving pre-training checkpoints before utd=1"""
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
    """the batch size of sample from the replay memory"""
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
    utd_ratio: float = 30.0
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
    actor_nonvision_input_dim: int = 14  #  14  # 8
    """the input dimension of the non-vision part of the actor network"""
    actor_output_dim: int = 2
    """the output dimension of the actor network"""
    actor_q_hidden_dim: int = 128
    """the hidden dimension of the actor and Q-function networks"""

    actor_std: float = 0.03
    """the fixed standard deviation for the actor's action distribution"""

    shared_encoder_gradient: bool = True
    """if true, the gradients from the actor and critic will be backpropagated through the shared encoder"""
    pre_train_perfect: bool = False
    """if true, pre-train the Q-fn with perfect actions before training the RL agent"""

    custom_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.02646138,
                0.35014075,
                -0.03229037,
                -2.3585236,
                0.0231909,
                2.713794,
                0.70426035,
            ]
        )
    )
    """custom home position for the robot end-effector"""
    goal_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.5417, -0.031, 0.08956918])
    )  # 0.54262590, -0.030810941, 0.051836114 # 0.5422, -0.03125
    """the ground truth goal position in the real world"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.5100313, -0.03166383, 0.07772372])
    )  # 0.5106526, -0.03026352, 0.04147444
    """the ground truth grasp position in the real world"""
    demo_grasp_pose_estimation_pose_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.51002705, -0.03166126, 0.07771762, 3.1398149, -0.00529997, 0.00330537]
        )
    )
    """the position used during pose estimation for grasping during demo"""

    demo_goal_pose_estimation_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.52480197, -0.03154222, 0.0966146, -3.1412485, -0.00335749, 0.00383465]
        )
    )
    """whether to include the demo pose in the pose estimation computation"""
    pose_estimation_assumed_orientation: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [
                    [-0.02442653477191925, -0.997778058052063, -0.0619833804666996],
                    [-0.9342041015625, 0.0007091141305863857, 0.35673803091049194],
                    [-0.35590144991874695, 0.06661905348300934, -0.9321458339691162],
                ]
            ]
        )
    )
    """the assumed orientation (rotation matrix) for pose estimation"""
    demo_grasped_pose_lavender: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.99956242, -0.01966487, -0.02209571, 0.51189277],
                [0.02016189, 0.99954325, 0.02250109, -0.03525977],
                [0.02164308, -0.02293679, 0.99950251, 0.05194422],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    """the pose matrix extracted from the demo for the lavender brick before grasping"""
