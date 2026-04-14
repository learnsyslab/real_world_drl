#
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
class LegoConfig:
    # Algorithm specific arguments
    env_name: str = "FrankaCartesianEnv"
    # env_name: str = "Pendulum-v1"
    """the environment id of the task"""
    use_cameras: bool = True
    """if true, use camera image observations"""
    episode_length: int = 150
    """the maximum length of an episode"""
    control_frequency: int = 15
    """the frequency at which the control commands are sent to the robot"""

    n_cameras: int = 1
    """the number of cameras to use for observations"""

    # whether to zoom in after cropping and how faroxxxxxcocyyyyycoo

    # positions / relative rotations for
    # * first PE pos
    # * grasping,
    # * n waypoints after grasping with tolerances when to go to next
    # * n points after second pose estimation (if n=0, do not do second PE)
    # * goal position (demo)

    # insertion axis / controlled dof

    # poses:
    # * demo grasped object pose (optionally: make estimate better with multiple images)
    # * pose estimation uncertainty

    # ?? Allow for relative x / z rotations within small tolerance?

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


DELTA_Z_TEST = 0.0


@dataclass
class SiemensConfig:
    # Algorithm specific arguments
    env_name: str = "FrankaCartesianEnv"
    # env_name: str = "Pendulum-v1"
    """the environment id of the task"""
    use_cameras: bool = True
    """if true, use camera image observations"""
    episode_length: int = 150
    """the maximum length of an episode"""
    control_frequency: int = 15
    """the frequency at which the control commands are sent to the robot"""

    n_cameras: int = 1
    """the number of cameras to use for observations"""

    # whether to zoom in after cropping and how far

    # positions / relative rotations for
    # * first PE pos
    # * grasping,
    # * n waypoints after grasping with tolerances when to go to next
    # * n points after second pose estimation (if n=0, do not do second PE)
    # * goal position (demo)

    # insertion axis / controlled dof

    # poses:
    # * demo grasped object pose (optionally: make estimate better with multiple images)
    # * pose estimation uncertainty

    # ?? Allow for relative x / z rotations within small tolerance?

    custom_first_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.3649075,
                0.29130843,
                0.13225512,
                -2.0765812,
                -0.06310042,
                2.3661199,
                1.3285213,
            ]
        )
    )
    custom_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.40216795,
                0.5880874,
                0.05695235,
                -1.9398501,
                -0.11787688,
                2.573576,
                1.3399832,
            ]
        )
    )

    custom_home_position_pe: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.3726598,
                0.19371478,
                0.22794928,
                -1.9073441,
                -0.06185328,
                2.0956953,
                1.4201956,
            ]
        )
    )
    """custom home position for the robot end-effector"""
    goal_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.510, 0.1997, 1.2374244e-01 + DELTA_Z_TEST])
    )  # 0.54262590, -0.030810941, 0.051836114 # 0.5422, -0.03125
    """the ground truth goal position in the real world"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.56637836, 0.2764249, 0.07775394 + DELTA_Z_TEST]
        )
    )  # 0.5106526, -0.03026352, 0.04147444
    """the ground truth grasp position in the real world"""
    demo_grasp_pose_estimation_pose_euler: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """the position used during pose estimation for grasping during demo"""
    pose_estimation_assumed_orientation: np.ndarray = field(
        default_factory=lambda: np.array([[]])
    )
    """the assumed orientation (rotation matrix) for pose estimation"""
    relative_motion_before_grasp: list = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            0.0,
            np.deg2rad(-3),
            0.0,
        ]  # ry np.deg2rad(-4)
    )
    """motion before moving to the grasp position"""
    gripper_grasp_position: float = 0.2
    """gripper position during grasping"""
    relative_motion_after_grasp: list = field(
        default_factory=lambda: [-0.01, 0.0, 0.005, 0.0, 0.0, 0.0]
    )
    relative_motion_after_grasp_pe: list = field(
        default_factory=lambda: [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
    )
    """motion after grasping"""
    waypoints_after_grasp: list = field(
        default_factory=lambda: [
            ([0.505, 0.26139268, 0.1362928 + DELTA_Z_TEST, 0.0, 0.0, 0.0], 0.002),
            (
                [
                    0.505,
                    0.19800043,
                    0.1336346 + DELTA_Z_TEST,
                    0.0,
                    np.deg2rad(+3),
                    0.0,
                ],
                0.001,
            ),
            (
                [
                    5.05e-01,
                    1.9927530e-01,
                    0.1255438 + DELTA_Z_TEST,  # 1.245438e-01 + DELTA_Z_TEST,
                    0.0,
                    0.0,
                    0.0,
                ],
                0.0004,
            ),
        ]
    )
    """waypoints to go to after grasping, rotations are relative and open loop, positions closed loop and offset by delta grasp (xz) in global frame at identical orientation"""

    relative_motion_after_rl_train: list = field(
        default_factory=lambda: [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
    )
    """motion to do after episode finishes"""
    waypoints_after_rl_train: list = field(
        default_factory=lambda: [
            ([0.5222315, 0.26139268, 0.1362928 + DELTA_Z_TEST, 0.0, 0.0, 0.0], 0.002)
        ]
    )
    """waypoints to go to after rl training (after that go to position after (relative motion after grasp), then undo relative motion after grasp)"""

    dropoff_point: list = field(
        default_factory=lambda: [
            0.55748737,
            0.2779482,
            0.08486971 + DELTA_Z_TEST,
        ]
    )

    estimate_goal_position: bool = False

    insertion_axis_index: int = 0
    insertion_axis_sign: float = 1.0
    insertion_forcetorque: float = -0.21  # Nm or N
    insertion_forcetorque_index = 4
    ft_controller_lever_arm: float = 0.16  # set to 1 if force control
    ft_controller_k: float = (
        5000 / 1.5
    )  # use value smaller than real k to overcome friction
    rl_axis_indices: list = field(default_factory=lambda: [1, 2])

    demo_w_D_w_o: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.0108812, -0.9968131, 0.07902758, 0.5531513],
                [0.9999358, -0.0105985, 0.00399576, 0.27225485],
                [-0.00314545, 0.07906599, 0.9968645, 0.049977],
            ]
        )
    )

    demo_t_D_t_o: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.03220806, -0.98976707, 0.13900985, -0.01457414],
                [-0.99935341, 0.02966688, -0.02031458, 0.00354214],
                [0.0159827, -0.13957429, -0.9900825, 0.0273347],
            ]
        )
    )
