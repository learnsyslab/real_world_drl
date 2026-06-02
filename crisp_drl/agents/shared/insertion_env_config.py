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

    brick_size: str = "2x2"
    """LEGO brick size — selects which lego_<size>_*_up.obj mesh FoundationPose
    consumes. '2x2' is the legacy/trained-on size. '2x4' uses the longer 2x4
    bricks; calibration constants below are still 2x2-tuned and need
    re-recording for any 2x4 grasp/stack execution."""

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
        #default_factory=lambda: np.array([0.54803634, -0.02683013,  0.08606622]) # 4x4 lego
    )  # 0.54262590, -0.030810941, 0.051836114 # 0.5422, -0.03125
    """the ground truth goal position in the real world"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.5100313, -0.03166383, 0.07772372])
    )  # 0.5106526, -0.03026352, 0.04147444
    """the ground truth grasp position in the real world"""
    demo_grasp_pose_estimation_pose_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.51002705, -0.03166126, 0.07771762, 3.1398149, -0.00529997, 0.00330537] # 2x2 lego            
        )
    )
    """the position used during pose estimation for grasping during demo"""

    demo_goal_pose_estimation_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.52480197, -0.03154222, 0.0966146, -3.1412485, -0.00335749, 0.00383465] # 2x2 lego           
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

    # ---- 6DoF PE pipeline (used by InsertionWrapperLegoPE) ----
    # Wide PE pose, home, and after-grasp lift are reused from existing LEGO
    # constants (demo_goal_pose_estimation_euler, custom_home_position,
    # after_grasp_lift_height=0.016 in InsertionWrapper). Only the values that
    # have no analog in the original wrapper need calibration:

    diagonal_hover_above_purple_offset: np.ndarray = field(
        default_factory=lambda: np.array([-0.012, 0.0, 0.045])
    )
    """[dx,dy,dz] offset (world frame) from purple's wide-PE position to the
    diagonal hover where the post-grasp close-up PE happens. TODO calibrate
    from a hardware test — purple must remain in frame from this viewpoint."""

    place_z_offset: float = 0.0
    """extra z to add when stacking lavender on purple (e.g. one stud height).
    Set to 0 first; if lavender lands too low/high after a successful PE,
    nudge in 1 mm steps."""

    place_xy_correction: list = field(default_factory=lambda: [0.0, 0.0])
    """[dx, dy] world-frame correction (m) added to goal_xy after PE to compensate
    for systematic offset (gripper-occlusion bias or mesh-origin vs stud-center).
    Measure actual contact error, then negate it here in 1 mm steps."""

    after_grasp_lift_height_pe: float = 0.016
    """vertical lift right after closing the gripper, before moving to the
    diagonal hover above purple. Mirrors InsertionWrapper.after_grasp_lift_height."""

    alignment_clip_angle_rad: float = float(np.deg2rad(25.0))
    """axis-angle clip on the 6DoF grasp orientation alignment (mirrors
    InsertionWrapperSiemensPE.alignment_clip_angle_rad)."""

    alignment_pitch_bias: float = 0.0
    """small pitch bias added to the demo->estimated relative rotation
    before clipping (analog of SiemensConfig.alignment_pitch_bias, zero for
    LEGO since the lavender brick is grasped top-down)."""


@dataclass
class LegoConfig2x4(LegoConfig):
    """LegoConfig variant for 2x4 bricks. All poses calibrated from hardware PE."""

    brick_size: str = "2x4"
    demo_grasped_pose_lavender: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.01557534, -0.98369759, 0.17915489, 0.51451883],
                [0.99898551, -0.00773778, 0.04436334, -0.03187948],
                [-0.04225385, 0.17966410, 0.98282014, 0.05242069],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    goal_position_ground_truth: np.ndarray = field(
        # default_factory=lambda: np.array([0.5417, -0.031, 0.08956918])
        default_factory=lambda: np.array([0.54803634, -0.027500,  0.08606622]) # 4x4 lego        
    )  # 0.54262590, -0.030810941, 0.051836114 # 0.5422, -0.03125
   
    grasp_position_ground_truth: np.ndarray = field(
        # default_factory=lambda: np.array([0.51613176, -0.02726734, 0.07760019])
        default_factory=lambda: np.array([0.51668704, -0.027500,  0.07762174]) # 4x4 lego
        #default_factory=lambda: np.array([0.51668704, -0.02606872,  0.07762174]) # 4x4 lego
        
    )   
    demo_goal_pose_estimation_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [5.1669e-01, -2.7500e-02,  1.01464644e-01,
             3.13255072e+00,  1.15732178e-02,  2.18608044e-03]
        )
    )



LEGO_BRICK_CONFIGS = {
    "2x2": LegoConfig,
    "2x4": LegoConfig2x4,
}


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
                0.22355211,
                0.30581114,
                0.12800646,
                -2.27951312,
                -0.05865505,
                2.58594489,
                1.18903553,
            ]
        )
    )

    # prev: [0.3649075, 0.29130843, 0.13225512, -2.0765812, -0.06310042, 2.3661199, 1.3285213]
    custom_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            # eval
            [ 
                0.22355211,
                0.30581114,
                0.12800646,
                -2.27951312,
                -0.05865505,
                2.58594489,
                1.18903553,
            ]
            # data collection
            #[
            #    0.18534106,
            #    0.35176319,
            #    0.10533369,
            #    -2.30630183,
            #    -0.05615016,
            #    2.65457749,
            #    1.12368488,
            #]
        )
    )
    # prev: [0.40216795, 0.5880874, 0.05695235, -1.9398501, -0.11787688, 2.573576, 1.3399832]

    custom_home_position_pe: np.ndarray = field(
        default_factory=lambda: np.array(
            [ 
                0.22355211,
                0.30581114,
                0.12800646,
                -2.27951312,
                -0.05865505,
                2.58594489,
                1.18903553,
            ]            
        )
    )
    # prev: [0.3726598, 0.19371478, 0.22794928, -1.9073441, -0.06185328, 2.0956953, 1.4201956]
    """custom home position for the robot end-effector"""
    goal_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array(
            #[0.52555341, -0.00133, 0.12857927] # new
            [0.52555341, -0.000, 0.12957927] #0.52555341, -0.000, 0.12857927
        
        )
    )
    # prev: [0.52500522, -0.00301577, 0.13028786]; older: [0.510, 0.1997, 1.2374244e-01]
    """the ground truth goal position in the real world"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.516003445, 0.15649372, 0.07702135]
        )
    )
    # prev: [0.56637836, 0.2764249, 0.07775394]
    """the ground truth grasp position in the real world"""
    grasp_orientation_ground_truth_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [-3.14114428, 0.00215048, -0.00711133]
        )
    )
    # prev: [3.1398149, -0.00529997, 0.00330537]
    """the ground truth grasp orientation in roll, pitch, yaw"""
    demo_grasp_pose_estimation_pose_euler: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """the position used during pose estimation for grasping during demo"""
    goal_orientation_ground_truth_euler: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.13331290, -0.00359125, -0.00325737]
        )
    )
    # prev: [-3.14082360, -0.00211094, -0.00701243]; older: [-3.1412485, -0.00335749, 0.00383465]
    """the ground truth goal orientation in roll, pitch, yaw"""
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
            #(
                #[
                #    0.52082772,
                #    0.15709122,
                #    0.10043685,
                #    6.27520990,
                #    0.00381359,
                #    -0.00101191,
                #],
                #0.001,
            #),
            (
                [
                    0.52660835,
                    0.01853190,
                    0.13558331,
                    6.27223349,
                    np.deg2rad(4),  # +5° about Y, /5
                    -0.00290716,
                ],
                0.001,
            ),
            (
                [
                    0.52849388,
                    0.000, #-0.00133,
                    0.13057927,
                    0,
                    0,
                    0,
                ],
                0.001,
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
            (
                [
                    0.49750429,
                    0.15999970,
                    0.09169126 + DELTA_Z_TEST,
                    0.0,
                    0.0,
                    0.0,
                ],
                0.002,
            )
        ]
    )  # prev: [0.5222315, 0.26139268, 0.1362928, 0,0,0] (old layout)
       # set to waypoints_after_grasp[0] (the new lift-off above the brick) so
       # the post-rollout sweep stays in-workspace before going to dropoff.
    """waypoints to go to after rl training (after that go to position after (relative motion after grasp), then undo relative motion after grasp)"""

    dropoff_point: list = field(
        default_factory=lambda: [
            0.5195256882,
            0.157009372,
            0.07783166 + DELTA_Z_TEST,
        ]
    )  # prev: [0.55748737, 0.2779482, 0.08486971]

    estimate_goal_position: bool = False

    insertion_axis_index: int = 0
    insertion_axis_sign: float = 1.0
    insertion_forcetorque: float = -0.11  # Nm or N, default: -0.21
    insertion_forcetorque_index = 4
    ft_controller_lever_arm: float = 0.16  # set to 1 if force control
    ft_controller_k: float = (
        5000 / 1.5
    )  # use value smaller than real k to overcome friction
    rl_axis_indices: list = field(default_factory=lambda: [1, 2])

    demo_w_D_w_o: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.04309517, -0.99750481, 0.05591680, 0.50349407],
                [0.99893628, -0.04210279, 0.01880613, 0.15427599],
                [-0.01640494, 0.05666775, 0.99825820, 0.04920582],
            ]
        )
    )

    demo_t_D_t_o: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.01184618, -0.99333894, 0.11461804, -0.01245333],
                [-0.99992973, 0.01181975, -0.00091017, 0.00156189],
                [-0.00045065, -0.11462073, -0.99340919, 0.02711663],
            ]
        )
    )


@dataclass
class ShelfBoxConfig:
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
                -0.16125268,
                -0.19791204,
                -0.32023647,
                -2.4182775,
                -0.08884472,
                2.2305312,
                0.36234158,
            ]
        )
    )
    custom_home_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.29983243,
                0.38195968,
                -0.15804778,
                -2.2734513,
                0.08058647,
                2.640415,
                0.26108682,
            ]
        )
    )

    custom_home_position_pe: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                # -1.6601315e-01,
                # 7.1060816e-03,
                # -2.8422508e-01,
                # -2.0143490e00,
                # 1.0296288e-03,
                # 2.0213308e00,
                # 3.2737154e-01,
                ##
                # 0.00895363,
                # -0.2211075,
                # 0.08787364,
                # -2.2996998,
                # 0.02361965,
                # 2.0857165,
                # 0.84811467,
                -0.36450416,
                0.17809568,
                -0.16299678,
                -1.7158662,
                0.03090358,
                1.889031,
                0.24577244,
            ]
        )
    )
    """custom home position for the robot end-effector"""
    goal_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.339, -0.242, 0.168]
        )  # trained with y=0.2445, z=0.166 eval uses -0.244, yellow grasp eval -0.242y 0.168z
    )  # 0.54262590, -0.030810941, 0.051836114 # 0.5422, -0.03125
    """the ground truth goal position in the real world"""
    goal_position_from_barcode_offset_world: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.008, 0.0482, 0.0535]  # [0.011, 0.056, 0.061]  # before 0.008 / 0.056
        )
        + np.array(
            [-np.sqrt(2 / 3) * 0.0015 - 0.001, 0.0, 1 / np.sqrt(3) * 0.0015]
        )  # works for RL w/o approach: np.array([0.007, 0.051, 0.056])
    )  # works for 0.065 final blue distance 0.009, 0.048, 0.054
    """offset to add to the position obtained from the barcode for the goal position in the world frame"""
    grasp_position_ground_truth: np.ndarray = field(
        default_factory=lambda: np.array([0.475, -0.242, 0.085])
    )  # 0.5106526, -0.03026352, 0.04147444
    """the ground truth grasp position in the real world"""
    grasp_position_offset_from_barcode_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0505, 0.001, -0.015])
    )
    """offset to add to the position obtained from the barcode for the grasp position in the local frame of the barcode"""
    gripper_grasp_position: float = 0.52
    """gripper position during grasping"""
    relative_motions_after_grasp: list = field(
        default_factory=lambda: [
            np.array([-0.01, 0.0, 0.1, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-1)]),
            np.array([0.0, 0.0, 0.0, 0.0, np.deg2rad(30), 0.0]),
        ]
    )
    relative_motion_after_grasp_pe: list = field(
        default_factory=lambda: [
            np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-1)]),
            np.array([0.0, 0.0, 0.0, 0.0, np.deg2rad(30), 0.0]),
        ],
    )
    """motion after grasping"""
    relative_motion_after_rl_train: list = field(
        default_factory=lambda: [0.05 * 0.5, 0.0, 0.05 * 0.7071, 0.0, 0.0, 0.0]
    )
    """motion to do after episode finishes; in tilted plane -> z-axis is tilted"""
    waypoints_after_rl_train: list = field(
        default_factory=lambda: [
            ([0.475 - 0.01, -0.242, 0.085 + 0.1, 0.0, 0.0, 0.0], 0.002),
            ([0.475 - 0.01, -0.242, 0.085 + 0.1, 0.0, -np.deg2rad(30), 0.0], 0.002),
            ([0.475 - 0.01, -0.242, 0.085 + 0.1, 0.0, 0.0, np.deg2rad(1)], 0.002),
        ]
    )
    """waypoints to go to after rl training (after that go to position after (relative motion after grasp), then undo relative motion after grasp)"""

    dropoff_point: list = field(
        default_factory=lambda: [0.475 - 0.01, -0.242, 0.085 + 0.01]
    )

    insertion_forcetorque: float = -1.0  # Nm or N # trained with -0.66
    ft_controller_lever_arm: float = 1.0  # set to 1 if force control
    ft_controller_k: float = (
        5000 / 1.5
    )  # use value smaller than real k to overcome friction

    demo_yellow_grasped_pose_tcp: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.051413118839263916,
                -0.002239066641777754,
                -0.020148837938904762,
            ]
        )
    )
