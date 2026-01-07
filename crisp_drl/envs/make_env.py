import functools
import logging
import gymnasium as gym
import torch
import torch.multiprocessing as mp
import multiprocessing
import rclpy
import time
import random
import numpy as np

from crisp_gym.envs.manipulator_env_config import NoCamFrankaEnvConfig, FrankaEnvConfig
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
from crisp_gym.util.rl_utils import load_actions_safe
from crisp_gym.config.home import home_close_to_table
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from copy import deepcopy


from crisp_drl.agents.shared.config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor
from crisp_drl.agents.shared.env_wrappers import (
    BelowZTerminationWrapper,
    CLIWrapper,
    CustomTerminationWrapper,
    DictObservationToInfoMover,
    ContainerWatcherWrapper,
    FarAwayTerminationWrapper,
    ImageEncoderWrapper,
    DinoImageEncoderWrapper,
    InsertionResetWrapper,
    NaiveToGoalPositionWrapper,
    NaiveZForceWrapper,
    NoRotationActionWrapper,
    NoRotationNoGripperNoZActionClippedWrapperSim,
    NoRotationNoGripperNoZActionWrapper,
    ObservationFormatterWrapper,
    SafetyBoxWrapperXY,
    StepLimitEnforcerWrapper,
    TimeMeasurementWrapper,
    observation_has_z_pressure,
    observation_has_z_pressure_or_below,
)
from crisp_drl.data.utils import crisp_batch_concat_obs_to_tensor, crisp_obs_to_tensor
from crisp_drl.training.training_cli import clear_terminal
import mujid.env.env as mujid_env

from crisp_gym.envs.env_wrapper import (
    InsertionWrapper,
    LastObservationWrapper,
    NoRotationNoGripperActionWrapper,
    ActionTimeStampWrapper,
)


def create_real_env() -> gym.Env:
    """Create a new environment instance."""
    env = make_env("my_env")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = InsertionResetWrapper(
        env,
        initial_pos=np.array([0.200, -0.020, -0.200]),
        grasp_randomization_bounds=(
            np.array([-0.002, -0.002, -0.001]),
            np.array([0.002, 0.002, 0.001]),
        ),
        insert_randomization_bounds=(
            np.array([-0.002, -0.002, 0.0]),
            np.array([0.002, 0.002, 0.0]),
        ),
        action_sequence_to_grasp=load_actions_safe("v4_go_to_pick.json"),
        action_sequence_after_grasp=load_actions_safe("v4_after_pick.json"),
    )
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    env = CLIWrapper(env, termination_fn=lambda _obs: False)
    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    env = NaiveToGoalPositionWrapper(
        env,
        coarse=True,
        randomize=False,
        step_size_xy=0.01,
        step_size_z=0.00025,
        xy_threshold=0.1,
        base_goal_position=np.array([0.541, -0.034, 0.0435]),
        ideal_grasp_position=np.array([0.57155, -0.03254, 0.04243]),
    )  # [0.58833, -0.13817,  0.04229]))
    env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    env = DictObservationToInfoMover(env)
    # env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0),
    # ('observation.previous.action', (6,7), 20.0), ('observation.velocity.cartesian', (0, 3), 100.0),
    # ('observation.error.cartesian', (0, 3), 10.0), ('observation.velocity.gripper', (0, 1), 20.0),
    # ('observation.error.gripper', (0, 1), 20.0), ('observation.state.gripper', (0, 1), 1.0),
    # ('observation.target.gripper', (0, 1), 1.0), ('observation.images.wrist_camera', (0, 512), 1.0),
    # ('observation.images.side_camera', (0, 512), 1.0)])
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (0, 2), 10.0),
            ("observation.previous.error.cartesian", (0, 3), 10.0),
            ("observation.velocity.cartesian", (0, 3), 100.0),
            ("observation.error.cartesian", (0, 3), 10.0),
            ("observation.images.wrist_camera", (0, 512), 1.0),
            # ('observation.images.side_camera', (0, 512), 1.0)
        ],
    )  # 268 or 1036
    env = NoRotationNoGripperNoZActionWrapper(env)

    return env


def create_real_env_v3(config: Config) -> gym.Env:
    env = make_env("my_env_v3")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoRotationNoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapper(
        env,
        grasp_randomisation_z_range=(0.0005, 0.0015),
        home_config=config.custom_home_position,
        grasp_position_ground_truth=config.grasp_position_ground_truth,
        goal_position_ground_truth=config.goal_position_ground_truth,
        step_limit=config.episode_length,
    )
    env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    # maybe something with z velocity
    env = CLIWrapper(env)

    env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    env = DictObservationToInfoMover(env)

    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.error.cartesian", (0, 2), 1000.0),
            ("observation.velocity.cartesian", (0, 2), 1000.0),
            ("observation.error.cartesian", (0, 2), 1000.0),
            ("observation.images.wrist_camera", (0, 512), 1.0),
        ],
    )
    return env


def create_simulated_env(config: dict) -> gym.Env:
    """Create a new environment instance."""
    sac_config = Config()
    config["n_cameras"] = sac_config.n_cameras
    env = mujid_env.MujidEnv(
        config=config,
    )
    env = StepLimitEnforcerWrapper(env, max_steps=sac_config.episode_length)
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = NaiveZForceWrapper(
        env,
        step_size=0.00025,
        max_z_error=0.001,
    )
    env = SafetyBoxWrapperXY(
        env,
        step_size=0.0005,
        box_radius=0.004,
        randomization_box_radius=0.0024,
        base_goal_position=np.array([0.6, 0.0]),
        ideal_grasp_position=np.array([0.0, 0.0]),
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination)
    env = DinoImageEncoderWrapper(
        env, n_cameras=sac_config.n_cameras, image_size=(256, 256)
    )
    # env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    env = DictObservationToInfoMover(env)
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.error.cartesian", (0, 3), 1000.0),
            ("observation.velocity.cartesian", (0, 3), 1000.0),
            ("observation.error.cartesian", (0, 3), 1000.0),
            ("observation.images.wrist_camera_1", (0, 512), 1.0),
            # ("observation.images.wrist_camera_2", (0, 512), 1.0),
        ],
    )
    env = NoRotationNoGripperNoZActionClippedWrapperSim(env, clip=0.0003)

    return env


def custom_sim_termination(obs):
    if obs["observation.state.target"][2] < 0.125:
        print("E_FAIL (Z)")
        return "E_FAIL"

    fixed_box_pos = np.array([0.6, 0.0, 0.1198])
    moving_box_pos = obs["observation.state.moving_brick"]
    delta = np.abs(moving_box_pos - fixed_box_pos)
    err = np.abs(obs["observation.error.cartesian"])
    if delta[2] < 14e-3 and err[2] > 0.8e-3:
        if delta[0] < 1e-3 and delta[1] < 1e-3:
            # print("E_SUCCESS")
            return "E_SUCCESS"
        else:
            print("E_FAIL (Stuck)")
            return "E_FAIL"
