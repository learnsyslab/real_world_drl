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

from crisp_drl.agents.shared.insertion_env_config import (
    LEGO_BRICK_CONFIGS,
    LegoConfig,
    SiemensConfig,
)
from crisp_drl.agents.shared.insertion_wrapper_s import (
    InsertionWrapperSiemens,
    InsertionWrapperSiemensPE,
)
from crisp_drl.agents.shared.insertion_wrapper_lego import InsertionWrapperLegoPE
from crisp_drl.agents.shared.insertion_wrapper_lego_2x4 import (
    InsertionWrapperLego2x4PE,
)
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


from crisp_drl.agents.shared.algorithm_config import Config
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
    InsertionWrapperSim,
    InsertionWrapperSim3D,
    InsertionWrapperSim3DoFRotZ,
    InsertionWrapperSim5DoF,
    MotionPlannerWrapper,
    NaiveToGoalPositionWrapper,
    NaiveZForceWrapper,
    NoRotationActionWrapper,
    NoRotationNoGripperNoZActionClippedWrapperSim,
    NoRotationNoGripperNoZActionWrapper,
    NoGripperActionWrapper,
    NoRotationNoGripperWrapperSim,
    ObservationFormatterWrapper,
    ZeroFTInjectorWrapper,
    SafetyBoxWrapperXY,
    StepLimitEnforcerWrapper,
    SuccessClassificationWrapper,
    TimeMeasurementWrapper,
    observation_has_z_pressure,
    observation_has_z_pressure_or_below,
)
from crisp_drl.agents.shared.insertion_wrapper import (
    InsertionWrapper,
    SensorTareWrapper,
)
from crisp_drl.data.utils import crisp_batch_concat_obs_to_tensor, crisp_obs_to_tensor
from crisp_drl.training.training_cli import clear_terminal
import mujid.env.env as mujid_env

from crisp_gym.envs.env_wrapper import (
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
    env = CLIWrapper(env)
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


def create_real_env_v3(config: Config, args=None) -> gym.Env:
    env = make_env("my_env_v3")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoRotationNoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    # env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    is_eval = False if args is None else args.eval
    env = InsertionWrapper(
        env,
        config=config,
        grasp_randomisation_z_range=(0.0005, 0.0015),
        step_limit=config.episode_length if not is_eval else 2 * config.episode_length,
        is_eval=is_eval,
        use_pose_estimation=True
        if args is not None and args.use_pose_estimation
        else False,
    )
    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    # maybe something with z velocity
    env = CLIWrapper(env)

    env = DinoImageEncoderWrapper(
        env,
        n_cameras=config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={
            "observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)
        },  #  (227, 483, 138, 394)},
        rescales={"observation.images.wrist_camera": 1.5},
    )

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
            ("observation.features.wrist_camera", (0, 512), 1.0),
        ],
    )
    return env


def create_real_env_v3_timo(config: Config) -> gym.Env:
    env = make_env("my_env_v3")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoRotationNoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    # env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))

    env = InsertionWrapper(
        env,
        config=config,
        grasp_randomisation_z_range=(0.0005, 0.0015),
        step_limit=config.episode_length,
        is_eval=False,
        use_pose_estimation=True,
    )

    env = CLIWrapper(env)

    env = DinoImageEncoderWrapper(
        env,
        n_cameras=config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={
            "observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)
        },  #  (227, 483, 138, 394)},
        rescales={"observation.images.wrist_camera": 1.5},
    )

    return env


def create_real_env_v4(config: Config, args=None) -> gym.Env:
    no_ft = bool(args is not None and getattr(args, "no_ft_sensor", False))
    env = make_env("my_env_v4_no_ft" if no_ft else "my_env_v4")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoRotationNoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    # env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    if no_ft:
        env = ZeroFTInjectorWrapper(env)
    else:
        env = SensorTareWrapper(
            env,
            sensor_key="observation.state.sensors_bota_ft_sensor",
            sensor_data_shape=(6,),
        )
    is_eval = False if args is None else args.eval
    env = InsertionWrapper(
        env,
        config=config,
        grasp_randomisation_z_range=(0.0005, 0.0015) if is_eval else (0.00025, 0.00175),
        grasp_randomisation_x_range=(-0.0015, 0.0015) if is_eval else (-0.002, 0.002),
        safety_box_radius=0.004 if is_eval else 0.003,
        step_limit=config.episode_length if not is_eval else 2 * config.episode_length,
        is_eval=is_eval,
        use_pose_estimation=True
        if args is not None and args.use_pose_estimation
        else False,
        use_ft_controller=not no_ft,
    )

    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    # maybe something with z velocity
    env = CLIWrapper(env)

    env = DinoImageEncoderWrapper(
        env,
        n_cameras=config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={
            "observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)
        },  #  (227, 483, 138, 394)},
        rescales={"observation.images.wrist_camera": 1.5},
    )

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
            ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1),
            ("observation.features.wrist_camera", (0, 512), 1.0),
        ],
    )
    if no_ft:
        mp_backend = getattr(args, "mp_backend", "quintic") if args is not None else "quintic"
        env = MotionPlannerWrapper(env, backend=mp_backend)
    return env


def create_real_env_v4_pe(
    alg_config: Config,
    env_config: LegoConfig | None = None,
    args=None,
    brick_size: str = "2x2",
) -> gym.Env:
    """LEGO twin of create_real_env_s1_pe: 6DoF PE-driven grasp + placement."""
    no_ft = bool(args is not None and getattr(args, "no_ft_sensor", False))
    env = make_env("my_env_v4_no_ft" if no_ft else "my_env_v4")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)  # 6D action so PE wrapper can rotate EE
    env = LastObservationWrapper(env)
    if no_ft:
        env = ZeroFTInjectorWrapper(env)
    else:
        env = SensorTareWrapper(
            env,
            sensor_key="observation.state.sensors_bota_ft_sensor",
            sensor_data_shape=(6,),
        )
    is_eval = False if args is None else args.eval
    resolved_brick_size = getattr(args, "brick_size", brick_size)
    if env_config is None:
        cfg_cls = LEGO_BRICK_CONFIGS.get(resolved_brick_size, LegoConfig)
        env_config = cfg_cls()
    wrapper_cls = (
        InsertionWrapperLego2x4PE
        if getattr(env_config, "brick_size", "2x2") == "2x4"
        else InsertionWrapperLegoPE
    )

    env = wrapper_cls(
        env,
        alg_config=alg_config,
        env_config=env_config,
        safety_box_radius=0.004 if is_eval else 0.003,
        safety_box_step_size=0.0005,
        step_limit=env_config.episode_length
        if not is_eval
        else 2 * env_config.episode_length,
        use_ft_controller=not no_ft,
        pose_viz_dir=getattr(args, "pose_viz_dir", None) if args is not None else None,
    )

    env = DinoImageEncoderWrapper(
        env,
        n_cameras=env_config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={"observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)},
        rescales={"observation.images.wrist_camera": 1.5},
    )

    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    # Match create_real_env_v4 obs schema: 2D action/error slices, 6D FT, 512-D
    # DINO features (rescaled flow keeps the same encoder dim as v4).
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.error.cartesian", (0, 2), 1000.0),
            ("observation.velocity.cartesian", (0, 2), 1000.0),
            ("observation.error.cartesian", (0, 2), 1000.0),
            ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1),
            ("observation.features.wrist_camera", (0, 512), 1.0),
        ],
    )
    success_threshold = (
        getattr(args, "no_ft_success_threshold", 4.0)
        if no_ft
        else getattr(args, "success_threshold", 8.0)
    )
    if args is not None and getattr(args, "load_policy", None):
        env = SuccessClassificationWrapper(
            env,
            args=args,
            sac_config=alg_config,
            threshold=success_threshold,
        )
    env = CLIWrapper(env)
    mp_backend = getattr(args, "mp_backend", "quintic") if args is not None else "quintic"
    env = MotionPlannerWrapper(env, backend=mp_backend)
    return env


def create_real_env_s1(
    alg_config: Config, env_config: SiemensConfig, args=None
) -> gym.Env:
    no_ft = bool(args is not None and getattr(args, "no_ft_sensor", False))
    use_6dof_grasp = bool(args is not None and getattr(args, "use_6dof_grasp", False))
    dof_slice_end = 6 if getattr(alg_config, "actor_output_dim", 2) >= 5 else 3
    env = make_env("my_env_v4_no_ft" if no_ft else "my_env_v4")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    # env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    if no_ft:
        env = ZeroFTInjectorWrapper(env)
    else:
        env = SensorTareWrapper(
            env,
            sensor_key="observation.state.sensors_bota_ft_sensor",
            sensor_data_shape=(6,),
        )
    is_eval = False if args is None else args.eval
    env = InsertionWrapperSiemens(
        env,
        alg_config=alg_config,
        env_config=env_config,
        grasp_randomisation_x_range=(-0.00175, 0.00175)
        if not is_eval
        else (-0.0015, 0.0015),
        grasp_randomisation_z_range=(-0.001, 0.001) if not is_eval else (-0.0, 0.0),
        safety_box_radius=0.003,
        safety_box_step_size=0.0004,
        minimal_start_goal_distance=0.002,
        step_limit=env_config.episode_length
        if not is_eval
        else 2 * env_config.episode_length,
        is_eval=is_eval,
        use_pose_estimation=True
        if args is not None and args.use_pose_estimation
        else False,
        use_ft_controller=not no_ft,
        use_6dof_grasp=use_6dof_grasp,
    )

    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    # maybe something with z velocity
    env = CLIWrapper(env)
    env = DinoImageEncoderWrapper(
        env,
        n_cameras=env_config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={"observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)},
    )

    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (1, dof_slice_end), 1000.0),
            ("observation.previous.error.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.velocity.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.error.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1),
            ("observation.features.wrist_camera", (0, 512), 1.0),
        ],
    )
    return env


def create_real_env_s1_pe(
    alg_config: Config, env_config: SiemensConfig, args=None
) -> gym.Env:
    no_ft = bool(args is not None and getattr(args, "no_ft_sensor", False))
    use_6dof_grasp = bool(args is not None and getattr(args, "use_6dof_grasp", False))
    dof_slice_end = 6 if getattr(alg_config, "actor_output_dim", 2) >= 5 else 3
    env = make_env("my_env_v4_no_ft" if no_ft else "my_env_v4")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    # env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
    if no_ft:
        env = ZeroFTInjectorWrapper(env)
    else:
        env = SensorTareWrapper(
            env,
            sensor_key="observation.state.sensors_bota_ft_sensor",
            sensor_data_shape=(6,),
        )
    is_eval = False if args is None else args.eval
    env = InsertionWrapperSiemensPE(
        env,
        alg_config=alg_config,
        env_config=env_config,
        safety_box_radius=0.003,
        safety_box_step_size=0.0004,
        step_limit=env_config.episode_length
        if not is_eval
        else 2 * env_config.episode_length,
        use_ft_controller=not no_ft,
        use_6dof_grasp=use_6dof_grasp,
        pose_viz_dir=getattr(args, "pose_viz_dir", None) if args is not None else None,
    )

    # obs["observation.state.cartesian"][2] < 0.049)
    #  # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003,
    #  # min_z_height=0.055, terminate_z_height = 0.0475))
    # maybe something with z velocity

    env = DinoImageEncoderWrapper(
        env,
        n_cameras=env_config.n_cameras,
        image_keys=["observation.images.wrist_camera"],
        image_size=(256, 256),
        crops={"observation.images.wrist_camera": (175, 175 + 224, 346, 346 + 224)},
    )

    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=[
            ("observation.previous.action", (1, dof_slice_end), 1000.0),
            ("observation.previous.error.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.velocity.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.error.cartesian", (1, dof_slice_end), 1000.0),
            ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1),
            ("observation.features.wrist_camera", (0, 384), 1.0),
        ],
    )
    success_threshold = (
        getattr(args, "no_ft_success_threshold", 4.0)
        if no_ft
        else getattr(args, "success_threshold", 8.0)
    )
    env = SuccessClassificationWrapper(
        env,
        args=args,
        sac_config=alg_config,
        threshold=success_threshold,
    )
    env = CLIWrapper(env)
    mp_backend = getattr(args, "mp_backend", "quintic") if args is not None else "quintic"
    env = MotionPlannerWrapper(env, backend=mp_backend)
    return env


def create_simulated_env(
    mujid_config: dict,
    sac_config: Config = Config(),
    is_eval=False,
    use_ft=True,
    pe_accuracy=0.0015,
) -> gym.Env:
    """Create a new environment instance."""
    mujid_config["n_cameras"] = sac_config.n_cameras
    env = mujid_env.MujidEnv(
        config=mujid_config,
    )
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapperSim(
        env,
        config=sac_config,
        grasp_randomisation_z_range=(-pe_accuracy / 3 + 0.001, pe_accuracy / 3 + 0.001)
        if is_eval
        else (-pe_accuracy / 3 + 0.001 - 0.00025, pe_accuracy / 3 + 0.001 + 0.00025),
        grasp_randomisation_x_range=(-pe_accuracy, pe_accuracy)
        if is_eval
        else (-pe_accuracy - 0.00025, pe_accuracy + 0.00025),
        safety_box_radius=2 * pe_accuracy + 0.001 if is_eval else 2 * pe_accuracy,
        goal_position_randomisation_xy_range=(
            -2 * pe_accuracy * 0.9,
            2 * pe_accuracy * 0.9,
        ),
        minimal_start_goal_distance=2 * pe_accuracy,
        step_limit=sac_config.episode_length
        if not is_eval
        else 2 * sac_config.episode_length,
        is_eval=is_eval,
        grasp_randomisation_mode="box",
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination)
    image_keys = [
        "observation.images.wrist_camera_1",
    ]
    if sac_config.n_cameras > 1:
        image_keys.append("observation.images.wrist_camera_2")
    env = DinoImageEncoderWrapper(
        env,
        n_cameras=sac_config.n_cameras,
        image_size=(256, 256),
        crops={k: (0, 256, 0, 256) for k in image_keys},
        image_keys=image_keys,
    )
    # env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    key_ranges_scales = (
        [
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.error.cartesian", (0, 2), 1000.0),
            ("observation.velocity.cartesian", (0, 2), 1000.0),
            ("observation.error.cartesian", (0, 2), 1000.0),
        ]
        + (
            [("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1)]
            if use_ft
            else []
        )
        + [(k.replace("images", "features"), (0, 512), 1.0) for k in image_keys]
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=key_ranges_scales,
    )
    env = NoRotationNoGripperNoZActionClippedWrapperSim(env, clip=0.001)

    return env


def create_simulated_env_3dof(
    mujid_config: dict,
    sac_config: Config = Config(),
    is_eval=False,
    use_ft=True,
    pe_accuracy=0.0015,
) -> gym.Env:
    """Create a new environment instance."""
    mujid_config["n_cameras"] = sac_config.n_cameras
    env = mujid_env.MujidEnv3D(
        config=mujid_config,
    )
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapperSim3D(
        env,
        config=sac_config,
        grasp_randomisation_z_range=(-pe_accuracy / 3 + 0.001, pe_accuracy / 3 + 0.001)
        if is_eval
        else (-pe_accuracy / 3 + 0.001 - 0.00025, pe_accuracy / 3 + 0.001 + 0.00025),
        grasp_randomisation_x_range=(-pe_accuracy, pe_accuracy)
        if is_eval
        else (-pe_accuracy - 0.00025, pe_accuracy + 0.00025),
        safety_box_radius=2 * pe_accuracy + 0.001 if is_eval else 2 * pe_accuracy,
        safety_box_height=2 * pe_accuracy + 0.002,
        goal_position_randomisation_xyz_range=(
            -2 * pe_accuracy * 0.9,
            2 * pe_accuracy * 0.9,
        ),
        minimal_start_goal_distance=2 * pe_accuracy,
        step_limit=sac_config.episode_length
        if not is_eval
        else 2 * sac_config.episode_length,
        is_eval=is_eval,
        grasp_randomisation_mode="box",
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination_3dof)
    image_keys = [
        "observation.images.wrist_camera_1",
    ]
    if sac_config.n_cameras > 1:
        image_keys.append("observation.images.wrist_camera_2")
    env = DinoImageEncoderWrapper(
        env,
        n_cameras=sac_config.n_cameras,
        image_size=(256, 256),
        crops={k: (0, 256, 0, 256) for k in image_keys},
        image_keys=image_keys,
    )
    # env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    key_ranges_scales = (
        [
            ("observation.previous.action", (0, 3), 1000.0),
            ("observation.previous.error.cartesian", (0, 3), 1000.0),
            ("observation.velocity.cartesian", (0, 3), 1000.0),
            ("observation.error.cartesian", (0, 3), 1000.0),
        ]
        + (
            [("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1)]
            if use_ft
            else []
        )
        + [(k.replace("images", "features"), (0, 512), 1.0) for k in image_keys]
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=key_ranges_scales,
    )
    env = NoRotationNoGripperWrapperSim(env)

    return env


def create_simulated_env_3dof_rz(
    mujid_config: dict,
    sac_config: Config = Config(),
    is_eval=False,
    use_ft=True,
    pe_accuracy=0.0015,
    pe_accuracy_angular=np.deg2rad(3),
) -> gym.Env:
    """Create a new environment instance."""
    mujid_config["n_cameras"] = sac_config.n_cameras
    env = mujid_env.MujidEnv5D(
        config=mujid_config,
    )
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapperSim3DoFRotZ(
        env,
        config=sac_config,
        grasp_randomisation_z_range=(-pe_accuracy / 3 + 0.001, pe_accuracy / 3 + 0.001)
        if is_eval
        else (-pe_accuracy / 3 + 0.001 - 0.00025, pe_accuracy / 3 + 0.001 + 0.00025),
        grasp_randomisation_x_range=(-pe_accuracy, pe_accuracy)
        if is_eval
        else (-pe_accuracy - 0.00025, pe_accuracy + 0.00025),
        safety_box_radius=2 * pe_accuracy + 0.001 if is_eval else 2 * pe_accuracy,
        safety_box_angular_radius=np.deg2rad(
            2 * pe_accuracy_angular + 1 if is_eval else 2 * pe_accuracy_angular
        ),
        goal_position_randomisation_xy_range=(
            -2 * pe_accuracy * 0.9,
            2 * pe_accuracy * 0.9,
        ),
        goal_orientation_randomisation_angle=np.deg2rad(2 * pe_accuracy_angular * 0.9),
        minimal_start_goal_distance=2 * pe_accuracy,
        minimal_start_goal_angle=np.deg2rad(2 * pe_accuracy_angular),
        safety_box_angular_step_size=np.deg2rad(1),
        safety_box_step_size=0.0005,
        step_limit=sac_config.episode_length
        if not is_eval
        else 2 * sac_config.episode_length,
        is_eval=is_eval,
        grasp_randomisation_mode="box",
        target_z_error=0.05,
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination)
    image_keys = [
        "observation.images.wrist_camera_1",
    ]
    if sac_config.n_cameras > 1:
        image_keys.append("observation.images.wrist_camera_2")
    env = DinoImageEncoderWrapper(
        env,
        n_cameras=sac_config.n_cameras,
        image_size=(256, 256),
        crops={k: (0, 256, 0, 256) for k in image_keys},
        image_keys=image_keys,
    )
    # env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    # rotations have larger magnitude, ~ 34x, scale them up less; 18-dim.
    key_ranges_scales = (
        [
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.action", (5, 6), 40.0),
            ("observation.velocity.cartesian", (0, 2), 1000.0),
            ("observation.velocity.angular", (2, 3), 40.0),
            ("observation.error.cartesian", (0, 2), 1000.0),
            ("observation.error.angular", (2, 3), 40.0),
            ("observation.previous.error.cartesian", (0, 2), 1000.0),
            ("observation.previous.error.angular", (2, 3), 40.0),
        ]
        + (
            [("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1)]
            if use_ft
            else []
        )
        + [(k.replace("images", "features"), (0, 512), 1.0) for k in image_keys]
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=key_ranges_scales,
    )

    return env


def create_simulated_env_5dof(
    mujid_config: dict,
    sac_config: Config = Config(),
    is_eval=False,
    use_ft=True,
    pe_accuracy=0.0015,
    pe_accuracy_angular=np.deg2rad(3),
) -> gym.Env:
    """Create a new environment instance."""
    mujid_config["n_cameras"] = sac_config.n_cameras
    env = mujid_env.MujidEnv5D(
        config=mujid_config,
    )
    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)
    env = InsertionWrapperSim5DoF(
        env,
        config=sac_config,
        grasp_randomisation_z_range=(-pe_accuracy / 3 + 0.001, pe_accuracy / 3 + 0.001)
        if is_eval
        else (-pe_accuracy / 3 + 0.001 - 0.00025, pe_accuracy / 3 + 0.001 + 0.00025),
        grasp_randomisation_x_range=(-pe_accuracy, pe_accuracy)
        if is_eval
        else (-pe_accuracy - 0.00025, pe_accuracy + 0.00025),
        safety_box_radius=2 * pe_accuracy + 0.001 if is_eval else 2 * pe_accuracy,
        safety_box_angular_radius=np.deg2rad(
            2 * pe_accuracy_angular + 1 if is_eval else 2 * pe_accuracy_angular
        ),
        goal_position_randomisation_xy_range=(
            -2 * pe_accuracy * 0.9,
            2 * pe_accuracy * 0.9,
        ),
        goal_orientation_randomisation_angle=np.deg2rad(2 * pe_accuracy_angular * 0.9),
        minimal_start_goal_distance=2 * pe_accuracy,
        minimal_start_goal_angle=np.deg2rad(2 * pe_accuracy_angular),
        safety_box_angular_step_size=np.deg2rad(1),
        safety_box_step_size=0.0005,
        step_limit=sac_config.episode_length
        if not is_eval
        else 2 * sac_config.episode_length,
        is_eval=is_eval,
        grasp_randomisation_mode="box",
        target_z_error=0.05,
    )
    env = CustomTerminationWrapper(env, termination_fn=custom_sim_termination)
    image_keys = [
        "observation.images.wrist_camera_1",
    ]
    if sac_config.n_cameras > 1:
        image_keys.append("observation.images.wrist_camera_2")
    env = DinoImageEncoderWrapper(
        env,
        n_cameras=sac_config.n_cameras,
        image_size=(256, 256),
        crops={k: (0, 256, 0, 256) for k in image_keys},
        image_keys=image_keys,
    )
    # env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
    assert torch.cuda.is_available(), (
        "CUDA must be available to use ObservationFormatterWrapper"
    )
    # rotations have larger magnitude, ~ 34x, scale them up less; 18-dim.
    key_ranges_scales = (
        [
            ("observation.previous.action", (0, 2), 1000.0),
            ("observation.previous.action", (3, 6), 40.0),
            ("observation.velocity.cartesian", (0, 2), 1000.0),
            ("observation.velocity.angular", (0, 3), 40.0),
            ("observation.error.cartesian", (0, 2), 1000.0),
            ("observation.error.angular", (0, 3), 40.0),
            ("observation.previous.error.cartesian", (0, 2), 1000.0),
            ("observation.previous.error.angular", (0, 3), 40.0),
        ]
        + (
            [("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1)]
            if use_ft
            else []
        )
        + [(k.replace("images", "features"), (0, 512), 1.0) for k in image_keys]
    )
    env = ObservationFormatterWrapper(
        env,
        "cuda",
        keys_ranges_scales=key_ranges_scales,
    )

    return env


def custom_sim_termination_3dof(obs):
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


def custom_sim_termination(obs):
    if obs["observation.state.cartesian"][2] < 0.115:
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
