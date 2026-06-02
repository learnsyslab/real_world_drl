import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser

from crisp_drl.agents.shared.actor import SACActor
from crisp_drl.agents.sac_rlpd.learner import SACLearner
from crisp_drl.agents.shared.algorithm_config import Config

import signal
from contextlib import contextmanager

from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig, SiemensConfig
from crisp_drl.agents.shared.insertion_wrapper import install_stop_handler
from crisp_drl.envs import make_env, make_rew

ROLLOUT_LENGTH = 150
ROLLOUT_LENGTH_EVAL = 30


def _override_config_for_task(args, config):
    """Apply per-task config overrides.

    Note: ``launch_processes`` re-instantiates ``Config()`` (existing pattern),
    so per-task overrides must be re-applied inside ``launch_actor`` /
    ``launch_learner`` after the spawn boundary as well as in ``main``.
    """
    if args is None:
        return config
    task = getattr(args, "task", "siemens")
    if task in ("lego_3dof_rz", "lego_3dof_rz_pe"):
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = np.array(
            [
                0.00025,
                0.00025,
                np.deg2rad(0.5),
            ]
        )
    if getattr(args, "use_6dof_grasp", False):
        if task != "siemens":
            raise ValueError(
                "--use_6dof_grasp is currently only supported for --task siemens."
            )
        config.actor_output_dim = 5
        config.actor_nonvision_input_dim = 26
        config.max_action = np.array(
            [
                0.00025,
                0.00025,
                np.deg2rad(0.25),
                np.deg2rad(0.25),
                np.deg2rad(0.25),
            ]
        )
    return config


@contextmanager
def no_interrupts():
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)


def launch_processes(args, config):
    rclpy.init()
    ctx = mp.get_context("spawn")

    # for s, a, r, ns experience from the environment
    data_queue = ctx.Queue()
    # for policy parameters
    parameters_queue = ctx.Queue()

    config = Config()
    config = _override_config_for_task(args, config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    # start actor
    actor_process = ctx.Process(
        target=launch_actor,
        args=(
            args,
            config,
            data_queue,
            parameters_queue,
            run_name,
        ),
    )
    actor_process.start()
    logging.info(f"RLPD actor process started with PID: {actor_process.pid}")

    if not args.eval:
        # start learner
        learner_process = ctx.Process(
            target=launch_learner,
            args=(args, config, data_queue, parameters_queue, run_name),
        )
        learner_process.start()
        logging.info(f"RLPD learner process started with PID: {learner_process.pid}")
    try:
        time.sleep(100000000)

    except KeyboardInterrupt:
        logging.info("Ending Processes")

    finally:
        with no_interrupts():
            if "actor_process" in locals():
                actor_process.join(timeout=10)
                if not args.eval:
                    learner_process.join(timeout=60)  # pyright: ignore[reportPossiblyUnboundVariable]

                if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
                    rclpy.shutdown()

                if actor_process.is_alive():
                    logging.info("Trying to force terminating Actor Process.")
                    actor_process.terminate()
                    actor_process.join()
                logging.info("SAC actor process terminated succesfully.")

                if not args.eval:
                    if learner_process.is_alive():  # pyright: ignore[reportPossiblyUnboundVariable]
                        logging.info(
                            f"Trying to force terminating Learner Process with PID: {learner_process.pid}."  # pyright: ignore[reportPossiblyUnboundVariable]
                        )
                        learner_process.terminate()  # pyright: ignore[reportPossiblyUnboundVariable]
                        learner_process.join()  # pyright: ignore[reportPossiblyUnboundVariable]
                    logging.info("SAC learner process terminated succesfully.")
                logging.info("All nodes terminated.")


def launch_actor(
    args,
    config,
    data_queue,
    parameters_queue,
    run_name,
):
    logging.basicConfig(level=logging.INFO)
    try:
        # env = make_env.create_simulated_env(
        #     {
        #         "initial_keyframe": 2,
        #         "lego_shift_range": (-0.002, 0.002),
        #         "initial_position_range": (
        #             np.array([-1.0, -1.0, -2.0]) * 1e-3,  # np.zeros(3),
        #             np.array([1.0, 1.0, -1.9]) * 1e-3,  # np.zeros(3),
        #         ),
        #         "live_view": False,
        #     }
        # )
        task = getattr(args, "task", "siemens") if args is not None else "siemens"
        if task == "lego_3dof_rz":
            env = make_env.create_real_env_v4_3dof_rz(config, args=args)
        elif task == "lego_3dof_rz_pe":
            env = make_env.create_real_env_v4_3dof_rz_pe(config, args=args)
        elif task == "lego":
            env = make_env.create_real_env_v4(config, args=args)
        elif task in ("b1", "b1_pe"):
            from crisp_drl.agents.shared.siemens_config_demo import (
                COMBINED_FIRST_PE_POSITION,
            )
            env_config = ShelfBoxConfig(
                episode_length=ROLLOUT_LENGTH if not args.eval else ROLLOUT_LENGTH_EVAL
            )
            env_config.custom_first_home_position = COMBINED_FIRST_PE_POSITION.copy()
            env_config.custom_home_position_pe = COMBINED_FIRST_PE_POSITION.copy()
            env_config.custom_home_position = COMBINED_FIRST_PE_POSITION.copy()
            env = (
                make_env.create_real_env_b1(config, env_config=env_config, args=args)
                if not args or not args.use_pose_estimation
                else make_env.create_real_env_b1_pe(
                    alg_config=config, env_config=env_config, args=args
                )
            )
        else:
            env = (
                make_env.create_real_env_s1(
                    config, env_config=SiemensConfig(), args=args
                )
                if not args or not args.use_pose_estimation
                else make_env.create_real_env_s1_pe(
                    alg_config=config, env_config=SiemensConfig(), args=args
                )
            )
        # rclpy overwrites Python's SIGINT handler on init; reinstall so that
        # Ctrl+C raises KeyboardInterrupt inside movement loops.
        install_stop_handler()
        # rew_fn = make_rew.create_sim_reward_fn(  # noqa: F821
        #     self.config,
        #     ideal_goal_pos_xy=np.array([0.6, 0.0]),
        #     ideal_grasp_pos_xy=np.array([0.0, 0.0]),
        #     event_reward_map={
        #         "E_SUCCESS": 0.1 / (1 - self.config.gamma) * 3,
        #         "E_FAIL": -0.1 / (1 - self.config.gamma) / 2 * 3,
        #         "E_SAFETY_BOX_VIOLATION": 0.0,  # -0.05,
        #     },
        # )
        rew_fn = make_rew.create_real_reward_fn(
            config,
            event_reward_map={
                "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
                "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
                "E_SAFETY_BOX_VIOLATION": 0.0,  # -0.05,
                "E_ROLLOUT_UNUSABLE": -100,
                "E_CONTROLLER_ISSUE": -100,
                "E_TORQUE": -100,
            },
        )
        actor = SACActor(args, config, parameters_queue, run_name, env, rew_fn)
    except Exception as e:
        logging.error(f"Failed to initialize SAC Actor: {e}", exc_info=True)
        return

    try:
        task = getattr(args, "task", "siemens") if args is not None else "siemens"
        use_pe = (args and args.use_pose_estimation) or task == "lego_3dof_rz_pe"
        if use_pe:
            actor.run_pe(data_queue)
        else:
            actor.run(data_queue)
    finally:
        actor.close()


def launch_learner(args, config, data_queue, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)

    try:
        learner = SACLearner(
            args,
            config=config,
            parameters_queue=parameters_queue,
            run_name=run_name,
        )
    except Exception as e:
        logging.info(f"Failed to initialize SAC Learner: {e}", exc_info=True)
        return

    try:
        learner.run(data_queue)
    finally:
        learner.close()


def main():
    logging.basicConfig(level=logging.INFO)
    argparse = ArgumentParser()
    argparse.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Set the checkpoint name for the experiment. Per default the timestamp is used.",
    )
    argparse.add_argument(
        "--pre_train",
        type=str,
        default=None,
        help="Set the path to a pre-train buffer.",
    )
    argparse.add_argument(
        "--expert_buffer_path",
        type=str,
        default=None,
        help="Path to the prerecorded expert buffer.",
    )
    argparse.add_argument(
        "--load_policy", type=str, help="Checkpoint name of policy to be loaded."
    )
    argparse.add_argument(
        "--resume_training",
        type=str,
        help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.",
    )
    argparse.add_argument(
        "--cli_training",
        action="store_true",
        help="Run training loop with interactive command line interface.",
    )
    argparse.add_argument(
        "--eval", action="store_true", help="Run model in evaluation mode."
    )
    argparse.add_argument(
        "--use_pose_estimation",
        action="store_true",
        help="Use pose estimation module in the environment.",
    )
    argparse.add_argument(
        "--pe_align_gripper",
        action="store_true",
        help="If set, align gripper yaw (Z) to PE-detected brick yaw during reset (3DoF only).",
    )
    # argument for maximum number of episodes to run
    argparse.add_argument(
        "--max_episodes",
        type=int,
        default=1000000,
        help="Maximum number of episodes to run.",
    )
    argparse.add_argument(
        "--load_encoder", type=str, help="Checkpoint name of encoder to be loaded."
    )
    argparse.add_argument(
        "--success_threshold",
        type=float,
        default=9.3,
        help="Threshold on mean Q(s, pi(s)) for appending E_SUCCESS and terminating.",
    )
    argparse.add_argument(
        "--task",
        type=str,
        choices=["siemens", "lego", "lego_3dof_rz", "lego_3dof_rz_pe", "b1", "b1_pe"],
        default="siemens",
        help="Which task/env builder to use. 'siemens' keeps the current default "
        "(create_real_env_s1[_pe]). 'lego' switches to create_real_env_v4. "
        "'lego_3dof_rz' uses create_real_env_v4_3dof_rz (XY + yaw, 3-D action). "
        "'lego_3dof_rz_pe' uses create_real_env_v4_3dof_rz_pe (same action space "
        "but with PE-driven grasp and goal from FoundationPose).",
    )
    argparse.add_argument(
        "--no_ft_sensor",
        action="store_true",
        help="Run without the BOTA force-torque sensor. Uses my_env_v4_no_ft.yaml "
        "(empty sensor_configs), injects zero FT features so the trained policy's "
        "obs shape is preserved, and disables the FT-driven axis controller in "
        "the active InsertionWrapper (no contact-establishing phase; substituted "
        "axis action fixed at 0). Works with both --task lego and --task siemens.",
    )
    argparse.add_argument(
        "--no_ft_success_threshold",
        type=float,
        default=4.0,
        help="Classifier success threshold used when --no_ft_sensor is active "
        "(siemens PE path only). Lower than --success_threshold because zero "
        "FT features push Q(s, pi(s)) estimates out of the trained distribution. "
        "Ignored unless --no_ft_sensor is set.",
    )
    argparse.add_argument(
        "--use_6dof_grasp",
        action="store_true",
        help="Enable the 6DoF grasp path for the Siemens real environment. "
        "The grasp motion keeps roll/pitch/yaw targets and pose estimation stops "
        "forcing a fixed orientation.",
    )
    argparse.add_argument(
        "--pose_viz_dir",
        type=str,
        default=None,
        help="If set, save per-reset pose-estimation overlays (PNG) + raw NPZ "
        "artifacts into this directory for offline inspection.",
    )
    argparse.add_argument(
        "--snap_reinforce",
        action="store_true",
        help="After the snap-push hold, execute a reinforce cycle: open gripper → "
        "lift 6mm → close → press down → lift → open → re-grasp lego → lift. "
        "Helps confirm full LEGO seating before reset.",
    )
    argparse.add_argument(
        "--3dof",
        action="store_true",
        dest="pe_3dof",
        help="PE 3DOF mode: flat-lego assumption. Wide PE gives coarse X-Y hover; "
        "close-up PE refines X-Y and extracts yaw; goal computed from target brick "
        "X-Y + yaw only (Z from fixed offset). Gripper roll/pitch unchanged throughout.",
    )
    argparse.add_argument(
        "--gt_target_goal",
        action="store_true",
        help="Use ground-truth goal position and zero yaw for the target (yellow) brick "
        "instead of the PE estimate. Useful for isolating grasp-side PE errors.",
    )
    argparse.add_argument(
        "--grasp_z_offset",
        type=float,
        default=None,
        help="Additive Z offset on the final grasp height for the 3DOF PE branch "
        "(metres; positive = gripper closes higher above the table). "
        "If omitted, defaults to 2 mm in --eval and 0 mm otherwise.",
    )
    args = argparse.parse_args()
    config = Config()
    config = _override_config_for_task(args, config)
    launch_processes(args, config)


if __name__ == "__main__":
    main()
