import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser
from gymnasium import spaces

from crisp_drl.agents.sac_rlpd.learner import SACLearner
from crisp_drl.agents.shared.algorithm_config import Config


def _override_config_for_task(args, config):
    if args is None:
        return config
    task = getattr(args, "task", None)
    if task == "lego_3dof_rz":
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = np.array([0.00025, 0.00025, np.deg2rad(0.5)])
    elif task == "lego":
        pass  # default dims apply
    elif task == "siemens":
        pass  # default dims apply


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
        "--load_encoder", type=str, help="Checkpoint name of encoder to be loaded."
    )
    argparse.add_argument(
        "--seed", type=int, default=None, help="Random seed for training."
    )
    argparse.add_argument(
        "--utd_ratio", type=float, default=None, help="Random seed for training."
    )
    argparse.add_argument(
        "--n_cameras", type=int, default=None, help="Random seed for training."
    )
    argparse.add_argument(
        "--actor_nonvision_input_dim",
        type=int,
        default=None,
        help="Non-vision observation dimension for the actor network.",
    )
    argparse.add_argument(
        "--actor_output_dim",
        type=int,
        default=None,
        help="Action dimension for the actor network (e.g. 2 for XY, 3 for lego_3dof_rz, 5 for 6dof).",
    )
    argparse.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["siemens", "lego", "lego_3dof_rz"],
        help="Apply per-task config overrides (dims, max_action). Mirrors run_sac.py --task.",
    )
    args = argparse.parse_args()

    rclpy.init()
    kwargs = {}
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.utd_ratio is not None:
        kwargs["utd_ratio"] = args.utd_ratio
    if args.n_cameras is not None:
        kwargs["n_cameras"] = args.n_cameras
    if args.actor_nonvision_input_dim is not None:
        kwargs["actor_nonvision_input_dim"] = args.actor_nonvision_input_dim
    if args.actor_output_dim is not None:
        kwargs["actor_output_dim"] = args.actor_output_dim
    config = Config(**kwargs)
    _override_config_for_task(args, config)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    learner = SACLearner(
        args,
        config,
        parameters_queue=None,  # type: ignore
        run_name=run_name,
    )
    learner.pre_train()
    learner.close()


if __name__ == "__main__":
    main()
