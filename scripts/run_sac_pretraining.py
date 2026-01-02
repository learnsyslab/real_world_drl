import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser
from gymnasium import spaces

from crisp_drl.agents.sac_rlpd.learner import SACLearner
from crisp_drl.agents.shared.config import Config


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
    args = argparse.parse_args()

    rclpy.init()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        config = Config()
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    learner = SACLearner(
        args,
        parameters_queue=None,
        run_name=run_name,
    )
    learner.pre_train()
    learner.close()


if __name__ == "__main__":
    main()
