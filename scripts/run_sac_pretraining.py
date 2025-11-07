import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser
from gymnasium import spaces

from crisp_drl.agents.sac.actor_linus import SACActor
from crisp_drl.agents.sac.learner_linus import SACLearner
from crisp_drl.agents.sac.config import SAC_Config

import signal
from contextlib import contextmanager

@contextmanager
def no_interrupts():
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)


def launch_processes(args):
    rclpy.init()
    ctx = mp.get_context("spawn")

    config = SAC_Config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    try:
        # start learner
        learner_process = ctx.Process(target=launch_learner, args=(args,
                                                                None,
                                                                run_name))
        learner_process.start()
        logging.info(f"SAC learner process started with PID: {learner_process.pid}")

        time.sleep(100000000)

    except KeyboardInterrupt:
        logging.info("Ending Processes")

    finally:
        with no_interrupts():

            learner_process.join(timeout=60)
            if rclpy.ok():
                rclpy.shutdown()

            if learner_process.is_alive():
                logging.info(f"Trying to force terminating Learner Process with PID: {learner_process.pid}.")
                learner_process.terminate()
                learner_process.join()
            logging.info("SAC learner process terminated succesfully.")

def launch_learner(args, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)
    action_space = spaces.Box(-np.inf, np.inf, (2,))
    observation_space_networks = spaces.Box(-np.inf, np.inf, (27,))
    observation_space_buffers = spaces.Box(-np.inf, np.inf, (523,))

    try:
        learner = SACLearner(args,
                             action_space, 
                             observation_space_networks=observation_space_networks, observation_space_buffers=observation_space_buffers,
                             parameters_queue=parameters_queue, run_name=run_name, n_cameras=1)
    except Exception as e:
        logging.info(f"Failed to initialize SAC Learner: {e}", exc_info=True)
        return

    try:
        learner.pre_train()
    except Exception as e:
        logging.error(f"An error occurred in the SAC Learner: {e}", exc_info=True)
    finally:
        learner.close()


def main():
    logging.basicConfig(level=logging.INFO)
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--pre_train", type=str, default=None, help="Set the path to a pre-train buffer.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    argparse.add_argument("--cli_training", action="store_true", help="Run training loop with interactive command line interface.")
    argparse.add_argument("--eval", action="store_true", help="Run model in evaluation mode.")
    args = argparse.parse_args()
    launch_processes(args)


if __name__ == "__main__":
    main()