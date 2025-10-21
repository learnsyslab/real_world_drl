import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser

from crisp_drl.agents.rlpd.actor import RLPDActor
from crisp_drl.agents.rlpd.learner import RLPDLearner
from crisp_drl.agents.rlpd.config import RLPD_Config
from crisp_drl.training.training_cli import TrainingCLI


def launch_processes(args):
    rclpy.init()
    ctx = mp.get_context("spawn")

    # for s, a, r, ns experience from the environment
    data_queue = ctx.Queue()
    # for policy parameters
    parameters_queue = ctx.Queue()

    config = RLPD_Config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    try:
        # start actor
        actor_process = ctx.Process(target=launch_actor, args=(args,
                                                               data_queue, 
                                                               parameters_queue,
                                                               run_name))
        actor_process.start()
        logging.info(f"RLPD actor process started with PID: {actor_process.pid}")

        if not args.eval:
            # start learner
            learner_process = ctx.Process(target=launch_learner, args=(args,
                                                                    data_queue, 
                                                                    parameters_queue,
                                                                    run_name))
            learner_process.start()
            logging.info(f"RLPD learner process started with PID: {learner_process.pid}")


    except KeyboardInterrupt:
        logging.info("Ending Processes")

    finally:
        if "actor_process" in locals():
            actor_process.join(timeout=2)
            if not args.eval:
                learner_process.join(timeout=2)

            if rclpy.ok():
                rclpy.shutdown()

            if actor_process.is_alive():
                logging.info(f"Trying to force terminating Actor Process.")
                actor_process.terminate()
                actor_process.join()
            logging.info("RLPD actor process terminated succesfully.")

            if not args.eval:
                if learner_process.is_alive():
                    logging.info(f"Trying to force terminating Learner Process with PID: {learner_process.pid}.")
                    learner_process.terminate()
                    learner_process.join()
                logging.info("RLPD learner process terminated succesfully.")
            logging.info("All nodes terminated.")


def launch_actor(args, data_queue, parameters_queue, run_name,):
    logging.basicConfig(level=logging.INFO)
    try:
        actor = RLPDActor(args, parameters_queue, run_name)
    except Exception as e:
        logging.error(f"Failed to initialize RLPD Actor: {e}", exc_info=True)
        return

    try:
        actor.run(data_queue)
    finally:
        actor.close()
        

def launch_learner(args, data_queue, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)

    # get environment info from actor node TODO
    action_space = 4
    observation_space = 523
    try:
        learner = RLPDLearner(args,
                             action_space, 
                             observation_space,
                             parameters_queue,
                             run_name)
    except Exception as e:
        logging.info(f"Failed to initialize RLPD Learner: {e}", exc_info=True)
        return

    try:
        learner.run(data_queue)
    finally:
        learner.close()


def main():
    logging.basicConfig(level=logging.INFO)
    argparse = ArgumentParser()
    argparse.add_argument("expert_buffer_path", type=str, help="Path to the prerecorded expert buffer.")
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    argparse.add_argument("--cli_training", action="store_true", help="Run training loop with interactive command line interface.")
    argparse.add_argument("--eval", action="store_true", help="Run model in evaluation mode.")
    args = argparse.parse_args()
    launch_processes(args)


if __name__ == "__main__":
    main()


# Launch Learner node with data channels

# Launch Actor node with data channels
# create actor with learnable layers
# Create env, call env.wait_until_ready()
# Repeatedly run env until done; delete episode or truncate, compute rewards, send to learner
# cli wrapper: after reset wait for ready; during/after rollout handle cli events


# TODO: Collect expert data
# TODO: Fix buffer implementation