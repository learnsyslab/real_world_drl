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
    # for environment info
    env_info_queue = ctx.Queue()
    # for policy parameters
    parameters_queue = ctx.Queue()
    # queue for sparse rewards from training CLI
    reward_queue = ctx.Queue()
    # event for training CLI
    continue_training_event = ctx.Event()
    # event to signal episode truncation
    truncation_event = ctx.Event()
    # event to signal the end of an episode to the learner
    episode_is_running = ctx.Event()

    config = RLPD_Config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{config.env_name}__{algo_name}__{timestamp}"

    try:
        if args.cli_training:
            cli = TrainingCLI(continue_training_event, reward_queue, config)
            cli.menu()
        # start actor
        actor_process = ctx.Process(target=launch_actor, args=(args,
                                                               data_queue, 
                                                               env_info_queue,
                                                               parameters_queue,
                                                               run_name,
                                                               continue_training_event,
                                                               truncation_event,
                                                               episode_is_running,
                                                               reward_queue))
        actor_process.start()
        logging.info(f"RLPD actor process started with PID: {actor_process.pid}")

        if not args.eval:
            # start learner
            learner_process = ctx.Process(target=launch_learner, args=(args,
                                                                    data_queue, 
                                                                    env_info_queue,
                                                                    parameters_queue,
                                                                    run_name,
                                                                    episode_is_running))
            learner_process.start()
            logging.info(f"RLPD learner process started with PID: {learner_process.pid}")

        try:
            while True:
                if truncation_event.is_set():
                    if args.cli_training:
                        truncation_event.clear()
                        cli.menu()
                time.sleep(1/30)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating processes...")

    except KeyboardInterrupt:
        logging.info("Closing CLI.")

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


def launch_actor(args, data_queue, env_info_queue, parameters_queue, run_name, continue_training_event, truncation_event, episode_is_running, reward_queue):
    logging.basicConfig(level=logging.INFO)
    try:
        actor = RLPDActor(args, env_info_queue, parameters_queue, run_name, reward_queue)
    except Exception as e:
        logging.error(f"Failed to initialize RLPD Actor: {e}", exc_info=True)
        return

    try:
        actor.run(data_queue, continue_training_event, truncation_event, episode_is_running)
    finally:
        actor.close()
        

def launch_learner(args, data_queue, env_info_queue, parameters_queue, run_name, episode_is_running):
    logging.basicConfig(level=logging.INFO)

    # get environment info from actor node
    action_space = env_info_queue.get()
    observation_space = env_info_queue.get()
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
        learner.run(data_queue, episode_is_running)
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