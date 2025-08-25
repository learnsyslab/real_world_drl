import torch.multiprocessing as mp
import time
import rclpy
import logging

from rlmb.agents.sac.actor import SACActor
from rlmb.agents.sac.learner import SACLearner
from rlmb.agents.sac.config import SAC_Config

def launch_processes():
    rclpy.init()
    ctx = mp.get_context("spawn")

    # for s, a, r, ns experience from the environment
    data_queue = ctx.Queue()
    # for environment info
    env_info_queue = ctx.Queue()
    # for policy parameters
    parameters_queue = ctx.Queue()

    config = SAC_Config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.env_name}__{config.exp_name}__{timestamp}"

    try:
        # start actor
        actor_process = ctx.Process(target=launch_actor, args=(data_queue, 
                                                               env_info_queue,
                                                               parameters_queue,
                                                               run_name))
        actor_process.start()
        logging.info(f"RLPD actor process started with PID: {actor_process.pid}")

        # start learner
        learner_process = ctx.Process(target=launch_learner, args=(data_queue, 
                                                                   env_info_queue,
                                                                   parameters_queue,
                                                                   run_name))
        learner_process.start()
        logging.info(f"RLPD learner process started with PID: {learner_process.pid}")

        try:
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating processes...")

    finally:
        actor_process.join(timeout=2)
        learner_process.join(timeout=2)

        if rclpy.ok():
            rclpy.shutdown()

        if actor_process.is_alive():
            logging.info(f"Trying to force terminating Actor Process.")
            actor_process.terminate()
            actor_process.join()
        logging.info("RLPD actor process terminated succesfully.")

        if learner_process.is_alive():
            logging.info(f"Trying to force terminating Learner Process with PID: {learner_process.pid}.")
            learner_process.terminate()
            learner_process.join()
        logging.info("RLPD learner process terminated succesfully.")
        logging.info("All nodes terminated.")


def launch_actor(data_queue, env_info_queue, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)
    try:
        actor = SACActor(env_info_queue, parameters_queue, run_name)
    except Exception as e:
        logging.error(f"Failed to initialize SAC Actor: {e}", exc_info=True)
        return

    try:
        actor.run(data_queue)
    finally:
        actor.close()
        

def launch_learner(data_queue, env_info_queue, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)

    # get environment info from actor node
    action_space = env_info_queue.get()
    observation_space = env_info_queue.get()
    try:
        learner = SACLearner(action_space, 
                             observation_space,
                             parameters_queue,
                             run_name)
    except Exception as e:
        logging.info(f"Failed to initialize SAC Learner: {e}", exc_info=True)
        return

    try:
        learner.run(data_queue)
    finally:
        learner.close()


def main():
    logging.basicConfig(level=logging.INFO)
    launch_processes()


if __name__ == "__main__":
    main()