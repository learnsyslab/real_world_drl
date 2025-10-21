import pickle
import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser

from rlmb.agents.rlpd.actor import RLPDActor
from rlmb.agents.rlpd.env_wrappers import ActionTimeStampWrapper, CLIWrapper, ContainerWatcherWrapper, FarAwayTerminationWrapper, ImageEncoderWrapper, InsertionResetWrapper, LastObservationWrapper, ObservationConcatWrapper, TimeMeasurementWrapper
from rlmb.agents.rlpd.learner import RLPDLearner
from rlmb.agents.rlpd.config import RLPD_Config
from rlmb.training.training_cli import TrainingCLI
from crisp_gym.manipulator_env import make_env
from crisp_gym.util.rl_utils import load_actions_safe

# ...existing code...
import rclpy
import traceback
import sys

# Wrap rclpy.shutdown to print a stack trace when it's called so you can find the caller
_original_rclpy_shutdown = getattr(rclpy, "shutdown", None)


def _log_and_call_rclpy_shutdown(*args, **kwargs):
    print("rclpy.shutdown() called — stack trace (most recent call last):", file=sys.stderr, flush=True)
    traceback.print_stack(file=sys.stderr)
    if _original_rclpy_shutdown:
        return _original_rclpy_shutdown(*args, **kwargs)

rclpy.shutdown = _log_and_call_rclpy_shutdown


def launch_processes(args):
    rclpy.init()
    ctx = mp.get_context("spawn")


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
            
                                                               run_name))
        actor_process.start()
        logging.info(f"RLPD actor process started with PID: {actor_process.pid}")


    except KeyboardInterrupt:
        logging.info("Ending Processes")
    except Exception as e:
        print(e)

    finally:
        if "actor_process" in locals():
            actor_process.join(timeout=2)


            if rclpy.ok():
                rclpy.shutdown()

            if actor_process.is_alive():
                logging.info(f"Trying to force terminating Actor Process.")
                actor_process.terminate()
                actor_process.join()
            logging.info("RLPD actor process terminated succesfully.")

            logging.info("All nodes terminated.")


class PlaybackActionSource:
    def __init__(self, actions):
        self.i = 0
        self.actions = actions

    def reset(self):
        self.i = 0

    def next(self, _obs):
        if self.i < len(self.actions):
            self.i += 1
            return self.actions[self.i-1]

def launch_actor(args, run_name,):
    logging.basicConfig(level=logging.INFO)
    env = make_env("my_env")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = ImageEncoderWrapper(env)
    env = LastObservationWrapper(env)
    env = ObservationConcatWrapper(env, key_ranges=[('observation.previous.action', (0,3)), ('observation.previous.action', (6,7)), ('observation.velocity.cartesian', (None, None)), ('observation.error.cartesian', (None, None)), ('observation.velocity.gripper', (None, None)),
                                                ('observation.error.gripper', (None, None)), ('observation.state.gripper', (None, None)), ('observation.target.gripper', (None, None)), ('observation.images.wrist_camera', (None, None)), ('observation.images.side_camera', (None, None))])
    env = InsertionResetWrapper(env, initial_pos=np.array([0.200, -0.020, -0.200]), grasp_randomization_bounds=(np.array([-0.005, -0.005, -0.002]), np.array([0.005, 0.005, 0.002])), 
                                insert_randomization_bounds=(np.array([-0.03, -0.03, -0.003]), np.array([0.03, 0.03, 0.003])), action_sequence=load_actions_safe("v3_pick_up.json"))
    env = ActionTimeStampWrapper(env)
    env = FarAwayTerminationWrapper(env, approximate_goal_pos=np.array([593.94, -34.18,  53.97]) * 0.001, max_distance=0.1)
    env = ContainerWatcherWrapper(env, ctx=mp.get_context("spawn"))
    env = CLIWrapper(env, gripper_threshold=0.33)

    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    args = argparse.parse_args()

    alg_name = "PreProgrammedPolicy"

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{alg_name}__{time.strftime('%Y%m%d-%H%M%S')}"

    os.mkdir(f"rollouts/{run_name}")
    run_number = 0


    action_source = PlaybackActionSource(load_actions_safe("v2_crafted_rel_acts_insert_plus32.json"))

    
    while True:
        obs, info = env.reset()
        action_source.reset()


        played_actions = []
        action_timestamps = []
        all_observations = [obs]
        all_infos = [info]
        # collect rollout
        while True:
            # step environment
            action = action_source.next(obs)
            if action is None:
                print("Out of actions.")
                break

            played_actions.append(action)
            action_timestamps.append(time.time())
            obs, _reward, terminated, truncated, info = env.step(action)
            all_observations.append(obs)
            all_infos.append(info)
            
            if truncated: 
                break


        # Save data for complete rollout
        with open(f"rollouts/{run_name}/act_{run_number}.pkl", "wb") as f:
            pickle.dump(played_actions, f)
        with open(f"rollouts/{run_name}/tmp_{run_number}.pkl", "wb") as f:
            pickle.dump(action_timestamps, f)
        with open(f"rollouts/{run_name}/obs_{run_number}.pkl", "wb") as f:
            pickle.dump(all_observations, f)
        with open(f"rollouts/{run_name}/inf_{run_number}.pkl", "wb") as f:
            pickle.dump(all_infos, f)
            run_number += 1





def main():
    logging.basicConfig(level=logging.INFO)
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    argparse.add_argument("--cli_training", action="store_true", help="Run training loop with interactive command line interface.")
    argparse.add_argument("--eval", action="store_true", help="Run model in evaluation mode.")
    args = argparse.parse_args()
    # launch_processes(args)
    env = make_env("my_env")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = TimeMeasurementWrapper(env, 0)
    env = InsertionResetWrapper(env, initial_pos=np.array([0.200, -0.020, -0.200]), grasp_randomization_bounds=(np.array([-0.005, -0.005, -0.002]), np.array([0.005, 0.005, 0.002])), 
                                insert_randomization_bounds=(np.array([-0.03, -0.03, -0.003]), np.array([0.03, 0.03, 0.003])), action_sequence=[]) # load_actions_safe("v3_pick_up.json")
    env = TimeMeasurementWrapper(env, 1)
    env = CLIWrapper(env, gripper_threshold=1.0) # 0.33
    env = TimeMeasurementWrapper(env, 2)
    env = ImageEncoderWrapper(env)
    env = TimeMeasurementWrapper(env, 3)
    env = LastObservationWrapper(env)
    env = TimeMeasurementWrapper(env, 4)

    env = ActionTimeStampWrapper(env)
    env = TimeMeasurementWrapper(env, 4)
    # env = FarAwayTerminationWrapper(env, approximate_goal_pos=np.array([593.94, -34.18,  53.97]) * 0.001, max_distance=0.1)
    # env = TimeMeasurementWrapper(env, 5)
    env = ContainerWatcherWrapper(env, ctx=mp.get_context("spawn"))
    env = TimeMeasurementWrapper(env, 5)

    env = ObservationConcatWrapper(env, key_ranges=[('observation.previous.action', (0,3)), ('observation.previous.action', (6,7)), ('observation.velocity.cartesian', (None, None)), ('observation.error.cartesian', (None, None)), ('observation.velocity.gripper', (None, None)),
                                                ('observation.error.gripper', (None, None)), ('observation.state.gripper', (None, None)), ('observation.target.gripper', (None, None)), ('observation.images.wrist_camera', (None, None)), ('observation.images.side_camera', (None, None))])
    env = TimeMeasurementWrapper(env, 6)

    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    args = argparse.parse_args()

    alg_name = "PreProgrammedPolicy"

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{alg_name}__{time.strftime('%Y%m%d-%H%M%S')}"

    os.mkdir(f"rollouts/{run_name}")
    run_number = 0

    # load_actions_safe("v2_crafted_rel_acts_insert_plus32.json")

    action_source = PlaybackActionSource([np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0])] * 150)

    
    while True:
        obs, info = env.reset()
        action_source.reset()


        played_actions = []
        action_timestamps = []
        all_observations = [obs]
        all_infos = [info]
        # collect rollout
        while True:
            # step environment
            action = action_source.next(obs)
            if action is None:
                print("Out of actions.")
                break

            played_actions.append(action)
            action_timestamps.append(time.time())
            try:
                obs, _reward, terminated, truncated, info = env.step(action, block=True)
            except Exception as e:
                print(e)
                print("Interrupted")
                env.close()
                sys.exit(0)
            all_observations.append(obs)
            all_infos.append(info)
            
            if truncated: 
                break


        # Save data for complete rollout
        with open(f"rollouts/{run_name}/act_{run_number}.pkl", "wb") as f:
            pickle.dump(played_actions, f)
        with open(f"rollouts/{run_name}/tmp_{run_number}.pkl", "wb") as f:
            pickle.dump(action_timestamps, f)
        with open(f"rollouts/{run_name}/obs_{run_number}.pkl", "wb") as f:
            pickle.dump(all_observations, f)
        with open(f"rollouts/{run_name}/inf_{run_number}.pkl", "wb") as f:
            pickle.dump(all_infos, f)
            run_number += 1



if __name__ == "__main__":
    main()

# ts = np.sort(ts)  # just in case
# dts_s = np.diff(ts)        # seconds
# dts_ms = dts_s * 1000.0    # milliseconds

# # Basic stats
# n = dts_ms.size
# mean_ms = dts_ms.mean()
# median_ms = np.median(dts_ms)
# min_ms = dts_ms.min()
# max_ms = dts_ms.max()

# print(f"dts (ms) count={n}, mean={mean_ms:.6f} ms, median={median_ms:.6f} ms, min={min_ms:.6f} ms, max={max_ms:.6f} ms")
# print("dts (ms) array:", np.round(dts_ms, 6))

# # Plot histogram
# plt.figure(figsize=(7,4))
# plt.hist(dts_ms, bins=30, color='C0', alpha=0.8, edgecolor='k')
# plt.axvline(mean_ms, color='C1', linestyle='--', linewidth=1.5, label=f"mean {mean_ms:.3f} ms")
# plt.axvline(median_ms, color='C2', linestyle=':', linewidth=1.5, label=f"median {median_ms:.3f} ms")
# plt.xlabel("dt (ms)")
# plt.ylabel("count")
# plt.title("Histogram of inter-timestamp dts")
# plt.legend()
# plt.grid(axis='y', alpha=0.25)
# plt.tight_layout()
# plt.show()