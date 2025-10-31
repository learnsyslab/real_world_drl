import functools
import pickle
import numpy as np
import torch.multiprocessing as mp
import time
import rclpy
import logging
import os

from argparse import ArgumentParser

from crisp_drl.agents.rlpd.actor import RLPDActor
from crisp_drl.agents.rlpd.env_wrappers import ActionTimeStampWrapper, BelowZTerminationWrapper, CLIWrapper, DictObservationToInfoMover, ContainerWatcherWrapper, FarAwayTerminationWrapper, ImageEncoderWrapper, InsertionResetWrapper, LastObservationWrapper, ObservationFormatterWrapper, PrintCartesianInfoWrapper, TimeMeasurementWrapper, VideoWrapper, observation_has_z_pressure
from crisp_drl.agents.rlpd.learner import RLPDLearner
from crisp_drl.agents.rlpd.config import RLPD_Config
from crisp_drl.agents.rlpd.rewards import place_reward, prune_after_async_termination
from crisp_gym.manipulator_env import make_env
from crisp_gym.util.rl_utils import load_actions_safe

# ...existing code...
import rclpy
import traceback
import sys

# Wrap rclpy.shutdown to print a stack trace when it's called so you can find the caller
# _original_rclpy_shutdown = getattr(rclpy, "shutdown", None)


# def _log_and_call_rclpy_shutdown(*args, **kwargs):
#     print("rclpy.shutdown() called — stack trace (most recent call last):", file=sys.stderr, flush=True)
#     traceback.print_stack(file=sys.stderr)
#     if _original_rclpy_shutdown:
#         return _original_rclpy_shutdown(*args, **kwargs)

# rclpy.shutdown = _log_and_call_rclpy_shutdown


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
        

class NaiveToGoalPositionPolicy:
    def __init__(self, goal_position, step_size=0.001):
        self.i = 0
        self.goal_position = goal_position
        self.step_size = step_size

    def next(self, obs):
        # if xy close, go directly to goal, otherwise go in xy direction
        current_pos = obs['observation.state.cartesian'][:3]
        delta = self.goal_position - current_pos
        action = np.zeros(7)
        norm_xy = np.linalg.norm(delta[:2])
        action[:2] = delta[:2] * min(self.step_size / norm_xy, 1.0)
        if norm_xy <= 0.005:
            action[2] = delta[2] * min(self.step_size / 2 / abs(delta[2]), 1.0)
        return action

def launch_actor(args, run_name,):
    logging.basicConfig(level=logging.INFO)
    pass

def main():
    logging.basicConfig(level=logging.INFO)
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    argparse.add_argument("--cli_training", action="store_true", help="Run training loop with interactive command line interface.")
    argparse.add_argument("--eval", action="store_true", help="Run model in evaluation mode.")
    args = argparse.parse_args()
    alg_name = "PreProgrammedPolicy"

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{alg_name}__{time.strftime('%Y%m%d-%H%M%S')}"

    os.mkdir(f"rollouts/{run_name}")
    run_number = 0

    # launch_processes(args)
    env = make_env("my_env")
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    env = InsertionResetWrapper(env, initial_pos=np.array([0.200, -0.020, -0.200]), grasp_randomization_bounds=(np.array([-0.003, -0.003, -0.001]), np.array([0.003, 0.003, 0.0015])), 
                                insert_randomization_bounds=(np.array([-0.01, -0.01, 0.0]), np.array([0.01, 0.01, 0.005])), action_sequence_to_grasp=load_actions_safe("v3_go_to_pick.json"), action_sequence_after_grasp=load_actions_safe("v3_after_pick.json"))
    env = ActionTimeStampWrapper(env)
    # env = PrintCartesianInfoWrapper(env)
    env = LastObservationWrapper(env)
    env = BelowZTerminationWrapper(env, min_z=0.0475)
    env = FarAwayTerminationWrapper(env, approximate_goal_pos=np.array([539.75, -33,  50]) * 0.001, max_distance=0.055)
    env = ContainerWatcherWrapper(env, ctx=mp.get_context("spawn"))

    env = CLIWrapper(env, termination_fn = functools.partial(observation_has_z_pressure, error_threshold=0.005, previous_error_threshold=0.003, min_z_height=0.055))
    env = VideoWrapper(env, video_dir=f"recordings/{run_name}", camera_keys = ["observation.images.wrist_camera", "observation.images.side_camera"], fps=15)

    env = ImageEncoderWrapper(env, n_cameras=2, image_size=(256, 256))
    env = DictObservationToInfoMover(env)
    # env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0), ('observation.previous.action', (6,7), 20.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), ('observation.velocity.gripper', (0, 1), 20.0),
    #                                         ('observation.error.gripper', (0, 1), 20.0), ('observation.state.gripper', (0, 1), 1.0), ('observation.target.gripper', (0, 1), 1.0), ('observation.images.wrist_camera', (0, 512), 1.0), ('observation.images.side_camera', (0, 512), 1.0)])
    env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0), ('observation.previous.error.cartesian', (0,3), 10.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), 
                                                               ('observation.images.wrist_camera', (0, 512), 1.0), ('observation.images.side_camera', (0, 512), 1.0)]) # 268 or 1036

    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    args = argparse.parse_args()



    # insert_trajectory = load_actions_safe("v4_put_down_closed.json")

    # action_source = PlaybackActionSource([np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0])] * 150)

    # Actor loop: collect rollout, compute/truncate rewards, send to learner and check if new weights from learner are available (if not collect next rollout)
    #   actor class everything in eval mode; actor has one ffnn per camera; one shared for all features
    # Learner loop: wait for rollout, add to buffer, do up to utd-x samples (every n check for new rollout, every N send new weights)


    while True:
        obs, info = env.reset()

        reset_action = -info["reset.randomize.insert"]
        actual_grasp_pos = info["reset.grasped.position"]
        # action_source = PlaybackActionSource(np.concatenate(([reset_action / 10] * 10, insert_trajectory), axis=0))

        # compute actual goal position based on where the object was grasped
        goal_position = np.array([0.541, -0.034,  0.0435])
        ideal_grasp_pos = np.array([0.58833, -0.13817,  0.04229])
        goal_position[0] += actual_grasp_pos[0] - ideal_grasp_pos[0] 
        goal_position[2] += actual_grasp_pos[2] - ideal_grasp_pos[2]
        print(f"actual grasped pos: {(actual_grasp_pos-ideal_grasp_pos) * 1000} mm")
        policy = NaiveToGoalPositionPolicy(goal_position=goal_position, step_size=0.001)


        all_actions = []
        action_timestamps = []
        all_observations = [obs]
        all_infos = []
        # collect rollout
        while True:
            # step environment
            action = policy.next(info["observation"])
            if action is None:
                print("Out of actions.")
                break

            all_actions.append(action)
            action_timestamps.append(time.time())
            try:
                # print(action)
                obs, _reward, terminated, truncated, info = env.step(action, block=True)
                # print(f"Error: {obs[7:10] * 1000} mm")
            except Exception as e:
                print(e)
                print("Interrupted")
                env.close()
                sys.exit(0)
            all_observations.append(obs)
            all_infos.append(info)
            
            if truncated or terminated: 
                break

        if truncated:
            print("Truncated, continuing.")
            continue
        all_actions, all_observations, all_infos = prune_after_async_termination(all_actions, all_observations, all_infos, {"E_CONTROLLER_ISSUE", "E_TORQUE"})
        all_rewards = place_reward(all_actions, all_observations, all_infos, {"E_TORQUE": -10.0, "E_FAR_AWAY": -3.0, "E_BELOW_Z": -3.0, "E_SUCCESS": 10.0, "E_FAIL": -1.0, "E_BAD_BEHAVIOR": -5.0, "E_CONTROLLER_ISSUE": 0.0})
        if all_rewards[-1] > 0:
            print(f"Got reward {all_rewards[-1]}, saving run")
            # Save data for complete rollout
            with open(f"rollouts/{run_name}/{run_number}.pkl", "wb") as f:
                pickle.dump({"actions": all_actions, "rewards": all_rewards, "observations": all_observations, "infos": all_infos, "actual_grasped_pos": actual_grasp_pos}, f)
                run_number += 1
        else:
            print(f"Got negative reward {all_rewards[-1]}, skipping run")





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