import json
import logging
import os
from turtle import fd
import gymnasium as gym
from pyparsing import line
import torch
import torch.multiprocessing as mp

import rclpy
import time
import random
import numpy as np

from crisp_drl.data import utils

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


from crisp_drl.agents.shared.rewards import (
    dense_place_reward,
    sparse_event_reward,
    prune_after_async_termination,
    xy_action_magnitude_dense_reward,
    xy_dense_simple_place_reward,
)
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor, SharedEncoder
from crisp_drl.envs import make_rew


class SACActor:
    def __init__(
        self,
        args,
        config: Config,
        parameters_queue: mp.Queue,
        run_name: str,
        env,
        rew_fn,
    ):
        self.config = config
        self.args = args
        self.run_name = run_name
        self.checkpoint_path = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # setting the seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.parameters_queue = parameters_queue

        # check gym environment
        self.is_gymnasium_env = self.config.env_name in list(gym.envs.registry.keys())  # type: ignore # gym envs registry has no typing

        self.use_camera_inputs = self.config.use_cameras
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )
        self.learning_starts = self.config.learning_starts
        self.env = env
        self.reward_fn = rew_fn

        # checkpoint names of model to be loaded
        self.load_model = args.resume_training or args.load_policy

        # policy
        self.actor = Actor(self.config).to(self.device)
        if self.load_model is not None:
            self.actor.load_state_dict(
                torch.load(f"checkpoints/{self.load_model}/actor_state_dict.pth")
            )

        # image encoders
        self.shared_encoder = SharedEncoder(self.config).to(self.device)
        assert self.load_model is None or args.load_encoder is None, (
            "Either load entire model or only encoder, not both."
        )
        if self.load_model is not None:
            self.shared_encoder.load_state_dict(
                torch.load(
                    f"checkpoints/{self.load_model}/shared_encoder_state_dict.pth"
                )
            )
        elif args.load_encoder is not None:
            assert self.config.n_cameras == 1, (
                "Loading single encoder only works with one camera."
            )
            self.shared_encoder.image_encoders[0].load_state_dict(
                utils.ae_state_dict_from_file(args.load_encoder)
            )
        self.shared_encoder.eval()

        # summary writer for tensorboard
        runs_path = Path(__file__).resolve().parent.parent.parent.parent / "runs_actor"
        runs_path.mkdir(parents=True, exist_ok=True)
        if not self.args.eval:
            self.writer = SummaryWriter(runs_path / run_name)

            self._sync_nodes()  # Wait for the learner to put the initial policy parameters in the queue

        self.global_step = 0
        self.episode_num = 0
        if self.load_model is not None:
            with open(f"checkpoints/{self.load_model}/global_step", "r") as f:
                self.global_step, self.episode_num = map(int, f.read().split(" "))

    def run(self, data_queue: mp.Queue):
        """Main process loop for the SAC actor.
        This method will execture actions in the environment
        and send the s,a,r,s' tuples to the data queue."""

        try:
            # reset the episode variables
            obs, reset_info = self.env.reset(seed=self.config.seed)
            perfect_action = obs["observation.perfect_action"]
            obs = obs["observation.formatted"]
            actual_grasp_pos = reset_info["reset.grasped.position"]

            all_actions = []
            all_rewards = []
            all_observations = [obs]
            all_infos = [reset_info]
            t3 = None
            dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}
            last_episode_rewards = []
            last_episode_successes = []
            all_episode_successes = []

            episode_length = 0
            sum_of_returns = 0.0
            sum_of_squared_returns = 0.0
            self.episode_num = 0

            while self.global_step < self.config.total_timesteps:
                self.global_step += 1
                t0 = time.perf_counter()
                if t3 is not None:
                    dts["out"].append(t0 - t3)
                obs_input = utils.shared_encode(
                    self.shared_encoder,
                    obs.view(1, -1),
                    self.config.shared_encoder_gradient,
                )
                t1 = time.perf_counter()
                action, _ = self.actor(obs_input)
                action = action.view(-1).detach()

                t2 = time.perf_counter()
                all_actions.append(action)
                obs, reward, termination, truncation, info = self.env.step(
                    action.cpu().numpy()
                )
                perfect_action = obs["observation.perfect_action"]
                obs = obs["observation.formatted"]
                all_observations.append(obs)
                all_infos.append(info)
                all_rewards.append(reward)
                done = termination or truncation

                t3_ = time.perf_counter()
                if not done:
                    if t3 is not None:
                        dts["loop"].append(t3_ - t3)
                    t3 = t3_
                    dts["enc"].append(t1 - t0)
                    dts["actor"].append(t2 - t1)
                    dts["step"].append(t3 - t2)

                episode_length += 1

                if done:
                    if "custom_events" in info and "E_ROLLOUT_UNUSABLE" in map(
                        lambda entry: entry[1], info["custom_events"]
                    ):
                        self.global_step -= episode_length
                        obs, reset_info = self.env.reset()
                        perfect_action = obs["observation.perfect_action"]
                        obs = obs["observation.formatted"]
                        all_actions = []
                        all_rewards = []
                        all_observations = [obs]
                        all_infos = [reset_info]
                        actual_grasp_pos = reset_info["reset.grasped.position"]
                        t3 = None
                        dts = {
                            "enc": [],
                            "actor": [],
                            "step": [],
                            "out": [],
                            "loop": [],
                        }
                        continue

                    all_actions, all_observations, all_rewards, all_infos = (
                        self.reward_fn(
                            all_actions,
                            all_observations,
                            all_rewards,
                            all_infos,
                            actual_grasp_pos_xy=actual_grasp_pos[:2],
                        )
                    )

                    episode_successful = "custom_events" in all_infos[
                        -1
                    ] and "E_SUCCESS" in map(
                        lambda x: x[1], all_infos[-1]["custom_events"]
                    )

                    if self.args.eval:
                        # save "reset.grasped.delta", "reset.goal_position.offset", "observation.perfect_action"
                        # in a thread-safe way
                        fd = os.open(
                            os.path.join(self.checkpoint_path, "eval_infos.jsonl"),
                            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                            0o644,
                        )
                        # Use explicit None check — Python's `or` calls
                        # __bool__ on the LHS, which raises on numpy arrays
                        # ("truth value ambiguous").
                        grasp_delta = all_infos[0].get("reset.grasped.delta")
                        if grasp_delta is None:
                            grasp_delta = all_infos[0].get(
                                "reset.grasped.delta_estimated"
                            )
                        line = json.dumps(
                            {
                                "reset.grasped.delta": grasp_delta.tolist() if grasp_delta is not None else None,
                                "reset.goal_position.offset": all_infos[0][
                                    "reset.goal_position.offset"
                                ].tolist(),
                                "observation.perfect_action": perfect_action.tolist(),
                                "episode_successful": episode_successful,
                                "length": episode_length,
                                "datetime": time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime()
                                ),
                            }
                        )
                        os.write(fd, (line + "\n").encode())
                        os.close(fd)
                    if episode_successful:
                        last_episode_successes.append(1)
                        all_episode_successes.append(1)
                    else:
                        last_episode_successes.append(0)
                        all_episode_successes.append(0)
                    if len(last_episode_successes) > 20:
                        last_episode_successes.pop(0)

                    data_queue.put(
                        (all_actions, all_observations, all_rewards, termination)
                    )
                    if termination:
                        print(f"Episode {self.episode_num} terminated.")
                    elif truncation:
                        print(f"Episode {self.episode_num} truncated.")
                    self.episode_num += 1
                    if self.episode_num >= self.args.max_episodes > 0:
                        print("Reached maximum number of episodes. Stopping actor.")
                        print(
                            f"Mean success rate: {np.mean(all_episode_successes) * 100:.2f} %"
                        )
                        break
                    episode_return = sum(all_rewards)

                    last_episode_rewards.append(episode_return)
                    if len(last_episode_rewards) > 10:
                        last_episode_rewards.pop(0)
                    # print(all_infos)

                    sum_of_returns = sum(last_episode_rewards)
                    sum_of_squared_returns = sum(
                        map(lambda x: x**2, last_episode_rewards)
                    )
                    reward_window_size = len(last_episode_rewards)
                    eps_return_std = (
                        sum_of_squared_returns / reward_window_size
                        - (sum_of_returns / reward_window_size) ** 2
                        + 1e-10
                    ) ** 0.5

                    if not self.args.eval:
                        self.writer.add_scalar(
                            "charts/episodic_return", episode_return, self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", episode_length, self.global_step
                        )

                        self.writer.add_scalar(
                            "charts/avg_return",
                            sum_of_returns / reward_window_size,
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/eps_return_std", eps_return_std, self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/rollout_success",
                            sum(last_episode_successes) / len(last_episode_successes),
                            self.episode_num,
                        )

                        self.writer.add_scalar(
                            "charts/t_encoding", np.mean(dts["enc"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_encoding_std",
                            np.std(dts["enc"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_actor", np.mean(dts["actor"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_actor_std",
                            np.std(dts["actor"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_stepping", np.mean(dts["step"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_stepping_std",
                            np.std(dts["step"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_outside", np.mean(dts["out"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_outside_std",
                            np.std(dts["out"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_loop", np.mean(dts["loop"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_loop_std", np.std(dts["loop"]), self.global_step
                        )

                    # reset the episode variables
                    logging.info(f"Episode length: {episode_length}")
                    episode_length = 0

                    obs, reset_info = self.env.reset()
                    perfect_action = obs["observation.perfect_action"]
                    obs = obs["observation.formatted"]
                    actual_grasp_pos = reset_info["reset.grasped.position"]

                    all_actions = []
                    all_rewards = []
                    all_observations = [obs]
                    all_infos = [reset_info]
                    t3 = None
                    dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}

                    if not self.args.eval:
                        logging.info(
                            "Waiting for the learner to finish gradient updates..."
                        )
                        self.save_tb_additional_info()
                        self._sync_nodes()
                        logging.info("Learner fininshed gradient updates.")
                    logging.info(
                        f"success? {last_episode_successes[-1] == 1}; n_rollouts: {len(last_episode_successes)}; last success rate: {np.mean(last_episode_successes)}; all success rate: {np.mean(all_episode_successes)}"
                    )

        except SystemExit:
            logging.info("[run fnc] Quit Training request received. Terminating actor process...")
        except KeyboardInterrupt:
            logging.info("[run fnc] Keyboard interrupt received. Terminating actor process...")
        except Exception as e:
            logging.error(f"[run fnc] An error occurred in the SAC Actor: {e}", exc_info=True)
        finally:
            self.close()

    def run_pe(self, data_queue: mp.Queue):
        """Main process loop for the SAC actor.
        This method will execture actions in the environment"""
        print(f"[SACActor] Starting run_pe")

        try:
            # reset the episode variables
            obs, reset_info = self.env.reset(seed=self.config.seed)
            obs = obs["observation.formatted"]
            actual_grasp_pos = reset_info["reset.grasped.position"]

            all_actions = []
            all_rewards = []
            all_observations = [obs]
            all_infos = [reset_info]
            t3 = None
            dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}
            last_episode_rewards = []
            last_episode_successes = []
            all_episode_successes = []

            episode_length = 0
            sum_of_returns = 0.0
            sum_of_squared_returns = 0.0
            self.episode_num = 0

            while self.global_step < self.config.total_timesteps:
                self.global_step += 1
                t0 = time.perf_counter()
                if t3 is not None:
                    dts["out"].append(t0 - t3)
                obs_input = utils.shared_encode(
                    self.shared_encoder,
                    obs.view(1, -1),
                    self.config.shared_encoder_gradient,
                )
                t1 = time.perf_counter()
                action, _ = self.actor(obs_input)
                action = action.view(-1).detach()

                t2 = time.perf_counter()
                all_actions.append(action)
                obs, reward, termination, truncation, info = self.env.step(
                    action.cpu().numpy()
                )
                obs = obs["observation.formatted"]
                all_observations.append(obs)
                all_infos.append(info)
                all_rewards.append(reward)
                done = termination or truncation

                t3_ = time.perf_counter()
                if not done:
                    if t3 is not None:
                        dts["loop"].append(t3_ - t3)
                    t3 = t3_
                    dts["enc"].append(t1 - t0)
                    dts["actor"].append(t2 - t1)
                    dts["step"].append(t3 - t2)

                episode_length += 1

                if done:
                    if "custom_events" in info and "E_ROLLOUT_UNUSABLE" in map(
                        lambda entry: entry[1], info["custom_events"]
                    ):
                        self.global_step -= episode_length
                        obs, reset_info = self.env.reset()
                        obs = obs["observation.formatted"]
                        all_actions = []
                        all_rewards = []
                        all_observations = [obs]
                        all_infos = [reset_info]
                        actual_grasp_pos = reset_info["reset.grasped.position"]
                        t3 = None
                        dts = {
                            "enc": [],
                            "actor": [],
                            "step": [],
                            "out": [],
                            "loop": [],
                        }
                        continue

                    all_actions, all_observations, all_rewards, all_infos = (
                        self.reward_fn(
                            all_actions,
                            all_observations,
                            all_rewards,
                            all_infos,
                            actual_grasp_pos_xy=actual_grasp_pos[:2],
                        )
                    )

                    episode_successful = "custom_events" in all_infos[
                        -1
                    ] and "E_SUCCESS" in map(
                        lambda x: x[1], all_infos[-1]["custom_events"]
                    )

                    if self.args.eval:
                        # save "reset.grasped.delta", "reset.goal_position.offset", "observation.perfect_action"
                        # in a thread-safe way
                        fd = os.open(
                            os.path.join(self.checkpoint_path, "eval_infos.jsonl"),
                            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                            0o644,
                        )
                        line = json.dumps(
                            {
                                "reset.grasped.delta_estimated": all_infos[0][
                                    "reset.grasped.delta_estimated"
                                ].tolist(),
                                # "reset.goal_position.offset": all_infos[0][
                                #     "reset.goal_position.offset"
                                # ].tolist(),
                                "episode_successful": episode_successful,
                                "length": episode_length,
                                "datetime": time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime()
                                ),
                            }
                        )
                        os.write(fd, (line + "\n").encode())
                        os.close(fd)
                    if episode_successful:
                        last_episode_successes.append(1)
                        all_episode_successes.append(1)
                    else:
                        last_episode_successes.append(0)
                        all_episode_successes.append(0)
                    if len(last_episode_successes) > 20:
                        last_episode_successes.pop(0)

                    data_queue.put(
                        (all_actions, all_observations, all_rewards, termination)
                    )
                    if termination:
                        print(f"Episode {self.episode_num} terminated.")
                    elif truncation:
                        print(f"Episode {self.episode_num} truncated.")
                    self.episode_num += 1
                    if self.episode_num >= self.args.max_episodes > 0:
                        print("Reached maximum number of episodes. Stopping actor.")
                        print(
                            f"Mean success rate: {np.mean(all_episode_successes) * 100:.2f} %"
                        )
                        break
                    episode_return = sum(all_rewards)

                    last_episode_rewards.append(episode_return)
                    if len(last_episode_rewards) > 10:
                        last_episode_rewards.pop(0)
                    # print(all_infos)

                    sum_of_returns = sum(last_episode_rewards)
                    sum_of_squared_returns = sum(
                        map(lambda x: x**2, last_episode_rewards)
                    )
                    reward_window_size = len(last_episode_rewards)
                    eps_return_std = (
                        sum_of_squared_returns / reward_window_size
                        - (sum_of_returns / reward_window_size) ** 2
                        + 1e-10
                    ) ** 0.5

                    if not self.args.eval:
                        self.writer.add_scalar(
                            "charts/episodic_return", episode_return, self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", episode_length, self.global_step
                        )

                        self.writer.add_scalar(
                            "charts/avg_return",
                            sum_of_returns / reward_window_size,
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/eps_return_std", eps_return_std, self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/rollout_success",
                            sum(last_episode_successes) / len(last_episode_successes),
                            self.episode_num,
                        )

                        self.writer.add_scalar(
                            "charts/t_encoding", np.mean(dts["enc"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_encoding_std",
                            np.std(dts["enc"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_actor", np.mean(dts["actor"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_actor_std",
                            np.std(dts["actor"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_stepping", np.mean(dts["step"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_stepping_std",
                            np.std(dts["step"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_outside", np.mean(dts["out"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_outside_std",
                            np.std(dts["out"]),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/t_loop", np.mean(dts["loop"]), self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/t_loop_std", np.std(dts["loop"]), self.global_step
                        )

                    # reset the episode variables
                    logging.info(f"Episode length: {episode_length}")
                    episode_length = 0

                    obs, reset_info = self.env.reset()
                    obs = obs["observation.formatted"]
                    actual_grasp_pos = reset_info["reset.grasped.position"]

                    all_actions = []
                    all_rewards = []
                    all_observations = [obs]
                    all_infos = [reset_info]
                    t3 = None
                    dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}

                    if not self.args.eval:
                        logging.info(
                            "Waiting for the learner to finish gradient updates..."
                        )
                        self.save_tb_additional_info()
                        self._sync_nodes()
                        logging.info("Learner fininshed gradient updates.")
                    logging.info(
                        f"success? {last_episode_successes[-1] == 1}; n_rollouts: {len(last_episode_successes)}; last success rate: {np.mean(last_episode_successes)}; all success rate: {np.mean(all_episode_successes)}"
                    )

        except SystemExit:
            logging.info("[run_pe fnc]Quit Training request received. Terminating actor process...")
        except KeyboardInterrupt:
            logging.info("[run_pe fnc] Keyboard interrupt received. Terminating actor process...")
        except Exception as e:
            logging.error(f"[run_pe fnc] An error occurred in the SAC Actor: {e}", exc_info=True)
        finally:
            self.close()

    def save_tb_additional_info(self):
        with open(os.path.join(self.checkpoint_path, "global_step"), "w") as f:
            f.write(f"{self.global_step} {self.episode_num}")

    def close(self):
        logging.info("Executing SAC Actor closing behavior...")
        if not self.args.eval:
            self.save_tb_additional_info()
            self.writer.close()
        # Clean up the environment
        self.env.reset(
            options={"last_reset": True}
        )  # reset to a known state before closing
        self.env.close()
        if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
            rclpy.shutdown()

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        policy_parameters, shared_encoder_parameters, shared_encoder_bn = (
            self.parameters_queue.get()
        )
        self.actor.load_state_dict(policy_parameters)
        self.shared_encoder.load_state_dict(shared_encoder_parameters)
        self.shared_encoder.load_batchnorm_stats(shared_encoder_bn)

        logging.info(
            "Actor received updated parameters for the policy and vision encoder."
        )
