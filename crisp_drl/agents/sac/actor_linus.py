import functools
import logging
import os
import gymnasium as gym
import torch
import torch.multiprocessing as mp
import multiprocessing
import rclpy
import time
import random
import numpy as np

from crisp_gym.manipulator_env_config import NoCamFrankaEnvConfig, FrankaEnvConfig
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env
from crisp_gym.util.rl_utils import load_actions_safe
from crisp_gym.config.home import home_close_to_table
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from copy import deepcopy


from crisp_drl.agents.rlpd.rewards import (
    dense_place_reward,
    sparse_place_reward,
    prune_after_async_termination,
    xy_action_magnitude_dense_reward,
    xy_dense_place_reward,
    xy_dense_simple_place_reward,
)
from crisp_drl.agents.sac.config import SAC_Config
from crisp_drl.agents.sac.networks_cleanrl import Actor
from crisp_drl.agents.rlpd.env_wrappers import (
    ActionTimeStampWrapper,
    BelowZTerminationWrapper,
    CLIWrapper,
    DictObservationToInfoMover,
    ContainerWatcherWrapper,
    FarAwayTerminationWrapper,
    ImageEncoderWrapper,
    InsertionResetWrapper,
    LastObservationWrapper,
    NaiveToGoalPositionWrapper,
    NoRotationActionWrapper,
    NoRotationNoGripperActionWrapper,
    NoRotationNoGripperNoZActionWrapper,
    ObservationFormatterWrapper,
    TimeMeasurementWrapper,
    observation_has_z_pressure,
    observation_has_z_pressure_or_below,
)
from crisp_drl.data.utils import crisp_batch_concat_obs_to_tensor, crisp_obs_to_tensor
from crisp_drl.training.training_cli import clear_terminal


class SACActor:
    def __init__(self, args, parameters_queue: mp.Queue, run_name: str, env):
        self.config = SAC_Config()
        self.args = args

        # setting the seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.parameters_queue = parameters_queue

        # check gym environment
        self.is_gymnasium_env = self.config.env_name in list(gym.envs.registry.keys())

        self.use_camera_inputs = self.config.use_cameras
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )
        self.update_policy_after = self.config.update_policy_after
        self.learning_starts = self.config.learning_starts
        self.env = env

        # checkpoint names of model to be loaded
        self.load_model = args.resume_training or args.load_policy

        # policy
        self.actor = Actor(self.env.action_space, self.config).to(self.device)
        if self.load_model is not None:
            self.actor.load_state_dict(
                torch.load(f"checkpoints/{self.load_model}/actor_state_dict.pth")
            )

        # image encoders
        if self.use_camera_inputs:
            self.image_encoders = [
                torch.nn.Sequential(
                    torch.nn.Linear(self.config.vision_head_input_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, self.config.vision_head_output_dim),
                ).to(self.device)
                for _ in range(len(self.env.cameras))
            ]
            if self.load_model is not None:
                for i, image_encoder in enumerate(self.image_encoders):
                    image_encoder.load_state_dict(
                        torch.load(
                            f"checkpoints/{self.load_model}/image_encoder_{i}_state_dict.pth"
                        )
                    )
        else:
            self.image_encoders = []

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
            actual_grasp_pos = reset_info["reset.grasped.position"]

            all_actions = []
            all_observations = [obs]
            all_infos = [reset_info]
            t3 = None
            dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}
            last_episode_rewards = []
            last_episode_successes = []

            episode_length = 0
            sum_of_returns = 0.0
            sum_of_squared_returns = 0.0
            self.episode_num = 0

            while self.global_step < self.config.total_timesteps:
                self.global_step += 1
                t0 = time.perf_counter()
                if t3 is not None:
                    dts["out"].append(t0 - t3)
                obs_input = crisp_batch_concat_obs_to_tensor(
                    obs.view(1, -1),
                    self.image_encoders,
                    self.config.vision_head_input_dim,
                )
                t1 = time.perf_counter()
                action, _, _ = self.actor.get_action(obs_input)
                action = action.view(-1).detach()

                t2 = time.perf_counter()
                all_actions.append(action)
                obs, _reward, termination, truncation, info = self.env.step(
                    action.cpu().numpy(), block=True
                )
                all_observations.append(obs)
                all_infos.append(info)
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
                        all_actions = []
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
                    if not self.args.eval:
                        all_actions, all_observations, all_infos = (
                            prune_after_async_termination(
                                all_actions,
                                all_observations,
                                all_infos,
                                {"E_CONTROLLER_ISSUE", "E_TORQUE"},
                            )
                        )
                        # all_rewards = dense_place_reward(all_actions, all_observations, all_infos, {"E_TORQUE": -10.0, "E_FAR_AWAY": -3.0, "E_BELOW_Z": -3.0, "E_SUCCESS": 10.0, "E_FAIL": -1.0, "E_BAD_BEHAVIOR": -5.0, "E_CONTROLLER_ISSUE": 0.0},
                        #              max_rew=0.01, ideal_goal_pos=[0.53975, -0.033,  0.05], ideal_grasp_pos=np.array([0.58833, -0.13817,  0.04229]), actual_grasp_pos=actual_grasp_pos, k_xy=0.005, k_z=0.002)
                        # all_rewards = sparse_place_reward(
                        #     all_actions, all_observations, all_infos
                        # )
                        # all_rewards = xy_dense_place_reward(all_actions, all_observations, all_infos, max_rew=0.01, max_action_magnitude=0.0008,
                        #                                      ideal_goal_pos_xy=np.array([0.53975, -0.033]), ideal_grasp_pos_xy=np.array([0.58833, -0.13817]),
                        #                                      actual_grasp_pos_xy=actual_grasp_pos[:2])
                        all_rewards = xy_dense_simple_place_reward(
                            all_actions,
                            all_observations,
                            all_infos,
                            max_rew=0.1,
                            gamma=self.config.gamma,
                            ideal_goal_pos_xy=np.array([0.6, 0.0]),
                            ideal_grasp_pos_xy=np.array([0.0, 0.0]),
                            actual_grasp_pos_xy=actual_grasp_pos[:2],
                            event_reward_map={
                                "E_SUCCESS": 0.1 / (1 - self.config.gamma) / 2 * 3,
                                "E_FAIL": -0.1 / (1 - self.config.gamma) / 2 * 3,
                            },
                        )
                        all_rewards = xy_action_magnitude_dense_reward(
                            all_rewards,
                            all_actions,
                            threshold=0.000251,
                            reward=-0.05,
                        )

                        data_queue.put(
                            (all_observations, all_actions, all_rewards, termination)
                        )

                        self.episode_num += 1
                        episode_return = sum(all_rewards)

                        self.writer.add_scalar(
                            "charts/episodic_return", episode_return, self.global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", episode_length, self.global_step
                        )
                        last_episode_rewards.append(episode_return)
                        if len(last_episode_rewards) > 10:
                            last_episode_rewards.pop(0)
                        # print(all_infos)
                        if "custom_events" in all_infos[-1] and "E_SUCCESS" in map(
                            lambda x: x[1], all_infos[-1]["custom_events"]
                        ):
                            last_episode_successes.append(1)
                        else:
                            last_episode_successes.append(0)
                        if len(last_episode_successes) > 20:
                            last_episode_successes.pop(0)

                        sum_of_returns = sum(last_episode_rewards)
                        sum_of_squared_returns = sum(
                            map(lambda x: x**2, last_episode_rewards)
                        )
                        reward_window_size = len(last_episode_rewards)
                        eps_return_std = (
                            sum_of_squared_returns / reward_window_size
                            - (sum_of_returns / reward_window_size) ** 2
                        ) ** 0.5
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
                    actual_grasp_pos = reset_info["reset.grasped.position"]

                    all_actions = []
                    all_observations = [obs]
                    all_infos = [reset_info]
                    t3 = None
                    dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}

                    if not self.args.eval:
                        logging.info(
                            "Waiting for the learner to finish gradient updates..."
                        )
                        self._sync_nodes()
                        logging.info("Learner fininshed gradient updates.")

        except SystemExit:
            logging.info("Quit Training request received. Terminating actor process...")
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating actor process...")
        except Exception as e:
            logging.error(f"An error occurred in the SAC Actor: {e}", exc_info=True)
        finally:
            self.close()

    def close(self):
        logging.info("Executing SAC Actor closing behavior...")
        if not self.args.eval:
            self.writer.close()
            self.run_name = os.path.basename(self.writer.log_dir)
            self.checkpoint_path = os.path.join("checkpoints", self.run_name)
            with open(os.path.join(self.checkpoint_path, "global_step"), "w") as f:
                f.write(f"{self.global_step} {self.episode_num}")
        # Clean up the environment
        self.env.close()
        if rclpy.ok():
            rclpy.shutdown()

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        policy_parameters, proj_head_parameters = self.parameters_queue.get()
        self.actor.load_state_dict(policy_parameters)
        if self.use_camera_inputs:
            for i, params in enumerate(proj_head_parameters):
                self.image_encoders[i].load_state_dict(params)

        logging.info(
            "Actor received updated parameters for the policy and vision encoder."
        )
