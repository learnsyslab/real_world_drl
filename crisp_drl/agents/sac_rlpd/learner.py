import joblib
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import random
import numpy as np
import time

from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from crisp_drl.data import utils
from crisp_drl.data import buffers_cleanrl
from crisp_drl.data.buffers_cleanrl import (
    ReplayBufferGpu,
)
from crisp_drl.agents.shared.config import Config
from crisp_drl.agents.shared.networks_cleanrl import SharedEncoder, SoftQNetwork, Actor


class SACLearner:
    def __init__(
        self,
        args,
        action_space,
        parameters_queue: mp.Queue,
        run_name: str,
    ):
        # Store the configuration parameters
        self.config = Config()
        assert int(self.config.update_policy_after * self.config.utd_ratio) > 0, (
            "Invalid combination of update_policy_after and utd_ratio. "
            "Ensure that int(update_policy_after * utd_ratio) > 0."
        )
        self.args = args

        # check gym environment
        self.use_camera_inputs = self.config.use_cameras

        # set seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.action_space = action_space
        self.parameters_queue = parameters_queue

        # summary writer for tensorboard
        runs_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "runs_learner"
        )
        runs_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(runs_path / run_name)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.config).items()]
                )
            ),
        )

        # checkpoint names of model to be loaded
        self.load_model = args.resume_training or args.load_policy

        # initializing networks
        self.actor = Actor(action_space, self.config).to(self.device)
        obs_dim_networks = (
            self.config.actor_nonvision_input_dim
            + self.config.vision_head_output_dim * self.config.n_cameras
        )
        self.q_networks = [
            SoftQNetwork(obs_dim_networks, self.action_space).to(self.device)
            for _ in range(self.config.num_critics)
        ]
        self.q_target_networks = [
            SoftQNetwork(obs_dim_networks, self.action_space).to(self.device)
            for _ in range(self.config.num_critics)
        ]

        # load model if specified
        if self.load_model is not None:
            self.actor.load_state_dict(
                torch.load(f"checkpoints/{self.load_model}/actor_state_dict.pth")
            )
            for idx in range(self.config.num_critics):
                self.q_networks[idx].load_state_dict(
                    torch.load(
                        f"checkpoints/{self.load_model}/qf{idx + 1}_state_dict.pth"
                    )
                )
                self.q_target_networks[idx].load_state_dict(
                    torch.load(
                        f"checkpoints/{self.load_model}/qf{idx + 1}_target_state_dict.pth"
                    )
                )
            logging.info(f"Loaded model from checkpoints/{self.load_model}")
        else:
            for idx, target_net in enumerate(self.q_target_networks):
                target_net.load_state_dict(self.q_networks[idx].state_dict())

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

        # optimizers
        if self.config.shared_encoder_gradient:
            self.shared_encoder_optimizer = optim.Adam(
                self.shared_encoder.parameters(), lr=self.config.policy_lr
            )
            if args.resume_training:
                self.shared_encoder_optimizer.load_state_dict(
                    torch.load(
                        f"checkpoints/{self.load_model}/shared_encoder_optimizer_state_dict.pth"
                    )
                )
        q_params = []
        for q_net in self.q_networks:
            q_params += list(q_net.parameters())
        self.q_optimizer = optim.Adam(q_params, lr=self.config.q_lr)
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.config.policy_lr
        )
        if args.resume_training:
            self.q_optimizer.load_state_dict(
                torch.load(f"checkpoints/{self.load_model}/q_optimizer_state_dict.pth")
            )
            self.actor_optimizer.load_state_dict(
                torch.load(
                    f"checkpoints/{self.load_model}/actor_optimizer_state_dict.pth"
                )
            )

        if self.args.pre_train is None:
            self._sync_nodes()

        # initializing the replay buffer
        if self.args.resume_training is not None:
            self.replay_buffer = joblib.load(
                f"checkpoints/{self.args.resume_training}/replay_buffer.joblib"
            )
            logging.info(
                f"Loaded replay buffer from checkpoints/{self.args.resume_training}/replay_buffer.joblib"
            )
        elif self.args.pre_train is not None:
            self.replay_buffer = joblib.load(self.args.pre_train)
            logging.info(f"Loaded pre-train buffer from {self.args.pre_train}")
        else:
            self.replay_buffer = ReplayBufferGpu(
                buffer_size=self.config.buffer_size,
                observation_dim=self.config.actor_nonvision_input_dim
                + self.config.vision_head_input_dim * self.config.n_cameras,
                action_space=action_space,
                device=self.device,
                n_step_return=self.config.n_step_return,
                gamma=self.config.gamma,
            )

        # Automatic entropy tuning
        if self.config.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_space.shape).to(self.device)
            ).item()
            if self.load_model is not None:
                self.log_alpha = (
                    torch.load(f"checkpoints/{self.load_model}/log_alpha.pth")
                    .to(self.device)
                    .requires_grad_(True)
                )
            else:
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.config.q_lr)
            if args.resume_training:
                self.a_optimizer.load_state_dict(
                    torch.load(
                        f"checkpoints/{self.load_model}/a_optimizer_state_dict.pth"
                    )
                )

        else:
            self.alpha = self.config.alpha

        self.current_training_step = 0
        self.episode_num = 0
        self.global_step = 0

        if args.resume_training is True:
            with open(f"checkpoints/{self.load_model}/current_training_step", "r") as f:
                self.current_training_step = int(f.read())
            with open(f"checkpoints/{self.load_model}/global_step", "r") as f:
                self.global_step, self.episode_num = map(int, f.read().split(" "))

        if self.args.expert_buffer_path is not None:
            self.expert_buffer = joblib.load(self.args.expert_buffer_path)
            logging.info(f"Loaded expert buffer from {self.args.expert_buffer_path}")
        else:
            self.expert_buffer = None

        # training checkpoint path
        self.run_name = os.path.basename(self.writer.log_dir)
        self.checkpoint_path = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def pre_train(self):
        """Main process loop for the SAC learner."""
        try:
            logging.info(
                f"Learner starts training... (total steps: {int(self.replay_buffer.size() * self.config.utd_ratio)})"
            )
            for self.current_training_step in range(
                int(self.replay_buffer.size() * self.config.utd_ratio)
            ):
                self._train_step(self.writer)
                if (self.current_training_step + 1) % self.replay_buffer.size() == 0:
                    logging.info(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} Pass {(self.current_training_step + 1) // self.replay_buffer.size()} through the pre-train buffer completed."
                    )
            print("Learner finished training.")

        except SystemExit:
            logging.info("Quit training request received. Closing learner process...")
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating learner process...")
        except Exception as e:
            logging.error(f"An error occurred in the RLPD Learner: {e}", exc_info=True)
        finally:
            self.close()

    def run(self, data_queue: mp.Queue):
        """Main process loop for the SAC learner."""
        try:
            global_time_step = 0
            _current_training_step = 0

            while self.current_training_step < self.config.total_timesteps:
                while data_queue.empty():
                    time.sleep(0.1)
                print("Learner starts training...")
                episode_len = 0
                while not data_queue.empty():
                    actions, observations, rewards, termination = data_queue.get()
                    episode_len = len(actions)
                    self.replay_buffer.add_rollout(
                        obs=observations,
                        action=actions,
                        reward=rewards,
                        terminated=termination,
                    )
                    self.episode_num += 1
                    self.global_step += episode_len
                if self.replay_buffer.size() < self.config.learning_starts:
                    print(
                        f"Not enough data in replay buffer yet, skipping training: {self.replay_buffer.size()} < {self.config.learning_starts}"
                    )
                    self._sync_nodes()
                    continue

                for _ in range(int(episode_len * self.config.utd_ratio)):
                    _current_training_step += 1.0 / self.config.utd_ratio
                    self.current_training_step = int(_current_training_step)
                    self._train_step(self.writer)
                    global_time_step += 1
                print("Learner finished training.")

                self._sync_nodes()

        except SystemExit:
            logging.info("Quit training request received. Closing learner process...")
            self.close()
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating learner process...")
            self.close()
        except Exception as e:
            self.close()
            logging.error(f"An error occurred in the RLPD Learner: {e}", exc_info=True)

    def _train_step(self, writer):
        """Perform a single training step using data from the replay buffer.
        This updates both the critics and the policy networks."""
        if self.expert_buffer is not None:  # Do RLPD training
            online_data = self.replay_buffer.sample(self.config.batch_size // 2)
            expert_data = self.expert_buffer.sample(self.config.batch_size // 2)

            observations = torch.cat(
                (online_data.observations, expert_data.observations), dim=0
            )
            actions = torch.cat((online_data.actions, expert_data.actions), dim=0)
            rewards = torch.cat((online_data.rewards, expert_data.rewards), dim=0)
            next_observations = torch.cat(
                (online_data.next_observations, expert_data.next_observations), dim=0
            )
            dones = torch.cat((online_data.dones, expert_data.dones), dim=0)
            data = buffers_cleanrl.ReplayBufferSamples(
                observations, actions, next_observations, dones, rewards
            )
        else:  # Standard SAC training
            data = self.replay_buffer.sample(self.config.batch_size)
        with torch.no_grad():
            next_obs = self.shared_encoder(data.next_observations)
            next_state_actions, next_state_log_pis = self.actor(next_obs)
            # pick two random Q-networks from the ensemble
            ensemble_samples = random.sample(
                range(self.config.num_critics), self.config.critic_subset_size
            )
            qf_next_targets = [
                self.q_target_networks[i](next_obs, next_state_actions).view(-1)
                for i in ensemble_samples
            ]
            min_qf_next_targets = torch.min(torch.stack(qf_next_targets), dim=0)[
                0
            ] - self.alpha * next_state_log_pis.view(-1)
            next_q_values = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * self.config.gamma * (min_qf_next_targets).view(-1)

        obs = utils.shared_encode(
            self.shared_encoder, data.observations, self.config.shared_encoder_gradient
        )

        qf_a_values = [
            self.q_networks[i](obs, data.actions).view(-1)
            for i in range(self.config.num_critics)
        ]
        qf_losses = [
            F.mse_loss(qf_a_values[i], next_q_values)
            for i in range(self.config.num_critics)
        ]
        qf_loss = sum(qf_losses)

        # optimize the q functions, shared encoder and policy
        if self.config.shared_encoder_gradient:
            self.shared_encoder_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        if self.config.shared_encoder_gradient:
            # retain graph for shared encoder update
            qf_loss.backward(retain_graph=True)  # type: ignore # PyTorch bug workaround
        else:
            qf_loss.backward()  # type: ignore # PyTorch bug workaround
        self.q_optimizer.step()

        if self.config.shared_encoder_gradient:
            self.shared_encoder_optimizer.step()

        obs = obs.clone().detach()
        pi, log_pi = self.actor(obs)
        qf_pi = [self.q_networks[i](obs, pi) for i in range(self.config.num_critics)]
        min_qf_pi = qf_pi[0]
        for qf in qf_pi[1:]:
            min_qf_pi = torch.min(min_qf_pi, qf)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        # update temperature if needed
        if self.config.autotune:
            alpha_loss = (
                -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
            ).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update target networks
        self._update_targets()

        if self.current_training_step % 100 == 0:
            writer.add_scalar(
                "losses/qf0_values",
                qf_a_values[0].mean().item(),
                self.current_training_step,
            )
            writer.add_scalar(
                "losses/qf1_values",
                qf_a_values[1].mean().item(),
                self.current_training_step,
            )
            writer.add_scalar(
                "losses/qf0_loss", qf_losses[0].item(), self.current_training_step
            )
            writer.add_scalar(
                "losses/qf1_loss", qf_losses[1].item(), self.current_training_step
            )
            writer.add_scalar(
                "losses/qf_loss_average",
                qf_loss.item() / self.config.num_critics,  # type: ignore # PyTorch bug workaround
                self.current_training_step,
            )
            writer.add_scalar(
                "losses/actor_loss", actor_loss.item(), self.current_training_step
            )
            writer.add_scalar("losses/alpha", self.alpha, self.current_training_step)
            if self.config.autotune:
                writer.add_scalar(
                    "losses/alpha_loss",
                    alpha_loss.item(),  # type: ignore # PyTorch bug workaround
                    self.current_training_step,
                )
            if self.use_camera_inputs:
                writer.add_scalar(
                    "weights_img_encoder",
                    self.shared_encoder.image_encoders[0]
                    .fc1.weight.data.norm()  # type: ignore # insufficient type info
                    .cpu()
                    .item(),
                    self.current_training_step,
                )
            writer.add_scalar(
                "entropy", -log_pi.mean().item(), self.current_training_step
            )

            # model checkpoint
            torch.save(
                self.actor.state_dict(),
                os.path.join(self.checkpoint_path, "actor_state_dict.pth"),
            )

    def _update_targets(self):
        """Update the target networks' parameters using exponential moving average."""
        with torch.no_grad():
            for i in range(self.config.num_critics):
                for param, target_param in zip(
                    self.q_networks[i].parameters(),
                    self.q_target_networks[i].parameters(),
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data
                        + (1 - self.config.tau) * target_param.data
                    )

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        actor_parameters = self.actor.state_dict()
        shared_encoder_parameters = self.shared_encoder.state_dict()
        shared_encoder_batchnorm_stats = self.shared_encoder.get_batchnorm_stats()
        self.parameters_queue.put(
            (
                actor_parameters,
                shared_encoder_parameters,
                shared_encoder_batchnorm_stats,
            )
        )

    def close(self):
        """Close the learner and clean up resources."""
        logging.info("Executing SAC Learner closing behavior...")

        # Save the model parameters
        torch.save(
            self.actor.state_dict(),
            os.path.join(self.checkpoint_path, "actor_state_dict.pth"),
        )
        for idx in range(self.config.num_critics):
            torch.save(
                self.q_networks[idx].state_dict(),
                os.path.join(self.checkpoint_path, f"qf{idx + 1}_state_dict.pth"),
            )
            torch.save(
                self.q_target_networks[idx].state_dict(),
                os.path.join(
                    self.checkpoint_path, f"qf{idx + 1}_target_state_dict.pth"
                ),
            )
        if self.config.autotune:
            torch.save(
                self.log_alpha, os.path.join(self.checkpoint_path, "log_alpha.pth")
            )
            torch.save(
                self.a_optimizer.state_dict(),
                os.path.join(self.checkpoint_path, "a_optimizer_state_dict.pth"),
            )
        torch.save(
            self.shared_encoder.state_dict(),
            os.path.join(self.checkpoint_path, "shared_encoder_state_dict.pth"),
        )

        torch.save(
            self.q_optimizer.state_dict(),
            os.path.join(self.checkpoint_path, "q_optimizer_state_dict.pth"),
        )
        if self.config.shared_encoder_gradient:
            torch.save(
                self.shared_encoder_optimizer.state_dict(),
                os.path.join(
                    self.checkpoint_path, "shared_encoder_optimizer_state_dict.pth"
                ),
            )
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(self.checkpoint_path, "actor_optimizer_state_dict.pth"),
        )
        with open(
            os.path.join(self.checkpoint_path, "current_training_step"), "w"
        ) as f:
            f.write(str(self.current_training_step))

        if not self.args.eval:
            with open(os.path.join(self.checkpoint_path, "global_step"), "w") as f:
                f.write(f"{self.global_step} {self.episode_num}")
            self.writer.close()

        self.replay_buffer.save_buffer(self.checkpoint_path)
        torch.cuda.empty_cache()
        logging.info("Model parameters saved successfully.")
        return
