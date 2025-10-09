import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import random
import numpy as np
import gymnasium as gym

from torch import multiprocessing as mp
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from copy import deepcopy

from rlmb.data.buffers_cleanrl import ReplayBuffer, load_buffer_from_file
from rlmb.agents.rlpd.config import RLPD_Config
from rlmb.agents.rlpd.networks_cleanrl import SoftQNetwork, Actor

class RLPDLearner:
    def __init__(self,
                 args,
                 action_space,
                 observation_space,
                 parameters_queue: mp.Queue,
                 run_name: str):
        
        # Store the configuration parameters
        self.config = RLPD_Config
        assert int(self.config.update_policy_after * self.config.utd_ratio) > 0, \
            "Invalid combination of update_policy_after and utd_ratio. " \
            "Ensure that int(update_policy_after * utd_ratio) > 0."
        self.args = args
        
        # check gym environment
        self.use_camera_inputs = self.config.use_cameras

        # set seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.action_space = action_space
        self.observation_space = observation_space
        self.parameters_queue = parameters_queue

        # summary writer for tensorboard
        runs_path = Path(__file__).resolve().parent.parent.parent.parent / "runs_learner"
        runs_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(runs_path / run_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.cuda else "cpu")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.config).items()])),
        )

        # checkpoint names of model to be loaded
        self.load_model = args.resume_training or args.load_policy

        # initializing networks
        self.actor = Actor(observation_space, action_space, self.config).to(self.device)
        self.q_networks = [SoftQNetwork(self.observation_space, self.action_space).to(self.device) for _ in range(self.config.num_critics)]
        self.q_target_networks = [SoftQNetwork(self.observation_space, self.action_space).to(self.device) for _ in range(self.config.num_critics)]
        
        # load model if specified
        if self.load_model is not None:
            self.actor.load_state_dict(torch.load(f"checkpoints/{self.load_model}/actor_state_dict.pth"))
            for idx in range(self.config.num_critics):
                self.q_networks[idx].load_state_dict(torch.load(f"checkpoints/{self.load_model}/qf{idx+1}_state_dict.pth"))
                self.q_target_networks[idx].load_state_dict(torch.load(f"checkpoints/{self.load_model}/qf{idx+1}_target_state_dict.pth"))
            logging.info(f"Loaded model from checkpoints/{self.load_model}")
        else:
            for idx, target_net in enumerate(self.q_target_networks):
                target_net.load_state_dict(self.q_networks[idx].state_dict())

        # image encoders (only relevant for crisp_gym environments)
        if self.use_camera_inputs:
            projection_head = torch.nn.Sequential(
                    torch.nn.Linear(512, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128)
                )
            resnet_18 = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval().requires_grad_(False)
            resnet_18.fc = projection_head
            self.image_encoders = [
                deepcopy(resnet_18).to(self.device) for name in self.observation_space.keys() if "image" in name
            ]
            # Load image encoder weights
            if self.load_model is not None:
                for i, encoder in enumerate(self.image_encoders):
                    encoder.fc.load_state_dict(torch.load(f"checkpoints/{self.load_model}/image_encoder_{i}_state_dict.pth"))
            
            # image encoder optimizer
            params = []
            for encoder in self.image_encoders:
                params += list(encoder.fc.parameters())
            self.img_encoder_optimizer = optim.Adam(params, lr=self.config.policy_lr)
        else:
            self.image_encoders = None
        
        # optimizers
        q_params = []
        for q_net in self.q_networks:
            q_params += list(q_net.parameters())
        self.q_optimizer = optim.Adam(q_params, lr=self.config.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.config.policy_lr)
        self._sync_nodes()

        # initializing the replay buffer
        if self.args.resume_training is not None:
            self.replay_buffer = load_buffer_from_file(f"checkpoints/{self.args.resume_training}/replay_buffer.pkl", self.image_encoders)
            logging.info(f"Loaded replay buffer from checkpoints/{self.args.resume_training}/replay_buffer.pkl")
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size=self.config.buffer_size,
                observation_space=observation_space,
                image_encoders=self.image_encoders,
                action_space=action_space,
                device=self.device,
                handle_timeout_termination=False
            )
        # load expert buffer
        self.expert_buffer = load_buffer_from_file(self.args.expert_buffer_path, self.image_encoders)
        logging.info(f"Loaded expert buffer from {self.args.expert_buffer_path}")

        # Automatic entropy tuning
        if self.config.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            if self.load_model is not None:
                self.log_alpha = torch.load(f"checkpoints/{self.load_model}/log_alpha.pth").to(self.device).requires_grad_(True)
            else:
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.config.q_lr)
        else:
            self.alpha = self.config.alpha

        # training checkpoint path
        self.run_name = os.path.basename(self.writer.log_dir)
        self.checkpoint_path = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
  
    def run(self, data_queue: mp.Queue, episode_is_running: mp.Event):
        """Main process loop for the RLPD learner."""
        try:
            global_time_step = 0
            # check if a model was loaded (in that case we do not need to fill the buffer)
            if self.load_model is None:
                # Wait for enough data in the replay buffer before starting training
                logging.info("Learner is waiting for the replay buffer to fill with initial exploration samples...")
                for _ in range(self.config.learning_starts):
                    obs, action, reward, new_obs, terminated, truncated, info = data_queue.get()
                    self.replay_buffer.add(
                        obs=obs, next_obs=new_obs, action=action, reward=reward, done=terminated or truncated, infos=[info]
                    )
                    global_time_step += 1
            
            current_training_step = 0
            logging.info("Learner starts training...")

            while global_time_step < self.config.total_timesteps:
                episode_is_running.wait()  # Wait until the actor signals that an episode is running
                while episode_is_running.is_set() or not data_queue.empty():
                    obs, action, reward, new_obs, terminated, truncated, info = data_queue.get()
                    self.replay_buffer.add(
                            obs=obs, next_obs=new_obs, action=action, reward=reward,
                            done=terminated or truncated, infos=info
                        )
                    self._train_step(self.writer, current_training_step)
                    current_training_step += 1
                    global_time_step += 1

                # episode end
                self._sync_nodes()

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating learner process...")
            self.close()
        except Exception as e:
            self.close()
            logging.error(f"An error occurred in the RLPD Learner: {e}", exc_info=True)

    def _train_step(self, writer, current_training_step):
        """Perform a single training step using data from the replay buffer.
        This updates both the critics and the policy networks."""
        for step in range(int(self.config.utd_ratio)):
            online_data = self.replay_buffer.sample(self.config.batch_size // 2)
            expert_data = self.expert_buffer.sample(self.config.batch_size // 2)

            observations = torch.cat((online_data.observations, expert_data.observations), dim=0)
            actions = torch.cat((online_data.actions, expert_data.actions), dim=0)
            rewards = torch.cat((online_data.rewards, expert_data.rewards), dim=0)
            next_observations = torch.cat((online_data.next_observations, expert_data.next_observations), dim=0)
            dones = torch.cat((online_data.dones, expert_data.dones), dim=0) 

            breakpoint()
            with torch.no_grad():
                next_state_actions, next_state_log_pis, _ = self.actor.get_action(next_observations)
                # pick two random Q-networks from the ensemble
                ensemble_samples = random.sample(range(self.config.num_critics), self.config.critic_subset_size)
                qf_next_targets = [self.q_target_networks[i](next_observations, next_state_actions).view(-1) for i in ensemble_samples]
                min_qf_next_targets = torch.min(torch.stack(qf_next_targets), dim=0)[0] - self.alpha * next_state_log_pis.view(-1)
                next_q_values = rewards.flatten() + (1 - dones.flatten()) * self.config.gamma * (min_qf_next_targets).view(-1)

            qf_a_values = [self.q_networks[i](observations, actions).view(-1) for i in range(self.config.num_critics)]
            qf_losses = [F.mse_loss(qf_a_values[i], next_q_values) for i in range(self.config.num_critics)]
            qf_loss = sum(qf_losses)

            # optimize the q functions, image encoder and policy
            if self.use_camera_inputs:
                self.img_encoder_optimizer.zero_grad()
            self.q_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            if self.use_camera_inputs and step == int(self.config.utd_ratio) - 1:
                # retain graph for image encoder update
                qf_loss.backward(retain_graph=True)
            else:
                qf_loss.backward()
            self.q_optimizer.step()

            if self.use_camera_inputs:
                self.img_encoder_optimizer.step()
        
        obs = observations.clone().detach()
        pi, log_pi, _ = self.actor.get_action(obs)
        qf_pi = [self.q_networks[i](obs, pi) for i in range(self.config.num_critics)]
        actor_loss = ((self.alpha * log_pi) - (1 / self.config.num_critics) * sum(qf_pi)).mean()

        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update temperature if needed
        if self.config.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update target networks
        self._update_targets()

        if current_training_step % 100 == 0:
            writer.add_scalar("losses/qf0_values", qf_a_values[0].mean().item(), current_training_step)
            writer.add_scalar("losses/qf0_loss", qf_losses[0].item(), current_training_step)
            writer.add_scalar("losses/qf_loss_average", qf_loss.item() / self.config.num_critics, current_training_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), current_training_step)
            writer.add_scalar("losses/alpha", self.alpha, current_training_step)
            if self.config.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), current_training_step)
            if self.use_camera_inputs:
                writer.add_scalar("weights_img_encoder", self.image_encoders[0].fc[0].weight.data.norm().cpu().item(), current_training_step)
            writer.add_scalar("entropy", -log_pi.mean().item(), current_training_step)
            
            # model checkpoint
            torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_path, "actor_state_dict.pth"))

    def _update_targets(self):
        """Update the target networks' parameters using exponential moving average."""
        with torch.no_grad():
            for i in range(self.config.num_critics):
                for param, target_param in zip(self.q_networks[i].parameters(), self.q_target_networks[i].parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        actor_parameters = self.actor.state_dict()

        proj_heads_parameters = None
        if self.use_camera_inputs:
            proj_heads_parameters = [encoder.fc.state_dict() for encoder in self.image_encoders]
        self.parameters_queue.put((actor_parameters, proj_heads_parameters))

    def close(self):
        """Close the learner and clean up resources."""
        logging.info("Executing RLPD Learner closing behavior...")
        self.writer.close()

        # Save the model parameters
        torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_path, "actor_state_dict.pth"))
        for idx in range(self.config.num_critics):
            torch.save(self.q_networks[idx].state_dict(), os.path.join(self.checkpoint_path, f"qf{idx+1}_state_dict.pth"))
            torch.save(self.q_target_networks[idx].state_dict(), os.path.join(self.checkpoint_path, f"qf{idx+1}_target_state_dict.pth"))
        torch.save(self.log_alpha, os.path.join(self.checkpoint_path, "log_alpha.pth"))
        for i, encoder in enumerate(self.image_encoders):
            torch.save(encoder.fc.state_dict(), os.path.join(self.checkpoint_path, f"image_encoder_{i}_state_dict.pth"))
        self.replay_buffer.save_buffer(self.checkpoint_path)
        torch.cuda.empty_cache()
        logging.info("Model parameters saved successfully.")
        return
    