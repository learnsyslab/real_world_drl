import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import random
import numpy as np
import time
import gymnasium as gym

from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from copy import deepcopy

from rlmb.data.buffers_cleanrl import ReplayBuffer
from rlmb.agents.sac.config import SAC_Config
from rlmb.agents.sac.networks_cleanrl import SoftQNetwork, Actor

class SACLearner:
    def __init__(self,
                 action_space,
                 observation_space,
                 parameters_queue: mp.Queue,
                 run_name: str):
        
        # Store the configuration parameters
        self.config = SAC_Config
        assert int(self.config.update_policy_after * self.config.utd_ratio) > 0, \
            "Invalid combination of update_policy_after and utd_ratio. " \
            "Ensure that int(update_policy_after * utd_ratio) > 0."
        
        # check gym environment
        self.is_gymnasium_env = self.config.env_name in list(gym.envs.registry.keys())

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

        # initializing networks
        self.actor = Actor(observation_space, action_space, self.config).to(self.device)
        self.qf1 = SoftQNetwork(self.observation_space, self.action_space).to(self.device)
        self.qf2 = SoftQNetwork(self.observation_space, self.action_space).to(self.device)
        self.qf1_target = SoftQNetwork(self.observation_space, self.action_space).to(self.device)
        self.qf2_target = SoftQNetwork(self.observation_space, self.action_space).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        if not self.is_gymnasium_env:
        # image encoders (only relevant for crisp_gym environments)
            projection_head = torch.nn.Sequential(
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128)
                )
            resnet_18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).requires_grad_(False)
            resnet_18.fc = projection_head
            self.image_encoders = [
                deepcopy(resnet_18).to(self.device) for name in self.observation_space.keys() if "image" in name
            ]
            params = []
            for encoder in self.image_encoders:
                params += list(encoder.fc.parameters())
            self.img_encoder_optimizer = optim.Adam(params, lr=self.config.policy_lr)
        else:
            self.image_encoders = None
        
        # optimizers
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.config.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.config.policy_lr)
        self._sync_nodes()

        # initializing the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config.buffer_size,
            observation_space=observation_space,
            image_encoders=self.image_encoders,
            action_space=action_space,
            device=self.device,
            handle_timeout_termination=False
        )

        # Automatic entropy tuning
        if self.config.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()    
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.config.q_lr)
        else:
            self.alpha = self.config.alpha

        # training checkpoint path
        self.run_name = os.path.basename(self.writer.log_dir)
        self.checkpoint_path = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
  
    def run(self, data_queue: mp.Queue):
        """Main process loop for the RLPD learner."""
        try:
            global_time_step = 0
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
                for _ in range(int(self.config.update_policy_after * self.config.utd_ratio)):
                    # perform enough training steps to match the utd ratio
                    self._train_step(self.writer, current_training_step)
                    current_training_step += 1
                # send updated policy parameters to the actor
                self._sync_nodes()

                for _ in range(self.config.update_policy_after):
                    # move next batch of data from the queue to the replay buffer
                    obs, action, reward, new_obs, terminated, truncated, info = data_queue.get()
                    self.replay_buffer.add(
                        obs=obs, next_obs=new_obs, action=action, reward=reward,
                        done=terminated or truncated, infos=info
                    )
                    global_time_step += 1
                
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating learner process...")
            self.close()
        except Exception as e:
            self.close()
            logging.error(f"An error occurred in the RLPD Learner: {e}", exc_info=True)

    def _train_step(self, writer, current_training_step):
        """Perform a single training step using data from the replay buffer.
        This updates both the critics and the policy networks."""
        data = self.replay_buffer.sample(self.config.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pis, _ = self.actor.get_action(data.next_observations)
            qf1_next_targets = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_targets = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_targets = torch.min(qf1_next_targets, qf2_next_targets) - self.alpha * next_state_log_pis
            next_q_values = data.rewards.flatten() + (1 - data.dones.flatten()) * self.config.gamma * (min_qf_next_targets).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_values)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_values)
        qf_loss = qf1_loss + qf2_loss

        # optimize the q functions, image encoder and policy
        if not self.is_gymnasium_env:
            self.img_encoder_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        if not self.is_gymnasium_env:
            # retain graph for image encoder update
            qf_loss.backward(retain_graph=True)
        else:
            qf_loss.backward()
        self.q_optimizer.step()
        
        pi, log_pi, _ = self.actor.get_action(data.observations)
        qf1_pi = self.qf1(data.observations, pi)
        qf2_pi = self.qf2(data.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        self.actor_optimizer.step()
        if not self.is_gymnasium_env:
            self.img_encoder_optimizer.step()

        # update temperature if needed
        if self.config.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(data.observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update target networks
        self._update_targets()

        if current_training_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), current_training_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), current_training_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), current_training_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), current_training_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, current_training_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), current_training_step)
            writer.add_scalar("losses/alpha", self.alpha, current_training_step)
            if self.config.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), current_training_step)
            
            # model checkpoint
            torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_path, "actor_state_dict.pth"))

    def _update_targets(self):
        """Update the target networks' parameters using exponential moving average."""
        with torch.no_grad():
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            
            for param2, target_param2 in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param2.data.copy_(self.config.tau * param2.data + (1 - self.config.tau) * target_param2.data)

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        actor_parameters = self.actor.state_dict()

        proj_heads_parameters = None
        if not self.is_gymnasium_env:
            proj_heads_parameters = [encoder.fc.state_dict() for encoder in self.image_encoders]
        self.parameters_queue.put((actor_parameters, proj_heads_parameters))

    def close(self):
        """Close the learner and clean up resources."""
        logging.info("Executing RLPD Learner closing behavior...")
        self.writer.close()

        # Save the model parameters
        torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_path, "actor_state_dict.pth"))
        torch.cuda.empty_cache()
        logging.info("Model parameters saved successfully.")
        return
    