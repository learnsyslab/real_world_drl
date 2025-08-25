import logging
import gymnasium as gym
import torch
import torch.multiprocessing as mp
import rclpy
import time
import random
import numpy as np

from crisp_gym.manipulator_env_config import NoCamFrankaEnvConfig
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from rlmb.agents.sac.config import SAC_Config
from rlmb.agents.sac.networks_cleanrl import Actor
from rlmb.agents.sac.env_wrappers import MaximizeHeightRewardWrapper
from rlmb.data.utils import crisp_obs_to_tensor

class SACActor:
    def __init__(self, 
                 env_info_queue: mp.Queue,
                 parameters_queue: mp.Queue,
                 run_name: str):
        
        self.config = SAC_Config()

        # setting the seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.env_info_queue = env_info_queue
        self.parameters_queue = parameters_queue

        # check gym environment
        self.is_gymnasium_env = self.config.env_name in list(gym.envs.registry.keys())

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.cuda else "cpu")
        self.policy_update_after = self.config.update_policy_after
        self.learning_starts = self.config.learning_starts
        self.env = self._create_env()

        # policy
        self.actor = Actor(self.env.observation_space, self.env.action_space, self.config).to(self.device)
        
        # summary writer for tensorboard
        runs_path = Path(__file__).resolve().parent.parent.parent.parent / "runs_actor"
        runs_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(runs_path / run_name)
  
        self._sync_policy() # Wait for the learner to put the initial policy parameters in the queue

    def run(self, data_queue: mp.Queue):
        """Main process loop for the RLPD actor.
        This method will execture actions in the environment
        and send the s,a,r,s' tuples to the data queue."""
    
        try:
            # reset the episode variables
            if not self.is_gymnasium_env:
                self.env.home()
                obs, _ = self.env.reset()
            else:
                obs, _ = self.env.reset(seed = self.config.seed)

            episode_return = 0.0
            episode_length = 0
            sum_of_returns = 0.0
            sum_of_squared_returns = 0.0
            episode_num = 0

            for global_step in range(self.config.total_timesteps):   
                if global_step < self.learning_starts: 
                    # Take random actions for the first few steps
                    action = self.env.action_space.sample()
                else:              
                    # transform observation to torch Tensor
                    if not self.is_gymnasium_env:
                        obs_input = crisp_obs_to_tensor(obs).to(self.device)
                    else:
                        obs_input = torch.Tensor(obs).to(self.device).view(1, -1)

                    action, _, _ = self.actor.get_action(obs_input)
                    action = action.view(-1).detach().cpu().numpy()
                    
                    # sync policy every "self.policy_update_after" steps
                    if (global_step - self.learning_starts) % self.policy_update_after == 0:
                        self._sync_policy()

                next_obs, reward, termination, truncation, info = self.env.step(action)
                episode_return += reward
                episode_length += 1

                real_next_obs = next_obs.copy()
                # send experience data to the learner
                data_queue.put((obs, action, reward, real_next_obs, termination, truncation, info))

                obs = next_obs

                done = termination or truncation
                if done:
                    self.writer.add_scalar(f"charts/episodic_return", episode_return, global_step)
                    self.writer.add_scalar(f"charts/episodic_length", episode_length, global_step)
                    sum_of_returns += episode_return
                    sum_of_squared_returns += episode_return ** 2
                    episode_num += 1
                    eps_return_std = (sum_of_squared_returns / episode_num - (sum_of_returns / episode_num) ** 2) ** 0.5    
                    self.writer.add_scalar(f"charts/avg_return", sum_of_returns / episode_num, global_step)
                    self.writer.add_scalar(f"charts/eps_return_std", eps_return_std, global_step)

                    # reset the episode variables
                    episode_return = 0
                    episode_length = 0
                    if not self.is_gymnasium_env:
                        self.env.home()
                    obs, _ = self.env.reset()
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Terminating actor process...")
            self.close()
        except Exception as e:
            logging.error(f"An error occurred in the RLPD Actor: {e}", exc_info=True)
            self.close()

    def close(self):
        logging.info("Executing RLPD Actor closing behavior...")
        self.writer.close()
        # Clean up the environment
        self.env.close()
        if rclpy.ok():
            rclpy.shutdown()

    def _create_env(self) -> ManipulatorCartesianEnv:
        """Create a new environment instance."""
        if self.is_gymnasium_env:
            env = gym.make(self.config.env_name)
        else:
            manipulator_env_config = NoCamFrankaEnvConfig(max_episode_steps=100, control_frequency=10)
            env = ManipulatorCartesianEnv(config = manipulator_env_config)
            env = MaximizeHeightRewardWrapper(env)
        env.observation_space.dtype = np.float32
        self.env_info_queue.put(env.action_space)
        self.env_info_queue.put(env.observation_space)
        return env

    def _sync_policy(self):
        """Update the actor policy to match the learner policy."""
        # Get the latest policy parameters from the queue
        policy_parameters = self.parameters_queue.get()
        self.actor.load_state_dict(policy_parameters)

        logging.info("Actor received an updated policy")
