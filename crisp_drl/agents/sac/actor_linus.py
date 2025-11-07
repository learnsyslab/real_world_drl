import functools
import logging
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


from crisp_drl.agents.rlpd.rewards import dense_place_reward, sparse_place_reward, prune_after_async_termination, xy_dense_place_reward
from crisp_drl.agents.sac.config import SAC_Config
from crisp_drl.agents.sac.networks_cleanrl import Actor
from crisp_drl.agents.rlpd.env_wrappers import ActionTimeStampWrapper, BelowZTerminationWrapper, CLIWrapper, DictObservationToInfoMover, ContainerWatcherWrapper, FarAwayTerminationWrapper, ImageEncoderWrapper, InsertionResetWrapper, LastObservationWrapper, NaiveToGoalPositionWrapper, NoRotationActionWrapper, NoRotationNoGripperActionWrapper, NoRotationNoGripperNoZActionWrapper, ObservationFormatterWrapper, TimeMeasurementWrapper, observation_has_z_pressure, observation_has_z_pressure_or_below
from crisp_drl.data.utils import crisp_batch_concat_obs_to_tensor, crisp_obs_to_tensor
from crisp_drl.training.training_cli import clear_terminal


class SACActor:
    def __init__(self, 
                 args,
                 parameters_queue: mp.Queue,
                 observation_space,
                 run_name: str):
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.cuda else "cpu")
        self.update_policy_after = self.config.update_policy_after
        self.learning_starts = self.config.learning_starts
        self.env = self._create_env()

        # checkpoint names of model to be loaded
        self.load_model = args.resume_training or args.load_policy

        # policy
        self.actor = Actor(observation_space, self.env.action_space, self.config).to(self.device)
        if self.load_model is not None:
            self.actor.load_state_dict(torch.load(f"checkpoints/{self.load_model}/actor_state_dict.pth"))

        # image encoders
        if self.use_camera_inputs:
            self.image_encoders = [
                torch.nn.Sequential(
                    torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Linear(128, 16)
                ).to(self.device)
                for _ in range(len(self.env.cameras))
            ]
            if self.load_model is not None:
                for i, image_encoder in enumerate(self.image_encoders):
                    image_encoder.load_state_dict(
                        torch.load(f"checkpoints/{self.load_model}/image_encoder_{i}_state_dict.pth")
                    )
        else:
            self.image_encoders = None

        # summary writer for tensorboard
        runs_path = Path(__file__).resolve().parent.parent.parent.parent / "runs_actor"
        runs_path.mkdir(parents=True, exist_ok=True)
        if not self.args.eval:
            self.writer = SummaryWriter(runs_path / run_name)
  
            self._sync_nodes() # Wait for the learner to put the initial policy parameters in the queue

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

            episode_length = 0
            sum_of_returns = 0.0
            sum_of_squared_returns = 0.0
            episode_num = 0

            for global_step in range(self.config.total_timesteps):
                t0 = time.perf_counter()
                if t3 is not None:
                    dts["out"].append(t0-t3)
                obs_input = crisp_batch_concat_obs_to_tensor(torch.Tensor(obs).view(1, -1),  self.image_encoders, self.device)
                t1 = time.perf_counter()
                action, _, _ = self.actor.get_action(obs_input)
                action = action.view(-1).detach().cpu().numpy()

                t2 = time.perf_counter()
                all_actions.append(action)
                obs, _reward, termination, truncation, info = self.env.step(action, block=True)
                all_observations.append(obs)
                all_infos.append(info)
                done = termination or truncation


                t3_ = time.perf_counter()
                if not done:
                    if t3 is not None:
                        dts["loop"].append(t3_-t3)
                    t3 = t3_
                    dts["enc"].append(t1-t0)
                    dts["actor"].append(t2-t1)
                    dts["step"].append(t3-t2)

                episode_length += 1

                if done:

                    if "custom_events" in info and "E_ROLLOUT_UNUSABLE" in map(lambda entry: entry[1], info["custom_events"]):
                        global_step -= 1
                        obs, reset_info = self.env.reset() 
                        all_actions = []
                        all_observations = [obs]
                        all_infos = [reset_info] 
                        actual_grasp_pos = reset_info["reset.grasped.position"]
                        t3 = None
                        dts = {"enc": [], "actor": [], "step": [], "out": [], "loop": []}
                        continue

                    if not self.args.eval:
                        all_actions, all_observations, all_infos = prune_after_async_termination(all_actions, all_observations, all_infos, {"E_CONTROLLER_ISSUE", "E_TORQUE"})
                        # all_rewards = dense_place_reward(all_actions, all_observations, all_infos, {"E_TORQUE": -10.0, "E_FAR_AWAY": -3.0, "E_BELOW_Z": -3.0, "E_SUCCESS": 10.0, "E_FAIL": -1.0, "E_BAD_BEHAVIOR": -5.0, "E_CONTROLLER_ISSUE": 0.0},
                        #              max_rew=0.01, ideal_goal_pos=[0.53975, -0.033,  0.05], ideal_grasp_pos=np.array([0.58833, -0.13817,  0.04229]), actual_grasp_pos=actual_grasp_pos, k_xy=0.005, k_z=0.002)
                        all_rewards = sparse_place_reward(all_actions, all_observations, all_infos)
                        # all_rewards = xy_dense_place_reward(all_actions, all_observations, all_infos, max_rew=0.01, max_action_magnitude=0.0008,
                        #                                      ideal_goal_pos_xy=np.array([0.53975, -0.033]), ideal_grasp_pos_xy=np.array([0.58833, -0.13817]), 
                        #                                      actual_grasp_pos_xy=actual_grasp_pos[:2])
                    
                        
                        data_queue.put((all_observations, all_actions, all_rewards, termination))
                        
                        episode_return = sum(all_rewards)

                        self.writer.add_scalar(f"charts/episodic_return", episode_return, global_step)
                        self.writer.add_scalar(f"charts/episodic_length", episode_length, global_step)
                        sum_of_returns += episode_return
                        sum_of_squared_returns += episode_return ** 2
                        episode_num += 1
                        eps_return_std = (sum_of_squared_returns / episode_num - (sum_of_returns / episode_num) ** 2) ** 0.5    
                        self.writer.add_scalar(f"charts/avg_return", sum_of_returns / episode_num, global_step)
                        self.writer.add_scalar(f"charts/eps_return_std", eps_return_std, global_step)

                        self.writer.add_scalar(f"charts/t_encoding", np.mean(dts["enc"]), global_step)
                        self.writer.add_scalar(f"charts/t_encoding_std", np.std(dts["enc"]), global_step)
                        self.writer.add_scalar(f"charts/t_actor", np.mean(dts["actor"]), global_step)
                        self.writer.add_scalar(f"charts/t_actor_std", np.std(dts["actor"]), global_step)
                        self.writer.add_scalar(f"charts/t_stepping", np.mean(dts["step"]), global_step)
                        self.writer.add_scalar(f"charts/t_stepping_std", np.std(dts["step"]), global_step)
                        self.writer.add_scalar(f"charts/t_outside", np.mean(dts["out"]), global_step)
                        self.writer.add_scalar(f"charts/t_outside_std", np.std(dts["out"]), global_step)
                        self.writer.add_scalar(f"charts/t_loop", np.mean(dts["loop"]), global_step)
                        self.writer.add_scalar(f"charts/t_loop_std", np.std(dts["loop"]), global_step)

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
                        logging.info(f"Waiting for the learner to finish gradient updates...")
                        self._sync_nodes()
                        logging.info(f"Learner fininshed gradient updates.")

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
        # Clean up the environment
        self.env.close()
        if rclpy.ok():
            rclpy.shutdown()

    def _create_env(self) -> ManipulatorCartesianEnv:
        """Create a new environment instance."""
        env = make_env("my_env")
        print("Env created.")
        env.wait_until_ready()
        print("Env ready.")

        env = InsertionResetWrapper(env, initial_pos=np.array([0.200, -0.020, -0.200]), grasp_randomization_bounds=(np.array([-0.002, -0.002, -0.001]), np.array([0.002, 0.002, 0.001])),                             
                                    insert_randomization_bounds=(np.array([-0.002, -0.002, 0.0]), np.array([0.002, 0.002, 0.0])), action_sequence_to_grasp=load_actions_safe("v4_go_to_pick.json"), action_sequence_after_grasp=load_actions_safe("v4_after_pick.json"))
        env = ActionTimeStampWrapper(env)
        env = LastObservationWrapper(env)
        env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
        env = CLIWrapper(env, termination_fn = lambda _obs: False) # obs["observation.state.cartesian"][2] < 0.049) # functools.partial(observation_has_z_pressure_or_below, error_threshold=0.005, previous_error_threshold=0.003, min_z_height=0.055, terminate_z_height = 0.0475))
        env = NaiveToGoalPositionWrapper(env, coarse=True, randomize=False, step_size_xy=0.01, step_size_z=0.00025, xy_threshold=0.1, 
                                         base_goal_position=np.array([0.541, -0.034,  0.0435]), ideal_grasp_position=np.array([0.57155, -0.03254,  0.04243])) # [0.58833, -0.13817,  0.04229]))
        env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))
        env = DictObservationToInfoMover(env)
        # env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0), ('observation.previous.action', (6,7), 20.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), ('observation.velocity.gripper', (0, 1), 20.0),
        #                                         ('observation.error.gripper', (0, 1), 20.0), ('observation.state.gripper', (0, 1), 1.0), ('observation.target.gripper', (0, 1), 1.0), ('observation.images.wrist_camera', (0, 512), 1.0), ('observation.images.side_camera', (0, 512), 1.0)])
        env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,2), 10.0), ('observation.previous.error.cartesian', (0,3), 10.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), 
                                                            ('observation.images.wrist_camera', (0, 512), 1.0), 
                                                            # ('observation.images.side_camera', (0, 512), 1.0)
                                                            ]) # 268 or 1036
        env = NoRotationNoGripperNoZActionWrapper(env)

        return env

    def _sync_nodes(self):
        """Sync all shared model parameters between actor and learner."""
        policy_parameters, proj_head_parameters = self.parameters_queue.get()
        self.actor.load_state_dict(policy_parameters)
        if self.use_camera_inputs:
            for i, params in enumerate(proj_head_parameters):
                self.image_encoders[i].load_state_dict(params)

        logging.info("Actor received updated parameters for the policy and vision encoder.")
