import multiprocessing
import time
from typing import Any
from gymnasium import RewardWrapper, ActionWrapper, ObservationWrapper, Wrapper
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import threading
import subprocess
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.multiprocessing as mp
from pynput import keyboard
from crisp_gym.manipulator_env import ManipulatorCartesianEnv

# Make cuDNN deterministic for consistent inference
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from crisp_drl.agents.rlpd.config import RLPD_Config

class MaximizeHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper

    def reward(self, reward, action):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        #reward = 10 * action[2] - 0.1 * np.linalg.norm(action)
        reward = 10 * z - np.linalg.norm(action)

        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, action), terminated, truncated, info
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()


class SparseHeightRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper

    def reward(self, reward, action):
        """
        The reward is set to the scaled height of the eef in the environment.
        """
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        #if z > 0.9:
        #    reward = 100.0
        #else:
        #    reward = - 0.1 * np.linalg.norm(action)
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation.pop("observation.state.joint")
        observation.pop("task")
        observation.pop("observation.state.target")
        eef_pose = self.robot.end_effector_pose
        z = eef_pose.position[2]
        if z > 0.9:
            terminated = True
        return observation, self.reward(reward, action), terminated, truncated, info
    
    def close(self):
        self.env.close()

    def reset(self, seed = None):
        obs, info = self.env.reset(seed=seed)
        obs.pop("observation.state.joint")
        obs.pop("task")
        obs.pop("observation.state.target")
        return obs, info

    def home(self):
        self.env.home()


class SafetyBoundingBoxWrapper(ActionWrapper):
    def __init__(self, env, config: SparseHeightRewardWrapper):
        super().__init__(env)

        self.robot = self.env.robot
        self.cameras = self.env.cameras
        self.gripper = self.env.gripper
        self.config = config

        self.x_min = 0.25
        self.x_max = 0.60
        self.y_min = -0.15
        self.y_max = 0.15
        self.z_min = 0.08
        self.z_max = 0.50
    
    def action(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        eef_pose = self.robot.end_effector_pose
        current_pos = np.array(eef_pose.position)

        # Predict next position based on action (assuming action is a displacement)
        next_pos = current_pos + action[:3]  # Extract position part of the action

        # Check if next position violates bounds
        clipped_pos = np.clip(next_pos, 
                              [self.x_min, self.y_min, self.z_min],
                              [self.x_max, self.y_max, self.z_max])

        # If position would be clipped, adjust the action accordingly
        if not np.array_equal(next_pos, clipped_pos):
            delta = clipped_pos - current_pos
            action = np.concatenate([delta, action[3:]])  # Keep orientation part unchanged
        
        if np.any((action[:3] < -self.config.max_action) | (action[:3] > self.config.max_action)):
            raise RuntimeError(f"Action {action} exceeds the maximum action!")
        return action
    
    def close(self):
        self.env.close()

    def home(self):
        self.env.home()

def append_or_insert(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

# class ImageEncoderWrapper(ObservationWrapper):
#     def __init__(self, env):
#         self.device = 'cuda:0'
#         self.resnet18_transform = ResNet18_Weights.DEFAULT.transforms(antialias=True).to(self.device)
#         self.resnet_18 = (
#             resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
#             .eval()
#             .requires_grad_(False)
#         )
#         self.resnet_18.fc = torch.nn.modules.Identity()
#         self.resnet_18.to(self.device)
        
#         super().__init__(env)

#     def observation(self, observation):
#         t0 = time.time()
#         image_keys = list(filter(lambda k: k.startswith("observation.images."), observation.keys()))
#         # Batch images and rescale to [0.0,1.0]
#         image_tensor = torch.tensor(np.array(list(map(lambda key: observation[key], image_keys))), dtype=torch.float) / 255.0
#         t1 = time.time()
#         # Reshape to B,C,H,W
#         if image_tensor.shape[1] != 3:
#             image_tensor = image_tensor.permute(0, 3, 1, 2)
#         # Apply pre-transform and resnet
#         t2 = time.time()
#         img_tensor = image_tensor.to(self.device)
#         t3 = time.time()
#         img_tensor = self.resnet18_transform(img_tensor)
#         t4 = time.time()
#         features = self.resnet_18(img_tensor)
#         t5 = time.time()
#         features = features.cpu()
#         t6 = time.time()
#         print(f"Image pre-processing took: Batching {(t1-t0)*1000:.3f}, Reshaping {(t2-t1)*1000:3f}, cpu->gpu {(t3-t2)*1000:.3f}, Pre-processing {(t4-t3)*1000:.3f}, resnet: {(t5-t4)*1000:.3f}, gpu->cpu {(t6-t5)*1000:.3f}")
#         # un-batch images
#         for i, key in enumerate(image_keys):
#             observation[key] = features[i]
#         return observation
    
    # def step(
    #     self, action, block=False
    # ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    #     observation, reward, terminated, truncated, info = self.env.step(action, block=block)
    #     return self.observation(observation), reward, terminated, truncated, info

class ImageEncoderWrapper(ObservationWrapper):
    # def __init__(self, env):
    #     self.device = 'cuda:0'

    #     # Pre-allocate a tensor on GPU for input batches
    #     self._gpu_input = torch.zeros(
    #         (2, 3, 256, 256),
    #         dtype=torch.float32,
    #         device=self.device
    #     )

    #     super().__init__(env)

    #     # Load MobileNetV3 and remove classifier
    #     model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    #     model.eval()
    #     model.classifier = nn.Identity()
    #     model.to(self.device)

    #     # TorchScript trace for fast inference
    #     dummy_input = torch.zeros(2, 3, 256, 256).to(self.device)
    #     self.model = torch.jit.trace(model, dummy_input)
    #     self.model.eval()

    #     # Pre-allocate GPU input tensor
    #     self._gpu_input = torch.zeros(
    #         (2, 3, 256, 256),
    #         dtype=torch.float32,
    #         device=self.device
    #     )

    #     # Normalization constants 
    #     self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
    #     self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)



    # def observation(self, observation):
    #     t0 = time.time()

    #     # Collect image keys
    #     image_keys = [k for k in observation.keys() if k.startswith("observation.images.")]
    #     n_images = len(image_keys)
    #     t1 = time.time()

    #     # Convert images to GPU tensor directly and resize
    #     for i, key in enumerate(image_keys):
    #         img = torch.tensor(observation[key], dtype=torch.float32, device=self.device) / 255.0
    #         if img.shape[0] != 3:
    #             img = img.permute(2, 0, 1)  # HWC -> CHW
    #         self._gpu_input[i] = img
    #     t2 = time.time()

    #     # Normalize batch
    #     batch_input = (self._gpu_input[:n_images] - self._mean) / self._std
    #     t3 = time.time()

    #     # Forward pass with TorchScript model
    #     with torch.no_grad():
    #         torch.cuda.synchronize()
    #         features = self.model(batch_input)
    #         torch.cuda.synchronize()
    #     t4 = time.time()

    #     # Move features to CPU and assign back to observation
    #     features_cpu = features.cpu()
    #     t5 = time.time()
    #     for i, key in enumerate(image_keys):
    #         observation[key] = features_cpu[i]
    #     t6 = time.time()

    #     # Print detailed timings
    #     print(f"Image pre-processing took: "
    #           f"Collect keys {(t1-t0)*1000:.3f} ms, "
    #           f"CPU->GPU + resize {(t2-t1)*1000:.3f} ms, "
    #           f"Normalization {(t3-t2)*1000:.3f} ms, "
    #           f"Inference {(t4-t3)*1000:.3f} ms, "
    #           f"GPU->CPU {(t5-t4)*1000:.3f} ms, "
    #           f"Assign back {(t6-t5)*1000:.3f} ms")

    #     return observation

    def __init__(self, env):
        self.device = 'cuda:0'

        super().__init__(env)

        # Load ResNet-18 and remove classifier
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.eval()
        model.fc = nn.Identity()  # remove the final classification layer
        model.to(self.device)

        # TorchScript trace for fast inference
        dummy_input = torch.zeros(2, 3, 224, 224).to(self.device)
        self.model = torch.jit.trace(model, dummy_input)
        self.model.eval()

        # Pre-allocate GPU input tensor
        self._gpu_input = torch.zeros(
            (2, 3, 224, 224),
            dtype=torch.float32,
            device=self.device
        )

        # Normalization constants for ResNet
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

    def _center_crop(self, img, size=224):
        _, h, w = img.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return img[:, top:top+size, left:left+size]

    def observation(self, observation):
        t0 = time.time()

        # Collect image keys
        image_keys = [k for k in observation.keys() if k.startswith("observation.images.")]
        n_images = len(image_keys)
        t1 = time.time()

        # Convert images to GPU tensor directly and center crop
        for i, key in enumerate(image_keys):
            img = torch.tensor(observation[key], dtype=torch.float32, device=self.device) / 255.0
            if img.shape[0] != 3:
                img = img.permute(2, 0, 1)  # HWC -> CHW
            img = self._center_crop(img, size=224)
            self._gpu_input[i] = img
        t2 = time.time()

        # Normalize batch
        batch_input = (self._gpu_input[:n_images] - self._mean) / self._std
        t3 = time.time()

        # Forward pass with TorchScript model
        with torch.no_grad():
            torch.cuda.synchronize()
            features = self.model(batch_input)
            torch.cuda.synchronize()
        t4 = time.time()

        # Move features to CPU and assign back to observation
        features_cpu = features.cpu()
        t5 = time.time()
        for i, key in enumerate(image_keys):
            observation[key] = features_cpu[i]
        t6 = time.time()

        # Print detailed timings
        print(f"Image pre-processing took: "
              f"Collect keys {(t1-t0)*1000:.3f} ms, "
              f"CPU->GPU + crop {(t2-t1)*1000:.3f} ms, "
              f"Normalization {(t3-t2)*1000:.3f} ms, "
              f"Inference {(t4-t3)*1000:.3f} ms, "
              f"GPU->CPU {(t5-t4)*1000:.3f} ms, "
              f"Assign back {(t6-t5)*1000:.3f} ms")

        return observation

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        return self.observation(observation), reward, terminated, truncated, info

    
class LastObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_cartesian_state = None
        self.last_gripper_state = None
    
    def reset(self, *, seed = None, options = None):
        observation, info = self.env.reset(seed=seed, options=options)
        current_cartesian_state = observation['observation.state.cartesian'][:3]
        current_gripper_state = observation["observation.state.gripper"]
        observation["observation.previous.action"] = np.zeros(self.env.action_space.shape)

        observation["observation.velocity.cartesian"] = np.zeros_like(current_cartesian_state)
        observation["observation.error.cartesian"] = observation["observation.state.target"][:3] - current_cartesian_state
        observation["observation.velocity.gripper"] = np.zeros_like(current_gripper_state)
        observation["observation.error.gripper"] = observation["observation.target.gripper"] - current_gripper_state
        
        self.last_cartesian_state = current_cartesian_state
        self.last_gripper_state = current_gripper_state
        return observation, info
    
    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        current_cartesian_state = observation['observation.state.cartesian'][:3]
        current_gripper_state = observation["observation.state.gripper"]

        observation["observation.previous.action"] = action

        observation["observation.velocity.cartesian"] = current_cartesian_state - self.last_cartesian_state if self.last_cartesian_state is not None else np.zeros_like(current_cartesian_state)
        observation["observation.error.cartesian"] = observation["observation.state.target"][:3] - current_cartesian_state
        observation["observation.velocity.gripper"] = current_gripper_state - self.last_gripper_state if self.last_gripper_state is not None else np.zeros_like(current_gripper_state)
        observation["observation.error.gripper"] = observation["observation.target.gripper"] - current_gripper_state
        
        self.last_cartesian_state = current_cartesian_state
        self.last_gripper_state = current_gripper_state
        return observation, reward, terminated, truncated, info


class ObservationConcatWrapper(ObservationWrapper):
    def __init__(self, env, key_ranges):
        super().__init__(env)
        self.key_ranges = key_ranges

    def observation(self, observation):
        return np.concatenate((list(map(lambda kr: np.array(observation[kr[0]])[kr[1][0]:kr[1][1]], self.key_ranges))))
    
    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        return self.observation(observation), reward, terminated, truncated, info


class InsertionResetWrapper(Wrapper):
    def __init__(self, env, initial_pos, grasp_randomization_bounds, insert_randomization_bounds, action_sequence):
        super().__init__(env)
        self.initial_pos = initial_pos
        self.grasp_randomization_bounds = grasp_randomization_bounds
        self.insert_randomization_bounds = insert_randomization_bounds
        self.action_sequence = action_sequence

    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.env.home()
        obs, _ = self.env.reset()
        time.sleep(0.5)
        print("Homed.")

        randomize_insert_action = np.concatenate((np.random.uniform(self.insert_randomization_bounds[0], self.insert_randomization_bounds[1]), [0.0, 0.0, 0.0, 0.0]))
        randomize_grasp_action = np.concatenate((np.random.uniform(self.grasp_randomization_bounds[0], self.grasp_randomization_bounds[1]), [0.0, 0.0, 0.0, 0.0]))
        pos = randomize_grasp_action[:3] + self.initial_pos
        
        obs, *_ = self.env.step(np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0]), block=True)
        while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > 0.003:
            obs, *_ = self.env.step(np.zeros(7), block=True)
        time.sleep(0.5)
        print("Reached starting state.")

        for act in self.action_sequence:
            obs, *_ = self.env.step(act, block=True)
        obs, *_ = self.env.step(-randomize_grasp_action, block=True)
        obs, *_ = self.env.step(randomize_insert_action, block=True)
        while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > 0.003:
            obs, *_ = self.env.step(np.zeros(7), block=True)
        time.sleep(0.5)
        print("Executed reset-sequence")

        obs, info = self.env.reset(seed=seed, options=options)
        time.sleep(0.5)
        return obs, info
    
    def step(self, action, block=False):
        return self.env.step(action, block=block)


class ActionTimeStampWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action, block=False) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        time_stamp = time.time()
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        info["action_t"] = time_stamp
        return observation, reward, terminated, truncated, info
    
class FarAwayTerminationWrapper(Wrapper):
    def __init__(self, env, approximate_goal_pos, max_distance):
        super().__init__(env)
        self.approximate_goal_pos = approximate_goal_pos
        self.max_distance = max_distance

    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)

        if np.linalg.norm(observation['observation.state.cartesian'][:3] - self.approximate_goal_pos) > self.max_distance:
            terminated = True
            append_or_insert(info, "custom_events", "E_FAR_AWAY")
            print(f"Terminated for being far away ({observation['observation.state.cartesian'][:3]} for goal pose {self.approximate_goal_pos})")

        return observation, reward, terminated, truncated, info
    


def controller_container_watcher(out_queue):
    # until first line of current.log split at " " changes
    while True:
        t = threading.Timer(1.0, lambda: None)
        t.start()
        error_string = subprocess.run(["ssh", "linusschwarz@franka", r"grep -P 'cartesian_reflex|communication_constraints_violation|franka::NetworkException' /home/linusschwarz/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout
        if len(error_string) < 5:
            t.join()
            continue
        
        # look for the time code when the container crashed
        crash_time_str = error_string.split(" ")[0].strip("[]") # Time code in format 2025-10-16_10:45:26.554086 
        t.cancel()
        crash_time_struct = time.strptime(crash_time_str.split(".")[0], "%Y-%m-%d_%H:%M:%S")
        crash_timestamp = time.mktime(crash_time_struct) + float("0." + crash_time_str.split(".")[1])


        last_start_time_containing_string = subprocess.run(["ssh", "linusschwarz@franka", r"head -n 1 /home/linusschwarz/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout.split(" ")[0]

        if 'cartesian_reflex' in error_string:
            out_queue.put((crash_timestamp, "E_TORQUE"))
        else:
            out_queue.put((crash_timestamp, "E_CONTROLLER_ISSUE"))
        
        time.sleep(20)

        # wait until new container has launched
        while subprocess.run(["ssh", "linusschwarz@franka", r"head -n 1 /home/linusschwarz/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout.split(" ")[0] == last_start_time_containing_string:
            time.sleep(2)
        
        # wait until topics are available
        while "/joint_trajectory_controller/state" not in subprocess.run(["ssh", "linusschwarz@franka", r"source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID=101 ros2 topic list"], capture_output=True, text=True).stdout:
            time.sleep(5)
        out_queue.put((time.time(), "E_CONTROLLER_READY"))

class ContainerWatcherWrapper(Wrapper):
    def __init__(self, env, ctx: mp.SpawnContext):
        super().__init__(env)
        self.controller_container_watcher_event_queue = ctx.Queue()
        self.controller_container_watcher_thread = multiprocessing.Process(target=controller_container_watcher, args=(self.controller_container_watcher_event_queue,))
        self.controller_container_watcher_thread.start()
        self.is_running = True


    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        # check for controller container events
        temp_events = []
        while not self.controller_container_watcher_event_queue.empty():
            _timestamp, event = self.controller_container_watcher_event_queue.get()
            print(f"[CONTROLLER] Event: {event}")
            temp_events.append(event)
        if len(temp_events) > 0 and temp_events[-1] != "E_CONTROLLER_READY":
            self.is_running = False

        # Wait for E_READY
        while not self.is_running:
            while self.controller_container_watcher_event_queue.empty():
                time.sleep(0.3)
            _timestamp, event = self.controller_container_watcher_event_queue.get()
            if event == "E_CONTAINER_READY":
                print(f"[CONTROLLER] Event: {event}")
                self.is_running = True
            else:
                print(f"[CONTROLLER] Warning Skipping {event}, should have been container ready")
        
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)

        # check for controller container events
        temp_events = []
        while not self.controller_container_watcher_event_queue.empty():
            timestamp, event = self.controller_container_watcher_event_queue.get()
            print(f"[CONTROLLER] Event: {event}")
            temp_events.append((timestamp, event))

        # Terminater on Torque limit, truncate on container issue
        if len(temp_events) > 0:
            assert len(temp_events) == 1, f"At most one concurrent container event allowed, found {temp_events}"

            append_or_insert(info, "custom_events", temp_events[0])
            if temp_events[0][1] == "E_TORQUE":
                terminated = True
            else:
                truncated = True
        
        return observation, reward, terminated, truncated, info

    def close(self):
        self.controller_container_watcher_thread.terminate()
        self.env.close()

class TimeMeasurementWrapper(Wrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        
    def step(self, action, block=False):
        before = time.time()
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        after = time.time()
        info[f"dt_{self.n}"] = after - before
        return observation, reward, terminated, truncated, info

class CLIWrapper(Wrapper):
    def  __init__(self, env, gripper_threshold):
        super().__init__(env)
        self.gripper_threshold = gripper_threshold
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.ready_key = 'r'
        self.environment_ready = False
        self.other_key = None
        self.other_keys_to_check = ['s', 'f', 't']
        self.other_key_to_event = {'s': "E_SUCCESS", 'f': "E_FAIL", 't': "E_TRUNCATE"}


    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(action, block=block)
        # wait for s/f when gripper open
        if observation["observation.state.gripper"] > self.gripper_threshold:
            terminated = True
            print("Place successful? ([s]uccess/[f]ail)")
            while True:
                while self.other_key is None:
                    time.sleep(0.05)
                if self.other_key not in "sf":
                    print(f"[CLI] Ignoring {self.other_key}, waiting for whether the run was success.")
                    self.other_key = None
                    continue
                now = time.time()
                event = self.other_key_to_event[self.other_key]
                self.other_key = None

                print(f"[CLI] {event}")
                append_or_insert(info, "custom_events", (now, event))
                break

        elif self.other_key == "t":
            truncated = True
            event = self.other_key_to_event[self.other_key]
            append_or_insert(info, "custom_events", (time.time(), event))
            self.other_key = None
            print(f"[CLI] {event}")
                
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        # wait for key "r"
        print(f"[CLI] Waiting for ready")
        while not self.environment_ready:
            time.sleep(0.1)
        self.environment_ready = False
        return self.env.reset(seed=seed, options=options)



    def on_press(self, key):
        try:
            if key.char == self.ready_key:
                self.environment_ready = True
            if key.char in self.other_keys_to_check:
                print(f"{key.char} is pressed!")
        except AttributeError:
            pass  # Special keys like shift, ctrl, etc.

    def close(self):
        self.listener.stop()
        self.env.close()