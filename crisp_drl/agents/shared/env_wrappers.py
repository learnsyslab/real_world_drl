import copy
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, SupportsFloat
import cv2
from gymnasium import RewardWrapper, ActionWrapper, ObservationWrapper, Wrapper

import numpy as np
import torch
import torch.nn as nn
import threading
import subprocess
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.multiprocessing as mp

from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper


try:
    from pynput import keyboard
except ImportError:
    print("pynput not installed, keyboard interrupt will not be available.")
from gymnasium import spaces
import imageio
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor, SharedEncoder, SoftQNetwork


# Make cuDNN deterministic for consistent inference
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
        # reward = 10 * action[2] - 0.1 * np.linalg.norm(action)
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
        # if z > 0.9:
        #    reward = 100.0
        # else:
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

    def reset(self, seed=None):
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
        clipped_pos = np.clip(
            next_pos,
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_max],
        )

        # If position would be clipped, adjust the action accordingly
        if not np.array_equal(next_pos, clipped_pos):
            delta = clipped_pos - current_pos
            action = np.concatenate(
                [delta, action[3:]]
            )  # Keep orientation part unchanged

        if np.any(
            (action[:3] < -self.config.max_action)
            | (action[:3] > self.config.max_action)
        ):
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

    def __init__(self, env, n_cameras, image_size):
        self.device = "cuda:0"
        self.n_cameras = n_cameras
        assert image_size == (
            256,
            256,
        ), f"Only image size 256x256 supported, {image_size} was provided."

        super().__init__(env)
        # Normalization constants for ResNet
        self._mean = (
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(self.device)
            * 255.0
        )
        self._std = (
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(self.device)
            * 255.0
        )

        # Load ResNet-18 and remove classifier
        class Normalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                # register buffers so they move with the model (.to(device)) and are not trained
                self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

            def forward(self, x: torch.Tensor):
                return (x.permute(0, 3, 1, 2) - self.mean) / self.std

        # Load ResNet-18 backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()  # remove final classification layer

        # Wrap model with normalization
        mean = (
            np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        )  # ImageNet means
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0  # ImageNet stds
        model = nn.Sequential(Normalize(mean, std), backbone)

        # model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # model.eval()
        # model.fc = nn.Identity()  # remove the final classification layer
        model.to(self.device)

        # TorchScript trace for fast inference
        dummy_input = torch.zeros(self.n_cameras, 224, 224, 3, dtype=torch.float32).to(
            self.device
        )
        self.model: torch.jit.ScriptModule = torch.jit.trace(model, dummy_input)
        self.model.eval()

        self._gpu_input = torch.zeros(
            (self.n_cameras, 224, 224, 3), dtype=torch.float32, device=self.device
        )

    def observation(self, observation):
        # Collect image keys
        image_keys = [
            k for k in observation.keys() if k.startswith("observation.images.")
        ]
        assert len(image_keys) == self.n_cameras, (
            f"Found {len(image_keys)} cameras, but specified was {self.n_cameras}"
        )

        t0 = time.time()
        # torch.cuda.synchronize()
        t1 = time.time()
        for i, key in enumerate(image_keys):
            self._gpu_input[i] = torch.from_numpy(observation[key][16:240, 16:240])
        # torch.cuda.synchronize()
        t2 = time.time()

        # Forward pass with TorchScript model
        with torch.no_grad():
            # torch.cuda.synchronize()
            features = self.model(self._gpu_input)
            # torch.cuda.synchronize()
        t4 = time.time()

        # Move features to CPU and assign back to observation
        features_cpu = features.cpu()
        t5 = time.time()
        # features_cpu = torch.zeros((self.n_cameras, 1024))
        for i, key in enumerate(image_keys):
            observation[key] = features_cpu[i]

        # Print detailed timings
        # print(f"Image pre-processing took: "
        #       F"Total: {(t5-t0)*1000:.3f} ms, "
        #       # F"Sync: {(t1-t0)*1000:.3f} ms, "
        #       f"to GPU {(t2-t1)*1000:.3f} ms, "
        #       f"Inference {(t4-t2)*1000:.3f} ms, "
        #       f"GPU->CPU {(t5-t4)*1000:.3f} ms, "
        #     )
        return observation

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        return self.observation(observation), reward, terminated, truncated, info


class DinoVAEImageEncoderWrapper(ObservationWrapper):
    """Image encoder using DINOv2 backbone + trained VAE encoder.

    This wrapper:
    1. Resizes images to 224x224
    2. Passes through DINOv2 (ViT-S/14 with registers) to get 384-dim features
    3. Normalizes features using saved mean/std from VAE training
    4. Passes through trained VAE encoder to get low-dimensional latent representation

    Args:
        env: The environment to wrap
        n_cameras: Number of cameras in the observation
        image_size: Expected input image size (must be 224x224 or 256x256)
        vae_checkpoint_path: Path to trained VAE checkpoint (.pt file)
        normalization_path: Path to normalization stats (.npz file with mean/std)
        dino_model_name: DINOv2 model name (default: dinov2_vits14_reg)
        device: Device to run inference on (default: cuda:0)
    """

    def __init__(
        self,
        env,
        n_cameras: int,
        image_size: tuple[int, int],
        vae_checkpoint_path: str,
        normalization_path: str,
        dino_model_name: str = "dinov2_vits14_reg",
        device: str = "cuda:0",
    ):
        self.device = device
        self.n_cameras = n_cameras
        self.image_size = image_size

        super().__init__(env)

        # ---------------------------------------------------------------------------
        # Load DINOv2 backbone
        # ---------------------------------------------------------------------------
        print(f"Loading DINOv2 model: {dino_model_name}...")

        class Normalize(nn.Module):
            """ImageNet normalization for DINOv2."""

            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Expect input as (N, H, W, C) in range [0, 1]
                x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
                return (x - self.mean) / self.std

        dino_backbone = torch.hub.load(
            "facebookresearch/dinov2", dino_model_name, trust_repo=True
        ).to(self.device)
        dino_backbone.eval()

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.dino_model = nn.Sequential(Normalize(mean, std), dino_backbone).to(
            self.device
        )

        # Try to trace for faster inference
        try:
            dummy = torch.zeros(
                n_cameras, 224, 224, 3, dtype=torch.float32, device=self.device
            )
            self.dino_model = torch.jit.trace(self.dino_model, dummy)
            self.dino_model.eval()
            print("  DINOv2 traced with TorchScript")
        except Exception as e:
            print(f"  TorchScript trace failed, using eager mode: {e}")

        # ---------------------------------------------------------------------------
        # Load VAE encoder
        # ---------------------------------------------------------------------------
        print(f"Loading VAE from: {vae_checkpoint_path}")
        checkpoint = torch.load(vae_checkpoint_path, map_location=self.device)
        vae_config = checkpoint["config"]

        # Recreate VAE encoder architecture
        class VAEEncoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.ln1 = nn.LayerNorm(hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.ln2 = nn.LayerNorm(hidden_dim)
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = nn.functional.gelu(self.ln1(self.fc1(x)))
                h = nn.functional.gelu(self.ln2(self.fc2(h)))
                return self.fc_mu(h)  # Return mean for inference

        self.vae_encoder = VAEEncoder(
            input_dim=vae_config["input_dim"],
            hidden_dim=vae_config["hidden_dim"],
            latent_dim=vae_config["latent_dim"],
        ).to(self.device)

        # Load encoder weights from full VAE state dict
        encoder_state = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("encoder."):
                # Map encoder weights, skip logvar head
                new_key = key.replace("encoder.", "")
                if "fc_logvar" not in new_key:
                    encoder_state[new_key] = value
        self.vae_encoder.load_state_dict(encoder_state)
        self.vae_encoder.eval()

        self.latent_dim = vae_config["latent_dim"]
        print(
            f"  VAE: {vae_config['input_dim']} -> {vae_config['hidden_dim']} -> {self.latent_dim}"
        )

        # ---------------------------------------------------------------------------
        # Load normalization stats
        # ---------------------------------------------------------------------------
        print(f"Loading normalization from: {normalization_path}")
        norm_data = np.load(normalization_path)
        self.feat_mean = torch.from_numpy(norm_data["mean"]).float().to(self.device)
        self.feat_std = torch.from_numpy(norm_data["std"]).float().to(self.device)

        # ---------------------------------------------------------------------------
        # Pre-allocate GPU tensors
        # ---------------------------------------------------------------------------
        self._gpu_input = torch.zeros(
            (n_cameras, 224, 224, 3), dtype=torch.float32, device=self.device
        )

        print(f"DinoVAEImageEncoderWrapper initialized:")
        print(f"  Cameras: {n_cameras}, Image size: {image_size}")
        print(f"  Output dim per camera: {self.latent_dim}")

    def observation(self, observation):
        # Collect image keys
        image_keys = [
            k for k in observation.keys() if k.startswith("observation.images.")
        ]
        assert len(image_keys) == self.n_cameras, (
            f"Found {len(image_keys)} cameras, but specified was {self.n_cameras}"
        )

        # Load images to GPU (crop to 224x224 if needed)
        for i, key in enumerate(image_keys):
            img = observation[key]
            # Handle different input sizes
            if img.shape[0] == 256 and img.shape[1] == 256:
                # Center crop from 256x256 to 224x224
                img = img[16:240, 16:240]
            elif img.shape[0] != 224 or img.shape[1] != 224:
                raise ValueError(
                    f"Unexpected image size {img.shape[:2]}, expected 224x224 or 256x256"
                )
            # Normalize to [0, 1]
            self._gpu_input[i] = torch.from_numpy(img).float() / 255.0

        with torch.no_grad():
            # DINOv2 forward pass -> (n_cameras, 384)
            dino_features = self.dino_model(self._gpu_input)

            # Normalize features
            dino_features_norm = (dino_features - self.feat_mean) / self.feat_std

            # VAE encoder forward pass -> (n_cameras, latent_dim)
            latent = self.vae_encoder(dino_features_norm)

        # Move to CPU and assign back to observation
        latent_cpu = latent.cpu()
        for i, key in enumerate(image_keys):
            observation[key] = latent_cpu[i]

        return observation

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        return self.observation(observation), reward, terminated, truncated, info


class DinoImageEncoderWrapper(ObservationWrapper):
    """Image encoder using only DINOv2 backbone (no VAE).

    This wrapper passes images through DINOv2 to get 384-dim feature vectors.
    Useful as a baseline or when you don't have a trained VAE.

    Args:
        env: The environment to wrap
        n_cameras: Number of cameras in the observation
        image_size: Expected input image size (must be 224x224 or 256x256)
        dino_model_name: DINOv2 model name (default: dinov2_vits14_reg)
        device: Device to run inference on (default: cuda:0)
    """

    def __init__(
        self,
        env,
        n_cameras: int,
        image_size: tuple[int, int],
        image_keys: list[str],
        crops: dict[str, tuple[int, int, int, int]],
        rescales: dict[str, float] = {},
        dino_model_name: str = "dinov2_vits14_reg",
        device: str = "cuda:0",
    ):
        super().__init__(env)
        self.device = device
        self.n_cameras = n_cameras
        self.image_size = image_size
        self.crops = crops
        self.rescales = rescales
        self.image_keys = image_keys
        assert len(self.image_keys) == self.n_cameras, (
            f"Found {len(self.image_keys)} cameras, but specified was {self.n_cameras}"
        )

        print(f"Loading DINOv2 model: {dino_model_name}...")

        class Normalize(nn.Module):
            """ImageNet normalization for DINOv2."""

            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
                return (x - self.mean) / self.std

        dino_backbone = torch.hub.load(
            "/home/linusschwarz/.cache/torch/hub/facebookresearch_dinov2_main",
            dino_model_name,
            trust_repo=True,
            source="local",
        ).to(self.device)
        dino_backbone.eval()

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.model = nn.Sequential(Normalize(mean, std), dino_backbone).to(self.device)

        # TorchScript trace for faster inference
        try:
            dummy = torch.zeros(
                n_cameras, 224, 224, 3, dtype=torch.float32, device=self.device
            )
            self.model = torch.jit.trace(self.model, dummy)
            self.model.eval()
            print("  DINOv2 traced with TorchScript")
        except Exception as e:
            print(f"  TorchScript trace failed, using eager mode: {e}")

        self._gpu_input = torch.zeros(
            (n_cameras, 224, 224, 3), dtype=torch.float32, device=self.device
        )

        print("DinoImageEncoderWrapper initialized:")
        print(f"  Cameras: {n_cameras}, Image size: {image_size}")
        print("  Output dim per camera: 384")
        self.first_image = True

    def observation(self, observation):
        for i, key in enumerate(self.image_keys):
            img = observation[key]
            if self.first_image:
                print(f"Image key: {key}, shape: {img.shape}")

            if key in self.crops:
                crop = self.crops[key]
                img = img[crop[0] : crop[1], crop[2] : crop[3]]
            if img.shape[0] == 256 and img.shape[1] == 256:
                img = img[16:240, 16:240]
            elif img.shape[0] != 224 or img.shape[1] != 224:
                raise ValueError(
                    f"Unexpected image size {img.shape[:2]}, expected 224x224 or 256x256"
                )
            if key in self.rescales:
                scale = self.rescales[key]
                new_size = int(224 * scale)
                img = cv2.resize(
                    img,
                    dsize=(
                        new_size,
                        new_size,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Center crop or pad to 224x224
                start = (new_size - 224) // 2
                img = img[start : start + 224, start : start + 224]
            # save first image for debugging
            if self.first_image:
                imageio.imwrite(f"test_images/debug_first_image_{key}.png", img)
            self._gpu_input[i] = torch.from_numpy(img).float() / 255.0
        self.first_image = False

        with torch.no_grad():
            features = self.model(self._gpu_input)

        features_cpu = features.cpu()
        for i, key in enumerate(self.image_keys):
            observation[key.replace("images", "features")] = features_cpu[i]

        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info


class LastObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_cartesian_state = None
        self.last_gripper_state = None
        self.last_cartesian_error = None
        self.last_gripper_error = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        current_cartesian_state = observation["observation.state.cartesian"][:3]
        # current_gripper_state = observation["observation.state.gripper"]
        observation["observation.previous.action"] = np.zeros(
            self.env.action_space.shape
        )
        observation["observation.previous.error.cartesian"] = np.zeros(3)
        observation["observation.previous.error.gripper"] = np.zeros(1)

        observation["observation.velocity.cartesian"] = np.zeros_like(
            current_cartesian_state
        )
        observation["observation.error.cartesian"] = (
            observation["observation.state.target"][:3] - current_cartesian_state
        )
        # observation["observation.velocity.gripper"] = np.zeros_like(
        #     current_gripper_state
        # )
        # observation["observation.error.gripper"] = (
        #     observation["observation.target.gripper"] - current_gripper_state
        # )

        self.last_cartesian_state = current_cartesian_state
        # self.last_gripper_state = current_gripper_state
        self.last_cartesian_error = observation["observation.error.cartesian"]
        # self.last_gripper_error = observation["observation.error.gripper"]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_cartesian_state = observation["observation.state.cartesian"][:3]
        # current_gripper_state = observation["observation.state.gripper"]

        observation["observation.previous.action"] = action
        observation["observation.previous.error.cartesian"] = self.last_cartesian_error
        # observation["observation.previous.error.gripper"] = self.last_gripper_error

        observation["observation.velocity.cartesian"] = (
            current_cartesian_state - self.last_cartesian_state
            if self.last_cartesian_state is not None
            else np.zeros_like(current_cartesian_state)
        )
        observation["observation.error.cartesian"] = (
            observation["observation.state.target"][:3] - current_cartesian_state
        )
        # observation["observation.velocity.gripper"] = (
        #     current_gripper_state - self.last_gripper_state
        #     if self.last_gripper_state is not None
        #     else np.zeros_like(current_gripper_state)
        # )
        # observation["observation.error.gripper"] = (
        #     observation["observation.target.gripper"] - current_gripper_state
        # )

        self.last_cartesian_state = current_cartesian_state
        # self.last_gripper_state = current_gripper_state
        self.last_cartesian_error = observation["observation.error.cartesian"]
        # self.last_gripper_error = observation["observation.error.gripper"]
        return observation, reward, terminated, truncated, info


class ObservationFormatterWrapper(ObservationWrapper):
    def __init__(
        self, env, device, keys_ranges_scales: list[tuple[str, tuple[int, int], float]]
    ):
        super().__init__(env)
        self.device = device
        self.keys_ranges_scales = keys_ranges_scales
        n_dim = sum(map(lambda krs: krs[1][1] - krs[1][0], self.keys_ranges_scales))
        self.observation_space = spaces.Box(
            low=np.full(n_dim, -np.inf), high=np.full(n_dim, np.inf)
        )

    def observation(self, observation):
        obs_list = []
        for key, range_, scale in self.keys_ranges_scales:
            extracted = torch.tensor(
                observation[key][range_[0] : range_[1]],
                dtype=torch.float32,
                device=self.device,
            )
            obs_list.append(extracted if scale == 1.0 else extracted * scale)
        observation["observation.formatted"] = torch.concatenate(obs_list)
        return observation


class NoRotationActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (4,))

    def action(self, action):
        used_action = np.zeros(7)
        used_action[:3] = action[:3]
        used_action[6] = action[3]
        return used_action

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action), block=block)


class NoRotationNoGripperActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))

    def action(self, action):
        return np.concatenate((action, np.zeros(4)))

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))


class NoRotationNoGripperNoZActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))

    def action(self, action):
        return np.concatenate((action, np.zeros(5)))

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))  # pyright: ignore[reportReturnType]


class NoGripperActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (6,))

    def action(self, action):
        return np.concatenate((action, [0]))

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))  # pyright: ignore[reportReturnType]


class NoRotationNoGripperNoZActionClippedWrapperSim(Wrapper):
    def __init__(self, env, clip=0.00025):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.clip = clip

    def action(self, action):
        action = np.concatenate((np.clip(action, -self.clip, self.clip), np.zeros(5)))
        action[3] = 1.0  # No rotation quaternion w=1
        return action

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))


class NoRotationNoGripperWrapperSim(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))

    def action(self, action):
        action = np.concatenate((action, np.zeros(4)))
        action[3] = 1.0  # No rotation quaternion w=1
        return action

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))


class InsertionResetWrapper(Wrapper):
    def __init__(
        self,
        env,
        initial_pos,
        grasp_randomization_bounds,
        insert_randomization_bounds,
        action_sequence_to_grasp,
        action_sequence_after_grasp,
    ):
        super().__init__(env)
        self.initial_pos = initial_pos
        self.grasp_randomization_bounds = grasp_randomization_bounds
        self.insert_randomization_bounds = insert_randomization_bounds
        self.action_sequence_to_grasp = action_sequence_to_grasp
        self.action_sequence_after_grasp = action_sequence_after_grasp

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.env.home()
        obs, _ = self.env.reset()
        time.sleep(0.5)
        print("Homed.")

        randomize_insert_action = np.concatenate(
            (
                np.random.uniform(
                    self.insert_randomization_bounds[0],
                    self.insert_randomization_bounds[1],
                ),
                [0.0, 0.0, 0.0, 0.0],
            )
        )
        randomize_grasp_action = np.concatenate(
            (
                np.random.uniform(
                    self.grasp_randomization_bounds[0],
                    self.grasp_randomization_bounds[1],
                ),
                [0.0, 0.0, 0.0, 0.0],
            )
        )
        pos = randomize_grasp_action[:3] + self.initial_pos

        obs, *_ = self.env.step(
            np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0]), block=True
        )
        while (
            np.linalg.norm(
                obs["observation.state.cartesian"][:3]
                - obs["observation.state.target"][:3]
            )
            > 0.003
        ):
            obs, *_ = self.env.step(np.zeros(7), block=True)
        time.sleep(0.5)
        print("Reached starting state.")

        for act in self.action_sequence_to_grasp:
            obs, *_ = self.env.step(act, block=True)
        time.sleep(0.5)
        obs = self.env._get_obs()
        reset_grasped_position = obs["observation.state.cartesian"][:3].copy()
        print("Executed grasp sequence")
        for act in self.action_sequence_after_grasp:
            obs, *_ = self.env.step(act, block=True)

        randomize_grasp_action[0] = 0
        randomize_grasp_action[2] = 0

        obs, *_ = self.env.step(-randomize_grasp_action, block=True)
        obs, *_ = self.env.step(randomize_insert_action, block=True)
        while (
            np.linalg.norm(
                obs["observation.state.cartesian"][:3]
                - obs["observation.state.target"][:3]
            )
            > 0.003
        ):
            obs, *_ = self.env.step(np.zeros(7), block=True)
        time.sleep(0.5)
        print("Executed randomization sequence")

        obs, info = self.env.reset(seed=seed, options=options)
        time.sleep(0.5)
        info["reset.randomize.insert"] = randomize_insert_action
        info["reset.grasped.position"] = reset_grasped_position
        return obs, info

    def step(self, action, block=False):
        return self.env.step(action, block=block)


class PrintCartesianInfoWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        cartesian_error = (
            observation["observation.state.target"][:3]
            - observation["observation.state.cartesian"][:3]
        )
        cartessian_state = observation["observation.state.cartesian"][:3]
        with np.printoptions(precision=2, suppress=True):
            print(f"Cartesian state: {cartessian_state * 1000} mm")
            print(f"Cartesian error: {cartesian_error * 1000} mm")
        return observation, reward, terminated, truncated, info


class ActionTimeStampWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        time_stamp = time.time()
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["action_t"] = time_stamp
        return observation, reward, terminated, truncated, info


class FarAwayTerminationWrapper(Wrapper):
    def __init__(self, env, approximate_goal_pos, max_distance):
        super().__init__(env)
        self.approximate_goal_pos = approximate_goal_pos
        self.max_distance = max_distance

    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )

        if (
            np.linalg.norm(
                observation["observation.state.cartesian"][:3]
                - self.approximate_goal_pos
            )
            > self.max_distance
        ):
            append_or_insert(info, "custom_events", (time.time(), "E_FAR_AWAY"))
            terminated = True
            print(
                f"Terminated for being far away ({observation['observation.state.cartesian'][:3]} for goal pose {self.approximate_goal_pos})"
            )

        return observation, reward, terminated, truncated, info


class BelowZTerminationWrapper(Wrapper):
    def __init__(self, env, min_z):
        super().__init__(env)
        self.min_z = min_z

    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )

        if observation["observation.state.cartesian"][2] < self.min_z:
            append_or_insert(info, "custom_events", (time.time(), "E_BELOW_Z"))
            terminated = True
            print(
                f"Terminated for being too low ({observation['observation.state.cartesian'][2]} for min_z {self.min_z})"
            )

        return observation, reward, terminated, truncated, info


class DictObservationToInfoMover(Wrapper):
    def __init__(self, env, keep_raw_images=False):
        super().__init__(env)
        self.keep_raw_images = keep_raw_images

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["observation"] = self.obs_info(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["observation"] = self.obs_info(obs)
        return obs, info

    def obs_info(self, obs):
        image_keys = [k for k in obs.keys() if k.startswith("observation.images.")]
        obs_copy = copy.copy(obs)
        for key in image_keys:
            if not self.keep_raw_images or not key.endswith("_raw"):
                obs_copy.pop(key)
        return obs_copy


class VideoWrapper(Wrapper):
    def __init__(self, env, video_dir, fps=30, camera_keys=[]):
        super().__init__(env)
        self.video_dir = video_dir
        self.fps = fps
        self.camera_keys = camera_keys
        self.frames = [[] for _ in camera_keys]
        os.makedirs(self.video_dir, exist_ok=True)
        self.i = 0

    def step(self, action, block=False):
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        for cam_idx, cam_key in enumerate(self.camera_keys):
            frame = observation[cam_key]
            self.frames[cam_idx].append(frame)
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if len(self.frames[0]) > 0:
            for cam_idx, cam_key in enumerate(self.camera_keys):
                # with open(os.path.join(self.video_dir, f"episode_{self.i}_{cam_key}.mp4"), "w") as f:

                imageio.mimwrite(
                    os.path.join(self.video_dir, f"episode_{self.i}_{cam_key}.mp4"),
                    self.frames[cam_idx],
                    format="mp4",
                    fps=self.fps,
                    codec="libx264",
                )  # type: ignore
            self.frames = [[obs[cam_key]] for cam_key in self.camera_keys]
            self.i += 1
        return obs, info


class NaiveToGoalPositionWrapper(ActionWrapper):
    def __init__(
        self,
        env,
        step_size_xy=0.001,
        step_size_z=0.00033,
        xy_threshold=0.005,
        base_goal_position=np.array([0.541, -0.034, 0.0435]),
        ideal_grasp_position=np.array([0.58833, -0.13817, 0.04229]),
        coarse=False,
        randomize=False,
    ):
        super().__init__(env)
        self.step_size_xy = step_size_xy
        self.step_size_z = step_size_z
        self.xy_threshold = xy_threshold
        self.base_goal_position = base_goal_position
        self.ideal_grasp_pos = ideal_grasp_position
        self.goal_position = None
        self._obs = None
        self.randomize = randomize
        self.coarse = coarse

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        actual_grasp_pos = info["reset.grasped.position"]
        # compute actual goal position based on where the object was grasped
        self.goal_position = self.base_goal_position.copy()
        self.goal_position[0] += actual_grasp_pos[0] - self.ideal_grasp_pos[0]
        self.goal_position[2] += actual_grasp_pos[2] - self.ideal_grasp_pos[2]
        if self.randomize:
            self.goal_position[:2] += np.random.uniform(
                -self.xy_threshold / 2, self.xy_threshold / 2
            )
        self._obs = obs
        return obs, info

    def step(
        self, base_action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # if xy close, go directly to goal, otherwise move only in xy direction
        current_pos = self._obs["observation.state.cartesian"][:3]
        action = np.zeros(7)
        delta = self.goal_position - current_pos
        norm_xy = np.linalg.norm(delta[:2])

        if not self.coarse or norm_xy > self.xy_threshold:
            action[:2] = delta[:2] * min(self.step_size_xy / norm_xy, 1.0)
        else:  # elif abs(self._obs['observation.state.target'][2] - self._obs['observation.state.cartesian'][2]) < 0.00175:
            action[2] = delta[2] * min(self.step_size_z / abs(delta[2]), 1.0)

        observation, reward, terminated, truncated, info = self.env.step(
            base_action + action, block=block
        )
        # info['action.naive'] = np.copy(action)
        self._obs = observation
        return observation, reward, terminated, truncated, info


class SafetyBoxWrapperXY(ActionWrapper):
    def __init__(
        self,
        env,
        step_size=0.00025,
        box_radius=0.003,
        base_goal_position=np.array([0.541, -0.034]),
        ideal_grasp_position=np.array([0.58833, -0.13817]),
        coarse=True,
        randomize=True,
        randomization_box_radius=None,
    ):
        super().__init__(env)
        self.step_size_xy = step_size
        self.box_radius = box_radius
        self.base_goal_position = base_goal_position
        self.ideal_grasp_pos = ideal_grasp_position
        self.goal_position = None
        self.ideal_goal_position = None
        self._obs = None
        self.randomize = randomize
        self.coarse = coarse
        if randomization_box_radius is not None:
            self.randomization_box_radius = randomization_box_radius
        else:
            self.randomization_box_radius = box_radius * 0.8

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        actual_grasp_pos = info["reset.grasped.position"]
        # compute actual goal position based on where the object was grasped
        self.goal_position = self.base_goal_position.copy()
        self.goal_position[0] += actual_grasp_pos[0] - self.ideal_grasp_pos[0]
        self.ideal_goal_position = self.goal_position.copy()
        if self.randomize:
            self.goal_position += np.random.uniform(
                -self.randomization_box_radius, self.randomization_box_radius, size=(2,)
            )
        self._obs = obs
        return obs, info

    def step(
        self, base_action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # if xy close, go directly to goal, otherwise move only in xy direction
        current_pos = self._obs["observation.state.cartesian"][:2]
        action = np.zeros(7)
        delta = self.goal_position - current_pos
        norm_ideal = np.linalg.norm(self.ideal_goal_position - current_pos)
        norm_xy = np.linalg.norm(delta)

        if not self.coarse or norm_xy > self.box_radius:
            action[:2] = delta * min(self.step_size_xy / norm_xy, 1.0)

        observation, reward, terminated, truncated, info = self.env.step(
            base_action + action, block=block
        )
        if not self.coarse or (
            norm_xy > self.box_radius and norm_ideal > self.box_radius * 0.75
        ):
            append_or_insert(
                info, "custom_events", (time.time(), "E_SAFETY_BOX_VIOLATION")
            )
        # info['action.naive'] = np.copy(action)
        self._obs = observation
        return observation, reward, terminated, truncated, info


class NaiveZForceWrapper(ActionWrapper):
    def __init__(
        self,
        env,
        step_size=0.00033,
        max_z_error=0.002,
    ):
        super().__init__(env)
        self.step_size = step_size
        self.max_z_error = max_z_error

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if abs(self._obs["observation.error.cartesian"][2]) < self.max_z_error:
            action[2] -= self.step_size
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        self._obs = observation
        return observation, reward, terminated, truncated, info


class NaiveZVelocityWrapper(ActionWrapper):
    def __init__(
        self,
        env,
        step_size=0.00033,
    ):
        super().__init__(env)
        self.step_size = step_size

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action[2] -= self.step_size
        return self.env.step(action, block=block)


def controller_container_watcher(out_queue):
    # until first line of current.log split at " " changes
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("DYLD_LIBRARY_PATH", None)  # For macOS
    env.pop("SSL_CERT_FILE", None)
    env.pop("OPENSSL_CONF", None)
    ppid = os.getppid()
    while True:
        if os.getppid() != ppid:
            print("[CONTROLLER] Parent process changed, exiting container watcher")
            return
        error_out = subprocess.run(
            [
                "ssh",
                "linusschwarz@franka",
                r"grep -P 'cartesian_reflex|joint_velocity_violation|communication_constraints_violation|franka::NetworkException' /home/linusschwarz/crisp_controllers_demos/current.log",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        error_string = error_out.stdout
        if len(error_string) < 5:
            error_err = error_out.stderr
            if len(error_err) > 0:
                print(f"Container watcher finished with error {error_err}")
            time.sleep(0.5)
            continue

        print(f"[CONTROLLER] Error: {error_string}")

        # look for the time code when the container crashed
        crash_time_str = error_string.split(" ")[0].strip(
            "[]"
        )  # Time code in format 2025-10-16_10:45:26.554086
        crash_time_struct = time.strptime(
            crash_time_str.split(".")[0], "%Y-%m-%d_%H:%M:%S"
        )
        crash_timestamp = time.mktime(crash_time_struct) + float(
            "0." + crash_time_str.split(".")[1]
        )

        last_start_time_containing_proc = subprocess.run(
            [
                "ssh",
                "linusschwarz@franka",
                r"head -n 1 /home/linusschwarz/crisp_controllers_demos/current.log",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        last_start_time_containing_string = (
            last_start_time_containing_proc.stdout.split(" ")[0]
        )

        if (
            "cartesian_reflex" in error_string
            or "joint_velocity_violation" in error_string
        ):
            print(f"[CONTROLLER] Cartesian reflex/joint velocity violation")
            out_queue.put((crash_timestamp, "E_TORQUE"))
        else:
            out_queue.put((crash_timestamp, "E_CONTROLLER_ISSUE"))
            print(f"[CONTROLLER] Controller issue")

        last_start_time_containing_proc_err = last_start_time_containing_proc.stderr
        if len(last_start_time_containing_proc_err) > 0:
            print(
                f"Last time code extraction finished with error {last_start_time_containing_proc_err}"
            )

        time.sleep(20)

        # wait until new container has launched
        while (
            subprocess.run(
                [
                    "ssh",
                    "linusschwarz@franka",
                    r"head -n 1 /home/linusschwarz/crisp_controllers_demos/current.log",
                ],
                capture_output=True,
                text=True,
                env=env,
            ).stdout.split(" ")[0]
            == last_start_time_containing_string
        ):
            time.sleep(2)

        # wait until topics are available
        while True:
            topic_check = subprocess.run(
                [
                    "ssh",
                    "linusschwarz@franka",
                    r"source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID=101 ros2 topic list",
                ],
                capture_output=True,
                text=True,
                env=env,
            ).stdout
            if (
                "/cartesian_impedance_controller/transition_event" in topic_check
                and "/joint_trajectory_controller/state" in topic_check
            ):
                break

            time.sleep(5)
        out_queue.put((time.time(), "E_CONTROLLER_READY"))
        print(f"[CONTROLLER] Controller READY")


class ContainerWatcherWrapper(Wrapper):
    def __init__(self, env, ctx: multiprocessing.context.BaseContext):
        super().__init__(env)
        self.controller_container_watcher_event_queue = ctx.Queue()
        self.controller_container_watcher_thread = multiprocessing.Process(
            target=controller_container_watcher,
            args=(self.controller_container_watcher_event_queue,),
        )
        self.controller_container_watcher_thread.daemon = True
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
            print(f"[CONTROLLER] [RESET-backlog] Event: {event}")
            temp_events.append(event)
        if len(temp_events) > 0 and temp_events[-1] != "E_CONTROLLER_READY":
            self.is_running = False

        # Wait for E_READY
        while not self.is_running:
            print(f"[CONTROLLER] [RESET] Not ready, waiting...")
            while self.controller_container_watcher_event_queue.empty():
                time.sleep(0.3)
            _timestamp, event = self.controller_container_watcher_event_queue.get()
            if event == "E_CONTROLLER_READY":
                print(f"[CONTROLLER] [RESET-wait-ready] Event: {event}")
                self.is_running = True
                self.env.unwrapped.wait_until_ready()  # type: ignore
            else:
                print(
                    f"[CONTROLLER] Warning Skipping {event}, should have been container ready"
                )

        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        # check for controller container events
        temp_events = []
        while not self.controller_container_watcher_event_queue.empty():
            timestamp, event = self.controller_container_watcher_event_queue.get()
            print(f"[CONTROLLER] [STEP] Event: {event}")
            temp_events.append((timestamp, event))

        # Terminater on Torque limit, truncate on container issue
        if len(temp_events) > 0:
            assert len(temp_events) == 1, (
                f"At most one concurrent container event allowed, found {temp_events}"
            )
            self.is_running = False

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
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        after = time.time()
        info[f"dt_{self.n}"] = after - before
        return observation, reward, terminated, truncated, info


def observation_has_z_pressure(
    observation,
    error_threshold=0.005,
    previous_error_threshold=0.003,
    min_z_height=0.055,
):
    return (
        abs(observation["observation.error.cartesian"][2]) > error_threshold
        and abs(observation["observation.previous.error.cartesian"][2])
        > previous_error_threshold
        and observation["observation.state.cartesian"][2] < min_z_height
    )


def observation_has_z_pressure_or_below(
    observation,
    error_threshold=0.005,
    previous_error_threshold=0.003,
    min_z_height=0.055,
    terminate_z_height=0.0475,
):
    return (
        abs(observation["observation.error.cartesian"][2]) > error_threshold
        and abs(observation["observation.previous.error.cartesian"][2])
        > previous_error_threshold
        and observation["observation.state.cartesian"][2] < min_z_height
    ) or observation["observation.state.cartesian"][2] < terminate_z_height


class CLIWrapper(Wrapper):
    def __init__(self, env):  # , termination_fn):
        super().__init__(env)
        # self.termination_fn = termination_fn
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.ready_key = "r"
        self.environment_ready = False
        self.other_key = None
        self.other_keys_to_check = "sfub"

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # wait for s/f when gripper open
        # if self.termination_fn(observation):
        #     terminated = True
        #     print("Place successful? ([s]uccess/[f]ail)")
        #     while True:
        #         while self.other_key is None:
        #             time.sleep(0.05)
        #         if self.other_key not in "sfu":
        #             print(
        #                 f"[CLI] Ignoring {self.other_key}, waiting for whether the run was success."
        #             )
        #             self.other_key = None
        #             continue
        #         now = time.time()
        #         event = {"s": "E_SUCCESS", "f": "E_FAIL", "u": "E_ROLLOUT_UNUSABLE"}[
        #             self.other_key
        #         ]
        #         self.other_key = None

        #         print(f"[CLI] {event}")
        #         append_or_insert(info, "custom_events", (now, event))
        #         break
        # elif
        if truncated:
            time.sleep(2.0)
        if self.other_key is not None:
            if self.other_key == "u":
                truncated = True
                append_or_insert(
                    info, "custom_events", (time.time(), "E_ROLLOUT_UNUSABLE")
                )
            elif self.other_key == "b":
                terminated = True
                append_or_insert(info, "custom_events", (time.time(), "E_BAD_BEHAVIOR"))
            elif self.other_key == "s":
                terminated = True
                append_or_insert(info, "custom_events", (time.time(), "E_SUCCESS"))
            elif self.other_key == "f":
                terminated = True
                append_or_insert(info, "custom_events", (time.time(), "E_FAIL"))
            print(f"[CLI] processed {self.other_key}")
            self.other_key = None

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # wait for key "r"
        # print(f"[CLI] Waiting for ready")
        # while not self.environment_ready:
        #     time.sleep(0.1)
        self.environment_ready = False
        obs, info = self.env.reset(seed=seed, options=options)
        if self.other_key is not None and self.other_key != "u":
            print(f"[CLI] Resetting other_key on reset from {self.other_key}")
            self.other_key = None
        return obs, info

    def on_press(self, key):
        try:
            char = key.char
            if char == self.ready_key:
                self.environment_ready = True
            elif char is not None and char in self.other_keys_to_check:
                print(f"{key.char} is pressed!")
                self.other_key = key.char
        except AttributeError:
            pass  # Special keys like shift, ctrl, etc.

    def close(self):
        self.listener.stop()
        self.env.close()


class CustomTerminationWrapper(Wrapper):
    def __init__(self, env, termination_fn):
        super().__init__(env)
        self.termination_fn = termination_fn

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        termination = self.termination_fn(observation)
        if termination is not None:
            append_or_insert(info, "custom_events", (time.time(), termination))
            terminated = True

        return observation, reward, terminated, truncated, info


class SuccessClassificationWrapper(Wrapper):
    def __init__(self, env, args, sac_config: Config, threshold: float):
        super().__init__(env)
        classifier_dir = Path("checkpoints") / getattr(args, "load_policy", "")
        if classifier_dir is None:
            raise ValueError("SuccessClassificationWrapper requires args.load_policy.")

        self.threshold = threshold
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and sac_config.cuda else "cpu"
        )
        self.sac_config = sac_config
        self.max_val = -float("inf")

        self.shared_encoder = SharedEncoder(sac_config).to(self.device)
        self.shared_encoder.load_state_dict(
            torch.load(
                os.path.join(classifier_dir, "shared_encoder_state_dict.pth"),
                map_location=self.device,
            )
        )
        self.shared_encoder.eval()

        self.actor = Actor(sac_config).to(self.device)
        self.actor.load_state_dict(
            torch.load(
                os.path.join(classifier_dir, "actor_state_dict.pth"),
                map_location=self.device,
            )
        )
        self.actor.eval()

        self.q_functions = [
            SoftQNetwork(sac_config).to(self.device)
            for _ in range(self.sac_config.num_critics)
        ]
        for idx, qf in enumerate(self.q_functions):
            qf.load_state_dict(
                torch.load(
                    os.path.join(classifier_dir, f"qf{idx + 1}_state_dict.pth"),
                    map_location=self.device,
                )
            )
            qf.eval()

    def _actor_mean_action(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        h = self.actor.net(encoded_obs)
        mean = self.actor.fc_mean(h)
        squashed_mean = torch.tanh(mean)
        return squashed_mean * self.actor.action_scale + self.actor.action_bias

    def _success_score(self, observation: Any) -> float:
        if isinstance(observation, dict):
            if "observation.formatted" not in observation:
                raise KeyError(
                    "Observation dict must contain 'observation.formatted' for success classification."
                )
            obs_raw = observation["observation.formatted"]
        else:
            obs_raw = observation

        if isinstance(obs_raw, torch.Tensor):
            obs_tensor = obs_raw.to(self.device, dtype=torch.float32).view(1, -1)
        else:
            obs_tensor = torch.as_tensor(
                obs_raw, dtype=torch.float32, device=self.device
            ).view(1, -1)

        with torch.no_grad():
            encoded_obs = self.shared_encoder(obs_tensor)
            action = self._actor_mean_action(encoded_obs)
            q_values = torch.stack(
                [qf(encoded_obs, action).squeeze(-1) for qf in self.q_functions], dim=0
            )
            return float(q_values.mean().item())

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        score = self._success_score(observation)
        self.max_val = max(score, self.max_val)
        print(f"[CLASSIFIER SCORE]: {score:.3f}")
        if score >= self.threshold or self.max_val > 8 and score < 6:
            t = time.time()
            append_or_insert(info, "custom_events", (t, "E_SUCCESS"))
            append_or_insert(info, "custom_events", (t, "E_SUCCESS_CLS"))
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self, options=None, seed=None):
        v = self.env.reset(options=options, seed=seed)
        self.max_val = -float("inf")
        return v


class StepLimitEnforcerWrapper(Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action, block=False
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            action, block=block
        )
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        return observation, reward, terminated, truncated, info


class ObservationNormalizerWrapper(ObservationWrapper):
    def __init__(self, env, means: torch.Tensor, stds: torch.Tensor, device):
        super().__init__(env)
        self.means = means.to(device)
        self.stds = stds.to(device)

    def observation(self, observation):
        return (observation - self.means) / self.stds


class PoseEstimationEvalWrapper(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        z_step_size=0.00025 * 0.1,
        max_z_error_deviation=0.001 * 0.1,
        target_z_error=0.025 * 0.08,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        is_eval=False,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.z_step_size = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error = target_z_error
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=config.pose_estimation_assumed_orientation
        )
        self.pose_estimation_position_euler = np.array(
            config.demo_pose_estimation_euler
        )
        print("[InsertionWrapper] [__init__] Eval mode:", is_eval)

        self.delta_z_push_reset = 0.05 * 0.08
        self.delta_z_push_reset_careful_threshold_distance = 0.02 * 0.1
        self.delta_z_push_reset_careful_threshold_velocity = 0.003
        self.delta_z_push_reset_step_size = 0.004 * 0.2
        self.delta_z_contact_error = 0.03 * 0.08
        self.delta_z_contact_step_size = 0.002 * 0.3
        self.delta_z_contact_careful_threshold_velocity = 0.004
        self.delta_z_contact_careful_threshold_distance = 0.01 * 0.1
        self.delta_z_i_clipping = 0.01 * 0.1
        self.reset_lift_height = 0.01  # 0.035
        self.after_grasp_lift_height = 0.03  # 0.05

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if not self.first_reset:
            # lift up
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs, *_ = self.env.step(
                np.array([0.0, 0.0, self.reset_lift_height + delta_z])
            )
            time.sleep(0.2)

            # go to grasping position
            delta = (
                self.actual_grasp_position[:2]
                - self.obs["observation.state.cartesian"][:2]
            )
            for _ in range(10):
                self.obs, *_ = self.env.step(
                    np.array([delta[0] / 10.0, delta[1] / 10.0, -0.02 / 10]),
                )

            # push down
            while (
                abs(
                    self.obs["observation.state.cartesian"][2]
                    - self.obs["observation.state.target"][2]
                )
                < self.delta_z_push_reset
            ):
                delta_xy = (
                    self.actual_grasp_position[0:2]
                    - self.obs["observation.state.cartesian"][0:2]
                )
                delta_z = (
                    -self.delta_z_push_reset_step_size
                    if self.obs["observation.velocity.cartesian"][2]
                    > -self.delta_z_push_reset_careful_threshold_velocity
                    or abs(
                        self.actual_grasp_position[2]
                        - self.obs["observation.state.cartesian"][2]
                    )
                    > self.delta_z_push_reset_careful_threshold_distance
                    else 0.0
                )
                self.obs, *_ = self.env.step(
                    np.array([delta_xy[0], delta_xy[1], delta_z])
                )

            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs, *_ = self.env.step(np.array([0.0, 0.0, delta_z * 0.8]))
            self.env.unwrapped.gripper.home()  # type: ignore
            time.sleep(0.5)
            self.n_since_last_home += 1
        if self.n_since_last_home >= 4 or self.first_reset:
            print(
                f"self.n_since_last_home={self.n_since_last_home}, first_reset={self.first_reset}, homing..."
            )
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
            self.n_since_last_home = 0
            self.first_reset = False

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        self.target_grasp_position = np.copy(self.grasp_position_ground_truth)

        grasp_randomisation_x = np.random.uniform(
            self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
        )
        grasp_randomisation_z = np.random.uniform(
            self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
        )

        self.target_grasp_position[0] += grasp_randomisation_x
        self.target_grasp_position[2] += grasp_randomisation_z

        print("Moving to grasp position...")
        for _ in range(15):
            delta = (
                self.target_grasp_position - self.obs["observation.state.cartesian"][:3]
            )
            self.obs, *_ = self.env.step(np.array([delta[0], delta[1], delta[2]]))
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(1.0)
        self.obs, *_ = self.env.step(np.zeros(3))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        # pick up quickly
        print("Picking up...")
        self.obs, *_ = self.env.step(np.array([0.0, 0.0, self.after_grasp_lift_height]))

        # compute goal position
        # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position
        print("Moving to pose estimation position...")

        for _ in range(10):
            delta = (
                self.pose_estimation_position_euler[0:3]
                - self.obs["observation.state.cartesian"][0:3]
            )
            self.obs, *_ = self.env.step(delta / 10.0)

        time.sleep(0.5)
        print("Refining pose estimation position...")

        self.obs, *_ = self.env.step(np.zeros(3))
        for _ in range(10):
            delta = (
                self.pose_estimation_position_euler[0:3]
                - self.obs["observation.state.cartesian"][0:3]
            )
            print(f"Compensating for {delta}")
            self.obs, *_ = self.env.step(
                np.clip(delta / 2.0, -self.delta_z_i_clipping, self.delta_z_i_clipping)
            )
        time.sleep(0.5)
        print("Finalizing pose estimation position...")

        self.obs, *_ = self.env.step(np.zeros(3))
        self.actual_estimation_position = np.copy(
            self.obs["observation.state.cartesian"]
        )
        pose_estimation_joint_state = np.copy(self.obs["observation.state.joints"])
        print("Estimating pose...")
        lavender_pose, purple_pose = (
            self.pose_estimation_helper.estimate_two_lego_bricks_absolute(
                self.obs["observation.images.wrist_camera"],
                self.obs["observation.images.wrist_depth_camera"],
                self.actual_estimation_position,
            )
        )

        # raise NotImplementedError("Check this formula again carefully!")
        # e = estiation, g = goal, d = demo, i = inference

        predicted_goal_position = (
            self.actual_estimation_position[:3]
            + purple_pose[:3, 3]
            - lavender_pose[:3, 3]
        )

        self.obs, reset_info = self.env.reset()

        print("Reset complete.")
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )
        adjusted_ground_truth_goal_position = (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        goal_position_error = (
            predicted_goal_position - adjusted_ground_truth_goal_position
        )
        # Populate reset info
        reset_info["pose_estimation.lavender"] = lavender_pose
        reset_info["pose_estimation.purple"] = purple_pose
        reset_info["pose_estimation.joint_state"] = pose_estimation_joint_state
        reset_info["pose_estimation.cartesian"] = self.actual_estimation_position
        reset_info["pose_estimation.predicted_goal_position"] = predicted_goal_position
        reset_info["pose_estimation.ground_truth_goal_position"] = (
            adjusted_ground_truth_goal_position
        )
        reset_info["pose_estimation.goal_position_error"] = goal_position_error
        reset_info["pose_estimation.grasp_delta"] = self.reset_grasp_delta
        reset_info["pose_estimation.actual_grasp_position"] = self.actual_grasp_position

        print(f"[PoseEstimationEvalWrapper] Goal position error: {goal_position_error}")
        print(
            f"[PoseEstimationEvalWrapper] Error norm: {np.linalg.norm(goal_position_error[:2]):.6f} (xy), {abs(goal_position_error[2]):.6f} (z)"
        )

        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Step is not needed for pose estimation evaluation, just pass through."""
        return self.env.step(action)


def random_point_in_ellipse(x_range, y_range):
    rho = np.random.random()
    phi = np.random.random() * 2 * np.pi
    x = np.sqrt(rho) * np.cos(phi)
    y = np.sqrt(rho) * np.sin(phi)
    width = x_range[1] - x_range[0]
    x_center = (x_range[1] + x_range[0]) / 2.0
    height = y_range[1] - y_range[0]
    y_center = (y_range[1] + y_range[0]) / 2.0
    x = x * width / 2.0 + x_center
    y = y * height / 2.0 + y_center
    return np.array([x, y])


class InsertionWrapperSim(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        grasp_randomisation_mode="box",
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        z_step_size=0.00025,
        max_z_error_deviation=0.001,
        target_z_error=0.0025,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        is_eval=False,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = np.array([0.6, 0.0, 0.0])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode = grasp_randomisation_mode
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.z_step_size = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error = target_z_error
        self.step_limit = step_limit
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        print("[InsertionWrapperSim] [__init__] Eval mode:", is_eval)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.grasp_randomisation_mode == "box":
            grasp_randomisation_x = np.random.uniform(
                self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
            )
            grasp_randomisation_z = np.random.uniform(
                self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
            )
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_randomisation_x, grasp_randomisation_z = random_point_in_ellipse(
                self.grasp_randomisation_x_range,
                self.grasp_randomisation_z_range,
            )

        self.grasp_position = np.array(
            [grasp_randomisation_x, 0.0, grasp_randomisation_z]
        )

        self.goal_position = np.copy(self.goal_position_ground_truth)
        goal_position_randomisation_xy = np.zeros(2)
        while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
            goal_position_randomisation_xy = np.random.uniform(
                self.goal_position_randomisation_xy_range[0],
                self.goal_position_randomisation_xy_range[1],
                size=2,
            )

        self.goal_position[:2] += goal_position_randomisation_xy
        self.goal_position[0] += self.grasp_position[0]  # ground truth is at 0
        self.goal_position[2] += self.grasp_position[2]  # ground truth is at 0

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
        else:
            self.start_position = self.goal_position.copy()

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={
                "start_position": self.start_position,
                "grasp_position": self.grasp_position,
            },
        )

        reset_info["reset.grasped.delta"] = self.grasp_position
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.grasp_position
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.array([action[0], action[1], 0.0, 0.0, 0.0, 0.0])

        z_error = self.obs["observation.error.cartesian"][2]
        if z_error > -self.target_z_error + self.max_z_error_deviation:
            action[2] -= self.z_step_size
        elif z_error < -self.target_z_error - self.max_z_error_deviation:
            action[2] += self.z_step_size

        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action[:2] = delta_xy * self.safety_box_step_size / norm_xy

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.grasp_position
        )
        return obs


class InsertionWrapperSim3DoFRotZ(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(
            0.0005,
            0.002,
        ),  # ry is not needed as the object is assumed to lie on a flat surface initially
        grasp_randomisation_mode="box",
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        goal_orientation_randomisation_angle=np.deg2rad(3),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        safety_box_angular_radius=np.deg2rad(6),
        safety_box_angular_step_size=np.deg2rad(1),
        z_step_size=0.00025,
        max_z_error_deviation=0.001,
        target_z_error=0.0025,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        minimal_start_goal_angle=np.deg2rad(6),
        is_eval=False,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = np.array([0.6, 0.0, 0.0])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode = grasp_randomisation_mode
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.goal_orientation_randomisation_angle = goal_orientation_randomisation_angle
        self.safety_box_radius = safety_box_radius
        self.safety_box_angular_radius = safety_box_angular_radius
        self.safety_box_step_size = safety_box_step_size
        self.safety_box_angular_step_size = safety_box_angular_step_size
        self.z_step_size = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error = target_z_error
        self.step_limit = step_limit
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.minimal_start_goal_angle = minimal_start_goal_angle
        self.is_eval = is_eval
        print("[InsertionWrapperSim] [__init__] Eval mode:", is_eval)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.grasp_randomisation_mode == "box":
            grasp_randomisation_x = np.random.uniform(
                self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
            )
            grasp_randomisation_z = np.random.uniform(
                self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
            )
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_randomisation_x, grasp_randomisation_z = random_point_in_ellipse(
                self.grasp_randomisation_x_range,
                self.grasp_randomisation_z_range,
            )

        self.grasp_position = np.array(
            [grasp_randomisation_x, 0.0, grasp_randomisation_z]
        )

        self.goal_position = np.copy(self.goal_position_ground_truth)
        goal_position_randomisation_xy = np.zeros(2)
        while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
            goal_position_randomisation_xy = np.random.uniform(
                self.goal_position_randomisation_xy_range[0],
                self.goal_position_randomisation_xy_range[1],
                size=2,
            )

        goal_position_randomisation_rz = 0.0
        while (
            np.abs(goal_position_randomisation_rz) < 0.1 * self.minimal_start_goal_angle
        ):
            goal_position_randomisation_rz = np.random.uniform(
                -self.goal_orientation_randomisation_angle,
                self.goal_orientation_randomisation_angle,
            )

        self.goal_position[:2] += goal_position_randomisation_xy
        self.goal_position[0] += self.grasp_position[0]  # ground truth is at 0
        self.goal_position[2] += self.grasp_position[2]  # ground truth is at 0
        self.goal_rotation_z = goal_position_randomisation_rz

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            self.start_rotation_z = 0.0
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            while np.abs(self.start_rotation_z) < self.minimal_start_goal_angle:
                self.start_rotation_z = self.goal_rotation_z + np.random.uniform(
                    -self.safety_box_angular_radius, self.safety_box_angular_radius
                )
        else:
            self.start_position = self.goal_position.copy()
            self.start_rotation_z = self.goal_rotation_z

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={
                "start_position": self.start_position,
                "start_so3": np.array([0.0, 0.0, self.start_rotation_z]),
                "grasp_position": self.grasp_position,
            },
        )

        reset_info["reset.grasped.delta"] = self.grasp_position
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.grasp_position
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        reset_info["reset.goal_orientation.rotation_z"] = self.goal_rotation_z
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.array([action[0], action[1], 0.0, 0.0, 0.0, action[2]])
        # print(f"z-error: {self.obs['observation.state.target'][2]}")

        z_error = self.obs["observation.error.cartesian"][2]
        if z_error > -self.target_z_error + self.max_z_error_deviation:
            action[2] -= self.z_step_size
        elif z_error < -self.target_z_error - self.max_z_error_deviation:
            action[2] += self.z_step_size

        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action[:2] = delta_xy * self.safety_box_step_size / norm_xy

        # check for rotation: project current rotation error onto rotation axis; if angle is larger than safety_box_angular_radius, apply angular action towards goal orientation
        current_rotation_z = self.obs["observation.state.cartesian"][5]
        rotation_z_error = self.goal_rotation_z - current_rotation_z
        rotation_z_error = (rotation_z_error + np.pi) % (
            2 * np.pi
        ) - np.pi  # wrap to [-pi, pi]
        if abs(rotation_z_error) > self.safety_box_angular_radius:
            action[5] = np.sign(rotation_z_error) * self.safety_box_angular_step_size

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.grasp_position
        )
        obs["observation.perfect_rotation"] = -obs["observation.state.cartesian"][3:]
        return obs


def random_axis():
    v = np.random.normal(size=3)  # Sample from N(0,1)
    v /= np.linalg.norm(v)  # Normalize to unit length
    return v


def axis_angle_from_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(mat) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    if theta < 1e-8:
        return np.zeros(3)

    axis = np.array(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ]
    )
    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-8:
        return np.zeros(3)

    axis = axis / denom
    return axis * theta


def rotation_matrix_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3)

    axis = axis_angle / theta
    kx, ky, kz = axis
    k = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )

    identity = np.eye(3)
    return identity + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


class InsertionWrapperSim5DoF(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(
            0.0005,
            0.002,
        ),  # ry is not needed as the object is assumed to lie on a flat surface initially
        grasp_randomisation_mode="box",
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        goal_orientation_randomisation_angle=np.deg2rad(3),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        safety_box_angular_radius=np.deg2rad(6),
        safety_box_angular_step_size=np.deg2rad(1),
        z_step_size=0.00025,
        max_z_error_deviation=0.001,
        target_z_error=0.0025,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        minimal_start_goal_angle=np.deg2rad(6),
        is_eval=False,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = np.array([0.6, 0.0, 0.0])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode = grasp_randomisation_mode
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.goal_orientation_randomisation_angle = goal_orientation_randomisation_angle
        self.safety_box_radius = safety_box_radius
        self.safety_box_angular_radius = safety_box_angular_radius
        self.safety_box_step_size = safety_box_step_size
        self.safety_box_angular_step_size = safety_box_angular_step_size
        self.z_step_size = z_step_size
        self.max_z_error_deviation = max_z_error_deviation
        self.target_z_error = target_z_error
        self.step_limit = step_limit
        self.action_space = spaces.Box(-np.inf, np.inf, (5,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.minimal_start_goal_angle = minimal_start_goal_angle
        self.is_eval = is_eval
        print("[InsertionWrapperSim] [__init__] Eval mode:", is_eval)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.grasp_randomisation_mode == "box":
            grasp_randomisation_x = np.random.uniform(
                self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
            )
            grasp_randomisation_z = np.random.uniform(
                self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
            )
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_randomisation_x, grasp_randomisation_z = random_point_in_ellipse(
                self.grasp_randomisation_x_range,
                self.grasp_randomisation_z_range,
            )

        self.grasp_position = np.array(
            [grasp_randomisation_x, 0.0, grasp_randomisation_z]
        )

        self.goal_position = np.copy(self.goal_position_ground_truth)
        goal_position_randomisation_xy = np.zeros(2)
        while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
            goal_position_randomisation_xy = np.random.uniform(
                self.goal_position_randomisation_xy_range[0],
                self.goal_position_randomisation_xy_range[1],
                size=2,
            )
        # -> gererate random axis, generate random angle > min angle
        goal_orientation_randomisation_axis = random_axis()
        goal_orientation_randomisation_angle = 0.0
        while (
            np.abs(goal_orientation_randomisation_angle)
            < 0.1 * self.minimal_start_goal_angle
        ):
            goal_orientation_randomisation_angle = np.random.uniform(
                -self.goal_orientation_randomisation_angle,
                self.goal_orientation_randomisation_angle,
            )

        self.goal_position[:2] += goal_position_randomisation_xy
        self.goal_position[0] += self.grasp_position[0]  # ground truth is at 0
        self.goal_position[2] += self.grasp_position[2]  # ground truth is at 0
        self.goal_orientation = (
            goal_orientation_randomisation_angle * goal_orientation_randomisation_axis
        )

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            # start orientation: start from goal orientation -> add another random rotation on top -> until angle to gt_goal > min angle
            self.start_orientation = np.zeros(3)
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            while (
                np.linalg.norm(self.start_orientation) < self.minimal_start_goal_angle
            ):
                # axis_angle_from_R(R_from_axis_angle(random_axis_angle) @ R_from_axis_angle(goal_orientation))
                start_orientation_randomisation_angle = np.random.uniform(
                    -self.safety_box_angular_radius, self.safety_box_angular_radius
                )

                self.start_orientation = axis_angle_from_rotation_matrix(
                    rotation_matrix_from_axis_angle(
                        random_axis() * start_orientation_randomisation_angle
                    )
                    @ rotation_matrix_from_axis_angle(self.goal_orientation)
                )
        else:
            self.start_position = self.goal_position.copy()
            self.start_orientation = self.goal_orientation.copy()

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={
                "start_position": self.start_position,
                "start_so3": self.start_orientation,
                "grasp_position": self.grasp_position,
            },
        )

        reset_info["reset.grasped.delta"] = self.grasp_position
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.grasp_position
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        reset_info["reset.goal_orientation"] = self.goal_orientation
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.array([action[0], action[1], 0.0, action[2], action[3], action[4]])
        # print(f"z-error: {self.obs['observation.state.target'][2]}")

        z_error = self.obs["observation.error.cartesian"][2]
        if z_error > -self.target_z_error + self.max_z_error_deviation:
            action[2] -= self.z_step_size
        elif z_error < -self.target_z_error - self.max_z_error_deviation:
            action[2] += self.z_step_size

        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action[:2] = delta_xy * self.safety_box_step_size / norm_xy

        # check for rotation: if angle is larger than safety_box_angular_radius, apply angular action towards goal orientation
        current_orientation = self.obs["observation.state.cartesian"][3:6]
        angle_error = np.linalg.norm(current_orientation)
        if angle_error > self.safety_box_angular_radius:
            action[3:6] = (
                -current_orientation / angle_error * self.safety_box_angular_step_size
            )

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.grasp_position
        )
        obs["observation.perfect_rotation"] = -obs["observation.state.cartesian"][3:]
        return obs


class InsertionWrapperSim3D(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        grasp_randomisation_mode="box",
        goal_position_randomisation_xyz_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_height=0.005,
        safety_box_step_size=0.0005,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        is_eval=False,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = np.array([0.6, 0.0, 0.134])
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.grasp_randomisation_mode = grasp_randomisation_mode
        self.sbox_center_randomisation_xyz_range = goal_position_randomisation_xyz_range
        self.sbox_radius_xy = safety_box_radius
        self.sbox_height_z = safety_box_height
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        print("[InsertionWrapperSim] [__init__] Eval mode:", is_eval)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.grasp_randomisation_mode == "box":
            grasp_randomisation_x = np.random.uniform(
                self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
            )
            grasp_randomisation_z = np.random.uniform(
                self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
            )
        elif self.grasp_randomisation_mode == "ellipse":
            grasp_randomisation_x, grasp_randomisation_z = random_point_in_ellipse(
                self.grasp_randomisation_x_range,
                self.grasp_randomisation_z_range,
            )

        self.grasp_position = np.array(
            [grasp_randomisation_x, 0.0, grasp_randomisation_z]  # pyright: ignore[reportPossiblyUnboundVariable]
        )

        self.sbox_center = np.copy(self.goal_position_ground_truth)
        sbox_center_randomisation_xyz = np.zeros(2)
        while np.linalg.norm(sbox_center_randomisation_xyz) < 0.001:
            sbox_center_randomisation_xyz = np.random.uniform(
                self.sbox_center_randomisation_xyz_range[0],
                self.sbox_center_randomisation_xyz_range[1],
                size=3,
            )

        self.sbox_center += sbox_center_randomisation_xyz
        # self.sbox_center[2] += 0.002  # to compensate for stud height
        self.sbox_center[0] += self.grasp_position[0]  # ground truth is at 0
        self.sbox_center[2] += self.grasp_position[2]  # ground truth is at 0
        print(f"[3D reset] sbox center at {self.sbox_center}")

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            while (
                np.linalg.norm(
                    self.start_position[:3] - self.goal_position_ground_truth[:3]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.sbox_center[:2] + np.random.uniform(
                    -self.sbox_radius_xy, self.sbox_radius_xy, size=2
                )
                self.start_position[2] = self.sbox_center[2] + np.random.uniform(
                    0.0, self.sbox_height_z
                )
        else:
            self.start_position = self.sbox_center.copy()
        print(f"[3D reset] start position at {self.start_position}")

        self.obs, reset_info = self.env.reset(
            seed=seed,
            options={
                "start_position": self.start_position,
                "grasp_position": self.grasp_position,
            },
        )

        reset_info["reset.grasped.delta"] = self.grasp_position
        goal_position_offset = self.sbox_center - (
            self.goal_position_ground_truth + self.grasp_position
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        self.obs = self.add_perfect_action_to_obs(self.obs)
        self.n_steps = 0
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action = np.array([action[0], action[1], action[2], 1.0, 0.0, 0.0, 0.0])

        # current_pos_xyz = self.obs["observation.state.cartesian"][:3]
        # delta_xyz = self.goal_position[:3] - current_pos_xyz
        # norm_xyz = np.linalg.norm(delta_xyz)
        # if norm_xyz > self.safety_box_radius:
        #     action[:3] = delta_xyz * self.safety_box_step_size / norm_xyz

        # apply safety box
        current_pos_xyz = self.obs["observation.state.cartesian"][:3]
        delta_xyz = current_pos_xyz - self.sbox_center[:3]
        delta_xyz_clipped = np.clip(
            delta_xyz,
            [-self.sbox_radius_xy, -self.sbox_radius_xy, -self.sbox_height_z],
            [self.sbox_radius_xy, self.sbox_radius_xy, self.sbox_height_z],
        )
        if np.any(delta_xyz_clipped != delta_xyz):
            correcting_action = delta_xyz_clipped - delta_xyz
            action[:3] = correcting_action

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + self.grasp_position
        ) - obs["observation.state.cartesian"][:3]
        return obs
