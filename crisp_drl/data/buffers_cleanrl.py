# Copyright notice
#
# This file contains code adapted from cleanrl
# (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/buffers.py)
# licensed under the MIT License.
#
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from __future__ import annotations

from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from joblib import dump, load

from crisp_drl.data.utils import (
    crisp_batch_concat_obs_to_tensor,
    crisp_batch_obs_to_tensor,
)

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ReplayBufferSamplesWithPerfectActions(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    perfect_actions: th.Tensor
    next_observations: th.Tensor
    next_perfect_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
            if key
            not in ["observation.state.joint", "task", "observation.state.target"]
        }  # type: ignore[misc]
    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


# Deprecated function, use joblib.load instead
# def load_buffer_from_file(path: str, image_encoders) -> BaseBuffer:


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        image_encoders: list[th.nn.Module],
        action_space: spaces.Space,
        device: th.device | str = "auto",
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.image_encoders = image_encoders
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray
    ) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array, copy: bool = True) -> th.Tensor:
        """
        Convert buffer data array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """

        if copy:
            return th.tensor(array, device=self.device)
        else:
            return th.as_tensor(array, device=self.device)

    def obs_to_torch(self, array, copy: bool = True) -> th.Tensor:
        """
        Convert buffer data array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """

        if len(self.image_encoders) > 0:
            return crisp_batch_concat_obs_to_tensor(
                array, self.image_encoders, self.device
            )
        if copy:
            return th.tensor(array, device=self.device)
        else:
            return th.as_tensor(array, device=self.device)

    def save_buffer(self, path: str) -> None:
        """
        Save the buffer to a file.

        :param path: Path to the file
        """
        if not path.endswith(".joblib"):
            path = f"{path}/replay_buffer.joblib"
        dump(self, path)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param image_encoders: list of projection image encoders
    :param action_space: Action space
    :param device: PyTorch device
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        image_encoders: list[th.nn.Module],
        action_space: spaces.Space,
        device: th.device | str = "auto",
    ):
        super().__init__(
            buffer_size, observation_space, image_encoders, action_space, device
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros(
            (self.buffer_size, *self.obs_shape), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.buffer_size, *self.obs_shape), dtype=np.float32
        )

        self.actions = np.zeros(
            (self.buffer_size, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.terminateds.nbytes
            )

            total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add_rollout(
        self,
        obs: np.ndarray,  # shape B+1, D_OBS
        action: np.ndarray,  # shape B, D_ACT
        reward: np.ndarray,  # shape B,
        terminated: bool,
    ) -> None:
        if isinstance(obs, th.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(action, th.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, th.Tensor):
            reward = reward.cpu().numpy()
        batch_size, *_ = action.shape
        # First handle before runover
        if self.pos + batch_size > self.buffer_size:
            self.observations[self.pos :] = obs[: self.buffer_size - self.pos]
            self.next_observations[self.pos :] = obs[
                1 : self.buffer_size - self.pos + 1
            ]
            self.actions[self.pos :] = action[: self.buffer_size - self.pos]
            self.rewards[self.pos :] = reward[: self.buffer_size - self.pos]
            self.terminateds[self.pos :] = False

            obs = obs[self.buffer_size - self.pos :]
            action = action[self.buffer_size - self.pos :]
            reward = reward[self.buffer_size - self.pos :]

            batch_size -= self.buffer_size - self.pos
            self.pos = 0
            self.full = True

        self.observations[self.pos : self.pos + batch_size] = obs[:-1]
        self.next_observations[self.pos : self.pos + batch_size] = obs[1:]
        self.actions[self.pos : self.pos + batch_size] = action
        self.rewards[self.pos : self.pos + batch_size] = reward
        self.terminateds[self.pos : self.pos + batch_size - 1] = False
        self.terminateds[self.pos + batch_size - 1] = terminated

        self.pos += batch_size
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            self.obs_to_torch(self.observations[batch_inds, :]),
            self.to_torch(self.actions[batch_inds, :]),
            self.obs_to_torch(self.next_observations[batch_inds, :]),
            self.to_torch(self.terminateds[batch_inds].reshape(-1, 1)),
            self.to_torch(self.rewards[batch_inds].reshape(-1, 1)),
        )

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class ReplayBufferGpu:
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    """

    observation_dim: int
    obs_shape: tuple[int, ...]

    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor
    terminateds: th.Tensor
    timeouts: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_space: spaces.Space,
        n_step_return: int = 1,
        gamma: float = 0.99,
        device: th.device | str = "auto",
    ):
        self.buffer_size = max(buffer_size, 1)
        self.observation_dim = observation_dim
        self.action_space = action_space
        self.obs_shape = (observation_dim,)
        self.action_dim = get_action_dim(action_space)
        self.pos_obs = 0
        self.pos_acts = 0
        self.device = get_device(device)
        self.valid_indices_act = th.zeros(
            self.buffer_size, dtype=th.int32, device=self.device
        )
        self.valid_indices_obs = th.zeros(
            self.buffer_size, dtype=th.int32, device=self.device
        )
        self.num_valid = 0

        self.n_step_return = n_step_return
        self.gamma = gamma

        self.observations = th.zeros(
            (self.buffer_size, self.observation_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.actions = th.zeros(
            (self.buffer_size, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.rewards = th.zeros(
            (self.buffer_size), dtype=th.float32, device=self.device
        )
        self.terminateds = th.zeros(
            (self.buffer_size), dtype=th.float32, device=self.device
        )

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.terminateds.nbytes
            )
            mem_available = psutil.virtual_memory().available

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.pos_obs

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos_obs = 0
        self.num_valid = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        :param batch_size: Number of element to sample
        :return:
        """
        batch_inds = th.randint(0, self.num_valid, (batch_size,))
        return self._get_samples(batch_inds)

    def save_buffer(self, path: str) -> None:
        """
        Save the buffer to a file.

        :param path: Path to the file
        """
        if not path.endswith(".joblib"):
            path = f"{path}/replay_buffer.joblib"
        dump(self, path)

    def add_rollout(
        self,
        obs: list[th.Tensor],  # shape B+1, D_OBS
        action: list[th.Tensor],  # shape B, D_ACT
        reward: list[float],  # shape B,
        terminated: bool,
    ) -> None:
        assert all(isinstance(o, th.Tensor) for o in obs), (
            "obs should be a list of torch Tensors"
        )
        assert all(isinstance(a, th.Tensor) for a in action), (
            "action should be a list of torch Tensors"
        )
        assert all(isinstance(r, float) for r in reward), (
            "reward should be a list of floats"
        )

        obs: th.Tensor = th.stack(obs).to(self.device)
        action: th.Tensor = th.stack(action).to(self.device)
        reward: th.Tensor = th.tensor(reward, dtype=th.float32, device=self.device)

        batch_size_act, *_ = action.shape
        batch_size_obs, *_ = obs.shape
        # First handle before runover
        assert self.pos_obs + batch_size_obs <= self.buffer_size, (
            "Buffer overflow not handled yet"
        )

        self.observations[self.pos_obs : self.pos_obs + batch_size_obs] = obs
        self.actions[self.pos_acts : self.pos_acts + batch_size_act] = action
        self.rewards[self.pos_acts : self.pos_acts + batch_size_act] = reward
        self.terminateds[self.pos_obs : self.pos_obs + batch_size_obs] = False
        self.terminateds[self.pos_obs + batch_size_obs - 1] = terminated

        for n in range(0, batch_size_act - self.n_step_return + 1):
            self.valid_indices_act[self.num_valid] = self.pos_acts + n
            self.valid_indices_obs[self.num_valid] = self.pos_obs + n
            self.num_valid += 1

        self.pos_obs += batch_size_obs
        self.pos_acts += batch_size_act

    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples:
        batch_inds_act = self.valid_indices_act[batch_inds]
        batch_inds_obs = self.valid_indices_obs[batch_inds]
        rew_sum = th.clone(self.rewards[batch_inds_act])
        for n in range(1, self.n_step_return):
            rew_sum += self.rewards[batch_inds_act + n] * (self.gamma**n)
        return ReplayBufferSamples(
            self.observations[batch_inds_obs, :],
            self.actions[batch_inds_act, :],
            self.observations[batch_inds_obs + self.n_step_return, :],
            self.terminateds[batch_inds_obs + self.n_step_return].reshape(-1, 1),
            rew_sum.reshape(-1, 1),
        )


class ReplayBufferGpuWithPerfectActions:
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    """

    observation_dim: int
    obs_shape: tuple[int, ...]

    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: th.Tensor
    actions: th.Tensor
    perfect_actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor
    terminateds: th.Tensor
    timeouts: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_space: spaces.Space,
        n_step_return: int = 1,
        gamma: float = 0.99,
        device: th.device | str = "auto",
    ):
        self.buffer_size = max(buffer_size, 1)
        self.observation_dim = observation_dim
        self.action_space = action_space
        self.obs_shape = (observation_dim,)
        self.action_dim = get_action_dim(action_space)
        self.pos_obs = 0
        self.pos_acts = 0
        self.device = get_device(device)
        self.valid_indices_act = th.zeros(
            self.buffer_size, dtype=th.int32, device=self.device
        )
        self.valid_indices_obs = th.zeros(
            self.buffer_size, dtype=th.int32, device=self.device
        )
        self.num_valid = 0

        self.n_step_return = n_step_return
        self.gamma = gamma

        self.observations = th.zeros(
            (self.buffer_size, self.observation_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.actions = th.zeros(
            (self.buffer_size, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.perfect_actions = th.zeros(
            (self.buffer_size, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.rewards = th.zeros(
            (self.buffer_size), dtype=th.float32, device=self.device
        )
        self.terminateds = th.zeros(
            (self.buffer_size), dtype=th.float32, device=self.device
        )

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.perfect_actions.nbytes
                + self.rewards.nbytes
                + self.terminateds.nbytes
            )
            mem_available = psutil.virtual_memory().available

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.pos_obs

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos_obs = 0
        self.num_valid = 0

    def sample(self, batch_size: int) -> ReplayBufferSamplesWithPerfectActions:
        """
        :param batch_size: Number of element to sample
        :return:
        """
        batch_inds = th.randint(0, self.num_valid, (batch_size,))
        return self._get_samples(batch_inds)

    def save_buffer(self, path: str | Path) -> None:
        """
        Save the buffer to a file.

        :param path: Path to the file
        """
        if not str(path).endswith(".joblib"):
            path = Path(path) / "replay_buffer.joblib"
        dump(self, path)

    def add_rollout(
        self,
        obs: list[th.Tensor],  # shape B+1, D_OBS
        action: list[th.Tensor],  # shape B, D_ACT
        perfect_action: list[th.Tensor],  # shape B, D_ACT
        reward: list[float],  # shape B,
        terminated: bool,
    ) -> None:
        assert all(isinstance(o, th.Tensor) for o in obs), (
            "obs should be a list of torch Tensors"
        )
        assert all(isinstance(a, th.Tensor) for a in action), (
            "action should be a list of torch Tensors"
        )
        assert all(isinstance(p, th.Tensor) for p in perfect_action), (
            "perfect_action should be a list of torch Tensors"
        )
        assert all(isinstance(r, float) for r in reward), (
            "reward should be a list of floats"
        )

        obs: th.Tensor = th.stack(obs).to(self.device)
        action: th.Tensor = th.stack(action).to(self.device)
        perfect_action: th.Tensor = th.stack(perfect_action).to(self.device)
        reward: th.Tensor = th.tensor(reward, dtype=th.float32, device=self.device)

        batch_size_act, *_ = action.shape
        batch_size_obs, *_ = obs.shape
        # First handle before runover
        assert self.pos_obs + batch_size_obs <= self.buffer_size, (
            "Buffer overflow not handled yet"
        )

        self.observations[self.pos_obs : self.pos_obs + batch_size_obs] = obs
        self.actions[self.pos_acts : self.pos_acts + batch_size_act] = action
        self.perfect_actions[self.pos_acts : self.pos_acts + batch_size_act] = (
            perfect_action
        )
        self.rewards[self.pos_acts : self.pos_acts + batch_size_act] = reward
        self.terminateds[self.pos_obs : self.pos_obs + batch_size_obs] = False
        self.terminateds[self.pos_obs + batch_size_obs - 1] = terminated

        for n in range(0, batch_size_act - self.n_step_return + 1):
            self.valid_indices_act[self.num_valid] = self.pos_acts + n
            self.valid_indices_obs[self.num_valid] = self.pos_obs + n
            self.num_valid += 1

        self.pos_obs += batch_size_obs
        self.pos_acts += batch_size_act

    def change_n_steps(self, new_n_steps: int) -> None:
        """
        Change the number of steps to use for n-step returns.

        :param new_n_steps: New number of steps
        """
        assert new_n_steps > self.n_step_return, "Only increasing n_steps is supported"
        # Adjust valid indices by removing the last ones that are no longer valid
        valid_indices_act = self.valid_indices_act[: self.num_valid].tolist()
        valid_indices_obs = self.valid_indices_obs[: self.num_valid].tolist()
        i_old = 0
        i_new = 0
        while i_old < len(valid_indices_act) - (new_n_steps - self.n_step_return):
            if (
                valid_indices_act[i_old] + new_n_steps - self.n_step_return
                <= valid_indices_act[i_old + new_n_steps - self.n_step_return]
            ):
                self.valid_indices_act[i_new] = valid_indices_act[i_old]
                self.valid_indices_obs[i_new] = valid_indices_obs[i_old]
                i_new += 1
            i_old += 1

        self.num_valid = i_new

        self.n_step_return = new_n_steps

    def _get_samples(
        self, batch_inds: th.Tensor
    ) -> ReplayBufferSamplesWithPerfectActions:
        batch_inds_act = self.valid_indices_act[batch_inds]
        batch_inds_obs = self.valid_indices_obs[batch_inds]
        rew_sum = th.clone(self.rewards[batch_inds_act])
        for n in range(1, self.n_step_return):
            rew_sum += self.rewards[batch_inds_act + n] * (self.gamma**n)
        return ReplayBufferSamplesWithPerfectActions(
            self.observations[batch_inds_obs],
            self.actions[batch_inds_act],
            self.perfect_actions[batch_inds_act],
            self.observations[batch_inds_obs + self.n_step_return],
            self.perfect_actions[
                batch_inds_act + self.n_step_return
            ],  # validity: only used in non-terminal states
            self.terminateds[batch_inds_obs + self.n_step_return].reshape(-1, 1),
            rew_sum.reshape(-1, 1),
        )
