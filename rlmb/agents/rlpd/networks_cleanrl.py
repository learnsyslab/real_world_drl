# Copyright notice
#
# This file contains code adapted from cleanrl
# (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.spaces import Dict, Box

from rlmb.data.utils import get_input_size_from_dict_space


class SoftQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        if isinstance(observation_space, Dict):
            obs_size = get_input_size_from_dict_space(observation_space)
        elif isinstance(observation_space, Box):
            obs_size = observation_space.shape[0]
        else:
            raise NotImplementedError("Observation space type not supported")
        self.fc1 = nn.Linear(
            obs_size + np.prod(action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super().__init__()
        if isinstance(observation_space, Dict):
            input_size = get_input_size_from_dict_space(observation_space)
        elif isinstance(observation_space, Box):
            input_size = observation_space.shape[0]
        else:
            raise NotImplementedError("Observation space type not supported")
        self.config = config
        self.is_gymnasium_env = self.config.env_name in list(gym.envs.registry.keys())
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        if self.config.max_action:
            scale = self.config.max_action
            bias = 0.0
        else:
            scale = (action_space.high - action_space.low) / 2.0
            bias = (action_space.high + action_space.low) / 2.0
            
        self.register_buffer(
            "action_scale",
            torch.tensor(
                scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                bias,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
