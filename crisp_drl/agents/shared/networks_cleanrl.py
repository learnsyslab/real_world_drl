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
import torch.distributions as D

from crisp_drl.agents.shared.algorithm_config import Config


class SoftQNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        obs_size = (
            config.actor_nonvision_input_dim
            + config.vision_head_output_dim * config.n_cameras
        )

        self.fc1 = nn.Linear(
            obs_size + config.actor_output_dim,
            config.actor_q_hidden_dim,
        )
        self.fc2 = nn.Linear(config.actor_q_hidden_dim, config.actor_q_hidden_dim)
        self.fc3 = nn.Linear(config.actor_q_hidden_dim, 1)
        self.ln1 = nn.LayerNorm(config.actor_q_hidden_dim)
        self.ln2 = nn.LayerNorm(config.actor_q_hidden_dim)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class FixedNorm(nn.Module):
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        # store as buffers so they move with .to(device) but are not trainable
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)  # pyright: ignore[reportOperatorIssue]


class ImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, normalization: FixedNorm):
        super().__init__()
        # self.normalization = nn.BatchNorm1d(input_dim, affine=False)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x):
        # x = self.normalization(x)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.ln1(x)
        x = F.silu(x)
        # x = self.fc2(x)
        # x = self.ln2(x)
        # x = F.silu(x)
        x = self.fc_mu(x)
        return x


class SharedEncoder(nn.Module):
    def __init__(self, config: Config, vision_only=False):
        super().__init__()
        self.config = config

        self.image_encoders = nn.ModuleList(
            [
                ImageEncoder(
                    input_dim=self.config.vision_head_input_dim,
                    hidden_dim=self.config.vision_head_hidden_dim,
                    output_dim=self.config.vision_head_output_dim,
                    normalization=FixedNorm(
                        mean=torch.zeros(self.config.vision_head_input_dim),
                        std=torch.ones(self.config.vision_head_input_dim),
                    ),
                )
                for _ in range(config.n_cameras)
            ]
        )
        self.vision_only = vision_only
        # for i in range(self.config.n_cameras):
        #     seq: ImageEncoder = self.image_encoders[i]  # type: ignore # insufficient type info
        #     nn.init.xavier_uniform_(seq.fc1.weight)
        #     nn.init.zeros_(seq.fc1.bias)
        #     nn.init.xavier_uniform_(seq.fc_mu.weight)
        #     nn.init.zeros_(seq.fc_mu.bias)

    def get_batchnorm_stats(self):
        stats = []
        for i in range(self.config.n_cameras):
            seq: ImageEncoder = self.image_encoders[i]  # type: ignore # insufficient type info
            stats.append(
                {
                    # f"bn_running_mean_{i}": seq.normalization.running_mean.clone(),
                    # f"bn_running_var_{i}": seq.normalization.running_var.clone(),
                }
            )
        return stats

    def load_batchnorm_stats(self, bn_stats: list[dict]):
        for i in range(self.config.n_cameras):
            seq: ImageEncoder = self.image_encoders[i]  # type: ignore # insufficient type info
            stats = bn_stats[i]
            # seq.normalization.running_mean.data.copy_(stats[f"bn_running_mean_{i}"])
            # seq.normalization.running_var.data.copy_(stats[f"bn_running_var_{i}"])

    def load_state_dict_from_vaes(self, vae_state_dicts: list[dict]):
        for i in range(self.config.n_cameras):
            vae_state_dict = vae_state_dicts[i]
            seq: ImageEncoder = self.image_encoders[i]  # type: ignore # insufficient type info
            print(f"State dict keys: {list(vae_state_dict.keys())}")

            # Map VAE fc1 weights and bias to the first Linear layer in the Sequential
            seq.fc1.weight.data.copy_(vae_state_dict["encoder.fc1.weight"])
            seq.fc1.bias.data.copy_(vae_state_dict["encoder.fc1.bias"])

            # Map VAE LayerNorm weights and bias
            seq.ln1.weight.data.copy_(vae_state_dict["encoder.ln1.weight"])
            seq.ln1.bias.data.copy_(vae_state_dict["encoder.ln1.bias"])
            # Map VAE fc_mu weights and bias to the second Linear layer in the Sequential
            seq.fc_mu.weight.data.copy_(vae_state_dict["encoder.fc_mu.weight"])
            seq.fc_mu.bias.data.copy_(vae_state_dict["encoder.fc_mu.bias"])

    def load_state_dict_normalization(self, vae_normalizations: list[dict]):
        for i in range(self.config.n_cameras):
            seq: ImageEncoder = self.image_encoders[i]  # type: ignore # insufficient type info
            norm_params = vae_normalizations[i]
            # Load normalization parameters
            seq.normalization.mean.data.copy_(
                torch.tensor(norm_params["mean"]).flatten()
            )
            seq.normalization.std.data.copy_(torch.tensor(norm_params["std"]).flatten())

    def forward(self, x):
        # assume B, D_OBS input, last N_CAM * vision_head_input_dim are pre-encoded images
        _B, D_OBS = x.shape

        parts: list[torch.Tensor] = []

        vision_dim = self.config.vision_head_input_dim * self.config.n_cameras

        # non-image portion
        if not self.vision_only:
            parts.append(x[:, : D_OBS - vision_dim])

        start = D_OBS - vision_dim

        # image encoder outputs
        for i in range(self.config.n_cameras):
            slice_start = start + i * self.config.vision_head_input_dim
            slice_end = start + (i + 1) * self.config.vision_head_input_dim
            out = self.image_encoders[i](x[:, slice_start:slice_end])
            parts.append(out)

        return torch.cat(parts, dim=1)


class ActorFixedSigma(nn.Module):
    def __init__(self, config: Config, return_dist=False, scale=None):
        super().__init__()
        self.config = config
        self.return_dist = return_dist
        output_dim = config.actor_output_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(
                self.config.actor_nonvision_input_dim
                + self.config.vision_head_output_dim * self.config.n_cameras,
                config.actor_q_hidden_dim,
            ),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(config.actor_q_hidden_dim, config.actor_q_hidden_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(config.actor_q_hidden_dim, output_dim),
            nn.Tanh(),
        )
        if scale is not None:
            bias = 0.0
            scale = scale
        elif self.config.max_action is not None:
            scale = self.config.max_action
            bias = 0.0
        else:
            raise ValueError("Either scale or config.max_action must be provided")

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
        self.register_buffer("std", torch.tensor(config.actor_std, dtype=torch.float32))

    def compute_raw_action(self, x):
        return self.net(x)

    def compute_action(self, x):
        return self.net(x) * self.action_scale + self.action_bias

    def forward(self, x):
        B = x.size(0)

        h = self.net(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t) - torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        if self.return_dist:
            return action, log_prob, normal
        return action, log_prob


class Actor(nn.Module):
    def __init__(self, config: Config, return_dist=False, scale=None):
        super().__init__()
        self.config = config
        self.return_dist = return_dist
        self.output_dim = config.actor_output_dim

        self.net = nn.Sequential(
            nn.Linear(
                self.config.actor_nonvision_input_dim
                + self.config.vision_head_output_dim * self.config.n_cameras,
                config.actor_q_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.actor_q_hidden_dim, config.actor_q_hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(config.actor_q_hidden_dim, self.output_dim)
        # self.fc_chol_params = nn.Linear(
        #     config.actor_q_hidden_dim, self.output_dim * (self.output_dim + 1) // 2
        # )
        self.fc_logstd = nn.Linear(config.actor_q_hidden_dim, self.output_dim)
        # action rescaling
        if scale is not None:
            bias = 0.0
            scale = scale
        elif self.config.max_action is not None:
            scale = self.config.max_action
            bias = 0.0
        else:
            raise ValueError("Either scale or config.max_action must be provided")

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

        self.softplus = nn.Softplus()

    def compute_action_logprob(self, x, target):
        B = x.size(0)

        h = self.net(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        target_y_t = (target - self.action_bias) / self.action_scale
        target_x_t = torch.atanh(target_y_t)

        log_prob = normal.log_prob(target_x_t) - torch.log(
            (1 - target_y_t.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(1, keepdim=True)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # pyright: ignore[reportOperatorIssue]
        return action, log_prob

    def forward(self, x):
        B = x.size(0)

        h = self.net(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # pyright: ignore[reportOperatorIssue]

        log_prob = normal.log_prob(x_t) - torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        if self.return_dist:
            return action, log_prob, normal
        return action, log_prob

        # # Predict the Cholesky decomposition of the covariance matrix
        # chol_params = self.fc_chol_params(h)  # shape: (batch_size, num_tril_params)
        # # Total number of Cholesky parameters needed
        # tril_indices = torch.tril_indices(
        #     row=self.output_dim, col=self.output_dim, offset=0
        # )
        # diag_indices = torch.arange(self.output_dim, device=x.device)
        # L = torch.zeros(B, self.output_dim, self.output_dim, device=x.device)
        # L[:, tril_indices[0], tril_indices[1]] = chol_params

        # # L[:, diag_indices, diag_indices] = torch.exp(
        # #     torch.clamp(L[:, diag_indices, diag_indices], -20, 2)
        # # )
        # # L[:, diag_indices, diag_indices] = torch.exp(
        # #     LOG_STD_MIN
        # #     + (LOG_STD_MAX - LOG_STD_MIN)
        # #     * torch.sigmoid(L[:, diag_indices, diag_indices])
        # # )
        # L[:, diag_indices, diag_indices] = (
        #     self.softplus(L[:, diag_indices, diag_indices]) + 1e-5
        # )  # ensure positive diagonals

        # # Create the covariance matrix
        # dist = D.MultivariateNormal(
        #     mean * self.action_scale + self.action_bias,
        #     scale_tril=L * self.action_scale,  # type: ignore # insufficient type info
        # )
        # action = dist.rsample((B,))
        dist = D.Normal(
            mean * self.action_scale + self.action_bias,
            log_std.exp() * self.action_scale * self.action_scale,  # type: ignore # insufficient type info
        )
        dist = D.Independent(dist, 1)
        action = dist.rsample()

        log_prob = dist.log_prob(action)

        if self.return_dist:
            return action, log_prob, dist
        return action, log_prob

        # log_std = self.fc_logstd(h)
        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
        #     log_std + 1
        # )  # From SpinUp / Denis Yarats

        # std = log_std.exp()
        # normal = torch.distributions.Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias  # type: ignore # insufficient type info

        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)

        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias  # type: ignore # insufficient type info
        # return action, log_prob, mean

    # def get_action(self, x):
    #     mean, log_std = self(x)
    #     std = log_std.exp()
    #     normal = torch.distributions.Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias  # type: ignore # insufficient type info

    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
    #     log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)

    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias  # type: ignore # insufficient type info
    #     return action, log_prob, mean


# old workflow: mean+std -> normal -> sample -> tanh -> scale
# new workflow: mean+chol -> Mnormal -> sample.logprob -> scale => replace tanh with action clipping
# class SquashedNormalMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.mean = nn.Linear(hidden_dim, output_dim)
#         self.output_dim = output_dim
#         # self.log_std = nn.Linear(hidden_dim, output_dim)
#         self.chol_params = nn.Linear(hidden_dim, output_dim * (output_dim + 1) // 2)

#     def forward(self, x) -> D.MultivariateNormal:
#         B = x.size(0)

#         h = self.net(x)
#         mean = self.mean(h)

#         # log_std = self.log_std(h).clamp(-20, 2)  # clamp for stability
#         # std = log_std.exp()

#         # Predict the Cholesky decomposition of the covariance matrix
#         chol_params = self.chol_params(h)  # shape: (batch_size, num_tril_params)
#         # Total number of Cholesky parameters needed
#         tril_indices = torch.tril_indices(
#             row=self.output_dim, col=self.output_dim, offset=0
#         )
#         diag_indices = torch.arange(self.output_dim, device=x.device)
#         L = torch.zeros(B, self.output_dim, self.output_dim, device=x.device)

#         L[:, tril_indices[0], tril_indices[1]] = chol_params
#         # Ensure positive diagonals
#         L[:, diag_indices, diag_indices] = torch.exp(
#             torch.clamp(L[:, diag_indices, diag_indices], -20, 2)
#         )

#         # Create the covariance matrix
#         cov = torch.matmul(L, L.transpose(1, 2))
#         dist = D.MultivariateNormal(mean, covariance_matrix=cov)

#         sample = dist.rsample((B,))
#         log_prob = dist.log_prob(sample)
#         action = sample * self.action_scale + self.action_bias

#         return action, log_prob
