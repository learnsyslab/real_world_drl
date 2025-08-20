

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rlmb.data.buffers_cleanrl import ReplayBuffer
from rlmb.agents.sac.networks_cleanrl import SoftQNetwork, Actor
from rlmb.agents.sac.sac_config import SAC_Config

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

config = SAC_Config()
run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = config.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(config.env_id, config.seed + i, i, config.capture_video, run_name) for i in range(config.num_envs)]
)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

max_action = float(envs.single_action_space.high[0])

actor = Actor(envs).to(device)
qf1 = SoftQNetwork(envs).to(device)
qf2 = SoftQNetwork(envs).to(device)
qf1_target = SoftQNetwork(envs).to(device)
qf2_target = SoftQNetwork(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.q_lr)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.policy_lr)

# Automatic entropy tuning
if config.autotune:
    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=config.q_lr)
else:
    alpha = config.alpha

envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
    config.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    n_envs=config.num_envs,
    handle_timeout_termination=False,
)
start_time = time.time()

# TRY NOT TO MODIFY: start the game
obs, _ = envs.reset(seed=config.seed)
episode_returns = np.zeros((config.num_envs,), dtype=np.float32)
episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)

for global_step in range(config.total_timesteps):
    # ALGO LOGIC: put action logic here
    if global_step < config.learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    episode_returns += rewards
    episode_lengths += 1

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    done = np.logical_or(terminations, truncations)
    if np.any(done):
        for idx, d in enumerate(done):
            if d:
                writer.add_scalar(f"charts/episodic_return/env_{idx}", episode_returns[idx], global_step)
                writer.add_scalar(f"charts/episodic_length/env_{idx}", episode_lengths[idx], global_step)
                episode_returns[idx] = 0
                episode_lengths[idx] = 0

    # TRY NOT TO MODIFY: save data to reply buffer
    rb.add(obs, next_obs, actions, rewards, terminations, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > config.learning_starts:
        data = rb.sample(config.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
            qf1_next_target = qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * config.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = qf1(data.observations, data.actions).view(-1)
        qf2_a_values = qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % config.policy_update_after == 0:  # TD 3 Delayed update support
            for _ in range(
                config.policy_update_after
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if config.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target networks
        if global_step % config.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )
            if config.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            torch.save(actor.state_dict(), "model_weights.pth")

envs.close()
writer.close()