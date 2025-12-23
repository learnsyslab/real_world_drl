from pathlib import Path
import time
import numpy as np
import torch
from crisp_drl.agents.shared.config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor
from crisp_drl.agents.shared.networks_cleanrl import SharedEncoder
from crisp_drl.envs import make_env
import sys

exp_name = ""
if len(sys.argv) > 1:
    exp_name = sys.argv[1]

current_date = time.strftime("%Y%m%d-%H%M%S")

env = make_env.create_simulated_env(
    {
        "initial_keyframe": 2,
        "lego_shift_range": (-0.002, 0.002),
        "initial_position_range": (
            np.array([-1.0, -1.0, -2.0]) * 1e-3,  # np.zeros(3),
            np.array([1.0, 1.0, -1.9]) * 1e-3,  # np.zeros(3),
        ),
        "live_view": True,
    }
)


# Alternate between going to the goal position and moving randomly with probaility p
N_ROLLOUTS = 100
model_name = "1cam_128_16_full_pre_ds500v2_utd8"
ideal_goal_pos = np.array([0.6, 0.0])
ideal_grasp_pos = np.array([0.0, 0.0])
n_success = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(action_space=env.action_space, config=Config())
actor.load_state_dict(torch.load(f"checkpoints/{model_name}/actor_state_dict.pth"))
actor.to(device)
actor.eval()

shared_encoder_path = f"checkpoints/{model_name}/shared_encoder_state_dict.pth"
shared_encoder = SharedEncoder(Config())
shared_encoder.load_state_dict(torch.load(shared_encoder_path))
shared_encoder.to(device)
shared_encoder.eval()

for i in range(N_ROLLOUTS):
    # new reset wrapper: automatically place brick back and pick it up again
    obs, reset_info = env.reset()
    # compute goal position
    actual_grasp_pos = reset_info["reset.grasped.position"]
    goal_pos = ideal_goal_pos.copy()
    goal_pos[0] += actual_grasp_pos[0] - ideal_grasp_pos[0]

    episode_length = 0
    while True:
        episode_length += 1
        action, _ = actor(shared_encoder(obs.unsqueeze(0).to(device)))
        action = action.cpu().detach().squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if terminated:
                n_success += 1
            if i % 10 == 0:
                print(
                    f"Total episodes = {i + 1}, success rate = {n_success / (i + 1) * 100:.2f}%; Episode finished: Length = {episode_length}, Success: {terminated}"
                )

            break
print(
    f"Total episodes = {i + 1}, success rate = {n_success / (i + 1) * 100:.2f}%; Episode finished: Length = {episode_length}, Success: {terminated}"  # type: ignore
)
env.close()


# 1cam_128_16_full_pre_ds33999 -> 69%
# 1cam_128_16_full_pre_ds33999_utd8 -> 90-92% => Re-run with new reward and RLPD; worse: 60% with box violation term, 63% without, back to 88% with old reward
# 1cam_128_16_full_pre_ds8999 -> 82%
# 1cam_128_16_full_pre_ds8999_utd8 -> 76%
# 1cam_128_16_full_pre         -> 96%

# ds33999s -> 75-79% (sparse only); second seed: 71-74%
# ds33999sdd -> 52-55% sparse + delta place
# 79% with old reward?; with udt12 (until convergence) -> 75-77%
# with dense simple -> 61-70 %; utd8 63-69%


# ds5s -> 38-41%
# ds999s -> 36-42%
# mixed 33, 8, 9, 999 > 77-78%
# with half sized buffer + utd16 -> 36-60%??; utd8 -> 39-50-53%; utd32 -> 60-68%
# second run with full buffer: utd8 -> 45%
# 100 samples mixed utd8 -> 11%

# ds500v2 -> 97% utd8; 0.33, 0.8, 0.9, 0.95, 0.999 100 each
# ds500_0 -> % utd8; 0.0 500
