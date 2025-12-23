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
model_name = "1cam_128_16_full_pre_ds100_9_utd32"
ideal_goal_pos = np.array([0.6, 0.0])
ideal_grasp_pos = np.array([0.0, 0.0])
n_success = 0
total_successful_length = 0
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
                total_successful_length += episode_length
            if i % 10 == 9:
                print(
                    f"Total episodes = {i + 1}, success rate = {n_success / (i + 1) * 100:.2f}%; Episode finished: Length = {episode_length}, Success: {terminated}"
                )

            break
print(
    f"Total episodes = {i + 1}, success rate = {n_success / (i + 1) * 100:.2f}%; Episode finished: Length = {episode_length}, Success: {terminated}, Average Successful Length: {total_successful_length / n_success if n_success > 0 else 0:.2f}"  # type: ignore
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

# ds500v2 -> 96-97% l=18.69 utd8; 0.33, 0.8, 0.9, 0.95, 0.999 100 each
# ds500_0 -> 8% l=26 utd8; 0.0 500
# ds500_33 -> 61% l=41.85 utd8; 0.33 500
# ds500_8 -> 99-100% l=15.51-16.47 utd8; 0.8 500
# ds500_9 -> 99-100% l=18.09-20.30; utd8; 0.9 500
# ds500_999 -> 18% l=50.56; utd8; 0.999 500

# ds200_8 -> 66-68-73-77% l=29.58-31.94-35.44-35.80 utd20; 0.8 200
# ds200_9 -> 61-67-69-73% l=29-75-34.60-35.04-36.85 utd20; 0.9 200

# ds100_8 -> 35-37-40-43% l=30.23-31.88-33.24-36.86 utd32; 0.8 100
# ds100_9 -> 29-31-33-34% l=34.32-39.42-41.86-45.09 utd32; 0.9 100
