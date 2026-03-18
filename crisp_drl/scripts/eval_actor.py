import json
import os
import time
import numpy as np
import torch
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor
from crisp_drl.agents.shared.networks_cleanrl import SharedEncoder
from crisp_drl.envs import make_env
import argparse


# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate actor policy")
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Checkpoint run name, e.g. 1cft_d300v9/pretrain_10",
)
parser.add_argument("--n_cameras", type=int, default=1, help="Number of cameras")
parser.add_argument("--use_ft", action="store_true", help="Use force/torque sensor")
parser.add_argument(
    "--n_steps", type=int, default=150, help="Truncation limit for episodes"
)
parser.add_argument(
    "--pe_accuracy",
    type=float,
    default=0.0015,
    help="Accuracy of the pose estimation module",
)


args = parser.parse_args()

current_date = time.strftime("%Y%m%d-%H%M%S")

config = Config(
    n_cameras=args.n_cameras,
    actor_nonvision_input_dim=14 if args.use_ft else 8,
    episode_length=args.n_steps,
)
env = make_env.create_simulated_env(
    {"initial_keyframe": 2, "live_view": False},
    is_eval=True,
    use_ft=args.use_ft,
    sac_config=config,
    pe_accuracy=args.pe_accuracy,
)
# cutoff after 200 if <50% success rate, 400 for <75% success rate, 600 for <80%, 1000 otherwise
N_ROLLOUTS = 500
N_TRIALS = 1
model_name = args.run_name
ideal_goal_pos = np.array([0.6, 0.0])
ideal_grasp_pos = np.array([0.0, 0.0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(config)
actor.load_state_dict(torch.load(f"checkpoints/{model_name}/actor_state_dict.pth"))
actor.to(device)
actor.eval()

shared_encoder_path = f"checkpoints/{model_name}/shared_encoder_state_dict.pth"
shared_encoder = SharedEncoder(config)
shared_encoder.load_state_dict(torch.load(shared_encoder_path))
shared_encoder.to(device)
shared_encoder.eval()

for _ in range(N_TRIALS):
    n_success = 0
    total_successful_length = 0
    for i in range(N_ROLLOUTS):
        if i == 100 and n_success / (i + 1) < 0.5:
            print(
                f"Cutting off evaluation at {i} rollouts due to low success rate ({n_success / (i + 1) * 100:.2f}%)."
            )
            break
        elif i == 200 and n_success / (i + 1) < 0.75:
            print(
                f"Cutting off evaluation at {i} rollouts due to low success rate ({n_success / (i + 1) * 100:.2f}%)."
            )
            break
        elif i == 300 and n_success / (i + 1) < 0.8:
            print(
                f"Cutting off evaluation at {i} rollouts due to low success rate ({n_success / (i + 1) * 100:.2f}%)."
            )
            break
        # new reset wrapper: automatically place brick back and pick it up again
        obs, reset_info = env.reset()
        # compute goal position
        actual_grasp_pos = reset_info["reset.grasped.position"]
        goal_pos = ideal_goal_pos.copy()
        goal_pos[0] += actual_grasp_pos[0] - ideal_grasp_pos[0]

        episode_length = 0
        while True:
            episode_length += 1
            action, _ = actor(
                shared_encoder(obs["observation.formatted"].unsqueeze(0).to(device))
            )
            action = action.cpu().detach().squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episode_successful = "custom_events" in info and "E_SUCCESS" in map(
                    lambda x: x[1], info["custom_events"]
                )
                if episode_successful:
                    n_success += 1
                    total_successful_length += episode_length
                if i % 10 == 9:
                    print(
                        f"Total episodes = {i + 1}, success rate = {n_success / (i + 1) * 100:.2f}%; Episode finished: Length = {episode_length}, Success: {terminated}"
                    )

                break
        fd = os.open(
            os.path.join(f"checkpoints/{model_name}/", "eval_infos.jsonl"),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        line = json.dumps(
            {
                "reset.grasped.delta": reset_info["reset.grasped.delta"].tolist(),
                "reset.goal_position.offset": reset_info[
                    "reset.goal_position.offset"
                ].tolist(),
                "observation.perfect_action": obs[
                    "observation.perfect_action"
                ].tolist(),
                "episode_successful": terminated,
                "length": episode_length,
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        os.write(fd, (line + "\n").encode())
        os.close(fd)
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
# ds500_999 -> 18% l=50.56; utd8; 0.999 500

# ds100_9 -> 29-31-33-34% l=34.32-39.42-41.86-45.09 utd32; 0.9 100
# ds200_9 -> 61-67-69-73% l=29-75-34.60-35.04-36.85 utd20; 0.9 200
# ds500_9 -> 99-100% l=18.09-20.30; utd8; 0.9 500


# ds100_8 -> 35-37-40-43% l=30.23-31.88-33.24-36.86 utd32; 0.8 100
# ds200_8 -> 66-68-73-77% l=29.58-31.94-35.44-35.80 utd20; 0.8 200
# ds300_8 -> 90-92-95% l=21.95-25.37-26.34 utd12; 0.8 300
# ds400_8 -> 97-99% l=17.28-18.15 utd10; 0.8 400
# ds500_8 -> 99-100% l=15.51-16.47 utd8; 0.8 500

# ds200_75 -> 51-53-60% l=23.62-26.63-28.27 utd20; 0.75 200
# ds200_85 -> 80-83-84% l=33.15-34.80-35.49 utd20; 0.85 200

# ds100_85 -> 16% l=36.81 utd40; 0.85 100

# ds200_85e  -> 95-96% l=35.42-37.42 utd20; 0.85 200 extended safety box
# ds200_85es2 -> 91-93% l=27.33-30.96 utd20; 0.85 200 extended safety box, second subset, seed 2
# ds200_85e2 -> 59-60% l=35.49-36.77 utd20; 0.85 200 extended safety box, second subset

# ds100_85e  -> 59-65-66-73% l=35.44-38.42-39.17-40.76 utd40; 0.85 100 extended safety box
# ds100_85e2 -> 19-29% l=37.95-40.21 utd40; 0.85 100 extended safety box, second subset

# Data is important -> How does good data look like? Coverage? -> Compare datasets; use fixed randomisation scheme?
# Try more robust method (longer rollouts <-> perffect Q)

# with 2 q fns:
# ds200_85e2_q2 -> 54-56% l=33.50-36.61 utd20; 0.85 200 extended safety box, second subset, 2 q fns
# with half entropy coeff (alpha=0.0005):
# ds200_85e2_a05 -> 53-60% l=38.83-41.43 utd20; 0.85 200 extended safety box, second subset, alpha=0.0005

# with 3-step returns:
# ds200_85e2_nstp3 -> 15% l=36.87 utd20; 0.85 200 extended safety box, second subset, n_step_return=3
# ds200_85e2_nstp2 -> 47-56% l=31.00-31.02 utd20; 0.85 200 extended safety box, second subset, n_step_return=2

# with pre-training of Q-fn with perfect actions:
# ds200_85e2_pfq -> 54-60% l=33.32-34.04 utd20; 0.85 200 extended safety box, second subset, pre-train perfect actions for Q-fn targets

# compute dataset metrics: average rollout length, average successful length, success rate, coverage across state space
# success rate important?

# dss200_85e1 -> 64-71% l=36.46-37.81 utd20; 0.85 200 extended safety box
# dss200_85e2 -> 50-56% l=40.56-43.77 utd20; 0.85 200 extended safety box

# datasets: _e -> success rate 0.785; _e2 -> success rate 0.73
#         s_e1 -> success rate 0.765;s_e2 -> success rate 0.72

# with only completed rollouts in dataset:
# dss200_85ec -> 32-38% l=31.60-38.47 utd20; 0.85 200 extended safety box
# dss100_85ec -> 29-32% l=29.12-38.07 utd20; 0.85 200 extended safety box

# with 80/90% completed rollouts in dataset:
# ds200_85ec8 -> 62-66% l=37.71-41.89
# ds100_85ec9 -> 43-48% l=31.15-35.84


# after 200 rollouts rlpd:
# ds200_85e2 s2 -> 91-97% l=23.34-24.53 ; rlpd utd4
# ds200_85e2 s3 -> 95-99% l=25.26-25.33 ; rlpd utd4
# after many rollouts rlpd with utd20:
# ds200_85e2 s2 -> 59-61% l=33.59-39.56 ; rlpd utd20, 250 rollouts
# ds200_85e2 s3 -> 93-95% l=26.55-27.41 ; rlpd utd20, 400 rollouts

# longer demo rollouts (150):
# ds200_85el1 -> 93-97% l=27.13-31.98
# ds200_85el2 -> 95-96% l=35.23-38.81

# with only 100 demos:
# ds100_85el1 -> 34-42% l=23.23-26.97
# ds100_85el2 -> 38-51% l=31.06-41.39

# even longer demo rollouts (200, p=0.9):
# ds150_9exl1 -> 82-86% l=47.78-50.70
# ds150_9exl2 -> 94-96% l=35.17-36.06

# p=0.88
# ds150_88exl1 -> 31-35% l=35.32-41.43
# ds150_88exl2 -> 86-86% l=43.02-58.62

# baseline: p=0.85 l=150
# ds150_85el1 -> 96-96% l=32.83-32.10
# ds150_85el2 -> 88-91% l=37.05-40.40

# baseline: p=0.85 l=120
# ds120_85el1 -> 38-39% l=28.54-31.68
# ds120_85el2 -> 65-73% l=47.09-52.41

# with weight decay:
# ds120_85el1_w1e-4 -> 18-26% l=46.78-71.27
# ds120_85el1_w1e-5 -> 19-22% l=61.63-71.09

# with pre-trained image encoder:
# ds150_85el2_utd20 -> 68-73% l=40.15-46.94

# New env, broader coverage, start in contact, only xy error as non-vision input
# ds300v3_8_utd20 -> 88-93-94% l=26.27-32.44-32.85
#           utd30 -> 88-90% l=28.59-30.07

# with all 8 observations as non-vision input, n=200
# ds300v4_75 utd5 -> 92-93-96-97% l=37.47-39.29-39.70-39.80
# ds300v4_75 utd10 -> 90-91.5-94-96% l=30.92-33.59-33.99-35.61
# ds300v4_75 utd15 -> 89.5-91-94.5-96.5% l=34.94-39.46-41.26-41.44
# ds300v4_75 utd20 -> 89.5-91-92.5-97% l=36.94-38.70-42.46-42.67
# ds300v4_75 utd25 -> 89.5-93.5-94-95% l=32.68-34.22-37.41-37.44
# ds300v4_75 utd30 -> 91-91.5-93.5-94.5% l=34.25-35.79-36.91-38.08

# ds300v4_75 utd5 -> 91.5-93.5-95.5-97.5% l=28.28-28.29-29.50-30.36
# ds300v4_75 utd10 -> 92-93-94-94.5% l=27.56-27.64-27.65-28.29
# ds300v4_75 utd15 -> 93-93.5-93.5-95.5% l=28.29-28.45-29.06-33.66
# ds300v4_75 utd20 -> 91-92-92.5-94% l=27.10-27.71-28.58-30.12
# ds300v4_75 utd25 -> 91-93-94.5-96.5% l=31.57-34.23-34.13-35.95
# ds300v4_75 utd30 -> 90-91-92.5-94% l=29.23-29.72-31.50-35.57
