# 1) Load replay buffer with perfect actions
# 2) Pre-encode observations with VAE with frozen weights OR encode on-the-fly with trainable weights
# 3) Train actor to predict perfect actions from encoded observations and other info


import argparse
from pathlib import Path
import time
import gymnasium
import joblib
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from crisp_drl.agents.shared.networks_cleanrl import (
    Actor,
    ActorFixedSigma,
    SharedEncoder,
)
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.data.buffers_cleanrl import ReplayBufferGpuWithPerfectActions

"""Sample usage:
python crisp_drl/scripts/pre_train_actor_supervised.py --buffer_path rollout_data/collect_data/replay_buffer.joblib --vae_path checkpoints/vae/20251208_111636/best_model.pt --run_name run_3 --batch_size 512 --epochs 500 --lr 1e-3 --weight_decay 5e-4
"""

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_path", type=str, help="Path to the buffer file")
parser.add_argument("--vae_path", type=str, help="Path to the trained VAE file")
parser.add_argument(
    "--batch_size", type=int, default=256, help="Batch size for training"
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer"
)

parser.add_argument(
    "--run_name", type=str, default=None, help="Run name for TensorBoard logging"
)
parser.add_argument(
    "--val_split", type=float, default=0.1, help="Validation split ratio"
)
args = parser.parse_args()
run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load replay buffer
buffer: ReplayBufferGpuWithPerfectActions = joblib.load(args.buffer_path)
print(f"Loaded buffer from {args.buffer_path} with {buffer.size()} transitions.")

# Load VAE
vae_state_dict = torch.load(args.vae_path)["model_state_dict"]
# remove file from path, add normalization.npz
vae_normalization_path = str(args.vae_path).replace(
    "best_model.pt", "normalization.npz"
)
vae_normalization = np.load(vae_normalization_path)
print(f"Loaded VAE from {args.vae_path}.")

shared_encoder = SharedEncoder(Config())
# shared_encoder.load_state_dict_from_vaes([vae_state_dict])
# shared_encoder.initialize_randomly()
# shared_encoder.load_state_dict_normalization([vae_normalization])
shared_encoder.to(device)
assert torch.all(
    torch.isfinite(torch.cat([p.flatten() for p in shared_encoder.parameters()]))
), "Shared encoder has NaN or Inf values when initialized"

obs = buffer.observations.to(device)
actions = buffer.perfect_actions.to(device)
assert torch.all(torch.isfinite(obs)), "Observations have NaN or Inf values"
assert torch.all(torch.isfinite(actions)), "Actions have NaN or Inf values"


dataset = TensorDataset(obs, actions)

# Split dataset into train and val
val_size = int(len(dataset) * args.val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

log_dir = Path("tensorboard_logs/pretrain_actor") / run_name
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))
writer.add_text(
    "hparams",
    f"batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}",
    0,
)
print(f"TensorBoard logging to {log_dir}")

# Initialize actor
actor = ActorFixedSigma(
    action_space=gymnasium.spaces.Box(low=-1, high=1, shape=actions.shape[1:]),
    config=config,
    return_dist=True,
    scale=0.0003,
).to(device)

# Optimizer

params = list(actor.parameters()) + list(shared_encoder.parameters())

optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.8)

assert torch.all(
    torch.isfinite(torch.cat([p.flatten() for p in shared_encoder.parameters()]))
), "Shared encoder has NaN or Inf values before training"

# Track best validation alignment
best_val_align = -float("inf")
best_model_path = Path("checkpoints/pretrain_actor") / (run_name + "_best_align.pth")
best_shared_encoder_path = Path("checkpoints/pretrain_actor") / (
    run_name + "_best_align_shared_encoder.pth"
)

# Training loop
for epoch in range(args.epochs):
    # Training phase
    epoch_loss = 0.0
    epoch_l2 = 0.0
    epoch_align = 0.0

    for i, (batch_obs, norm_batch_actions) in enumerate(train_dataloader):
        batch_obs = batch_obs.to(device)
        norm_batch_actions = (
            norm_batch_actions.to(device) / 0.0003
        )  # scale actions to [-1, 1]

        # Add tiny random offsets to training observations for slight augmentation
        # batch_obs = batch_obs + torch.empty_like(batch_obs).uniform_(-1e-5, 1e-5)

        batch_obs = shared_encoder(batch_obs)

        # act, _, dist = actor(batch_obs)
        # loss = -dist.log_prob(batch_actions).mean()
        # act, logprob = actor.compute_action_logprob(batch_obs, batch_actions)
        # loss = -logprob.mean()
        raw_act = actor.compute_raw_action(batch_obs)
        loss = F.mse_loss(raw_act, norm_batch_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_l2 += loss.item() * batch_obs.size(
                0
            )  # F.mse_loss(act, batch_actions).item() * batch_obs.size(0)
            pred_unit = F.normalize(raw_act, dim=-1, eps=1e-8)
            target_unit = F.normalize(norm_batch_actions, dim=-1, eps=1e-8)
            epoch_align += (pred_unit * target_unit).sum(
                dim=-1
            ).mean().item() * batch_obs.size(0)

        epoch_loss += loss.item() * batch_obs.size(0)

    epoch_loss /= len(train_dataset)
    epoch_l2 = np.sqrt(epoch_l2 / len(train_dataset))
    epoch_align /= len(train_dataset)

    # Validation phase
    val_loss = 0.0
    val_l2 = 0.0
    val_align = 0.0

    actor.eval()
    shared_encoder.eval()
    with torch.no_grad():
        for batch_obs, norm_batch_actions in val_dataloader:
            batch_obs = batch_obs.to(device)
            norm_batch_actions = norm_batch_actions.to(device) / 0.0003

            batch_obs = shared_encoder(batch_obs)

            # act, _, dist = actor(batch_obs)
            # loss = -dist.log_prob(batch_actions).mean()
            raw_act = actor.compute_raw_action(batch_obs)
            loss = F.mse_loss(raw_act, norm_batch_actions)

            val_loss += loss.item() * batch_obs.size(0)
            val_l2 += F.mse_loss(raw_act, norm_batch_actions).item() * batch_obs.size(0)
            pred_unit = F.normalize(raw_act, dim=-1, eps=1e-8)
            target_unit = F.normalize(norm_batch_actions, dim=-1, eps=1e-8)
            val_align += (pred_unit * target_unit).sum(
                dim=-1
            ).mean().item() * batch_obs.size(0)

    actor.train()
    shared_encoder.train()

    val_loss /= len(val_dataset)
    val_l2 = np.sqrt(val_l2 / len(val_dataset))
    val_align /= len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {epoch_loss:.6f}, Train L2: {epoch_l2:.6f}, Train Align: {epoch_align:.6f}, Val Loss: {val_loss:.6f}, Val L2: {val_l2:.6f}, Val Align: {val_align:.6f}"
    )
    writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
    writer.add_scalar("L2/train", epoch_l2, epoch + 1)
    writer.add_scalar("Align/train", epoch_align, epoch + 1)
    writer.add_scalar("Loss/val", val_loss, epoch + 1)
    writer.add_scalar("L2/val", val_l2, epoch + 1)
    writer.add_scalar("Align/val", val_align, epoch + 1)

    # Save model if validation alignment improved
    if val_align > best_val_align:
        best_val_align = val_align
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), best_model_path)
        torch.save(shared_encoder.state_dict(), best_shared_encoder_path)
        print(f"  → Saved best model with val align: {best_val_align:.6f}")

writer.close()
# Save pre-trained actor
output_path = Path("checkpoints/pretrain_actor") / (run_name + ".pth")
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(actor.state_dict(), output_path)
shared_encoder_output_path = Path("checkpoints/pretrain_actor") / (
    run_name + "_shared_encoder.pth"
)
torch.save(shared_encoder.state_dict(), shared_encoder_output_path)
print(f"Saved pre-trained actor to {output_path}.")
print(f"Saved pre-trained shared encoder to {shared_encoder_output_path}.")
