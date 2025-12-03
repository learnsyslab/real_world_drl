"""Train a Variational Autoencoder on DINOv2 features.

This script trains a 2-layer VAE with modern training optimizations:
- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule with warmup
- KL annealing (cyclical or linear)
- Gradient clipping
- Early stopping with best model checkpointing
- Mixed precision training (optional)
- Exponential Moving Average (EMA) of model weights

Usage:
    python crisp_drl/scripts/train_vae.py --data-path datasets/dinov2_dataset/feat_dinov2_vits14_reg.npz --hidden-dim 128 --latent-dim 32 --epochs 500 --batch-size 256

Dependencies:
    - torch
    - numpy
    - tensorboard
    - tqdm (optional)
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ---------------------------------------------------------------------------
# VAE Model Definition
# ---------------------------------------------------------------------------


class VAEEncoder(nn.Module):
    """2-layer encoder with LayerNorm and GELU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Mean and log-variance heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.gelu(self.ln2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """2-layer decoder with LayerNorm and GELU activation."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(z)))
        h = F.gelu(self.ln2(self.fc2(h)))
        return self.fc_out(h)


class VAE(nn.Module):
    """Variational Autoencoder with 2-layer encoder and decoder."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent mean (for inference)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss with β-VAE formulation.

    Args:
        x: Input tensor
        x_recon: Reconstructed tensor
        mu: Latent mean
        logvar: Latent log-variance
        beta: KL weight (β-VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Learning Rate and KL Schedulers
# ---------------------------------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_kl_annealing_weight(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 10,
    mode: str = "linear",
    n_cycles: int = 4,
) -> float:
    """Compute KL annealing weight.

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (for linear mode)
        mode: 'linear', 'cyclical', or 'constant'
        n_cycles: Number of cycles for cyclical annealing

    Returns:
        KL weight (beta)
    """
    if mode == "constant":
        return 1.0
    elif mode == "linear":
        return min(1.0, epoch / warmup_epochs)
    elif mode == "cyclical":
        # Cyclical annealing from "Cyclical Annealing Schedule" paper
        tau = (epoch % (total_epochs // n_cycles)) / (total_epochs // n_cycles)
        return min(1.0, tau * 2)  # Linearly increase then hold at 1
    else:
        raise ValueError(f"Unknown KL annealing mode: {mode}")


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1 - self.decay) * param.data
            )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, original_params: dict):
        """Restore original weights."""
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Data
    data_path: str = "datasets/dinov2_dataset/feat_dinov2_vits14_reg.npz"
    val_split: float = 0.1

    # Model
    hidden_dim: int = 32
    latent_dim: int = 16

    # Training
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # LR Schedule
    warmup_epochs: int = 10
    min_lr_ratio: float = 0.01

    # KL Annealing
    kl_mode: str = "cyclical"  # linear, cyclical, constant
    kl_cycles: int = 4
    beta_max: float = 1.0  # Maximum β for β-VAE

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Mixed Precision
    use_amp: bool = True

    # Early Stopping
    patience: int = 50
    min_delta: float = 1e-6

    # Logging & Saving
    log_interval: int = 10
    save_dir: str = "checkpoints/vae"
    tensorboard_dir: str = "tensorboard_logs/vae"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    beta: float,
    config: TrainConfig,
    ema: Optional[EMA] = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch[0].to(config.device)
        optimizer.zero_grad()

        if config.use_amp and config.device == "cuda":
            with torch.cuda.amp.autocast():
                x_recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta)

            scaler.scale(loss).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta)

            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    beta: float,
    config: TrainConfig,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in val_loader:
        x = batch[0].to(config.device)
        x_recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def train(config: TrainConfig):
    """Main training function."""
    print(f"Training VAE with config:")
    print(f"  Data: {config.data_path}")
    print(f"  Hidden dim: {config.hidden_dim}, Latent dim: {config.latent_dim}")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  KL annealing: {config.kl_mode}")
    print()

    # current time
    current_time = time.strftime("%Y%m%d_%H%M%S")

    # Load data
    data = np.load(config.data_path)
    features = data["features"].astype(np.float32)
    print(f"Loaded features: {features.shape}")

    # Normalize features (important for VAE training stability)
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    # Create dataset and split
    dataset = TensorDataset(torch.from_numpy(features_norm))
    n_val = int(len(dataset) * config.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )

    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # Create model
    input_dim = features.shape[1]
    model = VAE(input_dim, config.hidden_dim, config.latent_dim).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # LR scheduler
    num_training_steps = config.epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, config.min_lr_ratio
    )

    # Mixed precision scaler
    scaler = (
        torch.cuda.amp.GradScaler()
        if config.use_amp and config.device == "cuda"
        else None
    )

    # EMA
    ema = EMA(model, config.ema_decay) if config.use_ema else None

    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    save_dir = Path(config.save_dir) / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save normalization stats
    np.savez(
        save_dir / "normalization.npz",
        mean=feat_mean,
        std=feat_std,
    )

    # TensorBoard writer
    tb_dir = Path(config.tensorboard_dir) / current_time
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Log hyperparameters
    hparams = {
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "kl_mode": config.kl_mode,
        "beta_max": config.beta_max,
        "use_ema": config.use_ema,
    }
    writer.add_text("hyperparameters", str(hparams), 0)

    print("\nStarting training...")
    print(f"TensorBoard logs: {tb_dir}")
    for epoch in range(1, config.epochs + 1):
        # Get KL annealing weight
        beta = config.beta_max * get_kl_annealing_weight(
            epoch, config.epochs, config.warmup_epochs, config.kl_mode, config.kl_cycles
        )

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, beta, config, ema
        )

        # Validate (with EMA weights if enabled)
        if ema is not None:
            original_params = {
                name: param.data.clone() for name, param in model.named_parameters()
            }
            ema.apply_shadow(model)

        val_metrics = validate(model, val_loader, beta, config)

        if ema is not None:
            ema.restore(model, original_params)

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard logging (every epoch)
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Loss/train_recon", train_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/train_kl", train_metrics["kl_loss"], epoch)
        writer.add_scalar("Loss/val_recon", val_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/val_kl", val_metrics["kl_loss"], epoch)
        writer.add_scalar("Schedule/beta", beta, epoch)
        writer.add_scalar("Schedule/learning_rate", current_lr, epoch)

        # Console logging
        if epoch % config.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(R: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"β: {beta:.3f} | LR: {current_lr:.2e}"
            )

        # Checkpointing
        if val_metrics["loss"] < best_val_loss - config.min_delta:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            # Save best model (with EMA weights if enabled)
            if ema is not None:
                ema.apply_shadow(model)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": config.hidden_dim,
                    "latent_dim": config.latent_dim,
                },
            }
            torch.save(checkpoint, save_dir / "best_model.pt")

            if ema is not None:
                ema.restore(model, original_params)

            print(f"  → Saved new best model (val_loss: {val_metrics['loss']:.6f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config.patience:
            print(
                f"\nEarly stopping after {epoch} epochs (patience: {config.patience})"
            )
            break

    # Save final model
    if ema is not None:
        original_params = {
            name: param.data.clone() for name, param in model.named_parameters()
        }
        ema.apply_shadow(model)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_metrics["loss"],
        "config": {
            "input_dim": input_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
        },
    }
    torch.save(checkpoint, save_dir / "final_model.pt")

    # Log final metrics to TensorBoard
    writer.add_hparams(
        hparams,
        {
            "hparam/best_val_loss": best_val_loss,
            "hparam/final_val_loss": val_metrics["loss"],
            "hparam/final_epoch": epoch,
        },
    )
    writer.close()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"TensorBoard logs: {tb_dir}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on DINOv2 features")

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/dinov2_dataset/feat_dinov2_vits14_reg.npz",
        help="Path to .npz file with 'features' key",
    )
    parser.add_argument("--val-split", type=float, default=0.1)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # LR Schedule
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)

    # KL Annealing
    parser.add_argument(
        "--kl-mode",
        type=str,
        default="cyclical",
        choices=["linear", "cyclical", "constant"],
    )
    parser.add_argument("--kl-cycles", type=int, default=4)
    parser.add_argument("--beta-max", type=float, default=1.0)

    # EMA
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    # Mixed Precision
    parser.add_argument("--no-amp", action="store_true")

    # Early Stopping
    parser.add_argument("--patience", type=int, default=500)

    # Logging & Saving
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints/vae")
    parser.add_argument("--tensorboard-dir", type=str, default="tensorboard_logs/vae")

    # Device
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(
        data_path=args.data_path,
        val_split=args.val_split,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        kl_mode=args.kl_mode,
        kl_cycles=args.kl_cycles,
        beta_max=args.beta_max,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        use_amp=not args.no_amp,
        patience=args.patience,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        tensorboard_dir=args.tensorboard_dir,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    train(config)


if __name__ == "__main__":
    main()
