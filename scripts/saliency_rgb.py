"""
RGB Saliency Map Script for understanding pixel-level feature importance in a trained RL policy.

This script loads a debug image (produced by DinoImageEncoderWrapper), encodes it through
DINOv2, and computes gradient-based saliency maps to visualize which RGB pixels are most
important for the policy's action outputs.
"""

import argparse
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v3 as imageio

from crisp_drl.data.buffers_cleanrl import ReplayBufferGpu
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.networks_cleanrl import SharedEncoder, Actor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RGB saliency maps for RL policy"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="test_images/debug_first_image_observation.images.front.png",
        help="Path to the input image (debug image from DinoImageEncoderWrapper)",
    )
    parser.add_argument(
        "--buffer_path",
        type=str,
        default="rollout_data/collect_data_real/env_v0_0.8/replay_buffer.joblib",
        help="Path to the replay buffer joblib file (for non-vision features)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/rw_1cam_128_16_full_pre_d300v3_8_utd30",
        help="Directory containing the checkpoint files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output images (default: checkpoint_dir)",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vits14_reg",
        help="DINOv2 model name",
    )
    return parser.parse_args()


def load_models(checkpoint_dir: str, config: Config, device: str):
    """Load the actor and shared encoder from checkpoints."""
    checkpoint_path = Path(checkpoint_dir)

    # Load shared encoder
    shared_encoder = SharedEncoder(config).to(device)
    shared_encoder_path = checkpoint_path / "shared_encoder_state_dict.pth"
    if shared_encoder_path.exists():
        shared_encoder.load_state_dict(
            torch.load(shared_encoder_path, map_location=device)
        )
        print(f"Loaded shared encoder from {shared_encoder_path}")
    else:
        print(f"Warning: Shared encoder not found at {shared_encoder_path}")

    # Load actor
    actor = Actor(config).to(device)
    actor_path = checkpoint_path / "actor_state_dict.pth"
    if actor_path.exists():
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        print(f"Loaded actor from {actor_path}")
    else:
        raise FileNotFoundError(f"Actor not found at {actor_path}")

    return shared_encoder, actor


def load_dino_model(dino_model_name: str, device: str):
    """Load DINOv2 model without JIT compilation."""
    print(f"Loading DINOv2 model: {dino_model_name} (no JIT)...")

    class Normalize(nn.Module):
        """ImageNet normalization for DINOv2."""

        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input: NHWC, Output: NCHW normalized
            x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
            return (x - self.mean) / self.std

    dino_backbone = torch.hub.load(
        "facebookresearch/dinov2", dino_model_name, trust_repo=True
    ).to(device)
    dino_backbone.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model = nn.Sequential(Normalize(mean, std), dino_backbone).to(device)
    model.eval()

    print("  DINOv2 loaded (eager mode, no JIT)")
    return model


def load_and_preprocess_image(image_path: str, device: str) -> torch.Tensor:
    """Load image and prepare for gradient computation."""
    print(f"Loading image from {image_path}...")
    img = imageio.imread(image_path)

    # Ensure image is 224x224
    if img.shape[0] != 224 or img.shape[1] != 224:
        raise ValueError(
            f"Expected image size 224x224, got {img.shape[:2]}. "
            "The debug image from DinoImageEncoderWrapper should already be 224x224."
        )

    # Convert to float tensor [0, 1] with shape (1, 224, 224, 3)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension

    print(f"  Image shape: {img_tensor.shape}")
    return img_tensor


def compute_rgb_saliency(
    image: torch.Tensor,
    nonvision_features: torch.Tensor,
    dino_model: nn.Module,
    shared_encoder: nn.Module,
    actor: nn.Module,
    config: Config,
    device: str,
):
    """
    Compute RGB saliency maps for action outputs.

    Returns gradients of action dimensions with respect to RGB input pixels.
    """
    shared_encoder.eval()
    actor.eval()
    # DINOv2 needs to be in train mode for gradient flow, or we ensure requires_grad works
    dino_model.eval()

    # Enable gradient computation for input image
    image_grad = image.clone().requires_grad_(True)

    # Forward pass through DINOv2
    dino_features = dino_model(image_grad)  # (1, 384)

    # Construct full observation: [nonvision_features, vision_features]
    # nonvision_features: (1, 2), dino_features: (1, 384)
    full_obs = torch.cat([nonvision_features, dino_features], dim=1)

    # Forward pass through shared encoder and actor
    encoded = shared_encoder(full_obs)
    actions, _ = actor(encoded)

    # Compute gradients for action x (first dimension)
    grad_action_x = torch.autograd.grad(
        outputs=actions[:, 0].sum(),
        inputs=image_grad,
        retain_graph=True,
    )[0]

    # Compute gradients for action y (second dimension)
    grad_action_y = torch.autograd.grad(
        outputs=actions[:, 1].sum(),
        inputs=image_grad,
    )[0]

    return grad_action_x, grad_action_y, actions


def create_saliency_visualization(
    image: np.ndarray,
    grad_action_x: torch.Tensor,
    grad_action_y: torch.Tensor,
    output_dir: Path,
):
    """Create and save RGB saliency visualizations."""
    # Convert gradients to numpy (1, H, W, 3) -> (H, W, 3)
    grad_x = grad_action_x.squeeze(0).cpu().numpy()  # (224, 224, 3)
    grad_y = grad_action_y.squeeze(0).cpu().numpy()  # (224, 224, 3)

    # Take absolute value for saliency
    saliency_x = np.abs(grad_x)
    saliency_y = np.abs(grad_y)

    # Combined saliency (sum of both actions)
    saliency_combined = saliency_x + saliency_y

    # --- Individual RGB Channel Saliency Maps ---
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row labels
    row_titles = ["Action X", "Action Y", "Combined"]
    saliencies = [saliency_x, saliency_y, saliency_combined]
    channel_names = ["Red", "Green", "Blue"]

    for row_idx, (saliency, row_title) in enumerate(zip(saliencies, row_titles)):
        # Original image in first column
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{row_title}\nOriginal Image")
        axes[row_idx, 0].axis("off")

        # RGB channels
        for ch_idx, ch_name in enumerate(channel_names):
            channel_saliency = saliency[:, :, ch_idx]

            # Normalize for visualization
            vmax = np.percentile(channel_saliency, 99)
            if vmax > 0:
                channel_saliency_norm = np.clip(channel_saliency / vmax, 0, 1)
            else:
                channel_saliency_norm = channel_saliency

            im = axes[row_idx, ch_idx + 1].imshow(
                channel_saliency_norm, cmap="hot", vmin=0, vmax=1
            )
            axes[row_idx, ch_idx + 1].set_title(f"{row_title}\n{ch_name} Channel")
            axes[row_idx, ch_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[row_idx, ch_idx + 1], fraction=0.046, pad=0.04)

    plt.suptitle("RGB Channel Saliency Maps", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_rgb_channels.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RGB channel saliency to {output_dir / 'saliency_rgb_channels.png'}")

    # --- Aggregated Saliency (sum over RGB) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Aggregate over RGB channels
    saliency_x_agg = saliency_x.sum(axis=2)
    saliency_y_agg = saliency_y.sum(axis=2)
    saliency_combined_agg = saliency_combined.sum(axis=2)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for idx, (sal, title) in enumerate(
        [
            (saliency_x_agg, "Action X"),
            (saliency_y_agg, "Action Y"),
            (saliency_combined_agg, "Combined"),
        ]
    ):
        vmax = np.percentile(sal, 99)
        if vmax > 0:
            sal_norm = np.clip(sal / vmax, 0, 1)
        else:
            sal_norm = sal

        im = axes[idx + 1].imshow(sal_norm, cmap="hot", vmin=0, vmax=1)
        axes[idx + 1].set_title(f"Saliency: {title}")
        axes[idx + 1].axis("off")
        plt.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)

    plt.suptitle("Aggregated Saliency Maps (sum over RGB)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_aggregated.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated saliency to {output_dir / 'saliency_aggregated.png'}")

    # --- Overlay Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (sal, title) in enumerate(
        [
            (saliency_x_agg, "Action X"),
            (saliency_y_agg, "Action Y"),
            (saliency_combined_agg, "Combined"),
        ]
    ):
        # Normalize saliency
        vmax = np.percentile(sal, 99)
        if vmax > 0:
            sal_norm = np.clip(sal / vmax, 0, 1)
        else:
            sal_norm = sal

        # Create overlay: blend original image with saliency heatmap
        axes[idx].imshow(image)
        im = axes[idx].imshow(sal_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[idx].set_title(f"Overlay: {title}")
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle("Saliency Overlay on Original Image", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay saliency to {output_dir / 'saliency_overlay.png'}")

    # --- Binned Overlay Visualization (16x16 bins) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Aggregate saliency into 16x16 bins (224/16 = 14 pixels per bin)
    n_bins = 16
    bin_size = image.shape[0] // n_bins  # 14 for 224x224 image

    for idx, (sal, title) in enumerate(
        [
            (saliency_x_agg, "Action X"),
            (saliency_y_agg, "Action Y"),
            (saliency_combined_agg, "Combined"),
        ]
    ):
        # Aggregate into bins by reshaping and summing
        # Reshape to (n_bins, bin_size, n_bins, bin_size) then sum over bin dimensions
        sal_binned = sal[: n_bins * bin_size, : n_bins * bin_size]  # Crop to exact size
        sal_binned = sal_binned.reshape(n_bins, bin_size, n_bins, bin_size)
        sal_binned = sal_binned.sum(axis=(1, 3))  # (16, 16)

        # Normalize binned saliency
        vmax = np.percentile(sal_binned, 99)
        if vmax > 0:
            sal_binned_norm = np.clip(sal_binned / vmax, 0, 1)
        else:
            sal_binned_norm = sal_binned

        # Upsample back to image size for overlay
        sal_upsampled = np.repeat(
            np.repeat(sal_binned_norm, bin_size, axis=0), bin_size, axis=1
        )
        # Pad if needed to match original image size
        if sal_upsampled.shape[0] < image.shape[0]:
            pad_h = image.shape[0] - sal_upsampled.shape[0]
            pad_w = image.shape[1] - sal_upsampled.shape[1]
            sal_upsampled = np.pad(sal_upsampled, ((0, pad_h), (0, pad_w)), mode="edge")

        # Create overlay
        axes[idx].imshow(image)
        im = axes[idx].imshow(sal_upsampled, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[idx].set_title(f"Binned Overlay: {title}")
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Draw grid lines to show bins
        for i in range(1, n_bins):
            axes[idx].axhline(y=i * bin_size, color="white", linewidth=0.3, alpha=0.5)
            axes[idx].axvline(x=i * bin_size, color="white", linewidth=0.3, alpha=0.5)

    plt.suptitle("Saliency Overlay (16x16 bins)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_dir / "saliency_overlay_binned.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(
        f"Saved binned overlay saliency to {output_dir / 'saliency_overlay_binned.png'}"
    )

    # --- Signed Saliency (showing positive vs negative gradients) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    grad_x_np = grad_action_x.squeeze(0).cpu().numpy()
    grad_y_np = grad_action_y.squeeze(0).cpu().numpy()

    for row_idx, (grad, title) in enumerate(
        [(grad_x_np, "Action X"), (grad_y_np, "Action Y")]
    ):
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{title}\nOriginal")
        axes[row_idx, 0].axis("off")

        for ch_idx, ch_name in enumerate(channel_names):
            channel_grad = grad[:, :, ch_idx]

            # Use symmetric colormap for signed gradients
            vmax = np.percentile(np.abs(channel_grad), 99)
            if vmax == 0:
                vmax = 1

            im = axes[row_idx, ch_idx + 1].imshow(
                channel_grad, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            axes[row_idx, ch_idx + 1].set_title(f"{title}\n{ch_name} (signed)")
            axes[row_idx, ch_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[row_idx, ch_idx + 1], fraction=0.046, pad=0.04)

    plt.suptitle("Signed Gradient Saliency (Red=positive, Blue=negative)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_signed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved signed saliency to {output_dir / 'saliency_signed.png'}")

    return {
        "saliency_x": saliency_x,
        "saliency_y": saliency_y,
        "saliency_combined": saliency_combined,
        "grad_x": grad_x_np,
        "grad_y": grad_y_np,
    }


def main():
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    # Load config
    config = Config()
    nonvision_dim = config.actor_nonvision_input_dim  # 2
    vision_dim = config.vision_head_input_dim * config.n_cameras  # 384

    print(f"\nConfig: nonvision_dim={nonvision_dim}, vision_dim={vision_dim}")

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load replay buffer for non-vision features
    print(f"\nLoading replay buffer from {args.buffer_path}...")
    buffer: ReplayBufferGpu = joblib.load(args.buffer_path)
    print(f"Buffer size: {buffer.size()} transitions")

    # Get first observation's non-vision features (first 2 dimensions)
    first_obs = buffer.observations[0].to(device)
    nonvision_features = first_obs[:nonvision_dim].unsqueeze(0)  # (1, 2)
    print(
        f"Non-vision features from first observation: {nonvision_features.cpu().numpy()}"
    )

    # Load input image
    image_tensor = load_and_preprocess_image(args.image_path, device)
    original_image = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    # Load DINOv2 model (no JIT)
    dino_model = load_dino_model(args.dino_model_name, device)

    # Load policy models
    print("\nLoading policy models...")
    shared_encoder, actor = load_models(args.checkpoint_dir, config, device)

    # Compute RGB saliency
    print("\nComputing RGB saliency maps...")
    grad_x, grad_y, actions = compute_rgb_saliency(
        image_tensor,
        nonvision_features,
        dino_model,
        shared_encoder,
        actor,
        config,
        device,
    )

    print(f"\nPredicted actions: {actions.detach().cpu().numpy()}")

    # Create visualizations
    print("\nGenerating saliency visualizations...")
    results = create_saliency_visualization(original_image, grad_x, grad_y, output_dir)

    # Save raw results
    np.savez(
        output_dir / "saliency_rgb_results.npz",
        saliency_x=results["saliency_x"],
        saliency_y=results["saliency_y"],
        saliency_combined=results["saliency_combined"],
        grad_x=results["grad_x"],
        grad_y=results["grad_y"],
        nonvision_features=nonvision_features.cpu().numpy(),
        actions=actions.detach().cpu().numpy(),
    )
    print(f"\nRaw results saved to {output_dir / 'saliency_rgb_results.npz'}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SALIENCY SUMMARY")
    print("=" * 60)

    for name, sal in [
        ("Action X", results["saliency_x"]),
        ("Action Y", results["saliency_y"]),
        ("Combined", results["saliency_combined"]),
    ]:
        print(f"\n{name}:")
        for ch_idx, ch_name in enumerate(["Red", "Green", "Blue"]):
            ch_sal = sal[:, :, ch_idx]
            print(
                f"  {ch_name}: mean={ch_sal.mean():.6f}, max={ch_sal.max():.6f}, "
                f"std={ch_sal.std():.6f}"
            )
        agg_sal = sal.sum(axis=2)
        print(
            f"  Aggregated: mean={agg_sal.mean():.6f}, max={agg_sal.max():.6f}, "
            f"std={agg_sal.std():.6f}"
        )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
