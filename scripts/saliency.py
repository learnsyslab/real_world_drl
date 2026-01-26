"""
Saliency analysis script for understanding feature importance in a trained RL policy.

This script loads a replay buffer, policy, shared encoder, and Q-function,
then computes gradient-based saliency maps to understand which input features
are most important for the policy's actions and Q-value predictions.
"""

import argparse
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from crisp_drl.data.buffers_cleanrl import ReplayBufferGpu
from crisp_drl.agents.shared.config import Config
from crisp_drl.agents.shared.networks_cleanrl import SharedEncoder, SoftQNetwork, Actor


def parse_args():
    parser = argparse.ArgumentParser(description="Compute saliency maps for RL policy")
    parser.add_argument(
        "--buffer_path",
        type=str,
        default="rollout_data/collect_data_real/env_v0_0.8/replay_buffer.joblib",
        help="Path to the replay buffer joblib file",
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
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for computing gradients",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)",
    )
    return parser.parse_args()


def load_models(checkpoint_dir: str, config: Config, device: str):
    """Load the actor, shared encoder, and Q-function from checkpoints."""
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

    # Load Q-function
    qf1 = SoftQNetwork(config).to(device)
    qf1_path = checkpoint_path / "qf1_state_dict.pth"
    if qf1_path.exists():
        qf1.load_state_dict(torch.load(qf1_path, map_location=device))
        print(f"Loaded Q-function from {qf1_path}")
    else:
        raise FileNotFoundError(f"Q-function not found at {qf1_path}")

    return shared_encoder, actor, qf1


def compute_feature_statistics(
    observations: torch.Tensor, nonvision_dim: int, vision_dim: int
):
    """Compute mean and std for non-vision and vision features separately."""
    N, D = observations.shape

    # Non-vision features (first nonvision_dim)
    nonvision_obs = observations[:, :nonvision_dim]
    nonvision_mean = nonvision_obs.mean(dim=0)
    nonvision_std = nonvision_obs.std(dim=0)

    # Vision features (last vision_dim)
    vision_obs = observations[:, -vision_dim:]
    vision_mean = vision_obs.mean(dim=0)
    vision_std = vision_obs.std(dim=0)

    return {
        "nonvision_mean": nonvision_mean,
        "nonvision_std": nonvision_std,
        "vision_mean": vision_mean,
        "vision_std": vision_std,
    }


def compute_action_gradients(
    observations: torch.Tensor,
    shared_encoder: nn.Module,
    actor: nn.Module,
    batch_size: int,
    device: str,
):
    """
    Compute gradients of actions with respect to observations.
    Returns sum of absolute gradients for each action dimension.
    """
    shared_encoder.eval()
    actor.eval()

    N, D = observations.shape
    num_batches = (N + batch_size - 1) // batch_size

    # Accumulators for gradient sums
    grad_sum_action_x = torch.zeros(D, device=device)
    grad_sum_action_y = torch.zeros(D, device=device)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        batch_obs = observations[start_idx:end_idx].clone().requires_grad_(True)

        # Forward pass through encoder and actor
        encoded = shared_encoder(batch_obs)
        actions, _ = actor(encoded)

        # Compute gradients for action x (first dimension)
        grad_x = torch.autograd.grad(
            outputs=actions[:, 0].sum(),
            inputs=batch_obs,
            retain_graph=True,
        )[0]
        grad_sum_action_x += grad_x.abs().sum(dim=0)

        # Compute gradients for action y (second dimension)
        grad_y = torch.autograd.grad(
            outputs=actions[:, 1].sum(),
            inputs=batch_obs,
        )[0]
        grad_sum_action_y += grad_y.abs().sum(dim=0)

        if (i + 1) % 10 == 0:
            print(f"  Processed batch {i + 1}/{num_batches}")

    return grad_sum_action_x, grad_sum_action_y


def compute_qvalue_gradients(
    observations: torch.Tensor,
    shared_encoder: nn.Module,
    actor: nn.Module,
    qf: nn.Module,
    batch_size: int,
    device: str,
):
    """
    Compute gradients of Q-values with respect to observations.
    Returns sum of absolute gradients.
    """
    shared_encoder.eval()
    actor.eval()
    qf.eval()

    N, D = observations.shape
    num_batches = (N + batch_size - 1) // batch_size

    # Accumulator for gradient sums
    grad_sum_q = torch.zeros(D, device=device)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        batch_obs = observations[start_idx:end_idx].clone().requires_grad_(True)

        # Forward pass through encoder, actor, and Q-function
        encoded = shared_encoder(batch_obs)
        actions, _ = actor(encoded)
        q_values = qf(encoded, actions)

        # Compute gradients for Q-value
        grad_q = torch.autograd.grad(
            outputs=q_values.sum(),
            inputs=batch_obs,
        )[0]
        grad_sum_q += grad_q.abs().sum(dim=0)

        if (i + 1) % 10 == 0:
            print(f"  Processed batch {i + 1}/{num_batches}")

    return grad_sum_q


def analyze_saliency(
    grad_sum: torch.Tensor,
    std: torch.Tensor,
    feature_names: list[str],
    title: str,
):
    """Analyze and print saliency results."""
    # Normalize by std (avoid division by zero)
    std_safe = std.clone()
    std_safe[std_safe < 1e-8] = 1e-8
    normalized_saliency = grad_sum / std_safe

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    print("\nRaw gradient sums (sum of |dOutput/dFeature|):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {grad_sum[i].item():.6f}")

    # Compute relative importance (percentage)
    total = grad_sum.sum()
    print("\nRelative importance raw (%):")
    for i, name in enumerate(feature_names):
        pct = 100.0 * grad_sum[i] / total
        print(f"  {name}: {pct.item():.2f}%")

    print("\nStd-normalized saliency (gradient_sum / std):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {normalized_saliency[i].item():.6f}")

    # Compute relative importance (percentage)
    total = normalized_saliency.sum()
    print("\nRelative importance normalized (%):")
    for i, name in enumerate(feature_names):
        pct = 100.0 * normalized_saliency[i] / total
        print(f"  {name}: {pct.item():.2f}%")

    return normalized_saliency


def main():
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    # Load config
    config = Config()
    nonvision_dim = config.actor_nonvision_input_dim  # 8
    vision_dim = config.vision_head_input_dim * config.n_cameras  # 384

    print(f"\nConfig: nonvision_dim={nonvision_dim}, vision_dim={vision_dim}")

    # Load replay buffer
    print(f"\nLoading replay buffer from {args.buffer_path}...")
    buffer: ReplayBufferGpu = joblib.load(args.buffer_path)
    print(f"Buffer size: {buffer.size()} transitions")

    # Get observations
    num_samples = args.num_samples if args.num_samples else buffer.pos_obs
    observations = buffer.observations[:num_samples].to(device)
    print(f"Using {num_samples} samples")

    # Verify dimensions
    N, D = observations.shape
    expected_dim = nonvision_dim + vision_dim
    print(f"Observation shape: {observations.shape}")
    print(f"Expected dimension: {expected_dim}")

    if D != expected_dim:
        print("Warning: Dimension mismatch! Adjusting vision_dim...")
        vision_dim = D - nonvision_dim
        print(f"Adjusted vision_dim: {vision_dim}")

    # Compute feature statistics
    print("\nComputing feature statistics...")
    stats = compute_feature_statistics(observations, nonvision_dim, vision_dim)

    print("\nNon-vision feature statistics:")
    print(f"  Mean: {stats['nonvision_mean'].cpu().numpy()}")
    print(f"  Std: {stats['nonvision_std'].cpu().numpy()}")

    print("\nVision feature statistics (summary):")
    print(
        f"  Mean: min={stats['vision_mean'].min().item():.4f}, max={stats['vision_mean'].max().item():.4f}, mean={stats['vision_mean'].mean().item():.4f}"
    )
    print(
        f"  Std: min={stats['vision_std'].min().item():.4f}, max={stats['vision_std'].max().item():.4f}, mean={stats['vision_std'].mean().item():.4f}"
    )

    # Load models
    print("\nLoading models...")
    shared_encoder, actor, qf1 = load_models(args.checkpoint_dir, config, device)

    # Compute action gradients
    print("\nComputing action gradients...")
    grad_sum_action_x, grad_sum_action_y = compute_action_gradients(
        observations, shared_encoder, actor, args.batch_size, device
    )

    # Compute Q-value gradients
    print("\nComputing Q-value gradients...")
    grad_sum_q = compute_qvalue_gradients(
        observations, shared_encoder, actor, qf1, args.batch_size, device
    )

    # Feature names for non-vision (first 8)
    nonvision_names = [f"nonvision_{i}" for i in range(nonvision_dim)]

    # === Analysis for Action X ===
    print("\n" + "=" * 80)
    print("ACTION X SALIENCY ANALYSIS")
    print("=" * 80)

    # Non-vision features
    nonvision_saliency_x = analyze_saliency(
        grad_sum_action_x[:nonvision_dim],
        stats["nonvision_std"],
        nonvision_names,
        "Action X - Non-vision Features",
    )

    # Vision features (aggregated)
    vision_grad_x = grad_sum_action_x[-vision_dim:]
    vision_std = stats["vision_std"]
    vision_std_safe = vision_std.clone()
    vision_std_safe[vision_std_safe < 1e-8] = 1e-8
    vision_normalized_x = vision_grad_x / vision_std_safe

    print(f"\nAction X - Vision Features (aggregated over {vision_dim} dimensions):")
    print(f"  Total raw gradient sum: {vision_grad_x.sum().item():.6f}")
    print(f"  Total std-normalized: {vision_normalized_x.sum().item():.6f}")
    print(f"  Mean std-normalized per feature: {vision_normalized_x.mean().item():.6f}")

    # === Analysis for Action Y ===
    print("\n" + "=" * 80)
    print("ACTION Y SALIENCY ANALYSIS")
    print("=" * 80)

    # Non-vision features
    nonvision_saliency_y = analyze_saliency(
        grad_sum_action_y[:nonvision_dim],
        stats["nonvision_std"],
        nonvision_names,
        "Action Y - Non-vision Features",
    )

    # Vision features (aggregated)
    vision_grad_y = grad_sum_action_y[-vision_dim:]
    vision_normalized_y = vision_grad_y / vision_std_safe

    print(f"\nAction Y - Vision Features (aggregated over {vision_dim} dimensions):")
    print(f"  Total raw gradient sum: {vision_grad_y.sum().item():.6f}")
    print(f"  Total std-normalized: {vision_normalized_y.sum().item():.6f}")
    print(f"  Mean std-normalized per feature: {vision_normalized_y.mean().item():.6f}")

    # === Analysis for Q-value ===
    print("\n" + "=" * 80)
    print("Q-VALUE SALIENCY ANALYSIS")
    print("=" * 80)

    # Non-vision features
    nonvision_saliency_q = analyze_saliency(
        grad_sum_q[:nonvision_dim],
        stats["nonvision_std"],
        nonvision_names,
        "Q-value - Non-vision Features",
    )

    # Vision features (aggregated)
    vision_grad_q = grad_sum_q[-vision_dim:]
    vision_normalized_q = vision_grad_q / vision_std_safe

    print(f"\nQ-value - Vision Features (aggregated over {vision_dim} dimensions):")
    print(f"  Total raw gradient sum: {vision_grad_q.sum().item():.6f}")
    print(f"  Total std-normalized: {vision_normalized_q.sum().item():.6f}")
    print(f"  Mean std-normalized per feature: {vision_normalized_q.mean().item():.6f}")

    # === Summary comparison ===
    print("\n" + "=" * 80)
    print("SUMMARY: Non-vision vs Vision Feature Importance")
    print("=" * 80)

    # Compute total importance for non-vision vs vision
    total_nonvision_x = nonvision_saliency_x.sum().item()
    total_vision_x = vision_normalized_x.sum().item()
    total_x = total_nonvision_x + total_vision_x

    total_nonvision_y = nonvision_saliency_y.sum().item()
    total_vision_y = vision_normalized_y.sum().item()
    total_y = total_nonvision_y + total_vision_y

    total_nonvision_q = nonvision_saliency_q.sum().item()
    total_vision_q = vision_normalized_q.sum().item()
    total_q = total_nonvision_q + total_vision_q

    print("\nAction X:")
    print(
        f"  Non-vision total: {total_nonvision_x:.4f} ({100 * total_nonvision_x / total_x:.1f}%)"
    )
    print(
        f"  Vision total: {total_vision_x:.4f} ({100 * total_vision_x / total_x:.1f}%)"
    )

    print("\nAction Y:")
    print(
        f"  Non-vision total: {total_nonvision_y:.4f} ({100 * total_nonvision_y / total_y:.1f}%)"
    )
    print(
        f"  Vision total: {total_vision_y:.4f} ({100 * total_vision_y / total_y:.1f}%)"
    )

    print("\nQ-value:")
    print(
        f"  Non-vision total: {total_nonvision_q:.4f} ({100 * total_nonvision_q / total_q:.1f}%)"
    )
    print(
        f"  Vision total: {total_vision_q:.4f} ({100 * total_vision_q / total_q:.1f}%)"
    )

    # === Top 20 Vision Features Analysis ===
    print("\n" + "=" * 80)
    print("TOP 20 VISION FEATURES vs REST")
    print("=" * 80)

    def analyze_top_k_vision(normalized_vision: torch.Tensor, name: str, k: int = 20):
        """Analyze top-k vision features compared to the rest."""
        # Get top-k indices by normalized saliency
        topk_values, topk_indices = torch.topk(normalized_vision, k)
        rest_mask = torch.ones(
            normalized_vision.shape[0],
            dtype=torch.bool,
            device=normalized_vision.device,
        )
        rest_mask[topk_indices] = False
        rest_values = normalized_vision[rest_mask]

        total = normalized_vision.sum().item()
        topk_sum = topk_values.sum().item()
        rest_sum = rest_values.sum().item()

        print(f"\n{name}:")
        print(f"  Top {k} features:")
        print(f"    Total saliency: {topk_sum:.4f} ({100 * topk_sum / total:.1f}%)")
        print(f"    Mean saliency: {topk_values.mean().item():.4f}")
        print(f"    Min in top-{k}: {topk_values.min().item():.4f}")
        print(f"    Max in top-{k}: {topk_values.max().item():.4f}")
        print(f"    Indices: {topk_indices.cpu().numpy()}")
        print(f"  Remaining {len(rest_values)} features:")
        print(f"    Total saliency: {rest_sum:.4f} ({100 * rest_sum / total:.1f}%)")
        print(f"    Mean saliency: {rest_values.mean().item():.4f}")
        print(f"    Min: {rest_values.min().item():.4f}")
        print(f"    Max: {rest_values.max().item():.4f}")
        print(f"  Ratio (top-{k} / rest): {topk_sum / rest_sum:.2f}x")

        return topk_indices.cpu().numpy(), topk_values.cpu().numpy()

    k_ = 100
    topk_indices_x, topk_values_x = analyze_top_k_vision(
        vision_normalized_x, "Action X - Vision", k=k_
    )
    topk_indices_y, topk_values_y = analyze_top_k_vision(
        vision_normalized_y, "Action Y - Vision", k=k_
    )
    topk_indices_q, topk_values_q = analyze_top_k_vision(
        vision_normalized_q, "Q-value - Vision", k=k_
    )

    # Check overlap between top features
    print("\n" + "-" * 60)
    print(f"Overlap Analysis of Top {k_} Vision Features")
    print("-" * 60)

    set_x = set(topk_indices_x)
    set_y = set(topk_indices_y)
    set_q = set(topk_indices_q)

    overlap_xy = set_x & set_y
    overlap_xq = set_x & set_q
    overlap_yq = set_y & set_q
    overlap_all = set_x & set_y & set_q

    print(f"  Action X ∩ Action Y: {len(overlap_xy)} features")
    print(f"  Action X ∩ Q-value: {len(overlap_xq)} features")
    print(f"  Action Y ∩ Q-value: {len(overlap_yq)} features")
    print(f"  All three: {len(overlap_all)} features")
    if overlap_all:
        print(f"    Common indices: {sorted(overlap_all)}")

    # === Pareto Plots for Vision Feature Saliency ===
    print("\n" + "=" * 80)
    print("GENERATING PARETO PLOTS FOR VISION FEATURES")
    print("=" * 80)

    def create_pareto_plot(
        normalized_saliency: torch.Tensor,
        title: str,
        output_path: Path,
    ):
        """Create a Pareto plot showing sorted saliency with cumulative percentage."""
        # Sort by saliency in descending order
        sorted_vals, sorted_indices = torch.sort(normalized_saliency, descending=True)
        sorted_vals = sorted_vals.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

        # Compute cumulative percentage
        total = sorted_vals.sum()
        cumulative = np.cumsum(sorted_vals) / total * 100

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Bar plot for individual saliency values
        x = np.arange(len(sorted_vals))
        bars = ax1.bar(x, sorted_vals, color="steelblue", alpha=0.7, label="Saliency")
        ax1.set_xlabel("Vision Feature (sorted by importance)", fontsize=12)
        ax1.set_ylabel("Normalized Saliency", color="steelblue", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="steelblue")

        # Line plot for cumulative percentage on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            cumulative,
            color="darkorange",
            linewidth=2,
            marker="",
            label="Cumulative %",
        )
        ax2.set_ylabel("Cumulative Percentage (%)", color="darkorange", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.set_ylim(0, 105)

        # Add horizontal lines at key percentages
        for pct in [50, 80, 90, 95]:
            ax2.axhline(y=pct, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            # Find how many features needed to reach this percentage
            n_features = np.searchsorted(cumulative, pct) + 1
            ax2.annotate(
                f"{pct}% @ {n_features} features",
                xy=(n_features, pct),
                xytext=(n_features + 20, pct + 2),
                fontsize=9,
                color="gray",
            )

        ax1.set_title(title, fontsize=14)
        ax1.set_xlim(-1, len(sorted_vals))

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Print summary statistics
        print(f"\n{title}:")
        print(f"  Features for 50% saliency: {np.searchsorted(cumulative, 50) + 1}")
        print(f"  Features for 80% saliency: {np.searchsorted(cumulative, 80) + 1}")
        print(f"  Features for 90% saliency: {np.searchsorted(cumulative, 90) + 1}")
        print(f"  Features for 95% saliency: {np.searchsorted(cumulative, 95) + 1}")
        print(f"  Saved plot to: {output_path}")

        return sorted_indices, cumulative

    checkpoint_path = Path(args.checkpoint_dir)

    # Create Pareto plots for each output
    pareto_indices_x, pareto_cumulative_x = create_pareto_plot(
        vision_normalized_x,
        "Pareto Plot: Vision Feature Saliency for Action X",
        checkpoint_path / "pareto_vision_action_x.png",
    )

    pareto_indices_y, pareto_cumulative_y = create_pareto_plot(
        vision_normalized_y,
        "Pareto Plot: Vision Feature Saliency for Action Y",
        checkpoint_path / "pareto_vision_action_y.png",
    )

    pareto_indices_q, pareto_cumulative_q = create_pareto_plot(
        vision_normalized_q,
        "Pareto Plot: Vision Feature Saliency for Q-value",
        checkpoint_path / "pareto_vision_q.png",
    )

    # Create combined Pareto plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (norm_sal, title_short) in zip(
        axes,
        [
            (vision_normalized_x, "Action X"),
            (vision_normalized_y, "Action Y"),
            (vision_normalized_q, "Q-value"),
        ],
    ):
        sorted_vals, _ = torch.sort(norm_sal, descending=True)
        sorted_vals = sorted_vals.cpu().numpy()
        cumulative = np.cumsum(sorted_vals) / sorted_vals.sum() * 100
        x = np.arange(len(sorted_vals))

        ax2 = ax.twinx()
        ax.bar(x, sorted_vals, color="steelblue", alpha=0.7)
        ax2.plot(x, cumulative, color="darkorange", linewidth=2)
        ax.set_title(title_short, fontsize=12)
        ax.set_xlabel("Feature rank")
        ax.set_ylabel("Saliency", color="steelblue")
        ax2.set_ylabel("Cumulative %", color="darkorange")
        ax2.set_ylim(0, 105)
        for pct in [80, 95]:
            ax2.axhline(y=pct, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Pareto Analysis: Vision Feature Saliency", fontsize=14)
    plt.tight_layout()
    combined_path = checkpoint_path / "pareto_vision_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved combined Pareto plot to: {combined_path}")

    # Save results
    results = {
        "nonvision_mean": stats["nonvision_mean"].cpu().numpy(),
        "nonvision_std": stats["nonvision_std"].cpu().numpy(),
        "vision_mean": stats["vision_mean"].cpu().numpy(),
        "vision_std": stats["vision_std"].cpu().numpy(),
        "grad_sum_action_x": grad_sum_action_x.cpu().numpy(),
        "grad_sum_action_y": grad_sum_action_y.cpu().numpy(),
        "grad_sum_q": grad_sum_q.cpu().numpy(),
        "normalized_saliency_action_x_nonvision": nonvision_saliency_x.cpu().numpy(),
        "normalized_saliency_action_y_nonvision": nonvision_saliency_y.cpu().numpy(),
        "normalized_saliency_q_nonvision": nonvision_saliency_q.cpu().numpy(),
        "normalized_saliency_action_x_vision": vision_normalized_x.cpu().numpy(),
        "normalized_saliency_action_y_vision": vision_normalized_y.cpu().numpy(),
        "normalized_saliency_q_vision": vision_normalized_q.cpu().numpy(),
        "top20_vision_indices_action_x": topk_indices_x,
        "top20_vision_indices_action_y": topk_indices_y,
        "top20_vision_indices_q": topk_indices_q,
        "top20_vision_values_action_x": topk_values_x,
        "top20_vision_values_action_y": topk_values_y,
        "top20_vision_values_q": topk_values_q,
    }

    output_path = Path(args.checkpoint_dir) / "saliency_results.npz"
    np.savez(output_path, **results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
