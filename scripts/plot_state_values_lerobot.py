import argparse
import re
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.networks_cleanrl import Actor, SharedEncoder, SoftQNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-rollout state values V(s)=Q(s, pi(s)) from a LeRobot dataset, "
            "split into successful and truncated episodes."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("rollout_data/collect_data_real/run_s_1b_0.82"),
        help="Path to the LeRobot dataset directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/rw_1cft_d300vs1b/pretrain_5"),
        help="Path to the checkpoint directory containing actor/shared encoder/Q-functions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for generated plots (default: plots/state_values_<dataset_name>).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference.",
    )
    return parser.parse_args()


def _extract_episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        return int(1e12)
    return int(match.group(1))


def find_episode_files(dataset_dir: Path) -> list[Path]:
    episode_files = sorted(
        dataset_dir.glob("data/chunk-*/episode_*.parquet"),
        key=_extract_episode_id,
    )
    if not episode_files:
        raise FileNotFoundError(
            f"No episode parquet files found under {dataset_dir / 'data'}."
        )
    return episode_files


def load_episode_observations_and_rewards(
    episode_parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        episode_parquet_path,
        columns=["observation.formatted", "reward"],
    )

    obs_raw = table.column("observation.formatted").to_pylist()
    observations = np.stack([np.asarray(x, dtype=np.float32) for x in obs_raw], axis=0)

    rew_raw = table.column("reward").to_pylist()
    rewards = np.asarray(
        [float(np.asarray(x).squeeze()) for x in rew_raw], dtype=np.float32
    )

    return observations, rewards


def infer_config_from_checkpoint(checkpoint_dir: Path, n_q_functions: int) -> Config:
    actor_state_dict = torch.load(
        checkpoint_dir / "actor_state_dict.pth", map_location="cpu"
    )
    shared_encoder_state_dict = torch.load(
        checkpoint_dir / "shared_encoder_state_dict.pth", map_location="cpu"
    )
    qf1_state_dict = torch.load(
        checkpoint_dir / "qf1_state_dict.pth", map_location="cpu"
    )

    camera_indices = set()
    for key in shared_encoder_state_dict.keys():
        match = re.match(r"image_encoders\.(\d+)\.fc1\.weight", key)
        if match:
            camera_indices.add(int(match.group(1)))
    n_cameras = len(camera_indices) if camera_indices else 1

    vision_head_input_dim = int(
        shared_encoder_state_dict["image_encoders.0.fc1.weight"].shape[1]
    )
    vision_head_hidden_dim = int(
        shared_encoder_state_dict["image_encoders.0.fc1.weight"].shape[0]
    )
    vision_head_output_dim = int(
        shared_encoder_state_dict["image_encoders.0.fc_mu.weight"].shape[0]
    )

    actor_input_dim = int(actor_state_dict["net.0.weight"].shape[1])
    actor_output_dim = int(actor_state_dict["fc_mean.weight"].shape[0])
    actor_nonvision_input_dim = actor_input_dim - vision_head_output_dim * n_cameras
    actor_q_hidden_dim = int(qf1_state_dict["fc1.weight"].shape[0])

    max_action_tensor = actor_state_dict.get("action_scale", None)
    max_action = (
        max_action_tensor.cpu().numpy()
        if isinstance(max_action_tensor, torch.Tensor)
        else Config().max_action
    )
    actor_std_tensor = actor_state_dict.get("std", None)
    actor_std = (
        float(actor_std_tensor.item())
        if isinstance(actor_std_tensor, torch.Tensor)
        else Config().actor_std
    )

    return Config(
        n_cameras=n_cameras,
        vision_head_input_dim=vision_head_input_dim,
        vision_head_hidden_dim=vision_head_hidden_dim,
        vision_head_output_dim=vision_head_output_dim,
        actor_nonvision_input_dim=actor_nonvision_input_dim,
        actor_output_dim=actor_output_dim,
        actor_q_hidden_dim=actor_q_hidden_dim,
        max_action=max_action,
        actor_std=actor_std,
        num_critics=n_q_functions,
    )


def get_q_paths(checkpoint_dir: Path) -> list[Path]:
    q_paths: list[Path] = []
    pattern = re.compile(r"qf(\d+)_state_dict\.pth$")

    for path in checkpoint_dir.glob("qf*_state_dict.pth"):
        if path.name.endswith("_target_state_dict.pth"):
            continue
        if pattern.match(path.name):
            q_paths.append(path)

    if not q_paths:
        raise FileNotFoundError(
            f"No Q-function state dicts found under {checkpoint_dir}."
        )

    def _q_index(path: Path) -> int:
        match = pattern.match(path.name)
        if match is None:
            return int(1e12)
        return int(match.group(1))

    q_paths = sorted(q_paths, key=_q_index)
    return q_paths


def load_models(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[SharedEncoder, Actor, list[SoftQNetwork], Config]:
    q_paths = get_q_paths(checkpoint_dir)
    config = infer_config_from_checkpoint(checkpoint_dir, n_q_functions=len(q_paths))

    shared_encoder = SharedEncoder(config).to(device)
    shared_encoder.load_state_dict(
        torch.load(
            checkpoint_dir / "shared_encoder_state_dict.pth", map_location=device
        )
    )
    shared_encoder.eval()

    actor = Actor(config).to(device)
    actor.load_state_dict(
        torch.load(checkpoint_dir / "actor_state_dict.pth", map_location=device)
    )
    actor.eval()

    q_functions: list[SoftQNetwork] = []
    for q_path in q_paths:
        qf = SoftQNetwork(config).to(device)
        qf.load_state_dict(torch.load(q_path, map_location=device))
        qf.eval()
        q_functions.append(qf)

    return shared_encoder, actor, q_functions, config


def actor_mean_action(actor: Actor, encoded_obs: torch.Tensor) -> torch.Tensor:
    hidden = actor.net(encoded_obs)
    mean = actor.fc_mean(hidden)
    squashed_mean = torch.tanh(mean)
    action_scale = cast(torch.Tensor, actor.action_scale)
    action_bias = cast(torch.Tensor, actor.action_bias)
    return squashed_mean * action_scale + action_bias


def compute_state_values(
    observations: np.ndarray,
    shared_encoder: SharedEncoder,
    actor: Actor,
    q_functions: list[SoftQNetwork],
    device: torch.device,
) -> np.ndarray:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)

    with torch.no_grad():
        encoded_obs = shared_encoder(obs_tensor)
        pi_actions = actor_mean_action(actor, encoded_obs)
        q_values = torch.stack(
            [qf(encoded_obs, pi_actions).squeeze(-1) for qf in q_functions], dim=0
        )
        v_values = q_values.mean(dim=0)

    return v_values.cpu().numpy()


def classify_episode(terminal_reward: float) -> str:
    if terminal_reward > 0.0:
        return "success"
    if np.isclose(terminal_reward, 0.0, atol=1e-8):
        return "truncated"
    return "other"


def plot_success_aligned(success_values: list[np.ndarray], output_path: Path) -> None:
    if not success_values:
        print("No successful episodes found. Skipping successful plot.")
        return

    max_len = max(len(v) for v in success_values)

    plt.figure(figsize=(12, 7))
    for values in success_values:
        x = np.arange(max_len - len(values), max_len)
        plt.plot(x, values, alpha=0.65, linewidth=1.0)

    plt.xlabel("Aligned timestep (all successful episodes end at same x)")
    plt.ylabel("State value V(s) = mean_i Q_i(s, pi(s))")
    plt.title(f"Successful rollouts (n={len(success_values)}), end-aligned")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved successful plot to: {output_path}")


def plot_truncated(truncated_values: list[np.ndarray], output_path: Path) -> None:
    if not truncated_values:
        print("No truncated episodes found. Skipping truncated plot.")
        return

    plt.figure(figsize=(12, 7))
    for values in truncated_values:
        x = np.arange(len(values))
        plt.plot(x, values, alpha=0.65, linewidth=1.0)

    plt.xlabel("Timestep")
    plt.ylabel("State value V(s) = mean_i Q_i(s, pi(s))")
    plt.title(f"Truncated rollouts (n={len(truncated_values)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved truncated plot to: {output_path}")


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {args.checkpoint_dir}"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("plots") / f"state_values_{args.dataset_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    episode_files = find_episode_files(args.dataset_dir)
    print(f"Found {len(episode_files)} episode files.")

    shared_encoder, actor, q_functions, config = load_models(
        args.checkpoint_dir, device
    )

    expected_obs_dim = (
        config.actor_nonvision_input_dim
        + config.vision_head_input_dim * config.n_cameras
    )
    print(
        "Loaded models with config: "
        f"n_cameras={config.n_cameras}, "
        f"obs_dim={expected_obs_dim}, "
        f"action_dim={config.actor_output_dim}, "
        f"n_q={len(q_functions)}"
    )

    success_values: list[np.ndarray] = []
    truncated_values: list[np.ndarray] = []
    other_count = 0

    for episode_path in episode_files:
        observations, rewards = load_episode_observations_and_rewards(episode_path)
        if observations.shape[1] != expected_obs_dim:
            raise ValueError(
                f"Observation dimension mismatch in {episode_path}: "
                f"got {observations.shape[1]}, expected {expected_obs_dim}."
            )

        finite_rewards = rewards[np.isfinite(rewards)]
        if finite_rewards.size == 0:
            continue
        terminal_reward = float(finite_rewards[-1])
        episode_type = classify_episode(terminal_reward)

        if episode_type == "success":
            if len(observations) <= 1:
                continue
            observations_for_values = observations[:-1]
        else:
            observations_for_values = observations

        state_values = compute_state_values(
            observations=observations_for_values,
            shared_encoder=shared_encoder,
            actor=actor,
            q_functions=q_functions,
            device=device,
        )

        if episode_type == "success":
            success_values.append(state_values)
        elif episode_type == "truncated":
            truncated_values.append(state_values)
        else:
            other_count += 1

    success_plot_path = output_dir / "state_values_success_aligned.png"
    truncated_plot_path = output_dir / "state_values_truncated.png"

    plot_success_aligned(success_values, success_plot_path)
    plot_truncated(truncated_values, truncated_plot_path)

    print("Summary:")
    print(f"  Successful episodes: {len(success_values)}")
    print(f"  Truncated episodes:  {len(truncated_values)}")
    print(f"  Other episodes:      {other_count}")


if __name__ == "__main__":
    main()
