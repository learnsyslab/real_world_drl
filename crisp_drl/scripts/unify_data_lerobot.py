u
"""Load a local LeRobot dataset and export it as a ReplayBufferGpu.

This replaces the legacy folder-based merger. It walks episodes in order,
cuts rewards/actions to align transitions (rewards[1:], actions[:-1]), and
stores observations as-is.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.data.buffers_cleanrl import ReplayBufferGpu


Episode = Tuple[list[np.ndarray], list[np.ndarray], list[float], list[bool]]


def iter_episodes(dataset: LeRobotDataset) -> Iterable[Episode]:
    """Yield full episodes from a LeRobotDataset in order.

    A new episode is recognized when the reward is NaN (marks the first frame).
    """

    obs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []

    for idx in range(len(dataset)):
        frame = dataset[idx]

        reward_val = float(np.asarray(frame["reward"]).squeeze())

        # NaN reward marks the first frame of a new episode
        if np.isnan(reward_val):
            if obs:
                yield obs, actions, rewards, terminals
            obs, actions, rewards, terminals = [], [], [], []

        obs.append(np.asarray(frame["observation.formatted"]))
        actions.append(np.asarray(frame["action"]))
        rewards.append(reward_val)
        terminals.append(bool(np.asarray(frame["is_terminal"]).squeeze()))

    # Yield the last episode if non-empty
    if obs:
        yield obs, actions, rewards, terminals
    else:
        print("[WARNING] No complete episode found at end of dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load one or more local LeRobot datasets and export them to a"
            " ReplayBufferGpu joblib."
        )
    )
    parser.add_argument(
        "--repo_id",
        nargs="+",
        type=str,
        required=True,
        metavar="REPO_ID",
        help=(
            "Local repo_id(s) inside the root directory (e.g."
            " collect_data_real/env_v0_0.8 collect_data_real/env_v0_0.9)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("rollout_data/collect_data_real"),
        help="Root directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path. If a directory, saves"
            " replay_buffer.joblib inside; if omitted, saves next to the"
            " dataset."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the replay buffer (cpu or cuda).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to convert.",
    )
    args = parser.parse_args()

    config = Config()

    episodes: list[Episode] = []
    total_obs = 0
    total_transitions = 0

    for repo_id in args.repo_id:
        dataset = LeRobotDataset(repo_id=repo_id, root=args.root / repo_id)

        for i, episode in enumerate(iter_episodes(dataset)):
            observations, actions, rewards, terminals = episode
            if len(observations) < 2:
                print(
                    f"[WARNING] Skipping episode {i} in {repo_id} with less"
                    " than 2 frames."
                )
                continue

            episodes.append(episode)
            total_obs += len(observations)
            total_transitions += len(observations) - 1

            if args.max_episodes is not None and len(episodes) >= args.max_episodes:
                break

        if args.max_episodes is not None and len(episodes) >= args.max_episodes:
            break

    if not episodes:
        raise RuntimeError("No complete episodes found in the selected datasets.")

    obs_dim = episodes[0][0][0].size
    action_dim = episodes[0][1][0].size

    print(
        f"Collected {len(episodes)} episodes, {total_transitions} transitions,"
        f" obs_dim={obs_dim}, action_dim={action_dim}."
    )

    replay_buffer = ReplayBufferGpu(
        buffer_size=total_obs,
        observation_dim=obs_dim,
        action_dim=action_dim,
        n_step_return=1,
        gamma=config.gamma,
        device=args.device,
    )

    for ep_idx, (observations, actions, rewards, terminals) in enumerate(episodes):
        # Align lengths: actions[:-1], rewards[1:], observations[:]
        trimmed_actions = actions[:-1]
        trimmed_rewards = rewards[1:]
        assert np.isfinite(trimmed_rewards).all(), (
            f"Non-finite rewards in episode {ep_idx}: {trimmed_rewards}"
        )

        if len(trimmed_actions) != len(trimmed_rewards):
            raise ValueError(
                f"Mismatch in episode {ep_idx}:"
                f" actions {len(trimmed_actions)} vs rewards {len(trimmed_rewards)}."
            )

        obs_tensors = [torch.as_tensor(o, dtype=torch.float32) for o in observations]
        act_tensors = [torch.as_tensor(a, dtype=torch.float32) for a in trimmed_actions]
        rew_list = [float(r) for r in trimmed_rewards]
        terminated = bool(terminals[-1])

        replay_buffer.add_rollout(
            obs=obs_tensors,
            action=act_tensors,
            reward=rew_list,
            terminated=terminated,
        )

    if len(args.repo_id) == 1:
        default_output = args.root / args.repo_id[0] / "replay_buffer.joblib"
    else:
        default_output = args.root / "replay_buffer.joblib"
    if args.output is None:
        output_path = default_output
    else:
        output_path = args.output
        if output_path.is_dir():
            output_path = output_path / "replay_buffer.joblib"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer.save_buffer(str(output_path))

    print(f"Saved replay buffer to {output_path}")


if __name__ == "__main__":
    main()
