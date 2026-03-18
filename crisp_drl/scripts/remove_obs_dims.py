#!/usr/bin/env python3
"""
Script to remove the first N dimensions from observations in a ReplayBufferGpu.

Usage:
    python remove_obs_dims.py <path_to_replay_buffer.joblib> [--dims N] [--output PATH]
"""

import argparse
from pathlib import Path

import torch as th
from joblib import dump, load


def remove_first_dims_from_buffer(
    buffer_path: str,
    dims_to_remove: int = 6,
    output_path: str | None = None,
) -> None:
    """
    Load a ReplayBufferGpu, remove the first N dimensions from observations,
    and save it again.

    Args:
        buffer_path: Path to the replay buffer joblib file
        dims_to_remove: Number of dimensions to remove from the start of observations
        output_path: Path to save the modified buffer (defaults to overwriting the original)
    """
    print(f"Loading buffer from: {buffer_path}")
    buffer = load(buffer_path)

    original_obs_dim = buffer.observation_dim
    original_obs_shape = buffer.obs_shape
    print(f"Original observation dim: {original_obs_dim}")
    print(f"Original obs_shape: {original_obs_shape}")
    print(f"Observations tensor shape: {buffer.observations.shape}")

    if dims_to_remove >= original_obs_dim:
        raise ValueError(
            f"Cannot remove {dims_to_remove} dimensions from observations "
            f"with only {original_obs_dim} dimensions"
        )

    # Remove first N dimensions from observations
    new_observations = buffer.observations[:, dims_to_remove:]
    new_obs_dim = original_obs_dim - dims_to_remove

    print(f"\nRemoving first {dims_to_remove} dimensions...")
    print(f"New observation dim: {new_obs_dim}")
    print(f"New observations tensor shape: {new_observations.shape}")

    # Update buffer attributes
    buffer.observations = new_observations
    buffer.observation_dim = new_obs_dim
    buffer.obs_shape = (new_obs_dim,)

    # Save the buffer
    save_path = output_path if output_path else buffer_path
    print(f"\nSaving buffer to: {save_path}")
    dump(buffer, save_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Remove the first N dimensions from observations in a ReplayBufferGpu"
    )
    parser.add_argument(
        "--buffer_path",
        type=str,
        help="Path to the replay buffer joblib file",
        required=True,
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=6,
        help="Number of dimensions to remove from the start (default: 6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the modified buffer (default: overwrite original)",
    )

    args = parser.parse_args()

    if not Path(args.buffer_path).exists():
        raise FileNotFoundError(f"Buffer file not found: {args.buffer_path}")

    remove_first_dims_from_buffer(
        buffer_path=args.buffer_path,
        dims_to_remove=args.dims,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
