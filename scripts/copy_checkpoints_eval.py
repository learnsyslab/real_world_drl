"""Copy checkpoint subtrees that contain `.jsonl` files into `checkpoints_eval/`."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_checkpoints_eval(
    source_dir: Path, destination_dir: Path, clean: bool = False
) -> int:
    """Copy only the folders that contain `.jsonl` files, plus the files themselves."""
    if clean and destination_dir.exists():
        shutil.rmtree(destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    for jsonl_path in source_dir.rglob("*.jsonl"):
        if not jsonl_path.is_file():
            continue

        relative_path = jsonl_path.relative_to(source_dir)
        target_path = destination_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jsonl_path, target_path)
        copied_files += 1

    return copied_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the checkpoints folder structure into checkpoints_eval/ but only for "
            "subtrees that contain .jsonl files."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "checkpoints",
        help="Source checkpoints directory (default: repository checkpoints/).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "checkpoints_eval",
        help="Destination directory (default: repository checkpoints_eval/).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the destination directory before copying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.source}")

    copied_files = copy_checkpoints_eval(
        args.source, args.destination, clean=args.clean
    )
    print(f"Copied {copied_files} .jsonl file(s) into {args.destination}")


if __name__ == "__main__":
    main()
