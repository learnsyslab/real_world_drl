"""Create a dataset of DINOv2 features from TIFF image stacks.

This script finds TIFF stacks matching a glob, loads them, resizes frames,
passes batches through a DINOv2 backbone, and saves per-file feature arrays.

Usage example:
  python crisp_drl/scripts/vae_dataset.py \
      --input-glob "/home/linusschwarz/workspaces/isaac_ros-dev/own_samples/*/camera_cropped.tif" \
      --output-dir /tmp/dinov2_dataset \
      --model dinov2_vits14_reg \
      --batch-size 8 \
      --image-size 224

Dependencies:
  - torch
  - torchvision
  - tifffile
  - numpy
  - pillow
  - tqdm (optional)

The script saves compressed NumPy `.npz` files with key `features`.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import tifffile
except Exception as e:  # pragma: no cover - helpful runtime message
    raise RuntimeError("Please install tifffile (pip install tifffile)") from e

try:
    from torchvision import transforms as T
except Exception:
    raise RuntimeError("Please install torchvision (pip install torchvision)")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional
    tqdm = lambda x, **kwargs: x


# ---------------------------------------------------------------------------
# Normalization module (ImageNet mean/std) — same as used in test_dino.py
# ---------------------------------------------------------------------------
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input as (N, H, W, C) in range [0, 1]
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return (x - self.mean) / self.std


def find_files(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def load_tiff_stack(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    # Expected shapes: (F, H, W) or (F, H, W, C)
    if arr.ndim == 3:
        # Add channel dimension
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        # replicate grayscale to 3 channels
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        # Drop alpha if present
        arr = arr[..., :3]
    return arr


def preprocess_frames(frames: np.ndarray, image_size: int) -> np.ndarray:
    # frames: (F, H, W, C) uint8 or float in [0,1]
    F = frames.shape[0]
    transform = T.Compose(
        [T.Resize((image_size, image_size), interpolation=Image.BILINEAR)]
    )

    out = []
    for i in range(F):
        frame = frames[i]
        # normalize dtype to uint8 for Pillow
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            # assume in [0,1]
            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame_u8 = frame.astype(np.uint8)
        img = Image.fromarray(frame_u8)
        img = transform(img)
        arr = np.array(img, copy=False)
        arr = arr.astype(np.float32) / 255.0
        out.append(arr)
    return np.stack(out, axis=0)


def batch_iterator(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


def infer_on_stack(
    model: nn.Module,
    frames: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    # frames: (F, H, W, C) float32 in [0,1]
    model.to(device)
    model.eval()
    feats_list = []
    F = frames.shape[0]
    with torch.no_grad():
        for a, b in batch_iterator(F, batch_size):
            batch = frames[a:b]
            t = torch.from_numpy(batch).to(device=device)
            out = model(t)
            # handle different output formats
            if isinstance(out, torch.Tensor):
                feats = out.detach().cpu().numpy()
            elif isinstance(out, (list, tuple)):
                feats = out[0].detach().cpu().numpy()
            elif isinstance(out, dict):
                # try to find a useful key
                for key in ("feat", "features", "x", "out"):
                    if key in out:
                        val = out[key]
                        feats = (
                            val.detach().cpu().numpy()
                            if isinstance(val, torch.Tensor)
                            else np.asarray(val)
                        )
                        break
                else:
                    # fallback: take first value
                    val = next(iter(out.values()))
                    feats = (
                        val.detach().cpu().numpy()
                        if isinstance(val, torch.Tensor)
                        else np.asarray(val)
                    )
            else:
                feats = np.asarray(out)
            feats_list.append(feats)
    return np.concatenate(feats_list, axis=0)


def build_model(
    model_name: str, device: str, use_jit: bool, image_size: int
) -> nn.Module:
    print(f"Loading model {model_name} on {device}...")
    backbone = torch.hub.load(
        "facebookresearch/dinov2", model_name, trust_repo=True
    ).to(device)
    backbone.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model = nn.Sequential(Normalize(mean, std), backbone).to(device)

    if use_jit:
        # try to trace with a dummy
        try:
            dummy = torch.zeros(1, image_size, image_size, 3, dtype=torch.float32).to(
                device
            )
            model = torch.jit.trace(model, dummy)
            model.eval()
            print("Traced model with TorchScript")
        except Exception as e:
            print("TorchScript trace failed — using eager model. Error:", e)
    return model


def process_files(
    input_glob: str,
    output_dir: str,
    model_name: str,
    batch_size: int,
    image_size: int,
    device: str,
    use_jit: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    files = find_files(input_glob)
    if not files:
        raise SystemExit(f"No files found for pattern: {input_glob}")

    model = build_model(model_name, device, use_jit, image_size)

    all_feats = []
    for path in tqdm(files, desc="files"):
        print(f"Processing {path}")
        frames = load_tiff_stack(path)  # (F, H, W, C)
        frames = preprocess_frames(frames, image_size)
        feats = infer_on_stack(model, frames, batch_size, device)
        all_feats.append(feats)

    out_path = os.path.join(output_dir, f"feat_{model_name}.npz")
    np.savez_compressed(out_path, features=np.concatenate(all_feats, axis=0))
    print(
        f"Saved features to {out_path}, features.shape={np.concatenate(all_feats, axis=0).shape}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DINOv2 dataset from TIFF stacks")
    p.add_argument(
        "--input-glob", type=str, required=True, help="glob for input TIFF stacks"
    )
    p.add_argument(
        "--output-dir", type=str, required=True, help="where to save feature .npz files"
    )
    p.add_argument(
        "--model", type=str, default="dinov2_vits14_reg", help="torch.hub model name"
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    p.add_argument(
        "--use-jit", action="store_true", help="trace model with TorchScript for speed"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_files(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        use_jit=args.use_jit,
    )


if __name__ == "__main__":
    main()

"""
python crisp_drl/scripts/vae_dataset.py \
  --input-glob "/home/linusschwarz/workspaces/isaac_ros-dev/own_samples/*/camera_cropped.tif" \
  --output-dir datasets/dinov2_dataset \
  --model dinov2_vits14_reg \
  --batch-size 8 \
  --image-size 224

  """
