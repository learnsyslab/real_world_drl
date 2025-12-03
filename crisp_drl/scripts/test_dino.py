import time
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Normalization module (ImageNet mean/std)
# ---------------------------------------------------------------------------
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # Expect input as (N, H, W, C)
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------
def benchmark_dinov2(
    model_name="dinov2_vits14_reg",
    batch_size=2,
    image_size=(224, 224),
    n_warmup=10,
    n_iters=50,
    use_jit=True,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")

    # -----------------------------------------------------------------------
    # Load model from torch.hub
    # -----------------------------------------------------------------------
    backbone = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
    backbone.eval()

    # DINOv2 expects images normalized between 0–1 (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model = nn.Sequential(Normalize(mean, std), backbone).to(device)

    # -----------------------------------------------------------------------
    # Dummy input (NHWC because we wrap a Normalize layer that permutes it)
    # -----------------------------------------------------------------------
    H, W = image_size
    dummy = torch.zeros(batch_size, H, W, 3, dtype=torch.float32).to(device)

    # -----------------------------------------------------------------------
    # JIT trace for speed
    # -----------------------------------------------------------------------
    if use_jit:
        print("Tracing with TorchScript...")
        model = torch.jit.trace(model, dummy)
        model.eval()

    # Warmup ---------------------------------------------------------------
    print("Running warmup...")
    for _ in range(n_warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs -----------------------------------------------------------
    print("Benchmarking...")
    times = []
    for _ in range(n_iters):
        start = time.time()
        _ = model(dummy)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    print("\n---------------- DINOv2 Benchmark ----------------")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Mean latency:  {np.mean(times):.3f} ms")
    print(f"Median latency:{np.median(times):.3f} ms")
    print(f"Min latency:   {np.min(times):.3f} ms")
    print(f"Max latency:   {np.max(times):.3f} ms")
    print("--------------------------------------------------")


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark_dinov2(
        model_name="dinov2_vits14_reg",  # or "dinov2_vits14"
        batch_size=2,
        image_size=(224, 224),
        use_jit=True,
    )
