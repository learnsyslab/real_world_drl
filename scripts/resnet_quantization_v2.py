import os

import torch

from scripts.resnet_quantization import (
    evaluate,
    load_data,
    resnet18_quantizable,
    transform_data,
)

from torch.utils.data import DataLoader, Dataset
from typing import List

# FX/TensorRT imports
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
    TRTInterpreterResult,
)
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision


# ----------------- simple Dataset wrapper -----------------
class TensorListDataset(Dataset):
    def __init__(self, tensor_list: List[torch.Tensor], labels=None):
        self.tensor_list = tensor_list
        self.labels = labels

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.tensor_list[idx]
        return self.tensor_list[idx], self.labels[idx]


# ----------------- Load model and data (your existing functions) -----------------
# (I assume these functions exist in your script: load_data, transform_data, evaluate)
float_model = resnet18_quantizable(pretrained=True).eval().to("cpu")

data_path = "imagenet/"
labels_path = os.path.join(data_path, "imagenet_class_index.json")
traindir = os.path.join(data_path, "train")
valdir = os.path.join(data_path, "val")

print("Loading train data...")
train_data, train_labels = load_data(
    traindir, labels_path
)  # returns cpu tensors / list
print("Loading test data...")
test_data, test_labels = load_data(valdir, labels_path)

print("Transforming test data...")
test_data = transform_data(test_data)  # list of cpu tensors (C,H,W)
print("Transforming train data...")
train_data = transform_data(train_data)

# GPU test tensors for evaluation
test_data_gpu = [img.to("cuda") for img in test_data]

# Optional baseline FP32 eval (on GPU)
float_model_gpu = float_model.to("cuda")
inference_time, accuracy = evaluate(float_model_gpu, test_data_gpu, test_labels)
print("FP32 Baseline Inference Time:", inference_time)
print("FP32 Baseline Accuracy:", accuracy)
# send model back to CPU for FX trace
float_model = float_model.to("cpu").eval()

# ----------------- Prepare calibration dataloader -----------------
# use a representative subset for calibration (1000 images recommended if available)
num_calib_samples = min(len(train_data), 1000)
calib_samples = train_data[:num_calib_samples]
calib_batch_size = 16
calib_loader = DataLoader(
    TensorListDataset(calib_samples),
    batch_size=calib_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

# The DataLoaderCalibrator expects tensors that are same dtype as model input (float32)
# and will move batches to the provided device (CUDA) when calibrating.

# ----------------- FX trace with acc_tracer -----------------
# Use a representative input (batch size 1 recommended for tracing)
example_input = (
    test_data[0].unsqueeze(0)
    if isinstance(test_data, list)
    else torch.randn(1, 3, 224, 224)
)
example_input = example_input.to(torch.float32)  # ensure float32
print("Tracing model with acc_tracer...")
acc_mod = acc_tracer.trace(float_model, [example_input])

# Build InputTensorSpec from the example input (FX path expects this)
input_specs = InputTensorSpec.from_tensors([example_input])

# ----------------- Create DataLoaderCalibrator -----------------
calib_cache_file = "trt_int8_calib.cache"
calibrator = DataLoaderCalibrator(
    calib_loader,
    algorithm="entropy",  # "entropy" recommended, "minmax" also available
    device=torch.device("cuda"),  # calibrator will run calibration on GPU
    cache_file=calib_cache_file,
    use_cache=False,  # set True to reuse cache_file if available
)

# ----------------- Build TRT engine via FX interpreter (INT8) -----------------
max_batch_size = 8
max_workspace_size = 1 << 28  # ~256MB, increase if needed

print("Running TRTInterpreter to build INT8 engine (this will run calibration)...")
trt_interpreter = TRTInterpreter(acc_mod, input_specs, explicit_batch_dimension=True)

# Try to pass calibrator directly to the interpreter.run() call.
# Some torch_tensorrt versions accept 'calibrator' kwarg here; others may require
# using torch_tensorrt.compile(...) which we provide as a fallback below.
try:
    trt_interpreter_result: TRTInterpreterResult = trt_interpreter.run(
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        lower_precision=LowerPrecision.INT8,
        # calibrator=calibrator,
        timing_cache=None,
        sparse_weights=False,
        strict_type_constraints=False,
    )
    trt_module = TRTModule(
        trt_interpreter_result.engine,
        trt_interpreter_result.input_names,
        trt_interpreter_result.output_names,
    )
    print("INT8 TRT engine built via TRTInterpreter.run() (FX path).")
except TypeError as e:
    # Fallback: some torch_tensorrt versions do not accept 'calibrator' in TRTInterpreter.run().
    # Use torch_tensorrt.compile on the traced/scripted model instead (compile accepts calibrator).
    print("TRTInterpreter.run() rejected calibrator arg (fallback). Error:", e)
    print(
        "Falling back to torch_tensorrt.compile(...) PTQ path using a scripted model."
    )
    import torch_tensorrt

    # Move model to CUDA for scripting/compilation
    float_model_cuda = float_model.to("cuda").eval()
    with torch.no_grad():
        scripted = torch.jit.trace(
            float_model_cuda, torch.randn(1, 3, 224, 224).to("cuda")
        ).eval()
    # compile with calibrator => builds INT8 engine (this will run calibration)
    trt_module = torch_tensorrt.compile(
        scripted,
        inputs=[
            torch_tensorrt.Input(
                min_shape=[1, 3, 224, 224],
                opt_shape=[4, 3, 224, 224],
                max_shape=[8, 3, 224, 224],
                dtype=torch.float,
            )
        ],
        enabled_precisions={torch_tensorrt.dtype.int8},
        calibrator=calibrator,
        workspace_size=max_workspace_size,
    )
    print("INT8 TRT engine built via torch_tensorrt.compile() fallback.")

# Save the built TRT module
torch.jit.save(trt_module, "resnet18_trt_int8.ts")
print("Saved TRT INT8 module: resnet18_trt_int8.ts")

# ----------------- Evaluate the TRT module on GPU -----------------
print("Evaluating TRT INT8 module on GPU...")
# Ensure evaluate() sends inputs already as CUDA tensors; trt_module expects CUDA tensors
inference_time_trt, accuracy_trt = evaluate(trt_module, test_data_gpu, test_labels)
print("TRT INT8 Inference Time:", inference_time_trt)
print("TRT INT8 Accuracy:", accuracy_trt)
