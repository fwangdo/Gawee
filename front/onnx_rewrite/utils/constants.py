from __future__ import annotations

from pathlib import Path


# we allow operations as below only
SUPPORTED_OPS: set[str] = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Reshape",
    "Transpose",
    "ReduceMean",
    "Sqrt",
    "Tanh",
    "Softmax",
    "Gather",
    "Concat",
    "Slice",
    "Unsqueeze",
    "Squeeze",
    "Cast",
}

PRIORITY_MODELS: dict[str, Path] = {
    "distilbert_base_uncased": Path("benchmarks/onnx/nlp/distilbert_base_uncased/onnx/model.onnx"),
    "mobilebert_uncased": Path("benchmarks/onnx/nlp/mobilebert_uncased/onnx/model.onnx"),
    "mobilenetv3_small": Path("benchmarks/onnx/vision/mobilenetv3_small.onnx"),
}

