from __future__ import annotations

from pathlib import Path


SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Sub",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "HardSigmoid",
        "HardSwish",
        "IsNaN",
        "LeakyRelu",
        "LessOrEqual",
        "MatMul",
        "Max",
        "MaxPool",
        "Min",
        "Mod",
        "Mul",
        "Neg",
        "Pad",
        "Pow",
        "Range",
        "Relu",
        "ReduceMax",
        "Reshape",
        "ReduceMean",
        "ReduceSum",
        "Resize",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Split",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Squeeze",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)


PRIORITY_MODELS: dict[str, Path] = {
    "resnet18": Path("benchmarks/onnx/vision/resnet18.onnx"),
    "distilbert_base_uncased": Path("benchmarks/onnx/nlp/distilbert_base_uncased/onnx/model.onnx"),
    "bert_tiny": Path("benchmarks/onnx/nlp/bert_tiny/onnx/model.onnx"),
}

EXTENDED_BENCHMARK_MODELS: dict[str, Path] = {
    # Small RoPE-based decoder-only LLM with an explicit ONNX opset>=13 export.
    "qwen3_0_6b": Path("benchmarks/onnx/nlp/qwen3_0_6b/model.onnx"),
    # Latest Ultralytics YOLO family candidate. We export it locally to ONNX opset 17.
    "yolo26_n": Path("benchmarks/onnx/vision/yolo26_n/model.onnx"),
}

ALL_BENCHMARK_MODELS: dict[str, Path] = {
    **PRIORITY_MODELS,
    **EXTENDED_BENCHMARK_MODELS,
}
