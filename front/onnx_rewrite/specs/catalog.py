from __future__ import annotations

from pathlib import Path


SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "Conv",
        "Sub",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gelu",
        "GlobalAveragePool",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Max",
        "MaxPool",
        "Min",
        "Mul",
        "Pad",
        "Relu",
        "Reshape",
        "ReduceMean",
        "ReduceSum",
        "Shape",
        "Sigmoid",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Squeeze",
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
