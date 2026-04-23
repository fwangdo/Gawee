from __future__ import annotations

from ..specs import SUPPORTED_OPS


# Supported ONNX ops in the current frontend contract.
OP_ADD = "Add"
OP_AVERAGE_POOL = "AveragePool"
OP_CAST = "Cast"
OP_CONCAT = "Concat"
OP_CONV = "Conv"
OP_SUB = "Sub"
OP_DIV = "Div"
OP_EQUAL = "Equal"
OP_ERF = "Erf"
OP_EXPAND = "Expand"
OP_GELU = "Gelu"
OP_GLOBAL_AVERAGE_POOL = "GlobalAveragePool"
OP_HARD_SIGMOID = "HardSigmoid"
OP_HARD_SWISH = "HardSwish"
OP_LEAKY_RELU = "LeakyRelu"
OP_MAX = "Max"
OP_MAX_POOL = "MaxPool"
OP_MIN = "Min"
OP_MUL = "Mul"
OP_PAD = "Pad"
OP_RELU = "Relu"
OP_RESHAPE = "Reshape"
OP_REDUCE_MEAN = "ReduceMean"
OP_SHAPE = "Shape"
OP_SIGMOID = "Sigmoid"
OP_SOFTMAX = "Softmax"
OP_SQRT = "Sqrt"
OP_TANH = "Tanh"
OP_SQUEEZE = "Squeeze"
OP_TRANSPOSE = "Transpose"
OP_UNSQUEEZE = "Unsqueeze"
OP_WHERE = "Where"

# Common ONNX ops that are currently unsupported but still appear in models.
OP_BATCH_NORMALIZATION = "BatchNormalization"
OP_CONSTANT = "Constant"
OP_CONSTANT_OF_SHAPE = "ConstantOfShape"
OP_CONV_TRANSPOSE = "ConvTranspose"
OP_CUMSUM = "CumSum"
OP_GATHER = "Gather"
OP_GEMM = "Gemm"
OP_IDENTITY = "Identity"
OP_LAYER_NORMALIZATION = "LayerNormalization"
OP_MATMUL = "MatMul"
OP_NOT = "Not"
OP_POW = "Pow"
OP_REDUCE_SUM = "ReduceSum"
OP_RESIZE = "Resize"
OP_SLICE = "Slice"
OP_SPLIT = "Split"


__all__ = [
    "OP_SUPPORTED_SET",
    "OP_ADD",
    "OP_AVERAGE_POOL",
    "OP_CAST",
    "OP_CONCAT",
    "OP_CONV",
    "OP_DIV",
    "OP_EQUAL",
    "OP_ERF",
    "OP_EXPAND",
    "OP_GELU",
    "OP_GLOBAL_AVERAGE_POOL",
    "OP_HARD_SIGMOID",
    "OP_HARD_SWISH",
    "OP_LEAKY_RELU",
    "OP_MAX",
    "OP_MAX_POOL",
    "OP_MIN",
    "OP_MUL",
    "OP_PAD",
    "OP_REDUCE_MEAN",
    "OP_RELU",
    "OP_RESHAPE",
    "OP_SHAPE",
    "OP_SIGMOID",
    "OP_SOFTMAX",
    "OP_SQRT",
    "OP_SQUEEZE",
    "OP_SUB",
    "OP_TANH",
    "OP_TRANSPOSE",
    "OP_UNSQUEEZE",
    "OP_WHERE",
    "OP_BATCH_NORMALIZATION",
    "OP_CONSTANT",
    "OP_CONSTANT_OF_SHAPE",
    "OP_CONV_TRANSPOSE",
    "OP_CUMSUM",
    "OP_GATHER",
    "OP_GEMM",
    "OP_IDENTITY",
    "OP_LAYER_NORMALIZATION",
    "OP_MATMUL",
    "OP_NOT",
    "OP_POW",
    "OP_REDUCE_SUM",
    "OP_RESIZE",
    "OP_SLICE",
    "OP_SPLIT",
]


OP_SUPPORTED_SET = SUPPORTED_OPS
