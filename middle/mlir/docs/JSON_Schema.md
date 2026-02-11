# Gawee Graph JSON Schema

This document describes the JSON format used by `gawee-translate` to emit MLIR.

## Top-Level Structure

```json
{
  "inputs":  ["x"],                // Graph input tensor names
  "outputs": ["fc"],               // Graph output tensor names
  "values":  { ... },              // Shape/dtype metadata for every tensor
  "nodes":   [ ... ]               // Operations in topological order
}
```

## Values

Every tensor (inputs, outputs, intermediates) has an entry in `values`:

```json
"x": {
  "id": "x",
  "shape": [1, 3, 224, 224],      // NCHW format
  "dtype": "torch.float32"
}
```

The emitter uses `shape` to determine the MLIR tensor type.

## Nodes

Each node represents one operation. All nodes share this structure:

```json
{
  "op_type": "Conv",               // Gawee op type (used for dispatch)
  "name": "conv1",                 // Module path in original PyTorch model
  "inputs": ["x"],                 // Input tensor names (references into values)
  "outputs": ["conv1"],            // Output tensor names (references into values)
  "attrs": { ... }                 // Op-specific attributes
}
```

`inputs` and `outputs` are lists of tensor names that must match keys in `values`.

---

## Op Types

### Conv

2D convolution. Inputs: 1 tensor. Has weight and bias parameters.

```json
{
  "op_type": "Conv",
  "inputs": ["x"],
  "outputs": ["conv1"],
  "attrs": {
    "stride": [2, 2],
    "padding": [3, 3],
    "dilation": [1, 1],
    "groups": 1,
    "weight": {
      "shape": [64, 3, 7, 7],       // [out_channels, in_channels, kH, kW]
      "dtype": "float32",
      "path": "weights/conv1_weight_0.bin"
    },
    "bias": {
      "shape": [64],                 // [out_channels]
      "dtype": "float32",
      "path": "weights/conv1_bias_1.bin"
    }
  }
}
```

### Relu

ReLU activation. Inputs: 1 tensor. No attributes needed.

```json
{
  "op_type": "Relu",
  "inputs": ["conv1"],
  "outputs": ["relu"],
  "attrs": {
    "inplace": true                  // ignored in compilation
  }
}
```

### Add

Elementwise addition (residual connection). Inputs: 2 tensors. No attributes.

```json
{
  "op_type": "Add",
  "inputs": ["layer1_0_conv2", "maxpool"],
  "outputs": ["add"],
  "attrs": {}
}
```

### MaxPool

Max pooling. Inputs: 1 tensor.

```json
{
  "op_type": "MaxPool",
  "inputs": ["relu"],
  "outputs": ["maxpool"],
  "attrs": {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "dilation": 1,
    "ceil_mode": false
  }
}
```

Note: `kernel_size`, `stride`, `padding`, `dilation` can be either a single integer
or a list of 2 integers (e.g., `[3, 3]`). A single integer means the same value
for both H and W.

### AdAvgPool (Adaptive Average Pooling)

Adaptive average pooling. Inputs: 1 tensor.

```json
{
  "op_type": "AdAvgPool",
  "inputs": ["layer4_1_relu_1"],
  "outputs": ["avgpool"],
  "attrs": {
    "output_size": [1, 1]            // target spatial dimensions
  }
}
```

In ResNet-18, `output_size` is always `[1, 1]` (global average pooling).

### flatten

Flatten dimensions. Inputs: 1 tensor.

```json
{
  "op_type": "flatten",
  "inputs": ["avgpool"],
  "outputs": ["flatten"],
  "attrs": {
    "start_dim": 1,
    "end_dim": -1
  }
}
```

- `start_dim`: First dimension to flatten (inclusive).
- `end_dim`: Last dimension to flatten (inclusive). `-1` means last dim.
- Example: `[1, 512, 1, 1]` with `start_dim=1, end_dim=-1` becomes `[1, 512]`.

### MatMul (Linear)

Fully-connected / linear layer. Inputs: 1 tensor. Has weight and bias parameters.

```json
{
  "op_type": "MatMul",
  "inputs": ["flatten"],
  "outputs": ["fc"],
  "attrs": {
    "in_features": 512,
    "out_features": 1000,
    "weight": {
      "shape": [1000, 512],          // [out_features, in_features]
      "dtype": "float32",
      "path": "weights/fc_weight_40.bin"
    },
    "bias": {
      "shape": [1000],               // [out_features]
      "dtype": "float32",
      "path": "weights/fc_bias_41.bin"
    }
  }
}
```

Note: Weight is stored as `[out_features, in_features]` (PyTorch convention),
so the lowering uses `MatmulTransposeBOp` to compute `input @ weight^T`.

---

## Weight Files

Weight tensors are stored as raw binary files (`.bin`) under `weights/` directory.
Each `weight` or `bias` object has:

- `shape`: Tensor dimensions
- `dtype`: Data type (always `float32` in ResNet-18)
- `path`: Relative path to the binary file

## ResNet-18 Op Count

| Op Type | Count |
|---------|-------|
| Conv | 20 |
| Relu | 16 |
| Add | 8 |
| MaxPool | 1 |
| AdAvgPool | 1 |
| flatten | 1 |
| MatMul | 1 |
| **Total** | **48** |
