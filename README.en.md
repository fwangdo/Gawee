# gawee

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.md)

`gawee` is a deep learning compiler project that takes ONNX models, rewrites
their graphs, lowers them through an MLIR pipeline
(`Gawee Dialect -> Linalg -> SCF/LLVM`), and builds AOT native executables.

The current priority is to reduce heavy frontend rewrites and instead make the
middle-end accept the real operation set needed by `resnet / bert_tiny / tinyllama`.

---

## Current Status

| Model | ONNX Emission | Gawee -> Linalg | Full LLVM/AOT | Notes |
| --- | --- | --- | --- | --- |
| ResNet-18 | pass | pass | pass | End-to-end AOT and numerical path available |
| bert_tiny | pass | pass | pass | Correctness passes after preserving semantic `Gather` |
| tinyllama_15m | pending | pending | pending | Tiny decoder LLM candidate with `RoPE` |

### What was added in this stage

- Split `MatMul` from `Gemm/Linear`
- Add explicit semantic ops to the `gawee` dialect:
  - `gawee.gather`
  - `gawee.gather_elements`
  - `gawee.range`
  - `gawee.resize`
  - `gawee.split`
  - `gawee.tile`
- Keep trivial ops as direct emitter decompositions:
  - `Pow`, `Neg`, `Sin`, `Cos`, `And`, `LessOrEqual`, `IsNaN`, `Mod`
  - `Constant`, `ConstantOfShape`

The point is to stop hiding unsupported benchmark ops behind aggressive
frontend rewrites and let the middle-end handle them explicitly.

---

## Pipeline

```text
ONNX Model
  -> Rewrite / Optimize (Python)
  -> MLIR Gawee Dialect
  -> Linalg
  -> Bufferization / SCF / Math / LLVM
  -> Native Binary
```

### Frontend (Python)

- ONNX graph rewrites and normalization
- constant folding and spec-driven rewrites
- unsupported-op audits per benchmark model

### Middle-end (C++ / MLIR)

- `gawee-onnx-translate`
  - reads ONNX protobuf directly
  - emits either `gawee.*` ops or direct `tensor/linalg/math` ops
- `gawee-opt`
  - `--convert-gawee-to-linalg`
  - `--gawee-to-loops`
  - `--gawee-to-llvm`

### Backend (C++)

- AOT executable generation
- ONNX Runtime based output comparison
- static-shape-bound evaluation path for NLP models

---

## Support Strategy

Not every op needs a new dialect op.

### 1. Keep semantic ops in `gawee`

These ops carry meaningful shape/axis/lookup semantics and are useful fallback
boundaries:

- `MatMul`
- `Gather`
- `GatherElements`
- `Range`
- `Resize`
- `Split`
- `Tile`

Implementation order:

1. define in `GaweeOps.td`
2. emit `gawee.*` in the ONNX emitter
3. lower in `GaweeToLinalg.cpp`

### 2. Directly decompose trivial ops

These are emitted straight to `tensor.generate`, `linalg.generic`, `arith`, or
`math` without adding new dialect ops:

- `Pow`
- `Neg`
- `Sin`
- `Cos`
- `And`
- `LessOrEqual`
- `IsNaN`
- `Mod`
- `Constant`
- `ConstantOfShape`

---

## Verification Notes

After the semantic-op expansion:

- `resnet18`
  - ONNX emission passes
  - `gawee-to-llvm` passes
- `bert_tiny`
  - ONNX emission passes
  - `gawee-to-llvm` passes
  - correctness passes
- `qwen3_0_6b`
  - ONNX emission passes
  - `convert-gawee-to-linalg` passes
  - fixed dynamic-length legalization for `gawee.range`

`distilbert_base_uncased` is removed from the default benchmark/eval set because
its CPU cost is too high for routine regression runs. It is being replaced by
`tinyllama_15m` as the small modern decoder candidate.

For `qwen`, the full LLVM pipeline is still primarily a scale/runtime issue.
The main milestone for this phase is that semantic ops are now legalized at the
MLIR stage instead of being rejected as unsupported.

---

## Learning Notes

- [MatMul Lowering Summary](middle/mlir/docs/kr/MatMulLowering_Summary.md)
- [MatMul Lowering Quiz](middle/mlir/docs/kr/MatMulLowering_Quiz.cpp)
- [Semantic Op Lowering Summary](middle/mlir/docs/kr/SemanticOpLowering_Summary.md)
- [Semantic Op Lowering Quiz](middle/mlir/docs/kr/SemanticOpLowering_Quiz.cpp)
