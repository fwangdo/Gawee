# gawee

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.ko.md)

A deep learning compiler that takes ONNX models, rewrites and optimizes the graph, and lowers through an MLIR-based pipeline (Gawee Dialect → Linalg → SCF → LLVM IR) to produce AOT-compiled native executables.

---

## Target Models

| Model | Frontend | MLIR Lowering | AOT Build | Correctness |
| --- | --- | --- | --- | --- |
| ResNet-18 | pass | pass | pass | pass (max_abs ≈ 3e-6) |
| bert_tiny | pass | pass | pass | WIP (numerical) |
| distilbert_base_uncased | pass | pass | pass | WIP (numerical) |

NLP models require static shape binding (batch=1, seq_len=128) before translation.

---

## Pipeline

```
ONNX Model → Rewrite/Optimize → MLIR (Gawee Dialect) → Linalg → SCF → LLVM IR → Native Binary
```

### Frontend (Python)

- Load ONNX models and apply graph-level rewrites (op fusion, constant folding)
- Spec-driven rewrite system with per-op correctness validation
- Export rewritten ONNX for downstream translation

### Middle-end (C++ / MLIR)

- Translate ONNX directly to MLIR Gawee Dialect (`gawee-onnx-translate`)
- Gawee Dialect → Linalg Dialect conversion (GaweeToLinalg)
- Decompose aggregated ops (e.g. linalg.softmax) into primitive linalg.generic
- Multi-stage lowering: Linalg → Bufferization → SCF loops → Math/LLVM Dialect
- CLI tools: `gawee-opt`, `gawee-onnx-translate`

### Backend (C++)

- AOT compilation: LLVM IR → native executable via `gawee-aot`
- Automatic launcher code generation from MLIR function signatures
- Correctness validation against ONNX Runtime (`eval_priority_models.py`)

---

## Key Concepts

### 1. Gawee IR Design

A custom IR that explicitly represents only the information needed for graph analysis and optimization.

- Clear separation of operation nodes and data flow
- Explicit shape / dtype / layout / data representation
- Graph: Nodes (ops) / Values (tensors)
- Node: op type / input / output / attributes / fx Node
- Value: shape / dtype / producer / consumers / data (only for constants)

---

### 2. Graph Analysis

Analysis performed for optimization:

- Shape inference
- Constant propagation
- Graph traversal (topological order)
- Cost estimation:
  - FLOPs
  - Memory access estimation (read/write)

---

### 3. Frontend Optimization

Only **graph-level optimizations** are performed in the frontend.

- Constant Folding — evaluate constant subgraphs at compile time
- Operator Fusion — combine consecutive op patterns into fused operators
  - e.g., Conv + BatchNorm, Conv + Add
- Eliminate Python ops from fx
- [Optimization pass details](docs/arch.md)

---

### 4. MLIR Gawee Dialect

Custom MLIR Dialect for DL ops defined using TableGen.

- Ops under `gawee` namespace: Conv2D / ReLU / Add / BatchNorm / MaxPool / AdAvgPool / Flatten / Linear
- Input/output types and attributes (stride, padding, dilation, etc.) declared in TableGen
- C++ boilerplate auto-generated from TableGen

---

### 5. ONNX → MLIR Conversion (ONNXMLIREmitter)

Translates ONNX protobuf directly to MLIR Gawee Dialect ops.

- Parse ONNX graph inputs, outputs, and initializers
- Small integer initializers become `arith.constant`; others become function arguments
- Traverse ONNX nodes in topological order to emit `gawee.*` ops
- Output an MLIR module as `func.func @forward(...)`

---

### 6. Gawee → Linalg Lowering

Convert Gawee Dialect ops to Linalg Dialect using OpConversionPattern.

- `gawee.conv` → `linalg.conv_2d_nchw_fchw` (with padding)
- `gawee.relu` → `linalg.generic` (max(x, 0) body)
- `gawee.add` → `linalg.add`
- `gawee.softmax` → `linalg.softmax` → decomposed to primitive ops
- `gawee.matmul` / `gawee.gather` / `gawee.layer_norm` etc. for NLP models
- Uses ConversionTarget, TypeConverter dialect conversion framework

---

### 7. Multi-stage Lowering

Multi-stage lowering pipeline from Linalg to LLVM IR.

- `--convert-gawee-to-linalg`: Gawee → Linalg
- `--gawee-to-loops`: Gawee → Linalg → Bufferization → SCF loops
- `--gawee-to-llvm`: Gawee → Linalg → Decompose → Bufferize → SCF → Math → LLVM (full pipeline)
- Includes softmax decomposition, tensor → memref bufferization, math-to-libm/llvm

---

## Optimization Results

### ResNet-18
```
Before: 69 nodes → After: 49 nodes
  - ConvBNFolding: 20 applications
  - Memory reads:  37.2MB → 27.3MB (26.7% reduction)
  - Memory writes: 32.9MB → 23.0MB (30.2% reduction)
```

### UNet
```
Before: 196 nodes → After: 116 nodes
  - IdentityElimination: 12, ConvBNFolding: 46, PythonOpElimination: 22
  - Memory reads:  136.6MB → 94.9MB (30.5% reduction)
  - Memory writes: 116.0MB → 83.7MB (27.9% reduction)
```

---

## Project Structure

```
gawee/
├── front/onnx_rewrite/        # Frontend (Python) — ONNX graph rewrite
│   ├── specs/                 #   Per-op rewrite specs and catalog
│   ├── passes/                #   Rewrite passes (matmul conversion, etc.)
│   └── utils/                 #   Constants and helpers
├── middle/mlir/               # Middle-end (C++ / MLIR)
│   ├── include/Gawee/         #   TableGen definitions (Dialect, Ops)
│   ├── lib/Gawee/             #   Dialect registration
│   ├── lib/Conversion/        #   Lowering: Gawee→Linalg, decomposition, bufferize
│   ├── lib/Emit/              #   ONNX→MLIR translation (ONNXMLIREmitter)
│   └── tools/                 #   gawee-opt, gawee-onnx-translate
├── back/                      # Backend — AOT compiler and eval harness
│   ├── gawee_aot.cpp          #   AOT builder (MLIR → native binary)
│   ├── runtime_support.h      #   MemRef descriptor and NPY I/O
│   └── eval_priority_models.py#   End-to-end correctness evaluation
├── scripts/                   # Utility scripts
└── docs/                      # Documentation
```

---

## References

- PyTorch fx documentation
- ONNX specification
- TVM architecture documentation
- MLIR documentation (Dialects, TableGen, Conversion)
