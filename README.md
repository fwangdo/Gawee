# gawee

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.ko.md)

A deep learning compiler project that converts PyTorch models into a custom IR, performs graph analysis and optimization, and lowers through an MLIR-based middle-end pipeline (Linalg → SCF → LLVM IR). Designed to tackle real-world problems found in production DL compilers (e.g., TVM).

---

## Target Models

- **ResNet-18**
  - Based on the standard ImageNet ResNet-18 architecture.
  - Goal: reduce graph nodes by fusing Conv / BatchNorm.
- **UNet**
  - Based on the standard ImageNet UNet architecture.
  - Goal: eliminate redundant Identity ops and removable Python ops (e.g., getitem, getattr).
- [Usage & Evaluation](docs/usage.md)

---

## Pipeline

```
PyTorch Model → FX Graph → Gawee IR → JSON → MLIR(Gawee Dialect) → Linalg → SCF → LLVM
```

### Frontend (Python)

- Parse torch fx graph into Gawee IR
- Measure baseline cost using predefined costs
- Optimize using passes defined in Gawee IR
- Export IR as JSON

### Middle-end (C++ / MLIR)

- Convert JSON graph to MLIR Gawee Dialect (MLIREmitter)
- Gawee Dialect → Linalg Dialect conversion (GaweeToLinalg)
- Multi-stage lowering: Linalg → Bufferization → SCF loops → LLVM Dialect
- Two CLI tools: `gawee-opt`, `gawee-translate`

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

### 5. JSON → MLIR Conversion (MLIREmitter)

Converts the frontend JSON graph to MLIR Gawee Dialect ops.

- Parse inputs, outputs, weights, and nodes from JSON
- Register weight tensors as function arguments
- Traverse nodes in topological order to emit `gawee.*` ops
- Output an MLIR module as `func.func @main(...)`

---

### 6. Gawee → Linalg Lowering

Convert Gawee Dialect ops to Linalg Dialect using OpConversionPattern.

- `gawee.conv` → `linalg.conv_2d_nchw_fchw` (with padding)
- `gawee.relu` → `linalg.generic` (max(x, 0) body)
- `gawee.add` → `linalg.add`
- Lowering patterns for BatchNorm / MaxPool / AdAvgPool / Flatten / Linear
- Uses ConversionTarget, TypeConverter dialect conversion framework

---

### 7. Multi-stage Lowering

Multi-stage lowering pipeline from Linalg to LLVM IR.

- `--convert-gawee-to-linalg`: Gawee → Linalg
- `--gawee-to-loops`: Gawee → Linalg → Bufferization → SCF loops
- `--gawee-to-llvm`: Gawee → Linalg → Bufferize → SCF → LLVM (full pipeline)
- Includes tensor → memref conversion (bufferization)

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
├── gawee_ir/                  # Frontend (Python)
│   ├── graph.py               #   Gawee IR definition (Graph / Node / Value)
│   ├── parser.py              #   PyTorch FX → Gawee IR conversion
│   ├── mapper.py              #   PyTorch op → Gawee op mapping
│   ├── translator.py          #   Gawee IR → JSON conversion
│   ├── analysis/              #   Shape inference, Cost analysis
│   └── passes/                #   Optimization passes (Conv-BN folding, etc.)
├── middle/mlir/               # Middle-end (C++ / MLIR)
│   ├── include/Gawee/         #   TableGen definitions (Dialect, Ops)
│   ├── lib/Gawee/             #   Dialect registration
│   ├── lib/Conversion/        #   Gawee → Linalg conversion patterns
│   ├── lib/Emit/              #   JSON → MLIR conversion (MLIREmitter)
│   └── tools/                 #   gawee-opt, gawee-translate
├── scripts/                   # Scripts
├── jsondata/                  # Frontend output JSON
└── docs/                      # Documentation
```

---

## References

- PyTorch fx documentation
- ONNX specification
- TVM architecture documentation
- MLIR documentation (Dialects, TableGen, Conversion)
