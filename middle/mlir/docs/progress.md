# Gawee MLIR Compiler - Progress Tracker

## Overview

Building an AI compiler from scratch using MLIR infrastructure.

```
Goal: Neural Network Model → Optimized Executable

Pipeline:
  Frontend → Gawee Dialect → Linalg → SCF/Affine → LLVM → Binary
```

---

## Phase 1: Infrastructure Setup ✅

| Task | Status | Files |
|------|--------|-------|
| LLVM/MLIR installation | ✅ Done | ~/llvm-install/ |
| CMake build system | ✅ Done | CMakeLists.txt |
| Build script | ✅ Done | build.sh |
| TableGen workaround (string literal bug) | ✅ Done | build.sh (sed fix) |
| compile_commands.json for IDE | ✅ Done | build.sh, .clangd |

**Key learnings:**
- CMake finds MLIR via `find_package(MLIR REQUIRED CONFIG)`
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` generates compile_commands.json
- TableGen has a bug in some versions - need sed workaround

---

## Phase 2: Gawee Dialect Definition ✅

| Task | Status | Files |
|------|--------|-------|
| Dialect declaration (.td) | ✅ Done | include/Gawee/GaweeDialect.td |
| Op definitions (.td) | ✅ Done | include/Gawee/GaweeOps.td |
| C++ dialect implementation | ✅ Done | lib/Gawee/GaweeDialect.cpp |
| Generated headers | ✅ Done | include/Gawee/generated/*.inc |

**Ops defined:**
- `gawee.conv` - 2D convolution (input, weight, strides, padding, dilation)
- `gawee.relu` - ReLU activation
- `gawee.add` - Elementwise addition

**Key learnings:**
- TableGen generates C++ from .td files
- `DenseI64ArrayAttr:$name` generates `getName()` and `getNameAttr()`
- Ops need arguments (ins) and results (outs)

---

## Phase 3: Gawee → Linalg Conversion ✅

| Task | Status | Files |
|------|--------|-------|
| ConvOpLowering | ✅ Done | lib/Conversion/GaweeToLinalg.cpp |
| ReluOpLowering | ✅ Done | lib/Conversion/GaweeToLinalg.cpp |
| AddOpLowering | ✅ Done | lib/Conversion/GaweeToLinalg.cpp |
| Pass definition | ✅ Done | lib/Conversion/GaweeToLinalg.cpp |
| Summary document | ✅ Done | docs/GaweeToLinalg_Summary.md |
| Quiz file | ✅ Done | docs/GaweeToLinalg_Quiz.cpp |

**Conversion mappings:**
```
gawee.conv  → linalg.conv_2d_nchw_fchw
gawee.relu  → linalg.generic (max(0, x))
gawee.add   → linalg.add
```

**Key learnings:**
- `OpConversionPattern<T>` rewrites op T to other ops
- `adaptor` = converted operands, `op` = original op with attributes
- Linalg uses destination-passing style (create empty output first)
- `linalg.generic` = Swiss army knife for custom elementwise ops
- `ConversionTarget`: legal = can remain, illegal = must convert

---

## Phase 4: gawee-opt Tool ✅

| Task | Status | Files |
|------|--------|-------|
| Create gawee-opt executable | ✅ Done | tools/gawee-opt.cpp |
| Register dialects | ✅ Done | tools/gawee-opt.cpp |
| Register passes | ✅ Done | tools/gawee-opt.cpp |
| Update CMakeLists.txt | ✅ Done | CMakeLists.txt |
| Add RTTI fix | ✅ Done | CMakeLists.txt |
| Add getDependentDialects | ✅ Done | lib/Conversion/GaweeToLinalg.cpp |
| Test with sample IR | ✅ Done | test/simple_test.mlir |
| Summary document | ✅ Done | docs/gawee-opt_Summary.md |
| Quiz file | ✅ Done | docs/gawee-opt_Quiz.cpp |

**Result:** Tool works correctly:
```bash
./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir
```

**Key learnings:**
- Dialects must be registered in `DialectRegistry`
- Passes must declare `getDependentDialects()` for dialects they create ops from
- RTTI must be disabled (`-fno-rtti`) to match LLVM's build
- `MlirOptMain` handles CLI, parsing, pass execution

---

## Phase 5: Linalg → Loops ✅

| Task | Status | Files |
|------|--------|-------|
| Add SCF dialect to gawee-opt | ✅ Done | tools/gawee-opt.cpp |
| Add bufferization support | ✅ Done | tools/gawee-opt.cpp |
| Full pipeline script | ✅ Done | scripts/full_pipeline.sh |
| Test all ops | ✅ Done | test/simple_test.mlir |

**Pipeline:**
```
Gawee → Linalg(tensor) → Bufferize → Linalg(memref) → SCF loops
```

**Key learnings:**
- Linalg-to-loops works on memref, not tensor - need bufferization first
- Bufferization converts tensor → memref (memory allocation)
- MLIR provides built-in passes: `--one-shot-bufferize`, `--convert-linalg-to-loops`
- Complex ops like conv2d become 7 nested loops

---

## Phase 6: C++ Graph → Gawee MLIR (Frontend Connection) ✅

| Task | Status | Files |
|------|--------|-------|
| Create MLIREmitter class | ✅ Done | include/Emit/MLIREmitter.h, lib/Emit/MLIREmitter.cpp |
| Emit gawee.conv from Graph::Node | ✅ Done | lib/Emit/MLIREmitter.cpp |
| Emit gawee.relu from Graph::Node | ✅ Done | lib/Emit/MLIREmitter.cpp |
| Emit gawee.add from Graph::Node | ✅ Done | lib/Emit/MLIREmitter.cpp |
| Create gawee-translate tool | ✅ Done | tools/gawee-translate.cpp |
| Test with subset of graph.json | ✅ Done | test/subset_graph.json |
| Summary document | ✅ Done | docs/MLIREmitter_Summary.md |
| Quiz file | ✅ Done | docs/MLIREmitter_Quiz.cpp |
| Build and test | ✅ Done | - |

**Result:** Full pipeline works:
```bash
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --convert-gawee-to-linalg
```

**Key learnings:**
- `mlir-translate` converts external formats ↔ MLIR
- `OpBuilder` creates ops at current insertion point
- Value mapping connects JSON string names to `mlir::Value`
- LLVM JSON API uses Optional returns - always check validity
- Topological order ensures inputs are defined before use
- Weights must be function arguments (constant tensors can't bufferize)

---

## Phase 7: SCF → LLVM → Binary ✅

| Task | Status | Files |
|------|--------|-------|
| SCF to CF conversion | ✅ Done | tools/gawee-opt.cpp |
| Arith to LLVM conversion | ✅ Done | tools/gawee-opt.cpp |
| MemRef to LLVM conversion | ✅ Done | tools/gawee-opt.cpp |
| CF to LLVM conversion | ✅ Done | tools/gawee-opt.cpp |
| Func to LLVM conversion | ✅ Done | tools/gawee-opt.cpp |
| LLVM dialect → LLVM IR script | ✅ Done | scripts/to_llvm_ir.sh |
| Test file | ✅ Done | test/llvm_test.mlir |
| Summary document | ✅ Done | docs/LLVMLowering_Summary.md |
| Quiz file | ✅ Done | docs/LLVMLowering_Quiz.cpp |

**Result:** Full pipeline to LLVM IR works:
```bash
./build/gawee-opt --scf-to-llvm test/llvm_test.mlir
./scripts/to_llvm_ir.sh test/llvm_test.mlir output.ll
```

**Key learnings:**
- Lowering is hierarchical: High-level → Mid-level → Low-level → Target
- SCF (for/while) → CF (branches) via `scf-to-cf` pass
- MemRef → LLVM struct with pointer, offset, sizes, strides
- Multiple conversion passes needed, order matters
- `reconcile-unrealized-casts` cleans up temporary markers
- `mlir-translate --mlir-to-llvmir` converts LLVM dialect to LLVM IR
- **Bufferization interfaces must be registered** for each dialect (arith, linalg, tensor, func)

---

## Phase 8: Extend for ResNet (User's Own Work)

**graph.json analysis (ResNet-18, 48 nodes):**

| Op in graph.json | Count | In GaweeOps.td? | Lowering? |
|-----------------|-------|-----------------|-----------|
| `Conv` | 20 | ✅ Yes | ✅ Done (+ bias broadcast) |
| `Relu` | 16 | ✅ Yes | ✅ Done |
| `Add` | 8 | ✅ Yes | ✅ Done |
| `MaxPool` | 1 | ✅ Yes | ✅ Done |
| `AdAvgPool` (AdaptiveAvgPool2d) | 1 | ✅ Yes | ✅ Done (sum pool + divf) |
| `flatten` | 1 | ✅ Yes | ✅ Done (collapse_shape) |
| `MatMul` (Linear) | 1 | ✅ Yes | ✅ Done (matmul_transpose_b + bias) |

**Note:** BatchNorm does not appear in graph.json because conv-bn fusion
was already applied at the frontend. Conv nodes carry fused weight+bias.

| Task | Status | Files |
|------|--------|-------|
| Add MaxPool op to dialect | ✅ Done | GaweeOps.td |
| Add AdaptiveAvgPool op to dialect | ✅ Done | GaweeOps.td |
| Add Flatten op to dialect | ✅ Done | GaweeOps.td |
| Add Linear op to dialect | ✅ Done | GaweeOps.td |
| Add bias to Conv lowering | ✅ Done | GaweeToLinalg.cpp |
| Implement MaxPool lowering | ✅ Done | GaweeToLinalg.cpp |
| Implement AdAvgPool lowering | ✅ Done | GaweeToLinalg.cpp |
| Implement Flatten lowering | ✅ Done | GaweeToLinalg.cpp |
| Implement Linear lowering | ✅ Done | GaweeToLinalg.cpp |
| Register all new patterns in pass | ✅ Done | GaweeToLinalg.cpp |
| Extend MLIREmitter for new ops | ⬚ Todo | MLIREmitter.cpp |
| Full ResNet inference | ⬚ Todo | - |

**Goal:** Full support for ResNet model. User will extend based on patterns learned.

**Note:** This follows the same patterns as Phase 2-3. Repeat the process for each new op.

---

## File Structure

```
middle/mlir/
├── CMakeLists.txt           # Build configuration
├── build.sh                 # Build script
├── .clangd                  # IDE configuration
├── include/
│   ├── Gawee/
│   │   ├── GaweeDialect.td  # Dialect TableGen
│   │   ├── GaweeDialect.h   # Dialect C++ header
│   │   ├── GaweeOps.td      # Ops TableGen
│   │   └── generated/       # Generated .inc files
│   └── Emit/
│       └── MLIREmitter.h    # JSON→MLIR emitter header
├── lib/
│   ├── Gawee/
│   │   └── GaweeDialect.cpp # Dialect implementation
│   ├── Conversion/
│   │   └── GaweeToLinalg.cpp # Conversion pass
│   └── Emit/
│       └── MLIREmitter.cpp  # JSON→MLIR emitter implementation
├── tools/
│   ├── gawee-opt.cpp        # Optimizer tool (incl. LLVM lowering)
│   └── gawee-translate.cpp  # Translator tool (JSON→MLIR)
├── scripts/
│   ├── full_pipeline.sh     # Gawee → Loops pipeline
│   └── to_llvm_ir.sh        # MLIR → LLVM IR pipeline
├── test/
│   ├── simple_test.mlir     # Test file (hand-written Gawee)
│   ├── llvm_test.mlir       # Test file for LLVM lowering
│   └── subset_graph.json    # Test JSON for translator
└── docs/
    ├── progress.md          # This file
    ├── todo.md              # Study checklist
    ├── GaweeToLinalg_Summary.md
    ├── GaweeToLinalg_Quiz.cpp
    ├── gawee-opt_Summary.md
    ├── gawee-opt_Quiz.cpp
    ├── LinalgToLoops_Summary.md
    ├── LinalgToLoops_Quiz.md
    ├── MLIREmitter_Summary.md
    ├── MLIREmitter_Quiz.cpp
    ├── LLVMLowering_Summary.md
    └── LLVMLowering_Quiz.cpp
```

---

## Commands Reference

```bash
# Build everything
./build.sh

# gawee-opt: Transform MLIR
./build/gawee-opt --help
./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir
./build/gawee-opt --gawee-to-loops test/simple_test.mlir
./build/gawee-opt --scf-to-llvm test/llvm_test.mlir

# gawee-translate: JSON → MLIR
./build/gawee-translate test/subset_graph.json
./build/gawee-translate test/subset_graph.json -o output.mlir

# Full pipeline: JSON → MLIR → Linalg
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --convert-gawee-to-linalg

# Full pipeline: JSON → MLIR → LLVM dialect
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --gawee-to-llvm

# Full pipeline: MLIR → LLVM IR
./scripts/to_llvm_ir.sh test/llvm_test.mlir output.ll
```

---

## Notes & Decisions

- Using LLVM 16+ (new cast syntax: `mlir::cast<T>(value)`)
- NCHW format for convolution (batch, channels, height, width)
- Linalg destination-passing style for all ops
- `rewriter.create<>()` is deprecated but still works (ignore warnings for now)
