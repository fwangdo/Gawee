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

## Phase 4: gawee-opt Tool 🔄 IN PROGRESS

| Task | Status | Files |
|------|--------|-------|
| Create gawee-opt executable | ⬚ Todo | tools/gawee-opt.cpp |
| Register dialects | ⬚ Todo | tools/gawee-opt.cpp |
| Register passes | ⬚ Todo | tools/gawee-opt.cpp |
| Update CMakeLists.txt | ⬚ Todo | CMakeLists.txt |
| Test with sample IR | ⬚ Todo | test/conv_test.mlir |

**Goal:** Create a tool to run passes on MLIR files:
```bash
./gawee-opt --convert-gawee-to-linalg input.mlir
```

---

## Phase 5: Linalg → Loops (Future)

| Task | Status | Files |
|------|--------|-------|
| Linalg to SCF conversion | ⬚ Todo | - |
| Linalg to Affine conversion | ⬚ Todo | - |
| Loop optimizations | ⬚ Todo | - |

**Goal:** Lower linalg ops to explicit loops (for/while).

---

## Phase 6: Loops → LLVM (Future)

| Task | Status | Files |
|------|--------|-------|
| SCF to LLVM conversion | ⬚ Todo | - |
| Arith to LLVM conversion | ⬚ Todo | - |
| MemRef to LLVM conversion | ⬚ Todo | - |

**Goal:** Lower to LLVM dialect for code generation.

---

## Phase 7: LLVM Backend (Future)

| Task | Status | Files |
|------|--------|-------|
| LLVM IR generation | ⬚ Todo | - |
| Target code generation | ⬚ Todo | - |
| JIT execution | ⬚ Todo | - |

**Goal:** Generate executable binary or run via JIT.

---

## Phase 8: Frontend Connection (Future)

| Task | Status | Files |
|------|--------|-------|
| Parser → Gawee MLIR emission | ⬚ Todo | middle/src/Parser.cpp |
| Model loading | ⬚ Todo | - |
| End-to-end test | ⬚ Todo | - |

**Goal:** Connect existing frontend to MLIR pipeline.

---

## File Structure

```
middle/mlir/
├── CMakeLists.txt           # Build configuration
├── build.sh                 # Build script
├── .clangd                  # IDE configuration
├── include/
│   └── Gawee/
│       ├── GaweeDialect.td  # Dialect TableGen
│       ├── GaweeDialect.h   # Dialect C++ header
│       ├── GaweeOps.td      # Ops TableGen
│       └── generated/       # Generated .inc files
├── lib/
│   ├── Gawee/
│   │   └── GaweeDialect.cpp # Dialect implementation
│   └── Conversion/
│       └── GaweeToLinalg.cpp # Conversion pass
├── tools/                   # (to create)
│   └── gawee-opt.cpp        # Optimizer tool
├── test/                    # (to create)
│   └── *.mlir               # Test files
└── docs/
    ├── progress.md          # This file
    ├── GaweeToLinalg_Summary.md
    └── GaweeToLinalg_Quiz.cpp
```

---

## Commands Reference

```bash
# Build everything
./build.sh

# After gawee-opt is created:
./build/gawee-opt --help
./build/gawee-opt --convert-gawee-to-linalg test/input.mlir
```

---

## Notes & Decisions

- Using LLVM 16+ (new cast syntax: `mlir::cast<T>(value)`)
- NCHW format for convolution (batch, channels, height, width)
- Linalg destination-passing style for all ops
- `rewriter.create<>()` is deprecated but still works (ignore warnings for now)
