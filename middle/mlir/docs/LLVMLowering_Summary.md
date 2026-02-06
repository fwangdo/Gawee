# LLVM Lowering (Phase 7) - Summary

## What You Need to Understand

### 1. The Full Lowering Pipeline

```
Gawee dialect
    ↓ (gawee-to-linalg)
Linalg dialect (on tensors)
    ↓ (one-shot-bufferize)
Linalg dialect (on memrefs)
    ↓ (convert-linalg-to-loops)
SCF dialect (loops) + MemRef + Arith
    ↓ (scf-to-cf)
ControlFlow dialect (branches) + MemRef + Arith
    ↓ (arith-to-llvm, memref-to-llvm, cf-to-llvm, func-to-llvm)
LLVM dialect
    ↓ (mlir-translate --mlir-to-llvmir)
LLVM IR (.ll file)
    ↓ (llc)
Assembly / Object file
    ↓ (clang/ld)
Executable binary
```

### 2. Dialect Hierarchy

```
High-level (domain-specific)
│  Gawee dialect       → Neural network ops (conv, relu, add)
│  Linalg dialect      → Linear algebra + structured ops
│
Mid-level (control flow)
│  SCF dialect         → Structured Control Flow (for, while, if)
│  CF dialect          → Unstructured Control Flow (br, cond_br)
│
Low-level (memory)
│  MemRef dialect      → Memory references (load, store, alloc)
│  Arith dialect       → Arithmetic operations (addf, mulf, etc.)
│
Target
   LLVM dialect        → LLVM IR operations (llvm.load, llvm.br, etc.)
```

### 3. Key Passes

| Pass | Input | Output | Purpose |
|------|-------|--------|---------|
| `convert-linalg-to-loops` | Linalg (memref) | SCF + MemRef | Convert linalg ops to explicit loops |
| `convert-scf-to-cf` | SCF | CF | Lower structured loops to branches |
| `convert-arith-to-llvm` | Arith | LLVM | Lower arithmetic ops |
| `finalize-memref-to-llvm` | MemRef | LLVM | Lower memory ops to LLVM |
| `convert-cf-to-llvm` | CF | LLVM | Lower branches to LLVM |
| `convert-func-to-llvm` | Func | LLVM | Lower function definitions |
| `reconcile-unrealized-casts` | Mixed | Clean | Remove type conversion markers |

### 4. SCF to CF Transformation

**Before (SCF):**
```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %val = memref.load %A[%i] : memref<4xf32>
  memref.store %val, %B[%i] : memref<4xf32>
}
```

**After (CF):**
```mlir
  cf.br ^header(%c0 : index)
^header(%i: index):
  %cond = arith.cmpi slt, %i, %c4 : index
  cf.cond_br %cond, ^body, ^exit
^body:
  %val = memref.load %A[%i] : memref<4xf32>
  memref.store %val, %B[%i] : memref<4xf32>
  %next = arith.addi %i, %c1 : index
  cf.br ^header(%next : index)
^exit:
  // continue...
```

### 5. MemRef to LLVM Transformation

**MemRef representation in LLVM:**
```
memref<4xf32> → struct {
  ptr allocated;     // Base allocation pointer
  ptr aligned;       // Aligned data pointer
  i64 offset;        // Offset from aligned ptr
  i64 sizes[1];      // Size in each dimension
  i64 strides[1];    // Stride in each dimension
}
```

**MemRef load:**
```mlir
// MLIR
%val = memref.load %A[%i] : memref<4xf32>

// LLVM dialect
%ptr = llvm.extractvalue %A[1]   // Get aligned pointer
%gep = llvm.getelementptr %ptr[%i]
%val = llvm.load %gep : !llvm.ptr -> f32
```

### 6. UnrealizedConversionCast

During conversion, different passes may use different type systems:
```mlir
// Temporary cast during conversion
%0 = builtin.unrealized_conversion_cast %arg : memref<4xf32> to !llvm.struct<...>
```

The `reconcile-unrealized-casts` pass removes these after all conversions are done.

### 7. LLVM Dialect vs LLVM IR

| LLVM Dialect (MLIR) | LLVM IR |
|---------------------|---------|
| `llvm.load %ptr : !llvm.ptr -> f32` | `%0 = load float, ptr %ptr` |
| `llvm.store %val, %ptr : f32, !llvm.ptr` | `store float %val, ptr %ptr` |
| `llvm.br ^bb1` | `br label %bb1` |
| `llvm.call @foo()` | `call void @foo()` |
| `llvm.return %val` | `ret i32 %val` |

### 8. Bufferization Interface Registration (Critical!)

One-shot bufferization needs **external models** to know how to bufferize each dialect:

```cpp
// Include bufferization interfaces
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

// Register interfaces in dialect registry
arith::registerBufferizableOpInterfaceExternalModels(registry);
linalg::registerBufferizableOpInterfaceExternalModels(registry);
tensor::registerBufferizableOpInterfaceExternalModels(registry);
bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
```

**Why is this needed?**
- One-shot bufferization is analysis-based (decides which tensors can alias)
- Each dialect must tell the pass how its ops behave with respect to memory
- Without registration: `error: op was not bufferized`

**CMake libraries:**
```cmake
MLIRArithTransforms
MLIRLinalgTransforms  # Already included for linalg-to-loops
MLIRTensorTransforms
```

### 9. Registration in gawee-opt

```cpp
// Include conversion passes
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// Register passes in pipeline
PassPipelineRegistration<>(
    "scf-to-llvm",
    "Lower SCF/MemRef to LLVM",
    [](OpPassManager &pm) {
      pm.addPass(createSCFToControlFlowPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createConvertControlFlowToLLVMPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });

// Register LLVM dialect
registry.insert<LLVM::LLVMDialect>();
registry.insert<cf::ControlFlowDialect>();
```

### 10. CMake Libraries for LLVM Lowering

```cmake
target_link_libraries(gawee-opt
  # ... existing libraries ...

  # LLVM dialect and conversions
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
)
```

### 11. Full Pipeline Commands

```bash
# Full pipeline: JSON -> MLIR (Gawee) -> LLVM dialect
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --gawee-to-llvm

# From SCF/memref to LLVM dialect (for testing)
./build/gawee-opt --scf-to-llvm test/llvm_test.mlir

# From LLVM dialect to LLVM IR
./build/gawee-translate test/subset_graph.json | \
  ./build/gawee-opt --gawee-to-llvm | \
  mlir-translate --mlir-to-llvmir > output.ll

# Compile to object file
llc output.ll -o output.o -filetype=obj

# Link to executable (needs runtime for malloc, etc.)
clang output.o -o output
```

### 12. mlir-translate Tool

MLIR provides `mlir-translate` for format conversion:

```bash
# MLIR (LLVM dialect) -> LLVM IR
mlir-translate --mlir-to-llvmir input.mlir -o output.ll

# LLVM IR -> MLIR (LLVM dialect)
mlir-translate --import-llvm input.ll -o output.mlir
```

## Key Takeaways

1. **Lowering is hierarchical**: High-level → Mid-level → Low-level → Target
2. **SCF uses structured control**: for/while/if → converted to branches (CF)
3. **MemRef becomes struct**: Contains pointer, offset, sizes, strides
4. **Multiple passes needed**: Each dialect needs its own conversion pass
5. **Order matters**: SCF→CF before CF→LLVM, Func→LLVM typically last
6. **Unrealized casts**: Temporary markers removed at the end
7. **mlir-translate**: Bridges MLIR LLVM dialect ↔ actual LLVM IR
8. **Bufferization interfaces**: Must register external models for each dialect
