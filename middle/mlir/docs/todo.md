# Study TODO List

This file tracks what you need to study and practice.

---

## Phase 2: Gawee Dialect Definition ✅ COMPLETED

### Files to Read
- [x] `include/Gawee/GaweeDialect.td` - Dialect TableGen definition
- [x] `include/Gawee/GaweeOps.td` - Op TableGen definitions
- [x] `lib/Gawee/GaweeDialect.cpp` - Dialect C++ implementation

### Study Materials
- [x] (No summary/quiz yet - dialect definition basics)

### Key Concepts
- TableGen syntax (.td files)
- Dialect declaration
- Op definition (ins, outs, arguments, results)
- Generated code (.inc files)

---

## Phase 3: Gawee → Linalg Conversion ✅ COMPLETED

### Files to Read
- [x] `lib/Conversion/GaweeToLinalg.cpp` - Conversion pass implementation

### Study Materials
- [x] **Read**: `docs/GaweeToLinalg_Summary.md`
- [x] **Practice**: `docs/GaweeToLinalg_Quiz.cpp`

### Key Concepts
- OpConversionPattern
- ConversionTarget (legal/illegal)
- Rewriter API
- Destination-passing style
- linalg.generic for custom ops

---

## Phase 4: gawee-opt Tool ✅ COMPLETED

### Files to Read
- [x] `tools/gawee-opt.cpp` - Optimizer tool

### Study Materials
- [x] **Read**: `docs/gawee-opt_Summary.md`
- [x] **Practice**: `docs/gawee-opt_Quiz.cpp`

### Key Concepts
- DialectRegistry
- PassPipelineRegistration
- MlirOptMain
- getDependentDialects
- RTTI settings

---

## Phase 5: Linalg → Loops ✅ COMPLETED

### Files to Read
- [x] `scripts/full_pipeline.sh` - Pipeline script

### Study Materials
- [x] **Read**: `docs/LinalgToLoops_Summary.md`
- [x] **Practice**: `docs/LinalgToLoops_Quiz.md`
- [x] **Bonus**: `docs/ShellScript_Summary.md`
- [x] **Bonus**: `docs/ShellScript_Quiz.sh`

### Key Concepts
- Bufferization (tensor → memref)
- Linalg to loops conversion
- MLIR built-in passes

---

## Phase 6: JSON → Gawee MLIR (Translator) 🔄 CURRENT

### Files to Read
- [ ] `include/Emit/MLIREmitter.h` - Emitter header
- [ ] `lib/Emit/MLIREmitter.cpp` - Emitter implementation
- [ ] `tools/gawee-translate.cpp` - Translator tool

### Study Materials
- [ ] **Read**: `docs/MLIREmitter_Summary.md`
- [ ] **Practice**: `docs/MLIREmitter_Quiz.cpp`

### Key Concepts
- mlir-opt vs mlir-translate
- OpBuilder usage
- Value mapping (string → mlir::Value)
- RankedTensorType creation
- LLVM JSON API
- Topological ordering

---

## Phase 7: SCF → LLVM → Binary

### Files to Read
- [ ] `tools/gawee-opt.cpp` - Look at `--scf-to-llvm` pipeline
- [ ] `scripts/to_llvm_ir.sh` - LLVM IR generation script
- [ ] `test/llvm_test.mlir` - Test file for LLVM lowering

### Study Materials
- [ ] **Read**: `docs/LLVMLowering_Summary.md`
- [ ] **Practice**: `docs/LLVMLowering_Quiz.cpp`

### Key Concepts
- Dialect hierarchy (high → low level)
- SCF → CF (structured → unstructured control flow)
- MemRef → LLVM struct representation
- Multiple conversion passes and order
- UnrealizedConversionCast
- mlir-translate (MLIR ↔ LLVM IR)

---

## Phase 8: Extension (Your Own Work)

### Guide Files (Comment-only scaffolds)
- [ ] **Read**: `docs/extension/00_Extension_Checklist.md` - Master checklist
- [ ] `docs/extension/01_GaweeOps_Extension.td` - Add MaxPool, BatchNorm ops
- [ ] `docs/extension/02_GaweeToLinalg_Extension.cpp` - Lowering patterns
- [ ] `docs/extension/03_MLIREmitter_Extension.cpp` - JSON emission

### Tasks
- [ ] Add `gawee.maxpool` op to dialect
- [ ] Add `gawee.batchnorm` op to dialect
- [ ] Implement `MaxPoolOpLowering`
- [ ] Implement `BatchNormOpLowering`
- [ ] Implement `emitMaxPool()` in MLIREmitter
- [ ] Implement `emitBatchNorm()` in MLIREmitter
- [ ] Test full pipeline with new ops
- [ ] (Optional) Add bias support to conv

### Goal
Full support for ResNet-like models with all common ops.

---

## Quick Reference

### Build Commands
```bash
./build.sh                                    # Build everything
```

### Test Commands
```bash
# Phase 3-5: Gawee → Linalg → Loops
./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir
./build/gawee-opt --gawee-to-loops test/simple_test.mlir

# Phase 6: JSON → MLIR
./build/gawee-translate test/subset_graph.json

# Phase 7: SCF → LLVM
./build/gawee-opt --scf-to-llvm test/llvm_test.mlir
./scripts/to_llvm_ir.sh test/llvm_test.mlir output.ll

# Full pipeline: JSON → LLVM
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --gawee-to-llvm

# Phase 8 (Extension): Test your new ops
./build/gawee-opt --convert-gawee-to-linalg test/extension_test.mlir
./build/gawee-translate test/extension_graph.json | ./build/gawee-opt --gawee-to-llvm
```

### Study Order (Recommended)
1. Phase 3 (GaweeToLinalg) - Core pattern rewriting
2. Phase 4 (gawee-opt) - Tool infrastructure
3. Phase 6 (MLIREmitter) - Building IR programmatically
4. Phase 7 (LLVMLowering) - Full lowering pipeline
5. Phase 5 (LinalgToLoops) - Bufferization details

---

## Progress Tracker

| Phase | Summary Read | Quiz Done | Code Understood |
|-------|--------------|-----------|-----------------|
| 2     | N/A          | N/A       | [x] ✅          |
| 3     | [x]          | [x]       | [x] ✅          |
| 4     | [x]          | [x]       | [x] ✅          |
| 5     | [x]          | [x]       | [x] ✅          |
| 6     | [ ] **NEXT** | [ ]       | [ ]             |
| 7     | [ ]          | [ ]       | [ ]             |
| 8     | N/A (guides) | N/A       | [ ] (implement) |
