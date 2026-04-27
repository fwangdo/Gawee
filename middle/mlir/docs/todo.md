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

### Pipeline Clarification
- `Gawee -> Linalg` and `Linalg -> SCF` are not the same step
- There is now an explicit middle slot for `Linalg`-level transforms
- Typical work in that slot: tiling, fusion, scheduling, vectorization prep

### Direction After Lowering
- Saying "I experienced AI compiler middle-end broadly" now requires more than legalization and bufferization
- The missing center of gravity is optimization-oriented middle-end work:
  - real tiling
  - real fusion
  - real scheduling
  - real vectorization prep / vector lowering
  - correctness + latency verification loop
- Current codebase status:
  - legalization/lowering: implemented
  - bufferization pipeline: wired
  - end-to-end AOT execution: wired
  - optimization passes: scaffold stage

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

### Updated Mental Model
- Step 1: `Gawee -> Linalg` legalization
- Step 2: tiling scaffold pass
  - current scaffold: `lib/Conversion/LinalgTransformScaffold.cpp`
  - intended ownership: tiling decisions before loop lowering
- Step 3: fusion scaffold pass
  - current scaffold: `lib/Conversion/LinalgFusionScaffold.cpp`
  - intended ownership: producer/consumer fusion and post-op fusion planning
- Step 4: scheduling scaffold pass
  - current scaffold: `lib/Conversion/LinalgSchedulingScaffold.cpp`
  - intended ownership: loop order / parallel / reduction scheduling decisions
- Step 5: vectorization scaffold pass
  - current scaffold: `lib/Conversion/LinalgVectorizationScaffold.cpp`
  - intended ownership: vectorization readiness and vector-lowering preparation
- Step 6: verification scaffold pass
  - current scaffold: `lib/Conversion/LinalgVerificationScaffold.cpp`
  - intended ownership: transform precondition checks and IR-side verification hooks
- Step 7: bufferization preparation
  - current scaffold: `lib/Conversion/BufferizePrepScaffold.cpp`
  - intended ownership: pre-bufferization cleanup and normalization
- Step 8: one-shot bufferization (`tensor -> memref`)
- Step 9: `Linalg(memref) -> SCF`
- Step 10: `SCF -> CF -> LLVM`

### Middle-end Completion Direction
- To honestly say "I covered the middle-end broadly", target this sequence:
  1. replace tiling scaffold with real conv/matmul tiling
  2. replace fusion scaffold with at least one real producer-consumer fusion path
  3. replace scheduling scaffold with at least one concrete loop reorder / parallel choice
  4. replace vectorization scaffold with a real vector-friendly lowering step
  5. connect verification scaffold to correctness/latency experiments in `back/`
- In other words:
  - "lowering works" is the starting point
  - "performance transforms + verification loop work" is the fuller middle-end story

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
