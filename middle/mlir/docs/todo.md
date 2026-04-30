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
  - optimization passes: implemented as heuristic planning / annotation passes
  - pre-bufferization cleanup: implemented for `tensor.empty` and no-op `tensor.cast`

---

## Phase 6: JSON → Gawee MLIR (Translator) ◻ OPTIONAL / LOW PRIORITY

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

### Why This Is Low Priority Now
- the emitter is useful, but comparatively straightforward
- it does not teach the hard part of the middle-end
- current study priority should be:
  - `Linalg` transform reasoning
  - bufferization
  - `SCF`
  - LLVM lowering
  - end-to-end IR reconstruction

---

## Phase 7: Linalg Transform → Bufferize → SCF → LLVM 🔄 CURRENT

### Files to Read
- [ ] `lib/Conversion/LinalgTransform.cpp`
- [ ] `lib/Conversion/LinalgFusion.cpp`
- [ ] `lib/Conversion/LinalgScheduling.cpp`
- [ ] `lib/Conversion/LinalgVectorization.cpp`
- [ ] `lib/Conversion/LinalgVerification.cpp`
- [ ] `lib/Conversion/BufferizePrep.cpp`
- [ ] `tools/gawee-opt.cpp` - Look at `--scf-to-llvm` pipeline
- [ ] `scripts/to_llvm_ir.sh` - LLVM IR generation script
- [ ] `test/llvm_test.mlir` - Test file for LLVM lowering

### Study Materials
- [ ] **Read**: `docs/LLVMLowering_Summary.md`
- [ ] **Practice**: `docs/LLVMLowering_Quiz.cpp`
- [x] `explanation/01_LinalgTransform.md` ~ `08_Pipeline_BigPicture.md`

### Key Concepts
- real tiling rewrite vs tiling metadata
- fusion planning vs real fusion rewrite
- scheduling hints vs actual loop reorder
- vectorization prep vs real vector lowering
- bufferization preparation
- Dialect hierarchy (high → low level)
- SCF → CF (structured → unstructured control flow)
- MemRef → LLVM struct representation
- Multiple conversion passes and order
- UnrealizedConversionCast
- mlir-translate (MLIR ↔ LLVM IR)

### Updated Mental Model
- Step 1: `Gawee -> Linalg` legalization
- Step 2: tiling pass
  - current file: `lib/Conversion/LinalgTransform.cpp`
  - current implementation:
    - inspect conv / matmul / generic families
    - choose tile-size hints
    - attach explicit `gawee.transform.*` attrs
  - intended ownership: tiling decisions before loop lowering
- Step 3: fusion pass
  - current file: `lib/Conversion/LinalgFusion.cpp`
  - current implementation:
    - detect simple single-use producer/consumer pairs
    - attach `gawee.fusion.group` / role attrs
  - intended ownership: producer/consumer fusion and post-op fusion planning
- Step 4: scheduling pass
  - current file: `lib/Conversion/LinalgScheduling.cpp`
  - current implementation:
    - count parallel vs reduction loops
    - attach loop-interchange and scheduling-hint attrs
  - intended ownership: loop order / parallel / reduction scheduling decisions
- Step 5: vectorization pass
  - current file: `lib/Conversion/LinalgVectorization.cpp`
  - current implementation:
    - attach vectorization kind + width hints
    - record static-result readiness
  - intended ownership: vectorization readiness and vector-lowering preparation
- Step 6: verification pass
  - current file: `lib/Conversion/LinalgVerification.cpp`
  - current implementation:
    - summarize linalg coverage at module level
    - mark per-op verification status attrs
  - intended ownership: transform precondition checks and IR-side verification hooks
- Step 7: bufferization preparation
  - current file: `lib/Conversion/BufferizePrep.cpp`
  - current implementation:
    - replace `tensor.empty` with `bufferization.alloc_tensor`
    - fold no-op `tensor.cast`
    - attach destination-style bufferization attrs
  - intended ownership: pre-bufferization cleanup and normalization
- Step 8: one-shot bufferization (`tensor -> memref`)
- Step 9: `Linalg(memref) -> SCF`
- Step 10: `SCF -> CF -> LLVM`

### Explanation Writing Plan
- [x] `explanation/01_LinalgTransform.md` ✅
  - explain why this is still `Linalg -> Linalg`, not `Linalg -> SCF`
  - explain planning structs, heuristics, remarks, and attrs
- [x] `explanation/02_LinalgFusion.md` ✅
  - explain producer/consumer detection, fusion groups, and current limits
- [x] `explanation/03_LinalgScheduling.md` ✅
  - explain loop classification, interchange hints, and scheduling attrs
- [x] `explanation/04_LinalgVectorization.md` ✅
  - explain width hints, static-result checks, and vector-prep intent
- [x] `explanation/05_LinalgVerification.md` ✅
  - explain module-level verification summary and per-op verification attrs
- [x] `explanation/06_BufferizePrep.md` ✅
  - explain why `tensor.empty` / `tensor.cast` are normalized before bufferization
- [x] `explanation/07_gawee-opt_pipeline.md` ✅
  - explain how the pass pipeline is assembled from `Linalg` transforms through LLVM lowering
- [x] `explanation/08_Pipeline_BigPicture.md` ✅
  - explain the end-to-end story:
    `Gawee -> Linalg -> transform/fusion/scheduling/vector/verify -> bufferize -> SCF -> CF -> LLVM`

### Middle-end Completion Direction
- To honestly say "I covered the middle-end broadly", target this sequence:
  1. replace attr-level tiling hints with real conv/matmul tiling rewrites
  2. replace fusion grouping attrs with at least one real producer-consumer fusion rewrite
  3. replace scheduling hints with at least one concrete loop reorder / parallel choice
  4. replace vectorization hints with a real vector-friendly lowering step
  5. connect verification attrs to correctness/latency experiments in `back/`
- In other words:
  - "lowering works" is the starting point
  - "performance transforms + verification loop work" is the fuller middle-end story

### Training Goal
- do not stop at "I know one lowering function"
- be able to rebuild, alone, the full path:
  - `Gawee op understanding`
  - `Gawee -> Linalg`
  - `Linalg transform/fusion/scheduling/vector/verify`
  - `BufferizePrep`
  - `OneShotBufferize`
  - `Linalg -> SCF`
  - `SCF -> LLVM`
  - `LLVM dialect -> LLVM IR`

### Concrete Practice Order
1. rebuild `Conv`, `Linear`, `Reshape`, `Transpose`, `Softmax` lowerings by hand
2. explain and re-implement the planning passes in plain words
3. trace one `resnet18` subgraph from `Linalg` to `SCF`
4. trace the same subgraph from `SCF` to LLVM dialect
5. dump final LLVM IR and explain where each major op went
6. only after that, return to emitter if needed

### Current Backend Status
- `resnet18`
  - `gawee-to-loops`: passes
  - `gawee-to-llvm`: passes
  - AOT runner build: passes
  - AOT runner execution: passes
  - output shape matches ONNX Runtime: `(1, 1000)`
  - correctness now matches ONNX Runtime on the saved evaluation input:
    - `max_abs_diff ~= 2.86e-06`
    - `mean_abs_diff ~= 6.07e-07`
    - `np.allclose(..., atol=1e-4, rtol=1e-4)` passes
  - current latency baseline:
    - Gawee AOT end-to-end runner: about `6.52s ~ 6.59s`
    - ONNX Runtime inference-only: about `15.4ms ~ 17.4ms`
  - interpretation:
    - the backend execution path is closed
    - the result is numerically aligned on the current saved-input check
    - the current AOT latency includes file I/O and process launch, so it is only a coarse baseline
- `bert_tiny`, `distilbert_base_uncased`
  - translator and several dynamic lowering blockers were improved
  - but end-to-end LLVM/AOT/correctness are not closed yet
  - next focus:
    - dynamic shape + broadcast cleanup
    - slice/reduce/reshape correctness under bufferization

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
2. Phase 4 (gawee-opt) - Pipeline structure
3. Phase 7 (Linalg transform → LLVM) - Main study target
4. Phase 5 (LinalgToLoops) - Bufferization and loop lowering details
5. Phase 6 (MLIREmitter) - Optional reinforcement, not main bottleneck

---

## Progress Tracker

| Phase | Summary Read | Quiz Done | Code Understood |
|-------|--------------|-----------|-----------------|
| 2     | N/A          | N/A       | [x] ✅          |
| 3     | [x]          | [x]       | [x] ✅          |
| 4     | [x]          | [x]       | [x] ✅          |
| 5     | [x]          | [x]       | [x] ✅          |
| 6     | [ ] optional | [ ]       | [ ]             |
| 7     | [ ] **NEXT** | [ ]       | [ ]             |
| 8     | N/A (guides) | N/A       | [ ] (implement) |
