# Study TODO List

This file tracks what you need to study and practice.

---

## Phase 2: Gawee Dialect Definition ✅ COMPLETED

### Files to Read
- [x] `include/Gawee/GaweeDialect.td` - Dialect TableGen definition
- [x] `include/Gawee/GaweeOps.td` - Op TableGen definitions
- [x] `lib/Gawee/GaweeDialect.cpp` - Dialect C++ implementation

### Key Concepts
- TableGen syntax (.td files)
- Dialect declaration
- Op definition (ins, outs, arguments, results)
- Generated code (.inc files)

---

## Phase 3: Gawee → Linalg Conversion ✅ COMPLETED

### Files to Read
- [x] `lib/Conversion/GaweeToLinalg.cpp` - Conversion pass implementation

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

### Key Concepts
- DialectRegistry
- PassPipelineRegistration
- MlirOptMain
- getDependentDialects

---

## Phase 5: Linalg → Loops ✅ COMPLETED

### Key Concepts
- Bufferization (tensor → memref)
- Linalg to loops conversion
- MLIR built-in passes

---

## Phase 7: Middle-End Optimization Pipeline 🔄 CURRENT

### 현재 상태 (2026-05-22)

#### Pass별 구현 상태

| Pass | 상태 | IR 변형 | API |
|------|------|---------|-----|
| LinalgTransform | ✅ 동작 | tiling + tile loop interchange | `scf::tileUsingSCF` |
| LinalgFusion | ✅ 동작 | elementwise generic fusion | `populateElementwiseOpsFusionPatterns` |
| LinalgScheduling | ✅ 동작 | generic interchange + loop peeling | `interchangeGenericOp`, `peelForLoopAndSimplifyBounds` |
| LinalgVectorization | ✅ 동작 | projected-permutation elementwise (broadcast 포함), maxDim≤32 | `linalg::vectorize` |
| LinalgVerification | ✅ 동작 | 분석/진단만 | attr + remark |
| Canonicalize + CSE | ✅ 동작 | cleanup (3곳 삽입) | `createCanonicalizerPass`, `createCSEPass` |
| LICM | ✅ 동작 | loop invariant hoist | `createLoopInvariantCodeMotionPass` |
| VectorToLLVM | ✅ 동작 | vector → LLVM lowering | `createConvertVectorToLLVMPass` |
| BufferizePrep | ✅ 동작 | tensor.empty→alloc_tensor | 커스텀 |
| DecomposeAggregated | ✅ 동작 | softmax 등 분해 (tiling 전으로 이동됨) | 커스텀 |

#### Pipeline 순서

```
GaweeToLinalg
  → DecomposeAggregated (softmax 등을 tiling 전에 분해)
  → LinalgTransform (tiling + interchange)
  → Canonicalize + CSE
  → LinalgFusion (elementwise fusion)
  → LinalgScheduling (generic interchange + peeling)
  → Canonicalize + CSE
  → LinalgVectorization (projected-permutation elementwise, maxDim≤32)
  → LinalgVerification
  → EmptyTensorToAllocTensor
  → BufferizePrep
  → OneShotBufferize
  → Canonicalize + CSE
  → VectorToSCF
  → LinalgToLoops
  → LICM
  → SCFToControlFlow
  → ExpandStridedMetadata + LowerAffine
  → CWrappers
  → MathToLibm → MathToLLVM → ArithToLLVM → CFToLLVM → VectorToLLVM → UBToLLVM → MemRefToLLVM → FuncToLLVM
  → ReconcileUnrealizedCasts
```

#### Correctness

| model | 상태 | max_abs_diff | atol |
|-------|------|-------------|------|
| resnet18 | ✅ PASS | 5.25e-06 | 1e-4 |
| bert_tiny | ✅ PASS | 1.79e-07 | 5e-4 |
| tinyllama_15m | ✅ PASS | 1.62e-05 | 5e-4 |

#### Latency (Gawee p50 vs ORT median, 2026-05-22)

| model | baseline (ms) | optimized (ms) | speedup | ORT (ms) | Gawee/ORT |
|-------|--------------|----------------|---------|----------|-----------|
| resnet18 | 6555 | 6743 | 0.97x | 16 | 421x |
| bert_tiny | 228 | 346 | 0.66x | 0.6 | 577x |
| tinyllama_15m | 97 | 106 | 0.92x | 1.6 | 66x |

---

### 성능 분석

#### 왜 baseline 대비 개선이 없는가

현재 optimized가 baseline보다 오히려 **약간 느리다** (resnet18: 0.96x, tinyllama: 0.93x).

원인:
1. **tiling이 loop overhead 추가** — outer loop + inner loop로 분할되면서 loop 제어 비용 증가
2. **그 overhead를 상쇄할 최적화 부재**:
   - fusion: 동작하지만 elementwise generic끼리만 합침. tiled loop 안의 producer를 끌어오는 tile-and-fuse는 미구현
   - vectorization: 동작하지만 maxDim=32 제한으로 대부분 op이 대상 밖
   - canonicalize/CSE/LICM: 정리할 redundant op이 적어서 실질 영향 없음
3. **tiling 자체의 가치**: tiling은 단독으로는 성능을 올리지 않는다.
   tiling + fusion + vectorization이 조합되어야 cache locality + SIMD가 동시에 달성됨

#### ORT 대비 100~600x 느린 원인 분해

| 원인 | 예상 배수 | 설명 |
|------|----------|------|
| Vectorization 제한적 | 8-16x | SIMD 미사용 — maxDim=32 제한으로 대부분 op 미적용 |
| Tile-and-Fuse 없음 | 2-4x | 중간 텐서를 매번 메모리에 write/read |
| Micro-kernel 없음 | 4-8x | ORT는 hand-tuned GEMM kernel 사용 |
| LLVM backend 차이 | 2-4x | ORT는 MKL/Eigen 수준 최적화 |

8 × 3 × 6 × 3 ≈ 400x, resnet18의 실측과 일치.

#### 성능 개선 가능 지점 (우선순위 순)

**1. Tile-and-Fuse — 가장 현실적 (예상 2-4x)**
- 현재: tiling(LinalgTransform)과 fusion(LinalgFusion)이 별도 pass
- 문제: tiling 후 생긴 scf.for 안의 tiled op은 elementwise fusion 대상이 아님
- 해결: `scf::tileConsumerAndFuseProducersUsingSCF()`로 tiling 시 producer를 동시에 fusion
- 효과: conv+bias+relu가 한 tile loop에서 실행 → 중간 full-size 텐서 할당 제거
- 복잡도: LinalgTransform.cpp의 tileConvLikeOps/tileMatmulLikeOps 재구성 필요

**2. Vectorization 확대 — 가장 큰 단일 개선 (예상 8-16x)**
- 파이프라인은 완성: vectorize → VectorToSCF → VectorToLLVM → UBToLLVM
- 현재 제한: maxDim=32이므로 tiling 전 원본 텐서는 대부분 대상 밖
- 핵심 전략: tile-and-fuse로 작은 tile 생성 → tiled op이 maxDim 이하 → vectorize 적용
- 추가 개선:
  a. constant-index map `(d0,d1,d2) -> (d0,d1,0)` 처리 (fusion에서 생성)
  b. named op (conv, matmul) vectorization 추가

**3. 조합 효과**
- tile-and-fuse + vectorization = 실질적 성능 전환점
- tiling으로 small tile → vectorize로 SIMD → fused loop에서 cache 재사용
- 이 조합이 ORT 대비 10-20x까지 좁힐 수 있는 현실적 목표

---

### 다음 단계

- [x] **vectorization 필터 완화** — projected permutation + 0-result map 허용, replacement 버그 수정
- [x] **vectorization pipeline 완성** — VectorToSCF, UBToLLVM 추가
- [ ] **tile-and-fuse replacement 디버깅** (소규모 IR로 검증 후 적용)
- [ ] **vectorization + tiling 조합** — tiled ops가 maxDim 이하가 되면 vectorize 가능

---

## Phase 8: Extension (Your Own Work)

### Tasks
- [ ] Add `gawee.maxpool` op to dialect
- [ ] Add `gawee.batchnorm` op to dialect
- [ ] Implement `MaxPoolOpLowering`
- [ ] Implement `BatchNormOpLowering`
- [ ] Test full pipeline with new ops

---

## Quick Reference

### Build Commands
```bash
cd middle/mlir && ./build.sh
```

### Test Commands
```bash
# Full pipeline: Gawee → LLVM
./build/gawee-opt --gawee-to-llvm test/simple_test.mlir

# Baseline (no linalg optimizations)
./build/gawee-opt --gawee-to-llvm-baseline test/simple_test.mlir

# Backend evaluation (all priority models)
python back/eval_priority_models.py
python back/eval_priority_models.py --baseline
```

---

## Progress Tracker

| Phase | Status |
|-------|--------|
| 2 - Dialect definition | ✅ |
| 3 - GaweeToLinalg | ✅ |
| 4 - gawee-opt tool | ✅ |
| 5 - Linalg → Loops | ✅ |
| 7 - Middle-end optimization | 🔄 tiling/fusion/scheduling/vectorization/CSE/LICM 동작, tile-and-fuse 미구현 |
| 8 - Extension ops | ◻ |
