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

### 현재 상태 (2026-05-03)

#### Pass별 구현 상태

| Pass | 상태 | IR 변형 | API |
|------|------|---------|-----|
| LinalgTransform | ✅ 동작 | tiling + tile loop interchange | `scf::tileUsingSCF` |
| LinalgFusion | ✅ 동작 | elementwise generic fusion | `populateElementwiseOpsFusionPatterns` |
| LinalgScheduling | ✅ 동작 | generic interchange + loop peeling | `interchangeGenericOp`, `peelForLoopAndSimplifyBounds` |
| LinalgVectorization | ⚠ 비활성 | (코드 있음, MLIR crash) | `linalg::vectorize` |
| LinalgVerification | ✅ 동작 | 분석/진단만 | attr + remark |
| Canonicalize + CSE | ✅ 동작 | cleanup (3곳 삽입) | `createCanonicalizerPass`, `createCSEPass` |
| LICM | ✅ 동작 | loop invariant hoist | `createLoopInvariantCodeMotionPass` |
| VectorToLLVM | ✅ 등록됨 | (vectorization 비활성이라 no-op) | `createConvertVectorToLLVMPass` |
| BufferizePrep | ✅ 동작 | tensor.empty→alloc_tensor | 커스텀 |
| DecomposeAggregated | ✅ 동작 | softmax 등 분해 | 커스텀 |

#### Pipeline 순서

```
GaweeToLinalg
  → LinalgTransform (tiling + interchange)
  → Canonicalize + CSE
  → LinalgFusion (elementwise fusion)
  → LinalgScheduling (generic interchange + peeling)
  → Canonicalize + CSE
  → LinalgVectorization (비활성)
  → LinalgVerification
  → DecomposeAggregated
  → EmptyTensorToAllocTensor
  → BufferizePrep
  → OneShotBufferize
  → Canonicalize + CSE
  → LinalgToLoops
  → LICM
  → SCFToControlFlow
  → ExpandStridedMetadata + LowerAffine
  → CWrappers
  → MathToLibm → MathToLLVM → ArithToLLVM → CFToLLVM → VectorToLLVM → MemRefToLLVM → FuncToLLVM
  → ReconcileUnrealizedCasts
```

#### Correctness

| model | 상태 | max_abs_diff | atol |
|-------|------|-------------|------|
| resnet18 | ✅ PASS | 5.25e-06 | 1e-4 |
| bert_tiny | ✅ PASS | 1.79e-07 | 5e-4 |
| tinyllama_15m | ✅ PASS | 1.62e-05 | 5e-4 |

#### Latency (Gawee p50 vs ORT median)

| model | baseline (ms) | optimized (ms) | ORT (ms) | Gawee/ORT |
|-------|--------------|----------------|----------|-----------|
| resnet18 | 6575 | 6814 | 17 | 400x |
| bert_tiny | AOT fail | 354 | 0.6 | 590x |
| tinyllama_15m | 100 | 108 | 1.7 | 63x |

---

### 성능 분석

#### 왜 baseline 대비 개선이 없는가

현재 optimized가 baseline보다 오히려 **약간 느리다** (resnet18: 0.96x, tinyllama: 0.93x).

원인:
1. **tiling이 loop overhead 추가** — outer loop + inner loop로 분할되면서 loop 제어 비용 증가
2. **그 overhead를 상쇄할 최적화 부재**:
   - fusion: 동작하지만 elementwise generic끼리만 합침. tiled loop 안의 producer를 끌어오는 tile-and-fuse는 미구현
   - vectorization: MLIR crash로 비활성. scalar 연산만 사용
   - canonicalize/CSE/LICM: 정리할 redundant op이 적어서 실질 영향 없음
3. **tiling 자체의 가치**: tiling은 단독으로는 성능을 올리지 않는다.
   tiling + fusion + vectorization이 조합되어야 cache locality + SIMD가 동시에 달성됨

#### ORT 대비 100~600x 느린 원인 분해

| 원인 | 예상 배수 | 설명 |
|------|----------|------|
| Vectorization 없음 | 8-16x | SIMD 미사용 (ARM NEON 128bit = f32 4개 동시) |
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

**2. Vectorization — 가장 큰 단일 개선 (예상 8-16x, 현재 blocked)**
- MLIR 내부 crash: `VectorizationState::getOrCreateMaskFor`에서 segfault
- broadcast indexing map이 있는 generic op에서 발생
- 코드는 작성 완료 (LinalgVectorization.cpp), 후보 선별 로직도 있음
- 해결 경로:
  a. MLIR 버전 업그레이드 (masked vectorization 버그 수정 기대)
  b. loop-level vectorization (bufferize → LinalgToLoops 후, innermost scf.for를 직접 변환)
  c. vectorize 대상을 더 좁게 — broadcast 없는 pure identity map generic만

**3. 조합 효과**
- tile-and-fuse + vectorization = 실질적 성능 전환점
- tiling으로 small tile → vectorize로 SIMD → fused loop에서 cache 재사용
- 이 조합이 ORT 대비 10-20x까지 좁힐 수 있는 현실적 목표

---

### 다음 단계

- [ ] **tile-and-fuse 구현** (LinalgTransform.cpp에서 `tileConsumerAndFuseProducersUsingSCF` 사용)
- [ ] **vectorization crash 우회** (MLIR 업그레이드 또는 loop-level 접근)
- [ ] explanation 파일 업데이트 (02_LinalgFusion.md 등 현재 코드 반영)
- [ ] delta.md에 tile-and-fuse 후 재측정 결과 기록

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
| 7 - Middle-end optimization | 🔄 tiling/fusion/scheduling/CSE/LICM 동작, vectorization blocked |
| 8 - Extension ops | ◻ |
