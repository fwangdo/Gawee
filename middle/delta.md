# Middle-End Optimization Delta

## Pass 상태

| Pass | 상태 | 설명 |
|------|------|------|
| LinalgTransform | 동작 | tiling (conv/matmul) + tile loop interchange |
| LinalgFusion | **동작** | elementwise generic fusion (`populateElementwiseOpsFusionPatterns`) |
| LinalgScheduling | 동작 | generic interchange + loop peeling |
| LinalgVectorization | 비활성 | `linalg::vectorize()` crash (masked vectorization bug in MLIR build) |
| LinalgVerification | 동작 | 검증/진단 |
| Canonicalize + CSE | **동작** | tiling 후, scheduling 후, bufferization 후에 삽입 |
| LICM | **동작** | LinalgToLoops 후 loop invariant code motion |
| VectorToLLVM | **동작** | pipeline에 등록됨 (vectorization 비활��이므로 현재 no-op) |

## Baseline vs Optimized Latency

Gawee p50 latency (ms), ORT median for reference. Date: 2026-05-03.

| model | baseline (ms) | optimized (ms) | speedup | ORT (ms) |
|-------|--------------|----------------|---------|----------|
| resnet18 | 6575.3 | 6814.2 | 0.96x | 17.3 |
| bert_tiny | AOT fail | 354.0 | n/a | 0.6 |
| tinyllama_15m | 100.4 | 107.6 | 0.93x | 1.7 |

**Analysis:**
- Optimized가 baseline보다 약간 느린 이유: tiling이 loop overhead를 추가하지만,
  fusion/vectorization이 아직 그 overhead를 상쇄하지 못함
- bert_tiny baseline: LLVM IR이 tiling 없이 너��� 커서 llc parse error
- ORT 대비 ~300x 느림: scalar loop 기반이라 SIMD/vectorization 없음

## 현재 적용된 최적화

- **tiling**: conv (N=1,C=8,H=8,W=8), matmul (M=32,N=32,K=16)
- **tile loop interchange**: conv (N,H,W,C_out) 순서로 spatial locality 개선
- **loop peeling**: tail iteration 분리 (vectorization 준비)
- **elementwise fusion**: generic op chains 합침 (중간 텐서 할당 제거)
- **canonicalize + CSE**: tiling/scheduling/bufferization 후 redundant op 정리
- **LICM**: loop lowering 후 invariant 연산 hoist

## 미구현 / 비활��� 최적화

### 1. Vectorization (LinalgVectorization.cpp) — 비활성
- **코드 작성 완료**, `linalg::vectorize()` 호출부 존재
- **현재 비활성**: MLIR의 `VectorizationState::getOrCreateMaskFor`에서 segfault
  - broadcast indexing map�� 있는 generic op에서 발생
  - identity map만 허용해도 crash (fusion 후 생성된 op 패턴 문제)
- **해결 방향**: MLIR 버전 업그레이드 또는 vectorize 가능 op subset을 더 좁게 선별
- **영향**: 가장 큰 성능 개선 요인 — SIMD 없이는 ORT 대비 100x+ 느림

### 2. Tile-and-Fuse
- 현재 tiling(LinalgTransform)과 fusion(LinalgFusion)이 별도 pass
- 정석: `scf::tileConsumerAndFuseProducersUsingSCF()`로 동시에 수행
- **효과**: tiling된 consumer 안에 producer를 합쳐서 중간 full-size 텐서 제거
- **복잡도**: LinalgTransform 재구성 필요

### 3. Buffer Deallocation
- **API:** `bufferization::createOwnershipBasedBufferDeallocationPass()`
- **효과**: 불필요한 memref alloc 해제
- **영향**: 메모리 사용량만, 성능에는 미미
