# Phase 9: Pipeline Improvement Summary

## 이번에 변경한 내용

### 1. Vectorization 완성 (LinalgVectorization.cpp)

**개념**: `linalg::vectorize()`는 linalg op의 scalar loop body를 vector dialect op으로 변환한다. 예를 들어 `arith.addf`가 `vector.broadcast` + `arith.addf`(vector 버전)로 바뀜.

**변경 내역**:
1. `runOnOperation()`의 no-op body를 `vectorizeEligibleOps()` 호출로 교체
2. 필터를 `isIdentity()` → `isProjectedPermutation()` + 0-result map 허용으로 완화
3. `linalg::vectorize()` 결과 처리 버그 수정: `rewriter.replaceOp(op, result->replacements)`
4. pipeline에 `VectorToSCF`, `UBToLLVM` pass 추가

**핵심 이해 포인트**:
- `linalg::vectorize(rewriter, op)`는 `FailureOr<VectorizationResult>`를 반환
- `VectorizationResult.replacements`를 `rewriter.replaceOp()`에 전달해야 원래 op이 교체됨 — 이걸 안 하면 vectorize해도 원래 op이 그대로 남음
- `isProjectedPermutation()`: identity, broadcast(`(d0,d1,d2,d3) -> (d1)`), permutation을 모두 허용
- 0-result map `(d0,d1,d2) -> ()`: rank-0 tensor (scalar broadcast)에 해당, `isProjectedPermutation()`이 false를 반환하므로 별도 처리 필요
- `ub.poison`: vectorize가 `vector.transfer_read`의 padding으로 생성. UB dialect 등록 + `UBToLLVM` pass 필요
- `VectorToSCF`: `vector.transfer_read/write`를 SCF loop으로 lowering. bufferization 후, LinalgToLoops 전에 배치

**현재 한계**: maxDim=32 — LLVM backend가 큰 벡터(예: 1792bit)를 `Cannot select`로 거부. 실제 모델 텐서가 대부분 32 초과이므로 vectorize되는 op이 적음.

### 2. DecomposeAggregated 이동 (gawee-opt.cpp)

**개념**: `linalg.softmax` 같은 aggregated op은 내부적으로 reduce-max, subtract, exp, reduce-sum, divide의 시퀀스. 이걸 먼저 분해해야 각 primitive op을 개별적으로 tiling/fusion/vectorization할 수 있음.

**변경**: `gawee-to-loops`와 `gawee-to-llvm` 두 파이프라인 모두에서 DecomposeAggregated를 `GaweeToLinalg` 직후, `LinalgTransform` 전으로 이동.

**핵심 이해 포인트**:
- pass 순서가 최적화 품질을 결정한다
- decompose → tile → fuse → vectorize 순서가 자연스러움
- 이전에는 tile → fuse → vectorize → decompose 순서였음 → softmax가 opaque 단위로 남아서 최적화 기회 상실

### 3. Tile-and-Fuse (LinalgTransform.cpp) — conv에 대해 부분 구현

**개념**: `tileUsingSCF`는 consumer op만 tiling. `tileConsumerAndFuseProducersUsingSCF`는 consumer를 tiling하면서 동시에 producer를 tile loop 안으로 끌어온다.

**왜 중요한가**:
```
// tiling만 (이전):
%fill = linalg.fill(...)           // full-size tensor 할당
%conv = conv(%input, %weight, %fill)  // tiled loop 안에서 실행
// fill은 매번 full tensor를 초기화 — 낭비!

// tile-and-fuse (현재):
scf.for ... {
  %tiled_fill = linalg.fill(...)   // tile 크기만큼만 초기화
  %tiled_conv = conv(..., %tiled_fill)
}
// fill이 tile loop 안에서 tile 크기만큼만 실행
```

**구현 핵심**:
```cpp
scf::SCFTileAndFuseOptions options;
options.setTilingOptions(tilingOptions);
options.fusionControlFn =
    [](tensor::ExtractSliceOp, OpResult originalProducer,
       bool isDestinationOperand) -> std::optional<ControlFnResult> {
      // destination operand의 fill만 fuse
      if (!isDestinationOperand) return std::nullopt;
      if (!isa<linalg::FillOp>(originalProducer.getOwner()))
        return std::nullopt;
      return ControlFnResult{/*yieldProducerReplacement=*/false};
    };
```

**fusionControlFn 3개 파라미터의 의미**:
1. `candidateSliceOp`: tile loop 안에서 생성된 `tensor.extract_slice` — 어떤 slice를 fuse할지
2. `originalProducer`: fuse 대상인 producer op의 result
3. `isDestinationOperand`: 이 producer가 consumer의 `outs()` (destination) 인지 `ins()` (input) 인지

**yieldProducerReplacement의 의미**:
- `true`: fused producer의 결과를 loop의 yield에 추가 → loop 밖에서 원래 producer 대신 사용 가능
- `false`: fused producer는 loop 안에서만 사용되고, loop 밖의 원래 producer는 그대로 남음
- fill의 경우: conv의 `outs()`로만 사용되므로 yield 불필요 → `false`가 적절

**왜 input producer fusion이 위험한가**:
- conv의 input은 stride/padding 때문에 output tile과 다른 크기의 slice가 필요
- 예: output tile [1,8,8,8]이면 input slice는 [1,3,10,10] (kernel 3x3 때문)
- input producer가 다른 conv라면, 그 conv의 output shape과 필요한 input slice shape이 안 맞을 수 있음

## 핵심 API 정리

| API | 역할 | 헤더 |
|-----|------|------|
| `linalg::vectorize(rewriter, op)` | linalg op → vector dialect, `FailureOr<VectorizationResult>` 반환 | `Linalg/Transforms/Transforms.h` |
| `scf::tileUsingSCF(rewriter, op, options)` | consumer만 tiling | `SCF/Transforms/TileUsingInterface.h` |
| `scf::tileConsumerAndFuseProducersUsingSCF(rewriter, op, options)` | tile + producer fusion | 같은 헤더 |
| `SCFTileAndFuseOptions::fusionControlFn` | 어떤 producer를 fuse할지 제어 | 같은 헤더 |
| `SCFTileAndFuseResult.replacements` | `DenseMap<Value, Value>` — 원래 값→대체 값 매핑 | 같은 헤더 |

### 4. Elementwise Fusion 제어 (LinalgFusion.cpp) — 성능 regression 수정

**문제**: `populateElementwiseOpsFusionPatterns`에 `return true` (greedy fusion)을 쓰면 bert_tiny가 226ms → 345ms로 1.53x 느려짐.

**원인 분석 (ablation testing)**:
```
no-opts (decompose만):     226ms, 479 loads+stores
fusion-only (greedy):       345ms, 607 loads+stores (+27%)
```
greedy fusion이 multi-use producer를 각 consumer에 복제 → 중간 값 재계산 → 메모리 접근 증가.

**해결**:
```cpp
// Before (greedy — 모든 operand를 fuse):
linalg::ControlFusionFn controlFn = [](OpOperand *) { return true; };

// After (single-use만 fuse):
linalg::ControlFusionFn controlFn = [](OpOperand *operand) {
  return operand->get().hasOneUse();
};
```

**핵심 이해 포인트**:
- `ControlFusionFn`은 `OpOperand *`를 받아서 bool 반환 — 이 operand의 producer를 consumer에 fuse할지 결정
- `operand->get()`: producer op의 result Value
- `hasOneUse()`: 이 Value를 사용하는 op이 정확히 1개인지 (= 이 consumer뿐인지)
- multi-use producer를 fuse하면: 각 consumer에 producer body가 복제 → 전체 연산량 증가
- single-use producer만 fuse하면: 중간 tensor 할당만 제거, 연산량 동일

**ablation testing 방법론**:
- gawee-opt.cpp에 ablation pipeline 등록 (특정 pass만 활성/비활성)
- 각 pipeline의 AOT binary를 수동 빌드 + latency 측정
- loads/stores 수 비교로 memory traffic 차이 확인

### 5. AOT 파이프라인에 `opt -O2` 추가 (gawee_aot.cpp) — 3.8x 개선

**문제**: `mlir-translate --mlir-to-llvmir`로 생성한 LLVM IR을 `llc`로 바로 codegen하고 있었음.
`llc`에 `-O2`를 줘도 효과 없었음.

**핵심 구분 — `opt` vs `llc`**:
- **`opt`** = LLVM **middle-end** optimizer. LLVM IR → 최적화된 LLVM IR.
  - LoopVectorize: scalar loop → SIMD (ARM NEON `fmla v0.4s, v1.4s, v2.4s`)
  - SLPVectorize: straight-line code를 벡터화
  - GVN, LICM, inlining, dead store elimination 등
- **`llc`** = LLVM **backend** code generator. LLVM IR → 기계어.
  - instruction selection (IR → target-specific 명령어)
  - register allocation
  - instruction scheduling
  - `-O2`는 codegen 품질만 올림, middle-end 최적화는 수행 안 함

**왜 `llc -O2`만으로는 효과가 없었나**:
- MLIR이 생성하는 LLVM IR은 scalar loop의 나열
- `llc`는 이 scalar loop을 그대로 기계어로 변환 — vectorization 없음
- `opt -O2`가 LoopVectorize를 실행해야 scalar loop → SIMD 변환이 일어남

**변경**:
```cpp
// 이전:  llc -O2 만 사용
llcCmd << llc << " -O2 -filetype=obj " << llvmIr << " -o " << obj;

// 이후:  opt -O2 → llc -O2
optCmd << opt << " -O2 -S " << llvmIr << " -o " << optimizedIr;
llcCmd << llc << " -O2 -filetype=obj " << optimizedIr << " -o " << obj;
```

**결과**:
- resnet18: 6490ms → 1722ms (**3.8x**)
- bert_tiny: 265ms → 62ms (**4.3x**)
- tinyllama: 101ms → 62ms (**1.6x**)

### 6. Elementwise Fusion 비활성화 (LinalgFusion.cpp) — opt -O2와의 간섭

**발견**: `opt -O2` 추가 후, LinalgFusion이 bert_tiny를 63ms → 111ms로 **1.8x 악화**.

**원인**: `populateElementwiseOpsFusionPatterns`이 여러 elementwise op을 하나의 `linalg.generic`으로 합치면서 복잡한 affine indexing map을 생성. 이것이 LLVM LoopVectorize를 방해:
- 단순한 별개 루프: LLVM이 각각 독립적으로 벡터화 가능
- fused 복잡한 루프: LLVM이 loop body 내의 복잡한 인덱싱을 분석하지 못해 vectorization 포기

**ablation 결과**:
```
fusion ON  + opt -O2:  bert_tiny = 111ms
fusion OFF + opt -O2:  bert_tiny =  62ms  (baseline = 63ms)
```

**교훈**: MLIR-level fusion은 LLVM backend의 auto-vectorization과 상충할 수 있다. Fusion은 중간 메모리 할당을 제거하지만, loop body를 복잡하게 만들어 LLVM의 LoopVectorize가 포기하게 만들 수 있음. 해결 방향: MLIR 수준에서 fuse + vectorize를 함께 해서 LLVM에 의존하지 않는 것.

## 다시 구현하려면 알아야 할 것

1. MLIR pass pipeline에서 pass 순서의 의미 — 어떤 pass가 어떤 IR 형태를 기대하는지
2. `linalg::vectorize()`의 제약: static shape, indexing map 형태, 반환값 처리
3. `tileConsumerAndFuseProducersUsingSCF`의 fusion control function 설계
   - 어떤 producer를 fuse할지 (destination only? input도?)
   - `yieldProducerReplacement`의 true/false 결정 기준
4. vector lowering pipeline: vectorize → VectorToSCF → VectorToLLVM → UBToLLVM
5. `ub.poison` dialect의 존재 이유와 등록 방법
6. **elementwise fusion의 `ControlFusionFn` 설계 — greedy vs selective**
   - `hasOneUse()` 조건으로 multi-use producer 재계산 방지
   - ablation testing으로 pass별 성능 영향 격리
7. **`opt` vs `llc`의 역할 구분**
   - `opt`: middle-end optimization (LoopVectorize, SLPVectorize, GVN, LICM)
   - `llc`: backend codegen (instruction selection, register allocation)
   - MLIR → LLVM IR → (opt -O2) → (llc -O2) → object 파이프라인
8. **MLIR fusion과 LLVM auto-vectorization의 간섭**
   - fused loop body가 복잡하면 LLVM LoopVectorize가 포기
   - ablation으로 각 pass의 영향을 격리해서 확인하는 방법론
