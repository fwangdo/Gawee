# ONNX Rewrite Report

## Scope

이 문서는 `front/onnx_rewrite` rewrite pipeline의 현재 성능/정확도 상태를 정리한다.

측정 대상 모델은 현재 priority 3종이다.

- `resnet18`
- `distilbert_base_uncased`
- `bert_tiny`

측정은 모두 같은 조건으로 수행했다.

- rewrite pipeline: `ConstantFolding -> EliminateId -> RewriteBN -> RewritePow -> RewriteGather -> RewriteGemm -> RewriteMatmul -> Cleanup`
- runtime: ONNX Runtime CPU
- latency setting: `warmup=10`, `repeat=40`
- throughput 기준: batch 1 기준 `1000 / median_ms` 를 `samples/s`로 계산
- correctness 기준: `front.onnx_rewrite.runtime.validation.compare_models`
- correctness gate: `max_abs_diff <= 1e-4`

원본 수치 JSON은 [artifacts/front_rewrite_bench/benchmark_results.json](/Users/hdy/code/portfolio/Gawee/artifacts/front_rewrite_bench/benchmark_results.json)에 저장했다.

## Summary

현재 rewrite pipeline은 priority 3개 모델 모두에 대해 `unsupported op = 0`을 달성한다.

다만 latency/throughput 관점에서는 NLP 모델에서 graph blow-up이 매우 크고, 그 결과 성능이 크게 악화된다.

또한 `distilbert_base_uncased`는 현재 correctness gate `1e-4`를 약간 초과한다.

요약하면:

- **coverage:** 목표 달성 (`unsupported op only`)
- **vision 성능:** 거의 유지
- **NLP 성능:** 현재 매우 나쁨
- **correctness:** `resnet18`, `bert_tiny`는 통과, `distilbert_base_uncased`는 경미한 수치 drift로 gate 실패

## Per-Model Results

| model | before nodes | after nodes | unsupported before | unsupported after | max abs diff | correctness | before median ms | after median ms | speedup | before throughput (samples/s) | after throughput (samples/s) |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 49 | 53 | `Gemm=1` | `{}` | `1.5259e-05` | pass | 47.169 | 47.566 | 0.992x | 21.200 | 21.024 |
| `distilbert_base_uncased` | 388 | 3862 | `Gather=12, MatMul=50, Pow=14` | `{}` | `1.7977e-04` | fail (`tol=1e-4`) | 5.690 | 63.145 | 0.090x | 175.739 | 15.837 |
| `bert_tiny` | 186 | 3559 | `Gather=23, Gemm=2, MatMul=16, Pow=5` | `{}` | `4.7684e-07` | pass | 0.065 | 4.941 | 0.013x | 15404.404 | 202.402 |

## Model Notes

### `resnet18`

- unsupported op는 `Gemm` 1개뿐이었고 rewrite 후 모두 제거된다.
- node 수 증가는 `49 -> 53`으로 작다.
- correctness는 통과한다.
- latency는 사실상 동일하다.

현재 상태에서 `resnet18`는 rewrite cost가 크지 않고, rewrite system의 baseline sanity check로 적절하다.

### `distilbert_base_uncased`

- rewrite 전 unsupported op는 `Gather`, `MatMul`, `Pow`가 섞여 있다.
- rewrite 후 unsupported op는 0이다.
- 그러나 node 수가 `388 -> 3862`로 약 10배 증가한다.
- latency는 `5.690ms -> 63.145ms`로 크게 악화된다.
- throughput도 `175.739 -> 15.837 samples/s`로 크게 감소한다.

correctness는 현재 gate를 통과하지 못했다.

- current `max_abs_diff`: `1.7976760864257812e-04`
- current tolerance: `1e-4`
- worst case: `dynamic_3`

다만 이 수치는 semantic 붕괴라기보다는 **경미한 수치 drift**에 가깝다.

추가 확인 결과:

- worst-case `mean_abs`: 약 `5.41e-05`
- worst-case `RMS`: 약 `6.25e-05`

즉 출력 전체가 크게 틀어진 것이 아니라, 일부 위치의 최대 오차가 gate를 조금 넘는 상황이다.

### `bert_tiny`

- rewrite 전 unsupported op는 `Gather`, `Gemm`, `MatMul`, `Pow`
- rewrite 후 unsupported op는 0
- correctness는 충분히 통과 (`4.7684e-07`)
- 하지만 node 수가 `186 -> 3559`로 크게 증가
- latency와 throughput은 매우 크게 악화

현재 `bert_tiny`는 legality는 해결했지만, 성능 관점에서는 실용적이지 않다.

## DistilBERT Correctness Detail

`distilbert_base_uncased`의 케이스별 `max_abs_diff`는 아래 범위다.

- `baseline`: `1.3113021850585938e-04`
- `dynamic_2`: `1.3685226440429688e-04`
- `dynamic_3`: `1.7976760864257812e-04`
- `mask_random`: `1.4400482177734375e-04`
- `mask_prefix`: `1.4972686767578125e-04`
- `mask_checker`: `1.3184547424316406e-04`
- `edge_indices`: `5.2928924560546875e-05`
- `low_band_vocab`: `1.367330551147461e-04`

이 값만 보면 현재 distilbert rewrite는:

- 명백한 semantic failure라기보다는
- decomposition rewrite가 누적되며 생긴 numerical drift로 보는 편이 타당하다.

따라서 다음 판단 후보는 두 가지다.

1. `distilbert`에 대해 pass-family ablation을 해서 어떤 rewrite가 drift를 키우는지 찾는다.
2. decomposition-heavy NLP 모델에 대해 correctness gate를 `1e-4`보다 약간 완화할지 검토한다.

## Interpretation

현재 rewrite system의 성격은 분명하다.

### 잘 되는 점

- priority 3개 모델 모두 `unsupported op only` 달성
- `Pow` rewrite 포함
- `Conv/ConvTranspose + BN` fusion path 구현 및 개별 검증 완료
- `resnet18`, `bert_tiny` correctness 통과

### 아직 부족한 점

- NLP 모델에서 `Gather` / `MatMul` rewrite가 graph를 지나치게 크게 만든다.
- 그 결과 latency와 throughput이 심각하게 악화된다.
- `distilbert`는 현재 tolerance 기준에서 correctness fail이다.

즉 현재 pipeline은 **legality-first rewrite system**으로는 의미가 있지만, **performance-preserving rewrite system**으로는 아직 미완성이다.

## Current Conclusion

현재 상태를 한 줄로 요약하면 다음과 같다.

> priority 3개 모델 모두 supported-op-only graph로 내리는 데는 성공했지만, NLP 모델에서는 graph expansion 비용이 너무 커서 runtime 성능이 크게 악화되며, distilbert는 현재 correctness gate도 약간 넘는다.

다음 우선순위는 명확하다.

1. `distilbert_base_uncased` numerical drift 원인 분해
2. `Gather` / `MatMul` rewrite의 graph blow-up 완화
3. NLP rewrite path의 latency/throughput 개선
