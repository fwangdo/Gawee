# ONNX Rewrite TODO

## Final Goal

현재 frontend scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

[ ] `resnet18`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료
[ ] `distilbert_base_uncased`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료
[ ] `bert_tiny`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료

## Priority

현재 가장 우선적인 목표는 `resnet18`과 `bert_tiny`의 rewrite completion이다.

이 모델에서 먼저 아래를 끝내야 한다.

[x] rewrite pass가 실제로 graph를 바꾸도록 완성
[ ] rewritten model이 ORT에서 실행 가능
[ ] correctness check 통과
[ ] latency 비교 report 생성

그 다음에 같은 흐름을 `distilbert_base_uncased`, `bert_tiny`로 확장한다.

## Rewrite Targets

현재 op set과 benchmark audit 기준으로, 구현해야 할 rewrite 우선순위는 아래와 같다.

### Phase 1: CNN first

[x] `Gemm -> Conv` 또는 `Gemm -> MatMul + bias` 정리

`resnet18` 기준으로는 `Gemm`가 핵심 blocker다.

### Phase 2: Transformer core

[x] `MatMul -> Conv` 또는 supported op 조합
[x] `Gather` rewrite 또는 support 전략 재판단
[x] `Slice`는 supported op로 유지
[ ] `Pow` rewrite 또는 support 전략 재판단

### Phase 3: Secondary cleanup

[ ] `Constant` folding
[ ] `ConstantOfShape` folding
[ ] `Identity` elimination
[ ] `Not` rewrite
[ ] `CumSum` rewrite 또는 support 전략 재판단

## RewriteMatmul Completion

`passes/rewrite_matmul.py`에서 남아 있는 일:

[x] generated nodes를 실제 graph에 삽입
[x] 기존 `MatMul` node 제거
[ ] initializer / shape update 정리
[ ] 2D / 3D / 4D 경로별 ORT 실행 검증
[ ] dynamic path correctness 검증
[x] 실패 케이스를 `log`에 남기고 pass 전체는 계속 진행

## Current ONNX Understanding Notes

현재 이해 수준은 "ONNX graph를 value-name 기반 DAG로 읽는 감각은 잡혀가고 있으나, op별 broadcasting / shape propagation / initializer와 runtime tensor의 구분은 계속 명시적 설명이 필요한 단계"로 본다.

특히 아래는 이해가 빠르게 개선된 지점이다.

[x] `graph.input`과 `initializer`가 둘 다 node input으로 참조 가능한 tensor value라는 점 이해
[x] node name과 tensor output name은 별도 namespace에 가깝고, 이름을 분리해야 디버깅이 쉬워진다는 점 이해
[x] `Unsqueeze`의 axes 입력은 스칼라가 아니라 int64 tensor/list 형태라는 점 이해
[x] `Gather(data, indices)`에서 `indices`는 보통 token 하나가 아니라 `[B, S]` 같은 sequence tensor일 수 있다는 점 이해
[x] `Equal(indices, scalar_vocab_id)`는 scalar를 indices shape로 broadcast해서 mask를 만든다는 점 이해
[x] `Unsqueeze(-1)`는 mask를 embedding row와 곱하기 위한 broadcast 축을 추가한다는 점 이해
[x] chunked gather는 `[B, S, C, H]` 중간 표현을 만들고 chunk axis를 줄여 `[B, S, H]`로 복원한다는 점 이해
[x] `ReduceMean * C`보다 `ReduceSum`이 직접적이고 현재 supported op set에 더 맞는 방식이라는 점 이해
[x] dynamic-data Gather는 `one-hot-like mask -> MatMul -> Reshape`로 표현할 수 있고, 뒤에서 `RewriteMatmul`이 다시 lowering할 수 있다는 점 이해

앞으로 설명할 때는 아래 관점으로 설명하는 것이 좋다.

[ ] ONNX node 설명은 항상 `input tensor names -> output tensor names`로 먼저 풀기
[ ] 각 rewrite는 중간 tensor shape를 단계별로 써서 설명하기
[ ] broadcasting이 일어나는 정확한 op와 shape pair를 반드시 명시하기
[ ] initializer 기반 static rewrite와 runtime tensor 기반 dynamic rewrite를 매번 구분하기
[ ] `axis`가 의미하는 차원을 concrete example로 먼저 고정하고 일반화하기
[ ] `Shape`, `Gather`, `Slice`, `Reshape` 같은 shape-manipulation op는 값 계산과 shape 계산을 분리해 설명하기
[ ] `MatMul -> Conv` lowering은 layout 변환과 수학적 동치성을 따로 설명하기
[ ] pass 순서가 왜 중요한지, 특히 `Gather -> MatMul -> RewriteMatmul` 같은 chained rewrite는 pipeline 관점으로 설명하기
[ ] supported op에 넣는 결정과 rewrite로 제거하는 결정의 차이를 "현실성 / graph blow-up / 구현 난이도" 축으로 설명하기
[ ] 검증 설명은 먼저 structural check, 그다음 ORT execution, 마지막 correctness metric 순서로 설명하기

## Validation / Eval

rewrite 후 평가는 아래 기준으로 수행한다.

[ ] runtime: ONNX Runtime
[ ] correctness: original vs rewritten output 비교
[ ] latency: original vs rewritten latency 비교

현재는 generic synthetic input 기반이므로, 모델군별 입력 정책도 필요하다.

[ ] CNN: image-shaped float input
[ ] NLP: token / mask / segment input policy

## Additional Benchmark Expansion

`opset >= 13` 추가 benchmark를 통해 아래 후속 과제가 확인됐다.

[ ] `distilbert_base_uncased_mnli`를 추가 benchmark 후보로 등록하고 기존 `distilbert`와 함께 NLP drift / latency 경향 비교
[ ] `vit_tiny_patch16_224`를 추가 benchmark 후보로 등록하고 ViT 계열의 rewrite correctness / latency 경향 확인
[ ] `vit_base_patch16_224`를 추가 benchmark 후보로 등록하고 model size 증가 시 graph blow-up 영향 비교
[ ] vision transformer 입력용 `pixel_values` 생성 규칙을 validation harness에 추가
[ ] vision 입력 기본 shape를 `(N, 3, H, W)`로 다루도록 `runtime/validation.py` 보강
[ ] `dinov3_convnext_tiny`를 modern ONNX stress benchmark로 유지
[ ] `LayerNormalization` rewrite 구현
[ ] `CastLike` rewrite 또는 support 전략 결정
[ ] `Loop` 지원 여부 결정 또는 benchmark scope에서 제외할 기준 정리
[ ] `SequenceEmpty` 포함 sequence 계열 op 지원 여부 결정
[ ] modern ONNX benchmark에 대해 "현재 rewrite 범위 밖"과 "우선 구현 대상"을 구분한 문서화

## Done Criteria

`resnet18`가 아래를 만족하면 1차 목표 달성으로 본다.

[ ] rewrite 성공
[ ] rewritten ONNX가 ORT에서 실행됨
[ ] correctness metric 통과
[ ] latency report 생성됨

최종적으로는 3개 모델 전부가 위 조건을 만족해야 한다.
