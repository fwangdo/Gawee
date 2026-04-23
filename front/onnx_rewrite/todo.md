# ONNX Rewrite TODO

## Final Goal

현재 frontend scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

[ ] `resnet18`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료
[ ] `distilbert_base_uncased`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료
[ ] `bert_tiny`에 대해 `rewrite -> ORT 실행 -> correctness 검증 -> latency 비교` 완료

## Priority

현재 가장 우선적인 목표는 `resnet18`이다.

이 모델에서 먼저 아래를 끝내야 한다.

[ ] rewrite pass가 실제로 graph를 바꾸도록 완성
[ ] rewritten model이 ORT에서 실행 가능
[ ] correctness check 통과
[ ] latency 비교 report 생성

그 다음에 같은 흐름을 `distilbert_base_uncased`, `bert_tiny`로 확장한다.

## Rewrite Targets

현재 op set과 benchmark audit 기준으로, 구현해야 할 rewrite 우선순위는 아래와 같다.

### Phase 1: CNN first

[ ] `Gemm -> Conv` 또는 `Gemm -> MatMul + bias` 정리

`resnet18` 기준으로는 `Gemm`가 핵심 blocker다.

### Phase 2: Transformer core

[ ] `MatMul -> Conv` 또는 supported op 조합
[ ] `Gather` rewrite 또는 support 전략 재판단
[x] `Slice`는 supported op로 유지
[ ] `Pow` rewrite 또는 support 전략 재판단

### Phase 3: Secondary cleanup

[ ] `Constant` folding
[ ] `ConstantOfShape` folding
[ ] `Identity` elimination
[ ] `Not` rewrite
[ ] `CumSum` rewrite 또는 support 전략 재판단

## ConvertMatmul Completion

`passes/convert_matmul.py`에서 남아 있는 일:

[ ] generated nodes를 실제 graph에 삽입
[ ] 기존 `MatMul` node 제거
[ ] initializer / shape update 정리
[ ] 2D / 3D / 4D 경로별 ORT 실행 검증
[ ] dynamic path correctness 검증
[ ] 실패 케이스를 `log`에 남기고 pass 전체는 계속 진행

## Validation / Eval

rewrite 후 평가는 아래 기준으로 수행한다.

[ ] runtime: ONNX Runtime
[ ] correctness: original vs rewritten output 비교
[ ] latency: original vs rewritten latency 비교

현재는 generic synthetic input 기반이므로, 모델군별 입력 정책도 필요하다.

[ ] CNN: image-shaped float input
[ ] NLP: token / mask / segment input policy

## Done Criteria

`resnet18`가 아래를 만족하면 1차 목표 달성으로 본다.

[ ] rewrite 성공
[ ] rewritten ONNX가 ORT에서 실행됨
[ ] correctness metric 통과
[ ] latency report 생성됨

최종적으로는 3개 모델 전부가 위 조건을 만족해야 한다.
