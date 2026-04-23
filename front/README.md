# Frontend

이 디렉토리는 `Gawee`의 frontend 전용 문서와 ONNX graph rewrite scaffold를 둔다.

현재 frontend의 핵심 목표는 단순하다.

> 입력 ONNX graph를 읽고, 최종 결과가 `supported op only` 계약을 만족하게 만든다.

즉 V1 frontend는 아래 계약을 가장 먼저 강제한다.

- 입력은 `.onnx` 파일이다.
- graph rewrite는 legality보다 먼저 `지원 op 집합 안으로 내리는 것`에 집중한다.
- 최종 산출물은 `supported op`만 포함해야 한다.
- unsupported op가 남아 있으면 rewrite는 실패로 간주한다.

## Current Priority Models

- `resnet18`
- `distilbert_base_uncased`
- `bert_tiny`

즉 현재 근시일 조합은 `vision 1개 + NLP 2개`이다.

Frontend의 현재 benchmark scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

## Supported Op Contract

현재 frontend rewrite의 목표 supported op set은 아래다.

- `Add`
- `AveragePool`
- `Cast`
- `Concat`
- `Conv`
- `Sub`
- `Div`
- `Equal`
- `Erf`
- `Expand`
- `Gelu`
- `GlobalAveragePool`
- `HardSigmoid`
- `HardSwish`
- `LeakyRelu`
- `Max`
- `MaxPool`
- `Min`
- `Mul`
- `Pad`
- `Relu`
- `Reshape`
- `ReduceMean`
- `ReduceSum`
- `Shape`
- `Sigmoid`
- `Softmax`
- `Sqrt`
- `Tanh`
- `Squeeze`
- `Transpose`
- `Unsqueeze`
- `Where`

세부 제약은 문서와 코드에서 같이 관리한다.

- `Softmax(last-axis only)`

## Layout

- [onnx_rewrite.md](/Users/hdy/code/portfolio/dl-base/Gawee/front/onnx_rewrite.md)
  `todo/onnx_rewrite.md`를 frontend 관점에서 복사해 둔 작업 문서
- [onnx_rewrite/run_onnx_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/run_onnx_rewrite.py)
  rewrite / audit 전용 CLI 진입점
- [onnx_rewrite/eval_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/eval_rewrite.py)
  rewrite 결과의 runtime diff / latency 평가 CLI
- [onnx_rewrite/core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py)
  ONNX -> rewrite -> ONNX 저장을 담당하는 핵심 실행체
- [onnx_rewrite/checker/op_checker.py](/Users/hdy/code/portfolio/dl-base/Gawee/front/onnx_rewrite/checker/op_checker.py)
  supported-op-only gate
- [onnx_rewrite/passes/passer.py](/Users/hdy/code/portfolio/dl-base/Gawee/front/onnx_rewrite/passes/passer.py)
  pass 실행 순서 관리
- [onnx_rewrite/passes/cleanup.py](/Users/hdy/code/portfolio/dl-base/Gawee/front/onnx_rewrite/passes/cleanup.py)
  마지막 validation slot
- [onnx_rewrite/analysis/audit.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/analysis/audit.py)
  ONNX op histogram / unsupported op audit
- [onnx_rewrite/specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py)
  지원 op 집합과 우선 모델 경로

## Current Scaffold Scope

현재 scaffold는 의도적으로 매우 얇다.

- ONNX load / save
- supported / unsupported op audit
- checker / passes / utils / optimizer 구조
- rewrite CLI와 분리된 ONNX Runtime 기반 before/after latency 측정
- rewrite CLI와 분리된 ONNX Runtime 기반 before/after output diff 측정
- after-check에서 unsupported op가 남으면 즉시 실패

아직 실제 decomposition rewrite는 거의 없다.
V1의 목적은 `rewrite를 붙일 수 있는 최소 골격`과 `supported-op-only gate`를 먼저 세우는 것이다.
