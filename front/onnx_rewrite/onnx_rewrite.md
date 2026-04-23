# TODO: Gawee ONNX Rewrite

이 문서는 [todo/onnx_rewrite.md](/Users/hdy/code/portfolio/dl-base/Gawee/todo/onnx_rewrite.md)를
frontend 작업 문맥으로 복사해 둔 것이다.
목표는 ONNX graph rewrite를 `Gawee` frontend의 명확한 ownership 영역으로 분리하는 것이다.

## What "Done" Means

이 프로젝트는 아래 조건을 모두 만족해야 완료로 본다.

- 공개 NLP 모델 5개와 vision 모델 2개가 준비되어 있다.
- dynamic shape를 실제로 쓰는 모델 2개에 대해 `torch -> ONNX` 변환이 재현 가능하다.
- NLP 5개는 최종적으로 supported op set 안으로 내려간다.
- lowering 전후 correctness 비교가 된다.
- rewrite 전후 latency 비교가 된다.
- 모델별 unsupported op before/after 표가 있다.
- README와 report만 읽어도 문제 정의, 접근 방식, 한계가 선명하다.

## Supported Op Set

이 프로젝트의 목표 supported op set은 아래다.

- arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
- tensor / layout: `Cast`, `Concat`, `Expand`, `Reshape`, `Shape`, `Squeeze`, `Transpose`, `Unsqueeze`
- reduction / normalization: `ReduceMean`, `ReduceSum`, `Softmax(last-axis only)`, `Sqrt`
- activation: `Erf`, `GELU`, `HardSigmoid`, `HardSwish`, `LeakyRelu`, `Relu`, `Sigmoid`, `Tanh`
- comparison / selection: `Equal`, `Where`
- spatial: `AveragePool`, `Conv`, `GlobalAveragePool`, `MaxPool`, `Pad`

Constant와 initializer는 계산 op로 세지 않는다.

## Benchmark Models

현재 frontend benchmark scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

### NLP: must support unsupported -> supported lowering

- `prajjwal1/bert-tiny`
- `distilbert-base-uncased`

### Vision: must show rewrite breadth and profiling ability

- `resnet18`

## Near-Term Focus

초기 graph rewrite 구현과 검증은 아래 3개 모델에 우선 집중한다.

- `resnet18`
- `distilbert-base-uncased`
- `prajjwal1/bert-tiny`

## Immediate Order of Work

이 프로젝트는 아래 순서로 진행한다.

1. 모델 확보와 baseline 실행
2. unsupported op 카운터 작성
3. validator 작성
4. LayerNorm / GELU lowering 구현
5. mask / Expand / shape cleanup 구현
6. 5개 NLP 모델 전체 lowering 달성
7. latency benchmark와 report 정리
8. vision 보조 rewrite와 profiling 추가

## Phase 0. Scope Freeze

### A. 문제 정의 확정

- README의 문제 정의를 한 문장으로 다시 읽고 더 줄일 필요가 없는지 확인
- README에 `이 프로젝트는 transformer accelerator primitive lowering 문제`라는 문장을 유지

### B. export 가정 확정

- batch size를 `1`로 고정
- 기본 sequence length를 `128`로 고정
- 추가 검증 sequence length를 `32`, `64`로 확정
- encoder-only, inference-only, no KV cache를 V1 가정으로 확정
- vision 모델의 기본 input shape를 문서화

### C. 모델 확보 전략 확정

- NLP 5개 모델의 원본 출처와 export 경로 정리
- vision 2개 모델의 원본 출처 정리
- dynamic shape를 실제로 사용하는 모델 2개 선정
- 각 모델에서 dynamic axis를 어디까지 열어둘지 정책 확정
- `torch.export` / `torch.onnx.export` 중 어떤 경로가 더 안정적인지 비교
- 현실적인 export 실패 원인(shape constraint, control flow, unsupported op, external data) 기록
- 모델 파일 저장 위치 규칙 정하기
- 모델명과 파일명 naming convention 정하기

## Phase 1. Environment and Baseline

### A. 환경

- `requirements.txt` 실제 버전 고정
- 로컬 `.venv` 생성 절차 README에 추가
- Apple Silicon M1에서 설치 가능한 패키지 조합 확인

### B. baseline runner

- `src/benchmark.py`가 실제 ONNX Runtime 세션을 열도록 구현
- session options에서 thread 수를 고정할 수 있게 구현
- NLP 모델용 dummy input 생성 함수 추가
- vision 모델용 dummy input 생성 함수 추가
- warmup / repeat 후 median, p95를 계산하도록 구현

### C. baseline metadata

- 모델별 node count 계산기 작성
- 모델별 op histogram 계산기 작성
- 모델별 file size 기록기 작성
- baseline 결과를 markdown / json 둘 다로 남기기

### D. baseline 실행

- NLP 5개 baseline 한 번씩 실행
- vision 2개 baseline 한 번씩 실행
- dynamic shape 모델 2개를 길이/배치 변화 케이스로 실제 export
- export된 dynamic model이 서로 다른 입력 shape에서 ORT로 로드/실행되는지 확인
- M1에서 모두 실제로 돈다는 것 확인
- baseline 실패 모델이 있으면 export 정책 수정

## Phase 2. Unsupported Op Audit

### A. op counter

- 모델 하나를 읽어 op 타입별 count를 출력하는 유틸 작성
- supported set 밖의 op만 따로 출력하는 기능 추가
- 모델별 unsupported summary를 markdown으로 저장

### B. lowering target map

- `LayerNormalization -> ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div/Mul/Add` 경로 정리
- `SkipLayerNormalization -> Add + LayerNormalization lowering` 경로 정리
- `GELU/FastGELU/BiasGELU -> tanh-based path` 정리
- `Where-based mask -> additive mask` 정리
- `Expand -> reshape/broadcast-friendly path` 정리
- generic `Gather -> axis=0 only form` 가능 조건 정리

### C. 현실성 검증

- 5개 NLP 모델 각각에서 unsupported op가 무엇인지 표로 확정
- unsupported op가 현재 rewrite 계획으로 커버되는지 체크
- 커버되지 않는 op가 있으면 export 방식 또는 supported set 재조정
- dynamic shape export에서 생긴 unsupported / brittle 패턴이 rewrite scope에 들어오는지 확인

## Phase 3. Validation Infrastructure

### A. input generation

- `input_ids` 생성기 작성
- `attention_mask` 생성기 작성
- optional `token_type_ids` 생성기 작성
- sequence length 32/64/128 case 생성
- vision용 random image tensor 생성기 작성

### B. correctness compare

- `src/validate.py`에 원본/변환 모델 동시 실행 구현
- deterministic multi-case 입력 생성 적용
- output count mismatch 즉시 실패 처리
- max abs error 하나만 correctness metric으로 사용
- global pass 기준 `max_abs_diff <= 1e-4` 적용
- worst-case case 이름 기록
- 모델별 semantic-preservation gate 문서화
- pass family별 기대 tolerance 문서화
- decomposition이 없는 pass는 더 엄격한 tolerance를 적용
- decomposition pass는 약간 넓은 tolerance를 적용하되 failure 시 원인 기록

### C. edge cases

- all-ones attention mask
- prefix mask
- sparse mask
- token_type_ids 없는 경우
- token_type_ids 있는 경우
- NLP와 vision 모두 deterministic seed 고정
- 가능하면 대표 중간 tensor를 output으로 추가해 subgraph-level compare 수행

### D. semantic preservation reporting

- output tensor별 max abs 표 생성
- case별 max abs 요약 생성
- pass family별 gate 통과 여부 표시
- 실패 시 어떤 output에서 왜 실패했는지 남기기
- ``rewrite correctness''와 ``task accuracy''를 혼동하지 않는 문구를 README/report에 유지
- dynamic shape 입력 2~3종에 대해 correctness가 유지되는지 별도 표로 기록

## Rule For V1 Scaffold

frontend scaffold의 현재 핵심 규칙은 아래 한 줄이다.

> 최종 ONNX graph는 supported op만 포함해야 한다.

따라서 현재 구현에서는 unsupported op가 하나라도 남아 있으면 rewrite pipeline은 실패해야 한다.
