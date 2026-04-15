# TODO: Gawee ONNX Rewrite

이 문서는 기존 `projects/onnx_graph_rewrite/TODO.md`를
`Gawee` 중심 포트폴리오 문맥으로 옮겨 놓은 것이다.
목표는 ONNX graph rewrite를 별도 작은 프로젝트로 남기기보다,
Gawee의 graph/IR/optimizer narrative 안으로 흡수하는 것이다.

## What "Done" Means

이 프로젝트는 아래 조건을 모두 만족해야 완료로 본다.

- 공개 NLP 모델 5개와 vision 모델 2개가 준비되어 있다.
- dynamic shape를 실제로 쓰는 모델 2개에 대해 `torch -> ONNX` 변환이 재현 가능하다.
- NLP 5개는 최종적으로 supported primitive op set 안으로 내려간다.
- lowering 전후 correctness 비교가 된다.
- rewrite 전후 latency 비교가 된다.
- 모델별 unsupported op before/after 표가 있다.
- README와 report만 읽어도 문제 정의, 접근 방식, 한계가 선명하다.

## Supported Primitive Op Set

이 프로젝트의 목표 op set은 아래다.

- [ ] `Add`
- [ ] `Sub`
- [ ] `Mul`
- [ ] `Div`
- [ ] `MatMul`
- [ ] `Reshape`
- [ ] `Transpose`
- [ ] `ReduceMean`
- [ ] `Sqrt`
- [ ] `Tanh`
- [ ] `Softmax(last-axis only)`
- [ ] `Gather(axis=0 only)`
- [ ] `Concat`
- [ ] `Slice`
- [ ] `Unsqueeze`
- [ ] `Squeeze`
- [ ] `Cast`

Constant와 initializer는 계산 op로 세지 않는다.

## Benchmark Models

### NLP: must support unsupported -> supported lowering

- [ ] `prajjwal1/bert-tiny`
- [ ] `distilbert-base-uncased`
- [ ] `microsoft/MiniLM-L12-H384-uncased`
- [ ] `google/mobilebert-uncased`
- [ ] `distilroberta-base`

### Vision: must show rewrite breadth and profiling ability

- [ ] `resnet18`
- [ ] `mobilenetv3-small`

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

- [ ] README의 문제 정의를 한 문장으로 다시 읽고 더 줄일 필요가 없는지 확인
- [ ] 특정 기업 과제와 겹치는 표현이 없는지 최종 점검
- [ ] README에 `이 프로젝트는 transformer accelerator primitive lowering 문제`라는 문장을 유지

### B. export 가정 확정

- [ ] batch size를 `1`로 고정
- [ ] 기본 sequence length를 `128`로 고정
- [ ] 추가 검증 sequence length를 `32`, `64`로 확정
- [ ] encoder-only, inference-only, no KV cache를 V1 가정으로 확정
- [ ] vision 모델의 기본 input shape를 문서화

### C. 모델 확보 전략 확정

- [ ] NLP 5개 모델의 원본 출처와 export 경로 정리
- [ ] vision 2개 모델의 원본 출처 정리
- [ ] dynamic shape를 실제로 사용하는 모델 2개 선정
- [ ] 각 모델에서 dynamic axis를 어디까지 열어둘지 정책 확정
- [ ] `torch.export` / `torch.onnx.export` 중 어떤 경로가 더 안정적인지 비교
- [ ] 현실적인 export 실패 원인(shape constraint, control flow, unsupported op, external data) 기록
- [ ] 모델 파일 저장 위치 규칙 정하기
- [ ] 모델명과 파일명 naming convention 정하기

## Phase 1. Environment and Baseline

### A. 환경

- [ ] `requirements.txt` 실제 버전 고정
- [ ] 로컬 `.venv` 생성 절차 README에 추가
- [ ] Apple Silicon M1에서 설치 가능한 패키지 조합 확인

### B. baseline runner

- [ ] `src/benchmark.py`가 실제 ONNX Runtime 세션을 열도록 구현
- [ ] session options에서 thread 수를 고정할 수 있게 구현
- [ ] NLP 모델용 dummy input 생성 함수 추가
- [ ] vision 모델용 dummy input 생성 함수 추가
- [ ] warmup / repeat 후 median, p95를 계산하도록 구현

### C. baseline metadata

- [ ] 모델별 node count 계산기 작성
- [ ] 모델별 op histogram 계산기 작성
- [ ] 모델별 file size 기록기 작성
- [ ] baseline 결과를 markdown / json 둘 다로 남기기

### D. baseline 실행

- [ ] NLP 5개 baseline 한 번씩 실행
- [ ] vision 2개 baseline 한 번씩 실행
- [ ] dynamic shape 모델 2개를 길이/배치 변화 케이스로 실제 export
- [ ] export된 dynamic model이 서로 다른 입력 shape에서 ORT로 로드/실행되는지 확인
- [ ] M1에서 모두 실제로 돈다는 것 확인
- [ ] baseline 실패 모델이 있으면 export 정책 수정

## Phase 2. Unsupported Op Audit

### A. op counter

- [ ] 모델 하나를 읽어 op 타입별 count를 출력하는 유틸 작성
- [ ] supported set 밖의 op만 따로 출력하는 기능 추가
- [ ] 모델별 unsupported summary를 markdown으로 저장

### B. lowering target map

- [ ] `LayerNormalization -> ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div/Mul/Add` 경로 정리
- [ ] `SkipLayerNormalization -> Add + LayerNormalization lowering` 경로 정리
- [ ] `GELU/FastGELU/BiasGELU -> tanh-based path` 정리
- [ ] `Where-based mask -> additive mask` 정리
- [ ] `Expand -> reshape/broadcast-friendly path` 정리
- [ ] generic `Gather -> axis=0 only form` 가능 조건 정리

### C. 현실성 검증

- [ ] 5개 NLP 모델 각각에서 unsupported op가 무엇인지 표로 확정
- [ ] unsupported op가 현재 rewrite 계획으로 커버되는지 체크
- [ ] 커버되지 않는 op가 있으면 export 방식 또는 supported set 재조정
- [ ] dynamic shape export에서 생긴 unsupported / brittle 패턴이 rewrite scope에 들어오는지 확인

## Phase 3. Validation Infrastructure

### A. input generation

- [ ] `input_ids` 생성기 작성
- [ ] `attention_mask` 생성기 작성
- [ ] optional `token_type_ids` 생성기 작성
- [ ] sequence length 32/64/128 case 생성
- [ ] vision용 random image tensor 생성기 작성

### B. correctness compare

- [ ] `src/validate.py`에 원본/변환 모델 동시 실행 구현
- [ ] output count mismatch 즉시 실패 처리
- [ ] max abs error 계산
- [ ] mean abs error 계산
- [ ] cosine similarity 계산
- [ ] SNR은 보조 지표로만 추가
- [ ] 모델별 semantic-preservation gate 문서화
- [ ] pass family별 기대 tolerance 문서화
- [ ] decomposition이 없는 pass는 더 엄격한 tolerance를 적용
- [ ] decomposition pass는 약간 넓은 tolerance를 적용하되 failure 시 원인 기록

### C. edge cases

- [ ] all-ones attention mask
- [ ] prefix mask
- [ ] sparse mask
- [ ] token_type_ids 없는 경우
- [ ] token_type_ids 있는 경우
- [ ] NLP와 vision 모두 deterministic seed 고정
- [ ] 가능하면 대표 중간 tensor를 output으로 추가해 subgraph-level compare 수행

### D. semantic preservation reporting

- [ ] output tensor별 max/mean/cosine/SNR 표 생성
- [ ] pass family별 gate 통과 여부 표시
- [ ] 실패 시 어떤 output에서 왜 실패했는지 남기기
- [ ] ``rewrite correctness''와 ``task accuracy''를 혼동하지 않는 문구를 README/report에 유지
- [ ] dynamic shape 입력 2~3종에 대해 correctness가 유지되는지 별도 표로 기록

## Phase 3.5. Dynamic Shape Export Challenges

### A. realistic export track

- [ ] dynamic shape 모델 2개를 선택해 `torch -> ONNX` export 재현
- [ ] batch 또는 sequence length가 달라질 때 같은 ONNX graph가 유지되는지 확인
- [ ] export 시 생기는 shape constraint warning을 수집
- [ ] exporter가 생성한 불필요한 shape op / guard / fallback 패턴을 기록
- [ ] 현실적인 failure case 1개 이상을 문서화하고 우회책을 정리

### B. acceptance criteria

- [ ] 모델별 export command를 README에 남기기
- [ ] 모델별 dynamic axes 정책을 표로 남기기
- [ ] 서로 다른 입력 shape 3종 이상에서 ORT 실행 성공 확인
- [ ] static export 대비 graph가 얼마나 복잡해졌는지 비교

## Phase 4. Core Graph Utilities

### A. graph access helpers

- [ ] model load/save
- [ ] initializer name -> tensor 조회
- [ ] node name -> producer/consumer 조회
- [ ] input/output name remap 유틸
- [ ] safe node replacement helper
- [ ] topological sort helper

### B. graph cleanup helpers

- [ ] unused initializer 제거
- [ ] dangling node 제거
- [ ] identity 제거 helper
- [ ] redundant transpose pair 제거 helper
- [ ] redundant squeeze/unsqueeze 제거 helper

### C. report helpers

- [ ] before/after node count diff
- [ ] before/after unsupported op diff
- [ ] pass log 구조 확정
- [ ] skip reason schema 확정

## Phase 5. Rewrite Implementation

### Pass 1. Cleanup

- [ ] Identity 제거 구현
- [ ] dead initializer 제거 구현
- [ ] dead node 제거 구현
- [ ] transpose pair elimination 구현
- [ ] squeeze/unsqueeze cleanup 구현
- [ ] cleanup unit tests 작성

### Pass 2. LayerNorm lowering

- [ ] LayerNorm의 입력/scale/bias shape 가정 정리
- [ ] mean 계산 축 확인
- [ ] `x - mean` 경로 생성
- [ ] variance 또는 squared mean 경로 생성
- [ ] `sqrt(var + eps)` 경로 생성
- [ ] normalize 후 gamma/beta 적용
- [ ] axis와 last-dim semantics 유지 검증
- [ ] small handcrafted ONNX graph로 unit test 작성

### Pass 3. SkipLayerNorm lowering

- [ ] SkipLayerNorm 패턴 매칭
- [ ] residual add 분리
- [ ] Add 결과를 LayerNorm lowering 경로로 연결
- [ ] output naming과 multiple outputs 처리
- [ ] unit test 작성

### Pass 4. GELU family lowering

- [ ] `GELU` 노드 처리
- [ ] `FastGELU` 노드 처리 여부 확인
- [ ] `BiasGELU`는 bias add + GELU로 분리할지 결정
- [ ] `Erf`, `Pow` 없이 tanh path만으로 구현
- [ ] approximation 공식과 numerical tolerance 문서화
- [ ] unit test 작성

### Pass 5. Mask rewrite

- [ ] Where-based attention mask 패턴 탐지
- [ ] boolean mask를 additive mask로 바꾸는 설계 확정
- [ ] softmax 직전 score tensor와 shape alignment 확인
- [ ] additive mask constant 값 정책 정하기
- [ ] edge case 테스트 작성

### Pass 6. Expand / shape cleanup

- [ ] Expand 사용 위치 분류
- [ ] 제거 가능한 Expand와 유지해야 할 Expand 구분
- [ ] reshape/unsqueeze/transpose 정리
- [ ] softmax가 last-axis에 오도록 필요 시 transpose 정리
- [ ] unit test 작성

### Pass 7. Generic Gather handling

- [ ] axis=0 embedding lookup이면 supported로 인정
- [ ] axis가 다른 Gather 패턴 조사
- [ ] reshape/transpose로 axis=0 equivalent form이 가능한 경우 구현
- [ ] 불가능한 경우 skip reason을 명확히 기록

## Phase 6. End-to-End Lowering

### NLP 5개 전체 적용

- [ ] bert-tiny에 전체 pipeline 적용
- [ ] distilbert에 전체 pipeline 적용
- [ ] MiniLM에 전체 pipeline 적용
- [ ] mobilebert에 전체 pipeline 적용
- [ ] distilroberta에 전체 pipeline 적용

### 모델별 완료 조건

- [ ] unsupported op count가 0인지 확인
- [ ] supported set 밖 op가 남아 있으면 리포트에 명시
- [ ] correctness 통과 여부 기록
- [ ] pipeline 중 어느 pass가 실제로 작동했는지 기록

## Phase 7. Benchmarking

### A. NLP

- [ ] sequence length 32에서 5개 모델 측정
- [ ] sequence length 64에서 5개 모델 측정
- [ ] sequence length 128에서 5개 모델 측정
- [ ] rewrite 전후 median latency 비교
- [ ] rewrite 전후 p95 latency 비교

### B. Vision

- [ ] resnet18 baseline 측정
- [ ] mobilenetv3-small baseline 측정
- [ ] cleanup/fusion pass 적용
- [ ] rewrite 전후 비교

### C. 해석

- [ ] latency가 좋아진 모델과 아닌 모델을 나누어 해석
- [ ] unsupported lowering은 됐지만 속도는 안 좋아진 사례 정리
- [ ] 왜 그런지 memory traffic / kernel launch / cache reuse 관점에서 설명

## Phase 8. Reporting and Packaging

### A. 모델별 report

- [ ] before/after op histogram
- [ ] before/after unsupported op summary
- [ ] before/after node count
- [ ] correctness 결과
- [ ] latency 결과
- [ ] 적용 pass log
- [ ] skip reason

### B. README polish

- [ ] 문제 정의를 더 짧고 강하게 다시 쓰기
- [ ] supported/unsupported op 표 정리
- [ ] 7개 벤치마크 모델 표 추가
- [ ] 왜 이 문제가 일반적인 현업 문제인지 설명
- [ ] 각 rewrite가 LaTeX 문서와 어떻게 연결되는지 짧게 설명

### C. Interview prep

- [ ] LayerNorm lowering을 shape level로 설명하는 1분 답변 작성
- [ ] GELU lowering을 numerical trade-off 포함해 설명하는 1분 답변 작성
- [ ] mask rewrite를 shape level로 설명하는 1분 답변 작성
- [ ] 왜 지원 op set을 이렇게 줄였는지 설명하는 30초 답변 작성
- [ ] vision 모델을 왜 같이 넣었는지 설명하는 30초 답변 작성
