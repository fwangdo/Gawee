# TODO: Gawee Quantization

이 문서는 기존 `projects/onnx_quantization/TODO.md`를
`Gawee` 중심 포트폴리오 문맥으로 옮겨 놓은 것이다.
목표는 quantization을 별도 포트폴리오로 남기기보다,
Gawee의 ONNX analysis / rewrite / validation 흐름과 연결되는 하위 축으로 관리하는 것이다.

## What "Done" Means

이 프로젝트는 아래 조건을 모두 만족해야 완료로 본다.

- 모델 3개 이상에 대해 quantization 결과가 있다.
- vision 최소 2개, NLP 최소 1개를 포함한다.
- FP32와 quantized 모델의 size / latency / numerical drift가 모두 기록된다.
- calibration 설정이 재현 가능하다.
- README만 읽어도 어떤 실험을 했고 무엇을 배웠는지 선명하다.

## Benchmark Models

### Vision

- [ ] `resnet18`
- [ ] `mobilenetv3-small`

### NLP

- [ ] `distilbert-base-uncased`

필요하면 `MiniLM`을 추가한다.

## Immediate Order of Work

1. 모델 확보와 baseline 실행
2. calibration 데이터 정책 확정
3. static PTQ 경로 구현
4. FP32 vs INT8 correctness 비교
5. FP32 vs INT8 latency 측정
6. per-channel / calibration ablation 추가
7. 보고서와 README 정리

## Phase 0. Scope Freeze

### A. 범위 확정

- [ ] V1을 `static PTQ 중심`으로 확정
- [ ] `dynamic quantization`은 NLP 비교용 보조 실험으로 둘지 결정
- [ ] task metric 대신 numerical drift를 기본 평가지표로 쓸지 확정
- [ ] provider를 우선 `CPUExecutionProvider`로 고정

### B. 모델 확정

- [ ] vision 2개 모델 출처와 확보 방식 정리
- [ ] NLP 1개 모델 출처와 export 방식 정리
- [ ] 모델 파일 저장 규칙 정하기
- [ ] 모델명과 출력 파일 naming convention 정하기

### C. calibration 정책 확정

- [ ] vision calibration sample 개수 확정
- [ ] NLP calibration sample 개수 확정
- [ ] 입력 생성 방식을 synthetic로 갈지 실제 샘플로 갈지 결정
- [ ] representative data requirements를 README에 적기

## Phase 1. Environment and Baseline

### A. 환경

- [ ] `requirements.txt` 실제 버전 고정
- [ ] 로컬 `.venv` 절차 문서화
- [ ] Apple Silicon M1에서 패키지 설치 확인

### B. baseline inference

- [ ] `src/benchmark.py`에 실제 ONNX Runtime 실행 구현
- [ ] vision 더미 입력 생성기 작성
- [ ] NLP 더미 입력 생성기 작성
- [ ] warmup / repeat / median / p95 계산 구현

### C. baseline metadata

- [ ] file size 기록기 작성
- [ ] node count 기록기 작성
- [ ] op histogram 기록기 작성
- [ ] baseline 결과 markdown/json 저장

### D. baseline 실행

- [ ] resnet18 baseline 측정
- [ ] mobilenetv3-small baseline 측정
- [ ] distilbert baseline 측정

## Phase 2. Calibration Pipeline

### A. vision calibration

- [ ] image-like calibration tensor 생성기 작성
- [ ] input normalization 정책 정하기
- [ ] sample count 8/32/128 실험 계획 세우기

### B. NLP calibration

- [ ] `input_ids` 생성기 작성
- [ ] `attention_mask` 생성기 작성
- [ ] sequence length 32/64/128 calibration 정책 정하기
- [ ] synthetic token distribution 정책 정하기

### C. 재현성

- [ ] calibration seed 고정
- [ ] calibration config를 json으로 저장
- [ ] calibration 재현 커맨드 README에 적기

## Phase 3. Quantization Implementation

### A. static PTQ

- [ ] `src/quantize.py`에 static PTQ 실제 구현
- [ ] quantized 모델 저장 구현
- [ ] QDQ graph 또는 quantized op path 생성 확인
- [ ] 실패 시 예외 메시지 정리

### B. dynamic quantization

- [ ] NLP 모델에 dynamic quantization 비교 실험 추가 여부 결정
- [ ] 추가한다면 distilbert에 한정해 구현

### C. granularity

- [ ] per-tensor weight quantization 경로 확인
- [ ] per-channel weight quantization 경로 확인
- [ ] 지원되는 모델에 대해 두 모드 비교

## Phase 4. Evaluation

### A. correctness compare

- [ ] FP32 vs quantized output 비교기 작성
- [ ] max abs error 계산
- [ ] mean abs error 계산
- [ ] cosine similarity 계산
- [ ] SNR은 보조 지표로만 추가
- [ ] vision과 NLP 모델에 서로 다른 acceptance gate 정의
- [ ] drift를 task-accuracy proxy로 해석한다는 문구를 README/report에 명시
- [ ] task metric을 붙일 수 있는 모델이 있으면 최소 1개 추가

### B. edge cases

- [ ] vision 입력 크기 변화 실험 필요 여부 결정
- [ ] NLP sequence length 32/64/128에서 drift 비교
- [ ] attention mask 패턴에 따른 drift 차이 확인

### C. failure analysis

- [ ] 어떤 모델에서 drift가 큰지 확인
- [ ] 어떤 layer가 민감할지 가설 세우기
- [ ] 필요 시 일부 layer exclude 실험 설계
- [ ] drift가 calibration 부족 때문인지 provider/op coverage 문제인지 구분
- [ ] quantized kernel이 실제로 사용되었는지 확인할 체크포인트 추가

### D. quantization correctness reporting

- [ ] output tensor별 max/mean/cosine/SNR 표 생성
- [ ] vision / NLP별 acceptance gate 통과 여부 표기
- [ ] ``semantic equivalence''가 아니라 ``accuracy retention''으로 표현 통일
- [ ] latency/size 이득과 drift를 함께 읽는 summary 작성

## Phase 5. Benchmarking

### A. protocol

- [ ] warmup 수 확정
- [ ] repeat 수 확정
- [ ] thread 수 고정
- [ ] provider 고정

### B. actual runs

- [ ] resnet18 FP32 vs INT8 측정
- [ ] mobilenetv3-small FP32 vs INT8 측정
- [ ] distilbert FP32 vs INT8 측정
- [ ] 필요 시 distilbert FP32 vs dynamic quantization 측정

### C. tables

- [ ] size reduction 표 생성
- [ ] latency 표 생성
- [ ] drift 표 생성

## Phase 6. Ablation

### A. calibration size

- [ ] calibration sample 8개
- [ ] calibration sample 32개
- [ ] calibration sample 128개
- [ ] sample 수가 drift에 미치는 영향 해석

### B. granularity

- [ ] per-tensor vs per-channel 비교
- [ ] vision에서 차이가 나는지 확인
- [ ] NLP에서 차이가 나는지 확인

### C. exclusion policy

- [ ] quantization 제외 레이어 후보 정리
- [ ] 제외했을 때 drift 개선되는지 확인
- [ ] 속도/크기 손해도 같이 기록

## Phase 7. Reporting and Packaging

### A. 모델별 report

- [ ] baseline metadata
- [ ] quantization mode
- [ ] calibration 설정
- [ ] quantized op summary
- [ ] size 변화
- [ ] latency 변화
- [ ] drift summary

### B. README polish

- [ ] quantization 목표를 한 문장으로 다시 쓰기
- [ ] 모델 목록 표 추가
- [ ] calibration 절차 설명 추가
- [ ] 왜 activation quantization이 어려운지 설명 추가
- [ ] 왜 INT8이 항상 빠르지 않은지 설명 추가

### C. Interview prep

- [ ] scale / zero-point를 30초로 설명하는 답변 작성
- [ ] per-channel이 왜 중요한지 30초 답변 작성
- [ ] calibration이 왜 중요한지 30초 답변 작성
- [ ] 왜 어떤 모델은 잘 되고 어떤 모델은 안 되는지 1분 답변 작성
- [ ] 왜 latency 개선이 제한적일 수 있는지 1분 답변 작성
