# gawee

[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)

`gawee`는 ONNX 모델을 받아 그래프 rewrite를 수행하고, MLIR 기반 파이프라인
(`Gawee Dialect -> Linalg -> SCF/LLVM`)으로 lowering해서 AOT 실행 파일까지
만드는 딥러닝 컴파일러 프로젝트입니다.

현재 우선순위는 "작은 vision 모델용 우회 rewrite"보다, `resnet / bert_tiny / tinyllama`
같은 실제 benchmark 모델이 middle-end에서 직접 지원되는 op 집합을 넓히는 것입니다.

---

## 현재 상태

| 모델 | ONNX Emission | Gawee -> Linalg | Full LLVM/AOT | 비고 |
| --- | --- | --- | --- | --- |
| ResNet-18 | pass | pass | pass | AOT 실행 및 수치 검증 경로 확보 |
| bert_tiny | pass | pass | pass | `Gather` semantic op 유지 후 correctness 통과 |
| tinyllama_15m | 준비 중 | 준비 중 | 준비 중 | `RoPE`가 들어간 초소형 decoder LLM 후보 |

### 이번 단계에서 늘린 지원 범위

- `MatMul`을 `Gemm/Linear`와 분리
- semantic op를 `gawee` dialect에 명시적으로 추가
  - `gawee.gather`
  - `gawee.gather_elements`
  - `gawee.range`
  - `gawee.resize`
  - `gawee.split`
  - `gawee.tile`
- trivial decomposition은 emitter에서 직접 lowering
  - `Pow`, `Neg`, `Sin`, `Cos`, `And`, `LessOrEqual`, `IsNaN`, `Mod`
  - `Constant`, `ConstantOfShape`

핵심 의도는 front에서 unsupported op를 과하게 rewrite해서 숨기지 않고,
middle-end가 benchmark에 실제로 등장하는 op semantics를 직접 받도록 만드는 것입니다.

---

## 파이프라인

```text
ONNX Model
  -> Rewrite / Optimize (Python)
  -> MLIR Gawee Dialect
  -> Linalg
  -> Bufferization / SCF / Math / LLVM
  -> Native Binary
```

### Frontend (Python)

- ONNX graph rewrite 및 정규화
- constant folding, spec-driven rewrite
- 모델별 unsupported op audit

### Middle-end (C++ / MLIR)

- `gawee-onnx-translate`
  - ONNX protobuf를 직접 읽어서 `gawee.*` 또는 direct `tensor/linalg/math` op 생성
- `gawee-opt`
  - `--convert-gawee-to-linalg`
  - `--gawee-to-loops`
  - `--gawee-to-llvm`

### Backend (C++)

- AOT 실행 파일 생성
- ONNX Runtime 기준 결과 비교
- NLP 모델은 static shape binding 경로를 사용

---

## 지원 전략

모든 연산을 새 dialect op로 만들지는 않습니다.

### 1. semantic op는 `gawee`에 남긴다

다음 op는 shape/axis/lookup 의미가 크고, 나중에 fallback 여부를 판단할 가치가 있습니다.

- `MatMul`
- `Gather`
- `GatherElements`
- `Range`
- `Resize`
- `Split`
- `Tile`

이런 op는:

1. `GaweeOps.td`에 정의
2. ONNX emitter에서 `gawee.*` 생성
3. `GaweeToLinalg.cpp`에서 lowering

순서로 구현합니다.

### 2. trivial op는 direct decomposition 한다

다음 op는 별도 dialect op 없이 emitter에서 바로 푸는 편이 낫습니다.

- `Pow`
- `Neg`
- `Sin`
- `Cos`
- `And`
- `LessOrEqual`
- `IsNaN`
- `Mod`
- `Constant`
- `ConstantOfShape`

이 경우 `tensor.generate`, `linalg.generic`, `arith`, `math` 조합으로 직접 낮춥니다.

---

## 현재 middle-end에서 중요한 op

현재 `Gawee` dialect와 ONNX emission 경로에서 중요한 축은 다음과 같습니다.

- CNN 경로
  - `conv`, `relu`, `add`, `max_pool`, `average_pool`, `global_average_pool`
- Transformer / LLM 경로
  - `matmul`, `reshape`, `transpose`, `expand`, `slice`, `softmax`
  - `gather`, `gather_elements`, `range`, `split`, `tile`, `resize`
- 공통 연산
  - `mul`, `div`, `sub`, `reduce_mean`, `reduce_sum`, `where`, `cast`

---

## 검증 메모

이번 semantic op 확장 이후 확인한 내용:

- `resnet18`
  - ONNX emission 통과
  - `gawee-to-llvm` 통과
- `bert_tiny`
  - ONNX emission 통과
  - `gawee-to-llvm` 통과
  - correctness 통과
- `qwen3_0_6b`
  - ONNX emission 통과
  - `convert-gawee-to-linalg` 통과
  - `gawee.range`의 dynamic length legalization 문제를 수정

`distilbert_base_uncased`는 CPU 사용량 때문에 기본 benchmark/eval 대상에서 제외했습니다.
그 자리는 `tinyllama_15m`로 대체해, 더 작은 체급에서 `RoPE`가 포함된 modern decoder 경로를 먼저 보려 합니다.

`qwen`은 모듈이 커서 full LLVM 파이프라인 자체가 오래 걸리므로,
현재는 확장 benchmark로 두고 "semantic op가 MLIR 단계에서 illegal로 남지 않는다"는 것을 우선 기준으로 봅니다.

---

## 학습 문서

이번 단계와 직접 연결되는 문서:

- [MatMul Lowering Summary](middle/mlir/docs/kr/MatMulLowering_Summary.md)
- [MatMul Lowering Quiz](middle/mlir/docs/kr/MatMulLowering_Quiz.cpp)
- [Semantic Op Lowering Summary](middle/mlir/docs/kr/SemanticOpLowering_Summary.md)
- [Semantic Op Lowering Quiz](middle/mlir/docs/kr/SemanticOpLowering_Quiz.cpp)

---

## 프로젝트 구조

```text
gawee/
├── front/onnx_rewrite/        # Python frontend / ONNX rewrite
├── middle/mlir/               # MLIR middle-end
│   ├── include/Gawee/         # TableGen dialect/op definitions
│   ├── include/Emit/          # ONNX emitter headers
│   ├── lib/Emit/              # ONNX -> MLIR
│   ├── lib/Conversion/        # Gawee -> Linalg / lowering pipeline
│   └── tools/                 # gawee-opt, gawee-onnx-translate
├── back/                      # AOT builder / evaluation
└── docs/                      # notes and reports
```

---

## 참고

- ONNX specification
- MLIR documentation
- LLVM documentation
