# ONNX Rewrite

이 문서는 `front/onnx_rewrite` 모듈의 목적, 구조, 사용법, 현재 구현 범위를 설명한다.

<details open>
<summary><strong>한국어</strong></summary>

## 개요

`front/onnx_rewrite`는 `Gawee`의 ONNX frontend rewrite 모듈이다.

현재 목표는 다음 한 줄로 요약할 수 있다.

> 입력 ONNX graph를 읽고, 최종 결과를 `supported-op-only` graph로 내린다.

즉 이 시스템은 일반적인 ONNX optimizer라기보다, 특정 하드웨어/중간 표현이 받아들일 수 있는 연산 집합으로 graph를 강제로 낮추는 legality-first rewrite pipeline에 가깝다.

현재는 correctness / validation / latency 계측까지 포함한 scaffold를 갖추고 있고, priority 모델 3종에 대해 unsupported op 제거를 달성했다.

## Priority 모델

- `resnet18`
- `distilbert_base_uncased`
- `bert_tiny`

모델 경로는 [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py)의 `PRIORITY_MODELS`에 정의되어 있다.

## Supported Op Contract

지원 대상으로 간주하는 op set은 [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py)의 `SUPPORTED_OPS`가 기준이다.

현재 포함 op:

- arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
- tensor/layout: `Cast`, `Concat`, `Expand`, `Reshape`, `Shape`, `Slice`, `Squeeze`, `Transpose`, `Unsqueeze`
- reduction/activation: `ReduceMean`, `ReduceSum`, `Erf`, `Gelu`, `HardSigmoid`, `HardSwish`, `LeakyRelu`, `Relu`, `Sigmoid`, `Softmax`, `Sqrt`, `Tanh`
- comparison/select: `Equal`, `Where`
- spatial: `AveragePool`, `Conv`, `GlobalAveragePool`, `MaxPool`, `Pad`

현재 시스템 계약은 명확하다.

- rewrite 이후 unsupported op가 하나라도 남으면 실패
- `onnx.checker.check_model`가 실패하면 실패

즉 “최대한 바꿔본다”가 아니라 supported-op-only 결과를 만들지 못하면 실패하는 구조다.

## 디렉토리 구조

- [run_onnx_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/run_onnx_rewrite.py)  
  rewrite / audit CLI
- [eval_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/eval_rewrite.py)  
  rewrite 후 correctness / latency 평가 CLI
- [core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py)  
  top-level optimize entrypoint
- [passes/passer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/passer.py)  
  pass 실행 순서
- [passes/folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py)  
  pass 공통 helper, shape inference, producer/consumer map
- [checker/op_checker.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/checker/op_checker.py)  
  supported-op-only checker
- [analysis/audit.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/analysis/audit.py)  
  ONNX histogram / unsupported audit
- [runtime/validation.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/validation.py)  
  ORT correctness 비교
- [runtime/benchmark.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/benchmark.py)  
  ORT latency 측정
- [report.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/report.md)  
  benchmark / correctness 상태 정리
- [explain.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/explain.md)  
  개인 학습용 설명 노트. git ignore 대상

## 실행 흐름

실제 rewrite 흐름은 아래와 같다.

1. ONNX model load
2. before unsupported histogram 계산
3. ordered passes 실행
4. `onnx.checker.check_model`
5. after unsupported histogram 계산
6. unsupported op가 남아 있으면 실패
7. model save / optional report write

핵심 구현은 [core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py)에 있다.

## Pass 순서

현재 [passes/passer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/passer.py) 기준 pass 순서는 다음과 같다.

1. `ConstantFolding`
2. `EliminateId`
3. `RewriteBN`
4. `RewritePow`
5. `RewriteGather`
6. `RewriteGemm`
7. `RewriteMatmul`
8. `Cleanup`

이 순서에는 의도가 있다.

- 먼저 constant / identity 같은 단순한 정리를 끝낸다.
- 그 다음 BN / Pow처럼 독립적으로 지울 수 있는 op를 제거한다.
- 그 다음 Gather / Gemm / MatMul처럼 graph shape를 크게 바꾸는 rewrite를 적용한다.
- 마지막에 cleanup과 validation을 수행한다.

## 각 pass의 역할

### `ConstantFolding`

- `Constant`
- `ConstantOfShape`

를 initializer로 접는다.

이 pass는 이후 rewrite에서 “input이 static인가?”를 판별할 수 있게 해준다.

### `EliminateId`

`Identity`를 제거하고 edge를 직접 연결한다.

핵심은 `Identity output -> original input` mapping을 만든 뒤, downstream input과 graph output을 rewiring하는 것이다.

### `RewriteBN`

세 가지 경우를 처리한다.

1. `BatchNormalization` 단독: depthwise `1x1 Conv`로 치환
2. `Conv + BatchNormalization`: fused `Conv`
3. `ConvTranspose + BatchNormalization`: fused `ConvTranspose`

핵심 수식:

```text
scale_factor = scale / sqrt(var + eps)
bias_factor  = bias - mean * scale_factor
```

Conv fusion에서는 output channel 축에 scale을 곱한다.

- `Conv`: weight output 축은 axis 0
- `ConvTranspose`: weight output 축은 axis 1

### `RewritePow`

현재는 scalar initializer exponent만 처리한다.

지원 패턴:

- `x^1`
- `x^0.5`
- `x^-0.5`
- `x^-1`
- `x^2`
- `x^3`
- `x^4`

즉 `Pow`를 `Sqrt`, `Div`, `Mul` 조합으로 내린다.

### `RewriteGather`

현재 가장 중요한 rewrite 중 하나다.

처리 패턴:

1. static gather fold
2. scalar-index gather -> `Slice + Reshape`
3. small vocab embedding-style gather -> `Equal + Cast + Unsqueeze + Mul + Add`
4. chunked vocab gather -> chunk 단위 `Equal + Cast + Unsqueeze + Mul + ReduceSum + Add`
5. dynamic axis-0 gather -> one-hot-like mask + `MatMul` + shape restore

즉 Gather는 단일 전략이 아니라 case split 기반 rewrite다.

### `RewriteGemm`

static weight `Gemm`를 equivalent `1x1 Conv` chain으로 변환한다.

고려 요소:

- `alpha`
- `beta`
- `transA`
- `transB`
- optional bias

### `RewriteMatmul`

두 갈래다.

1. static weight가 있으면 Conv 계열로 lowering
2. 둘 다 dynamic이면 `Unsqueeze + Mul + ReduceSum`

또한 `RewriteGather`의 dynamic path가 `MatMul`을 중간 결과로 만들기 때문에, 실제 pipeline에서는 `Gather -> MatMul -> RewriteMatmul` 연쇄가 중요하다.

### `Cleanup`

- unused initializer 제거
- topological sort
- `onnx.checker.check_model`

최종 정리와 validation slot이다.

## Correctness / Validation

[runtime/validation.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/validation.py)는 ONNX Runtime 기반으로 원본/변환 모델을 비교한다.

현재 방식:

- deterministic seed
- 여러 dynamic size
- 여러 mask mode
- 여러 integer input mode

결과는 `ValidationResult`로 구조화되어 반환된다.

핵심 metric:

- `max_abs_diff`
- `worst_case`
- pass / fail

현재 default gate는:

```text
max_abs_diff <= 1e-4
```

## Latency / Throughput 측정

[runtime/benchmark.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/benchmark.py)는 ONNX Runtime CPU 기준 median / p95 latency를 측정한다.

현재 benchmark 조건:

- single-process
- `intra_op_num_threads=1`
- `inter_op_num_threads=1`
- sequential execution

throughput은 현재 report에서 batch 1 기준:

```text
throughput(samples/s) = 1000 / median_ms
```

로 계산했다.

## 현재 상태

현재 priority 모델 3종에 대해서는 unsupported op 제거를 달성했다.

다만 성능 특성은 모델에 따라 다르다.

- `resnet18`: legality 달성 + latency 거의 유지
- `bert_tiny`: legality 달성 + correctness 통과, 하지만 graph blow-up으로 latency 악화
- `distilbert_base_uncased`: legality 달성, 하지만 graph blow-up이 크고 current correctness gate를 약간 초과

자세한 수치는 [report.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/report.md)를 보면 된다.

## 사용 예시

### 1. audit only

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --audit-only
```

### 2. rewrite 실행

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output artifacts/front_rewrite/resnet18.onnx
```

### 3. rewrite + correctness + latency 평가

```bash
python -m front.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output artifacts/front_rewrite/resnet18.onnx \
  --report artifacts/front_rewrite/resnet18_eval.json
```

</details>

<details>
<summary><strong>English</strong></summary>

## Overview

`front/onnx_rewrite` is the ONNX frontend rewrite module for `Gawee`.

Its current goal is simple:

> Read an input ONNX graph and lower it into a supported-op-only graph.

This is not a generic optimizer. It is a legality-first rewrite pipeline for a constrained operator set.

The module already includes correctness validation and latency measurement, and currently removes unsupported ops for the three priority models.

## Priority Models

- `resnet18`
- `distilbert_base_uncased`
- `bert_tiny`

The model registry lives in [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py).

## Supported Op Contract

The supported operator set is defined in [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py) as `SUPPORTED_OPS`.

Current categories include:

- arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
- tensor/layout: `Cast`, `Concat`, `Expand`, `Reshape`, `Shape`, `Slice`, `Squeeze`, `Transpose`, `Unsqueeze`
- reduction/activation: `ReduceMean`, `ReduceSum`, `Erf`, `Gelu`, `HardSigmoid`, `HardSwish`, `LeakyRelu`, `Relu`, `Sigmoid`, `Softmax`, `Sqrt`, `Tanh`
- comparison/select: `Equal`, `Where`
- spatial: `AveragePool`, `Conv`, `GlobalAveragePool`, `MaxPool`, `Pad`

The pipeline contract is strict:

- if any unsupported op remains after rewriting, optimization fails
- if `onnx.checker.check_model` fails, optimization fails

## Directory Layout

- [run_onnx_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/run_onnx_rewrite.py): rewrite / audit CLI
- [eval_rewrite.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/eval_rewrite.py): rewrite + correctness + latency evaluation
- [core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py): top-level optimization entrypoint
- [passes/passer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/passer.py): pass ordering
- [passes/folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py): shared pass helpers
- [checker/op_checker.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/checker/op_checker.py): supported-op-only checking
- [analysis/audit.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/analysis/audit.py): histogram / unsupported audit
- [runtime/validation.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/validation.py): ORT correctness comparison
- [runtime/benchmark.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/benchmark.py): ORT latency measurement
- [report.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/report.md): current benchmark and correctness summary
- [explain.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/explain.md): private learning notes, ignored by git

## Execution Flow

The rewrite pipeline runs as follows:

1. Load ONNX model
2. Compute pre-rewrite unsupported histogram
3. Run ordered passes
4. Run `onnx.checker.check_model`
5. Compute post-rewrite unsupported histogram
6. Fail if unsupported ops remain
7. Save model and optional reports

The main implementation lives in [core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py).

## Pass Order

Current pass order in [passes/passer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/passer.py):

1. `ConstantFolding`
2. `EliminateId`
3. `RewriteBN`
4. `RewritePow`
5. `RewriteGather`
6. `RewriteGemm`
7. `RewriteMatmul`
8. `Cleanup`

The intention is:

- clean up constants and no-ops first
- remove independent unsupported ops such as BN / Pow
- then apply larger structural rewrites such as Gather / Gemm / MatMul
- finally normalize and validate the graph

## Pass Summary

### `ConstantFolding`

Folds `Constant` and `ConstantOfShape` into initializers.

### `EliminateId`

Removes `Identity` by rewiring downstream users and graph outputs.

### `RewriteBN`

Handles:

1. standalone `BatchNormalization` -> depthwise `1x1 Conv`
2. `Conv + BatchNormalization` -> fused `Conv`
3. `ConvTranspose + BatchNormalization` -> fused `ConvTranspose`

Core formula:

```text
scale_factor = scale / sqrt(var + eps)
bias_factor  = bias - mean * scale_factor
```

### `RewritePow`

Rewrites scalar-constant `Pow` into supported ops such as `Sqrt`, `Div`, and `Mul`.

### `RewriteGather`

Supports:

1. static gather folding
2. scalar-index gather
3. small-vocabulary embedding-style lowering
4. chunked lowering for large vocabularies
5. dynamic axis-0 gather via one-hot-like masks and `MatMul`

### `RewriteGemm`

Converts static-weight `Gemm` into equivalent `1x1 Conv` chains.

### `RewriteMatmul`

Converts `MatMul` either into:

1. Conv-based chains when one side is static
2. `Unsqueeze + Mul + ReduceSum` when both sides are dynamic

### `Cleanup`

- remove unused initializers
- topologically sort nodes
- validate with `onnx.checker.check_model`

## Correctness / Validation

[runtime/validation.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/validation.py) compares original and rewritten models with ONNX Runtime.

Current validation strategy:

- deterministic seeds
- multiple dynamic sizes
- multiple mask modes
- multiple integer-input modes

Main metrics:

- `max_abs_diff`
- `worst_case`
- pass / fail

Current default gate:

```text
max_abs_diff <= 1e-4
```

## Latency / Throughput Measurement

[runtime/benchmark.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/benchmark.py) measures median and p95 latency with ONNX Runtime CPU.

Current benchmark setup:

- single-process
- `intra_op_num_threads=1`
- `inter_op_num_threads=1`
- sequential execution

Reported throughput is currently computed from batch-1 median latency:

```text
throughput(samples/s) = 1000 / median_ms
```

## Current Status

All three priority models currently reach unsupported-op count zero after rewriting.

However, the performance characteristics differ by model:

- `resnet18`: legality achieved with roughly preserved latency
- `bert_tiny`: legality achieved and correctness passes, but graph blow-up causes large slowdown
- `distilbert_base_uncased`: legality achieved, but graph blow-up is large and the current correctness gate is slightly exceeded

See [report.md](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/report.md) for the latest numbers.

## Usage Examples

### 1. Audit only

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --audit-only
```

### 2. Rewrite only

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output artifacts/front_rewrite/resnet18.onnx
```

### 3. Rewrite + correctness + latency evaluation

```bash
python -m front.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output artifacts/front_rewrite/resnet18.onnx \
  --report artifacts/front_rewrite/resnet18_eval.json
```

</details>
