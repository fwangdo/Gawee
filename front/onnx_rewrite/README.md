# ONNX Rewrite

이 문서는 `front/onnx_rewrite` 모듈의 목적, 구조, 사용법, 현재 구현 범위를 설명한다.

<details open>
<summary><strong>한국어</strong></summary>

## 개요

`front/onnx_rewrite`는 `Gawee`의 ONNX frontend rewrite 모듈이다.

현재 목표는 다음 한 줄로 요약할 수 있다.

> 입력 ONNX graph를 읽고, 최종 결과를 `supported-op-only` graph로 내린다.

즉 이 시스템은 일반적인 ONNX optimizer라기보다, 특정 하드웨어/중간 표현이 받아들일 수 있는 연산 집합으로 graph를 강제로 낮추는 legality-first rewrite pipeline에 가깝다.

현재는 correctness / validation / latency 계측까지 포함한 scaffold를 갖추고 있고, 현재 기본 priority 모델에 대해 unsupported op 제거를 목표로 둔다.

## Priority 모델

- `resnet18`
- `bert_tiny`
- `tinyllama_15m`

모델 경로는 [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py)의 `PRIORITY_MODELS`에 정의되어 있다.

## Extended benchmark 후보

- `qwen3_0_6b`
  `RoPE` 기반 decoder-only LLM benchmark 후보다.
- `yolo26_n`
  최신 `YOLO26` 계열 benchmark 후보다.

이 둘은 [specs/catalog.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/specs/catalog.py)의 `EXTENDED_BENCHMARK_MODELS`에 정의되어 있다.
`distilbert_base_uncased`는 CPU 사용량 문제로 기본 benchmark 대상에서 제외했다.

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

현재 기본 priority 모델 기준으로는 unsupported op 제거를 진행한다.

다만 성능 특성은 모델에 따라 다르다.

- `resnet18`: legality 달성 + latency 거의 유지
- `bert_tiny`: legality 달성 + correctness 통과, 하지만 graph blow-up으로 latency 악화
- `tinyllama_15m`: `RoPE`가 포함된 초소형 decoder LLM 후보로 교체 예정

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

The module already includes correctness validation and latency measurement, and targets unsupported-op removal for the default priority models.

## Priority Models

- `resnet18`
- `bert_tiny`
- `tinyllama_15m`

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

The default priority set is now `resnet18`, `bert_tiny`, and `tinyllama_15m`.

However, the performance characteristics differ by model:

- `resnet18`: legality achieved with roughly preserved latency
- `bert_tiny`: legality achieved and correctness passes, but graph blow-up causes large slowdown
- `tinyllama_15m`: selected to replace `distilbert` as the smaller RoPE-based decoder candidate

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
