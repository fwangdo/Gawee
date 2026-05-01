# End-to-End 실행 가이드

ONNX 모델을 입력받아 Gawee MLIR 파이프라인을 거쳐 네이티브 바이너리로 실행하고,
ONNX Runtime 기준으로 correctness를 검증하는 전체 과정.

---

## 전제 조건

```bash
# MLIR 빌드 (middle/mlir/build 아래 바이너리 생성)
cd middle/mlir && ./build.sh

# Backend 빌드 (back/build 아래 바이너리 생성)
cd back && ./build.sh

# Python 환경 (onnx, onnxruntime, numpy)
source .venv/bin/activate
```

---

## 파이프라인 단계별 실행

### 1. ONNX 모델 준비

원본 ONNX 모델을 사용한다. Frontend rewrite는 하지 않는다.
NLP 모델처럼 dynamic shape가 있으면 static shape binding을 먼저 수행한다.

```python
import onnx
from onnx import shape_inference

model = onnx.load("benchmarks/onnx/nlp/bert_tiny/onnx/model.onnx")

# Dynamic dim -> static dim 바인딩 (NLP 모델의 경우)
for inp in model.graph.input:
    if inp.name == "input_ids":
        for dim, val in zip(inp.type.tensor_type.shape.dim, [1, 128]):
            dim.ClearField("dim_param")
            dim.dim_value = val
    # attention_mask, token_type_ids도 동일하게 처리

inferred = shape_inference.infer_shapes(model)
onnx.save(inferred, "inferred.onnx")
```

Vision 모델(resnet18 등)은 이미 static shape이므로 shape inference만 수행하면 된다.

### 2. ONNX -> Gawee MLIR (Translation)

```bash
middle/mlir/build/gawee-onnx-translate inferred.onnx -o gawee.mlir
```

ONNX 노드가 `gawee.*` dialect op 또는 direct `tensor/linalg/math` op으로 변환된다.

### 3. Gawee -> Linalg -> Loops (Lowering)

```bash
# loops MLIR (AOT 빌더의 ABI source로 사용)
middle/mlir/build/gawee-opt --gawee-to-loops gawee.mlir > loops.mlir

# LLVM MLIR (AOT 빌더의 코드 생성 입력)
middle/mlir/build/gawee-opt --gawee-to-llvm gawee.mlir > llvm.mlir
```

`--gawee-to-loops`는 `Gawee -> Linalg -> Bufferize -> SCF` 까지,
`--gawee-to-llvm`는 거기서 `-> LLVM dialect`까지 내린다.

### 4. AOT Runner 빌드

```bash
back/build/gawee-aot build \
  --abi-source loops.mlir \
  --input llvm.mlir \
  --output forward_runner \
  --entry forward \
  --num-output-args 1
```

- `--abi-source`: memref 시그니처를 읽어 C 래퍼를 생성하는 데 사용
- `--num-output-args`: 출력 텐서 개수 (함수가 memref를 반환하면 0, 인자로 받으면 1)

### 5. 입력 데이터 준비

```bash
mkdir -p inputs/
```

입력 데이터는 NumPy `.npy` 파일로 저장한다.
파일명은 `arg0.npy`, `arg1.npy`, ... 순서로,
ONNX graph input 순서 뒤에 runtime initializer가 이어진다.

`eval_priority_models.py`의 `save_runner_inputs()`가 이 과정을 자동으로 처리한다.

### 6. 실행

```bash
mkdir -p outputs/
./forward_runner inputs/ outputs/
```

결과가 `outputs/output0.npy`에 저장된다.

### 7. Correctness 검증

```python
import numpy as np
import onnxruntime as ort

# Gawee 출력
gawee_output = np.load("outputs/output0.npy")

# ORT 기준 출력
session = ort.InferenceSession("inferred.onnx", providers=["CPUExecutionProvider"])
ort_outputs = session.run(None, feed_dict)  # feed_dict는 동일한 입력
ort_output = ort_outputs[0]

# 비교
max_abs = np.max(np.abs(gawee_output - ort_output))
close = np.allclose(gawee_output, ort_output, atol=5e-4, rtol=1e-4)
print(f"max_abs_diff: {max_abs}, allclose: {close}")
```

---

## 자동 평가 (권장)

위 과정을 모든 priority 모델에 대해 자동으로 수행하는 스크립트:

```bash
.venv/bin/python3 back/eval_priority_models.py
```

결과:
- `artifacts/back_eval/priority_models_report.json` - 상세 결과
- `artifacts/back_eval/priority_models_report.md` - 마크다운 요약
- `artifacts/back_eval/<model>/` - 각 모델별 중간 산출물

---

## 현재 벤치마크 결과

| 모델 | max_abs_diff | threshold | 결과 |
|------|-------------|-----------|------|
| resnet18 | 5.25e-06 | 1e-4 | PASS |
| bert_tiny | 1.79e-07 | 5e-4 | PASS |
| tinyllama_15m | 1.62e-05 | 5e-4 | PASS |

---

## 주의사항

- **원본 ONNX를 사용한다.** Frontend-rewritten ONNX는 노드 수가 늘어나면서 FP 오차가 누적된다.
- **distilbert_base_uncased는 제외.** 모델이 커서 dev 머신에서 시스템 행을 유발한다.
- **CPU 부하 주의.** 전체 eval은 모델 3개를 순차 실행하므로 시간이 걸린다. 단일 모델만 테스트하려면 스크립트를 수정하거나 위 단계별 명령을 직접 사용한다.
