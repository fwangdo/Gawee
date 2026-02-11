# Phase 5: Bufferization과 Linalg → Loops

## 개요

Phase 3에서 Gawee → Linalg(텐서 위의 연산)으로 변환했다.
이 Phase에서는 두 단계를 거친다:

1. **Bufferization**: `tensor` → `memref` (값 의미론 → 참조 의미론)
2. **Linalg → Loops**: `linalg.generic` 등 → `scf.for` 루프

```
tensor<1x64x112x112xf32>  →  memref<1x64x112x112xf32>
linalg.conv_2d_nchw_fchw  →  scf.for + scf.for + ... + arith.mulf + arith.addf
```

---

## 1. tensor vs memref

| | tensor | memref |
|---|---|---|
| 의미론 | 값 (value) | 참조 (reference) |
| 불변성 | 불변 (SSA) | 가변 (메모리에 저장) |
| 비유 | `const int x = 5` | `int *ptr` |
| 용도 | 고수준 최적화 | 실제 메모리 접근 |

### 왜 tensor를 먼저 쓰는가?

tensor는 **SSA(Static Single Assignment)** 형태라 최적화가 쉽다:
- 값이 변하지 않으므로 의존성 분석이 단순
- 메모리 할당/해제를 신경 쓸 필요 없음
- fusion, tiling 같은 최적화가 tensor 수준에서 더 쉬움

memref는 실제 메모리를 다루므로:
- alias 분석이 필요 (같은 메모리를 가리킬 수 있음)
- 할당/해제 관리 필요

---

## 2. One-Shot Bufferize

### 동작 원리

One-Shot Bufferize는 tensor 연산을 **in-place memref 연산**으로 변환한다:

```
변환 전 (tensor):
  %empty = tensor.empty [1, 64, 112, 112] : tensor<1x64x112x112xf32>
  %filled = linalg.fill ins(%zero) outs(%empty)
  %conv = linalg.conv_2d_nchw_fchw ins(%input, %weight) outs(%filled)

변환 후 (memref):
  %alloc = memref.alloc() : memref<1x64x112x112xf32>
  linalg.fill ins(%zero) outs(%alloc)
  linalg.conv_2d_nchw_fchw ins(%input_memref, %weight_memref) outs(%alloc)
```

### gawee-opt에서의 설정

```cpp
// tensor.empty → bufferization.alloc_tensor로 변환 (bufferize 준비)
pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

// 실제 bufferization 실행
bufferization::OneShotBufferizePassOptions bufOpts;
bufOpts.bufferizeFunctionBoundaries = true;  // 함수 경계도 bufferize
pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
```

### `bufferizeFunctionBoundaries = true`

함수 시그니처도 변환한다:
```
변환 전: func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>
변환 후: func @forward(%arg0: memref<1x3x224x224xf32>) -> memref<1x1000xf32>
```

이 옵션 없이는 함수 경계에서 tensor ↔ memref 변환(to_tensor/to_memref)이 삽입된다.

### Bufferization 인터페이스가 필요한 이유

각 dialect의 op마다 "이 op을 어떻게 bufferize할지" 정보가 필요하다.
gawee-opt에서 등록한 인터페이스들이 이 정보를 제공한다:

```cpp
arith::registerBufferizableOpInterfaceExternalModels(registry);
linalg::registerBufferizableOpInterfaceExternalModels(registry);
tensor::registerBufferizableOpInterfaceExternalModels(registry);
bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
```

---

## 3. Linalg → Loops 변환

```cpp
pm.addPass(createConvertLinalgToLoopsPass());
```

이 pass는 linalg op을 SCF(Structured Control Flow) 루프로 변환한다.

### 변환 예시: ReLU

```
변환 전 (linalg):
  linalg.generic {
    indexing_maps = [identity, identity],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input) outs(%output) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0
      %result = arith.maximumf %in, %zero
      linalg.yield %result
  }

변환 후 (SCF):
  scf.for %n = 0 to 1 {
    scf.for %c = 0 to 64 {
      scf.for %h = 0 to 112 {
        scf.for %w = 0 to 112 {
          %val = memref.load %input[%n, %c, %h, %w]
          %zero = arith.constant 0.0
          %result = arith.maximumf %val, %zero
          memref.store %result, %output[%n, %c, %h, %w]
        }
      }
    }
  }
```

### iterator_types의 영향

- `parallel`: 독립적인 루프 → `scf.for` (추후 병렬화 가능)
- `reduction`: 축소 루프 → 누적 변수가 필요

---

## 4. SCF Dialect

SCF는 **구조화된 제어 흐름**을 표현한다:

| SCF op | 의미 |
|--------|------|
| `scf.for` | for 루프 |
| `scf.while` | while 루프 |
| `scf.if` | 조건 분기 |

SCF는 CFG(Control Flow Graph)보다 높은 수준이라 분석/최적화가 쉽다.
Phase 7에서 `scf.for` → `cf.br` + `cf.cond_br` (기본 분기)로 한 단계 더 낮춘다.

---

## 5. 전체 파이프라인 흐름

```
Phase 3:  gawee.conv     →  linalg.conv_2d_nchw_fchw  (tensor 위)
Phase 5a: tensor         →  memref                     (bufferization)
Phase 5b: linalg.conv    →  scf.for + arith ops        (loop lowering)
```

gawee-opt의 `gawee-to-loops` 파이프라인:

```cpp
pm.addPass(gawee::createGaweeToLinalgPass());    // Gawee → Linalg
bufferization::OneShotBufferizePassOptions bufOpts;
bufOpts.bufferizeFunctionBoundaries = true;
pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));  // tensor → memref
pm.addPass(createConvertLinalgToLoopsPass());     // Linalg → SCF
```

---

## 핵심 개념 정리

- **Bufferization** = tensor(값) → memref(참조) 변환
- **One-Shot Bufferize** = MLIR의 표준 bufferization pass
- **bufferizeFunctionBoundaries** = 함수 시그니처도 memref로 변환
- **Bufferization 인터페이스** = 각 op의 bufferize 방법 정보
- **Linalg → Loops** = linalg op → scf.for 중첩 루프
- **SCF** = 구조화된 제어 흐름 dialect (for, while, if)
- **destination-passing이 여기서 빛나는 이유**: `outs` 텐서가 그대로 출력 memref가 됨
