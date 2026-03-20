# Gawee MLIR Compiler - UNet Progress Tracker

## Overview

Extending the Gawee compiler to support UNet inference.
ResNet-18 pipeline is complete (see `progress_resnet.md`).

```
Goal: UNet JSON (116 nodes) → Full MLIR → LLVM IR

Missing ops: cat (4 nodes), interpolate (5 nodes)
Existing ops already working: Conv (47), Relu (43), Add (16), MaxPool (1)
```

---

## Phase 9: gawee.cat — Tensor Concatenation

### What is `cat`?

`cat`은 **concatenate**의 줄임말이다. 여러 tensor를 **같은 축(axis) 기준으로 이어 붙여서** 하나의 더 큰 tensor를 만드는 연산이다.

중요:
- 원소끼리 더하는 연산이 아니다.
- 새로운 값을 계산하는 연산이라기보다, **기존 tensor들을 특정 차원에서 연결하는 재배치 연산**에 가깝다.
- 이어 붙이는 축을 제외한 나머지 차원 크기는 모두 같아야 한다.

예시 1:
```text
a shape = [1, 32, 64, 64]
b shape = [1, 32, 64, 64]
cat([a, b], axis=1) = [1, 64, 64, 64]
```

설명:
- `axis=1`은 channel 축이다.
- batch, height, width는 그대로 유지된다.
- channel만 32 + 32 = 64가 된다.

예시 2:
```text
a shape = [2, 3]
b shape = [2, 5]
cat([a, b], axis=1) = [2, 8]
```

반대로 아래는 불가능하다:
```text
a shape = [1, 32, 64, 64]
b shape = [1, 32, 32, 32]
cat([a, b], axis=1)   // 불가
```

이유:
- `axis=1`이 아닌 차원들(H, W)이 서로 다르기 때문이다.

### Why UNet uses `cat`

UNet은 downsampling 경로의 feature와 upsampling 경로의 feature를 **skip connection**으로 합친다.
이때 둘을 더하는 것이 아니라 **channel 방향으로 붙이는 경우가 많다**.

즉:
```text
encoder feature  [N, C1, H, W]
decoder feature  [N, C2, H, W]
cat(axis=1)
=> [N, C1 + C2, H, W]
```

그 다음 convolution이 붙어서, 합쳐진 feature를 다시 섞어준다.

### `cat` vs `add`

`add`:
- shape가 보통 동일해야 한다.
- 같은 위치의 값을 더한다.
- 결과 shape는 보통 입력 shape와 같다.

`cat`:
- 지정한 축을 제외하면 shape가 같아야 한다.
- 값을 더하지 않고 이어 붙인다.
- 결과 shape는 concat 축에서 더 커진다.

### 9-1. Op Definition (TableGen) ⬚

| Task | Status | Files |
|------|--------|-------|
| Define `gawee.cat` in TableGen | ⬚ Todo | include/Gawee/GaweeOps.td |
| Rebuild to generate .inc files | ⬚ Todo | build.sh |

**Op spec:**
```
Inputs:  Variadic<AnyTensor>:$inputs
Attrs:   I64Attr:$axis
Output:  AnyTensor:$result
```

**Key points:**
- Variadic input (여러 텐서를 받아야 함)
- `axis` = concatenation dimension (UNet에서 channel 축, dim=1)
- 기존 Add op과 유사하지만 입력이 가변적

---

### 9-2. MLIREmitter — emitCat ⬚

| Task | Status | Files |
|------|--------|-------|
| Add `emitCat()` to MLIREmitter | ⬚ Todo | lib/Emit/MLIREmitter.cpp |
| Handle variadic inputs from JSON | ⬚ Todo | lib/Emit/MLIREmitter.cpp |

**Key points:**
- JSON의 `inputs` 배열에서 여러 입력 텐서를 읽어야 함
- `attrs`에서 `axis` 추출
- 출력 shape 계산: concat 축만 합산, 나머지 동일

---

### 9-3. CatOpLowering (Gawee → Linalg) ⬚

| Task | Status | Files |
|------|--------|-------|
| Implement CatOpLowering | ⬚ Todo | lib/Conversion/GaweeToLinalg.cpp |
| Register pattern in pass | ⬚ Todo | lib/Conversion/GaweeToLinalg.cpp |

**Lowering strategy:**
```
gawee.cat(%a, %b, axis=1)
  → %out = tensor.empty(...)
  → %s0 = tensor.insert_slice %a into %out[0,0,...][Sa,...][1,1,...]
  → %s1 = tensor.insert_slice %b into %s0[0,Ca,...][Sb,...][1,1,...]
```

**Key points:**
- `tensor.insert_slice`를 입력 개수만큼 반복
- offset은 concat 축에서만 누적, 나머지 축은 0
- destination-passing style 유지

---

### 9-4. Cat 테스트 ⬚

| Task | Status | Files |
|------|--------|-------|
| Hand-written cat test MLIR | ⬚ Todo | test/cat_test.mlir |
| Cat end-to-end with gawee-opt | ⬚ Todo | - |
| Summary document | ⬚ Todo | docs/Cat_Summary.md |
| Quiz file | ⬚ Todo | docs/Cat_Quiz.cpp |

```bash
# 테스트 명령어
./build/gawee-opt --convert-gawee-to-linalg test/cat_test.mlir
```

---

## Phase 10: gawee.interpolate — Upsampling

### 10-1. Op Definition (TableGen) ⬚

| Task | Status | Files |
|------|--------|-------|
| Define `gawee.interpolate` in TableGen | ⬚ Todo | include/Gawee/GaweeOps.td |
| Rebuild to generate .inc files | ⬚ Todo | build.sh |

**Op spec:**
```
Inputs:  AnyTensor:$input
Attrs:   DenseI64ArrayAttr:$scale_factor, StrAttr:$mode
Output:  AnyTensor:$result
```

**Key points:**
- `scale_factor` = [2, 2] (UNet에서 2배 업샘플)
- `mode` = "nearest" 또는 "bilinear"
- nearest가 구현이 쉬움, bilinear은 인덱스 수학이 복잡

---

### 10-2. MLIREmitter — emitInterpolate ⬚

| Task | Status | Files |
|------|--------|-------|
| Add `emitInterpolate()` to MLIREmitter | ⬚ Todo | lib/Emit/MLIREmitter.cpp |
| Parse scale_factor and mode from JSON | ⬚ Todo | lib/Emit/MLIREmitter.cpp |

**Key points:**
- JSON `attrs`에서 `scale_factor`, `mode` 추출
- 출력 shape = 입력 H×scale, W×scale (N, C는 동일)

---

### 10-3. InterpolateOpLowering (Gawee → Linalg) ⬚

| Task | Status | Files |
|------|--------|-------|
| Implement InterpolateOpLowering (nearest) | ⬚ Todo | lib/Conversion/GaweeToLinalg.cpp |
| Register pattern in pass | ⬚ Todo | lib/Conversion/GaweeToLinalg.cpp |

**Lowering strategy (nearest):**
```
gawee.interpolate(%x, scale=[2,2], mode="nearest")
  → %out = tensor.empty([N, C, H*2, W*2])
  → linalg.generic {
      // output index → input index
      src_h = out_h / scale_h   (integer division = nearest neighbor)
      src_w = out_w / scale_w
      yield input[n, c, src_h, src_w]
    }
```

**Key points:**
- `linalg.generic`에서 `linalg.index` 사용하여 출력 좌표 계산
- nearest: `src = dst / scale` (정수 나눗셈)
- bilinear은 나중에 — nearest부터 구현
- 이 lowering이 전체에서 **가장 어려운 부분**

---

### 10-4. Interpolate 테스트 ⬚

| Task | Status | Files |
|------|--------|-------|
| Hand-written interpolate test MLIR | ⬚ Todo | test/interpolate_test.mlir |
| Interpolate end-to-end with gawee-opt | ⬚ Todo | - |
| Summary document | ⬚ Todo | docs/Interpolate_Summary.md |
| Quiz file | ⬚ Todo | docs/Interpolate_Quiz.cpp |

```bash
# 테스트 명령어
./build/gawee-opt --convert-gawee-to-linalg test/interpolate_test.mlir
```

---

## Phase 11: Full UNet Pipeline ⬚

| Task | Status | Files |
|------|--------|-------|
| gawee-translate with UNet JSON | ⬚ Todo | - |
| gawee-opt Gawee → Linalg | ⬚ Todo | - |
| gawee-opt full pipeline → LLVM | ⬚ Todo | - |
| Verify all 116 nodes convert | ⬚ Todo | - |

```bash
# Step-by-step verification
./build/gawee-translate jsondata/graph.json
./build/gawee-translate jsondata/graph.json | ./build/gawee-opt --convert-gawee-to-linalg
./build/gawee-translate jsondata/graph.json | ./build/gawee-opt --gawee-to-llvm
```

---

## Recommended Order

```
9-1  gawee.cat TableGen 정의                      ← Easy
9-2  emitCat (MLIREmitter)                        ← Easy
9-3  CatOpLowering (insert_slice)                 ← Medium
9-4  Cat 테스트                                    ← Verify

10-1 gawee.interpolate TableGen 정의              ← Easy
10-2 emitInterpolate (MLIREmitter)                ← Easy
10-3 InterpolateOpLowering (nearest)              ← Hard
10-4 Interpolate 테스트                            ← Verify

11   Full UNet 파이프라인 통합 테스트              ← Final verify
```

---

## Difficulty Assessment

| Task | Difficulty | Reason |
|------|-----------|--------|
| Cat op definition (9-1) | Easy | Add op과 유사, variadic만 다름 |
| Cat emitter (9-2) | Easy | 기존 패턴 따르면 됨 |
| Cat lowering (9-3) | Medium | `tensor.insert_slice` 여러 번, offset 계산 |
| Interpolate op definition (10-1) | Easy | 일반적인 TableGen |
| Interpolate emitter (10-2) | Easy | 기존 패턴 따르면 됨 |
| Interpolate lowering (10-3) | Hard | `linalg.generic` + index 연산 |
| Full pipeline test (11) | Medium | 116 노드 전체 통과 확인 |
