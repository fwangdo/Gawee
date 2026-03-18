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
