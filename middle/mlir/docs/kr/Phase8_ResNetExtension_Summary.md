# Phase 8: ResNet 확장 (새 Op 추가 패턴)

## 개요

Phase 3까지는 Conv, ReLU, Add만 있었다.
ResNet-18을 지원하려면 MaxPool, AdaptiveAvgPool, Flatten, Linear이 추가로 필요하다.
이 Phase에서는 **새 op을 추가하는 전체 패턴**을 배운다.

새 op 추가 = `.td 정의` + `lowering 패턴` + `emitter 함수` (3곳 수정)

---

## 1. 새 Op 추가 체크리스트

어떤 연산이든 추가할 때 이 3단계를 따른다:

### Step 1: GaweeOps.td에 Op 정의

```tablegen
def Gawee_MaxPoolOp : Gawee_Op<"max_pool", []> {
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernelSize,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    BoolAttr:$ceilMode
  );
  let results = (outs AnyTensor:$output);
}
```

### Step 2: GaweeToLinalg.cpp에 lowering 패턴

```cpp
struct MaxPoolOpLowering : public OpConversionPattern<gawee::MaxPoolOp> {
  // matchAndRewrite 구현
};

// runOnOperation에서 등록:
patterns.add<MaxPoolOpLowering>(ctx);
```

### Step 3: MLIREmitter.cpp에 emit 함수

```cpp
bool MLIREmitter::emitMaxPool(const llvm::json::Object &node,
                               const llvm::json::Object &values) {
  // JSON → Gawee op 생성
}

// emitNode에서 디스패치:
if (*opType == "MaxPool") return emitMaxPool(node, values);
```

### Step 4: build.sh 재실행

tblgen이 `.td` 변경을 반영하여 `.inc` 파일을 재생성한다.

---

## 2. MaxPool: -inf 패딩

### 핵심 개념: identity element

MaxPool의 identity element(항등원)는 **`-inf`**이다.
`max(x, -inf) = x`이므로, 패딩 위치의 `-inf`는 max 연산에 영향을 주지 않는다.

```cpp
// -inf 상수 생성
auto negInf = arith::ConstantOp::create(rewriter, loc,
    rewriter.getFloatAttr(elementType,
        -std::numeric_limits<double>::infinity()));
```

### window tensor

Pooling에는 **window tensor** (커널 크기만큼의 빈 텐서)가 필요하다:

```cpp
Value windowTensor = tensor::EmptyOp::create(
    rewriter, loc, adaptor.getKernelSize(), elementType);
```

이 텐서는 실제 값을 담지 않는다. Pooling op에 커널 크기를 알려주는 역할이다.

### 전체 lowering

```
gawee.max_pool
→ tensor.pad(input, -inf)        // 패딩
→ tensor.empty + linalg.fill(-inf) // output을 -inf로 초기화
→ tensor.empty (window)            // 커널 크기 텐서
→ linalg.pooling_nchw_max          // max pooling 실행
```

---

## 3. AdaptiveAvgPool: 분해(decomposition)

MLIR에 `adaptive_average_pool`이 없다. 직접 분해해야 한다.

### 분해 전략

Adaptive average pool(output_size=[1,1])은:
1. 전체 H, W에 대해 sum pooling
2. 결과를 `H * W`로 나눔

```
gawee.ad_avg_pool
→ tensor.empty + linalg.fill(0)    // output을 0으로 초기화
→ tensor.empty (window: H x W)     // 입력 전체 크기의 window
→ linalg.pooling_nchw_sum           // 합산
→ linalg.generic(div by H*W)        // 평균 계산
```

### linalg.generic으로 나눗셈

```cpp
int64_t count = H * W;  // 나눌 값
Value countVal = arith::ConstantOp::create(rewriter, loc, elementType,
    rewriter.getFloatAttr(elementType, static_cast<double>(count)));

linalg::GenericOp::create(rewriter, loc, TypeRange{outputType},
    ValueRange{sumPool->getResults()[0]},  // input: sum 결과
    ValueRange{divEmpty},                   // output: destination
    indexingMaps, iteratorTypes,
    [&](OpBuilder &builder, Location loc, ValueRange args) {
      Value avg = arith::DivFOp::create(builder, loc, args[0], countVal);
      linalg::YieldOp::create(builder, loc, avg);
    });
```

---

## 4. Flatten: tensor.collapse_shape와 ReassociationIndices

### ReassociationIndices란?

어떤 차원들을 하나로 합칠지 명시하는 배열의 배열:

```
입력: [1, 512, 1, 1]
flatten(startDim=1, endDim=-1)
결과: [1, 512]

reassociation = [[0], [1, 2, 3]]
  dim 0 → 그대로
  dim 1, 2, 3 → 합침 (512 * 1 * 1 = 512)
```

### 구현

```cpp
SmallVector<ReassociationIndices> reassociation;

// startDim 전: 각 차원을 독립적으로 유지
for (int64_t i = 0; i < startDim; i++)
  reassociation.push_back({i});

// startDim ~ endDim: 하나로 합침
ReassociationIndices mergedGroup;
for (int64_t i = startDim; i <= endDim; i++)
  mergedGroup.push_back(i);
reassociation.push_back(mergedGroup);

// endDim 후: 각 차원을 독립적으로 유지
for (auto i = endDim + 1; i < rank; i++)
  reassociation.push_back({i});

auto flattenOp = rewriter.create<tensor::CollapseShapeOp>(
    loc, outputType, input, reassociation);
```

### 음수 endDim 처리

```cpp
int64_t endDimInt = endDim.getInt();
if (endDimInt < 0)
  endDimInt += rank;  // -1 → 마지막 차원
```

Python의 음수 인덱싱과 동일한 규칙이다.

---

## 5. Linear: MatmulTransposeB

`y = xW^T + b`에서 weight가 `[out_features, in_features]` 형태이므로
일반 matmul 대신 **transpose-B matmul**을 사용한다:

```cpp
auto matmul = linalg::MatmulTransposeBOp::create(
    rewriter, loc, outputType,
    ValueRange{input, weight},  // input: [B, in], weight: [out, in]
    filledZero                   // output: [B, out]
);
```

`MatmulTransposeBOp`는 자동으로 두 번째 인자를 전치한다.
일반 `MatmulOp`을 쓰면 weight를 `[in_features, out_features]`로 전치해야 한다.

---

## 6. Emitter 주의사항

### JSON 필드 이름 불일치

| JSON (snake_case) | TableGen (camelCase) |
|---|---|
| `kernel_size` | `$kernelSize` |
| `ceil_mode` | `$ceilMode` |
| `start_dim` | `$startDim` |
| `output_size` | `$outputSize` |

### 스칼라 vs 배열 처리

MaxPool의 JSON에서 `kernel_size`가 정수 `3`일 수도, 배열 `[3, 3]`일 수도 있다:

```cpp
auto getI64OrArray = [&](const char *name) -> SmallVector<int64_t> {
  SmallVector<int64_t> result;
  if (const auto *arr = attrs->getArray(name)) {
    for (const auto &v : *arr)
      if (auto i = v.getAsInteger()) result.push_back(*i);
  } else if (auto scalar = attrs->getInteger(name)) {
    result.push_back(*scalar);
    result.push_back(*scalar);  // H, W에 동일하게 적용
  }
  return result;
};
```

### I64Attr vs DenseI64ArrayAttr

```cpp
// DenseI64ArrayAttr (배열): stride, padding 등
builder->getDenseI64ArrayAttr(strides)   // SmallVector<int64_t> → Attribute

// I64Attr (단일 값): startDim, endDim 등
builder->getI64IntegerAttr(startDim)     // int64_t → Attribute
```

---

## 7. 정리: Op별 lowering 대응표

| Gawee Op | Linalg/Tensor Op | 패딩값 | 특이사항 |
|----------|-----------------|--------|----------|
| conv | conv_2d_nchw_fchw + generic(bias) | 0 | AffineMap broadcast |
| relu | generic(max(0, x)) | 없음 | elementwise |
| add | linalg.add | 없음 | 직접 대응 |
| max_pool | pooling_nchw_max | -inf | window tensor 필요 |
| ad_avg_pool | pooling_nchw_sum + generic(div) | 0 | 분해 필요 |
| flatten | tensor.collapse_shape | 없음 | ReassociationIndices |
| linear | matmul_transpose_b + generic(bias) | 0 | TransposeB 사용 |

---

## 핵심 개념 정리

- **새 op 추가 패턴**: `.td` → lowering → emitter (항상 3곳)
- **Identity element**: 패딩/초기값은 연산의 항등원 (conv=0, max=-inf)
- **Window tensor**: pooling의 커널 크기를 지정하는 빈 텐서
- **분해(decomposition)**: MLIR에 없는 op은 기존 op 조합으로 구현
- **ReassociationIndices**: collapse_shape에서 합칠 차원 그룹 지정
- **MatmulTransposeB**: weight가 `[out, in]` 형태일 때 자동 전치
- **JSON snake_case ↔ TableGen camelCase**: 수동 매핑 필요
