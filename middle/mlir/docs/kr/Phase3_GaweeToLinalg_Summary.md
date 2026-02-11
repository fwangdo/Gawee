# Phase 3: Gawee → Linalg Lowering

## 개요

이 Phase에서는 고수준 Gawee op을 중간 수준 Linalg op으로 변환(lowering)한다.
핵심은 **OpConversionPattern** 클래스를 사용하여 각 Gawee op마다 변환 규칙을 정의하는 것이다.

변환 흐름:
```
gawee.conv      → tensor.pad + linalg.conv_2d_nchw_fchw + linalg.generic(bias)
gawee.relu      → linalg.generic(max(0, x))
gawee.add       → linalg.add
gawee.max_pool  → tensor.pad(-inf) + linalg.pooling_nchw_max
gawee.ad_avg_pool → linalg.pooling_nchw_sum + linalg.generic(div)
gawee.flatten   → tensor.collapse_shape
gawee.linear    → linalg.matmul_transpose_b + linalg.generic(bias)
```

---

## 1. OpConversionPattern의 구조

모든 lowering 패턴은 이 템플릿을 따른다:

```cpp
struct ConvOpLowering : public OpConversionPattern<gawee::ConvOp> {
  using OpConversionPattern::OpConversionPattern;  // 생성자 상속

  LogicalResult
  matchAndRewrite(gawee::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 1. adaptor에서 입력값 가져오기
    // 2. 새 op 생성
    // 3. rewriter.replaceOp(op, 결과)로 원본 교체
    return success();
  }
};
```

### op vs adaptor 차이

| | op | adaptor |
|---|---|---|
| 접근 대상 | 원본 Gawee op | 이미 변환된 값 |
| 용도 | 속성(Attribute) 읽기 | 피연산자(Value) 읽기 |
| 예시 | `op.getStridesAttr()` | `adaptor.getInput()` |

**왜 adaptor가 필요한가?** 변환 순서에 따라 입력 값이 이미 다른 타입으로 변환되어 있을 수 있다.
`adaptor`는 변환 후의 값을 제공하고, `op`는 원본 속성을 제공한다.

---

## 2. Destination-Passing Style

Linalg op은 **출력 텐서를 미리 만들어서 전달**해야 한다. 이것이 destination-passing style이다.

```cpp
// (1) 빈 텐서 생성
Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, shape, elementType);

// (2) 초기값으로 채우기 (conv는 0, max_pool은 -inf)
Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                        rewriter.getZeroAttr(elementType));
Value output = linalg::FillOp::create(rewriter, loc, zero, emptyTensor)
                   .getResult(0);

// (3) linalg op의 outs에 전달
auto conv = linalg::Conv2DNchwFchwOp::create(
    rewriter, loc, outputType,
    ValueRange{input, weight},  // ins (입력)
    output,                      // outs (출력 — 미리 만든 텐서)
    strides, dilations
);
```

**왜 destination-passing인가?**
- Bufferization(Phase 5)에서 `outs` 텐서를 그대로 memref로 변환할 수 있다.
- 별도 메모리 할당 없이 in-place 연산이 가능하다.

---

## 3. Padding 처리

`linalg.conv_2d_nchw_fchw`와 `linalg.pooling_nchw_max`는 **padding을 직접 처리하지 않는다.**
따라서 `tensor.pad`로 입력을 미리 패딩해야 한다.

### Conv의 padding (pad value = 0)

```cpp
int64_t padH = padding[0], padW = padding[1];
if (padH != 0 || padW != 0) {
  // NCHW 포맷: N과 C는 패딩 안 함, H와 W만 패딩
  SmallVector<int64_t> lowPad = {0, 0, padH, padW};
  SmallVector<int64_t> highPad = {0, 0, padH, padW};

  // 패딩 후 shape 계산
  SmallVector<int64_t> paddedShape(inputShape);
  paddedShape[2] += 2 * padH;  // H 차원
  paddedShape[3] += 2 * padW;  // W 차원

  // PadOp 생성
  auto padOp = rewriter.create<tensor::PadOp>(
      loc, paddedType, input, lowPad, highPad,
      ValueRange{}, ValueRange{});

  // body region: 패딩 위치에 채울 값 (zero) 지정
  auto &region = padOp.getRegion();
  auto *block = rewriter.createBlock(&region);
  for (int i = 0; i < rank; i++)
    block->addArgument(rewriter.getIndexType(), loc);
  rewriter.setInsertionPointToEnd(block);
  rewriter.create<tensor::YieldOp>(loc, zero);

  input = padOp.getResult();
  rewriter.setInsertionPointAfter(padOp);  // 삽입 지점 복원
}
```

### MaxPool의 padding (pad value = -inf)

Max pooling은 패딩 값이 **`-inf`**여야 한다.
0으로 패딩하면 음수값이 있을 때 패딩 위치의 0이 max로 선택되는 오류가 발생한다.

```cpp
auto negInf = arith::ConstantOp::create(rewriter, loc,
    rewriter.getFloatAttr(elementType, -std::numeric_limits<double>::infinity()));
```

### Identity Element (항등원) 개념

| 연산 | 항등원 | 이유 |
|------|--------|------|
| 덧셈 (conv, avg_pool) | 0 | x + 0 = x |
| 최대값 (max_pool) | -inf | max(x, -inf) = x |
| 곱셈 | 1 | x * 1 = x |

패딩 값과 output 초기값 모두 해당 연산의 항등원이어야 한다.

---

## 4. AffineMap과 Bias Broadcasting

Conv, Linear의 bias 더하기는 **차원이 다른** 텐서를 더해야 한다:
- Conv output: `[N, C, H, W]` (4D)
- Bias: `[C]` (1D)

`linalg.generic`의 `indexingMaps`로 broadcast를 정의한다:

```cpp
int64_t rank = 4;  // (n, c, h, w)
auto ctx = rewriter.getContext();

// bias: (n, c, h, w) -> (c)  — c 차원만 사용, n/h/w는 broadcast
AffineMap biasMap = AffineMap::get(rank, 0, {getAffineDimExpr(1, ctx)}, ctx);

// conv 결과 & output: (n, c, h, w) -> (n, c, h, w)  — 전체 차원 사용
AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

SmallVector<AffineMap> indexingMaps = {identityMap, biasMap, identityMap};
```

### AffineMap 매개변수 설명

`AffineMap::get(numDims, numSymbols, results, context)`
- `numDims`: 루프 차원 수 (4: n, c, h, w)
- `numSymbols`: 심볼 수 (보통 0)
- `results`: 어떤 차원을 사용하는지
- `getAffineDimExpr(1, ctx)`: dim1 = c 차원

### Iterator Types

```cpp
SmallVector<utils::IteratorType> iteratorTypes(
    rank, utils::IteratorType::parallel);
```

모든 차원이 `parallel`이면 독립적으로 병렬 처리 가능하다.
`reduction`이면 루프 간 데이터 의존성이 있다 (예: sum).

---

## 5. linalg.generic의 body

`linalg.generic`은 범용 루프 연산이다. body에서 element-wise 연산을 정의한다:

```cpp
auto genericOp = linalg::GenericOp::create(
    rewriter, loc,
    TypeRange{outputType},          // 결과 타입
    ValueRange{convResult, bias},   // inputs
    ValueRange{biasEmpty},          // outputs (destination)
    indexingMaps,                    // 각 텐서의 인덱싱
    iteratorTypes,                  // parallel or reduction
    // body builder — 각 원소에 대해 실행되는 코드
    [&](OpBuilder &builder, Location loc, ValueRange args) {
      // args[0] = convResult 원소, args[1] = bias 원소, args[2] = output (미사용)
      Value result = arith::AddFOp::create(builder, loc, args[0], args[1]);
      linalg::YieldOp::create(builder, loc, result);
    }
);
```

**args 순서**: inputs 순서 → outputs 순서. 위에서는 args[0]=conv, args[1]=bias, args[2]=dest.

---

## 6. 특수 lowering 패턴

### Flatten → tensor.collapse_shape

Flatten은 차원을 합치는 연산이다. `ReassociationIndices`로 어떤 차원들을 합칠지 명시한다:

```
입력: [1, 512, 1, 1]  flatten(startDim=1, endDim=-1)
결과: [1, 512]

reassociation = [[0], [1, 2, 3]]
  → dim 0은 그대로, dim 1+2+3을 합침
```

### AdaptiveAvgPool → PoolingNchwSum + Generic(div)

MLIR에 adaptive average pooling이 없으므로 직접 분해한다:
1. `linalg.pooling_nchw_sum`으로 합산
2. `linalg.generic`으로 원소별 나눗셈 (합 / 원소수)

### Linear → MatmulTransposeB + Generic(bias)

`y = xW^T + b` 연산:
1. `linalg.matmul_transpose_b`로 `xW^T` 계산
2. `linalg.generic`으로 bias 더하기 (Conv와 같은 broadcast 패턴)

---

## 7. Pass 정의 및 등록

모든 패턴을 묶어서 Pass로 만든다:

```cpp
struct GaweeToLinalgPass
    : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-gawee-to-linalg"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect>();   // 변환 후 허용
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<gawee::GaweeDialect>();   // 반드시 변환해야 함

    RewritePatternSet patterns(ctx);
    patterns.add<ConvOpLowering>(ctx);
    patterns.add<ReluOpLowering>(ctx);
    // ... 모든 패턴 등록

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
```

### ConversionTarget의 역할
- `addLegalDialect`: 변환 후 남아있어도 되는 dialect
- `addIllegalDialect`: 모든 op이 변환되어야 하는 dialect

`applyPartialConversion`은 illegal op이 남아있으면 실패를 보고한다.

---

## 핵심 개념 정리

- **OpConversionPattern** = 하나의 op 변환 규칙 (`matchAndRewrite`)
- **adaptor** = 이미 변환된 피연산자, **op** = 원본 속성
- **Destination-passing** = output 텐서를 미리 만들어 전달
- **Identity element** = 패딩/초기값에 사용 (conv=0, maxpool=-inf)
- **AffineMap** = 루프 인덱스와 텐서 인덱스의 매핑 (broadcasting 정의)
- **linalg.generic** = 범용 element-wise 연산, body에서 로직 정의
- **ReassociationIndices** = 어떤 차원을 합칠지 (collapse_shape에 사용)
