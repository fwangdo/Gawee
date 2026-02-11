// ==========================================================================
// Phase 8 퀴즈: ResNet 확장 — MaxPool, AdAvgPool, Flatten 구현
// ==========================================================================
//
// 세 가지 새 op의 lowering을 직접 구현하라.
// Phase 3에서 배운 패턴을 적용하되, 각 op의 고유한 특성을 처리해야 한다:
//   - MaxPool: -inf 패딩, window tensor
//   - AdAvgPool: 분해(sum + div), MLIR에 없는 op 처리
//   - Flatten: ReassociationIndices, 음수 인덱스 처리
//

#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include <limits>

using namespace mlir;

namespace {

// ==========================================================================
// 문제 1: MaxPoolOpLowering
// ==========================================================================
//
// Conv의 패딩과 비교하여, MaxPool의 패딩이 다른 점에 주의하라.
// 빈칸을 채워 완성하시오.

struct MaxPoolOpLowering : public OpConversionPattern<gawee::MaxPoolOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::MaxPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    auto strides = adaptor.getStridesAttr();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilationAttr();

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    // (a) MaxPool의 identity element는 무엇인가?
    // Conv는 0을 사용했다. MaxPool은 _____를 사용한다.
    // 이유: _____
    auto negInf = arith::ConstantOp::create(rewriter, loc,
        rewriter.getFloatAttr(elementType,
            _____));  // (b) C++에서 -inf 표현

    // (c) 패딩: Conv와 구조는 동일하지만 패딩 값이 다르다.
    int64_t padH = padding[0];
    int64_t padW = padding[1];
    if (padH != 0 || padW != 0) {
      SmallVector<int64_t> lowPad = {0, 0, padH, padW};
      SmallVector<int64_t> highPad = {0, 0, padH, padW};

      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> paddedShape(inputType.getShape());
      paddedShape[2] += 2 * padH;
      paddedShape[3] += 2 * padW;
      auto paddedType = RankedTensorType::get(paddedShape, elementType);

      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, input, lowPad, highPad,
          ValueRange{}, ValueRange{});
      auto &region = padOp.getRegion();
      auto *block = rewriter.createBlock(&region);
      for (int i = 0; i < paddedType.getRank(); i++)
        block->addArgument(rewriter.getIndexType(), loc);
      rewriter.setInsertionPointToEnd(block);
      // (d) Conv에서는 zero를 yield했다. MaxPool에서는 무엇을 yield?
      rewriter.create<tensor::YieldOp>(loc, _____.getResult());
      input = padOp.getResult();
      rewriter.setInsertionPointAfter(padOp);
    }

    // (e) output을 무엇으로 초기화해야 하는가?
    // Conv는 zero로 초기화했다. MaxPool은 _____로 초기화한다.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    auto filledOutput = linalg::FillOp::create(
        rewriter, loc, _____.getResult(), emptyTensor);  // (f) 초기화 값

    // (g) window tensor: pooling op에 커널 크기를 알려주는 빈 텐서
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, adaptor._____(), elementType);  // (h) 커널 크기 getter

    // (i) linalg의 어떤 op을 사용하는가?
    auto maxPoolOp = _____::create(  // 어떤 op?
        rewriter, loc, outputType,
        ValueRange{input, windowTensor},
        filledOutput.getResult(0),
        strides, dilation
    );

    rewriter.replaceOp(op, maxPoolOp.getResults());
    return success();
  }
};


// ==========================================================================
// 문제 2: AdAvgPoolOpLowering
// ==========================================================================
//
// MLIR에 adaptive average pooling이 없으므로 직접 분해한다.
// 분해 전략: sum pooling + 원소별 나눗셈
//
// 빈칸을 채워 완성하시오.

struct AdAvgOpLowering : public OpConversionPattern<gawee::AdAvgPoolOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::AdAvgPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();

    // (a) 입력의 H, W 차원 추출 (NCHW 포맷)
    int64_t H = inputType.getShape()[_____];  // H 차원 인덱스
    int64_t W = inputType.getShape()[_____];  // W 차원 인덱스

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    // (b) sum pooling을 위한 destination 준비
    // sum의 identity element는 _____이다.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getZeroAttr(elementType));
    Value zeroFilled = linalg::FillOp::create(
        rewriter, loc, zero, emptyTensor).getResult(0);

    // (c) window tensor 크기는?
    // adaptive avg pool(output_size=[1,1])이면 전체 H x W를 커버해야 한다.
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{_____, _____}, elementType);  // 크기는?

    auto strideAttr = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilationAttr = rewriter.getDenseI64ArrayAttr({1, 1});

    // (d) 어떤 linalg op으로 합산하는가?
    auto sumPool = _____::create(  // 어떤 op?
        rewriter, loc, outputType,
        ValueRange{input, windowTensor},
        ValueRange{zeroFilled},
        strideAttr, dilationAttr
    );

    // (e) 나눗셈: sum / (H * W)
    int64_t count = _____ * _____;  // 무엇을 곱하는가?
    Value countVal = arith::ConstantOp::create(
        rewriter, loc, elementType,
        rewriter.getFloatAttr(elementType, static_cast<double>(count)));

    Value divEmpty = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);

    int64_t rank = outputType.getRank();
    SmallVector<AffineMap, 2> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // (f) linalg.generic으로 나눗셈
    auto divOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{sumPool->getResults()[0]},
        ValueRange{divEmpty},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = sum 원소, args[1] = dest (미사용)
          Value avg = _____::create(builder, loc, args[0], countVal);  // (g) 나눗셈 op
          linalg::YieldOp::create(builder, loc, avg);
        }
    );

    rewriter.replaceOp(op, divOp.getResults());
    return success();
  }
};


// ==========================================================================
// 문제 3: FlattenOpLowering
// ==========================================================================
//
// tensor.collapse_shape와 ReassociationIndices를 사용한다.
// 빈칸을 채워 완성하시오.

struct FlattenOpLowering : public OpConversionPattern<gawee::FlattenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::FlattenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto startDim = adaptor.getStartDimAttr();
    auto endDim = adaptor.getEndDimAttr();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    // (a) 음수 endDim 처리
    int64_t endDimInt = endDim.getInt();
    int64_t rank = inputType.getRank();
    if (endDimInt < 0) {
      endDimInt _____ rank;  // 어떤 연산?
    }

    // (b) ReassociationIndices 구축
    // 예: shape=[1,512,1,1], startDim=1, endDim=3(또는 -1)
    // reassociation = [[0], [1, 2, 3]]
    SmallVector<ReassociationIndices> reassociation;

    // startDim 이전: 각 차원을 독립적으로 유지
    for (int64_t i = 0; i < startDim.getInt(); i++) {
      reassociation.push_back({_____});  // (c) 무엇을 push?
    }

    // startDim ~ endDim: 하나로 합침
    ReassociationIndices mergedGroup;
    for (int64_t i = _____; i <= _____; i++) {  // (d) 범위는?
      mergedGroup.push_back(_____);              // (e) 무엇을 push?
    }
    reassociation.push_back(mergedGroup);

    // endDim 이후: 각 차원을 독립적으로 유지
    for (auto i = _____ + 1; i < _____; i++) {  // (f) 범위는?
      reassociation.push_back({i});
    }

    // (g) 어떤 tensor op을 사용하는가?
    auto flattenOp = rewriter.create<tensor::_____>(  // 어떤 op?
        loc, outputType, input, reassociation);

    rewriter.replaceOp(op, flattenOp.getResult());
    return success();
  }
};


// ==========================================================================
// 문제 4: 개념 문제
// ==========================================================================
//
// (a) 새 op을 추가할 때 수정해야 하는 3개 파일은?
//     1. _____
//     2. _____
//     3. _____
//
// (b) Identity element(항등원)을 올바르게 짝지으시오:
//     덧셈(sum, conv):  _____
//     최대값(max):       _____
//     곱셈(multiply):    _____
//
// (c) window tensor의 역할은?
//     _____
//
// (d) tensor.collapse_shape vs tensor.reshape의 차이는?
//     _____
//
// (e) LinearOp에서 MatmulOp 대신 MatmulTransposeBOp를 사용하는 이유는?
//     _____
//
// (f) JSON의 kernel_size가 스칼라 3일 수도 있고 배열 [3,3]일 수도 있다.
//     이를 어떻게 처리하는가?
//     _____
//

} // namespace


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) -inf (음의 무한대). max(x, -inf) = x이므로 패딩 위치가 결과에 영향을 주지 않는다.
  (b) -std::numeric_limits<double>::infinity()
  (d) negInf  (negInf.getResult())
  (f) negInf  (negInf.getResult())
  (h) getKernelSize
  (i) linalg::PoolingNchwMaxOp

문제 2:
  (a) 2, 3 (NCHW에서 H=index 2, W=index 3)
  (b) 0 (합산의 항등원)
  (c) H, W
  (d) linalg::PoolingNchwSumOp
  (e) H * W
  (g) arith::DivFOp

문제 3:
  (a) += (endDimInt += rank)
  (c) i (각 차원을 단독 그룹으로)
  (d) startDim.getInt(); endDimInt
  (e) i (합칠 차원 인덱스)
  (f) endDimInt; rank
  (g) CollapseShapeOp

문제 4:
  (a) 1. GaweeOps.td (op 정의)
      2. GaweeToLinalg.cpp (lowering 패턴)
      3. MLIREmitter.cpp (JSON → Gawee emit)
  (b) 덧셈: 0
      최대값: -inf
      곱셈: 1
  (c) pooling op에 커널 크기를 알려주는 빈 텐서.
      실제 데이터는 담지 않으며, shape만 사용된다.
  (d) collapse_shape은 연속된 차원만 합칠 수 있다 (ReassociationIndices).
      reshape는 임의의 shape 변환이 가능하지만, 더 복잡한 lowering이 필요하다.
      Flatten처럼 연속 차원을 합치는 경우 collapse_shape이 적합하다.
  (e) PyTorch Linear의 weight shape이 [out_features, in_features]이기 때문.
      일반 Matmul(A x B)은 B가 [in, out] 형태여야 하므로 전치가 필요하다.
      MatmulTransposeBOp는 자동으로 B를 전치해준다.
  (f) getArray를 먼저 시도하고 실패하면 getInteger로 스칼라를 읽어
      [scalar, scalar]로 복제한다 (H, W에 동일 적용).

*/
