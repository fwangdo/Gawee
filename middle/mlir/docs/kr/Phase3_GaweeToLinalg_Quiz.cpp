// ==========================================================================
// Phase 3 퀴즈: ConvOpLowering 전체 재구현
// ==========================================================================
//
// Conv lowering을 처음부터 작성하라.
// padding 처리, destination-passing, AffineMap bias broadcast를 모두 포함해야 한다.
//
// 이 파일은 실제 빌드용이 아닌 학습 확인용이다.
//

#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// ==========================================================================
// 문제 1: ConvOpLowering 구조
// ==========================================================================
//
// OpConversionPattern을 상속하는 ConvOpLowering을 작성하라.
// 빈칸을 채우시오.

struct ConvOpLowering : public _____<gawee::ConvOp> {  // (a) 어떤 클래스를 상속?
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ConvOp op, _____ adaptor,    // (b) 어떤 타입?
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // ==========================================================================
    // 문제 2: 피연산자 가져오기
    // ==========================================================================
    // op과 adaptor의 차이를 이해하고 올바른 곳에서 값을 가져오라.

    // input은 이미 변환된 값이므로 _____에서 가져옴
    Value input = _____.getInput();           // (c)
    Value weight = _____.getWeight();         // (d)
    Value bias = _____.getBias();             // (e)

    // strides는 속성(Attribute)이므로 _____에서 가져옴
    auto strides = _____.getStridesAttr();    // (f)
    auto dilations = _____.getDilationAttr(); // (g)
    auto padding = _____.getPadding();        // (h) ArrayRef<int64_t>

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    // ==========================================================================
    // 문제 3: Padding 처리
    // ==========================================================================
    // linalg.conv_2d_nchw_fchw는 패딩을 자체 처리하지 않는다.
    // tensor.pad로 입력을 미리 패딩해야 한다.

    // conv의 패딩 값(identity element)은 _____
    Value zero = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getZeroAttr(elementType));  // (i)

    int64_t padH = padding[0];
    int64_t padW = padding[1];
    if (padH != 0 || padW != 0) {
      // NCHW 포맷에서 N, C는 패딩 안 하고 H, W만 패딩
      SmallVector<int64_t> lowPad = {_____, _____, _____, _____};   // (j) 각 값은?
      SmallVector<int64_t> highPad = {_____, _____, _____, _____};  // (k) 각 값은?

      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> paddedShape(inputType.getShape());
      paddedShape[_____] += 2 * padH;  // (l) 어떤 인덱스?
      paddedShape[_____] += 2 * padW;  // (m) 어떤 인덱스?
      auto paddedType = RankedTensorType::get(paddedShape, elementType);

      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, input, lowPad, highPad,
          ValueRange{}, ValueRange{});

      // PadOp의 body: 패딩 위치에 채울 값을 yield
      auto &region = padOp.getRegion();
      auto *block = rewriter.createBlock(&region);
      for (int i = 0; i < paddedType.getRank(); i++)
        block->addArgument(rewriter.getIndexType(), loc);
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<tensor::YieldOp>(loc, _____);  // (n) 무엇을 yield?

      input = padOp.getResult();
      rewriter.setInsertionPointAfter(padOp);
    }

    // ==========================================================================
    // 문제 4: Destination-Passing Style
    // ==========================================================================
    // output 텐서를 미리 만들어 conv op에 전달한다.

    // 빈 텐서 생성
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);

    // 초기값으로 채우기
    Value output = _____::create(rewriter, loc, zero, emptyTensor)  // (o) 어떤 op?
                       .getResult(0);

    // conv 생성
    auto conv = _____::create(              // (p) linalg의 어떤 op?
        rewriter, loc, outputType,
        ValueRange{input, weight},           // ins
        output,                              // outs (destination-passing!)
        strides, dilations
    );

    // ==========================================================================
    // 문제 5: Bias Broadcasting with AffineMap
    // ==========================================================================
    // bias는 [C] 형태, conv 출력은 [N, C, H, W] 형태.
    // AffineMap으로 broadcast를 정의한다.

    Value convResult = conv.getResults()[0];
    int64_t rank = outputType.getRank();

    Value biasEmpty = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);

    auto ctx = rewriter.getContext();

    // bias: (n, c, h, w) -> (c) 만 사용
    // getAffineDimExpr의 인자: n=0, c=1, h=2, w=3
    AffineMap biasMap = AffineMap::get(
        rank, 0,
        {getAffineDimExpr(_____, ctx)},  // (q) 몇 번째 dim?
        ctx
    );

    // conv 결과와 output은 전체 차원 사용
    AffineMap identityMap = AffineMap::_____(rank, ctx);  // (r) 어떤 팩토리 메서드?

    SmallVector<AffineMap> indexingMaps = {_____, _____, _____};  // (s) 순서: conv, bias, out

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::_____);  // (t) parallel or reduction?

    auto biasAdd = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{convResult, bias},   // inputs
        ValueRange{biasEmpty},          // output
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = _____, args[1] = _____, args[2] = _____  // (u) 각각 무엇?
          Value result = _____::create(builder, loc, args[0], args[1]);  // (v) 어떤 op?
          linalg::YieldOp::create(builder, loc, result);
        }
    );

    // ==========================================================================
    // 문제 6: 원본 op 교체
    // ==========================================================================

    rewriter._____(op, biasAdd.getResults());  // (w) 어떤 메서드?

    return _____();  // (x) 성공 반환
  }
};


// ==========================================================================
// 문제 7: Pass 등록
// ==========================================================================
// 빈칸을 채워 Pass를 완성하시오.

struct GaweeToLinalgPass
    : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "_____"; }  // (a) CLI 플래그

  void getDependentDialects(DialectRegistry &registry) const override {
    // (b) 이 pass가 생성하는 dialect 3개를 등록하시오
    registry.insert<_____>();
    registry.insert<_____>();
    registry.insert<_____>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // (c) 변환 후 합법적인 것과 불법적인 것을 정의
    ConversionTarget target(*ctx);
    target._____<linalg::LinalgDialect>();     // 합법
    target._____<gawee::GaweeDialect>();       // 불법

    RewritePatternSet patterns(ctx);
    patterns._____<ConvOpLowering>(ctx);       // (d) 패턴 추가

    if (_____(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();                     // (e) 실패 검사
  }
};

} // namespace


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) OpConversionPattern
  (b) OpAdaptor

문제 2:
  (c) adaptor
  (d) adaptor
  (e) adaptor
  (f) op
  (g) op
  (h) op

문제 3:
  (i) conv의 패딩 값(identity element)은 0 (덧셈의 항등원)
  (j) {0, 0, padH, padW}
  (k) {0, 0, padH, padW}
  (l) 2 (H 차원)
  (m) 3 (W 차원)
  (n) zero

문제 4:
  (o) linalg::FillOp
  (p) linalg::Conv2DNchwFchwOp

문제 5:
  (q) 1 (c 차원)
  (r) getMultiDimIdentityMap
  (s) {identityMap, biasMap, identityMap}  (conv=identity, bias=biasMap, out=identity)
  (t) parallel
  (u) args[0] = conv 원소, args[1] = bias 원소, args[2] = output (미사용)
  (v) arith::AddFOp

문제 6:
  (w) replaceOp
  (x) success

문제 7:
  (a) "convert-gawee-to-linalg"
  (b) linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect
  (c) addLegalDialect, addIllegalDialect
  (d) add
  (e) failed

*/
