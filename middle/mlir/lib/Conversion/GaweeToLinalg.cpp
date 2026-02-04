//===----------------------------------------------------------------------===//
// Gawee to Linalg Conversion Pass
//===----------------------------------------------------------------------===//
//
// LEARNING: Conversion Passes
//
// A conversion pass transforms ops from one dialect to another.
// This is "lowering" - going from high-level to low-level representation.
//
// Key MLIR concepts:
//   - ConversionPattern: describes how to rewrite one op
//   - TypeConverter: describes how to convert types
//   - ConversionTarget: specifies which ops are legal after conversion
//
// Flow:
//   gawee.conv -> linalg.conv_2d_nchw_fchw
//   gawee.relu -> linalg.generic (with max(0, x) body)
//   gawee.add  -> linalg.add
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
//
// LEARNING: Each pattern handles one op type.
//
// Pattern anatomy:
//   1. Match: find the op to convert (done by template parameter)
//   2. Rewrite: create new ops, replace old op
//

//===----------------------------------------------------------------------===//
// Conv2D Lowering
//===----------------------------------------------------------------------===//

struct ConvOpLowering : public OpConversionPattern<gawee::ConvOp> {
  using OpConversionPattern::OpConversionPattern;

  // LogicalResult returns success() / failure()
  LogicalResult
  matchAndRewrite(gawee::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get operands from adaptor (already converted values)
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    // Step 2: Get attributes (use *Attr() to get Attribute, not ArrayRef)
    auto strides = op.getStridesAttr();
    auto dilations = op.getDilationAttr();

    // Step 3: Create output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        outputType.getShape(),
        outputType.getElementType()
    );

    // Step 4: Create linalg.conv_2d_nchw_fchw
    auto conv = rewriter.create<linalg::Conv2DNchwFchwOp>(
        loc,
        outputType,
        ValueRange{input, weight},  // ins
        output,                      // outs
        strides,
        dilations
    );

    // Step 5: Replace original op with conv result
    rewriter.replaceOp(op, conv.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReLU Lowering
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpConversionPattern<gawee::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get input and its type
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    int64_t rank = inputType.getRank();

    // Step 2: Create output tensor (same shape as input)
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        inputType.getShape(),
        elementType
    );

    // Step 3: Create indexing maps (identity maps for elementwise op)
    SmallVector<AffineMap, 2> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
    );

    // Step 4: Create iterator types (all parallel for elementwise)
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel
    );

    // Step 5: Create linalg.generic op
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTypes=*/TypeRange{inputType},
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = output element (unused)
          Value inVal = args[0];

          // Create zero constant
          Value zero = builder.create<arith::ConstantOp>(
              loc,
              elementType,
              builder.getZeroAttr(elementType)
          );

          // max(0, x)
          Value result = builder.create<arith::MaximumFOp>(loc, inVal, zero);

          // Yield result
          builder.create<linalg::YieldOp>(loc, result);
        }
    );

    // Step 6: Replace original op
    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Add Lowering
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<gawee::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get operands
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        outputType.getShape(),
        outputType.getElementType()
    );

    // Step 3: Create linalg.add
    auto addOp = rewriter.create<linalg::AddOp>(
        loc,
        TypeRange{outputType},
        ValueRange{lhs, rhs},  // ins
        ValueRange{output}      // outs
    );

    // Step 4: Replace original op
    rewriter.replaceOp(op, addOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// TODO: Add more lowering patterns
//===----------------------------------------------------------------------===//
//
// - MaxPoolOpLowering -> linalg.pooling_nchw_max
// - LinearOpLowering -> linalg.matmul + linalg.broadcast(bias)
//

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
//
// LEARNING: A Pass wraps the conversion patterns and runs them.
//

struct GaweeToLinalgPass
    : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-gawee-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower Gawee dialect to Linalg dialect";
  }

  void runOnOperation() override {
    // LEARNING: Conversion setup
    //
    // 1. ConversionTarget: defines what ops are "legal" after conversion
    // 2. RewritePatternSet: collection of patterns to apply
    // 3. applyPartialConversion: run the conversion

    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Define what's legal after conversion
    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<gawee::GaweeDialect>();  // Gawee ops must be converted

    // Collect patterns
    RewritePatternSet patterns(ctx);
    patterns.add<ConvOpLowering>(ctx);
    patterns.add<ReluOpLowering>(ctx);
    patterns.add<AddOpLowering>(ctx);
    // TODO: Add more patterns

    // Run conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//
//
// LEARNING: Registration makes the pass available to mlir-opt tool.
//
// After registration:
//   mlir-opt --convert-gawee-to-linalg input.mlir
//

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass() {
  return std::make_unique<GaweeToLinalgPass>();
}
} // namespace mlir::gawee

// TODO: Add this to a Passes.h header for external use
