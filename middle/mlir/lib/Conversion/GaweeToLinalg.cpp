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
    // LEARNING: This is where you transform gawee.conv -> linalg.conv_2d
    //
    // Steps:
    //   1. Get input, weight tensors from adaptor
    //   2. Get attributes (strides, padding, dilation)
    //   3. Create output tensor (tensor.empty)
    //   4. Create linalg.conv_2d_nchw_fchw op
    //   5. Replace gawee.conv with the new ops
    //
    // Hints:
    //   - adaptor.getInput() gives the converted input value
    //   - op.getStrides() gives the strides attribute
    //   - rewriter.create<linalg::Conv2DNchwFchwOp>(...) creates new op
    //   - rewriter.replaceOp(op, newResult) replaces the old op

    Location loc = op.getLoc();

    // TODO: Get operands
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    // TODO: Get attributes
    auto strides = op.getStrides();
    auto padding = op.getPadding();

    // TODO: Compute output shape and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = rewriter.create<tensor::EmptyOp>(                                                                                                                                                                                                                                             
        loc,                                                                                                                                                                                                                                                                                     
        outputType.getShape(),                                                                                                                                                                                                                                                                   
        outputType.getElementType()                                                                                                                                                                                                                                                              
    );     
    // TODO: Create linalg.conv_2d
    // auto conv = rewriter.create<linalg::Conv2DNchwFchwOp>(
    //     loc, outputType, ValueRange{input, weight}, output, strides, dilations);

    // TODO: Replace original op
    // rewriter.replaceOp(op, conv.getResult(0));

    return failure(); // TODO: return success() when implemented
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
    // LEARNING: ReLU -> linalg.generic with max(0, x) body
    //
    // linalg.generic is the "Swiss army knife" of Linalg.
    // It can express any elementwise operation.
    //
    // Structure:
    //   linalg.generic {
    //     indexing_maps = [identity, identity],  // input and output maps
    //     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    //   } ins(%input) outs(%output) {
    //     ^bb0(%in: f32, %out: f32):
    //       %zero = arith.constant 0.0 : f32
    //       %result = arith.maximumf %in, %zero : f32
    //       linalg.yield %result : f32
    //   }
    //
    // Hints:
    //   - Create indexing maps with AffineMap::getMultiDimIdentityMap()
    //   - Use rewriter.create<linalg::GenericOp>(...)
    //   - Build the body block with rewriter.createBlock()

    Location loc = op.getLoc();

    // TODO: Implement ReLU as linalg.generic

    return failure(); // TODO: return success() when implemented
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
    // LEARNING: This one is easy - Linalg has linalg.add directly
    //
    // rewriter.replaceOpWithNewOp<linalg::AddOp>(op, inputs, outputs);

    // TODO: Implement

    return failure(); // TODO: return success() when implemented
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
