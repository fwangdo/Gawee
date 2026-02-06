//===----------------------------------------------------------------------===//
// GaweeToLinalg Quiz
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the conversion pass.
// This mirrors the actual structure of lib/Conversion/GaweeToLinalg.cpp
//
// After completing, compare with the real implementation.
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
// Q1: Conv2D Lowering (with zero-initialization!)
//===----------------------------------------------------------------------===//
//
// IMPORTANT: Conv is a reduction op - it accumulates into output.
// Output MUST be zero-initialized for correct results AND bufferization.
//

struct ConvOpLowering : public OpConversionPattern<gawee::ConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q1a: Get input and weight from adaptor
    Value input = adaptor.???();
    Value weight = adaptor.???();

    // Q1b: Get strides and dilations as Attributes
    // HINT: Use *Attr() suffix to get the attribute
    auto strides = op.???();
    auto dilations = op.???();

    // Q1c: Get output type
    auto outputType = mlir::cast<???>(op.getOutput().getType());
    auto elementType = outputType.???();

    // Q1d: Create empty output tensor
    // HINT: tensor::EmptyOp::create(rewriter, loc, shape, elementType)
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.???(),
        elementType
    );

    // Q1e: Create zero constant for initialization
    // HINT: arith::ConstantOp::create(rewriter, loc, type, zeroAttr)
    Value zero = arith::ConstantOp::create(
        rewriter, loc,
        elementType,
        rewriter.???(elementType)
    );

    // Q1f: Fill output with zeros using linalg.fill
    // WHY? Conv accumulates into output, so must start at zero
    // HINT: linalg::FillOp::create(rewriter, loc, fillValue, destTensor)
    Value output = linalg::FillOp::create(rewriter, loc, ???, ???)
                       .getResult(0);

    // Q1g: Create linalg.conv_2d_nchw_fchw
    auto conv = linalg::Conv2DNchwFchwOp::create(
        rewriter, loc,
        outputType,
        ValueRange{???, ???},  // ins: input, weight
        ???,                    // outs: zero-initialized output
        strides,
        dilations
    );

    // Q1h: Replace original op
    rewriter.???(op, conv.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Q2: ReLU Lowering (using linalg.generic)
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpConversionPattern<gawee::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q2a: Get input and its properties
    Value input = adaptor.???();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.???();
    int64_t rank = inputType.???();

    // Q2b: Create output tensor (same shape as input)
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        inputType.???(),
        ???
    );

    // Q2c: Create identity indexing maps for elementwise operation
    // HINT: Both input and output use the same identity map
    SmallVector<AffineMap, 2> indexingMaps(
        ???,  // How many maps? (1 input + 1 output)
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
    );

    // Q2d: Create iterator types (all parallel for elementwise)
    SmallVector<utils::IteratorType> iteratorTypes(
        rank,
        utils::IteratorType::???
    );

    // Q2e: Create linalg.generic with body
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{inputType},
        ValueRange{???},    // inputs
        ValueRange{???},    // outputs
        indexingMaps,
        iteratorTypes,
        // Body builder lambda
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = output element (unused)
          Value inVal = args[???];

          // Q2f: Create zero constant
          Value zero = builder.create<arith::ConstantOp>(
              loc, elementType, builder.???(elementType)
          );

          // Q2g: Compute max(input, zero) - this is ReLU!
          Value result = builder.create<arith::???>(loc, inVal, zero);

          // Q2h: Yield result
          builder.create<linalg::???>(loc, result);
        }
    );

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Q3: Add Lowering (using linalg.add)
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<gawee::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q3a: Get lhs and rhs from adaptor
    Value lhs = adaptor.???();
    Value rhs = adaptor.???();

    // Q3b: Get output type (from op, not adaptor!)
    // WHY from op? adaptor only has converted inputs, not output type
    auto outputType = mlir::cast<RankedTensorType>(op.???().getType());

    // Q3c: Create empty output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        outputType.getShape(),
        outputType.???()
    );

    // Q3d: Create linalg.add
    auto addOp = rewriter.create<linalg::AddOp>(
        loc,
        TypeRange{???},
        ValueRange{???, ???},  // ins: lhs, rhs
        ValueRange{???}        // outs: output
    );

    // Q3e: Replace and return
    rewriter.???(op, addOp.???());
    return ???();
  }
};

//===----------------------------------------------------------------------===//
// Q4: Pass Definition
//===----------------------------------------------------------------------===//

struct GaweeToLinalgPass
    : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-gawee-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower Gawee dialect to Linalg dialect";
  }

  // Q4a: Declare dialects that this pass will CREATE ops from
  // WHY? MLIR needs to load these dialects before the pass runs
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<???>();  // linalg ops
    registry.insert<???>();  // arith.constant
    registry.insert<???>();  // tensor.empty
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Q4b: Create conversion target
    ConversionTarget target(*ctx);

    // Q4c: Mark dialects as legal (output dialects)
    target.???<linalg::LinalgDialect>();
    target.???<arith::ArithDialect>();
    target.???<tensor::TensorDialect>();

    // Q4d: Mark Gawee dialect as illegal (must be converted away)
    target.???<gawee::GaweeDialect>();

    // Q4e: Collect patterns
    RewritePatternSet patterns(ctx);
    patterns.add<???>(ctx);
    patterns.add<???>(ctx);
    patterns.add<???>(ctx);

    // Q4f: Run conversion
    if (failed(???(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Q5: Pass Creation Function
//===----------------------------------------------------------------------===//

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass() {
  return std::make_unique<???>();
}
}

//===----------------------------------------------------------------------===//
// Q6: Conceptual Questions
//===----------------------------------------------------------------------===//
//
// Q6a: Why do we zero-initialize the conv output with linalg.fill?
//      A) For better performance
//      B) Conv is a reduction op that accumulates - must start at zero
//      C) MLIR requires it
//      D) It's optional
//
// Q6b: What's the difference between adaptor and op in matchAndRewrite?
//      A) No difference
//      B) adaptor has converted operands, op has original attributes
//      C) adaptor has attributes, op has operands
//      D) op is for input, adaptor is for output
//
// Q6c: Why use linalg.generic for ReLU instead of a named op?
//      A) linalg.generic is faster
//      B) There's no named linalg op for ReLU
//      C) Named ops don't support f32
//      D) Generic is easier to write
//
// Q6d: What does addIllegalDialect do?
//      A) Prevents the dialect from being used
//      B) Marks all ops from that dialect as illegal - must be converted
//      C) Deletes the dialect
//      D) Throws an error if dialect is present
//

//===----------------------------------------------------------------------===//
// Answer Key
//===----------------------------------------------------------------------===//
/*
Q1a: getInput, getWeight
Q1b: getStridesAttr, getDilationAttr
Q1c: RankedTensorType, getElementType
Q1d: getShape
Q1e: getZeroAttr
Q1f: zero, emptyTensor
Q1g: input, weight, output
Q1h: replaceOp

Q2a: getInput, getElementType, getRank
Q2b: getShape, elementType
Q2c: 2
Q2d: parallel
Q2e: input, output
Q2f: 0, getZeroAttr
Q2g: MaximumFOp
Q2h: YieldOp

Q3a: getLhs, getRhs
Q3b: getOutput
Q3c: getElementType
Q3d: outputType, lhs, rhs, output
Q3e: replaceOp, getResults, success

Q4a: linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect
Q4c: addLegalDialect (x3)
Q4d: addIllegalDialect
Q4e: ConvOpLowering, ReluOpLowering, AddOpLowering
Q4f: applyPartialConversion

Q5: GaweeToLinalgPass

Q6a: B - Conv accumulates partial sums, must start at zero
Q6b: B - adaptor gives converted inputs, op gives original attributes
Q6c: B - No named linalg op for elementwise max(0, x)
Q6d: B - All ops from illegal dialect must be converted away
*/
