//===----------------------------------------------------------------------===//
// GaweeToLinalg Quiz
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the conversion pass.
// After completing, compare with lib/Conversion/GaweeToLinalg.cpp
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
// Quiz 1: Conv2D Lowering
//===----------------------------------------------------------------------===//

struct ConvOpLowering : public OpConversionPattern<gawee::ConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  // adaptor -> already converted operand values. 
  matchAndRewrite(gawee::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q1-1: Get input and weight from adaptor
    Value input = adaptor.getInput();  
    Value weight = adaptor.getWeight();   

    // Q1-2: Get strides and dilations as Attributes (not ArrayRef!)
    // Hint: use *Attr() suffix
    // attr should be got by op. 
    auto strides = op.getStridesAttr();  
    auto dilations = op.getDilationAttr();  

    // Q1-3: Get output type using mlir::cast
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    // Q1-4: Create empty output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        outputType.getShape(),  // shape
        outputType.getElementType()  // element type
    );

    // Q1-5: Create linalg.conv_2d_nchw_fchw
    auto conv = rewriter.create<linalg::Conv2DNchwFchwOp>(
        loc,
        outputType,                    // result type
        ValueRange{input, weight},   // ins: input, weight
        output,                    // outs: output
        strides,                    // strides
        dilations                     // dilations
    );

    // Q1-6: Replace original op
    rewriter.replaceOp(op, conv.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quiz 2: ReLU Lowering (harder)
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpConversionPattern<gawee::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q2-1: Get input and its type
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType(); // dtype. 
    int64_t rank = inputType.getRank();

    // Q2-2: Create output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
        loc,
        inputType.getShape(),
        elementType                                            
    );

    // Q2-3: Create identity indexing maps for elementwise operation
    // Hint: AffineMap::getMultiDimIdentityMap(rank, context)
    SmallVector<AffineMap, 2> indexingMaps(
        2,  // how many maps? (input + output)
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
    );

    // Q2-4: Create iterator types (all parallel for elementwise)
    // all independent. 
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, 
        utils::IteratorType::parallel
    );

    // Q2-5: Create linalg.generic with body
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{inputType},           // result types
        ValueRange{input},          // inputs
        ValueRange{output},          // outputs
        indexingMaps,
        iteratorTypes,
        // rhs in each loop operation. It is the answer of calculation(in this case, relu. )  
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value inVal = args[0];  // 0 -> input, 1 -> output.  

          // Q2-6: Create zero constant
          Value zero = builder.create<arith::ConstantOp>(
              loc, elementType, builder.getZeroAttr(elementType)
          );

          // Q2-7: Compute max(input, zero) - ReLU operation 
          // the essence. MaximumFOp is max opeartion for fp datatype.  
          Value result = builder.create<arith::MaximumFOp>(loc, inVal, zero);

          // Q2-8: Yield result
          builder.create<linalg::YieldOp>(loc, result); // YieldOp means return value. 
        }
    );

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Qud Lowering
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<gawee::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Q3-1: Get lhs and rhs operands
    Value lhs = ???;
    Value rhs = ???;

    // Q3-2: Get output type and create empty tensor
    auto outputType = mlir::cast<RankedTensorType>(???.getType());
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, ???, ???
    );

    // Q3-3: Create linalg.add
    auto addOp = rewriter.create<linalg::AddOp>(
        loc,
        TypeRange{???},
        ValueRange{???, ???},  // ins
        ValueRange{???}        // outs
    );

    // Q3-4: Replace and return
    rewriter.???(op, addOp.getResults());
    return ???;
  }
};

//===----------------------------------------------------------------------===//
// Quiz 4: Pass Definition
//===----------------------------------------------------------------------===//

struct GaweeToLinalgPass
    : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-gawee-to-linalg"; }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Q4-1: Create conversion target
    ConversionTarget target(*ctx);

    // Q4-2: Mark dialects as legal (can exist after conversion)
    target.addLegalDialect<linalg::???>();
    target.addLegalDialect<arith::???>();
    target.addLegalDialect<tensor::???>();

    // Q4-3: Mark Gawee dialect as illegal (must be converted)
    target.add???Dialect<gawee::GaweeDialect>();

    // Q4-4: Add patterns
    RewritePatternSet patterns(ctx);
    patterns.add<???, ???, ???>(ctx);

    // Q4-5: Run conversion
    if (failed(???(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Answer Key (don't peek until you've tried!)
//===----------------------------------------------------------------------===//
/*
Q1-1: adaptor.getInput(), adaptor.getWeight()
Q1-2: op.getStridesAttr(), op.getDilationAttr()
Q1-3: RankedTensorType
Q1-4: outputType.getShape(), outputType.getElementType()
Q1-5: outputType, input, weight, output, strides, dilations
Q1-6: replaceOp, success()

Q2-1: adaptor.getInput(), input, getElementType(), getRank()
Q2-2: inputType.getShape(), elementType
Q2-3: 2, getMultiDimIdentityMap
Q2-4: rank, parallel
Q2-5: inputType, input, output
Q2-6: 0, getZeroAttr
Q2-7: MaximumFOp
Q2-8: YieldOp

Q3-1: adaptor.getLhs(), adaptor.getRhs()
Q3-2: op.getOutput(), outputType.getShape(), outputType.getElementType()
Q3-3: outputType, lhs, rhs, output
Q3-4: replaceOp, success()

Q4-1: (already done)
Q4-2: LinalgDialect, ArithDialect, TensorDialect
Q4-3: addIllegal
Q4-4: ConvOpLowering, ReluOpLowering, AddOpLowering
Q4-5: applyPartialConversion
*/
