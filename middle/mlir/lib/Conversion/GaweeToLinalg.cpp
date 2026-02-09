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
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

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

    // Step 3: Create output tensor and initialize to zero
    // Conv is a reduction op - it accumulates into the output, so we must
    // zero-initialize it for correct results and bufferization
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        elementType
    );

    // Create zero constant and fill the output tensor
    Value zero = arith::ConstantOp::create(
        rewriter, loc,
        elementType,
        rewriter.getZeroAttr(elementType)
    );
    Value output = linalg::FillOp::create(rewriter, loc, zero, emptyTensor)
                       .getResult(0);

    // Step 4: Create linalg.conv_2d_nchw_fchw
    auto conv = linalg::Conv2DNchwFchwOp::create(
        rewriter, loc,
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
    Value output = tensor::EmptyOp::create(
        rewriter, loc,
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
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{inputType},
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = output element (unused)
          Value inVal = args[0];

          // Create zero constant
          Value zero = arith::ConstantOp::create(
              builder, loc,
              elementType,
              builder.getZeroAttr(elementType)
          );

          // max(0, x)
          Value result = arith::MaximumFOp::create(builder, loc, inVal, zero);

          // Yield result
          linalg::YieldOp::create(builder, loc, result);
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
    Value output = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        outputType.getElementType()
    );

    // Step 3: Create linalg.add
    auto addOp = linalg::AddOp::create(
        rewriter, loc,
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

//===----------------------------------------------------------------------===//
// Maxpool lowering.  
//===----------------------------------------------------------------------===//

struct MaxPoolOpLowering : public OpConversionPattern<gawee::MaxPoolOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::MaxPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get all features.  
    Value input = adaptor.getInput();
    auto kernelSize = adaptor.getKernelSizeAttr(); 
    auto strides = adaptor.getStridesAttr(); 
    auto padding = adaptor.getPaddingAttr(); 
    auto dilation = adaptor.getDilationAttr();  
    auto ceilMode = adaptor.getCeilMode();

    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();  
    
    // scaffold. It's destination which calculated result will be saved.  
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter,loc,
        outputType.getShape(),
        elementType
    );

    // how to genreate neg inf?
    auto negInf = arith::ConstantOp::create(rewriter, loc,
      rewriter.getFloatAttr(elementType, -std::numeric_limits<double>::infinity())
    );
    auto filledOutput = linalg::FillOp::create(rewriter, loc, negInf.getResult(), emptyTensor);

    // window tensor to let mlir know the shape kernel.
    Value windowTensor = tensor::EmptyOp::create(rewriter, loc, kernelSize, elementType
    );


    // Step 3: Create linalg.pooling_nchw_max
    auto maxPoolOp = linalg::PoolingNchwMaxOp::create(
        rewriter, loc, outputType,
        // ValueRange is a container which has Values.
        ValueRange{input, windowTensor}, filledOutput.getResult(0),
        strides,
        dilation
    );

    // Step 4: Replace original op
    rewriter.replaceOp(op, maxPoolOp.getResults());

    return success();
  }
};


//===----------------------------------------------------------------------===//
// Adaptive average pooling lowering.  
//===----------------------------------------------------------------------===//

struct AdAvgOpLowering : public OpConversionPattern<gawee::AdAvgPoolOp> {
  // Note that, Adaptive average pooling is pytorch specific operator. 
  // Therefore, mlir does not have this operation. 
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::AdAvgPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Extract information.  
    auto input = adaptor.getInput(); 
    auto inputType = mlir::cast<RankedTensorType>(input.getType()); 
    auto elementType = inputType.getElementType();

    // shape of input
    int64_t H = inputType.getShape()[2]; 
    int64_t W = inputType.getShape()[3]; 

    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    // Step 3: Generate destination object.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType 
    ); 
    Value zero = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getZeroAttr(elementType)
    );
    Value zeroFilled = linalg::FillOp::create(
        rewriter, loc, zero, emptyTensor 
    ).getResult(0); 
    
    // Step 4: Create window tensor.
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{H, W}, elementType
    );

    // 5. Extract attributes for operation. 
    auto strideAttr = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilationAttr = rewriter.getDenseI64ArrayAttr({1,1});

    // 6. Generate Sumpool, which is the preliminary step for generating adaptive average.
    auto sumPool = linalg::PoolingNchwSumOp::create(
        rewriter, loc, outputType, ValueRange{input, windowTensor},
        ValueRange{zeroFilled}, strideAttr,
        dilationAttr
    );

    // 7. Createt the divisor constant which the number of total elements for one element(k_H * k_W).  
    int64_t count = H * W;
    Value countVal = arith::ConstantOp::create(
        rewriter, loc, elementType,
        rewriter.getFloatAttr(elementType, static_cast<double>(count)) 
    );

    // div empty
    Value divEmpty = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType 
    );

    // key factor: elementwise division. 
    int64_t rank = outputType.getRank();
    // template argument 2 means "assigning 2 slots(slot is decided by sizeof function) by default."
    SmallVector<AffineMap, 2> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
    );
    // it means that we can do parallel method for all dimensions. 
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel
    ); 

    //
    auto divOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{sumPool->getResults()[0]}, // ins
        ValueRange{divEmpty},                 // outs
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // this part operates one loop. 
          // args[0] is input, args[1] is output.
          // Yield records the result on proper output location. 
          Value avg = arith::DivFOp::create(builder, loc, args[0], countVal); 
          linalg::YieldOp::create(builder, loc, avg); 
      }
    );

    // Step 4: Replace original op
    rewriter.replaceOp(op, divOp.getResults());

    return success();
  }
};


//===----------------------------------------------------------------------===//
// Flatten operation lowering.  
//===----------------------------------------------------------------------===//

struct FlattenOpLowering : public OpConversionPattern<gawee::AddOp> {
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
    Value output = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        outputType.getElementType()
    );

    // Step 3: Create linalg.add
    auto addOp = linalg::AddOp::create(
        rewriter, loc,
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
// Linear lowering.  
//===----------------------------------------------------------------------===//

struct LinearOpLowering : public OpConversionPattern<gawee::AddOp> {
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
    Value output = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        outputType.getElementType()
    );

    // Step 3: Create linalg.add
    auto addOp = linalg::AddOp::create(
        rewriter, loc,
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

  // Declare dialects that this pass will produce ops from
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
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
