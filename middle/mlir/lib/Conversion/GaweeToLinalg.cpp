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
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
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
    Value bias = adaptor.getBias(); // TODO  

    // Step 2: Get attributes (use *Attr() to get Attribute, not ArrayRef)
    auto strides = op.getStridesAttr();
    auto dilations = op.getDilationAttr();
    auto padding = op.getPadding(); // ArrayRef<int64_t>

    // Step 3: Pad input if padding is non-zero.
    // linalg.conv_2d_nchw_fchw does NOT handle padding internally.
    // We must explicitly pad the input tensor on H and W dims with zeros.
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    Value zero = arith::ConstantOp::create(
        rewriter, loc,
        elementType,
        rewriter.getZeroAttr(elementType)
    );

    int64_t padH = padding[0];
    int64_t padW = padding[1];
    if (padH != 0 || padW != 0) {
      // Pad format: [N_low, C_low, H_low, W_low] and [N_high, C_high, H_high, W_high]
      // Only pad spatial dims (H, W), not batch (N) or channel (C).
      SmallVector<int64_t> lowPad = {0, 0, padH, padW};
      SmallVector<int64_t> highPad = {0, 0, padH, padW};

      // Compute padded input type
      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> paddedShape(inputType.getShape());
      paddedShape[2] += 2 * padH;
      paddedShape[3] += 2 * padW;
      auto paddedType = RankedTensorType::get(paddedShape, elementType);

      // PadOp needs a body region that yields the pad value (zero).
      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, input,
          lowPad, highPad,
          /*low=*/ValueRange{}, /*high=*/ValueRange{});

      // Build the body: yield zero for all padded positions
      auto &region = padOp.getRegion();
      auto *block = rewriter.createBlock(&region);
      for (int i = 0; i < paddedType.getRank(); i++)
        block->addArgument(rewriter.getIndexType(), loc);
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<tensor::YieldOp>(loc, zero);

      input = padOp.getResult();
      // Restore insertion point after the pad op
      rewriter.setInsertionPointAfter(padOp);
    }

    // Step 4: Create output tensor and initialize to zero
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        elementType
    );
    Value output = linalg::FillOp::create(rewriter, loc, zero, emptyTensor)
                       .getResult(0);

    // Step 5: Create linalg.conv_2d_nchw_fchw
    auto conv = linalg::Conv2DNchwFchwOp::create(
        rewriter, loc,
        outputType,
        ValueRange{input, weight},  // ins
        output,                      // outs
        strides,
        dilations
    );

    // Step 6: Add bias
    // bias shape: [out_channels] (1D), conv output: [N, C, H, W] (4D)
    // Need to broadcast bias across N, H, W dims using linalg.generic
    Value convResult = conv.getResults()[0];
    int64_t rank = outputType.getRank();

    Value biasEmpty = tensor::EmptyOp::create(
        rewriter, loc,
        outputType.getShape(),
        elementType
    );

    // Indexing maps:
    //   bias: (n, c, h, w) -> (c)       — broadcast across n, h, w
    //   conv: (n, c, h, w) -> (n, c, h, w)  — identity
    //   out:  (n, c, h, w) -> (n, c, h, w)  — identity
    auto ctx = rewriter.getContext();
    AffineMap biasMap = AffineMap::get(
        rank, 0, {getAffineDimExpr(1, ctx)}, ctx);  // (n,c,h,w) -> (c)
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

    SmallVector<AffineMap> indexingMaps = {identityMap, biasMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto biasAdd = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{outputType},
        ValueRange{convResult, bias},   // inputs
        ValueRange{biasEmpty},          // output (destination)
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = conv element, args[1] = bias element, args[2] = output (unused)
          Value result = arith::AddFOp::create(builder, loc, args[0], args[1]);
          linalg::YieldOp::create(builder, loc, result);
        }
    );

    // Step 7: Replace original op with conv + bias result
    rewriter.replaceOp(op, biasAdd.getResults());

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
    // note that, small vector is not a tensor but a container to convery information to MLIR framework. 
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


// As of this part, I define pass on my own. 
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
    auto strides = adaptor.getStridesAttr();
    auto padding = adaptor.getPadding(); // ArrayRef<int64_t>
    auto dilation = adaptor.getDilationAttr();

    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    // -inf is the identity element for max (same reason conv uses 0).
    auto negInf = arith::ConstantOp::create(rewriter, loc,
      rewriter.getFloatAttr(elementType, -std::numeric_limits<double>::infinity())
    );

    // Step 3: Pad input with -inf if padding is non-zero.
    // Same issue as Conv: linalg.pooling_nchw_max does NOT handle padding.
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

      // Pad with -inf (not 0!) so padded positions never win the max.
      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, input,
          lowPad, highPad,
          /*low=*/ValueRange{}, /*high=*/ValueRange{});
      auto &region = padOp.getRegion();
      auto *block = rewriter.createBlock(&region);
      for (int i = 0; i < paddedType.getRank(); i++)
        block->addArgument(rewriter.getIndexType(), loc);
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<tensor::YieldOp>(loc, negInf.getResult());
      input = padOp.getResult();
      rewriter.setInsertionPointAfter(padOp);
    }

    // Step 4: Create output filled with -inf and window tensor.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    auto filledOutput = linalg::FillOp::create(
        rewriter, loc, negInf.getResult(), emptyTensor);

    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, adaptor.getKernelSize(), elementType);

    // Step 5: Create linalg.pooling_nchw_max
    auto maxPoolOp = linalg::PoolingNchwMaxOp::create(
        rewriter, loc, outputType,
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

struct FlattenOpLowering : public OpConversionPattern<gawee::FlattenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::FlattenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get operands
    Value input = adaptor.getInput(); 
    auto inputType = mlir::cast<RankedTensorType>(input.getType()); 
    auto startDim = adaptor.getStartDimAttr(); 
    auto endDim = adaptor.getEndDimAttr(); 

    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());

    // Calculate start / end dimension.
    int64_t endDimInt = endDim.getInt(); 
    int64_t rank = inputType.getRank();  
    
    if (endDimInt < 0) {
      endDimInt += rank; 
    }
    
    SmallVector<ReassociationIndices> reassociation;

    // non-merge
    for (int64_t i = 0; i < startDim.getInt(); i++) {
      reassociation.push_back({i}); 
    }

    // merge
    ReassociationIndices mergedGroup; 
    for (int64_t i = startDim.getInt(); i <= endDimInt; i++) {
      mergedGroup.push_back(i); 
    }
    reassociation.push_back(mergedGroup);

    // non-merge.
    for (auto i = endDimInt + 1; i < rank; i++) {
      reassociation.push_back({i});
    }

    // Step 3: Create flatten operation.
    auto flattenOp = rewriter.create<tensor::CollapseShapeOp>(
        loc,
        outputType, 
        input, 
        reassociation
    );

    // Step 4: Replace original op
    rewriter.replaceOp(op, flattenOp.getResult());

    return success();
  }
};


//===----------------------------------------------------------------------===//
// Linear lowering.  
//===----------------------------------------------------------------------===//

struct LinearOpLowering : public OpConversionPattern<gawee::LinearOp> {
  // we need two operations which are matmul and add. 
  // because linear consists of matmul and add. 
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: Get operands
    Value input = adaptor.getInput(); 
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias(); 

    // Frist, we have to define matmul operation. 
    // Step 2: Get output type and create empty output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType 
    );

    // it should generate ssa value by getResult. But constantOp is an exception
    Value zero = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getZeroAttr(elementType)    
    );
    // getResult means getting nth ssa value from the defined operation. 
    Value filledZero = linalg::FillOp::create(
        rewriter, loc, zero, emptyTensor
    ).getResult(0);

    auto matmul = linalg::MatmulTransposeBOp::create(
        rewriter, loc, outputType, ValueRange{input, weight},
        filledZero
    );  

    // second. 
    Value matmulResult = matmul.getResult(0);  
    int64_t rank = outputType.getRank();

    Value biasEmpty =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                elementType
    );

    auto ctx = rewriter.getContext();
    // AffineMap defines object that has all information to calculate loop.  
    AffineMap biasMap = AffineMap::get(
      // getAffineDimExpr makes affine opearation for each loop. 
      // it means biasMap use only c(index 1) dimension like bias[c]
      rank, 0, getAffineDimExpr(1, ctx) , ctx
    ); 
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

    SmallVector<AffineMap> indexingMaps = {
      identityMap, // lhs. the value left to the operation. 
      biasMap, // rhs. the value right to the operation.
      identityMap // the destination. 
          }; 
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto linearOp = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{outputType},
        ValueRange{matmulResult, bias},   // inputs
        ValueRange{biasEmpty},          // output (destination)
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = conv element, args[1] = bias element, args[2] = output (unused)
          Value result = arith::AddFOp::create(builder, loc, args[0], args[1]);
          linalg::YieldOp::create(builder, loc, result);
        }
    );

    // Step 4: Replace original op
    rewriter.replaceOp(op, linearOp.getResults());

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
    patterns.add<MaxPoolOpLowering>(ctx); 
    patterns.add<AdAvgOpLowering>(ctx); 
    patterns.add<FlattenOpLowering>(ctx);
    patterns.add<LinearOpLowering>(ctx); 

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
