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
// Extension rule for this file:
//   - do NOT create a separate learning-only lowering file
//   - extend this production file directly
//   - when adding a new gawee op, first identify the closest existing pattern
//     and copy its structure conservatively
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

// -------------------------------------------------------------------------
// Lowering extension scaffold
//
// When you add a new op family, answer these before writing code:
//
//   1. Which existing lowering is the nearest reference?
//      - gawee.relu   -> see ReluOpLowering
//      - gawee.add    -> see AddOpLowering
//      - gawee.linear -> see LinearOpLowering
//      - gawee.conv   -> see ConvOpLowering
//
//   2. Is the target best expressed as:
//      - direct linalg op
//      - linalg.generic
//      - tensor.empty + linalg.fill + linalg op
//
//   3. Does the lowering need:
//      - broadcast indexing maps
//      - padding materialization
//      - reshape/transpose handling
//      - post-op bias add
//
// Suggested future candidates to think about:
//   - gawee.batch_norm
//   - gawee.cat
//   - any new op introduced by ONNXMLIREmitter expansion
//
// Do not implement new logic here mechanically.
// First write down:
//   input shape, output shape, target linalg op, and any broadcast rule.
// -------------------------------------------------------------------------

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
// Scaffold-only lowerings for the remaining gawee ops
//===----------------------------------------------------------------------===//
//
// Each gawee op should have a named lowering home in this file, even before
// the real rewrite exists. Replace these stubs incrementally with concrete
// linalg/tensor/arith rewrites as the dialect grows.

struct BatchNormOpLowering : public OpConversionPattern<gawee::BatchNormOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::BatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    Value runningMean = adaptor.getRunningMean();
    Value runningVar = adaptor.getRunningVar();

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    if (outputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          op, "batch_norm lowering currently expects rank-4 NCHW tensors");
    }

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = outputType.getElementType();
    Value output =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), elementType);

    auto ctx = rewriter.getContext();
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(4, ctx);
    AffineMap channelMap =
        AffineMap::get(4, 0, {getAffineDimExpr(1, ctx)}, ctx);
    SmallVector<AffineMap> indexingMaps = {identityMap, channelMap, channelMap,
                                           channelMap, channelMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        4, utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{input, weight, bias, runningMean, runningVar},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto epsAttr = builder.getFloatAttr(elementType, op.getEps());
          Value eps = arith::ConstantOp::create(builder, loc, epsAttr);
          Value shiftedVar = arith::AddFOp::create(builder, loc, args[4], eps);
          Value denom = math::SqrtOp::create(builder, loc, shiftedVar);
          Value centered = arith::SubFOp::create(builder, loc, args[0], args[3]);
          Value normalized = arith::DivFOp::create(builder, loc, centered, denom);
          Value scaled = arith::MulFOp::create(builder, loc, normalized, args[1]);
          Value shifted = arith::AddFOp::create(builder, loc, scaled, args[2]);
          linalg::YieldOp::create(builder, loc, shifted);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct AveragePoolOpLowering
    : public OpConversionPattern<gawee::AveragePoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::AveragePoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto strides = adaptor.getStridesAttr();
    auto padding = adaptor.getPadding();

    if (!op.getCountIncludePad() && (padding[0] != 0 || padding[1] != 0)) {
      return rewriter.notifyMatchFailure(
          op,
          "average_pool lowering currently supports padded case only when "
          "count_include_pad=true");
    }

    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value zero = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getZeroAttr(elementType));

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
          /*low=*/ValueRange{}, /*high=*/ValueRange{});
      auto &region = padOp.getRegion();
      auto *block = rewriter.createBlock(&region);
      for (int i = 0; i < paddedType.getRank(); ++i)
        block->addArgument(rewriter.getIndexType(), loc);
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<tensor::YieldOp>(loc, zero);
      input = padOp.getResult();
      rewriter.setInsertionPointAfter(padOp);
    }

    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    Value zeroFilled =
        linalg::FillOp::create(rewriter, loc, zero, emptyTensor).getResult(0);
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, adaptor.getKernelSize(), elementType);

    auto sumPool = linalg::PoolingNchwSumOp::create(
        rewriter, loc, outputType, ValueRange{input, windowTensor},
        ValueRange{zeroFilled}, strides, rewriter.getDenseI64ArrayAttr({1, 1}));

    int64_t kernelCount =
        adaptor.getKernelSize()[0] * adaptor.getKernelSize()[1];
    Value countVal = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elementType, static_cast<double>(kernelCount)));
    Value divEmpty = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap, 2> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto avgOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{sumPool.getResult(0)}, ValueRange{divEmpty}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value avg = arith::DivFOp::create(builder, loc, args[0], countVal);
          linalg::YieldOp::create(builder, loc, avg);
        });

    rewriter.replaceOp(op, avgOp.getResults());
    return success();
  }
};

struct CatOpLowering : public OpConversionPattern<gawee::CatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.cat from gawee to linalg");
  }
};

struct GlobalAveragePoolOpLowering
    : public OpConversionPattern<gawee::GlobalAveragePoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::GlobalAveragePoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.global_average_pool from gawee to linalg");
  }
};

struct SubOpLowering : public OpConversionPattern<gawee::SubOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = isa<FloatType>(elementType)
                             ? Value(arith::SubFOp::create(builder, loc, args[0],
                                                           args[1]))
                             : Value(arith::SubIOp::create(builder, loc, args[0],
                                                           args[1]));
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct MulOpLowering : public OpConversionPattern<gawee::MulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = isa<FloatType>(elementType)
                             ? Value(arith::MulFOp::create(builder, loc, args[0],
                                                           args[1]))
                             : Value(arith::MulIOp::create(builder, loc, args[0],
                                                           args[1]));
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct DivOpLowering : public OpConversionPattern<gawee::DivOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = isa<FloatType>(elementType)
                             ? Value(arith::DivFOp::create(builder, loc, args[0],
                                                           args[1]))
                             : Value(arith::DivSIOp::create(builder, loc, args[0],
                                                            args[1]));
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct ReduceMeanOpLowering
    : public OpConversionPattern<gawee::ReduceMeanOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.reduce_mean from gawee to linalg");
  }
};

struct ReduceSumOpLowering : public OpConversionPattern<gawee::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.reduce_sum from gawee to linalg");
  }
};

struct ReshapeOpLowering : public OpConversionPattern<gawee::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.reshape from gawee to linalg");
  }
};

struct TransposeOpLowering : public OpConversionPattern<gawee::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.transpose from gawee to linalg");
  }
};

struct SqueezeOpLowering : public OpConversionPattern<gawee::SqueezeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.squeeze from gawee to linalg");
  }
};

struct UnsqueezeOpLowering : public OpConversionPattern<gawee::UnsqueezeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.unsqueeze from gawee to linalg");
  }
};

struct ShapeOpLowering : public OpConversionPattern<gawee::ShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.shape from gawee to linalg");
  }
};

struct SoftmaxOpLowering : public OpConversionPattern<gawee::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.softmax from gawee to linalg");
  }
};

struct SqrtOpLowering : public OpConversionPattern<gawee::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = math::SqrtOp::create(builder, loc, args[0]);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct TanhOpLowering : public OpConversionPattern<gawee::TanhOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::TanhOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = math::TanhOp::create(builder, loc, args[0]);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct SigmoidOpLowering : public OpConversionPattern<gawee::SigmoidOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value one = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 1.0));
          Value neg = arith::NegFOp::create(builder, loc, args[0]);
          Value exp = math::ExpOp::create(builder, loc, neg);
          Value denom = arith::AddFOp::create(builder, loc, one, exp);
          Value result = arith::DivFOp::create(builder, loc, one, denom);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct HardSigmoidOpLowering
    : public OpConversionPattern<gawee::HardSigmoidOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::HardSigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value alpha = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, op.getAlpha()));
          Value beta = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, op.getBeta()));
          Value zero = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 0.0));
          Value one = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 1.0));
          Value scaled = arith::MulFOp::create(builder, loc, args[0], alpha);
          Value shifted = arith::AddFOp::create(builder, loc, scaled, beta);
          Value clippedLow =
              arith::MaximumFOp::create(builder, loc, shifted, zero);
          Value result =
              arith::MinimumFOp::create(builder, loc, clippedLow, one);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct HardSwishOpLowering : public OpConversionPattern<gawee::HardSwishOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::HardSwishOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value three = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 3.0));
          Value six = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 6.0));
          Value zero = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 0.0));
          Value shifted = arith::AddFOp::create(builder, loc, args[0], three);
          Value clippedLow =
              arith::MaximumFOp::create(builder, loc, shifted, zero);
          Value clippedHigh =
              arith::MinimumFOp::create(builder, loc, clippedLow, six);
          Value numer = arith::MulFOp::create(builder, loc, args[0], clippedHigh);
          Value result = arith::DivFOp::create(builder, loc, numer, six);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct LeakyReluOpLowering : public OpConversionPattern<gawee::LeakyReluOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::LeakyReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value zero = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 0.0));
          Value alpha = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, op.getAlpha()));
          Value scaled = arith::MulFOp::create(builder, loc, args[0], alpha);
          Value result = arith::MaximumFOp::create(builder, loc, args[0], scaled);
          Value isNeg = arith::CmpFOp::create(
              builder, loc, arith::CmpFPredicate::OLT, args[0], zero);
          Value selected =
              arith::SelectOp::create(builder, loc, isNeg, scaled, result);
          linalg::YieldOp::create(builder, loc, selected);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct GeluOpLowering : public OpConversionPattern<gawee::GeluOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value half = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 0.5));
          Value one = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 1.0));
          Value sqrt2 = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(elementType, 1.4142135623730951));
          Value div = arith::DivFOp::create(builder, loc, args[0], sqrt2);
          Value erf = math::ErfOp::create(builder, loc, div);
          Value plus = arith::AddFOp::create(builder, loc, one, erf);
          Value scaled = arith::MulFOp::create(builder, loc, args[0], plus);
          Value result = arith::MulFOp::create(builder, loc, half, scaled);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct ErfOpLowering : public OpConversionPattern<gawee::ErfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result = math::ErfOp::create(builder, loc, args[0]);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct EqualOpLowering : public OpConversionPattern<gawee::EqualOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result;
          if (isa<FloatType>(lhsType.getElementType())) {
            result = arith::CmpFOp::create(
                builder, loc, arith::CmpFPredicate::OEQ, args[0], args[1]);
          } else {
            result = arith::CmpIOp::create(
                builder, loc, arith::CmpIPredicate::eq, args[0], args[1]);
          }
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct WhereOpLowering : public OpConversionPattern<gawee::WhereOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value condition = adaptor.getCondition();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        4, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{condition, lhs, rhs},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result =
              arith::SelectOp::create(builder, loc, args[0], args[1], args[2]);
          linalg::YieldOp::create(builder, loc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct MaxOpLowering : public OpConversionPattern<gawee::MaxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.max from gawee to linalg");
  }
};

struct MinOpLowering : public OpConversionPattern<gawee::MinOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.min from gawee to linalg");
  }
};

struct ExpandOpLowering : public OpConversionPattern<gawee::ExpandOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.expand from gawee to linalg");
  }
};

struct SliceOpLowering : public OpConversionPattern<gawee::SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.slice from gawee to linalg");
  }
};

struct PadOpLowering : public OpConversionPattern<gawee::PadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.pad from gawee to linalg");
  }
};

struct CastOpLowering : public OpConversionPattern<gawee::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return rewriter.notifyMatchFailure(
        op, "TODO: lower gawee.cast from gawee to linalg");
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
    registry.insert<math::MathDialect>();
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
    target.addLegalDialect<math::MathDialect>();
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
    patterns.add<BatchNormOpLowering>(ctx);
    patterns.add<AveragePoolOpLowering>(ctx);
    patterns.add<CatOpLowering>(ctx);
    patterns.add<GlobalAveragePoolOpLowering>(ctx);
    patterns.add<SubOpLowering>(ctx);
    patterns.add<MulOpLowering>(ctx);
    patterns.add<DivOpLowering>(ctx);
    patterns.add<ReduceMeanOpLowering>(ctx);
    patterns.add<ReduceSumOpLowering>(ctx);
    patterns.add<ReshapeOpLowering>(ctx);
    patterns.add<TransposeOpLowering>(ctx);
    patterns.add<SqueezeOpLowering>(ctx);
    patterns.add<UnsqueezeOpLowering>(ctx);
    patterns.add<ShapeOpLowering>(ctx);
    patterns.add<SoftmaxOpLowering>(ctx);
    patterns.add<SqrtOpLowering>(ctx);
    patterns.add<TanhOpLowering>(ctx);
    patterns.add<SigmoidOpLowering>(ctx);
    patterns.add<HardSigmoidOpLowering>(ctx);
    patterns.add<HardSwishOpLowering>(ctx);
    patterns.add<LeakyReluOpLowering>(ctx);
    patterns.add<GeluOpLowering>(ctx);
    patterns.add<ErfOpLowering>(ctx);
    patterns.add<EqualOpLowering>(ctx);
    patterns.add<WhereOpLowering>(ctx);
    patterns.add<MaxOpLowering>(ctx);
    patterns.add<MinOpLowering>(ctx);
    patterns.add<ExpandOpLowering>(ctx);
    patterns.add<SliceOpLowering>(ctx);
    patterns.add<PadOpLowering>(ctx);
    patterns.add<CastOpLowering>(ctx);

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
