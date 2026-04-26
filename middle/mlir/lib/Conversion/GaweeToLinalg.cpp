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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

using namespace mlir;

namespace {

static int64_t normalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  return axis;
}

static Value makeScalarConstant(OpBuilder &builder, Location loc, Type type,
                                double floatValue, int64_t intValue) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getFloatAttr(floatType, floatValue));
  }
  if (isa<IndexType>(type)) {
    return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(intValue));
  }
  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getIntegerAttr(type, intValue));
}

static Value makeZeroValue(OpBuilder &builder, Location loc, Type type) {
  if (isa<FloatType>(type))
    return makeScalarConstant(builder, loc, type, 0.0, 0);
  return arith::ConstantOp::create(builder, loc, type, builder.getZeroAttr(type));
}

static Value buildElementwiseBinaryGeneric(
    ConversionPatternRewriter &rewriter, Location loc, RankedTensorType outputType,
    Value lhs, Value rhs,
    function_ref<Value(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  Value output = tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                         outputType.getElementType());
  int64_t rank = outputType.getRank();
  SmallVector<AffineMap> indexingMaps(
      3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(
      rank, utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
      ValueRange{output}, indexingMaps, iteratorTypes,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        Value result = bodyBuilder(builder, bodyLoc, args[0], args[1]);
        linalg::YieldOp::create(builder, bodyLoc, result);
      });
  return genericOp.getResult(0);
}

static FailureOr<SmallVector<int64_t>> getConstantI64Tensor(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return failure();

  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    SmallVector<int64_t> values;
    values.reserve(dense.getNumElements());
    for (APInt element : dense.getValues<APInt>())
      values.push_back(element.getSExtValue());
    return values;
  }

  return failure();
}

static Value extractTensorScalarAsIndex(OpBuilder &builder, Location loc,
                                        Value tensor, int64_t position) {
  Value idx = arith::ConstantOp::create(builder, loc, builder.getIndexAttr(position));
  Value extracted = tensor::ExtractOp::create(builder, loc, tensor, ValueRange{idx});
  if (isa<IndexType>(extracted.getType()))
    return extracted;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    extracted);
}

static Value extractScalarTensorValue(OpBuilder &builder, Location loc, Value tensor) {
  auto type = dyn_cast<RankedTensorType>(tensor.getType());
  if (!type)
    return tensor;
  if (type.getRank() == 0)
    return tensor::ExtractOp::create(builder, loc, tensor, ValueRange{});
  if (type.getRank() == 1 && type.getShape()[0] == 1) {
    Value zero =
        arith::ConstantOp::create(builder, loc, builder.getIndexAttr(0));
    return tensor::ExtractOp::create(builder, loc, tensor, ValueRange{zero});
  }
  return tensor;
}

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
    // adaptor: Get op safe. 
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
      // Q. need to know the meaning of low / high pad. 
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
    // ValueRange: list of Value. 
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
        // what is the role of this closure? 
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
    // AffineMap -> how to access index. 
    SmallVector<AffineMap, 2> indexingMaps(
        // getMultiDimidmap -> it should be the same. 
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
    );

    // Step 4: Create iterator types (all parallel for elementwise)
    SmallVector<utils::IteratorType> iteratorTypes(
        // Q. why do we need rank info? A. sometimes we need to write info for each dim. 
        rank, utils::IteratorType::parallel
    );

    // the meaning of closure -> calculation in loop.
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
      //
      // tensor::PadOp is not "just an attribute-only op".
      // It also owns a small region/body that answers this question:
      //
      //   "If execution touches a padded coordinate, what value should be
      //    produced there?"
      //
      // In this lowering, the answer is always "-inf".
      // So we first create the PadOp shell, then fill in its body below.
      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, input,
          lowPad, highPad,
          /*low=*/ValueRange{}, /*high=*/ValueRange{});

      // region:
      //   A region is MLIR's way to attach nested code to an op.
      //   Examples:
      //     - func.func owns a region for its function body
      //     - scf.for owns a region for its loop body
      //     - tensor.pad owns a region for "what value do we use in padded
      //       space?"
      //
      // For tensor.pad, this region is evaluated only for padded positions,
      // not for the original in-bounds input elements.
      auto &region = padOp.getRegion();

      // block:
      //   A region contains one or more basic blocks.
      //   A block is a sequence of MLIR ops plus block arguments.
      //
      // Here we create the single block that forms the body of tensor.pad.
      // This block will compute the value to place at a padded coordinate.
      auto *block = rewriter.createBlock(&region);

      // tensor.pad body receives one index argument per result dimension.
      // Intuition:
      //   if paddedType is rank-4, the body conceptually gets
      //     (%n, %c, %h, %w)
      //   so it knows "which padded output location am I filling right now?"
      //
      // In this particular lowering, we do not actually use those indices,
      // because every padded location gets the same constant value (-inf).
      // But the block signature still has to match what tensor.pad expects.
      for (int i = 0; i < paddedType.getRank(); i++)
        block->addArgument(rewriter.getIndexType(), loc);

      // The rewriter needs an insertion point, i.e. "where should newly
      // created IR go?"
      //
      // Up to now, the insertion point was in the outer rewrite stream.
      // To define the tensor.pad body, we temporarily move the insertion point
      // inside the newly created block.
      // it means cursur is moved to the end of the block. it is not needed. 
      rewriter.setInsertionPointToEnd(block);

      // tensor.yield is the "return" of the tensor.pad body.
      // It tells tensor.pad:
      //
      //   "For this padded coordinate, use negInf as the element value."
      //
      // Since this body ignores the block's index arguments, every padded
      // coordinate yields the same negInf constant.
      // yeild means returning region. 
      rewriter.create<tensor::YieldOp>(loc, negInf.getResult());

      // From this point on, the padded tensor becomes the new input to the
      // following pooling op.
      //
      // So the later linalg.pooling_nchw_max will read from:
      //   original input + explicit border filled with -inf
      // rather than from the original unpadded tensor.
      input = padOp.getResult();

      // After finishing the nested tensor.pad body, restore the insertion
      // point to the outer stream.
      //
      // If we do not do this, the next created ops would accidentally be
      // inserted inside the tensor.pad region instead of after the pad op.
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

    // Step 6: Replace original op
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
    // fillOp::create -> generate op. therefore, we need to extract the tensor by getResult(0); 
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
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto catOp = tensor::ConcatOp::create(rewriter, op.getLoc(), outputType,
                                          op.getAxis(), adaptor.getInputs());
    rewriter.replaceOp(op, catOp.getResult());
    return success();
  }
};

struct GlobalAveragePoolOpLowering
    : public OpConversionPattern<gawee::GlobalAveragePoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::GlobalAveragePoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    int64_t h = inputType.getShape()[2];
    int64_t w = inputType.getShape()[3];

    Value zero = makeZeroValue(rewriter, loc, elementType);
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    Value zeroFilled =
        linalg::FillOp::create(rewriter, loc, zero, emptyTensor).getResult(0);
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{h, w}, elementType);

    auto sumPool = linalg::PoolingNchwSumOp::create(
        rewriter, loc, outputType, ValueRange{input, windowTensor},
        ValueRange{zeroFilled}, rewriter.getDenseI64ArrayAttr({1, 1}),
        rewriter.getDenseI64ArrayAttr({1, 1}));

    Value countVal = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elementType, static_cast<double>(h * w)));
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
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();

    SmallVector<int64_t> axes;
    for (int64_t axis : op.getAxes())
      axes.push_back(normalizeAxis(axis, inputType.getRank()));

    Value zero = makeZeroValue(rewriter, loc, elementType);
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    Value init = linalg::FillOp::create(rewriter, loc, zero, emptyTensor)
                     .getResult(0);

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{input}, ValueRange{init}, axes,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          Value result = arith::AddFOp::create(builder, bodyLoc, args[0], args[1]);
          linalg::YieldOp::create(builder, bodyLoc, result);
        });

    int64_t count = 1;
    for (int64_t axis : axes)
      count *= inputType.getShape()[axis];
    Value countVal = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elementType, static_cast<double>(count)));

    Value divResult = buildElementwiseBinaryGeneric(
        rewriter, loc, outputType, reduceOp.getResult(0), reduceOp.getResult(0),
        [&](OpBuilder &builder, Location bodyLoc, Value lhs, Value) {
          return Value(arith::DivFOp::create(builder, bodyLoc, lhs, countVal));
        });
    rewriter.replaceOp(op, divResult);
    return success();
  }
};

struct ReduceSumOpLowering : public OpConversionPattern<gawee::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    SmallVector<int64_t> axes;
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    for (int64_t axis : op.getAxes())
      axes.push_back(normalizeAxis(axis, inputType.getRank()));

    Value zero = makeZeroValue(rewriter, loc, elementType);
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);
    Value init = linalg::FillOp::create(rewriter, loc, zero, emptyTensor)
                     .getResult(0);

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{input}, ValueRange{init}, axes,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          Value result = arith::AddFOp::create(builder, bodyLoc, args[0], args[1]);
          linalg::YieldOp::create(builder, bodyLoc, result);
        });

    rewriter.replaceOp(op, reduceOp.getResults());
    return success();
  }
};

struct ReshapeOpLowering : public OpConversionPattern<gawee::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<TensorType>(op.getOutput().getType());
    auto reshapeOp = tensor::ReshapeOp::create(rewriter, op.getLoc(), outputType,
                                               adaptor.getInput(),
                                               adaptor.getShape());
    rewriter.replaceOp(op, reshapeOp.getResult());
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<gawee::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value init = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    auto transposeOp = linalg::TransposeOp::create(
        rewriter, loc, adaptor.getInput(), init, op.getPerm());
    rewriter.replaceOp(op, transposeOp.getResults());
    return success();
  }
};

struct SqueezeOpLowering : public OpConversionPattern<gawee::SqueezeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    llvm::SmallDenseSet<int64_t> axes;
    for (int64_t axis : op.getAxes())
      axes.insert(normalizeAxis(axis, inputType.getRank()));

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices current;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      current.push_back(i);
      if (!axes.contains(i)) {
        reassociation.push_back(current);
        current.clear();
      }
    }
    if (!current.empty() && !reassociation.empty())
      reassociation.back().append(current.begin(), current.end());

    auto squeezeOp = tensor::CollapseShapeOp::create(
        rewriter, op.getLoc(), outputType, input, reassociation);
    rewriter.replaceOp(op, squeezeOp.getResult());
    return success();
  }
};

struct UnsqueezeOpLowering : public OpConversionPattern<gawee::UnsqueezeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    llvm::SmallDenseSet<int64_t> axes;
    for (int64_t axis : op.getAxes())
      axes.insert(normalizeAxis(axis, outputType.getRank()));

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices current;
    for (int64_t outDim = 0; outDim < outputType.getRank(); ++outDim) {
      current.push_back(outDim);
      bool nextIsInserted =
          (outDim + 1 < outputType.getRank()) && axes.contains(outDim + 1);
      if (!nextIsInserted) {
        reassociation.push_back(current);
        current.clear();
      }
    }

    auto unsqueezeOp = tensor::ExpandShapeOp::create(
        rewriter, op.getLoc(), outputType, input, reassociation);
    rewriter.replaceOp(op, unsqueezeOp.getResult());
    return success();
  }
};

struct ShapeOpLowering : public OpConversionPattern<gawee::ShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    int64_t rank = inputType.getRank();
    int64_t start = normalizeAxis(op.getStart(), rank);
    int64_t end = op.getEnd() < 0 ? op.getEnd() + rank : op.getEnd();
    end = std::min<int64_t>(end, rank);

    SmallVector<Value> dims;
    for (int64_t dim = start; dim < end; ++dim) {
      Value dimValue;
      if (!inputType.isDynamicDim(dim)) {
        dimValue = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(inputType.getShape()[dim]));
      } else {
        Value indexDim = tensor::DimOp::create(rewriter, loc, input, dim);
        dimValue = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getI64Type(), indexDim);
      }
      dims.push_back(dimValue);
    }

    auto shapeOp = tensor::FromElementsOp::create(rewriter, loc, outputType, dims);
    rewriter.replaceOp(op, shapeOp.getResult());
    return success();
  }
};

struct SoftmaxOpLowering : public OpConversionPattern<gawee::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    int64_t axis = normalizeAxis(op.getAxis(), outputType.getRank());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    auto softmaxOp = linalg::SoftmaxOp::create(
        rewriter, loc, TypeRange{outputType}, adaptor.getInput(), output, axis);
    rewriter.replaceOp(op, softmaxOp.getResults());
    return success();
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
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value current = adaptor.getInputs().front();
    for (Value next : adaptor.getInputs().drop_front()) {
      current = buildElementwiseBinaryGeneric(
          rewriter, op.getLoc(), outputType, current, next,
          [&](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
            if (isa<FloatType>(elementType))
              return Value(arith::MaximumFOp::create(builder, loc, lhs, rhs));
            return Value(arith::MaxSIOp::create(builder, loc, lhs, rhs));
          });
    }
    rewriter.replaceOp(op, current);
    return success();
  }
};

struct MinOpLowering : public OpConversionPattern<gawee::MinOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto elementType = outputType.getElementType();
    Value current = adaptor.getInputs().front();
    for (Value next : adaptor.getInputs().drop_front()) {
      current = buildElementwiseBinaryGeneric(
          rewriter, op.getLoc(), outputType, current, next,
          [&](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
            if (isa<FloatType>(elementType))
              return Value(arith::MinimumFOp::create(builder, loc, lhs, rhs));
            return Value(arith::MinSIOp::create(builder, loc, lhs, rhs));
          });
    }
    rewriter.replaceOp(op, current);
    return success();
  }
};

struct ExpandOpLowering : public OpConversionPattern<gawee::ExpandOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> inputExprs;
    int64_t outRank = outputType.getRank();
    int64_t inRank = inputType.getRank();
    int64_t leading = outRank - inRank;
    for (int64_t i = 0; i < inRank; ++i) {
      int64_t outDim = leading + i;
      if (inputType.getShape()[i] == 1 && outputType.getShape()[outDim] != 1)
        inputExprs.push_back(getAffineConstantExpr(0, ctx));
      else
        inputExprs.push_back(getAffineDimExpr(outDim, ctx));
    }
    AffineMap inputMap = AffineMap::get(outRank, 0, inputExprs, ctx);
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);
    SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        outRank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(builder, bodyLoc, args[0]);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct SliceOpLowering : public OpConversionPattern<gawee::SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto maybeStarts = getConstantI64Tensor(adaptor.getStarts());
    auto maybeEnds = getConstantI64Tensor(adaptor.getEnds());
    auto maybeAxes = getConstantI64Tensor(adaptor.getAxes());
    auto maybeSteps = getConstantI64Tensor(adaptor.getSteps());
    if (failed(maybeStarts) || failed(maybeEnds) || failed(maybeAxes) ||
        failed(maybeSteps)) {
      return rewriter.notifyMatchFailure(
          op, "slice lowering currently expects constant starts/ends/axes/steps");
    }

    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    int64_t rank = inputType.getRank();

    SmallVector<int64_t> offsets(rank, 0);
    SmallVector<int64_t> sizes(outputType.getShape().begin(),
                               outputType.getShape().end());
    SmallVector<int64_t> strides(rank, 1);

    for (size_t i = 0; i < maybeAxes->size(); ++i) {
      int64_t axis = normalizeAxis((*maybeAxes)[i], rank);
      int64_t start = (*maybeStarts)[i];
      int64_t step = (*maybeSteps)[i];
      if (start < 0)
        start += inputType.getShape()[axis];
      offsets[axis] = start;
      strides[axis] = step;
    }

    auto sliceOp = tensor::ExtractSliceOp::create(
        rewriter, op.getLoc(), outputType, input, ValueRange{}, ValueRange{},
        ValueRange{}, offsets, sizes, strides);
    rewriter.replaceOp(op, sliceOp.getResult());
    return success();
  }
};

struct PadOpLowering : public OpConversionPattern<gawee::PadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getMode() != "constant") {
      return rewriter.notifyMatchFailure(
          op, "pad lowering currently supports only constant mode");
    }

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    int64_t rank = inputType.getRank();

    SmallVector<int64_t> staticLow(rank, ShapedType::kDynamic);
    SmallVector<int64_t> staticHigh(rank, ShapedType::kDynamic);
    SmallVector<Value> dynamicLow;
    SmallVector<Value> dynamicHigh;
    for (int64_t dim = 0; dim < rank; ++dim) {
      dynamicLow.push_back(extractTensorScalarAsIndex(rewriter, loc, adaptor.getPads(),
                                                     dim));
      dynamicHigh.push_back(extractTensorScalarAsIndex(rewriter, loc, adaptor.getPads(),
                                                      rank + dim));
    }

    Value padValue = extractScalarTensorValue(rewriter, loc, adaptor.getConstantValue());
    auto padOp = rewriter.create<tensor::PadOp>(
        loc, outputType, input, staticLow, staticHigh,
        ValueRange{dynamicLow}, ValueRange{dynamicHigh});
    auto &region = padOp.getRegion();
    auto *block = rewriter.createBlock(&region);
    for (int i = 0; i < outputType.getRank(); ++i)
      block->addArgument(rewriter.getIndexType(), loc);
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<tensor::YieldOp>(loc, padValue);
    rewriter.replaceOp(op, padOp.getResult());
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<gawee::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gawee::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Type srcElem = inputType.getElementType();
    Type dstElem = outputType.getElementType();
    if (srcElem == dstElem) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType, input);
      return success();
    }

    Value output = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), dstElem);
    int64_t rank = outputType.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outputType}, ValueRange{input},
        ValueRange{output}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          Value result;
          if (isa<FloatType>(srcElem) && isa<FloatType>(dstElem)) {
            unsigned srcWidth = cast<FloatType>(srcElem).getWidth();
            unsigned dstWidth = cast<FloatType>(dstElem).getWidth();
            result = (dstWidth > srcWidth)
                         ? Value(arith::ExtFOp::create(builder, bodyLoc, dstElem,
                                                       args[0]))
                         : Value(arith::TruncFOp::create(builder, bodyLoc, dstElem,
                                                         args[0]));
          } else if (isa<IntegerType>(srcElem) && isa<FloatType>(dstElem)) {
            result = arith::SIToFPOp::create(builder, bodyLoc, dstElem, args[0]);
          } else if (isa<FloatType>(srcElem) && isa<IntegerType>(dstElem)) {
            result = arith::FPToSIOp::create(builder, bodyLoc, dstElem, args[0]);
          } else if (isa<IntegerType>(srcElem) && isa<IntegerType>(dstElem)) {
            unsigned srcWidth = cast<IntegerType>(srcElem).getWidth();
            unsigned dstWidth = cast<IntegerType>(dstElem).getWidth();
            if (dstWidth > srcWidth)
              result = arith::ExtSIOp::create(builder, bodyLoc, dstElem, args[0]);
            else if (dstWidth < srcWidth)
              result = arith::TruncIOp::create(builder, bodyLoc, dstElem, args[0]);
            else
              result = args[0];
          } else {
            result = args[0];
          }
          linalg::YieldOp::create(builder, bodyLoc, result);
        });
    rewriter.replaceOp(op, genericOp.getResults());
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
