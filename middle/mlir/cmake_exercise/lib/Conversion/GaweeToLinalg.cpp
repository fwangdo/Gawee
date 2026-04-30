#include "Conversion/GaweePasses.h"
#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional> 

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

static Value extractTensorScalarAsIndex(OpBuilder &builder, Location loc,
                                        Value tensor, int64_t position);

static Value createEmptyTensorFromSourceDims(OpBuilder &builder, Location loc,
                                             RankedTensorType outputType,
                                             Value source,
                                             ArrayRef<int64_t> sourceDimMap = {}) {
  SmallVector<Value> dynamicSizes;
  dynamicSizes.reserve(outputType.getNumDynamicDims());
  for (int64_t dim = 0; dim < outputType.getRank(); ++dim) {
    if (!outputType.isDynamicDim(dim))
      continue;
    int64_t sourceDim = sourceDimMap.empty() ? dim : sourceDimMap[dim];
    dynamicSizes.push_back(tensor::DimOp::create(builder, loc, source, sourceDim));
  }
  return tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                 outputType.getElementType(), dynamicSizes);
}

static Value createEmptyTensorFromShapeTensor(OpBuilder &builder, Location loc,
                                              RankedTensorType outputType,
                                              Value shapeTensor) {
  SmallVector<Value> dynamicSizes;
  dynamicSizes.reserve(outputType.getNumDynamicDims());
  for (int64_t dim = 0; dim < outputType.getRank(); ++dim) {
    if (!outputType.isDynamicDim(dim))
      continue;
    dynamicSizes.push_back(
        extractTensorScalarAsIndex(builder, loc, shapeTensor, dim));
  }
  return tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                 outputType.getElementType(), dynamicSizes);
}

static DenseIntElementsAttr makeI64ElementsAttr(OpBuilder &builder,
                                                ArrayRef<int64_t> values) {
  auto attrType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder.getI64Type());
  return DenseIntElementsAttr::get(attrType, values);
}

static AffineMap buildBroadcastMap(RankedTensorType inputType,
                                   RankedTensorType outputType,
                                   MLIRContext *ctx) {
  int64_t outRank = outputType.getRank();
  int64_t inRank = inputType.getRank();
  if (inRank == 0)
    return AffineMap::get(outRank, 0, {}, ctx);

  SmallVector<AffineExpr> exprs;
  exprs.reserve(inRank);
  int64_t leading = outRank - inRank;
  for (int64_t i = 0; i < inRank; ++i) {
    int64_t outDim = leading + i;
    if (!inputType.isDynamicDim(i) && inputType.getShape()[i] == 1 &&
        (outputType.isDynamicDim(outDim) || outputType.getShape()[outDim] != 1)) {
      exprs.push_back(getAffineConstantExpr(0, ctx));
      continue;
    }
    exprs.push_back(getAffineDimExpr(outDim, ctx));
  }
  return AffineMap::get(outRank, 0, exprs, ctx);
}

static Value chooseShapeCarrier(Value preferred, Value fallback) {
  auto preferredType = dyn_cast<RankedTensorType>(preferred.getType());
  if (preferredType && preferredType.getRank() > 0)
    return preferred;
  return fallback;
}

static Value buildElementwiseBinaryGeneric(
    ConversionPatternRewriter &rewriter, Location loc, RankedTensorType outputType,
    Value lhs, Value rhs,
    function_ref<Value(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
  auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());
  Value output = createEmptyTensorFromSourceDims(
      rewriter, loc, outputType, chooseShapeCarrier(lhs, rhs));
  int64_t rank = outputType.getRank();
  SmallVector<AffineMap> indexingMaps = {
      buildBroadcastMap(lhsType, outputType, rewriter.getContext()),
      buildBroadcastMap(rhsType, outputType, rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};
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
  // the reason of using "ValueRange" is that it can access multi index. 
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

static FailureOr<Value> expandReductionResultToKeepDims(
    ConversionPatternRewriter &rewriter, Location loc, Value reduced,
    RankedTensorType outputType, ArrayRef<int64_t> axes) {
  auto reducedType = dyn_cast<RankedTensorType>(reduced.getType());
  if (!reducedType)
    return failure();

  if (reducedType == outputType)
    return reduced;

  if (reducedType.getRank() == 0) {
    if (!outputType.hasStaticShape())
      return failure();
    auto shapeType =
        RankedTensorType::get({outputType.getRank()}, rewriter.getI64Type());
    auto shapeAttr = DenseElementsAttr::get(
        shapeType, ArrayRef<int64_t>(outputType.getShape().begin(),
                                     outputType.getShape().end()));
    Value shapeValue =
        arith::ConstantOp::create(rewriter, loc, shapeType, shapeAttr);
    return Value(
        tensor::ReshapeOp::create(rewriter, loc, outputType, reduced, shapeValue));
  }

  // the axes look like { 2, 3 }. it should be recovered. 
  llvm::SmallDenseSet<int64_t> reducedAxes;
  for (int64_t axis : axes)
    reducedAxes.insert(axis);

  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(reducedType.getRank());
  int64_t outDim = 0;
  for (int64_t inDim = 0; inDim < reducedType.getRank(); ++inDim) {
    ReassociationIndices group;

    while (outDim < outputType.getRank() && reducedAxes.contains(outDim))
      group.push_back(outDim++);

    if (outDim >= outputType.getRank())
      return failure();

    group.push_back(outDim++);

    while (outDim < outputType.getRank() && reducedAxes.contains(outDim))
      group.push_back(outDim++);

    reassociation.push_back(group);
  }

  if (outDim != outputType.getRank())
    return failure();

  return Value(tensor::ExpandShapeOp::create(rewriter, loc, outputType, reduced,
                                             reassociation));
}

// op lowering.
struct PadOpLowering : public OpConversionPattern<gawee::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(gawee::PadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewrtier) {
    Location loc = op.getLoc(); 
                                }
}

struct ConvOpLowering : public OpConversionPattern<gawee::ConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // get operand
    Value input = adaptor.getInput();
    // Value

    // get strides, dilation, padding. 

    
            }
}


struct GaweeToLinalgPass : public PassWrapper<GaweeToLinalgPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-gawee-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower Gawee dialect to Linalg dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>(); 
    registry.insert<arith::ArithDialect>(); 
    registry.insert<math::MathDialect>(); 
    registry.insert<scf::SCFDialect>();  
    registry.insert<tensor::TensorDialect>(); 
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext(); 
    ModuleOp module = getOperation();

    ConversionTarget target(*ctx); 
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<gawee::GaweeDialect>();  // Gawee ops must be converted

    RewritePatternSet patterns(ctx);

    // Run conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

  }
}

} // namespace.

// pass registration.

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass() {
  return std::make_unique<GaweeToLinalgPass>();
}
}
