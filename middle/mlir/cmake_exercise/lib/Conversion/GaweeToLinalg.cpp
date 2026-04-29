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

// preliminaries. 

// static means the function is for this file only.
static int64_t normalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    return axis + rank; 
  }
  return axis; 
}

static Value makeScalarConstant(OpBuilder &builder, Location loc, Type type,
                                double floatValue, int64_t intValue) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return arith::ConstantOp::create(builder, loc, builder.getFloatAttr(floatType, floatValue)); 
  }
  if (isa<IndexType>(type)) {
    return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(intValue)); 
  }
  return arith::ConstantOp::create(builder, loc, type, builder.getIntegerAttr(type, intValue)); 
}

static Value makeZeroValue(OpBuilder &builder, Location loc, Type type) {
  if (isa<FloatType>(type)) {
    return makeScalarConstant(builder, loc, type, 0.0, 0); 
  }
  return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type)); 
}

static Value extractTensorScalarAsIndex(OpBuilder &builder, Location loc, Value tensor, int64_t position);

static Value
createEmptyTensorFromSrouceDims(OpBuilder &builder, Location loc,
                                RankedTensorType outputType, Value source,
                                ArrayRef<int64_t> sourceDimMap = {}) {
  SmallVector<Value> dynamicSize; 
  dynamicSize.reserve(outputType.getNumDynamicDims());
  for (int64_t dim = 0; dim < outputType.getRank(); ++dim) {
    if (!outputType.isDynamicDim(dim)) {
      continue;
    }
    int64_t sourceDim = sourceDimMap.empty() ? dim : sourceDimMap[dim]; 
    dynamicSize.push_back(tensor::DimOp::create(builder, loc, source, sourceDim)); 
  }
  return tensor::EmptyOp::create(builder, loc, outputType.getShape(), outputType.getElementType(), dynamicSize); 
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
