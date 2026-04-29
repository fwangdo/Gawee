//===----------------------------------------------------------------------===//
// Linalg Vectorization Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for vectorization preparation work.
//
// Typical future responsibilities:
//   - identify vectorization-friendly linalg ops
//   - normalize shapes/layouts for vector-friendly access
//   - mark where vector.transfer / vector.contract style lowering should start
//   - connect middle-end scheduling decisions to backend SIMD opportunities
//
// Current behavior:
//   - no IR mutation
//   - emits remarks about simple vectorization readiness
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

static bool hasStaticTensorResult(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  auto rankedType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return rankedType && rankedType.hasStaticShape();
}

static int64_t chooseVectorWidthHint(linalg::LinalgOp op) {
  if (op->getNumResults() != 1)
    return 1;
  auto rankedType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rankedType || !rankedType.hasStaticShape() || rankedType.getRank() == 0)
    return 1;

  int64_t innermost = rankedType.getShape().back();
  if (ShapedType::isDynamic(innermost))
    return 1;
  if (innermost % 16 == 0)
    return 16;
  if (innermost % 8 == 0)
    return 8;
  if (innermost % 4 == 0)
    return 4;
  return 1;
}

static void analyzeVectorizationReadiness(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](linalg::LinalgOp op) {
    SmallString<128> message;
    llvm::raw_svector_ostream os(message);
    os << "vectorization plan: ";
    int64_t widthHint = chooseVectorWidthHint(op);
    if (hasStaticTensorResult(op.getOperation()))
      os << "static result shape available";
    else
      os << "dynamic or non-tensor result limits vector planning";

    StringRef kind = "generic";
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op.getOperation())) {
      os << ", contraction op is a prime vectorization candidate";
      kind = "contraction";
    } else if (isa<linalg::Conv2DNchwFchwOp>(op.getOperation())) {
      os << ", conv op likely needs layout/tile prep before vector lowering";
      kind = "convolution";
    } else {
      os << ", generic/vector transfer path should be evaluated";
    }

    op->setAttr("gawee.vector.kind", builder.getStringAttr(kind));
    op->setAttr("gawee.vector.width_hint",
                builder.getI64IntegerAttr(widthHint));
    op->setAttr("gawee.vector.static_result",
                builder.getBoolAttr(hasStaticTensorResult(op.getOperation())));
    op->emitRemark() << os.str();
  });
}

struct LinalgVectorizationPass
    : public PassWrapper<LinalgVectorizationPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-vectorization";
  }

  StringRef getDescription() const override {
    return "Post-lowering Linalg vectorization planning pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    analyzeVectorizationReadiness(getOperation());
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
} // namespace mlir::gawee
