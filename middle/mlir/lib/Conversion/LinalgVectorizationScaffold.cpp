//===----------------------------------------------------------------------===//
// Linalg Vectorization Scaffold Pass
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

static void analyzeVectorizationReadiness(ModuleOp module) {
  module.walk([&](linalg::LinalgOp op) {
    SmallString<128> message;
    llvm::raw_svector_ostream os(message);
    os << "vectorization scaffold: ";
    if (hasStaticTensorResult(op.getOperation()))
      os << "static result shape available";
    else
      os << "dynamic or non-tensor result limits vector planning";

    if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op.getOperation()))
      os << ", contraction op is a prime vectorization candidate";
    else if (isa<linalg::Conv2DNchwFchwOp>(op.getOperation()))
      os << ", conv op likely needs layout/tile prep before vector lowering";
    else
      os << ", generic/vector transfer path should be evaluated";
    op->emitRemark() << os.str();
  });
}

struct LinalgVectorizationScaffoldPass
    : public PassWrapper<LinalgVectorizationScaffoldPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-vectorization";
  }

  StringRef getDescription() const override {
    return "Scaffold pass for post-lowering Linalg vectorization planning";
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
std::unique_ptr<Pass> createLinalgVectorizationScaffoldPass() {
  return std::make_unique<LinalgVectorizationScaffoldPass>();
}
} // namespace mlir::gawee
