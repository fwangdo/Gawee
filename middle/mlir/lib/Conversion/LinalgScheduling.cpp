//===----------------------------------------------------------------------===//
// Linalg Scheduling Pass
//===----------------------------------------------------------------------===//
//
// This pass applies loop reordering decisions after tiling/fusion and before
// bufferization / loop lowering.
//
// Current behavior:
//   - for linalg.generic ops with mixed parallel/reduction iterators,
//     applies interchangeGenericOp to bring parallel loops before reductions
//   - named ops (matmul, conv) have fixed loop order and are left unchanged
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

/// Check if the interchange vector is the identity permutation.
static bool isIdentityPermutation(ArrayRef<unsigned> perm) {
  for (unsigned i = 0; i < perm.size(); ++i) {
    if (perm[i] != i)
      return false;
  }
  return true;
}

static void applyLoopInterchange(ModuleOp module) {
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp op) { candidates.push_back(op); });

  IRRewriter rewriter(module.getContext());
  for (linalg::GenericOp op : candidates) {
    auto iteratorTypes = op.getIteratorTypesArray();

    // Build interchange: parallel loops first, then reductions.
    SmallVector<unsigned> interchange;
    for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
      if (iteratorTypes[i] == utils::IteratorType::parallel)
        interchange.push_back(i);
    }
    for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
      if (iteratorTypes[i] == utils::IteratorType::reduction)
        interchange.push_back(i);
    }

    // Skip if already in the right order or no reductions present.
    if (interchange.size() != iteratorTypes.size() ||
        isIdentityPermutation(interchange))
      continue;

    rewriter.setInsertionPoint(op);
    FailureOr<linalg::GenericOp> result =
        linalg::interchangeGenericOp(rewriter, op, interchange);
    if (failed(result)) {
      op->emitRemark() << "loop interchange failed";
      continue;
    }
  }
}

struct LinalgSchedulingPass
    : public PassWrapper<LinalgSchedulingPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-scheduling"; }

  StringRef getDescription() const override {
    return "Post-lowering Linalg loop interchange pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    applyLoopInterchange(getOperation());
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgSchedulingPass() {
  return std::make_unique<LinalgSchedulingPass>();
}
} // namespace mlir::gawee
