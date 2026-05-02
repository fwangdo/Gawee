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
//   - peels tail iterations from scf.for loops so that main loop bodies
//     contain only full tiles (preparation for future vectorization)
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
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

/// Peel tail iterations from scf.for loops produced by tiling.
///
/// After tiling, a loop like `for i = 0 to 14 step 8` has iterations where
/// the last tile is partial (covers only indices 8..13 instead of a full
/// 8-element tile). Peeling splits this into:
///   - main loop:   for i = 0 to 8 step 8   (always full tiles)
///   - tail loop:   for i = 8 to 14 step 8  (partial remainder)
///
/// The main loop body can then be safely vectorized because every iteration
/// processes exactly `step` elements.
///
/// Peeling is skipped automatically when it is unnecessary:
///   - step == 1 (every iteration is trivially "full")
///   - bounds are already evenly divisible by step
///   - bounds or step are dynamic (conservative skip)
static void applyLoopPeeling(ModuleOp module) {
  // Collect ForOps first, because peeling creates new ForOps and we don't
  // want to visit those during the same walk.
  SmallVector<scf::ForOp> forOps;
  module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  IRRewriter rewriter(module.getContext());
  for (scf::ForOp forOp : forOps) {
    scf::ForOp partialIteration;
    // peelForLoopAndSimplifyBounds returns failure when peeling is not
    // applicable (step==1, already divides evenly, dynamic bounds, etc.).
    // That is expected — just skip.
    (void)scf::peelForLoopAndSimplifyBounds(rewriter, forOp,
                                            partialIteration);
  }
}

struct LinalgSchedulingPass
    : public PassWrapper<LinalgSchedulingPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-scheduling"; }

  StringRef getDescription() const override {
    return "Post-lowering Linalg loop interchange and peeling pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    applyLoopInterchange(module);
    applyLoopPeeling(module);
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgSchedulingPass() {
  return std::make_unique<LinalgSchedulingPass>();
}
} // namespace mlir::gawee
