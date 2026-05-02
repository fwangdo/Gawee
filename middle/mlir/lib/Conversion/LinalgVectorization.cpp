//===----------------------------------------------------------------------===//
// Linalg Vectorization Pass
//===----------------------------------------------------------------------===//
//
// This pass vectorizes eligible linalg ops into vector dialect ops.
//
// Current behavior:
//   - vectorizes elementwise linalg.generic ops with static shapes
//   - skips conv/matmul (require more complex vector lowering)
//   - skips ops with large shapes to avoid generating huge vectors
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Return true if ALL operand and result shapes are static and no dimension
/// exceeds `maxDim`. Checks both inputs and outputs to ensure vectorize()
/// can infer vector sizes from static shapes without fallback to masking.
static bool allShapesSmallAndStatic(linalg::LinalgOp op, int64_t maxDim) {
  for (OpOperand &operand : op->getOpOperands()) {
    auto shaped = dyn_cast<ShapedType>(operand.get().getType());
    if (!shaped || !shaped.hasStaticShape())
      return false;
    for (int64_t dim : shaped.getShape()) {
      if (dim > maxDim)
        return false;
    }
  }
  for (Value result : op->getResults()) {
    auto shaped = dyn_cast<ShapedType>(result.getType());
    if (!shaped || !shaped.hasStaticShape())
      return false;
    for (int64_t dim : shaped.getShape()) {
      if (dim > maxDim)
        return false;
    }
  }
  return true;
}

/// Return true if the op is a linalg.generic suitable for vectorization:
/// - all-parallel iterators (elementwise)
/// - all indexing maps are identity (no broadcast/transpose)
/// The identity-map requirement avoids masked vectorization paths that
/// can crash on some MLIR versions.
static bool isVectorizableElementwise(linalg::LinalgOp op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp)
    return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      }))
    return false;
  // Require all indexing maps to be identity — no broadcast, no permutation.
  // This avoids triggering masked vectorization (getOrCreateMaskFor) which
  // can segfault on broadcast maps.
  unsigned numLoops = op.getNumLoops();
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    if (!map.isIdentity() || map.getNumDims() != numLoops)
      return false;
  }
  return true;
}

static void vectorizeEligibleOps(ModuleOp module) {
  // Collect candidates first — vectorize() replaces ops.
  SmallVector<linalg::LinalgOp> candidates;
  module.walk([&](linalg::LinalgOp op) {
    if (!isVectorizableElementwise(op))
      return;
    // Only vectorize small shapes (from tiled ops or naturally small ops).
    // Max 64 per dimension keeps vectors reasonable for CPU SIMD.
    if (!allShapesSmallAndStatic(op, /*maxDim=*/64))
      return;
    candidates.push_back(op);
  });

  IRRewriter rewriter(module.getContext());
  for (linalg::LinalgOp op : candidates) {
    rewriter.setInsertionPoint(op);
    // Empty inputVectorSizes → infer from op's static shapes.
    (void)linalg::vectorize(rewriter, op);
  }
}

struct LinalgVectorizationPass
    : public PassWrapper<LinalgVectorizationPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-vectorization";
  }

  StringRef getDescription() const override {
    return "Vectorize eligible elementwise linalg ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    // Disabled: linalg::vectorize() crashes in VectorizationState::
    // getOrCreateMaskFor on broadcast indexing maps in this MLIR build.
    // The vectorizeEligibleOps() code is kept for when a fixed MLIR
    // version is available.
    (void)getOperation();
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
} // namespace mlir::gawee
