//===----------------------------------------------------------------------===//
// Linalg Vectorization Pass
//===----------------------------------------------------------------------===//
//
// This pass vectorizes eligible linalg ops into vector dialect ops.
//
// Current behavior:
//   - vectorizes elementwise linalg.generic ops with static shapes
//   - vectorizes tiled named ops (conv, matmul) with small static shapes
//   - skips ops with large shapes to avoid generating huge vectors
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
/// - all indexing maps are projected permutations (identity, broadcast, or
///   permutation — but no complex affine expressions)
static bool isVectorizableElementwise(linalg::LinalgOp op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp)
    return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      }))
    return false;
  // Allow projected permutations (includes identity and broadcast maps).
  // Also allow scalar maps like (d0,d1,d2) -> () for scalar broadcast operands.
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    if (map.getNumResults() == 0)
      continue; // scalar operand — always safe to broadcast
    if (!map.isProjectedPermutation())
      return false;
  }
  return true;
}

/// Return true if the op is a named linalg op that linalg::vectorize() knows
/// how to handle (conv, matmul, fill, etc.). These are typically the tiled
/// versions with small shapes that fit in vector registers.
static bool isVectorizableNamedOp(linalg::LinalgOp op) {
  return isa<linalg::Conv2DNchwFchwOp, linalg::MatmulOp,
             linalg::MatmulTransposeBOp, linalg::FillOp>(op.getOperation());
}

static void vectorizeEligibleOps(ModuleOp module) {
  // Collect candidates first — vectorize() replaces ops.
  SmallVector<linalg::LinalgOp> candidates;
  module.walk([&](linalg::LinalgOp op) {
    bool eligible = isVectorizableElementwise(op) || isVectorizableNamedOp(op);
    if (!eligible)
      return;
    // Only vectorize ops with small static shapes. Max 32 per dimension
    // keeps vectors within what LLVM backends can handle on CPU targets.
    if (!allShapesSmallAndStatic(op, /*maxDim=*/32))
      return;
    candidates.push_back(op);
  });

  IRRewriter rewriter(module.getContext());
  for (linalg::LinalgOp op : candidates) {
    rewriter.setInsertionPoint(op);
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op);
    if (succeeded(result)) {
      rewriter.replaceOp(op, result->replacements);
    }
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
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    vectorizeEligibleOps(getOperation());
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
} // namespace mlir::gawee
