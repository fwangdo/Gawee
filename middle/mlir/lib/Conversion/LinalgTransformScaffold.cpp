//===----------------------------------------------------------------------===//
// Linalg Transform Scaffold Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for middle-end transforms that operate after
// Gawee -> Linalg lowering and before bufferization.
//
// Important:
//   - this is NOT the Linalg -> SCF lowering pass
//   - this pass should usually keep the IR in the Linalg/tensor/arith world
//   - the goal here is to improve the Linalg IR before it is lowered further
//
// In other words:
//   Gawee -> Linalg             : legalization / first lowering
//   this pass                   : optimize / restructure Linalg IR
//   convert-linalg-to-loops     : actual Linalg -> SCF lowering
//
// Typical future responsibilities:
//   - tiling
//   - fusion
//   - scheduling / loop reordering
//   - canonicalization around linalg ops
//   - vectorization preparation
//
// How to think about implementation:
//   - the pass is the pipeline slot
//   - the real logic will usually be split by target op family
//   - e.g. one helper for conv tiling, one for matmul tiling, one for generic
//     fusion, etc.
//
// So yes: tiling is often op-specific.
// The pass should own "when this transform runs",
// while helper functions / pattern groups own "how this op is transformed".
//
// Current behavior:
//   - no IR mutation
//   - leaves explicit TODO markers in code only
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// -------------------------------------------------------------------------
// Future op-family helpers
//
// The pass should stay small. Real transform logic should move into helpers
// that each own one transform family or one target-op family.
// -------------------------------------------------------------------------

static bool isConvLikeOp(Operation *op) {
  return isa<linalg::Conv2DNchwFchwOp>(op);
}

static bool isMatmulLikeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op);
}

static bool isGenericOp(Operation *op) {
  return isa<linalg::GenericOp>(op);
}

static void tileConvLikeOps(ModuleOp module) {
  // TODO:
  // 1. collect conv-like ops
  // 2. choose tile sizes for N/C/H/W or output spatial loops
  // 3. decide whether padding / halo regions affect the strategy
  // 4. keep the result in Linalg/SCF-friendly form, not LLVM
  module.walk([&](Operation *op) {
    if (!isConvLikeOp(op)) {
      return;
    }
    // TODO: apply conv-specific tiling here.
  });
}

static void tileMatmulLikeOps(ModuleOp module) {
  // TODO:
  // 1. collect matmul-like ops
  // 2. choose tile sizes for M/N/K loops
  // 3. decide whether to tile only, or tile+fuse follow-up bias/add
  module.walk([&](Operation *op) {
    if (!isMatmulLikeOp(op)) {
      return;
    }
    // TODO: apply matmul-specific tiling here.
  });
}

static void scheduleOrFuseGenericOps(ModuleOp module) {
  // TODO:
  // 1. collect linalg.generic ops that are worth touching
  // 2. separate pure elementwise cases from reduction cases
  // 3. decide whether to fuse producer/consumer chains here or in a later pass
  module.walk([&](Operation *op) {
    if (!isGenericOp(op)) {
      return;
    }
    // TODO: apply generic-op scheduling / fusion here.
  });
}

struct LinalgTransformScaffoldPass
    : public PassWrapper<LinalgTransformScaffoldPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-transform"; }

  StringRef getDescription() const override {
    return "Scaffold pass for post-lowering Linalg tiling/scheduling transforms";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The pass owns transform order.
    // The helpers own per-op-family transform details.
    //
    // Current intended order:
    //   1. conv-like tiling
    //   2. matmul-like tiling
    //   3. generic-op scheduling / fusion
    //
    // This keeps the "pipeline-level decision" separate from the
    // "op-specific implementation" decision.

    tileConvLikeOps(module);
    tileMatmulLikeOps(module);
    scheduleOrFuseGenericOps(module);
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgTransformScaffoldPass() {
  return std::make_unique<LinalgTransformScaffoldPass>();
}
} // namespace mlir::gawee
