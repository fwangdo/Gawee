//===----------------------------------------------------------------------===//
// Bufferization Prep Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for transformations that should happen
// immediately before one-shot bufferization.
//
// Typical future responsibilities:
//   - normalize destination-passing style expectations
//   - insert / rewrite alloc_tensor-style producers when needed
//   - clean up patterns that confuse one-shot bufferize
//   - separate "bufferization preparation" from bufferization itself
//
// Current behavior:
//   - no IR mutation
//   - keeps the pipeline slot explicit
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static bool replaceTensorEmptyWithAllocTensor(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  IRRewriter rewriter(ctx);
  bool changed = false;

  module.walk([&](tensor::EmptyOp emptyOp) {
    rewriter.setInsertionPoint(emptyOp);
    auto allocTensor = rewriter.create<bufferization::AllocTensorOp>(
        emptyOp.getLoc(), cast<RankedTensorType>(emptyOp.getType()),
        emptyOp.getDynamicSizes());
    rewriter.replaceOp(emptyOp, allocTensor.getResult());
    changed = true;
  });

  return changed;
}

static bool foldNoOpTensorCast(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  IRRewriter rewriter(ctx);
  bool changed = false;

  module.walk([&](tensor::CastOp castOp) {
    if (castOp.getSource().getType() != castOp.getType())
      return;
    rewriter.replaceOp(castOp, castOp.getSource());
    changed = true;
  });

  return changed;
}

static void annotateDestinationStyleOps(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](linalg::LinalgOp op) {
    op->setAttr("gawee.bufferize.destination_count",
                builder.getI64IntegerAttr(op.getNumDpsInits()));
    op->setAttr("gawee.bufferize.input_count",
                builder.getI64IntegerAttr(op.getNumDpsInputs()));
    op->setAttr("gawee.bufferize.all_tensor_semantics",
                builder.getBoolAttr(op.hasPureTensorSemantics()));
  });
}

struct BufferizePrepPass
    : public PassWrapper<BufferizePrepPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-bufferize-prep"; }

  StringRef getDescription() const override {
    return "Pre-bufferization cleanup and normalization pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = false;

    changed |= foldNoOpTensorCast(module);
    changed |= replaceTensorEmptyWithAllocTensor(module);
    annotateDestinationStyleOps(module);

    Builder builder(module.getContext());
    module->setAttr("gawee.bufferize.prep_ran", builder.getUnitAttr());
    module->setAttr("gawee.bufferize.prep_changed",
                    builder.getBoolAttr(changed));
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeBufferizePrepPass() {
  return std::make_unique<BufferizePrepPass>();
}
} // namespace mlir::gawee
