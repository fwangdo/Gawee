//===- DecomposeAggregatedLinalgOps.cpp - Decompose linalg.softmax etc. ---===//
//
// Walks the module and calls decomposeOperation() on every op that implements
// AggregatedOpInterface (e.g. linalg.softmax).  This converts them into
// primitive linalg.generic / arith / math ops that the standard
// linalg-to-loops pipeline can lower.
//
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct DecomposeAggregatedPattern
    : public OpInterfaceRewritePattern<linalg::AggregatedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::AggregatedOpInterface op,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> result = op.decomposeOperation(rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

struct DecomposeAggregatedLinalgOpsPass
    : public PassWrapper<DecomposeAggregatedLinalgOpsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      DecomposeAggregatedLinalgOpsPass)

  StringRef getArgument() const override {
    return "decompose-aggregated-linalg-ops";
  }
  StringRef getDescription() const override {
    return "Decompose aggregated linalg ops (e.g. softmax) into primitives";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeAggregatedPattern>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::gawee::createDecomposeAggregatedLinalgOpsPass() {
  return std::make_unique<DecomposeAggregatedLinalgOpsPass>();
}
