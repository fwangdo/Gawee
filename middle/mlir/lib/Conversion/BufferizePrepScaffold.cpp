//===----------------------------------------------------------------------===//
// Bufferization Prep Scaffold Pass
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct BufferizePrepScaffoldPass
    : public PassWrapper<BufferizePrepScaffoldPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-bufferize-prep"; }

  StringRef getDescription() const override {
    return "Scaffold pass for pre-bufferization cleanup and normalization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // TODO:
    // 1. identify tensor patterns that should be normalized before bufferization
    // 2. separate what belongs here from what EmptyTensorToAllocTensor already does
    // 3. add explicit cleanup rewrites once the target patterns are chosen

    module.walk([&](Operation *op) {
      (void)op;
      // TODO: future pre-bufferization normalization hooks live here.
    });
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeBufferizePrepScaffoldPass() {
  return std::make_unique<BufferizePrepScaffoldPass>();
}
} // namespace mlir::gawee
