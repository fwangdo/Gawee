#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct DecomposeAggregatedLinalgOpsPass
    : public PassWrapper<DecomposeAggregatedLinalgOpsPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "decompose-aggregated-linalg-ops";
  }
  StringRef getDescription() const override {
    return "TODO: retype aggregated linalg decomposition pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark(
        "TODO: implement DecomposeAggregatedLinalgOpsPass");
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::gawee::createDecomposeAggregatedLinalgOpsPass() {
  return std::make_unique<DecomposeAggregatedLinalgOpsPass>();
}
