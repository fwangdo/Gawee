#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinalgSchedulingPass
    : public PassWrapper<LinalgSchedulingPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-scheduling"; }
  StringRef getDescription() const override {
    return "TODO: retype linalg scheduling pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement LinalgSchedulingPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createLinalgSchedulingPass() {
  return std::make_unique<LinalgSchedulingPass>();
}
