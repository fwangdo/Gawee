#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinalgFusionPass
    : public PassWrapper<LinalgFusionPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-fusion"; }
  StringRef getDescription() const override {
    return "TODO: retype linalg fusion pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement LinalgFusionPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createLinalgFusionPass() {
  return std::make_unique<LinalgFusionPass>();
}
