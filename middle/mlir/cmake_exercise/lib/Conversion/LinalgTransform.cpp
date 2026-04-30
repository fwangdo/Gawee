#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinalgTransformPass
    : public PassWrapper<LinalgTransformPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-transform"; }
  StringRef getDescription() const override {
    return "TODO: retype linalg transform pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement LinalgTransformPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createLinalgTransformPass() {
  return std::make_unique<LinalgTransformPass>();
}
