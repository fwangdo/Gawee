#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinalgVerificationPass
    : public PassWrapper<LinalgVerificationPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-verification";
  }
  StringRef getDescription() const override {
    return "TODO: retype linalg verification pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement LinalgVerificationPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createLinalgVerificationPass() {
  return std::make_unique<LinalgVerificationPass>();
}
