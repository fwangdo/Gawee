#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct BufferizePrepPass
    : public PassWrapper<BufferizePrepPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-bufferize-prep"; }
  StringRef getDescription() const override {
    return "TODO: retype bufferization prep pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement BufferizePrepPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createGaweeBufferizePrepPass() {
  return std::make_unique<BufferizePrepPass>();
}
