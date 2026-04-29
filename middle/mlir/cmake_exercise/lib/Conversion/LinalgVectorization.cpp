#include "Conversion/GaweePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LinalgVectorizationPass
    : public PassWrapper<LinalgVectorizationPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-vectorization";
  }
  StringRef getDescription() const override {
    return "TODO: retype linalg vectorization pass";
  }

  void runOnOperation() override {
    // # TODO
    getOperation()->emitRemark("TODO: implement LinalgVectorizationPass");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::gawee::createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
