//===----------------------------------------------------------------------===//
// Linalg Verification Scaffold Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for middle-end verification hooks.
//
// Typical future responsibilities:
//   - validate that expected transform preconditions are still true
//   - check whether intended tiling/fusion/vectorization candidates survived
//   - emit structured diagnostics before bufferization/loop lowering
//   - become the bridge between IR transforms and correctness/perf evaluation
//
// Current behavior:
//   - no IR mutation
//   - emits a module-level summary and simple per-op verification remarks
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static void emitVerificationSummary(ModuleOp module) {
  int64_t linalgCount = 0;
  int64_t genericCount = 0;
  int64_t structuredCount = 0;

  module.walk([&](linalg::LinalgOp op) {
    ++linalgCount;
    if (isa<linalg::GenericOp>(op.getOperation()))
      ++genericCount;
    else
      ++structuredCount;
  });

  module.emitRemark() << "verification scaffold summary: linalg_ops="
                      << linalgCount << ", generic_ops=" << genericCount
                      << ", structured_ops=" << structuredCount;
}

static void verifyPerOpExpectations(ModuleOp module) {
  module.walk([&](linalg::LinalgOp op) {
    if (op.getNumDpsInits() == 0) {
      op.emitRemark()
          << "verification scaffold: op has no destination operands; "
             "check whether destination style expectations still hold";
      return;
    }

    op.emitRemark()
        << "verification scaffold: destination-style op is present for later "
           "bufferization/perf checks";
  });
}

struct LinalgVerificationScaffoldPass
    : public PassWrapper<LinalgVerificationScaffoldPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-verification";
  }

  StringRef getDescription() const override {
    return "Scaffold pass for post-lowering Linalg verification hooks";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    emitVerificationSummary(module);
    verifyPerOpExpectations(module);
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgVerificationScaffoldPass() {
  return std::make_unique<LinalgVerificationScaffoldPass>();
}
} // namespace mlir::gawee
