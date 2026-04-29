//===----------------------------------------------------------------------===//
// Linalg Verification Pass
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static void emitVerificationSummary(ModuleOp module) {
  Builder builder(module.getContext());
  int64_t linalgCount = 0;
  int64_t genericCount = 0;
  int64_t structuredCount = 0;
  int64_t plannedCount = 0;

  module.walk([&](linalg::LinalgOp op) {
    ++linalgCount;
    if (isa<linalg::GenericOp>(op.getOperation()))
      ++genericCount;
    else
      ++structuredCount;
    if (op->hasAttr("gawee.transform.kind"))
      ++plannedCount;
  });

  module.emitRemark() << "verification summary: linalg_ops="
                      << linalgCount << ", generic_ops=" << genericCount
                      << ", structured_ops=" << structuredCount
                      << ", planned_ops=" << plannedCount;
  module->setAttr("gawee.verify.linalg_count",
                  builder.getI64IntegerAttr(linalgCount));
  module->setAttr("gawee.verify.planned_count",
                  builder.getI64IntegerAttr(plannedCount));
}

static void verifyPerOpExpectations(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](linalg::LinalgOp op) {
    bool hasTransformPlan = op->hasAttr("gawee.transform.kind");
    bool hasSchedule = op->hasAttr("gawee.schedule.kind");
    bool hasVectorInfo = op->hasAttr("gawee.vector.kind");
    op->setAttr("gawee.verify.has_transform_plan",
                builder.getBoolAttr(hasTransformPlan));
    op->setAttr("gawee.verify.has_schedule",
                builder.getBoolAttr(hasSchedule));
    op->setAttr("gawee.verify.has_vector_info",
                builder.getBoolAttr(hasVectorInfo));

    if (op.getNumDpsInits() == 0) {
      op.emitRemark()
          << "verification: op has no destination operands; "
             "check whether destination style expectations still hold";
      op->setAttr("gawee.verify.status",
                  builder.getStringAttr("needs-dps-review"));
      return;
    }

    op.emitRemark()
        << "verification: destination-style op is present for later "
           "bufferization/perf checks";
    op->setAttr("gawee.verify.status", builder.getStringAttr("ok"));
  });
}

struct LinalgVerificationPass
    : public PassWrapper<LinalgVerificationPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "gawee-linalg-verification";
  }

  StringRef getDescription() const override {
    return "Post-lowering Linalg verification pass";
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
std::unique_ptr<Pass> createLinalgVerificationPass() {
  return std::make_unique<LinalgVerificationPass>();
}
} // namespace mlir::gawee
