//===----------------------------------------------------------------------===//
// Linalg Scheduling Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for loop/order decisions that should happen
// after tiling/fusion planning and before bufferization / loop lowering.
//
// Typical future responsibilities:
//   - loop reordering
//   - parallel loop selection
//   - reduction-friendly scheduling
//   - deciding which loop structure should survive into SCF
//
// Current behavior:
//   - no IR mutation
//   - emits remarks about loop structure and scheduling pressure
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

static void analyzeLoopSchedulingNeeds(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](linalg::LinalgOp op) {
    auto iteratorTypes = op.getIteratorTypesArray();
    int64_t parallelCount = 0;
    int64_t reductionCount = 0;
    SmallVector<int64_t> parallelLoops;
    SmallVector<int64_t> reductionLoops;
    SmallVector<int64_t> interchange;
    interchange.reserve(iteratorTypes.size());

    for (int64_t i = 0, e = iteratorTypes.size(); i < e; ++i) {
      if (iteratorTypes[i] == utils::IteratorType::parallel)
        parallelLoops.push_back(i);
      if (iteratorTypes[i] == utils::IteratorType::reduction)
        reductionLoops.push_back(i);
    }
    interchange.append(parallelLoops.begin(), parallelLoops.end());
    interchange.append(reductionLoops.begin(), reductionLoops.end());

    for (utils::IteratorType iteratorType : iteratorTypes) {
      if (iteratorType == utils::IteratorType::parallel)
        ++parallelCount;
      if (iteratorType == utils::IteratorType::reduction)
        ++reductionCount;
    }

    SmallString<128> message;
    llvm::raw_svector_ostream os(message);
    os << "scheduling plan: parallel_loops=" << parallelCount
       << ", reduction_loops=" << reductionCount;
    if (reductionCount > 0)
      os << ", reduction ordering likely matters";
    else
      os << ", loop reorder/parallel split is the likely next lever";

    op->setAttr("gawee.schedule.parallel_loops",
                builder.getDenseI64ArrayAttr(parallelLoops));
    op->setAttr("gawee.schedule.reduction_loops",
                builder.getDenseI64ArrayAttr(reductionLoops));
    op->setAttr("gawee.schedule.interchange",
                builder.getDenseI64ArrayAttr(interchange));
    op->setAttr(
        "gawee.schedule.kind",
        builder.getStringAttr(reductionCount > 0 ? "reduction-aware"
                                                 : "parallel-first"));
    op->emitRemark() << os.str();
  });
}

struct LinalgSchedulingPass
    : public PassWrapper<LinalgSchedulingPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-scheduling"; }

  StringRef getDescription() const override {
    return "Post-lowering Linalg scheduling planning pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    analyzeLoopSchedulingNeeds(getOperation());
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgSchedulingPass() {
  return std::make_unique<LinalgSchedulingPass>();
}
} // namespace mlir::gawee
