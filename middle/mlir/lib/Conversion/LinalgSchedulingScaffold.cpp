//===----------------------------------------------------------------------===//
// Linalg Scheduling Scaffold Pass
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

static void analyzeLoopSchedulingNeeds(ModuleOp module) {
  module.walk([&](linalg::LinalgOp op) {
    auto iteratorTypes = op.getIteratorTypesArray();
    int64_t parallelCount = 0;
    int64_t reductionCount = 0;
    for (utils::IteratorType iteratorType : iteratorTypes) {
      if (iteratorType == utils::IteratorType::parallel)
        ++parallelCount;
      if (iteratorType == utils::IteratorType::reduction)
        ++reductionCount;
    }

    SmallString<128> message;
    llvm::raw_svector_ostream os(message);
    os << "scheduling scaffold: parallel_loops=" << parallelCount
       << ", reduction_loops=" << reductionCount;
    if (reductionCount > 0)
      os << ", reduction ordering likely matters";
    else
      os << ", loop reorder/parallel split is the likely next lever";
    op->emitRemark() << os.str();
  });
}

struct LinalgSchedulingScaffoldPass
    : public PassWrapper<LinalgSchedulingScaffoldPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-scheduling"; }

  StringRef getDescription() const override {
    return "Scaffold pass for post-lowering Linalg scheduling planning";
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
std::unique_ptr<Pass> createLinalgSchedulingScaffoldPass() {
  return std::make_unique<LinalgSchedulingScaffoldPass>();
}
} // namespace mlir::gawee
