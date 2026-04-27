//===----------------------------------------------------------------------===//
// Linalg Fusion Scaffold Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for producer/consumer fusion work that should
// happen after Gawee -> Linalg lowering and before bufferization.
//
// Typical future responsibilities:
//   - fuse elementwise producer/consumer chains
//   - fuse bias/add ops into nearby conv or matmul consumers when legal
//   - separate profitable fusion from "everything fuses" behavior
//   - make fusion decisions explicit before scheduling/vectorization
//
// Current behavior:
//   - no IR mutation
//   - collects likely fusion candidates and emits remarks
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

static bool isElementwiseGeneric(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  return llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
    return iteratorType == utils::IteratorType::parallel;
  });
}

static bool isConvOrMatmul(Operation *op) {
  return isa<linalg::Conv2DNchwFchwOp, linalg::MatmulOp,
             linalg::MatmulTransposeBOp>(op);
}

static bool hasSingleTensorResult(Operation *op) {
  return op->getNumResults() == 1 && isa<RankedTensorType>(op->getResult(0).getType());
}

static bool isLikelyFusionPair(Operation *producer, Operation *consumer) {
  if (!hasSingleTensorResult(producer))
    return false;
  if (producer->getBlock() != consumer->getBlock())
    return false;
  if (producer->getResult(0).hasOneUse() == false)
    return false;

  if (isConvOrMatmul(producer) && isElementwiseGeneric(consumer))
    return true;
  if (isElementwiseGeneric(producer) && isElementwiseGeneric(consumer))
    return true;
  return false;
}

static void analyzeProducerConsumerChains(ModuleOp module) {
  Builder builder(module.getContext());
  int64_t nextGroupId = 0;

  module.walk([&](Operation *op) {
    if (!isElementwiseGeneric(op) && !isConvOrMatmul(op))
      return;

    SmallString<128> message;
    llvm::raw_svector_ostream os(message);
    os << "fusion scaffold candidate";

    if (isConvOrMatmul(op))
      os << ": structured producer with likely post-op fusion opportunities";
    else
      os << ": elementwise generic op that may fuse with neighbors";

    if (hasSingleTensorResult(op)) {
      for (Operation *user : op->getResult(0).getUsers()) {
        if (!isa<linalg::LinalgOp>(user))
          continue;
        if (!isLikelyFusionPair(op, user))
          continue;

        op->setAttr("gawee.fusion.group",
                    builder.getI64IntegerAttr(nextGroupId));
        op->setAttr("gawee.fusion.role",
                    builder.getStringAttr("producer"));
        user->setAttr("gawee.fusion.group",
                      builder.getI64IntegerAttr(nextGroupId));
        user->setAttr("gawee.fusion.role",
                      builder.getStringAttr("consumer"));
        user->setAttr("gawee.fusion.source",
                      builder.getStringAttr(op->getName().getStringRef()));
        os << ", assigned fusion_group=" << nextGroupId;
        ++nextGroupId;
        break;
      }
    }

    op->emitRemark() << os.str();
  });

  module->setAttr("gawee.fusion.group_count",
                  builder.getI64IntegerAttr(nextGroupId));
}

struct LinalgFusionScaffoldPass
    : public PassWrapper<LinalgFusionScaffoldPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-fusion"; }

  StringRef getDescription() const override {
    return "Scaffold pass for post-lowering Linalg fusion planning";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Intended future order inside this pass:
    //   1. collect producer/consumer chains
    //   2. classify elementwise-only fusion opportunities
    //   3. classify structured-op + post-op fusion opportunities
    //   4. separate legal fusion from profitable fusion
    analyzeProducerConsumerChains(module);
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgFusionScaffoldPass() {
  return std::make_unique<LinalgFusionScaffoldPass>();
}
} // namespace mlir::gawee
