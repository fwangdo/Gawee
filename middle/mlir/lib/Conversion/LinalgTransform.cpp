//===----------------------------------------------------------------------===//
// Linalg Transform Pass
//===----------------------------------------------------------------------===//
//
// This pass is the intended home for middle-end transforms that operate after
// Gawee -> Linalg lowering and before bufferization.
//
// Important:
//   - this is NOT the Linalg -> SCF lowering pass
//   - this pass should usually keep the IR in the Linalg/tensor/arith world
//   - the goal here is to improve the Linalg IR before it is lowered further
//
// In other words:
//   Gawee -> Linalg             : legalization / first lowering
//   this pass                   : optimize / restructure Linalg IR
//   convert-linalg-to-loops     : actual Linalg -> SCF lowering
//
// Typical future responsibilities:
//   - tiling
//   - fusion
//   - scheduling / loop reordering
//   - canonicalization around linalg ops
//   - vectorization preparation
//
// How to think about implementation:
//   - the pass is the pipeline slot
//   - the real logic will usually be split by target op family
//   - e.g. one helper for conv tiling, one for matmul tiling, one for generic
//     fusion, etc.
//
// So yes: tiling is often op-specific.
// The pass should own "when this transform runs",
// while helper functions / pattern groups own "how this op is transformed".
//
// Current behavior:
//   - still no IR mutation
//   - but now the pass performs explicit analysis and planning
//   - it records "what would be transformed" and "why"
//   - that makes the future tiling/scheduling code much easier to add
//===----------------------------------------------------------------------===//

#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Why this file now has "analysis structs"
//
// Tiling is rarely "apply the same transform to every op".
// Even inside Linalg, different op families want different decisions:
//   - conv: usually care about output H/W and maybe channels
//   - matmul: usually care about M/N/K tiles
//   - generic: may be elementwise, may be reduction, may be a poor target
//
// A practical first implementation is:
//   1. collect candidate ops
//   2. inspect their shapes / iterator types
//   3. build an explicit "plan" object
//   4. later, plug real transform code into that plan
//
// This file therefore does real inspection work, but intentionally stops
// before rewriting IR. The goal is to make the transform intent obvious.
//===----------------------------------------------------------------------===//

enum class TransformPriority {
  Low,
  Medium,
  High,
};

static StringRef stringifyPriority(TransformPriority priority) {
  switch (priority) {
  case TransformPriority::Low:
    return "low";
  case TransformPriority::Medium:
    return "medium";
  case TransformPriority::High:
    return "high";
  }
  llvm_unreachable("unknown transform priority");
}

struct ConvTilingPlan {
  Operation *operation = nullptr;
  linalg::LinalgOp op;
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> interchangeVector;
  bool tileOutputSpatialLoops = false;
  bool tileChannelLoop = false;
  TransformPriority priority = TransformPriority::Low;
  SmallString<128> rationale;
};

struct MatmulTilingPlan {
  Operation *operation = nullptr;
  linalg::LinalgOp op;
  SmallVector<int64_t> tileSizes;
  SmallVector<int64_t> interchangeVector;
  bool tileReductionLoop = false;
  TransformPriority priority = TransformPriority::Low;
  SmallString<128> rationale;
};

struct GenericTransformPlan {
  Operation *operation = nullptr;
  linalg::GenericOp op;
  bool isElementwise = false;
  bool hasReduction = false;
  bool worthFusion = false;
  TransformPriority priority = TransformPriority::Low;
  SmallString<128> rationale;
};

// -------------------------------------------------------------------------
// Utility helpers
//
// These are deliberately small and readable. The point is not to be clever;
// the point is to make every decision easy to explain to a novice reader.
// -------------------------------------------------------------------------

static bool isStaticTensorShape(Value value) {
  auto rankedType = dyn_cast<RankedTensorType>(value.getType());
  return rankedType && rankedType.hasStaticShape();
}

static FailureOr<RankedTensorType> getSingleResultTensorType(Operation *op) {
  if (op->getNumResults() != 1) {
    return failure();
  }

  auto rankedType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rankedType) {
    return failure();
  }
  return rankedType;
}

static SmallVector<int64_t> buildUnitTileVector(unsigned count) {
  return SmallVector<int64_t>(count, 1);
}

static void appendText(SmallString<128> &buffer, StringRef text) {
  if (!buffer.empty()) {
    buffer += " ";
  }
  buffer += text;
}

static void emitPlanRemark(Operation *op, StringRef header, StringRef details) {
  op->emitRemark() << header << ": " << details;
}

static StringAttr makeStringAttr(Operation *op, StringRef text) {
  return StringAttr::get(op->getContext(), text);
}

static void setPlanMetadata(Operation *op, StringRef kind,
                            TransformPriority priority,
                            ArrayRef<int64_t> tileSizes) {
  Builder builder(op->getContext());
  op->setAttr("gawee.transform.kind", makeStringAttr(op, kind));
  op->setAttr("gawee.transform.priority",
              makeStringAttr(op, stringifyPriority(priority)));
  if (!tileSizes.empty()) {
    op->setAttr("gawee.transform.tile_sizes",
                builder.getDenseI64ArrayAttr(tileSizes));
  }
}

static void setPlanNote(Operation *op, StringRef attrName, StringRef text) {
  op->setAttr(attrName, makeStringAttr(op, text));
}

static bool isElementwiseGenericHeuristic(linalg::GenericOp genericOp) {
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      })) {
    return false;
  }

  // This pass uses a deliberately simple heuristic:
  // if every loop is parallel and all indexing maps are projected
  // permutations, then this generic op behaves like a plain elementwise op
  // for middle-end planning purposes.
  return llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isProjectedPermutation(); });
}

static bool isConvLikeOp(Operation *op) {
  return isa<linalg::Conv2DNchwFchwOp>(op);
}

static bool isMatmulLikeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op);
}

static bool isGenericOp(Operation *op) {
  return isa<linalg::GenericOp>(op);
}

static ConvTilingPlan buildConvPlan(linalg::LinalgOp linalgOp) {
  ConvTilingPlan plan;
  plan.operation = linalgOp.getOperation();
  plan.op = linalgOp;
  plan.parallelTileSizes =
      buildUnitTileVector(linalgOp.getNumParallelLoops());

  appendText(plan.rationale,
             "Start with unit tiles so the pass is always valid.");

  auto maybeResultType = getSingleResultTensorType(linalgOp.getOperation());
  if (failed(maybeResultType)) {
    appendText(plan.rationale,
               "Single ranked tensor result not found, so only minimal "
               "planning is possible.");
    return plan;
  }

  RankedTensorType resultType = *maybeResultType;
  ArrayRef<int64_t> shape = resultType.getShape();

  // NCHW output convention:
  //   dim 0: batch
  //   dim 1: output channels
  //   dim 2: output height
  //   dim 3: output width
  //
  // For many conv kernels, H/W tiling is the first obvious win because
  // spatial loops are usually large and independent.
  if (shape.size() == 4) {
    plan.tileOutputSpatialLoops = true;
    plan.priority = TransformPriority::High;

    // The loop order for the named Linalg conv op is not "tensor rank order"
    // by accident; it is chosen by the op definition. This pass keeps the
    // decision simple:
    //   - leave batch and channel loops as 1 for now
    //   - tile output H/W loops if they are statically known
    if (!ShapedType::isDynamic(shape[2])) {
      plan.parallelTileSizes[2] = std::min<int64_t>(shape[2], 8);
    }
    if (!ShapedType::isDynamic(shape[3])) {
      plan.parallelTileSizes[3] = std::min<int64_t>(shape[3], 8);
    }

    if (!ShapedType::isDynamic(shape[1]) && shape[1] >= 16) {
      plan.tileChannelLoop = true;
      plan.parallelTileSizes[1] = 8;
      appendText(plan.rationale,
                 "Output channel count is reasonably large, so channel tiling "
                 "is marked as worth trying.");
    } else {
      appendText(plan.rationale,
                 "Spatial tiling is preferred first; channel tiling remains "
                 "conservative.");
    }
  } else {
    appendText(plan.rationale,
               "Result is not rank-4 NCHW, so no conv-specific spatial plan "
               "is inferred.");
  }

  // Tile loop interchange for conv.
  // Conv2D NCHW_FCHW has 7 loops: N(0), C_out(1), H(2), W(3), C_in(4), KH(5), KW(6).
  // Default tiled loop order is (N, C_out, H, W, ...).
  // Reorder to (N, H, W, C_out, ...) so that spatial dims are outermost,
  // improving locality for NCHW output layout.
  unsigned numLoops = linalgOp.getNumLoops();
  if (numLoops >= 4 && plan.tileOutputSpatialLoops) {
    // Only interchange the tiled (parallel) dims: 0->0, 1->2, 2->3, 3->1.
    // Untiled reduction dims keep their relative order.
    plan.interchangeVector = {0, 2, 3, 1};
    for (unsigned i = 4; i < numLoops; ++i)
      plan.interchangeVector.push_back(i);
    appendText(plan.rationale,
               "Tile loop interchange: (N,H,W,C_out) for spatial locality.");
  }

  return plan;
}

static MatmulTilingPlan buildMatmulPlan(linalg::LinalgOp linalgOp) {
  MatmulTilingPlan plan;
  plan.operation = linalgOp.getOperation();
  plan.op = linalgOp;
  plan.tileSizes = buildUnitTileVector(linalgOp.getNumLoops());
  appendText(plan.rationale,
             "Start with unit tiles; later passes can replace them with "
             "hardware-aware values.");

  auto maybeResultType = getSingleResultTensorType(linalgOp.getOperation());
  if (failed(maybeResultType)) {
    appendText(plan.rationale,
               "Single ranked tensor result not found, so M/N planning stops "
               "early.");
    return plan;
  }

  RankedTensorType resultType = *maybeResultType;
  if (resultType.getRank() != 2) {
    appendText(plan.rationale,
               "Result is not rank-2, so this does not look like the usual "
               "matmul output shape.");
    return plan;
  }

  plan.priority = TransformPriority::High;

  ArrayRef<int64_t> shape = resultType.getShape();
  if (!ShapedType::isDynamic(shape[0])) {
    plan.tileSizes[0] = std::min<int64_t>(shape[0], 32);
  }
  if (!ShapedType::isDynamic(shape[1])) {
    plan.tileSizes[1] = std::min<int64_t>(shape[1], 32);
  }

  // The third loop in a standard matmul is the K reduction loop.
  if (linalgOp.getNumLoops() >= 3) {
    plan.tileReductionLoop = true;
    plan.tileSizes[2] = 16;
    appendText(plan.rationale,
               "Reduction tiling is explicitly marked because matmul often "
               "benefits from K blocking.");
  }

  // Matmul loop order (M, N, K) is already good: output dims (M, N) are
  // outermost and reduction (K) is innermost. No interchange needed.

  return plan;
}

static GenericTransformPlan buildGenericPlan(linalg::GenericOp genericOp) {
  GenericTransformPlan plan;
  plan.operation = genericOp.getOperation();
  plan.op = genericOp;

  auto iteratorTypes = genericOp.getIteratorTypesArray();
  plan.hasReduction = llvm::is_contained(iteratorTypes,
                                         utils::IteratorType::reduction);
  plan.isElementwise = isElementwiseGenericHeuristic(genericOp);

  if (plan.isElementwise) {
    plan.priority = TransformPriority::Medium;
    plan.worthFusion = true;
    appendText(plan.rationale,
               "Elementwise generic ops are classic producer/consumer fusion "
               "candidates.");
  }

  if (plan.hasReduction) {
    plan.priority = TransformPriority::High;
    plan.worthFusion = false;
    appendText(plan.rationale,
               "Reduction loops need more careful scheduling than plain "
               "elementwise loops.");
  }

  if (!plan.isElementwise && !plan.hasReduction) {
    appendText(plan.rationale,
               "Generic op is neither trivially elementwise nor clearly a "
               "reduction; leave it conservative.");
  }

  return plan;
}

static std::string formatIntList(ArrayRef<int64_t> values) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return text;
}

static void describeConvPlan(const ConvTilingPlan &plan) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "priority=" << stringifyPriority(plan.priority)
     << ", parallelTileSizes=" << formatIntList(plan.parallelTileSizes)
     << ", tileOutputSpatialLoops="
     << (plan.tileOutputSpatialLoops ? "true" : "false")
     << ", tileChannelLoop=" << (plan.tileChannelLoop ? "true" : "false")
     << ", rationale=\"" << plan.rationale << "\"";
  emitPlanRemark(plan.operation, "conv tiling plan", os.str());
  setPlanMetadata(plan.operation, "conv", plan.priority,
                  plan.parallelTileSizes);
  plan.operation->setAttr(
      "gawee.transform.tile_spatial",
      BoolAttr::get(plan.operation->getContext(), plan.tileOutputSpatialLoops));
  plan.operation->setAttr(
      "gawee.transform.tile_channel",
      BoolAttr::get(plan.operation->getContext(), plan.tileChannelLoop));
  setPlanNote(plan.operation, "gawee.transform.rationale", plan.rationale);
}

static void describeMatmulPlan(const MatmulTilingPlan &plan) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "priority=" << stringifyPriority(plan.priority)
     << ", tileSizes=" << formatIntList(plan.tileSizes)
     << ", tileReductionLoop="
     << (plan.tileReductionLoop ? "true" : "false")
     << ", rationale=\"" << plan.rationale << "\"";
  emitPlanRemark(plan.operation, "matmul tiling plan", os.str());
  setPlanMetadata(plan.operation, "matmul", plan.priority, plan.tileSizes);
  plan.operation->setAttr(
      "gawee.transform.tile_reduction",
      BoolAttr::get(plan.operation->getContext(), plan.tileReductionLoop));
  setPlanNote(plan.operation, "gawee.transform.rationale", plan.rationale);
}

static void describeGenericPlan(const GenericTransformPlan &plan) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "priority=" << stringifyPriority(plan.priority)
     << ", isElementwise=" << (plan.isElementwise ? "true" : "false")
     << ", hasReduction=" << (plan.hasReduction ? "true" : "false")
     << ", worthFusion=" << (plan.worthFusion ? "true" : "false")
     << ", rationale=\"" << plan.rationale << "\"";
  emitPlanRemark(plan.operation, "generic scheduling plan", os.str());
  setPlanMetadata(plan.operation, "generic", plan.priority, {});
  plan.operation->setAttr(
      "gawee.transform.elementwise",
      BoolAttr::get(plan.operation->getContext(), plan.isElementwise));
  plan.operation->setAttr(
      "gawee.transform.has_reduction",
      BoolAttr::get(plan.operation->getContext(), plan.hasReduction));
  plan.operation->setAttr(
      "gawee.transform.worth_fusion",
      BoolAttr::get(plan.operation->getContext(), plan.worthFusion));
  setPlanNote(plan.operation, "gawee.transform.rationale", plan.rationale);
}

/// Walk forward from `root` along single-use def chains, collecting
/// elementwise linalg.generic consumers (e.g. bias_add, relu).
///
/// Stop when a consumer has a tensor input that comes from outside the chain
/// (e.g. residual add with a skip connection). Such ops would require the
/// tile-and-fuse API to slice an unrelated producer, which can cause
/// incorrect memory access.
static SmallVector<Operation *> findElementwiseConsumerChain(Operation *root) {
  SmallVector<Operation *> chain;
  Operation *current = root;
  while (current->getNumResults() == 1 && current->getResult(0).hasOneUse()) {
    Operation *user = *current->getResult(0).getUsers().begin();
    auto genericOp = dyn_cast<linalg::GenericOp>(user);
    if (!genericOp || !isElementwiseGenericHeuristic(genericOp))
      break;

    // Check that every tensor input of this consumer either:
    //   (a) comes from `current` (the chain predecessor), or
    //   (b) is a scalar broadcast (rank-0 or rank-1 with broadcast map).
    // If any tensor input comes from outside the chain, stop here.
    bool allInputsFromChain = true;
    for (Value input : genericOp.getInputs()) {
      if (input == current->getResult(0))
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(input.getType());
      if (!tensorType)
        continue;
      // Allow scalar/1D broadcast inputs (e.g. bias tensor<64xf32>).
      if (tensorType.getRank() <= 1)
        continue;
      // This is a full-rank tensor input from outside the chain — stop.
      allInputsFromChain = false;
      break;
    }
    if (!allInputsFromChain)
      break;

    chain.push_back(user);
    current = user;
  }
  return chain;
}

static void tileAndFuseConvLikeOps(ModuleOp module) {
  // Process one conv at a time: tile-and-fuse may invalidate pointers to
  // subsequent ops (especially when consumer chains are erased/replaced).
  // Re-walk after each successful transformation.
  while (true) {
    Operation *nextConv = nullptr;
    module.walk([&](Operation *op) {
      if (isConvLikeOp(op) && !op->hasAttr("gawee.transform.tiled")) {
        nextConv = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!nextConv)
      break;

    ConvTilingPlan plan = buildConvPlan(cast<linalg::LinalgOp>(nextConv));
    describeConvPlan(plan);

    // Mark as visited so the re-walk does not pick up this conv again
    // (nor its fused copy inside the tile loop, which inherits all attrs).
    plan.operation->setAttr("gawee.transform.tiled",
                            BoolAttr::get(module.getContext(), true));

    // Look for elementwise consumer chain: conv → bias_add → relu ...
    // NOTE: Chain fusion is correct for isolated conv+chain patterns (verified
    // on tile_fuse_test.mlir, stride=2 conv, padded conv, two-conv chains).
    // However, it causes a runtime segfault on resnet18 (20 convs) that has
    // not been root-caused yet. Disabled until the issue is resolved.
    // TODO: enable and debug resnet18 segfault.
    SmallVector<Operation *> consumerChain;
    // SmallVector<Operation *> consumerChain =
    //     findElementwiseConsumerChain(plan.operation);

    // Decide tiling root: last consumer if chain exists, otherwise conv.
    Operation *tilingRoot =
        consumerChain.empty() ? plan.operation : consumerChain.back();

    auto tilingIface = dyn_cast<TilingInterface>(tilingRoot);
    if (!tilingIface) {
      tilingRoot->emitRemark() << "tiling root does not implement "
                                  "TilingInterface, skipping tiling";
      break;
    }

    // Build tile sizes for the tiling root.
    // If chain exists: the root is an elementwise generic with the same
    // output shape [N,C,H,W], so use parallelTileSizes directly.
    // If no chain: conv has 7 loops (4 parallel + 3 reduction), pad with 0s.
    SmallVector<int64_t> fullTileSizes;
    SmallVector<int64_t> interchangeVec;
    if (consumerChain.empty()) {
      unsigned numLoops =
          cast<linalg::LinalgOp>(plan.operation).getNumLoops();
      fullTileSizes.assign(numLoops, 0);
      for (unsigned i = 0;
           i < plan.parallelTileSizes.size() && i < numLoops; ++i) {
        fullTileSizes[i] = plan.parallelTileSizes[i];
      }
      interchangeVec = plan.interchangeVector;
    } else {
      // Elementwise generic root: all loops are parallel, same count as
      // parallelTileSizes (4 for NCHW).
      fullTileSizes = plan.parallelTileSizes;
      // Use only the parallel-dim interchange: {0,2,3,1}.
      if (plan.interchangeVector.size() >= 4)
        interchangeVec = {plan.interchangeVector[0],
                          plan.interchangeVector[1],
                          plan.interchangeVector[2],
                          plan.interchangeVector[3]};
    }

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(module.getContext(), fullTileSizes);

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    if (!interchangeVec.empty())
      tilingOptions.setInterchange(interchangeVec);

    // Tile-and-fuse: tile the root consumer and fuse producers backward.
    // When the root is the last elementwise consumer, the API walks backward
    // through bias_add → conv → fill, fusing each into the tile loop.
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);
    tileAndFuseOptions.fusionControlFn =
        [](tensor::ExtractSliceOp, OpResult originalProducer,
           bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *producer = originalProducer.getOwner();
      // fill: always safe to fuse (destination init).
      if (isa<linalg::FillOp>(producer))
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            /*yieldProducerReplacement=*/false};
      // conv: main compute producer in the chain.
      if (isa<linalg::Conv2DNchwFchwOp>(producer))
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            /*yieldProducerReplacement=*/false};
      // elementwise generic: bias_add and similar ops in the chain.
      if (auto genericOp = dyn_cast<linalg::GenericOp>(producer)) {
        if (isElementwiseGenericHeuristic(genericOp))
          return scf::SCFTileAndFuseOptions::ControlFnResult{
              /*yieldProducerReplacement=*/false};
      }
      return std::nullopt;
    };

    IRRewriter rewriter(module.getContext());
    rewriter.setInsertionPoint(tilingRoot);
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingIface,
                                                   tileAndFuseOptions);
    if (failed(result)) {
      tilingRoot->emitWarning() << "conv tile-and-fuse failed";
      break;
    }

    // Replace uses of the original tiling root's results with loop results.
    for (auto &[origVal, replacement] : result->replacements) {
      rewriter.replaceAllUsesWith(origVal, replacement);
    }

    // Clean up: erase dead ops (tiling root, conv, and all fused producers).
    if (tilingRoot->use_empty())
      rewriter.eraseOp(tilingRoot);
    if (tilingRoot != plan.operation && plan.operation->use_empty())
      rewriter.eraseOp(plan.operation);
    for (Operation *op : consumerChain) {
      if (op != tilingRoot && op->use_empty())
        rewriter.eraseOp(op);
    }
    for (Operation *fusedProducer : result->fusedProducers) {
      if (fusedProducer->use_empty())
        rewriter.eraseOp(fusedProducer);
    }
  }
}

static void tileMatmulLikeOps(ModuleOp module) {
  SmallVector<MatmulTilingPlan, 2> plans;
  module.walk([&](Operation *op) {
    if (!isMatmulLikeOp(op)) {
      return;
    }
    plans.push_back(buildMatmulPlan(cast<linalg::LinalgOp>(op)));
  });

  for (const MatmulTilingPlan &plan : plans) {
    describeMatmulPlan(plan);

    auto tilingIface = dyn_cast<TilingInterface>(plan.operation);
    if (!tilingIface) {
      plan.operation->emitRemark() << "matmul op does not implement "
                                      "TilingInterface, skipping tiling";
      continue;
    }

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(module.getContext(), plan.tileSizes);

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);
    if (!plan.interchangeVector.empty())
      options.setInterchange(plan.interchangeVector);

    IRRewriter rewriter(module.getContext());
    rewriter.setInsertionPoint(plan.operation);
    FailureOr<scf::SCFTilingResult> result =
        scf::tileUsingSCF(rewriter, tilingIface, options);
    if (failed(result)) {
      plan.operation->emitWarning() << "matmul tiling failed";
      continue;
    }

    rewriter.replaceOp(plan.operation, result->replacements);
  }
}

static void scheduleOrFuseGenericOps(ModuleOp module) {
  SmallVector<GenericTransformPlan> plans;
  module.walk([&](Operation *op) {
    if (!isGenericOp(op)) {
      return;
    }
    plans.push_back(buildGenericPlan(cast<linalg::GenericOp>(op)));
  });

  for (const GenericTransformPlan &plan : plans) {
    describeGenericPlan(plan);
    // Future work:
    //   - fuse elementwise producer/consumer chains
    //   - isolate reductions that need specialized scheduling
    //   - keep "plain generic cleanup" separate from heavyweight transforms
  }
}

static void summarizeModule(ModuleOp module) {
  int convCount = 0;
  int matmulCount = 0;
  int genericCount = 0;

  module.walk([&](Operation *op) {
    if (isConvLikeOp(op)) {
      ++convCount;
      return;
    }
    if (isMatmulLikeOp(op)) {
      ++matmulCount;
      return;
    }
    if (isGenericOp(op)) {
      ++genericCount;
    }
  });

  module.emitRemark() << "linalg transform summary: conv=" << convCount
                      << ", matmul=" << matmulCount
                      << ", generic=" << genericCount;
  Builder builder(module.getContext());
  module->setAttr("gawee.transform.conv_count",
                  builder.getI64IntegerAttr(convCount));
  module->setAttr("gawee.transform.matmul_count",
                  builder.getI64IntegerAttr(matmulCount));
  module->setAttr("gawee.transform.generic_count",
                  builder.getI64IntegerAttr(genericCount));
}

struct LinalgTransformPass
    : public PassWrapper<LinalgTransformPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "gawee-linalg-transform"; }

  StringRef getDescription() const override {
    return "Post-lowering Linalg tiling and scheduling planning pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The pass owns transform order.
    // The helpers own per-op-family transform details.
    //
    // Current intended order:
    //   1. conv-like tiling
    //   2. matmul-like tiling
    //   3. generic-op scheduling / fusion
    //
    // This keeps the "pipeline-level decision" separate from the
    // "op-specific implementation" decision.
    //
    // Extra note for readers:
    //   Nothing below changes IR yet.
    //   The pass currently does "inspection -> plan building -> remarks".
    //   That is intentional:
    //     1. understand each op family first
    //     2. make transform decisions explicit
    //     3. only then attach real rewrite code

    summarizeModule(module);

    tileAndFuseConvLikeOps(module);
    tileMatmulLikeOps(module);
    scheduleOrFuseGenericOps(module);
  }
};

} // namespace

namespace mlir::gawee {
std::unique_ptr<Pass> createLinalgTransformPass() {
  return std::make_unique<LinalgTransformPass>();
}
} // namespace mlir::gawee
