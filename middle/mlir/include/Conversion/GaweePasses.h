#ifndef GAWEE_CONVERSION_PASSES_H
#define GAWEE_CONVERSION_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::gawee {

std::unique_ptr<Pass> createGaweeToLinalgPass();
std::unique_ptr<Pass> createLinalgTransformPass();
std::unique_ptr<Pass> createLinalgFusionPass();
std::unique_ptr<Pass> createLinalgSchedulingPass();
std::unique_ptr<Pass> createLinalgVectorizationPass();
std::unique_ptr<Pass> createLinalgVerificationPass();
std::unique_ptr<Pass> createGaweeBufferizePrepPass();
std::unique_ptr<Pass> createDecomposeAggregatedLinalgOpsPass();

} // namespace mlir::gawee

#endif
