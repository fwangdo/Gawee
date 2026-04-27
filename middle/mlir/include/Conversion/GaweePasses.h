#ifndef GAWEE_CONVERSION_PASSES_H
#define GAWEE_CONVERSION_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::gawee {

std::unique_ptr<Pass> createGaweeToLinalgPass();
std::unique_ptr<Pass> createLinalgTransformScaffoldPass();
std::unique_ptr<Pass> createLinalgFusionScaffoldPass();
std::unique_ptr<Pass> createLinalgSchedulingScaffoldPass();
std::unique_ptr<Pass> createLinalgVectorizationScaffoldPass();
std::unique_ptr<Pass> createLinalgVerificationScaffoldPass();
std::unique_ptr<Pass> createGaweeBufferizePrepScaffoldPass();

} // namespace mlir::gawee

#endif
