#include "Gawee/GaweeDialect.h"
#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main() {
  DialectRegistry registry;
  registry.insert<gawee::GaweeDialect, arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, tensor::TensorDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral kModule = R"mlir(
module {
  func.func @toy(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %0 = "gawee.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(kModule, &context);
  if (!module) {
    llvm::errs() << "failed to parse exercise module\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(gawee::createGaweeToLinalgPass());
  pm.addPass(gawee::createLinalgTransformPass());
  pm.addPass(gawee::createLinalgFusionPass());
  pm.addPass(gawee::createLinalgSchedulingPass());
  pm.addPass(gawee::createLinalgVectorizationPass());
  pm.addPass(gawee::createLinalgVerificationPass());

  if (failed(pm.run(module.get()))) {
    llvm::errs() << "exercise pass pipeline failed\n";
    return 1;
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
