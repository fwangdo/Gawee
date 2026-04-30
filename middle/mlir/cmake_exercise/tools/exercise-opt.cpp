#include "Gawee/GaweeDialect.h"
#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<gawee::GaweeDialect, arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, tensor::TensorDialect>();

  // # TODO
  return failed(MlirOptMain(argc, argv, "Gawee CMake exercise tool\n",
                            registry));
}
