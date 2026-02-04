//===----------------------------------------------------------------------===//
// gawee-opt Quiz
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the opt tool.
// After completing, compare with tools/gawee-opt.cpp
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Declaration (defined elsewhere)
//===----------------------------------------------------------------------===//

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass();
}

//===----------------------------------------------------------------------===//
// Quiz: Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Q1: Register our conversion pass
  // This makes --convert-gawee-to-linalg available on command line
  // Hint: Use PassPipelineRegistration
  PassPipelineRegistration<>(
      "convert-gawee-to-linalg",           // CLI flag
      "Lower Gawee dialect to Linalg",     // Description
      [](OpPassManager &pm) {
        pm.addPass(gawee::createGaweeToLinalgPass());
      });

  // Q2: Create dialect registry
  // This tells MLIR which dialects we want to use
  DialectRegistry registry;

  // Q3: Register our dialect (the input dialect)
  registry.insert<gawee::GaweeDialect>();

  // Q4: Register target dialects (output of conversion)
  registry.insert<linalg::LinalgDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<func::FuncDialect>();

  // Q5: Use MLIR's standard opt main function
  // This handles CLI parsing, file I/O, pass execution
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Gawee MLIR Optimizer\n", registry));
}

//===----------------------------------------------------------------------===//
// Bonus Quiz: What goes in the pass's getDependentDialects?
//===----------------------------------------------------------------------===//
//
// Q6: Why do we need getDependentDialects() in our pass?
//
// A) To register dialects for parsing input
// B) To declare dialects that the pass will CREATE ops from
// C) To register dialects for printing output
// D) It's optional and not needed
//
// Q7: If you forget getDependentDialects(), what error do you get?
//
// A) Compilation error
// B) "dialect not registered"
// C) "op isn't known in this MLIRContext"
// D) Linker error
//

//===----------------------------------------------------------------------===//
// Answer Key
//===----------------------------------------------------------------------===//
/*
Q1: PassPipelineRegistration, OpPassManager, addPass
Q2: DialectRegistry
Q3: insert
Q4: insert, LinalgDialect, ArithDialect, TensorDialect, FuncDialect
Q5: MlirOptMain

Bonus:
Q6: B - The pass creates tensor.empty, linalg.add, etc. Those dialects must be loaded.
Q7: C - You get "Building op X but it isn't known in this MLIRContext"
*/
