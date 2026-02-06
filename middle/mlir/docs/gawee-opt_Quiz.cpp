//===----------------------------------------------------------------------===//
// gawee-opt Quiz
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the opt tool.
// This mirrors the actual structure of tools/gawee-opt.cpp
//
// After completing, compare with the real implementation.
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

// Dialect includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Q1: Bufferization interface includes (CRITICAL for one-shot-bufferize!)
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/???.h"      // Q1a
#include "mlir/Dialect/Arith/Transforms/???.h"       // Q1b
#include "mlir/Dialect/Tensor/Transforms/???.h"      // Q1c

// Conversion passes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

// Pass declaration
namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass();
}

//===----------------------------------------------------------------------===//
// Q2: Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {

  //-----------------------------------------------------------------------
  // Q2a: Register basic Gawee → Linalg pipeline
  //-----------------------------------------------------------------------
  PassPipelineRegistration<>(
      "convert-gawee-to-linalg",
      "Lower Gawee dialect to Linalg dialect",
      [](OpPassManager &pm) {
        pm.addPass(gawee::???());
      });

  //-----------------------------------------------------------------------
  // Q2b: Register full pipeline: Gawee → LLVM dialect
  //-----------------------------------------------------------------------
  PassPipelineRegistration<>(
      "gawee-to-llvm",
      "Full pipeline: Gawee -> Linalg -> SCF -> LLVM dialect",
      [](OpPassManager &pm) {
        // Step 1: Gawee -> Linalg (on tensors)
        pm.addPass(gawee::createGaweeToLinalgPass());

        // Step 2: Convert tensor.empty to bufferization.alloc_tensor
        pm.addPass(bufferization::???());

        // Step 3: Bufferize (tensor -> memref)
        bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.bufferizeFunctionBoundaries = ???;
        pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

        // Step 4: Linalg -> SCF loops
        pm.addPass(???());

        // Step 5: SCF -> ControlFlow (cf dialect)
        pm.addPass(???());

        // Step 6: Convert to LLVM dialect
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(???());  // CF to LLVM
        pm.addPass(???());  // MemRef to LLVM
        pm.addPass(???());  // Func to LLVM

        // Step 7: Clean up unrealized casts
        pm.addPass(???());
      });

  //-----------------------------------------------------------------------
  // Q3: Register all dialects
  //-----------------------------------------------------------------------
  DialectRegistry registry;

  // Our dialect
  registry.insert<gawee::GaweeDialect>();

  // Target dialects
  registry.insert<linalg::LinalgDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<???>();  // Q3a: LLVM dialect

  //-----------------------------------------------------------------------
  // Q4: Register bufferization interfaces (CRITICAL!)
  //-----------------------------------------------------------------------
  // WHY? One-shot-bufferize needs to know how each dialect's ops
  // behave with respect to memory (can they alias? are they writable?)
  //
  // Without these, you get: "error: op was not bufferized"

  arith::???ExternalModels(registry);     // Q4a
  linalg::???ExternalModels(registry);    // Q4b
  tensor::???ExternalModels(registry);    // Q4c
  bufferization::func_ext::???ExternalModels(registry);  // Q4d

  //-----------------------------------------------------------------------
  // Q5: Run the opt main
  //-----------------------------------------------------------------------
  return mlir::asMainReturnCode(
      mlir::???(argc, argv, "Gawee MLIR Optimizer\n", registry));
}

//===----------------------------------------------------------------------===//
// Q6: Conceptual Questions
//===----------------------------------------------------------------------===//
//
// Q6a: Why do we need to register bufferization interfaces?
//      A) For pretty printing
//      B) One-shot-bufferize needs to know how ops behave with memory
//      C) For parsing MLIR
//      D) For optimizations
//
// Q6b: What does bufferizeFunctionBoundaries = true do?
//      A) Ignores function boundaries
//      B) Also bufferizes function arguments and return values
//      C) Only bufferizes inside functions
//      D) Disables bufferization
//
// Q6c: Why run empty-tensor-to-alloc-tensor before one-shot-bufferize?
//      A) For performance
//      B) tensor.empty needs to become bufferization.alloc_tensor first
//      C) It's optional
//      D) For pretty printing
//
// Q6d: What is the purpose of reconcile-unrealized-casts?
//      A) Optimize casts
//      B) Remove temporary type conversion markers between passes
//      C) Add type casts
//      D) Verify types
//

//===----------------------------------------------------------------------===//
// Answer Key
//===----------------------------------------------------------------------===//
/*
Q1a: BufferizableOpInterfaceImpl
Q1b: BufferizableOpInterfaceImpl
Q1c: BufferizableOpInterfaceImpl

Q2a: createGaweeToLinalgPass
Q2b: createEmptyTensorToAllocTensorPass, true, createConvertLinalgToLoopsPass,
     createSCFToControlFlowPass, createConvertControlFlowToLLVMPass,
     createFinalizeMemRefToLLVMConversionPass, createConvertFuncToLLVMPass,
     createReconcileUnrealizedCastsPass

Q3a: LLVM::LLVMDialect

Q4a: registerBufferizableOpInterface
Q4b: registerBufferizableOpInterface
Q4c: registerBufferizableOpInterface
Q4d: registerBufferizableOpInterface

Q5: MlirOptMain

Q6a: B - Bufferization needs to understand op memory semantics
Q6b: B - Converts function tensor args/returns to memref
Q6c: B - tensor.empty must become alloc_tensor for bufferization
Q6d: B - Cleans up unrealized_conversion_cast markers
*/
