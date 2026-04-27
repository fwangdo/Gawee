//===----------------------------------------------------------------------===//
// gawee-opt: Gawee MLIR Optimizer Tool
//===----------------------------------------------------------------------===//
//
// This tool runs MLIR passes on Gawee dialect IR.
//
// Usage:
//   gawee-opt [options] <input.mlir>
//
// Examples:
//   gawee-opt input.mlir                           # Parse and print
//   gawee-opt --convert-gawee-to-linalg input.mlir # Run conversion pass
//   gawee-opt --gawee-to-llvm input.mlir           # Full pipeline to LLVM
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"
#include "Conversion/GaweePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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

//===----------------------------------------------------------------------===//
// Pass Declaration
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Register our conversion pass
  PassPipelineRegistration<>(
      "convert-gawee-to-linalg",
      "Lower Gawee dialect to Linalg dialect",
      [](OpPassManager &pm) {
        pm.addPass(gawee::createGaweeToLinalgPass());
      });

  // Register full pipeline: Gawee -> Linalg -> Loops
  PassPipelineRegistration<>(
      "gawee-to-loops",
      "Full pipeline: Gawee -> Linalg -> SCF loops",
      [](OpPassManager &pm) {
        // Step 1: Gawee -> Linalg (on tensors)
        pm.addPass(gawee::createGaweeToLinalgPass());

        // Step 2: Linalg-level transform slot (tiling / scheduling / fusion)
        pm.addPass(gawee::createLinalgTransformPass());
        pm.addPass(gawee::createLinalgFusionPass());
        pm.addPass(gawee::createLinalgSchedulingPass());
        pm.addPass(gawee::createLinalgVectorizationPass());
        pm.addPass(gawee::createLinalgVerificationPass());

        // Step 3: Bufferization preparation slot
        pm.addPass(gawee::createGaweeBufferizePrepPass());

        // Step 4: Bufferize (tensor -> memref)
        bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

        // Step 5: Linalg -> SCF loops
        pm.addPass(createConvertLinalgToLoopsPass());
      });

  // Register full pipeline: Gawee -> LLVM dialect
  PassPipelineRegistration<>(
      "gawee-to-llvm",
      "Full pipeline: Gawee -> Linalg -> SCF -> LLVM dialect",
      [](OpPassManager &pm) {
        // Step 1: Gawee -> Linalg (on tensors)
        pm.addPass(gawee::createGaweeToLinalgPass());

        // Step 2: Linalg-level transform slot (tiling / scheduling / fusion)
        pm.addPass(gawee::createLinalgTransformPass());
        pm.addPass(gawee::createLinalgFusionPass());
        pm.addPass(gawee::createLinalgSchedulingPass());
        pm.addPass(gawee::createLinalgVectorizationPass());
        pm.addPass(gawee::createLinalgVerificationPass());

        // Step 3: Convert tensor.empty to bufferization.alloc_tensor
        // (required for proper bufferization)
        pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

        // Step 4: Bufferization preparation slot
        pm.addPass(gawee::createGaweeBufferizePrepPass());

        // Step 5: Bufferize (tensor -> memref)
        bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

        // Step 6: Linalg -> SCF loops
        pm.addPass(createConvertLinalgToLoopsPass());

        // Step 7: SCF -> ControlFlow (cf dialect)
        pm.addPass(createSCFToControlFlowPass());

        // Step 8: Normalize memref metadata ops before LLVM conversion.
        pm.addPass(memref::createExpandStridedMetadataPass());
        pm.addPass(createLowerAffinePass());

        // Step 9: Convert to LLVM dialect
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(LLVM::createLLVMRequestCWrappersPass());
        pm.addPass(createConvertFuncToLLVMPass());

        // Step 10: Clean up unrealized casts
        pm.addPass(createReconcileUnrealizedCastsPass());
      });

  // Register SCF/memref -> LLVM pipeline (for testing without bufferization)
  PassPipelineRegistration<>(
      "scf-to-llvm",
      "Pipeline: SCF/MemRef/Arith -> LLVM dialect",
      [](OpPassManager &pm) {
        // Step 1: SCF -> ControlFlow
        pm.addPass(createSCFToControlFlowPass());

        // Step 2: Convert to LLVM dialect
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(LLVM::createLLVMRequestCWrappersPass());
        pm.addPass(createConvertFuncToLLVMPass());

        // Step 3: Clean up unrealized casts
        pm.addPass(createReconcileUnrealizedCastsPass());
      });

  // Register all dialects we need
  DialectRegistry registry;

  // Our dialect
  registry.insert<gawee::GaweeDialect>();

  // Target dialects (output of conversion)
  registry.insert<linalg::LinalgDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<LLVM::LLVMDialect>();

  // Register bufferization interfaces for each dialect
  // These tell one-shot-bufferize how to bufferize ops from each dialect
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  // Use MLIR's standard opt main function
  // This handles:
  //   - Command line parsing
  //   - Input file reading
  //   - Pass pipeline execution
  //   - Output printing
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Gawee MLIR Optimizer\n", registry));
}
