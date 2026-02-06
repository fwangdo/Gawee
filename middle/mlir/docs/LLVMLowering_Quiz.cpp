//===----------------------------------------------------------------------===//
// LLVM Lowering Quiz (Phase 7)
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the LLVM lowering pipeline.
// After completing, compare with tools/gawee-opt.cpp
//
//===----------------------------------------------------------------------===//

// Q1: What headers do you need for LLVM lowering?
// Fill in the missing includes:

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Conversion passes
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/???.h"           // Q1a
#include "mlir/Conversion/ControlFlowToLLVM/???.h"    // Q1b
#include "mlir/Conversion/MemRefToLLVM/???.h"         // Q1c
#include "mlir/Conversion/FuncToLLVM/???.h"           // Q1d
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Q2: Register the LLVM lowering pipeline
//===----------------------------------------------------------------------===//

void registerPipeline() {
  PassPipelineRegistration<>(
      "scf-to-llvm",
      "Lower SCF/MemRef to LLVM dialect",
      [](OpPassManager &pm) {
        // Q2a: First, convert SCF (for loops) to CF (branches)
        pm.addPass(???());

        // Q2b: Convert arithmetic operations to LLVM
        pm.addPass(???());

        // Q2c: Convert control flow (branches) to LLVM
        pm.addPass(???());

        // Q2d: Convert memref operations to LLVM
        pm.addPass(???());

        // Q2e: Convert function definitions to LLVM
        pm.addPass(???());

        // Q2f: Clean up temporary cast markers
        pm.addPass(???());
      });
}

//===----------------------------------------------------------------------===//
// Q3: Register required dialects
//===----------------------------------------------------------------------===//

void registerDialects(DialectRegistry &registry) {
  // Q3a: Register the LLVM dialect (target dialect)
  registry.insert<???>();

  // Q3b: Register the ControlFlow dialect (intermediate)
  registry.insert<???>();
}

//===----------------------------------------------------------------------===//
// Q4: Conceptual Questions
//===----------------------------------------------------------------------===//
//
// Q4a: What is the correct order of these passes?
//      A) func-to-llvm, scf-to-cf, arith-to-llvm
//      B) scf-to-cf, arith-to-llvm, func-to-llvm
//      C) arith-to-llvm, func-to-llvm, scf-to-cf
//
// Q4b: Why do we need `reconcile-unrealized-casts` at the end?
//      A) To optimize the code
//      B) To remove temporary type conversion markers between passes
//      C) To verify the IR is correct
//
// Q4c: How is `memref<4xf32>` represented in LLVM?
//      A) As a single pointer
//      B) As an array type
//      C) As a struct containing pointer, offset, sizes, and strides
//
// Q4d: What does `scf.for` become after lowering to CF dialect?
//      A) A `llvm.for` operation
//      B) A loop with header, body, and exit basic blocks using branches
//      C) A recursive function call
//
// Q4e: What tool converts MLIR LLVM dialect to actual LLVM IR?
//      A) gawee-opt
//      B) mlir-translate --mlir-to-llvmir
//      C) llc
//
// Q4f: Why must we call `linalg::registerBufferizableOpInterfaceExternalModels()`?
//      A) To enable printing of linalg ops
//      B) To tell one-shot-bufferize how to convert linalg ops from tensor to memref
//      C) To optimize linalg ops
//      D) It's optional, just for debugging
//

//===----------------------------------------------------------------------===//
// Q5: MLIR Code Transformation
//===----------------------------------------------------------------------===//
//
// Given this SCF code:
//
//   scf.for %i = %c0 to %c4 step %c1 {
//     %val = memref.load %A[%i] : memref<4xf32>
//     memref.store %val, %B[%i] : memref<4xf32>
//   }
//
// Q5a: After scf-to-cf, how many basic blocks will there be?
//      A) 1
//      B) 2
//      C) 3 (header, body, exit)
//      D) 4
//
// Q5b: What CF operation is used for the loop condition check?
//      A) cf.br
//      B) cf.cond_br
//      C) cf.switch
//
// Q5c: After memref-to-llvm, what LLVM operation replaces memref.load?
//      A) llvm.load
//      B) llvm.memref.load
//      C) llvm.read
//

//===----------------------------------------------------------------------===//
// Answer Key
//===----------------------------------------------------------------------===//
/*
Q1a: ArithToLLVM
Q1b: ControlFlowToLLVM
Q1c: MemRefToLLVM
Q1d: ConvertFuncToLLVMPass

Q2a: createSCFToControlFlowPass
Q2b: createArithToLLVMConversionPass
Q2c: createConvertControlFlowToLLVMPass
Q2d: createFinalizeMemRefToLLVMConversionPass
Q2e: createConvertFuncToLLVMPass
Q2f: createReconcileUnrealizedCastsPass

Q3a: LLVM::LLVMDialect
Q3b: cf::ControlFlowDialect

Q4a: B - scf-to-cf must come before CF ops can be lowered to LLVM
Q4b: B - Passes use unrealized_conversion_cast as temporary markers
Q4c: C - MemRef descriptor contains all metadata for strided access
Q4d: B - SCF loops become basic blocks with conditional branches
Q4e: B - mlir-translate converts between MLIR and LLVM IR formats
Q4f: B - Bufferization needs external models to know how each dialect's ops behave

Q5a: C - Header (condition), body (loop work), exit (after loop)
Q5b: B - cf.cond_br checks condition and branches to body or exit
Q5c: A - memref.load becomes llvm.load with getelementptr
*/
