//===----------------------------------------------------------------------===//
// Gawee Dialect Implementation
//===----------------------------------------------------------------------===//
//
// LEARNING: Dialect registration
//
// Every dialect must be registered with MLIR's context before use.
// This file:
//   1. Includes generated implementations
//   2. Implements the initialize() method
//   3. Adds any custom verifiers/canonicalizers
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

using namespace mlir;
using namespace mlir::gawee;

//===----------------------------------------------------------------------===//
// Include TableGen-generated implementations
//===----------------------------------------------------------------------===//

// Dialect implementation (constructor, etc.)
#include "Gawee/GaweeDialect.cpp.inc"

// Operation implementations
#define GET_OP_CLASSES
#include "Gawee/GaweeOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//
//
// LEARNING: This is called when the dialect is loaded.
// It registers all operations with the MLIR context.
//

void GaweeDialect::initialize() {
  // Register all ops defined in GaweeOps.td
  addOperations<
#define GET_OP_LIST
#include "Gawee/GaweeOps.cpp.inc"
  >();

  // TODO: Register custom types if you define any
  // addTypes<GaweeTensorType, ...>();

  // TODO: Register custom attributes if you define any
  // addAttributes<...>();
}

//===----------------------------------------------------------------------===//
// Operation verifiers (optional)
//===----------------------------------------------------------------------===//
//
// LEARNING: Verifiers check that ops are well-formed.
//
// Example: Conv should have 4D input tensor
//
// LogicalResult ConvOp::verify() {
//   auto inputType = getInput().getType().cast<RankedTensorType>();
//   if (inputType.getRank() != 4)
//     return emitOpError("input must be 4D tensor");
//   return success();
// }
//

// TODO: Implement verifiers for your ops

//===----------------------------------------------------------------------===//
// Canonicalization patterns (optional)
//===----------------------------------------------------------------------===//
//
// LEARNING: Canonicalizers simplify ops at compile time.
//
// Example: relu(relu(x)) -> relu(x)
//
// void ReluOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
//                                          MLIRContext *context) {
//   patterns.add<FuseDoubleRelu>(context);
// }
//

// TODO: Implement canonicalization patterns
