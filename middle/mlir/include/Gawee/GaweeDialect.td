//===----------------------------------------------------------------------===//
// Gawee Dialect Definition
//===----------------------------------------------------------------------===//
//
// LEARNING: TableGen (.td files)
//
// TableGen is MLIR's domain-specific language for defining dialects/ops.
// It generates C++ code automatically, reducing boilerplate.
//
// Flow:
//   GaweeDialect.td --> mlir-tblgen --> GaweeDialect.h.inc, GaweeDialect.cpp.inc
//
// Why TableGen?
//   - Declarative: describe WHAT, not HOW
//   - Consistent: all dialects follow same patterns
//   - Extensible: easy to add verification, canonicalization, etc.
//
//===----------------------------------------------------------------------===//

#ifndef GAWEE_DIALECT_TD
#define GAWEE_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//
//
// LEARNING: A dialect is a namespace for operations, types, and attributes.
//
// Think of it like a "language" within MLIR. Examples:
//   - `arith` dialect: arithmetic ops (add, mul, ...)
//   - `linalg` dialect: linear algebra ops (matmul, conv, ...)
//   - `gawee` dialect: YOUR neural network ops
//

def Gawee_Dialect : Dialect {
  // Dialect name - appears in IR as: gawee.conv, gawee.relu, etc.
  let name = "gawee";

  // Short description
  let summary = "Gawee dialect for neural network graphs";

  // Disable bytecode support (avoids missing header issues)
  let usePropertiesForAttributes = 0;

  // Longer description
  let description = [{
    The Gawee dialect represents high-level neural network operations.
    It is designed to be lowered to Linalg for execution.

    Example IR:
    ```mlir
    %0 = gawee.conv %input, %weight : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32> -> tensor<1x64x112x112xf32>
    %1 = gawee.relu %0 : tensor<1x64x112x112xf32>
    ```
  }];

  // C++ namespace for generated code
  let cppNamespace = "::mlir::gawee";

  // TODO: Uncomment when you implement custom types
  // let useDefaultTypePrinterParser = 1;

  // TODO: Uncomment when you implement custom attributes
  // let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base class for Gawee operations
//===----------------------------------------------------------------------===//
//
// LEARNING: Op inheritance
//
// All Gawee ops inherit from this base class.
// This ensures they all belong to the Gawee dialect.
//

class Gawee_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gawee_Dialect, mnemonic, traits>;

#endif // GAWEE_DIALECT_TD
