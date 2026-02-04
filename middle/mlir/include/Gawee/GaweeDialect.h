//===----------------------------------------------------------------------===//
// Gawee Dialect C++ Header
//===----------------------------------------------------------------------===//

#ifndef GAWEE_DIALECT_H
#define GAWEE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

// Dialect declaration
#include "generated/GaweeDialect.h.inc"

// Op declarations
#define GET_OP_CLASSES
#include "generated/GaweeOps.h.inc"

#endif // GAWEE_DIALECT_H
