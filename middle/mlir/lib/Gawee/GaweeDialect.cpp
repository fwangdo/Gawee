//===----------------------------------------------------------------------===//
// Gawee Dialect Implementation
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"

using namespace mlir;
using namespace mlir::gawee;

// Dialect implementation
#include "generated/GaweeDialect.cpp.inc"

// Op implementations
#define GET_OP_CLASSES
#include "generated/GaweeOps.cpp.inc"

void GaweeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "generated/GaweeOps.cpp.inc"
  >();
}
