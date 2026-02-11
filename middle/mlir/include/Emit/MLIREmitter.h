//===----------------------------------------------------------------------===//
// MLIREmitter - Translates JSON Graph to Gawee MLIR
//===----------------------------------------------------------------------===//
//
// This class reads a parsed JSON graph representation and emits
// Gawee dialect operations.
//
// Pipeline:
//   JSON file -> llvm::json::Value -> MLIREmitter -> gawee.mlir
//
//===----------------------------------------------------------------------===//

#ifndef GAWEE_EMIT_MLIREMITTER_H
#define GAWEE_EMIT_MLIREMITTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <unordered_map>

namespace mlir::gawee {

/// MLIREmitter translates a JSON graph to Gawee MLIR operations.
///
/// Usage:
///   MLIREmitter emitter(context);
///   auto module = emitter.emit(jsonGraph);
///   if (!module) { /* handle error */ }
///
class MLIREmitter {
public:
  // explicit does not allow implicit conversion like A a -> A(a) 
  // constructor. 
  explicit MLIREmitter(MLIRContext *context);

  /// Emit MLIR module from JSON graph.
  /// Returns nullptr on failure.
  OwningOpRef<ModuleOp> emit(const llvm::json::Object &graph);

  /// Get the last error message (if emit() returned nullptr).
  llvm::StringRef getError() const { return errorMsg; }

private:
  MLIRContext *ctx;
  std::unique_ptr<OpBuilder> builder;
  std::string errorMsg;

  /// Maps value name (e.g., "conv1") -> MLIR Value
  std::unordered_map<std::string, Value> valueMap;

  /// Weight tensors to add as function arguments (name, type)
  std::vector<std::pair<std::string, RankedTensorType>> weightArgs;

  /// Index for tracking which weight argument to use next
  size_t weightArgIndex = 0;

  /// Parse shape array from JSON and create RankedTensorType.
  RankedTensorType parseShape(const llvm::json::Array *shape);

  /// Emit a single node. Returns false on failure.
  bool emitNode(const llvm::json::Object &node,
                const llvm::json::Object &values);

  /// Emit specific operations
  bool emitConv(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitRelu(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitAdd(const llvm::json::Object &node, const llvm::json::Object &values);

  // Add. 
  bool emitMaxPool(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitAdAvgPool(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitFlatten(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitLinear(const llvm::json::Object &node, const llvm::json::Object &values);

  /// Helper: Look up input Value by name
  Value lookupValue(llvm::StringRef name);
  

  /// Helper: Set error message
  void setError(const llvm::Twine &msg);
};

} // namespace mlir::gawee

#endif // GAWEE_EMIT_MLIREMITTER_H
