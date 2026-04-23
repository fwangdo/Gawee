//===----------------------------------------------------------------------===//
// ONNXMLIREmitter - Translates ONNX Graphs to Gawee MLIR
//===----------------------------------------------------------------------===//
//
// This class is the ONNX-front parallel to MLIREmitter.
//
// Intended pipeline:
//   normalized.onnx -> ONNXMLIREmitter -> gawee.mlir
//
// Current status:
//   - scaffold only
//   - creates an empty module shell by default
//   - contains opt-in ONNX protobuf parsing path for Conv
//   - real multi-op coverage is still TODO
//
//===----------------------------------------------------------------------===//

#ifndef GAWEE_EMIT_ONNXMLIREMITTER_H
#define GAWEE_EMIT_ONNXMLIREMITTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir::gawee {

class ONNXMLIREmitter {
public:
  explicit ONNXMLIREmitter(MLIRContext *context);

  /// Emit MLIR module from an ONNX file path.
  /// Returns nullptr on failure.
  ///
  /// Default behavior:
  ///   - without GAWEE_ENABLE_ONNX_PROTO: returns nullptr with a scaffold error
  ///   - with    GAWEE_ENABLE_ONNX_PROTO: parses ONNX protobuf and emits Conv
  OwningOpRef<ModuleOp> emitFromFile(llvm::StringRef onnxPath);

  /// Get the last error message (if emitFromFile() returned nullptr).
  llvm::StringRef getError() const { return errorMsg; }

private:
  MLIRContext *ctx;
  std::unique_ptr<OpBuilder> builder;
  std::string errorMsg;

  void setError(const llvm::Twine &msg);
  OwningOpRef<ModuleOp> createEmptyModule(llvm::StringRef onnxPath);
};

} // namespace mlir::gawee

#endif // GAWEE_EMIT_ONNXMLIREMITTER_H
