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
//   - contains a Conv reference path and TODO extension points
//
//===----------------------------------------------------------------------===//

#ifndef GAWEE_EMIT_ONNXMLIREMITTER_H
#define GAWEE_EMIT_ONNXMLIREMITTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include <onnx/onnx_pb.h>
#include <memory>
#include <string>

namespace mlir::gawee {

class ONNXMLIREmitter {
public:
  explicit ONNXMLIREmitter(MLIRContext *context);

  /// Emit MLIR module from an ONNX file path.
  /// Returns nullptr on failure.
  OwningOpRef<ModuleOp> emitFromFile(llvm::StringRef onnxPath);

  /// Get the last error message (if emitFromFile() returned nullptr).
  llvm::StringRef getError() const { return errorMsg; }

private:
  MLIRContext *ctx;
  std::unique_ptr<OpBuilder> builder;
  std::string errorMsg;
  llvm::StringMap<llvm::SmallVector<int64_t>> i64TensorLiterals;

  // ONNX node helpers. Each helper should:
  //   1. read operands from valueMap
  //   2. read result types from tensorTypes
  //   3. create the matching gawee op
  //   4. write results back into valueMap
  //
  // Helper names follow ONNX op families where possible.
  bool emitRelu(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitAdd(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitAveragePool(const onnx::NodeProto &node,
                       llvm::StringMap<Value> &valueMap,
                       llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitCast(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitConcat(const onnx::NodeProto &node,
                  llvm::StringMap<Value> &valueMap,
                  llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitConv(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSub(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitDiv(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitEqual(const onnx::NodeProto &node,
                 llvm::StringMap<Value> &valueMap,
                 llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitErf(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitExpand(const onnx::NodeProto &node,
                  llvm::StringMap<Value> &valueMap,
                  llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitGelu(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitGlobalAveragePool(const onnx::NodeProto &node,
                             llvm::StringMap<Value> &valueMap,
                             llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitHardSigmoid(const onnx::NodeProto &node,
                       llvm::StringMap<Value> &valueMap,
                       llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitHardSwish(const onnx::NodeProto &node,
                     llvm::StringMap<Value> &valueMap,
                     llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitLeakyRelu(const onnx::NodeProto &node,
                     llvm::StringMap<Value> &valueMap,
                     llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitMax(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitMaxPool(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitMin(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitMul(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitPad(const onnx::NodeProto &node,
               llvm::StringMap<Value> &valueMap,
               llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitReduceMean(const onnx::NodeProto &node,
                      llvm::StringMap<Value> &valueMap,
                      llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitReduceSum(const onnx::NodeProto &node,
                     llvm::StringMap<Value> &valueMap,
                     llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitReshape(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitShape(const onnx::NodeProto &node,
                 llvm::StringMap<Value> &valueMap,
                 llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSigmoid(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSlice(const onnx::NodeProto &node,
                 llvm::StringMap<Value> &valueMap,
                 llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSoftmax(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSqrt(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitSqueeze(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitTanh(const onnx::NodeProto &node,
                llvm::StringMap<Value> &valueMap,
                llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitTranspose(const onnx::NodeProto &node,
                     llvm::StringMap<Value> &valueMap,
                     llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitUnsqueeze(const onnx::NodeProto &node,
                     llvm::StringMap<Value> &valueMap,
                     llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitWhere(const onnx::NodeProto &node,
                 llvm::StringMap<Value> &valueMap,
                 llvm::StringMap<RankedTensorType> &tensorTypes);

  // Existing gawee ops that are useful to keep visible in the ONNX scaffold.
  bool emitBatchNormalization(const onnx::NodeProto &node,
                              llvm::StringMap<Value> &valueMap,
                              llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitFlatten(const onnx::NodeProto &node,
                   llvm::StringMap<Value> &valueMap,
                   llvm::StringMap<RankedTensorType> &tensorTypes);
  bool emitLinearLike(const onnx::NodeProto &node,
                      llvm::StringMap<Value> &valueMap,
                      llvm::StringMap<RankedTensorType> &tensorTypes);

  void setError(const llvm::Twine &msg);
  OwningOpRef<ModuleOp> createEmptyModule(llvm::StringRef onnxPath);
};

} // namespace mlir::gawee

#endif // GAWEE_EMIT_ONNXMLIREMITTER_H
