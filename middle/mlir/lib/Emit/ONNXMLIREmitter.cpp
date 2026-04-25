//===----------------------------------------------------------------------===//
// ONNXMLIREmitter Implementation
//===----------------------------------------------------------------------===//

#include "Emit/ONNXMLIREmitter.h"
#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;
using namespace mlir::gawee;

namespace {

Value lookupMappedValue(llvm::StringRef name, llvm::StringMap<Value> &valueMap) {
  auto it = valueMap.find(name);
  if (it == valueMap.end()) {
    return Value();
  }
  return it->second;
}

RankedTensorType lookupTensorType(llvm::StringRef name,
                                  llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto it = tensorTypes.find(name);
  if (it == tensorTypes.end()) {
    return RankedTensorType();
  }
  return it->second;
}

DenseI64ArrayAttr getI64ArrayAttrFromNode(OpBuilder &builder,
                                          const onnx::NodeProto &node,
                                          llvm::StringRef attrName,
                                          ArrayRef<int64_t> defaultValue) {
  for (const auto &attr : node.attribute()) {
    if (attr.name() != attrName) {
      continue;
    }

    SmallVector<int64_t> values;
    if (attr.ints_size() > 0) {
      for (auto value : attr.ints()) {
        values.push_back(value);
      }
    } else if (attr.has_i()) {
      values.push_back(attr.i());
      values.push_back(attr.i());
    }
    return builder.getDenseI64ArrayAttr(values);
  }

  return builder.getDenseI64ArrayAttr(defaultValue);
}

} // namespace

ONNXMLIREmitter::ONNXMLIREmitter(MLIRContext *context) : ctx(context) {
  builder = std::make_unique<OpBuilder>(ctx);
}

bool ONNXMLIREmitter::emitRelu(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Relu expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Relu input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Relu output: " + node.output(0));
    return false;
  }

  auto reluOp = builder->create<ReluOp>(loc, resultType, input);
  valueMap[node.output(0)] = reluOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitAdd(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Add expects exactly 2 inputs and 1 output");
    return false;
  }

  Value lhs = lookupMappedValue(node.input(0), valueMap);
  if (!lhs) {
    setError("Add lhs value not found in emitter environment: " + node.input(0));
    return false;
  }

  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!rhs) {
    setError("Add rhs value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Add output: " + node.output(0));
    return false;
  }

  auto addOp = builder->create<AddOp>(loc, resultType, lhs, rhs);
  valueMap[node.output(0)] = addOp.getResult();
  return true;
}

#define GAWEE_ONNX_STUB(MethodName, OpName)                                    \
  bool ONNXMLIREmitter::MethodName(                                            \
      const onnx::NodeProto &node,                                             \
      llvm::StringMap<Value> &valueMap,                                        \
      llvm::StringMap<RankedTensorType> &tensorTypes) {                        \
    (void)node;                                                                \
    (void)valueMap;                                                            \
    (void)tensorTypes;                                                         \
    setError("TODO: implement " OpName " in ONNXMLIREmitter");                \
    return false;                                                              \
  }

GAWEE_ONNX_STUB(emitAveragePool, "emitAveragePool()")
GAWEE_ONNX_STUB(emitCast, "emitCast()")
GAWEE_ONNX_STUB(emitConcat, "emitConcat()")

bool ONNXMLIREmitter::emitConv(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 2 || node.output_size() < 1) {
    setError("Conv node is missing required inputs/outputs");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Conv input value not found in emitter environment: " + node.input(0));
    return false;
  }

  Value weight = lookupMappedValue(node.input(1), valueMap);
  if (!weight) {
    setError("Conv weight value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Conv output: " + node.output(0));
    return false;
  }

  Value biasValue;
  if (node.input_size() >= 3 && !node.input(2).empty()) {
    biasValue = lookupMappedValue(node.input(2), valueMap);
    if (!biasValue) {
      setError("Conv bias value not found in emitter environment: " + node.input(2));
      return false;
    }
  } else {
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    if (!weightType || weightType.getRank() < 1 || weightType.isDynamicDim(0)) {
      setError("Cannot synthesize Conv bias because weight output channel is unknown");
      return false;
    }

    int64_t outChannels = weightType.getShape()[0];
    auto biasType = RankedTensorType::get({outChannels}, weightType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(
        biasType, builder->getZeroAttr(weightType.getElementType()));
    biasValue = arith::ConstantOp::create(*builder, loc, biasType, zeroAttr);
  }

  auto strides = getI64ArrayAttrFromNode(*builder, node, "strides", {1, 1});
  auto padding = getI64ArrayAttrFromNode(*builder, node, "pads", {0, 0});
  auto dilation = getI64ArrayAttrFromNode(*builder, node, "dilations", {1, 1});

  SmallVector<int64_t> normalizedPadding(padding.asArrayRef().begin(),
                                         padding.asArrayRef().end());
  if (normalizedPadding.size() == 4) {
    normalizedPadding = {normalizedPadding[0], normalizedPadding[1]};
  }

  auto convOp = builder->create<ConvOp>(
      loc,
      resultType,
      input,
      weight,
      biasValue,
      strides,
      builder->getDenseI64ArrayAttr(normalizedPadding),
      dilation);

  valueMap[node.output(0)] = convOp.getResult();
  return true;
}

GAWEE_ONNX_STUB(emitSub, "emitSub()")
GAWEE_ONNX_STUB(emitDiv, "emitDiv()")
GAWEE_ONNX_STUB(emitEqual, "emitEqual()")
GAWEE_ONNX_STUB(emitErf, "emitErf()")
GAWEE_ONNX_STUB(emitExpand, "emitExpand()")
GAWEE_ONNX_STUB(emitGelu, "emitGelu()")
GAWEE_ONNX_STUB(emitGlobalAveragePool, "emitGlobalAveragePool()")
GAWEE_ONNX_STUB(emitHardSigmoid, "emitHardSigmoid()")
GAWEE_ONNX_STUB(emitHardSwish, "emitHardSwish()")
GAWEE_ONNX_STUB(emitLeakyRelu, "emitLeakyRelu()")
GAWEE_ONNX_STUB(emitMax, "emitMax()")
GAWEE_ONNX_STUB(emitMaxPool, "emitMaxPool()")
GAWEE_ONNX_STUB(emitMin, "emitMin()")
GAWEE_ONNX_STUB(emitMul, "emitMul()")
GAWEE_ONNX_STUB(emitPad, "emitPad()")
GAWEE_ONNX_STUB(emitReduceMean, "emitReduceMean()")
GAWEE_ONNX_STUB(emitReduceSum, "emitReduceSum()")
GAWEE_ONNX_STUB(emitReshape, "emitReshape()")
GAWEE_ONNX_STUB(emitShape, "emitShape()")
GAWEE_ONNX_STUB(emitSigmoid, "emitSigmoid()")
GAWEE_ONNX_STUB(emitSlice, "emitSlice()")
GAWEE_ONNX_STUB(emitSoftmax, "emitSoftmax()")
GAWEE_ONNX_STUB(emitSqrt, "emitSqrt()")
GAWEE_ONNX_STUB(emitSqueeze, "emitSqueeze()")
GAWEE_ONNX_STUB(emitTanh, "emitTanh()")
GAWEE_ONNX_STUB(emitTranspose, "emitTranspose()")
GAWEE_ONNX_STUB(emitUnsqueeze, "emitUnsqueeze()")
GAWEE_ONNX_STUB(emitWhere, "emitWhere()")
GAWEE_ONNX_STUB(emitBatchNormalization, "emitBatchNormalization()")
GAWEE_ONNX_STUB(emitFlatten, "emitFlatten()")

bool ONNXMLIREmitter::emitLinearLike(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  (void)node;
  (void)valueMap;
  (void)tensorTypes;

  // TODO: decide ONNX MatMul/Gemm -> gawee.linear contract.
  //
  // What to read first:
  //   - MLIREmitter::emitLinear() in middle/mlir/lib/Emit/MLIREmitter.cpp
  //   - rewrite_gemm.py and rewrite_matmul.py in front/onnx_rewrite/passes
  //   - Gawee_LinearOp in middle/mlir/include/Gawee/GaweeOps.td
  //
  // Questions to answer before coding:
  //   - will normalized ONNX still contain Gemm or only MatMul-like forms?
  //   - when can gawee.linear represent the ONNX node exactly?
  //   - if bias is absent, do we synthesize zero bias like Conv does?
  setError("TODO: implement emitLinearLike() in ONNXMLIREmitter");
  return false;
}

#undef GAWEE_ONNX_STUB

OwningOpRef<ModuleOp> ONNXMLIREmitter::emitFromFile(llvm::StringRef onnxPath) {
  // -----------------------------------------------------------------------
  // Basic ONNX -> gawee dialect path
  //
  // Goal of this implementation:
  //   - show how to deserialize ModelProto in C++
  //   - build MLIR function arguments from ONNX inputs/initializers
  //   - fully emit Conv nodes
  //   - leave other ops as explicit TODOs for incremental extension
  //
  // Design choice:
  //   - normalized ONNX is assumed as input
  //   - ONNXMLIREmitter is NOT responsible for graph surgery/rewrite
  //   - it only lowers an already-clean ONNX graph to gawee dialect
  // -----------------------------------------------------------------------

  onnx::ModelProto model;
  std::ifstream inputFile(std::string(onnxPath), std::ios::binary);

  // error handling case. 
  if (!inputFile.good()) {
    setError("Could not open ONNX file: " + onnxPath.str());
    return nullptr;
  }
  if (!model.ParseFromIstream(&inputFile)) {
    setError("Failed to parse ONNX ModelProto from file: " + onnxPath.str());
    return nullptr;
  }

  if (!model.has_graph()) {
    setError("ONNX model does not contain a graph");
    return nullptr;
  }

  const auto &graph = model.graph();

  auto loc = builder->getUnknownLoc();
  auto module = ModuleOp::create(loc);
  // cursor move. it moves to the end of module.getBody() 
  builder->setInsertionPointToEnd(module.getBody());

  // --- Local helpers ------------------------------------------------------
  // They are intentionally kept inside emitFromFile for now, so the first
  // complete path can be read in one place while experimenting with ONNX C++.

  auto getElementType = [&](int32_t elemType) -> Type {
    // Keep the initial mapping deliberately small.
    // Expand this as new models introduce more dtypes.
    switch (elemType) {
    case onnx::TensorProto_DataType_FLOAT:
      return builder->getF32Type();
    case onnx::TensorProto_DataType_FLOAT16:
      return builder->getF16Type();
    case onnx::TensorProto_DataType_INT64:
      return builder->getIntegerType(64);
    case onnx::TensorProto_DataType_INT32:
      return builder->getIntegerType(32);
    default:
      return Type();
    }
  };

  auto parseTensorShape = [&](const onnx::TensorShapeProto &shapeProto,
                              int32_t elemType) -> RankedTensorType {
    SmallVector<int64_t> shape;
    for (const auto &dim : shapeProto.dim()) {
      if (dim.has_dim_value()) {
        shape.push_back(dim.dim_value());
      } else {
        // Dynamic dims are allowed at the ONNX level.
        // MLIR models them as ShapedType::kDynamic.
        shape.push_back(ShapedType::kDynamic);
      }
    }

    auto elementType = getElementType(elemType);
    if (!elementType) {
      return RankedTensorType();
    }

    // only if we can return shape and valtype. 
    return RankedTensorType::get(shape, elementType);
  };

  auto parseValueInfoType = [&](const onnx::ValueInfoProto &valueInfo) -> RankedTensorType {
    if (!valueInfo.has_type() || !valueInfo.type().has_tensor_type()) {
      return RankedTensorType();
    }
    const auto &tensorType = valueInfo.type().tensor_type();
    return parseTensorShape(tensorType.shape(), tensorType.elem_type());
  };

  auto parseInitializerType = [&](const onnx::TensorProto &tensor) -> RankedTensorType {
    SmallVector<int64_t> shape;
    for (auto dim : tensor.dims()) {
      shape.push_back(dim);
    }
    auto elementType = getElementType(tensor.data_type());
    if (!elementType) {
      return RankedTensorType();
    }
    return RankedTensorType::get(shape, elementType);
  };

  // --- Collect types from ONNX graph -------------------------------------
  // We need names -> tensor types before creating the function signature.
  llvm::StringMap<RankedTensorType> tensorTypes;
  std::unordered_set<std::string> initializerNames;

  for (const auto &valueInfo : graph.input()) {
    auto type = parseValueInfoType(valueInfo);
    if (type) {
      tensorTypes[valueInfo.name()] = type;
    }
  }
  for (const auto &valueInfo : graph.value_info()) {
    auto type = parseValueInfoType(valueInfo);
    if (type) {
      tensorTypes[valueInfo.name()] = type;
    }
  }
  for (const auto &valueInfo : graph.output()) {
    auto type = parseValueInfoType(valueInfo);
    if (type) {
      tensorTypes[valueInfo.name()] = type;
    }
  }
  for (const auto &initializer : graph.initializer()) {
    auto type = parseInitializerType(initializer);
    if (type) {
      tensorTypes[initializer.name()] = type;
      initializerNames.insert(initializer.name());
    }
  }

  // --- Build function signature ------------------------------------------
  // Convention for the scaffold:
  //   - graph inputs become function arguments
  //   - initializers also become function arguments
  // This mirrors the current JSON emitter structure and keeps lowering simple
  // before deciding whether some constants should become arith.constant ops.
  SmallVector<Type> argTypes;
  SmallVector<std::string> argNames;

  for (const auto &input : graph.input()) {
    if (initializerNames.count(input.name()) != 0) {
      continue; // ONNX often duplicates initializers in graph.input; skip here.
    }
    auto it = tensorTypes.find(input.name());
    if (it == tensorTypes.end()) {
      setError("Missing type information for ONNX input: " + input.name());
      return nullptr;
    }
    argTypes.push_back(it->second);
    argNames.push_back(input.name());
  }

  for (const auto &initializer : graph.initializer()) {
    auto it = tensorTypes.find(initializer.name());
    if (it == tensorTypes.end()) {
      setError("Missing type information for ONNX initializer: " + initializer.name());
      return nullptr;
    }
    argTypes.push_back(it->second);
    argNames.push_back(initializer.name());
  }

  SmallVector<Type> resultTypes;
  for (const auto &output : graph.output()) {
    auto it = tensorTypes.find(output.name());
    if (it == tensorTypes.end()) {
      setError("Missing type information for ONNX output: " + output.name());
      return nullptr;
    }
    resultTypes.push_back(it->second);
  }

  auto funcType = builder->getFunctionType(argTypes, resultTypes);
  auto func = builder->create<func::FuncOp>(loc, "forward", funcType);
  auto *entryBlock = func.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  func->setAttr("gawee.onnx_source", builder->getStringAttr(onnxPath));

  llvm::StringMap<Value> valueMap;
  for (size_t i = 0; i < argNames.size(); ++i) {
    valueMap[argNames[i]] = entryBlock->getArgument(i);
  }

  // --- Emit nodes ---------------------------------------------------------
  for (const auto &node : graph.node()) {
    if (node.op_type() == "Relu") {
      if (!emitRelu(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Add") {
      if (!emitAdd(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "MatMul" || node.op_type() == "Gemm") {
      if (!emitLinearLike(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Conv") {
      if (!emitConv(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "AveragePool") {
      if (!emitAveragePool(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Cast") {
      if (!emitCast(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Concat") {
      if (!emitConcat(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Sub") {
      if (!emitSub(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Div") {
      if (!emitDiv(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Equal") {
      if (!emitEqual(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Erf") {
      if (!emitErf(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Expand") {
      if (!emitExpand(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Gelu") {
      if (!emitGelu(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "GlobalAveragePool") {
      if (!emitGlobalAveragePool(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "HardSigmoid") {
      if (!emitHardSigmoid(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "HardSwish") {
      if (!emitHardSwish(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "LeakyRelu") {
      if (!emitLeakyRelu(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Max") {
      if (!emitMax(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "MaxPool") {
      if (!emitMaxPool(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Min") {
      if (!emitMin(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Mul") {
      if (!emitMul(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Pad") {
      if (!emitPad(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "ReduceMean") {
      if (!emitReduceMean(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "ReduceSum") {
      if (!emitReduceSum(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Reshape") {
      if (!emitReshape(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Shape") {
      if (!emitShape(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Sigmoid") {
      if (!emitSigmoid(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Slice") {
      if (!emitSlice(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Softmax") {
      if (!emitSoftmax(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Sqrt") {
      if (!emitSqrt(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Squeeze") {
      if (!emitSqueeze(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Tanh") {
      if (!emitTanh(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Transpose") {
      if (!emitTranspose(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Unsqueeze") {
      if (!emitUnsqueeze(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Where") {
      if (!emitWhere(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "BatchNormalization") {
      if (!emitBatchNormalization(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Flatten") {
      if (!emitFlatten(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }

    setError("Unsupported op during emission: " + node.op_type());
    return nullptr;
  }

  // --- Return outputs -----------------------------------------------------
  SmallVector<Value> returnValues;
  for (const auto &output : graph.output()) {
    auto it = valueMap.find(output.name());
    if (it == valueMap.end()) {
      setError("Final output value not found: " + output.name());
      return nullptr;
    }
    returnValues.push_back(it->second);
  }
  builder->create<func::ReturnOp>(loc, returnValues);

  return module;
}

OwningOpRef<ModuleOp> ONNXMLIREmitter::createEmptyModule(llvm::StringRef onnxPath) {
  auto loc = builder->getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder->setInsertionPointToEnd(module.getBody());

  // Placeholder function so the emitter has the same structural target as the
  // JSON emitter. Real implementation should derive function type from ONNX IO.
  auto funcType = builder->getFunctionType(TypeRange{}, TypeRange{});
  auto func = builder->create<func::FuncOp>(loc, "forward", funcType);
  auto *entryBlock = func.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  builder->create<func::ReturnOp>(loc, ValueRange{});

  func->setAttr("gawee.onnx_source", builder->getStringAttr(onnxPath));
  return module;
}

void ONNXMLIREmitter::setError(const llvm::Twine &msg) {
  errorMsg = msg.str();
}
