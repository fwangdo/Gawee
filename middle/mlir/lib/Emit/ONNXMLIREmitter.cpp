//===----------------------------------------------------------------------===//
// ONNXMLIREmitter Implementation
//===----------------------------------------------------------------------===//

#include "Emit/ONNXMLIREmitter.h"
#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;
using namespace mlir::gawee;

namespace {

Value lookupMappedValue(llvm::StringRef name, llvm::StringMap<Value> &valueMap) {
  // get value. 
  auto it = valueMap.find(name);
  if (it == valueMap.end()) {
    return Value();
  }
  return it->second;
}

RankedTensorType lookupTensorType(llvm::StringRef name,
                                  llvm::StringMap<RankedTensorType> &tensorTypes) {
  // get tensor type 
  auto it = tensorTypes.find(name);
  if (it == tensorTypes.end()) {
    return RankedTensorType();
  }
  return it->second;
}

DenseI64ArrayAttr getI64ArrayAttrFromNode(OpBuilder &builder,
                                          const onnx::NodeProto &node,
                                          llvm::StringRef attrName,
                                          ArrayRef<int64_t> defaultValue,
                                          bool duplicateScalar = false) {
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
      if (duplicateScalar) {
        values.push_back(attr.i());
      }
    }
    return builder.getDenseI64ArrayAttr(values);
  }

  return builder.getDenseI64ArrayAttr(defaultValue);
}

int64_t getI64AttrFromNode(const onnx::NodeProto &node,
                           llvm::StringRef attrName,
                           int64_t defaultValue) {
  for (const auto &attr : node.attribute()) {
    if (attr.name() == attrName && attr.has_i()) {
      return attr.i();
    }
  }
  return defaultValue;
}

double getF64AttrFromNode(const onnx::NodeProto &node,
                          llvm::StringRef attrName,
                          double defaultValue) {
  for (const auto &attr : node.attribute()) {
    if (attr.name() != attrName) {
      continue;
    }
    if (attr.has_f()) {
      return attr.f();
    }
    if (attr.has_i()) {
      return static_cast<double>(attr.i());
    }
  }
  return defaultValue;
}

bool getBoolAttrFromNode(const onnx::NodeProto &node,
                         llvm::StringRef attrName,
                         bool defaultValue) {
  return getI64AttrFromNode(node, attrName, defaultValue ? 1 : 0) != 0;
}

std::string getStringAttrFromNode(const onnx::NodeProto &node,
                                  llvm::StringRef attrName,
                                  llvm::StringRef defaultValue) {
  for (const auto &attr : node.attribute()) {
    if (attr.name() == attrName && attr.has_s()) {
      return attr.s();
    }
  }
  return defaultValue.str();
}

template <typename T>
bool appendRawTensorData(const onnx::TensorProto &tensor, SmallVectorImpl<int64_t> &out) {
  const std::string &raw = tensor.raw_data();
  if (raw.size() % sizeof(T) != 0) {
    return false;
  }

  size_t count = raw.size() / sizeof(T);
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, raw.data() + i * sizeof(T), sizeof(T));
    out.push_back(static_cast<int64_t>(value));
  }
  return true;
}

bool extractI64TensorLiteral(const onnx::TensorProto &tensor,
                             SmallVectorImpl<int64_t> &out) {
  switch (tensor.data_type()) {
  case onnx::TensorProto_DataType_INT64:
    if (tensor.int64_data_size() > 0) {
      out.reserve(tensor.int64_data_size());
      for (auto value : tensor.int64_data()) {
        out.push_back(value);
      }
      return true;
    }
    if (!tensor.raw_data().empty()) {
      return appendRawTensorData<int64_t>(tensor, out);
    }
    return tensor.dims_size() == 0;
  case onnx::TensorProto_DataType_INT32:
    if (tensor.int32_data_size() > 0) {
      out.reserve(tensor.int32_data_size());
      for (auto value : tensor.int32_data()) {
        out.push_back(value);
      }
      return true;
    }
    if (!tensor.raw_data().empty()) {
      return appendRawTensorData<int32_t>(tensor, out);
    }
    return tensor.dims_size() == 0;
  default:
    return false;
  }
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

bool ONNXMLIREmitter::emitAveragePool(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("AveragePool expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("AveragePool input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for AveragePool output: " + node.output(0));
    return false;
  }

  auto kernelSize =
      getI64ArrayAttrFromNode(*builder, node, "kernel_shape", {}, true);
  if (kernelSize.empty()) {
    setError("AveragePool requires kernel_shape attribute");
    return false;
  }

  auto strides = getI64ArrayAttrFromNode(*builder, node, "strides", {1, 1}, true);
  auto padding = getI64ArrayAttrFromNode(*builder, node, "pads", {0, 0}, true);
  SmallVector<int64_t> normalizedPadding(padding.asArrayRef().begin(),
                                         padding.asArrayRef().end());
  if (normalizedPadding.size() == 4) {
    normalizedPadding = {normalizedPadding[0], normalizedPadding[1]};
  }

  auto averagePoolOp = builder->create<AveragePoolOp>(
      loc,
      resultType,
      input,
      kernelSize,
      strides,
      builder->getDenseI64ArrayAttr(normalizedPadding),
      builder->getBoolAttr(getBoolAttrFromNode(node, "ceil_mode", false)),
      builder->getBoolAttr(getBoolAttrFromNode(node, "count_include_pad", false)));
  valueMap[node.output(0)] = averagePoolOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitCast(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Cast expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Cast input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Cast output: " + node.output(0));
    return false;
  }

  auto castOp = builder->create<CastOp>(
      loc, resultType, input, builder->getI64IntegerAttr(getI64AttrFromNode(node, "to", 0)));
  valueMap[node.output(0)] = castOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitConcat(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("Concat expects at least 1 input and exactly 1 output");
    return false;
  }

  SmallVector<Value> inputs;
  inputs.reserve(node.input_size());
  for (const auto &inputName : node.input()) {
    Value input = lookupMappedValue(inputName, valueMap);
    if (!input) {
      setError("Concat input value not found in emitter environment: " + inputName);
      return false;
    }
    inputs.push_back(input);
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Concat output: " + node.output(0));
    return false;
  }

  int64_t axis = getI64AttrFromNode(node, "axis", 0);
  if (axis < 0 && resultType.getRank() >= 0) {
    axis += resultType.getRank();
  }

  auto catOp = builder->create<CatOp>(
      loc, resultType, inputs, builder->getI64IntegerAttr(axis));
  valueMap[node.output(0)] = catOp.getResult();
  return true;
}

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

  auto strides =
      getI64ArrayAttrFromNode(*builder, node, "strides", {1, 1}, true);
  auto padding =
      getI64ArrayAttrFromNode(*builder, node, "pads", {0, 0}, true);
  auto dilation =
      getI64ArrayAttrFromNode(*builder, node, "dilations", {1, 1}, true);

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

bool ONNXMLIREmitter::emitSub(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Sub expects exactly 2 inputs and 1 output");
    return false; 
  }

  Value lhs = lookupMappedValue(node.input(0), valueMap);
  if (!lhs) {
    setError("Sub lhs value not found in emitter environment: " + node.input(0));
    return false; 
  }

  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!rhs) {
    setError("Sub rhs value not found in emitter environment: " + node.input(1));
    return false; 
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes); 
  if (!resultType) {
    setError("Missing output tensor type for Sub output: " + node.output(0));
    return false; 
  }

  auto subOp = builder->create<SubOp>(loc, resultType, lhs, rhs); 
  valueMap[node.output(0)] = subOp.getResult();  
  return true;
}

bool ONNXMLIREmitter::emitDiv(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Div expects exactly 2 inputs and 1 output");
    return false; 
  }

  // lhs 
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  if (!lhs) {
    setError("Div lhs value not found in emitter environment: " + node.input(0));
    return false; 
  }

  // rhs
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!rhs) {
    setError("Div rhs value not found in emitter environment: " + node.input(1));
    return false; 
  }

  // resultType 
  auto resultType = lookupTensorType(node.output(0), tensorTypes); 
  if (!resultType) {
    setError("Missing output tensor type for Div output: " + node.output(0)); 
    return false; 
  }

  // enrollment.  
  auto divOp = builder->create<DivOp>(loc, resultType, lhs, rhs); 
  valueMap[node.output(0)] = divOp.getResult(); 
  return true;
}

bool ONNXMLIREmitter::emitEqual(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Equal expects exactly 2 inputs and 1 output");
    return false;
  }

  Value lhs = lookupMappedValue(node.input(0), valueMap);
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!lhs) {
    setError("Equal lhs value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!rhs) {
    setError("Equal rhs value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Equal output: " + node.output(0));
    return false;
  }

  auto equalOp = builder->create<EqualOp>(loc, resultType, lhs, rhs);
  valueMap[node.output(0)] = equalOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitErf(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Erf expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Erf input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Erf output: " + node.output(0));
    return false;
  }

  auto erfOp = builder->create<ErfOp>(loc, resultType, input);
  valueMap[node.output(0)] = erfOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitExpand(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Expand expects exactly 2 inputs and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value shape = lookupMappedValue(node.input(1), valueMap);
  if (!input) {
    setError("Expand input value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!shape) {
    setError("Expand shape value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Expand output: " + node.output(0));
    return false;
  }

  auto expandOp = builder->create<ExpandOp>(loc, resultType, input, shape);
  valueMap[node.output(0)] = expandOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitGelu(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Gelu expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Gelu input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Gelu output: " + node.output(0));
    return false;
  }

  auto geluOp = builder->create<GeluOp>(loc, resultType, input);
  valueMap[node.output(0)] = geluOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitGlobalAveragePool(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("GlobalAveragePool expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("GlobalAveragePool input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for GlobalAveragePool output: " + node.output(0));
    return false;
  }

  auto globalAveragePoolOp =
      builder->create<GlobalAveragePoolOp>(loc, resultType, input);
  valueMap[node.output(0)] = globalAveragePoolOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitHardSigmoid(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("HardSigmoid expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("HardSigmoid input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for HardSigmoid output: " + node.output(0));
    return false;
  }

  double alpha = getF64AttrFromNode(node, "alpha", 0.2);
  double beta = getF64AttrFromNode(node, "beta", 0.5);

  auto hardSigmoidOp = builder->create<HardSigmoidOp>(
      loc,
      resultType,
      input,
      builder->getF64FloatAttr(alpha),
      builder->getF64FloatAttr(beta));
  valueMap[node.output(0)] = hardSigmoidOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitHardSwish(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("HardSwish expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("HardSwish input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for HardSwish output: " + node.output(0));
    return false;
  }

  auto hardSwishOp = builder->create<HardSwishOp>(loc, resultType, input);
  valueMap[node.output(0)] = hardSwishOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitLeakyRelu(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("LeakyRelu expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("LeakyRelu input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for LeakyRelu output: " + node.output(0));
    return false;
  }

  double alpha = getF64AttrFromNode(node, "alpha", 0.01);
  auto leakyReluOp = builder->create<LeakyReluOp>(
      loc, resultType, input, builder->getF64FloatAttr(alpha));
  valueMap[node.output(0)] = leakyReluOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitMax(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("Max expects at least 1 input and exactly 1 output");
    return false;
  }

  SmallVector<Value> inputs;
  for (const auto &inputName : node.input()) {
    Value input = lookupMappedValue(inputName, valueMap);
    if (!input) {
      setError("Max input value not found in emitter environment: " + inputName);
      return false;
    }
    inputs.push_back(input);
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Max output: " + node.output(0));
    return false;
  }

  auto maxOp = builder->create<MaxOp>(loc, resultType, inputs);
  valueMap[node.output(0)] = maxOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitMaxPool(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("MaxPool expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("MaxPool input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for MaxPool output: " + node.output(0));
    return false;
  }

  auto kernelSize =
      getI64ArrayAttrFromNode(*builder, node, "kernel_shape", {}, true);
  if (kernelSize.empty()) {
    setError("MaxPool requires kernel_shape attribute");
    return false;
  }

  auto strides = getI64ArrayAttrFromNode(*builder, node, "strides", {1, 1}, true);
  auto padding = getI64ArrayAttrFromNode(*builder, node, "pads", {0, 0}, true);
  auto dilation = getI64ArrayAttrFromNode(*builder, node, "dilations", {1, 1}, true);
  SmallVector<int64_t> normalizedPadding(padding.asArrayRef().begin(),
                                         padding.asArrayRef().end());
  if (normalizedPadding.size() == 4) {
    normalizedPadding = {normalizedPadding[0], normalizedPadding[1]};
  }

  auto maxPoolOp = builder->create<MaxPoolOp>(
      loc,
      resultType,
      input,
      kernelSize,
      strides,
      builder->getDenseI64ArrayAttr(normalizedPadding),
      dilation,
      builder->getBoolAttr(getBoolAttrFromNode(node, "ceil_mode", false)));
  valueMap[node.output(0)] = maxPoolOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitMin(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("Min expects at least 1 input and exactly 1 output");
    return false;
  }

  SmallVector<Value> inputs;
  for (const auto &inputName : node.input()) {
    Value input = lookupMappedValue(inputName, valueMap);
    if (!input) {
      setError("Min input value not found in emitter environment: " + inputName);
      return false;
    }
    inputs.push_back(input);
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Min output: " + node.output(0));
    return false;
  }

  auto minOp = builder->create<MinOp>(loc, resultType, inputs);
  valueMap[node.output(0)] = minOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitMul(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {

  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Mul expects exactly 2 inputs and 1 output");
    return false; 
  }

  // lhs 
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  if (!lhs) {
    setError("Mul lhs value not found in emitter environment: " + node.input(0));
    return false; 
  }

  // rhs
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!rhs) {
    setError("Mul rhs value not found in emitter environment: " + node.input(1));
    return false; 
  }

  // resultType 
  auto resultType = lookupTensorType(node.output(0), tensorTypes); 
  if (!resultType) {
    setError("Missing output tensor type for Mul output: " + node.output(0)); 
    return false; 
  }

  // enrollment.  
  auto mulOp = builder->create<MulOp>(loc, resultType, lhs, rhs); 
  valueMap[node.output(0)] = mulOp.getResult(); 
  return true;
}

bool ONNXMLIREmitter::emitPad(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 2 || node.output_size() != 1) {
    setError("Pad expects at least 2 inputs and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value pads = lookupMappedValue(node.input(1), valueMap);
  if (!input) {
    setError("Pad input value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!pads) {
    setError("Pad pads value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Pad output: " + node.output(0));
    return false;
  }

  Value constantValue;
  if (node.input_size() >= 3 && !node.input(2).empty()) {
    constantValue = lookupMappedValue(node.input(2), valueMap);
    if (!constantValue) {
      setError("Pad constant value not found in emitter environment: " + node.input(2));
      return false;
    }
  } else {
    auto scalarType = RankedTensorType::get({}, resultType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(
        scalarType, builder->getZeroAttr(resultType.getElementType()));
    constantValue = arith::ConstantOp::create(*builder, loc, scalarType, zeroAttr);
  }

  auto padOp = builder->create<PadOp>(
      loc,
      resultType,
      input,
      pads,
      constantValue,
      builder->getStringAttr(getStringAttrFromNode(node, "mode", "constant")));
  valueMap[node.output(0)] = padOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitReduceMean(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("ReduceMean expects at least 1 input and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("ReduceMean input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for ReduceMean output: " + node.output(0));
    return false;
  }

  SmallVector<int64_t> axes;
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("ReduceMean axes must be a constant integer tensor: " + node.input(1));
      return false;
    }
    axes = it->second;
  } else {
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "axes" && attr.ints_size() > 0) {
        for (auto axis : attr.ints()) {
          axes.push_back(axis);
        }
      }
    }
    if (axes.empty()) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType) {
        setError("ReduceMean needs ranked input when axes are omitted");
        return false;
      }
      for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
        axes.push_back(axis);
      }
    }
  }

  auto reduceMeanOp = builder->create<ReduceMeanOp>(
      loc,
      resultType,
      input,
      builder->getDenseI64ArrayAttr(axes),
      builder->getBoolAttr(getBoolAttrFromNode(node, "keepdims", true)));
  valueMap[node.output(0)] = reduceMeanOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitReduceSum(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("ReduceSum expects at least 1 input and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("ReduceSum input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for ReduceSum output: " + node.output(0));
    return false;
  }

  SmallVector<int64_t> axes;
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("ReduceSum axes must be a constant integer tensor: " + node.input(1));
      return false;
    }
    axes = it->second;
  } else {
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "axes" && attr.ints_size() > 0) {
        for (auto axis : attr.ints()) {
          axes.push_back(axis);
        }
      }
    }
    if (axes.empty()) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType) {
        setError("ReduceSum needs ranked input when axes are omitted");
        return false;
      }
      for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
        axes.push_back(axis);
      }
    }
  }

  auto reduceSumOp = builder->create<ReduceSumOp>(
      loc,
      resultType,
      input,
      builder->getDenseI64ArrayAttr(axes),
      builder->getBoolAttr(getBoolAttrFromNode(node, "keepdims", true)));
  valueMap[node.output(0)] = reduceSumOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitReshape(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Reshape expects exactly 2 inputs and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value shape = lookupMappedValue(node.input(1), valueMap);
  if (!input) {
    setError("Reshape input value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!shape) {
    setError("Reshape shape value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Reshape output: " + node.output(0));
    return false;
  }

  auto reshapeOp = builder->create<ReshapeOp>(loc, resultType, input, shape);
  valueMap[node.output(0)] = reshapeOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitShape(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Shape expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Shape input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Shape output: " + node.output(0));
    return false;
  }

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  int64_t rank = inputType ? inputType.getRank() : resultType.getDimSize(0);
  int64_t start = getI64AttrFromNode(node, "start", 0);
  int64_t end = getI64AttrFromNode(node, "end", rank);
  if (start < 0 && rank >= 0)
    start += rank;
  if (end < 0 && rank >= 0)
    end += rank;

  auto shapeOp = builder->create<ShapeOp>(
      loc,
      resultType,
      input,
      builder->getI64IntegerAttr(start),
      builder->getI64IntegerAttr(end));
  valueMap[node.output(0)] = shapeOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitSigmoid(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Sigmoid expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Sigmoid input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Sigmoid output: " + node.output(0));
    return false;
  }

  auto sigmoidOp = builder->create<SigmoidOp>(loc, resultType, input);
  valueMap[node.output(0)] = sigmoidOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitSlice(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 3 || node.output_size() != 1) {
    setError("Slice expects at least 3 inputs and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value starts = lookupMappedValue(node.input(1), valueMap);
  Value ends = lookupMappedValue(node.input(2), valueMap);
  if (!input) {
    setError("Slice input value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!starts) {
    setError("Slice starts value not found in emitter environment: " + node.input(1));
    return false;
  }
  if (!ends) {
    setError("Slice ends value not found in emitter environment: " + node.input(2));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Slice output: " + node.output(0));
    return false;
  }

  auto makeI64TensorConst = [&](ArrayRef<int64_t> values) -> Value {
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(values.size())},
                              builder->getIntegerType(64));
    auto attr = DenseElementsAttr::get(tensorType, values);
    return arith::ConstantOp::create(*builder, loc, tensorType, attr);
  };

  auto startsType = dyn_cast<RankedTensorType>(starts.getType());
  int64_t numSlices = startsType && startsType.getRank() == 1 &&
                              !startsType.isDynamicDim(0)
                          ? startsType.getShape()[0]
                          : -1;

  Value axes;
  if (node.input_size() >= 4 && !node.input(3).empty()) {
    axes = lookupMappedValue(node.input(3), valueMap);
    if (!axes) {
      setError("Slice axes value not found in emitter environment: " + node.input(3));
      return false;
    }
  } else {
    if (numSlices < 0) {
      setError("Slice cannot synthesize default axes without static starts length");
      return false;
    }
    SmallVector<int64_t> defaultAxes;
    for (int64_t i = 0; i < numSlices; ++i)
      defaultAxes.push_back(i);
    axes = makeI64TensorConst(defaultAxes);
  }

  Value steps;
  if (node.input_size() >= 5 && !node.input(4).empty()) {
    steps = lookupMappedValue(node.input(4), valueMap);
    if (!steps) {
      setError("Slice steps value not found in emitter environment: " + node.input(4));
      return false;
    }
  } else {
    if (numSlices < 0) {
      setError("Slice cannot synthesize default steps without static starts length");
      return false;
    }
    SmallVector<int64_t> defaultSteps(numSlices, 1);
    steps = makeI64TensorConst(defaultSteps);
  }

  auto sliceOp = builder->create<SliceOp>(
      loc, resultType, input, starts, ends, axes, steps);
  valueMap[node.output(0)] = sliceOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitSoftmax(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Softmax expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Softmax input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Softmax output: " + node.output(0));
    return false;
  }

  int64_t axis = getI64AttrFromNode(node, "axis", -1);
  int64_t rank = resultType.getRank();
  if (axis < 0 && rank >= 0) {
    axis += rank;
  }

  auto softmaxOp = builder->create<SoftmaxOp>(
      loc, resultType, input, builder->getI64IntegerAttr(axis));
  valueMap[node.output(0)] = softmaxOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitSqrt(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Sqrt expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Sqrt input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Sqrt output: " + node.output(0));
    return false;
  }

  auto sqrtOp = builder->create<SqrtOp>(loc, resultType, input);
  valueMap[node.output(0)] = sqrtOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitSqueeze(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("Squeeze expects at least 1 input and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Squeeze input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Squeeze output: " + node.output(0));
    return false;
  }

  SmallVector<int64_t> axes;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "axes" && attr.ints_size() > 0) {
      for (auto axis : attr.ints())
        axes.push_back(axis);
    }
  }
  if (axes.empty() && node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("Squeeze axes must be a constant integer tensor: " + node.input(1));
      return false;
    }
    axes = it->second;
  }

  auto squeezeOp = builder->create<SqueezeOp>(
      loc, resultType, input, builder->getDenseI64ArrayAttr(axes));
  valueMap[node.output(0)] = squeezeOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitTanh(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Tanh expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Tanh input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Tanh output: " + node.output(0));
    return false;
  }

  auto tanhOp = builder->create<TanhOp>(loc, resultType, input);
  valueMap[node.output(0)] = tanhOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitTranspose(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Transpose expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Transpose input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Transpose output: " + node.output(0));
    return false;
  }

  SmallVector<int64_t> perm;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "perm" && attr.ints_size() > 0) {
      for (auto value : attr.ints())
        perm.push_back(value);
    }
  }
  if (perm.empty()) {
    for (int64_t dim = resultType.getRank() - 1; dim >= 0; --dim)
      perm.push_back(dim);
  }

  auto transposeOp = builder->create<TransposeOp>(
      loc, resultType, input, builder->getDenseI64ArrayAttr(perm));
  valueMap[node.output(0)] = transposeOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitUnsqueeze(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("Unsqueeze expects at least 1 input and exactly 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Unsqueeze input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Unsqueeze output: " + node.output(0));
    return false;
  }

  SmallVector<int64_t> axes;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "axes" && attr.ints_size() > 0) {
      for (auto axis : attr.ints())
        axes.push_back(axis);
    }
  }
  if (axes.empty() && node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("Unsqueeze axes must be a constant integer tensor: " + node.input(1));
      return false;
    }
    axes = it->second;
  }

  auto unsqueezeOp = builder->create<UnsqueezeOp>(
      loc, resultType, input, builder->getDenseI64ArrayAttr(axes));
  valueMap[node.output(0)] = unsqueezeOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitWhere(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 3 || node.output_size() != 1) {
    setError("Where expects exactly 3 inputs and 1 output");
    return false;
  }

  Value condition = lookupMappedValue(node.input(0), valueMap);
  Value lhs = lookupMappedValue(node.input(1), valueMap);
  Value rhs = lookupMappedValue(node.input(2), valueMap);
  if (!condition) {
    setError("Where condition value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!lhs) {
    setError("Where lhs value not found in emitter environment: " + node.input(1));
    return false;
  }
  if (!rhs) {
    setError("Where rhs value not found in emitter environment: " + node.input(2));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Where output: " + node.output(0));
    return false;
  }

  auto whereOp = builder->create<WhereOp>(loc, resultType, condition, lhs, rhs);
  valueMap[node.output(0)] = whereOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitBatchNormalization(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 5 || node.output_size() != 1) {
    setError("BatchNormalization expects exactly 5 inputs and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value weight = lookupMappedValue(node.input(1), valueMap);
  Value bias = lookupMappedValue(node.input(2), valueMap);
  Value runningMean = lookupMappedValue(node.input(3), valueMap);
  Value runningVar = lookupMappedValue(node.input(4), valueMap);
  if (!input || !weight || !bias || !runningMean || !runningVar) {
    setError("BatchNormalization operands not found in emitter environment");
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for BatchNormalization output: " + node.output(0));
    return false;
  }

  auto batchNormOp = builder->create<BatchNormOp>(
      loc,
      resultType,
      input,
      weight,
      bias,
      builder->getBoolAttr(true),
      runningMean,
      runningVar,
      builder->getF64FloatAttr(getF64AttrFromNode(node, "epsilon", 1.0e-5)));
  valueMap[node.output(0)] = batchNormOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitFlatten(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Flatten expects exactly 1 input and 1 output");
    return false;
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Flatten input value not found in emitter environment: " + node.input(0));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for Flatten output: " + node.output(0));
    return false;
  }

  int64_t axis = getI64AttrFromNode(node, "axis", 1);
  if (axis != 1) {
    setError("Flatten currently supports ONNX axis=1 only with gawee.flatten");
    return false;
  }

  auto flattenOp = builder->create<FlattenOp>(
      loc,
      resultType,
      input,
      builder->getI64IntegerAttr(1),
      builder->getI64IntegerAttr(-1));
  valueMap[node.output(0)] = flattenOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitLinearLike(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() < 2 || node.output_size() != 1) {
    setError("LinearLike expects at least 2 inputs and exactly 1 output");
    return false;
  }

  if (node.op_type() == "Gemm") {
    if (getI64AttrFromNode(node, "transA", 0) != 0 ||
        getI64AttrFromNode(node, "transB", 1) != 1 ||
        getF64AttrFromNode(node, "alpha", 1.0) != 1.0 ||
        getF64AttrFromNode(node, "beta", 1.0) != 1.0) {
      setError("Gemm currently supports only transA=0, transB=1, alpha=1, beta=1");
      return false;
    }
  }

  Value input = lookupMappedValue(node.input(0), valueMap);
  Value weight = lookupMappedValue(node.input(1), valueMap);
  if (!input) {
    setError("LinearLike input value not found in emitter environment: " + node.input(0));
    return false;
  }
  if (!weight) {
    setError("LinearLike weight value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for LinearLike output: " + node.output(0));
    return false;
  }

  Value biasValue;
  if (node.input_size() >= 3 && !node.input(2).empty()) {
    biasValue = lookupMappedValue(node.input(2), valueMap);
    if (!biasValue) {
      setError("LinearLike bias value not found in emitter environment: " + node.input(2));
      return false;
    }
  } else {
    int64_t outFeatures = -1;
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    if (weightType && weightType.getRank() == 2 && !weightType.isDynamicDim(0)) {
      outFeatures = weightType.getShape()[0];
    } else if (resultType.getRank() >= 1 && !resultType.isDynamicDim(resultType.getRank() - 1)) {
      outFeatures = resultType.getShape().back();
    }
    if (outFeatures < 0) {
      setError("Cannot synthesize LinearLike bias because output width is unknown");
      return false;
    }

    auto biasType = RankedTensorType::get({outFeatures}, resultType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(
        biasType, builder->getZeroAttr(resultType.getElementType()));
    biasValue = arith::ConstantOp::create(*builder, loc, biasType, zeroAttr);
  }

  auto linearOp = builder->create<LinearOp>(loc, resultType, input, weight, biasValue);
  valueMap[node.output(0)] = linearOp.getResult();
  return true;
}

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
