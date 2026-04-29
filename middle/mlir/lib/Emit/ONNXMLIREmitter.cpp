//===----------------------------------------------------------------------===//
// ONNXMLIREmitter Implementation
//===----------------------------------------------------------------------===//

#include "Emit/ONNXMLIREmitter.h"
#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
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

SmallVector<Value> materializeDynamicDims(OpBuilder &builder, Location loc,
                                          RankedTensorType resultType,
                                          function_ref<Value(int64_t)> dimBuilder) {
  SmallVector<Value> dynamicDims;
  dynamicDims.reserve(resultType.getNumDynamicDims());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (!resultType.isDynamicDim(dim))
      continue;
    dynamicDims.push_back(dimBuilder(dim));
  }
  return dynamicDims;
}

Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType resultType,
                        ArrayRef<Value> dynamicDims = {}) {
  return tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                 resultType.getElementType(), dynamicDims);
}

Value createScalarConstant(OpBuilder &builder, Location loc, Type type,
                           double floatValue, int64_t intValue) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getFloatAttr(floatType, floatValue));
  }
  if (isa<IndexType>(type)) {
    return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(intValue));
  }
  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getIntegerAttr(type, intValue));
}

std::optional<Value> getBroadcastedDimValue(OpBuilder &builder, Location loc,
                                            RankedTensorType inputType, Value input,
                                            int64_t resultRank, int64_t resultDim) {
  int64_t inputRank = inputType.getRank();
  if (inputRank == 0)
    return std::nullopt;
  int64_t leading = resultRank - inputRank;
  if (resultDim < leading)
    return std::nullopt;
  int64_t inputDim = resultDim - leading;
  int64_t staticDim = inputType.getShape()[inputDim];
  if (staticDim == 1)
    return std::nullopt;
  if (inputType.isDynamicDim(inputDim))
    return tensor::DimOp::create(builder, loc, input, inputDim).getResult();
  return createScalarConstant(builder, loc, builder.getIndexType(), 0.0, staticDim);
}

SmallVector<Value> materializeBroadcastedDynamicDims(OpBuilder &builder, Location loc,
                                                     RankedTensorType resultType,
                                                     Value lhs, Value rhs) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  SmallVector<Value> dynamicDims;
  dynamicDims.reserve(resultType.getNumDynamicDims());
  int64_t resultRank = resultType.getRank();
  for (int64_t dim = 0; dim < resultRank; ++dim) {
    if (!resultType.isDynamicDim(dim))
      continue;
    if (auto lhsDim = getBroadcastedDimValue(builder, loc, lhsType, lhs, resultRank, dim)) {
      dynamicDims.push_back(*lhsDim);
      continue;
    }
    if (auto rhsDim = getBroadcastedDimValue(builder, loc, rhsType, rhs, resultRank, dim)) {
      dynamicDims.push_back(*rhsDim);
      continue;
    }
    dynamicDims.push_back(
        createScalarConstant(builder, loc, builder.getIndexType(), 0.0, 1));
  }
  return dynamicDims;
}

Value extractTensorScalarAsIndex(OpBuilder &builder, Location loc, Value tensor,
                                 ArrayRef<Value> positions) {
  Value extracted = tensor::ExtractOp::create(builder, loc, tensor, positions);
  if (isa<IndexType>(extracted.getType()))
    return extracted;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(), extracted);
}

Value makeZeroTensorLike(OpBuilder &builder, Location loc,
                         RankedTensorType resultType,
                         ArrayRef<Value> dynamicDims = {}) {
  Value empty = createEmptyTensor(builder, loc, resultType, dynamicDims);
  Value zero = createScalarConstant(builder, loc, resultType.getElementType(), 0.0, 0);
  return linalg::FillOp::create(builder, loc, zero, empty).getResult(0);
}

std::string externalDataLocation(const onnx::TensorProto &tensor,
                                 llvm::StringRef key) {
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == key)
      return entry.value();
  }
  return "";
}

DenseElementsAttr buildDenseElementsAttr(OpBuilder &builder,
                                         const onnx::TensorProto &tensor,
                                         llvm::StringRef onnxDirectory) {
  SmallVector<int64_t> shape;
  for (auto dim : tensor.dims())
    shape.push_back(dim);

  Type elemType;
  switch (tensor.data_type()) {
  case onnx::TensorProto_DataType_FLOAT:
    elemType = builder.getF32Type();
    break;
  case onnx::TensorProto_DataType_FLOAT16:
    elemType = builder.getF16Type();
    break;
  case onnx::TensorProto_DataType_INT64:
    elemType = builder.getI64Type();
    break;
  case onnx::TensorProto_DataType_INT32:
    elemType = builder.getI32Type();
    break;
  case onnx::TensorProto_DataType_BOOL:
    elemType = builder.getI1Type();
    break;
  default:
    return DenseElementsAttr();
  }

  auto tensorType = RankedTensorType::get(shape, elemType);
  int64_t numElements = 1;
  for (int64_t dim : shape)
    numElements *= dim;
  if (shape.empty())
    numElements = 1;
  auto readExternalBytes = [&]() -> std::string {
    if (tensor.data_location() != onnx::TensorProto_DataLocation_EXTERNAL)
      return "";
    std::string location = externalDataLocation(tensor, "location");
    if (location.empty())
      return "";
    std::filesystem::path dataPath =
        std::filesystem::path(std::string(onnxDirectory)) / location;
    std::ifstream in(dataPath, std::ios::binary);
    if (!in)
      return "";
    std::string offsetText = externalDataLocation(tensor, "offset");
    std::string lengthText = externalDataLocation(tensor, "length");
    size_t offset = offsetText.empty() ? 0 : static_cast<size_t>(std::stoull(offsetText));
    size_t length = lengthText.empty() ? 0 : static_cast<size_t>(std::stoull(lengthText));
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::string raw(length, '\0');
    in.read(raw.data(), static_cast<std::streamsize>(length));
    if (static_cast<size_t>(in.gcount()) != length)
      return "";
    return raw;
  };
  if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT) {
    SmallVector<float> values;
    if (tensor.float_data_size() > 0) {
      values.reserve(tensor.float_data_size());
      for (auto value : tensor.float_data())
        values.push_back(value);
    } else if (!tensor.raw_data().empty()) {
      size_t count = tensor.raw_data().size() / sizeof(float);
      values.resize(count);
      std::memcpy(values.data(), tensor.raw_data().data(), tensor.raw_data().size());
    } else if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
      std::string raw = readExternalBytes();
      if (raw.empty())
        return DenseElementsAttr();
      size_t count = raw.size() / sizeof(float);
      values.resize(count);
      std::memcpy(values.data(), raw.data(), raw.size());
    } else if (numElements == 0) {
      return DenseElementsAttr::get(tensorType, ArrayRef<float>{});
    }
    return DenseElementsAttr::get(tensorType, ArrayRef<float>(values));
  }
  if (tensor.data_type() == onnx::TensorProto_DataType_BOOL) {
    SmallVector<int64_t> values;
    if (!tensor.raw_data().empty()) {
      values.reserve(tensor.raw_data().size());
      for (unsigned char byte : tensor.raw_data())
        values.push_back(byte != 0);
    } else if (numElements == 0) {
      return DenseElementsAttr::get(tensorType, ArrayRef<int64_t>{});
    }
    return DenseElementsAttr::get(tensorType, ArrayRef<int64_t>(values));
  }

  SmallVector<int64_t> i64Values;
  if (extractI64TensorLiteral(tensor, i64Values)) {
    if (tensor.data_type() == onnx::TensorProto_DataType_INT32) {
      SmallVector<int32_t> values;
      values.reserve(i64Values.size());
      for (int64_t value : i64Values)
        values.push_back(static_cast<int32_t>(value));
      return DenseElementsAttr::get(tensorType, ArrayRef<int32_t>(values));
    }
    if (tensor.data_type() == onnx::TensorProto_DataType_BOOL) {
      SmallVector<bool> values;
      values.reserve(i64Values.size());
      for (int64_t value : i64Values)
        values.push_back(value != 0);
      return DenseElementsAttr::get(tensorType, ArrayRef<bool>(values));
    }
    return DenseElementsAttr::get(tensorType, ArrayRef<int64_t>(i64Values));
  }
  if (numElements == 0) {
    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
      return DenseElementsAttr::get(tensorType, ArrayRef<int32_t>{});
    return DenseElementsAttr::get(tensorType, ArrayRef<int64_t>{});
  }
  if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL &&
      (tensor.data_type() == onnx::TensorProto_DataType_INT64 ||
       tensor.data_type() == onnx::TensorProto_DataType_INT32)) {
    std::string raw = readExternalBytes();
    if (raw.empty())
      return DenseElementsAttr();
    if (tensor.data_type() == onnx::TensorProto_DataType_INT64) {
      size_t count = raw.size() / sizeof(int64_t);
      i64Values.resize(count);
      std::memcpy(i64Values.data(), raw.data(), raw.size());
      return DenseElementsAttr::get(tensorType, ArrayRef<int64_t>(i64Values));
    }
    size_t count = raw.size() / sizeof(int32_t);
    SmallVector<int32_t> i32Values(count);
    std::memcpy(i32Values.data(), raw.data(), raw.size());
    return DenseElementsAttr::get(tensorType, ArrayRef<int32_t>(i32Values));
  }

  return DenseElementsAttr();
}

FailureOr<SmallVector<int64_t>> getConstantI64Values(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return failure();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    SmallVector<int64_t> values;
    values.reserve(dense.getNumElements());
    for (APInt v : dense.getValues<APInt>())
      values.push_back(v.getSExtValue());
    return values;
  }
  return failure();
}

Value createUnaryElementwise(OpBuilder &builder, Location loc, RankedTensorType resultType,
                             Value input, ArrayRef<Value> dynamicDims,
                             function_ref<Value(OpBuilder &, Location, Value)> bodyBuilder) {
  Value output = createEmptyTensor(builder, loc, resultType, dynamicDims);
  int64_t rank = resultType.getRank();
  SmallVector<AffineMap> indexingMaps(
      2, AffineMap::getMultiDimIdentityMap(rank, builder.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, ValueRange{input}, ValueRange{output},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange args) {
        Value result = bodyBuilder(nestedBuilder, bodyLoc, args[0]);
        linalg::YieldOp::create(nestedBuilder, bodyLoc, result);
      });
  return genericOp.getResult(0);
}

Value createBinaryElementwise(OpBuilder &builder, Location loc, RankedTensorType resultType,
                              Value lhs, Value rhs, ArrayRef<Value> dynamicDims,
                              function_ref<Value(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  Value output = createEmptyTensor(builder, loc, resultType, dynamicDims);
  int64_t rank = resultType.getRank();
  auto buildMap = [&](RankedTensorType inputType) {
    int64_t inRank = inputType.getRank();
    if (inRank == 0)
      return AffineMap::get(rank, 0, {}, builder.getContext());
    SmallVector<AffineExpr> exprs;
    int64_t leading = rank - inRank;
    for (int64_t i = 0; i < inRank; ++i) {
      int64_t outDim = leading + i;
      if (!inputType.isDynamicDim(i) && inputType.getShape()[i] == 1 &&
          (resultType.isDynamicDim(outDim) || resultType.getShape()[outDim] != 1)) {
        exprs.push_back(getAffineConstantExpr(0, builder.getContext()));
      } else {
        exprs.push_back(getAffineDimExpr(outDim, builder.getContext()));
      }
    }
    return AffineMap::get(rank, 0, exprs, builder.getContext());
  };
  SmallVector<AffineMap> indexingMaps = {
      buildMap(lhsType), buildMap(rhsType),
      AffineMap::getMultiDimIdentityMap(rank, builder.getContext())};
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{output},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange args) {
        Value result = bodyBuilder(nestedBuilder, bodyLoc, args[0], args[1]);
        linalg::YieldOp::create(nestedBuilder, bodyLoc, result);
      });
  return genericOp.getResult(0);
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

bool ONNXMLIREmitter::emitMatMul(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();

  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("MatMul expects exactly 2 inputs and 1 output");
    return false;
  }

  Value lhs = lookupMappedValue(node.input(0), valueMap);
  if (!lhs) {
    setError("MatMul lhs value not found in emitter environment: " + node.input(0));
    return false;
  }

  Value rhs = lookupMappedValue(node.input(1), valueMap);
  if (!rhs) {
    setError("MatMul rhs value not found in emitter environment: " + node.input(1));
    return false;
  }

  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!resultType) {
    setError("Missing output tensor type for MatMul output: " + node.output(0));
    return false;
  }

  auto matmulOp = builder->create<MatMulOp>(loc, resultType, lhs, rhs);
  valueMap[node.output(0)] = matmulOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitPow(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Pow expects exactly 2 inputs and 1 output");
    return false;
  }
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!lhs || !rhs || !resultType) {
    setError("Pow operands or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims =
      materializeBroadcastedDynamicDims(*builder, loc, resultType, lhs, rhs);
  Value result = createBinaryElementwise(
      *builder, loc, resultType, lhs, rhs, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value lhsVal, Value rhsVal) {
        return Value(math::PowFOp::create(nestedBuilder, bodyLoc, lhsVal, rhsVal));
      });
  valueMap[node.output(0)] = result;
  return true;
}

bool ONNXMLIREmitter::emitNeg(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Neg expects exactly 1 input and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !resultType) {
    setError("Neg input or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  Type elemType = resultType.getElementType();
  Value result = createUnaryElementwise(
      *builder, loc, resultType, input, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value arg) {
        if (isa<FloatType>(elemType))
          return Value(arith::NegFOp::create(nestedBuilder, bodyLoc, arg));
        return Value(arith::SubIOp::create(
            nestedBuilder, bodyLoc,
            createScalarConstant(nestedBuilder, bodyLoc, elemType, 0.0, 0), arg));
      });
  valueMap[node.output(0)] = result;
  return true;
}

bool ONNXMLIREmitter::emitAnd(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("And expects exactly 2 inputs and 1 output");
    return false;
  }
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!lhs || !rhs || !resultType) {
    setError("And operands or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims =
      materializeBroadcastedDynamicDims(*builder, loc, resultType, lhs, rhs);
  Value result = createBinaryElementwise(
      *builder, loc, resultType, lhs, rhs, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value lhsVal, Value rhsVal) {
        return Value(arith::AndIOp::create(nestedBuilder, bodyLoc, lhsVal, rhsVal));
      });
  valueMap[node.output(0)] = result;
  return true;
}

bool ONNXMLIREmitter::emitLessOrEqual(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("LessOrEqual expects exactly 2 inputs and 1 output");
    return false;
  }
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!lhs || !rhs || !resultType) {
    setError("LessOrEqual operands or result type missing");
    return false;
  }
  Type cmpType = cast<RankedTensorType>(lhs.getType()).getElementType();
  SmallVector<Value> dynamicDims =
      materializeBroadcastedDynamicDims(*builder, loc, resultType, lhs, rhs);
  Value result = createBinaryElementwise(
      *builder, loc, resultType, lhs, rhs, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value lhsVal, Value rhsVal) {
        if (isa<FloatType>(cmpType)) {
          return Value(arith::CmpFOp::create(
              nestedBuilder, bodyLoc, arith::CmpFPredicate::OLE, lhsVal, rhsVal));
        }
        return Value(arith::CmpIOp::create(
            nestedBuilder, bodyLoc, arith::CmpIPredicate::sle, lhsVal, rhsVal));
      });
  valueMap[node.output(0)] = result;
  return true;
}

bool ONNXMLIREmitter::emitIsNaN(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("IsNaN expects exactly 1 input and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !resultType) {
    setError("IsNaN input or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  Value result = createUnaryElementwise(
      *builder, loc, resultType, input, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value arg) {
        return Value(arith::CmpFOp::create(
            nestedBuilder, bodyLoc, arith::CmpFPredicate::UNO, arg, arg));
      });
  valueMap[node.output(0)] = result;
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

bool ONNXMLIREmitter::emitGather(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Gather expects exactly 2 inputs and 1 output");
    return false;
  }
  Value data = lookupMappedValue(node.input(0), valueMap);
  Value indices = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!data || !indices || !resultType) {
    setError("Gather operands or result type missing");
    return false;
  }
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType) {
    setError("Gather expects ranked tensors");
    return false;
  }
  int64_t axis = getI64AttrFromNode(node, "axis", 0);
  if (axis < 0)
    axis += dataType.getRank();
  auto gatherOp = builder->create<GatherOp>(
      loc, resultType, data, indices, builder->getI64IntegerAttr(axis));
  valueMap[node.output(0)] = gatherOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitGatherElements(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("GatherElements expects exactly 2 inputs and 1 output");
    return false;
  }
  Value data = lookupMappedValue(node.input(0), valueMap);
  Value indices = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!data || !indices || !resultType) {
    setError("GatherElements operands or result type missing");
    return false;
  }
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!indicesType) {
    setError("GatherElements expects ranked indices tensor");
    return false;
  }
  int64_t axis = getI64AttrFromNode(node, "axis", 0);
  if (axis < 0)
    axis += indicesType.getRank();
  auto gatherOp = builder->create<GatherElementsOp>(
      loc, resultType, data, indices, builder->getI64IntegerAttr(axis));
  valueMap[node.output(0)] = gatherOp.getResult();
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

bool ONNXMLIREmitter::emitReduceMax(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() < 1 || node.output_size() != 1) {
    setError("ReduceMax expects at least 1 input and exactly 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !resultType) {
    setError("ReduceMax input or result type missing");
    return false;
  }
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> axes;
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("ReduceMax axes must be a constant integer tensor");
      return false;
    }
    axes = it->second;
  } else {
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "axes" && attr.ints_size() > 0) {
        for (auto axis : attr.ints())
          axes.push_back(axis);
      }
    }
  }
  if (axes.empty()) {
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis)
      axes.push_back(axis);
  }
  for (int64_t &axis : axes) {
    if (axis < 0)
      axis += inputType.getRank();
  }
  SmallVector<int64_t> keptDims;
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (!llvm::is_contained(axes, dim))
      keptDims.push_back(dim);
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, keptDims[dim]); });
  Value init = createEmptyTensor(*builder, loc, resultType, dynamicDims);
  Type elemType = resultType.getElementType();
  Attribute minAttr;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    minAttr = builder->getFloatAttr(floatType, -std::numeric_limits<double>::infinity());
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    minAttr = builder->getIntegerAttr(intType, llvm::APInt::getSignedMinValue(intType.getWidth()));
  } else {
    setError("ReduceMax supports only float/int element types");
    return false;
  }
  Value seed = arith::ConstantOp::create(*builder, loc, cast<TypedAttr>(minAttr));
  init = linalg::FillOp::create(*builder, loc, seed, init).getResult(0);
  auto reduceOp = linalg::ReduceOp::create(
      *builder, loc, ValueRange{input}, ValueRange{init}, axes,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange args) {
        Value result;
        if (isa<FloatType>(elemType))
          result = arith::MaximumFOp::create(nestedBuilder, bodyLoc, args[0], args[1]);
        else
          result = arith::MaxSIOp::create(nestedBuilder, bodyLoc, args[0], args[1]);
        linalg::YieldOp::create(nestedBuilder, bodyLoc, result);
      });
  valueMap[node.output(0)] = reduceOp.getResult(0);
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

bool ONNXMLIREmitter::emitResize(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() < 3 || node.output_size() != 1) {
    setError("Resize expects at least 3 inputs and exactly 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  Value scales = lookupMappedValue(node.input(2), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !scales || !resultType) {
    setError("Resize operands or result type missing");
    return false;
  }
  if (getStringAttrFromNode(node, "mode", "nearest") != "nearest" ||
      getStringAttrFromNode(node, "coordinate_transformation_mode", "asymmetric") != "asymmetric" ||
      getStringAttrFromNode(node, "nearest_mode", "floor") != "floor") {
    setError("Resize currently supports only nearest/asymmetric/floor");
    return false;
  }
  auto resizeOp = builder->create<ResizeOp>(
      loc, resultType, input, scales,
      builder->getStringAttr(getStringAttrFromNode(node, "mode", "nearest")),
      builder->getStringAttr(getStringAttrFromNode(
          node, "coordinate_transformation_mode", "asymmetric")),
      builder->getStringAttr(getStringAttrFromNode(node, "nearest_mode", "floor")));
  valueMap[node.output(0)] = resizeOp.getResult();
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

bool ONNXMLIREmitter::emitRange(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 3 || node.output_size() != 1) {
    setError("Range expects exactly 3 inputs and 1 output");
    return false;
  }
  Value start = lookupMappedValue(node.input(0), valueMap);
  Value limit = lookupMappedValue(node.input(1), valueMap);
  Value delta = lookupMappedValue(node.input(2), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!start || !limit || !delta || !resultType) {
    setError("Range operands or result type missing");
    return false;
  }
  auto rangeOp = builder->create<RangeOp>(loc, resultType, start, limit, delta);
  valueMap[node.output(0)] = rangeOp.getResult();
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

  // gawee.slice expects all four index tensors explicitly:
  //   starts, ends, axes, steps
  // ONNX Slice can omit axes and steps, so the emitter may need to synthesize
  // those missing tensors before it can build the gawee op.
  auto makeI64TensorConst = [&](ArrayRef<int64_t> values) -> Value {
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(values.size())},
                              builder->getIntegerType(64));
    auto attr = DenseElementsAttr::get(tensorType, values);
    return arith::ConstantOp::create(*builder, loc, tensorType, attr);
  };

  // When ONNX omits axes/steps, their lengths must match the number of slice
  // entries. The easiest source for that count is the rank-1 length of
  // `starts`. If `starts` is not statically shaped, we cannot manufacture a
  // well-formed default tensor here.
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
    // ONNX default: if axes is absent, slice along axes [0, 1, 2, ...].
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
    // ONNX default: if steps is absent, every slice step is 1.
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

bool ONNXMLIREmitter::emitSin(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Sin expects exactly 1 input and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !resultType) {
    setError("Sin input or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  valueMap[node.output(0)] = createUnaryElementwise(
      *builder, loc, resultType, input, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value arg) {
        return Value(math::SinOp::create(nestedBuilder, bodyLoc, arg));
      });
  return true;
}

bool ONNXMLIREmitter::emitCos(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("Cos expects exactly 1 input and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !resultType) {
    setError("Cos input or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  valueMap[node.output(0)] = createUnaryElementwise(
      *builder, loc, resultType, input, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value arg) {
        return Value(math::CosOp::create(nestedBuilder, bodyLoc, arg));
      });
  return true;
}

bool ONNXMLIREmitter::emitSplit(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() < 1 || node.output_size() < 1) {
    setError("Split expects at least 1 input and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  if (!input) {
    setError("Split input missing");
    return false;
  }
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    setError("Split expects ranked input");
    return false;
  }
  int64_t axis = getI64AttrFromNode(node, "axis", 0);
  if (axis < 0)
    axis += inputType.getRank();

  SmallVector<int64_t> splitSizes;
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = i64TensorLiterals.find(node.input(1));
    if (it == i64TensorLiterals.end()) {
      setError("Split sizes must be constant integer tensor");
      return false;
    }
    splitSizes = it->second;
  } else {
    for (int i = 0; i < node.output_size(); ++i) {
      auto resultType = lookupTensorType(node.output(i), tensorTypes);
      splitSizes.push_back(resultType.getShape()[axis]);
    }
  }
  SmallVector<Type> resultTypes;
  resultTypes.reserve(node.output_size());
  for (int i = 0; i < node.output_size(); ++i)
    resultTypes.push_back(lookupTensorType(node.output(i), tensorTypes));
  auto splitOp = builder->create<SplitOp>(
      loc, TypeRange(resultTypes), input,
      builder->getDenseI64ArrayAttr(splitSizes),
      builder->getI64IntegerAttr(axis));
  for (int i = 0; i < node.output_size(); ++i)
    valueMap[node.output(i)] = splitOp->getResult(i);
  return true;
}

bool ONNXMLIREmitter::emitTile(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Tile expects exactly 2 inputs and 1 output");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  Value repeats = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!input || !repeats || !resultType) {
    setError("Tile operands or result type missing");
    return false;
  }
  auto repeatValues = getConstantI64Values(repeats);
  if (failed(repeatValues)) {
    setError("Tile repeats must be constant integer tensor");
    return false;
  }
  auto tileOp = builder->create<TileOp>(loc, resultType, input, repeats);
  valueMap[node.output(0)] = tileOp.getResult();
  return true;
}

bool ONNXMLIREmitter::emitConstant(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.output_size() != 1) {
    setError("Constant expects exactly 1 output");
    return false;
  }
  for (const auto &attr : node.attribute()) {
    if (attr.name() != "value" || !attr.has_t())
      continue;
    DenseElementsAttr dense = buildDenseElementsAttr(*builder, attr.t(), onnxDirectory);
    if (!dense) {
      std::string shapeText = "[";
      for (int i = 0; i < attr.t().dims_size(); ++i) {
        if (i)
          shapeText += ",";
        shapeText += std::to_string(attr.t().dims(i));
      }
      shapeText += "]";
      setError("Unsupported Constant tensor payload: name=" + node.name() +
               " dtype=" + std::to_string(attr.t().data_type()) +
               " shape=" + shapeText);
      return false;
    }
    Value constant = arith::ConstantOp::create(*builder, loc, dense.getType(), dense);
    valueMap[node.output(0)] = constant;
    SmallVector<int64_t> i64Values;
    if (extractI64TensorLiteral(attr.t(), i64Values))
      i64TensorLiterals[node.output(0)] = i64Values;
    tensorTypes[node.output(0)] = cast<RankedTensorType>(dense.getType());
    return true;
  }
  setError("Constant node without tensor value is unsupported");
  return false;
}

bool ONNXMLIREmitter::emitConstantOfShape(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 1 || node.output_size() != 1) {
    setError("ConstantOfShape expects exactly 1 input and 1 output");
    return false;
  }
  Value shape = lookupMappedValue(node.input(0), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!shape || !resultType) {
    setError("ConstantOfShape shape or result type missing");
    return false;
  }
  SmallVector<Value> dynamicDims = materializeDynamicDims(
      *builder, loc, resultType,
      [&](int64_t dim) {
        Value idx = arith::ConstantOp::create(*builder, loc, builder->getIndexAttr(dim));
        SmallVector<Value> positions{idx};
        return extractTensorScalarAsIndex(*builder, loc, shape, positions);
      });
  Attribute fillAttr = builder->getZeroAttr(resultType.getElementType());
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "value" && attr.has_t()) {
      DenseElementsAttr dense = buildDenseElementsAttr(*builder, attr.t(), onnxDirectory);
      if (!dense) {
        setError("Unsupported ConstantOfShape fill tensor");
        return false;
      }
      if (dense.getNumElements() != 1) {
        setError("ConstantOfShape fill tensor must be scalar");
        return false;
      }
      if (auto floatDense = dyn_cast<DenseFPElementsAttr>(dense)) {
        fillAttr = builder->getFloatAttr(
            cast<FloatType>(resultType.getElementType()),
            (*floatDense.getValues<APFloat>().begin()).convertToDouble());
      } else if (auto intDense = dyn_cast<DenseIntElementsAttr>(dense)) {
        fillAttr = builder->getIntegerAttr(
            resultType.getElementType(),
            (*intDense.getValues<APInt>().begin()).getSExtValue());
      } else {
        setError("Unsupported ConstantOfShape scalar attribute kind");
        return false;
      }
    }
  }
  Value output = createEmptyTensor(*builder, loc, resultType, dynamicDims);
  Value scalar = arith::ConstantOp::create(*builder, loc, cast<TypedAttr>(fillAttr));
  valueMap[node.output(0)] =
      linalg::FillOp::create(*builder, loc, scalar, output).getResult(0);
  return true;
}

bool ONNXMLIREmitter::emitTopK(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 2) {
    setError("TopK expects exactly 2 inputs and 2 outputs");
    return false;
  }
  Value input = lookupMappedValue(node.input(0), valueMap);
  Value kTensor = lookupMappedValue(node.input(1), valueMap);
  auto valueType = lookupTensorType(node.output(0), tensorTypes);
  auto indexType = lookupTensorType(node.output(1), tensorTypes);
  if (!input || !kTensor || !valueType || !indexType) {
    setError("TopK operands or result types missing");
    return false;
  }
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() != 2) {
    setError("TopK currently supports rank-2 input tensors only");
    return false;
  }
  int64_t axis = getI64AttrFromNode(node, "axis", -1);
  if (axis < 0)
    axis += inputType.getRank();
  if (axis != 1 || getI64AttrFromNode(node, "largest", 1) != 1 ||
      getI64AttrFromNode(node, "sorted", 1) != 1) {
    setError("TopK currently supports axis=-1, largest=1, sorted=1 only");
    return false;
  }
  auto kValues = getConstantI64Values(kTensor);
  if (failed(kValues) || kValues->empty()) {
    setError("TopK requires constant k");
    return false;
  }
  int64_t k = (*kValues)[0];
  Type elemType = inputType.getElementType();
  if (!isa<FloatType>(elemType)) {
    setError("TopK currently supports floating-point inputs only");
    return false;
  }

  SmallVector<Value> valueDynDims = materializeDynamicDims(
      *builder, loc, valueType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  SmallVector<Value> indexDynDims = materializeDynamicDims(
      *builder, loc, indexType,
      [&](int64_t dim) { return tensor::DimOp::create(*builder, loc, input, dim); });
  Value valuesInit = createEmptyTensor(*builder, loc, valueType, valueDynDims);
  Value indicesInit = createEmptyTensor(*builder, loc, indexType, indexDynDims);
  Value negInf = arith::ConstantOp::create(
      *builder, loc, builder->getFloatAttr(cast<FloatType>(elemType),
                                           -std::numeric_limits<double>::infinity()));
  Value zeroI64 = arith::ConstantOp::create(*builder, loc, builder->getI64IntegerAttr(0));
  valuesInit = linalg::FillOp::create(*builder, loc, negInf, valuesInit).getResult(0);
  indicesInit = linalg::FillOp::create(*builder, loc, zeroI64, indicesInit).getResult(0);

  Value c0 = arith::ConstantOp::create(*builder, loc, builder->getIndexAttr(0));
  Value c1 = arith::ConstantOp::create(*builder, loc, builder->getIndexAttr(1));
  Value batchDim = tensor::DimOp::create(*builder, loc, input, 0);
  Value inputLen = tensor::DimOp::create(*builder, loc, input, 1);
  Value kIndex = arith::ConstantOp::create(*builder, loc, builder->getIndexAttr(k));

  auto rowLoop = scf::ForOp::create(
      *builder, loc, c0, batchDim, c1, ValueRange{valuesInit, indicesInit},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value row,
          ValueRange rowState) {
        auto rowValueType = RankedTensorType::get({k}, elemType);
        auto rowIndexType = RankedTensorType::get({k}, rowBuilder.getI64Type());
        Value rowValues = createEmptyTensor(rowBuilder, rowLoc, rowValueType);
        Value rowIndices = createEmptyTensor(rowBuilder, rowLoc, rowIndexType);
        rowValues = linalg::FillOp::create(rowBuilder, rowLoc, negInf, rowValues).getResult(0);
        rowIndices = linalg::FillOp::create(rowBuilder, rowLoc, zeroI64, rowIndices).getResult(0);

        auto colLoop = scf::ForOp::create(
            rowBuilder, rowLoc, c0, inputLen, c1, ValueRange{rowValues, rowIndices},
            [&](OpBuilder &colBuilder, Location colLoc, Value col,
                ValueRange colState) {
              Value value = tensor::ExtractOp::create(colBuilder, colLoc, input,
                                                      ValueRange{row, col});
              Value insertPosInit = kIndex;
              auto scanLoop = scf::ForOp::create(
                  colBuilder, colLoc, c0, kIndex, c1, ValueRange{insertPosInit},
                  [&](OpBuilder &scanBuilder, Location scanLoc, Value j,
                      ValueRange scanState) {
                    Value currentPos = scanState[0];
                    Value hasPos = arith::CmpIOp::create(
                        scanBuilder, scanLoc, arith::CmpIPredicate::ult,
                        currentPos, kIndex);
                    Value existingValue = tensor::ExtractOp::create(
                        scanBuilder, scanLoc, colState[0], ValueRange{j});
                    Value existingIndex = tensor::ExtractOp::create(
                        scanBuilder, scanLoc, colState[1], ValueRange{j});
                    Value colI64 = arith::IndexCastOp::create(
                        scanBuilder, scanLoc, scanBuilder.getI64Type(), col);
                    Value isGreater = arith::CmpFOp::create(
                        scanBuilder, scanLoc, arith::CmpFPredicate::OGT, value,
                        existingValue);
                    Value isEqual = arith::CmpFOp::create(
                        scanBuilder, scanLoc, arith::CmpFPredicate::OEQ, value,
                        existingValue);
                    Value tieBreak = arith::CmpIOp::create(
                        scanBuilder, scanLoc, arith::CmpIPredicate::slt, colI64,
                        existingIndex);
                    Value shouldInsert = arith::OrIOp::create(
                        scanBuilder, scanLoc, isGreater,
                        arith::AndIOp::create(scanBuilder, scanLoc, isEqual, tieBreak));
                    Value finalCond = arith::AndIOp::create(
                        scanBuilder, scanLoc,
                        arith::XOrIOp::create(
                            scanBuilder, scanLoc, hasPos,
                            arith::ConstantOp::create(scanBuilder, scanLoc,
                                                      scanBuilder.getI1Type(),
                                                      scanBuilder.getBoolAttr(true))),
                        shouldInsert);
                    Value nextPos = arith::SelectOp::create(scanBuilder, scanLoc,
                                                            finalCond, j, currentPos);
                    scf::YieldOp::create(scanBuilder, scanLoc, nextPos);
                  });
              Value insertPos = scanLoop.getResult(0);
              auto newValues = tensor::GenerateOp::create(
                  colBuilder, colLoc, rowValueType, ValueRange{},
                  [&](OpBuilder &genBuilder, Location genLoc, ValueRange ivs) {
                    Value pos = ivs[0];
                    Value hasInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::ult, insertPos, kIndex);
                    Value equalsInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::eq, pos, insertPos);
                    Value beforeInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::slt, pos, insertPos);
                    Value prevPos = arith::SubIOp::create(genBuilder, genLoc, pos, c1);
                    Value sourcePos = arith::SelectOp::create(genBuilder, genLoc,
                                                              beforeInsert, pos, prevPos);
                    Value oldValue = tensor::ExtractOp::create(
                        genBuilder, genLoc, colState[0], ValueRange{sourcePos});
                    Value insertedOrShifted = arith::SelectOp::create(
                        genBuilder, genLoc, equalsInsert, value, oldValue);
                    Value outValue = arith::SelectOp::create(
                        genBuilder, genLoc, hasInsert, insertedOrShifted, oldValue);
                    tensor::YieldOp::create(genBuilder, genLoc, outValue);
                  });
              auto newIndices = tensor::GenerateOp::create(
                  colBuilder, colLoc, rowIndexType, ValueRange{},
                  [&](OpBuilder &genBuilder, Location genLoc, ValueRange ivs) {
                    Value pos = ivs[0];
                    Value hasInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::ult, insertPos, kIndex);
                    Value equalsInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::eq, pos, insertPos);
                    Value beforeInsert = arith::CmpIOp::create(
                        genBuilder, genLoc, arith::CmpIPredicate::slt, pos, insertPos);
                    Value prevPos = arith::SubIOp::create(genBuilder, genLoc, pos, c1);
                    Value sourcePos = arith::SelectOp::create(genBuilder, genLoc,
                                                              beforeInsert, pos, prevPos);
                    Value oldIndex = tensor::ExtractOp::create(
                        genBuilder, genLoc, colState[1], ValueRange{sourcePos});
                    Value colI64 = arith::IndexCastOp::create(
                        genBuilder, genLoc, genBuilder.getI64Type(), col);
                    Value insertedOrShifted = arith::SelectOp::create(
                        genBuilder, genLoc, equalsInsert, colI64, oldIndex);
                    Value outIndex = arith::SelectOp::create(
                        genBuilder, genLoc, hasInsert, insertedOrShifted, oldIndex);
                    tensor::YieldOp::create(genBuilder, genLoc, outIndex);
                  });
              scf::YieldOp::create(colBuilder, colLoc,
                                   ValueRange{newValues.getResult(), newIndices.getResult()});
            });

        auto writeLoop = scf::ForOp::create(
            rowBuilder, rowLoc, c0, kIndex, c1, ValueRange{rowState[0], rowState[1]},
            [&](OpBuilder &writeBuilder, Location writeLoc, Value j,
                ValueRange writeState) {
              Value rowValue = tensor::ExtractOp::create(
                  writeBuilder, writeLoc, colLoop.getResult(0), ValueRange{j});
              Value rowIndex = tensor::ExtractOp::create(
                  writeBuilder, writeLoc, colLoop.getResult(1), ValueRange{j});
              Value nextValues = tensor::InsertOp::create(
                  writeBuilder, writeLoc, rowValue, writeState[0], ValueRange{row, j});
              Value nextIndices = tensor::InsertOp::create(
                  writeBuilder, writeLoc, rowIndex, writeState[1], ValueRange{row, j});
              scf::YieldOp::create(writeBuilder, writeLoc,
                                   ValueRange{nextValues, nextIndices});
            });
        scf::YieldOp::create(rowBuilder, rowLoc, writeLoop.getResults());
      });

  valueMap[node.output(0)] = rowLoop.getResult(0);
  valueMap[node.output(1)] = rowLoop.getResult(1);
  return true;
}

bool ONNXMLIREmitter::emitMod(
    const onnx::NodeProto &node,
    llvm::StringMap<Value> &valueMap,
    llvm::StringMap<RankedTensorType> &tensorTypes) {
  auto loc = builder->getUnknownLoc();
  if (node.input_size() != 2 || node.output_size() != 1) {
    setError("Mod expects exactly 2 inputs and 1 output");
    return false;
  }
  Value lhs = lookupMappedValue(node.input(0), valueMap);
  Value rhs = lookupMappedValue(node.input(1), valueMap);
  auto resultType = lookupTensorType(node.output(0), tensorTypes);
  if (!lhs || !rhs || !resultType) {
    setError("Mod operands or result type missing");
    return false;
  }
  Type elemType = cast<RankedTensorType>(lhs.getType()).getElementType();
  SmallVector<Value> dynamicDims =
      materializeBroadcastedDynamicDims(*builder, loc, resultType, lhs, rhs);
  valueMap[node.output(0)] = createBinaryElementwise(
      *builder, loc, resultType, lhs, rhs, dynamicDims,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, Value lhsVal, Value rhsVal) {
        if (isa<FloatType>(elemType))
          return Value(arith::RemFOp::create(nestedBuilder, bodyLoc, lhsVal, rhsVal));
        return Value(arith::RemSIOp::create(nestedBuilder, bodyLoc, lhsVal, rhsVal));
      });
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

  SmallVector<int64_t> shapeValues(resultType.getShape().begin(),
                                   resultType.getShape().end());
  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(shapeValues.size())}, builder->getI64Type());
  auto shapeAttr = DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(shapeValues));
  Value shapeValue = arith::ConstantOp::create(*builder, loc, shapeType, shapeAttr);
  auto reshapeOp = builder->create<ReshapeOp>(loc, resultType, input, shapeValue);
  valueMap[node.output(0)] = reshapeOp.getResult();
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

  // gawee.linear models the common inference form:
  //   Y = X * W^T + B
  // The current Gemm support is intentionally narrow so that Gemm and MatMul
  // can both funnel into that same contract without extra rewrites.
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
    // If ONNX already provides bias, forwarding it is straightforward.
    biasValue = lookupMappedValue(node.input(2), valueMap);
    if (!biasValue) {
      setError("LinearLike bias value not found in emitter environment: " + node.input(2));
      return false;
    }
  } else {
    // gawee.linear always takes a bias operand, so MatMul / bias-less Gemm
    // must be rewritten into "same op, but with an explicit zero bias".
    //
    // We need the output width to build that tensor:
    //   bias shape = [outFeatures]
    // Prefer weight[0] because linear weights are typically [outFeatures,
    // inFeatures]. If that is not statically known, fall back to the last
    // dimension of the result type.
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

    // Build a 1-D tensor<[outFeatures] x elemTy> filled with zeros so the
    // downstream gawee.linear sees the same explicit 3-operand form every time.
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
  onnxDirectory = std::filesystem::path(std::string(onnxPath)).parent_path().string();
  i64TensorLiterals.clear();

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
    case onnx::TensorProto_DataType_BOOL:
      return builder->getI1Type();
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

  auto collectI64TensorLiteral = [&](const onnx::TensorProto &tensor) {
    SmallVector<int64_t> values;
    switch (tensor.data_type()) {
    case onnx::TensorProto_DataType_INT64:
      if (tensor.int64_data_size() > 0) {
        values.reserve(tensor.int64_data_size());
        for (auto value : tensor.int64_data())
          values.push_back(value);
      } else if (!tensor.raw_data().empty()) {
        size_t count = tensor.raw_data().size() / sizeof(int64_t);
        values.reserve(count);
        auto *raw = reinterpret_cast<const int64_t *>(tensor.raw_data().data());
        for (size_t i = 0; i < count; ++i)
          values.push_back(raw[i]);
      }
      break;
    case onnx::TensorProto_DataType_INT32:
      if (tensor.int32_data_size() > 0) {
        values.reserve(tensor.int32_data_size());
        for (auto value : tensor.int32_data())
          values.push_back(value);
      } else if (!tensor.raw_data().empty()) {
        size_t count = tensor.raw_data().size() / sizeof(int32_t);
        values.reserve(count);
        auto *raw = reinterpret_cast<const int32_t *>(tensor.raw_data().data());
        for (size_t i = 0; i < count; ++i)
          values.push_back(raw[i]);
      }
      break;
    default:
      break;
    }

    if (!values.empty())
      i64TensorLiterals[tensor.name()] = values;
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
    collectI64TensorLiteral(initializer);
  }

  // --- Build function signature ------------------------------------------
  // Convention for the scaffold:
  //   - graph inputs become function arguments
  //   - large runtime parameters (e.g. weights) remain function arguments
  //   - small integer initializers used for shape/axes/steps become constants
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
    if (i64TensorLiterals.count(initializer.name()) != 0) {
      continue;
    }
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

  for (const auto &initializer : graph.initializer()) {
    auto literalIt = i64TensorLiterals.find(initializer.name());
    if (literalIt == i64TensorLiterals.end()) {
      continue;
    }

    auto typeIt = tensorTypes.find(initializer.name());
    if (typeIt == tensorTypes.end()) {
      setError("Missing type information for ONNX integer initializer: " +
               initializer.name());
      return nullptr;
    }

    auto tensorType = typeIt->second;
    auto denseAttr = DenseIntElementsAttr::get(tensorType, literalIt->second);
    Value constant = arith::ConstantOp::create(*builder, loc, tensorType, denseAttr);
    valueMap[initializer.name()] = constant;
  }

  // --- Emit nodes ---------------------------------------------------------
  for (const auto &node : graph.node()) {
    if (node.op_type() == "Constant") {
      if (!emitConstant(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "ConstantOfShape") {
      if (!emitConstantOfShape(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
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
    if (node.op_type() == "MatMul") {
      if (!emitMatMul(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Gemm") {
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
    if (node.op_type() == "Pow") {
      if (!emitPow(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Neg") {
      if (!emitNeg(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Equal") {
      if (!emitEqual(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "And") {
      if (!emitAnd(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "LessOrEqual") {
      if (!emitLessOrEqual(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "IsNaN") {
      if (!emitIsNaN(node, valueMap, tensorTypes))
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
    if (node.op_type() == "Gather") {
      if (!emitGather(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "GatherElements") {
      if (!emitGatherElements(node, valueMap, tensorTypes))
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
    if (node.op_type() == "ReduceMax") {
      if (!emitReduceMax(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "ReduceSum") {
      if (!emitReduceSum(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Resize") {
      if (!emitResize(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Reshape") {
      if (!emitReshape(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Range") {
      if (!emitRange(node, valueMap, tensorTypes))
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
    if (node.op_type() == "Sin") {
      if (!emitSin(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Cos") {
      if (!emitCos(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "Split") {
      if (!emitSplit(node, valueMap, tensorTypes))
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
    if (node.op_type() == "Tile") {
      if (!emitTile(node, valueMap, tensorTypes))
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
    if (node.op_type() == "Mod") {
      if (!emitMod(node, valueMap, tensorTypes))
        return nullptr;
      continue;
    }
    if (node.op_type() == "TopK") {
      if (!emitTopK(node, valueMap, tensorTypes))
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
