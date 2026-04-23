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

#if defined(GAWEE_ENABLE_ONNX_PROTO)
#include <onnx/onnx_pb.h>
#endif

using namespace mlir;
using namespace mlir::gawee;

ONNXMLIREmitter::ONNXMLIREmitter(MLIRContext *context) : ctx(context) {
  builder = std::make_unique<OpBuilder>(ctx);
}

OwningOpRef<ModuleOp> ONNXMLIREmitter::emitFromFile(llvm::StringRef onnxPath) {
#if !defined(GAWEE_ENABLE_ONNX_PROTO)
  // Keep the build path open even before ONNX/protobuf dependencies are wired.
  // Once those dependencies are available, enable GAWEE_ENABLE_ONNX_PROTO and
  // the Conv path below becomes the reference implementation.
  (void)createEmptyModule(onnxPath);
  setError("ONNXMLIREmitter scaffold exists. Enable GAWEE_ENABLE_ONNX_PROTO and link ONNX/protobuf to parse ONNX files.");
  return nullptr;
#else
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

  auto getI64ArrayAttr = [&](const onnx::NodeProto &node, llvm::StringRef attrName,
                             SmallVector<int64_t> defaultValue) -> DenseI64ArrayAttr {
    for (const auto &attr : node.attribute()) {
      if (attr.name() != attrName) {
        continue;
      }

      SmallVector<int64_t> values;
      if (attr.ints_size() > 0) {
        for (auto v : attr.ints()) {
          values.push_back(v);
        }
      } else if (attr.has_i()) {
        // ONNX sometimes stores scalar forms for stride/padding-like attrs.
        // Normalize scalar -> 2D pair because current gawee.conv is 2D-focused.
        values.push_back(attr.i());
        values.push_back(attr.i());
      }
      return builder->getDenseI64ArrayAttr(values);
    }
    return builder->getDenseI64ArrayAttr(defaultValue);
  };

  // --- Collect types from ONNX graph -------------------------------------
  // We need names -> tensor types before creating the function signature.
  std::unordered_map<std::string, RankedTensorType> tensorTypes;
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

  std::unordered_map<std::string, Value> valueMap;
  for (size_t i = 0; i < argNames.size(); ++i) {
    valueMap[argNames[i]] = entryBlock->getArgument(i);
  }

  // --- Emit nodes ---------------------------------------------------------
  // Only Conv is implemented "for real" here.
  // Treat the rest as explicit TODOs so you can extend one op family at a time.
  for (const auto &node : graph.node()) {
    if (node.op_type() != "Conv") {
      setError("ONNXMLIREmitter currently implements Conv only. Unsupported op during emission: " + node.op_type());
      return nullptr;
    }

    if (node.input_size() < 2 || node.output_size() < 1) {
      setError("Conv node is missing required inputs/outputs");
      return nullptr;
    }

    auto inputIt = valueMap.find(node.input(0));
    auto weightIt = valueMap.find(node.input(1));
    if (inputIt == valueMap.end()) {
      setError("Conv input value not found in emitter environment: " + node.input(0));
      return nullptr;
    }
    if (weightIt == valueMap.end()) {
      setError("Conv weight value not found in emitter environment: " + node.input(1));
      return nullptr;
    }

    auto resultTypeIt = tensorTypes.find(node.output(0));
    if (resultTypeIt == tensorTypes.end()) {
      setError("Missing output tensor type for Conv output: " + node.output(0));
      return nullptr;
    }

    Value biasValue;
    if (node.input_size() >= 3 && !node.input(2).empty()) {
      auto biasIt = valueMap.find(node.input(2));
      if (biasIt == valueMap.end()) {
        setError("Conv bias value not found in emitter environment: " + node.input(2));
        return nullptr;
      }
      biasValue = biasIt->second;
    } else {
      // ONNX Conv allows bias to be omitted.
      // Our gawee.conv currently always takes a bias operand, so synthesize
      // a zero bias tensor of shape [Cout]. This keeps gawee dialect emission
      // simple while still accepting common Conv forms from normalized ONNX.
      auto weightType = dyn_cast<RankedTensorType>(weightIt->second.getType());
      if (!weightType || weightType.getRank() < 1 || weightType.isDynamicDim(0)) {
        setError("Cannot synthesize Conv bias because weight output channel is unknown");
        return nullptr;
      }

      int64_t outChannels = weightType.getShape()[0];
      auto biasType = RankedTensorType::get({outChannels}, weightType.getElementType());
      auto zeroAttr = DenseElementsAttr::get(
          biasType, builder->getZeroAttr(weightType.getElementType()));
      biasValue = arith::ConstantOp::create(*builder, loc, biasType, zeroAttr);
    }

    // Map core ONNX Conv attrs into gawee.conv attrs.
    // Defaults follow ONNX Conv defaults for the 2D case.
    auto strides = getI64ArrayAttr(node, "strides", {1, 1});
    auto padding = getI64ArrayAttr(node, "pads", {0, 0});
    auto dilation = getI64ArrayAttr(node, "dilations", {1, 1});

    // NOTE:
    // ONNX pads for 2D Conv is usually [top, left, bottom, right].
    // Current gawee.conv expects a simpler 2-element style inherited from the
    // existing JSON emitter. For the first pass we keep this scaffold minimal:
    // if ONNX gives 4 values, use the leading pair [top, left].
    // This is good enough to understand the plumbing, but should be upgraded to
    // a fully-specified padding contract once more ops are added.
    SmallVector<int64_t> normalizedPadding;
    for (auto v : padding.asArrayRef()) {
      normalizedPadding.push_back(v);
    }
    if (normalizedPadding.size() == 4) {
      normalizedPadding = {normalizedPadding[0], normalizedPadding[1]};
    }

    auto convOp = builder->create<ConvOp>(
        loc,
        resultTypeIt->second,
        inputIt->second,
        weightIt->second,
        biasValue,
        strides,
        builder->getDenseI64ArrayAttr(normalizedPadding),
        dilation);

    valueMap[node.output(0)] = convOp.getResult();
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
#endif
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
