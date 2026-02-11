//===----------------------------------------------------------------------===//
// MLIREmitter Implementation
//===----------------------------------------------------------------------===//

#include "Emit/MLIREmitter.h"
#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::gawee;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MLIREmitter::MLIREmitter(MLIRContext *context) : ctx(context) {
  builder = std::make_unique<OpBuilder>(ctx);
}

//===----------------------------------------------------------------------===//
// Main emit function
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> MLIREmitter::emit(const llvm::json::Object &graph) {
  // Clear state from previous runs
  valueMap.clear();
  errorMsg.clear();
  weightArgs.clear();

  // Create module
  auto loc = builder->getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder->setInsertionPointToEnd(module.getBody());

  // Get graph components
  const auto *inputs = graph.getArray("inputs");
  const auto *outputs = graph.getArray("outputs");
  const auto *values = graph.getObject("values");
  const auto *nodes = graph.getArray("nodes");

  if (!inputs || !outputs || !values || !nodes) {
    setError("Missing required fields in graph JSON");
    return nullptr;
  }

  // First pass: collect all weight/bias tensors from Conv and MatMul nodes.
  // These become function arguments because they are constant parameters,
  // not computed tensors flowing through the graph.
  for (const auto &nodeVal : *nodes) {
    const auto *node = nodeVal.getAsObject();
    if (!node) continue;
    auto opType = node->getString("op_type");
    const auto *attrs = node->getObject("attrs");
    if (!attrs) continue;
    auto nodeName = node->getString("name");

    if ((opType && *opType == "Conv") || (opType && *opType == "MatMul")) {
      // Collect weight
      const auto *weightInfo = attrs->getObject("weight");
      if (weightInfo) {
        auto weightType = parseShape(weightInfo->getArray("shape"));
        if (weightType) {
          std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";
          weightArgs.push_back({weightName, weightType});
        }
      }
      // Collect bias
      const auto *biasInfo = attrs->getObject("bias");
      if (biasInfo) {
        auto biasType = parseShape(biasInfo->getArray("shape"));
        if (biasType) {
          std::string biasName = nodeName ? nodeName->str() + "_bias" : "bias";
          weightArgs.push_back({biasName, biasType});
        }
      }
    }
  }

  // Build function signature
  // Input types from "values" metadata
  SmallVector<Type> inputTypes;
  for (const auto &input : *inputs) {
    auto inputName = input.getAsString();
    if (!inputName) {
      setError("Invalid input name");
      return nullptr;
    }
    const auto *valueInfo = values->getObject(*inputName);
    if (!valueInfo) {
      setError("Input not found in values: " + *inputName);
      return nullptr;
    }
    const auto *shape = valueInfo->getArray("shape");
    auto tensorType = parseShape(shape);
    if (!tensorType) {
      return nullptr;
    }
    inputTypes.push_back(tensorType);
  }

  // Add weight types to function signature
  for (const auto &[name, type] : weightArgs) {
    inputTypes.push_back(type);
  }

  // Output types from "values" metadata
  SmallVector<Type> outputTypes;
  for (const auto &output : *outputs) {
    auto outputName = output.getAsString();
    if (!outputName) {
      setError("Invalid output name");
      return nullptr;
    }
    const auto *valueInfo = values->getObject(*outputName);
    if (!valueInfo) {
      setError("Output not found in values: " + *outputName);
      return nullptr;
    }
    const auto *shape = valueInfo->getArray("shape");
    auto tensorType = parseShape(shape);
    if (!tensorType) {
      return nullptr;
    }
    outputTypes.push_back(tensorType);
  }

  // Create function
  auto funcType = builder->getFunctionType(inputTypes, outputTypes);
  auto func = builder->create<func::FuncOp>(loc, "forward", funcType);
  auto *entryBlock = func.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);

  // Map input names to block arguments
  for (size_t i = 0; i < inputs->size(); ++i) {
    auto inputName = (*inputs)[i].getAsString();
    valueMap[inputName->str()] = entryBlock->getArgument(i);
  }

  // Map weight arguments to block arguments
  size_t numInputs = inputs->size();
  for (size_t i = 0; i < weightArgs.size(); ++i) {
    valueMap[weightArgs[i].first] = entryBlock->getArgument(numInputs + i);
  }

  // Reset weight argument index for emitting nodes
  weightArgIndex = 0;

  // Emit each node
  for (const auto &nodeVal : *nodes) {
    const auto *node = nodeVal.getAsObject();
    if (!node) {
      setError("Invalid node in nodes array");
      return nullptr;
    }
    if (!emitNode(*node, *values)) {
      // Error already set
      return nullptr;
    }
  }

  // Create return statement
  SmallVector<Value> returnValues;
  for (const auto &output : *outputs) {
    auto outputName = output.getAsString();
    Value v = lookupValue(*outputName);
    if (!v) {
      setError("Output value not found: " + *outputName);
      return nullptr;
    }
    returnValues.push_back(v);
  }
  builder->create<func::ReturnOp>(loc, returnValues);

  return module;
}

//===----------------------------------------------------------------------===//
// Node dispatch
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitNode(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto opType = node.getString("op_type");
  if (!opType) {
    setError("Node missing op_type");
    return false;
  }

  if (*opType == "Conv") {
    return emitConv(node, values);
  } else if (*opType == "Relu") {
    return emitRelu(node, values);
  } else if (*opType == "Add") {
    return emitAdd(node, values);
  } else if (*opType == "MaxPool") {
    return emitMaxPool(node, values);
  } else if (*opType == "AdAvgPool") {
    return emitAdAvgPool(node, values);
  } else if (*opType == "flatten") {
    return emitFlatten(node, values);
  } else if (*opType == "MatMul") {
    return emitLinear(node, values);
  } else {
    setError("Unsupported op type: " + *opType);
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Conv emission
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitConv(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("Conv: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("Conv: input not found: " + *inputName);
    return false;
  }

  // Get output info for result type
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("Conv: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("Conv: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Get attributes
  const auto *attrs = node.getObject("attrs");
  if (!attrs) {
    setError("Conv: missing attrs");
    return false;
  }

  // Get weight and bias from function arguments
  // They were collected in first pass and added as function arguments
  auto nodeName = node.getString("name");
  std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";
  std::string biasName = nodeName ? nodeName->str() + "_bias" : "bias";
  Value weight = lookupValue(weightName);
  Value bias = lookupValue(biasName);
  if (!weight) {
    setError("Conv: weight not found: " + weightName);
    return false;
  }
  if (!bias) {
    setError("Conv: bias not found: " + biasName);
    return false;
  }

  // Extract conv attributes
  auto getArrayAttr = [&](const char *name) -> SmallVector<int64_t> {
    SmallVector<int64_t> result;
    if (const auto *arr = attrs->getArray(name)) {
      for (const auto &v : *arr) {
        if (auto i = v.getAsInteger()) {
          result.push_back(*i);
        }
      }
    }
    return result;
  };

  auto strides = getArrayAttr("stride");
  auto padding = getArrayAttr("padding");
  auto dilation = getArrayAttr("dilation");

  // Create gawee.conv op (input, weight, bias, strides, padding, dilation)
  auto convOp = builder->create<ConvOp>(
      loc, resultType, input, weight, bias,
      builder->getDenseI64ArrayAttr(strides),
      builder->getDenseI64ArrayAttr(padding),
      builder->getDenseI64ArrayAttr(dilation));

  // Map output name to result
  valueMap[outputName->str()] = convOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Relu emission
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitRelu(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("Relu: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("Relu: input not found: " + *inputName);
    return false;
  }

  // Get output info
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("Relu: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("Relu: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Create gawee.relu op
  auto reluOp = builder->create<ReluOp>(loc, resultType, input);

  // Map output name to result
  valueMap[outputName->str()] = reluOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Add emission
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitAdd(const llvm::json::Object &node,
                          const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get inputs (Add has 2 inputs)
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->size() < 2) {
    setError("Add: needs 2 inputs");
    return false;
  }
  auto lhsName = (*inputs)[0].getAsString();
  auto rhsName = (*inputs)[1].getAsString();

  Value lhs = lookupValue(*lhsName);
  Value rhs = lookupValue(*rhsName);
  if (!lhs) {
    setError("Add: lhs not found: " + *lhsName);
    return false;
  }
  if (!rhs) {
    setError("Add: rhs not found: " + *rhsName);
    return false;
  }

  // Get output info
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("Add: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("Add: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Create gawee.add op
  auto addOp = builder->create<AddOp>(loc, resultType, lhs, rhs);

  // Map output name to result
  valueMap[outputName->str()] = addOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Max pooling.  
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitMaxPool(const llvm::json::Object &node,
                          const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  // json returns pointer and it returns nullptr when there is no value. 
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("MaxPool: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("MaxPool: input not found: " + *inputName);
    return false;
  }

  // Get output info for result type
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("MaxPool: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("MaxPool: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Get attributes
  const auto *attrs = node.getObject("attrs");
  if (!attrs) {
    setError("MaxPool: missing attrs");
    return false;
  }

  // FIX: JSON field names use snake_case (kernel_size, ceil_mode),
  //   not camelCase (kernelSize, ceilMode).
  // FIX: MaxPool attrs in graph.json can be scalar int (3) or array ([3,3]).
  //   Must handle both — normalize scalar to 2-element array [v, v].
  auto getI64OrArray = [&](const char *name) -> SmallVector<int64_t> {
    SmallVector<int64_t> result;
    // Try as array first
    if (const auto *arr = attrs->getArray(name)) {
      for (const auto &v : *arr) {
        if (auto i = v.getAsInteger()) result.push_back(*i);
      }
    } else if (auto scalar = attrs->getInteger(name)) {
      // Scalar int → duplicate for H and W
      result.push_back(*scalar);
      result.push_back(*scalar);
    }
    return result;
  };

  auto ceilMode = attrs->getBoolean("ceil_mode");

  auto kernelSize = getI64OrArray("kernel_size");
  auto strides = getI64OrArray("stride");
  auto padding = getI64OrArray("padding");
  auto dilation = getI64OrArray("dilation");

  // Create gawee.max_pool op
  auto maxPoolOp = builder->create<MaxPoolOp>(
      loc, resultType, input,
      builder->getDenseI64ArrayAttr(kernelSize),
      builder->getDenseI64ArrayAttr(strides),
      builder->getDenseI64ArrayAttr(padding),
      builder->getDenseI64ArrayAttr(dilation),
      builder->getBoolAttr(ceilMode.value_or(false)));

  // Map output name to result
  valueMap[outputName->str()] = maxPoolOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Adaptive average pooling.  
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitAdAvgPool(const llvm::json::Object &node,
                          const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("AdAvgPool: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();  
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("AdAvgPool: input not found: " + *inputName);
    return false;
  }

  // Get output info for result type
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("AdAvgPool: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("AdAvgPool: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Get attributes
  const auto *attrs = node.getObject("attrs");
  if (!attrs) {
    setError("AdAvgPool: missing attrs");
    return false;
  }

  // Extract output_size array
  SmallVector<int64_t> outputSize;
  if (const auto *arr = attrs->getArray("output_size")) {
    for (const auto &v : *arr) {
      if (auto i = v.getAsInteger()) outputSize.push_back(*i);
    }
  }

  // FIX: Was MaxPoolOp (copy-paste). Must use AdAvgPoolOp.
  // FIX: JSON field is "output_size", not "outputSize".
  auto adAvgPoolOp = builder->create<AdAvgPoolOp>(
      loc, resultType, input,
      builder->getDenseI64ArrayAttr(outputSize));

  // Map output name to result
  valueMap[outputName->str()] = adAvgPoolOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Flatten Operation.  
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitFlatten(const llvm::json::Object &node,
                          const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("Flatten: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();  
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("Flatten: input not found: " + *inputName);
    return false;
  }

  // Get output info for result type
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("Flatten: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("Flatten: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // Get attributes
  const auto *attrs = node.getObject("attrs");
  if (!attrs) {
    setError("Flatten: missing attrs");
    return false;
  }

  // FIX: Was MaxPoolOp (copy-paste). Must use FlattenOp.
  // FIX: start_dim/end_dim are single ints (I64Attr), not arrays (DenseI64ArrayAttr).
  //   JSON field names: "start_dim", "end_dim" (snake_case).
  auto startDim = attrs->getInteger("start_dim").value_or(1);
  auto endDim = attrs->getInteger("end_dim").value_or(-1);

  // Create gawee.flatten op
  auto flattenOp = builder->create<FlattenOp>(
      loc, resultType, input,
      builder->getI64IntegerAttr(startDim),
      builder->getI64IntegerAttr(endDim));

  // Map output name to result
  valueMap[outputName->str()] = flattenOp.getResult();
  return true;
}


//===----------------------------------------------------------------------===//
// Linear emission  
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitLinear(const llvm::json::Object &node,
                             const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("Linear: missing inputs");
    return false;
  }
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("Linear: input not found: " + *inputName);
    return false;
  }

  // Get output info for result type
  const auto *outputs = node.getArray("outputs");
  if (!outputs || outputs->empty()) {
    setError("Linear: missing outputs");
    return false;
  }
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  if (!outputInfo) {
    setError("Linear: output not found in values: " + *outputName);
    return false;
  }
  auto resultType = parseShape(outputInfo->getArray("shape"));
  if (!resultType) {
    return false;
  }

  // FIX: Was missing weight and bias. LinearOp needs (input, weight, bias).
  //   Weight/bias are collected in first pass (same pattern as Conv).
  //   JSON op_type is "MatMul" but we emit gawee.linear.
  auto nodeName = node.getString("name");
  std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";
  std::string biasName = nodeName ? nodeName->str() + "_bias" : "bias";
  Value weight = lookupValue(weightName);
  Value bias = lookupValue(biasName);
  if (!weight) {
    setError("Linear: weight not found: " + weightName);
    return false;
  }
  if (!bias) {
    setError("Linear: bias not found: " + biasName);
    return false;
  }
  auto linearOp = builder->create<LinearOp>(
      loc, resultType, input, weight, bias);

  // Map output name to result
  valueMap[outputName->str()] = linearOp.getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

RankedTensorType MLIREmitter::parseShape(const llvm::json::Array *shape) {
  if (!shape) {
    setError("parseShape: null shape array");
    return nullptr;
  }
  SmallVector<int64_t> dims;
  for (const auto &dim : *shape) {
    if (auto i = dim.getAsInteger()) {
      dims.push_back(*i);
    } else {
      setError("parseShape: invalid dimension");
      return nullptr;
    }
  }
  return RankedTensorType::get(dims, Float32Type::get(ctx));
}

Value MLIREmitter::lookupValue(llvm::StringRef name) {
  auto it = valueMap.find(name.str());
  if (it != valueMap.end()) {
    return it->second;
  }
  return nullptr;
}

void MLIREmitter::setError(const llvm::Twine &msg) {
  errorMsg = msg.str();
}
