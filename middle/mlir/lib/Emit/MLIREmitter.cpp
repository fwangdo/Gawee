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

  // First pass: collect all weight tensors from Conv nodes
  // These will become function arguments
  for (const auto &nodeVal : *nodes) {
    const auto *node = nodeVal.getAsObject();
    if (!node) continue;
    auto opType = node->getString("op_type");
    if (opType && *opType == "Conv") {
      const auto *attrs = node->getObject("attrs");
      if (attrs) {
        const auto *weightInfo = attrs->getObject("weight");
        if (weightInfo) {
          auto weightType = parseShape(weightInfo->getArray("shape"));
          if (weightType) {
            // Generate unique weight name
            auto nodeName = node->getString("name");
            std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";
            weightArgs.push_back({weightName, weightType});
          }
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
  } else {
    // Skip unsupported ops (e.g., MaxPool, BatchNorm)
    // In partial support mode, we just ignore them
    llvm::errs() << "Warning: Skipping unsupported op: " << *opType << "\n";
    return true;
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

  // Get weight from function arguments
  // Weight was collected in first pass and added as function argument
  auto nodeName = node.getString("name");
  std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";
  Value weight = lookupValue(weightName);
  if (!weight) {
    setError("Conv: weight not found: " + weightName);
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

  // Create gawee.conv op
  auto convOp = builder->create<ConvOp>(
      loc, resultType, input, weight,
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
