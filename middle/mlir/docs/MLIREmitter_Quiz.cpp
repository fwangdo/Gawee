//===----------------------------------------------------------------------===//
// MLIREmitter Quiz
//===----------------------------------------------------------------------===//
//
// Fill in the blanks (marked with ???) to complete the emitter.
// This mirrors the actual structure of lib/Emit/MLIREmitter.cpp
//
// After completing, compare with the real implementation.
//
//===----------------------------------------------------------------------===//

#include "Gawee/GaweeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"
#include <cstddef>
#include <memory>
#include <unordered_map>

using namespace mlir;
using namespace mlir::gawee;

//===----------------------------------------------------------------------===//
// Q1: Class Structure
//===----------------------------------------------------------------------===//
//
// The emitter needs:
//   - MLIRContext pointer
//   - OpBuilder for creating ops
//   - Map from string names to Values
//   - Storage for weight arguments (name + type pairs)
//

class MLIREmitter {
public:
  // constructor. should make context. 
  explicit MLIREmitter(MLIRContext *context);

  // owner of operation reference. 
  OwningOpRef<ModuleOp> emit(const llvm::json::Object &graph);
  // get error.  
  llvm::StringRef getError() const { return errorMsg; }

private:
  MLIRContext *ctx;
  // codegen. 
  std::unique_ptr<OpBuilder> builder;
  std::string errorMsg;

  // Q1a: What type maps value names ("conv1") to MLIR Values?
  std::unordered_map<std::string, Value> valueMap;

  // Q1b: What type stores weight arguments as (name, type) pairs?
  // HINT: We need name (string) and type (RankedTensorType) for each weight
  std::vector<std::pair<std::string, RankedTensorType>> weightArgs;

  // Helper methods
  RankedTensorType parseShape(const llvm::json::Array *shape);
  bool emitNode(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitConv(const llvm::json::Object &node, const llvm::json::Object &values);
  bool emitRelu(const llvm::json::Object &node, const llvm::json::Object &values);
  Value lookupValue(llvm::StringRef name);
  void setError(const llvm::Twine &msg);
};

//===----------------------------------------------------------------------===//
// Q2: Constructor
//===----------------------------------------------------------------------===//

MLIREmitter::MLIREmitter(MLIRContext *context) : ctx(context) {
  // Q2: Create the OpBuilder
  // HINT: OpBuilder needs a context, use std::make_unique
  // Q. ctx's type is ptr of MLIRContext. How can it be OpBuilder?
  builder = std::make_unique<OpBuilder>(ctx);  
}

//===----------------------------------------------------------------------===//
// Q3: parseShape - Convert JSON array to RankedTensorType
//===----------------------------------------------------------------------===//

RankedTensorType MLIREmitter::parseShape(const llvm::json::Array *shape) {
  if (!shape) {
    setError("parseShape: null shape array");
    return nullptr;
  }

  SmallVector<int64_t> dims;
  for (const auto &dim : *shape) {
    // Q3a: Extract integer from JSON value
    if (auto i = dim.getAsInteger()) {
      // * is not de-reference. it's handler for optional type. 
      dims.push_back(*i);
    } else {
      setError("parseShape: invalid dimension");
      return nullptr;
    }
  }

  // Q3b: Create RankedTensorType with f32 element type
  // HINT: Float32Type::get(ctx) gives f32
  return RankedTensorType::get(dims, Float32Type::get(ctx));
}

//===----------------------------------------------------------------------===//
// Q4: lookupValue - Find Value by name
//===----------------------------------------------------------------------===//

Value MLIREmitter::lookupValue(llvm::StringRef name) {
  // Q4: Look up name in valueMap, return nullptr if not found
  auto it = valueMap.find(name.str());
  if (it != valueMap.end()) {
    return it->second;  
  }
  return nullptr;  
}

//===----------------------------------------------------------------------===//
// Q5: emit - Main entry point (IMPORTANT: weights as function arguments)
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> MLIREmitter::emit(const llvm::json::Object &graph) {
  // Clear state
  valueMap.clear();
  errorMsg.clear();
  weightArgs.clear();

  auto loc = builder->getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder->setInsertionPointToEnd(module.getBody());

  // Get graph components
  const auto *inputs = graph.getArray("inputs");
  const auto *outputs = graph.getArray("outputs");
  const auto *values = graph.getObject("values");
  const auto *nodes = graph.getArray("nodes");

  if (!inputs || !outputs || !values || !nodes) {
    setError("Missing required fields");
    return nullptr;
  }

  //-----------------------------------------------------------------------
  // Q5a: FIRST PASS - Collect weights from Conv nodes
  //-----------------------------------------------------------------------
  // Why? Constant tensors can't be bufferized, so weights must be function args
  //
  for (const auto &nodeVal : *nodes) {
    const auto *node = nodeVal.getAsObject();
    if (!node) continue;

    // Q5a-i: Get the op_type string from the node
    auto opType = node->getString("op_type");

    if (opType && *opType == "Conv") {
      const auto *attrs = node->getObject("attrs");
      if (attrs) {
        const auto *weightInfo = attrs->getObject("weight");
        if (weightInfo) {
          auto weightType = parseShape(weightInfo->getArray("shape"));
          if (weightType) {
            // Q5a-ii: Get node name for unique weight naming
            auto nodeName = node->getString("name");
            std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";

            // Q5a-iii: Store weight info for later
            // HINT: weightArgs is a vector of pairs
            weightArgs.push_back({weightName, weightType});
          }
        }
      }
    }
  }

  //-----------------------------------------------------------------------
  // Q5b: Build function signature
  //-----------------------------------------------------------------------
  SmallVector<Type> inputTypes;

  // Add input tensor types
  for (const auto &input : *inputs) {
    auto inputName = input.getAsString();
    const auto *valueInfo = values->getObject(*inputName);
    auto tensorType = parseShape(valueInfo->getArray("shape"));
    // it makes tensor type included in input type. 
    inputTypes.push_back(tensorType);
  }

  // Q5b: Add weight types to function signature
  // HINT: Iterate over weightArgs and add each type
  for (const auto &[name, type] : weightArgs) {
    inputTypes.push_back(type);
  }

  // Build output types (similar pattern, already done for you)
  SmallVector<Type> outputTypes;
  for (const auto &output : *outputs) {
    auto outputName = output.getAsString();
    const auto *valueInfo = values->getObject(*outputName);
    outputTypes.push_back(parseShape(valueInfo->getArray("shape")));
  }

  //-----------------------------------------------------------------------
  // Q5c: Create function and entry block
  //-----------------------------------------------------------------------

  // Q5c-i: Create function type from inputs and outputs
  auto funcType = builder->getFunctionType(inputTypes, outputTypes);

  // Q5c-ii: Create FuncOp named "forward"
  auto func = builder->create<func::FuncOp>(loc, "forward", funcType);

  // Q5c-iii: Add entry block (this creates block arguments)
  // func::FuncOp can generate entry block. 
  auto *entryBlock = func.addEntryBlock();

  builder->setInsertionPointToStart(entryBlock);

  //-----------------------------------------------------------------------
  // Q5d: Map inputs to block arguments
  //-----------------------------------------------------------------------
  for (size_t i = 0; i < inputs->size(); ++i) {
    auto inputName = (*inputs)[i].getAsString();
    // Q5d: Map input name to block argument
    valueMap[inputName->str()] = entryBlock->getArgument(i);
  }

  //-----------------------------------------------------------------------
  // Q5e: Map weights to block arguments
  //-----------------------------------------------------------------------
  size_t numInputs = inputs->size();
  for (size_t i = 0; i < weightArgs.size(); ++i) {
    // Q5e: Map weight name to block argument (offset by numInputs)
    valueMap[weightArgs[i].first] = entryBlock->getArgument(numInputs + i);
  }

  //-----------------------------------------------------------------------
  // Emit nodes and return (done for you)
  //-----------------------------------------------------------------------
  for (const auto &nodeVal : *nodes) {
    const auto *node = nodeVal.getAsObject();
    if (!emitNode(*node, *values)) return nullptr;
  }

  SmallVector<Value> returnValues;
  for (const auto &output : *outputs) {
    returnValues.push_back(lookupValue(*output.getAsString()));
  }
  builder->create<func::ReturnOp>(loc, returnValues);

  return module;
}

//===----------------------------------------------------------------------===//
// Q6: emitNode - Dispatch to specific emitters
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitNode(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  // Q6a: Get op_type from node
  auto opType = node.getString("op_type"); 
  if (!opType) {
    setError("Node missing op_type");
    return false;
  }

  // Q6b: Dispatch based on op type
  if (*opType == "Conv") {
    return emitConv(node, values);
  } else if (*opType == "Relu") {
    return emitRelu(node, values);
  } else if (*opType == "Add") {
    return emitRelu(node, values); 
  } else {
    // Skip unsupported ops
    return true;
  }
}

//===----------------------------------------------------------------------===//
// Q7: emitConv - Emit gawee.conv operation
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitConv(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Get input
  const auto *inputs = node.getArray("inputs");
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) {
    setError("Conv: input not found");
    return false;
  }

  // Get output type
  const auto *outputs = node.getArray("outputs");
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  auto resultType = parseShape(outputInfo->getArray("shape"));

  // Get attributes
  const auto *attrs = node.getObject("attrs");

  //-----------------------------------------------------------------------
  // Q7a: Get weight from valueMap (NOT create constant!)
  //-----------------------------------------------------------------------
  // Why? Weights are function arguments now, not inline constants
  //
  auto nodeName = node.getString("name");
  std::string weightName = nodeName ? nodeName->str() + "_weight" : "weight";

  // Q7a: Look up the weight value
  Value weight = lookupValue(weightName);
  if (!weight) {
    setError("Conv: weight not found");
    return false;
  }

  //-----------------------------------------------------------------------
  // Q7b: Extract attributes and create op
  //-----------------------------------------------------------------------
  auto getArrayAttr = [&](const char *name) -> SmallVector<int64_t> {
    SmallVector<int64_t> result;
    if (const auto *arr = attrs->getArray(name)) {
      for (const auto &v : *arr) {
        if (auto i = v.getAsInteger()) result.push_back(*i);
      }
    }
    return result;
  };

  auto strides = getArrayAttr("stride");
  auto padding = getArrayAttr("padding");
  auto dilation = getArrayAttr("dilation");

  // Q7b: Create the ConvOp
  // HINT: builder->create<ConvOp>(loc, resultType, input, weight, strides, padding, dilation)
  auto convOp = builder->create<ConvOp>(
      loc, resultType, input, weight,
      builder->getDenseI64ArrayAttr(strides),
      builder->getDenseI64ArrayAttr(padding),
      builder->getDenseI64ArrayAttr(dilation));

  // Q7c: Store result in valueMap
  valueMap[outputName->str()] = convOp.getResult();

  return true;
}

//===----------------------------------------------------------------------===//
// Q8: emitRelu - Emit gawee.relu operation
//===----------------------------------------------------------------------===//

bool MLIREmitter::emitRelu(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto loc = builder->getUnknownLoc();

  // Q8a: Get input
  const auto *inputs = node.getArray("inputs");
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);
  if (!input) return false;

  // Q8b: Get output type
  const auto *outputs = node.getArray("outputs");
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  auto resultType = parseShape(outputInfo->getArray("shape"));

  // Q8c: Create ReluOp
  auto reluOp = builder->create<ReluOp>(loc, resultType, input);

  // Q8d: Store result
  valueMap[outputName->str()] = reluOp.getResult();

  return true;
}

//===----------------------------------------------------------------------===//
// Q9: Conceptual Questions
//===----------------------------------------------------------------------===//
//
// Q9a: Why do we make weights function arguments instead of inline constants?
//      A) MLIR doesn't support constant tensors
//      B) Constant tensors fail one-shot-bufferize (tensor→memref conversion)
//      C) Function arguments are faster
//      D) JSON doesn't support inline weights
//
// Q9b: Why is there a "first pass" before building the function signature?
//      A) To validate the JSON
//      B) To collect weight shapes so they can be added as function arguments
//      C) To count the number of nodes
//      D) It's not necessary
//
// Q9c: What does valueMap["conv1"] = convOp.getResult() accomplish?
//      A) Saves the op to disk
//      B) Allows later nodes to reference this output by name
//      C) Registers the op with MLIR
//      D) Validates the op
//
// Q9d: Why process nodes[] in order?
//      A) JSON requires it
//      B) Nodes are in topological order - inputs defined before use (SSA)
//      C) MLIR requires alphabetical order
//      D) Performance optimization
//

//===----------------------------------------------------------------------===//
// Answer Key
//===----------------------------------------------------------------------===//
/*
Q1a: std::unordered_map<std::string, Value>
Q1b: std::vector<std::pair<std::string, RankedTensorType>>

Q2: std::make_unique<OpBuilder>(ctx)

Q3a: getAsInteger()
Q3b: Float32Type::get(ctx)

Q4: name.str(), it->second, nullptr

Q5a-i: getString, "op_type"
Q5a-ii: getString
Q5a-iii: weightName, weightType
Q5b: type
Q5c-i: getFunctionType
Q5c-ii: "forward"
Q5c-iii: addEntryBlock
Q5d: getArgument
Q5e: getArgument, numInputs

Q6a: getString, "op_type"
Q6b: emitConv, emitRelu

Q7a: lookupValue
Q7b: ConvOp
Q7c: outputName->str(), getResult

Q8a: getArray, "inputs", getAsString, lookupValue
Q8b: getObject, *outputName
Q8c: ReluOp
Q8d: outputName->str(), getResult

Q9a: B - Bufferization fails on constant tensors
Q9b: B - Need weight types before building function signature
Q9c: B - Later nodes look up inputs by name
Q9d: B - Topological order ensures SSA validity
*/
