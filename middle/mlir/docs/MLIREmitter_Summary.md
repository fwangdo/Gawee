# MLIREmitter (gawee-translate) - Summary

## What You Need to Understand

### 1. What is an MLIR Translator?

A "translate" tool converts between MLIR and external formats:

```
┌────────────┐    Translator    ┌─────────────┐
│ External   │  ───────────────→│   MLIR      │
│ Format     │                  │   IR        │
│ (JSON)     │                  │ (gawee.mlir)│
└────────────┘                  └─────────────┘
```

Contrast with **opt** (optimizer):
- **opt**: MLIR → MLIR (transform passes)
- **translate**: External ↔ MLIR (format conversion)

### 2. Emitter Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MLIREmitter                            │
├─────────────────────────────────────────────────────────────┤
│  1. Parse JSON        →  Extract graph structure            │
│  2. Create Module     →  Top-level MLIR container           │
│  3. Create Function   →  func.func with signature           │
│  4. Emit Nodes        →  gawee.conv, gawee.relu, gawee.add  │
│  5. Create Return     →  func.return with output values     │
└─────────────────────────────────────────────────────────────┘
```

### 3. Key Concepts

#### Value Mapping (Name → mlir::Value)
The JSON graph uses string names to reference values:
```json
{"inputs": ["conv1"], "outputs": ["relu1"]}
```

The emitter maintains a map:
```cpp
std::unordered_map<std::string, Value> valueMap;

// When creating an op:
valueMap["relu1"] = reluOp.getResult();

// When referencing an input:
Value input = valueMap["conv1"];
```

#### Shape → RankedTensorType
JSON shapes like `[1, 64, 56, 56]` become MLIR types:
```cpp
RankedTensorType parseShape(const llvm::json::Array *shape) {
  SmallVector<int64_t> dims;
  for (const auto &dim : *shape) {
    dims.push_back(*dim.getAsInteger());
  }
  return RankedTensorType::get(dims, Float32Type::get(ctx));
}
```

Result: `tensor<1x64x56x56xf32>`

#### OpBuilder Usage
`OpBuilder` is used to create MLIR operations:
```cpp
// Set where to insert new ops
builder->setInsertionPointToEnd(block);

// Create operations
auto convOp = builder->create<ConvOp>(
    loc,           // Location (for error messages)
    resultType,    // Output type
    input,         // Input value
    weight,        // Weight value
    strides,       // Attribute
    padding,       // Attribute
    dilation       // Attribute
);
```

### 4. JSON Graph Structure

```json
{
  "inputs": ["x"],              // Function argument names
  "outputs": ["out"],           // Return value names
  "values": {                   // Shape metadata for all values
    "x": {"shape": [1,3,224,224], "dtype": "torch.float32"},
    "conv1": {"shape": [1,64,112,112], "dtype": "torch.float32"}
  },
  "nodes": [                    // Operations in topological order
    {
      "op_type": "Conv",
      "inputs": ["x"],
      "outputs": ["conv1"],
      "attrs": {"stride": [2,2], "padding": [3,3], ...}
    }
  ]
}
```

### 5. Emitter Flow (Step by Step)

```
1. Create ModuleOp
   └── Empty container for all ops

2. Build function signature
   ├── Input types from values["x"].shape
   └── Output types from values["out"].shape

3. Create FuncOp with entry block
   └── Block arguments become function parameters

4. Map input names to block arguments
   └── valueMap["x"] = blockArg[0]

5. For each node in nodes[]:
   ├── Look up input values from valueMap
   ├── Create gawee.* op
   └── Map output name to result: valueMap["conv1"] = convOp.getResult()

6. Create return statement
   └── Look up output values and return them
```

### 6. Weights as Function Arguments

**Why not inline constants?**
Using `arith.constant dense<0.0> : tensor<...>` for weights fails during
bufferization because constant tensors can't be converted to memrefs.

**Solution**: Make weights function arguments:
```
1. First pass: Collect all weight tensors from Conv nodes
   └── weightArgs = [(conv1_weight, tensor<16x3x3x3xf32>), ...]

2. Build function signature with weights as extra arguments
   └── forward(input, conv1_weight, conv2_weight, ...) -> output

3. Map weight names to block arguments
   └── valueMap["conv1_weight"] = blockArg[1]

4. In emitConv: look up weight from valueMap (not create constant)
```

Code pattern:
```cpp
// First pass: collect weights
for (const auto &nodeVal : *nodes) {
  if (opType == "Conv") {
    auto weightType = parseShape(weightInfo->getArray("shape"));
    weightArgs.push_back({nodeName + "_weight", weightType});
  }
}

// Add to function signature
for (const auto &[name, type] : weightArgs) {
  inputTypes.push_back(type);
}

// Map to block arguments
for (size_t i = 0; i < weightArgs.size(); ++i) {
  valueMap[weightArgs[i].first] = entryBlock->getArgument(numInputs + i);
}
```

### 7. Generated MLIR

Input JSON:
```json
{"op_type": "Conv", "inputs": ["x"], "outputs": ["conv1"],
 "attrs": {"stride": [1,1], "padding": [1,1], "dilation": [1,1],
           "weight": {"shape": [16,3,3,3]}}}
```

Output MLIR (weights as function arguments):
```mlir
func.func @forward(%arg0: tensor<1x3x8x8xf32>,
                   %arg1: tensor<16x3x3x3xf32>) -> tensor<1x16x8x8xf32> {
  %0 = gawee.conv %arg0, %arg1 {strides = [1,1], padding = [1,1], dilation = [1,1]}
       : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x8x8xf32>
  func.return %0 : tensor<1x16x8x8xf32>
}
```

### 8. LLVM JSON API

MLIR uses LLVM's JSON parser:
```cpp
#include "llvm/Support/JSON.h"

// Parse JSON string
auto jsonOrErr = llvm::json::parse(buffer);

// Access object
const auto *obj = jsonOrErr->getAsObject();

// Access fields
auto str = obj->getString("name");      // Optional<StringRef>
auto arr = obj->getArray("inputs");     // const Array*
auto num = obj->getInteger("stride");   // Optional<int64_t>
auto nested = obj->getObject("attrs");  // const Object*

// Iterate array
for (const auto &elem : *arr) {
  auto s = elem.getAsString();
}
```

### 9. Common Patterns

#### Getting Input Values
```cpp
const auto *inputs = node.getArray("inputs");
auto inputName = (*inputs)[0].getAsString();
Value input = valueMap[inputName->str()];
```

#### Getting Attributes as Arrays
```cpp
SmallVector<int64_t> strides;
if (const auto *arr = attrs->getArray("stride")) {
  for (const auto &v : *arr) {
    strides.push_back(*v.getAsInteger());
  }
}
auto stridesAttr = builder->getDenseI64ArrayAttr(strides);
```

#### Storing Results
```cpp
auto reluOp = builder->create<ReluOp>(loc, resultType, input);
valueMap[outputName->str()] = reluOp.getResult();
```

### 10. Error Handling Pattern

```cpp
bool emitConv(...) {
  const auto *inputs = node.getArray("inputs");
  if (!inputs || inputs->empty()) {
    setError("Conv: missing inputs");
    return false;  // Caller checks return value
  }
  // ... continue
  return true;
}
```

### 11. Usage

```bash
# Build
./build.sh

# Translate JSON to MLIR
./build/gawee-translate test/subset_graph.json

# Save to file
./build/gawee-translate test/subset_graph.json -o output.mlir

# Full pipeline: JSON -> MLIR -> Linalg -> Loops
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --gawee-to-loops

# Full pipeline: JSON -> MLIR -> LLVM dialect
./build/gawee-translate test/subset_graph.json | ./build/gawee-opt --gawee-to-llvm
```

## Key Takeaways

1. **Value mapping** connects JSON string names to MLIR Values
2. **OpBuilder** creates ops at the current insertion point
3. **RankedTensorType** encodes shape + element type
4. **LLVM JSON API** uses Optional returns - always check for validity
5. **Topological order** in nodes[] means inputs are always defined before use
6. **Weights as function arguments** - enables bufferization (constant tensors don't bufferize)
