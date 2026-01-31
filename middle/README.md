# Gawee Runtime

C++ parser for Gawee IR JSON format.

## Structure

```
runtime/
├── CMakeLists.txt          # Build configuration
├── include/gawee/
│   ├── Graph.h             # Graph data structures
│   └── Parser.h            # Parser interface
├── src/
│   ├── Graph.cpp           # Graph implementation
│   ├── Parser.cpp          # JSON parser implementation
│   └── main.cpp            # Example usage
└── third_party/
    └── json.hpp            # nlohmann/json library
```

## Build

```bash
cd runtime
mkdir build && cd build
cmake ..
make
```

## Run

```bash
# From build directory
./gawee_parser ../../jsondata/graph.json
```

## Key Concepts

### Graph Structure
- **Value**: Tensor metadata (shape, dtype) flowing between operations
- **Node**: Operation (Conv, ReLU, etc.) with inputs, outputs, and attributes
- **WeightRef**: Reference to binary weight file

### JSON Format
```json
{
  "inputs": ["x"],
  "outputs": ["fc"],
  "values": {
    "x": {"id": "x", "shape": [1, 3, 224, 224], "dtype": "float32"}
  },
  "nodes": [
    {
      "op_type": "Conv",
      "name": "conv1",
      "inputs": ["x"],
      "outputs": ["conv1"],
      "attrs": {
        "kernel_size": [7, 7],
        "weight": {"shape": [64, 3, 7, 7], "path": "weights/conv1.bin"}
      }
    }
  ]
}
```

## Next Steps

1. **MLIR Generation**: Add MLIR dialect and lowering
2. **Execution**: Add runtime execution with weight loading
3. **Optimization**: Add graph optimization passes
