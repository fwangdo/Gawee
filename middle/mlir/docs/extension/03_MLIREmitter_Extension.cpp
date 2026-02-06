//===----------------------------------------------------------------------===//
// Extension: Add New Ops to MLIREmitter
//===----------------------------------------------------------------------===//
//
// GOAL: Emit MaxPool and BatchNorm ops from JSON graph to Gawee MLIR.
//
// REFERENCE: Look at emitConv, emitRelu, emitAdd in lib/Emit/MLIREmitter.cpp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO 1: Add emitMaxPool Method
//===----------------------------------------------------------------------===//
//
// JSON format for MaxPool (check your graph.json for exact format):
//
//   {
//     "op_type": "MaxPool",
//     "inputs": ["relu1"],
//     "outputs": ["maxpool1"],
//     "attrs": {
//       "kernel_size": [3, 3],
//       "stride": [2, 2],
//       "padding": [1, 1]
//     }
//   }
//
// Implementation steps:
//   1. Get input name from node["inputs"][0]
//   2. Look up input Value from valueMap
//   3. Get output shape from values[outputName]["shape"]
//   4. Extract attributes: kernel_size, stride, padding
//   5. Create gawee.maxpool op using builder
//   6. Store result in valueMap
//
// bool emitMaxPool(const llvm::json::Object &node,
//                  const llvm::json::Object &values) {
//   // YOUR IMPLEMENTATION HERE
//   //
//   // HINT: Very similar to emitRelu, but with attributes like emitConv
//   //
//   // Key differences:
//   //   - Has attributes (kernel_size, stride, padding)
//   //   - Output shape is different from input shape
// }
//

//===----------------------------------------------------------------------===//
// TODO 2: Add emitBatchNorm Method
//===----------------------------------------------------------------------===//
//
// JSON format for BatchNorm:
//
//   {
//     "op_type": "BatchNormalization",  // Note: ONNX uses this name
//     "inputs": ["conv1"],
//     "outputs": ["bn1"],
//     "attrs": {
//       "epsilon": 1e-5,
//       "gamma": {"shape": [64]},    // or provided as separate input
//       "beta": {"shape": [64]},
//       "mean": {"shape": [64]},
//       "variance": {"shape": [64]}
//     }
//   }
//
// Challenge: BatchNorm has 5 tensor inputs (input, gamma, beta, mean, var)
//
// Two approaches for gamma/beta/mean/var:
//
// Approach A: Make them function arguments (like conv weights)
//   - In emit(), first pass: collect BN parameters and add to weightArgs
//   - Map parameter names to block arguments
//   - In emitBatchNorm: look up from valueMap
//
// Approach B: If your JSON has them as separate node inputs
//   - Just look up each input from valueMap
//
// bool emitBatchNorm(const llvm::json::Object &node,
//                    const llvm::json::Object &values) {
//   // YOUR IMPLEMENTATION HERE
//   //
//   // Steps:
//   //   1. Get input tensor from node["inputs"][0]
//   //   2. Get gamma, beta, mean, variance (from attrs or inputs)
//   //   3. Get epsilon attribute
//   //   4. Get output type from values
//   //   5. Create gawee.batchnorm op
//   //   6. Store result in valueMap
// }
//

//===----------------------------------------------------------------------===//
// TODO 3: Update First Pass for BN Parameters (if using Approach A)
//===----------------------------------------------------------------------===//
//
// In MLIREmitter::emit(), the first pass collects weights for Conv.
// You may need to extend this for BatchNorm parameters.
//
// Look for this section in emit():
//
//   // First pass: collect all weight tensors from Conv nodes
//   for (const auto &nodeVal : *nodes) {
//     ...
//     if (opType && *opType == "Conv") {
//       // collect conv weights
//     }
//     // TODO: Add similar handling for BatchNormalization
//     //
//     // if (opType && *opType == "BatchNormalization") {
//     //   // Collect gamma, beta, mean, variance as function arguments
//     //   // weightArgs.push_back({nodeName + "_gamma", gammaType});
//     //   // weightArgs.push_back({nodeName + "_beta", betaType});
//     //   // ... etc
//     // }
//   }
//

//===----------------------------------------------------------------------===//
// TODO 4: Update emitNode Dispatch
//===----------------------------------------------------------------------===//
//
// In MLIREmitter::emitNode(), add dispatch for your new ops:
//
//   if (*opType == "Conv") {
//     return emitConv(node, values);
//   } else if (*opType == "Relu") {
//     return emitRelu(node, values);
//   } else if (*opType == "Add") {
//     return emitAdd(node, values);
//   }
//   // TODO: Add these:
//   // else if (*opType == "MaxPool") {
//   //   return emitMaxPool(node, values);
//   // } else if (*opType == "BatchNormalization") {
//   //   return emitBatchNorm(node, values);
//   // }
//

//===----------------------------------------------------------------------===//
// TODO 5: Update Header File
//===----------------------------------------------------------------------===//
//
// In include/Emit/MLIREmitter.h, add method declarations:
//
//   bool emitMaxPool(const llvm::json::Object &node,
//                    const llvm::json::Object &values);
//   bool emitBatchNorm(const llvm::json::Object &node,
//                      const llvm::json::Object &values);
//

//===----------------------------------------------------------------------===//
// VERIFICATION
//===----------------------------------------------------------------------===//
//
// Create a test JSON file (test/extension_graph.json) with your new ops:
//
//   {
//     "inputs": ["x"],
//     "outputs": ["out"],
//     "values": {
//       "x": {"shape": [1, 64, 32, 32], "dtype": "torch.float32"},
//       "pool1": {"shape": [1, 64, 16, 16], "dtype": "torch.float32"},
//       "out": {"shape": [1, 64, 16, 16], "dtype": "torch.float32"}
//     },
//     "nodes": [
//       {
//         "name": "pool1",
//         "op_type": "MaxPool",
//         "inputs": ["x"],
//         "outputs": ["pool1"],
//         "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]}
//       },
//       {
//         "name": "relu1",
//         "op_type": "Relu",
//         "inputs": ["pool1"],
//         "outputs": ["out"]
//       }
//     ]
//   }
//
// Test:
//   ./build/gawee-translate test/extension_graph.json
//
// Expected output should include:
//   gawee.maxpool %arg0 {kernel_size = [...], ...} : tensor<...> -> tensor<...>
//
