//===----------------------------------------------------------------------===//
// Extension: Add New Ops to Gawee Dialect
//===----------------------------------------------------------------------===//
//
// GOAL: Add MaxPool and BatchNorm operations to the Gawee dialect.
//
// REFERENCE: Look at existing ops in include/Gawee/GaweeOps.td
//
// After implementing, regenerate with: ./build.sh
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO 1: MaxPool2D Operation
//===----------------------------------------------------------------------===//
//
// MaxPool takes an input tensor and returns a smaller tensor by taking
// the maximum value in each pooling window.
//
// Input shape:  [N, C, H, W]
// Output shape: [N, C, H', W']  where H' = (H - kernel) / stride + 1
//
// Required attributes:
//   - kernel_size: array<i64, 2>  (e.g., [2, 2] for 2x2 pooling)
//   - strides: array<i64, 2>      (e.g., [2, 2] for stride 2)
//   - padding: array<i64, 2>      (e.g., [0, 0] for no padding)
//
// HINT: Follow the pattern of Gawee_ConvOp
//
// def Gawee_MaxPoolOp : Gawee_Op<"maxpool", [...]> {
//   // YOUR IMPLEMENTATION HERE
//   // 1. Add summary and description
//   // 2. Define arguments (input tensor + attributes)
//   // 3. Define results (output tensor)
//   // 4. Add assemblyFormat for pretty printing
// }
//

//===----------------------------------------------------------------------===//
// TODO 2: BatchNorm Operation
//===----------------------------------------------------------------------===//
//
// BatchNorm normalizes the input using learned parameters.
//
// Formula: output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
//
// During inference (our case), mean and variance are fixed (learned during training).
//
// Required inputs:
//   - input: tensor (the data to normalize)
//   - gamma: tensor (scale parameter, shape [C])
//   - beta: tensor (shift parameter, shape [C])
//   - mean: tensor (running mean, shape [C])
//   - variance: tensor (running variance, shape [C])
//
// Required attributes:
//   - epsilon: f64 (small constant for numerical stability, e.g., 1e-5)
//
// HINT: This op has 5 tensor inputs, not just 1-2 like conv/relu
//
// def Gawee_BatchNormOp : Gawee_Op<"batchnorm", [...]> {
//   // YOUR IMPLEMENTATION HERE
// }
//

//===----------------------------------------------------------------------===//
// TODO 3: (Optional) Add Bias to Conv
//===----------------------------------------------------------------------===//
//
// Current conv: output = conv(input, weight)
// With bias:    output = conv(input, weight) + bias
//
// Option A: Add optional bias argument to existing ConvOp
// Option B: Create separate ConvBiasOp
//
// If Option A:
//   - Change weight argument to handle optional bias
//   - Use OptionalAttr or make bias a separate optional input
//
// If Option B:
//   - Create new op with input, weight, bias
//   - Simpler but duplicates code
//
// DECISION: Choose one approach and implement
//

//===----------------------------------------------------------------------===//
// VERIFICATION
//===----------------------------------------------------------------------===//
//
// After implementing, test that TableGen generates correctly:
//
//   ./build.sh
//
// Check generated files in include/Gawee/generated/:
//   - GaweeOps.h.inc should contain your new op classes
//   - GaweeOps.cpp.inc should contain implementations
//
// Write a test file (test/extension_test.mlir) with your new ops:
//
//   func.func @test_maxpool(%input: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
//     %0 = gawee.maxpool %input {kernel_size = [2, 2], strides = [2, 2], padding = [0, 0]}
//          : tensor<1x64x32x32xf32> -> tensor<1x64x16x16xf32>
//     return %0 : tensor<1x64x16x16xf32>
//   }
//
