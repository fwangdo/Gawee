//===----------------------------------------------------------------------===//
// Extension: Add Lowering Patterns for New Ops
//===----------------------------------------------------------------------===//
//
// GOAL: Lower MaxPool and BatchNorm from Gawee dialect to Linalg dialect.
//
// REFERENCE: Look at existing patterns in lib/Conversion/GaweeToLinalg.cpp
//
// After implementing, add patterns to the pass and rebuild.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO 1: MaxPool Lowering
//===----------------------------------------------------------------------===//
//
// Target: linalg.pooling_nchw_max
//
// This is a named Linalg op (like linalg.conv_2d_nchw_fchw).
//
// Steps:
//   1. Get input from adaptor
//   2. Get attributes (kernel_size, strides, padding)
//   3. Create output tensor with tensor.empty()
//      - Calculate output shape: (H - K) / S + 1
//   4. Initialize output with -infinity (min float value)
//      - Use linalg.fill with -std::numeric_limits<float>::infinity()
//      - Or use arith.constant with FloatAttr::get(..., -INFINITY)
//   5. Create linalg.pooling_nchw_max
//   6. Replace original op
//
// struct MaxPoolOpLowering : public OpConversionPattern<gawee::MaxPoolOp> {
//   // YOUR IMPLEMENTATION HERE
//   //
//   // Key differences from ConvOpLowering:
//   //   - MaxPool needs -infinity init (not zero)
//   //   - Different Linalg op: linalg.pooling_nchw_max
//   //   - Kernel shape comes from attribute, not a weight tensor
//   //
//   // HINT: Check MLIR docs for linalg.pooling_nchw_max signature
// };
//

//===----------------------------------------------------------------------===//
// TODO 2: BatchNorm Lowering
//===----------------------------------------------------------------------===//
//
// Target: linalg.generic (custom elementwise computation)
//
// BatchNorm formula: output = gamma * (input - mean) / sqrt(variance + eps) + beta
//
// This is NOT a named Linalg op, so use linalg.generic like ReluOpLowering.
//
// Challenge: gamma, beta, mean, variance have shape [C] but input has [N,C,H,W]
//            Need to broadcast [C] to [N,C,H,W]
//
// Approach 1 (Simple): Use linalg.broadcast first, then linalg.generic
// Approach 2 (Direct): Use affine maps in linalg.generic for broadcasting
//
// For Approach 2, indexing maps would be:
//   - input:    (n, c, h, w) -> (n, c, h, w)  // identity
//   - gamma:    (n, c, h, w) -> (c)           // broadcast from [C]
//   - beta:     (n, c, h, w) -> (c)           // broadcast from [C]
//   - mean:     (n, c, h, w) -> (c)           // broadcast from [C]
//   - variance: (n, c, h, w) -> (c)           // broadcast from [C]
//   - output:   (n, c, h, w) -> (n, c, h, w)  // identity
//
// struct BatchNormOpLowering : public OpConversionPattern<gawee::BatchNormOp> {
//   // YOUR IMPLEMENTATION HERE
//   //
//   // Steps:
//   //   1. Get all 5 inputs from adaptor
//   //   2. Get epsilon attribute
//   //   3. Create output tensor with tensor.empty()
//   //   4. Build indexing maps (6 maps: 5 inputs + 1 output)
//   //   5. Create linalg.generic with body:
//   //      - %centered = arith.subf %in, %mean
//   //      - %var_eps = arith.addf %var, %eps_const
//   //      - %std = math.sqrt %var_eps
//   //      - %normalized = arith.divf %centered, %std
//   //      - %scaled = arith.mulf %normalized, %gamma
//   //      - %result = arith.addf %scaled, %beta
//   //      - linalg.yield %result
//   //   6. Replace original op
//   //
//   // HINT: You need to include "mlir/Dialect/Math/IR/Math.h" for math.sqrt
//   // HINT: Add MLIRMathDialect to CMakeLists.txt and register in gawee-opt
// };
//

//===----------------------------------------------------------------------===//
// TODO 3: (Optional) ConvBias Lowering
//===----------------------------------------------------------------------===//
//
// If you added bias support to conv, you need to lower it.
//
// Approach:
//   1. Lower conv part normally (linalg.conv_2d_nchw_fchw)
//   2. Add bias using linalg.generic or linalg.broadcast + linalg.add
//
// Bias shape is [C], output shape is [N,C,H,W], so need broadcasting.
//

//===----------------------------------------------------------------------===//
// TODO 4: Register New Patterns
//===----------------------------------------------------------------------===//
//
// In GaweeToLinalgPass::runOnOperation(), add your new patterns:
//
//   patterns.add<MaxPoolOpLowering>(ctx);
//   patterns.add<BatchNormOpLowering>(ctx);
//
// Also update getDependentDialects() if you added new dialects (e.g., math):
//
//   registry.insert<math::MathDialect>();
//

//===----------------------------------------------------------------------===//
// VERIFICATION
//===----------------------------------------------------------------------===//
//
// Test your lowering:
//
//   ./build/gawee-opt --convert-gawee-to-linalg test/extension_test.mlir
//
// Expected: Your gawee.maxpool should become linalg.pooling_nchw_max
// Expected: Your gawee.batchnorm should become linalg.generic with math ops
//
// Full pipeline test:
//
//   ./build/gawee-opt --gawee-to-llvm test/extension_test.mlir
//
