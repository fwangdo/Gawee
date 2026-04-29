// Test file for gawee.matmul lowering.
//
// Run:
//   ./build/gawee-opt --convert-gawee-to-linalg test/matmul_test.mlir
//   ./build/gawee-opt --gawee-to-llvm test/matmul_test.mlir

module {
  func.func @test_matmul_2d(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = "gawee.matmul"(%lhs, %rhs) : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @test_matmul_batched(%lhs: tensor<1x2x3x4xf32>, %rhs: tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32> {
    %0 = "gawee.matmul"(%lhs, %rhs) : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
    return %0 : tensor<1x2x3x5xf32>
  }
}
