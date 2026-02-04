// Test file for gawee-opt
// Run: ./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir

module {
  // Test gawee.add -> linalg.add
  func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "gawee.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test gawee.relu -> linalg.generic
  func.func @test_relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "gawee.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test gawee.conv -> linalg.conv_2d_nchw_fchw
  func.func @test_conv(%input: tensor<1x3x32x32xf32>, %weight: tensor<16x3x3x3xf32>) -> tensor<1x16x30x30xf32> {
    %0 = "gawee.conv"(%input, %weight) {
      strides = array<i64: 1, 1>,
      padding = array<i64: 0, 0, 0, 0>,
      dilation = array<i64: 1, 1>
    } : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x30x30xf32>
    return %0 : tensor<1x16x30x30xf32>
  }
}
