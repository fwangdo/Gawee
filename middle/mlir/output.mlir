#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  func.func @test_relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.000000e+00 : f32
      %2 = arith.maximumf %in, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  func.func @test_conv(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<1x16x30x30xf32> {
    %0 = tensor.empty() : tensor<1x16x30x30xf32>
    %1 = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>) outs(%0 : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
    return %1 : tensor<1x16x30x30xf32>
  }
}

