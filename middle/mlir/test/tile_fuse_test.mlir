// Test: tile-and-fuse on conv + bias_add + relu chain.
// Conv2D NCHW_FCHW output: 1x16x8x8, then elementwise bias + relu.
// This is the minimal pattern that tile-and-fuse should handle.
//
// Expected: conv is tiled, bias_add and relu are fused into the tile loop.
// Run: ./build/gawee-opt --gawee-linalg-transform test/tile_fuse_test.mlir

func.func @conv_bias_relu(
    %input: tensor<1x3x10x10xf32>,
    %weight: tensor<16x3x3x3xf32>,
    %bias: tensor<16xf32>) -> tensor<1x16x8x8xf32> {

  // Conv2D NCHW_FCHW: [1,3,10,10] * [16,3,3,3] -> [1,16,8,8]
  %zero = arith.constant 0.0 : f32
  %init_conv = tensor.empty() : tensor<1x16x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init_conv : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  %conv = linalg.conv_2d_nchw_fchw
    ins(%input, %weight : tensor<1x3x10x10xf32>, tensor<16x3x3x3xf32>)
    outs(%fill : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>

  // Bias add: broadcast bias[c] over [n,c,h,w]
  %init_bias = tensor.empty() : tensor<1x16x8x8xf32>
  %bias_add = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv, %bias : tensor<1x16x8x8xf32>, tensor<16xf32>)
    outs(%init_bias : tensor<1x16x8x8xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %add = arith.addf %in, %b : f32
    linalg.yield %add : f32
  } -> tensor<1x16x8x8xf32>

  // ReLU: max(0, x)
  %init_relu = tensor.empty() : tensor<1x16x8x8xf32>
  %relu = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bias_add : tensor<1x16x8x8xf32>)
    outs(%init_relu : tensor<1x16x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c0 = arith.constant 0.0 : f32
    %max = arith.maximumf %in, %c0 : f32
    linalg.yield %max : f32
  } -> tensor<1x16x8x8xf32>

  return %relu : tensor<1x16x8x8xf32>
}
