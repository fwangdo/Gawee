// Test: simple elementwise generic with broadcast map
// This should be vectorizable after the filter relaxation.
func.func @bias_add(%input: tensor<1x64x8x8xf32>, %bias: tensor<64xf32>) -> tensor<1x64x8x8xf32> {
  %empty = tensor.empty() : tensor<1x64x8x8xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input, %bias : tensor<1x64x8x8xf32>, tensor<64xf32>)
    outs(%empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %add = arith.addf %in, %b : f32
    linalg.yield %add : f32
  } -> tensor<1x64x8x8xf32>
  return %result : tensor<1x64x8x8xf32>
}
