// See generate1.sh
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @super_simple_reduction(%arg0: tensor<75600x5120xf32>, %arg1: tensor<75600xf32>) -> tensor<75600xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<75600x5120xf32>) outs(%arg1 : tensor<75600xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<75600xf32>
  return %0 : tensor<75600xf32>
}
