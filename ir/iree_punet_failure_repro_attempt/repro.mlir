#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>

func.func @foo(%arg0: tensor<8192x640xf32>, %arg1: tensor<8192xf32>, %arg2: tensor<640xf32>) -> tensor<8192x640xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<8192x640xf32>) outs(%arg1 : tensor<8192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.maximumf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<8192xf32>

  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %0 : tensor<8192x640xf32>, tensor<8192xf32>) outs(%arg1 : tensor<8192xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.subf %in, %in_0 : f32
    %5 = arith.mulf %4, %4 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8192xf32>

  %2 = tensor.empty() : tensor<8192x640xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<8192x640xf32>, tensor<8192xf32>, tensor<640xf32>) outs(%2 : tensor<8192x640xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    %5 = arith.addf %4, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<8192x640xf32>
  return %3 : tensor<8192x640xf32>
}
