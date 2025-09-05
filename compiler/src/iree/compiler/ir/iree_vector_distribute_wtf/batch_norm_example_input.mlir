// This is related to my ongoing work improving batch norm. This is one of the dispatches.
// Run generate.sh on this to observe the output.

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @batch_norm_reduction_example(%arg0: tensor<1152x384xf32>, %arg1: tensor<384xf32>) -> tensor<384xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<1152x384xf32>) outs(%arg1 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<384xf32>
  return %0 : tensor<384xf32>
}

// Current experiment:
