// This is general linalg input. to play with as I will.

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @funky_playtime(%arg0: tensor<1152x384xf32>, %arg1: tensor<384xf32>) -> tensor<384xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<1152x384xf32>) outs(%arg1 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<384xf32>
  return %0 : tensor<384xf32>
}

// Interesting findings.

////////////////////////////////////////////////
// tensor<64x384x128xf32> --> tensor<384xf32> //
////////////////////////////////////////////////
//
// Currently results in workgroups size of 64, and
// iterator_types = ["parallel", "reduction", "reduction"]
// partial_reduction = [0, 1, 128], thread = [0, 1, 2], workgroup = [1, 0, 0].
// The workgroup = [1, 0, 0] part is interesting. This means that the parallel dimension
// has workgroup tiling of 1 : 1 workgroup per parallel dimension.
// This means alot of workgroups, each not doing very much work.
