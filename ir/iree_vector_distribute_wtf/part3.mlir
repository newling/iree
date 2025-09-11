// See generate3.sh
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
func.func @rank_3_to_rank_1(%arg0: tensor<1024x20x2048xf32>, %arg1: tensor<20xf32>) -> tensor<20xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "reduction"]}
  ins(%arg0 : tensor<1024x20x2048xf32>) outs(%arg1 : tensor<20xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<20xf32>
  return %0 : tensor<20xf32>
}


// Side note: changing 1024->1000 and 2048->3000 means that we must use TileAndFuse instead of VectorDistribute.
