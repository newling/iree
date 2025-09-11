// See https://github.com/iree-org/iree/issues/21824#issuecomment-3246578632 for the source of this file.

// -----// IR Dump After AutoInputConversionPipelinePass (iree-auto-input-conversion) //----- //
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
module @module {
  util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.0000067817300193 : f64
    %cst_2 = arith.constant 9.000000e-01 : f64
    %cst_3 = arith.constant 1.000000e-01 : f64
    %cst_4 = arith.constant 1.000000e-05 : f64
    %cst_5 = arith.constant 1.474560e+05 : f64
    %cst_6 = arith.constant 1.474560e+05 : f32
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<128x24x48x384xbf16>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<384xbf16>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<384xbf16>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<384xbf16>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<384xbf16>
    %5 = tensor.empty() : tensor<128x384x24x48xbf16>
    %transposed = linalg.transpose ins(%0 : tensor<128x24x48x384xbf16>) outs(%5 : tensor<128x384x24x48xbf16>) permutation = [0, 3, 1, 2]
    %6 = tensor.empty() : tensor<128x384x24x48xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed : tensor<128x384x24x48xbf16>) outs(%6 : tensor<128x384x24x48xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %46 = arith.extf %in : bf16 to f32
      linalg.yield %46 : f32
    } -> tensor<128x384x24x48xf32>
    %8 = tensor.empty() : tensor<128x384x24x48xf64>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<128x384x24x48xf32>) outs(%8 : tensor<128x384x24x48xf64>) {
    ^bb0(%in: f32, %out: f64):
      %46 = arith.extf %in : f32 to f64
      linalg.yield %46 : f64
    } -> tensor<128x384x24x48xf64>
    %10 = tensor.empty() : tensor<1x384x1x1xf64>
    %11 = linalg.fill ins(%cst : f64) outs(%10 : tensor<1x384x1x1xf64>) -> tensor<1x384x1x1xf64>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%9 : tensor<128x384x24x48xf64>) outs(%11 : tensor<1x384x1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %46 = arith.addf %in, %out : f64
      linalg.yield %46 : f64
    } -> tensor<1x384x1x1xf64>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x384x1x1xf64>) outs(%10 : tensor<1x384x1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %46 = arith.divf %in, %cst_5 : f64
      linalg.yield %46 : f64
    } -> tensor<1x384x1x1xf64>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %13 : tensor<128x384x24x48xf64>, tensor<1x384x1x1xf64>) outs(%8 : tensor<128x384x24x48xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %46 = arith.subf %in, %in_11 : f64
      linalg.yield %46 : f64
    } -> tensor<128x384x24x48xf64>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %14 : tensor<128x384x24x48xf64>, tensor<128x384x24x48xf64>) outs(%8 : tensor<128x384x24x48xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %46 = arith.mulf %in, %in_11 : f64
      linalg.yield %46 : f64
    } -> tensor<128x384x24x48xf64>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%15 : tensor<128x384x24x48xf64>) outs(%11 : tensor<1x384x1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %46 = arith.addf %in, %out : f64
      linalg.yield %46 : f64
    } -> tensor<1x384x1x1xf64>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<1x384x1x1xf64>) outs(%10 : tensor<1x384x1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %46 = arith.divf %in, %cst_5 : f64
      linalg.yield %46 : f64
    } -> tensor<1x384x1x1xf64>
    %18 = tensor.empty() : tensor<1x384x1x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x384x1x1xf64>) outs(%18 : tensor<1x384x1x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %46 = arith.truncf %in : f64 to f32
      linalg.yield %46 : f32
    } -> tensor<1x384x1x1xf32>
    %20 = linalg.fill ins(%cst_0 : f32) outs(%18 : tensor<1x384x1x1xf32>) -> tensor<1x384x1x1xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%7 : tensor<128x384x24x48xf32>) outs(%20 : tensor<1x384x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.addf %in, %out : f32
      linalg.yield %46 : f32
    } -> tensor<1x384x1x1xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<1x384x1x1xf32>) outs(%18 : tensor<1x384x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.divf %in, %cst_6 : f32
      linalg.yield %46 : f32
    } -> tensor<1x384x1x1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19 : tensor<1x384x1x1xf32>) outs(%18 : tensor<1x384x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.truncf %cst_4 : f64 to f32
      %47 = arith.addf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<1x384x1x1xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x384x1x1xf32>) outs(%18 : tensor<1x384x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = math.rsqrt %in : f32
      linalg.yield %46 : f32
    } -> tensor<1x384x1x1xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed, %22 : tensor<128x384x24x48xbf16>, tensor<1x384x1x1xf32>) outs(%6 : tensor<128x384x24x48xf32>) {
    ^bb0(%in: bf16, %in_11: f32, %out: f32):
      %46 = arith.extf %in : bf16 to f32
      %47 = arith.subf %46, %in_11 : f32
      linalg.yield %47 : f32
    } -> tensor<128x384x24x48xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %24 : tensor<128x384x24x48xf32>, tensor<1x384x1x1xf32>) outs(%6 : tensor<128x384x24x48xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %46 = arith.mulf %in, %in_11 : f32
      linalg.yield %46 : f32
    } -> tensor<128x384x24x48xf32>
    %collapsed = tensor.collapse_shape %22 [[0, 1, 2, 3]] : tensor<1x384x1x1xf32> into tensor<384xf32>
    %collapsed_7 = tensor.collapse_shape %24 [[0, 1, 2, 3]] : tensor<1x384x1x1xf32> into tensor<384xf32>
    %27 = tensor.empty() : tensor<384xf32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed : tensor<384xf32>) outs(%27 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.truncf %cst_3 : f64 to f32
      %47 = arith.mulf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<384xf32>
    %29 = tensor.empty() : tensor<384xbf16>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%3 : tensor<384xbf16>) outs(%29 : tensor<384xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %46 = arith.truncf %cst_2 : f64 to bf16
      %47 = arith.mulf %in, %46 : bf16
      linalg.yield %47 : bf16
    } -> tensor<384xbf16>
    %31 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%28, %30 : tensor<384xf32>, tensor<384xbf16>) outs(%27 : tensor<384xf32>) {
    ^bb0(%in: f32, %in_11: bf16, %out: f32):
      %46 = arith.extf %in_11 : bf16 to f32
      %47 = arith.addf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<384xf32>
    %collapsed_8 = tensor.collapse_shape %19 [[0, 1, 2, 3]] : tensor<1x384x1x1xf32> into tensor<384xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_8 : tensor<384xf32>) outs(%27 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.truncf %cst_1 : f64 to f32
      %47 = arith.mulf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<384xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<384xf32>) outs(%27 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.truncf %cst_3 : f64 to f32
      %47 = arith.mulf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<384xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<384xbf16>) outs(%29 : tensor<384xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %46 = arith.truncf %cst_2 : f64 to bf16
      %47 = arith.mulf %in, %46 : bf16
      linalg.yield %47 : bf16
    } -> tensor<384xbf16>
    %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%33, %34 : tensor<384xf32>, tensor<384xbf16>) outs(%27 : tensor<384xf32>) {
    ^bb0(%in: f32, %in_11: bf16, %out: f32):
      %46 = arith.extf %in_11 : bf16 to f32
      %47 = arith.addf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<384xf32>
    %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xbf16> into tensor<384x1x1xbf16>
    %36 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %expanded : tensor<128x384x24x48xf32>, tensor<384x1x1xbf16>) outs(%6 : tensor<128x384x24x48xf32>) {
    ^bb0(%in: f32, %in_11: bf16, %out: f32):
      %46 = arith.extf %in_11 : bf16 to f32
      %47 = arith.mulf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<128x384x24x48xf32>
    %expanded_9 = tensor.expand_shape %2 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xbf16> into tensor<384x1x1xbf16>
    %37 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %expanded_9 : tensor<128x384x24x48xf32>, tensor<384x1x1xbf16>) outs(%6 : tensor<128x384x24x48xf32>) {
    ^bb0(%in: f32, %in_11: bf16, %out: f32):
      %46 = arith.extf %in_11 : bf16 to f32
      %47 = arith.addf %in, %46 : f32
      linalg.yield %47 : f32
    } -> tensor<128x384x24x48xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<128x384x24x48xf32>) outs(%5 : tensor<128x384x24x48xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %46 = arith.truncf %in : f32 to bf16
      linalg.yield %46 : bf16
    } -> tensor<128x384x24x48xbf16>
    %39 = tensor.empty() : tensor<128x24x48x384xbf16>
    %transposed_10 = linalg.transpose ins(%38 : tensor<128x384x24x48xbf16>) outs(%39 : tensor<128x24x48x384xbf16>) permutation = [0, 2, 3, 1]
    %40:5 = hal.tensor.barrier join(%transposed_10, %collapsed, %collapsed_7, %31, %35 : tensor<128x24x48x384xbf16>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) => %arg6 : !hal.fence
    %41 = hal.tensor.export %40#0 : tensor<128x24x48x384xbf16> -> !hal.buffer_view
    %42 = hal.tensor.export %40#1 : tensor<384xf32> -> !hal.buffer_view
    %43 = hal.tensor.export %40#2 : tensor<384xf32> -> !hal.buffer_view
    %44 = hal.tensor.export %40#3 : tensor<384xf32> -> !hal.buffer_view
    %45 = hal.tensor.export %40#4 : tensor<384xf32> -> !hal.buffer_view
    util.return %41, %42, %43, %44, %45 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
  util.func public @foo2(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1:5 = util.call @foo(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view)
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1#0, %1#1, %1#2, %1#3, %1#4 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}
