// This is the output of generate3.sh (with canonicalize and lifting the lowering config up).

#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
#config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 64], [0, 1, 2]], partial_reduction = [0, 1, 2048], subgroup_basis = [[1, 1, 8], [0, 1, 2]], thread = [0, 1, 4], workgroup = [1, 0, 0]}>
module {
  func.func @rank_3_to_rank_1_dispatch_0_reduction_20x1024x2048_f32() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x20x2048xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 20, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x20x2048xf32>> -> tensor<1024x20x2048xf32>
    %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [20], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xf32>> -> tensor<20xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%2 : tensor<1024x20x2048xf32>) outs(%3 : tensor<20xf32>) attrs =  {lowering_config = #config
    } {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<20xf32>
    iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0], sizes = [20], strides = [1] : tensor<20xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xf32>>
    return
  }
}
