// This is the result of running generate1.sh (and then running --canoncalize and pulling the config to the top).

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
#config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], partial_reduction = [0, 256], subgroup_basis = [[1, 1], [0, 1]], thread = [0, 4], workgroup = [1, 0]}>
// FIZZ: see comment in generate2.sh.
// #config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], partial_reduction = [0, 1024], subgroup_basis = [[1, 1], [0, 1]], thread = [0, 16], workgroup = [1, 0]}>

module {
  func.func @super_simple_reduction_dispatch_0_reduction_75600x5120_f32() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<75600x5120xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<75600xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [75600, 5120], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<75600x5120xf32>> -> tensor<75600x5120xf32>
    %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [75600], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<75600xf32>> -> tensor<75600xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<75600x5120xf32>) outs(%3 : tensor<75600xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<75600xf32>
    iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0], sizes = [75600], strides = [1] : tensor<75600xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<75600xf32>>
    return
  }
}
