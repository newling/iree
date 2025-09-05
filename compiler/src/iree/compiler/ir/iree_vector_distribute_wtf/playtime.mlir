 // /home/jnewling/workspace/builds/iree_build_0/tools/iree-opt --iree-gpu-test-target=gfx942 --mlir-print-ir-after-all    --iree-codegen-llvmgpu-use-vector-distribution   --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-executable-target))"

func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
  %5 = tensor.empty() : tensor<32000x2xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32000x2xf32>) -> tensor<32000x2xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<32000x4096xf16>, tensor<2x4096xf16>) outs(%6 : tensor<32000x2xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 512], subgroup_basis = [[1, 1, 2], [0, 1, 2]], thread = [0, 0, 4], lane_basis = [[1, 1, 64], [0, 1, 2]], workgroup = [16, 1, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %out, %10 : f32
    linalg.yield %11 : f32
  } -> tensor<32000x2xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 2], strides = [1, 1] : tensor<32000x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf32>>
  return
}


// /home/jnewling/workspace/builds/iree_build_0/tools/iree-opt --iree-gpu-test-target=gfx942 --mlir-print-ir-after-all   --mlir-print-ir-module-scope --mlir-disable-threading --iree-codegen-llvmgpu-use-vector-distribution   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy,func.func(iree-llvmgpu-lower-executable-target))" test2.mlir #  > after_all.mlir 2>&1
