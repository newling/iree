hal.executable public @torch_jit$async_dispatch_524 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>) {
    hal.executable.export public @torch_jit$async_dispatch_524_matmul_1x768x192_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_524_matmul_1x768x192_f32() {
        %cst = arith.constant 5.000000e-01 : f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %cst_1 = arith.constant 1.41421354 : f32
        %cst_2 = arith.constant 0.000000e+00 : f32
        %c1770240 = arith.constant 1770240 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4:2 = util.assume.int
            %2[<umin = 47378304, umax = 47378304, udiv = 47378304>, <umin = 49000320, umax = 49000320, udiv = 49000320>],
            %3[<umin = 67008, umax = 67008, udiv = 67008>, <umin = 61632, umax = 61632, udiv = 61632>]
          : index, index
        %5 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c1770240) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2305x1x192xf32>>
        %6 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%4#0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x768xf32>>
        %7 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%4#1) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x768xf32>>
        %8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x768xf32>>
        %9 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0], sizes = [192, 768], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x768xf32>> -> tensor<192x768xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [1, 768], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x768xf32>> -> tensor<1x768xf32>
        %11 = tensor.empty() : tensor<1x768xf32>
        %12 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 192], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2305x1x192xf32>> -> tensor<1x192xf32>
        %13 = linalg.fill ins(%cst_2 : f32) outs(%11 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %14 = linalg.matmul ins(%12, %9 : tensor<1x192xf32>, tensor<192x768xf32>) outs(%13 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %10 : tensor<1x768xf32>, tensor<1x768xf32>) outs(%11 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %16 = arith.addf %in, %in_3 : f32
          %17 = arith.divf %16, %cst_1 : f32
          %18 = math.erf %17 : f32
          %19 = arith.addf %18, %cst_0 : f32
          %20 = arith.mulf %16, %19 : f32
          %21 = arith.mulf %20, %cst : f32
          linalg.yield %21 : f32
        } -> tensor<1x768xf32>
        iree_tensor_ext.dispatch.tensor.store %15, %8, offsets = [0, 0], sizes = [1, 768], strides = [1, 1] : tensor<1x768xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x768xf32>>
        return
      }
    }
  }
}
