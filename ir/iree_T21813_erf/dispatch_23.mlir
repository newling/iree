hal.executable public @main_graph$async_dispatch_23 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>) {
    hal.executable.export public @main_graph$async_dispatch_23_batch_matmul_DxDx4096x1024_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_graph$async_dispatch_23_batch_matmul_DxDx4096x1024_f32() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.41421354 : f32
        %cst_1 = arith.constant 1.000000e+00 : f32
        %cst_2 = arith.constant 5.000000e-01 : f32
        %0 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
        %6 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
        %7 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
        %8 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(8) : i32
        %9 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(9) : i32
        %10 = hal.interface.constant.load layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(10) : i32
        %11 = arith.extui %0 : i32 to i64
        %12 = arith.extui %1 : i32 to i64
        %13 = arith.shli %12, %c32_i64 : i64
        %14 = arith.ori %11, %13 : i64
        %15 = arith.index_castui %14 : i64 to index
        %16 = arith.extui %2 : i32 to i64
        %17 = arith.extui %3 : i32 to i64
        %18 = arith.shli %17, %c32_i64 : i64
        %19 = arith.ori %16, %18 : i64
        %20 = arith.index_castui %19 : i64 to index
        %21 = arith.index_castui %4 : i32 to index
        %22 = arith.extui %5 : i32 to i64
        %23 = arith.extui %6 : i32 to i64
        %24 = arith.shli %23, %c32_i64 : i64
        %25 = arith.ori %22, %24 : i64
        %26 = arith.index_castui %25 : i64 to index
        %27 = arith.extui %7 : i32 to i64
        %28 = arith.extui %8 : i32 to i64
        %29 = arith.shli %28, %c32_i64 : i64
        %30 = arith.ori %27, %29 : i64
        %31 = arith.index_castui %30 : i64 to index
        %32 = arith.extui %9 : i32 to i64
        %33 = arith.extui %10 : i32 to i64
        %34 = arith.shli %33, %c32_i64 : i64
        %35 = arith.ori %32, %34 : i64
        %36 = arith.index_castui %35 : i64 to index
        %37:4 = util.assume.int
            %20[<umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>, <umin = 0, umax = 0>],
            %21[<umin = 83910656, umax = 83910656, udiv = 83910656>, <umin = 83939328, umax = 83939328, udiv = 83939328>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83984384, umax = 83984384, udiv = 83984384>, <umin = 83984384, umax = 83984384, udiv = 83984384>, <umin = 84008960, umax = 84008960, udiv = 84008960>, <umin = 83910656, umax = 83910656, udiv = 83910656>, <umin = 83910656, umax = 83910656, udiv = 83910656>, <umin = 84029440, umax = 84029440, udiv = 84029440>, <umin = 84049920, umax = 84049920, udiv = 84049920>, <umin = 84008960, umax = 84008960, udiv = 84008960>, <umin = 83939328, umax = 83939328, udiv = 83939328>, <umin = 83939328, umax = 83939328, udiv = 83939328>, <umin = 83939328, umax = 83939328, udiv = 83939328>, <umin = 83939328, umax = 83939328, udiv = 83939328>, <umin = 88276992, umax = 88276992, udiv = 88276992>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>, <umin = 83959808, umax = 83959808, udiv = 83959808>],
            %31<umin = 0, umax = 9007199254740991>,
            %36<umin = 0, umax = 9007199254740991>
          : index, index, index, index
        %38 = hal.interface.binding.subspan layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%37#1) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>>
        %39 = iree_tensor_ext.dispatch.workload.ordinal %37#2, 0 : index
        %40 = iree_tensor_ext.dispatch.workload.ordinal %37#3, 1 : index
        %41 = hal.interface.binding.subspan layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%15) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x1024xf32>>{%39, %40}
        %42 = hal.interface.binding.subspan layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%37#0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x4096xf32>>{%39}
        %43 = hal.interface.binding.subspan layout(<constants = 11, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%26) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x4096xf32>>{%39, %40}
        %44 = iree_tensor_ext.dispatch.tensor.load %41, offsets = [0, 0, 0], sizes = [%39, %40, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x1024xf32>>{%39, %40} -> tensor<?x?x1024xf32>
        %45 = iree_tensor_ext.dispatch.tensor.load %42, offsets = [0, 0, 0], sizes = [%39, 1024, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x4096xf32>>{%39} -> tensor<?x1024x4096xf32>
        %46 = iree_tensor_ext.dispatch.tensor.load %38, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
        %47 = tensor.empty(%39, %40) : tensor<?x?x4096xf32>
        %48 = linalg.fill ins(%cst : f32) outs(%47 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %49 = linalg.batch_matmul ins(%44, %45 : tensor<?x?x1024xf32>, tensor<?x1024x4096xf32>) outs(%48 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %49 : tensor<4096xf32>, tensor<?x?x4096xf32>) outs(%47 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %51 = arith.addf %in, %in_3 : f32
          %52 = arith.divf %51, %cst_0 : f32
          %53 = math.erf %52 : f32
          %54 = arith.addf %53, %cst_1 : f32
          %55 = arith.mulf %51, %54 : f32
          %56 = arith.mulf %55, %cst_2 : f32
          linalg.yield %56 : f32
        } -> tensor<?x?x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %50, %43, offsets = [0, 0, 0], sizes = [%39, %40, 4096], strides = [1, 1, 1] : tensor<?x?x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x4096xf32>>{%39, %40}
        return
      }
    }
  }
}
