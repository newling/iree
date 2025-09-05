// -----// IR Dump Before LLVMGPUVectorDistributePass (iree-llvmgpu-vector-distribute) //----- //
func.func @rank_3_to_rank_1_dispatch_0_reduction_20x1024x2048_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant dense<2048> : vector<2048xindex>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<2048xf32>
  %0 = ub.poison : f32
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align = memref.assume_alignment %1, 64 : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>>
  %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>> to memref<1024x20x2048xf32, #amdgpu.address_space<fat_raw_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<20xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align_1 = memref.assume_alignment %3, 64 : memref<20xf32, #hal.descriptor_type<storage_buffer>>
  %4 = amdgpu.fat_raw_buffer_cast %assume_align_1 resetOffset : memref<20xf32, #hal.descriptor_type<storage_buffer>> to memref<20xf32, #amdgpu.address_space<fat_raw_buffer>>
  scf.forall (%arg0) in (20) {
    %subview = memref.subview %4[%arg0] [1] [1] : memref<20xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<1xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %5 = vector.step : vector<2048xindex>
    %6 = arith.cmpi ult, %5, %cst : vector<2048xindex>
    %7 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %cst_0) -> (vector<2048xf32>) {
      %12 = vector.transfer_read %2[%arg1, %arg0, %c0], %0 {in_bounds = [true]} : memref<1024x20x2048xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<2048xf32>
      %13 = iree_vector_ext.to_layout %12 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [8], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [1], thread_strides = [1]>) : vector<2048xf32>
      %14 = iree_vector_ext.to_layout %arg2 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [8], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [1], thread_strides = [1]>) : vector<2048xf32>
      %15 = arith.select %6, %13, %cst_0 : vector<2048xi1>, vector<2048xf32>
      %16 = arith.addf %15, %14 : vector<2048xf32>
      %17 = iree_vector_ext.to_layout %16 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [8], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [1], thread_strides = [1]>) : vector<2048xf32>
      scf.yield %17 : vector<2048xf32>
    }
    %8 = vector.transfer_read %subview_2[], %0 : memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<f32>
    %9 = vector.extract %8[] : f32 from vector<f32>
    %10 = vector.multi_reduction <add>, %7, %9 [0] : vector<2048xf32> to f32
    %11 = vector.broadcast %10 : f32 to vector<f32>
    vector.transfer_write %11, %subview_2[] : vector<f32>, memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %subview_3 = memref.subview %4[%arg0] [1] [1] : memref<20xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    memref.copy %subview_2, %subview_3 : memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return
}

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
module {
  func.func @rank_3_to_rank_1_dispatch_0_reduction_20x1024x2048_f32() attributes {translation_info = #translation} {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %cst_1 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = ub.poison : f32
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x1x4xf32>
    %cst_3 = arith.constant dense<2048> : vector<1x1x4xindex>
    %thread_id_x = gpu.thread_id  x
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>>
    %assume_align = memref.assume_alignment %1, 64 : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>>
    %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<1024x20x2048xf32, #hal.descriptor_type<storage_buffer>> to memref<1024x20x2048xf32, #amdgpu.address_space<fat_raw_buffer>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<20xf32, #hal.descriptor_type<storage_buffer>>
    %assume_align_4 = memref.assume_alignment %3, 64 : memref<20xf32, #hal.descriptor_type<storage_buffer>>
    %4 = amdgpu.fat_raw_buffer_cast %assume_align_4 resetOffset : memref<20xf32, #hal.descriptor_type<storage_buffer>> to memref<20xf32, #amdgpu.address_space<fat_raw_buffer>>
    scf.forall (%arg0) in (20) {
      %5:3 = affine.delinearize_index %thread_id_x into (8, 64) : index, index, index
      %6:2 = affine.delinearize_index %thread_id_x into (64) : index, index
      %7 = arith.muli %5#1, %c256 : index
      %8 = vector.broadcast %7 : index to vector<4xindex>
      %9 = arith.addi %8, %cst_1 : vector<4xindex>
      %10 = arith.muli %6#1, %c4 : index
      %11 = vector.broadcast %10 : index to vector<4xindex>
      %12 = arith.addi %9, %11 : vector<4xindex>
      %13 = vector.shape_cast %12 : vector<4xindex> to vector<1x1x4xindex>
      %14 = arith.cmpi ult, %13, %cst_3 : vector<1x1x4xindex>
      %15 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %cst_2) -> (vector<1x1x4xf32>) {
        %31 = affine.linearize_index disjoint [%5#1, %6#1, %c0] by (8, 64, 4) : index
        %32 = vector.transfer_read %2[%arg1, %arg0, %31], %0 {in_bounds = [true]} : memref<1024x20x2048xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
        %33 = vector.insert_strided_slice %32, %cst_2 {offsets = [0, 0, 0], strides = [1]} : vector<4xf32> into vector<1x1x4xf32>
        %34 = arith.select %14, %33, %cst_2 : vector<1x1x4xi1>, vector<1x1x4xf32>
        %35 = arith.addf %34, %arg2 : vector<1x1x4xf32>
        scf.yield %35 : vector<1x1x4xf32>
      }
      %16 = vector.transfer_read %4[%arg0], %0 : memref<20xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<f32>
      %17 = vector.extract %16[] : f32 from vector<f32>
      %18 = vector.multi_reduction <add>, %15, %cst_0 [0, 1, 2] : vector<1x1x4xf32> to f32
      %19 = gpu.subgroup_reduce  add %18 cluster(size = 64) : (f32) -> f32
      %20 = vector.insert %19, %cst [0] : f32 into vector<1xf32>
      %alloc = memref.alloc() : memref<10xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      vector.transfer_write %20, %alloc[%5#1] {in_bounds = [true]} : vector<1xf32>, memref<10xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %21:2 = affine.delinearize_index %thread_id_x into (8) : index, index
      %22 = vector.transfer_read %alloc[%21#1], %cst_0 {in_bounds = [true]} : memref<10xf32, #gpu.address_space<workgroup>>, vector<1xf32>
      %23 = vector.extract %22[0] : f32 from vector<1xf32>
      %24 = arith.addf %23, %cst_0 : f32
      %25 = gpu.subgroup_reduce  add %24 cluster(size = 8) : (f32) -> f32
      %26 = vector.insert %25, %cst [0] : f32 into vector<1xf32>
      %27 = vector.broadcast %17 : f32 to vector<1xf32>
      %28 = arith.addf %26, %27 : vector<1xf32>
      %29 = vector.extract %28[0] : f32 from vector<1xf32>
      %30 = vector.broadcast %29 : f32 to vector<f32>
      vector.transfer_write %30, %4[%arg0] : vector<f32>, memref<20xf32, #amdgpu.address_space<fat_raw_buffer>>
    } {mapping = [#iree_codegen.workgroup_mapping<x>]}
    return
  }
}
