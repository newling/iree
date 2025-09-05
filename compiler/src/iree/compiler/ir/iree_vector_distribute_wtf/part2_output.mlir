// -----// IR Dump Before LLVMGPUVectorDistributePass (iree-llvmgpu-vector-distribute) //----- //
func.func @super_simple_reduction_dispatch_0_reduction_75600x5120_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant dense<5120> : vector<256xindex>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<256xf32>
  %0 = ub.poison : f32
  %c256 = arith.constant 256 : index
  %c5120 = arith.constant 5120 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align = memref.assume_alignment %1, 64 : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>>
  %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>> to memref<75600x5120xf32, #amdgpu.address_space<fat_raw_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<75600xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align_1 = memref.assume_alignment %3, 64 : memref<75600xf32, #hal.descriptor_type<storage_buffer>>
  %4 = amdgpu.fat_raw_buffer_cast %assume_align_1 resetOffset : memref<75600xf32, #hal.descriptor_type<storage_buffer>> to memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>>
  scf.forall (%arg0) in (75600) {
    %subview = memref.subview %4[%arg0] [1] [1] : memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<1xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %5 = vector.step : vector<256xindex>
    %6 = arith.cmpi ult, %5, %cst : vector<256xindex>
    %7 = scf.for %arg1 = %c0 to %c5120 step %c256 iter_args(%arg2 = %cst_0) -> (vector<256xf32>) {
      %12 = vector.transfer_read %2[%arg0, %arg1], %0 {in_bounds = [true]} : memref<75600x5120xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<256xf32>
      %13 = iree_vector_ext.to_layout %12 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [0], thread_strides = [1]>) : vector<256xf32>
      %14 = iree_vector_ext.to_layout %arg2 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [0], thread_strides = [1]>) : vector<256xf32>
      %15 = arith.select %6, %13, %cst_0 : vector<256xi1>, vector<256xf32>
      %16 = arith.addf %15, %14 : vector<256xf32>
      %17 = iree_vector_ext.to_layout %16 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1], batch_tile = [1], outer_tile = [1], thread_tile = [64], element_tile = [4], subgroup_strides = [0], thread_strides = [1]>) : vector<256xf32>
      scf.yield %17 : vector<256xf32>
    }
    %8 = vector.transfer_read %subview_2[], %0 : memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<f32>
    %9 = vector.extract %8[] : f32 from vector<f32>
    %10 = vector.multi_reduction <add>, %7, %9 [0] : vector<256xf32> to f32
    %11 = vector.broadcast %10 : f32 to vector<f32>
    vector.transfer_write %11, %subview_2[] : vector<f32>, memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %subview_3 = memref.subview %4[%arg0] [1] [1] : memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    memref.copy %subview_2, %subview_3 : memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, strided<[], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return
}

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
module {
  func.func @super_simple_reduction_dispatch_0_reduction_75600x5120_f32() attributes {translation_info = #translation} {
    %cst = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c5120 = arith.constant 5120 : index
    %c256 = arith.constant 256 : index
    %0 = ub.poison : f32
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x1x4xf32>
    %cst_3 = arith.constant dense<5120> : vector<1x1x4xindex>
    %thread_id_x = gpu.thread_id  x
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>>
    %assume_align = memref.assume_alignment %1, 64 : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>>
    %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<75600x5120xf32, #hal.descriptor_type<storage_buffer>> to memref<75600x5120xf32, #amdgpu.address_space<fat_raw_buffer>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<75600xf32, #hal.descriptor_type<storage_buffer>>
    %assume_align_4 = memref.assume_alignment %3, 64 : memref<75600xf32, #hal.descriptor_type<storage_buffer>>
    %4 = amdgpu.fat_raw_buffer_cast %assume_align_4 resetOffset : memref<75600xf32, #hal.descriptor_type<storage_buffer>> to memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>>
    scf.forall (%arg0) in (75600) {
      %5:2 = affine.delinearize_index %thread_id_x into (64) : index, index
      %6 = arith.muli %5#1, %c4 : index
      %7 = vector.broadcast %6 : index to vector<4xindex>
      %8 = arith.addi %7, %cst : vector<4xindex>
      %9 = vector.shape_cast %8 : vector<4xindex> to vector<1x1x4xindex>
      %10 = arith.cmpi ult, %9, %cst_3 : vector<1x1x4xindex>
      %11 = scf.for %arg1 = %c0 to %c5120 step %c256 iter_args(%arg2 = %cst_2) -> (vector<1x1x4xf32>) {
        %21 = affine.linearize_index [%5#1, %arg1] by (64, 4) : index
        %22 = vector.transfer_read %2[%arg0, %21], %0 {in_bounds = [true]} : memref<75600x5120xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
        %23 = vector.insert_strided_slice %22, %cst_2 {offsets = [0, 0, 0], strides = [1]} : vector<4xf32> into vector<1x1x4xf32>
        %24 = arith.select %10, %23, %cst_2 : vector<1x1x4xi1>, vector<1x1x4xf32>
        %25 = arith.addf %24, %arg2 : vector<1x1x4xf32>
        scf.yield %25 : vector<1x1x4xf32>
      }
      %12 = vector.transfer_read %4[%arg0], %0 : memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<f32>
      %13 = vector.extract %12[] : f32 from vector<f32>
      %14 = vector.multi_reduction <add>, %11, %cst_1 [0, 1, 2] : vector<1x1x4xf32> to f32
      %15 = gpu.subgroup_reduce  add %14 cluster(size = 64) : (f32) -> f32
      %16 = vector.insert %15, %cst_0 [0] : f32 into vector<1xf32>
      %17 = vector.broadcast %13 : f32 to vector<1xf32>
      %18 = arith.addf %16, %17 : vector<1xf32>
      %19 = vector.extract %18[0] : f32 from vector<1xf32>
      %20 = vector.broadcast %19 : f32 to vector<f32>
      vector.transfer_write %20, %4[%arg0] : vector<f32>, memref<75600xf32, #amdgpu.address_space<fat_raw_buffer>>
    } {mapping = [#iree_codegen.workgroup_mapping<x>]}
    return
  }
}
