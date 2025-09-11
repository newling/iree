#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @foo_dispatch_0_reduction_384x1152_f32() {
    %c15 = arith.constant 15 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %thread_id_x = gpu.thread_id  x
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<147456x384xbf16, #gpu.address_space<global>>
    %assume_align = memref.assume_alignment %0, 64 : memref<147456x384xbf16, #gpu.address_space<global>>
    %1 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<147456x384xbf16, #gpu.address_space<global>> to memref<147456x384xbf16, #amdgpu.address_space<fat_raw_buffer>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %2:2 = affine.delinearize_index %workgroup_id_x into (128, 384) : index, index
    %3 = affine.linearize_index disjoint [%2#0, %c0, %c0] by (128, 24, 48) : index
    %4:3 = affine.delinearize_index %thread_id_x into (3, 64) : index, index, index
    %5:2 = affine.delinearize_index %thread_id_x into (64) : index, index
    %6 = arith.cmpi eq, %5#1, %c15 : index
    %7 = arith.cmpi slt, %5#1, %c15 : index
    %8 = arith.cmpi eq, %4#1, %c2 : index
    %9 = arith.cmpi slt, %4#1, %c2 : index
    %14 = vector.broadcast %9 : i1 to vector<8xi1>
    // %10 = arith.select %7, %c8, %c0 : index
    // %11 = arith.select %6, %c8, %10 : index
    // %12 = arith.select %9, %c8, %c0 : index
    // %13 = arith.select %8, %11, %12 : index
    // %14 = vector.create_mask %13 : vector<8xi1>

    %15 = affine.linearize_index disjoint [%4#1, %5#1, %c0] by (3, 64, 8) : index
    %16 = affine.apply #map()[%3, %15]
    %17 = vector.transfer_read %1[%16, %2#1], %cst, %14 {in_bounds = [true], permutation_map = #map1} : memref<147456x384xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
    // %17 = vector.transfer_read %1[%16, %2#1], %cst  {in_bounds = [true], permutation_map = #map1} : memref<147456x384xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
    vector.transfer_write %17, %1[%c0, %c0] {in_bounds = [true], permutation_map = #map1} : vector<8xbf16>, memref<147456x384xbf16, #amdgpu.address_space<fat_raw_buffer>>
    return
  }
}
