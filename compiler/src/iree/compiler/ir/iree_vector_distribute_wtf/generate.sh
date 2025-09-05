# This accompanies reduction_input,mlir. It'll print 3 IRs critical to understanting the
# lowering the the vector-distribute pipeline.
#
# The flat 'iree-codegen-llvmgpu-use-vector-distribution' encourages the strategy to
# be used, but doesn't force it.

$IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 \
   --mlir-print-ir-after=iree-llvmgpu-select-lowering-strategy \
   --mlir-print-ir-before=iree-llvmgpu-vector-distribute \
   --mlir-print-ir-after=iree-llvmgpu-lower-executable-target \
   --iree-codegen-llvmgpu-use-vector-distribution \
   reduction_input.mlir  -o /dev/null
