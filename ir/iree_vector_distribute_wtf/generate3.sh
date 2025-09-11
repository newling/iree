 $IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 --mlir-print-ir-after=iree-llvmgpu-select-lowering-strategy part3.mlir  -o /dev/null
