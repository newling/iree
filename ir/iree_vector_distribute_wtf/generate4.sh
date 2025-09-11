# Similar idea to generate2.sh

$IREE_BUILD/tools/iree-opt --iree-gpu-test-target=gfx942  --mlir-print-ir-before=iree-llvmgpu-vector-distribute  \
      --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-executable-target))" part4.mlir

# What can we see in this example?
#
# The config suggests that the scf.for over the inner-most reduction dimension will not
# be present, because of the partial_reduction size of 2048 in that dimension (2048
# is the full size of that dimension). Sanity check: 2048 (partial reduction size)
# is the same as the thread size (4) * workgroup size (512).
#
# Second observation is that each workgroup is responsible for a single parallel dimension
# So each workgroup (512 threads) must reduce all 1024*2048 elements in this example.
# Atomics or multiple dispatches needed to distribute the reduction dimension to
# different workgroups.
