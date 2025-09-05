# This runs compilation on the IR just after the selection strategy has been set. It runs to to the end.
#
# We print 2 IR's here.
# 1) the IR just before vector distribution. Here we can see what each "workgroup" is doing.
# 2) After the pipeline is complete. Here we can see what each "thread" is doing.

$IREE_BUILD/tools/iree-opt --iree-gpu-test-target=gfx942  --mlir-print-ir-before=iree-llvmgpu-vector-distribute  \
      --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-executable-target))" part2.mlir

# What we have learnt looking at this simple reduction example.
#
# 1) The lowering strategy gives
#     - the reduction a workgroup size of 64.
#     - partial_reduction in the reduction is 256. This value (I think) always equals the
#       number of elements the thread processes (4 in this case) multiplied by the workgroup size.
#
#       Each thread has a loop like
#       accumulator = 0
#       for i = 0 : dimSize : partial_reduction (256):
#          load 4 elements
#          accumulate
#       reduce 4 elements
#       reduce across all elements.
#
# For example, see what happens when you change the line FIZZ in part2.mlir! Loads of 16.
#
#
