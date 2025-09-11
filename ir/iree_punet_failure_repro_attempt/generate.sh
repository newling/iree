#!/bin/bash

# This IR was related to a numerical failure in punet, which in the end turned out to be noise.
# What I did was, I ran the punet model dumping the dispatches to a directory.
# I did this before and after the numerical fauilre.
# I did a diff of the dispathces befreo and after, and focused in on the ones that were different.
# The IR in this directory now is a stripped down version of one of those dispatches.

# We expect 1 input. Confirm that there is 1 input, and give it the name INPUT_FN
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input-fn>"
    exit 1
fi
INPUT_FN=$1


$IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 \
   --mlir-print-ir-after=iree-llvmgpu-select-lowering-strategy \
   --mlir-print-ir-before=iree-llvmgpu-vector-distribute \
   --mlir-print-ir-after=iree-llvmgpu-lower-executable-target \
   --iree-codegen-llvmgpu-use-vector-distribution \
   $INPUT_FN -o /dev/null
