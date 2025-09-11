#!/bin/bash
set -ex

# Compile MLIR to VMFB
$IREE_BUILD/tools/iree-compile --iree-hal-target-device=hip[0] \
  --iree-hip-target=gfx942 \
  input.mlir -o abc.vmfb

# Generate inputs, but only if arg0.npy does not already exist
if [ ! -f arg0.npy ]; then
  python3 create_inputs.py 32x32xf16 32x32xf16 32x32xf32
fi

# Generate expected output, but only if expected0.npy does not already exist
if [ ! -f expected0.npy ]; then
  python3 create_expected.py arg0.npy arg1.npy arg2.npy
fi

# Run the module (first run without expected outputs to print and see)
$IREE_BUILD/tools/iree-run-module \
  --device=hip \
  --module=abc.vmfb \
  --input=@arg0.npy --input=@arg1.npy --input=@arg2.npy

# Run the module with expected outputs
$IREE_BUILD/tools/iree-run-module \
  --device=hip \
  --module=abc.vmfb \
  --input=@arg0.npy --input=@arg1.npy --input=@arg2.npy \
  --expected_output=@expected0.npy
