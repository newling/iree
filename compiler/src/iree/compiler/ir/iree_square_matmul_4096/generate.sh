# This script generates a few files.

#!/bin/bash
set -ex

$IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 input.mlir  --mlir-print-ir-after-all --mlir-print-ir-module-scope --mlir-disable-threading   --iree-hal-dump-executable-files-to=${PWD}  -o abc.vmfb > after_all.mlir 2>&1

# Get the MLIR changes using my script
echo nope > empty.mlir
generate-ir-diff empty.mlir after_all.mlir 1 > changes.mlir

# Capture the one file ending with .optimized.ll in current dir
shopt -s nullglob
files=(*.optimized.ll)
shopt -u nullglob
if (( ${#files[@]} != 1 )); then
  echo "Error: expected exactly one *.optimized.ll file in current directory" >&2
  exit 1
fi
OPTIMIZED_LL=${files[0]}


# Get the llc changes:
$LLVM_BUILD/bin/llc  -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942  ${OPTIMIZED_LL} --print-changed=diff --debug-pass-manager  -o second_run.s > llc_changes.s 2>&1
