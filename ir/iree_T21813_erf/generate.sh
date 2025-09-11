#!/bin/bash

# See for example
# https://github.com/iree-org/iree/issues/21813
# and
# https://github.com/iree-org/iree/pull/21817
export INPUT_FN=dispatch_small.mlir

$IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 $INPUT_FN --mlir-print-ir-after-all --mlir-print-ir-module-scope --mlir-disable-threading   --iree-hal-dump-executable-files-to=${PWD}  -o abc.vmfb > after_all.mlir 2>&1

# Get the MLIR changes using my script
echo nope > empty.mlir
generate-ir-diff empty.mlir after_all.mlir 1 > changes.mlir
