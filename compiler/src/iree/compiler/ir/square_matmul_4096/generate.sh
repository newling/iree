# This script generates a few files. 

#!/bin/bash
set -ex

$IREE_BUILD/tools/iree-compile --iree-hal-target-device=amdgpu --iree-hip-target=gfx942 input.mlir  --mlir-print-ir-after-all --mlir-print-ir-module-scope --mlir-disable-threading   --iree-hal-dump-executable-files-to=${PWD}  > abc.vmfb > after_all.mlir 2>&1

echo nope > empty.mlir

generate-ir-diff empty.mlir after_all.mlir 1 > diffs_1.mlir
