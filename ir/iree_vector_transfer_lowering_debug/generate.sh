# This IR accompanies branch 'hoist_if_hacky_version' where
# lowering could be improved in the case where the mask is
# either all true or all false.

$IREE_BUILD/tools/iree-opt  --pass-pipeline='builtin.module(func.func(iree-codegen-vector-transfer-lowering))' input.mlir --debug > debug.mlir 2>&1
