#!/bin/bash
set -ex

export SKIP_DISPATCH_SPECIFIC_GEN=0
export GENERATE_PRE_DISPATCH_IR=1

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi
OUTPUT_DIR=$1

INPUT_FN=from_rahul_linalg_entry.mlir
VMFB_FN=$OUTPUT_DIR/model.vmfb

# If the output directory does not exist, create it.
mkdir -p $OUTPUT_DIR

BASE_FLAGS="--iree-hal-target-device=hip[0] --iree-hal-executable-debug-level=3 --iree-hip-target=gfx942 --iree-dispatch-creation-target-split-reduction-size=1024 --iree-opt-level=O3"
# BASE_FLAGS="--iree-hal-target-device=hip[0] --iree-hal-executable-debug-level=3 --iree-hip-target=gfx942 --iree-opt-level=O3"

$IREE_BUILD/tools/iree-compile $BASE_FLAGS $INPUT_FN -o $VMFB_FN --iree-hal-dump-executable-sources-to=$OUTPUT_DIR

echo $GENERATE_PRE_DISPATCH_IR

if [ "$GENERATE_PRE_DISPATCH_IR" -eq 1 ]; then
   $IREE_BUILD/tools/iree-compile $BASE_FLAGS $INPUT_FN --mlir-print-ir-after-all --mlir-print-ir-module-scope --mlir-disable-threading --compile-to=executable-configurations > $OUTPUT_DIR/after_all.mlir 2>&1
   echo "none" > $OUTPUT_DIR/empty.mlir
   generate-ir-diff $OUTPUT_DIR/empty.mlir $OUTPUT_DIR/after_all.mlir 1 > $OUTPUT_DIR/diffs.mlir
fi

if [ "$SKIP_DISPATCH_SPECIFIC_GEN" -eq 1 ]; then
    echo "Skipping dispatch specific generation."
    exit 0
fi

# Now, there should be a bunch of files in OUTPUT_DIR which end dispatch_0.mlir'.
# Simplify the names.
COUNT=0
for FILE in $OUTPUT_DIR/*dispatch_*.mlir; do
    mv "$FILE" "$OUTPUT_DIR/dispatch_$COUNT.mlir"
    COUNT=$((COUNT + 1))
done

echo "Number of dispatches is $COUNT"
for ((i=0; i<COUNT; i++)); do
    DISPATCH_DIR=$OUTPUT_DIR/dispatch_$i
    mkdir -p $DISPATCH_DIR
    $IREE_BUILD/tools/iree-compile $BASE_FLAGS --mlir-print-ir-after-all  --mlir-print-ir-module-scope --mlir-disable-threading \
    --iree-hal-dump-executable-files-to=$DISPATCH_DIR \
    $OUTPUT_DIR/dispatch_$i.mlir -o $DISPATCH_DIR/abc.vmfb > $DISPATCH_DIR/after_all.mlir 2>&1
    echo "none" > $DISPATCH_DIR/empty.mlir
    generate-ir-diff $DISPATCH_DIR/empty.mlir $DISPATCH_DIR/after_all.mlir 1 > $DISPATCH_DIR/diffs.mlir
done

# Who uses which pipeline?
set +x
for ((i=0; i<COUNT; i++)); do
    echo "Dispatch $i:"
    if grep -q "reduction" $OUTPUT_DIR/dispatch_$i/diffs.mlir; then
        echo "  contains reduction: yes"
    else
        echo "  contains reduction: no"
    fi

    if grep -q "LLVMGPUVectorDistribute" $OUTPUT_DIR/dispatch_$i/diffs.mlir; then
        echo "  uses LLVMGPUVectorDistribute: yes"
    else
        echo "  uses LLVMGPUVectorDistribute: no"
    fi
done
set -x
