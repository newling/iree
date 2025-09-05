#!/bin/bash

OUTPUT_DIR=./artifacts
VMFB_FN=./artifacts/model.vmfb

# OUTPUT_FN is passed in as first arg. Check there is exactly one arg, and set OUTPUT_FN
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 OUTPUT_FN"
    exit 1
fi
OUTPUT_FN=$1

export TRACY_NO_EXIT=1

echo "Entering benchmark module block (run async)"
(
$IREE_BUILD/tools/iree-benchmark-module \
  --module=$VMFB_FN \
  --device=hip \
  --function=foo2 \
  --input=128x24x48x384xbf16=1 --input=384xbf16 --input=384xbf16 --input=384xbf16 --input=384xbf16
) & BENCH_PID=$!

echo "Sleeping"
sleep 1
echo "Capturing tracy"
$IREE_BUILD/tracy/iree-tracy-capture -o $OUTPUT_DIR/$OUTPUT_FN
echo "Done"
