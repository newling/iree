#!/bin/bash

echo "Make sure the iree build is set up correctly for tracy!"


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 OUTPUT_DIR"
    exit 1
fi

OUTPUT_DIR=$1
mkdir -p $OUTPUT_DIR

echo "The output directory is: $OUTPUT_DIR"
VMFB_FN=$OUTPUT_DIR/model.vmfb
OUTPUT_FN=$OUTPUT_DIR/trace.tracy

echo "The output file is: $OUTPUT_FN"

# Assert that the VMFB file exists
if [ ! -f $VMFB_FN ]; then
    echo "Error: $VMFB_FN does not exist."
    exit 1
fi

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
$IREE_BUILD/tracy/iree-tracy-capture -o $OUTPUT_FN
echo "Done"
