#!/bin/bash
set -ex

#Based on rocm-examples/HIP-Basic/assembly_to_executable/README.md

# Note that we can use upstream LLVM's clang for most everything, just not the final linking step it seems.
export LLVMDIR=$LLVM_BUILD
export myhip=$LLVM_BUILD/bin/clang

# --fuse-cuid=none is needed!
echo "Compiling the host-side code in main.hip"
$myhip -c --cuda-host-only -fuse-cuid=none main.hip


echo "Compiling the device-side code to fun.s (bring your own fun.s if you like, and skip this step)"
$myhip -S --cuda-device-only --offload-arch=gfx942 main.hip -o fun.s

echo "Compiling fun.s to fun.o"
$LLVMDIR/bin/clang  -target amdgcn-amd-amdhsa -mcpu=gfx942 fun.s -o fun.o

echo "Performing the offload bundle trick"
$LLVMDIR/bin/clang-offload-bundler -type=o -bundle-align=4096 \
            -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942  \
            -input=/dev/null \
            -input=fun.o \
            -output=offload_bundle.hipfb

echo "Performing the llvm-mc step"
$LLVMDIR/bin/llvm-mc -triple host-x86_64-unknown-linux-gnu -o main_device.o hip_obj_gen.mcin --filetype=obj

#The BASEDIR might not be needed, hipcc is probasbly in PATH.
# I guess this is the linking step?
echo "Creating the final executable"
export BASEDIR=/opt/rocm-6.4.2
$BASEDIR/hipcc -o main main.o main_device.o

echo "Done!"
