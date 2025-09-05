#!/bin/bash
set -ex

# Note that to generate for just the device using '--cuda-device-only'.
echo "Generating LLVMIR"
hipcc  -emit-llvm -S --offload-arch=gfx942 --std=c++17 main.hip -o host_and_device.ll
hipcc  -emit-llvm -S --cuda-device-only --offload-arch=gfx942 --std=c++17 main.hip -o device.ll

# Below, the use of the -g flag was suggested in
# https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
#
# """
# While --save-temps is sufficient to generate relevant ISA source files, adding
# the debug symbols flag -g will further annotate the ISA with the lines of the
# corresponding kernel code.
# """
echo "Generating assembly (asm/s) and an executable program"
hipcc -save-temps=obj -g --offload-arch=gfx942 --std=c++17 main.hip -o ${PWD}/main
