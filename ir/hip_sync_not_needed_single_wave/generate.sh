#!/bin/bash
set -ex

echo "Generating LLVMIR for the device"
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
# hipcc -save-temps=obj -g --offload-arch=gfx942 --std=c++17 main.hip -o ${PWD}/main
hipcc -save-temps=obj --offload-arch=gfx942 --std=c++17 main.hip -o ${PWD}/main
