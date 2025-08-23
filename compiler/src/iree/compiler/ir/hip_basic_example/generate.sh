#!/bin/bash
set -ex

echo "Generating LLVMIR"
hipcc  -emit-llvm -S --offload-arch=gfx942 --std=c++17 main.hip

#  Below, the use of the -g flag was suggested in
#  https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
echo "Generating asm and executable"
hipcc -save-temps=obj -g --offload-arch=gfx942 --std=c++17 main.hip -o ${PWD}/main
