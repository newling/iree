#!/usr/bin/env bash
set -euo pipefail

# Remove all files in the current directory ending in:
#   .mlir, .ll, .hsaco, .rocmasm, .vmfb
# except for:
#   input.mlir
#   and all files ending in .sh

find . -maxdepth 1 -type f \
  \( -name '*.s' -o  -name '*.mlir' -o -name '*.ll' -o -name '*.hsaco' -o -name '*.rocmasm' -o -name '*.vmfb' \) \
  ! -name '*input.mlir' \
  ! -name 'dispatch*mlir' \
  ! -name 'dispatch*mlir' \
  ! -name '*.sh' \
  -exec rm -v {} +
