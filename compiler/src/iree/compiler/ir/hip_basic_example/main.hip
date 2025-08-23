#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>

/// This is based on llvm_ir_to_exectuble/main.hip although it does not do
/// anything complicated. It is just an example of generating artefacts to
/// peruse for better asm understanding
///
/// Two steps of compiling, one to generate the LLVM IR, second to generate
/// assembly.
///
///  1) to generate LLVMIR:
///  ```
///  hipcc  -emit-llvm -S --offload-arch=gfx942 --std=c++17 main.hip
///  ```
///
///  2) to generate assembly and executable:
///  ```
///  export OUTDIR=./funk
///  mkdir $OUTDIR
///  hipcc -save-temps=obj -g --offload-arch=gfx942 --std=c++17 main.hip -o
///  $OUTDIR/main
///  ```
///  In the above, the use of the -g flag was suggested in
///  https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
///
/// From scratch (untested)
///
/// git clone https://github.com/newling/rocm-examples.git
/// cd rocm-examples
/// cmake -S . -B build - G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=1
/// cmake --build build # probs not needed
/// ln -s build/compile_commands.json .
/// # change to this directory and run the compile commands above.

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the
/// program with an error code.
constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                        \
    if (error != hipSuccess) {                                                 \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error)      \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;       \
      std::exit(error_exit_code);                                              \
    }                                                                          \
  }

/// \brief Transposes the matrix \p in and stores the result in \p out using
/// static shared memory.
template <const unsigned int Width = 64>
__global__ void matrix_transpose_kernel(float *out, const float *in) {
  // Allocate the necessary amount of shared memory to store the transpose of
  // the matrix.
  constexpr unsigned int size = Width * Width;
  __shared__ float shared_matrix_memory[size];

  // Compute the row and column indexes of the matrix element that each thread
  // is going to process.
  const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  // If not out of bounds, transpose element (x,y).
  if (x < Width && y < Width) {
    shared_matrix_memory[y * Width + x] = in[x * Width + y];
  }

  __syncthreads();

  // If not out of bounds, transpose element (x,y).
  if (x < Width && y < Width) {
    float loc = 0; // shared_matrix_memory[y * Width + x];

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        loc += shared_matrix_memory[y * Width + x + i + j];
      }
    }
    out[y * Width + x] = loc;
  }
}

int main() {
  float *d_in, *d_out;
  size_t size_in_bytes = 1234;
  HIP_CHECK(hipMalloc(&d_in, size_in_bytes));
  HIP_CHECK(hipMalloc(&d_out, size_in_bytes));

  // Set the number of blocks per kernel grid.
  constexpr unsigned int grid_size = 512;
  // Set the number of threads per kernel block.
  constexpr unsigned int threads_per_block = 256;

  std::cout << "info: launch 'vector_square_kernel' kernel\n";
  matrix_transpose_kernel<<<grid_size, threads_per_block, 0,
                            hipStreamDefault>>>(d_out, d_in);

  // Check that the kernel invocation was successful.
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  std::cout << "PASSED!\n";
}
