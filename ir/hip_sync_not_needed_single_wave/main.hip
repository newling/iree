#include <hip/hip_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

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

// Reverse the elements within the workgroup.
template <int NThreads, bool DoSync>
__global__ void reverso(float *out, const float *in, const unsigned N) {

  // One element per thread in the workgroup.
  __shared__ float workspace[NThreads];
  auto idFromFront = threadIdx.x;
  auto idFromBack = blockDim.x - threadIdx.x - 1;
  const unsigned int from = blockDim.x * blockIdx.x + idFromFront;
  const unsigned int to = blockDim.x * blockIdx.x + idFromBack;
  workspace[to] = in[from];
  if (DoSync) {
    __syncthreads();
  }
  out[from] = workspace[from];
}

template <int block_size, bool do_sync>
auto run() {

  constexpr unsigned int N = 1024;
  constexpr unsigned int size_bytes = N * sizeof(float);
  assert(N % block_size == 0 && "N must be multiple of block_size");
  auto grid_size = N / block_size;

  // Block and grid sizes in 2D.
  const dim3 block_dim(block_size);
  const dim3 grid_dim(grid_size);

  std::vector<float> matrix(N);
  std::iota(matrix.begin(), matrix.end(), 1.f);

  // Allocate input and output matrices on device.
  float *d_matrix{};
  float *d_reversed{};

  HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
  HIP_CHECK(hipMalloc(&d_reversed, size_bytes));

  // Copy input matrix data from host to device.
  HIP_CHECK(
      hipMemcpy(d_matrix, matrix.data(), size_bytes, hipMemcpyHostToDevice));
  std::vector<float> h_reversed(N, 10.0);
  reverso<N, do_sync>
      <<<grid_dim, block_dim, 0, hipStreamDefault>>>(d_reversed, d_matrix, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_reversed.data(), d_reversed, size_bytes,
                      hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_CHECK(hipFree(d_matrix));
  HIP_CHECK(hipFree(d_reversed));

  return h_reversed;
}

int main() {

  auto oneWaveNoSync = run<64, false>();
  auto oneWaveSync = run<64, true>();
  if (oneWaveNoSync != oneWaveSync) {
    std::cerr
        << "Mismatch between sync and no-sync for one wave -- very surprising "
        << std::endl;
    return error_exit_code;
  }

  auto multiWaveNoSync = run<512, false>();
  auto multiWaveSync = run<512, true>();
  if (multiWaveNoSync != multiWaveSync) {
    std::cerr << "Mismatch between sync and no-sync for multi wave -- expected"
              << std::endl;
  }

  return 0;
}
