#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(stmt)                                                   \
  do {                                                                    \
    hipError_t err = (stmt);                                              \
    if (err != hipSuccess) {                                              \
      fprintf(stderr, "HIP error %s at %s:%d : %s\n", #stmt, __FILE__,    \
              __LINE__, hipGetErrorString(err));                          \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

using float4v = float __attribute__((__vector_size__(4 * sizeof(float))));

/// Specifically target the mfma_f32_16x16x4f32 instruction, where thread mapping is
///
/// A: [[0, 16, 32, 48],
///     [1, 17, 33, 49],
///     [2, 18, 34, 50],
///     [..]
///     [...]]
///
/// B: [[0, 1, 2, ..., 15],
///     [16,17,18,...,31],
///     ...
/// C: [[0, 1, 2, ..., 15],
///     [0, 1, 2, ..., 15],
///     [0, 1, 2, ..., 15],
///     [16, 17,18,...,31],
///     [16, 17,18,...,31],
///     ...
static __global__ void mfma_test(const float *__restrict__ A,
                                 const float *__restrict__ B,
                                 float *__restrict__ C) {

  const int thread_id = threadIdx.x;

  auto a_row = thread_id % 16;
  auto a_col = thread_id / 16;

  auto b_row = thread_id / 16;
  auto b_col = thread_id % 16;

  auto c_row = thread_id / 16;
  auto c_col = thread_id % 16;

  auto a_row_stride = 4;
  auto b_row_stride = 16;
  auto c_row_stride = 16;


  // Accumulator (4 partials per lane). Start from zero.
  float4v d = {0.f, 0.f, 0.f, 0.f};

  const float a = A[a_row * a_row_stride + a_col];
  const float b = B[b_row * b_row_stride + b_col];

  // First scalar argument: cbsz. (broadcast size)
  // Second (and probably third) are closely related: abid and blgp.
  d = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, d, 0, 0, 0);

  // Write back to C from the registers.
  for (int i = 0; i < 4; ++i) {
    C[(c_row * 4 + i) * c_row_stride + c_col] = d[i];
  }
}

// Naive CPU GEMM for verification: C = A*B
static void host_gemm(const float *A, const float *B, float *C) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float acc = 0.0;
      for (int p = 0; p < 4; ++p) {
        acc += A[i * 4 + p] * B[p * 16 + j];
      }
      C[i * 16 + j] = acc;
    }
  }
}

// A: 16 * 4
// B: 4  * 16
// C: 16 * 16
int main() {
  // Host allocations
  std::vector<float> hA(64), hB(64), hC(256), hC_ref(256);

  // Use modern C++ random facilities to initialize the matrices with either 0
  // or 1 (sparsity 0.5)
  std::mt19937 gen(42);
  std::bernoulli_distribution dist(0.5);
  auto rand01 = [&]() { return dist(gen) ? 1.0f : 0.0f; };
  std::generate(hA.begin(), hA.end(), rand01);
  std::generate(hB.begin(), hB.end(), rand01);

  // Device allocations
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  HIP_CHECK(hipMalloc(&dA, 64 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, 64 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dC, 256 * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(dA, hA.data(), 64 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dB, hB.data(), 64 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, 256 * sizeof(float)));

  // Launch: one wavefront (16x4) per 16×16 tile
  dim3 block(64);
  dim3 grid(1);

  hipLaunchKernelGGL(mfma_test, grid, block, 0, 0, dA, dB, dC);
  HIP_CHECK(hipDeviceSynchronize());

  // Copy back
  HIP_CHECK(
      hipMemcpy(hC.data(), dC, 256 * sizeof(float), hipMemcpyDeviceToHost));

  // Reference
  host_gemm(hA.data(), hB.data(), hC_ref.data());

  auto printTensor = [&](const float *data, int nRows, int nCols) {
    for (int i = 0; i < nRows; ++i) {
      for (int j = 0; j < nCols; ++j) {
        printf("%6.2f ", data[i * nCols + j]);
      }
      printf("\n");
    }
    printf("\n");
  };

  printTensor(hA.data(), 16, 4);
  printTensor(hB.data(), 4, 16);
  printTensor(hC_ref.data(), 16, 16);

  // Compare
  //
  auto funk = [&]() {
    for (int i = 0; i < 256; ++i) {
      const float diff = std::abs(hC[i] - hC_ref[i]);
      if (diff > 0.01) {
        printf("Mismatch at index %d: got %.6e, expected %.6e\n", i, hC[i],
               hC_ref[i]);
        return false;
      }
    }
    return true;
  }();

  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  return funk;
}
