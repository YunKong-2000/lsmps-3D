#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace lsmps3d {
namespace detail {

inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }

  std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"
            << "  expression: " << expression << "\n"
            << "  location: " << file << ":" << line << std::endl;
  std::exit(EXIT_FAILURE);
}

template <typename Status, typename Success>
inline void generic_check(Status status,
                          Success success,
                          const char* api_name,
                          const char* expression,
                          const char* file,
                          int line) {
  if (static_cast<int>(status) == static_cast<int>(success)) {
    return;
  }

  std::cerr << api_name << " error code: " << static_cast<int>(status) << "\n"
            << "  expression: " << expression << "\n"
            << "  location: " << file << ":" << line << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace detail
}  // namespace lsmps3d

#define LSMPS3D_CUDA_CHECK(expr) \
  ::lsmps3d::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

#define LSMPS3D_CUDA_KERNEL_CHECK() \
  do {                              \
    LSMPS3D_CUDA_CHECK(cudaGetLastError()); \
    LSMPS3D_CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (false)

#define LSMPS3D_CUBLAS_CHECK(expr) \
  ::lsmps3d::detail::generic_check((expr), 0, "cuBLAS", #expr, __FILE__, __LINE__)

#define LSMPS3D_CUSOLVER_CHECK(expr) \
  ::lsmps3d::detail::generic_check((expr), 0, "cuSOLVER", #expr, __FILE__, __LINE__)

#define LSMPS3D_CUSPARSE_CHECK(expr) \
  ::lsmps3d::detail::generic_check((expr), 0, "cuSPARSE", #expr, __FILE__, __LINE__)

#define LSMPS3D_AMGX_CHECK(expr) \
  ::lsmps3d::detail::generic_check((expr), 0, "AMGX", #expr, __FILE__, __LINE__)
