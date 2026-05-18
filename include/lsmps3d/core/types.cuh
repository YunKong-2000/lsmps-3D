#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace lsmps3d {

#ifndef LSMPS3D_USE_DOUBLE
using real = float;
#else
using real = double;
#endif

using index_t = std::int32_t;
using size_type = std::size_t;

static_assert(sizeof(index_t) == 4, "index_t is expected to be a 32-bit integer.");

struct Vec3 {
  real x{};
  real y{};
  real z{};

  __host__ __device__ constexpr Vec3() = default;
  __host__ __device__ constexpr Vec3(real x_value, real y_value, real z_value)
      : x(x_value), y(y_value), z(z_value) {}
};

struct Int3 {
  index_t x{};
  index_t y{};
  index_t z{};

  __host__ __device__ constexpr Int3() = default;
  __host__ __device__ constexpr Int3(index_t x_value, index_t y_value, index_t z_value)
      : x(x_value), y(y_value), z(z_value) {}
};

__host__ __device__ inline Vec3 make_vec3(real x, real y, real z) {
  return Vec3{x, y, z};
}

__host__ __device__ inline Int3 make_int3(index_t x, index_t y, index_t z) {
  return Int3{x, y, z};
}

}  // namespace lsmps3d
