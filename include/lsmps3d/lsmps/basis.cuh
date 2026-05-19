#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

inline constexpr int kLsmpsTypeABasis3DSize = 9;
inline constexpr int kLsmpsTypeBBasis3DSize = 10;
inline constexpr int kLsmpsMaxBasis3DSize = kLsmpsTypeBBasis3DSize;

enum class LsmpsBasisKind : int {
  TypeA = 0,
  TypeB = 1,
};

enum class LsmpsWallMode : int {
  None = 0,
  DirichletSample = 1,
  PressureNeumann = 2,
};

[[nodiscard]] __host__ __device__ inline int lsmps_basis_size(LsmpsBasisKind basis_kind) {
  return basis_kind == LsmpsBasisKind::TypeB ? kLsmpsTypeBBasis3DSize : kLsmpsTypeABasis3DSize;
}

}  // namespace lsmps3d
