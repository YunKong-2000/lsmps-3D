#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

inline constexpr int kMomentTypeABasis3DSize = 9;
inline constexpr int kMomentTypeBBasis3DSize = 10;
inline constexpr int kMomentMaxBasis3DSize = kMomentTypeBBasis3DSize;

enum class MomentBasisKind : int {
  TypeA = 0,
  TypeB = 1,
};

enum class MomentWallMode : int {
  None = 0,
  DirichletSample = 1,
  PressureNeumann = 2,
};

[[nodiscard]] __host__ __device__ inline int moment_basis_size(MomentBasisKind basis_kind) {
  return basis_kind == MomentBasisKind::TypeB ? kMomentTypeBBasis3DSize : kMomentTypeABasis3DSize;
}

}  // namespace lsmps3d
