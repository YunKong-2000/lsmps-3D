#pragma once

#include "lsmps3d/core/types.cuh"
#include "lsmps3d/lsmps/basis.cuh"

namespace lsmps3d {

struct MomentMatrixWorkspaceView {
  size_type particle_count{};
  int matrix_size{kLsmpsTypeBBasis3DSize};
  real* matrices{};
  real* rhs{};
  int* info{};
};

}  // namespace lsmps3d
