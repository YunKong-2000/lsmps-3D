#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct CsrMatrixView {
  size_type rows{};
  size_type cols{};
  size_type nnz{};
  index_t* row_offsets{};
  index_t* col_indices{};
  real* values{};
};

struct PpeWorkspaceView {
  CsrMatrixView matrix{};
  real* rhs{};
  real* pressure{};
};

}  // namespace lsmps3d
