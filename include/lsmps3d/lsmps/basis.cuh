#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

inline constexpr int kLsmpsTypeABasis3DSize = 9;
inline constexpr int kLsmpsTypeBBasis3DSize = 10;

struct LsmpsBasisConfig {
  int basis_size{kLsmpsTypeBBasis3DSize};
  real regularization{static_cast<real>(1.0e-3)};
};

}  // namespace lsmps3d
