#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

inline constexpr real kDefaultCfl = static_cast<real>(0.2);
inline constexpr real kDefaultGravityZ = static_cast<real>(-9.81);
inline constexpr real kDefaultNeighborRadiusFactor = static_cast<real>(3.1);
inline constexpr real kDefaultSurfaceRadiusFactor = static_cast<real>(1.5);
inline constexpr index_t kInvalidIndex = static_cast<index_t>(-1);

}  // namespace lsmps3d
