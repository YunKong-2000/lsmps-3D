#pragma once

#include <string>

#include "lsmps3d/core/constants.hpp"
#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct SimulationConfig {
  real particle_spacing{static_cast<real>(0.01)};
  real smoothing_radius{static_cast<real>(0.031)};
  real time_step{static_cast<real>(1.0e-4)};
  real density{static_cast<real>(1000.0)};
  real kinematic_viscosity{static_cast<real>(1.0e-6)};
  Vec3 gravity{static_cast<real>(0.0), static_cast<real>(0.0), kDefaultGravityZ};
  std::string amgx_config_path{"configs/amgx_ppe.json"};

  [[nodiscard]] real neighbor_radius() const noexcept {
    return smoothing_radius;
  }

  [[nodiscard]] real particle_volume() const noexcept {
    return particle_spacing * particle_spacing * particle_spacing;
  }
};

}  // namespace lsmps3d
