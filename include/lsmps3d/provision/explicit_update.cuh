#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

struct ProvisionVelocityView {
  size_type count{};
  real* vx{};
  real* vy{};
  real* vz{};
};

class DeviceProvisionWorkspace {
 public:
  DeviceProvisionWorkspace() = default;
  explicit DeviceProvisionWorkspace(size_type fluid_capacity);
  DeviceProvisionWorkspace(size_type fluid_capacity, size_type wall_capacity);
  ~DeviceProvisionWorkspace();

  DeviceProvisionWorkspace(const DeviceProvisionWorkspace&) = delete;
  DeviceProvisionWorkspace& operator=(const DeviceProvisionWorkspace&) = delete;

  DeviceProvisionWorkspace(DeviceProvisionWorkspace&& other) noexcept;
  DeviceProvisionWorkspace& operator=(DeviceProvisionWorkspace&& other) noexcept;

  void resize(size_type fluid_capacity);
  void resize(size_type fluid_capacity, size_type wall_capacity);
  void release() noexcept;

  [[nodiscard]] size_type capacity() const noexcept {
    return fluid_view_.count;
  }

  [[nodiscard]] size_type wall_capacity() const noexcept {
    return wall_view_.count;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] ProvisionVelocityView view() noexcept {
    return fluid_view_;
  }

  [[nodiscard]] ProvisionVelocityView wall_view() noexcept {
    return wall_view_;
  }

 private:
  ProvisionVelocityView fluid_view_{};
  ProvisionVelocityView wall_view_{};
};

class DeviceProvisionExplicitUpdate {
 public:
  DeviceProvisionExplicitUpdate() = default;
  DeviceProvisionExplicitUpdate(size_type fluid_capacity, SimulationConfig config);

  void resize(size_type fluid_capacity);
  void set_config(SimulationConfig config);
  [[nodiscard]] const SimulationConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] size_type bytes() const noexcept;

  void compute_temporary_velocity(const FluidParticleSoA& fluid,
                                  const WallParticleSoA& walls,
                                  const NeighborListView& fluid_neighbors,
                                  const NeighborListView& wall_neighbors,
                                  DeviceMomentMatrix& moment_matrices,
                                  unsigned long long geometry_generation,
                                  FluidParticleSoA temporary_velocity,
                                  WallParticleSoA temporary_wall_velocity);

 private:
  SimulationConfig config_{};
  DeviceProvisionWorkspace workspace_{};
};

void compute_provision_temporary_velocity(const FluidParticleSoA& fluid,
                                          const SimulationConfig& config,
                                          const ProvisionVelocityView& velocity_laplacian,
                                          FluidParticleSoA temporary_velocity);

void compute_provision_temporary_wall_velocity(const WallParticleSoA& walls,
                                               const SimulationConfig& config,
                                               WallParticleSoA temporary_wall_velocity);

}  // namespace lsmps3d
