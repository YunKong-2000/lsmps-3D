#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

struct CorrectionVectorView {
  size_type count{};
  real* x{};
  real* y{};
  real* z{};
};

class DeviceCorrectionWorkspace {
 public:
  DeviceCorrectionWorkspace() = default;
  explicit DeviceCorrectionWorkspace(size_type fluid_capacity);
  ~DeviceCorrectionWorkspace();

  DeviceCorrectionWorkspace(const DeviceCorrectionWorkspace&) = delete;
  DeviceCorrectionWorkspace& operator=(const DeviceCorrectionWorkspace&) = delete;

  DeviceCorrectionWorkspace(DeviceCorrectionWorkspace&& other) noexcept;
  DeviceCorrectionWorkspace& operator=(DeviceCorrectionWorkspace&& other) noexcept;

  void resize(size_type fluid_capacity);
  void release() noexcept;

  [[nodiscard]] size_type capacity() const noexcept {
    return pressure_gradient_.count;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] CorrectionVectorView pressure_gradient() noexcept {
    return pressure_gradient_;
  }

  [[nodiscard]] CorrectionVectorView ps_displacement() noexcept {
    return ps_displacement_;
  }

  [[nodiscard]] CorrectionVectorView smoothed_velocity() noexcept {
    return smoothed_velocity_;
  }

 private:
  CorrectionVectorView pressure_gradient_{};
  CorrectionVectorView ps_displacement_{};
  CorrectionVectorView smoothed_velocity_{};
};

class DevicePressureCorrection {
 public:
  DevicePressureCorrection() = default;
  DevicePressureCorrection(size_type fluid_capacity, SimulationConfig config);

  void resize(size_type fluid_capacity);
  void set_config(SimulationConfig config);
  [[nodiscard]] const SimulationConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] size_type bytes() const noexcept;

  void apply(const FluidParticleSoA& fluid,
             const WallParticleSoA& walls,
             const NeighborListView& fluid_neighbors,
             const NeighborListView& wall_neighbors,
             const FluidParticleSoA& temporary_velocity,
             const real* pressure,
             DeviceMomentMatrix& moment_matrices,
             unsigned long long geometry_generation);

  [[nodiscard]] CorrectionVectorView pressure_gradient() noexcept {
    return workspace_.pressure_gradient();
  }

  [[nodiscard]] CorrectionVectorView ps_displacement() noexcept {
    return workspace_.ps_displacement();
  }

 private:
  SimulationConfig config_{};
  DeviceCorrectionWorkspace workspace_{};
};

void clamp_particle_pressure(const FluidParticleSoA& fluid, const real* pressure);

void compute_pressure_gradient(const FluidParticleSoA& fluid,
                               const WallParticleSoA& walls,
                               const NeighborListView& fluid_neighbors,
                               const NeighborListView& wall_neighbors,
                               const real* pressure,
                               const SimulationConfig& config,
                               const MomentMatrixView& pressure_moment,
                               CorrectionVectorView pressure_gradient);

void compute_particle_shifting_displacement(const FluidParticleSoA& fluid,
                                            const WallParticleSoA& walls,
                                            const NeighborListView& fluid_neighbors,
                                            const NeighborListView& wall_neighbors,
                                            const SimulationConfig& config,
                                            CorrectionVectorView ps_displacement);

void update_velocity_and_position(const FluidParticleSoA& fluid,
                                  const FluidParticleSoA& temporary_velocity,
                                  const SimulationConfig& config,
                                  CorrectionVectorView pressure_gradient,
                                  CorrectionVectorView ps_displacement);

void smooth_particle_velocity(const FluidParticleSoA& fluid,
                              const NeighborListView& fluid_neighbors,
                              const SimulationConfig& config,
                              CorrectionVectorView smoothed_velocity);

}  // namespace lsmps3d
