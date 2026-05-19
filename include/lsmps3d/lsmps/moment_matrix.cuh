#pragma once

#include <memory>

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/lsmps/basis.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

class DeviceLsmpsOperators {
 public:
  DeviceLsmpsOperators();
  DeviceLsmpsOperators(size_type fluid_capacity, SimulationConfig config);
  ~DeviceLsmpsOperators();

  DeviceLsmpsOperators(const DeviceLsmpsOperators&) = delete;
  DeviceLsmpsOperators& operator=(const DeviceLsmpsOperators&) = delete;

  DeviceLsmpsOperators(DeviceLsmpsOperators&& other) noexcept;
  DeviceLsmpsOperators& operator=(DeviceLsmpsOperators&& other) noexcept;

  void resize(size_type fluid_capacity);
  void release() noexcept;
  void set_config(SimulationConfig config);
  [[nodiscard]] const SimulationConfig& config() const noexcept;
  [[nodiscard]] size_type bytes() const noexcept;

  void prepare_matrices(const FluidParticleSoA& fluid,
                        const WallParticleSoA& walls,
                        const NeighborListView& fluid_neighbors,
                        const NeighborListView& wall_neighbors,
                        unsigned long long geometry_generation);

  void compute_pressure_gradient(const FluidParticleSoA& fluid,
                                 const WallParticleSoA& walls,
                                 const NeighborListView& fluid_neighbors,
                                 const NeighborListView& wall_neighbors,
                                 const real* pressure,
                                 real* gradient_x,
                                 real* gradient_y,
                                 real* gradient_z);

  void compute_near_surface_pressure_gradient(const FluidParticleSoA& fluid,
                                              const WallParticleSoA& walls,
                                              const NeighborListView& fluid_neighbors,
                                              const NeighborListView& wall_neighbors,
                                              const real* pressure,
                                              real* gradient_x,
                                              real* gradient_y,
                                              real* gradient_z);

  void compute_pressure_laplacian(const FluidParticleSoA& fluid,
                                  const WallParticleSoA& walls,
                                  const NeighborListView& fluid_neighbors,
                                  const NeighborListView& wall_neighbors,
                                  const real* pressure,
                                  real* laplacian);

  void compute_velocity_gradient(const FluidParticleSoA& fluid,
                                 const WallParticleSoA& walls,
                                 const NeighborListView& fluid_neighbors,
                                 const NeighborListView& wall_neighbors,
                                 const real* velocity_component,
                                 const real* wall_velocity_component,
                                 real* gradient_x,
                                 real* gradient_y,
                                 real* gradient_z);

  void compute_velocity_laplacian(const FluidParticleSoA& fluid,
                                  const WallParticleSoA& walls,
                                  const NeighborListView& fluid_neighbors,
                                  const NeighborListView& wall_neighbors,
                                  const real* velocity_component,
                                  const real* wall_velocity_component,
                                  real* laplacian);

  void compute_velocity_divergence(const FluidParticleSoA& fluid,
                                   const WallParticleSoA& walls,
                                   const NeighborListView& fluid_neighbors,
                                   const NeighborListView& wall_neighbors,
                                   real* divergence);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lsmps3d
