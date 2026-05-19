#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

struct SurfaceDetectionDiagnosticsView {
  index_t* fluid_neighbor_count{};
  index_t* wall_neighbor_count{};
  real* number_density{};
  real* number_density_ratio{};
  real* anisotropy{};
  real* air_open_ratio{};
  real* air_anisotropy{};
  real* surface_normal_x{};
  real* surface_normal_y{};
  real* surface_normal_z{};
};

struct VirtualLightDiagnosticsView {
  int* open_direction_count{};
  real* open_fraction{};
};

[[nodiscard]] real compute_uniform_reference_number_density(real particle_spacing,
                                                            real support_radius);

void classify_surface_particles(const FluidParticleSoA& fluid,
                                const WallParticleSoA& walls,
                                const NeighborListView& fluid_neighbors,
                                const NeighborListView& wall_neighbors,
                                const SimulationConfig& config,
                                SurfaceDetectionDiagnosticsView diagnostics = {});

void compute_virtual_light_diagnostics(const FluidParticleSoA& fluid,
                                       const WallParticleSoA& walls,
                                       const NeighborListView& fluid_neighbors,
                                       const NeighborListView& wall_neighbors,
                                       const SimulationConfig& config,
                                       VirtualLightDiagnosticsView diagnostics);

}  // namespace lsmps3d
