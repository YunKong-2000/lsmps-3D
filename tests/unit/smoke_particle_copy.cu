#include <array>
#include <cmath>
#include <iostream>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

int main() {
  constexpr lsmps3d::size_type kParticleCount = 4;

  lsmps3d::DeviceFluidParticles fluid(kParticleCount);
  lsmps3d::DeviceWallParticles walls(2);
  lsmps3d::DeviceNeighborList fluid_neighbors(kParticleCount, 8);
  lsmps3d::DeviceNeighborList wall_neighbors(kParticleCount, 6);
  const lsmps3d::WorkspaceSpec spec{
      kParticleCount,
      2,
      2,
      3,
      8,
  };
  lsmps3d::SimulationWorkspace workspace(spec);

  const std::array<lsmps3d::real, kParticleCount> host_x{
      static_cast<lsmps3d::real>(0.0),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.2),
      static_cast<lsmps3d::real>(0.3),
  };
  std::array<lsmps3d::real, kParticleCount> copied_x{};

  const auto view = fluid.view();
  LSMPS3D_CUDA_CHECK(cudaMemcpy(view.x,
                                host_x.data(),
                                host_x.size() * sizeof(lsmps3d::real),
                                cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(copied_x.data(),
                                view.x,
                                copied_x.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));

  for (lsmps3d::size_type i = 0; i < kParticleCount; ++i) {
    if (std::abs(copied_x[i] - host_x[i]) > static_cast<lsmps3d::real>(1.0e-6)) {
      std::cerr << "Particle copy mismatch at index " << i << std::endl;
      return 1;
    }
  }

  if (fluid.count() != kParticleCount || walls.count() != 2) {
    std::cerr << "Unexpected particle workspace size" << std::endl;
    return 1;
  }

  if (fluid_neighbors.view().particle_count != kParticleCount ||
      wall_neighbors.view().particle_count != kParticleCount) {
    std::cerr << "Unexpected neighbor list workspace size" << std::endl;
    return 1;
  }

  const auto workspace_view = workspace.view();
  if (workspace_view.fluid.count != kParticleCount || workspace_view.walls.count != 2 ||
      workspace_view.fluid_neighbors.neighbor_count != spec.fluid_neighbor_capacity() ||
      workspace_view.wall_neighbors.neighbor_count != spec.wall_neighbor_capacity() ||
      workspace_view.fluid_cells.cell_count != spec.cell_capacity ||
      workspace_view.wall_cells.cell_count != spec.cell_capacity) {
    std::cerr << "Unexpected simulation workspace capacity" << std::endl;
    return 1;
  }

  if (workspace.bytes() != spec.bytes()) {
    std::cerr << "Workspace byte estimate does not match allocated buffers" << std::endl;
    return 1;
  }

  return 0;
}
