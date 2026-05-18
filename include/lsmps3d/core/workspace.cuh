#pragma once

#include "lsmps3d/core/types.cuh"
#include "lsmps3d/neighbor/cell_list.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

struct WorkspaceSpec {
  size_type fluid_capacity{};
  size_type wall_capacity{};
  size_type max_fluid_neighbors_per_particle{};
  size_type max_wall_neighbors_per_particle{};
  size_type cell_capacity{};

  [[nodiscard]] size_type fluid_neighbor_capacity() const noexcept {
    return fluid_capacity * max_fluid_neighbors_per_particle;
  }

  [[nodiscard]] size_type wall_neighbor_capacity() const noexcept {
    return fluid_capacity * max_wall_neighbors_per_particle;
  }

  [[nodiscard]] size_type bytes() const noexcept;
};

struct SimulationWorkspaceView {
  FluidParticleSoA fluid{};
  WallParticleSoA walls{};
  NeighborListView fluid_neighbors{};
  NeighborListView wall_neighbors{};
  CellListView fluid_cells{};
  CellListView wall_cells{};
};

class SimulationWorkspace {
 public:
  SimulationWorkspace() = default;
  explicit SimulationWorkspace(const WorkspaceSpec& spec);

  SimulationWorkspace(const SimulationWorkspace&) = delete;
  SimulationWorkspace& operator=(const SimulationWorkspace&) = delete;
  SimulationWorkspace(SimulationWorkspace&&) noexcept = default;
  SimulationWorkspace& operator=(SimulationWorkspace&&) noexcept = default;

  void resize(const WorkspaceSpec& spec);
  void release() noexcept;

  [[nodiscard]] const WorkspaceSpec& spec() const noexcept {
    return spec_;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] SimulationWorkspaceView view() noexcept;

 private:
  WorkspaceSpec spec_{};
  DeviceFluidParticles fluid_{};
  DeviceWallParticles walls_{};
  DeviceNeighborList fluid_neighbors_{};
  DeviceNeighborList wall_neighbors_{};
  DeviceCellList fluid_cells_{};
  DeviceCellList wall_cells_{};
};

}  // namespace lsmps3d
