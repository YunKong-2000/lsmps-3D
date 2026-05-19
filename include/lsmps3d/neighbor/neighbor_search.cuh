#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/neighbor/cell_list.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

void build_cell_list(const FluidParticleSoA& particles,
                     const CellGrid& grid,
                     CellListView cells);

void build_cell_list(const WallParticleSoA& particles,
                     const CellGrid& grid,
                     CellListView cells);

void build_neighbor_lists(const FluidParticleSoA& fluid,
                          const WallParticleSoA& walls,
                          const SimulationConfig& config,
                          CellListView fluid_cells,
                          CellListView wall_cells,
                          NeighborListView fluid_neighbors,
                          NeighborListView wall_neighbors);

}  // namespace lsmps3d
