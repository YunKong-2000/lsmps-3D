#include "lsmps3d/neighbor/neighbor_search.cuh"

#include <cstdlib>
#include <iostream>
#include <limits>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "lsmps3d/core/cuda_check.cuh"

namespace lsmps3d {
namespace {

constexpr int kThreadsPerBlock = 128;

[[nodiscard]] int block_count(size_type count) {
  return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

void require_neighbor_capacity(const NeighborListView& neighbors, size_type required) {
  if (required <= neighbors.neighbor_count) {
    return;
  }

  std::cerr << "Neighbor list capacity exceeded: required " << required << ", capacity "
            << neighbors.neighbor_count << std::endl;
  std::exit(EXIT_FAILURE);
}

__host__ __device__ index_t flatten_cell(index_t ix, index_t iy, index_t iz, const Int3& dims) {
  return (iz * dims.y + iy) * dims.x + ix;
}

__device__ index_t clamp_cell_index(real coordinate, real origin, real cell_size, index_t dim) {
  const real scaled = (coordinate - origin) / cell_size;
  index_t index = static_cast<index_t>(floor(scaled));
  if (index < 0) {
    return 0;
  }
  if (index >= dim) {
    return dim - 1;
  }
  return index;
}

__device__ bool within_radius(real ax,
                              real ay,
                              real az,
                              real bx,
                              real by,
                              real bz,
                              real radius_squared) {
  const real dx = ax - bx;
  const real dy = ay - by;
  const real dz = az - bz;
  return dx * dx + dy * dy + dz * dz <= radius_squared;
}

template <typename Particles>
__global__ void assign_cells_kernel(Particles particles, CellGrid grid, CellListView cells) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= particles.count) {
    return;
  }

  const index_t ix = clamp_cell_index(particles.x[i], grid.origin.x, grid.cell_size, grid.dims.x);
  const index_t iy = clamp_cell_index(particles.y[i], grid.origin.y, grid.cell_size, grid.dims.y);
  const index_t iz = clamp_cell_index(particles.z[i], grid.origin.z, grid.cell_size, grid.dims.z);
  cells.cell_ids[i] = flatten_cell(ix, iy, iz, grid.dims);
  cells.sorted_indices[i] = static_cast<index_t>(i);
}

__global__ void reset_cell_ranges_kernel(CellListView cells) {
  const size_type cell = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cells.cell_count) {
    return;
  }

  cells.cell_begin[cell] = -1;
  cells.cell_end[cell] = -1;
}

__global__ void build_cell_ranges_kernel(CellListView cells) {
  const size_type sorted_pos = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sorted_pos >= cells.particle_count) {
    return;
  }

  const index_t cell = cells.cell_ids[sorted_pos];
  if (sorted_pos == 0 || cells.cell_ids[sorted_pos - 1] != cell) {
    cells.cell_begin[cell] = static_cast<index_t>(sorted_pos);
  }
  if (sorted_pos + 1 == cells.particle_count || cells.cell_ids[sorted_pos + 1] != cell) {
    cells.cell_end[cell] = static_cast<index_t>(sorted_pos + 1);
  }
}

template <typename SourceParticles>
__device__ index_t count_source_neighbors_for_fluid(size_type fluid_index,
                                                    const FluidParticleSoA& fluid,
                                                    const SourceParticles& source,
                                                    const CellListView& source_cells,
                                                    const CellGrid& grid,
                                                    real radius_squared,
                                                    bool exclude_self) {
  index_t count = 0;
  const real px = fluid.x[fluid_index];
  const real py = fluid.y[fluid_index];
  const real pz = fluid.z[fluid_index];
  const index_t cx = clamp_cell_index(px, grid.origin.x, grid.cell_size, grid.dims.x);
  const index_t cy = clamp_cell_index(py, grid.origin.y, grid.cell_size, grid.dims.y);
  const index_t cz = clamp_cell_index(pz, grid.origin.z, grid.cell_size, grid.dims.z);

  for (index_t dz = -1; dz <= 1; ++dz) {
    const index_t nz = cz + dz;
    if (nz < 0 || nz >= grid.dims.z) {
      continue;
    }
    for (index_t dy = -1; dy <= 1; ++dy) {
      const index_t ny = cy + dy;
      if (ny < 0 || ny >= grid.dims.y) {
        continue;
      }
      for (index_t dx = -1; dx <= 1; ++dx) {
        const index_t nx = cx + dx;
        if (nx < 0 || nx >= grid.dims.x) {
          continue;
        }

        const index_t cell = flatten_cell(nx, ny, nz, grid.dims);
        const index_t begin = source_cells.cell_begin[cell];
        const index_t end = source_cells.cell_end[cell];
        if (begin < 0 || end < 0) {
          continue;
        }

        for (index_t sorted_pos = begin; sorted_pos < end; ++sorted_pos) {
          const index_t source_index = source_cells.sorted_indices[sorted_pos];
          if (exclude_self && static_cast<size_type>(source_index) == fluid_index) {
            continue;
          }
          if (within_radius(px,
                            py,
                            pz,
                            source.x[source_index],
                            source.y[source_index],
                            source.z[source_index],
                            radius_squared)) {
            ++count;
          }
        }
      }
    }
  }

  return count;
}

template <typename SourceParticles>
__device__ void write_source_neighbors_for_fluid(size_type fluid_index,
                                                 const FluidParticleSoA& fluid,
                                                 const SourceParticles& source,
                                                 const CellListView& source_cells,
                                                 const CellGrid& grid,
                                                 real radius_squared,
                                                 bool exclude_self,
                                                 NeighborListView neighbors) {
  index_t out = neighbors.offsets[fluid_index];
  const real px = fluid.x[fluid_index];
  const real py = fluid.y[fluid_index];
  const real pz = fluid.z[fluid_index];
  const index_t cx = clamp_cell_index(px, grid.origin.x, grid.cell_size, grid.dims.x);
  const index_t cy = clamp_cell_index(py, grid.origin.y, grid.cell_size, grid.dims.y);
  const index_t cz = clamp_cell_index(pz, grid.origin.z, grid.cell_size, grid.dims.z);

  for (index_t dz = -1; dz <= 1; ++dz) {
    const index_t nz = cz + dz;
    if (nz < 0 || nz >= grid.dims.z) {
      continue;
    }
    for (index_t dy = -1; dy <= 1; ++dy) {
      const index_t ny = cy + dy;
      if (ny < 0 || ny >= grid.dims.y) {
        continue;
      }
      for (index_t dx = -1; dx <= 1; ++dx) {
        const index_t nx = cx + dx;
        if (nx < 0 || nx >= grid.dims.x) {
          continue;
        }

        const index_t cell = flatten_cell(nx, ny, nz, grid.dims);
        const index_t begin = source_cells.cell_begin[cell];
        const index_t end = source_cells.cell_end[cell];
        if (begin < 0 || end < 0) {
          continue;
        }

        for (index_t sorted_pos = begin; sorted_pos < end; ++sorted_pos) {
          const index_t source_index = source_cells.sorted_indices[sorted_pos];
          if (exclude_self && static_cast<size_type>(source_index) == fluid_index) {
            continue;
          }
          if (within_radius(px,
                            py,
                            pz,
                            source.x[source_index],
                            source.y[source_index],
                            source.z[source_index],
                            radius_squared)) {
            neighbors.indices[out++] = source_index;
          }
        }
      }
    }
  }
}

template <typename SourceParticles>
__global__ void count_neighbors_kernel(const FluidParticleSoA fluid,
                                       const SourceParticles source,
                                       const CellListView source_cells,
                                       const CellGrid grid,
                                       real radius_squared,
                                       bool exclude_self,
                                       NeighborListView neighbors) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  neighbors.offsets[i] = count_source_neighbors_for_fluid(
      i, fluid, source, source_cells, grid, radius_squared, exclude_self);
}

template <typename SourceParticles>
__global__ void write_neighbors_kernel(const FluidParticleSoA fluid,
                                       const SourceParticles source,
                                       const CellListView source_cells,
                                       const CellGrid grid,
                                       real radius_squared,
                                       bool exclude_self,
                                       NeighborListView neighbors) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  write_source_neighbors_for_fluid(
      i, fluid, source, source_cells, grid, radius_squared, exclude_self, neighbors);
}

template <typename Particles>
void build_cell_list_impl(const Particles& particles, const CellGrid& grid, CellListView cells) {
  if (grid.cell_size <= static_cast<real>(0) || grid.dims.x <= 0 || grid.dims.y <= 0 ||
      grid.dims.z <= 0) {
    std::cerr << "Invalid cell grid configuration" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (grid.cell_count() != static_cast<index_t>(cells.cell_count)) {
    std::cerr << "Cell list capacity does not match grid cell count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particles.count > cells.particle_count) {
    std::cerr << "Cell list particle capacity is smaller than particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particles.count > static_cast<size_type>(std::numeric_limits<index_t>::max())) {
    std::cerr << "Particle count exceeds 32-bit index capacity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  reset_cell_ranges_kernel<<<block_count(cells.cell_count), kThreadsPerBlock>>>(cells);
  LSMPS3D_CUDA_KERNEL_CHECK();

  if (particles.count == 0) {
    return;
  }

  assign_cells_kernel<<<block_count(particles.count), kThreadsPerBlock>>>(particles, grid, cells);
  LSMPS3D_CUDA_KERNEL_CHECK();

  thrust::device_ptr<index_t> cell_ids(cells.cell_ids);
  thrust::device_ptr<index_t> sorted_indices(cells.sorted_indices);
  thrust::sort_by_key(cell_ids, cell_ids + particles.count, sorted_indices);
  LSMPS3D_CUDA_CHECK(cudaGetLastError());

  build_cell_ranges_kernel<<<block_count(particles.count), kThreadsPerBlock>>>(cells);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

template <typename SourceParticles>
void build_one_neighbor_list(const FluidParticleSoA& fluid,
                             const SourceParticles& source,
                             const CellListView& source_cells,
                             const CellGrid& grid,
                             real radius,
                             bool exclude_self,
                             NeighborListView neighbors) {
  if (fluid.count > neighbors.particle_count) {
    std::cerr << "Neighbor list particle capacity is smaller than fluid count" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const real radius_squared = radius * radius;
  if (fluid.count == 0) {
    index_t zero = 0;
    LSMPS3D_CUDA_CHECK(
        cudaMemcpy(neighbors.offsets, &zero, sizeof(index_t), cudaMemcpyHostToDevice));
    return;
  }

  count_neighbors_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, source, source_cells, grid, radius_squared, exclude_self, neighbors);
  LSMPS3D_CUDA_KERNEL_CHECK();

  thrust::device_ptr<index_t> offsets(neighbors.offsets);
  thrust::exclusive_scan(offsets, offsets + fluid.count + 1, offsets);
  LSMPS3D_CUDA_CHECK(cudaGetLastError());

  index_t required = 0;
  LSMPS3D_CUDA_CHECK(cudaMemcpy(&required,
                                neighbors.offsets + fluid.count,
                                sizeof(index_t),
                                cudaMemcpyDeviceToHost));
  require_neighbor_capacity(neighbors, static_cast<size_type>(required));

  write_neighbors_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, source, source_cells, grid, radius_squared, exclude_self, neighbors);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

}  // namespace

void build_cell_list(const FluidParticleSoA& particles, const CellGrid& grid, CellListView cells) {
  build_cell_list_impl(particles, grid, cells);
}

void build_cell_list(const WallParticleSoA& particles, const CellGrid& grid, CellListView cells) {
  build_cell_list_impl(particles, grid, cells);
}

void build_neighbor_lists(const FluidParticleSoA& fluid,
                          const WallParticleSoA& walls,
                          const SimulationConfig& config,
                          CellListView fluid_cells,
                          CellListView wall_cells,
                          NeighborListView fluid_neighbors,
                          NeighborListView wall_neighbors) {
  const CellGrid grid = config.cell_grid();
  const real radius = config.neighbor_radius();
  if (radius <= static_cast<real>(0)) {
    std::cerr << "Neighbor search radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (radius > grid.cell_size) {
    std::cerr << "Neighbor search currently requires radius <= cell_size" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  build_cell_list(fluid, grid, fluid_cells);
  build_cell_list(walls, grid, wall_cells);
  build_one_neighbor_list(fluid, fluid, fluid_cells, grid, radius, true, fluid_neighbors);
  build_one_neighbor_list(fluid, walls, wall_cells, grid, radius, false, wall_neighbors);
}

}  // namespace lsmps3d
