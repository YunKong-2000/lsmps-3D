#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/neighbor/cell_list.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

#include <utility>

#include "lsmps3d/core/cuda_check.cuh"

namespace lsmps3d {
namespace {

template <typename T>
void device_alloc(T*& ptr, size_type count) {
  ptr = nullptr;
  if (count == 0) {
    return;
  }
  LSMPS3D_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
}

template <typename T>
void device_free(T*& ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
  cudaFree(ptr);
  ptr = nullptr;
}

void swap_fluid_views(FluidParticleSoA& lhs, FluidParticleSoA& rhs) noexcept {
  using std::swap;
  swap(lhs.count, rhs.count);
  swap(lhs.x, rhs.x);
  swap(lhs.y, rhs.y);
  swap(lhs.z, rhs.z);
  swap(lhs.vx, rhs.vx);
  swap(lhs.vy, rhs.vy);
  swap(lhs.vz, rhs.vz);
  swap(lhs.pressure, rhs.pressure);
  swap(lhs.surface_type, rhs.surface_type);
}

void swap_wall_views(WallParticleSoA& lhs, WallParticleSoA& rhs) noexcept {
  using std::swap;
  swap(lhs.count, rhs.count);
  swap(lhs.x, rhs.x);
  swap(lhs.y, rhs.y);
  swap(lhs.z, rhs.z);
  swap(lhs.vx, rhs.vx);
  swap(lhs.vy, rhs.vy);
  swap(lhs.vz, rhs.vz);
  swap(lhs.normal_x, rhs.normal_x);
  swap(lhs.normal_y, rhs.normal_y);
  swap(lhs.normal_z, rhs.normal_z);
}

void swap_neighbor_views(NeighborListView& lhs, NeighborListView& rhs) noexcept {
  using std::swap;
  swap(lhs.particle_count, rhs.particle_count);
  swap(lhs.neighbor_count, rhs.neighbor_count);
  swap(lhs.offsets, rhs.offsets);
  swap(lhs.indices, rhs.indices);
}

void swap_cell_views(CellListView& lhs, CellListView& rhs) noexcept {
  using std::swap;
  swap(lhs.particle_count, rhs.particle_count);
  swap(lhs.cell_count, rhs.cell_count);
  swap(lhs.cell_ids, rhs.cell_ids);
  swap(lhs.sorted_indices, rhs.sorted_indices);
  swap(lhs.cell_begin, rhs.cell_begin);
  swap(lhs.cell_end, rhs.cell_end);
}

}  // namespace

size_type WorkspaceSpec::bytes() const noexcept {
  const size_type fluid_bytes = fluid_capacity * (7 * sizeof(real) + sizeof(int));
  const size_type wall_bytes = wall_capacity * (9 * sizeof(real));
  const size_type fluid_neighbor_bytes =
      (fluid_capacity + 1) * sizeof(index_t) + fluid_neighbor_capacity() * sizeof(index_t);
  const size_type wall_neighbor_bytes =
      (fluid_capacity + 1) * sizeof(index_t) + wall_neighbor_capacity() * sizeof(index_t);
  const size_type fluid_cell_bytes =
      fluid_capacity * 2 * sizeof(index_t) + cell_capacity * 2 * sizeof(index_t);
  const size_type wall_cell_bytes =
      wall_capacity * 2 * sizeof(index_t) + cell_capacity * 2 * sizeof(index_t);
  return fluid_bytes + wall_bytes + fluid_neighbor_bytes + wall_neighbor_bytes + fluid_cell_bytes +
         wall_cell_bytes;
}

DeviceFluidParticles::DeviceFluidParticles(size_type count) {
  resize(count);
}

DeviceFluidParticles::~DeviceFluidParticles() {
  release();
}

DeviceFluidParticles::DeviceFluidParticles(DeviceFluidParticles&& other) noexcept {
  swap_fluid_views(view_, other.view_);
}

DeviceFluidParticles& DeviceFluidParticles::operator=(DeviceFluidParticles&& other) noexcept {
  if (this != &other) {
    release();
    swap_fluid_views(view_, other.view_);
  }
  return *this;
}

void DeviceFluidParticles::resize(size_type count) {
  if (count == view_.count) {
    return;
  }

  release();
  view_.count = count;
  device_alloc(view_.x, count);
  device_alloc(view_.y, count);
  device_alloc(view_.z, count);
  device_alloc(view_.vx, count);
  device_alloc(view_.vy, count);
  device_alloc(view_.vz, count);
  device_alloc(view_.pressure, count);
  device_alloc(view_.surface_type, count);
}

void DeviceFluidParticles::release() noexcept {
  device_free(view_.x);
  device_free(view_.y);
  device_free(view_.z);
  device_free(view_.vx);
  device_free(view_.vy);
  device_free(view_.vz);
  device_free(view_.pressure);
  device_free(view_.surface_type);
  view_.count = 0;
}

size_type DeviceFluidParticles::bytes() const noexcept {
  return view_.count * (7 * sizeof(real) + sizeof(int));
}

DeviceWallParticles::DeviceWallParticles(size_type count) {
  resize(count);
}

DeviceWallParticles::~DeviceWallParticles() {
  release();
}

DeviceWallParticles::DeviceWallParticles(DeviceWallParticles&& other) noexcept {
  swap_wall_views(view_, other.view_);
}

DeviceWallParticles& DeviceWallParticles::operator=(DeviceWallParticles&& other) noexcept {
  if (this != &other) {
    release();
    swap_wall_views(view_, other.view_);
  }
  return *this;
}

void DeviceWallParticles::resize(size_type count) {
  if (count == view_.count) {
    return;
  }

  release();
  view_.count = count;
  device_alloc(view_.x, count);
  device_alloc(view_.y, count);
  device_alloc(view_.z, count);
  device_alloc(view_.vx, count);
  device_alloc(view_.vy, count);
  device_alloc(view_.vz, count);
  device_alloc(view_.normal_x, count);
  device_alloc(view_.normal_y, count);
  device_alloc(view_.normal_z, count);
}

void DeviceWallParticles::release() noexcept {
  device_free(view_.x);
  device_free(view_.y);
  device_free(view_.z);
  device_free(view_.vx);
  device_free(view_.vy);
  device_free(view_.vz);
  device_free(view_.normal_x);
  device_free(view_.normal_y);
  device_free(view_.normal_z);
  view_.count = 0;
}

size_type DeviceWallParticles::bytes() const noexcept {
  return view_.count * (9 * sizeof(real));
}

DeviceNeighborList::DeviceNeighborList(size_type particle_count, size_type neighbor_capacity) {
  resize(particle_count, neighbor_capacity);
}

DeviceNeighborList::~DeviceNeighborList() {
  release();
}

DeviceNeighborList::DeviceNeighborList(DeviceNeighborList&& other) noexcept {
  swap_neighbor_views(view_, other.view_);
  using std::swap;
  swap(capacity_, other.capacity_);
}

DeviceNeighborList& DeviceNeighborList::operator=(DeviceNeighborList&& other) noexcept {
  if (this != &other) {
    release();
    swap_neighbor_views(view_, other.view_);
    using std::swap;
    swap(capacity_, other.capacity_);
  }
  return *this;
}

void DeviceNeighborList::resize(size_type particle_count, size_type neighbor_capacity) {
  if (particle_count == view_.particle_count && neighbor_capacity == capacity_) {
    return;
  }

  release();
  view_.particle_count = particle_count;
  view_.neighbor_count = neighbor_capacity;
  capacity_ = neighbor_capacity;
  device_alloc(view_.offsets, particle_count + 1);
  device_alloc(view_.indices, neighbor_capacity);
}

void DeviceNeighborList::release() noexcept {
  device_free(view_.offsets);
  device_free(view_.indices);
  view_.particle_count = 0;
  view_.neighbor_count = 0;
  capacity_ = 0;
}

size_type DeviceNeighborList::bytes() const noexcept {
  return (view_.particle_count + 1) * sizeof(index_t) + capacity_ * sizeof(index_t);
}

DeviceCellList::DeviceCellList(size_type particle_count, size_type cell_count) {
  resize(particle_count, cell_count);
}

DeviceCellList::~DeviceCellList() {
  release();
}

DeviceCellList::DeviceCellList(DeviceCellList&& other) noexcept {
  swap_cell_views(view_, other.view_);
}

DeviceCellList& DeviceCellList::operator=(DeviceCellList&& other) noexcept {
  if (this != &other) {
    release();
    swap_cell_views(view_, other.view_);
  }
  return *this;
}

void DeviceCellList::resize(size_type particle_count, size_type cell_count) {
  if (particle_count == view_.particle_count && cell_count == view_.cell_count) {
    return;
  }

  release();
  view_.particle_count = particle_count;
  view_.cell_count = cell_count;
  device_alloc(view_.cell_ids, particle_count);
  device_alloc(view_.sorted_indices, particle_count);
  device_alloc(view_.cell_begin, cell_count);
  device_alloc(view_.cell_end, cell_count);
}

void DeviceCellList::release() noexcept {
  device_free(view_.cell_ids);
  device_free(view_.sorted_indices);
  device_free(view_.cell_begin);
  device_free(view_.cell_end);
  view_.particle_count = 0;
  view_.cell_count = 0;
}

size_type DeviceCellList::bytes() const noexcept {
  return view_.particle_count * 2 * sizeof(index_t) + view_.cell_count * 2 * sizeof(index_t);
}

SimulationWorkspace::SimulationWorkspace(const WorkspaceSpec& spec) {
  resize(spec);
}

void SimulationWorkspace::resize(const WorkspaceSpec& spec) {
  if (spec.fluid_capacity == spec_.fluid_capacity && spec.wall_capacity == spec_.wall_capacity &&
      spec.max_fluid_neighbors_per_particle == spec_.max_fluid_neighbors_per_particle &&
      spec.max_wall_neighbors_per_particle == spec_.max_wall_neighbors_per_particle &&
      spec.cell_capacity == spec_.cell_capacity) {
    return;
  }

  fluid_.resize(spec.fluid_capacity);
  walls_.resize(spec.wall_capacity);
  fluid_neighbors_.resize(spec.fluid_capacity, spec.fluid_neighbor_capacity());
  wall_neighbors_.resize(spec.fluid_capacity, spec.wall_neighbor_capacity());
  fluid_cells_.resize(spec.fluid_capacity, spec.cell_capacity);
  wall_cells_.resize(spec.wall_capacity, spec.cell_capacity);
  spec_ = spec;
}

void SimulationWorkspace::release() noexcept {
  fluid_.release();
  walls_.release();
  fluid_neighbors_.release();
  wall_neighbors_.release();
  fluid_cells_.release();
  wall_cells_.release();
  spec_ = {};
}

size_type SimulationWorkspace::bytes() const noexcept {
  return fluid_.bytes() + walls_.bytes() + fluid_neighbors_.bytes() + wall_neighbors_.bytes() +
         fluid_cells_.bytes() + wall_cells_.bytes();
}

SimulationWorkspaceView SimulationWorkspace::view() noexcept {
  return SimulationWorkspaceView{
      fluid_.view(),
      walls_.view(),
      fluid_neighbors_.view(),
      wall_neighbors_.view(),
      fluid_cells_.view(),
      wall_cells_.view(),
  };
}

}  // namespace lsmps3d
