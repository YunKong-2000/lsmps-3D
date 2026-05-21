#include "lsmps3d/provision/explicit_update.cuh"

#include <cstdlib>
#include <iostream>
#include <utility>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace lsmps3d {
namespace {

constexpr int kThreadsPerBlock = 128;

[[nodiscard]] int block_count(size_type count) {
  return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

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

void swap_views(ProvisionVelocityView& lhs, ProvisionVelocityView& rhs) noexcept {
  using std::swap;
  swap(lhs.count, rhs.count);
  swap(lhs.vx, rhs.vx);
  swap(lhs.vy, rhs.vy);
  swap(lhs.vz, rhs.vz);
}

void validate_temporary_velocity_target(const FluidParticleSoA& fluid,
                                        const FluidParticleSoA& temporary_velocity) {
  if (temporary_velocity.count < fluid.count) {
    std::cerr << "Provision temporary velocity target is smaller than fluid particle count"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 &&
      (temporary_velocity.vx == nullptr || temporary_velocity.vy == nullptr ||
       temporary_velocity.vz == nullptr)) {
    std::cerr << "Provision temporary velocity target requires vx/vy/vz arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_temporary_wall_velocity_target(const WallParticleSoA& walls,
                                             const WallParticleSoA& temporary_wall_velocity) {
  if (temporary_wall_velocity.count < walls.count) {
    std::cerr << "Provision temporary wall velocity target is smaller than wall particle count"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (walls.count > 0 &&
      (temporary_wall_velocity.vx == nullptr || temporary_wall_velocity.vy == nullptr ||
       temporary_wall_velocity.vz == nullptr)) {
    std::cerr << "Provision temporary wall velocity target requires vx/vy/vz arrays"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_laplacian_workspace(const FluidParticleSoA& fluid,
                                  const ProvisionVelocityView& velocity_laplacian) {
  if (velocity_laplacian.count < fluid.count) {
    std::cerr << "Provision laplacian workspace is smaller than fluid particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 &&
      (velocity_laplacian.vx == nullptr || velocity_laplacian.vy == nullptr ||
       velocity_laplacian.vz == nullptr)) {
    std::cerr << "Provision laplacian workspace requires vx/vy/vz arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_fluid_velocity_inputs(const FluidParticleSoA& fluid) {
  if (fluid.count > 0 &&
      (fluid.vx == nullptr || fluid.vy == nullptr || fluid.vz == nullptr)) {
    std::cerr << "Provision update requires fluid vx/vy/vz arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_wall_velocity_inputs(const WallParticleSoA& walls) {
  if (walls.count > 0 && (walls.vx == nullptr || walls.vy == nullptr || walls.vz == nullptr)) {
    std::cerr << "Provision update requires wall vx/vy/vz arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_velocity_moment_matrix(const FluidParticleSoA& fluid,
                                     const MomentMatrixView& velocity_moment) {
  if (!velocity_moment.is_ready ||
      velocity_moment.kind != MomentMatrixKind::VelocityWallDirichletTypeA ||
      velocity_moment.basis_kind != MomentBasisKind::TypeA ||
      velocity_moment.matrix_size != kMomentTypeABasis3DSize ||
      velocity_moment.particle_count < fluid.count) {
    std::cerr << "Provision requires a prepared velocity Type-A moment matrix" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 && velocity_moment.inverse_matrices == nullptr) {
    std::cerr << "Provision velocity moment matrix inverse buffer is missing" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__device__ real provision_weight(real distance, real support_radius) {
  if (distance >= support_radius) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__device__ void provision_type_a_basis(real dx,
                                       real dy,
                                       real dz,
                                       real support_radius,
                                       real basis[kMomentTypeABasis3DSize]) {
  const real inv_support = static_cast<real>(1) / support_radius;
  const real sx = dx * inv_support;
  const real sy = dy * inv_support;
  const real sz = dz * inv_support;
  basis[0] = sx;
  basis[1] = sy;
  basis[2] = sz;
  basis[3] = sx * sx;
  basis[4] = sy * sy;
  basis[5] = sz * sz;
  basis[6] = sx * sy;
  basis[7] = sy * sz;
  basis[8] = sz * sx;
}

__device__ void accumulate_velocity_rhs(const real basis[kMomentTypeABasis3DSize],
                                        real weight,
                                        real delta_x,
                                        real delta_y,
                                        real delta_z,
                                        real rhs_x[kMomentTypeABasis3DSize],
                                        real rhs_y[kMomentTypeABasis3DSize],
                                        real rhs_z[kMomentTypeABasis3DSize]) {
  for (int row = 0; row < kMomentTypeABasis3DSize; ++row) {
    const real weighted_basis = weight * basis[row];
    rhs_x[row] += weighted_basis * delta_x;
    rhs_y[row] += weighted_basis * delta_y;
    rhs_z[row] += weighted_basis * delta_z;
  }
}

__device__ real laplacian_from_inverse_rhs(const real* inverse_matrix,
                                           const real rhs[kMomentTypeABasis3DSize],
                                           real support_radius) {
  real second_sum = static_cast<real>(0);
  for (int row = 3; row <= 5; ++row) {
    real coeff = static_cast<real>(0);
    for (int col = 0; col < kMomentTypeABasis3DSize; ++col) {
      coeff += inverse_matrix[row + col * kMomentTypeABasis3DSize] * rhs[col];
    }
    second_sum += coeff;
  }
  return static_cast<real>(2) * second_sum / (support_radius * support_radius);
}

__global__ void compute_velocity_laplacian_kernel(const FluidParticleSoA fluid,
                                                  const WallParticleSoA walls,
                                                  const NeighborListView fluid_neighbors,
                                                  const NeighborListView wall_neighbors,
                                                  MomentMatrixView velocity_moment,
                                                  ProvisionVelocityView laplacian) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  const real support_radius = velocity_moment.support_radius;
  real rhs_x[kMomentTypeABasis3DSize]{};
  real rhs_y[kMomentTypeABasis3DSize]{};
  real rhs_z[kMomentTypeABasis3DSize]{};

  for (index_t cursor = fluid_neighbors.offsets[i]; cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    const real dx = fluid.x[j] - xi;
    const real dy = fluid.y[j] - yi;
    const real dz = fluid.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = provision_weight(distance, support_radius);
    if (weight <= static_cast<real>(0)) {
      continue;
    }
    real basis[kMomentTypeABasis3DSize]{};
    provision_type_a_basis(dx, dy, dz, support_radius, basis);
    accumulate_velocity_rhs(basis,
                            weight,
                            fluid.vx[j] - fluid.vx[i],
                            fluid.vy[j] - fluid.vy[i],
                            fluid.vz[j] - fluid.vz[i],
                            rhs_x,
                            rhs_y,
                            rhs_z);
  }

  for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = wall_neighbors.indices[cursor];
    const real dx = walls.x[j] - xi;
    const real dy = walls.y[j] - yi;
    const real dz = walls.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = velocity_moment.wall_weight_scale * provision_weight(distance, support_radius);
    if (weight <= static_cast<real>(0)) {
      continue;
    }
    real basis[kMomentTypeABasis3DSize]{};
    provision_type_a_basis(dx, dy, dz, support_radius, basis);
    accumulate_velocity_rhs(basis,
                            weight,
                            walls.vx[j] - fluid.vx[i],
                            walls.vy[j] - fluid.vy[i],
                            walls.vz[j] - fluid.vz[i],
                            rhs_x,
                            rhs_y,
                            rhs_z);
  }

  const real* inverse_matrix =
      velocity_moment.inverse_matrices + i * kMomentTypeABasis3DSize * kMomentTypeABasis3DSize;
  laplacian.vx[i] = laplacian_from_inverse_rhs(inverse_matrix, rhs_x, support_radius);
  laplacian.vy[i] = laplacian_from_inverse_rhs(inverse_matrix, rhs_y, support_radius);
  laplacian.vz[i] = laplacian_from_inverse_rhs(inverse_matrix, rhs_z, support_radius);
}

__global__ void combine_explicit_velocity_kernel(const FluidParticleSoA fluid,
                                                 SimulationConfig config,
                                                 ProvisionVelocityView velocity_laplacian,
                                                 FluidParticleSoA temporary_velocity) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const bool is_splash =
      fluid.surface_type != nullptr && fluid.surface_type[i] == static_cast<int>(SurfaceType::Splash);
  const real viscosity_scale = is_splash ? static_cast<real>(0) : config.kinematic_viscosity;
  const real dt = config.time_step;

  temporary_velocity.vx[i] =
      fluid.vx[i] + dt * (viscosity_scale * velocity_laplacian.vx[i] + config.gravity.x);
  temporary_velocity.vy[i] =
      fluid.vy[i] + dt * (viscosity_scale * velocity_laplacian.vy[i] + config.gravity.y);
  temporary_velocity.vz[i] =
      fluid.vz[i] + dt * (viscosity_scale * velocity_laplacian.vz[i] + config.gravity.z);
}

__global__ void combine_explicit_wall_velocity_kernel(const WallParticleSoA walls,
                                                      SimulationConfig config,
                                                      WallParticleSoA temporary_wall_velocity) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= walls.count) {
    return;
  }

  const real dt = config.time_step;
  temporary_wall_velocity.vx[i] = walls.vx[i] + dt * config.gravity.x;
  temporary_wall_velocity.vy[i] = walls.vy[i] + dt * config.gravity.y;
  temporary_wall_velocity.vz[i] = walls.vz[i] + dt * config.gravity.z;
}

}  // namespace

DeviceProvisionWorkspace::DeviceProvisionWorkspace(size_type fluid_capacity) {
  resize(fluid_capacity);
}

DeviceProvisionWorkspace::DeviceProvisionWorkspace(size_type fluid_capacity, size_type wall_capacity) {
  resize(fluid_capacity, wall_capacity);
}

DeviceProvisionWorkspace::~DeviceProvisionWorkspace() {
  release();
}

DeviceProvisionWorkspace::DeviceProvisionWorkspace(DeviceProvisionWorkspace&& other) noexcept {
  swap_views(fluid_view_, other.fluid_view_);
  swap_views(wall_view_, other.wall_view_);
}

DeviceProvisionWorkspace& DeviceProvisionWorkspace::operator=(
    DeviceProvisionWorkspace&& other) noexcept {
  if (this != &other) {
    release();
    swap_views(fluid_view_, other.fluid_view_);
    swap_views(wall_view_, other.wall_view_);
  }
  return *this;
}

void DeviceProvisionWorkspace::resize(size_type fluid_capacity) {
  resize(fluid_capacity, wall_view_.count);
}

void DeviceProvisionWorkspace::resize(size_type fluid_capacity, size_type wall_capacity) {
  if (fluid_capacity == fluid_view_.count && wall_capacity == wall_view_.count) {
    return;
  }

  release();
  fluid_view_.count = fluid_capacity;
  device_alloc(fluid_view_.vx, fluid_capacity);
  device_alloc(fluid_view_.vy, fluid_capacity);
  device_alloc(fluid_view_.vz, fluid_capacity);
  wall_view_.count = wall_capacity;
  device_alloc(wall_view_.vx, wall_capacity);
  device_alloc(wall_view_.vy, wall_capacity);
  device_alloc(wall_view_.vz, wall_capacity);
}

void DeviceProvisionWorkspace::release() noexcept {
  device_free(fluid_view_.vx);
  device_free(fluid_view_.vy);
  device_free(fluid_view_.vz);
  fluid_view_.count = 0;
  device_free(wall_view_.vx);
  device_free(wall_view_.vy);
  device_free(wall_view_.vz);
  wall_view_.count = 0;
}

size_type DeviceProvisionWorkspace::bytes() const noexcept {
  return (fluid_view_.count + wall_view_.count) * 3 * sizeof(real);
}

DeviceProvisionExplicitUpdate::DeviceProvisionExplicitUpdate(size_type fluid_capacity,
                                                             SimulationConfig config)
    : config_(std::move(config)), workspace_(fluid_capacity) {}

void DeviceProvisionExplicitUpdate::resize(size_type fluid_capacity) {
  workspace_.resize(fluid_capacity, workspace_.wall_capacity());
}

void DeviceProvisionExplicitUpdate::set_config(SimulationConfig config) {
  config_ = std::move(config);
}

size_type DeviceProvisionExplicitUpdate::bytes() const noexcept {
  return workspace_.bytes();
}

void DeviceProvisionExplicitUpdate::compute_temporary_velocity(
    const FluidParticleSoA& fluid,
    const WallParticleSoA& walls,
    const NeighborListView& fluid_neighbors,
    const NeighborListView& wall_neighbors,
    DeviceMomentMatrix& moment_matrices,
    unsigned long long geometry_generation,
    FluidParticleSoA temporary_velocity,
    WallParticleSoA temporary_wall_velocity) {
  if (workspace_.capacity() < fluid.count || workspace_.wall_capacity() < walls.count) {
    workspace_.resize(fluid.count, walls.count);
  }
  validate_fluid_velocity_inputs(fluid);
  validate_wall_velocity_inputs(walls);
  validate_temporary_velocity_target(fluid, temporary_velocity);
  validate_temporary_wall_velocity_target(walls, temporary_wall_velocity);
  validate_laplacian_workspace(fluid, workspace_.view());

  moment_matrices.set_config(config_);
  moment_matrices.resize(fluid.count);
  moment_matrices.prepare_matrices(
      fluid, walls, fluid_neighbors, wall_neighbors, geometry_generation);
  const auto velocity_moment = moment_matrices.velocity_type_a();
  validate_velocity_moment_matrix(fluid, velocity_moment);
  const auto laplacian = workspace_.view();
  if (fluid.count > 0) {
    compute_velocity_laplacian_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
        fluid, walls, fluid_neighbors, wall_neighbors, velocity_moment, laplacian);
    LSMPS3D_CUDA_KERNEL_CHECK();
  }

  compute_provision_temporary_velocity(fluid, config_, laplacian, temporary_velocity);
  compute_provision_temporary_wall_velocity(walls, config_, temporary_wall_velocity);
}

void compute_provision_temporary_velocity(const FluidParticleSoA& fluid,
                                          const SimulationConfig& config,
                                          const ProvisionVelocityView& velocity_laplacian,
                                          FluidParticleSoA temporary_velocity) {
  validate_fluid_velocity_inputs(fluid);
  validate_laplacian_workspace(fluid, velocity_laplacian);
  validate_temporary_velocity_target(fluid, temporary_velocity);
  if (fluid.count == 0) {
    return;
  }

  combine_explicit_velocity_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, config, velocity_laplacian, temporary_velocity);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void compute_provision_temporary_wall_velocity(const WallParticleSoA& walls,
                                               const SimulationConfig& config,
                                               WallParticleSoA temporary_wall_velocity) {
  validate_wall_velocity_inputs(walls);
  validate_temporary_wall_velocity_target(walls, temporary_wall_velocity);
  if (walls.count == 0) {
    return;
  }

  combine_explicit_wall_velocity_kernel<<<block_count(walls.count), kThreadsPerBlock>>>(
      walls, config, temporary_wall_velocity);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

}  // namespace lsmps3d
