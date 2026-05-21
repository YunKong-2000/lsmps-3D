#include "lsmps3d/correction/pressure_correction.cuh"

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

void swap_vector_views(CorrectionVectorView& lhs, CorrectionVectorView& rhs) noexcept {
  using std::swap;
  swap(lhs.count, rhs.count);
  swap(lhs.x, rhs.x);
  swap(lhs.y, rhs.y);
  swap(lhs.z, rhs.z);
}

void validate_fluid_arrays(const FluidParticleSoA& fluid) {
  if (fluid.count > 0 &&
      (fluid.x == nullptr || fluid.y == nullptr || fluid.z == nullptr || fluid.vx == nullptr ||
       fluid.vy == nullptr || fluid.vz == nullptr)) {
    std::cerr << "Pressure correction requires fluid position and velocity arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_temporary_velocity(const FluidParticleSoA& fluid,
                                 const FluidParticleSoA& temporary_velocity) {
  if (temporary_velocity.count < fluid.count) {
    std::cerr << "Pressure correction temporary velocity target is too small" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 &&
      (temporary_velocity.vx == nullptr || temporary_velocity.vy == nullptr ||
       temporary_velocity.vz == nullptr)) {
    std::cerr << "Pressure correction requires temporary vx/vy/vz arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_neighbors(const FluidParticleSoA& fluid, const NeighborListView& fluid_neighbors) {
  if (fluid.count > 0 &&
      (fluid_neighbors.particle_count < fluid.count || fluid_neighbors.offsets == nullptr)) {
    std::cerr << "Pressure correction requires a valid fluid-neighbor CSR table" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_pressure_input(const FluidParticleSoA& fluid, const real* pressure) {
  if (fluid.count > 0 && pressure == nullptr) {
    std::cerr << "Pressure correction requires a pressure array" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_vector_workspace(const FluidParticleSoA& fluid,
                               const CorrectionVectorView& vector,
                               const char* label) {
  if (vector.count < fluid.count) {
    std::cerr << label << " workspace is smaller than fluid particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 && (vector.x == nullptr || vector.y == nullptr || vector.z == nullptr)) {
    std::cerr << label << " workspace requires x/y/z arrays" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_pressure_moment_matrix(const FluidParticleSoA& fluid,
                                     const MomentMatrixView& pressure_moment) {
  if (!pressure_moment.is_ready ||
      pressure_moment.kind != MomentMatrixKind::PressureWallNeumannTypeA ||
      pressure_moment.basis_kind != MomentBasisKind::TypeA ||
      pressure_moment.matrix_size != kMomentTypeABasis3DSize ||
      pressure_moment.particle_count < fluid.count) {
    std::cerr << "Pressure correction requires a prepared pressure Type-A moment matrix"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 && pressure_moment.inverse_matrices == nullptr) {
    std::cerr << "Pressure correction pressure moment inverse buffer is missing" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__device__ bool is_splash_particle(const FluidParticleSoA& fluid, size_type i) {
  return fluid.surface_type != nullptr &&
         fluid.surface_type[i] == static_cast<int>(SurfaceType::Splash);
}

__device__ bool is_surface_particle(const FluidParticleSoA& fluid, size_type i) {
  if (fluid.surface_type == nullptr) {
    return false;
  }
  const int type = fluid.surface_type[i];
  return type == static_cast<int>(SurfaceType::Surface) ||
         type == static_cast<int>(SurfaceType::Splash);
}

__device__ real correction_weight(real distance, real support_radius) {
  if (distance >= support_radius || distance <= static_cast<real>(0)) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__device__ real clamp_value(real value, real low, real high) {
  return value < low ? low : (value > high ? high : value);
}

__device__ void correction_type_a_basis(real dx,
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

__device__ void correction_wall_neumann_vector(real dx,
                                               real dy,
                                               real dz,
                                               real normal_x,
                                               real normal_y,
                                               real normal_z,
                                               real support_radius,
                                               real q[kMomentTypeABasis3DSize]) {
  const real inv_support = static_cast<real>(1) / support_radius;
  q[0] = normal_x;
  q[1] = normal_y;
  q[2] = normal_z;
  q[3] = static_cast<real>(2) * dx * normal_x * inv_support;
  q[4] = static_cast<real>(2) * dy * normal_y * inv_support;
  q[5] = static_cast<real>(2) * dz * normal_z * inv_support;
  q[6] = (dy * normal_x + dx * normal_y) * inv_support;
  q[7] = (dz * normal_y + dy * normal_z) * inv_support;
  q[8] = (dx * normal_z + dz * normal_x) * inv_support;
}

__device__ real first_coefficient_from_inverse_rhs(const real* inverse_matrix,
                                                   const real rhs[kMomentTypeABasis3DSize],
                                                   int row) {
  real coeff = static_cast<real>(0);
  for (int col = 0; col < kMomentTypeABasis3DSize; ++col) {
    coeff += inverse_matrix[row + col * kMomentTypeABasis3DSize] * rhs[col];
  }
  return coeff;
}

__global__ void clamp_fluid_pressure_kernel(FluidParticleSoA fluid, real* pressure) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }
  const real clamped = pressure[i] < static_cast<real>(0) ? static_cast<real>(0) : pressure[i];
  pressure[i] = clamped;
  if (fluid.pressure != nullptr && fluid.pressure != pressure) {
    fluid.pressure[i] = clamped;
  }
}

__global__ void compute_pressure_gradient_kernel(const FluidParticleSoA fluid,
                                                 const WallParticleSoA walls,
                                                 NeighborListView fluid_neighbors,
                                                 NeighborListView wall_neighbors,
                                                 const real* pressure,
                                                 SimulationConfig config,
                                                 MomentMatrixView pressure_moment,
                                                 CorrectionVectorView pressure_gradient) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  if (is_splash_particle(fluid, i)) {
    pressure_gradient.x[i] = static_cast<real>(0);
    pressure_gradient.y[i] = static_cast<real>(0);
    pressure_gradient.z[i] = static_cast<real>(0);
    return;
  }

  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  const real support_radius = pressure_moment.support_radius;
  real rhs[kMomentTypeABasis3DSize]{};

  for (index_t cursor = fluid_neighbors.offsets[i]; cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    const real dx = fluid.x[j] - xi;
    const real dy = fluid.y[j] - yi;
    const real dz = fluid.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = correction_weight(distance, support_radius);
    if (weight <= static_cast<real>(0)) {
      continue;
    }
    real basis[kMomentTypeABasis3DSize]{};
    correction_type_a_basis(dx, dy, dz, support_radius, basis);
    const real delta_p = pressure[j] - pressure[i];
    for (int row = 0; row < kMomentTypeABasis3DSize; ++row) {
      rhs[row] += weight * basis[row] * delta_p;
    }
  }

  for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = wall_neighbors.indices[cursor];
    const real dx = walls.x[j] - xi;
    const real dy = walls.y[j] - yi;
    const real dz = walls.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = pressure_moment.wall_weight_scale * correction_weight(distance, support_radius);
    if (weight <= static_cast<real>(0)) {
      continue;
    }
    real q[kMomentTypeABasis3DSize]{};
    correction_wall_neumann_vector(
        dx, dy, dz, walls.normal_x[j], walls.normal_y[j], walls.normal_z[j], support_radius, q);
    const real normal_gravity = config.gravity.x * walls.normal_x[j] +
                                config.gravity.y * walls.normal_y[j] +
                                config.gravity.z * walls.normal_z[j];
    const real wall_rhs = support_radius * config.density * normal_gravity;
    for (int row = 0; row < kMomentTypeABasis3DSize; ++row) {
      rhs[row] += weight * q[row] * wall_rhs;
    }
  }

  const real* inverse_matrix =
      pressure_moment.inverse_matrices + i * kMomentTypeABasis3DSize * kMomentTypeABasis3DSize;
  pressure_gradient.x[i] =
      first_coefficient_from_inverse_rhs(inverse_matrix, rhs, 0) / support_radius;
  pressure_gradient.y[i] =
      first_coefficient_from_inverse_rhs(inverse_matrix, rhs, 1) / support_radius;
  pressure_gradient.z[i] =
      first_coefficient_from_inverse_rhs(inverse_matrix, rhs, 2) / support_radius;
}

__global__ void compute_particle_shifting_kernel(const FluidParticleSoA fluid,
                                                 const WallParticleSoA walls,
                                                 NeighborListView fluid_neighbors,
                                                 NeighborListView wall_neighbors,
                                                 SimulationConfig config,
                                                 CorrectionVectorView ps_displacement) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  if (config.ps_displacement_scale <= static_cast<real>(0) || is_splash_particle(fluid, i)) {
    ps_displacement.x[i] = static_cast<real>(0);
    ps_displacement.y[i] = static_cast<real>(0);
    ps_displacement.z[i] = static_cast<real>(0);
    return;
  }

  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  const real min_distance = config.ps_min_distance_ratio * config.particle_spacing;
  const real max_shift = config.ps_max_displacement_ratio * config.particle_spacing;
  real shift_x = static_cast<real>(0);
  real shift_y = static_cast<real>(0);
  real shift_z = static_cast<real>(0);
  real normal_x = static_cast<real>(0);
  real normal_y = static_cast<real>(0);
  real normal_z = static_cast<real>(0);

  for (index_t cursor = fluid_neighbors.offsets[i]; cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    const real dx = xi - fluid.x[j];
    const real dy = yi - fluid.y[j];
    const real dz = zi - fluid.z[j];
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    if (distance <= static_cast<real>(0) || distance >= min_distance) {
      continue;
    }
    const real push = config.ps_displacement_scale * (min_distance - distance) / distance;
    shift_x += push * dx;
    shift_y += push * dy;
    shift_z += push * dz;
    normal_x += dx / distance;
    normal_y += dy / distance;
    normal_z += dz / distance;
  }

  for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = wall_neighbors.indices[cursor];
    const real dx = xi - walls.x[j];
    const real dy = yi - walls.y[j];
    const real dz = zi - walls.z[j];
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    if (distance <= static_cast<real>(0) || distance >= min_distance) {
      continue;
    }
    const real push = config.ps_displacement_scale * (min_distance - distance);
    shift_x += push * walls.normal_x[j];
    shift_y += push * walls.normal_y[j];
    shift_z += push * walls.normal_z[j];
  }

  if (is_surface_particle(fluid, i)) {
    const real normal_norm = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
    if (normal_norm > static_cast<real>(0)) {
      normal_x /= normal_norm;
      normal_y /= normal_norm;
      normal_z /= normal_norm;
      const real normal_component = shift_x * normal_x + shift_y * normal_y + shift_z * normal_z;
      shift_x -= normal_component * normal_x;
      shift_y -= normal_component * normal_y;
      shift_z -= normal_component * normal_z;
    }
  }

  const real shift_norm = sqrt(shift_x * shift_x + shift_y * shift_y + shift_z * shift_z);
  if (shift_norm > max_shift && shift_norm > static_cast<real>(0)) {
    const real scale = max_shift / shift_norm;
    shift_x *= scale;
    shift_y *= scale;
    shift_z *= scale;
  }

  ps_displacement.x[i] = shift_x;
  ps_displacement.y[i] = shift_y;
  ps_displacement.z[i] = shift_z;
}

__global__ void update_velocity_position_kernel(FluidParticleSoA fluid,
                                                FluidParticleSoA temporary_velocity,
                                                SimulationConfig config,
                                                CorrectionVectorView pressure_gradient,
                                                CorrectionVectorView ps_displacement) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const bool splash = is_splash_particle(fluid, i);
  const real old_vx = fluid.vx[i];
  const real old_vy = fluid.vy[i];
  const real old_vz = fluid.vz[i];
  real new_vx = temporary_velocity.vx[i];
  real new_vy = temporary_velocity.vy[i];
  real new_vz = temporary_velocity.vz[i];
  if (!splash) {
    const real pressure_scale = config.time_step / config.density;
    new_vx -= pressure_scale * pressure_gradient.x[i];
    new_vy -= pressure_scale * pressure_gradient.y[i];
    new_vz -= pressure_scale * pressure_gradient.z[i];
  }

  fluid.x[i] += static_cast<real>(0.5) * config.time_step * (old_vx + new_vx) +
                ps_displacement.x[i];
  fluid.y[i] += static_cast<real>(0.5) * config.time_step * (old_vy + new_vy) +
                ps_displacement.y[i];
  fluid.z[i] += static_cast<real>(0.5) * config.time_step * (old_vz + new_vz) +
                ps_displacement.z[i];
  fluid.vx[i] = new_vx;
  fluid.vy[i] = new_vy;
  fluid.vz[i] = new_vz;
}

__global__ void anti_penetration_kernel(FluidParticleSoA fluid,
                                        const WallParticleSoA walls,
                                        NeighborListView wall_neighbors,
                                        SimulationConfig config) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const real clearance = config.wall_clearance_ratio * config.particle_spacing;
  if (clearance <= static_cast<real>(0)) {
    return;
  }

  for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = wall_neighbors.indices[cursor];
    const real dx = fluid.x[i] - walls.x[j];
    const real dy = fluid.y[i] - walls.y[j];
    const real dz = fluid.z[i] - walls.z[j];
    const real signed_distance =
        dx * walls.normal_x[j] + dy * walls.normal_y[j] + dz * walls.normal_z[j];
    if (signed_distance >= clearance) {
      continue;
    }

    const real correction = clearance - signed_distance;
    fluid.x[i] += correction * walls.normal_x[j];
    fluid.y[i] += correction * walls.normal_y[j];
    fluid.z[i] += correction * walls.normal_z[j];

    const real normal_velocity = fluid.vx[i] * walls.normal_x[j] +
                                 fluid.vy[i] * walls.normal_y[j] +
                                 fluid.vz[i] * walls.normal_z[j];
    if (normal_velocity < static_cast<real>(0)) {
      fluid.vx[i] -= normal_velocity * walls.normal_x[j];
      fluid.vy[i] -= normal_velocity * walls.normal_y[j];
      fluid.vz[i] -= normal_velocity * walls.normal_z[j];
    }
  }
}

__global__ void smooth_velocity_kernel(const FluidParticleSoA fluid,
                                       NeighborListView fluid_neighbors,
                                       SimulationConfig config,
                                       CorrectionVectorView smoothed_velocity) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  if (config.velocity_smoothing_strength <= static_cast<real>(0) || is_splash_particle(fluid, i)) {
    smoothed_velocity.x[i] = fluid.vx[i];
    smoothed_velocity.y[i] = fluid.vy[i];
    smoothed_velocity.z[i] = fluid.vz[i];
    return;
  }

  real weight_sum = static_cast<real>(0);
  real avg_x = static_cast<real>(0);
  real avg_y = static_cast<real>(0);
  real avg_z = static_cast<real>(0);
  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  for (index_t cursor = fluid_neighbors.offsets[i]; cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    if (is_splash_particle(fluid, static_cast<size_type>(j))) {
      continue;
    }
    const real dx = fluid.x[j] - xi;
    const real dy = fluid.y[j] - yi;
    const real dz = fluid.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = correction_weight(distance, config.support_radius);
    if (weight <= static_cast<real>(0)) {
      continue;
    }
    weight_sum += weight;
    avg_x += weight * fluid.vx[j];
    avg_y += weight * fluid.vy[j];
    avg_z += weight * fluid.vz[j];
  }

  if (weight_sum <= static_cast<real>(0)) {
    smoothed_velocity.x[i] = fluid.vx[i];
    smoothed_velocity.y[i] = fluid.vy[i];
    smoothed_velocity.z[i] = fluid.vz[i];
    return;
  }

  const real alpha =
      clamp_value(config.velocity_smoothing_strength, static_cast<real>(0), static_cast<real>(1));
  smoothed_velocity.x[i] = (static_cast<real>(1) - alpha) * fluid.vx[i] +
                           alpha * avg_x / weight_sum;
  smoothed_velocity.y[i] = (static_cast<real>(1) - alpha) * fluid.vy[i] +
                           alpha * avg_y / weight_sum;
  smoothed_velocity.z[i] = (static_cast<real>(1) - alpha) * fluid.vz[i] +
                           alpha * avg_z / weight_sum;
}

__global__ void copy_smoothed_velocity_kernel(FluidParticleSoA fluid,
                                              CorrectionVectorView smoothed_velocity) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }
  fluid.vx[i] = smoothed_velocity.x[i];
  fluid.vy[i] = smoothed_velocity.y[i];
  fluid.vz[i] = smoothed_velocity.z[i];
}

}  // namespace

DeviceCorrectionWorkspace::DeviceCorrectionWorkspace(size_type fluid_capacity) {
  resize(fluid_capacity);
}

DeviceCorrectionWorkspace::~DeviceCorrectionWorkspace() {
  release();
}

DeviceCorrectionWorkspace::DeviceCorrectionWorkspace(DeviceCorrectionWorkspace&& other) noexcept {
  swap_vector_views(pressure_gradient_, other.pressure_gradient_);
  swap_vector_views(ps_displacement_, other.ps_displacement_);
  swap_vector_views(smoothed_velocity_, other.smoothed_velocity_);
}

DeviceCorrectionWorkspace& DeviceCorrectionWorkspace::operator=(
    DeviceCorrectionWorkspace&& other) noexcept {
  if (this != &other) {
    release();
    swap_vector_views(pressure_gradient_, other.pressure_gradient_);
    swap_vector_views(ps_displacement_, other.ps_displacement_);
    swap_vector_views(smoothed_velocity_, other.smoothed_velocity_);
  }
  return *this;
}

void DeviceCorrectionWorkspace::resize(size_type fluid_capacity) {
  if (fluid_capacity == pressure_gradient_.count) {
    return;
  }

  release();
  pressure_gradient_.count = fluid_capacity;
  ps_displacement_.count = fluid_capacity;
  smoothed_velocity_.count = fluid_capacity;
  device_alloc(pressure_gradient_.x, fluid_capacity);
  device_alloc(pressure_gradient_.y, fluid_capacity);
  device_alloc(pressure_gradient_.z, fluid_capacity);
  device_alloc(ps_displacement_.x, fluid_capacity);
  device_alloc(ps_displacement_.y, fluid_capacity);
  device_alloc(ps_displacement_.z, fluid_capacity);
  device_alloc(smoothed_velocity_.x, fluid_capacity);
  device_alloc(smoothed_velocity_.y, fluid_capacity);
  device_alloc(smoothed_velocity_.z, fluid_capacity);
}

void DeviceCorrectionWorkspace::release() noexcept {
  device_free(pressure_gradient_.x);
  device_free(pressure_gradient_.y);
  device_free(pressure_gradient_.z);
  pressure_gradient_.count = 0;
  device_free(ps_displacement_.x);
  device_free(ps_displacement_.y);
  device_free(ps_displacement_.z);
  ps_displacement_.count = 0;
  device_free(smoothed_velocity_.x);
  device_free(smoothed_velocity_.y);
  device_free(smoothed_velocity_.z);
  smoothed_velocity_.count = 0;
}

size_type DeviceCorrectionWorkspace::bytes() const noexcept {
  return pressure_gradient_.count * 9 * sizeof(real);
}

DevicePressureCorrection::DevicePressureCorrection(size_type fluid_capacity, SimulationConfig config)
    : config_(std::move(config)), workspace_(fluid_capacity) {}

void DevicePressureCorrection::resize(size_type fluid_capacity) {
  workspace_.resize(fluid_capacity);
}

void DevicePressureCorrection::set_config(SimulationConfig config) {
  config_ = std::move(config);
}

size_type DevicePressureCorrection::bytes() const noexcept {
  return workspace_.bytes();
}

void DevicePressureCorrection::apply(const FluidParticleSoA& fluid,
                                     const WallParticleSoA& walls,
                                     const NeighborListView& fluid_neighbors,
                                     const NeighborListView& wall_neighbors,
                                     const FluidParticleSoA& temporary_velocity,
                                     const real* pressure,
                                     DeviceMomentMatrix& moment_matrices,
                                     unsigned long long geometry_generation) {
  if (workspace_.capacity() < fluid.count) {
    workspace_.resize(fluid.count);
  }
  validate_fluid_arrays(fluid);
  validate_temporary_velocity(fluid, temporary_velocity);
  validate_neighbors(fluid, fluid_neighbors);
  validate_pressure_input(fluid, pressure);
  validate_vector_workspace(fluid, workspace_.pressure_gradient(), "Pressure gradient");
  validate_vector_workspace(fluid, workspace_.ps_displacement(), "PS displacement");
  validate_vector_workspace(fluid, workspace_.smoothed_velocity(), "Velocity smoothing");

  clamp_particle_pressure(fluid, pressure);

  moment_matrices.set_config(config_);
  moment_matrices.resize(fluid.count);
  moment_matrices.prepare_matrices(
      fluid, walls, fluid_neighbors, wall_neighbors, geometry_generation);
  const auto pressure_moment = moment_matrices.pressure_type_a();
  compute_pressure_gradient(fluid,
                            walls,
                            fluid_neighbors,
                            wall_neighbors,
                            pressure,
                            config_,
                            pressure_moment,
                            workspace_.pressure_gradient());
  compute_particle_shifting_displacement(
      fluid, walls, fluid_neighbors, wall_neighbors, config_, workspace_.ps_displacement());
  update_velocity_and_position(fluid,
                               temporary_velocity,
                               config_,
                               workspace_.pressure_gradient(),
                               workspace_.ps_displacement());
  if (fluid.count > 0) {
    anti_penetration_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
        fluid, walls, wall_neighbors, config_);
    LSMPS3D_CUDA_KERNEL_CHECK();
  }
  smooth_particle_velocity(fluid, fluid_neighbors, config_, workspace_.smoothed_velocity());
}

void clamp_particle_pressure(const FluidParticleSoA& fluid, const real* pressure) {
  validate_pressure_input(fluid, pressure);
  if (fluid.count == 0) {
    return;
  }
  clamp_fluid_pressure_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, const_cast<real*>(pressure));
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void compute_pressure_gradient(const FluidParticleSoA& fluid,
                               const WallParticleSoA& walls,
                               const NeighborListView& fluid_neighbors,
                               const NeighborListView& wall_neighbors,
                               const real* pressure,
                               const SimulationConfig& config,
                               const MomentMatrixView& pressure_moment,
                               CorrectionVectorView pressure_gradient) {
  validate_fluid_arrays(fluid);
  validate_neighbors(fluid, fluid_neighbors);
  validate_pressure_input(fluid, pressure);
  validate_vector_workspace(fluid, pressure_gradient, "Pressure gradient");
  validate_pressure_moment_matrix(fluid, pressure_moment);
  if (fluid.count == 0) {
    return;
  }
  compute_pressure_gradient_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(fluid,
                                                                                  walls,
                                                                                  fluid_neighbors,
                                                                                  wall_neighbors,
                                                                                  pressure,
                                                                                  config,
                                                                                  pressure_moment,
                                                                                  pressure_gradient);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void compute_particle_shifting_displacement(const FluidParticleSoA& fluid,
                                            const WallParticleSoA& walls,
                                            const NeighborListView& fluid_neighbors,
                                            const NeighborListView& wall_neighbors,
                                            const SimulationConfig& config,
                                            CorrectionVectorView ps_displacement) {
  validate_fluid_arrays(fluid);
  validate_neighbors(fluid, fluid_neighbors);
  validate_vector_workspace(fluid, ps_displacement, "PS displacement");
  if (fluid.count == 0) {
    return;
  }
  compute_particle_shifting_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, walls, fluid_neighbors, wall_neighbors, config, ps_displacement);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void update_velocity_and_position(const FluidParticleSoA& fluid,
                                  const FluidParticleSoA& temporary_velocity,
                                  const SimulationConfig& config,
                                  CorrectionVectorView pressure_gradient,
                                  CorrectionVectorView ps_displacement) {
  validate_fluid_arrays(fluid);
  validate_temporary_velocity(fluid, temporary_velocity);
  validate_vector_workspace(fluid, pressure_gradient, "Pressure gradient");
  validate_vector_workspace(fluid, ps_displacement, "PS displacement");
  if (fluid.count == 0) {
    return;
  }
  update_velocity_position_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, temporary_velocity, config, pressure_gradient, ps_displacement);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void smooth_particle_velocity(const FluidParticleSoA& fluid,
                              const NeighborListView& fluid_neighbors,
                              const SimulationConfig& config,
                              CorrectionVectorView smoothed_velocity) {
  validate_fluid_arrays(fluid);
  validate_neighbors(fluid, fluid_neighbors);
  validate_vector_workspace(fluid, smoothed_velocity, "Velocity smoothing");
  if (fluid.count == 0) {
    return;
  }
  smooth_velocity_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, fluid_neighbors, config, smoothed_velocity);
  LSMPS3D_CUDA_KERNEL_CHECK();
  copy_smoothed_velocity_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, smoothed_velocity);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

}  // namespace lsmps3d
