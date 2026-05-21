#include "lsmps3d/moment_matrix/moment_matrix.cuh"

#include <cstdlib>
#include <iostream>
#include <utility>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "lsmps3d/core/cuda_check.cuh"

namespace lsmps3d {
namespace {

constexpr int kThreadsPerBlock = 128;

enum class MomentPhysicalField : int {
  Pressure = 0,
  Velocity = 1,
};

enum class MomentBoundaryKind : int {
  None = 0,
  WallDirichlet = 1,
  WallPressureNeumann = 2,
};

struct MomentMatrixRequest {
  MomentPhysicalField field{MomentPhysicalField::Velocity};
  MomentBoundaryKind boundary{MomentBoundaryKind::None};
  MomentBasisKind basis_kind{MomentBasisKind::TypeA};
  MomentMatrixKind kind{MomentMatrixKind::VelocityWallDirichletTypeA};
  real support_radius{};
  real regularization{static_cast<real>(1.0e-8)};
  real wall_weight_scale{static_cast<real>(1)};
  unsigned long long geometry_generation{};
};

struct MomentMatrixCacheView {
  size_type particle_count{};
  int matrix_size{kMomentTypeABasis3DSize};
  MomentMatrixRequest request{};
  bool is_ready{false};
  real* matrices{};
  real* inverse_matrices{};
  real** matrix_ptrs{};
  real** inverse_matrix_ptrs{};
  int* info{};
  real* moment_trace{};
  real* regularization_added{};
  int* inversion_count{};
};

struct MatrixCache {
  MatrixCache() = default;
  MatrixCache(size_type particle_count, MomentBasisKind basis_kind) {
    resize(particle_count, basis_kind);
  }
  ~MatrixCache() {
    release();
  }

  MatrixCache(const MatrixCache&) = delete;
  MatrixCache& operator=(const MatrixCache&) = delete;

  MatrixCache(MatrixCache&& other) noexcept {
    using std::swap;
    swap(view, other.view);
  }

  MatrixCache& operator=(MatrixCache&& other) noexcept = delete;

  void resize(size_type particle_count, MomentBasisKind basis_kind);
  void release() noexcept;
  [[nodiscard]] size_type bytes() const noexcept;

  MomentMatrixCacheView view{};
};

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

void validate_matrix_request(const MomentMatrixRequest& request) {
  if (request.support_radius <= static_cast<real>(0)) {
    std::cerr << "Moment support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (request.regularization < static_cast<real>(0)) {
    std::cerr << "Moment regularization must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (request.wall_weight_scale < static_cast<real>(0)) {
    std::cerr << "Moment wall weight scale must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

bool same_matrix_request(const MomentMatrixRequest& lhs, const MomentMatrixRequest& rhs) {
  return lhs.field == rhs.field && lhs.boundary == rhs.boundary &&
         lhs.basis_kind == rhs.basis_kind && lhs.kind == rhs.kind &&
         lhs.support_radius == rhs.support_radius && lhs.regularization == rhs.regularization &&
         lhs.wall_weight_scale == rhs.wall_weight_scale &&
         lhs.geometry_generation == rhs.geometry_generation;
}

void validate_cache(const MomentMatrixCacheView& cache,
                    size_type particle_count,
                    MomentBasisKind basis_kind) {
  const int matrix_size = moment_basis_size(basis_kind);
  if (cache.particle_count < particle_count || cache.matrix_size != matrix_size) {
    std::cerr << "Invalid Moment matrix cache" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particle_count == 0) {
    return;
  }
  if (cache.matrices == nullptr || cache.inverse_matrices == nullptr ||
      cache.matrix_ptrs == nullptr || cache.inverse_matrix_ptrs == nullptr ||
      cache.info == nullptr || cache.moment_trace == nullptr ||
      cache.regularization_added == nullptr || cache.inversion_count == nullptr) {
    std::cerr << "Invalid Moment matrix cache buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__host__ __device__ real moment_weight(real distance, real support_radius) {
  if (distance >= support_radius) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__device__ int basis_vector(real dx,
                            real dy,
                            real dz,
                            MomentBasisKind basis_kind,
                            real support_radius,
                            real basis[kMomentMaxBasis3DSize]) {
  const real sx = dx / support_radius;
  const real sy = dy / support_radius;
  const real sz = dz / support_radius;
  int offset = 0;
  if (basis_kind == MomentBasisKind::TypeB) {
    basis[offset++] = static_cast<real>(1);
  }
  basis[offset++] = sx;
  basis[offset++] = sy;
  basis[offset++] = sz;
  basis[offset++] = sx * sx;
  basis[offset++] = sy * sy;
  basis[offset++] = sz * sz;
  basis[offset++] = sx * sy;
  basis[offset++] = sy * sz;
  basis[offset++] = sz * sx;
  return offset;
}

__device__ int wall_neumann_vector(real dx,
                                   real dy,
                                   real dz,
                                   real normal_x,
                                   real normal_y,
                                   real normal_z,
                                   MomentBasisKind basis_kind,
                                   real support_radius,
                                   real q[kMomentMaxBasis3DSize]) {
  int offset = 0;
  const real inv_support = static_cast<real>(1) / support_radius;
  if (basis_kind == MomentBasisKind::TypeB) {
    q[offset++] = static_cast<real>(0);
  }
  q[offset++] = normal_x;
  q[offset++] = normal_y;
  q[offset++] = normal_z;
  q[offset++] = static_cast<real>(2) * dx * normal_x * inv_support;
  q[offset++] = static_cast<real>(2) * dy * normal_y * inv_support;
  q[offset++] = static_cast<real>(2) * dz * normal_z * inv_support;
  q[offset++] = (dy * normal_x + dx * normal_y) * inv_support;
  q[offset++] = (dz * normal_y + dy * normal_z) * inv_support;
  q[offset++] = (dx * normal_z + dz * normal_x) * inv_support;
  return offset;
}

__device__ void add_outer_product(const real* vector, int n, real weight, real* matrix) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      matrix[row + col * n] += weight * vector[row] * vector[col];
    }
  }
}

__device__ void add_sample_to_matrix(real dx,
                                     real dy,
                                     real dz,
                                     real weight_scale,
                                     const MomentMatrixRequest& config,
                                     real* matrix) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }
  const real distance = sqrt(distance_squared);
  const real weight = weight_scale * moment_weight(distance, config.support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }
  real basis[kMomentMaxBasis3DSize]{};
  const int n = basis_vector(dx, dy, dz, config.basis_kind, config.support_radius, basis);
  add_outer_product(basis, n, weight, matrix);
}

__device__ void add_wall_neumann_to_matrix(real dx,
                                           real dy,
                                           real dz,
                                           real normal_x,
                                           real normal_y,
                                           real normal_z,
                                           const MomentMatrixRequest& config,
                                           real* matrix) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }
  const real distance = sqrt(distance_squared);
  const real weight = config.wall_weight_scale * moment_weight(distance, config.support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }
  real q[kMomentMaxBasis3DSize]{};
  const int n = wall_neumann_vector(
      dx, dy, dz, normal_x, normal_y, normal_z, config.basis_kind, config.support_radius, q);
  add_outer_product(q, n, weight, matrix);
}

__global__ void build_matrix_cache_kernel(const FluidParticleSoA fluid,
                                          const WallParticleSoA walls,
                                          const NeighborListView fluid_neighbors,
                                          const NeighborListView wall_neighbors,
                                          const MomentMatrixRequest config,
                                          MomentMatrixCacheView cache) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const int n = cache.matrix_size;
  real* matrix = cache.matrices + i * n * n;
  for (int entry = 0; entry < n * n; ++entry) {
    matrix[entry] = static_cast<real>(0);
  }

  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  for (index_t cursor = fluid_neighbors.offsets[i]; cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    add_sample_to_matrix(
        fluid.x[j] - xi, fluid.y[j] - yi, fluid.z[j] - zi, static_cast<real>(1), config, matrix);
  }

  if (config.boundary != MomentBoundaryKind::None) {
    for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
         ++cursor) {
      const index_t j = wall_neighbors.indices[cursor];
      const real dx = walls.x[j] - xi;
      const real dy = walls.y[j] - yi;
      const real dz = walls.z[j] - zi;
      if (config.boundary == MomentBoundaryKind::WallPressureNeumann) {
        add_wall_neumann_to_matrix(
            dx, dy, dz, walls.normal_x[j], walls.normal_y[j], walls.normal_z[j], config, matrix);
      } else {
        add_sample_to_matrix(dx, dy, dz, config.wall_weight_scale, config, matrix);
      }
    }
  }

  real trace = static_cast<real>(0);
  for (int row = 0; row < n; ++row) {
    trace += matrix[row + row * n];
  }
  const real regularization =
      config.regularization *
      (trace > static_cast<real>(0) ? trace / static_cast<real>(n) : static_cast<real>(1));
  for (int row = 0; row < n; ++row) {
    matrix[row + row * n] += regularization;
  }
  cache.moment_trace[i] = trace;
  cache.regularization_added[i] = regularization;
}

__global__ void setup_matrix_pointer_kernel(MomentMatrixCacheView cache) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cache.particle_count) {
    return;
  }

  const int n = cache.matrix_size;
  cache.matrix_ptrs[i] = cache.matrices + i * n * n;
  cache.inverse_matrix_ptrs[i] = cache.inverse_matrices + i * n * n;
}

__global__ void setup_inverse_column_pointer_kernel(MomentMatrixCacheView cache, int column) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cache.particle_count) {
    return;
  }

  const int n = cache.matrix_size;
  cache.inverse_matrix_ptrs[i] = cache.inverse_matrices + i * n * n + column * n;
}

__global__ void fill_identity_inverse_kernel(MomentMatrixCacheView cache) {
  const size_type matrix_entry = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_type n = static_cast<size_type>(cache.matrix_size);
  const size_type total = cache.particle_count * n * n;
  if (matrix_entry >= total) {
    return;
  }
  const size_type local = matrix_entry % (n * n);
  const size_type row = local % n;
  const size_type col = local / n;
  cache.inverse_matrices[matrix_entry] = row == col ? static_cast<real>(1) : static_cast<real>(0);
}

__global__ void increment_inversion_count_kernel(MomentMatrixCacheView cache) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && cache.inversion_count != nullptr) {
    ++cache.inversion_count[0];
  }
}

}  // namespace

void MatrixCache::resize(size_type particle_count, MomentBasisKind basis_kind) {
  const int matrix_size = moment_basis_size(basis_kind);
  if (particle_count == view.particle_count && matrix_size == view.matrix_size) {
    return;
  }

  release();
  view.particle_count = particle_count;
  view.matrix_size = matrix_size;
  view.request.basis_kind = basis_kind;
  device_alloc(view.matrices, particle_count * matrix_size * matrix_size);
  device_alloc(view.inverse_matrices, particle_count * matrix_size * matrix_size);
  device_alloc(view.matrix_ptrs, particle_count);
  device_alloc(view.inverse_matrix_ptrs, particle_count);
  device_alloc(view.info, particle_count);
  device_alloc(view.moment_trace, particle_count);
  device_alloc(view.regularization_added, particle_count);
  device_alloc(view.inversion_count, 1);
  if (view.inversion_count != nullptr) {
    LSMPS3D_CUDA_CHECK(cudaMemset(view.inversion_count, 0, sizeof(int)));
  }
}

void MatrixCache::release() noexcept {
  device_free(view.matrices);
  device_free(view.inverse_matrices);
  device_free(view.matrix_ptrs);
  device_free(view.inverse_matrix_ptrs);
  device_free(view.info);
  device_free(view.moment_trace);
  device_free(view.regularization_added);
  device_free(view.inversion_count);
  view.particle_count = 0;
  view.matrix_size = kMomentTypeABasis3DSize;
  view.request = {};
  view.is_ready = false;
}

size_type MatrixCache::bytes() const noexcept {
  const size_type n = static_cast<size_type>(view.matrix_size);
  return view.particle_count * (2 * n * n * sizeof(real) + 2 * sizeof(real*) +
                                sizeof(int) + 2 * sizeof(real)) +
         sizeof(int);
}

void ensure_matrix_internal(const FluidParticleSoA& fluid,
                            const WallParticleSoA& walls,
                            const NeighborListView& fluid_neighbors,
                            const NeighborListView& wall_neighbors,
                            const MomentMatrixRequest& request,
                            MomentMatrixCacheView& cache) {
  validate_matrix_request(request);
  validate_cache(cache, fluid.count, request.basis_kind);
  if (fluid.count > fluid_neighbors.particle_count || fluid.count > wall_neighbors.particle_count) {
    std::cerr << "Moment matrix neighbor lists are smaller than fluid particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (same_matrix_request(cache.request, request) && cache.is_ready) {
    return;
  }
  if (fluid.count == 0) {
    cache.request = request;
    cache.is_ready = true;
    return;
  }

  build_matrix_cache_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, walls, fluid_neighbors, wall_neighbors, request, cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  setup_matrix_pointer_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  cache.request = request;
  cache.is_ready = false;
  fill_identity_inverse_kernel<<<block_count(fluid.count * cache.matrix_size * cache.matrix_size),
                                 kThreadsPerBlock>>>(cache);
  LSMPS3D_CUDA_KERNEL_CHECK();

  cusolverDnHandle_t solver{};
  LSMPS3D_CUSOLVER_CHECK(cusolverDnCreate(&solver));
  const int batch_size = static_cast<int>(cache.particle_count);
  const int n = cache.matrix_size;
#ifndef LSMPS3D_USE_DOUBLE
  LSMPS3D_CUSOLVER_CHECK(cusolverDnSpotrfBatched(
      solver, CUBLAS_FILL_MODE_LOWER, n, cache.matrix_ptrs, n, cache.info, batch_size));
#else
  LSMPS3D_CUSOLVER_CHECK(cusolverDnDpotrfBatched(
      solver, CUBLAS_FILL_MODE_LOWER, n, cache.matrix_ptrs, n, cache.info, batch_size));
#endif
#ifndef LSMPS3D_USE_DOUBLE
  for (int column = 0; column < n; ++column) {
    setup_inverse_column_pointer_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(cache, column);
    LSMPS3D_CUDA_KERNEL_CHECK();
    LSMPS3D_CUSOLVER_CHECK(cusolverDnSpotrsBatched(solver,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   n,
                                                   1,
                                                   cache.matrix_ptrs,
                                                   n,
                                                   cache.inverse_matrix_ptrs,
                                                   n,
                                                   cache.info,
                                                   batch_size));
  }
#else
  for (int column = 0; column < n; ++column) {
    setup_inverse_column_pointer_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(cache, column);
    LSMPS3D_CUDA_KERNEL_CHECK();
    LSMPS3D_CUSOLVER_CHECK(cusolverDnDpotrsBatched(solver,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   n,
                                                   1,
                                                   cache.matrix_ptrs,
                                                   n,
                                                   cache.inverse_matrix_ptrs,
                                                   n,
                                                   cache.info,
                                                   batch_size));
  }
#endif
  LSMPS3D_CUSOLVER_CHECK(cusolverDnDestroy(solver));
  increment_inversion_count_kernel<<<1, 1>>>(cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  cache.is_ready = true;
}

MomentMatrixView make_public_view(const MomentMatrixCacheView& cache) {
  return MomentMatrixView{cache.particle_count,
                          cache.matrix_size,
                          cache.request.basis_kind,
                          cache.request.kind,
                          cache.request.support_radius,
                          cache.request.wall_weight_scale,
                          cache.request.geometry_generation,
                          cache.is_ready,
                          cache.inverse_matrices,
                          cache.moment_trace,
                          cache.regularization_added,
                          cache.info,
                          cache.inversion_count};
}

MomentMatrixRequest make_matrix_request(MomentMatrixKind kind,
                                       const SimulationConfig& config,
                                       unsigned long long geometry_generation) {
  switch (kind) {
    case MomentMatrixKind::VelocityWallDirichletTypeA:
      return MomentMatrixRequest{MomentPhysicalField::Velocity,
                                MomentBoundaryKind::WallDirichlet,
                                MomentBasisKind::TypeA,
                                kind,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case MomentMatrixKind::PressureWallNeumannTypeA:
      return MomentMatrixRequest{MomentPhysicalField::Pressure,
                                MomentBoundaryKind::WallPressureNeumann,
                                MomentBasisKind::TypeA,
                                kind,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case MomentMatrixKind::PressureWallNeumannTypeB:
      return MomentMatrixRequest{MomentPhysicalField::Pressure,
                                MomentBoundaryKind::WallPressureNeumann,
                                MomentBasisKind::TypeB,
                                kind,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case MomentMatrixKind::FluidOnlyTypeA:
      return MomentMatrixRequest{MomentPhysicalField::Velocity,
                                MomentBoundaryKind::None,
                                MomentBasisKind::TypeA,
                                kind,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
  }

  std::cerr << "Unknown Moment matrix kind" << std::endl;
  std::exit(EXIT_FAILURE);
}

struct DeviceMomentMatrix::Impl {
  explicit Impl(SimulationConfig initial_config) : config(std::move(initial_config)) {}

  SimulationConfig config{};
  unsigned long long geometry_generation{};
  bool has_prepared_generation{false};
  MatrixCache velocity_type_a;
  MatrixCache pressure_type_a;
  MatrixCache pressure_type_b;
  MatrixCache fluid_only_type_a;
  void resize(size_type fluid_capacity) {
    velocity_type_a.resize(fluid_capacity, MomentBasisKind::TypeA);
    pressure_type_a.resize(fluid_capacity, MomentBasisKind::TypeA);
    pressure_type_b.resize(fluid_capacity, MomentBasisKind::TypeB);
    fluid_only_type_a.resize(fluid_capacity, MomentBasisKind::TypeA);
    has_prepared_generation = false;
  }

  void release() noexcept {
    velocity_type_a.release();
    pressure_type_a.release();
    pressure_type_b.release();
    fluid_only_type_a.release();
    has_prepared_generation = false;
  }

  [[nodiscard]] size_type bytes() const noexcept {
    return velocity_type_a.bytes() + pressure_type_a.bytes() + pressure_type_b.bytes() +
           fluid_only_type_a.bytes();
  }

  void require_prepared() const {
    if (!has_prepared_generation) {
      std::cerr << "Moment operators must prepare matrices before computing operators" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

DeviceMomentMatrix::DeviceMomentMatrix()
    : impl_(std::make_unique<Impl>(SimulationConfig{})) {}

DeviceMomentMatrix::DeviceMomentMatrix(size_type fluid_capacity, SimulationConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  resize(fluid_capacity);
}

DeviceMomentMatrix::~DeviceMomentMatrix() = default;

DeviceMomentMatrix::DeviceMomentMatrix(DeviceMomentMatrix&& other) noexcept = default;

DeviceMomentMatrix& DeviceMomentMatrix::operator=(DeviceMomentMatrix&& other) noexcept =
    default;

void DeviceMomentMatrix::resize(size_type fluid_capacity) {
  impl_->resize(fluid_capacity);
}

void DeviceMomentMatrix::release() noexcept {
  impl_->release();
}

void DeviceMomentMatrix::set_config(SimulationConfig config) {
  impl_->config = std::move(config);
  impl_->has_prepared_generation = false;
}

const SimulationConfig& DeviceMomentMatrix::config() const noexcept {
  return impl_->config;
}

size_type DeviceMomentMatrix::bytes() const noexcept {
  return impl_->bytes();
}

void DeviceMomentMatrix::prepare_matrices(const FluidParticleSoA& fluid,
                                            const WallParticleSoA& walls,
                                            const NeighborListView& fluid_neighbors,
                                            const NeighborListView& wall_neighbors,
                                            unsigned long long geometry_generation) {
  impl_->geometry_generation = geometry_generation;
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          MomentMatrixKind::VelocityWallDirichletTypeA, impl_->config, geometry_generation),
      impl_->velocity_type_a.view);
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          MomentMatrixKind::PressureWallNeumannTypeA, impl_->config, geometry_generation),
      impl_->pressure_type_a.view);
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          MomentMatrixKind::PressureWallNeumannTypeB, impl_->config, geometry_generation),
      impl_->pressure_type_b.view);
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          MomentMatrixKind::FluidOnlyTypeA, impl_->config, geometry_generation),
      impl_->fluid_only_type_a.view);
  impl_->has_prepared_generation = true;
}

MomentMatrixView DeviceMomentMatrix::velocity_type_a() const {
  impl_->require_prepared();
  return make_public_view(impl_->velocity_type_a.view);
}

MomentMatrixView DeviceMomentMatrix::pressure_type_a() const {
  impl_->require_prepared();
  return make_public_view(impl_->pressure_type_a.view);
}

MomentMatrixView DeviceMomentMatrix::pressure_type_b() const {
  impl_->require_prepared();
  return make_public_view(impl_->pressure_type_b.view);
}

MomentMatrixView DeviceMomentMatrix::fluid_only_type_a() const {
  impl_->require_prepared();
  return make_public_view(impl_->fluid_only_type_a.view);
}

}  // namespace lsmps3d
