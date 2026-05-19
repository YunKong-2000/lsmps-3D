#include "lsmps3d/lsmps/moment_matrix.cuh"

#include <cstdlib>
#include <iostream>
#include <utility>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "lsmps3d/core/cuda_check.cuh"

namespace lsmps3d {
namespace {

constexpr int kThreadsPerBlock = 128;

enum class LsmpsPhysicalField : int {
  Pressure = 0,
  Velocity = 1,
};

enum class LsmpsBoundaryKind : int {
  None = 0,
  WallDirichlet = 1,
  WallPressureNeumann = 2,
};

enum class LsmpsOperatorMatrixKind : int {
  VelocityWallDirichletTypeA = 0,
  PressureWallNeumannTypeA = 1,
  PressureWallNeumannTypeB = 2,
  FluidOnlyTypeA = 3,
};

struct LsmpsMatrixRequest {
  LsmpsPhysicalField field{LsmpsPhysicalField::Velocity};
  LsmpsBoundaryKind boundary{LsmpsBoundaryKind::None};
  LsmpsBasisKind basis_kind{LsmpsBasisKind::TypeA};
  real support_radius{};
  real regularization{static_cast<real>(1.0e-8)};
  real wall_weight_scale{static_cast<real>(1)};
  unsigned long long geometry_generation{};
};

struct LsmpsMatrixCacheView {
  size_type particle_count{};
  int matrix_size{kLsmpsTypeABasis3DSize};
  LsmpsMatrixRequest request{};
  bool is_factorized{false};
  real* matrices{};
  real** matrix_ptrs{};
  int* info{};
  real* moment_trace{};
  real* regularization_added{};
  int* factorization_count{};
};

struct LsmpsOperatorWorkspaceView {
  size_type particle_count{};
  real* rhs{};
  real* solution{};
  real** rhs_ptrs{};
};

struct InternalCoefficientsView {
  real* coeffs{};
  int stride{kLsmpsTypeBBasis3DSize};
};

struct MatrixCache {
  MatrixCache() = default;
  MatrixCache(size_type particle_count, LsmpsBasisKind basis_kind) {
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

  void resize(size_type particle_count, LsmpsBasisKind basis_kind);
  void release() noexcept;
  [[nodiscard]] size_type bytes() const noexcept;

  LsmpsMatrixCacheView view{};
};

struct OperatorWorkspace {
  OperatorWorkspace() = default;
  OperatorWorkspace(size_type particle_count, LsmpsBasisKind basis_kind) {
    resize(particle_count, basis_kind);
  }
  ~OperatorWorkspace() {
    release();
  }

  OperatorWorkspace(const OperatorWorkspace&) = delete;
  OperatorWorkspace& operator=(const OperatorWorkspace&) = delete;

  OperatorWorkspace(OperatorWorkspace&& other) noexcept {
    using std::swap;
    swap(view, other.view);
    swap(matrix_size, other.matrix_size);
  }

  OperatorWorkspace& operator=(OperatorWorkspace&& other) noexcept = delete;

  void resize(size_type particle_count, LsmpsBasisKind basis_kind);
  void release() noexcept;
  [[nodiscard]] size_type bytes() const noexcept;

  LsmpsOperatorWorkspaceView view{};
  int matrix_size{kLsmpsTypeABasis3DSize};
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

void validate_matrix_request(const LsmpsMatrixRequest& request) {
  if (request.support_radius <= static_cast<real>(0)) {
    std::cerr << "LSMPS support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (request.regularization < static_cast<real>(0)) {
    std::cerr << "LSMPS regularization must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (request.wall_weight_scale < static_cast<real>(0)) {
    std::cerr << "LSMPS wall weight scale must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_rhs_parameters(real support_radius, real pressure_density) {
  if (support_radius <= static_cast<real>(0)) {
    std::cerr << "LSMPS RHS support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pressure_density < static_cast<real>(0)) {
    std::cerr << "LSMPS pressure density must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_cache(const LsmpsMatrixCacheView& cache,
                    size_type particle_count,
                    LsmpsBasisKind basis_kind) {
  const int matrix_size = lsmps_basis_size(basis_kind);
  if (cache.particle_count < particle_count || cache.matrix_size != matrix_size) {
    std::cerr << "Invalid LSMPS matrix cache" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particle_count == 0) {
    return;
  }
  if (cache.matrices == nullptr || cache.matrix_ptrs == nullptr || cache.info == nullptr ||
      cache.moment_trace == nullptr || cache.regularization_added == nullptr ||
      cache.factorization_count == nullptr) {
    std::cerr << "Invalid LSMPS matrix cache buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_workspace(const LsmpsOperatorWorkspaceView& workspace,
                        size_type particle_count,
                        LsmpsBasisKind basis_kind) {
  const int matrix_size = lsmps_basis_size(basis_kind);
  if (workspace.particle_count < particle_count) {
    std::cerr << "Invalid LSMPS operator workspace" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particle_count == 0) {
    return;
  }
  if (workspace.rhs == nullptr || workspace.solution == nullptr || workspace.rhs_ptrs == nullptr) {
    std::cerr << "Invalid LSMPS operator workspace buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  (void)matrix_size;
}

bool same_matrix_request(const LsmpsMatrixRequest& lhs, const LsmpsMatrixRequest& rhs) {
  return lhs.field == rhs.field && lhs.boundary == rhs.boundary &&
         lhs.basis_kind == rhs.basis_kind && lhs.support_radius == rhs.support_radius &&
         lhs.regularization == rhs.regularization &&
         lhs.wall_weight_scale == rhs.wall_weight_scale &&
         lhs.geometry_generation == rhs.geometry_generation;
}

__host__ __device__ real lsmps_weight(real distance, real support_radius) {
  if (distance >= support_radius) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__host__ __device__ int basis_offset(LsmpsBasisKind basis_kind) {
  return basis_kind == LsmpsBasisKind::TypeB ? 1 : 0;
}

__device__ int basis_vector(real dx,
                            real dy,
                            real dz,
                            LsmpsBasisKind basis_kind,
                            real support_radius,
                            real basis[kLsmpsMaxBasis3DSize]) {
  const real sx = dx / support_radius;
  const real sy = dy / support_radius;
  const real sz = dz / support_radius;
  int offset = 0;
  if (basis_kind == LsmpsBasisKind::TypeB) {
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
                                   LsmpsBasisKind basis_kind,
                                   real support_radius,
                                   real q[kLsmpsMaxBasis3DSize]) {
  const real sx = dx / support_radius;
  const real sy = dy / support_radius;
  const real sz = dz / support_radius;
  int offset = 0;
  if (basis_kind == LsmpsBasisKind::TypeB) {
    q[offset++] = static_cast<real>(0);
  }
  q[offset++] = normal_x;
  q[offset++] = normal_y;
  q[offset++] = normal_z;
  q[offset++] = static_cast<real>(2) * sx * normal_x;
  q[offset++] = static_cast<real>(2) * sy * normal_y;
  q[offset++] = static_cast<real>(2) * sz * normal_z;
  q[offset++] = sy * normal_x + sx * normal_y;
  q[offset++] = sz * normal_y + sy * normal_z;
  q[offset++] = sx * normal_z + sz * normal_x;
  return offset;
}

__device__ void accumulate_matrix_basis(real dx,
                                        real dy,
                                        real dz,
                                        real weight_scale,
                                        const LsmpsMatrixRequest& config,
                                        real* matrix) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }

  const real distance = sqrt(distance_squared);
  const real weight = weight_scale * lsmps_weight(distance, config.support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }

  real basis[kLsmpsMaxBasis3DSize]{};
  const int n = basis_vector(dx, dy, dz, config.basis_kind, config.support_radius, basis);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      matrix[row + col * n] += weight * basis[row] * basis[col];
    }
  }
}

__device__ void accumulate_neumann_matrix(real dx,
                                          real dy,
                                          real dz,
                                          real normal_x,
                                          real normal_y,
                                          real normal_z,
                                          const LsmpsMatrixRequest& config,
                                          real* matrix) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }

  const real distance = sqrt(distance_squared);
  const real weight = config.wall_weight_scale * lsmps_weight(distance, config.support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }

  real q[kLsmpsMaxBasis3DSize]{};
  const int n = wall_neumann_vector(
      dx, dy, dz, normal_x, normal_y, normal_z, config.basis_kind, config.support_radius, q);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      matrix[row + col * n] += weight * q[row] * q[col];
    }
  }
}

__device__ void accumulate_rhs_basis(real dx,
                                     real dy,
                                     real dz,
                                     real delta,
                                     real weight_scale,
                                     LsmpsBasisKind basis_kind,
                                     real support_radius,
                                     real* rhs) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }

  const real distance = sqrt(distance_squared);
  const real weight = weight_scale * lsmps_weight(distance, support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }

  real basis[kLsmpsMaxBasis3DSize]{};
  const int n = basis_vector(dx, dy, dz, basis_kind, support_radius, basis);
  for (int row = 0; row < n; ++row) {
    rhs[row] += weight * basis[row] * delta;
  }
}

__device__ void accumulate_pressure_neumann_rhs(real dx,
                                                real dy,
                                                real dz,
                                                real normal_x,
                                                real normal_y,
                                                real normal_z,
                                                LsmpsBasisKind basis_kind,
                                                real support_radius,
                                                real pressure_density,
                                                Vec3 gravity,
                                                real* rhs) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }

  const real distance = sqrt(distance_squared);
  const real weight = lsmps_weight(distance, support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }

  real q[kLsmpsMaxBasis3DSize]{};
  const int n = wall_neumann_vector(
      dx, dy, dz, normal_x, normal_y, normal_z, basis_kind, support_radius, q);
  const real pressure_normal_gradient =
      support_radius * pressure_density *
      (gravity.x * normal_x + gravity.y * normal_y + gravity.z * normal_z);
  for (int row = 0; row < n; ++row) {
    rhs[row] += weight * q[row] * pressure_normal_gradient;
  }
}

__global__ void build_matrix_cache_kernel(const FluidParticleSoA fluid,
                                          const WallParticleSoA walls,
                                          const NeighborListView fluid_neighbors,
                                          const NeighborListView wall_neighbors,
                                          const LsmpsMatrixRequest config,
                                          LsmpsMatrixCacheView cache) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const int n = cache.matrix_size;
  real* matrix = cache.matrices + i * n * n;
  for (int entry = 0; entry < n * n; ++entry) {
    matrix[entry] = static_cast<real>(0);
  }

  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];
  for (index_t out = fluid_neighbors.offsets[i]; out < fluid_neighbors.offsets[i + 1]; ++out) {
    const index_t j = fluid_neighbors.indices[out];
    accumulate_matrix_basis(
        fluid.x[j] - px, fluid.y[j] - py, fluid.z[j] - pz, static_cast<real>(1), config, matrix);
  }

  if (config.boundary != LsmpsBoundaryKind::None) {
    for (index_t out = wall_neighbors.offsets[i]; out < wall_neighbors.offsets[i + 1]; ++out) {
      const index_t j = wall_neighbors.indices[out];
      const real dx = walls.x[j] - px;
      const real dy = walls.y[j] - py;
      const real dz = walls.z[j] - pz;
      if (config.boundary == LsmpsBoundaryKind::WallPressureNeumann) {
        accumulate_neumann_matrix(
            dx, dy, dz, walls.normal_x[j], walls.normal_y[j], walls.normal_z[j], config, matrix);
      } else {
        accumulate_matrix_basis(dx, dy, dz, static_cast<real>(1), config, matrix);
      }
    }
  }

  real trace = static_cast<real>(0);
  for (int row = 0; row < n; ++row) {
    trace += matrix[row + row * n];
  }
  const real regularization = config.regularization *
                              (trace > static_cast<real>(0) ? trace / static_cast<real>(n)
                                                            : static_cast<real>(1));
  for (int row = 0; row < n; ++row) {
    matrix[row + row * n] += regularization;
  }
  cache.moment_trace[i] = trace;
  cache.regularization_added[i] = regularization;
}

__global__ void setup_matrix_pointer_kernel(LsmpsMatrixCacheView cache) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cache.particle_count) {
    return;
  }

  const int n = cache.matrix_size;
  cache.matrix_ptrs[i] = cache.matrices + i * n * n;
}

__global__ void increment_factorization_count_kernel(LsmpsMatrixCacheView cache) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && cache.factorization_count != nullptr) {
    ++cache.factorization_count[0];
  }
}

__global__ void setup_rhs_pointer_kernel(LsmpsOperatorWorkspaceView workspace, int matrix_size) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= workspace.particle_count) {
    return;
  }

  workspace.rhs_ptrs[i] = workspace.rhs + i * matrix_size;
}

__global__ void build_rhs_kernel(const FluidParticleSoA fluid,
                                 const WallParticleSoA walls,
                                 const NeighborListView fluid_neighbors,
                                 const NeighborListView wall_neighbors,
                                 const real* field,
                                 const real* wall_field,
                                 LsmpsBasisKind basis_kind,
                                 LsmpsBoundaryKind boundary,
                                 real support_radius,
                                 real reference_value,
                                 real pressure_density,
                                 Vec3 gravity,
                                 LsmpsOperatorWorkspaceView workspace) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const int n = lsmps_basis_size(basis_kind);
  real* rhs = workspace.rhs + i * n;
  for (int row = 0; row < n; ++row) {
    rhs[row] = static_cast<real>(0);
  }

  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];
  const real center_value = field[i];
  for (index_t out = fluid_neighbors.offsets[i]; out < fluid_neighbors.offsets[i + 1]; ++out) {
    const index_t j = fluid_neighbors.indices[out];
    const real delta =
        basis_kind == LsmpsBasisKind::TypeB ? field[j] - reference_value : field[j] - center_value;
    accumulate_rhs_basis(fluid.x[j] - px,
                         fluid.y[j] - py,
                         fluid.z[j] - pz,
                         delta,
                         static_cast<real>(1),
                         basis_kind,
                         support_radius,
                         rhs);
  }

  if (boundary != LsmpsBoundaryKind::None) {
    for (index_t out = wall_neighbors.offsets[i]; out < wall_neighbors.offsets[i + 1]; ++out) {
      const index_t j = wall_neighbors.indices[out];
      const real dx = walls.x[j] - px;
      const real dy = walls.y[j] - py;
      const real dz = walls.z[j] - pz;
      if (boundary == LsmpsBoundaryKind::WallPressureNeumann) {
        accumulate_pressure_neumann_rhs(dx,
                                        dy,
                                        dz,
                                        walls.normal_x[j],
                                        walls.normal_y[j],
                                        walls.normal_z[j],
                                        basis_kind,
                                        support_radius,
                                        pressure_density,
                                        gravity,
                                        rhs);
      } else {
        const real wall_value = wall_field != nullptr ? wall_field[j] : center_value;
        const real delta = basis_kind == LsmpsBasisKind::TypeB
                               ? wall_value - reference_value
                               : wall_value - center_value;
        accumulate_rhs_basis(
            dx, dy, dz, delta, static_cast<real>(1), basis_kind, support_radius, rhs);
      }
    }
  }
}

__global__ void evaluate_lsmps_operators_kernel(InternalCoefficientsView coefficients,
                                                size_type particle_count,
                                                LsmpsBasisKind basis_kind,
                                                real support_radius,
                                                real* gradient_x,
                                                real* gradient_y,
                                                real* gradient_z,
                                                real* laplacian) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= particle_count) {
    return;
  }

  const int offset = basis_offset(basis_kind);
  const real* coeff = coefficients.coeffs + i * coefficients.stride;
  if (gradient_x != nullptr) {
    gradient_x[i] = coeff[offset] / support_radius;
  }
  if (gradient_y != nullptr) {
    gradient_y[i] = coeff[offset + 1] / support_radius;
  }
  if (gradient_z != nullptr) {
    gradient_z[i] = coeff[offset + 2] / support_radius;
  }
  if (laplacian != nullptr) {
    const real inv_scale_squared = static_cast<real>(1) / (support_radius * support_radius);
    laplacian[i] =
        static_cast<real>(2) * (coeff[offset + 3] + coeff[offset + 4] + coeff[offset + 5]) *
        inv_scale_squared;
  }
}

__global__ void evaluate_lsmps_divergence_kernel(InternalCoefficientsView velocity_x_coefficients,
                                                 InternalCoefficientsView velocity_y_coefficients,
                                                 InternalCoefficientsView velocity_z_coefficients,
                                                 size_type particle_count,
                                                 LsmpsBasisKind basis_kind,
                                                 real support_radius,
                                                 real* divergence) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= particle_count) {
    return;
  }

  const int offset = basis_offset(basis_kind);
  divergence[i] = (velocity_x_coefficients.coeffs[i * velocity_x_coefficients.stride + offset] +
                   velocity_y_coefficients.coeffs[i * velocity_y_coefficients.stride + offset + 1] +
                   velocity_z_coefficients.coeffs[i * velocity_z_coefficients.stride + offset + 2]) /
                  support_radius;
}

}  // namespace

void MatrixCache::resize(size_type particle_count, LsmpsBasisKind basis_kind) {
  const int matrix_size = lsmps_basis_size(basis_kind);
  if (particle_count == view.particle_count && matrix_size == view.matrix_size) {
    return;
  }

  release();
  view.particle_count = particle_count;
  view.matrix_size = matrix_size;
  view.request.basis_kind = basis_kind;
  device_alloc(view.matrices, particle_count * matrix_size * matrix_size);
  device_alloc(view.matrix_ptrs, particle_count);
  device_alloc(view.info, particle_count);
  device_alloc(view.moment_trace, particle_count);
  device_alloc(view.regularization_added, particle_count);
  device_alloc(view.factorization_count, 1);
  if (view.factorization_count != nullptr) {
    LSMPS3D_CUDA_CHECK(cudaMemset(view.factorization_count, 0, sizeof(int)));
  }
}

void MatrixCache::release() noexcept {
  device_free(view.matrices);
  device_free(view.matrix_ptrs);
  device_free(view.info);
  device_free(view.moment_trace);
  device_free(view.regularization_added);
  device_free(view.factorization_count);
  view.particle_count = 0;
  view.matrix_size = kLsmpsTypeABasis3DSize;
  view.request = {};
  view.is_factorized = false;
}

size_type MatrixCache::bytes() const noexcept {
  const size_type n = static_cast<size_type>(view.matrix_size);
  return view.particle_count * (n * n * sizeof(real) + sizeof(real*) + sizeof(int) +
                                2 * sizeof(real)) +
         sizeof(int);
}

void OperatorWorkspace::resize(size_type particle_count, LsmpsBasisKind basis_kind) {
  const int basis_size = lsmps_basis_size(basis_kind);
  if (particle_count == view.particle_count && basis_size == matrix_size) {
    return;
  }

  release();
  view.particle_count = particle_count;
  matrix_size = basis_size;
  device_alloc(view.rhs, particle_count * basis_size);
  device_alloc(view.solution, particle_count * basis_size);
  device_alloc(view.rhs_ptrs, particle_count);
}

void OperatorWorkspace::release() noexcept {
  device_free(view.rhs);
  device_free(view.solution);
  device_free(view.rhs_ptrs);
  view.particle_count = 0;
  matrix_size = kLsmpsTypeABasis3DSize;
}

size_type OperatorWorkspace::bytes() const noexcept {
  return view.particle_count * (2 * static_cast<size_type>(matrix_size) * sizeof(real) +
                                sizeof(real*));
}

void compute_pressure_gradient_internal(const FluidParticleSoA& fluid,
                                        const WallParticleSoA& walls,
                                        const NeighborListView& fluid_neighbors,
                                        const NeighborListView& wall_neighbors,
                                        const real* pressure,
                                        const LsmpsMatrixCacheView& matrix,
                                        real density,
                                        Vec3 gravity,
                                        LsmpsOperatorWorkspaceView workspace,
                                        real* gradient_x,
                                        real* gradient_y,
                                        real* gradient_z);

void compute_pressure_laplacian_internal(const FluidParticleSoA& fluid,
                                         const WallParticleSoA& walls,
                                         const NeighborListView& fluid_neighbors,
                                         const NeighborListView& wall_neighbors,
                                         const real* pressure,
                                         const LsmpsMatrixCacheView& matrix,
                                         real density,
                                         Vec3 gravity,
                                         LsmpsOperatorWorkspaceView workspace,
                                         real* laplacian);

void compute_velocity_gradient_internal(const FluidParticleSoA& fluid,
                                        const WallParticleSoA& walls,
                                        const NeighborListView& fluid_neighbors,
                                        const NeighborListView& wall_neighbors,
                                        const real* velocity_component,
                                        const real* wall_velocity_component,
                                        const LsmpsMatrixCacheView& matrix,
                                        LsmpsOperatorWorkspaceView workspace,
                                        real* gradient_x,
                                        real* gradient_y,
                                        real* gradient_z);

void compute_velocity_laplacian_internal(const FluidParticleSoA& fluid,
                                         const WallParticleSoA& walls,
                                         const NeighborListView& fluid_neighbors,
                                         const NeighborListView& wall_neighbors,
                                         const real* velocity_component,
                                         const real* wall_velocity_component,
                                         const LsmpsMatrixCacheView& matrix,
                                         LsmpsOperatorWorkspaceView workspace,
                                         real* laplacian);

void compute_velocity_divergence_internal(const FluidParticleSoA& fluid,
                                          const WallParticleSoA& walls,
                                          const NeighborListView& fluid_neighbors,
                                          const NeighborListView& wall_neighbors,
                                          const LsmpsMatrixCacheView& matrix,
                                          LsmpsOperatorWorkspaceView workspace_x,
                                          LsmpsOperatorWorkspaceView workspace_y,
                                          LsmpsOperatorWorkspaceView workspace_z,
                                          real* divergence);

void ensure_matrix_internal(const FluidParticleSoA& fluid,
                            const WallParticleSoA& walls,
                            const NeighborListView& fluid_neighbors,
                            const NeighborListView& wall_neighbors,
                            const LsmpsMatrixRequest& request,
                            LsmpsMatrixCacheView& cache) {
  validate_matrix_request(request);
  validate_cache(cache, fluid.count, request.basis_kind);
  if (fluid.count > fluid_neighbors.particle_count || fluid.count > wall_neighbors.particle_count) {
    std::cerr << "LSMPS matrix neighbor lists are smaller than fluid particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (same_matrix_request(cache.request, request) && cache.is_factorized) {
    return;
  }
  if (fluid.count == 0) {
    cache.request = request;
    cache.is_factorized = true;
    return;
  }

  build_matrix_cache_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, walls, fluid_neighbors, wall_neighbors, request, cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  setup_matrix_pointer_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  cache.request = request;
  cache.is_factorized = false;

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
  LSMPS3D_CUSOLVER_CHECK(cusolverDnDestroy(solver));
  increment_factorization_count_kernel<<<1, 1>>>(cache);
  LSMPS3D_CUDA_KERNEL_CHECK();
  cache.is_factorized = true;
}

void build_lsmps_rhs_internal(const FluidParticleSoA& fluid,
                              const WallParticleSoA& walls,
                              const NeighborListView& fluid_neighbors,
                              const NeighborListView& wall_neighbors,
                              const real* field,
                              const real* wall_field,
                              LsmpsBasisKind basis_kind,
                              LsmpsBoundaryKind boundary,
                              real support_radius,
                              real reference_value,
                              real pressure_density,
                              Vec3 gravity,
                              LsmpsOperatorWorkspaceView workspace) {
  validate_rhs_parameters(support_radius, pressure_density);
  validate_workspace(workspace, fluid.count, basis_kind);
  if (fluid.count > fluid_neighbors.particle_count || fluid.count > wall_neighbors.particle_count) {
    std::cerr << "LSMPS RHS neighbor lists are smaller than fluid particle count" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (field == nullptr) {
    std::cerr << "LSMPS field values are required" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (boundary == LsmpsBoundaryKind::WallDirichlet && walls.count > 0 && wall_field == nullptr) {
    std::cerr << "LSMPS Dirichlet wall RHS requires wall field values" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count == 0) {
    return;
  }

  build_rhs_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(fluid,
                                                                   walls,
                                                                   fluid_neighbors,
                                                                   wall_neighbors,
                                                                   field,
                                                                   wall_field,
                                                                   basis_kind,
                                                                   boundary,
                                                                   support_radius,
                                                                   reference_value,
                                                                   pressure_density,
                                                                   gravity,
                                                                   workspace);
  LSMPS3D_CUDA_KERNEL_CHECK();
  setup_rhs_pointer_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      workspace, lsmps_basis_size(basis_kind));
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void solve_with_matrix_internal(const LsmpsMatrixCacheView& cache, LsmpsOperatorWorkspaceView workspace) {
  validate_cache(cache, cache.particle_count, cache.request.basis_kind);
  validate_workspace(workspace, cache.particle_count, cache.request.basis_kind);
  if (!cache.is_factorized) {
    std::cerr << "LSMPS matrix cache must be factorized before solve" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (cache.particle_count == 0) {
    return;
  }

  cusolverDnHandle_t solver{};
  LSMPS3D_CUSOLVER_CHECK(cusolverDnCreate(&solver));
  const int batch_size = static_cast<int>(cache.particle_count);
  const int n = cache.matrix_size;
#ifndef LSMPS3D_USE_DOUBLE
  LSMPS3D_CUSOLVER_CHECK(cusolverDnSpotrsBatched(solver,
                                                 CUBLAS_FILL_MODE_LOWER,
                                                 n,
                                                 1,
                                                 cache.matrix_ptrs,
                                                 n,
                                                 workspace.rhs_ptrs,
                                                 n,
                                                 cache.info,
                                                 batch_size));
#else
  LSMPS3D_CUSOLVER_CHECK(cusolverDnDpotrsBatched(solver,
                                                 CUBLAS_FILL_MODE_LOWER,
                                                 n,
                                                 1,
                                                 cache.matrix_ptrs,
                                                 n,
                                                 workspace.rhs_ptrs,
                                                 n,
                                                 cache.info,
                                                 batch_size));
#endif
  LSMPS3D_CUSOLVER_CHECK(cusolverDnDestroy(solver));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(workspace.solution,
                                workspace.rhs,
                                cache.particle_count * n * sizeof(real),
                                cudaMemcpyDeviceToDevice));
}

void evaluate_lsmps_operators_internal(InternalCoefficientsView coefficients,
                                       size_type particle_count,
                                       LsmpsBasisKind basis_kind,
                                       real support_radius,
                                       real* gradient_x,
                                       real* gradient_y,
                                       real* gradient_z,
                                       real* laplacian) {
  if (support_radius <= static_cast<real>(0)) {
    std::cerr << "LSMPS operator support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particle_count == 0) {
    return;
  }
  if (coefficients.coeffs == nullptr || coefficients.stride < lsmps_basis_size(basis_kind)) {
    std::cerr << "Invalid LSMPS coefficient buffer" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (gradient_x == nullptr && gradient_y == nullptr && gradient_z == nullptr &&
      laplacian == nullptr) {
    std::cerr << "LSMPS operator evaluation requires at least one output buffer" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  evaluate_lsmps_operators_kernel<<<block_count(particle_count), kThreadsPerBlock>>>(
      coefficients, particle_count, basis_kind, support_radius, gradient_x, gradient_y, gradient_z, laplacian);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void evaluate_lsmps_divergence_internal(InternalCoefficientsView velocity_x_coefficients,
                                        InternalCoefficientsView velocity_y_coefficients,
                                        InternalCoefficientsView velocity_z_coefficients,
                                        size_type particle_count,
                                        LsmpsBasisKind basis_kind,
                                        real support_radius,
                                        real* divergence) {
  if (support_radius <= static_cast<real>(0)) {
    std::cerr << "LSMPS divergence support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (divergence == nullptr || velocity_x_coefficients.coeffs == nullptr ||
      velocity_y_coefficients.coeffs == nullptr || velocity_z_coefficients.coeffs == nullptr) {
    std::cerr << "Invalid LSMPS divergence buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (particle_count == 0) {
    return;
  }

  evaluate_lsmps_divergence_kernel<<<block_count(particle_count), kThreadsPerBlock>>>(
      velocity_x_coefficients,
      velocity_y_coefficients,
      velocity_z_coefficients,
      particle_count,
      basis_kind,
      support_radius,
      divergence);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

void require_matrix_kind(const LsmpsMatrixCacheView& matrix,
                         LsmpsPhysicalField field,
                         LsmpsBoundaryKind boundary) {
  if (matrix.request.field != field || matrix.request.boundary != boundary) {
    std::cerr << "LSMPS operator received incompatible matrix cache" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void compute_scalar_operator_internal(const FluidParticleSoA& fluid,
                                      const WallParticleSoA& walls,
                                      const NeighborListView& fluid_neighbors,
                                      const NeighborListView& wall_neighbors,
                                      const real* field,
                                      const real* wall_field,
                                      const LsmpsMatrixCacheView& matrix,
                                      LsmpsBoundaryKind boundary,
                                      real reference_value,
                                      real density,
                                      Vec3 gravity,
                                      LsmpsOperatorWorkspaceView workspace,
                                      real* gradient_x,
                                      real* gradient_y,
                                      real* gradient_z,
                                      real* laplacian) {
  build_lsmps_rhs_internal(fluid,
                           walls,
                           fluid_neighbors,
                           wall_neighbors,
                           field,
                           wall_field,
                           matrix.request.basis_kind,
                           boundary,
                           matrix.request.support_radius,
                           reference_value,
                           density,
                           gravity,
                           workspace);
  solve_with_matrix_internal(matrix, workspace);
  evaluate_lsmps_operators_internal(InternalCoefficientsView{workspace.solution, matrix.matrix_size},
                                   fluid.count,
                                   matrix.request.basis_kind,
                                   matrix.request.support_radius,
                                   gradient_x,
                                   gradient_y,
                                   gradient_z,
                                   laplacian);
}

LsmpsMatrixRequest make_matrix_request(LsmpsOperatorMatrixKind kind,
                                       const SimulationConfig& config,
                                       unsigned long long geometry_generation) {
  switch (kind) {
    case LsmpsOperatorMatrixKind::VelocityWallDirichletTypeA:
      return LsmpsMatrixRequest{LsmpsPhysicalField::Velocity,
                                LsmpsBoundaryKind::WallDirichlet,
                                LsmpsBasisKind::TypeA,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case LsmpsOperatorMatrixKind::PressureWallNeumannTypeA:
      return LsmpsMatrixRequest{LsmpsPhysicalField::Pressure,
                                LsmpsBoundaryKind::WallPressureNeumann,
                                LsmpsBasisKind::TypeA,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case LsmpsOperatorMatrixKind::PressureWallNeumannTypeB:
      return LsmpsMatrixRequest{LsmpsPhysicalField::Pressure,
                                LsmpsBoundaryKind::WallPressureNeumann,
                                LsmpsBasisKind::TypeB,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
    case LsmpsOperatorMatrixKind::FluidOnlyTypeA:
      return LsmpsMatrixRequest{LsmpsPhysicalField::Velocity,
                                LsmpsBoundaryKind::None,
                                LsmpsBasisKind::TypeA,
                                config.support_radius,
                                config.lsmps_regularization,
                                config.lsmps_wall_weight_scale,
                                geometry_generation};
  }

  std::cerr << "Unknown LSMPS matrix kind" << std::endl;
  std::exit(EXIT_FAILURE);
}

struct DeviceLsmpsOperators::Impl {
  explicit Impl(SimulationConfig initial_config) : config(std::move(initial_config)) {}

  SimulationConfig config{};
  unsigned long long geometry_generation{};
  bool has_prepared_generation{false};
  MatrixCache velocity_type_a;
  MatrixCache pressure_type_a;
  MatrixCache pressure_type_b;
  MatrixCache fluid_only_type_a;
  OperatorWorkspace workspace_a;
  OperatorWorkspace workspace_b;
  OperatorWorkspace workspace_c;

  void resize(size_type fluid_capacity) {
    velocity_type_a.resize(fluid_capacity, LsmpsBasisKind::TypeA);
    pressure_type_a.resize(fluid_capacity, LsmpsBasisKind::TypeA);
    pressure_type_b.resize(fluid_capacity, LsmpsBasisKind::TypeB);
    fluid_only_type_a.resize(fluid_capacity, LsmpsBasisKind::TypeA);
    workspace_a.resize(fluid_capacity, LsmpsBasisKind::TypeB);
    workspace_b.resize(fluid_capacity, LsmpsBasisKind::TypeA);
    workspace_c.resize(fluid_capacity, LsmpsBasisKind::TypeA);
    has_prepared_generation = false;
  }

  void release() noexcept {
    velocity_type_a.release();
    pressure_type_a.release();
    pressure_type_b.release();
    fluid_only_type_a.release();
    workspace_a.release();
    workspace_b.release();
    workspace_c.release();
    has_prepared_generation = false;
  }

  [[nodiscard]] size_type bytes() const noexcept {
    return velocity_type_a.bytes() + pressure_type_a.bytes() + pressure_type_b.bytes() +
           fluid_only_type_a.bytes() + workspace_a.bytes() + workspace_b.bytes() +
           workspace_c.bytes();
  }

  void require_prepared() const {
    if (!has_prepared_generation) {
      std::cerr << "LSMPS operators must prepare matrices before computing operators" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

DeviceLsmpsOperators::DeviceLsmpsOperators()
    : impl_(std::make_unique<Impl>(SimulationConfig{})) {}

DeviceLsmpsOperators::DeviceLsmpsOperators(size_type fluid_capacity, SimulationConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  resize(fluid_capacity);
}

DeviceLsmpsOperators::~DeviceLsmpsOperators() = default;

DeviceLsmpsOperators::DeviceLsmpsOperators(DeviceLsmpsOperators&& other) noexcept = default;

DeviceLsmpsOperators& DeviceLsmpsOperators::operator=(DeviceLsmpsOperators&& other) noexcept =
    default;

void DeviceLsmpsOperators::resize(size_type fluid_capacity) {
  impl_->resize(fluid_capacity);
}

void DeviceLsmpsOperators::release() noexcept {
  impl_->release();
}

void DeviceLsmpsOperators::set_config(SimulationConfig config) {
  impl_->config = std::move(config);
  impl_->has_prepared_generation = false;
}

const SimulationConfig& DeviceLsmpsOperators::config() const noexcept {
  return impl_->config;
}

size_type DeviceLsmpsOperators::bytes() const noexcept {
  return impl_->bytes();
}

void DeviceLsmpsOperators::prepare_matrices(const FluidParticleSoA& fluid,
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
          LsmpsOperatorMatrixKind::VelocityWallDirichletTypeA, impl_->config, geometry_generation),
      impl_->velocity_type_a.view);
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          LsmpsOperatorMatrixKind::PressureWallNeumannTypeA, impl_->config, geometry_generation),
      impl_->pressure_type_a.view);
  ensure_matrix_internal(
      fluid,
      walls,
      fluid_neighbors,
      wall_neighbors,
      make_matrix_request(
          LsmpsOperatorMatrixKind::PressureWallNeumannTypeB, impl_->config, geometry_generation),
      impl_->pressure_type_b.view);
  impl_->has_prepared_generation = true;
}

void DeviceLsmpsOperators::compute_pressure_gradient(const FluidParticleSoA& fluid,
                                                     const WallParticleSoA& walls,
                                                     const NeighborListView& fluid_neighbors,
                                                     const NeighborListView& wall_neighbors,
                                                     const real* pressure,
                                                     real* gradient_x,
                                                     real* gradient_y,
                                                     real* gradient_z) {
  impl_->require_prepared();
  compute_pressure_gradient_internal(fluid,
                                     walls,
                                     fluid_neighbors,
                                     wall_neighbors,
                                     pressure,
                                     impl_->pressure_type_a.view,
                                     impl_->config.density,
                                     impl_->config.gravity,
                                     impl_->workspace_a.view,
                                     gradient_x,
                                     gradient_y,
                                     gradient_z);
}

void DeviceLsmpsOperators::compute_near_surface_pressure_gradient(
    const FluidParticleSoA& fluid,
    const WallParticleSoA& walls,
    const NeighborListView& fluid_neighbors,
    const NeighborListView& wall_neighbors,
    const real* pressure,
    real* gradient_x,
    real* gradient_y,
    real* gradient_z) {
  impl_->require_prepared();
  compute_pressure_gradient_internal(fluid,
                                     walls,
                                     fluid_neighbors,
                                     wall_neighbors,
                                     pressure,
                                     impl_->pressure_type_b.view,
                                     impl_->config.density,
                                     impl_->config.gravity,
                                     impl_->workspace_a.view,
                                     gradient_x,
                                     gradient_y,
                                     gradient_z);
}

void DeviceLsmpsOperators::compute_pressure_laplacian(const FluidParticleSoA& fluid,
                                                      const WallParticleSoA& walls,
                                                      const NeighborListView& fluid_neighbors,
                                                      const NeighborListView& wall_neighbors,
                                                      const real* pressure,
                                                      real* laplacian) {
  impl_->require_prepared();
  compute_pressure_laplacian_internal(fluid,
                                      walls,
                                      fluid_neighbors,
                                      wall_neighbors,
                                      pressure,
                                      impl_->pressure_type_a.view,
                                      impl_->config.density,
                                      impl_->config.gravity,
                                      impl_->workspace_a.view,
                                      laplacian);
}

void DeviceLsmpsOperators::compute_velocity_gradient(const FluidParticleSoA& fluid,
                                                     const WallParticleSoA& walls,
                                                     const NeighborListView& fluid_neighbors,
                                                     const NeighborListView& wall_neighbors,
                                                     const real* velocity_component,
                                                     const real* wall_velocity_component,
                                                     real* gradient_x,
                                                     real* gradient_y,
                                                     real* gradient_z) {
  impl_->require_prepared();
  compute_velocity_gradient_internal(fluid,
                                     walls,
                                     fluid_neighbors,
                                     wall_neighbors,
                                     velocity_component,
                                     wall_velocity_component,
                                     impl_->velocity_type_a.view,
                                     impl_->workspace_b.view,
                                     gradient_x,
                                     gradient_y,
                                     gradient_z);
}

void DeviceLsmpsOperators::compute_velocity_laplacian(const FluidParticleSoA& fluid,
                                                      const WallParticleSoA& walls,
                                                      const NeighborListView& fluid_neighbors,
                                                      const NeighborListView& wall_neighbors,
                                                      const real* velocity_component,
                                                      const real* wall_velocity_component,
                                                      real* laplacian) {
  impl_->require_prepared();
  compute_velocity_laplacian_internal(fluid,
                                      walls,
                                      fluid_neighbors,
                                      wall_neighbors,
                                      velocity_component,
                                      wall_velocity_component,
                                      impl_->velocity_type_a.view,
                                      impl_->workspace_b.view,
                                      laplacian);
}

void DeviceLsmpsOperators::compute_velocity_divergence(const FluidParticleSoA& fluid,
                                                       const WallParticleSoA& walls,
                                                       const NeighborListView& fluid_neighbors,
                                                       const NeighborListView& wall_neighbors,
                                                       real* divergence) {
  impl_->require_prepared();
  compute_velocity_divergence_internal(fluid,
                                       walls,
                                       fluid_neighbors,
                                       wall_neighbors,
                                       impl_->velocity_type_a.view,
                                       impl_->workspace_b.view,
                                       impl_->workspace_c.view,
                                       impl_->workspace_a.view,
                                       divergence);
}

void compute_pressure_gradient_internal(const FluidParticleSoA& fluid,
                                        const WallParticleSoA& walls,
                                        const NeighborListView& fluid_neighbors,
                                        const NeighborListView& wall_neighbors,
                                        const real* pressure,
                                        const LsmpsMatrixCacheView& matrix,
                                        real density,
                                        Vec3 gravity,
                                        LsmpsOperatorWorkspaceView workspace,
                                        real* gradient_x,
                                        real* gradient_y,
                                        real* gradient_z) {
  require_matrix_kind(
      matrix, LsmpsPhysicalField::Pressure, LsmpsBoundaryKind::WallPressureNeumann);
  compute_scalar_operator_internal(fluid,
                                   walls,
                                   fluid_neighbors,
                                   wall_neighbors,
                                   pressure,
                                   {},
                                   matrix,
                                   LsmpsBoundaryKind::WallPressureNeumann,
                                   static_cast<real>(0),
                                   density,
                                   gravity,
                                   workspace,
                                   gradient_x,
                                   gradient_y,
                                   gradient_z,
                                   nullptr);
}

void compute_pressure_laplacian_internal(const FluidParticleSoA& fluid,
                                         const WallParticleSoA& walls,
                                         const NeighborListView& fluid_neighbors,
                                         const NeighborListView& wall_neighbors,
                                         const real* pressure,
                                         const LsmpsMatrixCacheView& matrix,
                                         real density,
                                         Vec3 gravity,
                                         LsmpsOperatorWorkspaceView workspace,
                                         real* laplacian) {
  require_matrix_kind(
      matrix, LsmpsPhysicalField::Pressure, LsmpsBoundaryKind::WallPressureNeumann);
  compute_scalar_operator_internal(fluid,
                                   walls,
                                   fluid_neighbors,
                                   wall_neighbors,
                                   pressure,
                                   {},
                                   matrix,
                                   LsmpsBoundaryKind::WallPressureNeumann,
                                   static_cast<real>(0),
                                   density,
                                   gravity,
                                   workspace,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   laplacian);
}

void compute_velocity_gradient_internal(const FluidParticleSoA& fluid,
                                        const WallParticleSoA& walls,
                                        const NeighborListView& fluid_neighbors,
                                        const NeighborListView& wall_neighbors,
                                        const real* velocity_component,
                                        const real* wall_velocity_component,
                                        const LsmpsMatrixCacheView& matrix,
                                        LsmpsOperatorWorkspaceView workspace,
                                        real* gradient_x,
                                        real* gradient_y,
                                        real* gradient_z) {
  require_matrix_kind(matrix, LsmpsPhysicalField::Velocity, LsmpsBoundaryKind::WallDirichlet);
  compute_scalar_operator_internal(fluid,
                                   walls,
                                   fluid_neighbors,
                                   wall_neighbors,
                                   velocity_component,
                                   wall_velocity_component,
                                   matrix,
                                   LsmpsBoundaryKind::WallDirichlet,
                                   static_cast<real>(0),
                                   static_cast<real>(1),
                                   {},
                                   workspace,
                                   gradient_x,
                                   gradient_y,
                                   gradient_z,
                                   nullptr);
}

void compute_velocity_laplacian_internal(const FluidParticleSoA& fluid,
                                         const WallParticleSoA& walls,
                                         const NeighborListView& fluid_neighbors,
                                         const NeighborListView& wall_neighbors,
                                         const real* velocity_component,
                                         const real* wall_velocity_component,
                                         const LsmpsMatrixCacheView& matrix,
                                         LsmpsOperatorWorkspaceView workspace,
                                         real* laplacian) {
  require_matrix_kind(matrix, LsmpsPhysicalField::Velocity, LsmpsBoundaryKind::WallDirichlet);
  compute_scalar_operator_internal(fluid,
                                   walls,
                                   fluid_neighbors,
                                   wall_neighbors,
                                   velocity_component,
                                   wall_velocity_component,
                                   matrix,
                                   LsmpsBoundaryKind::WallDirichlet,
                                   static_cast<real>(0),
                                   static_cast<real>(1),
                                   {},
                                   workspace,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   laplacian);
}

void compute_velocity_divergence_internal(const FluidParticleSoA& fluid,
                                          const WallParticleSoA& walls,
                                          const NeighborListView& fluid_neighbors,
                                          const NeighborListView& wall_neighbors,
                                          const LsmpsMatrixCacheView& matrix,
                                          LsmpsOperatorWorkspaceView workspace_x,
                                          LsmpsOperatorWorkspaceView workspace_y,
                                          LsmpsOperatorWorkspaceView workspace_z,
                                          real* divergence) {
  require_matrix_kind(matrix, LsmpsPhysicalField::Velocity, LsmpsBoundaryKind::WallDirichlet);
  build_lsmps_rhs_internal(fluid,
                           walls,
                           fluid_neighbors,
                           wall_neighbors,
                           fluid.vx,
                           walls.vx,
                           matrix.request.basis_kind,
                           LsmpsBoundaryKind::WallDirichlet,
                           matrix.request.support_radius,
                           static_cast<real>(0),
                           static_cast<real>(1),
                           {},
                           workspace_x);
  solve_with_matrix_internal(matrix, workspace_x);
  build_lsmps_rhs_internal(fluid,
                           walls,
                           fluid_neighbors,
                           wall_neighbors,
                           fluid.vy,
                           walls.vy,
                           matrix.request.basis_kind,
                           LsmpsBoundaryKind::WallDirichlet,
                           matrix.request.support_radius,
                           static_cast<real>(0),
                           static_cast<real>(1),
                           {},
                           workspace_y);
  solve_with_matrix_internal(matrix, workspace_y);
  build_lsmps_rhs_internal(fluid,
                           walls,
                           fluid_neighbors,
                           wall_neighbors,
                           fluid.vz,
                           walls.vz,
                           matrix.request.basis_kind,
                           LsmpsBoundaryKind::WallDirichlet,
                           matrix.request.support_radius,
                           static_cast<real>(0),
                           static_cast<real>(1),
                           {},
                           workspace_z);
  solve_with_matrix_internal(matrix, workspace_z);
  evaluate_lsmps_divergence_internal(InternalCoefficientsView{workspace_x.solution, matrix.matrix_size},
                                     InternalCoefficientsView{workspace_y.solution, matrix.matrix_size},
                                     InternalCoefficientsView{workspace_z.solution, matrix.matrix_size},
                                     fluid.count,
                                     matrix.request.basis_kind,
                                     matrix.request.support_radius,
                                     divergence);
}

}  // namespace lsmps3d
