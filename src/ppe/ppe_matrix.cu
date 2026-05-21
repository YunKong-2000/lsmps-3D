#include "lsmps3d/ppe/ppe_matrix.cuh"

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#if defined(LSMPS3D_ENABLE_AMGX)
#include <amgx_c.h>
#endif

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

void swap_csr_views(CsrMatrixView& lhs, CsrMatrixView& rhs) noexcept {
  using std::swap;
  swap(lhs.rows, rhs.rows);
  swap(lhs.cols, rhs.cols);
  swap(lhs.nnz, rhs.nnz);
  swap(lhs.row_offsets, rhs.row_offsets);
  swap(lhs.col_indices, rhs.col_indices);
  swap(lhs.values, rhs.values);
}

void swap_ppe_views(PpeWorkspaceView& lhs, PpeWorkspaceView& rhs) noexcept {
  using std::swap;
  swap_csr_views(lhs.matrix, rhs.matrix);
  swap(lhs.rhs, rhs.rhs);
  swap(lhs.pressure, rhs.pressure);
  swap(lhs.divergence, rhs.divergence);
  swap(lhs.pressure_laplacian, rhs.pressure_laplacian);
}

void validate_matrix_workspace(const FluidParticleSoA& fluid,
                               const NeighborListView& fluid_neighbors,
                               const PpeWorkspaceView& workspace) {
  if (workspace.matrix.rows < fluid.count || workspace.matrix.cols < fluid.count ||
      workspace.matrix.nnz < fluid.count + fluid_neighbors.neighbor_count) {
    std::cerr << "PPE matrix workspace is smaller than the requested fluid system" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count == 0) {
    return;
  }
  if (workspace.matrix.row_offsets == nullptr || workspace.matrix.col_indices == nullptr ||
      workspace.matrix.values == nullptr || workspace.rhs == nullptr || workspace.pressure == nullptr ||
      workspace.pressure_laplacian == nullptr) {
    std::cerr << "PPE workspace requires matrix, rhs, pressure, and diagnostic buffers"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid_neighbors.particle_count < fluid.count || fluid_neighbors.offsets == nullptr) {
    std::cerr << "PPE assembly requires a valid fluid-neighbor CSR table" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_rhs_workspace(const FluidParticleSoA& fluid,
                            const FluidParticleSoA& temporary_velocity,
                            const WallParticleSoA& temporary_wall_velocity,
                            const PpeWorkspaceView& workspace) {
  if (fluid.count == 0) {
    return;
  }
  if (temporary_velocity.vx == nullptr || temporary_velocity.vy == nullptr ||
      temporary_velocity.vz == nullptr || workspace.rhs == nullptr ||
      workspace.divergence == nullptr) {
    std::cerr << "PPE RHS assembly requires temporary velocity, divergence and rhs buffers"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (temporary_wall_velocity.count > 0 &&
      (temporary_wall_velocity.vx == nullptr || temporary_wall_velocity.vy == nullptr ||
       temporary_wall_velocity.vz == nullptr)) {
    std::cerr << "PPE RHS assembly requires temporary wall velocity buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void validate_ppe_moment_matrix(const FluidParticleSoA& fluid,
                                const MomentMatrixView& moment,
                                MomentMatrixKind expected_kind,
                                const char* label) {
  if (!moment.is_ready || moment.kind != expected_kind ||
      moment.basis_kind != MomentBasisKind::TypeA ||
      moment.matrix_size != kMomentTypeABasis3DSize || moment.particle_count < fluid.count) {
    std::cerr << "PPE requires a prepared " << label << " Type-A moment matrix" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > 0 && moment.inverse_matrices == nullptr) {
    std::cerr << "PPE moment matrix inverse buffer is missing for " << label << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__device__ bool is_pressure_dirichlet_particle(const FluidParticleSoA& fluid, size_type i) {
  if (fluid.surface_type == nullptr) {
    return false;
  }
  const int type = fluid.surface_type[i];
  return type == static_cast<int>(SurfaceType::Surface) ||
         type == static_cast<int>(SurfaceType::Splash);
}

__global__ void build_ppe_row_offsets_kernel(const FluidParticleSoA fluid,
                                             NeighborListView fluid_neighbors,
                                             CsrMatrixView matrix) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }
  const index_t row_neighbors = fluid_neighbors.offsets[i + 1] - fluid_neighbors.offsets[i];
  matrix.row_offsets[i] = static_cast<index_t>(i) + fluid_neighbors.offsets[i];
  if (i + 1 == fluid.count) {
    matrix.row_offsets[i + 1] = static_cast<index_t>(i + 1) + fluid_neighbors.offsets[i + 1];
  }
  (void)row_neighbors;
}

__device__ real moment_weight(real distance, real support_radius) {
  if (distance >= support_radius) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__device__ void ppe_type_a_basis(real dx,
                                 real dy,
                                 real dz,
                                 real support_radius,
                                 real basis[9]) {
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

__device__ void ppe_wall_neumann_vector(real dx,
                                        real dy,
                                        real dz,
                                        real normal_x,
                                        real normal_y,
                                        real normal_z,
                                        real support_radius,
                                        real q[9]) {
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

__device__ real dot9(const real lhs[9], const real rhs[9]) {
  real value = static_cast<real>(0);
  for (int k = 0; k < 9; ++k) {
    value += lhs[k] * rhs[k];
  }
  return value;
}

__device__ void compute_pressure_laplacian_row(
    const MomentMatrixView pressure_moment,
    size_type i,
    real laplacian_row[kMomentTypeABasis3DSize]) {
  const real* inverse_matrix =
      pressure_moment.inverse_matrices + i * kMomentTypeABasis3DSize * kMomentTypeABasis3DSize;
  for (int col = 0; col < kMomentTypeABasis3DSize; ++col) {
    laplacian_row[col] =
        inverse_matrix[3 + col * kMomentTypeABasis3DSize] +
        inverse_matrix[4 + col * kMomentTypeABasis3DSize] +
        inverse_matrix[5 + col * kMomentTypeABasis3DSize];
  }
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

__device__ real first_coefficient_from_inverse_rhs(const real* inverse_matrix,
                                                   const real rhs[kMomentTypeABasis3DSize],
                                                   int row) {
  real coeff = static_cast<real>(0);
  for (int col = 0; col < kMomentTypeABasis3DSize; ++col) {
    coeff += inverse_matrix[row + col * kMomentTypeABasis3DSize] * rhs[col];
  }
  return coeff;
}

__global__ void assemble_ppe_system_kernel(const FluidParticleSoA fluid,
                                           const WallParticleSoA walls,
                                           SimulationConfig config,
                                           NeighborListView fluid_neighbors,
                                           NeighborListView wall_neighbors,
                                           const FluidParticleSoA temporary_velocity,
                                           const WallParticleSoA temporary_wall_velocity,
                                           MomentMatrixView velocity_moment,
                                           MomentMatrixView pressure_moment,
                                           PpeWorkspaceView workspace) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const index_t row_begin = workspace.matrix.row_offsets[i];
  const index_t neighbor_begin = fluid_neighbors.offsets[i];
  const index_t neighbor_end = fluid_neighbors.offsets[i + 1];

  if (is_pressure_dirichlet_particle(fluid, i)) {
    workspace.matrix.col_indices[row_begin] = static_cast<index_t>(i);
    workspace.matrix.values[row_begin] = static_cast<real>(1);
    for (index_t cursor = neighbor_begin; cursor < neighbor_end; ++cursor) {
      const index_t out = row_begin + 1 + cursor - neighbor_begin;
      workspace.matrix.col_indices[out] = fluid_neighbors.indices[cursor];
      workspace.matrix.values[out] = static_cast<real>(0);
    }
    workspace.rhs[i] = static_cast<real>(0);
    workspace.divergence[i] = static_cast<real>(0);
    if (workspace.pressure_laplacian != nullptr) {
      workspace.pressure_laplacian[i] = static_cast<real>(0);
    }
    return;
  }

  const real xi = fluid.x[i];
  const real yi = fluid.y[i];
  const real zi = fluid.z[i];
  const real support_radius = pressure_moment.support_radius;
  real laplacian_row[kMomentTypeABasis3DSize]{};
  compute_pressure_laplacian_row(pressure_moment, i, laplacian_row);
  real rhs_x[kMomentTypeABasis3DSize]{};
  real rhs_y[kMomentTypeABasis3DSize]{};
  real rhs_z[kMomentTypeABasis3DSize]{};
  real diagonal = static_cast<real>(0);

  for (index_t cursor = neighbor_begin; cursor < neighbor_end; ++cursor) {
    const index_t j = fluid_neighbors.indices[cursor];
    const real dx = fluid.x[j] - xi;
    const real dy = fluid.y[j] - yi;
    const real dz = fluid.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real weight = moment_weight(distance, support_radius);
    real off_diagonal = static_cast<real>(0);
    if (weight > static_cast<real>(0)) {
      real basis[kMomentTypeABasis3DSize]{};
      ppe_type_a_basis(dx, dy, dz, support_radius, basis);
      off_diagonal = static_cast<real>(2) * weight * dot9(laplacian_row, basis) /
                     (support_radius * support_radius * config.density);
      diagonal -= off_diagonal;
      accumulate_velocity_rhs(basis,
                              weight,
                              temporary_velocity.vx[j] - temporary_velocity.vx[i],
                              temporary_velocity.vy[j] - temporary_velocity.vy[i],
                              temporary_velocity.vz[j] - temporary_velocity.vz[i],
                              rhs_x,
                              rhs_y,
                              rhs_z);
    }
    const index_t out = row_begin + 1 + cursor - neighbor_begin;
    workspace.matrix.col_indices[out] = j;
    workspace.matrix.values[out] = off_diagonal;
  }

  real wall_rhs = static_cast<real>(0);
  for (index_t cursor = wall_neighbors.offsets[i]; cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const index_t j = wall_neighbors.indices[cursor];
    const real dx = walls.x[j] - xi;
    const real dy = walls.y[j] - yi;
    const real dz = walls.z[j] - zi;
    const real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const real base_weight = moment_weight(distance, support_radius);
    if (base_weight <= static_cast<real>(0)) {
      continue;
    }
    real basis[kMomentTypeABasis3DSize]{};
    ppe_type_a_basis(dx, dy, dz, support_radius, basis);
    accumulate_velocity_rhs(basis,
                            velocity_moment.wall_weight_scale * base_weight,
                            temporary_wall_velocity.vx[j] - temporary_velocity.vx[i],
                            temporary_wall_velocity.vy[j] - temporary_velocity.vy[i],
                            temporary_wall_velocity.vz[j] - temporary_velocity.vz[i],
                            rhs_x,
                            rhs_y,
                            rhs_z);
    const real normal_gravity = config.gravity.x * walls.normal_x[j] +
                                config.gravity.y * walls.normal_y[j] +
                                config.gravity.z * walls.normal_z[j];
    real q[kMomentTypeABasis3DSize]{};
    ppe_wall_neumann_vector(dx,
                            dy,
                            dz,
                            walls.normal_x[j],
                            walls.normal_y[j],
                            walls.normal_z[j],
                            support_radius,
                            q);
    wall_rhs += static_cast<real>(2) * pressure_moment.wall_weight_scale * base_weight *
                normal_gravity * dot9(laplacian_row, q) / support_radius;
  }

  const real* velocity_inverse =
      velocity_moment.inverse_matrices + i * kMomentTypeABasis3DSize * kMomentTypeABasis3DSize;
  const real divergence =
      (first_coefficient_from_inverse_rhs(velocity_inverse, rhs_x, 0) +
       first_coefficient_from_inverse_rhs(velocity_inverse, rhs_y, 1) +
       first_coefficient_from_inverse_rhs(velocity_inverse, rhs_z, 2)) /
      support_radius;

  workspace.matrix.col_indices[row_begin] = static_cast<index_t>(i);
  workspace.matrix.values[row_begin] = diagonal;
  workspace.divergence[i] = divergence;
  workspace.rhs[i] = divergence / config.time_step - wall_rhs;
  if (workspace.pressure_laplacian != nullptr) {
    workspace.pressure_laplacian[i] = wall_rhs;
  }
}

#if defined(LSMPS3D_ENABLE_AMGX)
__global__ void clamp_pressure_kernel(size_type count, real* pressure) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  pressure[i] = pressure[i] < static_cast<real>(0) ? static_cast<real>(0) : pressure[i];
}

[[nodiscard]] AMGX_Mode amgx_mode() {
#if defined(LSMPS3D_USE_DOUBLE)
  return AMGX_mode_dDDI;
#else
  return AMGX_mode_fFFI;
#endif
}

struct AmgxHandles {
  AMGX_config_handle config{};
  AMGX_resources_handle resources{};
  AMGX_solver_handle solver{};
  AMGX_matrix_handle matrix{};
  AMGX_vector_handle rhs{};
  AMGX_vector_handle solution{};
};
#endif

}  // namespace

DevicePpeWorkspace::DevicePpeWorkspace(size_type fluid_capacity, size_type matrix_nnz_capacity) {
  resize(fluid_capacity, matrix_nnz_capacity);
}

DevicePpeWorkspace::~DevicePpeWorkspace() {
  release();
}

DevicePpeWorkspace::DevicePpeWorkspace(DevicePpeWorkspace&& other) noexcept {
  swap_ppe_views(view_, other.view_);
  using std::swap;
  swap(matrix_nnz_capacity_, other.matrix_nnz_capacity_);
}

DevicePpeWorkspace& DevicePpeWorkspace::operator=(DevicePpeWorkspace&& other) noexcept {
  if (this != &other) {
    release();
    swap_ppe_views(view_, other.view_);
    using std::swap;
    swap(matrix_nnz_capacity_, other.matrix_nnz_capacity_);
  }
  return *this;
}

void DevicePpeWorkspace::resize(size_type fluid_capacity, size_type matrix_nnz_capacity) {
  if (fluid_capacity == view_.matrix.rows && matrix_nnz_capacity == matrix_nnz_capacity_) {
    return;
  }

  release();
  view_.matrix.rows = fluid_capacity;
  view_.matrix.cols = fluid_capacity;
  view_.matrix.nnz = matrix_nnz_capacity;
  matrix_nnz_capacity_ = matrix_nnz_capacity;
  device_alloc(view_.matrix.row_offsets, fluid_capacity + 1);
  device_alloc(view_.matrix.col_indices, matrix_nnz_capacity);
  device_alloc(view_.matrix.values, matrix_nnz_capacity);
  device_alloc(view_.rhs, fluid_capacity);
  device_alloc(view_.pressure, fluid_capacity);
  device_alloc(view_.divergence, fluid_capacity);
  device_alloc(view_.pressure_laplacian, fluid_capacity);
}

void DevicePpeWorkspace::set_active_matrix_nnz(size_type matrix_nnz) {
  if (matrix_nnz > matrix_nnz_capacity_) {
    std::cerr << "PPE active matrix nnz exceeds allocated matrix capacity" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  view_.matrix.nnz = matrix_nnz;
}

void DevicePpeWorkspace::release() noexcept {
  device_free(view_.matrix.row_offsets);
  device_free(view_.matrix.col_indices);
  device_free(view_.matrix.values);
  device_free(view_.rhs);
  device_free(view_.pressure);
  device_free(view_.divergence);
  device_free(view_.pressure_laplacian);
  view_ = {};
  matrix_nnz_capacity_ = 0;
}

size_type DevicePpeWorkspace::bytes() const noexcept {
  return (view_.matrix.rows + 1) * sizeof(index_t) +
         matrix_nnz_capacity_ * (sizeof(index_t) + sizeof(real)) + view_.matrix.rows * 4 * sizeof(real);
}

DevicePpeMatrixAssembler::DevicePpeMatrixAssembler(size_type fluid_capacity,
                                                   size_type matrix_nnz_capacity,
                                                   SimulationConfig config)
    : config_(std::move(config)), workspace_(fluid_capacity, matrix_nnz_capacity) {}

void DevicePpeMatrixAssembler::resize(size_type fluid_capacity, size_type matrix_nnz_capacity) {
  workspace_.resize(fluid_capacity, matrix_nnz_capacity);
}

void DevicePpeMatrixAssembler::set_config(SimulationConfig config) {
  config_ = std::move(config);
}

size_type DevicePpeMatrixAssembler::bytes() const noexcept {
  return workspace_.bytes();
}

void DevicePpeMatrixAssembler::assemble(const FluidParticleSoA& fluid,
                                        const WallParticleSoA& walls,
                                        const NeighborListView& fluid_neighbors,
                                        const NeighborListView& wall_neighbors,
                                        const FluidParticleSoA& temporary_velocity,
                                        const WallParticleSoA& temporary_wall_velocity,
                                        DeviceMomentMatrix& moment_matrices,
                                        unsigned long long geometry_generation) {
  const size_type required_nnz = fluid.count + fluid_neighbors.neighbor_count;
  if (workspace_.fluid_capacity() < fluid.count ||
      workspace_.matrix_nnz_capacity() < required_nnz) {
    resize(fluid.count, required_nnz);
  }
  workspace_.set_active_matrix_nnz(required_nnz);
  auto workspace_view = workspace_.view();
  moment_matrices.set_config(config_);
  moment_matrices.resize(fluid.count);
  moment_matrices.prepare_matrices(
      fluid, walls, fluid_neighbors, wall_neighbors, geometry_generation);
  assemble_ppe_matrix_and_rhs(fluid,
                              walls,
                              config_,
                              fluid_neighbors,
                              wall_neighbors,
                              temporary_velocity,
                              temporary_wall_velocity,
                              moment_matrices.velocity_type_a(),
                              moment_matrices.pressure_type_a(),
                              workspace_view);
}

void assemble_ppe_matrix_and_rhs(const FluidParticleSoA& fluid,
                                 const WallParticleSoA& walls,
                                 const SimulationConfig& config,
                                 const NeighborListView& fluid_neighbors,
                                 const NeighborListView& wall_neighbors,
                                 const FluidParticleSoA& temporary_velocity,
                                 const WallParticleSoA& temporary_wall_velocity,
                                 const MomentMatrixView& velocity_moment,
                                 const MomentMatrixView& pressure_moment,
                                 const PpeWorkspaceView& workspace) {
  validate_matrix_workspace(fluid, fluid_neighbors, workspace);
  validate_rhs_workspace(fluid, temporary_velocity, temporary_wall_velocity, workspace);
  validate_ppe_moment_matrix(
      fluid, velocity_moment, MomentMatrixKind::VelocityWallDirichletTypeA, "velocity");
  validate_ppe_moment_matrix(
      fluid, pressure_moment, MomentMatrixKind::PressureWallNeumannTypeA, "pressure");
  if (fluid.count == 0) {
    return;
  }

  build_ppe_row_offsets_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, fluid_neighbors, workspace.matrix);
  LSMPS3D_CUDA_KERNEL_CHECK();
  assemble_ppe_system_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(fluid,
                                                                            walls,
                                                                            config,
                                                                            fluid_neighbors,
                                                                            wall_neighbors,
                                                                            temporary_velocity,
                                                                            temporary_wall_velocity,
                                                                            velocity_moment,
                                                                            pressure_moment,
                                                                            workspace);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

struct AmgxPpeSolver::Impl {
  explicit Impl(std::filesystem::path path) : config_path(std::move(path)) {}

  std::filesystem::path config_path;
#if defined(LSMPS3D_ENABLE_AMGX)
  AmgxHandles handles{};
  bool initialized{false};
#endif
};

AmgxPpeSolver::AmgxPpeSolver(std::filesystem::path config_path)
    : impl_(std::make_unique<Impl>(std::move(config_path))) {}

AmgxPpeSolver::~AmgxPpeSolver() = default;

AmgxPpeSolver::AmgxPpeSolver(AmgxPpeSolver&&) noexcept = default;

AmgxPpeSolver& AmgxPpeSolver::operator=(AmgxPpeSolver&&) noexcept = default;

bool AmgxPpeSolver::is_available() noexcept {
#if defined(LSMPS3D_ENABLE_AMGX)
  return true;
#else
  return false;
#endif
}

const std::filesystem::path& AmgxPpeSolver::config_path() const noexcept {
  return impl_->config_path;
}

void AmgxPpeSolver::solve(const CsrMatrixView& matrix, const real* rhs, real* pressure) {
  if (matrix.rows == 0) {
    return;
  }
  if (matrix.rows != matrix.cols || matrix.row_offsets == nullptr || matrix.col_indices == nullptr ||
      matrix.values == nullptr || rhs == nullptr || pressure == nullptr) {
    std::cerr << "AMGX PPE solve requires a square CSR matrix, rhs, and solution buffer"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

#if defined(LSMPS3D_ENABLE_AMGX)
  AMGX_initialize();
  const AMGX_Mode mode = amgx_mode();
  const std::string config_source =
      impl_->config_path.empty() ? std::string{} : impl_->config_path.string();
  if (config_source.empty()) {
    LSMPS3D_AMGX_CHECK(AMGX_config_create(
        &impl_->handles.config,
        "config_version=2,solver(preconditioner)=NOSOLVER,preconditioner:scope=amg,"
        "solver=GMRES,use_scalar_norm=1,print_solve_stats=1,obtain_timings=1,"
        "monitor_residual=1,convergence=RELATIVE_INI,scope=main,max_iters=100,"
        "tolerance=1e-6,norm=L2"));
  } else {
    LSMPS3D_AMGX_CHECK(AMGX_config_create_from_file(&impl_->handles.config, config_source.c_str()));
  }
  LSMPS3D_AMGX_CHECK(AMGX_resources_create_simple(&impl_->handles.resources, impl_->handles.config));
  LSMPS3D_AMGX_CHECK(AMGX_matrix_create(&impl_->handles.matrix, impl_->handles.resources, mode));
  LSMPS3D_AMGX_CHECK(AMGX_vector_create(&impl_->handles.rhs, impl_->handles.resources, mode));
  LSMPS3D_AMGX_CHECK(AMGX_vector_create(&impl_->handles.solution, impl_->handles.resources, mode));
  LSMPS3D_AMGX_CHECK(AMGX_solver_create(
      &impl_->handles.solver, impl_->handles.resources, mode, impl_->handles.config));

  std::vector<index_t> host_row_offsets(matrix.rows + 1);
  std::vector<index_t> host_col_indices(matrix.nnz);
  std::vector<real> host_values(matrix.nnz);
  std::vector<real> host_rhs(matrix.rows);
  std::vector<real> host_pressure(matrix.rows, static_cast<real>(0));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(host_row_offsets.data(),
                                matrix.row_offsets,
                                host_row_offsets.size() * sizeof(index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(host_col_indices.data(),
                                matrix.col_indices,
                                host_col_indices.size() * sizeof(index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(host_values.data(),
                                matrix.values,
                                host_values.size() * sizeof(real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(host_rhs.data(), rhs, host_rhs.size() * sizeof(real), cudaMemcpyDeviceToHost));

  const int rows = static_cast<int>(matrix.rows);
  const int nnz = static_cast<int>(matrix.nnz);
  LSMPS3D_AMGX_CHECK(AMGX_matrix_upload_all(
      impl_->handles.matrix,
      rows,
      nnz,
      1,
      1,
      host_row_offsets.data(),
      host_col_indices.data(),
      host_values.data(),
      nullptr));
  LSMPS3D_AMGX_CHECK(AMGX_vector_upload(impl_->handles.rhs, rows, 1, host_rhs.data()));
  LSMPS3D_AMGX_CHECK(
      AMGX_vector_upload(impl_->handles.solution, rows, 1, host_pressure.data()));
  LSMPS3D_AMGX_CHECK(AMGX_solver_setup(impl_->handles.solver, impl_->handles.matrix));
  LSMPS3D_AMGX_CHECK(
      AMGX_solver_solve(impl_->handles.solver, impl_->handles.rhs, impl_->handles.solution));
  LSMPS3D_AMGX_CHECK(AMGX_vector_download(impl_->handles.solution, host_pressure.data()));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      pressure, host_pressure.data(), host_pressure.size() * sizeof(real), cudaMemcpyHostToDevice));
  clamp_pressure_kernel<<<block_count(matrix.rows), kThreadsPerBlock>>>(matrix.rows, pressure);
  LSMPS3D_CUDA_KERNEL_CHECK();

  AMGX_solver_destroy(impl_->handles.solver);
  AMGX_vector_destroy(impl_->handles.solution);
  AMGX_vector_destroy(impl_->handles.rhs);
  AMGX_matrix_destroy(impl_->handles.matrix);
  AMGX_resources_destroy(impl_->handles.resources);
  AMGX_config_destroy(impl_->handles.config);
  AMGX_finalize();
  impl_->handles = {};
#else
  std::cerr << "AMGX support is not enabled in this build. Configure with AMGX headers and "
               "library available to use AmgxPpeSolver."
            << std::endl;
  std::exit(EXIT_FAILURE);
#endif
}

}  // namespace lsmps3d
