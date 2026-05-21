#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

struct CsrMatrixView {
  size_type rows{};
  size_type cols{};
  size_type nnz{};
  index_t* row_offsets{};
  index_t* col_indices{};
  real* values{};
};

struct PpeWorkspaceView {
  CsrMatrixView matrix{};
  real* rhs{};
  real* pressure{};
  real* divergence{};
  real* pressure_laplacian{};
};

class DevicePpeWorkspace {
 public:
  DevicePpeWorkspace() = default;
  DevicePpeWorkspace(size_type fluid_capacity, size_type matrix_nnz_capacity);
  ~DevicePpeWorkspace();

  DevicePpeWorkspace(const DevicePpeWorkspace&) = delete;
  DevicePpeWorkspace& operator=(const DevicePpeWorkspace&) = delete;

  DevicePpeWorkspace(DevicePpeWorkspace&& other) noexcept;
  DevicePpeWorkspace& operator=(DevicePpeWorkspace&& other) noexcept;

  void resize(size_type fluid_capacity, size_type matrix_nnz_capacity);
  void set_active_matrix_nnz(size_type matrix_nnz);
  void release() noexcept;

  [[nodiscard]] size_type fluid_capacity() const noexcept {
    return view_.matrix.rows;
  }

  [[nodiscard]] size_type matrix_nnz_capacity() const noexcept {
    return matrix_nnz_capacity_;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] PpeWorkspaceView view() noexcept {
    return view_;
  }

 private:
  PpeWorkspaceView view_{};
  size_type matrix_nnz_capacity_{};
};

class DevicePpeMatrixAssembler {
 public:
  DevicePpeMatrixAssembler() = default;
  DevicePpeMatrixAssembler(size_type fluid_capacity,
                           size_type matrix_nnz_capacity,
                           SimulationConfig config);

  void resize(size_type fluid_capacity, size_type matrix_nnz_capacity);
  void set_config(SimulationConfig config);
  [[nodiscard]] const SimulationConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] size_type bytes() const noexcept;

  void assemble(const FluidParticleSoA& fluid,
                const WallParticleSoA& walls,
                const NeighborListView& fluid_neighbors,
                const NeighborListView& wall_neighbors,
                const FluidParticleSoA& temporary_velocity,
                const WallParticleSoA& temporary_wall_velocity,
                DeviceMomentMatrix& moment_matrices,
                unsigned long long geometry_generation);

  [[nodiscard]] PpeWorkspaceView workspace() noexcept {
    return workspace_.view();
  }

 private:
  SimulationConfig config_{};
  DevicePpeWorkspace workspace_{};
};

class AmgxPpeSolver {
 public:
  explicit AmgxPpeSolver(std::filesystem::path config_path = {}, bool print_solve_stats = false);
  ~AmgxPpeSolver();

  AmgxPpeSolver(const AmgxPpeSolver&) = delete;
  AmgxPpeSolver& operator=(const AmgxPpeSolver&) = delete;

  AmgxPpeSolver(AmgxPpeSolver&&) noexcept;
  AmgxPpeSolver& operator=(AmgxPpeSolver&&) noexcept;

  [[nodiscard]] static bool is_available() noexcept;
  [[nodiscard]] const std::filesystem::path& config_path() const noexcept;

  void solve(const CsrMatrixView& matrix, const real* rhs, real* pressure);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void assemble_ppe_matrix_and_rhs(const FluidParticleSoA& fluid,
                                 const WallParticleSoA& walls,
                                 const SimulationConfig& config,
                                 const NeighborListView& fluid_neighbors,
                                 const NeighborListView& wall_neighbors,
                                 const FluidParticleSoA& temporary_velocity,
                                 const WallParticleSoA& temporary_wall_velocity,
                                 const MomentMatrixView& velocity_moment,
                                 const MomentMatrixView& pressure_moment,
                                 const PpeWorkspaceView& workspace);

}  // namespace lsmps3d
