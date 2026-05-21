#pragma once

#include <memory>

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/moment_matrix/basis.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace lsmps3d {

enum class MomentMatrixKind : int {
  VelocityWallDirichletTypeA = 0,
  PressureWallNeumannTypeA = 1,
  PressureWallNeumannTypeB = 2,
  FluidOnlyTypeA = 3,
};

struct MomentMatrixView {
  size_type particle_count{};
  int matrix_size{kMomentTypeABasis3DSize};
  MomentBasisKind basis_kind{MomentBasisKind::TypeA};
  MomentMatrixKind kind{MomentMatrixKind::VelocityWallDirichletTypeA};
  real support_radius{};
  real wall_weight_scale{};
  unsigned long long geometry_generation{};
  bool is_ready{false};
  const real* inverse_matrices{};
  const real* moment_trace{};
  const real* regularization_added{};
  const int* info{};
  const int* inversion_count{};
};

class DeviceMomentMatrix {
 public:
  DeviceMomentMatrix();
  DeviceMomentMatrix(size_type fluid_capacity, SimulationConfig config);
  ~DeviceMomentMatrix();

  DeviceMomentMatrix(const DeviceMomentMatrix&) = delete;
  DeviceMomentMatrix& operator=(const DeviceMomentMatrix&) = delete;

  DeviceMomentMatrix(DeviceMomentMatrix&& other) noexcept;
  DeviceMomentMatrix& operator=(DeviceMomentMatrix&& other) noexcept;

  void resize(size_type fluid_capacity);
  void release() noexcept;
  void set_config(SimulationConfig config);
  [[nodiscard]] const SimulationConfig& config() const noexcept;
  [[nodiscard]] size_type bytes() const noexcept;

  void prepare_matrices(const FluidParticleSoA& fluid,
                        const WallParticleSoA& walls,
                        const NeighborListView& fluid_neighbors,
                        const NeighborListView& wall_neighbors,
                        unsigned long long geometry_generation);

  [[nodiscard]] MomentMatrixView velocity_type_a() const;
  [[nodiscard]] MomentMatrixView pressure_type_a() const;
  [[nodiscard]] MomentMatrixView pressure_type_b() const;
  [[nodiscard]] MomentMatrixView fluid_only_type_a() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lsmps3d
