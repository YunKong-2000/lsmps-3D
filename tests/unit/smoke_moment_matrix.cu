#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

int main() {
  constexpr lsmps3d::size_type kCount = 27;
  constexpr lsmps3d::size_type kCenter = 13;

  std::array<lsmps3d::real, kCount> x{};
  std::array<lsmps3d::real, kCount> y{};
  std::array<lsmps3d::real, kCount> z{};
  lsmps3d::size_type out = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        x[out] = static_cast<lsmps3d::real>(ix);
        y[out] = static_cast<lsmps3d::real>(iy);
        z[out] = static_cast<lsmps3d::real>(iz);
        ++out;
      }
    }
  }

  lsmps3d::DeviceFluidParticles fluid(kCount);
  lsmps3d::DeviceWallParticles walls(0);
  lsmps3d::DeviceNeighborList fluid_neighbors(kCount, kCount * (kCount - 1));
  lsmps3d::DeviceNeighborList wall_neighbors(kCount, 0);
  auto fluid_view = fluid.view();
  auto wall_view = walls.view();
  auto fluid_neighbor_view = fluid_neighbors.view();
  auto wall_neighbor_view = wall_neighbors.view();

  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.x, x.data(), x.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.y, y.data(), y.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.z, z.data(), z.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  std::vector<lsmps3d::index_t> offsets(kCount + 1);
  std::vector<lsmps3d::index_t> indices;
  indices.reserve(kCount * (kCount - 1));
  for (lsmps3d::size_type i = 0; i < kCount; ++i) {
    offsets[i] = static_cast<lsmps3d::index_t>(indices.size());
    for (lsmps3d::size_type j = 0; j < kCount; ++j) {
      if (i != j) {
        indices.push_back(static_cast<lsmps3d::index_t>(j));
      }
    }
  }
  offsets[kCount] = static_cast<lsmps3d::index_t>(indices.size());
  std::vector<lsmps3d::index_t> wall_offsets(kCount + 1, 0);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(fluid_neighbor_view.offsets,
                                offsets.data(),
                                offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(fluid_neighbor_view.indices,
                                indices.data(),
                                indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(wall_neighbor_view.offsets,
                                wall_offsets.data(),
                                wall_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyHostToDevice));

  lsmps3d::SimulationConfig config;
  config.support_radius = static_cast<lsmps3d::real>(4.0);
  config.cell_size = static_cast<lsmps3d::real>(4.0);
  config.lsmps_regularization = static_cast<lsmps3d::real>(1.0e-10);
  config.lsmps_wall_weight_scale = static_cast<lsmps3d::real>(1.0);
  config.density = static_cast<lsmps3d::real>(1.0);
  config.gravity = {};
  lsmps3d::DeviceMomentMatrix lsmps(kCount, config);

  lsmps.prepare_matrices(fluid_view, wall_view, fluid_neighbor_view, wall_neighbor_view, 1);
  const auto bytes_after_first_prepare = lsmps.bytes();
  lsmps.prepare_matrices(fluid_view, wall_view, fluid_neighbor_view, wall_neighbor_view, 1);
  if (bytes_after_first_prepare != lsmps.bytes()) {
    std::cerr << "Moment matrix storage changed when reusing generation" << std::endl;
    return 1;
  }

  const auto velocity = lsmps.velocity_type_a();
  const auto pressure = lsmps.pressure_type_a();
  const auto pressure_b = lsmps.pressure_type_b();
  const auto fluid_only = lsmps.fluid_only_type_a();
  if (!velocity.is_ready || !pressure.is_ready || !pressure_b.is_ready || !fluid_only.is_ready) {
    std::cerr << "Moment matrix views were not prepared" << std::endl;
    return 1;
  }
  if (velocity.kind != lsmps3d::MomentMatrixKind::VelocityWallDirichletTypeA ||
      pressure.kind != lsmps3d::MomentMatrixKind::PressureWallNeumannTypeA ||
      pressure_b.kind != lsmps3d::MomentMatrixKind::PressureWallNeumannTypeB ||
      fluid_only.kind != lsmps3d::MomentMatrixKind::FluidOnlyTypeA) {
    std::cerr << "Moment matrix view kinds are incorrect" << std::endl;
    return 1;
  }
  if (velocity.matrix_size != lsmps3d::kMomentTypeABasis3DSize ||
      pressure.matrix_size != lsmps3d::kMomentTypeABasis3DSize ||
      pressure_b.matrix_size != lsmps3d::kMomentTypeBBasis3DSize ||
      fluid_only.matrix_size != lsmps3d::kMomentTypeABasis3DSize) {
    std::cerr << "Moment matrix basis sizes are incorrect" << std::endl;
    return 1;
  }
  if (velocity.inverse_matrices == nullptr || pressure.inverse_matrices == nullptr ||
      pressure_b.inverse_matrices == nullptr || fluid_only.inverse_matrices == nullptr) {
    std::cerr << "Moment matrix inverse buffers are missing" << std::endl;
    return 1;
  }

  std::array<lsmps3d::real, lsmps3d::kMomentTypeABasis3DSize * lsmps3d::kMomentTypeABasis3DSize>
      center_inverse{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(center_inverse.data(),
                                velocity.inverse_matrices +
                                    kCenter * lsmps3d::kMomentTypeABasis3DSize *
                                        lsmps3d::kMomentTypeABasis3DSize,
                                center_inverse.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  for (const auto value : center_inverse) {
    if (!std::isfinite(value)) {
      std::cerr << "Moment matrix inverse contains non-finite values" << std::endl;
      return 1;
    }
  }

  return 0;
}
