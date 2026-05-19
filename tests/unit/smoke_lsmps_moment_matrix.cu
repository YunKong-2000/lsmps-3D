#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/lsmps/moment_matrix.cuh"
#include "lsmps3d/particle/neighbor_list.cuh"
#include "lsmps3d/particle/particle_data.cuh"

namespace {

lsmps3d::real field_value(lsmps3d::real x, lsmps3d::real y, lsmps3d::real z) {
  return static_cast<lsmps3d::real>(5.0) + static_cast<lsmps3d::real>(2.0) * x -
         static_cast<lsmps3d::real>(3.0) * y + static_cast<lsmps3d::real>(4.0) * z +
         static_cast<lsmps3d::real>(0.5) * static_cast<lsmps3d::real>(6.0) * x * x +
         static_cast<lsmps3d::real>(0.5) * static_cast<lsmps3d::real>(-8.0) * y * y +
         static_cast<lsmps3d::real>(0.5) * static_cast<lsmps3d::real>(10.0) * z * z +
         static_cast<lsmps3d::real>(1.5) * x * y -
         static_cast<lsmps3d::real>(2.0) * y * z + static_cast<lsmps3d::real>(0.75) * z * x;
}

bool nearly_equal(lsmps3d::real actual,
                  lsmps3d::real expected,
                  lsmps3d::real tolerance,
                  const char* label) {
  if (std::abs(actual - expected) <= tolerance) {
    return true;
  }

  std::cerr << label << " mismatch: actual=" << actual << " expected=" << expected
            << " tolerance=" << tolerance << std::endl;
  return false;
}

}  // namespace

int main() {
  constexpr lsmps3d::size_type kCount = 27;
  constexpr lsmps3d::size_type kCenter = 13;

  std::array<lsmps3d::real, kCount> x{};
  std::array<lsmps3d::real, kCount> y{};
  std::array<lsmps3d::real, kCount> z{};
  std::array<lsmps3d::real, kCount> values{};
  lsmps3d::size_type out = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        x[out] = static_cast<lsmps3d::real>(ix);
        y[out] = static_cast<lsmps3d::real>(iy);
        z[out] = static_cast<lsmps3d::real>(iz);
        values[out] = field_value(x[out], y[out], z[out]);
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
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      fluid_view.pressure, values.data(), values.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.vx, values.data(), values.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.vy, values.data(), values.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(fluid_view.vz, values.data(), values.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));

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

  lsmps3d::DeviceFluidParticles operator_buffers(kCount);
  auto operator_view = operator_buffers.view();
  lsmps3d::SimulationConfig config;
  config.support_radius = static_cast<lsmps3d::real>(4.0);
  config.cell_size = static_cast<lsmps3d::real>(4.0);
  config.lsmps_regularization = static_cast<lsmps3d::real>(1.0e-10);
  config.lsmps_wall_weight_scale = static_cast<lsmps3d::real>(1.0);
  config.density = static_cast<lsmps3d::real>(1.0);
  config.gravity = {};
  lsmps3d::DeviceLsmpsOperators lsmps(kCount, config);

  lsmps.prepare_matrices(fluid_view, wall_view, fluid_neighbor_view, wall_neighbor_view, 1);
  const auto bytes_after_first_prepare = lsmps.bytes();
  lsmps.prepare_matrices(fluid_view, wall_view, fluid_neighbor_view, wall_neighbor_view, 1);
  if (bytes_after_first_prepare != lsmps.bytes()) {
    std::cerr << "LSMPS unified operator storage changed when reusing generation" << std::endl;
    return 1;
  }

  lsmps.compute_near_surface_pressure_gradient(
      fluid_view,
      wall_view,
      fluid_neighbor_view,
      wall_neighbor_view,
      fluid_view.pressure,
      operator_view.x,
      operator_view.y,
      operator_view.z);
  lsmps.compute_pressure_laplacian(fluid_view,
                                   wall_view,
                                   fluid_neighbor_view,
                                   wall_neighbor_view,
                                   fluid_view.pressure,
                                   operator_view.pressure);

  const lsmps3d::real tolerance = static_cast<lsmps3d::real>(2.0e-3);
  std::array<lsmps3d::real, kCount> grad_x{};
  std::array<lsmps3d::real, kCount> grad_y{};
  std::array<lsmps3d::real, kCount> grad_z{};
  std::array<lsmps3d::real, kCount> laplacian{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      grad_x.data(), operator_view.x, grad_x.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      grad_y.data(), operator_view.y, grad_y.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      grad_z.data(), operator_view.z, grad_z.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(laplacian.data(),
                                operator_view.pressure,
                                laplacian.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  if (!nearly_equal(grad_x[kCenter], static_cast<lsmps3d::real>(2.0), tolerance, "grad x") ||
      !nearly_equal(grad_y[kCenter], static_cast<lsmps3d::real>(-3.0), tolerance, "grad y") ||
      !nearly_equal(grad_z[kCenter], static_cast<lsmps3d::real>(4.0), tolerance, "grad z") ||
      !nearly_equal(laplacian[kCenter], static_cast<lsmps3d::real>(8.0), tolerance, "laplacian")) {
    return 1;
  }

  lsmps.compute_velocity_divergence(fluid_view,
                                    wall_view,
                                    fluid_neighbor_view,
                                    wall_neighbor_view,
                                    operator_view.vx);
  std::array<lsmps3d::real, kCount> divergence{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(divergence.data(),
                                operator_view.vx,
                                divergence.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  if (!nearly_equal(divergence[kCenter],
                    static_cast<lsmps3d::real>(3.0),
                    tolerance,
                    "divergence")) {
    return 1;
  }

  const lsmps3d::Vec3 pressure_gravity{
      static_cast<lsmps3d::real>(-9.8), static_cast<lsmps3d::real>(0.0), static_cast<lsmps3d::real>(0.0)};
  std::array<lsmps3d::real, kCount> linear_pressure{};
  for (lsmps3d::size_type i = 0; i < kCount; ++i) {
    linear_pressure[i] = static_cast<lsmps3d::real>(-9800.0) * x[i];
  }
  LSMPS3D_CUDA_CHECK(cudaMemcpy(fluid_view.pressure,
                                linear_pressure.data(),
                                linear_pressure.size() * sizeof(lsmps3d::real),
                                cudaMemcpyHostToDevice));
  lsmps3d::SimulationConfig pressure_config = config;
  pressure_config.density = static_cast<lsmps3d::real>(1000.0);
  pressure_config.gravity = pressure_gravity;
  lsmps3d::DeviceLsmpsOperators pressure_wall_lsmps(kCount, pressure_config);
  pressure_wall_lsmps.prepare_matrices(
      fluid_view, wall_view, fluid_neighbor_view, wall_neighbor_view, 2);
  pressure_wall_lsmps.compute_pressure_gradient(
      fluid_view,
      wall_view,
      fluid_neighbor_view,
      wall_neighbor_view,
      fluid_view.pressure,
      operator_view.x,
      operator_view.y,
      operator_view.z);
  std::array<lsmps3d::real, kCount> pressure_wall_grad_x{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(pressure_wall_grad_x.data(),
                                operator_view.x,
                                pressure_wall_grad_x.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  if (!nearly_equal(pressure_wall_grad_x[kCenter],
                    static_cast<lsmps3d::real>(-9800.0),
                    tolerance,
                    "pressure Neumann wall gradient")) {
    return 1;
  }

  return 0;
}
