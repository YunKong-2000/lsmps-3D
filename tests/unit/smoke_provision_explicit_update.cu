#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/provision/explicit_update.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

template <typename T, std::size_t N>
void copy_to_device(T* dst, const std::array<T, N>& src) {
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

lsmps3d::real velocity_x(lsmps3d::real x) {
  return static_cast<lsmps3d::real>(1.5) + static_cast<lsmps3d::real>(0.5) * x +
         static_cast<lsmps3d::real>(2.0) * x * x;
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
  constexpr lsmps3d::size_type kWallCount = 2;

  std::array<lsmps3d::real, kCount> x{};
  std::array<lsmps3d::real, kCount> y{};
  std::array<lsmps3d::real, kCount> z{};
  std::array<lsmps3d::real, kCount> vx{};
  std::array<lsmps3d::real, kCount> vy{};
  std::array<lsmps3d::real, kCount> vz{};
  std::array<int, kCount> surface_type{};
  std::array<lsmps3d::real, kWallCount> wall_x{static_cast<lsmps3d::real>(-2),
                                               static_cast<lsmps3d::real>(2)};
  std::array<lsmps3d::real, kWallCount> wall_y{};
  std::array<lsmps3d::real, kWallCount> wall_z{};
  std::array<lsmps3d::real, kWallCount> wall_vx{static_cast<lsmps3d::real>(0.4),
                                                static_cast<lsmps3d::real>(-0.1)};
  std::array<lsmps3d::real, kWallCount> wall_vy{static_cast<lsmps3d::real>(0.2),
                                                static_cast<lsmps3d::real>(0.3)};
  std::array<lsmps3d::real, kWallCount> wall_vz{static_cast<lsmps3d::real>(-0.5),
                                                static_cast<lsmps3d::real>(0.7)};

  lsmps3d::size_type out = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        x[out] = static_cast<lsmps3d::real>(ix);
        y[out] = static_cast<lsmps3d::real>(iy);
        z[out] = static_cast<lsmps3d::real>(iz);
        vx[out] = velocity_x(x[out]);
        vy[out] = static_cast<lsmps3d::real>(-2.0);
        vz[out] = static_cast<lsmps3d::real>(0.25) * z[out];
        surface_type[out] = static_cast<int>(lsmps3d::SurfaceType::Inner);
        ++out;
      }
    }
  }
  surface_type[0] = static_cast<int>(lsmps3d::SurfaceType::Splash);

  lsmps3d::DeviceFluidParticles fluid(kCount);
  lsmps3d::DeviceFluidParticles temporary(kCount);
  lsmps3d::DeviceWallParticles walls(kWallCount);
  lsmps3d::DeviceWallParticles temporary_walls(kWallCount);
  lsmps3d::DeviceNeighborList fluid_neighbors(kCount, kCount * (kCount - 1));
  lsmps3d::DeviceNeighborList wall_neighbors(kCount, 0);
  auto fluid_view = fluid.view();
  auto temporary_view = temporary.view();
  auto wall_view = walls.view();
  auto temporary_wall_view = temporary_walls.view();
  auto fluid_neighbor_view = fluid_neighbors.view();
  auto wall_neighbor_view = wall_neighbors.view();

  copy_to_device(fluid_view.x, x);
  copy_to_device(fluid_view.y, y);
  copy_to_device(fluid_view.z, z);
  copy_to_device(fluid_view.vx, vx);
  copy_to_device(fluid_view.vy, vy);
  copy_to_device(fluid_view.vz, vz);
  copy_to_device(fluid_view.surface_type, surface_type);
  copy_to_device(wall_view.x, wall_x);
  copy_to_device(wall_view.y, wall_y);
  copy_to_device(wall_view.z, wall_z);
  copy_to_device(wall_view.vx, wall_vx);
  copy_to_device(wall_view.vy, wall_vy);
  copy_to_device(wall_view.vz, wall_vz);

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
  config.kinematic_viscosity = static_cast<lsmps3d::real>(0.25);
  config.time_step = static_cast<lsmps3d::real>(0.2);
  config.gravity = lsmps3d::Vec3{static_cast<lsmps3d::real>(0.1),
                                 static_cast<lsmps3d::real>(-0.2),
                                 static_cast<lsmps3d::real>(0.3)};

  lsmps3d::DeviceMomentMatrix lsmps(kCount, config);
  lsmps3d::DeviceProvisionExplicitUpdate provision(kCount, config);
  provision.compute_temporary_velocity(fluid_view,
                                       wall_view,
                                       fluid_neighbor_view,
                                       wall_neighbor_view,
                                       lsmps,
                                       1,
                                       temporary_view,
                                       temporary_wall_view);

  std::array<lsmps3d::real, kCount> temporary_x{};
  std::array<lsmps3d::real, kCount> temporary_y{};
  std::array<lsmps3d::real, kCount> temporary_z{};
  std::array<lsmps3d::real, kWallCount> temporary_wall_x{};
  std::array<lsmps3d::real, kWallCount> temporary_wall_y{};
  std::array<lsmps3d::real, kWallCount> temporary_wall_z{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_x.data(),
                                temporary_view.vx,
                                temporary_x.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_y.data(),
                                temporary_view.vy,
                                temporary_y.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_z.data(),
                                temporary_view.vz,
                                temporary_z.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_wall_x.data(),
                                temporary_wall_view.vx,
                                temporary_wall_x.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_wall_y.data(),
                                temporary_wall_view.vy,
                                temporary_wall_y.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(temporary_wall_z.data(),
                                temporary_wall_view.vz,
                                temporary_wall_z.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));

  const lsmps3d::real tolerance = static_cast<lsmps3d::real>(2.0e-3);
  const lsmps3d::real expected_laplacian_x = static_cast<lsmps3d::real>(4.0);
  const lsmps3d::real expected_center_x =
      vx[kCenter] +
      config.time_step *
          (config.kinematic_viscosity * expected_laplacian_x + config.gravity.x);
  const lsmps3d::real expected_center_y = vy[kCenter] + config.time_step * config.gravity.y;
  const lsmps3d::real expected_center_z = vz[kCenter] + config.time_step * config.gravity.z;
  if (!nearly_equal(temporary_x[kCenter], expected_center_x, tolerance, "temporary vx") ||
      !nearly_equal(temporary_y[kCenter], expected_center_y, tolerance, "temporary vy") ||
      !nearly_equal(temporary_z[kCenter], expected_center_z, tolerance, "temporary vz")) {
    return 1;
  }

  const lsmps3d::real expected_splash_x = vx[0] + config.time_step * config.gravity.x;
  if (!nearly_equal(temporary_x[0], expected_splash_x, tolerance, "splash temporary vx")) {
    return 1;
  }

  const lsmps3d::real expected_wall_x = wall_vx[0] + config.time_step * config.gravity.x;
  const lsmps3d::real expected_wall_y = wall_vy[0] + config.time_step * config.gravity.y;
  const lsmps3d::real expected_wall_z = wall_vz[0] + config.time_step * config.gravity.z;
  if (!nearly_equal(temporary_wall_x[0], expected_wall_x, tolerance, "temporary wall vx") ||
      !nearly_equal(temporary_wall_y[0], expected_wall_y, tolerance, "temporary wall vy") ||
      !nearly_equal(temporary_wall_z[0], expected_wall_z, tolerance, "temporary wall vz")) {
    return 1;
  }

  if (provision.bytes() != (kCount + kWallCount) * 3 * sizeof(lsmps3d::real)) {
    std::cerr << "Provision workspace byte accounting mismatch" << std::endl;
    return 1;
  }

  return 0;
}
