#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/correction/pressure_correction.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

template <typename T, std::size_t N>
void copy_to_device(T* dst, const std::array<T, N>& src) {
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
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
  std::array<lsmps3d::real, kCount> vx{};
  std::array<lsmps3d::real, kCount> vy{};
  std::array<lsmps3d::real, kCount> vz{};
  std::array<lsmps3d::real, kCount> pressure{};
  std::array<int, kCount> surface_type{};

  lsmps3d::size_type out = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        x[out] = static_cast<lsmps3d::real>(ix);
        y[out] = static_cast<lsmps3d::real>(iy);
        z[out] = static_cast<lsmps3d::real>(iz);
        vx[out] = static_cast<lsmps3d::real>(0.5);
        vy[out] = static_cast<lsmps3d::real>(-0.25);
        vz[out] = static_cast<lsmps3d::real>(0.75);
        pressure[out] = static_cast<lsmps3d::real>(20) +
                        static_cast<lsmps3d::real>(2) * x[out] -
                        static_cast<lsmps3d::real>(3) * y[out] +
                        static_cast<lsmps3d::real>(4) * z[out];
        surface_type[out] = static_cast<int>(lsmps3d::SurfaceType::Inner);
        ++out;
      }
    }
  }
  surface_type[0] = static_cast<int>(lsmps3d::SurfaceType::Splash);

  lsmps3d::DeviceFluidParticles fluid(kCount);
  lsmps3d::DeviceWallParticles walls(0);
  lsmps3d::DeviceFluidParticles temporary_velocity(kCount);
  lsmps3d::DeviceNeighborList fluid_neighbors(kCount, kCount * (kCount - 1));
  lsmps3d::DeviceNeighborList wall_neighbors(kCount, 0);
  auto fluid_view = fluid.view();
  auto temporary_view = temporary_velocity.view();
  auto wall_view = walls.view();
  auto fluid_neighbor_view = fluid_neighbors.view();
  auto wall_neighbor_view = wall_neighbors.view();

  copy_to_device(fluid_view.x, x);
  copy_to_device(fluid_view.y, y);
  copy_to_device(fluid_view.z, z);
  copy_to_device(fluid_view.vx, vx);
  copy_to_device(fluid_view.vy, vy);
  copy_to_device(fluid_view.vz, vz);
  copy_to_device(fluid_view.pressure, pressure);
  copy_to_device(fluid_view.surface_type, surface_type);
  copy_to_device(temporary_view.vx, vx);
  copy_to_device(temporary_view.vy, vy);
  copy_to_device(temporary_view.vz, vz);

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
  config.support_radius = static_cast<lsmps3d::real>(4);
  config.cell_size = static_cast<lsmps3d::real>(4);
  config.particle_spacing = static_cast<lsmps3d::real>(1);
  config.lsmps_regularization = static_cast<lsmps3d::real>(1.0e-10);
  config.time_step = static_cast<lsmps3d::real>(0.5);
  config.density = static_cast<lsmps3d::real>(2);
  config.ps_displacement_scale = static_cast<lsmps3d::real>(0);
  config.wall_clearance_ratio = static_cast<lsmps3d::real>(0);
  config.velocity_smoothing_strength = static_cast<lsmps3d::real>(0);

  lsmps3d::DeviceMomentMatrix lsmps(kCount, config);
  lsmps3d::DevicePressureCorrection correction(kCount, config);
  correction.apply(fluid_view,
                   wall_view,
                   fluid_neighbor_view,
                   wall_neighbor_view,
                   temporary_view,
                   fluid_view.pressure,
                   lsmps,
                   1);

  std::array<lsmps3d::real, kCount> out_x{};
  std::array<lsmps3d::real, kCount> out_y{};
  std::array<lsmps3d::real, kCount> out_z{};
  std::array<lsmps3d::real, kCount> out_vx{};
  std::array<lsmps3d::real, kCount> out_vy{};
  std::array<lsmps3d::real, kCount> out_vz{};
  std::array<lsmps3d::real, kCount> out_pressure{};
  std::array<lsmps3d::real, kCount> grad_x{};
  std::array<lsmps3d::real, kCount> grad_y{};
  std::array<lsmps3d::real, kCount> grad_z{};
  auto gradient = correction.pressure_gradient();
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(out_x.data(), fluid_view.x, out_x.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(out_y.data(), fluid_view.y, out_y.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(out_z.data(), fluid_view.z, out_z.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      out_vx.data(), fluid_view.vx, out_vx.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      out_vy.data(), fluid_view.vy, out_vy.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      out_vz.data(), fluid_view.vz, out_vz.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(out_pressure.data(),
                                fluid_view.pressure,
                                out_pressure.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(grad_x.data(), gradient.x, grad_x.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(grad_y.data(), gradient.y, grad_y.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(
      cudaMemcpy(grad_z.data(), gradient.z, grad_z.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));

  const lsmps3d::real tolerance = static_cast<lsmps3d::real>(2.0e-3);
  if (!nearly_equal(grad_x[kCenter], static_cast<lsmps3d::real>(2), tolerance, "pressure grad x") ||
      !nearly_equal(grad_y[kCenter], static_cast<lsmps3d::real>(-3), tolerance, "pressure grad y") ||
      !nearly_equal(grad_z[kCenter], static_cast<lsmps3d::real>(4), tolerance, "pressure grad z")) {
    return 1;
  }

  const lsmps3d::real scale = config.time_step / config.density;
  const lsmps3d::real expected_vx = vx[kCenter] - scale * static_cast<lsmps3d::real>(2);
  const lsmps3d::real expected_vy = vy[kCenter] - scale * static_cast<lsmps3d::real>(-3);
  const lsmps3d::real expected_vz = vz[kCenter] - scale * static_cast<lsmps3d::real>(4);
  if (!nearly_equal(out_vx[kCenter], expected_vx, tolerance, "corrected vx") ||
      !nearly_equal(out_vy[kCenter], expected_vy, tolerance, "corrected vy") ||
      !nearly_equal(out_vz[kCenter], expected_vz, tolerance, "corrected vz")) {
    return 1;
  }

  if (!nearly_equal(out_x[kCenter],
                    x[kCenter] + static_cast<lsmps3d::real>(0.5) * config.time_step *
                                    (vx[kCenter] + expected_vx),
                    tolerance,
                    "updated x")) {
    return 1;
  }

  if (correction.bytes() != kCount * 9 * sizeof(lsmps3d::real)) {
    std::cerr << "Correction workspace byte accounting mismatch" << std::endl;
    return 1;
  }

  return 0;
}
