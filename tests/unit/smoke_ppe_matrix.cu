#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/ppe/ppe_matrix.cuh"
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

bool file_contains(const char* path, const char* needle) {
  std::ifstream input(path);
  if (!input) {
    return false;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str().find(needle) != std::string::npos;
}

std::string amgx_config_path() {
  if (file_contains("configs/amgx_ppe.json", "\"solver\": \"GMRES\"")) {
    return "configs/amgx_ppe.json";
  }
  if (file_contains("../configs/amgx_ppe.json", "\"solver\": \"GMRES\"")) {
    return "../configs/amgx_ppe.json";
  }
  return "/workspace/configs/amgx_ppe.json";
}

bool run_amgx_solve_checks() {
  if (!lsmps3d::AmgxPpeSolver::is_available()) {
    return true;
  }

  std::array<lsmps3d::index_t, 3> row_offsets{0, 2, 4};
  std::array<lsmps3d::index_t, 4> col_indices{0, 1, 0, 1};
  std::array<lsmps3d::real, 4> values{static_cast<lsmps3d::real>(4),
                                      static_cast<lsmps3d::real>(-1),
                                      static_cast<lsmps3d::real>(-1),
                                      static_cast<lsmps3d::real>(4)};
  std::array<lsmps3d::real, 2> rhs{static_cast<lsmps3d::real>(3),
                                  static_cast<lsmps3d::real>(3)};
  std::array<lsmps3d::real, 2> host_pressure{};

  lsmps3d::index_t* device_row_offsets{};
  lsmps3d::index_t* device_col_indices{};
  lsmps3d::real* device_values{};
  lsmps3d::real* device_rhs{};
  lsmps3d::real* device_pressure{};
  LSMPS3D_CUDA_CHECK(cudaMalloc(&device_row_offsets, row_offsets.size() * sizeof(lsmps3d::index_t)));
  LSMPS3D_CUDA_CHECK(cudaMalloc(&device_col_indices, col_indices.size() * sizeof(lsmps3d::index_t)));
  LSMPS3D_CUDA_CHECK(cudaMalloc(&device_values, values.size() * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMalloc(&device_rhs, rhs.size() * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMalloc(&device_pressure, host_pressure.size() * sizeof(lsmps3d::real)));

  copy_to_device(device_row_offsets, row_offsets);
  copy_to_device(device_col_indices, col_indices);
  copy_to_device(device_values, values);
  copy_to_device(device_rhs, rhs);

  lsmps3d::CsrMatrixView matrix{2, 2, 4, device_row_offsets, device_col_indices, device_values};
  lsmps3d::AmgxPpeSolver solver(amgx_config_path());
  solver.solve(matrix, device_rhs, device_pressure);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(host_pressure.data(),
                                device_pressure,
                                host_pressure.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));

  const lsmps3d::real solve_tolerance = static_cast<lsmps3d::real>(2.0e-5);
  bool ok = nearly_equal(host_pressure[0], static_cast<lsmps3d::real>(1), solve_tolerance, "AMGX p0") &&
            nearly_equal(host_pressure[1], static_cast<lsmps3d::real>(1), solve_tolerance, "AMGX p1");

  rhs = {static_cast<lsmps3d::real>(-3), static_cast<lsmps3d::real>(-3)};
  copy_to_device(device_rhs, rhs);
  solver.solve(matrix, device_rhs, device_pressure);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(host_pressure.data(),
                                device_pressure,
                                host_pressure.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  ok = ok &&
       nearly_equal(host_pressure[0], static_cast<lsmps3d::real>(0), solve_tolerance, "AMGX clamp p0") &&
       nearly_equal(host_pressure[1], static_cast<lsmps3d::real>(0), solve_tolerance, "AMGX clamp p1");

  cudaFree(device_pressure);
  cudaFree(device_rhs);
  cudaFree(device_values);
  cudaFree(device_col_indices);
  cudaFree(device_row_offsets);
  return ok;
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
  std::array<int, kCount> surface_type{};

  lsmps3d::size_type out = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        x[out] = static_cast<lsmps3d::real>(ix);
        y[out] = static_cast<lsmps3d::real>(iy);
        z[out] = static_cast<lsmps3d::real>(iz);
        vx[out] = x[out];
        vy[out] = static_cast<lsmps3d::real>(2) * y[out];
        vz[out] = static_cast<lsmps3d::real>(3) * z[out];
        surface_type[out] = static_cast<int>(lsmps3d::SurfaceType::Inner);
        ++out;
      }
    }
  }
  surface_type[0] = static_cast<int>(lsmps3d::SurfaceType::Surface);

  lsmps3d::DeviceFluidParticles fluid(kCount);
  lsmps3d::DeviceWallParticles walls(0);
  lsmps3d::DeviceFluidParticles temporary_velocity(kCount);
  lsmps3d::DeviceWallParticles temporary_wall_velocity(0);
  lsmps3d::DeviceNeighborList fluid_neighbors(kCount, kCount * (kCount - 1));
  lsmps3d::DeviceNeighborList wall_neighbors(kCount, 0);
  auto fluid_view = fluid.view();
  auto wall_view = walls.view();
  auto temporary_velocity_view = temporary_velocity.view();
  auto temporary_wall_velocity_view = temporary_wall_velocity.view();
  auto fluid_neighbor_view = fluid_neighbors.view();
  auto wall_neighbor_view = wall_neighbors.view();

  copy_to_device(fluid_view.x, x);
  copy_to_device(fluid_view.y, y);
  copy_to_device(fluid_view.z, z);
  copy_to_device(fluid_view.vx, vx);
  copy_to_device(fluid_view.vy, vy);
  copy_to_device(fluid_view.vz, vz);
  copy_to_device(fluid_view.surface_type, surface_type);
  copy_to_device(temporary_velocity_view.vx, vx);
  copy_to_device(temporary_velocity_view.vy, vy);
  copy_to_device(temporary_velocity_view.vz, vz);

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
  config.lsmps_regularization = static_cast<lsmps3d::real>(1.0e-10);
  config.time_step = static_cast<lsmps3d::real>(0.5);
  config.density = static_cast<lsmps3d::real>(2);
  config.gravity = {};

  const lsmps3d::size_type expected_nnz = kCount + indices.size();
  lsmps3d::DeviceMomentMatrix lsmps(kCount, config);
  lsmps3d::DevicePpeMatrixAssembler assembler(kCount, expected_nnz, config);
  assembler.assemble(fluid_view,
                     wall_view,
                     fluid_neighbor_view,
                     wall_neighbor_view,
                     temporary_velocity_view,
                     temporary_wall_velocity_view,
                     lsmps,
                     1);
  auto ppe = assembler.workspace();

  std::vector<lsmps3d::index_t> row_offsets(kCount + 1);
  std::vector<lsmps3d::index_t> col_indices(expected_nnz);
  std::vector<lsmps3d::real> values(expected_nnz);
  std::array<lsmps3d::real, kCount> rhs{};
  std::array<lsmps3d::real, kCount> divergence{};
  LSMPS3D_CUDA_CHECK(cudaMemcpy(row_offsets.data(),
                                ppe.matrix.row_offsets,
                                row_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(col_indices.data(),
                                ppe.matrix.col_indices,
                                col_indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(values.data(),
                                ppe.matrix.values,
                                values.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      rhs.data(), ppe.rhs, rhs.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(divergence.data(),
                                ppe.divergence,
                                divergence.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));

  if (ppe.matrix.nnz != expected_nnz || row_offsets.front() != 0 ||
      row_offsets.back() != static_cast<lsmps3d::index_t>(expected_nnz)) {
    std::cerr << "PPE CSR offsets or nnz are inconsistent" << std::endl;
    return 1;
  }
  if (col_indices[row_offsets[0]] != 0 ||
      !nearly_equal(values[row_offsets[0]], static_cast<lsmps3d::real>(1), 0, "surface diag") ||
      !nearly_equal(rhs[0], static_cast<lsmps3d::real>(0), 0, "surface rhs")) {
    return 1;
  }

  const lsmps3d::real tolerance = static_cast<lsmps3d::real>(2.0e-3);
  if (!nearly_equal(divergence[kCenter], static_cast<lsmps3d::real>(6), tolerance, "divergence") ||
      !nearly_equal(rhs[kCenter],
                    static_cast<lsmps3d::real>(6) / config.time_step,
                    tolerance,
                    "rhs")) {
    return 1;
  }

  lsmps3d::real center_diag = static_cast<lsmps3d::real>(0);
  lsmps3d::real center_offdiag_sum = static_cast<lsmps3d::real>(0);
  for (lsmps3d::index_t cursor = row_offsets[kCenter]; cursor < row_offsets[kCenter + 1];
       ++cursor) {
    if (col_indices[cursor] == static_cast<lsmps3d::index_t>(kCenter)) {
      center_diag = values[cursor];
    } else {
      center_offdiag_sum += values[cursor];
    }
  }
  if (!std::isfinite(center_diag) || !std::isfinite(center_offdiag_sum) ||
      std::abs(center_diag) <= static_cast<lsmps3d::real>(0)) {
    std::cerr << "PPE LSMPS center row is invalid" << std::endl;
    return 1;
  }

  const lsmps3d::size_type expected_bytes =
      (kCount + 1) * sizeof(lsmps3d::index_t) +
      expected_nnz * (sizeof(lsmps3d::index_t) + sizeof(lsmps3d::real)) +
      kCount * 4 * sizeof(lsmps3d::real);
  if (assembler.bytes() != expected_bytes) {
    std::cerr << "PPE workspace byte accounting mismatch" << std::endl;
    return 1;
  }

  if (!file_contains("configs/amgx_ppe.json", "\"solver\": \"GMRES\"") &&
      !file_contains("../configs/amgx_ppe.json", "\"solver\": \"GMRES\"")) {
    std::cerr << "PPE AMGX config must use GMRES for the non-symmetric CSR system" << std::endl;
    return 1;
  }

  if (lsmps3d::AmgxPpeSolver::is_available()) {
    lsmps3d::AmgxPpeSolver solver(config.amgx_config_path);
    (void)solver.config_path();
    if (!run_amgx_solve_checks()) {
      return 1;
    }
  }

  return 0;
}
