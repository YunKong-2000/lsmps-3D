#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/io/vtk_writer.hpp"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"
#include "lsmps3d/ppe/ppe_matrix.cuh"
#include "lsmps3d/provision/explicit_update.cuh"
#include "lsmps3d/surface/surface_detection.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

struct HydrostaticCase {
  std::vector<lsmps3d::real> fluid_x;
  std::vector<lsmps3d::real> fluid_y;
  std::vector<lsmps3d::real> fluid_z;
  std::vector<lsmps3d::real> wall_x;
  std::vector<lsmps3d::real> wall_y;
  std::vector<lsmps3d::real> wall_z;
  std::vector<lsmps3d::real> wall_normal_x;
  std::vector<lsmps3d::real> wall_normal_y;
  std::vector<lsmps3d::real> wall_normal_z;
};

struct CsrRowDiagnostics {
  std::vector<lsmps3d::real> diagonal;
  std::vector<lsmps3d::real> row_sum;
  std::vector<int> row_nnz;
};

struct CsrResidualDiagnostics {
  std::vector<lsmps3d::real> residual;
  std::vector<lsmps3d::real> residual_plus_wall_rhs;
  std::vector<lsmps3d::real> residual_minus_wall_rhs;
  std::vector<lsmps3d::real> residual_plus_two_wall_rhs;
  std::vector<lsmps3d::real> residual_minus_two_wall_rhs;
};

const char* surface_type_name(int type) {
  switch (static_cast<lsmps3d::SurfaceType>(type)) {
    case lsmps3d::SurfaceType::Inner:
      return "Inner";
    case lsmps3d::SurfaceType::NearSurface:
      return "NearSurface";
    case lsmps3d::SurfaceType::Surface:
      return "Surface";
    case lsmps3d::SurfaceType::Splash:
      return "Splash";
  }
  return "Unknown";
}

HydrostaticCase make_half_full_box(lsmps3d::real spacing,
                                   lsmps3d::real box_size,
                                   lsmps3d::real liquid_height) {
  HydrostaticCase result;
  const int cells_per_side = static_cast<int>(std::lround(box_size / spacing));
  const int fluid_layers = static_cast<int>(std::lround(liquid_height / spacing));

  result.fluid_x.reserve(static_cast<std::size_t>(cells_per_side) * cells_per_side * fluid_layers);
  result.fluid_y.reserve(result.fluid_x.capacity());
  result.fluid_z.reserve(result.fluid_x.capacity());

  for (int iz = 0; iz < fluid_layers; ++iz) {
    for (int iy = 0; iy < cells_per_side; ++iy) {
      for (int ix = 0; ix < cells_per_side; ++ix) {
        result.fluid_x.push_back((static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) *
                                 spacing);
        result.fluid_y.push_back((static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                                 spacing);
        result.fluid_z.push_back((static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                                 spacing);
      }
    }
  }

  std::vector<std::tuple<int, int, int, lsmps3d::real, lsmps3d::real, lsmps3d::real>> wall_indices;
  wall_indices.reserve(static_cast<std::size_t>(5) * (cells_per_side + 1) * (cells_per_side + 1));
  const int last = cells_per_side;

  for (int iz = 0; iz <= cells_per_side; ++iz) {
    for (int iy = 0; iy <= cells_per_side; ++iy) {
      wall_indices.emplace_back(0, iy, iz, static_cast<lsmps3d::real>(1), static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(0));
      wall_indices.emplace_back(last, iy, iz, static_cast<lsmps3d::real>(-1), static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(0));
    }
    for (int ix = 1; ix < last; ++ix) {
      wall_indices.emplace_back(ix, 0, iz, static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(1), static_cast<lsmps3d::real>(0));
      wall_indices.emplace_back(ix, last, iz, static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(-1), static_cast<lsmps3d::real>(0));
    }
  }
  for (int iy = 1; iy < last; ++iy) {
    for (int ix = 1; ix < last; ++ix) {
      wall_indices.emplace_back(ix, iy, 0, static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(1));
    }
  }

  result.wall_x.reserve(wall_indices.size());
  result.wall_y.reserve(wall_indices.size());
  result.wall_z.reserve(wall_indices.size());
  result.wall_normal_x.reserve(wall_indices.size());
  result.wall_normal_y.reserve(wall_indices.size());
  result.wall_normal_z.reserve(wall_indices.size());
  for (const auto& [ix, iy, iz, nx, ny, nz] : wall_indices) {
    result.wall_x.push_back(static_cast<lsmps3d::real>(ix) * spacing);
    result.wall_y.push_back(static_cast<lsmps3d::real>(iy) * spacing);
    result.wall_z.push_back(static_cast<lsmps3d::real>(iz) * spacing);
    result.wall_normal_x.push_back(nx);
    result.wall_normal_y.push_back(ny);
    result.wall_normal_z.push_back(nz);
  }

  return result;
}

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src) {
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_from_device(const T* src, std::size_t count) {
  std::vector<T> dst(count);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return dst;
}

void write_csv(const std::filesystem::path& path,
               const lsmps3d::HostParticleSnapshot& particles,
               const std::vector<lsmps3d::index_t>& fluid_neighbor_count,
               const std::vector<lsmps3d::index_t>& wall_neighbor_count,
               const std::vector<lsmps3d::real>& number_density,
               const std::vector<lsmps3d::real>& number_density_ratio,
               const std::vector<lsmps3d::real>& anisotropy,
               const std::vector<lsmps3d::real>& air_open_ratio,
               const std::vector<lsmps3d::real>& air_anisotropy,
               const std::vector<lsmps3d::real>& surface_normal_x,
               const std::vector<lsmps3d::real>& surface_normal_y,
               const std::vector<lsmps3d::real>& surface_normal_z,
               const std::vector<int>& surface_type) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open CSV output: " + path.string());
  }

  out << std::setprecision(9);
  out << "particle_id,x,y,z,fluid_neighbor_count,wall_neighbor_count,number_density,"
         "number_density_ratio,anisotropy,air_open_ratio,air_anisotropy,surface_normal_x,"
         "surface_normal_y,surface_normal_z,surface_type,surface_type_name\n";
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    out << i << ',' << particles.x[i] << ',' << particles.y[i] << ',' << particles.z[i] << ','
        << fluid_neighbor_count[i] << ',' << wall_neighbor_count[i] << ',' << number_density[i]
        << ',' << number_density_ratio[i] << ',' << anisotropy[i] << ',' << air_open_ratio[i]
        << ',' << air_anisotropy[i] << ',' << surface_normal_x[i] << ',' << surface_normal_y[i]
        << ',' << surface_normal_z[i] << ',' << surface_type[i] << ','
        << surface_type_name(surface_type[i]) << '\n';
  }
}

std::vector<lsmps3d::real> make_hydrostatic_pressure(const std::vector<lsmps3d::real>& z,
                                                     lsmps3d::real liquid_height,
                                                     lsmps3d::real density,
                                                     lsmps3d::real gravity_magnitude,
                                                     lsmps3d::real particle_spacing) {
  std::vector<lsmps3d::real> pressure(z.size());
  const lsmps3d::real free_surface_center_z =
      liquid_height - static_cast<lsmps3d::real>(0.5) * particle_spacing;
  for (std::size_t i = 0; i < z.size(); ++i) {
    pressure[i] = density * gravity_magnitude *
                  std::max(free_surface_center_z - z[i], static_cast<lsmps3d::real>(0));
  }
  return pressure;
}

std::vector<lsmps3d::real> absolute_values(const std::vector<lsmps3d::real>& values) {
  std::vector<lsmps3d::real> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = std::abs(values[i]);
  }
  return result;
}

std::vector<lsmps3d::real> subtract_values(const std::vector<lsmps3d::real>& lhs,
                                           const std::vector<lsmps3d::real>& rhs) {
  std::vector<lsmps3d::real> result(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] - rhs[i];
  }
  return result;
}

lsmps3d::real max_value(const std::vector<lsmps3d::real>& values) {
  lsmps3d::real result = static_cast<lsmps3d::real>(0);
  for (const lsmps3d::real value : values) {
    result = std::max(result, value);
  }
  return result;
}

std::filesystem::path resolve_amgx_config_path() {
  const std::array<std::filesystem::path, 4> candidates{
      std::filesystem::path("configs/amgx_ppe.json"),
      std::filesystem::path("../configs/amgx_ppe.json"),
      std::filesystem::path("../../configs/amgx_ppe.json"),
      std::filesystem::path("/workspace/configs/amgx_ppe.json"),
  };
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return candidates.front();
}

CsrRowDiagnostics make_csr_row_diagnostics(const lsmps3d::CsrMatrixView& matrix) {
  std::vector<lsmps3d::index_t> row_offsets(matrix.rows + 1);
  std::vector<lsmps3d::index_t> col_indices(matrix.nnz);
  std::vector<lsmps3d::real> values(matrix.nnz);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(row_offsets.data(),
                                matrix.row_offsets,
                                row_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(col_indices.data(),
                                matrix.col_indices,
                                col_indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      values.data(), matrix.values, values.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));

  CsrRowDiagnostics diagnostics;
  diagnostics.diagonal.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.row_sum.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.row_nnz.assign(matrix.rows, 0);
  for (lsmps3d::size_type row = 0; row < matrix.rows; ++row) {
    const lsmps3d::index_t row_begin = row_offsets[row];
    const lsmps3d::index_t row_end = row_offsets[row + 1];
    diagnostics.row_nnz[row] = static_cast<int>(row_end - row_begin);
    for (lsmps3d::index_t cursor = row_begin; cursor < row_end; ++cursor) {
      const lsmps3d::real value = values[cursor];
      diagnostics.row_sum[row] += value;
      if (col_indices[cursor] == static_cast<lsmps3d::index_t>(row)) {
        diagnostics.diagonal[row] = value;
      }
    }
  }
  return diagnostics;
}

CsrResidualDiagnostics make_csr_residual_diagnostics(
    const lsmps3d::CsrMatrixView& matrix,
    const std::vector<lsmps3d::real>& pressure,
    const std::vector<lsmps3d::real>& rhs,
    const std::vector<lsmps3d::real>& wall_rhs) {
  std::vector<lsmps3d::index_t> row_offsets(matrix.rows + 1);
  std::vector<lsmps3d::index_t> col_indices(matrix.nnz);
  std::vector<lsmps3d::real> values(matrix.nnz);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(row_offsets.data(),
                                matrix.row_offsets,
                                row_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(col_indices.data(),
                                matrix.col_indices,
                                col_indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      values.data(), matrix.values, values.size() * sizeof(lsmps3d::real), cudaMemcpyDeviceToHost));

  CsrResidualDiagnostics diagnostics;
  diagnostics.residual.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.residual_plus_wall_rhs.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.residual_minus_wall_rhs.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.residual_plus_two_wall_rhs.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  diagnostics.residual_minus_two_wall_rhs.assign(matrix.rows, static_cast<lsmps3d::real>(0));
  for (lsmps3d::size_type row = 0; row < matrix.rows; ++row) {
    lsmps3d::real ap = static_cast<lsmps3d::real>(0);
    for (lsmps3d::index_t cursor = row_offsets[row]; cursor < row_offsets[row + 1]; ++cursor) {
      const lsmps3d::index_t col = col_indices[cursor];
      ap += values[cursor] * pressure[static_cast<std::size_t>(col)];
    }
    const lsmps3d::real residual = ap - rhs[static_cast<std::size_t>(row)];
    const lsmps3d::real wall = wall_rhs[static_cast<std::size_t>(row)];
    diagnostics.residual[static_cast<std::size_t>(row)] = residual;
    diagnostics.residual_plus_wall_rhs[static_cast<std::size_t>(row)] = residual + wall;
    diagnostics.residual_minus_wall_rhs[static_cast<std::size_t>(row)] = residual - wall;
    diagnostics.residual_plus_two_wall_rhs[static_cast<std::size_t>(row)] =
        residual + static_cast<lsmps3d::real>(2) * wall;
    diagnostics.residual_minus_two_wall_rhs[static_cast<std::size_t>(row)] =
        residual - static_cast<lsmps3d::real>(2) * wall;
  }
  return diagnostics;
}

__device__ lsmps3d::real lsmps_weight(lsmps3d::real distance, lsmps3d::real support_radius) {
  if (distance >= support_radius) {
    return static_cast<lsmps3d::real>(0);
  }
  return static_cast<lsmps3d::real>(1) - distance / support_radius;
}

__device__ void type_a_basis(lsmps3d::real dx,
                             lsmps3d::real dy,
                             lsmps3d::real dz,
                             lsmps3d::real support_radius,
                             lsmps3d::real basis[lsmps3d::kMomentTypeABasis3DSize]) {
  const lsmps3d::real inv_support = static_cast<lsmps3d::real>(1) / support_radius;
  const lsmps3d::real sx = dx * inv_support;
  const lsmps3d::real sy = dy * inv_support;
  const lsmps3d::real sz = dz * inv_support;
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

__device__ void wall_neumann_vector(lsmps3d::real dx,
                                    lsmps3d::real dy,
                                    lsmps3d::real dz,
                                    lsmps3d::real normal_x,
                                    lsmps3d::real normal_y,
                                    lsmps3d::real normal_z,
                                    lsmps3d::real support_radius,
                                    lsmps3d::real q[lsmps3d::kMomentTypeABasis3DSize]) {
  const lsmps3d::real inv_support = static_cast<lsmps3d::real>(1) / support_radius;
  q[0] = normal_x;
  q[1] = normal_y;
  q[2] = normal_z;
  q[3] = static_cast<lsmps3d::real>(2) * dx * normal_x * inv_support;
  q[4] = static_cast<lsmps3d::real>(2) * dy * normal_y * inv_support;
  q[5] = static_cast<lsmps3d::real>(2) * dz * normal_z * inv_support;
  q[6] = (dy * normal_x + dx * normal_y) * inv_support;
  q[7] = (dz * normal_y + dy * normal_z) * inv_support;
  q[8] = (dx * normal_z + dz * normal_x) * inv_support;
}

__device__ lsmps3d::real inverse_rhs_row(
    const lsmps3d::real* inverse_matrix,
    const lsmps3d::real rhs[lsmps3d::kMomentTypeABasis3DSize],
    int row) {
  lsmps3d::real value = static_cast<lsmps3d::real>(0);
  for (int col = 0; col < lsmps3d::kMomentTypeABasis3DSize; ++col) {
    value += inverse_matrix[row + col * lsmps3d::kMomentTypeABasis3DSize] * rhs[col];
  }
  return value;
}

__global__ void compute_pressure_operator_kernel(const lsmps3d::FluidParticleSoA fluid,
                                                 const lsmps3d::WallParticleSoA walls,
                                                 const lsmps3d::NeighborListView fluid_neighbors,
                                                 const lsmps3d::NeighborListView wall_neighbors,
                                                 lsmps3d::MomentMatrixView pressure_moment,
                                                 lsmps3d::real density,
                                                 lsmps3d::Vec3 gravity,
                                                 lsmps3d::real* gradient_x,
                                                 lsmps3d::real* gradient_y,
                                                 lsmps3d::real* gradient_z,
                                                 lsmps3d::real* laplacian) {
  constexpr int kBasisSize = lsmps3d::kMomentTypeABasis3DSize;
  const lsmps3d::size_type i =
      static_cast<lsmps3d::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const lsmps3d::real xi = fluid.x[i];
  const lsmps3d::real yi = fluid.y[i];
  const lsmps3d::real zi = fluid.z[i];
  const lsmps3d::real support_radius = pressure_moment.support_radius;
  lsmps3d::real rhs[kBasisSize]{};

  for (lsmps3d::index_t cursor = fluid_neighbors.offsets[i];
       cursor < fluid_neighbors.offsets[i + 1];
       ++cursor) {
    const lsmps3d::index_t j = fluid_neighbors.indices[cursor];
    const lsmps3d::real dx = fluid.x[j] - xi;
    const lsmps3d::real dy = fluid.y[j] - yi;
    const lsmps3d::real dz = fluid.z[j] - zi;
    const lsmps3d::real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const lsmps3d::real weight = lsmps_weight(distance, support_radius);
    if (weight <= static_cast<lsmps3d::real>(0)) {
      continue;
    }
    lsmps3d::real basis[kBasisSize]{};
    type_a_basis(dx, dy, dz, support_radius, basis);
    const lsmps3d::real delta_pressure = fluid.pressure[j] - fluid.pressure[i];
    for (int row = 0; row < kBasisSize; ++row) {
      rhs[row] += weight * basis[row] * delta_pressure;
    }
  }

  for (lsmps3d::index_t cursor = wall_neighbors.offsets[i];
       cursor < wall_neighbors.offsets[i + 1];
       ++cursor) {
    const lsmps3d::index_t j = wall_neighbors.indices[cursor];
    const lsmps3d::real dx = walls.x[j] - xi;
    const lsmps3d::real dy = walls.y[j] - yi;
    const lsmps3d::real dz = walls.z[j] - zi;
    const lsmps3d::real distance = sqrt(dx * dx + dy * dy + dz * dz);
    const lsmps3d::real weight =
        pressure_moment.wall_weight_scale * lsmps_weight(distance, support_radius);
    if (weight <= static_cast<lsmps3d::real>(0)) {
      continue;
    }
    lsmps3d::real q[kBasisSize]{};
    wall_neumann_vector(
        dx, dy, dz, walls.normal_x[j], walls.normal_y[j], walls.normal_z[j], support_radius, q);
    const lsmps3d::real normal_gravity = gravity.x * walls.normal_x[j] +
                                         gravity.y * walls.normal_y[j] +
                                         gravity.z * walls.normal_z[j];
    const lsmps3d::real wall_rhs = support_radius * density * normal_gravity;
    for (int row = 0; row < kBasisSize; ++row) {
      rhs[row] += weight * q[row] * wall_rhs;
    }
  }

  const lsmps3d::real* inverse_matrix =
      pressure_moment.inverse_matrices + i * kBasisSize * kBasisSize;
  const lsmps3d::real c1 = inverse_rhs_row(inverse_matrix, rhs, 0);
  const lsmps3d::real c2 = inverse_rhs_row(inverse_matrix, rhs, 1);
  const lsmps3d::real c3 = inverse_rhs_row(inverse_matrix, rhs, 2);
  const lsmps3d::real c4 = inverse_rhs_row(inverse_matrix, rhs, 3);
  const lsmps3d::real c5 = inverse_rhs_row(inverse_matrix, rhs, 4);
  const lsmps3d::real c6 = inverse_rhs_row(inverse_matrix, rhs, 5);
  gradient_x[i] = c1 / support_radius;
  gradient_y[i] = c2 / support_radius;
  gradient_z[i] = c3 / support_radius;
  laplacian[i] = static_cast<lsmps3d::real>(2) * (c4 + c5 + c6) /
                 (support_radius * support_radius);
}

void compute_pressure_operator(const lsmps3d::FluidParticleSoA& fluid,
                               const lsmps3d::WallParticleSoA& walls,
                               const lsmps3d::NeighborListView& fluid_neighbors,
                               const lsmps3d::NeighborListView& wall_neighbors,
                               const lsmps3d::MomentMatrixView& pressure_moment,
                               const lsmps3d::SimulationConfig& config,
                               lsmps3d::FluidParticleSoA output) {
  if (!pressure_moment.is_ready ||
      pressure_moment.kind != lsmps3d::MomentMatrixKind::PressureWallNeumannTypeA ||
      pressure_moment.basis_kind != lsmps3d::MomentBasisKind::TypeA ||
      pressure_moment.matrix_size != lsmps3d::kMomentTypeABasis3DSize ||
      pressure_moment.particle_count < fluid.count) {
    throw std::runtime_error("pressure operator diagnostics require a prepared pressure Type-A moment matrix");
  }
  if (fluid.count == 0) {
    return;
  }
  if (output.x == nullptr || output.y == nullptr || output.z == nullptr ||
      output.pressure == nullptr) {
    throw std::runtime_error("pressure operator diagnostics require gradient and laplacian buffers");
  }

  constexpr int kThreadsPerBlock = 128;
  const int blocks = static_cast<int>((fluid.count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  compute_pressure_operator_kernel<<<blocks, kThreadsPerBlock>>>(fluid,
                                                                 walls,
                                                                 fluid_neighbors,
                                                                 wall_neighbors,
                                                                 pressure_moment,
                                                                 config.density,
                                                                 config.gravity,
                                                                 output.x,
                                                                 output.y,
                                                                 output.z,
                                                                 output.pressure);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_dir =
      argc > 1 ? std::filesystem::path(argv[1])
               : std::filesystem::path("output/hydrostatic_surface_diagnostics");
  std::filesystem::create_directories(output_dir);

  constexpr lsmps3d::real kBoxSize = static_cast<lsmps3d::real>(1.0);
  constexpr lsmps3d::real kLiquidHeight = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real kSpacing = static_cast<lsmps3d::real>(0.02);
  constexpr lsmps3d::real kSupportRadius = static_cast<lsmps3d::real>(3.1) * kSpacing;
  constexpr lsmps3d::real kNearSurfaceRadius = static_cast<lsmps3d::real>(2.0) * kSpacing;
  constexpr lsmps3d::real kDensity = static_cast<lsmps3d::real>(1000.0);
  constexpr lsmps3d::real kGravityMagnitude = static_cast<lsmps3d::real>(9.81);

  const HydrostaticCase hydrostatic = make_half_full_box(kSpacing, kBoxSize, kLiquidHeight);
  const lsmps3d::size_type fluid_count = hydrostatic.fluid_x.size();
  const lsmps3d::size_type wall_count = hydrostatic.wall_x.size();

  const int grid_dim =
      static_cast<int>(std::ceil((kBoxSize + static_cast<lsmps3d::real>(2.0) * kSupportRadius) /
                                 kSupportRadius));
  const lsmps3d::WorkspaceSpec spec{
      fluid_count,
      wall_count,
      256,
      256,
      static_cast<lsmps3d::size_type>(grid_dim * grid_dim * grid_dim),
  };
  lsmps3d::SimulationWorkspace workspace(spec);
  auto view = workspace.view();

  copy_to_device(view.fluid.x, hydrostatic.fluid_x);
  copy_to_device(view.fluid.y, hydrostatic.fluid_y);
  copy_to_device(view.fluid.z, hydrostatic.fluid_z);
  LSMPS3D_CUDA_CHECK(cudaMemset(view.fluid.vx, 0, fluid_count * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMemset(view.fluid.vy, 0, fluid_count * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMemset(view.fluid.vz, 0, fluid_count * sizeof(lsmps3d::real)));
  const auto hydrostatic_pressure =
      make_hydrostatic_pressure(
          hydrostatic.fluid_z, kLiquidHeight, kDensity, kGravityMagnitude, kSpacing);
  copy_to_device(view.fluid.pressure, hydrostatic_pressure);
  copy_to_device(view.walls.x, hydrostatic.wall_x);
  copy_to_device(view.walls.y, hydrostatic.wall_y);
  copy_to_device(view.walls.z, hydrostatic.wall_z);
  copy_to_device(view.walls.normal_x, hydrostatic.wall_normal_x);
  copy_to_device(view.walls.normal_y, hydrostatic.wall_normal_y);
  copy_to_device(view.walls.normal_z, hydrostatic.wall_normal_z);
  LSMPS3D_CUDA_CHECK(cudaMemset(view.walls.vx, 0, wall_count * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMemset(view.walls.vy, 0, wall_count * sizeof(lsmps3d::real)));
  LSMPS3D_CUDA_CHECK(cudaMemset(view.walls.vz, 0, wall_count * sizeof(lsmps3d::real)));

  lsmps3d::SimulationConfig config;
  config.particle_spacing = kSpacing;
  config.support_radius = kSupportRadius;
  config.near_surface_radius = kNearSurfaceRadius;
  config.cell_origin = lsmps3d::Vec3{-kSupportRadius, -kSupportRadius, -kSupportRadius};
  config.cell_size = kSupportRadius;
  config.cell_dims = lsmps3d::Int3{grid_dim, grid_dim, grid_dim};
  config.splash_neighbor_threshold = 12;
  config.number_density_ratio_threshold = static_cast<lsmps3d::real>(0.85);
  config.air_open_ratio_threshold = static_cast<lsmps3d::real>(0.33);
  config.air_anisotropy_threshold = static_cast<lsmps3d::real>(0.05);
  config.include_wall_neighbors = true;
  config.wall_normal_independence_threshold = static_cast<lsmps3d::real>(0.25);
  config.density = kDensity;
  config.gravity = lsmps3d::Vec3{static_cast<lsmps3d::real>(0),
                                 static_cast<lsmps3d::real>(0),
                                 -kGravityMagnitude};
  config.amgx_config_path = resolve_amgx_config_path().string();
  lsmps3d::build_neighbor_lists(view.fluid,
                                view.walls,
                                config,
                                view.fluid_cells,
                                view.wall_cells,
                                view.fluid_neighbors,
                                view.wall_neighbors);

  lsmps3d::DeviceFluidParticles real_diagnostics(fluid_count);
  lsmps3d::DeviceFluidParticles normal_diagnostics(fluid_count);
  lsmps3d::DeviceNeighborList count_diagnostics(fluid_count, fluid_count);
  const auto real_diag = real_diagnostics.view();
  const auto normal_diag = normal_diagnostics.view();
  const auto count_diag = count_diagnostics.view();

  const lsmps3d::SurfaceDetectionDiagnosticsView surface_diagnostics{
      count_diag.offsets,
      count_diag.indices,
      real_diag.x,
      real_diag.y,
      real_diag.z,
      real_diag.vx,
      real_diag.vy,
      normal_diag.x,
      normal_diag.y,
      normal_diag.z,
  };
  lsmps3d::classify_surface_particles(view.fluid,
                                      view.walls,
                                      view.fluid_neighbors,
                                      view.wall_neighbors,
                                      config,
                                      surface_diagnostics);

  lsmps3d::DeviceMomentMatrix lsmps(fluid_count, config);
  lsmps.prepare_matrices(view.fluid, view.walls, view.fluid_neighbors, view.wall_neighbors, 1);
  lsmps3d::DeviceFluidParticles pressure_operator_buffers(fluid_count);
  auto pressure_operator_view = pressure_operator_buffers.view();
  compute_pressure_operator(view.fluid,
                            view.walls,
                            view.fluid_neighbors,
                            view.wall_neighbors,
                            lsmps.pressure_type_a(),
                            config,
                            pressure_operator_view);

  lsmps3d::DeviceFluidParticles temporary_velocity(fluid_count);
  lsmps3d::DeviceWallParticles temporary_wall_velocity(wall_count);
  const auto temporary_velocity_view = temporary_velocity.view();
  const auto temporary_wall_velocity_view = temporary_wall_velocity.view();
  lsmps3d::DeviceProvisionExplicitUpdate provision(fluid_count, config);
  provision.compute_temporary_velocity(view.fluid,
                                       view.walls,
                                       view.fluid_neighbors,
                                       view.wall_neighbors,
                                       lsmps,
                                       1,
                                       temporary_velocity_view,
                                       temporary_wall_velocity_view);

  const lsmps3d::size_type ppe_nnz =
      fluid_count + view.fluid_neighbors.neighbor_count;
  lsmps3d::DevicePpeMatrixAssembler ppe_assembler(fluid_count, ppe_nnz, config);
  ppe_assembler.assemble(view.fluid,
                         view.walls,
                         view.fluid_neighbors,
                         view.wall_neighbors,
                         temporary_velocity_view,
                         temporary_wall_velocity_view,
                         lsmps,
                         1);
  auto ppe_workspace = ppe_assembler.workspace();
  const bool has_ppe_solve = lsmps3d::AmgxPpeSolver::is_available();
  if (has_ppe_solve) {
    lsmps3d::AmgxPpeSolver solver(config.amgx_config_path);
    solver.solve(ppe_workspace.matrix, ppe_workspace.rhs, ppe_workspace.pressure);
  } else {
    LSMPS3D_CUDA_CHECK(cudaMemset(ppe_workspace.pressure, 0, fluid_count * sizeof(lsmps3d::real)));
  }

  lsmps3d::HostParticleSnapshot particles{
      hydrostatic.fluid_x,
      hydrostatic.fluid_y,
      hydrostatic.fluid_z,
  };
  const auto fluid_neighbor_count =
      copy_from_device(count_diag.offsets, static_cast<std::size_t>(fluid_count));
  const auto wall_neighbor_count =
      copy_from_device(count_diag.indices, static_cast<std::size_t>(fluid_count));
  const auto number_density = copy_from_device(real_diag.x, static_cast<std::size_t>(fluid_count));
  const auto number_density_ratio =
      copy_from_device(real_diag.y, static_cast<std::size_t>(fluid_count));
  const auto anisotropy = copy_from_device(real_diag.z, static_cast<std::size_t>(fluid_count));
  const auto air_open_ratio = copy_from_device(real_diag.vx, static_cast<std::size_t>(fluid_count));
  const auto air_anisotropy = copy_from_device(real_diag.vy, static_cast<std::size_t>(fluid_count));
  const auto surface_normal_x =
      copy_from_device(normal_diag.x, static_cast<std::size_t>(fluid_count));
  const auto surface_normal_y =
      copy_from_device(normal_diag.y, static_cast<std::size_t>(fluid_count));
  const auto surface_normal_z =
      copy_from_device(normal_diag.z, static_cast<std::size_t>(fluid_count));
  const auto surface_type =
      copy_from_device(view.fluid.surface_type, static_cast<std::size_t>(fluid_count));
  const auto ppe_pressure =
      copy_from_device(ppe_workspace.pressure, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_x =
      copy_from_device(pressure_operator_view.x, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_y =
      copy_from_device(pressure_operator_view.y, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_z =
      copy_from_device(pressure_operator_view.z, static_cast<std::size_t>(fluid_count));
  const auto pressure_laplacian =
      copy_from_device(pressure_operator_view.pressure, static_cast<std::size_t>(fluid_count));
  const auto ppe_rhs = copy_from_device(ppe_workspace.rhs, static_cast<std::size_t>(fluid_count));
  const auto ppe_divergence =
      copy_from_device(ppe_workspace.divergence, static_cast<std::size_t>(fluid_count));
  const auto ppe_wall_neumann_rhs =
      copy_from_device(ppe_workspace.pressure_laplacian, static_cast<std::size_t>(fluid_count));
  const auto ppe_matrix_diagnostics = make_csr_row_diagnostics(ppe_workspace.matrix);
  const auto ppe_residual_diagnostics = make_csr_residual_diagnostics(
      ppe_workspace.matrix, hydrostatic_pressure, ppe_rhs, ppe_wall_neumann_rhs);
  const auto ppe_pressure_error = subtract_values(ppe_pressure, hydrostatic_pressure);
  const auto ppe_pressure_abs_error = absolute_values(ppe_pressure_error);
  const std::vector<lsmps3d::real> expected_gradient_x(
      static_cast<std::size_t>(fluid_count), static_cast<lsmps3d::real>(0));
  const std::vector<lsmps3d::real> expected_gradient_y(
      static_cast<std::size_t>(fluid_count), static_cast<lsmps3d::real>(0));
  const std::vector<lsmps3d::real> expected_gradient_z(
      static_cast<std::size_t>(fluid_count), -kDensity * kGravityMagnitude);
  const std::vector<lsmps3d::real> expected_laplacian(
      static_cast<std::size_t>(fluid_count), static_cast<lsmps3d::real>(0));
  const auto pressure_gradient_x_abs_error =
      absolute_values(subtract_values(pressure_gradient_x, expected_gradient_x));
  const auto pressure_gradient_y_abs_error =
      absolute_values(subtract_values(pressure_gradient_y, expected_gradient_y));
  const auto pressure_gradient_z_abs_error =
      absolute_values(subtract_values(pressure_gradient_z, expected_gradient_z));
  const auto pressure_laplacian_abs_error =
      absolute_values(subtract_values(pressure_laplacian, expected_laplacian));

  lsmps3d::HostVtkPointFields point_fields;
  point_fields.add_scalar("fluid_neighbor_count",
                          std::vector<int>(fluid_neighbor_count.begin(), fluid_neighbor_count.end()));
  point_fields.add_scalar("wall_neighbor_count",
                          std::vector<int>(wall_neighbor_count.begin(), wall_neighbor_count.end()));
  point_fields.add_scalar("hydrostatic_pressure", hydrostatic_pressure);
  point_fields.add_scalar("pressure_gradient_x", pressure_gradient_x);
  point_fields.add_scalar("pressure_gradient_y", pressure_gradient_y);
  point_fields.add_scalar("pressure_gradient_z", pressure_gradient_z);
  point_fields.add_scalar("pressure_laplacian", pressure_laplacian);
  point_fields.add_scalar("pressure_gradient_x_abs_error", pressure_gradient_x_abs_error);
  point_fields.add_scalar("pressure_gradient_y_abs_error", pressure_gradient_y_abs_error);
  point_fields.add_scalar("pressure_gradient_z_abs_error", pressure_gradient_z_abs_error);
  point_fields.add_scalar("pressure_laplacian_abs_error", pressure_laplacian_abs_error);
  point_fields.add_scalar("ppe_pressure", ppe_pressure);
  point_fields.add_scalar("ppe_pressure_error", ppe_pressure_error);
  point_fields.add_scalar("ppe_pressure_abs_error", ppe_pressure_abs_error);
  point_fields.add_scalar("ppe_rhs", ppe_rhs);
  point_fields.add_scalar("ppe_wall_neumann_rhs", ppe_wall_neumann_rhs);
  point_fields.add_scalar("ppe_velocity_divergence", ppe_divergence);
  point_fields.add_scalar("ppe_matrix_diagonal", ppe_matrix_diagnostics.diagonal);
  point_fields.add_scalar("ppe_matrix_row_sum", ppe_matrix_diagnostics.row_sum);
  point_fields.add_scalar("ppe_matrix_row_nnz", ppe_matrix_diagnostics.row_nnz);
  point_fields.add_scalar("ppe_hydrostatic_residual", ppe_residual_diagnostics.residual);
  point_fields.add_scalar("ppe_hydrostatic_residual_plus_wall_rhs",
                          ppe_residual_diagnostics.residual_plus_wall_rhs);
  point_fields.add_scalar("ppe_hydrostatic_residual_minus_wall_rhs",
                          ppe_residual_diagnostics.residual_minus_wall_rhs);
  point_fields.add_scalar("ppe_hydrostatic_residual_plus_two_wall_rhs",
                          ppe_residual_diagnostics.residual_plus_two_wall_rhs);
  point_fields.add_scalar("ppe_hydrostatic_residual_minus_two_wall_rhs",
                          ppe_residual_diagnostics.residual_minus_two_wall_rhs);
  point_fields.add_scalar("surface_type", surface_type);

  config.output_directory = output_dir;
  config.vtk_file_prefix = "hydrostatic_surface";
  config.vtk_write_point_fields = true;
  const lsmps3d::LegacyVtkWriter vtk_writer(config);
  vtk_writer.write(0, particles, point_fields);
  const auto csv_path = output_dir / "hydrostatic_surface_debug.csv";
  write_csv(csv_path,
            particles,
            fluid_neighbor_count,
            wall_neighbor_count,
            number_density,
            number_density_ratio,
            anisotropy,
            air_open_ratio,
            air_anisotropy,
            surface_normal_x,
            surface_normal_y,
            surface_normal_z,
            surface_type);

  std::array<std::size_t, 4> type_counts{};
  std::size_t top_layer_fluid_count = 0;
  std::size_t top_layer_surface_count = 0;
  std::size_t below_top_surface_count = 0;
  for (const int type : surface_type) {
    if (type >= 0 && type < static_cast<int>(type_counts.size())) {
      ++type_counts[static_cast<std::size_t>(type)];
    }
  }
  for (std::size_t i = 0; i < surface_type.size(); ++i) {
    const bool is_top_layer =
        particles.z[i] >= kLiquidHeight - static_cast<lsmps3d::real>(0.75) * kSpacing;
    if (is_top_layer) {
      ++top_layer_fluid_count;
    }
    if (surface_type[i] == static_cast<int>(lsmps3d::SurfaceType::Surface)) {
      if (is_top_layer) {
        ++top_layer_surface_count;
      } else {
        ++below_top_surface_count;
      }
    }
  }

  const lsmps3d::real reference_number_density =
      lsmps3d::compute_uniform_reference_number_density(kSpacing, kSupportRadius);
  std::cout << "Hydrostatic half-full box diagnostics\n"
            << "  fluid particles: " << fluid_count << '\n'
            << "  wall particles: " << wall_count << '\n'
            << "  spacing: " << kSpacing << " m\n"
            << "  support radius: " << kSupportRadius << " m\n"
            << "  reference number density n0: " << reference_number_density << '\n'
            << "  Inner: " << type_counts[static_cast<int>(lsmps3d::SurfaceType::Inner)] << '\n'
            << "  NearSurface: " << type_counts[static_cast<int>(lsmps3d::SurfaceType::NearSurface)]
            << '\n'
            << "  Surface: " << type_counts[static_cast<int>(lsmps3d::SurfaceType::Surface)] << '\n'
            << "  Splash: " << type_counts[static_cast<int>(lsmps3d::SurfaceType::Splash)] << '\n'
            << "  Top-layer fluid particles: " << top_layer_fluid_count << '\n'
            << "  Top-layer Surface particles: " << top_layer_surface_count << '\n'
            << "  Below-top Surface particles: " << below_top_surface_count << '\n'
            << "  Hydrostatic pressure max: " << max_value(hydrostatic_pressure) << " Pa\n"
            << "  Pressure gradient z max abs error: "
            << max_value(pressure_gradient_z_abs_error) << " Pa/m\n"
            << "  Pressure laplacian max abs error: "
            << max_value(pressure_laplacian_abs_error) << " Pa/m^2\n"
            << "  PPE A*p_true-b max abs: "
            << max_value(absolute_values(ppe_residual_diagnostics.residual)) << '\n'
            << "  PPE residual+wall_rhs max abs: "
            << max_value(absolute_values(ppe_residual_diagnostics.residual_plus_wall_rhs)) << '\n'
            << "  PPE residual-wall_rhs max abs: "
            << max_value(absolute_values(ppe_residual_diagnostics.residual_minus_wall_rhs)) << '\n'
            << "  PPE residual+2*wall_rhs max abs: "
            << max_value(absolute_values(ppe_residual_diagnostics.residual_plus_two_wall_rhs)) << '\n'
            << "  PPE residual-2*wall_rhs max abs: "
            << max_value(absolute_values(ppe_residual_diagnostics.residual_minus_two_wall_rhs)) << '\n'
            << "  PPE pressure solve: " << (has_ppe_solve ? "AMGX" : "skipped (AMGX unavailable)")
            << '\n'
            << "  PPE pressure max abs error: " << max_value(ppe_pressure_abs_error) << " Pa\n"
            << "  VTK: " << vtk_writer.make_path(0) << '\n'
            << "  CSV: " << csv_path << std::endl;

  return 0;
}
