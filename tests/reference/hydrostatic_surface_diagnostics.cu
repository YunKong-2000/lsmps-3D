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
#include "lsmps3d/lsmps/moment_matrix.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"
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
                                                     lsmps3d::real gravity_magnitude) {
  std::vector<lsmps3d::real> pressure(z.size());
  for (std::size_t i = 0; i < z.size(); ++i) {
    pressure[i] = density * gravity_magnitude *
                  std::max(liquid_height - z[i], static_cast<lsmps3d::real>(0));
  }
  return pressure;
}

std::vector<lsmps3d::real> vector_magnitude(const std::vector<lsmps3d::real>& x,
                                            const std::vector<lsmps3d::real>& y,
                                            const std::vector<lsmps3d::real>& z) {
  std::vector<lsmps3d::real> magnitude(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    magnitude[i] = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
  }
  return magnitude;
}

std::vector<lsmps3d::real> absolute_values(const std::vector<lsmps3d::real>& values) {
  std::vector<lsmps3d::real> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = std::abs(values[i]);
  }
  return result;
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
  const auto hydrostatic_pressure =
      make_hydrostatic_pressure(hydrostatic.fluid_z, kLiquidHeight, kDensity, kGravityMagnitude);
  copy_to_device(view.fluid.pressure, hydrostatic_pressure);
  copy_to_device(view.walls.x, hydrostatic.wall_x);
  copy_to_device(view.walls.y, hydrostatic.wall_y);
  copy_to_device(view.walls.z, hydrostatic.wall_z);
  copy_to_device(view.walls.normal_x, hydrostatic.wall_normal_x);
  copy_to_device(view.walls.normal_y, hydrostatic.wall_normal_y);
  copy_to_device(view.walls.normal_z, hydrostatic.wall_normal_z);

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

  lsmps3d::DeviceFluidParticles pressure_operator_buffers(fluid_count);
  auto pressure_operator_view = pressure_operator_buffers.view();
  lsmps3d::DeviceLsmpsOperators lsmps(fluid_count, config);
  lsmps.prepare_matrices(view.fluid, view.walls, view.fluid_neighbors, view.wall_neighbors, 1);
  lsmps.compute_pressure_gradient(
      view.fluid,
      view.walls,
      view.fluid_neighbors,
      view.wall_neighbors,
      view.fluid.pressure,
      pressure_operator_view.x,
      pressure_operator_view.y,
      pressure_operator_view.z);
  lsmps.compute_pressure_laplacian(view.fluid,
                                   view.walls,
                                   view.fluid_neighbors,
                                   view.wall_neighbors,
                                   view.fluid.pressure,
                                   pressure_operator_view.pressure);

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
  const auto pressure_gradient_x =
      copy_from_device(pressure_operator_view.x, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_y =
      copy_from_device(pressure_operator_view.y, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_z =
      copy_from_device(pressure_operator_view.z, static_cast<std::size_t>(fluid_count));
  const auto pressure_laplacian =
      copy_from_device(pressure_operator_view.pressure, static_cast<std::size_t>(fluid_count));
  const auto pressure_gradient_magnitude =
      vector_magnitude(pressure_gradient_x, pressure_gradient_y, pressure_gradient_z);
  const auto pressure_laplacian_magnitude = absolute_values(pressure_laplacian);

  lsmps3d::HostVtkPointFields point_fields;
  point_fields.add_scalar("fluid_neighbor_count",
                          std::vector<int>(fluid_neighbor_count.begin(), fluid_neighbor_count.end()));
  point_fields.add_scalar("wall_neighbor_count",
                          std::vector<int>(wall_neighbor_count.begin(), wall_neighbor_count.end()));
  point_fields.add_scalar("number_density", number_density);
  point_fields.add_scalar("number_density_ratio", number_density_ratio);
  point_fields.add_scalar("anisotropy", anisotropy);
  point_fields.add_scalar("air_open_ratio", air_open_ratio);
  point_fields.add_scalar("air_anisotropy", air_anisotropy);
  point_fields.add_scalar("pressure", hydrostatic_pressure);
  point_fields.add_scalar("pressure_gradient_magnitude", pressure_gradient_magnitude);
  point_fields.add_scalar("pressure_laplacian", pressure_laplacian);
  point_fields.add_scalar("pressure_laplacian_magnitude", pressure_laplacian_magnitude);
  point_fields.add_scalar("surface_type", surface_type);
  point_fields.add_vector("surface_normal", surface_normal_x, surface_normal_y, surface_normal_z);
  point_fields.add_vector(
      "pressure_gradient", pressure_gradient_x, pressure_gradient_y, pressure_gradient_z);

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
            << "  VTK: " << vtk_writer.make_path(0) << '\n'
            << "  CSV: " << csv_path << std::endl;

  return 0;
}
