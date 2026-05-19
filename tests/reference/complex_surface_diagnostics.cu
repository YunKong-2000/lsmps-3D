#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/io/vtk_writer.hpp"
#include "lsmps3d/neighbor/neighbor_search.cuh"
#include "lsmps3d/surface/surface_detection.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

constexpr lsmps3d::real kPi = static_cast<lsmps3d::real>(3.14159265358979323846);

struct CaseData {
  std::string name;
  std::vector<lsmps3d::real> fluid_x;
  std::vector<lsmps3d::real> fluid_y;
  std::vector<lsmps3d::real> fluid_z;
  std::vector<lsmps3d::real> wall_x;
  std::vector<lsmps3d::real> wall_y;
  std::vector<lsmps3d::real> wall_z;
  std::vector<lsmps3d::real> wall_normal_x;
  std::vector<lsmps3d::real> wall_normal_y;
  std::vector<lsmps3d::real> wall_normal_z;
  std::vector<int> expected_surface;
  std::vector<lsmps3d::real> expected_normal_x;
  std::vector<lsmps3d::real> expected_normal_y;
  std::vector<lsmps3d::real> expected_normal_z;
};

struct Diagnostics {
  std::vector<lsmps3d::index_t> fluid_neighbor_count;
  std::vector<lsmps3d::index_t> wall_neighbor_count;
  std::vector<lsmps3d::real> number_density;
  std::vector<lsmps3d::real> number_density_ratio;
  std::vector<lsmps3d::real> anisotropy;
  std::vector<lsmps3d::real> air_open_ratio;
  std::vector<lsmps3d::real> air_anisotropy;
  std::vector<lsmps3d::real> surface_normal_x;
  std::vector<lsmps3d::real> surface_normal_y;
  std::vector<lsmps3d::real> surface_normal_z;
  std::vector<int> surface_type;
};

struct Stats {
  std::array<std::size_t, 4> type_counts{};
  std::size_t expected_surface_count{};
  std::size_t expected_surface_detected{};
  std::size_t false_surface_count{};
  lsmps3d::real mean_normal_angle_degrees{};
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

void add_fluid(CaseData& data,
               lsmps3d::real x,
               lsmps3d::real y,
               lsmps3d::real z,
               bool expected_surface,
               lsmps3d::real nx = static_cast<lsmps3d::real>(0),
               lsmps3d::real ny = static_cast<lsmps3d::real>(0),
               lsmps3d::real nz = static_cast<lsmps3d::real>(0)) {
  data.fluid_x.push_back(x);
  data.fluid_y.push_back(y);
  data.fluid_z.push_back(z);
  data.expected_surface.push_back(expected_surface ? 1 : 0);
  data.expected_normal_x.push_back(nx);
  data.expected_normal_y.push_back(ny);
  data.expected_normal_z.push_back(nz);
}

void add_wall(CaseData& data,
              lsmps3d::real x,
              lsmps3d::real y,
              lsmps3d::real z,
              lsmps3d::real nx,
              lsmps3d::real ny,
              lsmps3d::real nz) {
  data.wall_x.push_back(x);
  data.wall_y.push_back(y);
  data.wall_z.push_back(z);
  data.wall_normal_x.push_back(nx);
  data.wall_normal_y.push_back(ny);
  data.wall_normal_z.push_back(nz);
}

void add_box_walls(CaseData& data, int n, lsmps3d::real spacing, bool include_top = false) {
  for (int iz = 0; iz <= n; ++iz) {
    for (int iy = 0; iy <= n; ++iy) {
      add_wall(data, static_cast<lsmps3d::real>(0), iy * spacing, iz * spacing,
               static_cast<lsmps3d::real>(1), static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(0));
      add_wall(data, n * spacing, iy * spacing, iz * spacing, static_cast<lsmps3d::real>(-1),
               static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(0));
    }
    for (int ix = 1; ix < n; ++ix) {
      add_wall(data, ix * spacing, static_cast<lsmps3d::real>(0), iz * spacing,
               static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(1),
               static_cast<lsmps3d::real>(0));
      add_wall(data, ix * spacing, n * spacing, iz * spacing, static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(-1), static_cast<lsmps3d::real>(0));
    }
  }

  for (int iy = 1; iy < n; ++iy) {
    for (int ix = 1; ix < n; ++ix) {
      add_wall(data, ix * spacing, iy * spacing, static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(1));
      if (include_top) {
        add_wall(data, ix * spacing, iy * spacing, n * spacing, static_cast<lsmps3d::real>(0),
                 static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(-1));
      }
    }
  }
}

void normalize(lsmps3d::real& x, lsmps3d::real& y, lsmps3d::real& z) {
  const lsmps3d::real length = std::sqrt(x * x + y * y + z * z);
  if (length <= static_cast<lsmps3d::real>(0)) {
    return;
  }
  x /= length;
  y /= length;
  z /= length;
}

CaseData make_inclined_plane(lsmps3d::real spacing) {
  CaseData data;
  data.name = "inclined_plane";
  const int n = static_cast<int>(std::lround(static_cast<lsmps3d::real>(1) / spacing));
  add_box_walls(data, n, spacing);
  constexpr lsmps3d::real slope = static_cast<lsmps3d::real>(0.25);
  lsmps3d::real nx = -slope;
  lsmps3d::real ny = static_cast<lsmps3d::real>(0);
  lsmps3d::real nz = static_cast<lsmps3d::real>(1);
  normalize(nx, ny, nz);

  for (int iz = 0; iz < n; ++iz) {
    const lsmps3d::real z = (static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int iy = 0; iy < n; ++iy) {
      const lsmps3d::real y = (static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                              spacing;
      for (int ix = 0; ix < n; ++ix) {
        const lsmps3d::real x =
            (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) * spacing;
        const lsmps3d::real height = static_cast<lsmps3d::real>(0.35) + slope * x;
        if (z <= height) {
          add_fluid(data, x, y, z, height - z <= spacing, nx, ny, nz);
        }
      }
    }
  }
  return data;
}

CaseData make_sine_wave(lsmps3d::real spacing) {
  CaseData data;
  data.name = "sine_wave";
  const int n = static_cast<int>(std::lround(static_cast<lsmps3d::real>(1) / spacing));
  add_box_walls(data, n, spacing);

  for (int iz = 0; iz < n; ++iz) {
    const lsmps3d::real z = (static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int iy = 0; iy < n; ++iy) {
      const lsmps3d::real y = (static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                              spacing;
      for (int ix = 0; ix < n; ++ix) {
        const lsmps3d::real x =
            (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) * spacing;
        const lsmps3d::real sx = std::sin(static_cast<lsmps3d::real>(2) * kPi * x);
        const lsmps3d::real sy = std::sin(static_cast<lsmps3d::real>(2) * kPi * y);
        const lsmps3d::real cx = std::cos(static_cast<lsmps3d::real>(2) * kPi * x);
        const lsmps3d::real cy = std::cos(static_cast<lsmps3d::real>(2) * kPi * y);
        const lsmps3d::real amplitude = static_cast<lsmps3d::real>(0.08);
        const lsmps3d::real height = static_cast<lsmps3d::real>(0.5) + amplitude * sx * sy;
        if (z <= height) {
          lsmps3d::real nx = -amplitude * static_cast<lsmps3d::real>(2) * kPi * cx * sy;
          lsmps3d::real ny = -amplitude * static_cast<lsmps3d::real>(2) * kPi * sx * cy;
          lsmps3d::real nz = static_cast<lsmps3d::real>(1);
          normalize(nx, ny, nz);
          add_fluid(data, x, y, z, height - z <= spacing, nx, ny, nz);
        }
      }
    }
  }
  return data;
}

CaseData make_droplet(lsmps3d::real spacing) {
  CaseData data;
  data.name = "droplet";
  const int n = static_cast<int>(std::lround(static_cast<lsmps3d::real>(1) / spacing));
  add_box_walls(data, n, spacing);
  constexpr lsmps3d::real cx = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real cy = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real cz = static_cast<lsmps3d::real>(0.25);
  constexpr lsmps3d::real radius = static_cast<lsmps3d::real>(0.28);

  for (int iz = 0; iz < n; ++iz) {
    const lsmps3d::real z = (static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int iy = 0; iy < n; ++iy) {
      const lsmps3d::real y = (static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                              spacing;
      for (int ix = 0; ix < n; ++ix) {
        const lsmps3d::real x =
            (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) * spacing;
        const lsmps3d::real dx = x - cx;
        const lsmps3d::real dy = y - cy;
        const lsmps3d::real dz = z - cz;
        const lsmps3d::real r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r <= radius && z >= static_cast<lsmps3d::real>(0)) {
          lsmps3d::real nx = dx;
          lsmps3d::real ny = dy;
          lsmps3d::real nz = dz;
          normalize(nx, ny, nz);
          add_fluid(data, x, y, z, radius - r <= spacing, nx, ny, nz);
        }
      }
    }
  }
  return data;
}

CaseData make_cylinder(lsmps3d::real spacing) {
  CaseData data;
  data.name = "cylinder";
  const int n = static_cast<int>(std::lround(static_cast<lsmps3d::real>(1) / spacing));
  add_box_walls(data, n, spacing);
  constexpr lsmps3d::real cx = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real cy = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real radius = static_cast<lsmps3d::real>(0.12);
  constexpr lsmps3d::real liquid_height = static_cast<lsmps3d::real>(0.5);

  for (int iz = 0; iz <= n; ++iz) {
    const lsmps3d::real z = static_cast<lsmps3d::real>(iz) * spacing;
    for (int angle = 0; angle < 48; ++angle) {
      const lsmps3d::real theta =
          static_cast<lsmps3d::real>(angle) * static_cast<lsmps3d::real>(2) * kPi /
          static_cast<lsmps3d::real>(48);
      const lsmps3d::real nx = std::cos(theta);
      const lsmps3d::real ny = std::sin(theta);
      add_wall(data, cx + radius * nx, cy + radius * ny, z, nx, ny, static_cast<lsmps3d::real>(0));
    }
  }

  for (int iz = 0; iz < n; ++iz) {
    const lsmps3d::real z = (static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    if (z > liquid_height) {
      continue;
    }
    for (int iy = 0; iy < n; ++iy) {
      const lsmps3d::real y = (static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                              spacing;
      for (int ix = 0; ix < n; ++ix) {
        const lsmps3d::real x =
            (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) * spacing;
        const lsmps3d::real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (r >= radius + static_cast<lsmps3d::real>(0.5) * spacing) {
          add_fluid(data,
                    x,
                    y,
                    z,
                    liquid_height - z <= spacing,
                    static_cast<lsmps3d::real>(0),
                    static_cast<lsmps3d::real>(0),
                    static_cast<lsmps3d::real>(1));
        }
      }
    }
  }
  return data;
}

CaseData make_stepped_box(lsmps3d::real spacing) {
  CaseData data;
  data.name = "stepped_box";
  const int n = static_cast<int>(std::lround(static_cast<lsmps3d::real>(1) / spacing));
  add_box_walls(data, n, spacing);
  constexpr lsmps3d::real liquid_height = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real step_x = static_cast<lsmps3d::real>(0.5);
  constexpr lsmps3d::real step_height = static_cast<lsmps3d::real>(0.18);

  for (int iy = 0; iy <= n; ++iy) {
    const lsmps3d::real y = static_cast<lsmps3d::real>(iy) * spacing;
    for (int iz = 0; iz <= static_cast<int>(std::lround(step_height / spacing)); ++iz) {
      const lsmps3d::real z = static_cast<lsmps3d::real>(iz) * spacing;
      add_wall(data, step_x, y, z, static_cast<lsmps3d::real>(1), static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(0));
    }
  }
  for (int ix = 0; ix <= static_cast<int>(std::lround(step_x / spacing)); ++ix) {
    const lsmps3d::real x = static_cast<lsmps3d::real>(ix) * spacing;
    for (int iy = 0; iy <= n; ++iy) {
      const lsmps3d::real y = static_cast<lsmps3d::real>(iy) * spacing;
      add_wall(data, x, y, step_height, static_cast<lsmps3d::real>(0),
               static_cast<lsmps3d::real>(0), static_cast<lsmps3d::real>(1));
    }
  }

  for (int iz = 0; iz < n; ++iz) {
    const lsmps3d::real z = (static_cast<lsmps3d::real>(iz) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int iy = 0; iy < n; ++iy) {
      const lsmps3d::real y = (static_cast<lsmps3d::real>(iy) + static_cast<lsmps3d::real>(0.5)) *
                              spacing;
      for (int ix = 0; ix < n; ++ix) {
        const lsmps3d::real x =
            (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) * spacing;
        const lsmps3d::real bottom = x < step_x ? step_height : static_cast<lsmps3d::real>(0);
        if (z >= bottom && z <= liquid_height) {
          add_fluid(data,
                    x,
                    y,
                    z,
                    liquid_height - z <= spacing,
                    static_cast<lsmps3d::real>(0),
                    static_cast<lsmps3d::real>(0),
                    static_cast<lsmps3d::real>(1));
        }
      }
    }
  }
  return data;
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

Diagnostics run_surface_detection(const CaseData& data, lsmps3d::real spacing) {
  constexpr lsmps3d::real kBoxSize = static_cast<lsmps3d::real>(1.0);
  const lsmps3d::real support_radius = static_cast<lsmps3d::real>(3.1) * spacing;
  const int grid_dim = static_cast<int>(
      std::ceil((kBoxSize + static_cast<lsmps3d::real>(2.0) * support_radius) / support_radius));
  const lsmps3d::WorkspaceSpec spec{
      data.fluid_x.size(),
      data.wall_x.size(),
      256,
      256,
      static_cast<lsmps3d::size_type>(grid_dim * grid_dim * grid_dim),
  };
  lsmps3d::SimulationWorkspace workspace(spec);
  auto view = workspace.view();

  copy_to_device(view.fluid.x, data.fluid_x);
  copy_to_device(view.fluid.y, data.fluid_y);
  copy_to_device(view.fluid.z, data.fluid_z);
  copy_to_device(view.walls.x, data.wall_x);
  copy_to_device(view.walls.y, data.wall_y);
  copy_to_device(view.walls.z, data.wall_z);
  copy_to_device(view.walls.normal_x, data.wall_normal_x);
  copy_to_device(view.walls.normal_y, data.wall_normal_y);
  copy_to_device(view.walls.normal_z, data.wall_normal_z);

  lsmps3d::SimulationConfig config;
  config.particle_spacing = spacing;
  config.support_radius = support_radius;
  config.near_surface_radius = static_cast<lsmps3d::real>(2.0) * spacing;
  config.cell_origin = lsmps3d::Vec3{-support_radius, -support_radius, -support_radius};
  config.cell_size = support_radius;
  config.cell_dims = lsmps3d::Int3{grid_dim, grid_dim, grid_dim};
  config.splash_neighbor_threshold = 12;
  config.number_density_ratio_threshold = static_cast<lsmps3d::real>(0.85);
  config.air_open_ratio_threshold = static_cast<lsmps3d::real>(0.33);
  config.air_anisotropy_threshold = static_cast<lsmps3d::real>(0.033);
  config.include_wall_neighbors = true;
  config.wall_normal_independence_threshold = static_cast<lsmps3d::real>(0.25);
  lsmps3d::build_neighbor_lists(view.fluid,
                                view.walls,
                                config,
                                view.fluid_cells,
                                view.wall_cells,
                                view.fluid_neighbors,
                                view.wall_neighbors);

  lsmps3d::DeviceFluidParticles real_diagnostics(data.fluid_x.size());
  lsmps3d::DeviceFluidParticles normal_diagnostics(data.fluid_x.size());
  lsmps3d::DeviceNeighborList count_diagnostics(data.fluid_x.size(), data.fluid_x.size());
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

  Diagnostics diagnostics;
  diagnostics.fluid_neighbor_count =
      copy_from_device(count_diag.offsets, data.fluid_x.size());
  diagnostics.wall_neighbor_count = copy_from_device(count_diag.indices, data.fluid_x.size());
  diagnostics.number_density = copy_from_device(real_diag.x, data.fluid_x.size());
  diagnostics.number_density_ratio = copy_from_device(real_diag.y, data.fluid_x.size());
  diagnostics.anisotropy = copy_from_device(real_diag.z, data.fluid_x.size());
  diagnostics.air_open_ratio = copy_from_device(real_diag.vx, data.fluid_x.size());
  diagnostics.air_anisotropy = copy_from_device(real_diag.vy, data.fluid_x.size());
  diagnostics.surface_normal_x = copy_from_device(normal_diag.x, data.fluid_x.size());
  diagnostics.surface_normal_y = copy_from_device(normal_diag.y, data.fluid_x.size());
  diagnostics.surface_normal_z = copy_from_device(normal_diag.z, data.fluid_x.size());
  diagnostics.surface_type = copy_from_device(view.fluid.surface_type, data.fluid_x.size());
  return diagnostics;
}

void write_csv(const std::filesystem::path& path, const CaseData& data, const Diagnostics& diagnostics) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open CSV output: " + path.string());
  }

  out << std::setprecision(9);
  out << "particle_id,x,y,z,fluid_neighbor_count,wall_neighbor_count,number_density,"
         "number_density_ratio,anisotropy,air_open_ratio,air_anisotropy,surface_normal_x,"
         "surface_normal_y,surface_normal_z,surface_type,surface_type_name,expected_surface,"
         "expected_normal_x,expected_normal_y,expected_normal_z\n";
  for (std::size_t i = 0; i < data.fluid_x.size(); ++i) {
    out << i << ',' << data.fluid_x[i] << ',' << data.fluid_y[i] << ',' << data.fluid_z[i] << ','
        << diagnostics.fluid_neighbor_count[i] << ',' << diagnostics.wall_neighbor_count[i] << ','
        << diagnostics.number_density[i] << ',' << diagnostics.number_density_ratio[i] << ','
        << diagnostics.anisotropy[i] << ',' << diagnostics.air_open_ratio[i] << ','
        << diagnostics.air_anisotropy[i] << ',' << diagnostics.surface_normal_x[i] << ','
        << diagnostics.surface_normal_y[i] << ',' << diagnostics.surface_normal_z[i] << ','
        << diagnostics.surface_type[i] << ',' << surface_type_name(diagnostics.surface_type[i])
        << ',' << data.expected_surface[i] << ',' << data.expected_normal_x[i] << ','
        << data.expected_normal_y[i] << ',' << data.expected_normal_z[i] << '\n';
  }
}

void write_outputs(const std::filesystem::path& output_dir,
                   const CaseData& data,
                   const Diagnostics& diagnostics) {
  std::filesystem::create_directories(output_dir);
  lsmps3d::HostVtkPointFields fields;
  fields.add_scalar("fluid_neighbor_count",
                    std::vector<int>(diagnostics.fluid_neighbor_count.begin(),
                                     diagnostics.fluid_neighbor_count.end()));
  fields.add_scalar("wall_neighbor_count",
                    std::vector<int>(diagnostics.wall_neighbor_count.begin(),
                                     diagnostics.wall_neighbor_count.end()));
  fields.add_scalar("number_density", diagnostics.number_density);
  fields.add_scalar("number_density_ratio", diagnostics.number_density_ratio);
  fields.add_scalar("anisotropy", diagnostics.anisotropy);
  fields.add_scalar("air_open_ratio", diagnostics.air_open_ratio);
  fields.add_scalar("air_anisotropy", diagnostics.air_anisotropy);
  fields.add_scalar("surface_type", diagnostics.surface_type);
  fields.add_scalar("expected_surface", data.expected_surface);
  fields.add_vector("surface_normal",
                    diagnostics.surface_normal_x,
                    diagnostics.surface_normal_y,
                    diagnostics.surface_normal_z);
  fields.add_vector("expected_normal",
                    data.expected_normal_x,
                    data.expected_normal_y,
                    data.expected_normal_z);

  lsmps3d::SimulationConfig config;
  config.output_directory = output_dir;
  config.vtk_file_prefix = data.name;
  config.vtk_write_point_fields = true;
  const lsmps3d::LegacyVtkWriter writer(config);
  writer.write(0, lsmps3d::HostParticleSnapshot{data.fluid_x, data.fluid_y, data.fluid_z}, fields);

  lsmps3d::HostVtkPointFields wall_fields;
  wall_fields.add_vector("wall_normal", data.wall_normal_x, data.wall_normal_y, data.wall_normal_z);
  config.vtk_file_prefix = data.name + "_walls";
  const lsmps3d::LegacyVtkWriter wall_writer(config);
  wall_writer.write(0, lsmps3d::HostParticleSnapshot{data.wall_x, data.wall_y, data.wall_z}, wall_fields);

  write_csv(output_dir / (data.name + "_debug.csv"), data, diagnostics);
}

Stats compute_stats(const CaseData& data, const Diagnostics& diagnostics) {
  Stats stats;
  lsmps3d::real angle_sum = static_cast<lsmps3d::real>(0);
  std::size_t angle_count = 0;
  for (std::size_t i = 0; i < diagnostics.surface_type.size(); ++i) {
    const int type = diagnostics.surface_type[i];
    if (type >= 0 && type < static_cast<int>(stats.type_counts.size())) {
      ++stats.type_counts[static_cast<std::size_t>(type)];
    }
    const bool expected = data.expected_surface[i] != 0;
    const bool detected = type == static_cast<int>(lsmps3d::SurfaceType::Surface);
    if (expected) {
      ++stats.expected_surface_count;
      if (detected) {
        ++stats.expected_surface_detected;
      }
    } else if (detected) {
      ++stats.false_surface_count;
    }

    if (expected && detected) {
      const lsmps3d::real dot =
          diagnostics.surface_normal_x[i] * data.expected_normal_x[i] +
          diagnostics.surface_normal_y[i] * data.expected_normal_y[i] +
          diagnostics.surface_normal_z[i] * data.expected_normal_z[i];
      const lsmps3d::real clamped =
          std::max(static_cast<lsmps3d::real>(-1), std::min(static_cast<lsmps3d::real>(1), dot));
      angle_sum += std::acos(clamped) * static_cast<lsmps3d::real>(180) / kPi;
      ++angle_count;
    }
  }
  if (angle_count > 0) {
    stats.mean_normal_angle_degrees = angle_sum / static_cast<lsmps3d::real>(angle_count);
  }
  return stats;
}

void print_stats(const CaseData& data, const Stats& stats, const std::filesystem::path& output_dir) {
  const auto detected_ratio =
      stats.expected_surface_count > 0
          ? static_cast<double>(stats.expected_surface_detected) /
                static_cast<double>(stats.expected_surface_count)
          : 0.0;
  std::cout << "Complex surface diagnostics: " << data.name << '\n'
            << "  fluid particles: " << data.fluid_x.size() << '\n'
            << "  wall particles: " << data.wall_x.size() << '\n'
            << "  Inner: " << stats.type_counts[static_cast<int>(lsmps3d::SurfaceType::Inner)]
            << '\n'
            << "  NearSurface: "
            << stats.type_counts[static_cast<int>(lsmps3d::SurfaceType::NearSurface)] << '\n'
            << "  Surface: " << stats.type_counts[static_cast<int>(lsmps3d::SurfaceType::Surface)]
            << '\n'
            << "  Splash: " << stats.type_counts[static_cast<int>(lsmps3d::SurfaceType::Splash)]
            << '\n'
            << "  Expected-surface particles: " << stats.expected_surface_count << '\n'
            << "  Expected-surface detected: " << stats.expected_surface_detected << " ("
            << detected_ratio << ")\n"
            << "  False Surface particles: " << stats.false_surface_count << '\n'
            << "  Mean normal angle error: " << stats.mean_normal_angle_degrees << " deg\n"
            << "  VTK: " << (output_dir / (data.name + "_000000.vtk")) << '\n'
            << "  Wall VTK: " << (output_dir / (data.name + "_walls_000000.vtk")) << '\n'
            << "  CSV: " << (output_dir / (data.name + "_debug.csv")) << std::endl;
}

CaseData make_case(const std::string& name, lsmps3d::real spacing) {
  if (name == "inclined_plane") {
    return make_inclined_plane(spacing);
  }
  if (name == "sine_wave") {
    return make_sine_wave(spacing);
  }
  if (name == "droplet") {
    return make_droplet(spacing);
  }
  if (name == "cylinder") {
    return make_cylinder(spacing);
  }
  if (name == "stepped_box") {
    return make_stepped_box(spacing);
  }
  throw std::invalid_argument("Unknown case: " + name);
}

}  // namespace

int main(int argc, char** argv) {
  constexpr lsmps3d::real kSpacing = static_cast<lsmps3d::real>(0.02);
  const std::string case_name = argc > 1 ? argv[1] : "inclined_plane";
  const std::filesystem::path base_output_dir =
      argc > 2 ? std::filesystem::path(argv[2])
               : std::filesystem::path("output/complex_surface_diagnostics");

  if (case_name == "all") {
    const std::array<std::string, 5> cases{
        "inclined_plane", "sine_wave", "droplet", "cylinder", "stepped_box"};
    for (const auto& name : cases) {
      const CaseData data = make_case(name, kSpacing);
      const Diagnostics diagnostics = run_surface_detection(data, kSpacing);
      const auto output_dir = base_output_dir / name;
      write_outputs(output_dir, data, diagnostics);
      print_stats(data, compute_stats(data, diagnostics), output_dir);
    }
    return 0;
  }

  const CaseData data = make_case(case_name, kSpacing);
  const Diagnostics diagnostics = run_surface_detection(data, kSpacing);
  const auto output_dir = base_output_dir / data.name;
  write_outputs(output_dir, data, diagnostics);
  print_stats(data, compute_stats(data, diagnostics), output_dir);
  return 0;
}
