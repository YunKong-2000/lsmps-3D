#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lsmps3d/io/vtk_writer.hpp"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

bool file_contains(const std::filesystem::path& path, const std::string& token) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to open generated VTK file: " << path << std::endl;
    return false;
  }

  const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (contents.find(token) == std::string::npos) {
    std::cerr << "Generated VTK file is missing token: " << token << std::endl;
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const auto output_dir = std::filesystem::temp_directory_path() / "lsmps3d_vtk_writer_smoke";
  std::filesystem::remove_all(output_dir);

  lsmps3d::SimulationConfig config;
  config.output_directory = output_dir;
  config.vtk_file_prefix = "particles";
  config.vtk_write_point_fields = true;
  const lsmps3d::LegacyVtkWriter writer(config);
  const lsmps3d::HostParticleSnapshot particles{
      {0.0F, 1.0F, 0.0F},
      {0.0F, 0.0F, 1.0F},
      {0.0F, 0.0F, 0.5F},
  };

  lsmps3d::HostVtkPointFields point_fields;
  point_fields.add_vector("velocity", {1.0F, 0.0F, -1.0F}, {0.0F, 2.0F, 0.0F}, {0.0F, 0.0F, 3.0F});
  point_fields.add_vector("pressure_gradient", {0.5F, 0.0F, -0.5F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.5F});
  point_fields.add_real_scalar("pressure", {0.0F, 10.0F, 20.0F});
  point_fields.add_real_scalar("number_density", {1.0F, 0.55F, 0.0F});
  point_fields.add_real_scalar("anisotropy", {0.1F, 0.8F, 0.0F});
  point_fields.add_int_scalar("surface_type",
                              {static_cast<int>(lsmps3d::SurfaceType::Inner),
                               static_cast<int>(lsmps3d::SurfaceType::Surface),
                               static_cast<int>(lsmps3d::SurfaceType::Splash)});
  point_fields.add_int_scalar("fluid_neighbor_count", {6, 4, 0});
  point_fields.add_int_scalar("wall_neighbor_count", {2, 1, 0});

  writer.write(7, particles, point_fields);
  const auto path = writer.make_path(7);

  if (!std::filesystem::exists(path)) {
    std::cerr << "VTK writer did not create expected file: " << path << std::endl;
    return 1;
  }

  if (!file_contains(path, "DATASET POLYDATA") || !file_contains(path, "POINTS 3 float") ||
      !file_contains(path, "VECTORS velocity float") ||
      !file_contains(path, "VECTORS pressure_gradient float") ||
      !file_contains(path, "SCALARS pressure float 1") ||
      !file_contains(path, "SCALARS surface_type int 1") ||
      !file_contains(path, "SCALARS fluid_neighbor_count int 1") ||
      !file_contains(path, "SCALARS wall_neighbor_count int 1") ||
      !file_contains(path, "SCALARS number_density float 1") ||
      !file_contains(path, "SCALARS anisotropy float 1")) {
    return 1;
  }

  lsmps3d::HostVtkPointFields bad_fields = point_fields;
  bad_fields.vectors.front().z.pop_back();
  try {
    writer.write(8, particles, bad_fields);
    std::cerr << "VTK writer accepted a mismatched vector field length" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  std::filesystem::remove_all(output_dir);
  return 0;
}
