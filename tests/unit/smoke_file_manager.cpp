#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lsmps3d/io/file_manager.hpp"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

bool almost_equal(lsmps3d::real lhs, lsmps3d::real rhs) {
  return std::abs(lhs - rhs) <= static_cast<lsmps3d::real>(1.0e-10);
}

bool file_contains(const std::filesystem::path& path, const std::string& token) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "Failed to open generated file: " << path << std::endl;
    return false;
  }

  const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (contents.find(token) == std::string::npos) {
    std::cerr << "Generated file is missing token: " << token << std::endl;
    return false;
  }
  return true;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  const fs::path root = fs::temp_directory_path() / "lsmps3d_file_manager_smoke";
  fs::remove_all(root);
  fs::create_directories(root);

  const fs::path fluid_path = root / "fluid.csv";
  {
    std::ofstream fluid(fluid_path);
    fluid << "# x,y,z,vx,vy,vz\n"
          << "x,y,z,vx,vy,vz\n"
          << "0,0,0,0,0,0\n"
          << "1,0,0,0.5,0.25,0.125\n";
  }

  const fs::path wall_path = root / "wall.csv";
  {
    std::ofstream wall(wall_path);
    wall << "x,y,z,vx,vy,vz,nx,ny,nz\n"
         << "0,-1,0,0,0,0,0,1,0\n"
         << "1,-1,0,0.1,0.2,0.3,0,1,0\n";
  }

  lsmps3d::SimulationConfig config;
  config.fluid_particle_file = fluid_path;
  config.wall_particle_file = wall_path;
  config.output_directory = root / "vtk";
  config.vtk_file_prefix = "case";
  config.vtk_write_point_fields = true;

  const lsmps3d::FileManager manager(config);
  const lsmps3d::ParticleInputData input = manager.load_particle_input();

  if (input.fluid.count() != 2 || input.walls.count() != 2 ||
      !almost_equal(input.fluid.vx[0], static_cast<lsmps3d::real>(0)) ||
      !almost_equal(input.fluid.vx[1], static_cast<lsmps3d::real>(0.5)) ||
      !almost_equal(input.fluid.pressure[1], static_cast<lsmps3d::real>(0)) ||
      input.fluid.surface_type[0] != static_cast<int>(lsmps3d::SurfaceType::Inner) ||
      input.fluid.surface_type[1] != static_cast<int>(lsmps3d::SurfaceType::Inner) ||
      !almost_equal(input.walls.normal_y[0], static_cast<lsmps3d::real>(1)) ||
      !almost_equal(input.walls.vz[1], static_cast<lsmps3d::real>(0.3))) {
    std::cerr << "FileManager did not load CSV particle data as expected" << std::endl;
    return 1;
  }

  const fs::path saved_config_path = root / "saved.ini";
  manager.save_config(config, saved_config_path);
  const lsmps3d::SimulationConfig reloaded_config = manager.load_config(saved_config_path);
  if (reloaded_config.fluid_particle_file != fluid_path ||
      reloaded_config.wall_particle_file != wall_path ||
      reloaded_config.output_directory != config.output_directory) {
    std::cerr << "FileManager config round-trip did not preserve file paths" << std::endl;
    return 1;
  }

  lsmps3d::HostVtkPointFields fluid_fields;
  fluid_fields.add_vector("velocity", input.fluid.vx, input.fluid.vy, input.fluid.vz);
  fluid_fields.add_real_scalar("pressure", input.fluid.pressure);
  fluid_fields.add_int_scalar("surface_type", input.fluid.surface_type);
  manager.write_fluid_result(3, input.fluid, fluid_fields);

  lsmps3d::HostVtkPointFields wall_fields;
  wall_fields.add_vector("normal", input.walls.normal_x, input.walls.normal_y, input.walls.normal_z);
  wall_fields.add_vector("velocity", input.walls.vx, input.walls.vy, input.walls.vz);
  manager.write_wall_result(3, input.walls, wall_fields);

  const fs::path fluid_vtk = config.output_directory / "case_fluid_000003.vtk";
  const fs::path wall_vtk = config.output_directory / "case_wall_000003.vtk";
  if (!file_contains(fluid_vtk, "VECTORS velocity float") ||
      !file_contains(fluid_vtk, "SCALARS pressure float 1") ||
      !file_contains(wall_vtk, "VECTORS normal float")) {
    return 1;
  }

  const fs::path bad_path = root / "bad.csv";
  {
    std::ofstream bad(bad_path);
    bad << "x,y,z,vx,vy,vz\n"
        << "0,not-a-number,0,0,0,0\n";
  }
  try {
    (void)manager.load_fluid_particles(bad_path);
    std::cerr << "FileManager accepted invalid fluid CSV input" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  const fs::path extra_column_path = root / "extra_column.csv";
  {
    std::ofstream extra(extra_column_path);
    extra << "x,y,z,vx,vy,vz,pressure\n"
          << "0,0,0,0,0,0,100\n";
  }
  try {
    (void)manager.load_fluid_particles(extra_column_path);
    std::cerr << "FileManager accepted extra fluid CSV columns" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  fs::remove_all(root);
  return 0;
}
