#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "lsmps3d/core/config.hpp"

namespace {

bool almost_equal(lsmps3d::real lhs, lsmps3d::real rhs) {
  return std::abs(lhs - rhs) <= static_cast<lsmps3d::real>(1.0e-10);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  const lsmps3d::SimulationConfig defaults = lsmps3d::default_simulation_config();
  if (!almost_equal(defaults.neighbor_radius(), defaults.support_radius) ||
      !almost_equal(defaults.support_radius_ratio(), static_cast<lsmps3d::real>(3.1)) ||
      !almost_equal(defaults.particle_volume(), static_cast<lsmps3d::real>(1.0e-6))) {
    std::cerr << "Unexpected default simulation config values" << std::endl;
    return 1;
  }

  const fs::path input_path = fs::temp_directory_path() / "lsmps3d_smoke_config.ini";
  {
    std::ofstream input(input_path);
    input << "# comments and whitespace are allowed\n"
          << "[geometry]\n"
          << "particle_spacing = 0.02\n"
          << "support_radius = 0.062\n"
          << "near_surface_radius = 0.04\n"
          << "cell_size = 0.062\n\n"
          << "[simulation]\n"
          << "time_step = 0.0005\n"
          << "min_time_step = 0.0001\n"
          << "max_time_step = 0.002\n"
          << "time_step_growth_factor = 1.2\n"
          << "final_time = 0.5\n"
          << "output_interval = 0.05\n"
          << "density = 998\n"
          << "kinematic_viscosity = 0.0000011\n"
          << "cfl = 0.25\n"
          << "gravity_z = -9.8\n\n"
          << "[surface]\n"
          << "splash_neighbor_threshold = 12\n"
          << "number_density_ratio_threshold = 0.82\n"
          << "air_open_ratio_threshold = 0.31\n"
          << "air_anisotropy_threshold = 0.04\n"
          << "include_wall_neighbors = true\n"
          << "virtual_light_cone_cosine = 0.8\n\n"
          << "[lsmps]\n"
          << "regularization = 0.000000001\n"
          << "wall_weight_scale = 0.75\n\n"
          << "[correction]\n"
          << "ps_displacement_scale = 0.02\n"
          << "ps_min_distance_ratio = 0.8\n"
          << "ps_max_displacement_ratio = 0.15\n"
          << "wall_clearance_ratio = 0.3\n"
          << "velocity_smoothing_strength = 0.05\n\n"
          << "[files]\n"
          << "output_directory = /tmp/lsmps3d_output\n"
          << "vtk_file_prefix = smoke\n"
          << "vtk_write_point_fields = false\n"
          << "amgx_config_path = configs/custom.json\n"
          << "amgx_print_solve_stats = true\n";
  }

  const lsmps3d::SimulationConfig loaded = lsmps3d::load_simulation_config(input_path);
  if (!almost_equal(loaded.particle_spacing, static_cast<lsmps3d::real>(0.02)) ||
      !almost_equal(loaded.support_radius, static_cast<lsmps3d::real>(0.062)) ||
      !almost_equal(loaded.cell_size, static_cast<lsmps3d::real>(0.062)) ||
      loaded.vtk_write_point_fields ||
      !loaded.amgx_print_solve_stats || loaded.vtk_file_prefix != "smoke" ||
      loaded.amgx_config_path != "configs/custom.json" ||
      !almost_equal(loaded.min_time_step, static_cast<lsmps3d::real>(0.0001)) ||
      !almost_equal(loaded.max_time_step, static_cast<lsmps3d::real>(0.002)) ||
      !almost_equal(loaded.time_step_growth_factor, static_cast<lsmps3d::real>(1.2)) ||
      !almost_equal(loaded.final_time, static_cast<lsmps3d::real>(0.5)) ||
      !almost_equal(loaded.output_interval, static_cast<lsmps3d::real>(0.05)) ||
      !almost_equal(loaded.ps_displacement_scale, static_cast<lsmps3d::real>(0.02)) ||
      !almost_equal(loaded.velocity_smoothing_strength, static_cast<lsmps3d::real>(0.05))) {
    std::cerr << "Loaded simulation config does not match expected overrides" << std::endl;
    return 1;
  }

  const fs::path saved_path = fs::temp_directory_path() / "lsmps3d_smoke_config_saved.ini";
  lsmps3d::save_simulation_config(loaded, saved_path);
  const lsmps3d::SimulationConfig reloaded = lsmps3d::load_simulation_config(saved_path);
  if (!almost_equal(reloaded.support_radius, loaded.support_radius) ||
      reloaded.output_directory != loaded.output_directory ||
      reloaded.vtk_file_prefix != loaded.vtk_file_prefix) {
    std::cerr << "Saved simulation config did not round-trip" << std::endl;
    return 1;
  }

  lsmps3d::SimulationConfig invalid = defaults;
  invalid.support_radius = static_cast<lsmps3d::real>(-1);
  try {
    lsmps3d::validate_simulation_config(invalid);
    std::cerr << "Invalid simulation config was accepted" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  invalid = defaults;
  invalid.min_time_step = static_cast<lsmps3d::real>(0.01);
  invalid.max_time_step = static_cast<lsmps3d::real>(0.001);
  try {
    lsmps3d::validate_simulation_config(invalid);
    std::cerr << "Invalid time step limits were accepted" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  const fs::path bad_path = fs::temp_directory_path() / "lsmps3d_smoke_config_bad.ini";
  {
    std::ofstream bad(bad_path);
    bad << "[geometry]\nunknown_key=1\n";
  }
  try {
    (void)lsmps3d::load_simulation_config(bad_path);
    std::cerr << "Unknown config key was accepted" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  fs::remove(input_path);
  fs::remove(saved_path);
  fs::remove(bad_path);
  return 0;
}
