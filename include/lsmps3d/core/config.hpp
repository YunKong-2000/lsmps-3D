#pragma once

#include <filesystem>
#include <string>

#include "lsmps3d/core/constants.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/neighbor/cell_list.cuh"

namespace lsmps3d {

struct SimulationConfig {
  // Geometry.
  real particle_spacing{static_cast<real>(0.01)};
  real support_radius{static_cast<real>(0.031)};
  real near_surface_radius{static_cast<real>(0.015)};
  Vec3 cell_origin{};
  real cell_size{support_radius};
  Int3 cell_dims{1, 1, 1};

  // Simulation.
  real time_step{static_cast<real>(1.0e-4)};
  real min_time_step{static_cast<real>(1.0e-6)};
  real max_time_step{static_cast<real>(1.0e-3)};
  real time_step_growth_factor{static_cast<real>(1.05)};
  real final_time{static_cast<real>(1.0)};
  real output_interval{static_cast<real>(0.01)};
  real density{static_cast<real>(1000.0)};
  real kinematic_viscosity{static_cast<real>(1.0e-6)};
  real cfl{kDefaultCfl};
  Vec3 gravity{static_cast<real>(0.0), static_cast<real>(0.0), kDefaultGravityZ};

  // Surface and virtual-light diagnostics.
  index_t splash_neighbor_threshold{4};
  real number_density_ratio_threshold{static_cast<real>(0.85)};
  real air_open_ratio_threshold{static_cast<real>(0.33)};
  real air_anisotropy_threshold{static_cast<real>(0.033)};
  bool include_wall_neighbors{true};
  real wall_normal_independence_threshold{static_cast<real>(0.25)};
  real virtual_light_cone_cosine{static_cast<real>(0.8660254037844386)};

  // LSMPS.
  real lsmps_regularization{static_cast<real>(1.0e-8)};
  real lsmps_wall_weight_scale{static_cast<real>(1)};

  // Pressure correction, particle shifting, and smoothing.
  real ps_displacement_scale{static_cast<real>(0.05)};
  real ps_min_distance_ratio{static_cast<real>(0.85)};
  real ps_max_displacement_ratio{static_cast<real>(0.2)};
  real wall_clearance_ratio{static_cast<real>(0.25)};
  real velocity_smoothing_strength{static_cast<real>(0.1)};

  // Files and diagnostics.
  std::filesystem::path fluid_particle_file{"input/fluid_particles.csv"};
  std::filesystem::path wall_particle_file{"input/wall_particles.csv"};
  std::filesystem::path output_directory{"output"};
  std::string vtk_file_prefix{"lsmps3d"};
  bool vtk_write_point_fields{true};
  std::string amgx_config_path{"configs/amgx_ppe.json"};
  bool amgx_print_solve_stats{false};

  [[nodiscard]] real neighbor_radius() const noexcept {
    return support_radius;
  }

  [[nodiscard]] real particle_volume() const noexcept {
    return particle_spacing * particle_spacing * particle_spacing;
  }

  [[nodiscard]] real support_radius_ratio() const noexcept {
    return support_radius / particle_spacing;
  }

  [[nodiscard]] CellGrid cell_grid() const noexcept {
    return CellGrid{cell_origin, cell_size, cell_dims};
  }

  [[nodiscard]] size_type cell_capacity() const noexcept {
    return static_cast<size_type>(cell_dims.x) * static_cast<size_type>(cell_dims.y) *
           static_cast<size_type>(cell_dims.z);
  }
};

[[nodiscard]] SimulationConfig default_simulation_config() noexcept;

void validate_simulation_config(const SimulationConfig& config);

[[nodiscard]] SimulationConfig load_simulation_config(const std::filesystem::path& path);

void save_simulation_config(const SimulationConfig& config, const std::filesystem::path& path);

}  // namespace lsmps3d
