#include "lsmps3d/core/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lsmps3d {
namespace {

std::string trim(std::string_view value) {
  const auto is_not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
  const auto begin = std::find_if(value.begin(), value.end(), is_not_space);
  const auto end = std::find_if(value.rbegin(), value.rend(), is_not_space).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

real parse_real(std::string_view key, std::string_view value) {
  std::string text = trim(value);
  std::size_t parsed = 0;
  try {
    const double result = std::stod(text, &parsed);
    if (parsed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return static_cast<real>(result);
  } catch (const std::exception& error) {
    std::ostringstream message;
    message << "Invalid real value for '" << key << "': '" << value << "' (" << error.what()
            << ")";
    throw std::invalid_argument(message.str());
  }
}

index_t parse_index(std::string_view key, std::string_view value) {
  std::string text = trim(value);
  std::size_t parsed = 0;
  try {
    const long result = std::stol(text, &parsed, 10);
    if (parsed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return static_cast<index_t>(result);
  } catch (const std::exception& error) {
    std::ostringstream message;
    message << "Invalid integer value for '" << key << "': '" << value << "' (" << error.what()
            << ")";
    throw std::invalid_argument(message.str());
  }
}

bool parse_bool(std::string_view key, std::string_view value) {
  std::string text = trim(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (text == "true" || text == "1" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "false" || text == "0" || text == "no" || text == "off") {
    return false;
  }

  std::ostringstream message;
  message << "Invalid boolean value for '" << key << "': '" << value << "'";
  throw std::invalid_argument(message.str());
}

void require_positive(real value, std::string_view key) {
  if (value <= static_cast<real>(0)) {
    std::ostringstream message;
    message << "SimulationConfig field '" << key << "' must be positive";
    throw std::invalid_argument(message.str());
  }
}

void require_positive(index_t value, std::string_view key) {
  if (value <= 0) {
    std::ostringstream message;
    message << "SimulationConfig field '" << key << "' must be positive";
    throw std::invalid_argument(message.str());
  }
}

void require_non_negative(real value, std::string_view key) {
  if (value < static_cast<real>(0)) {
    std::ostringstream message;
    message << "SimulationConfig field '" << key << "' must be non-negative";
    throw std::invalid_argument(message.str());
  }
}

void write_real(std::ostream& output, std::string_view key, real value) {
  output << key << '=' << std::setprecision(17) << static_cast<double>(value) << '\n';
}

void write_index(std::ostream& output, std::string_view key, index_t value) {
  output << key << '=' << value << '\n';
}

void write_bool(std::ostream& output, std::string_view key, bool value) {
  output << key << '=' << (value ? "true" : "false") << '\n';
}

std::string scoped_key(std::string_view section, std::string_view key) {
  if (section.empty()) {
    return std::string(key);
  }
  std::string result(section);
  result.push_back('.');
  result.append(key);
  return result;
}

void apply_config_value(SimulationConfig& config,
                        std::string_view section,
                        std::string_view key,
                        std::string_view value,
                        std::size_t line_number) {
  const std::string full_key = scoped_key(section, key);

  if (section == "geometry") {
    if (key == "particle_spacing") {
      config.particle_spacing = parse_real(full_key, value);
    } else if (key == "support_radius") {
      config.support_radius = parse_real(full_key, value);
    } else if (key == "near_surface_radius") {
      config.near_surface_radius = parse_real(full_key, value);
    } else if (key == "cell_origin_x") {
      config.cell_origin.x = parse_real(full_key, value);
    } else if (key == "cell_origin_y") {
      config.cell_origin.y = parse_real(full_key, value);
    } else if (key == "cell_origin_z") {
      config.cell_origin.z = parse_real(full_key, value);
    } else if (key == "cell_size") {
      config.cell_size = parse_real(full_key, value);
    } else if (key == "cell_dim_x") {
      config.cell_dims.x = parse_index(full_key, value);
    } else if (key == "cell_dim_y") {
      config.cell_dims.y = parse_index(full_key, value);
    } else if (key == "cell_dim_z") {
      config.cell_dims.z = parse_index(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else if (section == "simulation") {
    if (key == "time_step") {
      config.time_step = parse_real(full_key, value);
    } else if (key == "min_time_step") {
      config.min_time_step = parse_real(full_key, value);
    } else if (key == "max_time_step") {
      config.max_time_step = parse_real(full_key, value);
    } else if (key == "time_step_growth_factor") {
      config.time_step_growth_factor = parse_real(full_key, value);
    } else if (key == "final_time") {
      config.final_time = parse_real(full_key, value);
    } else if (key == "output_interval") {
      config.output_interval = parse_real(full_key, value);
    } else if (key == "density") {
      config.density = parse_real(full_key, value);
    } else if (key == "kinematic_viscosity") {
      config.kinematic_viscosity = parse_real(full_key, value);
    } else if (key == "cfl") {
      config.cfl = parse_real(full_key, value);
    } else if (key == "gravity_x") {
      config.gravity.x = parse_real(full_key, value);
    } else if (key == "gravity_y") {
      config.gravity.y = parse_real(full_key, value);
    } else if (key == "gravity_z") {
      config.gravity.z = parse_real(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else if (section == "surface") {
    if (key == "splash_neighbor_threshold") {
      config.splash_neighbor_threshold = parse_index(full_key, value);
    } else if (key == "number_density_ratio_threshold") {
      config.number_density_ratio_threshold = parse_real(full_key, value);
    } else if (key == "air_open_ratio_threshold") {
      config.air_open_ratio_threshold = parse_real(full_key, value);
    } else if (key == "air_anisotropy_threshold") {
      config.air_anisotropy_threshold = parse_real(full_key, value);
    } else if (key == "include_wall_neighbors") {
      config.include_wall_neighbors = parse_bool(full_key, value);
    } else if (key == "wall_normal_independence_threshold") {
      config.wall_normal_independence_threshold = parse_real(full_key, value);
    } else if (key == "virtual_light_cone_cosine") {
      config.virtual_light_cone_cosine = parse_real(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else if (section == "lsmps") {
    if (key == "regularization") {
      config.lsmps_regularization = parse_real(full_key, value);
    } else if (key == "wall_weight_scale") {
      config.lsmps_wall_weight_scale = parse_real(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else if (section == "correction") {
    if (key == "ps_displacement_scale") {
      config.ps_displacement_scale = parse_real(full_key, value);
    } else if (key == "ps_min_distance_ratio") {
      config.ps_min_distance_ratio = parse_real(full_key, value);
    } else if (key == "ps_max_displacement_ratio") {
      config.ps_max_displacement_ratio = parse_real(full_key, value);
    } else if (key == "wall_clearance_ratio") {
      config.wall_clearance_ratio = parse_real(full_key, value);
    } else if (key == "velocity_smoothing_strength") {
      config.velocity_smoothing_strength = parse_real(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else if (section == "files") {
    if (key == "fluid_particle_file") {
      config.fluid_particle_file = trim(value);
    } else if (key == "wall_particle_file") {
      config.wall_particle_file = trim(value);
    } else if (key == "output_directory") {
      config.output_directory = trim(value);
    } else if (key == "vtk_file_prefix") {
      config.vtk_file_prefix = trim(value);
    } else if (key == "vtk_write_point_fields") {
      config.vtk_write_point_fields = parse_bool(full_key, value);
    } else if (key == "amgx_config_path") {
      config.amgx_config_path = trim(value);
    } else if (key == "amgx_print_solve_stats") {
      config.amgx_print_solve_stats = parse_bool(full_key, value);
    } else {
      throw std::invalid_argument("Unknown simulation config key '" + full_key + "' on line " +
                                  std::to_string(line_number));
    }
  } else {
    throw std::invalid_argument("Unknown simulation config section '" + std::string(section) +
                                "' on line " + std::to_string(line_number));
  }
}

}  // namespace

SimulationConfig default_simulation_config() noexcept {
  return SimulationConfig{};
}

void validate_simulation_config(const SimulationConfig& config) {
  require_positive(config.particle_spacing, "particle_spacing");
  require_positive(config.support_radius, "support_radius");
  require_non_negative(config.near_surface_radius, "near_surface_radius");
  require_positive(config.cell_size, "cell_size");
  require_positive(config.time_step, "time_step");
  require_positive(config.min_time_step, "min_time_step");
  require_positive(config.max_time_step, "max_time_step");
  require_positive(config.time_step_growth_factor, "time_step_growth_factor");
  require_non_negative(config.final_time, "final_time");
  require_positive(config.output_interval, "output_interval");
  require_positive(config.density, "density");
  require_positive(config.kinematic_viscosity, "kinematic_viscosity");
  require_positive(config.cfl, "cfl");
  require_non_negative(config.lsmps_regularization, "lsmps_regularization");
  require_positive(config.lsmps_wall_weight_scale, "lsmps_wall_weight_scale");
  require_non_negative(config.ps_displacement_scale, "ps_displacement_scale");
  require_positive(config.ps_min_distance_ratio, "ps_min_distance_ratio");
  require_non_negative(config.ps_max_displacement_ratio, "ps_max_displacement_ratio");
  require_non_negative(config.wall_clearance_ratio, "wall_clearance_ratio");
  require_non_negative(config.velocity_smoothing_strength, "velocity_smoothing_strength");

  if (config.support_radius > config.cell_size) {
    throw std::invalid_argument("SimulationConfig requires support_radius <= cell_size");
  }
  if (config.min_time_step > config.max_time_step) {
    throw std::invalid_argument("SimulationConfig requires min_time_step <= max_time_step");
  }
  if (config.time_step_growth_factor < static_cast<real>(1)) {
    throw std::invalid_argument("SimulationConfig field 'time_step_growth_factor' must be >= 1");
  }
  if (config.number_density_ratio_threshold <= static_cast<real>(0)) {
    throw std::invalid_argument("SimulationConfig field 'number_density_ratio_threshold' must be "
                                "positive");
  }
  if (config.virtual_light_cone_cosine < static_cast<real>(-1) ||
      config.virtual_light_cone_cosine > static_cast<real>(1)) {
    throw std::invalid_argument("SimulationConfig field 'virtual_light_cone_cosine' must be in "
                                "[-1, 1]");
  }
  if (config.velocity_smoothing_strength > static_cast<real>(1)) {
    throw std::invalid_argument("SimulationConfig field 'velocity_smoothing_strength' must be in "
                                "[0, 1]");
  }

  if (config.amgx_config_path.empty()) {
    throw std::invalid_argument("SimulationConfig field 'amgx_config_path' must not be empty");
  }
  if (config.fluid_particle_file.empty()) {
    throw std::invalid_argument("SimulationConfig field 'fluid_particle_file' must not be empty");
  }
  if (config.wall_particle_file.empty()) {
    throw std::invalid_argument("SimulationConfig field 'wall_particle_file' must not be empty");
  }
  if (config.vtk_file_prefix.empty()) {
    throw std::invalid_argument("SimulationConfig field 'vtk_file_prefix' must not be empty");
  }
}

SimulationConfig load_simulation_config(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    std::ostringstream message;
    message << "Unable to open simulation config: " << path;
    throw std::runtime_error(message.str());
  }

  SimulationConfig config = default_simulation_config();
  std::string line;
  std::string section;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }

    const std::string statement = trim(line);
    if (statement.empty()) {
      continue;
    }

    if (statement.front() == '[' && statement.back() == ']') {
      section = trim(std::string_view(statement).substr(1, statement.size() - 2));
      if (section.empty()) {
        throw std::invalid_argument("Invalid config line " + std::to_string(line_number) +
                                    ": empty section");
      }
      continue;
    }

    const auto separator_pos = statement.find('=');
    if (separator_pos == std::string::npos) {
      std::ostringstream message;
      message << "Invalid config line " << line_number << ": expected key=value";
      throw std::invalid_argument(message.str());
    }

    const std::string key = trim(std::string_view(statement).substr(0, separator_pos));
    const std::string value = trim(std::string_view(statement).substr(separator_pos + 1));
    if (key.empty()) {
      std::ostringstream message;
      message << "Invalid config line " << line_number << ": empty key";
      throw std::invalid_argument(message.str());
    }
    apply_config_value(config, section, key, value, line_number);
  }

  validate_simulation_config(config);
  return config;
}

void save_simulation_config(const SimulationConfig& config, const std::filesystem::path& path) {
  validate_simulation_config(config);

  std::ofstream output(path);
  if (!output) {
    std::ostringstream message;
    message << "Unable to write simulation config: " << path;
    throw std::runtime_error(message.str());
  }

  output << "# lsmps3d simulation config\n";
  output << "\n[geometry]\n";
  write_real(output, "particle_spacing", config.particle_spacing);
  write_real(output, "support_radius", config.support_radius);
  write_real(output, "near_surface_radius", config.near_surface_radius);
  write_real(output, "cell_size", config.cell_size);

  output << "\n[simulation]\n";
  write_real(output, "time_step", config.time_step);
  write_real(output, "min_time_step", config.min_time_step);
  write_real(output, "max_time_step", config.max_time_step);
  write_real(output, "time_step_growth_factor", config.time_step_growth_factor);
  write_real(output, "final_time", config.final_time);
  write_real(output, "output_interval", config.output_interval);
  write_real(output, "density", config.density);
  write_real(output, "kinematic_viscosity", config.kinematic_viscosity);
  write_real(output, "cfl", config.cfl);
  write_real(output, "gravity_x", config.gravity.x);
  write_real(output, "gravity_y", config.gravity.y);
  write_real(output, "gravity_z", config.gravity.z);

  output << "\n[surface]\n";
  write_index(output, "splash_neighbor_threshold", config.splash_neighbor_threshold);
  write_real(output, "number_density_ratio_threshold", config.number_density_ratio_threshold);
  write_real(output, "air_open_ratio_threshold", config.air_open_ratio_threshold);
  write_real(output, "air_anisotropy_threshold", config.air_anisotropy_threshold);
  write_bool(output, "include_wall_neighbors", config.include_wall_neighbors);
  write_real(output,
             "wall_normal_independence_threshold",
             config.wall_normal_independence_threshold);
  write_real(output, "virtual_light_cone_cosine", config.virtual_light_cone_cosine);

  output << "\n[lsmps]\n";
  write_real(output, "regularization", config.lsmps_regularization);
  write_real(output, "wall_weight_scale", config.lsmps_wall_weight_scale);

  output << "\n[correction]\n";
  write_real(output, "ps_displacement_scale", config.ps_displacement_scale);
  write_real(output, "ps_min_distance_ratio", config.ps_min_distance_ratio);
  write_real(output, "ps_max_displacement_ratio", config.ps_max_displacement_ratio);
  write_real(output, "wall_clearance_ratio", config.wall_clearance_ratio);
  write_real(output, "velocity_smoothing_strength", config.velocity_smoothing_strength);

  output << "\n[files]\n";
  output << "fluid_particle_file=" << config.fluid_particle_file.string() << '\n';
  output << "wall_particle_file=" << config.wall_particle_file.string() << '\n';
  output << "output_directory=" << config.output_directory.string() << '\n';
  output << "vtk_file_prefix=" << config.vtk_file_prefix << '\n';
  write_bool(output, "vtk_write_point_fields", config.vtk_write_point_fields);
  output << "amgx_config_path=" << config.amgx_config_path << '\n';
  write_bool(output, "amgx_print_solve_stats", config.amgx_print_solve_stats);
}

}  // namespace lsmps3d
