#include "lsmps3d/io/file_manager.hpp"

#include "lsmps3d/surface/surface_type.cuh"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

std::vector<std::string> split_csv_line(std::string_view line) {
  std::vector<std::string> values;
  std::size_t start = 0;
  while (start <= line.size()) {
    const std::size_t end = line.find(',', start);
    const std::size_t token_end = (end == std::string_view::npos) ? line.size() : end;
    values.push_back(trim(line.substr(start, token_end - start)));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return values;
}

bool is_header_row(const std::vector<std::string>& values) {
  if (values.empty()) {
    return false;
  }
  std::size_t parsed = 0;
  try {
    (void)std::stod(values.front(), &parsed);
  } catch (const std::exception&) {
    return true;
  }
  return parsed != values.front().size();
}

real parse_real_value(const std::string& text,
                      const std::filesystem::path& path,
                      std::size_t line_number,
                      std::string_view column_name) {
  std::size_t parsed = 0;
  try {
    const double value = std::stod(text, &parsed);
    if (parsed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return static_cast<real>(value);
  } catch (const std::exception& error) {
    std::ostringstream message;
    message << "Invalid real value in " << path << " line " << line_number << " column "
            << column_name << ": '" << text << "' (" << error.what() << ")";
    throw std::invalid_argument(message.str());
  }
}

void require_column_count(const std::vector<std::string>& values,
                          std::size_t expected_count,
                          const std::filesystem::path& path,
                          std::size_t line_number,
                          std::string_view format_name) {
  if (values.size() != expected_count) {
    std::ostringstream message;
    message << "Invalid " << format_name << " CSV row in " << path << " line " << line_number
            << ": expected " << expected_count << " columns, got " << values.size();
    throw std::invalid_argument(message.str());
  }
}

std::ifstream open_input_file(const std::filesystem::path& path, std::string_view label) {
  std::ifstream input(path);
  if (!input) {
    std::ostringstream message;
    message << "Unable to open " << label << ": " << path;
    throw std::runtime_error(message.str());
  }
  return input;
}

void validate_snapshot_count(const HostParticleSnapshot& snapshot, size_type expected, std::string_view label) {
  if (snapshot.x.size() != expected || snapshot.y.size() != expected || snapshot.z.size() != expected) {
    std::ostringstream message;
    message << label << " particle coordinate arrays must have matching lengths";
    throw std::invalid_argument(message.str());
  }
}

SimulationConfig result_config(SimulationConfig config, std::string prefix_suffix) {
  config.vtk_file_prefix += std::move(prefix_suffix);
  return config;
}

}  // namespace

FileManager::FileManager(SimulationConfig config) : config_(std::move(config)) {}

void FileManager::set_config(SimulationConfig config) {
  config_ = std::move(config);
}

SimulationConfig FileManager::load_config(const std::filesystem::path& path) const {
  return load_simulation_config(path);
}

void FileManager::save_config(const SimulationConfig& config, const std::filesystem::path& path) const {
  save_simulation_config(config, path);
}

HostFluidParticles FileManager::load_fluid_particles(const std::filesystem::path& path) const {
  auto input = open_input_file(path, "fluid particle CSV");
  HostFluidParticles particles;

  std::string line;
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

    const std::vector<std::string> values = split_csv_line(statement);
    if (particles.count() == 0 && is_header_row(values)) {
      continue;
    }
    require_column_count(values, 6, path, line_number, "fluid particle");

    particles.x.push_back(parse_real_value(values[0], path, line_number, "x"));
    particles.y.push_back(parse_real_value(values[1], path, line_number, "y"));
    particles.z.push_back(parse_real_value(values[2], path, line_number, "z"));
    particles.vx.push_back(parse_real_value(values[3], path, line_number, "vx"));
    particles.vy.push_back(parse_real_value(values[4], path, line_number, "vy"));
    particles.vz.push_back(parse_real_value(values[5], path, line_number, "vz"));
    particles.pressure.push_back(static_cast<real>(0));
    particles.surface_type.push_back(static_cast<int>(SurfaceType::Inner));
  }

  return particles;
}

HostWallParticles FileManager::load_wall_particles(const std::filesystem::path& path) const {
  auto input = open_input_file(path, "wall particle CSV");
  HostWallParticles particles;

  std::string line;
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

    const std::vector<std::string> values = split_csv_line(statement);
    if (particles.count() == 0 && is_header_row(values)) {
      continue;
    }
    require_column_count(values, 9, path, line_number, "wall particle");

    particles.x.push_back(parse_real_value(values[0], path, line_number, "x"));
    particles.y.push_back(parse_real_value(values[1], path, line_number, "y"));
    particles.z.push_back(parse_real_value(values[2], path, line_number, "z"));
    particles.vx.push_back(parse_real_value(values[3], path, line_number, "vx"));
    particles.vy.push_back(parse_real_value(values[4], path, line_number, "vy"));
    particles.vz.push_back(parse_real_value(values[5], path, line_number, "vz"));
    particles.normal_x.push_back(parse_real_value(values[6], path, line_number, "normal_x"));
    particles.normal_y.push_back(parse_real_value(values[7], path, line_number, "normal_y"));
    particles.normal_z.push_back(parse_real_value(values[8], path, line_number, "normal_z"));
  }

  return particles;
}

ParticleInputData FileManager::load_particle_input(const std::filesystem::path& fluid_path,
                                                   const std::filesystem::path& wall_path) const {
  return ParticleInputData{load_fluid_particles(fluid_path), load_wall_particles(wall_path)};
}

ParticleInputData FileManager::load_particle_input() const {
  return load_particle_input(config_.fluid_particle_file, config_.wall_particle_file);
}

void FileManager::write_fluid_result(size_type step,
                                     const HostFluidParticles& particles,
                                     const HostVtkPointFields& point_fields) const {
  const HostParticleSnapshot snapshot = particles.snapshot();
  validate_snapshot_count(snapshot, particles.count(), "Fluid");
  LegacyVtkWriter(result_config(config_, "_fluid")).write(step, snapshot, point_fields);
}

void FileManager::write_wall_result(size_type step,
                                    const HostWallParticles& particles,
                                    const HostVtkPointFields& point_fields) const {
  const HostParticleSnapshot snapshot = particles.snapshot();
  validate_snapshot_count(snapshot, particles.count(), "Wall");
  LegacyVtkWriter(result_config(config_, "_wall")).write(step, snapshot, point_fields);
}

}  // namespace lsmps3d
