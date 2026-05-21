#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Vec3 {
  double x{};
  double y{};
  double z{};
};

struct HydrostaticBox {
  std::vector<Vec3> fluid_positions;
  std::vector<Vec3> wall_positions;
  std::vector<Vec3> wall_normals;
};

struct CaseConfig {
  double box_size{1.0};
  double liquid_height{0.5};
  double spacing{0.02};
  std::filesystem::path output_directory{"input"};
};

void require_positive(double value, const char* name) {
  if (!(value > 0.0) || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be a positive finite value");
  }
}

int rounded_divisions(double length, double spacing, const char* name) {
  require_positive(length, name);
  require_positive(spacing, "spacing");

  const double raw = length / spacing;
  const double rounded = std::round(raw);
  if (std::abs(raw - rounded) > 1.0e-9) {
    throw std::invalid_argument(std::string(name) + " must be an integer multiple of spacing");
  }
  return static_cast<int>(rounded);
}

HydrostaticBox make_hydrostatic_box(const CaseConfig& config) {
  const int cells_per_side = rounded_divisions(config.box_size, config.spacing, "box_size");
  const int fluid_layers = rounded_divisions(config.liquid_height, config.spacing, "liquid_height");
  const int last = cells_per_side;

  HydrostaticBox result;
  result.fluid_positions.reserve(
      static_cast<std::size_t>(cells_per_side) * cells_per_side * fluid_layers);

  for (int iz = 0; iz < fluid_layers; ++iz) {
    for (int iy = 0; iy < cells_per_side; ++iy) {
      for (int ix = 0; ix < cells_per_side; ++ix) {
        result.fluid_positions.push_back(Vec3{(static_cast<double>(ix) + 0.5) * config.spacing,
                                             (static_cast<double>(iy) + 0.5) * config.spacing,
                                             (static_cast<double>(iz) + 0.5) * config.spacing});
      }
    }
  }

  std::vector<std::tuple<int, int, int, Vec3>> wall_samples;
  wall_samples.reserve(static_cast<std::size_t>(5) * (cells_per_side + 1) * (cells_per_side + 1));

  for (int iz = 0; iz <= cells_per_side; ++iz) {
    for (int iy = 0; iy <= cells_per_side; ++iy) {
      wall_samples.emplace_back(0, iy, iz, Vec3{1.0, 0.0, 0.0});
      wall_samples.emplace_back(last, iy, iz, Vec3{-1.0, 0.0, 0.0});
    }
    for (int ix = 1; ix < last; ++ix) {
      wall_samples.emplace_back(ix, 0, iz, Vec3{0.0, 1.0, 0.0});
      wall_samples.emplace_back(ix, last, iz, Vec3{0.0, -1.0, 0.0});
    }
  }

  for (int iy = 1; iy < last; ++iy) {
    for (int ix = 1; ix < last; ++ix) {
      wall_samples.emplace_back(ix, iy, 0, Vec3{0.0, 0.0, 1.0});
    }
  }

  result.wall_positions.reserve(wall_samples.size());
  result.wall_normals.reserve(wall_samples.size());
  for (const auto& [ix, iy, iz, normal] : wall_samples) {
    result.wall_positions.push_back(Vec3{static_cast<double>(ix) * config.spacing,
                                         static_cast<double>(iy) * config.spacing,
                                         static_cast<double>(iz) * config.spacing});
    result.wall_normals.push_back(normal);
  }

  return result;
}

void write_fluid_particles(const std::filesystem::path& path, const std::vector<Vec3>& positions) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open fluid particle output: " + path.string());
  }

  out << std::setprecision(12);
  out << "x,y,z,vx,vy,vz\n";
  for (const Vec3& point : positions) {
    out << point.x << ',' << point.y << ',' << point.z << ",0,0,0\n";
  }
}

void write_wall_particles(const std::filesystem::path& path,
                          const std::vector<Vec3>& positions,
                          const std::vector<Vec3>& normals) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open wall particle output: " + path.string());
  }

  out << std::setprecision(12);
  out << "x,y,z,vx,vy,vz,nx,ny,nz\n";
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const Vec3& point = positions[i];
    const Vec3& normal = normals[i];
    out << point.x << ',' << point.y << ',' << point.z << ",0,0,0," << normal.x << ','
        << normal.y << ',' << normal.z << '\n';
  }
}

CaseConfig parse_args(int argc, char** argv) {
  CaseConfig config;
  if (argc > 2) {
    throw std::invalid_argument("Usage: generate_hydrostatic_box [output_directory]");
  }
  if (argc == 2) {
    config.output_directory = argv[1];
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CaseConfig config = parse_args(argc, argv);
    const HydrostaticBox box = make_hydrostatic_box(config);

    std::filesystem::create_directories(config.output_directory);
    const std::filesystem::path fluid_path = config.output_directory / "fluid_particles.csv";
    const std::filesystem::path wall_path = config.output_directory / "wall_particles.csv";
    write_fluid_particles(fluid_path, box.fluid_positions);
    write_wall_particles(wall_path, box.wall_positions, box.wall_normals);

    std::cout << "Generated hydrostatic half-full box input\n"
              << "  box size: " << config.box_size << " m\n"
              << "  liquid height: " << config.liquid_height << " m\n"
              << "  spacing: " << config.spacing << " m\n"
              << "  fluid particles: " << box.fluid_positions.size() << '\n'
              << "  wall particles: " << box.wall_positions.size() << '\n'
              << "  fluid CSV: " << fluid_path << '\n'
              << "  wall CSV: " << wall_path << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "generate_hydrostatic_box failed: " << error.what() << std::endl;
    return 1;
  }
}
