#include "lsmps3d/io/vtk_writer.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace lsmps3d {
namespace {

template <typename T>
bool has_count(const std::vector<T>& values, size_type count) noexcept {
  return values.size() == count;
}

size_type particle_count(const HostParticleSnapshot& particles) {
  const size_type count = particles.x.size();
  if (!has_count(particles.y, count) || !has_count(particles.z, count)) {
    throw std::invalid_argument("HostParticleSnapshot fields must have matching lengths.");
  }
  return count;
}

void require_field_name(const std::string& name) {
  if (name.empty()) {
    throw std::invalid_argument("VTK point field name must not be empty.");
  }
}

template <typename T>
void write_scalar_field(std::ofstream& out,
                        const std::string& name,
                        const std::vector<T>& values,
                        const std::string& vtk_type) {
  out << "SCALARS " << name << ' ' << vtk_type << " 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto value : values) {
    out << value << '\n';
  }
}

void write_vector_field(std::ofstream& out, const HostVectorField& field, size_type count) {
  out << "VECTORS " << field.name << " float\n";
  for (size_type i = 0; i < count; ++i) {
    out << field.x[i] << ' ' << field.y[i] << ' ' << field.z[i] << '\n';
  }
}

}  // namespace

void HostVtkPointFields::add_real_scalar(std::string name, std::vector<real> values) {
  real_scalars.push_back(HostRealScalarField{std::move(name), std::move(values)});
}

void HostVtkPointFields::add_int_scalar(std::string name, std::vector<int> values) {
  int_scalars.push_back(HostIntScalarField{std::move(name), std::move(values)});
}

void HostVtkPointFields::add_scalar(std::string name, std::vector<real> values) {
  add_real_scalar(std::move(name), std::move(values));
}

void HostVtkPointFields::add_scalar(std::string name, std::vector<int> values) {
  add_int_scalar(std::move(name), std::move(values));
}

void HostVtkPointFields::add_vector(std::string name,
                                    std::vector<real> x_values,
                                    std::vector<real> y_values,
                                    std::vector<real> z_values) {
  vectors.push_back(HostVectorField{
      std::move(name),
      std::move(x_values),
      std::move(y_values),
      std::move(z_values),
  });
}

LegacyVtkWriter::LegacyVtkWriter(VtkWriterConfig config) : config_(std::move(config)) {}

std::filesystem::path LegacyVtkWriter::make_path(size_type step) const {
  std::ostringstream name;
  name << config_.file_prefix << '_' << std::setw(6) << std::setfill('0') << step << ".vtk";
  return config_.output_directory / name.str();
}

void LegacyVtkWriter::write(size_type step,
                            const HostParticleSnapshot& particles,
                            const HostVtkPointFields& point_fields) const {
  const size_type count = particle_count(particles);
  for (const auto& scalar : point_fields.real_scalars) {
    require_field_name(scalar.name);
    if (!has_count(scalar.values, count)) {
      throw std::invalid_argument("Real scalar field '" + scalar.name +
                                  "' length must match particle count.");
    }
  }
  for (const auto& scalar : point_fields.int_scalars) {
    require_field_name(scalar.name);
    if (!has_count(scalar.values, count)) {
      throw std::invalid_argument("Integer scalar field '" + scalar.name +
                                  "' length must match particle count.");
    }
  }
  for (const auto& vector : point_fields.vectors) {
    require_field_name(vector.name);
    if (!has_count(vector.x, count) || !has_count(vector.y, count) || !has_count(vector.z, count)) {
      throw std::invalid_argument("Vector field '" + vector.name +
                                  "' components must match particle count.");
    }
  }

  std::filesystem::create_directories(config_.output_directory);
  const auto path = make_path(step);
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open VTK output file: " + path.string());
  }

  out << std::setprecision(9);
  out << "# vtk DataFile Version 3.0\n";
  out << "lsmps3d particle diagnostics\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << count << " float\n";
  for (size_type i = 0; i < count; ++i) {
    out << particles.x[i] << ' ' << particles.y[i] << ' ' << particles.z[i] << '\n';
  }

  out << "VERTICES " << count << ' ' << count * 2 << '\n';
  for (size_type i = 0; i < count; ++i) {
    out << "1 " << i << '\n';
  }

  out << "POINT_DATA " << count << '\n';
  if (config_.write_point_fields) {
    for (const auto& vector : point_fields.vectors) {
      write_vector_field(out, vector, count);
    }
    for (const auto& scalar : point_fields.real_scalars) {
      write_scalar_field(out, scalar.name, scalar.values, "float");
    }
    for (const auto& scalar : point_fields.int_scalars) {
      write_scalar_field(out, scalar.name, scalar.values, "int");
    }
  }
}

}  // namespace lsmps3d
