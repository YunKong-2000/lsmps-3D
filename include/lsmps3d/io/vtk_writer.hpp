#pragma once

#include "lsmps3d/core/types.cuh"

#include <filesystem>
#include <string>
#include <vector>

namespace lsmps3d {

struct VtkWriterConfig {
  std::filesystem::path output_directory{"output"};
  std::string file_prefix{"lsmps3d"};
  bool write_point_fields{true};
};

struct HostParticleSnapshot {
  std::vector<real> x;
  std::vector<real> y;
  std::vector<real> z;
};

struct HostRealScalarField {
  std::string name;
  std::vector<real> values;
};

struct HostIntScalarField {
  std::string name;
  std::vector<int> values;
};

struct HostVectorField {
  std::string name;
  std::vector<real> x;
  std::vector<real> y;
  std::vector<real> z;
};

struct HostVtkPointFields {
  std::vector<HostRealScalarField> real_scalars;
  std::vector<HostIntScalarField> int_scalars;
  std::vector<HostVectorField> vectors;

  void add_real_scalar(std::string name, std::vector<real> values);
  void add_int_scalar(std::string name, std::vector<int> values);
  void add_scalar(std::string name, std::vector<real> values);
  void add_scalar(std::string name, std::vector<int> values);
  void add_vector(std::string name,
                  std::vector<real> x_values,
                  std::vector<real> y_values,
                  std::vector<real> z_values);
};

class LegacyVtkWriter {
 public:
  explicit LegacyVtkWriter(VtkWriterConfig config = {});

  [[nodiscard]] std::filesystem::path make_path(size_type step) const;

  void write(size_type step,
             const HostParticleSnapshot& particles,
             const HostVtkPointFields& point_fields = {}) const;

 private:
  VtkWriterConfig config_{};
};

}  // namespace lsmps3d
