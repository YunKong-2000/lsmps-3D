#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/io/vtk_writer.hpp"

#include <filesystem>
#include <vector>

namespace lsmps3d {

struct HostFluidParticles {
  std::vector<real> x;
  std::vector<real> y;
  std::vector<real> z;
  std::vector<real> vx;
  std::vector<real> vy;
  std::vector<real> vz;
  std::vector<real> pressure;
  std::vector<int> surface_type;

  [[nodiscard]] size_type count() const noexcept {
    return x.size();
  }

  [[nodiscard]] HostParticleSnapshot snapshot() const {
    return HostParticleSnapshot{x, y, z};
  }
};

struct HostWallParticles {
  std::vector<real> x;
  std::vector<real> y;
  std::vector<real> z;
  std::vector<real> normal_x;
  std::vector<real> normal_y;
  std::vector<real> normal_z;
  std::vector<real> vx;
  std::vector<real> vy;
  std::vector<real> vz;

  [[nodiscard]] size_type count() const noexcept {
    return x.size();
  }

  [[nodiscard]] HostParticleSnapshot snapshot() const {
    return HostParticleSnapshot{x, y, z};
  }
};

struct ParticleInputData {
  HostFluidParticles fluid;
  HostWallParticles walls;
};

class FileManager {
 public:
  explicit FileManager(SimulationConfig config = {});

  [[nodiscard]] const SimulationConfig& config() const noexcept {
    return config_;
  }

  void set_config(SimulationConfig config);

  [[nodiscard]] SimulationConfig load_config(const std::filesystem::path& path) const;
  void save_config(const SimulationConfig& config, const std::filesystem::path& path) const;

  [[nodiscard]] HostFluidParticles load_fluid_particles(const std::filesystem::path& path) const;
  [[nodiscard]] HostWallParticles load_wall_particles(const std::filesystem::path& path) const;
  [[nodiscard]] ParticleInputData load_particle_input() const;
  [[nodiscard]] ParticleInputData load_particle_input(const std::filesystem::path& fluid_path,
                                                      const std::filesystem::path& wall_path) const;

  void write_fluid_result(size_type step,
                          const HostFluidParticles& particles,
                          const HostVtkPointFields& point_fields = {}) const;
  void write_wall_result(size_type step,
                         const HostWallParticles& particles,
                         const HostVtkPointFields& point_fields = {}) const;

 private:
  SimulationConfig config_{};
};

}  // namespace lsmps3d
