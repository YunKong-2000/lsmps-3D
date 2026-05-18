#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct FluidParticleSoA {
  size_type count{};
  real* x{};
  real* y{};
  real* z{};
  real* vx{};
  real* vy{};
  real* vz{};
  real* pressure{};
  int* surface_type{};
};

struct WallParticleSoA {
  size_type count{};
  real* x{};
  real* y{};
  real* z{};
  real* vx{};
  real* vy{};
  real* vz{};
  real* normal_x{};
  real* normal_y{};
  real* normal_z{};
};

class DeviceFluidParticles {
 public:
  DeviceFluidParticles() = default;
  explicit DeviceFluidParticles(size_type count);
  ~DeviceFluidParticles();

  DeviceFluidParticles(const DeviceFluidParticles&) = delete;
  DeviceFluidParticles& operator=(const DeviceFluidParticles&) = delete;

  DeviceFluidParticles(DeviceFluidParticles&& other) noexcept;
  DeviceFluidParticles& operator=(DeviceFluidParticles&& other) noexcept;

  void resize(size_type count);
  void release() noexcept;

  [[nodiscard]] size_type count() const noexcept {
    return view_.count;
  }

  [[nodiscard]] size_type capacity() const noexcept {
    return view_.count;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] FluidParticleSoA view() noexcept {
    return view_;
  }

  [[nodiscard]] const FluidParticleSoA& view() const noexcept {
    return view_;
  }

 private:
  FluidParticleSoA view_{};
};

class DeviceWallParticles {
 public:
  DeviceWallParticles() = default;
  explicit DeviceWallParticles(size_type count);
  ~DeviceWallParticles();

  DeviceWallParticles(const DeviceWallParticles&) = delete;
  DeviceWallParticles& operator=(const DeviceWallParticles&) = delete;

  DeviceWallParticles(DeviceWallParticles&& other) noexcept;
  DeviceWallParticles& operator=(DeviceWallParticles&& other) noexcept;

  void resize(size_type count);
  void release() noexcept;

  [[nodiscard]] size_type count() const noexcept {
    return view_.count;
  }

  [[nodiscard]] size_type capacity() const noexcept {
    return view_.count;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] WallParticleSoA view() noexcept {
    return view_;
  }

  [[nodiscard]] const WallParticleSoA& view() const noexcept {
    return view_;
  }

 private:
  WallParticleSoA view_{};
};

}  // namespace lsmps3d
