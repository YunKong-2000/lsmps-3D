#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct NeighborListView {
  size_type particle_count{};
  size_type neighbor_count{};
  index_t* offsets{};
  index_t* indices{};
};

class DeviceNeighborList {
 public:
  DeviceNeighborList() = default;
  DeviceNeighborList(size_type particle_count, size_type neighbor_capacity);
  ~DeviceNeighborList();

  DeviceNeighborList(const DeviceNeighborList&) = delete;
  DeviceNeighborList& operator=(const DeviceNeighborList&) = delete;

  DeviceNeighborList(DeviceNeighborList&& other) noexcept;
  DeviceNeighborList& operator=(DeviceNeighborList&& other) noexcept;

  void resize(size_type particle_count, size_type neighbor_capacity);
  void release() noexcept;

  [[nodiscard]] size_type particle_capacity() const noexcept {
    return view_.particle_count;
  }

  [[nodiscard]] size_type neighbor_capacity() const noexcept {
    return capacity_;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] NeighborListView view() noexcept {
    return view_;
  }

  [[nodiscard]] const NeighborListView& view() const noexcept {
    return view_;
  }

 private:
  NeighborListView view_{};
  size_type capacity_{};
};

}  // namespace lsmps3d
