#pragma once

#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct CellGrid {
  Vec3 origin{};
  real cell_size{};
  Int3 dims{};

  [[nodiscard]] __host__ __device__ index_t cell_count() const {
    return dims.x * dims.y * dims.z;
  }
};

struct CellListView {
  size_type particle_count{};
  size_type cell_count{};
  index_t* cell_ids{};
  index_t* sorted_indices{};
  index_t* cell_begin{};
  index_t* cell_end{};
};

class DeviceCellList {
 public:
  DeviceCellList() = default;
  DeviceCellList(size_type particle_count, size_type cell_count);
  ~DeviceCellList();

  DeviceCellList(const DeviceCellList&) = delete;
  DeviceCellList& operator=(const DeviceCellList&) = delete;

  DeviceCellList(DeviceCellList&& other) noexcept;
  DeviceCellList& operator=(DeviceCellList&& other) noexcept;

  void resize(size_type particle_count, size_type cell_count);
  void release() noexcept;

  [[nodiscard]] size_type particle_capacity() const noexcept {
    return view_.particle_count;
  }

  [[nodiscard]] size_type cell_capacity() const noexcept {
    return view_.cell_count;
  }

  [[nodiscard]] size_type bytes() const noexcept;

  [[nodiscard]] CellListView view() noexcept {
    return view_;
  }

  [[nodiscard]] const CellListView& view() const noexcept {
    return view_;
  }

 private:
  CellListView view_{};
};

}  // namespace lsmps3d
