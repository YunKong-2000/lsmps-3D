#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"

namespace {

bool expect_offsets(const std::vector<lsmps3d::index_t>& actual,
                    const std::vector<lsmps3d::index_t>& expected,
                    const char* label) {
  if (actual == expected) {
    return true;
  }

  std::cerr << label << " offsets mismatch\nactual: ";
  for (const auto value : actual) {
    std::cerr << value << ' ';
  }
  std::cerr << "\nexpected: ";
  for (const auto value : expected) {
    std::cerr << value << ' ';
  }
  std::cerr << std::endl;
  return false;
}

bool expect_neighbor_rows(const std::vector<lsmps3d::index_t>& offsets,
                          const std::vector<lsmps3d::index_t>& indices,
                          const std::vector<std::vector<lsmps3d::index_t>>& expected,
                          const char* label) {
  for (std::size_t row = 0; row < expected.size(); ++row) {
    std::vector<lsmps3d::index_t> actual(indices.begin() + offsets[row],
                                         indices.begin() + offsets[row + 1]);
    std::sort(actual.begin(), actual.end());
    auto expected_row = expected[row];
    std::sort(expected_row.begin(), expected_row.end());
    if (actual != expected_row) {
      std::cerr << label << " row " << row << " mismatch\nactual: ";
      for (const auto value : actual) {
        std::cerr << value << ' ';
      }
      std::cerr << "\nexpected: ";
      for (const auto value : expected_row) {
        std::cerr << value << ' ';
      }
      std::cerr << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr lsmps3d::size_type kFluidCount = 5;
  constexpr lsmps3d::size_type kWallCount = 3;
  constexpr lsmps3d::size_type kFluidNeighborCapacity = 16;
  constexpr lsmps3d::size_type kWallNeighborCapacity = 12;

  const lsmps3d::WorkspaceSpec spec{
      kFluidCount,
      kWallCount,
      4,
      3,
      27,
  };
  lsmps3d::SimulationWorkspace workspace(spec);
  auto view = workspace.view();

  const std::array<lsmps3d::real, kFluidCount> fluid_x{
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.1),
      static_cast<lsmps3d::real>(1.4),
      static_cast<lsmps3d::real>(2.6),
  };
  const std::array<lsmps3d::real, kFluidCount> fluid_y{
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.7),
      static_cast<lsmps3d::real>(2.6),
  };
  const std::array<lsmps3d::real, kFluidCount> fluid_z{
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(2.6),
  };
  const std::array<lsmps3d::real, kWallCount> wall_x{
      static_cast<lsmps3d::real>(0.2),
      static_cast<lsmps3d::real>(1.2),
      static_cast<lsmps3d::real>(2.5),
  };
  const std::array<lsmps3d::real, kWallCount> wall_y{
      static_cast<lsmps3d::real>(0.5),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(2.6),
  };
  const std::array<lsmps3d::real, kWallCount> wall_z{
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(0.1),
      static_cast<lsmps3d::real>(2.6),
  };

  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.fluid.x, fluid_x.data(), fluid_x.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.fluid.y, fluid_y.data(), fluid_y.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.fluid.z, fluid_z.data(), fluid_z.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.walls.x, wall_x.data(), wall_x.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.walls.y, wall_y.data(), wall_y.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(
      view.walls.z, wall_z.data(), wall_z.size() * sizeof(lsmps3d::real), cudaMemcpyHostToDevice));

  const lsmps3d::NeighborSearchConfig config{
      lsmps3d::CellGrid{
          lsmps3d::Vec3{static_cast<lsmps3d::real>(0.0),
                        static_cast<lsmps3d::real>(0.0),
                        static_cast<lsmps3d::real>(0.0)},
          static_cast<lsmps3d::real>(1.0),
          lsmps3d::Int3{3, 3, 3},
      },
      static_cast<lsmps3d::real>(0.75),
  };

  lsmps3d::build_neighbor_lists(view.fluid,
                                view.walls,
                                config,
                                view.fluid_cells,
                                view.wall_cells,
                                view.fluid_neighbors,
                                view.wall_neighbors);

  std::vector<lsmps3d::index_t> fluid_offsets(kFluidCount + 1);
  std::vector<lsmps3d::index_t> wall_offsets(kFluidCount + 1);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(fluid_offsets.data(),
                                view.fluid_neighbors.offsets,
                                fluid_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(wall_offsets.data(),
                                view.wall_neighbors.offsets,
                                wall_offsets.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));

  const std::vector<lsmps3d::index_t> expected_fluid_offsets{0, 1, 3, 5, 6, 6};
  const std::vector<lsmps3d::index_t> expected_wall_offsets{0, 1, 2, 3, 4, 5};
  if (!expect_offsets(fluid_offsets, expected_fluid_offsets, "fluid") ||
      !expect_offsets(wall_offsets, expected_wall_offsets, "wall")) {
    return 1;
  }

  std::vector<lsmps3d::index_t> fluid_indices(kFluidNeighborCapacity);
  std::vector<lsmps3d::index_t> wall_indices(kWallNeighborCapacity);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(fluid_indices.data(),
                                view.fluid_neighbors.indices,
                                fluid_indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(wall_indices.data(),
                                view.wall_neighbors.indices,
                                wall_indices.size() * sizeof(lsmps3d::index_t),
                                cudaMemcpyDeviceToHost));

  const std::vector<std::vector<lsmps3d::index_t>> expected_fluid_rows{
      {1},
      {0, 2},
      {1, 3},
      {2},
      {},
  };
  const std::vector<std::vector<lsmps3d::index_t>> expected_wall_rows{
      {0},
      {0},
      {1},
      {1},
      {2},
  };
  if (!expect_neighbor_rows(fluid_offsets, fluid_indices, expected_fluid_rows, "fluid") ||
      !expect_neighbor_rows(wall_offsets, wall_indices, expected_wall_rows, "wall")) {
    return 1;
  }

  return 0;
}
