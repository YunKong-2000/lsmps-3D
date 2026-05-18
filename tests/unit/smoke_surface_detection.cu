#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"
#include "lsmps3d/surface/surface_detection.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace {

template <typename T, std::size_t N>
void copy_to_device(T* dst, const std::array<T, N>& src) {
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

bool expect_surface_types(const std::vector<int>& actual, const std::vector<int>& expected) {
  if (actual == expected) {
    return true;
  }

  std::cerr << "surface type mismatch\nactual: ";
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

}  // namespace

int main() {
  constexpr lsmps3d::size_type kFluidCount = 8;
  constexpr lsmps3d::size_type kWallCount = 10;

  const lsmps3d::WorkspaceSpec spec{
      kFluidCount,
      kWallCount,
      6,
      4,
      64,
  };
  lsmps3d::SimulationWorkspace workspace(spec);
  auto view = workspace.view();

  const std::array<lsmps3d::real, kFluidCount> fluid_x{
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.6),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(3.5),
      static_cast<lsmps3d::real>(2.5),
  };
  const std::array<lsmps3d::real, kFluidCount> fluid_y{
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.6),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(3.5),
      static_cast<lsmps3d::real>(2.5),
  };
  const std::array<lsmps3d::real, kFluidCount> fluid_z{
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.6),
      static_cast<lsmps3d::real>(3.5),
      static_cast<lsmps3d::real>(1.0),
  };
  const std::array<lsmps3d::real, kWallCount> wall_x{
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.9),
      static_cast<lsmps3d::real>(3.1),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(2.5),
  };
  const std::array<lsmps3d::real, kWallCount> wall_y{
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(1.9),
      static_cast<lsmps3d::real>(3.1),
      static_cast<lsmps3d::real>(2.5),
      static_cast<lsmps3d::real>(2.5),
  };
  const std::array<lsmps3d::real, kWallCount> wall_z{
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.6),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(1.0),
      static_cast<lsmps3d::real>(0.4),
      static_cast<lsmps3d::real>(1.6),
  };

  copy_to_device(view.fluid.x, fluid_x);
  copy_to_device(view.fluid.y, fluid_y);
  copy_to_device(view.fluid.z, fluid_z);
  copy_to_device(view.walls.x, wall_x);
  copy_to_device(view.walls.y, wall_y);
  copy_to_device(view.walls.z, wall_z);

  const lsmps3d::NeighborSearchConfig neighbor_config{
      lsmps3d::CellGrid{
          lsmps3d::Vec3{static_cast<lsmps3d::real>(0.0),
                        static_cast<lsmps3d::real>(0.0),
                        static_cast<lsmps3d::real>(0.0)},
          static_cast<lsmps3d::real>(1.0),
          lsmps3d::Int3{4, 4, 4},
      },
      static_cast<lsmps3d::real>(0.75),
  };
  lsmps3d::build_neighbor_lists(view.fluid,
                                view.walls,
                                neighbor_config,
                                view.fluid_cells,
                                view.wall_cells,
                                view.fluid_neighbors,
                                view.wall_neighbors);

  lsmps3d::DeviceFluidParticles diagnostics(kFluidCount);
  lsmps3d::DeviceNeighborList diagnostic_counts(kFluidCount, kFluidCount);
  auto diagnostic_fluid = diagnostics.view();
  auto diagnostic_neighbors = diagnostic_counts.view();

  const lsmps3d::SurfaceDetectionConfig surface_config{
      static_cast<lsmps3d::real>(0.75),
      static_cast<lsmps3d::real>(0.75),
      static_cast<lsmps3d::real>(0.6),
      static_cast<lsmps3d::real>(0.0),
      2,
      static_cast<lsmps3d::real>(0.75),
      static_cast<lsmps3d::real>(0.10),
      static_cast<lsmps3d::real>(0.12),
      true,
      static_cast<lsmps3d::real>(0.25),
  };
  const lsmps3d::real reference_number_density =
      lsmps3d::compute_uniform_reference_number_density(surface_config.particle_spacing,
                                                        surface_config.support_radius);
  if (reference_number_density <= static_cast<lsmps3d::real>(0)) {
    std::cerr << "invalid reference number density" << std::endl;
    return 1;
  }
  const lsmps3d::SurfaceDetectionDiagnosticsView surface_diagnostics{
      diagnostic_neighbors.offsets,
      diagnostic_neighbors.indices,
      diagnostic_fluid.x,
      diagnostic_fluid.y,
      diagnostic_fluid.z,
      diagnostic_fluid.vx,
      diagnostic_fluid.vy,
      diagnostic_fluid.vz,
      diagnostic_fluid.pressure,
      nullptr,
  };
  lsmps3d::classify_surface_particles(view.fluid,
                                      view.walls,
                                      view.fluid_neighbors,
                                      view.wall_neighbors,
                                      surface_config,
                                      surface_diagnostics);

  std::vector<int> surface_types(kFluidCount);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(surface_types.data(),
                                view.fluid.surface_type,
                                surface_types.size() * sizeof(int),
                                cudaMemcpyDeviceToHost));

  const std::vector<int> expected_key_surface_types{
      static_cast<int>(lsmps3d::SurfaceType::NearSurface),
      static_cast<int>(lsmps3d::SurfaceType::Surface),
      static_cast<int>(lsmps3d::SurfaceType::Splash),
      static_cast<int>(lsmps3d::SurfaceType::Inner),
  };
  const std::vector<int> actual_key_surface_types{
      surface_types[0],
      surface_types[1],
      surface_types[6],
      surface_types[7],
  };
  if (!expect_surface_types(actual_key_surface_types, expected_key_surface_types)) {
    return 1;
  }

  std::vector<lsmps3d::real> number_density(kFluidCount);
  std::vector<lsmps3d::real> number_density_ratio(kFluidCount);
  std::vector<lsmps3d::real> anisotropy(kFluidCount);
  std::vector<lsmps3d::real> air_open_ratio(kFluidCount);
  std::vector<lsmps3d::real> air_anisotropy(kFluidCount);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(number_density.data(),
                                diagnostic_fluid.x,
                                number_density.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(number_density_ratio.data(),
                                diagnostic_fluid.y,
                                number_density_ratio.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(anisotropy.data(),
                                diagnostic_fluid.z,
                                anisotropy.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(air_open_ratio.data(),
                                diagnostic_fluid.vx,
                                air_open_ratio.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(air_anisotropy.data(),
                                diagnostic_fluid.vy,
                                air_anisotropy.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  const lsmps3d::real expected_ratio = number_density[1] / reference_number_density;
  if (std::abs(number_density_ratio[1] - expected_ratio) > static_cast<lsmps3d::real>(1.0e-5) ||
      number_density_ratio[0] <= number_density_ratio[1] || anisotropy[1] <= anisotropy[0] ||
      air_open_ratio[1] <= static_cast<lsmps3d::real>(0) ||
      air_anisotropy[1] <= static_cast<lsmps3d::real>(0)) {
    std::cerr << "surface diagnostics did not capture expected density ratio/anisotropy contrast"
              << std::endl;
    return 1;
  }

  lsmps3d::DeviceFluidParticles virtual_light_buffers(kFluidCount);
  auto virtual_light_view = virtual_light_buffers.view();
  const lsmps3d::VirtualLightConfig light_config{
      static_cast<lsmps3d::real>(0.75),
      static_cast<lsmps3d::real>(0.8660254),
      true,
  };
  const lsmps3d::VirtualLightDiagnosticsView light_diagnostics{
      virtual_light_view.surface_type,
      virtual_light_view.x,
  };
  lsmps3d::compute_virtual_light_diagnostics(view.fluid,
                                             view.walls,
                                             view.fluid_neighbors,
                                             view.wall_neighbors,
                                             light_config,
                                             light_diagnostics);

  std::vector<int> open_direction_count(kFluidCount);
  std::vector<lsmps3d::real> open_fraction(kFluidCount);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(open_direction_count.data(),
                                virtual_light_view.surface_type,
                                open_direction_count.size() * sizeof(int),
                                cudaMemcpyDeviceToHost));
  LSMPS3D_CUDA_CHECK(cudaMemcpy(open_fraction.data(),
                                virtual_light_view.x,
                                open_fraction.size() * sizeof(lsmps3d::real),
                                cudaMemcpyDeviceToHost));
  if (open_direction_count[1] <= open_direction_count[0] ||
      open_fraction[6] <= static_cast<lsmps3d::real>(0.99)) {
    std::cerr << "virtual light diagnostics did not detect expected open directions" << std::endl;
    return 1;
  }

  return 0;
}
