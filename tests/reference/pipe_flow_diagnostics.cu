#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/io/vtk_writer.hpp"
#include "lsmps3d/lsmps/moment_matrix.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"

namespace {

struct PipeFlowCase {
  std::vector<lsmps3d::real> fluid_x;
  std::vector<lsmps3d::real> fluid_y;
  std::vector<lsmps3d::real> fluid_z;
  std::vector<lsmps3d::real> velocity_x;
  std::vector<lsmps3d::real> analytic_grad_x;
  std::vector<lsmps3d::real> analytic_grad_y;
  std::vector<lsmps3d::real> analytic_grad_z;
  std::vector<lsmps3d::real> analytic_laplacian;
  std::vector<lsmps3d::real> wall_x;
  std::vector<lsmps3d::real> wall_y;
  std::vector<lsmps3d::real> wall_z;
  std::vector<lsmps3d::real> wall_normal_x;
  std::vector<lsmps3d::real> wall_normal_y;
  std::vector<lsmps3d::real> wall_normal_z;
  std::vector<lsmps3d::real> wall_velocity_x;
};

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src) {
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_from_device(const T* src, std::size_t count) {
  std::vector<T> dst(count);
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return dst;
}

std::vector<lsmps3d::real> magnitude(const std::vector<lsmps3d::real>& x,
                                     const std::vector<lsmps3d::real>& y,
                                     const std::vector<lsmps3d::real>& z) {
  std::vector<lsmps3d::real> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
  }
  return out;
}

std::vector<lsmps3d::real> absolute_error(const std::vector<lsmps3d::real>& actual,
                                          const std::vector<lsmps3d::real>& expected) {
  std::vector<lsmps3d::real> out(actual.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    out[i] = std::abs(actual[i] - expected[i]);
  }
  return out;
}

PipeFlowCase make_pipe_flow_case(lsmps3d::real spacing,
                                 lsmps3d::real length,
                                 lsmps3d::real radius,
                                 lsmps3d::real max_velocity) {
  PipeFlowCase result;
  const int nx = static_cast<int>(std::lround(length / spacing));
  const int radial_extent = static_cast<int>(std::ceil(radius / spacing));
  const lsmps3d::real radius_squared = radius * radius;
  const lsmps3d::real inv_radius_squared = static_cast<lsmps3d::real>(1) / radius_squared;

  for (int ix = 0; ix < nx; ++ix) {
    const lsmps3d::real x = (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int iy = -radial_extent; iy <= radial_extent; ++iy) {
      const lsmps3d::real y = static_cast<lsmps3d::real>(iy) * spacing;
      for (int iz = -radial_extent; iz <= radial_extent; ++iz) {
        const lsmps3d::real z = static_cast<lsmps3d::real>(iz) * spacing;
        const lsmps3d::real r_squared = y * y + z * z;
        if (r_squared > radius_squared) {
          continue;
        }

        result.fluid_x.push_back(x);
        result.fluid_y.push_back(y);
        result.fluid_z.push_back(z);
        result.velocity_x.push_back(max_velocity *
                                    (static_cast<lsmps3d::real>(1) - r_squared * inv_radius_squared));
        result.analytic_grad_x.push_back(static_cast<lsmps3d::real>(0));
        result.analytic_grad_y.push_back(-static_cast<lsmps3d::real>(2) * max_velocity * y *
                                         inv_radius_squared);
        result.analytic_grad_z.push_back(-static_cast<lsmps3d::real>(2) * max_velocity * z *
                                         inv_radius_squared);
        result.analytic_laplacian.push_back(-static_cast<lsmps3d::real>(4) * max_velocity *
                                            inv_radius_squared);
      }
    }
  }

  const int wall_segments = std::max(24, static_cast<int>(std::ceil(2 * M_PI * radius / spacing)));
  for (int ix = 0; ix < nx; ++ix) {
    const lsmps3d::real x = (static_cast<lsmps3d::real>(ix) + static_cast<lsmps3d::real>(0.5)) *
                            spacing;
    for (int segment = 0; segment < wall_segments; ++segment) {
      const lsmps3d::real theta = static_cast<lsmps3d::real>(2.0 * M_PI) *
                                  static_cast<lsmps3d::real>(segment) /
                                  static_cast<lsmps3d::real>(wall_segments);
      const lsmps3d::real normal_y = -std::cos(theta);
      const lsmps3d::real normal_z = -std::sin(theta);
      result.wall_x.push_back(x);
      result.wall_y.push_back(radius * std::cos(theta));
      result.wall_z.push_back(radius * std::sin(theta));
      result.wall_normal_x.push_back(static_cast<lsmps3d::real>(0));
      result.wall_normal_y.push_back(normal_y);
      result.wall_normal_z.push_back(normal_z);
      result.wall_velocity_x.push_back(static_cast<lsmps3d::real>(0));
    }
  }

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_dir =
      argc > 1 ? std::filesystem::path(argv[1])
               : std::filesystem::path("output/pipe_flow_diagnostics");
  std::filesystem::create_directories(output_dir);

  constexpr lsmps3d::real kSpacing = static_cast<lsmps3d::real>(0.025);
  constexpr lsmps3d::real kLength = static_cast<lsmps3d::real>(0.6);
  constexpr lsmps3d::real kRadius = static_cast<lsmps3d::real>(0.2);
  constexpr lsmps3d::real kMaxVelocity = static_cast<lsmps3d::real>(1.0);
  constexpr lsmps3d::real kSupportRadius = static_cast<lsmps3d::real>(3.1) * kSpacing;

  const PipeFlowCase pipe = make_pipe_flow_case(kSpacing, kLength, kRadius, kMaxVelocity);
  const lsmps3d::size_type fluid_count = pipe.fluid_x.size();
  const lsmps3d::size_type wall_count = pipe.wall_x.size();

  const lsmps3d::Vec3 domain_min{-kSupportRadius,
                                 -kRadius - kSupportRadius,
                                 -kRadius - kSupportRadius};
  const lsmps3d::Vec3 domain_max{kLength + kSupportRadius,
                                 kRadius + kSupportRadius,
                                 kRadius + kSupportRadius};
  const lsmps3d::Int3 grid_dims{
      static_cast<lsmps3d::index_t>(std::ceil((domain_max.x - domain_min.x) / kSupportRadius)),
      static_cast<lsmps3d::index_t>(std::ceil((domain_max.y - domain_min.y) / kSupportRadius)),
      static_cast<lsmps3d::index_t>(std::ceil((domain_max.z - domain_min.z) / kSupportRadius)),
  };
  const lsmps3d::size_type cell_count =
      static_cast<lsmps3d::size_type>(grid_dims.x) * grid_dims.y * grid_dims.z;

  const lsmps3d::WorkspaceSpec spec{
      fluid_count,
      wall_count,
      256,
      128,
      cell_count,
  };
  lsmps3d::SimulationWorkspace workspace(spec);
  auto view = workspace.view();

  copy_to_device(view.fluid.x, pipe.fluid_x);
  copy_to_device(view.fluid.y, pipe.fluid_y);
  copy_to_device(view.fluid.z, pipe.fluid_z);
  copy_to_device(view.fluid.vx, pipe.velocity_x);
  copy_to_device(view.walls.x, pipe.wall_x);
  copy_to_device(view.walls.y, pipe.wall_y);
  copy_to_device(view.walls.z, pipe.wall_z);
  copy_to_device(view.walls.normal_x, pipe.wall_normal_x);
  copy_to_device(view.walls.normal_y, pipe.wall_normal_y);
  copy_to_device(view.walls.normal_z, pipe.wall_normal_z);
  copy_to_device(view.walls.vx, pipe.wall_velocity_x);

  lsmps3d::SimulationConfig config;
  config.support_radius = kSupportRadius;
  config.cell_origin = domain_min;
  config.cell_size = kSupportRadius;
  config.cell_dims = grid_dims;
  config.density = static_cast<lsmps3d::real>(1);
  config.gravity = {};
  lsmps3d::build_neighbor_lists(view.fluid,
                                view.walls,
                                config,
                                view.fluid_cells,
                                view.wall_cells,
                                view.fluid_neighbors,
                                view.wall_neighbors);

  lsmps3d::DeviceFluidParticles operator_buffers(fluid_count);
  auto operator_view = operator_buffers.view();

  lsmps3d::DeviceLsmpsOperators lsmps(fluid_count, config);
  lsmps.prepare_matrices(view.fluid, view.walls, view.fluid_neighbors, view.wall_neighbors, 1);
  lsmps.compute_velocity_gradient(
      view.fluid,
      view.walls,
      view.fluid_neighbors,
      view.wall_neighbors,
      view.fluid.vx,
      view.walls.vx,
      operator_view.x,
      operator_view.y,
      operator_view.z);
  lsmps.compute_velocity_laplacian(view.fluid,
                                   view.walls,
                                   view.fluid_neighbors,
                                   view.wall_neighbors,
                                   view.fluid.vx,
                                   view.walls.vx,
                                   operator_view.pressure);

  const auto gradient_x = copy_from_device(operator_view.x, fluid_count);
  const auto gradient_y = copy_from_device(operator_view.y, fluid_count);
  const auto gradient_z = copy_from_device(operator_view.z, fluid_count);
  const auto laplacian = copy_from_device(operator_view.pressure, fluid_count);
  const auto gradient_magnitude = magnitude(gradient_x, gradient_y, gradient_z);
  const auto analytic_gradient_magnitude =
      magnitude(pipe.analytic_grad_x, pipe.analytic_grad_y, pipe.analytic_grad_z);
  const auto gradient_y_error = absolute_error(gradient_y, pipe.analytic_grad_y);
  const auto gradient_z_error = absolute_error(gradient_z, pipe.analytic_grad_z);
  const auto laplacian_error = absolute_error(laplacian, pipe.analytic_laplacian);

  lsmps3d::HostParticleSnapshot particles{
      pipe.fluid_x,
      pipe.fluid_y,
      pipe.fluid_z,
  };
  lsmps3d::HostVtkPointFields point_fields;
  point_fields.add_scalar("velocity_x", pipe.velocity_x);
  point_fields.add_scalar("velocity_gradient_magnitude", gradient_magnitude);
  point_fields.add_scalar("analytic_velocity_gradient_magnitude", analytic_gradient_magnitude);
  point_fields.add_scalar("velocity_laplacian", laplacian);
  point_fields.add_scalar("analytic_velocity_laplacian", pipe.analytic_laplacian);
  point_fields.add_scalar("velocity_gradient_y_error", gradient_y_error);
  point_fields.add_scalar("velocity_gradient_z_error", gradient_z_error);
  point_fields.add_scalar("velocity_laplacian_error", laplacian_error);
  point_fields.add_vector("velocity", pipe.velocity_x, std::vector<lsmps3d::real>(fluid_count), std::vector<lsmps3d::real>(fluid_count));
  point_fields.add_vector("velocity_gradient", gradient_x, gradient_y, gradient_z);
  point_fields.add_vector(
      "analytic_velocity_gradient", pipe.analytic_grad_x, pipe.analytic_grad_y, pipe.analytic_grad_z);

  config.output_directory = output_dir;
  config.vtk_file_prefix = "pipe_flow";
  config.vtk_write_point_fields = true;
  const lsmps3d::LegacyVtkWriter vtk_writer(config);
  vtk_writer.write(0, particles, point_fields);

  std::cout << "Pipe flow diagnostics\n"
            << "  fluid particles: " << fluid_count << '\n'
            << "  wall particles: " << wall_count << '\n'
            << "  radius: " << kRadius << " m\n"
            << "  spacing: " << kSpacing << " m\n"
            << "  support radius: " << kSupportRadius << " m\n"
            << "  analytic laplacian: " << -static_cast<lsmps3d::real>(4) * kMaxVelocity /
                                                (kRadius * kRadius)
            << '\n'
            << "  VTK: " << vtk_writer.make_path(0) << std::endl;

  return 0;
}
