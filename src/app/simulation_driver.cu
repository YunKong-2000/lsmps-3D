#include "lsmps3d/app/simulation_driver.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/core/time_step.hpp"
#include "lsmps3d/core/types.cuh"
#include "lsmps3d/core/workspace.cuh"
#include "lsmps3d/correction/pressure_correction.cuh"
#include "lsmps3d/io/file_manager.hpp"
#include "lsmps3d/moment_matrix/moment_matrix.cuh"
#include "lsmps3d/neighbor/neighbor_search.cuh"
#include "lsmps3d/ppe/ppe_matrix.cuh"
#include "lsmps3d/provision/explicit_update.cuh"
#include "lsmps3d/surface/surface_detection.cuh"

namespace lsmps3d {
namespace {

constexpr size_type kMaxFluidNeighborsPerParticle = 256;
constexpr size_type kMaxWallNeighborsPerParticle = 256;
constexpr int kThreadsPerBlock = 128;

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_from_device(const T* src, size_type count) {
  std::vector<T> dst(static_cast<std::size_t>(count));
  if (count == 0) {
    return dst;
  }
  LSMPS3D_CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return dst;
}

void include_bounds(const std::vector<real>& x,
                    const std::vector<real>& y,
                    const std::vector<real>& z,
                    Vec3& min_corner,
                    Vec3& max_corner,
                    bool& has_particles) {
  if (x.size() != y.size() || x.size() != z.size()) {
    throw std::invalid_argument("Particle coordinate arrays must have matching lengths");
  }

  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(static_cast<double>(x[i])) || !std::isfinite(static_cast<double>(y[i])) ||
        !std::isfinite(static_cast<double>(z[i]))) {
      throw std::invalid_argument("Particle coordinates must be finite");
    }

    if (!has_particles) {
      min_corner = Vec3{x[i], y[i], z[i]};
      max_corner = min_corner;
      has_particles = true;
      continue;
    }

    min_corner.x = std::min(min_corner.x, x[i]);
    min_corner.y = std::min(min_corner.y, y[i]);
    min_corner.z = std::min(min_corner.z, z[i]);
    max_corner.x = std::max(max_corner.x, x[i]);
    max_corner.y = std::max(max_corner.y, y[i]);
    max_corner.z = std::max(max_corner.z, z[i]);
  }
}

index_t cell_dim_for_axis(real max_value, real origin, real cell_size) {
  const real span = max_value - origin;
  const auto dim = static_cast<long long>(std::ceil(static_cast<double>(span / cell_size))) + 1;
  if (dim <= 0 || dim > static_cast<long long>(std::numeric_limits<index_t>::max())) {
    throw std::invalid_argument("Derived cell-grid dimension is outside supported index range");
  }
  return static_cast<index_t>(dim);
}

void derive_cell_grid_from_particles(SimulationConfig& config, const ParticleInputData& input) {
  Vec3 min_corner{};
  Vec3 max_corner{};
  bool has_particles = false;
  include_bounds(input.fluid.x, input.fluid.y, input.fluid.z, min_corner, max_corner, has_particles);
  include_bounds(input.walls.x, input.walls.y, input.walls.z, min_corner, max_corner, has_particles);
  if (!has_particles) {
    throw std::invalid_argument("Cannot derive cell grid from empty particle input");
  }

  const real padding = config.support_radius;
  config.cell_origin = Vec3{min_corner.x - padding, min_corner.y - padding, min_corner.z - padding};
  config.cell_dims = Int3{
      cell_dim_for_axis(max_corner.x + padding, config.cell_origin.x, config.cell_size),
      cell_dim_for_axis(max_corner.y + padding, config.cell_origin.y, config.cell_size),
      cell_dim_for_axis(max_corner.z + padding, config.cell_origin.z, config.cell_size)};
}

__global__ void velocity_magnitude_kernel(FluidParticleSoA fluid, real* magnitudes) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const real vx = fluid.vx[i];
  const real vy = fluid.vy[i];
  const real vz = fluid.vz[i];
  magnitudes[i] = sqrt(vx * vx + vy * vy + vz * vz);
}

real max_velocity(const FluidParticleSoA& fluid, DeviceFluidParticles& diagnostics) {
  if (fluid.count == 0) {
    return static_cast<real>(0);
  }
  if (diagnostics.count() < fluid.count) {
    diagnostics.resize(fluid.count);
  }

  auto diag = diagnostics.view();
  const int blocks = static_cast<int>((fluid.count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  velocity_magnitude_kernel<<<blocks, kThreadsPerBlock>>>(fluid, diag.x);
  LSMPS3D_CUDA_KERNEL_CHECK();

  const auto magnitudes = copy_from_device(diag.x, fluid.count);
  real result = static_cast<real>(0);
  for (const auto value : magnitudes) {
    if (std::isfinite(static_cast<double>(value))) {
      result = std::max(result, value);
    }
  }
  return result;
}

void copy_input_to_workspace(const ParticleInputData& input, SimulationWorkspaceView view) {
  if (input.fluid.count() != view.fluid.count || input.walls.count() != view.walls.count) {
    throw std::invalid_argument("Particle input count does not match simulation workspace capacity");
  }

  copy_to_device(view.fluid.x, input.fluid.x);
  copy_to_device(view.fluid.y, input.fluid.y);
  copy_to_device(view.fluid.z, input.fluid.z);
  copy_to_device(view.fluid.vx, input.fluid.vx);
  copy_to_device(view.fluid.vy, input.fluid.vy);
  copy_to_device(view.fluid.vz, input.fluid.vz);
  copy_to_device(view.fluid.pressure, input.fluid.pressure);
  copy_to_device(view.fluid.surface_type, input.fluid.surface_type);

  copy_to_device(view.walls.x, input.walls.x);
  copy_to_device(view.walls.y, input.walls.y);
  copy_to_device(view.walls.z, input.walls.z);
  copy_to_device(view.walls.vx, input.walls.vx);
  copy_to_device(view.walls.vy, input.walls.vy);
  copy_to_device(view.walls.vz, input.walls.vz);
  copy_to_device(view.walls.normal_x, input.walls.normal_x);
  copy_to_device(view.walls.normal_y, input.walls.normal_y);
  copy_to_device(view.walls.normal_z, input.walls.normal_z);
}

HostFluidParticles copy_fluid_to_host(const FluidParticleSoA& fluid) {
  HostFluidParticles host;
  host.x = copy_from_device(fluid.x, fluid.count);
  host.y = copy_from_device(fluid.y, fluid.count);
  host.z = copy_from_device(fluid.z, fluid.count);
  host.vx = copy_from_device(fluid.vx, fluid.count);
  host.vy = copy_from_device(fluid.vy, fluid.count);
  host.vz = copy_from_device(fluid.vz, fluid.count);
  host.pressure = copy_from_device(fluid.pressure, fluid.count);
  host.surface_type = copy_from_device(fluid.surface_type, fluid.count);
  return host;
}

HostWallParticles copy_walls_to_host(const WallParticleSoA& walls) {
  HostWallParticles host;
  host.x = copy_from_device(walls.x, walls.count);
  host.y = copy_from_device(walls.y, walls.count);
  host.z = copy_from_device(walls.z, walls.count);
  host.vx = copy_from_device(walls.vx, walls.count);
  host.vy = copy_from_device(walls.vy, walls.count);
  host.vz = copy_from_device(walls.vz, walls.count);
  host.normal_x = copy_from_device(walls.normal_x, walls.count);
  host.normal_y = copy_from_device(walls.normal_y, walls.count);
  host.normal_z = copy_from_device(walls.normal_z, walls.count);
  return host;
}

HostVtkPointFields make_fluid_fields(const HostFluidParticles& fluid,
                                     DevicePressureCorrection& correction,
                                     bool include_correction_fields) {
  HostVtkPointFields fields;
  fields.add_vector("velocity", fluid.vx, fluid.vy, fluid.vz);
  fields.add_real_scalar("pressure", fluid.pressure);
  fields.add_int_scalar("surface_type", fluid.surface_type);

  if (!include_correction_fields) {
    return fields;
  }

  const auto pressure_gradient = correction.pressure_gradient();
  if (pressure_gradient.count >= fluid.count()) {
    fields.add_vector("pressure_gradient",
                      copy_from_device(pressure_gradient.x, fluid.count()),
                      copy_from_device(pressure_gradient.y, fluid.count()),
                      copy_from_device(pressure_gradient.z, fluid.count()));
  }

  const auto ps_displacement = correction.ps_displacement();
  if (ps_displacement.count >= fluid.count()) {
    fields.add_vector("ps_displacement",
                      copy_from_device(ps_displacement.x, fluid.count()),
                      copy_from_device(ps_displacement.y, fluid.count()),
                      copy_from_device(ps_displacement.z, fluid.count()));
  }

  return fields;
}

HostVtkPointFields make_wall_fields(const HostWallParticles& walls) {
  HostVtkPointFields fields;
  fields.add_vector("velocity", walls.vx, walls.vy, walls.vz);
  fields.add_vector("normal", walls.normal_x, walls.normal_y, walls.normal_z);
  return fields;
}

void write_fluid_results(const FileManager& file_manager,
                         size_type output_step,
                         const SimulationWorkspaceView& view,
                         DevicePressureCorrection& correction,
                         bool include_correction_fields) {
  const HostFluidParticles fluid = copy_fluid_to_host(view.fluid);
  file_manager.write_fluid_result(
      output_step, fluid, make_fluid_fields(fluid, correction, include_correction_fields));
}

void write_wall_results(const FileManager& file_manager,
                        size_type output_step,
                        const SimulationWorkspaceView& view) {
  const HostWallParticles walls = copy_walls_to_host(view.walls);
  file_manager.write_wall_result(output_step, walls, make_wall_fields(walls));
}

void rebuild_geometry(SimulationWorkspaceView view,
                      const SimulationConfig& config,
                      unsigned long long& geometry_generation) {
  build_neighbor_lists(view.fluid,
                       view.walls,
                       config,
                       view.fluid_cells,
                       view.wall_cells,
                       view.fluid_neighbors,
                       view.wall_neighbors);
  ++geometry_generation;
  classify_surface_particles(view.fluid, view.walls, view.fluid_neighbors, view.wall_neighbors, config);
}

void print_startup_summary(const std::filesystem::path& config_path,
                           const SimulationConfig& config,
                           size_type fluid_count,
                           size_type wall_count,
                           const SimulationWorkspace& workspace) {
  std::cout << "LSMPS3D simulation\n"
            << "  config: " << config_path << '\n'
            << "  fluid particles: " << fluid_count << '\n'
            << "  wall particles: " << wall_count << '\n'
            << "  cell origin: (" << config.cell_origin.x << ", " << config.cell_origin.y << ", "
            << config.cell_origin.z << ")\n"
            << "  cell size: " << config.cell_size << '\n'
            << "  cell dims: (" << config.cell_dims.x << ", " << config.cell_dims.y << ", "
            << config.cell_dims.z << ")\n"
            << "  workspace bytes: " << workspace.bytes() << '\n'
            << "  max fluid neighbors/particle: " << kMaxFluidNeighborsPerParticle << '\n'
            << "  max wall neighbors/particle: " << kMaxWallNeighborsPerParticle << '\n'
            << "  AMGX pressure solve: "
            << (AmgxPpeSolver::is_available() ? "enabled" : "unavailable; pressure set to zero")
            << '\n'
            << "  output directory: " << config.output_directory << std::endl;
}

}  // namespace

void print_simulation_usage(std::ostream& output, const char* program_name) {
  output << "Usage: " << program_name << " [config/simulation.ini]\n";
}

int run_simulation(const std::filesystem::path& config_path) {
  SimulationConfig config = load_simulation_config(config_path);

  FileManager file_manager(config);
  const ParticleInputData input = file_manager.load_particle_input();
  const size_type fluid_count = input.fluid.count();
  const size_type wall_count = input.walls.count();
  derive_cell_grid_from_particles(config, input);

  const WorkspaceSpec spec{
      fluid_count,
      wall_count,
      kMaxFluidNeighborsPerParticle,
      kMaxWallNeighborsPerParticle,
      config.cell_capacity(),
  };
  SimulationWorkspace workspace(spec);
  SimulationWorkspaceView view = workspace.view();
  copy_input_to_workspace(input, view);

  DeviceFluidParticles temporary_velocity(fluid_count);
  DeviceWallParticles temporary_wall_velocity(wall_count);
  DeviceMomentMatrix moment_matrices(fluid_count, config);
  DeviceProvisionExplicitUpdate provision(fluid_count, config);
  DevicePpeMatrixAssembler ppe_assembler(
      fluid_count, fluid_count + spec.fluid_neighbor_capacity(), config);
  DevicePressureCorrection correction(fluid_count, config);
  DeviceFluidParticles velocity_diagnostics(fluid_count);
  SimulationTimeManager time_manager(config);

  print_startup_summary(config_path, config, fluid_count, wall_count, workspace);

  unsigned long long geometry_generation = 0;
  rebuild_geometry(view, config, geometry_generation);

  const auto initial_output = time_manager.mark_initial_output();
  if (initial_output.should_output) {
    write_fluid_results(file_manager, initial_output.step_index, view, correction, false);
    write_wall_results(file_manager, initial_output.step_index, view);
  }

  while (!time_manager.finished()) {
    const real step_start_max_speed = max_velocity(view.fluid, velocity_diagnostics);
    const auto status = time_manager.advance(step_start_max_speed);

    config.time_step = status.time_step;
    provision.set_config(config);
    ppe_assembler.set_config(config);
    correction.set_config(config);
    moment_matrices.set_config(config);

    provision.compute_temporary_velocity(view.fluid,
                                         view.walls,
                                         view.fluid_neighbors,
                                         view.wall_neighbors,
                                         moment_matrices,
                                         geometry_generation,
                                         temporary_velocity.view(),
                                         temporary_wall_velocity.view());

    ppe_assembler.assemble(view.fluid,
                           view.walls,
                           view.fluid_neighbors,
                           view.wall_neighbors,
                           temporary_velocity.view(),
                           temporary_wall_velocity.view(),
                           moment_matrices,
                           geometry_generation);

    auto ppe = ppe_assembler.workspace();
    if (AmgxPpeSolver::is_available()) {
      AmgxPpeSolver solver(config.amgx_config_path, config.amgx_print_solve_stats);
      solver.solve(ppe.matrix, ppe.rhs, ppe.pressure);
    } else if (fluid_count > 0) {
      LSMPS3D_CUDA_CHECK(cudaMemset(ppe.pressure, 0, fluid_count * sizeof(real)));
    }

    correction.apply(view.fluid,
                     view.walls,
                     view.fluid_neighbors,
                     view.wall_neighbors,
                     temporary_velocity.view(),
                     ppe.pressure,
                     moment_matrices,
                     geometry_generation);

    LSMPS3D_CUDA_CHECK(cudaDeviceSynchronize());

    rebuild_geometry(view, config, geometry_generation);

    if (status.should_output || status.reached_final_time) {
      write_fluid_results(file_manager, status.step_index, view, correction, true);
    }

    if (status.should_output || status.reached_final_time) {
      std::cout << "step " << status.step_index << " time " << status.current_time << " dt "
                << status.time_step << " max_velocity " << status.max_velocity << std::endl;
    }
  }

  std::cout << "Simulation complete at time " << time_manager.current_time() << " after "
            << time_manager.step_index() << " steps" << std::endl;
  return EXIT_SUCCESS;
}

}  // namespace lsmps3d
