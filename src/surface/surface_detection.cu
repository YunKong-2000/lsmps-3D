#include "lsmps3d/surface/surface_detection.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "lsmps3d/core/cuda_check.cuh"
#include "lsmps3d/surface/surface_type.cuh"

namespace lsmps3d {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kVirtualLightDirectionCount = 14;
constexpr int kMaxWallBasisDirections = 3;

[[nodiscard]] int block_count(size_type count) {
  return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

__device__ real safe_rsqrt(real value) {
  constexpr real kEpsilon = static_cast<real>(1.0e-20);
  return rsqrt(value > kEpsilon ? value : kEpsilon);
}

__host__ __device__ real particle_weight(real distance, real support_radius) {
  if (distance >= support_radius) {
    return static_cast<real>(0);
  }
  return static_cast<real>(1) - distance / support_radius;
}

__device__ void accumulate_neighbor_moments(real dx,
                                            real dy,
                                            real dz,
                                            real support_radius,
                                            real& number_density,
                                            real& sx,
                                            real& sy,
                                            real& sz,
                                            int& weighted_count) {
  const real distance_squared = dx * dx + dy * dy + dz * dz;
  if (distance_squared <= static_cast<real>(0)) {
    return;
  }

  const real distance = sqrt(distance_squared);
  const real weight = particle_weight(distance, support_radius);
  if (weight <= static_cast<real>(0)) {
    return;
  }

  const real inv_distance = static_cast<real>(1) / distance;
  number_density += weight;
  sx += weight * dx * inv_distance;
  sy += weight * dy * inv_distance;
  sz += weight * dz * inv_distance;
  ++weighted_count;
}

__device__ void accumulate_fluid_neighbors(size_type i,
                                           const FluidParticleSoA& fluid,
                                           const NeighborListView& neighbors,
                                           real support_radius,
                                           real& number_density,
                                           real& sx,
                                           real& sy,
                                           real& sz,
                                           int& weighted_count) {
  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = neighbors.offsets[i]; out < neighbors.offsets[i + 1]; ++out) {
    const index_t j = neighbors.indices[out];
    accumulate_neighbor_moments(fluid.x[j] - px,
                                fluid.y[j] - py,
                                fluid.z[j] - pz,
                                support_radius,
                                number_density,
                                sx,
                                sy,
                                sz,
                                weighted_count);
  }
}

__device__ void accumulate_wall_neighbors(size_type i,
                                          const FluidParticleSoA& fluid,
                                          const WallParticleSoA& walls,
                                          const NeighborListView& neighbors,
                                          real support_radius,
                                          real& number_density,
                                          real& sx,
                                          real& sy,
                                          real& sz,
                                          int& weighted_count) {
  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = neighbors.offsets[i]; out < neighbors.offsets[i + 1]; ++out) {
    const index_t j = neighbors.indices[out];
    accumulate_neighbor_moments(walls.x[j] - px,
                                walls.y[j] - py,
                                walls.z[j] - pz,
                                support_radius,
                                number_density,
                                sx,
                                sy,
                                sz,
                                weighted_count);
  }
}

struct WallBasis {
  int count{};
  real x[kMaxWallBasisDirections]{};
  real y[kMaxWallBasisDirections]{};
  real z[kMaxWallBasisDirections]{};
};

__device__ void add_wall_basis_direction(WallBasis& basis,
                                         real x,
                                         real y,
                                         real z,
                                         real independence_threshold) {
  real rx = x;
  real ry = y;
  real rz = z;
  for (int k = 0; k < basis.count; ++k) {
    const real projection = rx * basis.x[k] + ry * basis.y[k] + rz * basis.z[k];
    rx -= projection * basis.x[k];
    ry -= projection * basis.y[k];
    rz -= projection * basis.z[k];
  }

  const real length_squared = rx * rx + ry * ry + rz * rz;
  if (length_squared <= independence_threshold * independence_threshold ||
      basis.count >= kMaxWallBasisDirections) {
    return;
  }

  const real inv_length = safe_rsqrt(length_squared);
  basis.x[basis.count] = rx * inv_length;
  basis.y[basis.count] = ry * inv_length;
  basis.z[basis.count] = rz * inv_length;
  ++basis.count;
}

__device__ WallBasis build_wall_basis(size_type i,
                                      const FluidParticleSoA& fluid,
                                      const WallParticleSoA& walls,
                                      const NeighborListView& wall_neighbors,
                                      const SimulationConfig& config) {
  WallBasis basis{};
  if (!config.include_wall_neighbors) {
    return basis;
  }

  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = wall_neighbors.offsets[i]; out < wall_neighbors.offsets[i + 1]; ++out) {
    const index_t j = wall_neighbors.indices[out];
    const real dx = walls.x[j] - px;
    const real dy = walls.y[j] - py;
    const real dz = walls.z[j] - pz;
    const real distance_squared = dx * dx + dy * dy + dz * dz;
    if (distance_squared <= static_cast<real>(0)) {
      continue;
    }

    const real distance = sqrt(distance_squared);
    if (particle_weight(distance, config.support_radius) <= static_cast<real>(0)) {
      continue;
    }

    const real normal_length_squared = walls.normal_x[j] * walls.normal_x[j] +
                                       walls.normal_y[j] * walls.normal_y[j] +
                                       walls.normal_z[j] * walls.normal_z[j];
    if (normal_length_squared <= static_cast<real>(0)) {
      continue;
    }

    // Wall normals point from wall particles to fluid, so the solid-side missing direction is -n.
    const real inv_normal_length = safe_rsqrt(normal_length_squared);
    add_wall_basis_direction(basis,
                             -walls.normal_x[j] * inv_normal_length,
                             -walls.normal_y[j] * inv_normal_length,
                             -walls.normal_z[j] * inv_normal_length,
                             config.wall_normal_independence_threshold);
  }

  return basis;
}

__global__ void classify_primary_surface_kernel(const FluidParticleSoA fluid,
                                                const WallParticleSoA walls,
                                                const NeighborListView fluid_neighbors,
                                                const NeighborListView wall_neighbors,
                                                const SimulationConfig config,
                                                real reference_number_density,
                                                SurfaceDetectionDiagnosticsView diagnostics) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  const index_t fluid_count = fluid_neighbors.offsets[i + 1] - fluid_neighbors.offsets[i];
  const index_t wall_count = wall_neighbors.offsets[i + 1] - wall_neighbors.offsets[i];
  const index_t classification_count =
      config.include_wall_neighbors ? fluid_count + wall_count : fluid_count;

  real number_density = static_cast<real>(0);
  real sx = static_cast<real>(0);
  real sy = static_cast<real>(0);
  real sz = static_cast<real>(0);
  int weighted_count = 0;

  accumulate_fluid_neighbors(
      i, fluid, fluid_neighbors, config.support_radius, number_density, sx, sy, sz, weighted_count);
  if (config.include_wall_neighbors) {
    accumulate_wall_neighbors(i,
                              fluid,
                              walls,
                              wall_neighbors,
                              config.support_radius,
                              number_density,
                              sx,
                              sy,
                              sz,
                              weighted_count);
  }

  real anisotropy = static_cast<real>(0);
  if (weighted_count > 0) {
    anisotropy = sqrt(sx * sx + sy * sy + sz * sz) / static_cast<real>(weighted_count);
  }
  const real number_density_ratio = number_density / reference_number_density;

  const real missing_x = -sx;
  const real missing_y = -sy;
  const real missing_z = -sz;
  const real missing_length_squared =
      missing_x * missing_x + missing_y * missing_y + missing_z * missing_z;
  real air_x = missing_x;
  real air_y = missing_y;
  real air_z = missing_z;
  if (missing_length_squared > static_cast<real>(0)) {
    const WallBasis wall_basis = build_wall_basis(i, fluid, walls, wall_neighbors, config);
    for (int k = 0; k < wall_basis.count; ++k) {
      const real projection = air_x * wall_basis.x[k] + air_y * wall_basis.y[k] +
                              air_z * wall_basis.z[k];
      if (projection > static_cast<real>(0)) {
        air_x -= projection * wall_basis.x[k];
        air_y -= projection * wall_basis.y[k];
        air_z -= projection * wall_basis.z[k];
      }
    }
  }
  const real air_length_squared = air_x * air_x + air_y * air_y + air_z * air_z;
  const real air_length = sqrt(air_length_squared);
  const real missing_length = sqrt(missing_length_squared);
  const real air_open_ratio =
      missing_length > static_cast<real>(0) ? air_length / missing_length : static_cast<real>(0);
  const real air_anisotropy =
      weighted_count > 0 ? air_length / static_cast<real>(weighted_count) : static_cast<real>(0);
  real surface_normal_x = static_cast<real>(0);
  real surface_normal_y = static_cast<real>(0);
  real surface_normal_z = static_cast<real>(0);
  if (air_length_squared > static_cast<real>(0)) {
    const real inv_air_length = safe_rsqrt(air_length_squared);
    surface_normal_x = air_x * inv_air_length;
    surface_normal_y = air_y * inv_air_length;
    surface_normal_z = air_z * inv_air_length;
  }

  int type = static_cast<int>(SurfaceType::Inner);
  if (classification_count < config.splash_neighbor_threshold) {
    type = static_cast<int>(SurfaceType::Splash);
  } else if (number_density_ratio < config.number_density_ratio_threshold &&
             air_open_ratio > config.air_open_ratio_threshold &&
             air_anisotropy > config.air_anisotropy_threshold) {
    type = static_cast<int>(SurfaceType::Surface);
  }
  fluid.surface_type[i] = type;

  if (diagnostics.fluid_neighbor_count != nullptr) {
    diagnostics.fluid_neighbor_count[i] = fluid_count;
  }
  if (diagnostics.wall_neighbor_count != nullptr) {
    diagnostics.wall_neighbor_count[i] = wall_count;
  }
  if (diagnostics.number_density != nullptr) {
    diagnostics.number_density[i] = number_density;
  }
  if (diagnostics.number_density_ratio != nullptr) {
    diagnostics.number_density_ratio[i] = number_density_ratio;
  }
  if (diagnostics.anisotropy != nullptr) {
    diagnostics.anisotropy[i] = anisotropy;
  }
  if (diagnostics.air_open_ratio != nullptr) {
    diagnostics.air_open_ratio[i] = air_open_ratio;
  }
  if (diagnostics.air_anisotropy != nullptr) {
    diagnostics.air_anisotropy[i] = air_anisotropy;
  }
  if (diagnostics.surface_normal_x != nullptr) {
    diagnostics.surface_normal_x[i] = surface_normal_x;
  }
  if (diagnostics.surface_normal_y != nullptr) {
    diagnostics.surface_normal_y[i] = surface_normal_y;
  }
  if (diagnostics.surface_normal_z != nullptr) {
    diagnostics.surface_normal_z[i] = surface_normal_z;
  }
}

__global__ void expand_near_surface_kernel(const FluidParticleSoA fluid,
                                           const NeighborListView fluid_neighbors,
                                           real near_surface_radius_squared) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count || fluid.surface_type[i] != static_cast<int>(SurfaceType::Inner)) {
    return;
  }

  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = fluid_neighbors.offsets[i]; out < fluid_neighbors.offsets[i + 1]; ++out) {
    const index_t j = fluid_neighbors.indices[out];
    if (fluid.surface_type[j] != static_cast<int>(SurfaceType::Surface)) {
      continue;
    }

    const real dx = fluid.x[j] - px;
    const real dy = fluid.y[j] - py;
    const real dz = fluid.z[j] - pz;
    if (dx * dx + dy * dy + dz * dz <= near_surface_radius_squared) {
      fluid.surface_type[i] = static_cast<int>(SurfaceType::NearSurface);
      return;
    }
  }
}

__device__ Vec3 virtual_light_direction(int index) {
  constexpr real kInvSqrt2 = static_cast<real>(0.7071067811865475);
  switch (index) {
    case 0:
      return Vec3{static_cast<real>(1), static_cast<real>(0), static_cast<real>(0)};
    case 1:
      return Vec3{static_cast<real>(-1), static_cast<real>(0), static_cast<real>(0)};
    case 2:
      return Vec3{static_cast<real>(0), static_cast<real>(1), static_cast<real>(0)};
    case 3:
      return Vec3{static_cast<real>(0), static_cast<real>(-1), static_cast<real>(0)};
    case 4:
      return Vec3{static_cast<real>(0), static_cast<real>(0), static_cast<real>(1)};
    case 5:
      return Vec3{static_cast<real>(0), static_cast<real>(0), static_cast<real>(-1)};
    case 6:
      return Vec3{kInvSqrt2, kInvSqrt2, static_cast<real>(0)};
    case 7:
      return Vec3{kInvSqrt2, -kInvSqrt2, static_cast<real>(0)};
    case 8:
      return Vec3{-kInvSqrt2, kInvSqrt2, static_cast<real>(0)};
    case 9:
      return Vec3{-kInvSqrt2, -kInvSqrt2, static_cast<real>(0)};
    case 10:
      return Vec3{kInvSqrt2, static_cast<real>(0), kInvSqrt2};
    case 11:
      return Vec3{kInvSqrt2, static_cast<real>(0), -kInvSqrt2};
    case 12:
      return Vec3{static_cast<real>(0), kInvSqrt2, kInvSqrt2};
    default:
      return Vec3{static_cast<real>(0), kInvSqrt2, -kInvSqrt2};
  }
}

__device__ bool fluid_neighbor_blocks_light(size_type i,
                                            int direction_index,
                                            const FluidParticleSoA& fluid,
                                            const NeighborListView& neighbors,
                                            const SimulationConfig& config) {
  const Vec3 dir = virtual_light_direction(direction_index);
  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = neighbors.offsets[i]; out < neighbors.offsets[i + 1]; ++out) {
    const index_t j = neighbors.indices[out];
    const real dx = fluid.x[j] - px;
    const real dy = fluid.y[j] - py;
    const real dz = fluid.z[j] - pz;
    const real distance_squared = dx * dx + dy * dy + dz * dz;
    if (distance_squared <= static_cast<real>(0) ||
        distance_squared > config.support_radius * config.support_radius) {
      continue;
    }

    const real projection = (dx * dir.x + dy * dir.y + dz * dir.z) * safe_rsqrt(distance_squared);
    if (projection >= config.virtual_light_cone_cosine) {
      return true;
    }
  }

  return false;
}

__device__ bool wall_neighbor_blocks_light(size_type i,
                                           int direction_index,
                                           const FluidParticleSoA& fluid,
                                           const WallParticleSoA& walls,
                                           const NeighborListView& neighbors,
                                           const SimulationConfig& config) {
  const Vec3 dir = virtual_light_direction(direction_index);
  const real px = fluid.x[i];
  const real py = fluid.y[i];
  const real pz = fluid.z[i];

  for (index_t out = neighbors.offsets[i]; out < neighbors.offsets[i + 1]; ++out) {
    const index_t j = neighbors.indices[out];
    const real dx = walls.x[j] - px;
    const real dy = walls.y[j] - py;
    const real dz = walls.z[j] - pz;
    const real distance_squared = dx * dx + dy * dy + dz * dz;
    if (distance_squared <= static_cast<real>(0) ||
        distance_squared > config.support_radius * config.support_radius) {
      continue;
    }

    const real projection = (dx * dir.x + dy * dir.y + dz * dir.z) * safe_rsqrt(distance_squared);
    if (projection >= config.virtual_light_cone_cosine) {
      return true;
    }
  }

  return false;
}

__global__ void virtual_light_kernel(const FluidParticleSoA fluid,
                                     const WallParticleSoA walls,
                                     const NeighborListView fluid_neighbors,
                                     const NeighborListView wall_neighbors,
                                     const SimulationConfig config,
                                     const VirtualLightDiagnosticsView diagnostics) {
  const size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= fluid.count) {
    return;
  }

  int open_count = 0;
  for (int direction = 0; direction < kVirtualLightDirectionCount; ++direction) {
    bool blocked =
        fluid_neighbor_blocks_light(i, direction, fluid, fluid_neighbors, config);
    if (!blocked && config.include_wall_neighbors) {
      blocked = wall_neighbor_blocks_light(i, direction, fluid, walls, wall_neighbors, config);
    }
    if (!blocked) {
      ++open_count;
    }
  }

  if (diagnostics.open_direction_count != nullptr) {
    diagnostics.open_direction_count[i] = open_count;
  }
  if (diagnostics.open_fraction != nullptr) {
    diagnostics.open_fraction[i] =
        static_cast<real>(open_count) / static_cast<real>(kVirtualLightDirectionCount);
  }
}

void validate_surface_inputs(const FluidParticleSoA& fluid,
                             const NeighborListView& fluid_neighbors,
                             const NeighborListView& wall_neighbors,
                             real support_radius) {
  if (support_radius <= static_cast<real>(0)) {
    std::cerr << "Surface detection support radius must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count > fluid_neighbors.particle_count || fluid.count > wall_neighbors.particle_count) {
    std::cerr << "Surface detection neighbor lists are smaller than fluid particle count"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

real compute_uniform_reference_number_density(real particle_spacing, real support_radius) {
  if (particle_spacing <= static_cast<real>(0) || support_radius <= static_cast<real>(0)) {
    std::cerr << "Reference number density requires positive spacing and support radius"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const index_t extent = static_cast<index_t>(std::ceil(support_radius / particle_spacing));
  real reference_density = static_cast<real>(0);
  for (index_t iz = -extent; iz <= extent; ++iz) {
    for (index_t iy = -extent; iy <= extent; ++iy) {
      for (index_t ix = -extent; ix <= extent; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }

        const real dx = static_cast<real>(ix) * particle_spacing;
        const real dy = static_cast<real>(iy) * particle_spacing;
        const real dz = static_cast<real>(iz) * particle_spacing;
        const real distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        reference_density += particle_weight(distance, support_radius);
      }
    }
  }

  return reference_density;
}

void classify_surface_particles(const FluidParticleSoA& fluid,
                                const WallParticleSoA& walls,
                                const NeighborListView& fluid_neighbors,
                                const NeighborListView& wall_neighbors,
                                const SimulationConfig& config,
                                SurfaceDetectionDiagnosticsView diagnostics) {
  validate_surface_inputs(fluid, fluid_neighbors, wall_neighbors, config.support_radius);
  if (config.near_surface_radius < static_cast<real>(0)) {
    std::cerr << "Near-surface radius must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const real reference_number_density =
      compute_uniform_reference_number_density(config.particle_spacing, config.support_radius);
  if (reference_number_density <= static_cast<real>(0)) {
    std::cerr << "Reference number density must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count == 0) {
    return;
  }

  classify_primary_surface_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, walls, fluid_neighbors, wall_neighbors, config, reference_number_density, diagnostics);
  LSMPS3D_CUDA_KERNEL_CHECK();

  if (config.near_surface_radius > static_cast<real>(0)) {
    const real radius_squared = config.near_surface_radius * config.near_surface_radius;
    expand_near_surface_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
        fluid, fluid_neighbors, radius_squared);
    LSMPS3D_CUDA_KERNEL_CHECK();
  }
}

void compute_virtual_light_diagnostics(const FluidParticleSoA& fluid,
                                       const WallParticleSoA& walls,
                                       const NeighborListView& fluid_neighbors,
                                       const NeighborListView& wall_neighbors,
                                       const SimulationConfig& config,
                                       VirtualLightDiagnosticsView diagnostics) {
  validate_surface_inputs(fluid, fluid_neighbors, wall_neighbors, config.support_radius);
  if (config.virtual_light_cone_cosine < static_cast<real>(-1) ||
      config.virtual_light_cone_cosine > static_cast<real>(1)) {
    std::cerr << "Virtual light cone cosine must be in [-1, 1]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (diagnostics.open_direction_count == nullptr && diagnostics.open_fraction == nullptr) {
    std::cerr << "Virtual light diagnostics require at least one output buffer" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (fluid.count == 0) {
    return;
  }

  virtual_light_kernel<<<block_count(fluid.count), kThreadsPerBlock>>>(
      fluid, walls, fluid_neighbors, wall_neighbors, config, diagnostics);
  LSMPS3D_CUDA_KERNEL_CHECK();
}

}  // namespace lsmps3d
