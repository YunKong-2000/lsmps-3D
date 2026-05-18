# Implementation Notes

## Stage 0 Architecture Summary

- Changed: Added the initial CMake project, module directory layout, public headers, reusable GPU buffer owners, and a CUDA smoke test.
- Reason: The first implementation stage fixes ownership boundaries before neighbor search, LSMPS operators, PPE assembly, AMGX solving, and VTK output are added.
- CUDA libraries used or considered: CUDA Runtime is used for initial device allocation and host-device copy validation. cuBLAS, cuSOLVER, cuSPARSE, Thrust/CUB, and AMGX are intentionally left as future module dependencies.
- Memory strategy: `DeviceFluidParticles`, `DeviceWallParticles`, `DeviceNeighborList`, and `DeviceCellList` own device buffers and support explicit `resize`/`release` plus move-only RAII lifetime. This keeps allocation outside future timestep hot paths.
- Data layout: Fluid and wall particles use SoA arrays. Fluid and wall neighbors are represented as separate CSR lists so indices always refer to a single particle collection.
- Mathematical formulation: No numerical operators are implemented in stage 0. The LSMPS headers reserve 3D linear and quadratic basis sizes for later moment matrix construction.
- Tests: Configure with CMake and run `ctest`; the smoke test allocates fluid, wall, and neighbor workspaces and verifies a small host-device particle coordinate copy.
- Review notes: AMGX and VTK classes are boundary stubs only. Solver setup, matrix assembly, output writing, and numerical kernels remain future work.

## Stage 1 Data Workspace Summary

- Changed: Added `SimulationWorkspace`, `WorkspaceSpec`, and explicit byte accounting for fluid particles, wall particles, neighbor CSR lists, and cell-list buffers.
- Reason: Time-step modules need one reusable ownership boundary for all primary device buffers before neighbor search, surface classification, LSMPS, PPE, and correction code are added.
- CUDA libraries used or considered: CUDA Runtime remains the only dependency here. Thrust/CUB will be considered in the neighbor-search implementation for sorting and scans, but no algorithmic workspace is allocated in this stage.
- Memory strategy: `SimulationWorkspace` owns move-only RAII buffers and allocates them during `resize`. The intended lifecycle is initialize once from the maximum problem size, reuse across time steps, then release at simulation shutdown. Capacity is determined by fluid particle count, wall particle count, maximum fluid/wall neighbors per fluid particle, and cell count.
- Data layout: Single-phase fluid data is stored as SoA fields `x/y/z`, `vx/vy/vz`, `pressure`, and `surface_type`; density remains a scalar configuration value. Wall data is a separate SoA with `x/y/z`, `vx/vy/vz`, and `normal_x/normal_y/normal_z`. Fluid and wall neighbors use separate CSR buffers sized by `fluid_capacity * max_*_neighbors_per_particle`.
- Capacity estimate: `WorkspaceSpec::bytes()` reports the current peak for persistent stage-1 buffers: particle arrays, two neighbor CSR lists, and fluid/wall cell lists. Later solver, LSMPS, sort scratch, and diagnostic arrays should add their own terms instead of hiding allocations in hot paths.
- Tests: `smoke_particle_copy` now also constructs a `SimulationWorkspace`, checks view capacities, and verifies allocated persistent bytes match the spec estimate.
- Review notes: The workspace records capacities, not active counts. Future modules that compact, sort, or deactivate particles should track active ranges separately while preserving these maximum allocations.

## Stage 2 Neighbor Search Summary

- Changed: Added a 3D background-grid neighbor-search module that builds fluid and wall cell lists, then emits separate fluid-fluid and fluid-wall CSR neighbor tables for each fluid particle.
- Reason: Later surface classification, LSMPS operators, PPE assembly, and correction modules need stable neighbor rows whose indices refer to exactly one particle collection.
- CUDA libraries used or considered: Thrust is used for `sort_by_key` and `exclusive_scan`; custom kernels handle cell assignment, cell-range construction, and radius-filtered CSR writes.
- Memory strategy: Persistent particle, cell-list, and neighbor-list buffers remain owned by `SimulationWorkspace`; the search reuses those buffers and checks CSR capacity before writing neighbor indices.
- Data layout: Input coordinates stay in SoA fields. Cell IDs and sorted particle indices are stored per particle collection, while fluid and wall neighbors are kept in separate CSR lists.
- Mathematical formulation: Particles are mapped to `floor((x - origin) / cell_size)` with clamping to the grid domain. Each fluid particle scans the surrounding `3x3x3` cells and accepts neighbors satisfying `|x_i - x_j|^2 <= r^2`; self is excluded only for fluid-fluid neighbors.
- Tests: `smoke_neighbor_search` validates CSR offsets and row indices for a small 3D fluid/wall case with radius truncation.
- Review notes: The first implementation requires `radius <= cell_size`, so one layer of adjacent cells is sufficient. Larger support radii should generalize the scan stencil before being enabled.

## Stage 3 Surface Detection Summary

- Changed: Added GPU hybrid surface classification and an optional virtual-light diagnostic module.
- Reason: Downstream LSMPS, PPE, and correction stages need stable `Inner`, `NearSurface`, `Surface`, and `Splash` labels before applying free-surface and splash-specific logic.
- CUDA libraries used or considered: CUDA Runtime kernels are used because the first classifier is a per-particle CSR traversal with small reductions. No new third-party dependency is needed.
- Memory strategy: The classifier writes into the existing fluid SoA `surface_type` field. Optional diagnostics are caller-owned device arrays, so no hidden allocation occurs inside timestep kernels.
- Data layout: Inputs remain SoA particle coordinates plus separate fluid and wall CSR neighbor lists. Diagnostics are separate SoA-style arrays for neighbor counts, number density, density ratio, raw anisotropy, air-open ratio, air anisotropy, surface normal, and virtual-light openness.
- Mathematical formulation: The hybrid classifier uses `neighbor_count < threshold` for `Splash`, otherwise labels `Surface` when the normalized linear kernel density `n_i / n_0` is below a ratio threshold and the wall-corrected air opening is strong enough. Here `n_i = sum(max(0, 1 - |r_ij| / h))`, and `n_0` is computed on the CPU from a uniform orthogonal particle lattice with the configured particle spacing and support radius. The raw missing-neighbor vector is `m = -sum(w_ij r_ij / |r_ij|)`. Near wall boundaries, up to three independent solid-side wall-normal directions are built from wall neighbors, the component of `m` explained by those directions is removed, and the residual `m_air` defines `air_open_ratio = |m_air| / |m|`, `air_anisotropy = |m_air| / n`, and the diagnostic `surface_normal`. A second kernel expands remaining `Inner` particles to `NearSurface` when a `Surface` neighbor lies within the configured radius.
- Optional diagnostic: The 3D virtual-light path samples 14 fixed directions and reports the fraction of directions whose cone is not blocked by fluid or wall neighbors. It is intentionally diagnostic-only in this stage.
- Tests: `smoke_surface_detection` validates all four surface labels, density/anisotropy diagnostic contrast, and virtual-light open-direction output on a compact 3D fixture.
- Reference diagnostic: `hydrostatic_surface_diagnostics` builds a 1 m x 1 m x 1 m box filled to 0.5 m with 0.02 m spacing, runs neighbor search and surface classification, then writes VTK and CSV fields for fluid neighbor count, wall neighbor count, number density, density ratio, raw anisotropy, air-open ratio, air anisotropy, surface normal, and surface type.
- Review notes: Density thresholds are dimensionless ratios so they are less sensitive to particle resolution, but still need calibration against 3D breaking-flow cases. Wall-boundary suppression requires valid wall normals pointing from wall particles toward the fluid. The virtual-light direction set is deliberately small for diagnostics; it should be expanded or stratified before being used as a primary classifier.

## Stage 2 VTK Diagnostics Summary

- Changed: Added a host-side legacy VTK particle writer with points plus extensible named point fields. Modules can append arbitrary real scalars, integer scalars, and 3D vector fields such as velocity, pressure gradient, neighbor counts, surface type, number density, and anisotropy.
- Reason: Neighbor search and surface detection need an early ParaView-readable output path for checking particle geometry, boundary truncation, and free-surface threshold behavior before the full solver loop exists.
- CUDA libraries used or considered: No CUDA-side IO dependency is used. The writer consumes host mirrors after callers copy SoA data and diagnostics back from device memory, keeping file output outside CUDA kernels.
- Memory strategy: `LegacyVtkWriter` owns only output configuration and does not allocate persistent GPU memory. Particle coordinates and point fields are caller-owned host vectors, so timestep modules can reuse their existing device buffers and choose which debug mirrors to materialize.
- Data layout: The writer accepts coordinate SoA arrays for points. All point data is appended through `HostVtkPointFields` as named SoA scalar or vector arrays instead of a fixed diagnostics struct.
- Mathematical formulation: No numerical operator is introduced; fields are serialized as provided by neighbor, surface, LSMPS, PPE, or correction modules.
- Tests: `smoke_vtk_writer` validates generated legacy VTK structure, dynamic scalar/vector fields, deterministic file naming, and mismatched field-length rejection.
- Review notes: This first writer targets ASCII debug output for small and medium checkpoints. Large production output should add binary VTK/VTP or batched writer support later.
