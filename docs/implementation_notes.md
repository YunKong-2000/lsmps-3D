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
- Reference diagnostic: `hydrostatic_surface_diagnostics` builds a 1 m x 1 m x 1 m box filled to 0.5 m with 0.02 m spacing, runs neighbor search and surface classification, then writes VTK and CSV fields for fluid neighbor count, wall neighbor count, number density, density ratio, raw anisotropy, air-open ratio, air anisotropy, surface normal, and surface type. `complex_surface_diagnostics` adds inclined-plane, sine-wave, droplet, cylinder-obstacle, and stepped-box geometries with expected-surface labels and expected normals for accuracy-oriented diagnostics; each complex case writes separate fluid and wall VTK files, with wall normals included in the wall output.
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

## Stage 5 Moment Matrix Summary

- Changed: Renamed the LSMPS operator module to `moment_matrix`. `DeviceMomentMatrix` now owns only prepared moment inverse matrices and exposes read-only `MomentMatrixView` objects for consuming modules.
- Reason: Provision, PPE assembly, and pressure correction should own their physical operator discretization. The shared module now prepares geometry-dependent $M^{-1}$ data once and does not compute gradients, divergence, or laplacians itself.
- CUDA libraries used or considered: CUDA Runtime kernels assemble per-particle dense moment matrices. cuSOLVER batched Cholesky factors each matrix and solves against identity RHS columns to store explicit `M^{-1}`.
- Memory strategy: `DeviceMomentMatrix` allocates `VelocityWallDirichletTypeA`, `PressureWallNeumannTypeA`, `PressureWallNeumannTypeB`, and `FluidOnlyTypeA` caches. Each cache keeps a factorization scratch matrix, an explicit inverse matrix, batched pointer arrays, and diagnostics.
- Data layout: Particle inputs remain SoA plus separate fluid and wall CSR lists. Consumers receive `MomentMatrixView` with contiguous per-particle inverse matrices and metadata.
- Mathematical formulation: Matrix assembly uses `M_i += w_ij p_ij p_ij^T` for fluid and Dirichlet wall samples, or `N_i += w_ij^B q_ij q_ij^T` for pressure Neumann wall constraints. Consumers form their RHS locally and multiply by `M_i^{-1}`.
- Tests: `smoke_moment_matrix` builds a 27-point 3D stencil, verifies repeated `prepare_matrices(...)` reuse, checks all public views, and copies a representative inverse matrix to validate finite values.

## Stage 6 Provision Explicit Update Summary

- Changed: Added the provision module for temporary velocity prediction from explicit viscosity and gravity.
- Reason: The solver needs `u*` before PPE assembly and pressure correction, while particle position updates remain deferred to later correction stages.
- CUDA libraries used or considered: The module consumes `DeviceMomentMatrix::velocity_type_a()` and uses CUDA Runtime kernels to build velocity RHS vectors, multiply by prepared `M^{-1}`, extract laplacians, and combine the explicit update.
- Memory strategy: `DeviceProvisionExplicitUpdate` owns one reusable `DeviceProvisionWorkspace` with three fluid laplacian arrays and three wall temporary velocity arrays (`vx/vy/vz`). No allocation occurs inside the per-component combine kernels.
- Data layout: Fluid, wall, laplacian, and temporary velocity fields all use SoA arrays. Fluid and wall temporary velocities are caller-owned so later PPE and correction modules can decide how to retain or swap them.
- Mathematical formulation: For non-splash fluid particles the predictor applies `u*_f = u_f + dt * (nu * Laplacian(u_f) + g)` component-wise. Splash particles keep gravity but skip viscosity. Wall temporary samples use `u*_w = u_w + dt * g` for consistent near-wall `div(u*)` evaluation; this does not advance wall positions or replace prescribed wall motion.
- Tests: `smoke_provision_explicit_update` verifies the x-velocity predictor against an analytical quadratic laplacian, checks gravity-only y/z updates, validates splash viscosity skipping, checks wall temporary velocity samples, and checks workspace byte accounting.
- Review notes: Wall repulsive acceleration for splash particles is still a later extension because wall contact/collision policy belongs with the correction and anti-penetration stages.

## Stage 7 PPE CSR And AMGX GMRES Summary

- Changed: Added PPE CSR workspace ownership, unified GPU CSR/RHS assembly from prepared moment inverse matrices, and an `AmgxPpeSolver` wrapper configured for GMRES. CMake now detects AMGX optionally and keeps PPE assembly buildable when AMGX is not installed.
- Reason: Pressure projection needs a global non-symmetric sparse system after the provision step, while AMGX availability should remain deployment-dependent instead of blocking module development.
- CUDA libraries used or considered: CUDA Runtime kernels assemble CSR rows and RHS values. AMGX GMRES is used when `amgx_c.h` and `libamgx`/`libamgxsh` are available, with the default configuration based on the official AMGX `GMRES.json` sample; otherwise the solver wrapper reports that AMGX is not enabled. cuSPARSE remains a possible future fallback for custom iterative solvers but is not used in this stage.
- Memory strategy: `DevicePpeWorkspace` owns reusable `row_offsets`, `col_indices`, `values`, `rhs`, `pressure`, `divergence`, and diagnostic laplacian buffers. Matrix capacity is allocated up front, while each assembly sets the active `nnz` before solver upload.
- Data layout: The PPE matrix uses general CSR storage for a non-symmetric sparse operator, with one diagonal entry plus one off-diagonal slot for every fluid-fluid neighbor row. Geometry, temporary velocity, pressure, and diagnostics remain SoA arrays.
- Mathematical formulation: The RHS is `b_i = rho / dt * div(u*)` for non-free-surface particles and zero for `Surface`/`Splash` Dirichlet rows. The divergence operator uses fixed fluid/wall geometry but samples `u*_f` for fluid neighbors and `u*_w` for wall Dirichlet values, avoiding artificial near-wall divergence in hydrostatic prediction. The first matrix stencil uses weighted graph-Laplacian-style coefficients `A_ij = -2 V w_ij / h^2`, `A_ii = sum_j -A_ij + epsilon`, with linear kernel `w_ij = max(0, 1 - |r_ij| / h)`, but the solver path treats the assembled PPE matrix as a general non-symmetric CSR system.
- Tests: `smoke_ppe_matrix` validates CSR offsets, active nnz, free-surface Dirichlet rows, LSMPS divergence-driven RHS from explicit temporary velocity inputs, row conservation, GMRES configuration, AMGX wrapper availability, and workspace byte accounting.
- Review notes: The first PPE stencil is a conservative sparse projection scaffold. Later pressure-correction validation should tune the Laplacian coefficients against the selected LSMPS pressure operator and expand boundary treatment for wall Neumann terms without assuming matrix symmetry.

## Unified Configuration Summary

- Changed: Replaced module-specific public parameter structs with the single `SimulationConfig` in `core/config.hpp`, and added INI-style load/save/validation helpers.
- Reason: Geometry, simulation, file output, surface, virtual-light, LSMPS, VTK, and AMGX parameters now have one ownership boundary instead of each module exposing a separate configuration class.
- CUDA libraries used or considered: No new CUDA or third-party dependency is used. Configuration parsing is host-side C++17 with standard library streams and filesystem paths.
- Memory strategy: The configuration module owns no GPU memory and performs no hidden device allocation. Modules read scalar settings from `SimulationConfig` before launching kernels or constructing reusable workspaces.
- Data layout: Particle and diagnostic SoA layouts are unchanged. Configuration is scalar host state; single-phase density remains a global simulation parameter, not a per-particle field.
- File format: Config files use fixed INI sections with simple `key=value` entries. Missing keys keep code defaults, unknown keys fail fast, and saved files are emitted in a stable order. The simulation driver derives `cell_origin` and `cell_dim` from the loaded particle coordinates and `cell_size`, so those grid bounds no longer need to be hand-written in the runtime configuration.
- Example:

```ini
[geometry]
particle_spacing=0.02
support_radius=0.062
cell_size=0.062

[simulation]
time_step=0.0001
density=1000
kinematic_viscosity=0.000001
gravity_z=-9.81

[surface]
splash_neighbor_threshold=12
number_density_ratio_threshold=0.85

[files]
output_directory=output
vtk_file_prefix=lsmps3d
amgx_config_path=configs/amgx_ppe.json
```

- Tests: `smoke_config` validates defaults, overrides, validation failures, unknown-key failures, and save/load round-tripping. Existing neighbor, surface, LSMPS, and VTK tests now pass one `SimulationConfig` instead of constructing module-specific parameter structs.
- Review notes: The first file format deliberately avoids nested includes, profile selection, expression evaluation, and unit parsing so configuration remains easy to read and diff.

## Stage 8 Pressure Correction Summary

- Changed: Added the correction module for negative-pressure clamping, LSMPS pressure-gradient evaluation, pressure velocity correction, PS displacement, wall anti-penetration, trapezoidal position update, and final neighbor velocity smoothing.
- Reason: The solver now has the final projection-and-motion stage after PPE pressure solve, so particle velocity and position updates are centralized instead of being spread across provision or PPE code.
- CUDA libraries used or considered: CUDA Runtime kernels are used because each step is a per-particle CSR traversal or vector update. Existing `DeviceMomentMatrix` pressure Type-A inverses are reused for gradients; no new third-party dependency is required.
- Memory strategy: `DeviceCorrectionWorkspace` owns reusable SoA buffers for pressure gradients, PS displacements, and smoothed velocities. Callers allocate particle, neighbor, pressure, and temporary velocity buffers outside the timestep hot path.
- Data layout: All correction diagnostics and outputs use SoA arrays. Pressure can come from the PPE workspace or the fluid pressure field; clamping also mirrors into `fluid.pressure` when it is a separate buffer.
- Mathematical formulation: The pressure correction uses `u = u* - dt / rho * grad(p)` for non-splash particles. Positions use trapezoidal integration `x += 0.5 dt (u_old + u_new) + delta_x_ps`. PS displacement repels overly close fluid neighbors, applies wall-normal clearance near walls, projects free-surface displacement to the local tangent direction when a surface normal can be inferred, and caps displacement by a configurable particle-spacing ratio. Final velocity smoothing blends each non-splash velocity with a weighted neighbor average.
- Tests: `smoke_pressure_correction` validates a linear pressure field gradient, velocity correction, trapezoidal position update, and workspace byte accounting.
- Review notes: The first PS and anti-penetration policy is conservative and local. Production free-surface cases will still need coefficient calibration and diagnostics for wall normals, displacement magnitude, and smoothing strength.

## Stage 9 Dynamic Time Step Summary

- Changed: Added a host-side `SimulationTimeManager` and extended `SimulationConfig` with minimum/maximum time step, growth factor, final simulation time, and output interval settings.
- Reason: The solver loop needs one independent module to advance variable simulation time, clamp time steps by CFL and user limits, trigger output frames, and stop exactly at the requested final time.
- CUDA libraries used or considered: No CUDA library is needed because the manager consumes the correction stage's scalar maximum velocity and owns no device data.
- Memory strategy: The module is scalar host state only. It performs no allocation and introduces no hidden work in timestep kernels.
- Data layout: Existing particle SoA layouts are unchanged. The module exchanges only `real` scalars and a `TimeStepStatus` value with the simulation driver.
- Mathematical formulation: The next step is `dt = clamp(min(growth_factor * dt_previous, cfl * particle_spacing / max_velocity), min_time_step, max_time_step)`, then clipped to the remaining final-time interval. Zero or non-finite maximum velocity leaves CFL non-restrictive.
- Tests: `smoke_time_step` validates initial output behavior, growth-limited steps, CFL-limited steps, minimum clamp behavior, final-time clipping, and invalid limit rejection. `smoke_config` now covers the new INI fields and validation.
- Review notes: The correction module still needs to publish the maximum velocity scalar to the driver before each `advance(...)` call.

## Stage 10 File Management Summary

- Changed: Added `FileManager`, host-side fluid/wall particle input containers, CSV preprocessing input readers, and result-output helpers that wrap the existing legacy VTK writer.
- Reason: The simulation driver needs one file-management boundary for configuration files, preprocessed particle layouts, and timestep outputs instead of scattering path handling and parser logic across modules.
- CUDA libraries used or considered: No CUDA IO dependency is used. CSV and INI parsing are host-side C++17 standard-library code; VTK output reuses `LegacyVtkWriter`.
- Memory strategy: The file manager owns only scalar configuration state. CSV particle data is stored in host SoA vectors and is intended to be copied into preallocated device workspaces before the timestep loop.
- Data layout: Fluid CSV rows use fixed columns `x,y,z,vx,vy,vz`; pressure and surface type are initialized on the host as zero and `Inner`. Wall CSV rows use fixed columns `x,y,z,vx,vy,vz,nx,ny,nz`. Result output keeps fluid and wall VTK files separate with `_fluid` and `_wall` filename suffixes.
- File format: CSV readers allow one header row, blank lines, and `#` comments. Invalid column counts or values report the input path and line number.
- Tests: `smoke_file_manager` validates fluid/wall CSV parsing, default field filling, configuration path round-tripping, VTK output routing, and invalid-value rejection.
- Review notes: The first CSV parser deliberately avoids quoted fields and nested includes because preprocessing files are numeric tables. Large production input can later add binary or VTK/VTU readers behind the same `FileManager` interface.

## Stage 11 Simulation Driver Summary

- Changed: Added the `lsmps3d` executable driver that loads `config/simulation.ini`, reads preprocessed fluid/wall CSV inputs, allocates reusable GPU workspaces, runs neighbor search, surface classification, moment matrix preparation, provision, PPE assembly/solve, pressure correction, dynamic time-step advancement, and VTK result output. The `main.cu` entry point now only handles CLI usage, exception reporting, and dispatch to `run_simulation(...)`; driver helpers live in `app/simulation_driver`.
- Reason: The individual modules now need one production-facing orchestration path to run a complete flow simulation instead of only module smoke tests and diagnostics.
- CUDA libraries used or considered: The driver reuses the existing CUDA Runtime module calls plus optional AMGX-backed PPE solve. No new third-party dependency is added.
- Memory strategy: Particle, neighbor, moment, provision, PPE, correction, and diagnostic buffers are allocated before the time loop and reused. The driver uses fixed first-pass CSR capacities of 256 fluid neighbors and 256 wall neighbors per fluid particle, matching existing reference diagnostics.
- Data layout: Host CSV data is copied into the existing device SoA workspace. Each output frame copies current SoA fields back to host and writes separate fluid/wall VTK files with velocity, pressure, surface type, pressure-gradient, particle-shift, and wall-normal fields.
- Grid setup: The driver derives the background cell-list origin and dimensions from the combined fluid/wall particle bounding box after loading CSV input, padded by one support radius in each direction. `cell_size` remains configurable and still must be at least the support radius.
- Mathematical formulation: The timestep sequence is `neighbor/surface -> u* -> PPE -> pressure correction/particle motion -> neighbor/surface`, with `dt` selected by `SimulationTimeManager` before each physical update from the current maximum velocity.
- Tests: Build the `lsmps3d` target with CMake. Runtime execution requires valid `input/fluid_particles.csv` and `input/wall_particles.csv` files matching the documented numeric CSV formats.
- Review notes: Neighbor-list capacities are driver constants in this first executable. Large or denser cases should promote these capacities to configuration after the expected preprocessing envelope is known.

## Hydrostatic Box Preprocessor Summary

- Changed: Added the `generate_hydrostatic_box` host-side tool under `tool/` with its own standalone CMake project. By default it writes `input/fluid_particles.csv` and `input/wall_particles.csv` for a 1 m x 1 m x 1 m box filled to 0.5 m with 0.02 m particle spacing.
- Reason: The executable simulation driver needs reproducible CSV inputs for the standard hydrostatic validation case instead of relying on test-only in-memory geometry construction.
- CUDA libraries used or considered: No CUDA dependency is needed because this is a deterministic CPU preprocessor.
- Memory strategy: The tool builds host vectors once, writes CSV files, and exits; simulation-time GPU allocation remains owned by the driver workspace.
- Data layout: Fluid CSV rows use `x,y,z,vx,vy,vz` with zero initial velocity. Wall CSV rows use `x,y,z,vx,vy,vz,nx,ny,nz`, with wall normals pointing into the fluid domain.
- Mathematical formulation: Fluid samples are placed at cell centers `(i + 0.5) dx` for `0 <= z < 0.5 m`. Wall samples lie on the five solid boundary planes of the open-top box at multiples of `dx`; duplicate wall edges and corners are avoided by assigning them to one face sample.
- Tests: Configured and built `generate_hydrostatic_box`, then ran it to generate 62,500 fluid particles and 12,601 wall particles in `input/`.
- Review notes: The first tool exposes only an optional output directory argument; geometry constants are fixed to the requested hydrostatic case.
