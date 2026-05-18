# lsmps3D

`lsmps3D` is an early-stage CUDA/C++17 implementation path for a 3D single-phase LSMPS/MPS free-surface fluid solver. The current codebase focuses on stable project structure, SoA particle storage, reusable GPU workspaces, background-grid neighbor search, hybrid surface classification, and host-side VTK diagnostics.

## Current Status

Implemented modules:

- Core CUDA utilities, shared numeric types, configuration structures, and reusable workspace ownership.
- Fluid and wall particle SoA buffers, plus separate CSR neighbor lists for fluid-fluid and fluid-wall interactions.
- 3D cell-list neighbor search using CUDA kernels and Thrust sorting/scans.
- Hybrid free-surface classification with `Inner`, `NearSurface`, `Surface`, and `Splash` labels.
- Optional virtual-light surface diagnostic for small validation cases.
- Legacy ASCII VTK particle writer for ParaView diagnostics.
- Unit smoke tests and a hydrostatic reference diagnostic case.

Planned modules include LSMPS local operators, explicit provision updates, PPE assembly, AMGX-based pressure solve, pressure correction, particle shifting, and wall anti-penetration handling.

## Repository Layout

```text
include/lsmps3d/   Public CUDA/C++ headers
src/               Module implementations
tests/unit/        Focused smoke tests
tests/reference/   Small diagnostic/reference programs
docs/              Implementation notes and review summaries
```

## Requirements

- Linux with an NVIDIA GPU and CUDA-capable driver
- CMake 3.24 or newer
- CUDA Toolkit with C++17 support
- A C++17 compiler supported by the CUDA Toolkit

The helper script `start_cuda_container.sh` can launch a CUDA development container with the current directory mounted at `/workspace`.

## Build

```bash
cmake -S . -B build -DLSMPS3D_BUILD_TESTS=ON
cmake --build build -j
```

If the default CUDA architecture list is not suitable for the local GPU, pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` during configuration.

## Test

```bash
ctest --test-dir build --output-on-failure
```

Available tests currently cover workspace allocation/copy, neighbor search, surface detection, and VTK writing. The hydrostatic diagnostic executable is built with the tests and can be run from the build directory to generate debug VTK/CSV output.

## Development Notes

The project follows a staged implementation plan:

1. Keep GPU-facing data in SoA form.
2. Allocate persistent CUDA buffers through explicit workspace/context objects.
3. Keep fluid and wall neighbor CSR tables separate.
4. Prefer CUDA Runtime, Thrust/CUB, cuBLAS, cuSOLVER, cuSPARSE, and AMGX before custom infrastructure when those libraries fit.
5. Add focused tests and diagnostic output with each numerical module.

See `docs/implementation_notes.md` for the current module summaries and validation notes.
