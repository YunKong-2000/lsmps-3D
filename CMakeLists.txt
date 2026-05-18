cmake_minimum_required(VERSION 3.24)

project(
  lsmps3d
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

option(LSMPS3D_BUILD_TESTS "Build LSMPS3D tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real 80-real 86-real 89-real)
endif()

add_library(lsmps3d_core
  src/core/runtime.cu
  src/io/vtk_writer.cpp
  src/neighbor/neighbor_search.cu
  src/surface/surface_detection.cu
)

target_include_directories(lsmps3d_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_features(lsmps3d_core PUBLIC cxx_std_17 cuda_std_17)

target_compile_options(lsmps3d_core
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(lsmps3d_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

if(LSMPS3D_BUILD_TESTS)
  enable_testing()

  add_executable(smoke_particle_copy tests/unit/smoke_particle_copy.cu)
  target_link_libraries(smoke_particle_copy PRIVATE lsmps3d_core)

  add_executable(smoke_neighbor_search tests/unit/smoke_neighbor_search.cu)
  target_link_libraries(smoke_neighbor_search PRIVATE lsmps3d_core)

  add_executable(smoke_surface_detection tests/unit/smoke_surface_detection.cu)
  target_link_libraries(smoke_surface_detection PRIVATE lsmps3d_core)

  add_executable(smoke_vtk_writer tests/unit/smoke_vtk_writer.cpp)
  target_link_libraries(smoke_vtk_writer PRIVATE lsmps3d_core)

  add_executable(hydrostatic_surface_diagnostics tests/reference/hydrostatic_surface_diagnostics.cu)
  target_link_libraries(hydrostatic_surface_diagnostics PRIVATE lsmps3d_core)

  add_test(NAME smoke_particle_copy COMMAND smoke_particle_copy)
  add_test(NAME smoke_neighbor_search COMMAND smoke_neighbor_search)
  add_test(NAME smoke_surface_detection COMMAND smoke_surface_detection)
  add_test(NAME smoke_vtk_writer COMMAND smoke_vtk_writer)
endif()
