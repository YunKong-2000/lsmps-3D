cmake_minimum_required(VERSION 3.16)

project(lsmps3d_preprocess_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(generate_hydrostatic_box generate_hydrostatic_box.cpp)
