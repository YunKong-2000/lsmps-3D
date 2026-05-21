#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>

#include "lsmps3d/app/simulation_driver.cuh"

int main(int argc, char** argv) {
  try {
    if (argc > 2) {
      lsmps3d::print_simulation_usage(std::cerr, argv[0]);
      return EXIT_FAILURE;
    }

    const std::filesystem::path config_path =
        argc == 2 ? std::filesystem::path(argv[1]) : std::filesystem::path("config/simulation.ini");
    return lsmps3d::run_simulation(config_path);
  } catch (const std::exception& error) {
    std::cerr << "Simulation failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
