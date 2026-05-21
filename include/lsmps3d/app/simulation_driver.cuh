#pragma once

#include <filesystem>
#include <ostream>

namespace lsmps3d {

void print_simulation_usage(std::ostream& output, const char* program_name);

int run_simulation(const std::filesystem::path& config_path);

}  // namespace lsmps3d
