#include <cmath>
#include <iostream>
#include <stdexcept>

#include "lsmps3d/core/time_step.hpp"

namespace {

bool nearly_equal(lsmps3d::real actual,
                  lsmps3d::real expected,
                  lsmps3d::real tolerance,
                  const char* label) {
  if (std::abs(actual - expected) <= tolerance) {
    return true;
  }

  std::cerr << label << " mismatch: actual=" << actual << " expected=" << expected
            << " tolerance=" << tolerance << std::endl;
  return false;
}

}  // namespace

int main() {
  lsmps3d::SimulationConfig config;
  config.particle_spacing = static_cast<lsmps3d::real>(0.02);
  config.time_step = static_cast<lsmps3d::real>(0.001);
  config.min_time_step = static_cast<lsmps3d::real>(0.0001);
  config.max_time_step = static_cast<lsmps3d::real>(0.01);
  config.time_step_growth_factor = static_cast<lsmps3d::real>(2.0);
  config.cfl = static_cast<lsmps3d::real>(0.25);
  config.final_time = static_cast<lsmps3d::real>(0.012);
  config.output_interval = static_cast<lsmps3d::real>(0.004);

  lsmps3d::SimulationTimeManager clock(config);
  if (!clock.output_due()) {
    std::cerr << "Initial output should be due" << std::endl;
    return 1;
  }

  const auto initial = clock.mark_initial_output();
  if (!initial.should_output || initial.output_index != 1 || clock.output_due()) {
    std::cerr << "Initial output marker did not update output state" << std::endl;
    return 1;
  }

  auto status = clock.advance(static_cast<lsmps3d::real>(2.0));
  if (!nearly_equal(status.time_step,
                    static_cast<lsmps3d::real>(0.002),
                    static_cast<lsmps3d::real>(1.0e-12),
                    "growth-limited dt") ||
      status.should_output || status.reached_final_time) {
    return 1;
  }

  status = clock.advance(static_cast<lsmps3d::real>(2.0));
  if (!nearly_equal(status.time_step,
                    static_cast<lsmps3d::real>(0.0025),
                    static_cast<lsmps3d::real>(1.0e-12),
                    "CFL-limited dt") ||
      !status.should_output || status.output_index != 2) {
    return 1;
  }

  status = clock.advance(static_cast<lsmps3d::real>(1000.0));
  if (!nearly_equal(status.time_step,
                    config.min_time_step,
                    static_cast<lsmps3d::real>(1.0e-12),
                    "minimum-clamped dt")) {
    return 1;
  }

  int guard = 0;
  while (!status.reached_final_time && guard < 20) {
    status = clock.advance(static_cast<lsmps3d::real>(0.0));
    ++guard;
  }
  if (!status.reached_final_time ||
      !nearly_equal(clock.current_time(),
                    config.final_time,
                    static_cast<lsmps3d::real>(1.0e-12),
                    "final time")) {
    return 1;
  }

  lsmps3d::TimeStepLimits invalid = lsmps3d::make_time_step_limits(config);
  invalid.growth_factor = static_cast<lsmps3d::real>(0.5);
  try {
    lsmps3d::validate_time_step_limits(invalid);
    std::cerr << "Invalid growth factor was accepted" << std::endl;
    return 1;
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
