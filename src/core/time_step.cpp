#include "lsmps3d/core/time_step.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lsmps3d {
namespace {

[[nodiscard]] real clamp_time_step(real value, const TimeStepLimits& limits) noexcept {
  return std::clamp(value, limits.min_time_step, limits.max_time_step);
}

[[nodiscard]] bool is_positive_finite(real value) noexcept {
  return std::isfinite(static_cast<double>(value)) && value > static_cast<real>(0);
}

[[nodiscard]] bool is_non_negative_finite(real value) noexcept {
  return std::isfinite(static_cast<double>(value)) && value >= static_cast<real>(0);
}

void require_positive_finite(real value, const char* name) {
  if (is_positive_finite(value)) {
    return;
  }

  std::ostringstream message;
  message << "TimeStepLimits field '" << name << "' must be positive and finite";
  throw std::invalid_argument(message.str());
}

void require_non_negative_finite(real value, const char* name) {
  if (is_non_negative_finite(value)) {
    return;
  }

  std::ostringstream message;
  message << "TimeStepLimits field '" << name << "' must be non-negative and finite";
  throw std::invalid_argument(message.str());
}

}  // namespace

TimeStepLimits make_time_step_limits(const SimulationConfig& config) noexcept {
  TimeStepLimits limits{};
  limits.current_time_step = config.time_step;
  limits.min_time_step = config.min_time_step;
  limits.max_time_step = config.max_time_step;
  limits.growth_factor = config.time_step_growth_factor;
  limits.cfl = config.cfl;
  limits.particle_spacing = config.particle_spacing;
  limits.final_time = config.final_time;
  limits.output_interval = config.output_interval;
  return limits;
}

void validate_time_step_limits(const TimeStepLimits& limits) {
  require_positive_finite(limits.current_time_step, "current_time_step");
  require_positive_finite(limits.min_time_step, "min_time_step");
  require_positive_finite(limits.max_time_step, "max_time_step");
  require_positive_finite(limits.growth_factor, "growth_factor");
  require_positive_finite(limits.cfl, "cfl");
  require_positive_finite(limits.particle_spacing, "particle_spacing");
  require_non_negative_finite(limits.final_time, "final_time");
  require_positive_finite(limits.output_interval, "output_interval");

  if (limits.growth_factor < static_cast<real>(1)) {
    throw std::invalid_argument("TimeStepLimits field 'growth_factor' must be >= 1");
  }
  if (limits.min_time_step > limits.max_time_step) {
    throw std::invalid_argument("TimeStepLimits requires min_time_step <= max_time_step");
  }
}

SimulationTimeManager::SimulationTimeManager(const TimeStepLimits& limits) {
  reset(limits);
}

SimulationTimeManager::SimulationTimeManager(const SimulationConfig& config) {
  reset(config);
}

void SimulationTimeManager::reset(const TimeStepLimits& limits) {
  validate_time_step_limits(limits);
  limits_ = limits;
  current_time_ = static_cast<real>(0);
  current_time_step_ = clamp_time_step(limits.current_time_step, limits);
  next_output_time_ = static_cast<real>(0);
  step_index_ = 0;
  output_index_ = 0;
  initial_output_pending_ = true;
}

void SimulationTimeManager::reset(const SimulationConfig& config) {
  reset(make_time_step_limits(config));
}

bool SimulationTimeManager::output_due() const noexcept {
  return initial_output_pending_ || current_time_ >= next_output_time_;
}

real SimulationTimeManager::cfl_limited_time_step(real max_velocity) const noexcept {
  if (!(max_velocity > static_cast<real>(0)) ||
      !std::isfinite(static_cast<double>(max_velocity))) {
    return limits_.max_time_step;
  }

  return limits_.cfl * limits_.particle_spacing / max_velocity;
}

real SimulationTimeManager::next_time_step(real max_velocity) const noexcept {
  const real growth_limited = current_time_step_ * limits_.growth_factor;
  const real cfl_limited = cfl_limited_time_step(max_velocity);
  real dt = clamp_time_step(std::min(growth_limited, cfl_limited), limits_);
  const real remaining_time = limits_.final_time - current_time_;
  if (remaining_time > static_cast<real>(0) && dt > remaining_time) {
    dt = remaining_time;
  }
  return dt;
}

TimeStepStatus SimulationTimeManager::advance(real max_velocity) {
  const real previous_time = current_time_;
  const real cfl_dt = cfl_limited_time_step(max_velocity);
  const real dt = next_time_step(max_velocity);
  current_time_ += dt;
  current_time_step_ = dt;
  ++step_index_;

  bool should_output = false;
  if (initial_output_pending_ || current_time_ >= next_output_time_) {
    should_output = true;
    initial_output_pending_ = false;
    ++output_index_;
    while (next_output_time_ <= current_time_) {
      next_output_time_ += limits_.output_interval;
    }
  }

  return TimeStepStatus{step_index_,
                        output_index_,
                        previous_time,
                        current_time_,
                        dt,
                        cfl_dt,
                        max_velocity,
                        should_output,
                        finished()};
}

TimeStepStatus SimulationTimeManager::mark_initial_output() {
  bool should_output = false;
  if (initial_output_pending_) {
    should_output = true;
    initial_output_pending_ = false;
    ++output_index_;
    next_output_time_ = limits_.output_interval;
  }

  return TimeStepStatus{step_index_,
                        output_index_,
                        current_time_,
                        current_time_,
                        current_time_step_,
                        std::numeric_limits<real>::infinity(),
                        static_cast<real>(0),
                        should_output,
                        finished()};
}

}  // namespace lsmps3d
