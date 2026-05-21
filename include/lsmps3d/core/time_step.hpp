#pragma once

#include "lsmps3d/core/config.hpp"
#include "lsmps3d/core/types.cuh"

namespace lsmps3d {

struct TimeStepLimits {
  real current_time_step{};
  real min_time_step{};
  real max_time_step{};
  real growth_factor{};
  real cfl{};
  real particle_spacing{};
  real final_time{};
  real output_interval{};
};

struct TimeStepStatus {
  size_type step_index{};
  size_type output_index{};
  real previous_time{};
  real current_time{};
  real time_step{};
  real cfl_time_step{};
  real max_velocity{};
  bool should_output{};
  bool reached_final_time{};
};

class SimulationTimeManager {
 public:
  SimulationTimeManager() = default;
  explicit SimulationTimeManager(const TimeStepLimits& limits);
  explicit SimulationTimeManager(const SimulationConfig& config);

  void reset(const TimeStepLimits& limits);
  void reset(const SimulationConfig& config);

  [[nodiscard]] const TimeStepLimits& limits() const noexcept {
    return limits_;
  }

  [[nodiscard]] real current_time() const noexcept {
    return current_time_;
  }

  [[nodiscard]] real current_time_step() const noexcept {
    return current_time_step_;
  }

  [[nodiscard]] size_type step_index() const noexcept {
    return step_index_;
  }

  [[nodiscard]] size_type output_index() const noexcept {
    return output_index_;
  }

  [[nodiscard]] bool finished() const noexcept {
    return current_time_ >= limits_.final_time;
  }

  [[nodiscard]] bool output_due() const noexcept;

  [[nodiscard]] real cfl_limited_time_step(real max_velocity) const noexcept;
  [[nodiscard]] real next_time_step(real max_velocity) const noexcept;
  TimeStepStatus advance(real max_velocity);
  TimeStepStatus mark_initial_output();

 private:
  TimeStepLimits limits_{};
  real current_time_{};
  real current_time_step_{};
  real next_output_time_{};
  size_type step_index_{};
  size_type output_index_{};
  bool initial_output_pending_{true};
};

[[nodiscard]] TimeStepLimits make_time_step_limits(const SimulationConfig& config) noexcept;
void validate_time_step_limits(const TimeStepLimits& limits);

}  // namespace lsmps3d
