#pragma once

#include "lsmps3d/core/config.hpp"

namespace lsmps3d {

class AmgxSolver {
 public:
  explicit AmgxSolver(SimulationConfig config);
  ~AmgxSolver();

  AmgxSolver(const AmgxSolver&) = delete;
  AmgxSolver& operator=(const AmgxSolver&) = delete;

 private:
  SimulationConfig config_{};
};

}  // namespace lsmps3d
