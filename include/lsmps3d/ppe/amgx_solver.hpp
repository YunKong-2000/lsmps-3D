#pragma once

#include <string>

namespace lsmps3d {

struct AmgxSolverConfig {
  std::string config_path{"configs/amgx_ppe.json"};
  bool print_solve_stats{false};
};

class AmgxSolver {
 public:
  explicit AmgxSolver(AmgxSolverConfig config);
  ~AmgxSolver();

  AmgxSolver(const AmgxSolver&) = delete;
  AmgxSolver& operator=(const AmgxSolver&) = delete;

 private:
  AmgxSolverConfig config_{};
};

}  // namespace lsmps3d
