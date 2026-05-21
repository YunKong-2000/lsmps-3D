# 动态时间步模块
## 目的
模拟中时间步长动态可调整，时间步长受限于 CFL 数以及人为给定的最大、最小时间步长。
## 具体实现
1、模拟开始时具有一个初始时间步长（配置文件给定），开始后时间步长以一个系数指数增长（系数大于等于 1，配置文件给定）。
2、时间步长受限于 CFL 数，程序会在 correction 步中找出最大的速度模长，时间步长$\Delta t$满足限制
$$
\Delta t < \frac{CFL * \Delta r}{|\mathbf{u}|_{max}}
$$
3、时间步长还受限于人为给定的最大和最小时间步长
$$
\Delta t_{min} < \Delta t < \Delta t_{max}
$$
4、该模块除了调整以及统计模拟时间外，还需判定程序是否输出模拟结果，当时间累计至一定间隔后，程序输出模拟结果文件。还需判定程序模拟是否结束，当到达模拟时间后停止模拟。

## 接口设计
- 配置项统一放在 `SimulationConfig` 的 `[simulation]` 段：`time_step`、`min_time_step`、`max_time_step`、`time_step_growth_factor`、`final_time`、`output_interval`、`cfl` 和 `particle_spacing`。
- `SimulationTimeManager` 是 host 侧状态管理器，不拥有 GPU 内存。每个时间步由 correction 步传入当前 `max_velocity`，模块计算
  $$
  \Delta t_{n+1} = min(growth * \Delta t_n,\ \frac{CFL * \Delta r}{|\mathbf{u}|_{max}})
  $$
  然后夹限到 $[\Delta t_{min}, \Delta t_{max}]$，最后裁剪到不超过剩余模拟时间。
- 当速度最大值为 0 或不可用时，CFL 限制视为不收紧，时间步只受增长系数和最大时间步约束。
- 输出判定通过 `mark_initial_output()` 和 `advance()` 的 `TimeStepStatus::should_output` 返回。初始状态默认允许输出第 0 帧，之后每跨过 `output_interval` 触发一次输出。
- 终止判定通过 `TimeStepStatus::reached_final_time` 或 `SimulationTimeManager::finished()` 返回，到达 `final_time` 后停止模拟循环。
