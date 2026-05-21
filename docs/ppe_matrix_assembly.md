# PPE 系数矩阵装配说明

本文档说明当前程序中 PPE（Pressure Poisson Equation）系数矩阵和右端项的实际装配方式。对应实现主要位于：

- `src/ppe/ppe_matrix.cu`
- `include/lsmps3d/ppe/ppe_matrix.cuh`
- `src/lsmps/moment_matrix.cu`

## 当前离散系统

当前 PPE 求解的是一个 CSR 稀疏线性系统：

```text
A p = b
```

其中：

- `p` 是流体粒子的压力未知量，未知数数量为流体粒子数 `Nf`。
- `A` 是 `Nf x Nf` 的 CSR 矩阵。
- `b` 是 PPE 右端项。
- 矩阵行只对应流体粒子，不为壁面粒子建立压力未知量。

当前矩阵容量预估为：

```text
nnz = Nf + num_fluid_neighbors
```

也就是每个流体粒子行预留：

- 1 个主对角线元素 `A_ii`
- 每个流体邻居 1 个非对角元素 `A_ij`

注意：当前 `nnz` 不包含壁面邻居贡献，因此壁面压力边界条件还没有进入 PPE 矩阵结构。

## 行偏移构建

对第 `i` 个流体粒子，流体邻居 CSR 为：

```text
fluid_neighbor_offsets[i] ... fluid_neighbor_offsets[i + 1]
```

当前 PPE 矩阵行起点为：

```text
row_offsets[i] = i + fluid_neighbor_offsets[i]
```

因此第 `i` 行长度为：

```text
row_nnz_i = 1 + fluid_neighbor_count_i
```

最后一个偏移为：

```text
row_offsets[Nf] = Nf + fluid_neighbor_offsets[Nf]
```

## 自由面/飞溅粒子的 Dirichlet 行

当前代码把 `Surface` 和 `Splash` 粒子作为压力 Dirichlet 行处理：

```text
p_i = 0
```

对应矩阵和右端项为：

```text
A_ii = 1
A_ij = 0,  j 为该行流体邻居
b_i  = 0
```

这等价于压力自由面条件：

```text
p = 0
```

当前没有对 `NearSurface` 粒子设置 Dirichlet，`NearSurface` 仍走普通内部行装配。

## 普通流体粒子的矩阵行

对非 `Surface` / `Splash` 粒子，当前使用一个图 Laplacian 风格的稀疏模板。

设：

```text
r_ij = x_j - x_i
d_ij = |r_ij|
h    = support_radius
V    = dx^3 = particle_spacing^3
eps  = lsmps_regularization
```

当前权函数为线性核：

```text
w_ij = max(0, 1 - d_ij / h)
```

代码中只有当：

```text
0 < d_ij < h
```

时才产生非零邻居系数。

非对角线系数为：

```text
A_ij = - 2 V w_ij / h^2
```

主对角线累加为：

```text
A_ii = sum_j ( - A_ij ) + eps
```

因此普通流体行可以写为：

```text
A_ii p_i + sum_j A_ij p_j = b_i
```

也就是：

```text
(sum_j 2 V w_ij / h^2 + eps) p_i
- sum_j (2 V w_ij / h^2) p_j
= b_i
```

其中 `j` 只遍历流体邻居。

## 右端项装配

PPE 右端项当前为：

```text
b_i = rho / dt * div(u*)_i
```

对于压力 Dirichlet 粒子：

```text
b_i = 0
```

其中 `div(u*)` 由 LSMPS 速度散度算子计算：

```text
div(u*) = du*_x/dx + du*_y/dy + du*_z/dz
```

当前散度计算使用：

- 固定的流体/壁面几何位置。
- 流体侧速度采样 `u*_f`。
- 壁面侧速度采样 `u*_w`。

在 provision 阶段：

```text
u*_f = u_f + dt (nu Laplacian(u_f) + g)
u*_w = u_w + dt g
```

这里 `u*_w` 只用于临时速度散度的壁面 Dirichlet 采样，不表示壁面真实运动被重力推进。

## 当前壁面边界处理状态

当前程序已经在 `div(u*)` 的 LSMPS 速度散度中使用壁面临时速度 `u*_w`，用于减少静水场近壁虚假散度。

但是，**壁面压力边界条件尚未代入 PPE 系数矩阵或右端项**。

具体表现为：

1. PPE 矩阵 `A` 只为流体邻居写入非对角项。
2. `matrix_nnz` 只按 `Nf + num_fluid_neighbors` 分配。
3. `assemble_ppe_matrix_kernel` 不遍历 `wall_neighbors`。
4. 壁面 Neumann 条件没有形成额外的矩阵项或 RHS 项。
5. 静水问题中，即使 AMGX 正常求解，若 `div(u*)` 接近零，则 RHS 接近零，系统自然会得到接近零的压力解，而不是解析静水压。

## 静水问题中缺失的压力边界条件

静水压力满足：

```text
grad p = rho g_body
```

若重力为：

```text
g_body = (0, 0, -g)
```

则解析压力为：

```text
p(z) = rho g max(H - z, 0)
```

壁面不可穿透边界下，压力 Neumann 条件通常来自法向动量平衡。若壁面法向 `n` 指向流体域内，静水下需要满足类似：

```text
dp/dn = rho g_body . n
```

或根据项目中法向定义采用相反符号：

```text
dp/dn = - rho g_body . n
```

符号需要结合壁面法向方向和压力梯度算子定义验证。

当前 PPE 装配没有把该边界条件写入：

```text
A p = b
```

因此静水压力无法由 PPE 系统恢复出来。

## 后续需要补充的装配方向

后续应把壁面压力 Neumann 条件纳入 PPE。至少需要明确：

1. 壁面邻居是否增加矩阵行中的隐式贡献。
2. Neumann 边界项进入 RHS 的符号和尺度。
3. 是否复用 LSMPS 压力 Neumann 算子中已有的 `rho g . n` 项。
4. PPE 矩阵容量是否需要从：

```text
Nf + num_fluid_neighbors
```

扩展为包含壁面约束所需的额外 nnz，或者将壁面贡献折算进主对角线/RHS 而不增加列。

当前静水诊断算例 `tests/reference/hydrostatic_surface_diagnostics.cu` 已输出以下字段用于检查：

- `hydrostatic_pressure`
- `ppe_pressure`
- `ppe_pressure_error`
- `ppe_pressure_abs_error`
- `ppe_rhs`
- `ppe_velocity_divergence`
- `ppe_matrix_diagonal`
- `ppe_matrix_row_sum`
- `ppe_matrix_row_nnz`
- `fluid_neighbor_count`
- `wall_neighbor_count`

这些字段可用于确认当前 RHS 是否接近零、近壁行矩阵系数是否缺少壁面贡献，以及 AMGX 求解压力与解析静水压之间的差异。
