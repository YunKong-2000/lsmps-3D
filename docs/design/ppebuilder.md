# PPE系数矩阵及右边项构建
## 目的
根据粒子的位置关系和临时速度，算出PPE系数矩阵元素和右边项的大小
## 输入
1、代入压力边界条件的$M^{-1}$矩阵
2、对应速度的$M^{-1}$矩阵
3、流体粒子及壁面粒子的临时速度

## 实现
1、根据粒子的拉普拉斯算子离散格式构建系数矩阵以及右边项，根据粒子的速度散度构建右边项
2、对于靠近壁面的流体粒子而言，邻域内壁面粒子的贡献分为两种，一种是在离散压力拉普拉斯算子时，壁面压力边界条件被代入，此时会改变右边项大小；另一种是在计算粒子速度散度时的贡献，此时会改变右边项大小
3、实际计算时不用区分是否靠近壁面，对于每一个流体粒子都进行邻域内流体粒子和壁面粒子的遍历。遍历流体粒子时就能确定系数矩阵的主对角线元素和非对角线元素；然后再遍历壁面粒子，最终确定右边项的大小。

## 关键
1、输入中必须包含 moment_matrix 模块中已经准备好的 $M^{-1}$ 矩阵（两种，一种对应压力，一种对应速度），搭建 PPE 时不需要重复求逆。
2、PPE 系数矩阵和右边项的主计算逻辑应放在同一个组装路径中。对每个流体粒子统一遍历邻域，同时完成压力拉普拉斯 CSR 系数、速度散度 RHS、壁面 Neumann RHS 的累加。
3、对于自由面粒子直接使用最简单的替换行的主对角线元素和右边项的方法强制令其压力求解结果为零。
4、最终的PPE形式如下
$$
\begin{align*}
    &\left(-\frac{2}{r_e^2 \rho} \sum_{j \in \mathrm{fluid}} w_{ij} [\mathbf{C}_4 + \mathbf{C}_5 + \mathbf{C}_6] \mathbf{P}_{ij}\right) p_i + \sum_{j \in \mathrm{fluid}} \left(\frac{2}{r_e^2 \rho} w_{ij} [\mathbf{C}_4 + \mathbf{C}_5 + \mathbf{C}_6] \mathbf{P}_{ij}\right) p_j \\
    &= \frac{1}{\Delta t r_e} \sum_{j \in \mathrm{fluid}} w_{ij} (\mathbf{u}^{*}_j - \mathbf{u}^{*}_i) \begin{bmatrix} \mathbf{C}_1 \\ \mathbf{C}_2 \\ \mathbf{C}_3 \end{bmatrix} \mathbf{P}_{ij} \\
    &\quad + \frac{1}{\Delta t r_e} \sum_{j \in \mathrm{wall}} w_{ij} (\mathbf{u}^{*}_{\mathrm{wall}} - \mathbf{u}^{*}_i)\begin{bmatrix} \mathbf{C}_1 \\ \mathbf{C}_2 \\ \mathbf{C}_3\end{bmatrix} \mathbf{P}_{ij} \\
    &\quad + \frac{2}{r_e \rho} \sum_{j \in \mathrm{wall}} w_{ij} (\rho \mathbf{n}_{\mathrm{wall}} \cdot \mathbf{g}) [\mathbf{C}_4 + \mathbf{C}_5 + \mathbf{C}_6] \mathbf{P}_{ij}
\end{align*}
$$