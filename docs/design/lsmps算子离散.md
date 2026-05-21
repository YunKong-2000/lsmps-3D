三维 LSMPS 算子离散及壁面边界条件耦合数学说明书1. 空间多项式展开与自由度基础在无网格粒子方法中，针对中心粒子 $i$ 及其邻域内的粒子 $j$，定义相对位移向量为 $\mathbf{x}_{ij} = (x_j - x_i, y_j - y_i, z_j - z_i)^T$。为了保证矩阵的条件数稳定，引入无量纲特征缩放参数 $r_s$。在三维空间中，当采用 2阶完全多项式近似（$p=2$） 时：Type-A 格式：通过显式引入函数差值 $\phi_j - \phi_i$ 消去零阶常数项，基函数向量为 9 维。Type-B 格式：保留零阶项（常数 1），将中心粒子的物理量作为未知的待拟合参数，基函数向量扩展至 10 维。2. 三维 LSMPS Type-A 格式详述Type-A 格式通过拟合中心粒子 $i$ 处的泰勒残差来提取空间导数。2.1 基函数向量 $p_{ij}$ 与系数向量 $c_i^\phi$对于二阶展开，9 维局部基函数列向量 $p_{ij}$ 显式定义为：$$p_{ij} = \begin{bmatrix}
\frac{x_{ij}}{r_s} \\
\frac{y_{ij}}{r_s} \\
\frac{z_{ij}}{r_s} \\
\frac{x_{ij}^2}{r_s^2} \\
\frac{y_{ij}^2}{r_s^2} \\
\frac{z_{ij}^2}{r_s^2} \\
\frac{x_{ij}y_{ij}}{r_s^2} \\
\frac{y_{ij}z_{ij}}{r_s^2} \\
\frac{z_{ij}x_{ij}}{r_s^2}
\end{bmatrix}$$对应的待求空间微商系数向量 $c_i^\phi$ 为 9 维列向量：$$c_i^\phi = \begin{bmatrix} c_1 \\ c_2 \\ c_3 \\ c_4 \\ c_5 \\ c_6 \\ c_7 \\ c_8 \\ c_9 \end{bmatrix}$$2.2 Moment 矩阵与右端项通过加权最小二乘法，导出 $9 \times 9$ 的正规方程组：$$M_i^{\text{fluid}} c_i^\phi = f_i^{\text{fluid}}$$其中，Moment 矩阵 $M_i^{\text{fluid}}$ 为 $9 \times 9$ 对称阵：$$M_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} p_{ij} p_{ij}^T$$右端项向量 $f_i^{\text{fluid}}$ 为 $9 \times 1$ 列向量：$$f_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} p_{ij} (\phi_j - \phi_i)$$2.3 核心离散算子的最终计算式通过 $c_i^\phi = (M_i^{\text{fluid}})^{-1} f_i^{\text{fluid}}$ 求解后，提取物理导数：三维梯度算子 $\nabla \phi$：$$\langle \nabla \phi \rangle_i = \frac{1}{r_s} \begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix}$$三维散度算子 $\nabla \cdot \mathbf{u}$（速度场 $\mathbf{u} = (u, v, w)^T$）：$$\langle \nabla \cdot \mathbf{u} \rangle_i = \frac{1}{r_s} \left( c_1^u + c_2^v + c_3^w \right)$$三维拉普拉斯算子 $\nabla^2 \phi$：$$\langle \nabla^2 \phi \rangle_i = \frac{2}{r_s^2} \left( c_4 + c_5 + c_6 \right)$$3. 三维 LSMPS Type-B 格式详述Type-B 格式常用于施加强约束边界（如自由面 Dirichlet 边界）或避免动量交换时的拉伸不稳定。3.1 基函数向量 $\hat{p}_{ij}$包含零阶信息的基函数扩展为 10 维列向量：$$\hat{p}_{ij} = \begin{bmatrix}
1 \\
\frac{x_{ij}}{r_s} \\
\frac{y_{ij}}{r_s} \\
\frac{z_{ij}}{r_s} \\
\frac{x_{ij}^2}{r_s^2} \\
\frac{y_{ij}^2}{r_s^2} \\
\frac{z_{ij}^2}{r_s^2} \\
\frac{x_{ij}y_{ij}}{r_s^2} \\
\frac{y_{ij}z_{ij}}{r_s^2} \\
\frac{z_{ij}x_{ij}}{r_s^2}
\end{bmatrix}$$3.2 Moment 矩阵与修正右端项求解 $10 \times 10$ 的线性方程组：$\hat{M}_i^{\text{fluid}} \hat{c}_i^\phi = \hat{f}_i^{\text{fluid}}$。矩阵计算如下：$$\hat{M}_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} \hat{p}_{ij} \hat{p}_{ij}^T$$工程零阶误差修正：为避免绝对场值导致的截断误差，引入参考基准值 $\phi_0$：$$\hat{f}_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} \hat{p}_{ij} (\phi_j - \phi_0)$$4. 壁面 Neumann 边界条件的矩阵级隐式耦合LSMPS 方法通过将法向梯度约束转化为惩罚项，直接累加到 Moment 矩阵中，实现第二类边界条件的高精度耦合。
4.1 法向梯度向量 $q_{ij}$设壁面点 $j$ 处的单位法向量为 $\mathbf{n}_j = (n_{j,x}, n_{j,y}, n_{j,z})^T$。定义方向导数向量 $q_{ij}$：$$q_{ij} = r_s \left( n_{j,x} \frac{\partial p_{ij}}{\partial x} + n_{j,y} \frac{\partial p_{ij}}{\partial y} + n_{j,z} \frac{\partial p_{ij}}{\partial z} \right)$$以 Type-A 的 9 维基函数为例，显式求导后得到 $9 \times 1$ 列向量：$$q_{ij} = \begin{bmatrix}
n_{j,x} \\
n_{j,y} \\
n_{j,z} \\
\frac{2 x_{ij} n_{j,x}}{r_s} \\
\frac{2 y_{ij} n_{j,y}}{r_s} \\
\frac{2 z_{ij} n_{j,z}}{r_s} \\
\frac{y_{ij} n_{j,x} + x_{ij} n_{j,y}}{r_s} \\
\frac{z_{ij} n_{j,y} + y_{ij} n_{j,z}}{r_s} \\
\frac{x_{ij} n_{j,z} + z_{ij} n_{j,x}}{r_s}
\end{bmatrix}$$
4.2 耦合了压力壁面边界条件的最终方程组对于壁面压力边界条件，使用第二类静水压平衡约束： $\frac{\partial P}{\partial \mathbf{n}_j} = \rho \mathbf{g} \cdot \mathbf{n}_j$。通过将边界残差并入最小二乘目标泛函，可得最终的复合方程组：$$\left( M_i^{\text{fluid}} + N_i^{\text{wall}} \right) c_i^P = f_i^{\text{fluid}} + f_i^{\text{wall}}$$其中，壁面贡献的修正矩阵 $N_i^{\text{wall}}$ 与修正右端项 $f_i^{\text{wall}}$ 分别为：
$$N_i^{\text{wall}} = \sum_{j \in \Lambda_{\text{wall}}} w_{ij}^B q_{ij} q_{ij}^T$$
$$f_i^{\text{wall}} = \sum_{j \in \Lambda_{\text{wall}}} w_{ij}^B q_{ij} \left( r_s \rho \mathbf{g} \cdot \mathbf{n}_j \right)$$
5 代入 Neumann 边界条件的算子显式公式在实际求解时，我们需要直接将压力场 $P$ 从系数向量 $c_i^P$ 的求解过程中提取出来，以构建关于 $P$ 的线性组合算子。定义中心粒子 $i$ 处耦合了壁面影响的 求逆复合矩阵 $W_i$（维度 $9 \times 9$）：$$W_i = \left( M_i^{\text{fluid}} + N_i^{\text{wall}} \right)^{-1}$$则中心粒子 $i$ 的系数向量 $c_i^P$ 可以完全展开为流体域与壁面域的两部分线性叠加：$$c_i^P = W_i \left[ \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} p_{ij} (P_j - P_i) + \sum_{j \in \Lambda_{\text{wall}}} w_{ij}^B q_{ij} (r_s \rho \mathbf{g} \cdot \mathbf{n}_j) \right]$$5.1 耦合边界的压力梯度算子 ($\nabla P$)定义梯度提取矩阵 $\mathbf{E}_{\text{grad}}$（维度 $3 \times 9$）：$$\mathbf{E}_{\text{grad}} = \frac{1}{r_s} \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$将 $c_i^P$ 代入，得到包含壁面重力支撑项的压力梯度显式表达式：$$\langle \nabla P \rangle_i = \sum_{j \in \Lambda_{\text{fluid}}} \left( \mathbf{E}_{\text{grad}} W_i w_{ij} p_{ij} \right) (P_j - P_i) + \sum_{j \in \Lambda_{\text{wall}}} \left( \mathbf{E}_{\text{grad}} W_i w_{ij}^B q_{ij} \right) (r_s \rho \mathbf{g} \cdot \mathbf{n}_j)$$5.2 耦合边界的压力拉普拉斯算子 ($\nabla^2 P$)定义拉普拉斯提取向量 $\mathbf{m}_{\text{lap}}^T$（维度 $1 \times 9$）：$$\mathbf{m}_{\text{lap}}^T = \frac{2}{r_s^2} \begin{bmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 0 \end{bmatrix}$$将 $c_i^P$ 代入，得到压力拉普拉斯算子的显式线性组合式：$$\langle \nabla^2 P \rangle_i = \sum_{j \in \Lambda_{\text{fluid}}} \left( \mathbf{m}_{\text{lap}}^T W_i w_{ij} p_{ij} \right) (P_j - P_i) + \sum_{j \in \Lambda_{\text{wall}}} \left( \mathbf{m}_{\text{lap}}^T W_i w_{ij}^B q_{ij} \right) (r_s \rho \mathbf{g} \cdot \mathbf{n}_j)$$此式清晰表明：壁面的 Neumann 边界条件已经不再作为传统方法的显式推断项，而是被吸收成拉普拉斯算子内部的常数源项偏移。  
6. 压力泊松方程 (PPE) 的全局离散形式在不可压缩流体的投影法求解中，压力泊松方程（PPE）的连续形式为：$$\nabla^2 P = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$$其中 $\mathbf{u}^*$ 为不含压力梯度修正的中间预测速度场。  
6.1 构建全局线性方程组 ($AP=b$)针对计算域内的每一个流体粒子 $i$，将其拉普拉斯算子展开式代入 PPE 中。为了形成形如 $\sum A_{ij} P_j = b_i$ 的标准线性系统，我们需要提取未知数 $P_j$ 和 $P_i$ 的系数。  
6.2 系数矩阵 $A_{ij}$ 的提取对于流体内部粒子 $i$ 及其流体邻域粒子 $j \in \Lambda_{\text{fluid}}$，稀疏矩阵的非对角线系数 $A_{ij}$ 为：$$A_{ij} = \mathbf{m}_{\text{lap}}^T W_i w_{ij} p_{ij} \quad (j \ne i)$$主对角线系数 $A_{ii}$ 则等于流体邻居对角系数之和的负值：$$A_{ii} = - \sum_{j \in \Lambda_{\text{fluid}}, j \ne i} A_{ij}$$6.3 右边源项 $b_i$ 的计算式除了速度散度主导的体积变化源项外，我们在 5.2 节中推导出的壁面边界修正项（常数标量）必须移至方程等式右边（作为已知源项处理）。最终中心粒子 $i$ 对应的方程右端项 $b_i$ 定义为：$$b_i = \frac{\rho}{\Delta t} \langle \nabla \cdot \mathbf{u}^* \rangle_i - \sum_{j \in \Lambda_{\text{wall}}} \left( \mathbf{m}_{\text{lap}}^T W_i w_{ij}^B q_{ij} \right) (r_s \rho \mathbf{g} \cdot \mathbf{n}_j)$$