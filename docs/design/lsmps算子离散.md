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
\end{bmatrix}$$3.2 Moment 矩阵与修正右端项求解 $10 \times 10$ 的线性方程组：$\hat{M}_i^{\text{fluid}} \hat{c}_i^\phi = \hat{f}_i^{\text{fluid}}$。矩阵计算如下：$$\hat{M}_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} \hat{p}_{ij} \hat{p}_{ij}^T$$工程零阶误差修正：为避免绝对场值导致的截断误差，引入参考基准值 $\phi_0$：$$\hat{f}_i^{\text{fluid}} = \sum_{j \in \Lambda_{\text{fluid}}} w_{ij} \hat{p}_{ij} (\phi_j - \phi_0)$$4. 壁面 Neumann 边界条件的矩阵级隐式耦合LSMPS 方法通过将法向梯度约束转化为惩罚项，直接累加到 Moment 矩阵中，实现第二类边界条件的高精度耦合。4.1 法向梯度向量 $q_{ij}$设壁面点 $j$ 处的单位法向量为 $\mathbf{n}_j = (n_{j,x}, n_{j,y}, n_{j,z})^T$。定义方向导数向量 $q_{ij}$：$$q_{ij} = r_s \left( n_{j,x} \frac{\partial p_{ij}}{\partial x} + n_{j,y} \frac{\partial p_{ij}}{\partial y} + n_{j,z} \frac{\partial p_{ij}}{\partial z} \right)$$以 Type-A 的 9 维基函数为例，显式求导后得到 $9 \times 1$ 列向量：$$q_{ij} = \begin{bmatrix}
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
4.2 对于壁面压力边界条件，使用第二类边界条件： $\frac{\partial p}{\partial \mathbf{n_j}} = \rho \mathbf{g} \cdot \mathbf{n}_j$。通过将边界残差并入最小二乘目标泛函，可得最终的耦合方程组：$$\left( M_i^{\text{fluid}} + N_i^{\text{wall}} \right) c_i^\phi = f_i^{\text{fluid}} + f_i^{\text{wall}}$$其中，壁面贡献的修正矩阵 $N_i^{\text{wall}}$ 与右端项 $f_i^{\text{wall}}$ 为：$$N_i^{\text{wall}} = \sum_{j \in \Lambda_{\text{wall}}} w_{ij}^B q_{ij} q_{ij}^T$$
$$f_i^{\text{wall}} = \sum_{j \in \Lambda_{\text{wall}}} w_{ij}^B q_{ij} \left( r_s \rho \mathbf{g} \cdot \mathbf{n_j} \right)$$
直接对上述复合后的 $9 \times 9$ （或 $10 \times 10$） 矩阵求逆解出的 $c_i^\phi$，将自动满足边界的法向导数要求。