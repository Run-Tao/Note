---
title: Level 7 机器学习中的线性代数
type: concept
subject: linear-algebra
level: Level 7
tags:
  - mathematics
  - linear-algebra
  - machine-learning
created: 2026-08-23
source: Learning_Roadmap.md Level 7 推荐资料
---

# Level 7 机器学习中的线性代数

> 本章是**第一次"反向"使用线性代数**：前六章我们学线代本身，这一章让线代为机器学习服务。你会发现机器学习里几乎每个名词——数据、损失、正则化、决策边界、降维——背后都有一个线代对象：矩阵、范数、投影、特征分解。
>
> 知识主线：**数据矩阵 → 最小二乘与正规方程 → 梯度下降的矩阵形式 → 岭回归（正则化）→ 分类（逻辑回归 / Softmax / LDA）→ PCA 与 K-means → 核方法 → 高斯分布与协方差**

## 学习目标

完成本章后，你应该能够：

- 把"数据集"看作设计矩阵，把"一条样本"看作行向量，"一个特征"看作列向量；
- 理解正规方程 $X^TX\mathbf{w}=X^T\mathbf{y}$ 的几何含义：残差垂直于列空间（投影）；
- 理解岭回归 $(X^TX+\lambda I)\mathbf{w}=X^T\mathbf{y}$ 为什么能"救回"奇异矩阵，以及它如何从奇异值视角收缩解；
- 看懂梯度下降的向量化写法，知道一次批量更新用了几次矩阵乘法；
- 理解决策边界、法向量、投影、协方差特征分解在分类与降维中的角色；
- 能用一句话解释核技巧为什么"免费获得高维"；
- 读懂经典模型（回归、SVM、PCA）核心推导中的每一步线代操作。

---

## 第一部分：数据的矩阵视角

### 1. 设计矩阵：一行一样本

**设计矩阵 / 数据矩阵（design matrix）** $X$ 是机器学习里数据的第一种"线代形态"：

$$
X=\begin{pmatrix}
- & \mathbf{x}_1^T & -\\
- & \mathbf{x}_2^T & -\\
& \vdots & \\
- & \mathbf{x}_n^T & -
\end{pmatrix}\in\mathbb{R}^{n\times d}
$$

- **行** = 样本（sample）：第 $i$ 行 $\mathbf{x}_i\in\mathbb{R}^d$ 是一条样本的特征向量（Level 1 的"数据即向量"正式落地）；
- **列** = 特征（feature）：第 $j$ 列是所有样本在第 $j$ 个特征上的取值；
- 通常再补一列全 1（截距项 bias），此时 $X\in\mathbb{R}^{n\times (d+1)}$，参数向量 $\mathbf{w}\in\mathbb{R}^{d+1}$。

**批量计算的本质**：对 $n$ 条样本做同一线性变换 $\mathbf{x}_i\mapsto\mathbf{w}^T\mathbf{x}_i$，等价于一次矩阵乘法

$$
\hat{\mathbf{y}}=X\mathbf{w}\in\mathbb{R}^n
$$

即"预测全部样本"= **一次矩阵–向量乘法**。这正是 Level 1 预告的"矩阵乘法 = 批量计算"：循环交给线性代数，速度交给 BLAS。

> [!tip] 行向量还是列向量？
> 约定"行 = 样本"是 ML 惯例（scikit-learn、Pandas 均如此）。若反过来"列 = 样本"，则预测写作 $\mathbf{w}^TX^T$——公式形态不同，但数学等价。**先统一约定，再读公式**。

### 2. 特征缩放与标准化：对列做向量运算

特征尺度差异巨大时（年龄 0–100 与收入 0–10 万），模型会被"大尺度"特征主导。**标准化（standardization / z-score）**对每一列做"平移 + 缩放"：

$$
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
$$

其中 $\mu_j$ 是第 $j$ 列的均值、$\sigma_j$ 是标准差。写成向量运算：$X_{\text{std}}=(X-\mathbf{1}\boldsymbol{\mu}^T)\operatorname{diag}(1/\boldsymbol{\sigma})$——即"整体平移均值向量 + 按列缩放（对角阵右乘）"，全是 Level 1 的矩阵运算。

**为什么重要**：

- **梯度下降**：特征尺度一致时，损失函数的等值线接近圆，梯度方向直指圆心，收敛快（§6）；
- **正则化**：$\|\mathbf{w}\|^2$ 惩罚对不同特征"一视同仁"，若特征尺度差 100 倍，惩罚就失真——标准化让惩罚公平；
- **距离类方法**（K-means、KNN、RBF 核）：距离 $\|\mathbf{x}_i-\mathbf{x}_j\|$ 会被大尺度特征支配，必须标准化。

> [!warning] 关键细节
> 标准化所用的 $\mu_j,\sigma_j$ **只能从训练集计算**，再应用到验证集/测试集。用全数据算均值属于"数据泄露"（data leakage）——测试集的信息偷偷进入了训练流程。

### 3. 训练 / 验证 / 测试：集合的划分

把样本集合 $D$（$n$ 个样本）划分成互不相交的三个子集：

$$
D=D_{\text{train}}\ \dot\cup\ D_{\text{val}}\ \dot\cup\ D_{\text{test}}
$$

这是 Level 0 集合思想的一次真实应用：**不相交**（$\dot\cup$）保证"用没见过的数据检验"。训练集用于拟合参数（解 $X^T X\mathbf{w}=X^T\mathbf{y}$ 等），验证集用于调超参数（如 $\lambda$、学习率），测试集只做最终评估。超参数调过头也会过拟合——验证集就是"第二道防线"。

---

## 第二部分：线性回归与最小二乘

### 4. 最小二乘回顾：正规方程的几何

**问题**：找 $\mathbf{w}$ 使预测 $\hat{\mathbf{y}}=X\mathbf{w}$ 尽量接近观测 $\mathbf{y}$，即最小化残差平方和：

$$
\min_{\mathbf{w}}\ \|X\mathbf{w}-\mathbf{y}\|^2
$$

**几何读法（Level 4 的投影）**：$\mathbf{y}\in\mathbb{R}^n$ 一般不在列空间 $\operatorname{Col}(X)$ 里（方程 $X\mathbf{w}=\mathbf{y}$ 无解，超定系统）。最佳近似就是 $\mathbf{y}$ 在 $\operatorname{Col}(X)$ 上的**正交投影**：残差 $\mathbf{r}=\mathbf{y}-X\hat{\mathbf{w}}$ 垂直于列空间：

$$
X^T(\mathbf{y}-X\hat{\mathbf{w}})=\mathbf{0}\quad\Longrightarrow\quad \boxed{\,X^TX\mathbf{w}=X^T\mathbf{y}\,}
$$

这就是**正规方程（normal equations）**。$\hat{\mathbf{y}}=X\hat{\mathbf{w}}$ 是 $\mathbf{y}$ 到列空间的投影，$X^T\mathbf{y}$ 是"投影的代数化身"（把 $\mathbf{y}$ 打到各列方向的系数）。若 $X$ 列满秩，$X^TX$ 可逆，解为

$$
\mathbf{w}^*=(X^TX)^{-1}X^T\mathbf{y}=X^{+}\mathbf{y}
$$

其中 $X^{+}$ 是伪逆（Level 6）——正规方程与伪逆是同一件事的两种写法。

> [!example] 简单例子
> 拟合 $y=w_1x_1+w_2x_2$，$X=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$，$\mathbf{y}=(1,2,4)^T$。则 $X^TX=\begin{pmatrix}2&1\\1&2\end{pmatrix}$，$X^T\mathbf{y}=(5,6)^T$，解得 $\mathbf{w}^*=\left(\tfrac43,\tfrac73\right)^T$。验证投影性质：$\hat{\mathbf{y}}=X\mathbf{w}^*=\left(\tfrac43,\tfrac73,\tfrac{11}{3}\right)^T$，残差 $\mathbf{r}=\left(-\tfrac13,-\tfrac13,\tfrac13\right)^T$ 满足 $X^T\mathbf{r}=\mathbf{0}$——残差确实垂直于列空间。

### 5. $X^TX$ 的结构：对称与半正定

$X^TX$ 不是随便一个矩阵，它有两个由构造保证的性质（Level 5 谱定理的用武之地）：

1. **对称**：$(X^TX)^T=X^T(X^T)^T=X^TX$；
2. **半正定**：对任意 $\mathbf{v}$，$\mathbf{v}^T(X^TX)\mathbf{v}=(X\mathbf{v})^T(X\mathbf{v})=\|X\mathbf{v}\|^2\ge 0$。若 $X$ 列满秩则正定，所有特征值 $>0$，可逆。

**意义**：对称 → 可正交对角化（$X^TX=Q\Lambda Q^T$）；半正定 → 特征值非负，损失函数 $\|X\mathbf{w}-\mathbf{y}\|^2$ 关于 $\mathbf{w}$ 是凸的（碗形，只有一个谷底）。这保证梯度下降"怎么走都能到"同一全局最优——**凸性是线性回归好训的根本原因**。

> [!warning] 秩不足：特征多、样本少
> $n<d$（样本少于特征）时 $X$ 列必线性相关，$X^TX$ 奇异，正规方程无唯一解——这就是欠定系统（Level 2 预告过）。**正则化（第三部分）正是为此时准备的**。

### 6. 梯度下降的矩阵形式：向量化批量计算

正规方程一步到位但要解线性系统（$O(nd^2)$ 以上），数据大时改用**梯度下降**。损失取平均形式：

$$
J(\mathbf{w})=\frac{1}{2n}\|X\mathbf{w}-\mathbf{y}\|^2,\qquad
\nabla J(\mathbf{w})=\frac{1}{n}X^T(X\mathbf{w}-\mathbf{y})
$$

更新规则：

$$
\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla J(\mathbf{w})
$$

**向量化的三步**（这就是"批量梯度下降"的一步）：

1. 前向：$\hat{\mathbf{y}}=X\mathbf{w}$（1 次矩阵–向量乘）；
2. 残差：$\mathbf{e}=\hat{\mathbf{y}}-\mathbf{y}$（向量减法）；
3. 梯度：$X^T\mathbf{e}$（1 次矩阵–向量乘）。

整个数据集被"打包"进两个矩阵乘，循环被消灭——这正是"向量化计算"（vectorization）的含义：**把对样本的循环改写为对矩阵的运算**，由 BLAS/GPU 并行执行。逻辑回归的梯度 $X^T(\mathbf{p}-\mathbf{y})$ 与这里结构完全相同（"特征转置 × 残差"），记住这一个模板，后续模型全是它的变体。

### 7. 计算图：前向与反向的矩阵视角（预告）

深度学习的训练流程可以画成**计算图（computational graph）**：节点是张量，边是运算。线性回归的计算图只有一条链：

$$
X,\ \mathbf{w}\ \xrightarrow{\text{matmul}}\ \hat{\mathbf{y}}\ \xrightarrow{\text{sub}}\ \mathbf{e}\ \xrightarrow{\text{sq, mean}}\ J
$$

前向传播按图正向算 $J$；反向传播按**矩阵链式法则**把 $\partial J/\partial \mathbf{w}$ 逐层传回（每一步都是矩阵乘/转置）。这是 Level 8 反向传播的雏形——本章先"看见"梯度是矩阵表达式，下一章再严格拆解。

---

## 第三部分：正则化

### 8. 岭回归：推导与解

**岭回归（ridge regression）**在损失上加 $\mathbf{w}$ 的 L2 范数惩罚：

$$
J(\mathbf{w})=\|X\mathbf{w}-\mathbf{y}\|^2+\lambda\|\mathbf{w}\|^2,\qquad \lambda>0
$$

求导置零：

$$
\nabla J=2X^T(X\mathbf{w}-\mathbf{y})+2\lambda\mathbf{w}=\mathbf{0}\ \Longrightarrow\ \boxed{\,(X^TX+\lambda I)\mathbf{w}=X^T\mathbf{y}\,}
$$

**为什么能救奇异矩阵**：$X^TX$ 的特征值 $\lambda_i\ge 0$，而 $X^TX+\lambda I$ 的特征值是 $\lambda_i+\lambda>0$——**正定、必然可逆**（Level 5：特征值全正 ⇔ 正定 ⇔ 可逆）。即使样本少于特征、$X^TX$ 奇异，岭回归也总有唯一解。这是欠定问题的标准解药（Level 2 预告的"正则化动机"在此兑现）。

### 9. 几何直觉：约束球与误差椭球

岭回归等价于带约束的最小二乘：

$$
\min_{\mathbf{w}}\|X\mathbf{w}-\mathbf{y}\|^2\quad\text{s.t.}\quad\|\mathbf{w}\|^2\le t
$$

（$\lambda$ 与 $t$ 一一对应，拉格朗日乘子）。几何上：

- 无约束最小二乘的解 $\mathbf{w}_{\text{ols}}$ 是误差等值线椭球的中心；
- 约束 $\|\mathbf{w}\|^2\le t$ 是以原点为球心的**球**；
- 最优解在**球与最小可及椭球相切**处——切点处椭球法向与球法向平行，即 $X^T(X\mathbf{w}-\mathbf{y})=-\lambda\mathbf{w}$，正是上面的方程。

一句话：**岭回归 = 把解从"远离原点的最优"拉回"离原点不太远的最优"**——远离原点的方向通常是噪声方向。

### 10. 奇异值视角：收缩的方向

用 SVD $X=U\Sigma V^T$（Level 6）看岭回归，能看清它"收缩了什么"：

$$
\mathbf{w}_{\text{ridge}}=\sum_{i=1}^{d}\frac{\sigma_i}{\sigma_i^2+\lambda}\,(u_i^T\mathbf{y})\,v_i
=\sum_{i=1}^{d}\underbrace{\frac{\sigma_i^2}{\sigma_i^2+\lambda}}_{\text{收缩因子}}\cdot\frac{u_i^T\mathbf{y}}{\sigma_i}\,v_i
$$

与普通最小二乘 $\mathbf{w}_{\text{ols}}=\sum_i \frac{u_i^T\mathbf{y}}{\sigma_i}v_i$ 相比，**每个奇异方向 $v_i$ 上的分量乘上因子 $\sigma_i^2/(\sigma_i^2+\lambda)\in(0,1)$**：

- 大奇异值方向（信号方向）：因子 ≈ 1，几乎不动；
- 小奇异值方向（噪声方向）：因子 → 0，被强烈压缩。

这就是正则化的本质：**沿数据方差小的方向收缩**，而不是均匀收缩。$\lambda$ 越大，收缩越狠（偏差增大、方差减小——偏差-方差权衡的代数形态）。

### 11. L1 与 L2：稀疏 vs 收缩

**Lasso（L1 正则化）**：$\min\|X\mathbf{w}-\mathbf{y}\|^2+\lambda\|\mathbf{w}\|_1$，其中 $\|\mathbf{w}\|_1=\sum_i|w_i|$。

二维下看约束区域：L2 是圆盘，L1 是**菱形（顶点在坐标轴上）**。误差椭球与约束区域的切点：

- 与圆相切：切点几乎不可能恰好落在坐标轴上 → **L2 只收缩、不置零**；
- 与菱形相切：顶点是"角"，切点**很容易落在顶点** → 该坐标分量为零 → **L1 产生稀疏解**（特征选择）。

> [!tip] 一句话区分
> L2（Ridge）：把权重按比例缩小，所有特征都保留；L1（Lasso）：把不重要的权重直接压成 0，自动做特征选择。几何原因：菱形有"角"，圆没有。

---

## 第四部分：分类

### 12. 逻辑回归：决策边界与法向量

**逻辑回归（logistic regression）**把线性得分过一层 sigmoid 变成概率：

$$
P(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^T\mathbf{x}+b)=\frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}
$$

**决策边界是超平面** $\mathbf{w}^T\mathbf{x}+b=0$，而权重向量 $\mathbf{w}$ 正是这个超平面的**法向量**（Level 1 的解析几何：平面方程系数即法向量）——它指向"类别 1"一侧，$\|\mathbf{w}\|$ 决定边界两侧的"陡峭程度"。训练用交叉熵损失，其批量梯度仍是模板 $X^T(\mathbf{p}-\mathbf{y})$（$\mathbf{p}$ 为预测概率向量）。

> [!example] 线代直觉
> 若 $\mathbf{w}=(2,-1)^T,\ b=1$，边界是直线 $2x_1-x_2+1=0$。任取边界上两点，其差向量 $\Delta$ 满足 $\mathbf{w}^T\Delta=0$——**$\mathbf{w}$ 垂直于边界上所有方向**，这就是"法向量"。

### 13. Softmax：多类分类的线性得分

多类问题把"每类一个得分"变成概率分布：

$$
P(y=k\mid\mathbf{x})=\frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^T\mathbf{x}}}
$$

**几何读法**：每个类别 $k$ 有一个权重向量 $\mathbf{w}_k$（类别"模板"），预测 = 把 $\mathbf{x}$ 投影到 $K$ 个模板方向，取得分最高者：$\hat{y}=\arg\max_k \mathbf{w}_k^T\mathbf{x}$。分类就是**在 $K$ 个方向上的投影比赛**。Softmax 分类器即多类逻辑回归，是深度网络输出层的标准组件。

### 14. LDA：投影的思想

**线性判别分析（Linear Discriminant Analysis, LDA）**找一条投影方向 $\mathbf{w}$，把数据投成一维后**类间距离尽可能大、类内散度尽可能小**：

$$
\max_{\mathbf{w}}\ \frac{[\mathbf{w}^T(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)]^2}{\mathbf{w}^TS_W\mathbf{w}},\qquad \mathbf{w}^*\propto S_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)
$$

其中 $\boldsymbol{\mu}_k$ 是类别均值，$S_W$ 是类内散度矩阵（各类协方差的加权和）。$\mathbf{w}^*$ 的推导用到了"瑞利商最大化 → 广义特征值问题"（Level 5 的进阶应用）。

> [!tip] LDA 与 PCA 的分工
> PCA 只找**方差最大**的方向（无监督，不关心类别）；LDA 找**最能分开类别**的方向（有监督）。同一个投影数学，两个不同的优化目标——"投影"这个线代动作可以服务于完全不同的任务。

---

## 第五部分：降维与聚类

### 15. PCA：协方差矩阵的特征分解

**主成分分析（Principal Component Analysis, PCA）**：把 $d$ 维数据投影到方差最大的 $k$ 个正交方向上。步骤：

1. **中心化**：$X_c=X-\mathbf{1}\boldsymbol{\mu}^T$（每列减均值）；
2. 计算样本协方差矩阵 $C=\frac{1}{n}X_c^TX_c\in\mathbb{R}^{d\times d}$；
3. **特征分解** $C=Q\Lambda Q^T$（$C$ 对称半正定，谱定理保证 $Q$ 正交、$\Lambda$ 对角非负）：
   - 特征向量 $q_1,\dots,q_d$ = 主方向（数据变化最大的方向）；
   - 特征值 $\lambda_1\ge\cdots\ge\lambda_d$ = 沿各主方向的方差；
4. 投影：$Z=X_cQ_k\in\mathbb{R}^{n\times k}$（$Q_k$ 取前 $k$ 列）。

**为什么特征向量是主方向**：沿方向 $\mathbf{v}$（$\|\mathbf{v}\|=1$）的投影方差是 $\mathbf{v}^TC\mathbf{v}$，最大化它得到 $\max_{\|\mathbf{v}\|=1}\mathbf{v}^TC\mathbf{v}=\lambda_1$，取到 $\mathbf{v}=q_1$——这是**瑞利商**（Rayleigh quotient）的结论，也是 Level 5 特征值意义的直接应用。**解释方差比例**：前 $k$ 个主成分解释 $\frac{\lambda_1+\cdots+\lambda_k}{\lambda_1+\cdots+\lambda_d}$。

### 16. SVD 实现：数值稳定

实际中不用"先算 $X^TX$ 再特征分解"（平方会放大数值误差、$d$ 大时开销高），而是直接对中心化矩阵做 SVD：

$$
X_c=U\Sigma V^T\ \Longrightarrow\ C=\frac{1}{n}X_c^TX_c=\frac{1}{n}V\Sigma^2 V^T
$$

**$V$ 的列就是主方向，奇异值平方 = 特征值（差 $\frac1n$ 因子）**。主成分得分 $Z=X_cV_k$。这就是"PCA 用 SVD 实现"（Level 6 预告兑现）：SVD 不要求 $X^TX$ 可逆、对病态数据更稳。

> [!example] 应用出处
> scikit-learn 的 `PCA` 默认用 `svd` 求解器；图像压缩、去噪、人脸识别（特征脸 Eigenfaces = 人脸数据 PCA 主方向）都是这一节内容的直接应用。

### 17. K-means：质心与距离

**K-means** 把 $n$ 个样本分到 $K$ 个簇，最小化"样本到质心距离平方和"：

$$
\min_{\{C_k\},\{\boldsymbol{\mu}_k\}}\ \sum_{k=1}^{K}\sum_{\mathbf{x}_i\in C_k}\|\mathbf{x}_i-\boldsymbol{\mu}_k\|^2
$$

交替优化两步（每次迭代都单调不增，收敛到局部最优）：

1. **指派**：$\mathbf{x}_i$ 归入距离最近的质心 $k^*=\arg\min_k\|\mathbf{x}_i-\boldsymbol{\mu}_k\|$——这是**范数 / 距离度量**的应用；
2. **更新**：$\boldsymbol{\mu}_k=\frac{1}{|C_k|}\sum_{\mathbf{x}_i\in C_k}\mathbf{x}_i$——簇内平均。

**为什么质心是均值**：对固定簇，$\sum_i\|\mathbf{x}_i-\mathbf{c}\|^2$ 的最小值点恰是均值 $\boldsymbol{\mu}$（方差分解 $\sum_i\|\mathbf{x}_i-\mathbf{c}\|^2=\sum_i\|\mathbf{x}_i-\boldsymbol{\mu}\|^2+|C_k|\|\mathbf{c}-\boldsymbol{\mu}\|^2$，第二项非负）。距离度量可换（如马氏距离 §22），但欧氏距离 + 均值是默认组合。

---

## 第六部分：核方法与 SVM

### 18. 升维以线性化

有些数据在低维**线性不可分**，但映射到高维后**线性可分**。特征映射（feature map）$\varphi:\mathbb{R}^d\to\mathbb{R}^D$（$D\gg d$）把非线性边界"拉直"成高维超平面。

> [!example] XOR 问题
> 二维平面上的异或（XOR）数据线性不可分；映射 $\varphi(x_1,x_2)=(x_1,x_2,x_1x_2)$ 后，四个点变为三维点，存在分离平面 $x_1+x_2-2x_1x_2=0.5$——**在原空间是曲线的决策边界，在特征空间是一条直线**。这就是"升维以线性化"。

**代价**：显式升维会让数据维数爆炸（二次特征就有 $\sim d^2/2$ 维），维度灾难随之而来——除非我们不真的算 $\varphi$。

### 19. 核技巧：不显式升维

**核函数（kernel）**是两个样本在特征空间的内积：

$$
k(\mathbf{x},\mathbf{x}')=\langle\varphi(\mathbf{x}),\varphi(\mathbf{x}')\rangle
$$

**核技巧（kernel trick）**：若算法只用内积（SVM、岭回归、PCA 均可改写为内积形式），就把内积 $\langle\mathbf{x},\mathbf{x}'\rangle$ 替换成 $k(\mathbf{x},\mathbf{x}')$——**高维空间从未显式构建，但等价于在那里做线性模型**。

- 多项式核 $k(\mathbf{x},\mathbf{x}')=(\mathbf{x}^T\mathbf{x}'+1)^2$：对应包含 $x_i x_j$ 等二次项的特征映射；
- 高斯核（RBF）$k(\mathbf{x},\mathbf{x}')=e^{-\|\mathbf{x}-\mathbf{x}'\|^2/2\sigma^2}$：对应**无限维**特征映射（泰勒展开可见），却只需 $O(d)$ 计算。

> [!note] 核 = 高维内积
> 数学上，只要 $k$ 满足 Mercer 条件（对应的"核矩阵"半正定），就存在某个 $\varphi$ 使 $k(\mathbf{x},\mathbf{x}')=\langle\varphi(\mathbf{x}),\varphi(\mathbf{x}')\rangle$。核方法 = **"隐式升维"**：效果上拥有了高维，代价上只付低维。

### 20. SVM 对偶的直觉

**支持向量机（SVM）**找间隔（margin）最大的分离超平面：约束 $y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1$ 下最小化 $\frac12\|\mathbf{w}\|^2$（间隔 $=2/\|\mathbf{w}\|$，最大间隔 = 最小范数）。

**对偶问题（dual problem）**的结论（拉格朗日对偶，推导留待建模库）：

$$
\mathbf{w}=\sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i,\qquad
\text{决策：}\ \operatorname{sign}\!\Big(\sum_{i}\alpha_i y_i\langle\mathbf{x}_i,\mathbf{x}\rangle+b\Big)
$$

两个关键线代事实：

1. **$\mathbf{w}$ 是训练样本的线性组合**，且只有 $\alpha_i>0$ 的样本起作用——**支持向量（support vector）**：决策边界只由贴着边界的少数点决定；
2. **对偶目标只含内积** $\langle\mathbf{x}_i,\mathbf{x}_j\rangle$ → 直接替换为核函数 $k(\mathbf{x}_i,\mathbf{x}_j)$，得到**核 SVM**：高维特征空间的线性分类器，复杂度只依赖样本数 $n$。

---

## 第七部分：高斯分布与协方差

### 21. $\Sigma$ 的几何：特征分解解读椭圆

**协方差矩阵（covariance matrix）** $\Sigma$ 是"随机向量偏离均值的期望外积"（Level 1 外积的概率版）：

$$
\Sigma=\mathbb{E}\big[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T\big]\in\mathbb{R}^{d\times d}
$$

多元高斯分布 $p(\mathbf{x})\propto\exp\!\big(-\tfrac12(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)$ 的**等高线是椭圆**：

$$
(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})=c
$$

对对称正定 $\Sigma$ 做特征分解 $\Sigma=Q\Lambda Q^T$：

- **特征向量 $q_i$ = 椭圆的主轴方向**；
- **特征值 $\lambda_i$ = 沿该轴的方差**，轴半长 $\propto\sqrt{\lambda_i}$；
- $\lambda_1=\lambda_2$ 时椭圆退化为圆（各向同性）；$\Sigma\propto I$ 即"各方向等方差"。

这与 PCA 是同一套数学：**PCA 的 $C$ 就是"标准化后的 $\Sigma$"，PCA 主方向就是高斯的椭圆主轴**。特征分解（Level 5）在此把概率分布"翻译"成了几何图形。

### 22. 马氏距离：坐标变换后的欧氏距离

**马氏距离（Mahalanobis distance）**：

$$
d_\Sigma(\mathbf{x},\boldsymbol{\mu})=\sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}
$$

**白化（whitening）**变换 $z=\Sigma^{-1/2}(\mathbf{x}-\boldsymbol{\mu})$（其中 $\Sigma^{-1/2}=Q\Lambda^{-1/2}Q^T$）先把数据"拉伸成各向同性"，再算欧氏距离：

$$
d_\Sigma(\mathbf{x},\boldsymbol{\mu})=\|z\|=\|\Sigma^{-1/2}(\mathbf{x}-\boldsymbol{\mu})\|
$$

**直觉**：欧氏距离把每个方向一视同仁；马氏距离沿主轴方向按方差"打折"——方差大的方向（数据本来就散）容忍更大的偏差，方差小的方向（数据紧凑）要求更贴近。它是"以数据形状为尺子的距离"，用于异常检测、K-means 的距离度量、LDA 的 $S_W^{-1}$ 也同此理。

### 23. 线代与概率的衔接：最小二乘 = 高斯噪声下的最大似然

假设 $y_i=\mathbf{w}^T\mathbf{x}_i+\varepsilon_i$，噪声 $\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$ 独立。则数据的似然为

$$
p(\mathbf{y}\mid X,\mathbf{w})\propto\exp\!\Big(-\frac{1}{2\sigma^2}\|X\mathbf{w}-\mathbf{y}\|^2\Big)
$$

最大化似然 $\Longleftrightarrow$ 最小化 $\|X\mathbf{w}-\mathbf{y}\|^2$——**最小二乘正是高斯噪声假设下的最大似然估计（MLE）**。而岭回归 = 高斯噪声 + 高斯先验 $\mathbf{w}\sim\mathcal{N}(\mathbf{0},\frac{\sigma^2}{\lambda}I)$ 的最大后验估计（MAP）。**一个公式，两门语言**：统计学家看"噪声分布"，线代视角看"投影与范数"。这正是 Level 7 作为"线代 × 概率"衔接点的意义。

---

## 常见误区汇总

1. **"设计矩阵每一行是特征"** → 行 = 样本、列 = 特征；$\hat{\mathbf{y}}=X\mathbf{w}$ 的一次乘法同时预测 $n$ 条样本；
2. **"正规方程是背下来的公式"** → 它是"残差垂直于列空间"这一几何条件的代数形式：$X^T\mathbf{r}=\mathbf{0}$；最小二乘解就是投影；
3. **"正则化只是防过拟合的小技巧"** → 岭回归等价于带约束的优化，从奇异值看是**沿小奇异值方向收缩**；它还让奇异矩阵变得可逆；
4. **"标准化只是让数字好看"** → 特征尺度影响梯度下降收敛速度、正则化的公平性、距离类方法的结果；且 $\mu,\sigma$ 只能用训练集算；
5. **"PCA 是挑几个重要特征"** → PCA 是**旋转坐标系到方差最大方向再丢弃低方差坐标**，新坐标是原特征的线性组合，不是特征子集；
6. **"L1 与 L2 只是惩罚系数不同"** → 约束区域的形状不同（菱形有角、圆无角）导致**稀疏 vs 收缩**的本质区别；
7. **"核方法就是升维"** → 显式升维会维度爆炸；核技巧是**不构造高维空间就计算高维内积**，复杂度只依赖样本数；
8. **"协方差矩阵只在对角线上有意义"** → 非对角元编码特征间相关性，特征分解给出椭球主轴——对角线视角会丢掉全部"形状"信息。

---

## 知识结构图

```mermaid
graph TD
    A[设计矩阵 X<br/>n 样本 × d 特征] --> B[特征缩放与标准化]
    A --> C[线性回归：最小二乘]
    C --> D[正规方程 XᵀXw = Xᵀy<br/>残差 ⊥ 列空间]
    C --> E[梯度下降矩阵形式<br/>∇J = Xᵀ(Xw − y)/n]
    E --> F[计算图 / 反向传播<br/>Level 8]
    D --> G[岭回归 XᵀX + λI<br/>约束球 × 误差椭球]
    G --> H[SVD 视角：收缩小奇异值方向]
    G --> I[L1 稀疏 vs L2 收缩]
    A --> J[逻辑回归 / Softmax<br/>决策边界 w 为法向量]
    J --> K[LDA：投影类间/类内]
    A --> L[PCA：C = XᵀX 特征分解]
    L --> M[SVD 实现：X = UΣVᵀ]
    A --> N[K-means：质心 = 均值<br/>距离 = 范数]
    J --> O[SVM：w = Σαᵢyᵢxᵢ]
    O --> P[核技巧：k(x,x′) = ⟨φ(x),φ(x′)⟩]
    P --> Q[升维以线性化]
    L --> R[高斯分布：Σ = QΛQᵀ<br/>椭圆主轴 = 特征向量]
    R --> S[马氏距离：白化后欧氏距离]
    R --> T[最小二乘 = 高斯噪声下 MLE<br/>概率 × 线代]
```

## 思考问题

1. 为什么正规方程的解 $\hat{\mathbf{w}}$ 一定是 $\mathbf{y}$ 到 $\operatorname{Col}(X)$ 的投影？从"残差垂直于列空间"出发给出几何解释。
2. $X^TX$ 为什么必然对称且半正定？当 $X$ 列满秩时它为什么可逆？这与你学过的哪个定理（Level 5）直接相关？
3. 岭回归为什么能让 $X^TX$ 奇异时仍有唯一解？从特征值角度解释 $(X^TX+\lambda I)$ 的可逆性。
4. 特征尺度相差很大时，梯度下降为什么收敛慢？试着从损失等值线的形状（长椭球 vs 圆）说明。
5. LDA 的目标函数里，分子分母分别衡量什么？为什么 $\mathbf{w}^*\propto S_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$ 有"白化"的味道（联系马氏距离）？
6. PCA 为什么要先中心化？不中心化时第一个主成分会被什么主导？
7. K-means 的质心为什么是簇内均值？用 $\sum_i\|\mathbf{x}_i-\mathbf{c}\|^2$ 的方差分解说明。
8. （预告）神经网络的计算图里，"反向传播"其实就是矩阵链式法则：若损失 $J$ 经 $z=W\mathbf{a}$ 传递，$\partial J/\partial W$ 应该是什么形状的矩阵？这将是 Level 8 的核心问题。

## 总结

**Level 7 的知识网络**：

**数据矩阵 X（行=样本、列=特征）→ 标准化（按列向量运算）→ 最小二乘（投影几何 + 正规方程）→ 梯度下降（矩阵批量计算 + 计算图）→ 岭回归（约束球几何 + SVD 收缩 + L1/L2 对比）→ 分类（决策边界法向量 / Softmax 模板方向 / LDA 投影）→ 降维（PCA = 协方差特征分解 = SVD）→ 聚类（K-means 质心与距离）→ 核方法（隐式升维、内积替换）→ 高斯与协方差（椭圆几何 + 马氏距离 + MLE 衔接）**

一句话总结：**机器学习是把"数据、损失、约束、分布"全部翻译成线性代数对象的工程——数据是矩阵，误差是范数，最优解是投影，正则化是约束几何，降维是特征分解，而核方法与高斯分布告诉我们：线代与概率在机器学习里是同一套语言的两副面孔。** 下一章（Level 8）将把这套"矩阵语言"搬进神经网络：张量、批量乘法与反向传播。

## 推荐资料

- **Stanford CS229 讲义与课程视频（Andrew Ng）**——难度：入门–中等。理由：线性回归、逻辑回归、SVM、PCA 的推导全部用矩阵记号，与本章内容一一对应，是"线代 → ML"最经典的教学材料。
- **《统计学习导论》(ISLR)**——难度：入门。理由：R 实现 + 直觉并重，回归、分类、重采样、降维的章节适合对照本章概念做第一遍验证。
- **周志华《机器学习》（西瓜书）**——难度：中等。理由：线性模型、支持向量机、降维、聚类各章与本章结构完全吻合，中文表述严谨，适合精读推导。
- **《深度学习》（花书）第 2 章与线性模型部分**——难度：中等。理由：第 2 章"线性代数"是给 ML 场景定制的线代复习，PCA/协方差/范数视角与本章互补，且直接通向 Level 8。

> 若遇到解决不了的疑问，请在笔记中用 `> [!question]` 标记，按 config.md 的规范跟踪解决。

## Related Notes

- [[Level6_Matrix_Decomposition|Level 6 矩阵分解]] —— 前置基础
- [[Level8_Linear_Algebra_in_DL|Level 8 深度学习中的线性代数]] —— 下一章
- [[Linear Algebra MOC]] —— 知识库总览
- [[Learning_Roadmap]] —— 学习路线
- [[Glossary]] —— 术语表
- [[Mathematical Modeling MOC]] —— 建模库（SVM、聚类、PCA 的实践实现）
