---
title: Level 5 特征值与特征向量
type: concept
subject: linear-algebra
level: Level 5
tags:
  - mathematics
  - linear-algebra
  - eigenvalues
created: 2026-08-23
source: Learning_Roadmap.md Level 5 推荐资料
---

# Level 5 特征值与特征向量

> 前面我们反复问"矩阵做了什么"：Level 3 问它把空间映射到哪（列空间）、把哪些方向压扁（零空间），Level 4 问它如何度量方向（正交与投影）。本章问一个更"挑剔"的问题：**哪些向量被 $A$ 作用后方向一点不变、只是被拉长或缩短？** 这些"不变的方向"就是特征向量，对应的缩放倍数就是特征值。
>
> 知识主线：**矩阵作用 → 不变方向（特征向量）→ 特征方程与计算 → 特征值的整体性质 → 相似矩阵与对角化（换基化简）→ 对称矩阵与谱定理（正交对角化）→ 正定矩阵（能量与凸性）→ AI 应用**
>
> 前置知识：Level 3（行列式、线性变换、基与坐标）与 Level 4（正交、投影、正交矩阵）。

## 学习目标

完成本章后，你应该能够：

- 用几何语言解释"特征向量方向不变、只被缩放"，并能对缩放、剪切、旋转、投影等具体矩阵说出特征向量是什么；
- 通过特征方程 $\det(A-\lambda I)=0$ 计算特征值与特征向量，并会用迹、行列式校验结果；
- 理解相似矩阵 = 同一个变换在不同基下的表达，掌握对角化 $A=PDP^{-1}$ 的条件与几何意义；
- 理解实对称矩阵的特殊性：特征值为实数、特征向量可正交、$A=Q\Lambda Q^T$（谱定理）；
- 判断正定 / 半正定矩阵，并联系能量、凸性、高斯分布与最优化；
- 建立"特征分解 = 拆解矩阵作用"的思想，能说出 PCA、PageRank、谱聚类、RNN 稳定性中的具体公式出处。

---

## 第一部分：特征值与特征向量的定义与几何直觉

### 1. 从"矩阵作用"到"不变方向"

Level 1–4 我们一直在回答"矩阵把向量变成什么"。对一般的向量，答案很复杂：方向和长度通常都变。但几乎每个矩阵都有一些"幸运"的方向：作用之后**方向不变，只有长度按固定倍数缩放**。这些方向就是矩阵的"骨架"——抓住它们，整个变换就变得容易理解。

> [!abstract] 本章的核心思想
> **特征分解 = 拆解矩阵作用**：把一个复杂的矩阵作用拆成若干个"各管一个方向、每个方向只是缩放"的最简单作用。就像分析一台机器时先找出它的"主轴"，每个主轴上的运动规律都最简单。

### 2. 定义：特征值与特征向量

**定义**：设 $A$ 是 $n \times n$ 方阵。若非零向量 $\mathbf{v}$ 和标量 $\lambda$ 满足

$$
A\mathbf{v} = \lambda\mathbf{v},
$$

则称 $\lambda$ 是 $A$ 的**特征值**（eigenvalue），$\mathbf{v}$ 是 $A$ 的对应于 $\lambda$ 的**特征向量**（eigenvector）。

用自己的话复述：**$A$ 把 $\mathbf{v}$ 所在的直线（过原点的方向）整体保持不动，只按 $\lambda$ 倍缩放**：

- $\lambda > 1$：拉长；$0 < \lambda < 1$：缩短；$\lambda < 0$：反向（仍回到同一条直线）；$\lambda = 0$：压成零向量（$\mathbf{v}$ 落在零空间里，Level 3）。

> [!warning] 两个必须记住的细节
> 1. **特征向量必须非零**：零向量满足 $A\mathbf{0} = \lambda\mathbf{0}$ 对**任何** $\lambda$ 都成立，若允许它，每个数都成了"特征值"，定义失去意义；
> 2. 特征向量不唯一：$\mathbf{v}$ 是特征向量，则任何非零倍数 $c\mathbf{v}$ 也是——我们真正关心的是**方向（直线）**，不是具体向量。

**特征空间（eigenspace）**：对应于同一 $\lambda$ 的所有特征向量加上零向量，构成一个子空间

$$
E_\lambda = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\} = \ker(A - \lambda I),
$$

它的维数称为 $\lambda$ 的**几何重数**（geometric multiplicity）。把它写成零空间的形式，就自动说明它是子空间（Level 3 的结论直接用上）。

### 3. 四个几何例子：缩放、剪切、旋转、投影

> [!example] 例 1：缩放矩阵（对角矩阵）
> $A = \begin{pmatrix}3 & 0\\0 & 2\end{pmatrix}$ 把 $x$ 轴方向放大 3 倍、$y$ 轴方向放大 2 倍。
> - 特征值 $\lambda_1 = 3$（特征向量 $(1,0)$ 所在直线）、$\lambda_2 = 2$（特征向量 $(0,1)$ 所在直线）；
> - 注意：向量 $(1,1)$ **不是**特征向量——它被映射到 $(3,2)$，方向变了。只有坐标轴方向"方向不变"。对角元互异时，对角矩阵的特征向量就是标准基方向。

> [!example] 例 2：剪切矩阵
> $A = \begin{pmatrix}1 & 1\\0 & 1\end{pmatrix}$（水平剪切）：把 $y$ 坐标"抹到" $x$ 上。
> - 特征方程 $(1-\lambda)^2 = 0$，特征值 $\lambda = 1$（代数重数 2）；
> - 但 $(A - I) = \begin{pmatrix}0&1\\0&0\end{pmatrix}$ 的零空间只有 $\operatorname{span}\{(1,0)\}$——**只有一个方向**（$x$ 轴）保持不变，其余方向都被剪切移动。几何重数 1 < 代数重数 2。这个例子将在第三部分说明"为什么不能对角化"。

> [!example] 例 3：旋转矩阵
> $A = \begin{pmatrix}0 & -1\\1 & 0\end{pmatrix}$（逆时针旋转 $90^\circ$）：**没有实特征值**——没有任何方向在旋转 $90^\circ$ 后保持原直线。特征方程 $\lambda^2 + 1 = 0$ 在 $\mathbb{R}$ 上无解，在 $\mathbb{C}$ 上有 $\lambda = \pm i$。
> 这说明：特征值问题天然要求我们走出实数域；但**实对称矩阵永远不会遇到这个问题**（第四部分）。

> [!example] 例 4（与 Level 4 呼应）：正交投影
> $P = \begin{pmatrix}0.5 & 0.5\\0.5 & 0.5\end{pmatrix}$ 是到直线 $y = x$ 的正交投影。特征值 1（直线 $y=x$ 上的向量保持不变）与 0（垂直线 $y=-x$ 上的向量被压成零）。**投影 = 保持目标子空间、消灭正交补**——这正是 Level 4 投影的"特征向量语言"。

### 4. AI 应用（一）：PCA 的方向直觉——特征向量 = "数据的主轴"

- **解决什么问题**：高维数据（如 $d$ 维特征）冗余大，想找"最重要的少数几个方向"做降维；
- **用到的线代工具**：数据协方差矩阵 $\Sigma = \frac{1}{n-1}X^TX$ 的**特征向量与特征值**（PCA 的数学定义；sklearn 内部用 SVD 实现，Level 6 解释原因）；
- **为什么有效**：协方差矩阵的特征向量给出数据变化最大的方向（特征值 = 该方向上的方差），沿最大特征值方向投影损失的信息最少；
- **对应概念**：特征向量 = "不变方向"，在这里具体化为"数据最伸展的方向"；
- **理解深度**：U（能写出 $\Sigma = \frac{1}{n-1}X^TX$ 与"方差 = 特征值"；完整推导在第四部分谱定理之后）。

---

## 第二部分：特征多项式与计算

### 5. 特征方程：为什么是 det(A − λI) = 0

特征值的判定条件可以直接从定义推出来：

$$
A\mathbf{v} = \lambda\mathbf{v} \iff (A - \lambda I)\mathbf{v} = \mathbf{0},\quad \mathbf{v} \neq \mathbf{0}.
$$

这是一个**齐次线性方程组**（Level 2）。非零解存在 $\iff$ 系数矩阵奇异 $\iff$ 行列式为 0（Level 2–3 的结论）：

$$
\boxed{\ \det(A - \lambda I) = 0\ }
$$

展开 $\det(A - \lambda I)$ 得到关于 $\lambda$ 的 $n$ 次多项式，称为**特征多项式**（characteristic polynomial）$p(\lambda)$；方程 $p(\lambda) = 0$ 称为**特征方程**（characteristic equation），它的根就是特征值。由代数基本定理，$n$ 次多项式在复数域恰有 $n$ 个根（计重数）——**$n \times n$ 矩阵在 $\mathbb{C}$ 上恒有 $n$ 个特征值（计代数重数）**。

> [!note] 计算流程
> 1. 写出 $A - \lambda I$，计算 $\det(A - \lambda I)$ 得特征多项式；
> 2. 解 $p(\lambda) = 0$ 得特征值（记录每个的代数重数）；
> 3. 对每个 $\lambda$，解齐次方程组 $(A - \lambda I)\mathbf{v} = \mathbf{0}$ 得特征空间（用自由变量写出基，Level 2 的方法）。

**特例（重要）**：**三角矩阵（上三角 / 下三角 / 对角）的特征值就是对角元**——因为 $\det(A - \lambda I)$ 是对角线上 $(a_{ii} - \lambda)$ 的乘积。

### 6. 完整计算例子

> [!example] 2×2 例子
> $A = \begin{pmatrix}4 & 1\\2 & 3\end{pmatrix}$：
> $$p(\lambda) = \det\begin{pmatrix}4-\lambda & 1\\2 & 3-\lambda\end{pmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda-2)(\lambda-5).$$
> 特征值 $\lambda_1 = 2,\ \lambda_2 = 5$。
> - $\lambda = 2$：解 $\begin{pmatrix}2&1\\2&1\end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_1 = (1,-2)$；
> - $\lambda = 5$：解 $\begin{pmatrix}-1&1\\2&-2\end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_2 = (1,1)$。
>
> 校验（见第 7 节）：$\operatorname{tr}(A) = 4+3 = 7 = 2+5$ ✓；$\det(A) = 12-2 = 10 = 2\times5$ ✓。

> [!example] 3×3 例子（三角矩阵）
> $A = \begin{pmatrix}1&7&2\\0&3&-1\\0&0&3\end{pmatrix}$ 是上三角矩阵，特征值直接读对角元：$\lambda = 1, 3, 3$。不需要展开特征多项式。

### 7. 特征值的整体性质

以下性质对任何方阵成立（$\lambda_1, \dots, \lambda_n$ 为特征值，**计代数重数**）：

**① 迹 = 特征值之和**
$$
\operatorname{tr}(A) = \sum_i a_{ii} = \lambda_1 + \cdots + \lambda_n.
$$

**② 行列式 = 特征值之积**
$$
\det(A) = \lambda_1 \lambda_2 \cdots \lambda_n.
$$

**③ 可逆性**：$A$ 可逆 $\iff$ 所有特征值非零。因为 $\det(A) = \prod_i \lambda_i$，而 $A$ 可逆 $\iff \det(A) \neq 0$（Level 3）。

> [!tip] 为什么成立（2×2 完整推导）
> 对 $A = \begin{pmatrix}a&b\\c&d\end{pmatrix}$，特征多项式
> $$p(\lambda) = \lambda^2 - (a+d)\lambda + (ad-bc) = \lambda^2 - \operatorname{tr}(A)\,\lambda + \det(A).$$
> 设两根为 $\lambda_1, \lambda_2$，则 $p(\lambda) = (\lambda-\lambda_1)(\lambda-\lambda_2) = \lambda^2 - (\lambda_1+\lambda_2)\lambda + \lambda_1\lambda_2$。比较系数（Vieta 定理）即得 ①②。一般 $n$ 的情形同理：比较 $p(\lambda)$ 中 $\lambda^{n-1}$ 项与常数项的系数（常数项恰为 $p(0) = \det(A)$）。

**④ 由特征值导出新矩阵的特征值**：若 $A\mathbf{v} = \lambda\mathbf{v}$，则
- $A^k \mathbf{v} = \lambda^k \mathbf{v}$（直接递推，$k$ 次幂）；
- $A$ 可逆时 $A^{-1}\mathbf{v} = \lambda^{-1}\mathbf{v}$（两边同乘 $A^{-1}$，注意 $\lambda \neq 0$）；
- $(A + cI)\mathbf{v} = (\lambda + c)\mathbf{v}$（平移不改变特征向量）。

> [!warning] 注意
> 特征值与矩阵的**元素**没有直接关系——"特征值等于对角元"只在三角矩阵成立。例 1 的 $A$ 特征值 2、5 与对角元 4、3 完全不同。

### 8. AI 应用（二）：PageRank——网络世界的主特征向量

- **解决什么问题**：Google 给网页按"重要性"排序（PageRank，Brin & Page 1998）；
- **用到的线代工具**：把链接关系写成随机矩阵 $S$（$S_{ij}$ = 从页 $j$ 跳到页 $i$ 的概率，列和为 1），加上"随机跳转"（teleport）得到**Google 矩阵** $G = \alpha S + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T$（$\alpha \approx 0.85$）；
- **为什么有效**：网页重要性的平稳分布 $\boldsymbol{\pi}$ 满足 $\boldsymbol{\pi} = G\boldsymbol{\pi}$——**$\boldsymbol{\pi}$ 是 $G$ 的特征值 1 的特征向量**。由 Perron–Frobenius 定理，随机矩阵的特征值 1 是单根且模最大（谱半径 = 1），所以幂迭代 $\boldsymbol{\pi}_{t+1} \leftarrow G\boldsymbol{\pi}_t$ 必然收敛——这让 PageRank 在十亿级网页上也能用"反复矩阵-向量乘法"算出来（每步 $O(\text{边数})$）；
- **对应概念**：特征值 = 缩放倍数（这里是 1，分布"不变"）；幂迭代收敛依赖"主特征值唯一且占优"（第 11 节）；
- **理解深度**：U（能写出 $G$、$\boldsymbol{\pi} = G\boldsymbol{\pi}$ 与幂迭代；Perron–Frobenius 的严格证明属于谱理论，暂记 I→U）。

---

## 第三部分：相似矩阵与对角化

### 9. 相似矩阵：同一个变换，不同的坐标系

**定义**：若存在可逆矩阵 $P$ 使 $B = P^{-1}AP$，则称 $A$ 与 $B$ **相似**（similar）。

几何意义（Level 3 的"换基"）：$P$ 的列就是一组**新基**（用旧基坐标写出），$P^{-1}AP$ 是同一个线性变换在**新基下**的矩阵。相似 = "同一变换换了个坐标系描述"，变换本身没变。

**相似不变量**（$A$ 与 $P^{-1}AP$ 相同）：特征多项式、特征值（计重数）、迹、行列式、秩。理由：$\det(P^{-1}AP - \lambda I) = \det\big(P^{-1}(A-\lambda I)P\big) = \det(A-\lambda I)$（用到 $\det(P^{-1})\det(P)=1$），其余性质随之而来。

> [!warning] 特征向量不是不变量
> 若 $A\mathbf{v} = \lambda\mathbf{v}$，则 $P^{-1}AP\cdot(P^{-1}\mathbf{v}) = \lambda (P^{-1}\mathbf{v})$：**特征向量要跟着坐标系一起变换**——"同一个方向在新坐标下的表示"变了，但特征值不变。

### 10. 对角化：A = PDP⁻¹

**定义**：若存在可逆矩阵 $P$ 与对角矩阵 $D$ 使 $A = PDP^{-1}$，则称 $A$ **可对角化**（diagonalizable）。

**推导**：把 $D = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ 代入 $AP = PD$（对 $A = PDP^{-1}$ 两边右乘 $P$）：

- $AP = [A\mathbf{v}_1, \dots, A\mathbf{v}_n]$，$PD = [\lambda_1\mathbf{v}_1, \dots, \lambda_n\mathbf{v}_n]$；
- 所以 $A\mathbf{v}_i = \lambda_i\mathbf{v}_i$：**$P$ 的列是特征向量，$D$ 的对角元是特征值**；$P$ 可逆要求这 $n$ 个特征向量线性无关，即它们构成 $\mathbb{R}^n$ 的一组基，称为**特征基**（eigenbasis）。

**几何意义**：在特征基下看，$A$ 的作用是"每个方向独立缩放 $\lambda_i$ 倍"——**换基后，复杂的矩阵作用变成最简单的一维缩放**。这就是 Level 3 预告的"换坐标系让问题变简单"，也是"特征分解 = 拆解矩阵作用"的直接体现。

**威力：矩阵幂**。若 $A = PDP^{-1}$，则

$$
A^k = (PDP^{-1})^k = PD^kP^{-1}, \qquad D^k = \operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k).
$$

幂运算从"反复矩阵乘法"变成"对角元各自取 $k$ 次幂"——这是动力系统分析（RNN、马尔可夫链）的引擎。

### 11. 何时可对角化

**代数重数**（algebraic multiplicity）：$\lambda$ 作为特征多项式根的重数；**几何重数**：$\dim E_\lambda = \dim\ker(A - \lambda I)$。恒有 几何重数 ≤ 代数重数。

**可对角化的充要条件**：对**每个**特征值，几何重数 = 代数重数（即所有特征值"凑够"独立特征向量，总计 $n$ 个）。

**充分条件（最常用）**：$n$ 个特征值**互不相同** ⇒ 可对角化。

> [!tip] 为什么互异特征值的特征向量线性无关（两个的情形）
> 设 $A\mathbf{v} = \lambda\mathbf{v}$，$A\mathbf{w} = \mu\mathbf{w}$，$\lambda \neq \mu$，且 $c_1\mathbf{v} + c_2\mathbf{w} = \mathbf{0}$。左乘 $A$：$c_1\lambda\mathbf{v} + c_2\mu\mathbf{w} = \mathbf{0}$；减去 $\lambda$ 倍的第一个等式：$c_2(\mu - \lambda)\mathbf{w} = \mathbf{0}$，因 $\mathbf{w} \neq 0$ 且 $\mu \neq \lambda$ 得 $c_2 = 0$，同理 $c_1 = 0$。一般情形用数学归纳法，同样的"逐个消元"思路。

> [!warning] 反例：剪切矩阵不可对角化
> $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$：特征值只有 $\lambda = 1$（代数重数 2），但 $E_1 = \ker(A-I) = \operatorname{span}\{(1,0)\}$ 只有一维（几何重数 1）。凑不出 2 个无关特征向量，无法对角化。几何上：剪切只保住了 $x$ 轴一个方向，其余方向都被滑动，没有任何直线保持不动。

> [!note] 幂迭代（数值计算的雏形，为 Level 6 铺垫）
> 若 $A$ 有**唯一占优特征值** $|\lambda_1| > |\lambda_2| \ge \cdots$，对几乎任意初始向量 $\mathbf{v}_0$，迭代 $\mathbf{v}_{t+1} = A\mathbf{v}_t$ 的方向收敛到 $\lambda_1$ 的特征向量——用 $A^t = PD^tP^{-1}$ 验证：第 $i$ 个特征方向按 $\lambda_i^t$ 增长，模最大的 $\lambda_1$ 最终主导。这就是**幂迭代**（power iteration），PageRank 与 PCA 大规模实现的核心算法。

### 12. AI 应用（三）：RNN 稳定性——为什么梯度会消失 / 爆炸

- **解决什么问题**：RNN 训练中长期依赖难以学习（梯度消失）或数值爆炸（梯度爆炸）；
- **用到的线代工具**：隐状态动力系统 $\mathbf{h}_{t+1} = W\mathbf{h}_t$（忽略非线性激活的线性化），解为 $\mathbf{h}_t = W^t\mathbf{h}_0$；若 $W = PDP^{-1}$，则 $\mathbf{h}_t = PD^tP^{-1}\mathbf{h}_0$，第 $i$ 个特征方向按 $\lambda_i^t$ 增长或衰减；
- **为什么有效**：定义**谱半径**（spectral radius）$\rho(W) = \max_i |\lambda_i|$：
  - $\rho(W) < 1$：所有方向指数衰减（稳定，但信息迅速遗忘 → 梯度消失）；
  - $\rho(W) > 1$：某些方向指数爆炸（梯度爆炸）；
  - 反向传播沿时间展开就是反复乘以 $W^T$（每步还乘激活函数的导数），梯度随步数按 $\sim \rho(W)^t$ 缩放——这就是"长程依赖难学"的数学本质（Hochreiter 1991 / Bengio et al. 1994 的经典分析）；
  - 缓解手段（Level 8 展开）：梯度裁剪（clipping）、正交初始化（让 $\rho(W) \approx 1$）、LSTM 门控；
- **对应概念**：$A^k$ 的特征值 = $\lambda^k$（第 7 节性质④）、主特征值决定长期行为；
- **理解深度**：U（能写出 $\mathbf{h}_t = W^t\mathbf{h}_0$ 与谱半径判据）。

---

## 第四部分：对称矩阵与谱定理

### 13. 实对称矩阵的特征值必为实数

**定理**：若 $A = A^T$ 是实对称矩阵，则 $A$ 的所有特征值都是实数。

**证明**：设 $\lambda$ 是 $A$ 的（可能是复的）特征值，$\mathbf{v} \neq \mathbf{0}$ 是对应（复）特征向量。一方面

$$
\bar{\mathbf{v}}^T A \mathbf{v} = \bar{\mathbf{v}}^T(\lambda\mathbf{v}) = \lambda \|\mathbf{v}\|^2;
$$

另一方面，对 $A\mathbf{v} = \lambda\mathbf{v}$ 取共轭转置（用 $A^T = A$ 且 $A$ 为实矩阵）得 $\bar{\mathbf{v}}^T A = \lambda^* \bar{\mathbf{v}}^T$（$^*$ 表示共轭），于是

$$
\bar{\mathbf{v}}^T A \mathbf{v} = \lambda^* \|\mathbf{v}\|^2.
$$

两式相减：$(\lambda - \lambda^*)\|\mathbf{v}\|^2 = 0$，而 $\|\mathbf{v}\|^2 > 0$，故 $\lambda = \lambda^*$，即 $\lambda \in \mathbb{R}$。∎

**几何直觉**：对称矩阵只"沿互相垂直的方向拉伸 / 压缩 / 翻转"，从不"旋转"（对比例 3 的旋转矩阵），所以不会出现复数特征值。

### 14. 不同特征值的特征向量正交

**定理**：实对称矩阵属于**不同特征值**的特征向量正交。

**证明**：设 $A\mathbf{v} = \lambda\mathbf{v}$，$A\mathbf{w} = \mu\mathbf{w}$，$\lambda \neq \mu$。用两种顺序计算 $\mathbf{w}^T A \mathbf{v}$（利用 $A = A^T$）：

$$
\mathbf{w}^T A \mathbf{v} = \mathbf{w}^T(\lambda\mathbf{v}) = \lambda\,\mathbf{w}^T\mathbf{v},
$$

同时

$$
\mathbf{w}^T A \mathbf{v} = (A^T\mathbf{w})^T\mathbf{v} = (A\mathbf{w})^T\mathbf{v} = \mu\,\mathbf{w}^T\mathbf{v}.
$$

两式相减得 $(\lambda - \mu)\,\mathbf{w}^T\mathbf{v} = 0$，因 $\lambda \neq \mu$，故 $\mathbf{w}^T\mathbf{v} = 0$，即 $\mathbf{v} \perp \mathbf{w}$。∎

### 15. 谱定理：A = QΛQᵀ

**谱定理（Spectral Theorem）**：实对称矩阵 $A$ 可以**正交对角化**：存在正交矩阵 $Q$（$Q^TQ = I$）与实对角矩阵 $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$，使

$$
\boxed{\ A = Q\Lambda Q^T\ }
$$

等价表述：$\mathbb{R}^n$ 存在由 $A$ 的特征向量组成的**标准正交基**。

**为什么成立（证明思路，归纳法）**：

1. 由第 13 节，$A$ 有实特征值 $\lambda_1$，取单位特征向量 $\mathbf{q}_1$；
2. 令 $W = \mathbf{q}_1^{\perp}$（$\mathbf{q}_1$ 的正交补，Level 4）。$W$ 在 $A$ 下**不变**：对 $\mathbf{x} \perp \mathbf{q}_1$，$\mathbf{q}_1^T A\mathbf{x} = (A\mathbf{q}_1)^T\mathbf{x} = \lambda_1 \mathbf{q}_1^T \mathbf{x} = 0$，即 $A\mathbf{x} \perp \mathbf{q}_1$；
3. 把 $A$ 限制在 $W$ 上（限制后仍是对称的），对 $\dim W = n-1$ 归纳：得到 $W$ 内一组正交单位特征向量；与 $\mathbf{q}_1$ 合起来就是 $\mathbb{R}^n$ 的标准正交特征基。∎

（反向几乎平凡：若 $A = Q\Lambda Q^T$ 且 $Q^TQ = I$，则 $A^T = Q\Lambda^T Q^T = Q\Lambda Q^T = A$——所以"实对称 ⟺ 可正交对角化"是等价命题。）

**对比一般矩阵**：一般矩阵只能保证"相似对角化"（$P$ 不必正交，且可能不可对角化，见例 2）；对称矩阵**一定**可以正交对角化。正交矩阵 $Q$ 意味着换到特征基的变换是"刚体旋转"，不引入畸变——与 Level 4 的正交性一脉相承。

**谱（spectrum）**：特征值集合 $\{\lambda_1, \dots, \lambda_n\}$ 称为 $A$ 的谱。"谱定理"的意思是：**对称矩阵完全由它的谱（加特征基）决定**——这也解释了为什么本章核心概念叫"特征值"（eigenvalue，德语"自身的值"）。

### 16. AI 应用（四）：PCA——协方差矩阵的谱分解

- **解决什么问题**：降维、去相关、可视化（人脸识别 Eigenfaces、基因表达分析等的基础）；
- **用到的线代工具**：中心化数据矩阵 $X \in \mathbb{R}^{n\times d}$（每列均值为 0）的样本协方差矩阵 $\Sigma = \frac{1}{n-1}X^TX$，它是**实对称、半正定**矩阵（第五部分），谱分解
$$
\Sigma = Q\Lambda Q^T, \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0;
$$
- **为什么有效**：$\Sigma$ 的特征向量 $\mathbf{q}_i$（**主成分**）给出数据方差最大的正交方向，$\lambda_i$ = 该方向上的方差（由 $Q^T\Sigma Q = \Lambda$ 的对角元直接读出，见第 19 节瑞利商）；把数据投影到前 $k$ 个主成分 $Z = XQ_k$，丢弃的信息 = 被丢弃方向上的方差和 $\sum_{i>k}\lambda_i$；
- **具体公式出处**：explained variance ratio（sklearn `PCA.explained_variance_ratio_`）$= \lambda_i / \sum_j \lambda_j = \lambda_i / \operatorname{tr}(\Sigma)$；总方差守恒：$\sum_i \lambda_i = \operatorname{tr}(\Sigma) = \sum_j \operatorname{Var}(x_j)$；
- **对应概念**：谱定理（主成分正交、方差可加）、半正定（方差不为负）；
- **理解深度**：U（能写出 $\Sigma = \frac{1}{n-1}X^TX$ 与其谱分解；sklearn 用 SVD 实现是数值稳定性问题，Level 6 再解释）。

> [!example] AI 应用（五）：谱聚类——图拉普拉斯的低特征向量
> - **解决什么问题**：数据呈"嵌套 / 流形"形状时，K-means 按欧氏距离聚不好，改用图上的连通性聚类；
> - **用到的线代工具**：相似度矩阵 $W$（如高斯核 $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$）、度矩阵 $D = \operatorname{diag}(W\mathbf{1})$、**图拉普拉斯** $L = D - W$（或归一化版 $L_{sym} = I - D^{-1/2}WD^{-1/2}$）。$L$ 是对称半正定矩阵；
> - **为什么有效**：取 $L$ 的**最小 $k$ 个特征值对应的特征向量**，把每个点嵌入 $\mathbb{R}^k$（$L$ 的零特征值个数 = 连通分量个数；第二个最小特征值 $\lambda_2$ 对应的向量叫 **Fiedler 向量**，其符号给出"最佳二分"），再在这个嵌入上跑 K-means，就能聚出非凸形状的簇。数学根源：最小化归一化割（normalized cut）松弛为 $L$ 的瑞利商问题（Shi & Malik 2000）；
> - **对应概念**：对称矩阵的谱分解（第四部分）、瑞利商与最小特征值（第 19 节）；
> - **理解深度**：U（能写出 $L = D - W$ 与"最小特征向量嵌入"；割-松弛的严格等价需要谱图理论，暂记 I→U）。

---

## 第五部分：正定与半正定矩阵

### 17. 定义：能量的语言

以下约定 $A$ 为**实对称**矩阵（二次型 $\mathbf{x}^T A \mathbf{x}$ 只依赖 $A$ 的对称部分 $(A + A^T)/2$，通常直接要求 $A = A^T$）。

**定义**：$A$ 称为**正定**（positive definite, PD），若对一切 $\mathbf{x} \neq \mathbf{0}$ 有 $\mathbf{x}^T A \mathbf{x} > 0$；称为**半正定**（positive semidefinite, PSD），若 $\mathbf{x}^T A \mathbf{x} \ge 0$ 恒成立。

为什么这个式子重要？$\mathbf{x}^T A \mathbf{x}$ 是"沿方向 $\mathbf{x}$ 的**能量 / 二次型**"：

- 力学中它是势能（$A$ = 刚度矩阵，$\frac12\mathbf{x}^TA\mathbf{x}$ = 形变能）；
- 统计中 $\Sigma$ 的二次型出现在高斯密度的指数里（第 20 节）；
- 最优化中 $\frac12\mathbf{x}^TA\mathbf{x} - \mathbf{b}^T\mathbf{x}$ 是最简单的二次损失。

> [!note] 与特征值的关系（关键推导）
> 由谱定理 $A = Q\Lambda Q^T$，令 $\mathbf{y} = Q^T\mathbf{x}$（正交变换保长度：$\|\mathbf{y}\| = \|\mathbf{x}\|$）：
> $$\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T Q \Lambda Q^T \mathbf{x} = \mathbf{y}^T \Lambda \mathbf{y} = \sum_{i=1}^{n} \lambda_i y_i^2.$$
> 因为 $\mathbf{y}$ 可取任意向量（$Q$ 可逆），"恒 $> 0$" $\iff$ **所有 $\lambda_i > 0$**；"恒 $\ge 0$" $\iff$ **所有 $\lambda_i \ge 0$**。正定与"特征值全正"是同一件事的两种说法。

### 18. 判定准则

设 $A$ 为 $n \times n$ 实对称矩阵：

| 判定 | 正定（PD） | 半正定（PSD） |
| --- | --- | --- |
| 特征值 | 全部 $> 0$ | 全部 $\ge 0$ |
| 主子式 | 顺序主子式全 $> 0$（Sylvester 准则） | 所有主子式 $\ge 0$（无简单的"顺序"版） |
| 分解 | 存在可逆 $C$ 使 $A = C^TC$（Cholesky 分解 $A = LL^T$，$L$ 下三角） | 存在 $B$ 使 $A = B^TB$ |
| 逆 | $A^{-1}$ 存在且同样正定 | 不一定可逆 |

**"分解"判定的推导**（重要，也是 AI 应用里最常见的形式）：

$$
\mathbf{x}^T (B^T B) \mathbf{x} = (B\mathbf{x})^T (B\mathbf{x}) = \|B\mathbf{x}\|^2 \ge 0.
$$

所以 $B^TB$ 恒半正定；当且仅当 $B\mathbf{x} = \mathbf{0}$ 只有零解（$B$ **列满秩**，即零空间平凡）时为正定。**协方差矩阵 $\Sigma = \frac{1}{n-1}X^TX$ 半正定**就是这个结论的直接推论（取 $B = X/\sqrt{n-1}$）；当数据"撑满"整个特征空间时它正定。

> [!warning] 常见陷阱
> "对角元全为正"**不是**正定：$A = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ 对角元全正，但特征值为 $3, -1$，取 $\mathbf{x} = (1,-1)$ 得 $\mathbf{x}^TA\mathbf{x} = -2 < 0$。判定必须看特征值或主子式。

### 19. 几何直觉：椭圆、凸性、瑞利商

**椭圆（等高线）**：对正定 $A$，集合 $\{\mathbf{x} : \mathbf{x}^T A \mathbf{x} = 1\}$ 是一个**椭球**：在特征基 $Q$ 下方程化为 $\sum \lambda_i y_i^2 = 1$，主轴沿特征向量方向 $\mathbf{q}_i$，半轴长 $1/\sqrt{\lambda_i}$。特征值越大 → 该方向越"窄"；特征值越小 → 越"扁"。**这就是"矩阵的几何形状"**——谱定理把所有方向的信息压缩进 $n$ 个数字（谱）。

**凸性（最优化）**：函数 $f(\mathbf{x}) = \frac12\mathbf{x}^TA\mathbf{x} - \mathbf{b}^T\mathbf{x}$ 的 Hessian 就是 $A$。$A$ 半正定 $\iff f$ 是**凸函数**（只有一个全局最小点）；$A$ 正定 $\iff f$ **严格凸**，唯一最小值 $\mathbf{x}^* = A^{-1}\mathbf{b}$——正是 Level 2 正规方程的形状。梯度下降在特征值差异大时收敛慢：沿小特征值方向曲率小、"看不见"曲率，收敛速度受**条件数** $\kappa = \lambda_{\max}/\lambda_{\min}$ 控制（Level 6 深入）。

**瑞利商（Rayleigh quotient）**：

$$
\lambda_{\min} \le \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}} \le \lambda_{\max},
$$

等号在 $\mathbf{x}$ 取对应特征向量时达到（由 $\sum \lambda_i y_i^2 / \sum y_i^2$ 直接看出）。它给出"矩阵作用在某方向的放大倍数"的界——谱聚类的最小特征值、谱范数（Level 6）、稳定性的谱半径都建立在这个不等式上。

### 20. AI 应用（六 / 七 / 八）：高斯分布、优化与对抗扰动

> [!example] 高斯模型：协方差矩阵的椭圆几何
> - **解决什么问题**：用多元高斯分布刻画数据分布（GMM 聚类、VAE 先验、卡尔曼滤波的基础）；
> - **用到的线代工具**：协方差矩阵 $\Sigma$ 的谱分解；
> - **为什么有效**：密度 $p(\mathbf{x}) \propto \exp\left(-\frac12(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$ 的等高线是椭球：轴沿 $\Sigma$ 的特征向量，半轴长 $\propto \sqrt{\lambda_i}$。**特征向量 = 数据的独立变化轴，特征值 = 各轴的变化幅度**；白化（whitening）$\Sigma^{-1/2} = Q\Lambda^{-1/2}Q^T$ 把椭圆"撑圆"、使特征去相关——深度学习输入预处理、VAE 的重参数化 $\mathbf{z} = \boldsymbol{\mu} + \Sigma^{1/2}\boldsymbol{\epsilon}$ 都直接用它；
> - **对应概念**：正定矩阵（$\Sigma \succ 0$）、谱分解的几何；
> - **理解深度**：U。

> [!example] 最优化：损失函数的曲率与条件数
> - **解决什么问题**：梯度下降为什么在有些方向上"磨蹭"，如何诊断收敛慢；
> - **用到的线代工具**：损失函数 Hessian $H$ 的谱分解（对最小二乘 $\|X\mathbf{w} - \mathbf{y}\|^2$ 有 $H = 2X^TX$ 半正定，Level 7 展开）；
> - **为什么有效**：沿 $H$ 的特征向量方向，损失是"独立的抛物线"（$\sum \lambda_i y_i^2$ 分解），曲率即 $\lambda_i$；曲率差异大（$\kappa = \lambda_{\max}/\lambda_{\min}$ 大）时，梯度下降在小曲率方向步长太小、大曲率方向又震荡——**条件数大 = 病态**，这正是预处理（白化 / 归一化）与牛顿法的动机；
> - **对应概念**：正定性与凸性、瑞利商；
> - **理解深度**：U。

> [!example] 对抗扰动的几何理解
> - **解决什么问题**：为什么加上人眼不可见的微小扰动 $\boldsymbol{\delta}$ 就能让分类器出错（Goodfellow et al. 2014 的 FGSM：$\boldsymbol{\delta} = \epsilon\,\operatorname{sign}(\nabla_{\mathbf{x}}L)$）；
> - **用到的线代工具**：数据协方差 $\Sigma$ 与损失 Hessian 的谱；
> - **为什么有效**（两个互补的视角）：
>   1. **数据几何**：自然数据集中在 $\Sigma$ 的大特征值方向（"信号方向"）；小特征值方向方差极小，数据"几乎没有分布在那里"——在这些方向加扰动不改变数据流形上的感知（人眼不易察觉），却足以把样本推过决策边界。**小特征值方向 = 对抗样本的藏身处**；
>   2. **损失曲率**：损失在 Hessian 最大特征值方向弯曲最剧烈，沿该方向微移一点损失变化最大——FGSM 的梯度方向与 Hessian 主导方向高度相关（更严谨的谱分析与对抗鲁棒性正则化属于研究前沿，Level 8 可再深入）；
> - **对应概念**：特征向量 = "特殊方向"（最大曲率 / 最小方差方向）、谱分解；
> - **理解深度**：I（直觉层面）。

---

## 常见误区汇总

1. **"零向量也是特征向量"** → 特征向量必须非零；否则 $A\mathbf{0} = \lambda\mathbf{0}$ 对一切 $\lambda$ 成立，定义崩塌；
2. **"特征值就是矩阵的对角元"** → 只有**三角矩阵**的特征值才等于对角元；一般矩阵要用 $\det(A - \lambda I) = 0$ 求；
3. **"方阵都能对角化"** → 需要 $n$ 个线性无关特征向量（每个特征值的几何重数 = 代数重数）；剪切矩阵 $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 就是反例；
4. **"特征向量一定两两正交"** → 一般矩阵的特征向量只是线性无关（且可任意缩放）；**只有实对称矩阵**（及更一般的正规矩阵）能选出正交的特征向量组；
5. **"实矩阵的特征值一定是实数"** → 旋转矩阵 $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ 的特征值为 $\pm i$；**实对称矩阵**的特征值才保证全为实数；
6. **"迹、行列式是'算出来的数字'，与特征值无关"** → $\operatorname{tr}(A) = \sum_i \lambda_i$、$\det(A) = \prod_i \lambda_i$，它们是相似不变量，也是校验计算结果是否正确的利器；
7. **"对角元全为正 ⇒ 正定"** → 必要不充分（$\begin{pmatrix}1&2\\2&1\end{pmatrix}$ 反例）；必须所有特征值 $> 0$（或顺序主子式全正）；
8. **"特征向量是唯一的"** → 特征向量可缩放、同一特征值的特征向量张成特征空间（子空间）；唯一的是**特征值**（以及特征空间的维数）。

---

## 知识结构图

```mermaid
graph TD
    A[矩阵作用 Av] --> B[不变方向: 特征向量 v]
    B --> C[特征方程 det(A-λI)=0]
    C --> D[特征值 λ 与代数重数]
    B --> E[特征空间 ker(A-λI) 与几何重数]
    D --> F[迹 = Σλᵢ, 行列式 = Πλᵢ]
    D --> G[可逆性: λᵢ ≠ 0]
    B --> H[对角化 A = PDP⁻¹]
    H --> I[相似矩阵: 同一变换换基]
    H --> J[Aᵏ = PDᵏP⁻¹ 动力系统/RNN]
    H --> K[可对角化: 几何重数 = 代数重数]
    K --> L[对称矩阵 A = Aᵀ]
    L --> M[谱定理 A = QΛQᵀ]
    M --> N[特征值实数, 特征向量正交]
    M --> O[正定矩阵: 特征值全正]
    O --> P[能量与凸性 xᵀAx]
    M --> Q[PCA: 协方差特征分解]
    M --> R[谱聚类: 拉普拉斯低特征向量]
    D --> S[PageRank: 特征向量 λ = 1]
    J --> S
    P --> T[高斯分布椭圆等高线]
    P --> U[最优化: Hessian 曲率/条件数]
    L --> V[Level 4 正交与投影]
    M --> W[Level 6 SVD: 特征分解的推广]
```

## 思考问题

1. 为什么特征向量必须非零？如果允许 $\mathbf{0}$，定义会发生什么？
2. 旋转矩阵没有实特征值，几何上说明了什么？为什么实对称矩阵永远不会遇到这种"麻烦"？
3. 用"换基"的语言解释 $A = PDP^{-1}$：$P$ 的列、$D$ 的对角元、$P^{-1}AP$ 三者分别是什么？
4. 剪切矩阵不可对角化的本质是什么？"代数重数 2、几何重数 1"用几何语言怎么描述？
5. 若 $|\lambda_1| > |\lambda_2| \ge \cdots$，为什么 $\mathbf{v}_{t+1} = A\mathbf{v}_t$ 最终会指向 $\lambda_1$ 的特征向量？这如何解释 RNN 的梯度消失 / 爆炸？
6. 为什么 PCA 要选协方差矩阵最大特征值对应的特征向量？"方差 = 特征值、总方差 = 迹"这两个等式如何帮你理解降维损失？
7. 思考（预告）：特征分解要求矩阵**方阵且可对角化**；但数据矩阵几乎都是长方阵。下一章（Level 6 矩阵分解）的 **SVD** 对任何矩阵都成立——你猜猜"奇异值"如何推广"特征值"？（提示：考虑 $A^TA$ 与 $AA^T$ 的特征分解）

## 总结

**Level 5 的知识网络**：

**矩阵作用 → 不变方向（特征向量）→ 特征方程与特征多项式 → 特征值的整体性质（迹 / 行列式 / 可逆性）→ 相似矩阵与对角化（换基化简为独立缩放）→ 幂运算与动力系统 → 对称矩阵与谱定理（正交对角化、实特征值、正交特征向量）→ 正定 / 半正定（能量、凸性、椭圆）→ PCA / 谱聚类 / PageRank / RNN 稳定性 / 高斯分布 / 对抗扰动**

一句话总结：**特征值 / 特征向量是矩阵的"主方向与主尺度"——把矩阵作用拆成若干"各管一个不变方向的一维缩放"；对称矩阵还保证这些方向互相垂直，正定矩阵则保证"所有方向都向上弯"（能量恒正）。** 有了这把"拆解"的钥匙：PCA 是协方差矩阵的谱分解，PageRank 是主特征向量，RNN 的生死由谱半径决定。下一章（Level 6 矩阵分解）将把"特征分解"推广到**任何**矩阵（SVD），并研究数值稳定地算出这些分解的方法。

## 推荐资料

- **3Blue1Brown《线性代数的本质》"特征向量与特征值"集**——难度：零基础。理由：几何直觉的权威来源，"方向不变只缩放"的动画一眼即懂；建议连同"对角化 / 特征基"相关几集一起看。
- **Strang《Introduction to Linear Algebra》第 5–6 章**——难度：入门–中等。理由：特征值计算、对角化、对称矩阵与正定矩阵的经典体系，例题规范；配 MIT 18.06 第 21–27 讲效果最佳。
- **MIT 18.06（Strang 课程视频）第 21–27 讲**——难度：中等。理由：课堂推导完整，尤其对称矩阵与正定矩阵的讲解是"概念与直觉并重"的范本。
- **Axler《线性代数应该这样学》（Linear Algebra Done Right）第 5、7 章**——难度：进阶。理由：不用行列式重构特征值理论，谱定理的证明最干净，适合精读时与本笔记的证明互证。
- **《深度学习》（花书）第 2.11 节"特征分解"**——难度：中等。理由：直接把特征分解写成机器学习语境（$A = Q\Lambda Q^T$、谱半径与收敛性），是本笔记 AI 应用的权威延伸。

> 若遇到解决不了的疑问，请在笔记中用 `> [!question]` 标记，按 config.md 的规范跟踪解决。

## Related Notes

- [[Level4_Orthogonality_and_Projection|Level 4 正交与投影]] —— 前置基础（正交、投影、正交矩阵）
- [[Level6_Matrix_Decomposition|Level 6 矩阵分解]] —— 下一章（SVD：特征分解的推广）
- [[Linear Algebra MOC]] —— 知识库总览
- [[Learning_Roadmap]] —— 学习路线
- [[Glossary]] —— 术语表
