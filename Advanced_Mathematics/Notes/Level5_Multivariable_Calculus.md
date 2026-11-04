---
title: Level 5 多元微积分
type: concept
subject: advanced-mathematics
level: Level 5
tags:
  - mathematics
  - calculus
  - multivariable
  - gradient
created: 2026-11-04
updated: 2026-11-04
source: 按 config.md 的 Level 模板整理；定义与公式据同济《高等数学》第七版下册 §8–§10、MIT OCW 18.02 相关章节；本库 `Exercises/` 暂无 Level 5 作业，本篇为体系补链，重点对接 AI / 优化场景所需的 Jacobian、Hessian 与约束极值。
---

# Level 5 多元微积分

> 承接 [[Level2_Derivatives_and_Differentials]]：一元里"可导 $\iff$ 可微"是巧合，多元里它彻底失效，必须分清**偏导存在 / 方向导数存在 / 可微**三个层次。
> 主线：把"局部线性化"从一条直线升级成一个线性映射（Jacobian 矩阵）。

## 直觉

一元函数放大后像直线；二元函数放大后像一张**切平面**。这张平面由一个线性变换 $A$ 描述：$f(x_0+h)\approx f(x_0)+A h$。$A$ 就是导数在高维的替代品。梯度 $\nabla f$ 只是 $A$ 在标量值函数（$1\times n$ 矩阵）时的转置写法。

"梯度指向上升最快的方向"这句常见说法，成立的前提是**可微**；只有偏导数时，沿坐标轴之外的方向可以完全失控。

## 定义

**偏导数（partial derivative）** 固定其余变量，只对一个变量求导：

$$\frac{\partial f}{\partial x_i}(x_0)=\lim_{h\to0}\frac{f(x_0+h e_i)-f(x_0)}{h}.$$

**方向导数（directional derivative）** 单位向量 $u$：$D_u f(x_0)=\lim_{t\to0^+}\frac{f(x_0+tu)-f(x_0)}t$（单侧极限）。

**可微（Frechet differentiable）** 存在线性映射 $A$ 使

$$f(x_0+h)-f(x_0)=A h+o(\|h\|),\qquad h\to0.$$

此时 $A=(\partial f/\partial x_1,\dots,\partial f/\partial x_n)$ 唯一，全微分 $\D f=\sum_i\frac{\partial f}{\partial x_i}\D x_i=\nabla f\cdot\D x$。

**关系链**（只有右向箭头，反向都不成立）：

$$\text{偏导连续}\ \Rightarrow\ \text{可微}\ \Rightarrow\ \begin{cases}\text{连续}\\ \text{各偏导存在}\\ \text{各方向导数存在，且 }D_u f=\nabla f\cdot u\end{cases}\ \Rightarrow\ \text{偏导存在}.$$

反例：$f(x,y)=\frac{xy}{x^2+y^2}$（补 $f(0,0)=0$）在原点两个偏导都是 0，但沿 $y=x$ 不连续；$f=\sqrt{x^2+y^2}$ 在原点各方向导数都存在（$=1$），但不可微。

**Jacobian 与 Hessian** 向量值 $F:\R^n\to\R^m$ 的导数是 $J_F=\bigl[\partial F_i/\partial x_j\bigr]_{m\times n}$；标量 $f$ 的二阶导是 Hessian $H_f=\bigl[\partial^2 f/\partial x_i\partial x_j\bigr]_{n\times n}$。若二阶偏导连续，则 $H_f$ 对称（Clairaut / Schwarz 定理）。

**链式法则** $z=f(u,v)$，$u=u(x,y),v=v(x,y)$：

$$\frac{\partial z}{\partial x}=\frac{\partial f}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial x};$$

一般形式 $J_{F\circ G}(x)=J_F(G(x))\,J_G(x)$ —— 矩阵乘法顺序不能颠倒。

**隐函数定理** $F(x,y)=0$、$F$ 连续可微、$F(x_0,y_0)=0$、$F_y(x_0,y_0)\ne0$ $\Rightarrow$ 在邻域内唯一确定可微函数 $y=y(x)$，且 $\frac{\D y}{\D x}=-\frac{F_x}{F_y}$。方程组情形把 $F_y\ne0$ 换成 Jacobian 行列式 $\det\frac{\partial(F_1,\dots,F_n)}{\partial(y_1,\dots,y_n)}\ne0$。

## 关键公式与结论

**极值与 Hessian 判别** 设 $\nabla f(x_0)=0$（驻点），$D=f_{xx}f_{yy}-f_{xy}^2$（二元）：

- $D>0$ 且 $f_{xx}>0$ → 极小；$D>0$ 且 $f_{xx}<0$ → 极大；
- $D<0$ → 鞍点；$D=0$ → 失效，须回到定义或更高阶。

$n$ 元版本：$H$ 正定 → 极小，负定 → 极大，不定 → 鞍点；用顺序主子式（Sylvester 判据）或特征值判号。

**拉格朗日乘子法（Lagrange multipliers）** 在约束 $g(x)=c$ 下求 $f$ 的极值，构造 $L(x,\lambda)=f(x)-\lambda\bigl(g(x)-c\bigr)$，解 $\nabla_x L=0,\ \nabla_\lambda L=0$。几何含义：极值处 $\nabla f\parallel\nabla g$（等值面相切）。多个约束各配一个 $\lambda_i$；不等式约束需 KKT 条件。

**二重积分与 Fubini** 矩形区域上 $\iint_D f=\int_a^b\D x\int_c^d f\D y=\int_c^d\D y\int_a^b f\D x$（$f$ 连续或绝对可积）。极坐标换元：

$$\iint_D f(x,y)\D x\D y=\iint_{D^*} f(r\cos\theta,r\sin\theta)\,r\,\D r\D\theta,$$

雅可比因子 $r$ 不能漏；一般换元 $(x,y)=(x(u,v),y(u,v))$ 时 $\D x\D y=\left|\det J\right|\D u\D v$。

**三大积分公式的统一形式**（$\partial\Omega$ 为边界，均要求相应光滑性）

$$\oint_{\partial D}P\D x+Q\D y=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)\D x\D y\ \text{(Green)},\qquad \iiint_\Omega\nabla\cdot\mathbf F\,\D V=\oiint_{\partial\Omega}\mathbf F\cdot\mathbf n\,\D S\ \text{(Gauss)},$$

$$\iint_\Sigma(\nabla\times\mathbf F)\cdot\mathbf n\,\D S=\oint_{\partial\Sigma}\mathbf F\cdot\D\mathbf r\ \text{(Stokes)}.$$

三者都是"内部某种导数的积分 = 边界上的量"，是 Stokes 定理 $\int_\Omega \D\omega=\int_{\partial\Omega}\omega$ 在低维的特例。

## 常见误解

| 错误 | 正确 |
| --- | --- |
| 偏导存在 $\Rightarrow$ 可微 / 连续 | 都不成立；偏导只沿坐标轴看，控制不了斜向逼近 |
| 各方向导数存在 $\Rightarrow$ 可微 | 方向导数甚至不能保证连续（$f=\frac{x^2y}{x^4+y^2}$ 型反例） |
| $\nabla f$ 存在就有"最速上升" | 需可微；否则 $\nabla f\cdot u$ 未必等于 $D_u f$ |
| $f_{xy}=f_{yx}$ 总成立 | 需**二阶混合偏导连续**；存在反例（$f=\frac{xy(x^2-y^2)}{x^2+y^2}$ 在原点两混合偏导不等） |
| $D=0$ 时判"不是极值" | 判别法失效，需另判（$x^4+y^4$ 有极小，$x^3+y^3$ 没有） |
| 极坐标换元直接把 $\D x\D y$ 写成 $\D r\D\theta$ | 必须乘雅可比 $r$ |
| 拉格朗日乘子解出的点一定是极值 | 只是候选点；还要比较边界、验证约束规格（$\nabla g\ne0$） |

## AI / 工程应用

- **反向传播 = Jacobian 链式乘积**：一层是 $h=W x$，$\partial L/\partial x=W^\top\partial L/\partial h$；梯度消失/爆炸就是这些 Jacobian 的奇异值连乘。
- **二阶优化与曲率**：Hessian 最大特征值决定梯度下降的稳定步长（$\eta<2/\lambda_{\max}$）；牛顿法用 $H^{-1}\nabla f$ 修正方向；大模型里用 K-FAC / 低秩近似避免 $O(n^2)$ 存储。
- **约束优化与 KKT**：SVM 的对偶、带约束的强化学习、模型压缩中的稀疏约束都写成拉格朗日形式；$\lambda$ 有"影子价格"解释（约束放松一单位目标改善多少）。
- **概率与高维积分**：多维高斯归一化、变分推断里的 ELBO 都是高维积分；重参数化技巧本质是换元并乘上 Jacobian 行列式（归一化流的核心）。
- **计算机视觉 / 图形学**：图像梯度 $\nabla I$ 做边缘检测，法向量与光照用 $\nabla F$；Stokes / 散度定理用于流体仿真与物理量守恒。

## 遗留疑问

> [!question] 未解决：隐函数定理（含方程组版本）的压缩映射证明尚未自己写一遍，只会用结论。

> [!question] 未解决：Stokes 定理在微分形式语言下的统一表述（$\int_\Omega \D\omega=\int_{\partial\Omega}\omega$）与三大公式的逐一对应，待补。

> [!question] 未解决：KKT 条件中约束规格（LICQ / Slater）的作用，以及非凸情形下"驻点"与"全局最优"的差距，尚未整理。

## 相关

- [[Level2_Derivatives_and_Differentials]] · [[Level4_Integrals]] · [[Level6_Series_and_ODEs]] · [[Advanced Mathematics MOC]]
- 一元可导与可微的等价性对照：[[Level2_Derivatives_and_Differentials]] 定义节
