---
title: Level 6 级数与常微分方程
type: concept
subject: advanced-mathematics
level: Level 6
tags:
  - mathematics
  - calculus
  - series
  - ode
created: 2026-11-04
updated: 2026-11-04
source: 按 config.md 的 Level 模板整理；定义与公式据同济《高等数学》第七版下册 §11–§12、§7 微分方程，MIT OCW 18.03 相关章节；本库 `Exercises/` 暂无 Level 6 作业，本篇为体系补链，重点对接数值逼近与动力系统。
---

# Level 6 级数与常微分方程

> 承接 [[Level3_Mean_Value_Theorems]] 的 Taylor 公式与 [[Level4_Integrals]] 的广义积分：把"有限项求和 / 有限区间累积"推广到**无限**，就得到级数；把"已知导数求原函数"推广到"导数由方程隐式给出"，就得到微分方程。
> 主线：无限过程是否可控，取决于**收敛性**。

## 直觉

级数是"把无穷多个数加起来"，但加法只对有限项定义，所以必须先用部分和 $S_n$ 转成一个**数列极限**问题 —— 整个 Level 1 的工具立刻可用。审敛法其实都是在回答："余项 $R_n=S-S_n$ 是否趋于 0、以多快速度趋于 0"。

ODE 反过来：不知道函数，只知道它满足的变化规律 $\dot y=F(t,y)$。解的存在唯一性、数值解的误差，都是"用局部线性化一步步推进"（Euler 法就是 [[Level3_Mean_Value_Theorems]] 的一阶 Taylor）。

## 定义

**数项级数** $\sum_{n=1}^\infty a_n$ 收敛 $\iff$ 部分和数列 $S_n=\sum_{k=1}^n a_k$ 收敛；和为 $S=\lim S_n$。必要条件 $a_n\to0$（不充分，调和级数反例）。

**绝对收敛 / 条件收敛** $\sum|a_n|$ 收敛称绝对收敛；$\sum a_n$ 收敛但 $\sum|a_n|$ 发散称条件收敛。绝对收敛可任意重排且级数和不变；条件收敛重排可得到任意实数甚至发散（Riemann 重排定理）。

**幂级数** $\sum_{n=0}^\infty a_n(x-x_0)^n$，收敛半径 $R=\dfrac1{\limsup\sqrt[n]{|a_n|}}$（或 $R=\lim\left|\frac{a_n}{a_{n+1}}\right|$，极限存在时）。$|x-x_0|<R$ 绝对收敛，$>R$ 发散，$=\pm R$ 端点须单独判别。在收敛区间内可逐项求导、逐项积分，且收敛半径不变（端点可能变）。

**函数项一致收敛** $S_n(x)\to S(x)$ 一致 $\iff\sup_x|S_n(x)-S(x)|\to0$。一致收敛保证：极限与积分/求导可交换、极限函数连续（配合各项连续）。Weierstrass M-判别法：$\sum\sup|u_n(x)|$ 收敛 $\Rightarrow\sum u_n$ 一致收敛。

**ODE 基本概念** 方程 $F(t,y,y',\dots,y^{(n)})=0$；最高阶导数的阶数称阶；含 $n$ 个任意常数的解称通解；由初值确定常数的问题叫初值问题（Cauchy 问题）。若 $F$ 对 $y$ 满足 Lipschitz 条件且连续，则初值问题局部存在唯一解（Picard–Lindelöf）。

## 关键公式与结论

**正项级数审敛法**（$\sum a_n,\ a_n\ge0$）

| 方法 | 判据 | 备注 |
| --- | --- | --- |
| 比较 / 极限比较 | $a_n\le C b_n$，或 $\lim a_n/b_n=c\in(0,\infty)$ | 常与 $p$-级数 $\sum n^{-p}$（$p>1$ 收敛）比 |
| 比值（d'Alembert） | $\lim a_{n+1}/a_n=\rho<1$ 收敛，$>1$ 发散 | $\rho=1$ **失效**（$1/n$ 与 $1/n^2$ 都是 1） |
| 根值（Cauchy） | $\lim\sqrt[n]{a_n}=\rho<1$ 收敛 | 比比值法适用范围略广 |
| 积分判别 | $a_n=f(n)$、$f$ 正递减，则 $\sum a_n$ 与 $\int_1^\infty f$ 同敛散 | 给出 $\sum n^{-p}$ 的完整结论 |
| Raabe | $\rho=1$ 时用 $n\left(\frac{a_n}{a_{n+1}}-1\right)>1$ 收敛 | 比值法的补位 |

**交错级数 Leibniz 判别** $a_n\downarrow0\Rightarrow\sum(-1)^{n-1}a_n$ 收敛，且 $|R_n|\le a_{n+1}$，余项符号与首项同。误差估计是这一判别法最实用的部分。

**五个必背 Taylor 级数**（$|x|<1$ 除特别注明）

$$e^x=\sum_{n=0}^\infty\frac{x^n}{n!}\ (x\in\R),\quad \sin x=\sum_{n=0}^\infty\frac{(-1)^nx^{2n+1}}{(2n+1)!},\quad \cos x=\sum_{n=0}^\infty\frac{(-1)^nx^{2n}}{(2n)!},$$

$$\ln(1+x)=\sum_{n=1}^\infty\frac{(-1)^{n-1}x^n}{n}\ (-1<x\le1),\qquad (1+x)^\alpha=\sum_{n=0}^\infty\binom\alpha n x^n\ (-1<x<1).$$

几何级数 $\sum_{n=0}^\infty q^n=\frac1{1-q}$（$|q|<1$）是所有收敛比较的基准；$\sum\frac1{n(n+1)}=1$ 是裂项相消的原型。

**Fourier 级数（$2\pi$ 周期）**

$$f(x)\sim\frac{a_0}2+\sum_{n=1}^\infty\bigl(a_n\cos nx+b_n\sin nx\bigr),\quad a_n=\frac1\pi\int_{-\pi}^\pi f\cos nx\,\D x,\quad b_n=\frac1\pi\int_{-\pi}^\pi f\sin nx\,\D x.$$

Dirichlet 条件收敛定理：分段光滑时级数收敛到 $\frac{f(x^+)+f(x^-)}2$（间断点取左右平均）。

**一阶 ODE 可解类型**

| 类型 | 形式 | 解法 |
| --- | --- | --- |
| 可分离变量 | $y'=f(x)g(y)$ | $\int\frac{\D y}{g(y)}=\int f(x)\D x+C$（注意 $g(y)=0$ 的常值解） |
| 齐次 | $y'=\varphi(y/x)$ | 令 $u=y/x$，$y'=u+xu'$ 化为可分离 |
| 一阶线性 | $y'+P(x)y=Q(x)$ | $y=e^{-\int P}\left(\int Q\,e^{\int P}\D x+C\right)$ |
| Bernoulli | $y'+P y=Q y^n$ | 令 $z=y^{1-n}$ 化为线性 |
| 恰当方程 | $M\D x+N\D y=0,\ M_y=N_x$ | 找势函数 $u$ 使 $\D u=M\D x+N\D y$ |

**二阶常系数线性** $y''+py'+qy=0$，特征方程 $r^2+pr+q=0$：

- 两相异实根：$y=C_1e^{r_1x}+C_2e^{r_2x}$；
- 重根 $r$：$y=(C_1+C_2x)e^{rx}$；
- 共轭复根 $\alpha\pm i\beta$：$y=e^{\alpha x}(C_1\cos\beta x+C_2\sin\beta x)$。

非齐次用待定系数法（右端为 $e^{\lambda x}P_m(x)$ / $\sin\beta x$ 等）或常数变易法；共振时乘 $x^k$。

**Euler 法与局部误差** $y_{n+1}=y_n+hF(t_n,y_n)$，局部截断误差 $O(h^2)$、全局 $O(h)$；RK4 全局 $O(h^4)$。这些阶数全部由 Taylor 余项给出。

## 常见误解

| 错误 | 正确 |
| --- | --- |
| $a_n\to0\Rightarrow\sum a_n$ 收敛 | 必要不充分；$\sum 1/n$ 发散，$\sum 1/n^2$ 收敛 |
| 比值法 $\rho=1$ 判"发散" | 完全失效，换 Raabe / 比较 / 积分判别 |
| 收敛半径内端点自动收敛 | 端点必须单独判（$\ln(1+x)$ 在 $x=1$ 收敛、$x=-1$ 发散） |
| 条件收敛级数可随意换序、逐项相加 | 重排可改变和；逐项运算需绝对收敛（函数项需一致收敛） |
| 逐项求导总合法 | 需**导函数级数一致收敛**（仅逐点收敛不够） |
| 通解含所有解 | 可分离变量时 $g(y)=0$ 的奇解常漏；Bernoulli 变换也可能丢 $y=0$ |
| 二阶方程两个解线性无关即可 | 需 Wronskian 非零；重根时第二个解必须带 $x$ 因子 |
| Euler 法步长越小越好 | 截断误差降但舍入误差升，且刚性方程需极小步长才稳定（改用隐式方法） |

## AI / 工程应用

- **数值逼近与特殊函数**：$e^x,\sin x,\ln(1+x)$ 的库函数实现用 Taylor / Chebyshev 展开加区间收缩；误差界来自余项或交错级数估计。
- **优化算法的收敛级数**：步长 $\eta_k$ 满足 $\sum\eta_k=\infty,\ \sum\eta_k^2<\infty$（Robbins–Monro）是随机逼近收敛的经典条件，级数敛散直接决定"能否走到最优点又不被噪声带偏"。
- **Fourier 与信号 / 卷积**：频域滤波、FFT 加速卷积、位置编码（Transformer 用 $\sin/\cos$ 基）都建立在 Fourier 级数/变换上。
- **动力系统与神经 ODE**：$\dot h=f_\theta(h,t)$ 把残差网络视为连续深度；伴随法反向传播等价于解一个反向 ODE；稳定性分析用特征值（对应二阶常系数方程的特征根判号）。
- **生成模型**：归一化流中的连续流（FFJORD）、扩散模型的概率流 ODE、评分匹配的 SDE 都以 ODE / SDE 为语言。

## 遗留疑问

> [!question] 未解决：一致收敛判别（Weierstrass M、Dirichlet / Abel 对函数项级数）与"逐项求导三条件"尚未整理成可操作的自查流程。

> [!question] 未解决：Riemann 重排定理的构造性证明只理解了思路，未自己写完整。

> [!question] 未解决：Sturm–Liouville 本征值问题与 Fourier 级数完备性的关系，待后续课程补齐。

## 相关

- [[Level3_Mean_Value_Theorems]] · [[Level4_Integrals]] · [[Level5_Multivariable_Calculus]] · [[数列极限与新压轴_题型专题]]
- [[Advanced Mathematics MOC]] —— Level 0–6 路线总览
