---
title: Level 4 积分
type: concept
subject: advanced-mathematics
level: Level 4
tags:
  - mathematics
  - calculus
  - integral
created: 2026-11-04
updated: 2026-11-04
source: 按 config.md 的 Level 模板整理；定义与公式据同济《高等数学》第七版上册 §4.1–§4.5、下册 §9 章积分应用、MIT OCW 18.001 相关章节；本库 `Exercises/` 暂无 Level 4 作业，故本篇为体系补链，不替代后续课程讲义。
---

# Level 4 积分

> 承接 [[Level3_Mean_Value_Theorems]]：导数回答"变化多快"，积分回答"累积多少"。
> 主线只有一句：**微积分基本定理把"求面积"和"求原函数"接成同一件事**。

## 直觉

把区间切成很多小段，每段用"高 × 宽"近似，加起来再让宽度趋于 0，得到定积分。反过来，若已知变化率 $f$，求总量就是找 $F$ 使 $F'=f$ —— 这是同一个问题的两面。

几何上 $\int_a^b f$ 是带符号面积；物理上速度的时间积分是位移；概率上密度的积分是概率。三者都用同一个 $\int$ 符号，因为都是"把局部量按测度累加"。

## 定义

**不定积分** $F'=f$ 时称 $F$ 为 $f$ 的一个原函数，全体原函数记 $\int f(x)\D x=F(x)+C$。连续函数必有原函数；有原函数不一定可积，可积也不一定有原函数（如符号函数在含 0 区间上 Riemann 可积但无原函数，因为导函数有介值性）。

**定积分（Riemann 积分）** 设 $f$ 在 $[a,b]$ 有界，任取分割 $T:a=x_0<\cdots<x_n=b$ 与介点 $\xi_k\in[x_{k-1},x_k]$，令 $\lambda=\max\Delta x_k$，若

$$\lim_{\lambda\to0}\sum_{k=1}^n f(\xi_k)\Delta x_k$$

存在且与分割、介点取法无关，称 $f$ 在 $[a,b]$ 可积，值记 $\int_a^b f(x)\D x$。可积的充分条件：$f$ 连续，或 $f$ 有界且只有有限个间断点；必要条件：$f$ 有界。

**变上限积分** $\Phi(x)=\int_a^x f(t)\D t$；$f$ 连续时 $\Phi$ 可导且 $\Phi'(x)=f(x)$。

**反常（广义）积分** 无穷区间或无界函数，用极限定义：$\int_a^{+\infty}f=\lim_{A\to+\infty}\int_a^A f$；瑕积分在瑕点处取单侧极限。收敛与绝对收敛要区分（$\int_1^\infty\frac{\sin x}{x}\D x$ 条件收敛）。

## 关键公式与结论

**微积分基本定理（Newton–Leibniz）** 若 $f$ 连续且 $F'=f$：

$$\int_a^b f(x)\D x=F(b)-F(a),\qquad \frac{\D}{\D x}\int_{\alpha(x)}^{\beta(x)}f(t)\D t=f(\beta(x))\beta'(x)-f(\alpha(x))\alpha'(x).$$

被积函数含参变量 $x$ 时须先换元把 $x$ 移出积分号，或用 Leibniz 积分求导法则 $\frac{\D}{\D x}\int_{\alpha}^{\beta}f(x,t)\D t=\int_\alpha^\beta f_x\D t+f(x,\beta)\beta'-f(x,\alpha)\alpha'$。

**两大计算方法**

$$\int f(g(x))g'(x)\D x=\int f(u)\D u,\qquad \int u\D v=uv-\int v\D u.$$

定积分换元**必须同步换限**；$\D x=\varphi'(t)\D t$ 且要保证换元在积分区间上单调（否则分段）。分部积分优先级口诀："反对幂三指"（谁更靠前就取作 $u$）。

**常用结论**

$$\int_{-a}^{a}f(x)\D x=\begin{cases}2\int_0^a f(x)\D x,&f\ \text{偶}\\0,&f\ \text{奇}\end{cases},\qquad \int_a^{a+T}f(x)\D x=\int_0^T f(x)\D x\ (f\ \text{周期}\ T),$$

$$\int_0^{\pi/2}\sin^n x\D x=\int_0^{\pi/2}\cos^n x\D x=I_n=\frac{n-1}{n}I_{n-2},\quad I_1=1,\ I_0=\frac\pi2,$$

$$\int_0^\infty e^{-x^2}\D x=\frac{\sqrt\pi}2,\qquad \int_0^1 x^{p-1}(1-x)^{q-1}\D x=B(p,q)=\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}.$$

**积分中值定理** $f$ 连续、$g$ 不变号可积 $\Rightarrow\exists\xi$ 使 $\int_a^b fg=f(\xi)\int_a^b g$；取 $g\equiv1$ 得 $\frac1{b-a}\int_a^b f=f(\xi)$（平均值）。

**几何量** 平面面积 $\int_a^b|f-g|\D x$；弧长 $\int_a^b\sqrt{1+f'^2}\D x$（参数式 $\int\sqrt{x'^2+y'^2}\D t$）；旋转体体积 $\pi\int_a^b f^2\D x$（圆盘法）。

**定积分定义求数列极限** $\lim_{n\to\infty}\frac1n\sum_{k=1}^n f\!\left(\frac kn\right)=\int_0^1 f(x)\D x$；这是 [[数列极限与新压轴_题型专题]] 里的一条主力套路。

## 常见误解

| 错误 | 正确 |
| --- | --- |
| $\int f(x)\D x$ 是一个函数 | 是**一族**函数，别漏 $+C$；$\int_a^x f(t)\D t$ 才是确定的那一个（且在 $a$ 处为 0） |
| 有原函数 $\iff$ 可积 | 互不蕴含；可积要求有界，原函数要求 $f$ 具介值性（达布定理） |
| $\frac{\D}{\D x}\int_a^{x^2} f(t)\D t=f(x^2)$ | 漏链式：$2x\,f(x^2)$ |
| 换元只换被积式 | 上下限同步换；不换限就必须把新变量回代后再用原限 |
| $\int_{-\infty}^{\infty}f=\lim_{A\to\infty}\int_{-A}^{A}f$ | 这是 Cauchy 主值；反常积分要求两侧**分别**收敛 |
| $\int_a^b f<0$ 就是"面积为负" | 面积是 $\int|f|$；带符号面积才可为负 |
| 奇函数在对称区间积分为 0，所以任何反常积分都这样 | 前提是该积分收敛；发散情形不能套对称性 |

## AI / 工程应用

- **概率与统计**：密度归一化 $\int p=1$、期望 $\mathbb E[X]=\int xp(x)\D x$；KL 散度、交叉熵都是积分/期望形式，Monte Carlo 用采样均值逼近积分。
- **数值积分**：梯形法误差 $O(h^2)$、Simpson 法 $O(h^4)$（都可由 Taylor 余项推出，见 [[Level3_Mean_Value_Theorems]]）；高维时网格法指数爆炸，改用 Monte Carlo（误差 $O(1/\sqrt N)$ 与维数无关）。
- **信号处理**：卷积 $(f*g)(t)=\int f(\tau)g(t-\tau)\D\tau$ 是 CNN 与线性时不变系统的数学核心；Fourier 变换 $\hat f(\omega)=\int f(t)e^{-i\omega t}\D t$ 是同一类积分算子。
- **连续归一化流 / 神经常微分方程**：把残差网络看作 ODE $\dot h=f_\theta(h,t)$，则输出是 $\int f_\theta\D t$，反向传播用伴随灵敏度法（adjoint method），本质仍是积分与变上限求导。

## 遗留疑问

> [!question] 未解决：Lebesgue 积分与 Riemann 积分的确切包含关系、以及"几乎处处连续 $\iff$ Riemann 可积"（Lebesgue 判据）尚未自己证明。

> [!question] 未解决：反常积分的 Dirichlet / Abel 审敛法只记了结论，未整理证明与典型反例。

> [!question] 未解决：含参变量积分一致收敛与"求导/积分换序"的充分条件，待 [[Level6_Series_and_ODEs]] 的函数项级数一致收敛一节对照补全。

## 相关

- [[Level3_Mean_Value_Theorems]] · [[Level5_Multivariable_Calculus]] · [[Level6_Series_and_ODEs]] · [[数列极限与新压轴_题型专题]]
- [[Advanced Mathematics MOC]] —— Level 路线与概念索引
