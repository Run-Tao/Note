---
title: Level 2 导数与微分
type: concept
subject: advanced-mathematics
level: Level 2
tags:
  - mathematics
  - calculus
  - derivative
created: 2026-11-04
updated: 2026-11-04
source: 按 config.md 的 Level 模板整理；定义与公式据同济《高等数学》第七版上册 §2.1–§2.5、本库 `Exercises/作业10–13`、期中复习 `sec2c.tex` 导数应用总表；真题口径据 `Resources/真题考点索引_2008-2025`。题型做法另见 [[导数计算_题型专题]]，本篇只承载概念与结论。
---

# Level 2 导数与微分

> 承接 [[Level1_Limits_and_Continuity]]：极限回答"趋近什么"，导数回答"变化多快"。
> 全篇主线是**局部线性化**（local linearization）：在一点附近用一条直线代替曲线，误差是比 $\Delta x$ 更高阶的无穷小。

## 直觉

把一个函数放大到足够小的尺度，光滑的曲线看起来就是直线。导数 $f'(x_0)$ 是这条直线的斜率，微分 $\D y=f'(x_0)\Delta x$ 是用直线算出的**近似增量**。

这件事的价值不在"会求导"，而在于：任何复杂函数在某点附近都可以被一次多项式控制，于是"局部的性质"（升降、凹凸、极值）和"整体的估计"（中值定理、Taylor）都有了着力点。工程中数值差分、反向传播、灵敏度分析用的都是同一个想法。

## 定义

**导数（derivative）** 设 $f$ 在 $x_0$ 的某邻域有定义，若

$$f'(x_0)=\lim_{\Delta x\to0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$$

存在（有限），称 $f$ 在 $x_0$ 可导。等价写法 $f'(x_0)=\lim_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}$。

- **右导数 / 左导数**：把 $\Delta x\to0$ 换成 $\Delta x\to0^+$ / $0^-$，记 $f'_+(x_0)$、$f'_-(x_0)$。可导 $\iff$ 两者都存在且相等。
- **可导 $\Rightarrow$ 连续**：由 $f(x)-f(x_0)=\frac{f(x)-f(x_0)}{x-x_0}(x-x_0)\to f'(x_0)\cdot0=0$。逆命题不成立（$f=|x|$ 在 0 处连续不可导）。
- **微分（differential）**：若 $\Delta y=f'(x_0)\Delta x+o(\Delta x)$，称 $f$ 在 $x_0$ 可微，线性主部 $\D y=f'(x_0)\D x$ 为微分。一元函数里**可导 $\iff$ 可微**（多元不成立，见 [[Level5_Multivariable_Calculus]]）。
- **高阶导数**：$f^{(n)}=(f^{(n-1)})'$；$f''$ 描述斜率的变化，几何上对应凹凸。

## 关键公式与结论

**基本求导法则**（下列各函数均在相应点可导）

$$\left(\frac uv\right)'=\frac{u'v-uv'}{v^2}\ (v\ne0),\qquad \bigl(f\circ g\bigr)'(x)=f'(g(x))\,g'(x),\qquad (f^{-1})'(y)=\frac1{f'(x)}.$$

反函数求导要求 $f'(x)\ne0$；链式法则是反向传播（backpropagation）的数学原型。

**隐函数与参数方程** 设 $F(x,y)=0$ 且 $F_y\ne0$：

$$\frac{\D y}{\D x}=-\frac{F_x}{F_y},\qquad \frac{\D^2 y}{\D x^2}=-\frac{F_{xx}F_y^2-2F_{xy}F_xF_y+F_{yy}F_x^2}{F_y^3}.$$

参数方程 $x=x(t),\ y=y(t)$，须 $x'(t)\ne0$：

$$\frac{\D y}{\D x}=\frac{y'(t)}{x'(t)},\qquad \frac{\D^2 y}{\D x^2}=\frac{\D}{\D t}\!\left(\frac{y'}{x'}\right)\!\Big/ x'=\frac{y''x'-y'x''}{(x')^3}.$$

极坐标 $r=r(\theta)$ 用 $x=r\cos\theta,\ y=r\sin\theta$ 化为参数形式：

$$\frac{\D y}{\D x}=\frac{r'\sin\theta+r\cos\theta}{r'\cos\theta-r\sin\theta}.$$

**Leibniz 公式与常用 $n$ 阶导**

$$(uv)^{(n)}=\sum_{k=0}^n\binom nk u^{(k)}v^{(n-k)},$$

$$(e^{ax})^{(n)}=a^ne^{ax},\quad (\sin x)^{(n)}=\sin\!\left(x+\frac{n\pi}2\right),\quad \bigl(\ln(1+x)\bigr)^{(n)}=\frac{(-1)^{n-1}(n-1)!}{(1+x)^n},\quad \left(\frac1{1+x}\right)^{(n)}=\frac{(-1)^n n!}{(1+x)^{n+1}}.$$

考场上高频的"**代点消项**"：求 $(uv)^{(n)}(x_0)$ 时，若某个因子在 $x_0$ 处连同高阶导大量为零，展开式只剩少数几项。例：$(x^2 e^x)^{(10)}(0)=\binom{10}{2}\cdot2!\cdot e^0=90$（只有 $k=2$ 那一项存活）。

**一阶微分形式不变性** 无论 $x$ 是自变量还是中间变量，$\D y=f'(x)\D x$ 形式相同；**二阶微分没有这个性质**（$\D^2y=f''(x)\D x^2+f'(x)\D^2x$，当 $x$ 为自变量时 $\D^2x=0$）。

**导函数的特殊性质：达布定理（Darboux）** 若 $f$ 在区间上处处可导，则 $f'$ 具有介值性，**不可能有第一类间断点**（跳跃不可能）。注意边界：只在去心邻域可导不保证，如 $f=|x|$ 的导函数在 0 处跳跃；$f'$ 可以有第二类间断点。

## 常见误解

| 错误 | 正确 |
| --- | --- |
| 连续 $\Rightarrow$ 可导 | 反例 $|x|$、$\sqrt[3]x$（后者在 0 处竖直切线，导数为无穷大即不可导） |
| $f'(x_0)=\lim_{x\to x_0}f'(x)$ | 右端可能不存在（$f=x^2\sin\frac1x$，$x\ne0$，$f(0)=0$ 可导但 $f'$ 在 0 处不连续）；求 $f'(x_0)$ 一定回到定义或"先验连续再求导" |
| $\dfrac{\D^2 y}{\D x^2}=\dfrac{y''(t)}{x''(t)}$ | 必须是 $\dfrac{\D}{\D t}(y'/x')$ 再除以 $x'$；直接相除是最高频错题 |
| 分段点直接套导函数公式 | 分段点必须用**定义**算左右导数，并先验连续性 |
| $F_y=0$ 也能用隐函数公式 | 公式分母为零；此时可能 $y$ 不是 $x$ 的函数，或存在竖直切线 |
| 极坐标中 $\theta$ 可以随便取 | $r<0$ 时点落在反向射线上，题目给的 $\theta$ 范围决定是哪个点；先算 $(x,y)$ 再求斜率 |
| $f''(x_0)=0\Rightarrow$ 拐点 | 须左右凹凸性改变（$x^4$ 在 0 处反例）；$f''$ 不存在的点也可能是拐点 |

## AI / 工程应用

- **反向传播**：复合函数的链式法则在計算图上反向累乘 Jacobian，本质是同一条公式；"梯度消失"就是连乘的导数绝对值长期小于 1。
- **梯度下降与学习率**：$f(x-\eta\nabla f)\approx f(x)-\eta\|\nabla f\|^2$ 是 Taylor 一阶近似，步长上界由二阶信息（Lipschitz 常数）给出，见 [[Level3_Mean_Value_Theorems]]。
- **自动微分 vs 数值差分**：前向差分 $\frac{f(x+h)-f(x)}h$ 截断误差 $O(h)$，中心差分 $O(h^2)$，但 $h$ 过小会被浮点舍入淹没；自动微分按链式法则传播，无截断误差。
- **灵敏度 / 条件数**：$\frac{|\Delta y|/|y|}{|\Delta x|/|x|}\approx\left|\frac{x f'(x)}{f(x)}\right|$，用来判断一个数值算法对输入扰动是否放大误差。

## 遗留疑问

> [!question] 未解决：达布定理的完整证明（用闭区间上连续函数最大值 + 导数极值的费马引理）尚未自己写一遍；目前只会用结论。

> [!question] 未解决：一元"可导 $\iff$ 可微"在多元失效的几何解释（偏导存在但沿不同方向逼近不一致），待 [[Level5_Multivariable_Calculus]] 补全后回看。

> [!question] 未解决：隐函数定理中 "$F$ 连续可微且 $F_y\ne0$" 的充分性证明，以及它在反函数存在性里的对偶形式。

## 相关

- [[Level1_Limits_and_Continuity]] · [[Level3_Mean_Value_Theorems]] · [[导数计算_题型专题]] · [[函数性态与作图_题型专题]]
- 讲义：`Resources/Downloads/讲义_导数计算专题.pdf`；真题索引：[[真题考点索引_2008-2025]]
