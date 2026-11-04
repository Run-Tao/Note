---
title: Level 3 中值定理与泰勒公式
type: concept
subject: advanced-mathematics
level: Level 3
tags:
  - mathematics
  - calculus
  - mean-value-theorem
  - taylor
created: 2026-11-04
updated: 2026-11-04
source: 按 config.md 的 Level 模板整理；定义与定理据同济《高等数学》第七版上册 §3.1–§3.3、本库 `Homework/Advanced Mathematics/中值定理专题/微分中值定理专题讲义.tex`（25 页专题讲义）与 `Exercises/作业14–16`；真题口径据 [[真题考点索引_2008-2025]]。本篇只写概念层与结论，证明套路与 24 道母题一律以专题讲义为准，不复制。
---

# Level 3 中值定理与泰勒公式

> 承接 [[Level2_Derivatives_and_Differentials]]：导数只是**一点**的信息，中值定理把一点的信息变成**一段区间**上的结论。
> 一句话概括：如果函数在两端之间光滑，那么"整体的平均变化率"一定 somewhere 等于"某点的瞬时变化率"。

## 直觉

开车从 A 到 B，平均时速 60，那么途中至少某一瞬间时速恰好是 60 —— 这就是拉格朗日中值定理。它把"函数差 $f(b)-f(a)$"这个整体量，用某点导数 $f'(\xi)$ 这个局部量表达出来，于是**只要能估计导数，就能估计函数**。

Taylor 公式是同一个想法的高阶版本：不只匹配一阶变化率，还匹配到 $n$ 阶，于是把函数写成多项式加一个可控余项。数值计算、优化算法的收敛性证明、极限计算，本质上都在用这个"多项式 + 余项"的结构。

## 定义

**费马引理** $f$ 在 $x_0$ 的邻域有定义、在 $x_0$ 可导，且 $x_0$ 是极值点 $\Rightarrow f'(x_0)=0$。这是全部中值定理的地基。

**罗尔定理（Rolle）** $f\in C[a,b]$、在 $(a,b)$ 可导、$f(a)=f(b)$ $\Rightarrow\exists\,\xi\in(a,b)$ 使 $f'(\xi)=0$。

**拉格朗日中值定理（Lagrange）** 去掉 $f(a)=f(b)$：

$$\exists\,\xi\in(a,b):\quad f(b)-f(a)=f'(\xi)(b-a).$$

**柯西中值定理（Cauchy）** 两个函数 $f,g$ 一起看（$g'(x)\ne0$）：

$$\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(\xi)}{g'(\xi)}.$$

取 $g(x)=x$ 退回 Lagrange；L'Hôpital 法则正是从 Cauchy 中值定理推出的。

**泰勒公式（Taylor）** 设 $f$ 在含 $x_0$ 的区间上 $n+1$ 阶可导：

$$f(x)=\sum_{k=0}^{n}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k+R_n(x),\qquad R_n=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}\ \text{（Lagrange 余项）},$$

其中 $\xi$ 介于 $x_0$ 与 $x$ 之间。若只要求 $f$ 在 $x_0$ 处 $n$ 阶可导，则有余项更弱的 Peano 形式 $R_n=o\bigl((x-x_0)^n\bigr)$ —— **极限计算用 Peano，不等式估计用 Lagrange**，两者条件不同，别混。

$n=0$ 时 Taylor 就是 Lagrange 中值定理；带 Peano 余项的 $n$ 阶展开是求 $\tfrac00$ 型极限最锋利的工具。

## 关键公式与结论

**六个必背麦克劳林展开**（$x\to0$，写 Peano 余项）

$$e^x=1+x+\frac{x^2}{2!}+\cdots+\frac{x^n}{n!}+o(x^n),\qquad \sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}+o(x^5),$$

$$\cos x=1-\frac{x^2}{2!}+\frac{x^4}{4!}+o(x^4),\qquad \ln(1+x)=x-\frac{x^2}2+\frac{x^3}3+o(x^3),$$

$$(1+x)^\alpha=1+\alpha x+\frac{\alpha(\alpha-1)}{2!}x^2+\cdots+\binom\alpha n x^n+o(x^n),\qquad \tan x=x+\frac{x^3}3+o(x^3).$$

一般项系数：$f(x)=(1+x)^\alpha$ 时 $f^{(n)}(0)=\alpha(\alpha-1)\cdots(\alpha-n+1)$（$n$ 个因子）。注意复合函数要先"提因子"再展开，例如 $\sqrt[3]{\sin x^3}=x\bigl(1-\frac{x^6}6+\cdots\bigr)^{1/3}$，详见 [[函数极限计算与泰勒_专项练习]] §4.5。

**辅助函数构造的四个原型**（证明 $f'(\xi)+\lambda f(\xi)=0$ 一类结论时倒推）

| 目标式 | 构造 | 依据 |
| --- | --- | --- |
| $f'(\xi)+\lambda f(\xi)=0$ | $F(x)=e^{\lambda x}f(x)$ | 乘积求导后再用 Rolle |
| $\xi f'(\xi)+n f(\xi)=0$ | $F(x)=x^n f(x)$ | 同上 |
| $\xi f'(\xi)-n f(\xi)=0$ | $F(x)=f(x)/x^n$ | 商求导 |
| $f'(\xi)+g'(\xi)f(\xi)=0$ | $F(x)=e^{g(x)}f(x)$ | 指数因子一般化 |

完整的万能构造表、15 类母题与 24 道训练题在 `Downloads/微分中值定理专题讲义.pdf` §3–§5，本篇不重复。

**由中值定理直接得到的三条常用结论**

1. $f'\equiv0$ 于区间 $\Rightarrow f\equiv C$（常数）；由此可证恒等式。
2. $f'>0$ 于区间 $\Rightarrow f$ 严格递增（逆不真：$x^3$ 严格递增但 $f'(0)=0$）。
3. 若 $|f'(x)|\le M$，则 $|f(b)-f(a)|\le M|b-a|$（Lipschitz 估计，是收敛性证明的常用一步）。

**L'Hôpital 法则的条件** 只用于 $\tfrac00$ 与 $\tfrac\infty\infty$；要求分母导数在去心邻域不为零、导数比的极限存在（或为 $\infty$）；加减式应先通分/有理化变成乘除再用；数列极限须先连续化（或改用 Stolz）。

## 常见误解

| 错误 | 正确 |
| --- | --- |
| 中值点 $\xi$ 可以算出来 | 定理只保证存在；题目要"求 $\xi$"时是另一类题，别把存在性当方程解 |
| 条件差不多就行 | 连续、可导、端点相等三条缺一不可；分段点、$|x|$、$x^{2/3}$ 是经典反例 |
| 洛必达对任何 $\tfrac00$ 都能用 | 导数比极限不存在时法则失效（如 $x^2\sin\frac1x/x$），要回泰勒 |
| 加减式里逐项等价替换 | 只有当**各项极限都存在**时才可拆；主部相消时必须展到更高阶（见 [[限时卷计算题补强_三型]]） |
| Peano 余项用来证不等式 | Peano 是定性 $o(\cdot)$，不能给数值上界；不等式要 Lagrange 余项 |
| 泰勒展开阶数随便取 | 原则：**展到分母同阶**；$1^\infty$ 里只把 $u-1$ 展到与 $1/v$ 同阶 |
| 把 $f^{(n)}(0)=\alpha(\alpha-1)\cdots$ 写成 $n-1$ 个因子 | 是 $n$ 个因子，从 $\alpha$ 递减到 $\alpha-n+1$ |

## AI / 工程应用

- **下降引理（descent lemma）**：若 $\nabla f$ 是 $L$-Lipschitz，则 $f(y)\le f(x)+\langle\nabla f(x),y-x\rangle+\frac L2\|y-x\|^2$，正是带 Lagrange 余项的一阶 Taylor 展开；由此得梯度下降步长 $\eta\le1/L$ 时函数值单调不增。
- **牛顿法收敛阶**：$f$ 在根附近二阶展开，余项 $\sim\frac{f''(\xi)}{2f'(x_k)}e_k^2$，解释二次收敛与初值敏感性。
- **数值误差与截断**：有限差分、Simpson 求积、ODE 单步法的局部截断误差都由 Taylor 余项给出；"阶数"就是余项里 $h$ 的幂。
- **模型近似 / 蒸馏**：用低阶多项式（或一阶线性化）在局部替代复杂模型，代价由余项控制；这是可解释性方法（如 LIME 的局部线性代理）的理论依据。

## 遗留疑问

> [!question] 未解决：柯西中值定理的几何解释（参数曲线 $(g(t),f(t))$ 的弦与切线平行）尚未自己画图验证。

> [!question] 未解决：Peano 余项要求"$x_0$ 处 $n$ 阶可导"、Lagrange 余项要求"区间上 $n+1$ 阶可导"，两者条件差一个全局性要求，尚未整理成可判定的自查清单。

> [!question] 未解决：L'Hôpital 法则在导数比极限不存在但原极限存在的例子（ oscillatory 反例），只见过结论没自己写完。

## 相关

- [[Level2_Derivatives_and_Differentials]] · [[Level4_Integrals]] · [[函数极限计算与泰勒_专项练习]] · [[中值定理与不等式证明_题型专题]]
- 专题讲义：`Resources/Downloads/微分中值定理专题讲义.pdf`（§2 三定理 / §3 构造法 / §4 母题 / §5 真题 / §6 不等式）
