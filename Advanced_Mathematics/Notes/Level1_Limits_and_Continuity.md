---
title: Level 1 极限与连续
type: concept
subject: advanced-mathematics
level: Level 1
tags:
  - mathematics
  - calculus
created: 2026-09-20
source: 整理自 Inbox/2026-09-20（数列极限草稿）；按高等数学教材通用定义补全
---

# Level 1 极限与连续

> 本章从**数列的极限**开始：极限是微积分的基石，导数、积分、级数都建立在它之上。
> 本笔记当前整理"数列极限"部分；函数极限与连续性待后续补充。

## 直觉

极限描述"无限接近"：我们不关心第 $n$ 项具体是多少，而关心 $n$ 越来越大时 $x_n$ 趋近于谁。
它把"趋于""大约"这类模糊说法变成可以严格证明的命题——这就是 $\epsilon$-$N$ 语言存在的意义。

## 1. 数列：定义在正整数集上的函数

> [!note] 我的原始理解（未验证）
> $$
> \begin{aligned}
> & f: \mathbb{N}_+ \rightarrow \mathbb{R} \\
> & n \mapsto x_n = f(n), \quad n \in \mathbb{N}_+
> \end{aligned}
> $$

- **数列（sequence）**：定义域为正整数集 $\mathbb{N}_+$（有时取 $\mathbb{N}$）的函数；
- 记作 $\{x_n\}$ 或 $(x_n)$，$x_n$ 称为通项；
- 因为定义域离散有序，数列可以理解为"按顺序排好的一列数"；
- 把它看作函数，就能直接沿用函数的单调性、有界性等概念。

## 2. 数列的单调性与有界性

- **单调性**：
  - 递增：$x_n \le x_{n+1}$；严格递增：$x_n < x_{n+1}$；
  - 递减 / 严格递减同理。
- **有界性**：存在 $M>0$，使 $|x_n| \le M$ 对所有 $n$ 成立；
  - 等价于同时有上界与下界；
  - 例：$x_n = (-1)^n$ 有界但不收敛；$x_n = n$ 无界。
- **单调有界准则**（预告）：单调且有界的数列一定收敛（依赖实数的完备性）——本章后续补充证明。

## 3. 数列极限的 ε-N 定义

> [!note] 我的原始理解（未验证，原始笔记未写完）
> 对数列 $x_n$，若存在常数 $A \in \mathbb{R}$，满足 $\forall \epsilon > 0, \exists N \in \mathbb{N}$，使得当 $n>N$ 时……

> [!info] Agent 补充（补全定义）
> **定义**：设 $\{x_n\}$ 为数列，$A \in \mathbb{R}$。若
> $$\forall \epsilon > 0,\ \exists N \in \mathbb{N},\ \forall n > N:\ |x_n - A| < \epsilon$$
> 则称数列 $\{x_n\}$ **收敛**于 $A$，$A$ 称为它的**极限**，记作
> $$\lim_{n\to\infty} x_n = A \quad \text{或} \quad x_n \to A\ (n \to \infty)$$
> 若极限不存在，则称数列**发散**。

**定义拆解**（每一段都在回答一个问题）：

| 符号 | 含义 |
| --- | --- |
| $\forall \epsilon > 0$ | 对任意给定的接近程度（无论多小） |
| $\exists N \in \mathbb{N}$ | 总能找到某一项作为起点（$N$ 可以依赖 $\epsilon$） |
| $\forall n > N$ | 从这一项之后的所有项 |
| $\lvert x_n - A\rvert < \epsilon$ | 都落在 $A$ 的 $\epsilon$ 邻域 $(A-\epsilon, A+\epsilon)$ 内 |

**几何解释**：在数轴上以 $A$ 为中心画任意窄的带子 $(A-\epsilon, A+\epsilon)$，数列从某一项起就**全都**落在带子里，不再跑出去。

## 4. 收敛数列的基本性质

- **唯一性**：极限若存在则唯一；
- **有界性**：收敛 $\Rightarrow$ 有界（反之不成立：$(-1)^n$ 有界但发散）；
- **保号性**：若 $\lim x_n = A > 0$，则存在 $N$，当 $n>N$ 时 $x_n > 0$（取 $\epsilon = A/2$ 即可证明）。

## 常见误解

1. **"$\epsilon$ 是某个很小的数"** → 错。$\epsilon$ 是**任意**小的正数，必须对一切 $\epsilon>0$ 都成立。
2. **"$N$ 与 $\epsilon$ 无关"** → 错。一般 $N$ 依赖于 $\epsilon$：$\epsilon$ 越小，$N$ 往往越大。
3. **"从某项起很接近"** → 不够。要求从某项起**一直**保持在 $\epsilon$ 带子里（是"以后都接近"，不是"偶尔接近"）。
4. **"有界就收敛"** → 错。单调 + 有界才收敛；$(-1)^n$ 有界但发散。
5. **"极限值必须是数列中的某一项"** → 错。例如 $x_n = 1/n$ 的极限是 0，但 0 不是任何一项。

## AI / 工程应用

- **迭代算法的收敛性**：梯度下降、牛顿法、幂法等都产生迭代序列 $\{x_n\}$，其收敛性用极限语言（有时用误差界）分析；
- **数值计算**：级数求和、数值积分把结果定义为某个序列的极限；
- **概率与统计**：大数定律说样本均值依概率收敛——同一套"趋于"的思想。

## 遗留疑问

> [!question] 未解决：用 $\epsilon$-$N$ 定义**证明**具体数列极限的例题待补（如 $\lim \frac{1}{n}=0$、$\lim \frac{2n+1}{n}=2$、$\lim q^n = 0\ (|q|<1)$）。

## Related Notes

- [[Level0_Functions_and_Mappings|Level 0 函数与映射]] —— 前置
- [[Level2_Derivatives_and_Differentials|Level 2 导数与微分]] —— 下一章
- [[Learning_Roadmap]] —— 学习路线
- [[Advanced Mathematics MOC]] —— 知识库总览