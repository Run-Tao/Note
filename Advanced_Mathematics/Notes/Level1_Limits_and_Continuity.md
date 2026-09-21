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

> [!note] 我的原始理解（未验证，原始笔记）
> - **唯一性**：数列收敛，极限唯一。
> - **有界性**：数列收敛，则数列有界；逆否命题：若数列无界，则数列一定发散。
> - **保序性**：若 $\lim x_n = A,\ \lim y_n = B$，且 $A > B$，则 $\exists N \in \mathbb{N}$，当 $n>N$ 时 $x_n > y_n$。
> - **保号性**：
> - 子列

### 4.1 唯一性

**定理**：若数列 $\{x_n\}$ 收敛，则极限唯一：设 $\lim x_n = A$ 且 $\lim x_n = B$，则 $A = B$。

> [!info] Agent 补充：反证法证明
> 假设 $A \ne B$，不妨设 $A < B$，取 $\epsilon = \dfrac{B-A}{2} > 0$。
> 由 $\lim x_n = A$，存在 $N_1$，当 $n>N_1$ 时 $|x_n-A|<\epsilon$，即 $x_n < A+\epsilon = \dfrac{A+B}{2}$；
> 由 $\lim x_n = B$，存在 $N_2$，当 $n>N_2$ 时 $|x_n-B|<\epsilon$，即 $x_n > B-\epsilon = \dfrac{A+B}{2}$。
> 取 $N=\max\{N_1,N_2\}$，当 $n>N$ 时同时有 $x_n < \dfrac{A+B}{2}$ 与 $x_n > \dfrac{A+B}{2}$，矛盾。故 $A=B$。

### 4.2 有界性

**定理**：若 $\{x_n\}$ 收敛，则 $\{x_n\}$ 有界。

> [!info] Agent 补充：证明思路
> 设 $\lim x_n = A$。取 $\epsilon = 1$，存在 $N$，当 $n>N$ 时 $|x_n-A|<1$，于是 $|x_n| \le |A|+1$。
> 前有限项中取 $M_1=\max\{|x_1|,\dots,|x_N|\}$，令 $M=\max\{M_1,|A|+1\}$，则 $|x_n|\le M$ 对所有 $n$ 成立。

**逆否命题**：若数列无界，则一定发散。它说明"有界"是收敛的**必要条件**（但不充分）。

### 4.3 保序性

**定理**：设 $\lim x_n = A,\ \lim y_n = B$，且 $A > B$，则存在 $N \in \mathbb{N}$，当 $n>N$ 时 $x_n > y_n$。

> [!info] Agent 补充：证明思路
> 取 $\epsilon = \dfrac{A-B}{2} > 0$。存在 $N$，当 $n>N$ 时 $|x_n-A|<\epsilon$ 且 $|y_n-B|<\epsilon$，于是
> $$x_n > A-\epsilon = \frac{A+B}{2} = B+\epsilon > y_n.$$

> [!warning] 注意严格不等号
> 若只知 $A \ge B$，不能推出 $x_n \ge y_n$；反过来，若从某项起恒有 $x_n \ge y_n$ 且两极限存在，取极限只能得到 $A \ge B$（严格性可能丢失）。

### 4.4 保号性

**定理**：若 $\lim x_n = A > 0$，则存在 $N$，当 $n>N$ 时 $x_n > 0$。

> [!info] Agent 补充：证明
> 取 $\epsilon = \dfrac{A}{2} > 0$，存在 $N$，当 $n>N$ 时 $|x_n-A|<\dfrac{A}{2}$，即 $x_n > A-\dfrac{A}{2} = \dfrac{A}{2} > 0$。若 $A<0$，同理从某项起 $x_n<0$。

### 4.5 子列

- **子列（subsequence）**：按**严格递增**的下标 $n_1<n_2<\cdots$ 从原数列中选出的数列，记作 $\{x_{n_k}\}$；
- **基本性质**：若 $\{x_n\}$ 收敛于 $A$，则它的**任何子列**都收敛于 $A$；
- **逆否用法（判断发散）**：若能找到两个收敛于不同极限的子列，或找到一个发散的子列，则原数列发散。例如 $(-1)^n$ 的奇数项子列趋于 $-1$、偶数项子列趋于 $1$，故发散；
- **待补**：Weierstrass 致密性定理（有界数列必有收敛子列）——见"遗留疑问"。

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
> [!question] 未解决：子列与 Weierstrass 致密性定理（有界数列必有收敛子列）待补充证明。

## Related Notes

- [[Level0_Functions_and_Mappings|Level 0 函数与映射]] —— 前置
- [[Level2_Derivatives_and_Differentials|Level 2 导数与微分]] —— 下一章
- [[Learning_Roadmap]] —— 学习路线
- [[Advanced Mathematics MOC]] —— 知识库总览