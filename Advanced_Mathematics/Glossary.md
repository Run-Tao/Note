---
title: Advanced Mathematics Glossary
type: index
subject: advanced-mathematics
level: overview
tags:
  - mathematics
  - calculus
  - glossary
---

# Glossary 术语表

> 高等数学重要术语中英文对照。按主题分组，随学习进度补充。
> 术语定义详见对应笔记，这里只给一句话定位。

## 函数与映射（Level 0）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 映射 | mapping | 每输入恰好对应一个输出 | [[Level0_Functions_and_Mappings]] |
| 函数 | function | 两个非空数集之间的映射 | [[Level0_Functions_and_Mappings]] |
| 定义域 | domain $D(f)$ | 自变量的取值范围 | [[Level0_Functions_and_Mappings]] |
| 陪域 | codomain $Y$ | 声明输出所在的集合 | [[Level0_Functions_and_Mappings]] |
| 值域 | range $R(f)$ | 实际取到的输出集合，$R(f)\subseteq Y$ | [[Level0_Functions_and_Mappings]] |
| 自变量 / 因变量 | independent / dependent variable | 输入 / 输出 | [[Level0_Functions_and_Mappings]] |
| 显函数 / 隐函数 | explicit / implicit function | 直接写出 $y=f(x)$ / 由 $F(x,y)=0$ 隐含定义 | [[Level0_Functions_and_Mappings]] |
| 图像 | graph | 点集 $\{(x,f(x))\}$；不一定能画成曲线 | [[Level0_Functions_and_Mappings]] |
| 复合函数 | composite function | $g \circ f$，先 $f$ 后 $g$ | [[Level0_Functions_and_Mappings]] |
| 单射 / 满射 / 双射 | injective / surjective / bijective | 不同输入不同输出 / 值域=陪域 / 一一对应 | [[Level0_Functions_and_Mappings]] |
| 反函数 | inverse function | 双射的逆映射，$D(f^{-1})=R(f)$ | [[Level0_Functions_and_Mappings]] |
| 奇偶性 | parity | 偶：$f(-x)=f(x)$；奇：$f(-x)=-f(x)$ | [[Level0_Functions_and_Mappings]] |
| 单调性 | monotonicity | 严格单调 ⟹ 定义域到值域的双射 | [[Level0_Functions_and_Mappings]] |
| 有界性 | boundedness | 存在 $M$ 使 $|f(x)|\le M$ | [[Level0_Functions_and_Mappings]] |
| 周期性 | periodicity | 存在 $T>0$ 使 $f(x+T)=f(x)$ | [[Level0_Functions_and_Mappings]] |

## 极限与连续（Level 1）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 数列 | sequence | 定义在正整数集上的函数 $x_n=f(n)$ | [[Level1_Limits_and_Continuity]] |
| 数列极限 | limit of a sequence | 存在 $A$，$\forall\epsilon>0,\exists N$，$n>N$ 时 $|x_n-A|<\epsilon$ | [[Level1_Limits_and_Continuity]] |
| 收敛 / 发散 | convergent / divergent | 极限存在 / 极限不存在 | [[Level1_Limits_and_Continuity]] |
| ε-N 语言 | epsilon-N definition | 描述"无限接近"的严格语言 | [[Level1_Limits_and_Continuity]] |
| 单调有界准则 | monotone convergence theorem | 单调有界数列必收敛 | [[Level1_Limits_and_Continuity]] |

## 预告（后续 Level 展开）

| 中文 | English | 一句话 |
| --- | --- | --- |
| 极限 | limit | 自变量趋近时函数值的趋近目标（Level 1） |
| 连续 | continuity | 极限等于函数值（Level 1） |
| 导数 | derivative | 瞬时变化率 / 切线斜率（Level 2） |
| 微分 | differential | 局部线性近似（Level 2） |
| 中值定理 | mean value theorem | 用平均变化率控制瞬时变化率（Level 3） |
| 定积分 | definite integral | 黎曼和的极限，累积量（Level 4） |
| 微积分基本定理 | fundamental theorem of calculus | 微分与积分互为逆运算（Level 4） |
| 偏导数 | partial derivative | 只对一个变量求导（Level 5） |
| 梯度 | gradient | 各偏导组成的向量，指向增长最快方向（Level 5） |
