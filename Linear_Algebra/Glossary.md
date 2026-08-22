---
title: Glossary 术语表
type: index
subject: linear-algebra
level: overview
tags:
  - mathematics
  - linear-algebra
  - glossary
---

# Glossary 术语表

> 重要数学术语中英文对照。按主题分组，已覆盖 Level 0–2 现有笔记；后续随学习进度补充。
> 术语定义详见对应笔记，这里只给一句话定位。
> **学习者背景**：高中数学已扎实，以下"数学基础"组属高中已掌握范畴，仅保留对照，不再展开。

## 数学基础（Level 0，高中已掌握，仅对照）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 变量 | variable | 可取不同值的量 | [[Level0_Mathematical_Foundation]] |
| 常量 | constant | 取值固定的量 | [[Level0_Mathematical_Foundation]] |
| 参数 | parameter | 同一问题中固定的可调量 | [[Level0_Mathematical_Foundation]] |
| 未知量 | unknown | 方程中待求解的量 | [[Level0_Mathematical_Foundation]] |
| 集合 | set | 确定对象的整体；无序、不重复 | [[Level0_Mathematical_Foundation]] |
| 子集 | subset | 每个元素都属于另一集合 | [[Level0_Mathematical_Foundation]] |
| 笛卡尔积 | Cartesian product | 所有有序对的集合 $A\times B$ | [[Level0_Mathematical_Foundation]] |
| 映射 | mapping / function | 每个输入恰好对应一个输出 | [[Level0_Mathematical_Foundation]] |
| 定义域 / 陪域 / 值域 | domain / codomain / range | 输入集合 / 声明输出集合 / 实际输出集合 | [[Level0_Mathematical_Foundation]] |
| 单射 / 满射 / 双射 | injective / surjective / bijective | 不同输入不同输出 / 值域=陪域 / 一一对应 | [[Level0_Mathematical_Foundation]] |
| 命题 / 定理 / 引理 / 推论 | proposition / theorem / lemma / corollary | 可判真假的陈述 / 被证的重要命题 / 中间结论 / 直接推论 | [[Level0_Mathematical_Foundation]] |

## 向量与矩阵（Level 1）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 向量 | vector | 可相加、可数乘的对象；带位置的数值组 | [[Level1_Vectors_and_Matrices]] |
| 标量 | scalar | 单个的数（向量之外的普通量） | [[Level1_Vectors_and_Matrices]] |
| 分量 | component | 向量的第 $i$ 个元素 $x_i$ | [[Level1_Vectors_and_Matrices]] |
| 转置 | transpose | 行列互换，$A^T$ | [[Level1_Vectors_and_Matrices]] |
| 零向量 | zero vector | 全部分量为 0，长度为 0 | [[Level1_Vectors_and_Matrices]] |
| 范数 | norm | 长度；欧氏范数 $\lVert x\rVert=\sqrt{\sum x_i^2}$ | [[Level1_Vectors_and_Matrices]] |
| 单位向量 | unit vector | 长度为 1 的向量 | [[Level1_Vectors_and_Matrices]] |
| 点积 / 内积 | dot product / inner product | 分量乘积之和 $x^Ty$，编码长度与夹角 | [[Level1_Vectors_and_Matrices]] |
| 余弦相似度 | cosine similarity | 归一化的点积，只衡量方向 | [[Level1_Vectors_and_Matrices]] |
| 正交 | orthogonal | 点积为零，即垂直 | [[Level1_Vectors_and_Matrices]] |
| 线性组合 | linear combination | 先缩放再相加：$\sum a_i v_i$ | [[Level1_Vectors_and_Matrices]] |
| 张成空间 | span | 所有线性组合的集合 | [[Level1_Vectors_and_Matrices]] |
| 线性相关 / 无关 | linearly dependent / independent | 存在冗余 / 无冗余（严格判断回到定义） | [[Level1_Vectors_and_Matrices]] |
| 矩阵 | matrix | 排成矩形的数表；数据 / 方程组 / 变换三重身份 | [[Level1_Vectors_and_Matrices]] |
| 方阵 | square matrix | 行数 = 列数的矩阵 | [[Level1_Vectors_and_Matrices]] |
| 零矩阵 | zero matrix | 全为 0；加法的"零" | [[Level1_Vectors_and_Matrices]] |
| 单位矩阵 | identity matrix $I$ | 对角线 1 其余 0；乘法的"一" | [[Level1_Vectors_and_Matrices]] |
| 对角矩阵 | diagonal matrix | 非对角全 0；各坐标轴独立缩放 | [[Level1_Vectors_and_Matrices]] |
| 对称矩阵 | symmetric matrix | $A^T=A$；如协方差矩阵 | [[Level1_Vectors_and_Matrices]] |
| 矩阵乘法 | matrix multiplication | 行×列点积；对应变换的复合 | [[Level1_Vectors_and_Matrices]] |
| 线性变换 | linear transformation | 保持加法与数乘的映射；矩阵=变换的坐标表示 | [[Level1_Vectors_and_Matrices]] |

## 线性方程组（Level 2）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 线性方程组 | linear system | $A\mathbf{x}=\mathbf{b}$ | [[Level2_Linear_Systems]] |
| 系数矩阵 | coefficient matrix | 未知量系数组成的矩阵 $A$ | [[Level2_Linear_Systems]] |
| 增广矩阵 | augmented matrix | $[A\mid b]$，消元同时作用于两边 | [[Level2_Linear_Systems]] |
| 初等行变换 | elementary row operation | 换行 / 行乘非零数 / 行加另一行倍数 | [[Level2_Linear_Systems]] |
| 高斯消元 | Gaussian elimination | 消元到行阶梯形后回代 | [[Level2_Linear_Systems]] |
| 高斯-约旦消元 | Gauss-Jordan elimination | 消元到 RREF，解直接读出 | [[Level2_Linear_Systems]] |
| 行阶梯形 | row echelon form (REF) | 主元逐行右移的阶梯形 | [[Level2_Linear_Systems]] |
| 最简行阶梯形 | reduced row echelon form (RREF) | 主元为 1 且所在列其余为 0 | [[Level2_Linear_Systems]] |
| 主元 | pivot | 每行第一个非零元 | [[Level2_Linear_Systems]] |
| 自由变量 | free variable | 非主元列变量，取值自由 | [[Level2_Linear_Systems]] |
| 秩 | rank | 主元个数 = 独立信息量 | [[Level2_Linear_Systems]] |
| 齐次方程组 | homogeneous system | $A\mathbf{x}=\mathbf{0}$，解集过原点 | [[Level2_Linear_Systems]] |
| 特解 / 通解 | particular solution / general solution | 一个具体解 / 全部解 = 特解 + 齐次通解 | [[Level2_Linear_Systems]] |
| 超定 / 欠定 | overdetermined / underdetermined | 方程多 / 未知数多 | [[Level2_Linear_Systems]] |

## 预告（后续 Level 展开）

| 中文 | English | 一句话 |
| --- | --- | --- |
| 特征值 / 特征向量 | eigenvalue / eigenvector | 变换下方向不变、只被缩放的向量（Level 5） |
| 奇异值分解 | SVD | 把矩阵拆成 $U\Sigma V^T$（Level 6） |
| 主成分分析 | PCA | 协方差矩阵特征分解找主方向（Level 6/7） |
| 零空间 / 列空间 | null space / column space | 齐次解集 / 列张成的子空间（Level 3） |
| 向量空间 / 子空间 | vector space / subspace | 满足封闭性的集合（Level 3） |
| 基 / 维数 | basis / dimension | 极小生成集 / 基的大小（Level 3） |
| 投影 / 最小二乘 | projection / least squares | 最佳近似 = 投影（Level 4） |
| 嵌入 | embedding | 把离散对象映射为向量，语义相似 ⇔ 距离近（AI 应用） |
