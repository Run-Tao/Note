# Notes 学习笔记

按 Level 组织，**一个 Level 一个主要文件**（如 `Level1_Vectors_and_Matrices.md`）；仅 SVD、PCA、特征值等独立价值高且被多次引用的概念才单独建笔记。

## 当前已有笔记

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [[Level0_Mathematical_Foundation\|Level0_Mathematical_Foundation]] | Level 0 数学基础（数学语言 / 集合 / 映射 / 函数 / 空间维度 / 证明） | 完成 |
| [[Level1_Vectors_and_Matrices\|Level1_Vectors_and_Matrices]] | Level 1 向量与矩阵（向量 / 线性组合 / 矩阵运算 / 线性变换 / NumPy） | 完成 |
| [[Level2_Linear_Systems\|Level2_Linear_Systems]] | Level 2 线性方程组（Ax=b / 消元 / 秩 / 解的结构 / NumPy） | 完成 |
| [[Level3_Vector_Spaces\|Level3_Vector_Spaces]] | Level 3 向量空间（向量空间 / 子空间 / 基与维数 / 线性变换 / 核与像 / 行列式） | 完成 |
| [[Level4_Orthogonality_and_Projection\|Level4_Orthogonality_and_Projection]] | Level 4 正交与投影（内积空间 / 投影 / 最小二乘 / Gram-Schmidt / QR） | 完成 |
| [[Level5_Eigenvalues_and_Eigenvectors\|Level5_Eigenvalues_and_Eigenvectors]] | Level 5 特征值与特征向量（特征分解 / 对角化 / 谱定理 / 正定矩阵） | 完成 |
| [[Level6_Matrix_Decomposition\|Level6_Matrix_Decomposition]] | Level 6 矩阵分解（LU / QR / SVD / 伪逆 / 条件数） | 完成 |
| [[Level7_Linear_Algebra_in_ML\|Level7_Linear_Algebra_in_ML]] | Level 7 机器学习中的线性代数（回归 / 正则化 / PCA / SVM / 协方差） | 完成 |
| [[Level8_Linear_Algebra_in_DL\|Level8_Linear_Algebra_in_DL]] | Level 8 深度学习中的线性代数（张量 / 反向传播 / 注意力 / 卷积 / LoRA） | 完成 |

## 写作规范

- 每篇笔记遵循 [config.md](../config.md) 的模板：直觉 → 定义 → 关键公式 → 常见误解 → AI 应用 → 遗留疑问；
- 用自己的话复述，禁止大段照抄教材；记录日期与来源；
- **遗留疑问**：在笔记内用 `> [!question] 未解决` 标记，由 MOC 汇总跟踪，不设独立目录；
- 原始笔记从 `Inbox/` 整理而来，整理流程见根目录 [[Knowledge_Workflow]]。
