---
title: Mathematical Modeling README
type: index
subject: mathematical-modeling
level: overview
tags:
  - mathematics
  - mathematical-modeling
---

# Mathematical Modeling 数学建模知识库

> 面向大学生数学建模竞赛（国赛 / 美赛等）的个人知识库。
> 目标不是收集"数学模型大全"，而是训练**完整建模能力**：从读懂现实问题开始，到写出可复现的模型论文为止。

## 核心能力主线

现实问题 → 理解问题 → 问题重述 → 问题拆解 → 合理假设 → 变量与参数定义 → 建立数学模型 → 选择求解方法 → Python 实现 → 结果分析 → 模型检验 → 敏感性 / 稳健性分析 → 模型改进 → 论文表达

## 目录结构

```
Mathematical_Modeling/
├── README.md                    本文件（介绍与总览）
├── Learning_Roadmap.md          学习路线（Level 0–12）
├── 00 - MOC/                    总览页 Mathematical Modeling MOC
├── 01 - Modeling Fundamentals/  建模基础：思维、问题分析、数学表达
├── 02 - Data Analysis/          数据分析与可视化
├── 03 - Mathematical Models/    经典数学模型（方程、差分等）
├── 04 - Optimization/           优化模型
├── 05 - Algorithms/             算法（图论、启发式、蒙特卡洛等）
├── 06 - Numerical Methods/      数值计算
├── 07 - Machine Learning/       机器学习模型
├── 08 - Python/                 建模相关 Python（NumPy/Pandas/Matplotlib/SciPy/scikit-learn）
├── 09 - Case Studies/           真题与案例分析
├── 10 - Competition/            比赛实战（选题、分工、时间管理、检查清单）
├── 11 - Paper Writing/          论文写作体系
├── Inbox/                       原始笔记（每天一个 YYYY-MM-DD.md，无格式要求；整理后归档 Processed/）
├── Exercises/                   练习与错题（记录错误原因）
├── Projects/                    综合项目（模型 → 实现 → 验证 → 论文）
└── Resources/                   资料索引
```

## 七大重点能力

1. **问题分析**：问题重述、问题拆解、识别变量 / 目标 / 约束、判断问题类型；
2. **建模**：基本假设、符号定义、变量设计、目标函数、约束条件、模型建立；
3. **模型选择**：回答"面对这个问题，为什么选择这个模型"，而不是背公式；
4. **求解**：解析方法、数值方法、优化算法、Python 计算；
5. **模型验证**：误差分析、模型检验、敏感性分析、稳健性分析、对比实验、极端情况分析；
6. **模型改进**：认识模型局限、修改假设、加入新变量、提升预测或优化效果；
7. **论文表达**：把"我的计算过程"转化为"别人能理解、验证和复现的数学模型"。

## 学习原则

整理任何建模知识时，优先回答这 10 个问题：

1. 这个方法解决什么问题？
2. 什么情况下应该使用？
3. 为什么适合这个问题？
4. 数学模型是什么？
5. 如何求解？
6. 如何用 Python 实现？
7. 如何验证？
8. 有什么缺点？
9. 有什么替代方法？
10. 如何写进数学建模论文？

避免"公式 + 定义 + 一个例题"的机械式笔记。

## 与其他知识库的连接

- **数学基础 ↔ 建模**：[[Level1_Vectors_and_Matrices|向量与矩阵]]、[[Level2_Linear_Systems|线性方程组]]（$A\mathbf{x}=\mathbf{b}$ 是回归与优化的公共骨架）；微积分、概率统计知识库规划中（链接占位，不建空文件）；
- **建模 ↔ Python**：建模所需库集中在 `08 - Python/`，Python 学习服务于建模；
- **建模 ↔ AI**：`07 - Machine Learning/`；线性代数侧的 AI 应用见 [[Linear Algebra MOC]]；
- **维护规范**：遵循 [[config]] 中的 Obsidian 规范（frontmatter、Wiki Links、Callout、MOC、增量维护）。

## 维护原则

- 不批量生成笔记；按学习进度逐步创建；
- 模型按**问题类型**组织，不追求模型数量；
- 案例必须包含"我的思路 / 我的错误 / 与优秀方案的对比 / 可迁移性"，不保存标准答案了事；
- 论文格式以当前官方竞赛规范为准，不使用过时格式。
