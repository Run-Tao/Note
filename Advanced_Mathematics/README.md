---
title: 高等数学 README
type: index
subject: advanced-mathematics
level: overview
tags:
  - mathematics
  - calculus
---

# 高等数学知识库

> 面向人工智能 / 计算机方向本科生的个人高等数学知识库（函数与映射 → 极限 → 微分 → 积分 → 多元微积分）。
>
> 目标不是应付考试，而是掌握描述"变化与累积"的数学语言，并把它真正用到机器学习、优化与数值计算中。

## 这个知识库要解决什么问题

微积分是"变化的数学"：极限描述逼近，导数描述变化率，积分描述累积。深度学习里的梯度下降、损失函数、反向传播，本质上都建立在这套语言之上。

这个知识库坚持三个原则：

1. **直觉优先** —— 先理解"为什么需要它、几何意义是什么"，再补严格定义与计算；
2. **建立知识体系** —— 从函数与映射出发，把极限、导数、积分串成一条线，与线性代数互相引用；
3. **学以致用** —— 每个核心概念都要能说出它在 AI / 优化 / 数值计算中的真实用途。

## 目录结构

```
Advanced_Mathematics/
├── 00 - MOC/                 知识库总览（Advanced Mathematics MOC）
├── README.md                 知识库介绍（本文件）
├── config.md                 知识库规范：长期目标与写作规范（维护时默认遵守）
├── Learning_Roadmap.md       学习路线（Level 0 – Level 6）
├── Inbox/                    原始笔记（每天一个 YYYY-MM-DD.md，无格式要求；整理后归档 Processed/）
├── Notes/                    学习笔记（按 Level 与主题组织）
├── Exercises/                练习题与错题本
├── Resources/                教材、课程、视频等资料索引
└── Glossary.md               术语表（单文件，中英文对照）
```

## 各目录用途

| 目录 | 用途 |
| --- | --- |
| Inbox/ | 原始笔记：随手记的碎片化想法，每天一个 YYYY-MM-DD.md，无格式要求，永不删除；Agent 提取内容整理后，整个文件移入 Inbox/Processed/ 归档 |
| Notes/ | 按 Level 和主题记录笔记，一主题一文件；未解决疑问用 `> [!question]` 标记 |
| Exercises/ | 概念题、计算题、证明题与错题记录 |
| Resources/ | 汇总教材、课程、视频与文档，标注难度与推荐理由 |
| Glossary.md | 术语表单文件：重要数学术语中英文对照，并链接相关笔记 |

## 使用建议

0. **维护知识库时默认遵守 [config.md](config.md)**：新建任何笔记或练习前，先读一遍规范；
1. 先打开总览页 [[Advanced Mathematics MOC]] 或 [Learning_Roadmap.md](Learning_Roadmap.md)，确定当前 Level；
2. 学完一个主题后按循环走一遍：**写笔记 → 做练习 → 找应用案例 → 解决遗留疑问**；
3. 与 [[Linear Algebra MOC|线性代数]] 互相引用：线代提供线性结构与矩阵语言，高数提供变化率与累积；
4. 每次学习结束时在个人层学习日志（[[Learning Log/README|00 - Personal/Learning Log]]）记录进度与遗留问题。

## 学习路线概览

| Level | 主题 | 一句话目标 |
| --- | --- | --- |
| Level 0 | 函数与映射基础 | 说清函数、映射、反函数与基本性质 |
| Level 1 | 极限与连续 | 理解"无限逼近"的严格语言 |
| Level 2 | 导数与微分 | 用导数描述变化率与局部线性化 |
| Level 3 | 微分中值定理与导数应用 | 用中值定理分析函数形态与最值 |
| Level 4 | 不定积分与定积分 | 理解累积、面积与"微积分基本定理" |
| Level 5 | 多元函数微分学 | 掌握偏导、梯度与多元极值 |
| Level 6 | 级数与微分方程（选学） | 用无穷过程与方程描述系统演化 |

详细路线见 [Learning_Roadmap.md](Learning_Roadmap.md)。

## Related

- [[Linear Algebra MOC|线性代数知识库]] —— 矩阵与线性变换
- [[Mathematical Modeling MOC|数学建模知识库]] —— 建模、优化与 Python 实践
