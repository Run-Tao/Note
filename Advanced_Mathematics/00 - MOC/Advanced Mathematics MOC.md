---
title: Advanced Mathematics MOC
type: moc
subject: advanced-mathematics
level: overview
tags:
  - mathematics
  - calculus
---

# Advanced Mathematics MOC

> 高等数学知识库总览。目标：面向 AI / 计算机方向，长期维护、相互连接、可扩展的数学知识体系。

## Learning Roadmap

[[Level0_Functions_and_Mappings|Level 0 - Functions and Mappings]]
↓
[[Level1_Limits_and_Continuity|Level 1 - Limits and Continuity]]
↓
[[Level2_Derivatives_and_Differentials|Level 2 - Derivatives and Differentials]]
↓
[[Level3_Mean_Value_Theorems|Level 3 - Mean Value Theorems]]
↓
[[Level4_Integrals|Level 4 - Integrals]]
↓
[[Level5_Multivariable_Calculus|Level 5 - Multivariable Calculus]]
↓
[[Level6_Series_and_ODEs|Level 6 - Series and ODEs]]

> 路线已建立（2026-09）；**Level 0–6 主笔记已全部建成**（Level 2–6 于 2026-11-04 一次性补齐，统一按 config.md 六段模板）。Level 1 已补齐数列极限、函数极限、连续性与间断点分类（2026-11-01），并补入 ε-X 与无穷大 M-型定义（2026-11-02）。期中复习仍以 Level 1–3 为主，Level 4–6 是长期概念底座，不进本期中考前排期。

## 核心概念（随知识库发展补充）

- [[反函数|Inverse Function]] —— Level 0
- [[极限|Limit]] —— Level 1
- [[导数|Derivative]] —— Level 2
- [[梯度|Gradient]] —— Level 5
- [[微积分基本定理|Fundamental Theorem of Calculus]] —— Level 4
- [[Taylor 公式|Taylor's Theorem]] —— Level 3

## 题型专篇

- [[极限定义证明_题型专题]] —— 第 11 题 8 分，17/17 套真题必考；三种定义形式的证明模板、真题实例与陷阱
- [[左右极限求法与间断点判型]] —— 方法笔记：左右极限**怎么算**（八类必分左右的结构、$x=a\mp t$ 替换法、单侧下合法的工具）+ 17 套真题左右极限复核值全表
- [[函数极限计算与泰勒_专项练习]] —— 练习笔记：泰勒决策卡（默写级）+ A 组 12 题 / B 组 13 题 / 限时卷 10 题答案速查 + 11/3 时间表与验收线
- [[导数计算_题型专题]] —— 六型第一动作 D1–D6 + 默写卡 + 分段可导/隐函数二阶导/Leibniz/极坐标切线 + 错因回炉
- [[函数性态与作图_题型专题]] —— 七步判定口径 + 渐近线四类 + G1–G3 自测 + 四套真题 Q17 速查 + 十二条高频错误
- [[中值定理与不等式证明_题型专题]] —— 六个辅助函数默写卡 + Rolle 答题模板 + 不等式三条路 + Q16 范例 + 25 页专题讲义页码级索引
- [[数列极限与新压轴_题型专题]] —— 夹逼/单调有界/定积分定义/Stolz 四法 + 2024-25 起抽象函数方程与介值定理压轴
- [[D-6_间断点四步卡与求导检查表]]、[[限时卷计算题补强_三型]]、[[D-5_作图七步与导数保温]] —— 三张执行/补强卡（默写级 + 题号索引 + 验收清单）
- **讲义（PDF）**：`Resources/Downloads/讲义_函数极限与间断点.pdf` —— 极限计算三步决策 + 间断点判型 + 16 道真题精讲 + 12 条高频错误 + **60 分钟限时卷（含详解）** + 17 套全表；配套 `Resources/Downloads/讲义_极限定义证明.pdf`（第 11 题）、`Resources/Downloads/讲义_练习_函数极限计算与泰勒.pdf`（D-7 练习册：泰勒决策卡 + 35 题 + 解析）
- **新增专题讲义（2026-11-04）**：`讲义_导数计算专题.pdf`（5 页，12 道例题 + 20 分钟限时卷 T1–T8）、`讲义_函数性态与作图.pdf`（4 页，6 道例题 + 30 分钟限时卷 L1–L2）、`讲义_数列极限与压轴新题型.pdf`（4 页，10 道例题 + 15 分钟限时卷 V1–V5）；均含详解、十二条高频错误与考前速查卡，数值结论经 `verify_new_materials.py` 93 项复核。

## 未解决问题

- [[Level0_Functions_and_Mappings]]：一般方程 $F(x,y)=0$ 何时能局部确定隐函数、如何求导？（待 Level 2 / Level 5）
- [[Level1_Limits_and_Continuity|极限与连续]]：ε-N 定义下证明数列极限的例题待补。
- [[Level1_Limits_and_Continuity|极限与连续]]：子列与 Weierstrass 致密性定理（有界数列必有收敛子列）待补证明。
- [[Level1_Limits_and_Continuity|极限与连续]]：函数版本的双侧极限证明（左右极限 → 双侧极限的 $\varepsilon$-$\delta$ 论证）待补。
- [[Level1_Limits_and_Continuity|极限与连续]]：复合函数 $g\circ f$ 的间断点如何从 $f$ 的间断点传递，待补。
- [[Level1_Limits_and_Continuity|极限与连续]]：无穷间断点与竖直渐近线的等价关系待补证明。
- [[极限定义证明_题型专题]]：13 套真题的官方解答路径未逐份核对（本文范例为自证版本）；2024-25 官方 $\delta$ 偏大是否影响评分，未查评分细则。

## 入口与规范

- [[README]] —— 知识库介绍
- [[Learning_Roadmap]] —— 学习路线（Level 0–6）
- [[config]] —— 知识库规范（长期目标 + 写作规范）
- 子目录：Notes / Exercises / Resources；术语表见 [[Glossary]]；学习日志见个人层 [[Learning Log/README|Learning Log]]

## 相关知识库

- [[Linear Algebra MOC|线性代数]] —— 向量 / 矩阵 / 线性变换（与本库互相引用）
- [[Mathematical Modeling MOC|数学建模]] —— 建模、优化与 Python 实践
## 练习与作业

- [[作业1_实数与函数]]、[[作业2_反函数与曲线]] —— Level 0
- [[作业3_数列极限]]、[[作业4_极限计算与夹逼定理]] —— Level 1
- [[作业5_单调有界准则与函数极限定义]]、[[作业6_函数极限与左右极限]]、[[作业7_极限综合与无穷小阶数]]、[[作业8_连续性与间断点分类]] —— Level 1
- [[作业9_闭区间连续函数性质与导数定义]] —— Level 1/2
- [[作业10_求导法则与微分]]、[[作业11_隐函数与参数方程求导]]、[[作业12_二阶导数与高阶导数]]、[[作业13_高阶导数与罗尔定理应用]] —— Level 2
- [[作业14_拉格朗日中值定理与不等式]]、[[作业15_泰勒展开与连续性讨论]]、[[作业16_泰勒公式求极限与性质证明]] —— Level 3

## 复习资料与真题

- [[Resources/README|Resources 索引]] —— 讲义、模拟卷、真题位置
- [[真题考点索引_2008-2025]] —— 17 套真题（2008–2025）的骨架、考点频次与刷题靶子
- [[高数I期中冲刺计划_2026-11]] —— 9 天冲刺计划（个人层）
- 模拟卷 C（`Resources/Downloads/模拟卷C.pdf` + `模拟卷C_答案.pdf`）—— 18 题全原创全真卷，与 A（真题重组）、B（原创）不重复；第 18 题专测压轴新题型，答案册含给分点与回炉路径