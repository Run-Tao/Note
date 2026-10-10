---
title: Mathematical Modeling MOC
type: moc
subject: mathematical-modeling
level: overview
tags:
  - mathematics
  - mathematical-modeling
---

# Mathematical Modeling MOC

> 数学建模知识库总览（索引页）。随学习逐步更新，不写成教材。

## Learning Roadmap

[[Level 0 - Modeling Thinking|Level 0 - 数学建模思维]]
↓
[[Level1_Problem_Analysis|Level 1 - 问题分析与数学表达]]
↓
[[Level2_Data_Analysis|Level 2 - 数据分析与可视化]]
↓
[[Level3_Classical_Models|Level 3 - 经典数学模型]]
↓
[[Level4_Statistical_Prediction|Level 4 - 统计与预测模型]]
↓
[[Level5_Evaluation_Decision|Level 5 - 评价与决策模型]]
↓
[[Level6_Optimization|Level 6 - 优化模型]]
↓
[[Level7_Graph_Network_Dynamic|Level 7 - 图论、网络与动态模型]]
↓
[[Level8_Numerical_Algorithms|Level 8 - 数值计算与算法]]
↓
[[Level9_Machine_Learning|Level 9 - 机器学习模型]]
↓
[[Level10_Model_Validation|Level 10 - 模型检验与模型改进]]
↓
[[Level11_Past_Problems|Level 11 - 历年真题训练]]
↓
[[Level12_Competition_Writing|Level 12 - 比赛实战与论文写作]]

> 详细目标见 [[Learning_Roadmap]]；未创建的文件为规划占位，不建空白笔记。

## 模型分类（按问题类型）

- **描述与统计**：描述性统计、相关性分析、聚类、PCA；
- **预测**：线性 / 非线性回归、时间序列、灰色预测、机器学习预测；
- **评价**：AHP、TOPSIS、熵权法、PCA 评价、灰色关联；
- **优化**：线性 / 整数 / 非线性 / 多目标 / 动态规划、遗传算法、模拟退火、蚁群、粒子群；
- **其他**：图论、网络流、排队、微分方程、蒙特卡洛、马尔可夫。

## 算法分类

- **解析 / 数值**：消元、插值与拟合、数值积分微分、迭代法；
- **优化算法**：精确（单纯形、分支定界）、启发式（GA / SA / ACO / PSO）；
- **机器学习**：回归、分类、聚类算法族；
- **模拟**：蒙特卡洛。

## 案例（09 - Case Studies）

- 真题结构：题目背景 → 问题重述 → 问题拆解 → 假设 → 符号 → 数据分析 → 建模 → 求解 → Python 实现 → 结果 → 检验 → 敏感性分析 → 评价 → 改进；
- 必含"我的思路 / 我的错误 / 与优秀方案的对比 / 可迁移性"。

## 比赛准备（10 - Competition）

- 赛前准备、读题方法、选题策略、三人分工、时间管理、建模 / 编程 / 论文流程、最终检查、常见错误；
- [[备赛记录|备赛记录（2026 高教社杯）]]：备赛活动日志（agent 测试、模拟、选题、分工等）。

## 项目（Projects）

- 数学概念 / 模型 → Python 实验 → 可视化 / 应用 → 项目记录。

## 教案索引（Agent 备课产出 · 04 / 05）

> 由"每日备课 Agent"持续产出（流水线与进度见 [[Teaching_Plan]]），共 **37 讲**，均经 `.venv` 实跑 + 断言核验 + 成文教案。
> ⚠️ **这些是"待消费"资料，不是"已学"进度**——学习进度以 Inbox 记录为准（见 [[Learning_Status]]）；当前产出:消费 ≈ 37:1。

### 优化与随机规划（`04 - Optimization`，18 讲）

- [[Day4 - 整数规划ILP分支定界与0-1项目选择|Day4 整数规划 ILP / 分支定界]]
- [[Day5 - 非线性规划NLP_均值方差投资组合|Day5 非线性规划 NLP / 均值-方差]]
- [[Day16 - 动态规划DP设备更新与01背包|Day16 动态规划 DP / 0-1 背包]]
- [[Day17 - 多目标优化加权和ε约束与Pareto前沿|Day17 多目标优化 / Pareto 前沿]]
- [[Day18 - 随机动态优化排队论MMc与MDP值迭代|Day18 排队论 M/M/c 与 MDP]]（↔ [[Level1_Limits_and_Continuity|高数：收敛]]）
- [[Day26 - 线性规划对偶理论影子价格与敏感性分析|Day26 LP 对偶 / 影子价格]]（↔ [[Level1_Vectors_and_Matrices|线代：转置 $A^\top$]]）
- [[Day27 - 报童模型Newsvendor随机需求下的最优订货|Day27 报童模型 Newsvendor]]
- [[Day28 - 两阶段随机规划与样本均值近似SAA|Day28 两阶段随机规划 / SAA]]
- [[Day29 - CVaR风险厌恶两阶段随机规划|Day29 CVaR 风险厌恶随机规划]]
- [[Day30 - 机会约束规划Chance-Constrained与安全等级代价|Day30 机会约束规划]]
- [[Day31 - 分布鲁棒优化DRO与Wasserstein模糊集|Day31 分布鲁棒优化 DRO]]
- [[Day33 - 无约束非线性优化梯度下降家族与拟牛顿法|Day33 梯度下降 / 拟牛顿法]]（↔ [[Level1_Limits_and_Continuity|高数：迭代收敛]]）
- [[Day34 - 约束非线性优化罚函数与内点法|Day34 KKT / 罚函数 / 内点法]]
- [[Day35 - 拉格朗日松弛与次梯度法|Day35 拉格朗日松弛 / 次梯度]]
- [[Day36 - 列生成法与Dantzig-Wolfe分解|Day36 列生成 / Dantzig-Wolfe]]
- [[Day37 - 割平面法Gomory割与整数规划的削顶点求解|Day37 割平面 / Gomory 割]]
- [[Day38 - Benders分解两阶段随机规划的主问题与子问题|Day38 Benders 分解]]（↔ [[Day35 - 拉格朗日松弛与次梯度法|切约束]]、[[Day36 - 列生成法与Dantzig-Wolfe分解|切列]]）
- [[Day40 - 最大权匹配KM算法与人员任务最优配对|Day40 最大权匹配 / KM 算法]]（↔ [[Day22 - 指派问题匈牙利算法与二分图匹配|匈牙利加权版]]、[[Day26 - 线性规划对偶理论影子价格与敏感性分析|对偶/影子价格]]）

### 算法 · 机器学习 · 评价（`05 - Algorithms`，19 讲）

- [[Day6 - 启发式元启发式GA与SA求解TSP|Day6 遗传算法 / 模拟退火（TSP）]]
- [[Day8 - 无监督聚类KMeans选址与定K|Day8 K-Means 聚类]]
- [[Day9 - 监督分类KNN信贷风险与定K|Day9 KNN 分类]]
- [[Day10 - 监督分类Logistic回归信贷概率与阈值权衡|Day10 Logistic 回归]]
- [[Day11 - 决策树与随机森林信贷规则与集成|Day11 决策树 / 随机森林]]
- [[Day12 - 梯度提升GBDT信贷风控与Boosting集成|Day12 梯度提升 GBDT]]
- [[Day13 - 支持向量机SVM信贷最大间隔与支持向量|Day13 支持向量机 SVM]]
- [[Day14 - 图论三件套最短路最小生成树最大流|Day14 图论三件套]]
- [[Day15 - 评价与决策AHP熵权法TOPSIS|Day15 AHP / 熵权法 / TOPSIS]]（↔ [[Level5_Eigenvalues_and_Eigenvectors|线代：特征向量 / Perron]]）
- [[Day19 - 组合赋权VIKOR与灰色关联综合评价|Day19 VIKOR / 灰色关联]]
- [[Day20 - 模糊综合评价CRITIC客观赋权与PCA综合评价|Day20 模糊综合评价 / CRITIC]]
- [[Day21 - TSP精确求解Held-Karp动态规划与分支定界|Day21 TSP 精确求解 / Held-Karp]]
- [[Day22 - 指派问题匈牙利算法与二分图匹配|Day22 匈牙利算法]]
- [[Day23 - 运输问题最小费用流与西北角Vogel法|Day23 运输问题 / 最小费用流]]
- [[Day24 - 稳定婚配Gale-Shapley与延迟接受|Day24 稳定婚配 Gale-Shapley]]
- [[Day25 - 二部图最大匹配与König定理|Day25 二部图匹配 / König 定理]]
- [[Day32 - 主成分分析PCA与主成分回归PCR|Day32 PCA / 主成分回归]]（↔ [[Level5_Eigenvalues_and_Eigenvectors|线代：特征值分解]]）
- [[Day39 - 一般图最大匹配Blossom带花树与奇环收缩|Day39 一般图最大匹配 / Blossom 带花树]]（↔ [[Day22 - 指派问题匈牙利算法与二分图匹配|二分图特例]]）
- [[Day41 - 中国邮递员问题最小权完美匹配与欧拉回路|Day41 中国邮递员问题 / 最小权完美匹配 + 欧拉回路]]（↔ [[Day14 - 图论三件套最短路最小生成树最大流|最短路]]、[[Day40 - 最大权匹配KM算法与人员任务最优配对|KM 匹配]]、[[Day21 - TSP精确求解Held-Karp动态规划与分支定界|TSP 对照]]）

> 配套可运行代码与图：`08 - Python/daily/<YYYY-MM-DD>/`（每讲含脚本 + 图 + README）。

## 未解决问题（整理汇总）

> 依 [[Knowledge_Workflow]] §4–5：疑问在相关笔记用 `> [!question]` 标记，并在此汇总。

| 问题 | 状态 | 说明 |
| --- | --- | --- |
| **产出:消费 ≈ 37:1** | ⚠️ 首要 | 37 讲 + 日课产出 vs 用户消费基本为零；**消费**是本库当前第一优先 |
| **Day7 编号缺口** | ✅ 已核定 | 计划草表 Day4+ 已被实际主线取代；09-07 的 GA/SA 已作为课程内 Day6 成稿，**非内容缺失**（详见 [[Teaching_Plan]]） |
| **时间序列 / ARIMA 无覆盖** | ⚠️ 待补 | 预测类赛题核心方法，全仓零命中 → 建议学预测专题时补写 |
| **Day36 脚本输出路径** | ✅ 已修复 | 硬编码仓库根相对路径 → 改为 `__file__` 锚定（2026-10-06 整理） |
| **跨库正文级 Wiki Link** | ⚠️ 长期为零 | 触点已在上表标注（线代 / 高数），待落到教案正文 |
| **投递故障 / 报告窗口错峰** | ⚠️ 运维项 | cron `delivery.mode=none`；报告与备课同刻 23:00 → 见 [[Automation_Plan]] |

## 与其他知识库的连接

- 数学基础：[[Level1_Vectors_and_Matrices|向量与矩阵]]、[[Level2_Linear_Systems|线性方程组]]；微积分 / 概率统计（规划占位）；
- AI：[[Linear Algebra MOC]]；
- 维护规范：[[config]]。
