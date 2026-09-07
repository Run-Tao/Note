---
title: 启发式与元启发式：遗传算法(GA) + 模拟退火(SA) 求解 TSP 城市巡回
type: lesson
subject: mathematical-modeling
level: L6
roadmap: Level 6 - 优化模型 / 算法
tags:
  - algorithms
  - metaheuristic
  - genetic-algorithm
  - simulated-annealing
  - tsp
  - combinatorial-optimization
date: 2026-09-07
day: 7
---

# Day 7 — 当"数目太多解不完"：元启发式(GA/SA) 硬啃 TSP

> 锚点：[[Learning_Roadmap|Learning Roadmap]] Level 6 · 优化模型 / 元启发式算法。
> 前几课(LP→ILP→NLP)都有精确求解器兜底；今天专治那些"**精确求解器直接炸掉**"的决策——TSP 旅行商。25 个城市要巡回送货/巡检，问哪条顺序最短？$25!\approx1.55\times10^{25}$ 种排列，穷举/分支定界都物理性跑不完。于是放弃"必最优"，改用**遗传算法 GA** 与**模拟退火 SA** 两类元启发式找"足够好"的路线。仓库 `.venv` 实际跑通。

## 代码与图

- 脚本：`heuristic_ga_sa_tsp.py`（GA：编码→适应度→锦标赛选择→顺序交叉OX→变异→精英保留；SA：2-opt 邻域 O(1) 增量 + Metropolis 接受 + 指数降温；对照最近邻贪心 NN）
- 图：`routes_ga_sa.png`（NN/GA/SA 三条路线并排对比）、`convergence_ga_sa.png`（best-so-far 收敛曲线 + NN 基准虚线）
- 配套教案：[[Day6 - 启发式元启发式GA与SA求解TSP|05 - Algorithms/Day6 教案]]（标题沿用课程内 Day6 连续编号，本轮产出日期 2026-09-07）

## 结果（实际跑通）

```
穷举路线数 N! = 1.55e+25  —— 精确搜索不可行
最近邻贪心 NN :  473.0   (构造启发式, 秒出, 作基准)
遗传算法  GA :  403.6   优于NN 14.7%
模拟退火  SA :  414.0   优于NN 12.5%
[断言全部通过] GA/SA 均显著优于最近邻贪心(跳出局部贪心陷阱)
```

## 一句话要点

TSP 是"排列型"NP-hard 组合优化，精确求解几十城就炸；**GA/SA 的核心是"允许有纪律地变坏"**——GA 靠群体+顺序交叉并行重组路线块，SA 靠 Metropolis"以概率 $e^{-\Delta/T}$ 接受更差解"再随温度指数降温收网——结果都显著优于"每步只吃最近"的短视贪心(约优 13–15%)，正是"先大步跳坑、再小步精修"的元启发式指纹。
