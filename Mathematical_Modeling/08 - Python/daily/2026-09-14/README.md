---
title: Day14 代码说明 - 图论三件套（最短路/最小生成树/最大流）
type: code-readme
subject: mathematical-modeling
level: L5
tags:
  - graph
  - shortest-path
  - mst
  - max-flow
  - logistics
date: 2026-09-14
---

# Day 14 代码说明：图论三件套

- **脚本**：`graph_algorithms_logistics.py`
- **图**：`graph_algorithms_logistics.png`
- **教案**：`05 - Algorithms/Day14 - 图论三件套最短路最小生成树最大流.md`
- **运行**：`cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-14/graph_algorithms_logistics.py"`

## 这个脚本回答的 10 问

1. **解决什么问题**：同一张区域物流网的配送时效 / 建网成本 / 运力三类规划决策。
2. **何时使用**：赛题涉及选址、路径、管网、物流、网络连通、瓶颈容量等图结构问题。
3. **为什么适合**：三类问题分别天然对应 SSSP / MST / Max-Flow 三个经典模型，都有最优性或多项式算法保证。
4. **数学模型**：最短路 $\min\sum_P w(e)$；MST $\min\sum_T w(e)$ 且连通；最大流容量约束 + 流量守恒。
5. **如何求解**：Dijkstra（堆）、Kruskal（并查集）/Prim（堆）、Edmonds-Karp（BFS 增广）。
6. **Python 实现**：手写三算法 + 最小割提取；可选 networkx 交叉对照。
7. **如何验证**：Kruskal=Prim 总长一致；MST 边数=n−1；最小割容量=最大流（最大流最小割定理）；断言全过。
8. **缺点**：Dijkstra 不支持负权；MST 只保连通最省；Edmonds-Karp 规模大时慢（应换 Dinic）。
9. **替代方法**：Bellman-Ford/Floyd（负权/多源）、最小费用最大流、Dinic、Steiner 树。
10. **写进论文**：作为"现实网络→图模型"的建模范例，展示模型选择与瓶颈分析（最小割=投资优先级）。

## 关键实测结论（8 节点 / 13 边）

- **最短路**：仓库→网点G 10.95 km 为全网最远（时效瓶颈）；路径 仓库→B→D→G。
- **MST**：Kruskal = Prim = **21.56 km**（7 条边）；全 13 条边总长 47.50 km，**省 55% 线缆**。
- **最大流**：仓库→网点G **21 件/晚**；**最小割 = {网点D-G(12), 网点E-G(9)}，容量 21 = 最大流**（定理实测成立）。
- 决策洞察：扩运力不必全线加宽，扩这两条割边性价比最高。
- 本机未装 networkx，交叉对照自动跳过；手写三算法 100% 自洽。
