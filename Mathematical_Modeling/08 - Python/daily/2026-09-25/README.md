---
title: Day25 - 二部图最大匹配与 König 定理（代码）
type: code-readme
subject: mathematical-modeling
level: L6 优化模型 / L7 图论、网络与动态模型
tags:
  - mathematics
  - mathematical-modeling
  - graph-theory
  - matching
  - network-flow
created: 2026-09-25
---

# Day25 · 二部图最大匹配与 König 定理

> 教案正文见 [[Day25 - 二部图最大匹配与König定理]]（`05 - Algorithms/`）。
> 代码：`max_matching_konig_theorem.py`；图：`max_matching_konig_theorem.png`。

## 一句话

**二分图里有三个看似无关的量其实是同一个数**：$|最大匹配| = |最小点覆盖| = |V| - |最大独立集|$。手写**增广路（Kuhn/匈牙利无权版）**求最大匹配，再用**交错可达**构造性验证 König 定理，并与**最大流（Dinic）**、**暴力穷举**三方交叉核对。

## 真实决策背景

5 个岗位 A1–A5、4 位候选人 B1–B4，"能胜任"关系有限（A2、A3 都只能干 B1 的活）。问：**最多能安排几个岗位有人干？卡在哪里？** 与 Day22 指派对照：那里默认人人有活、求总成本最小；这里先回答**"到底配不配得满、最多几个"**。

## 关键结果（仓库 `.venv` 实跑）

**最大匹配 = 4**（Kuhn 增广路 = Dinic 最大流 = 暴力穷举，三方一致）：

| 匹配边 | 岗位 | 候选人 |
|---|---|---|
| 1 | A1 | B2 |
| 2 | A2 | B1 |
| 3 | A4 | B3 |
| 4 | A5 | B4 |

- **瓶颈**：未匹配岗位 = **A3**。因 A2、A3 的邻居都只有 B1（Hall 条件 {A2,A3}→{B1}，1<2），注定空一岗。
- **König 定理实测**：交错可达搜索得 不可达左点 = {A1,A4,A5}、可达右点 = {B1} → **最小点覆盖 = {A1,A4,A5,B1}，大小 4 = 最大匹配**；穷举出的最小覆盖大小同为 4。
- **最大独立集** = {A2,A3,B2,B3,B4}，**大小 5 = |V|(9) − |最小点覆盖|(4)**，内部无相邻边。
- **多图定理验证**：随机生成 **300 组**二部图，逐组验证 |最小点覆盖| = |最大匹配| 且覆盖合法，**300/300 = 100%**。
- **敏感性**：移除 A1/A4/A5 → 匹配数降到 **3**；移除 A2/A3 → 仍为 **4**（点出关键岗位 A1、A4、A5）。

## 代码要点（手写、零专用匹配库）

- `kuhn_max_matching()`：DFS 递归增广路，翻转匹配状态，匹配数 +1（Berge 定理）。
- `Dinic`：源→左部→右部→汇、边容量全 1，最大流值 = 最大匹配数。
- `konig_min_vertex_cover()`：从未匹配左点出发交错可达，取（不可达左点 ∪ 可达右点）构造最小点覆盖。
- `brute_force_max_matching()` / `brute_force_min_vertex_cover()`：小图穷举真值。
- **断言**：三法最大匹配一致、覆盖大小 = 匹配数、覆盖合法、独立集无内边、$|独立集| = |V|−|覆盖|$，全部通过。

## 复现

```bash
cd /home/ubuntu/Note
./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-25/max_matching_konig_theorem.py"
```

## 图说明

`max_matching_konig_theorem.png`（2×2）：
- 左上：二部图全貌（岗位在左、候选人在右、灰线为"能胜任"）；
- 右上：最大匹配（蓝粗线）共 4 对；
- 左下：最小点覆盖（红点）大小 4 = 最大匹配；
- 右下：最大独立集（橙圈）大小 5 = |V| − |最小点覆盖|。

已注册系统 Noto Sans CJK SC 字体，避免中文乱码。
