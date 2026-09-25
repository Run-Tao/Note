---
title: Day27 - 报童模型 Newsvendor 随机需求下的单周期最优订货（代码）
type: code-readme
subject: mathematical-modeling
level: L6 优化模型 / 随机优化（报童模型）
tags:
  - mathematics
  - mathematical-modeling
  - optimization
  - stochastic-optimization
  - newsvendor
created: 2026-09-25
---

# Day27 · 报童模型（Newsvendor）随机需求下的单周期最优订货

> 教案正文见 [[Day27 - 报童模型Newsvendor随机需求下的最优订货]]（`04 - Optimization/`）。
> 代码：`newsvendor_stochastic.py`；图：`newsvendor_stochastic.png`。

## 一句话

**报童模型问的不是"需求会是多少"，而是"缺货和滞销哪个更肉疼"**——最优订货量是需求分布的一个分位数 $\frac{C_u}{C_u+C_o}$，缺货越贵订得越多。

## 真实决策背景

一家社区面包店每天清晨决定烤多少只"当日限定可颂"：成本 3 元、售价 8 元、隔夜清仓残值 1.5 元；当日需求 $\sim N(200, 40^2)$。问：每天烤多少只，期望利润最大？

- 缺货成本 $C_u = p-c = 5$ 元；过剩成本 $C_o = c-s = 1.5$ 元；
- 临界分位点 $\frac{C_u}{C_u+C_o} = 0.7692$。

## 关键结果（仓库 `.venv` 实跑）

| 方法 | $Q^*$ | 期望利润（元/天） |
|---|---|---|
| 解析公式 | 229.45 | 920.90 |
| 网格穷举 | 229.50 | 920.50 |
| 边际分析 | 229.45 | — |
| 蒙特卡洛 30 万天 | 按 229.45 经营 | 920.93 |

- **四法一致**（断言全过）；
- **服务水平** $P(D\le Q^*) = 0.7692$ = 临界分位点；
- **朴素对照**：只按均值 $Q=\mu=200$ 订 → 896.35 元/天，**每天少赚 24.58 元（2.74%）**；
- **敏感性 A**（均值估偏）：低估 20 只亏 11.0、高估 20 只亏 8.6 —— **低估需求比高估更贵**；
- **敏感性 B**（波动估偏）：$\sigma$ 40→20 亏 5.8、40→80 亏 17.4 元/天。

## 代码要点（手写、零专用库存库）

- `analytical_optimal()`：$Q^*=\mu+\sigma z$，期望利润用标准正态**损失函数** $L(z)=\phi(z)-z(1-\Phi(z))$ 闭式算；
- `grid_search()`：$Q$ 步长 0.5 网格 + 40 万需求样本估期望利润，取最大；
- `marginal_analysis()`：二分求"多订一只期望收益=0"临界点；
- `simulate_policy()`：30 万天蒙特卡洛（平均利润/波动/缺货率）；
- `sensitivity_to_forecast_error()` / `sensitivity_to_sigma()`：**参数估偏但未察觉**时的真实利润惩罚；
- **断言**：解析≈穷举、边际=解析、仿真≈解析、报童优于按均值订、服务水平=临界分位点，全部通过。

## 复现

```bash
cd /home/ubuntu/Note
./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-25/newsvendor_stochastic.py"
```

## 图说明

`newsvendor_stochastic.png`（2×2）：
- 左上：期望利润随 $Q$ 的曲线 + 解析/穷举/均值三线定位；
- 右上：需求分布 + 临界分位点（蓝=不缺货、红=缺货）；
- 左下：报童最优 vs 按均值订的利润分布对照（整体右移）；
- 右下：均值估偏的利润损失（绿=正确、红=偏），**低估比高估更贵**。

已注册系统 Noto Sans CJK 字体，避免中文乱码。
