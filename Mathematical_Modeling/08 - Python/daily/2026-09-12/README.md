---
title: Day12 · GBDT 信贷风控（Boosting 对 Bagging）
type: code-daily
subject: mathematical-modeling
level: L8
roadmap: Level 8 - 数据分析与机器学习
tags:
  - gradient-boosting
  - gbdt
  - ensemble
  - boosting
  - credit-scoring
date: 2026-09-12
day: 12
---

# Day12 — 梯度提升 GBDT：弱树串行接力，逐层削偏差

对应教案：[[Day12 - 梯度提升GBDT信贷风控与Boosting集成]]（`05 - Algorithms/`）

## 今天做什么

同一个"信贷风控该贷还是该拒"的决策，接 Day11（随机森林=Bagging）演 **Boosting 的另一半**：

- 从**对数几率初值** $F_0=\log\frac{p}{1-p}$ 出发；
- 每轮算**残差** $r_i=y_i-\text{sigmoid}(F_i)$（= LogLoss 的负梯度）；
- 训练一棵**浅回归树**（深度 2 的"树桩"）去拟合残差；
- 乘**学习率**后累加进总分 $F$，最后 sigmoid 得违约概率。

## 文件

| 文件 | 说明 |
|---|---|
| `gbdt_credit_boosting.py` | 手写回归树桩 + GBDT 主循环 + 轮数/学习率扫描 + sklearn 对照 + 三图 |

## 运行

```bash
cd /home/ubuntu/Note && ./.venv/bin/python \
  "Mathematical_Modeling/08 - Python/daily/2026-09-12/gbdt_credit_boosting.py"
```

## 实测结果（仓库 .venv，全部断言通过）

- 训练 336 / 验证 84 / 测试 180；
- 主模型（120 树 × lr=0.1 × 深度 2）：训练 **0.8006**、测试 **0.7778**，训练 logloss 0.6741→0.4389；
- 混淆矩阵：好判对 65 / 拒错好客户 29 / **漏放坏账 11** / 坏判对 75；违约者平均概率 0.715 vs 守约者 0.352；
- 特征重要性：**负债率 0.728 ≫ 收入 0.272**（与 Day11 同向）；
- 轮数效应：1 棵树测试仅 0.478（步长不够）→ 20 棵 0.778（偏差快速下降）→ 120 棵训练升到 0.801（过拟合征兆，验证见顶即停）；
- 学习率效应：lr 0.05→1.0 测试 0.778→0.789 但训练 0.789→0.851（lr 大更快过拟合）；
- **交叉对照**：sklearn GBDT 测试 **0.7778**，与手写完全一致。

## 图

- 左：训练/验证 logloss 随树数（+ 验证最优竖线）；
- 中：GBDT 违约概率场 + 0.5 判定黑线 + 测试点；
- 右：特征重要性。

字体已注册系统 Noto Sans CJK SC，中文无乱码。

## 一句话要点

**Boosting = 在函数空间做梯度下降**，每棵浅树拟合残差、小步累加，削**偏差**；与 Bagging（并行平均削**方差**）互补，调参直觉相反——森林树多无妨，GBDT 树多要早停，**学习率 × 树数是一对跷跷板**。
