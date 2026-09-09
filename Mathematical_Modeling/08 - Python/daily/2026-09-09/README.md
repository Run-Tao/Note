---
title: 监督分类 KNN：信贷风险预筛 + 手写=sklearn + 定 K
type: lesson
subject: mathematical-modeling
level: L8
roadmap: Level 8 - 数据分析与机器学习
tags:
  - classification
  - knn
  - supervised-learning
  - sklearn
  - standardization
date: 2026-09-09
day: 9
---

# Day 9 — 信贷风控：新贷款申请该贷还是该拒？（KNN 近邻分类）

配套教案见 [[05 - Algorithms/Day9 - 监督分类KNN信贷风险与定K]]；图 `knn_decision_boundary_k.png`。

```sh
cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-09/knn_credit_risk.py"
```

## 跑通要点输出

```
数据: 600 名历史客户(好/坏各300), 特征=月收入 x 负债率; 420 训练/180 留出测试
[手写 vs sklearn] k=7 两边准确率 0.894、指派一致率 100%
[定K]  峰值 K=11 (测试准确率 0.9278)  K=1:0.894(过拟合锯齿) K≥15 回落
[混淆矩阵@K=11] 好90/坏77判对; 拒错好客户8(8.2%)、漏放坏账5(6.1%)
[违约倾向分] 真实违约者平均0.920 vs 守约者0.136 —— 可设阈值差异化授信
```

- **手写 = sklearn**：逐 K 准确率一致、指派 100% 吻合 → 实现正确；
- **K 甜点在中间**：K=1 太贴单点噪声(过拟合)、K 太大被远处异类稀释(欠拟合)，K=11 最佳；
- **标准化成关键**：收入(几十量纲)会淹没负债率(0~1)，必须 z-score 后算距离；
- **误差读代价**：拒错(FP)失营收 vs 放坏账(FN)失本金，可调判定线换平衡。

## 文件

- `knn_credit_risk.py` — 可运行脚本（手写 KNN + sklearn 对照 + 定 K + 混淆矩阵 + 决策边界图）
- `knn_decision_boundary_k.png` — 决策边界 + 测试点 / 准确率随 K
