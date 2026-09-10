---
title: Day10 代码 — Logistic 回归信贷评分卡（手写梯度下降 + sklearn 对照 + 阈值权衡）
type: code
subject: mathematical-modeling
level: L8
roadmap: Level 8 - 数据分析与机器学习
tags:
  - logistic-regression
  - classification
  - gradient-descent
  - credit-scoring
date: 2026-09-10
day: 10
---

# Day 10 代码 — Logistic 回归信贷评分卡

## 文件

- `logistic_credit_scoring.py` — 主脚本
- `logistic_credit_scoring.png` — 输出图（概率热力图 + sigmoid + 损失收敛）

## 运行

```bash
cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-10/logistic_credit_scoring.py"
```

## 脚本头部标注

- **subject**: mathematical-modeling
- **level**: L8（数据分析与机器学习 / 监督分类）
- **anchor**: Day10 — 接 Day9 KNN，监督分类的"参数化"对照：Logistic 回归
- **goal**: 把线性打分经 sigmoid 压成违约概率，手写梯度下降训练并与 sklearn 对齐；用业务阈值平衡误拒/漏放两类错误。

## 做了什么（10 问成文的精简版）

1. **决策问题**：600 名历史客户(收入×负债率, 已知好坏)，新申请给"违约概率"评分卡 → 贷/拒。
2. **模型**：$p=\sigma(w^\top x+b)$，log-odds 线性；极大似然 ⇒ 交叉熵损失(凸)。
3. **求解**：手写 sigmoid + 批量梯度下降；与 `sklearn.LogisticRegression` 交叉验证。
4. **工程**：训练/测试切分 + 用训练集统计量标准化(防泄漏)；阈值扫描做敏感性分析。
5. **结果**：测试准确率 0.8222，手写系数与 sklearn **逐位一致**；坏客户平均概率 0.741 vs 好客户 0.309。
6. **检验**：系数符号合理、损失下降、训练测试一致、断言全过。

## 关键输出

```
w_收入 = -1.966  w_负债率 = +1.234  b = -0.163
交叉熵损失 0.6931 → 0.4209
测试准确率 手写 0.8222  vs  sklearn 0.8222
混淆矩阵: 好判对70 误拒19 漏放13 坏判对78
阈值 0.5 时准确率最高(0.8222)
```

## 与 Day9 对照

KNN(记忆型/硬标签/不规则边界) vs Logistic(参数化/概率/直线边界/可解释系数)。
