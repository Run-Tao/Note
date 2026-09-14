---
title: Day13 代码说明 - SVM 信贷最大间隔
type: code-readme
subject: mathematical-modeling
level: L8
tags:
  - classification
  - svm
  - hinge-loss
  - credit-scoring
date: 2026-09-13
---

# Day 13 代码说明：支持向量机 SVM / 最大间隔

- **脚本**：`svm_credit_margin.py`
- **图**：`svm_credit_margin.png`
- **教案**：`05 - Algorithms/Day13 - 支持向量机SVM信贷最大间隔与支持向量.md`
- **运行**：`cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-13/svm_credit_margin.py"`

## 这个脚本回答的 10 问

1. **解决什么问题**：样本少 / 高维 / 怕离群点带偏边界时，如何得到一条"留有余量"的分类边界。
2. **何时使用**：线性可分或近似可分、维数高样本少、关注安全边际而非概率。
3. **为什么适合**：最大化间隔等价于只让"贴着边界的少数支持向量"说话，带外的点对权重零贡献，天然稳健。
4. **数学模型**：$\min\ 0.5\lambda\|w\|^2 + \frac1n\sum_i\max(0,1-y_i(w^\top x_i+b))$（hinge loss + L2 正则 = 软间隔最大间隔）。
5. **如何求解**：对 hinge loss 做带动量的次梯度下降（凸问题，可稳定收敛）。
6. **Python 实现**：手写 `svm_hinge_fit` → `decision_function` / `predict`，并用 sklearn `LinearSVC`、`LogisticRegression` 交叉对照。
7. **如何验证**：手写 $w$ 与 LinearSVC $w$ 余弦相似度 0.9999；只用支持向量重训方向不变；测试准确率 0.7778 明显高于随机猜。
8. **缺点**：对特征尺度敏感（必须标准化）、对正则强度敏感（太强→全员支持向量）、不直接给概率。
9. **替代方法**：Logistic 回归（要概率）、KNN、树模型；核 SVM 可处理非线性。
10. **写进论文**：作为"最大间隔 vs 最大似然"两种分类哲学对照，佐证模型选择的合理性。

## 关键实测结论

- 最优 lam=0.5，测试准确率 **0.7778**；lam 过大 (>5) 正则过强 → 边界坍塌欠拟合。
- 手写 SVM ‖w‖=0.68 比 Logistic ‖w‖=2.17 更小 = **间隔更宽**。
- 与 sklearn LinearSVC 方向余弦 **0.9999**（实现正确性互证）。
- 决策值：好客户均值 −0.471 vs 坏客户 +0.491（给"分得多开"，非概率）。
- 支持向量子集重训方向不变 → 实测坐实"支持向量决定边界"。
