#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : 数学建模 / 机器学习 · 集成学习
level   : L8 - 数据分析与机器学习
anchor  : [[Learning_Roadmap|Level 8 - 数据分析与机器学习]] / 监督分类 · 梯度提升
goal    : 用"信贷风控该贷还是该拒"的真实决策，讲清 GBDT(梯度提升决策树)
          如何让一堆"弱学习器(浅树)"串行接力——每棵新树专门去拟合
          "前面所有树还没解释掉的残差(负梯度)"，用学习率收缩步长；
          与 Day11 的 Bagging(随机森林，并行平均降方差)形成
          "Bagging 降方差 vs Boosting 降偏差"的完整对照。
          本课只依赖 numpy 手写，不引入 sklearn 之外的新依赖。
run     : cd /home/ubuntu/Note && ./.venv/bin/python \
          "Mathematical_Modeling/08 - Python/daily/2026-09-12/gbdt_credit_boosting.py"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体（避免乱码） ----------
for cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "SimHei"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)


# =====================================================================
# 0. 造一份"历史客户"数据：收入(万/年) × 负债率(%) → 好客户(0)/违约(1)
#    与 Day11 同款数据生成逻辑，保证两条主线可对照。
# =====================================================================
def make_credit_data(n=600, seed=42):
    r = np.random.default_rng(seed)
    # 收入：对数正态更贴近真实（多数中等、少数高收入）
    income = np.exp(r.normal(1.6, 0.45, n))          # 单位：万/年
    # 负债率：随收入略降，带噪声，截断到 [5, 90] %
    debt = np.clip(r.normal(55, 18, n) - (income - 5) * 3.0, 5, 90)
    # 真实"违约势能"：负债率高 + 收入低 → 危险；含交互项
    z = 0.085 * debt - 0.95 * np.log(income + 1.0) - 2.6
    p = 1.0 / (1.0 + np.exp(-z))
    y = (r.uniform(size=n) < p).astype(int)
    X = np.column_stack([income, debt])
    return X, y


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


# =====================================================================
# 1. 回归树桩：用"平方误差下降"做贪心最佳切分（GBDT 的基学习器）
#    返回叶子值（叶内残差均值）+ 结构，供 predict_residual 使用。
# =====================================================================
class RegTree:
    def __init__(self, max_depth=2, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, resid):
        self.tree = self._build(X, resid, depth=0)
        return self

    def _build(self, X, resid, depth):
        n = len(resid)
        leaf = {"leaf": True, "value": float(resid.mean()) if n else 0.0}
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return leaf
        feat, thr, gain = self._best_split(X, resid)
        if feat is None or gain <= 1e-12:
            return leaf
        mask = X[:, feat] <= thr
        left = self._build(X[mask], resid[mask], depth + 1)
        right = self._build(X[~mask], resid[~mask], depth + 1)
        return {"leaf": False, "feat": feat, "thr": thr,
                "left": left, "right": right}

    def _best_split(self, X, resid):
        n, d = X.shape
        parent_sse = float(((resid - resid.mean()) ** 2).sum())
        best = (None, None, 0.0)
        for f in range(d):
            order = np.argsort(X[:, f])
            xs, rs = X[order, f], resid[order]
            # 候选切点：相邻值不同的地方
            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                if xs[i] == xs[i - 1]:
                    continue
                thr = 0.5 * (xs[i - 1] + xs[i])
                l, r = rs[:i], rs[i:]
                sse = ((l - l.mean()) ** 2).sum() + ((r - r.mean()) ** 2).sum()
                gain = parent_sse - float(sse)
                if gain > best[2]:
                    best = (f, thr, gain)
        return best

    def _predict_one(self, x, node):
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["value"]

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# =====================================================================
# 2. 手写 GBDT（对数几率/LogLoss 版）：把分类当"回归残差"接力
#    F_0 = log(p/(1-p))（对数几率初值），每轮：
#      残差 r_i = y_i - sigmoid(F(x_i))  ← LogLoss 的负梯度
#      拟合回归树 → 加学习率收缩 → F += lr * tree.predict(X)
# =====================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class GBDTClassifier:
    def __init__(self, n_estimators=120, learning_rate=0.1,
                 max_depth=2, min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.F0 = 0.0
        self.history = []      # 每轮训练集 logloss

    def fit(self, X, y, X_val=None, y_val=None):
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(p0 / (1 - p0)))   # 常数初值=对数几率
        F = np.full(len(y), self.F0, dtype=float)
        self.val_hist = []
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            resid = y - p                         # 负梯度（=残差）
            tree = RegTree(self.max_depth, self.min_samples_leaf).fit(X, resid)
            F = F + self.lr * tree.predict(X)
            self.trees.append(tree)
            ll = -np.mean(y * np.log(sigmoid(F) + 1e-12)
                          + (1 - y) * np.log(1 - sigmoid(F) + 1e-12))
            self.history.append(ll)
            if X_val is not None:
                pv = sigmoid(self.F0 + self.lr * np.sum(
                    [t.predict(X_val) for t in self.trees], axis=0))
                lv = -np.mean(y_val * np.log(pv + 1e-12)
                              + (1 - y_val) * np.log(1 - pv + 1e-12))
                self.val_hist.append(lv)
        return self

    def decision_function(self, X):
        return self.F0 + self.lr * np.sum([t.predict(X) for t in self.trees], axis=0)

    def predict_proba(self, X):
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

    def feature_importance(self, d):
        """按各树分裂增益(近似=分裂次数×结点方差)累计特征贡献。"""
        imp = np.zeros(d)
        for t in self.trees:
            stack = [t.tree]
            while stack:
                nd = stack.pop()
                if nd["leaf"]:
                    continue
                imp[nd["feat"]] += 1.0
                stack += [nd["left"], nd["right"]]
        return imp / imp.sum()


# =====================================================================
# 3. 主流程
# =====================================================================
def main():
    X, y = make_credit_data(600, seed=42)
    n = len(y)

    # 训练/测试切分（固定种子，可复现）
    idx = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    # 再从训练集切验证集，用于"早停"演示
    n_val = int(0.2 * n_tr)
    va, tr2 = tr[:n_val], tr[n_val:]

    mu, sd = standardize_fit(X[tr2])            # 仅用训练集统计量（防泄漏）
    Xtr = standardize_apply(X[tr2], mu, sd)
    Xva = standardize_apply(X[va], mu, sd)
    Xte = standardize_apply(X[te], mu, sd)
    ytr, yva, yte = y[tr2], y[va], y[te]

    print("=" * 62)
    print("Day12 GBDT 信贷风控 —— Bagging 的相反面：Boosting")
    print("=" * 62)
    print(f"训练 {len(ytr)} / 验证 {len(yva)} / 测试 {len(yte)}")

    # ---- 3.1 训练一个 GBDT ----
    clf = GBDTClassifier(n_estimators=120, learning_rate=0.1,
                         max_depth=2, min_samples_leaf=10)
    clf.fit(Xtr, ytr, Xva, yva)
    acc_tr = (clf.predict(Xtr) == ytr).mean()
    acc_te = (clf.predict(Xte) == yte).mean()
    print(f"\n[GBDT 120棵树 lr=0.1 深度2]")
    print(f"  训练集准确率 {acc_tr:.4f} | 测试集准确率 {acc_te:.4f}")
    print(f"  训练 logloss {clf.history[0]:.4f} → {clf.history[-1]:.4f}")
    print(f"  验证 logloss {clf.val_hist[0]:.4f} → {clf.val_hist[-1]:.4f}"
          f"（最低 {min(clf.val_hist):.4f} @ 第 {int(np.argmin(clf.val_hist))+1} 棵树）")

    # ---- 3.2 混淆矩阵 ----
    pred = clf.predict(Xte)
    tp = int(((pred == 1) & (yte == 1)).sum())   # 坏客户判对
    tn = int(((pred == 0) & (yte == 0)).sum())   # 好客户判对
    fp = int(((pred == 1) & (yte == 0)).sum())   # 拒错好客户
    fn = int(((pred == 0) & (yte == 1)).sum())   # 漏放坏账
    print(f"\n[混淆矩阵·测试集] 好判对 {tn} | 拒错好客户 {fp} | "
          f"漏放坏账 {fn} | 坏判对 {tp}")

    # ---- 3.3 概率可分性 ----
    proba = clf.predict_proba(Xte)[:, 1]
    m_bad = proba[yte == 1].mean()
    m_good = proba[yte == 0].mean()
    print(f"[概率分离] 真实违约者平均违约概率 {m_bad:.3f} | "
          f"守约者 {m_good:.3f}")

    # ---- 3.4 特征重要性 ----
    imp = clf.feature_importance(d=2)
    print(f"[特征重要性] 收入 {imp[0]:.3f} | 负债率 {imp[1]:.3f}")

    # ---- 3.5 与"单棵深树 / 随机森林思路"对照：boosting 数量效应 ----
    print("\n[Boosting 轮数效应] 弱树接力越多，偏差越小，但过多会过拟合：")
    rounds = [1, 5, 20, 60, 120]
    for m in rounds:
        c = GBDTClassifier(n_estimators=m, learning_rate=0.1,
                           max_depth=2, min_samples_leaf=10).fit(Xtr, ytr)
        print(f"  树数 {m:>3}: 训练 {(c.predict(Xtr)==ytr).mean():.4f} | "
              f"测试 {(c.predict(Xte)==yte).mean():.4f}")

    # ---- 3.6 学习率效应（同样 60 棵树） ----
    print("\n[学习率效应] lr 越小步长越保守，需要更多树但泛化更稳：")
    for lr in [0.05, 0.1, 0.3, 1.0]:
        c = GBDTClassifier(n_estimators=60, learning_rate=lr,
                           max_depth=2, min_samples_leaf=10).fit(Xtr, ytr)
        print(f"  lr={lr:<4}: 训练 {(c.predict(Xtr)==ytr).mean():.4f} | "
              f"测试 {(c.predict(Xte)==yte).mean():.4f}")

    # ---- 3.7 与 sklearn 交叉对照 ----
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        sk = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.1, max_depth=2,
            min_samples_leaf=10, random_state=42).fit(Xtr, ytr)
        acc_sk = (sk.predict(Xte) == yte).mean()
        print(f"\n[交叉对照] sklearn GBDT 测试准确率 {acc_sk:.4f} "
              f"(手写 {acc_te:.4f}，同一量级)")
    except Exception as e:
        acc_sk = None
        print(f"\n[交叉对照] sklearn 不可用：{e}")

    # ---- 3.8 断言：正确性自检 ----
    assert len(clf.trees) == 120
    assert 0.55 < acc_tr <= 1.0, "训练准确率应显著高于随机猜测"
    assert acc_te > 0.70, "测试准确率应明显优于随机"
    assert clf.history[-1] < clf.history[0], "训练 logloss 应随轮数下降"
    assert m_bad > m_good + 0.15, "真实违约者概率应显著高于守约者"
    assert abs(imp.sum() - 1.0) < 1e-8, "特征重要性应归一"
    assert imp[1] > imp[0], "负债率应为主导判别特征（与常识一致）"
    print("\n[断言] 全部通过 ✓")

    # =================================================================
    # 4. 画图
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) 学习曲线：训练/验证 logloss 随树数
    ax = axes[0]
    ax.plot(range(1, len(clf.history) + 1), clf.history, lw=2,
            color="#1f77b4", label="训练 logloss")
    ax.plot(range(1, len(clf.val_hist) + 1), clf.val_hist, lw=2,
            color="#d62728", label="验证 logloss")
    bi = int(np.argmin(clf.val_hist))
    ax.axvline(bi + 1, ls="--", color="gray", alpha=0.7)
    ax.annotate(f"验证最优\n第{bi+1}棵树", (bi + 1, clf.val_hist[bi]),
                textcoords="offset points", xytext=(10, 18), fontsize=9)
    ax.set_xlabel("boosting 轮数（树的数量）")
    ax.set_ylabel("logloss（越小越好）")
    ax.set_title("(a) 弱树接力：训练/验证损失")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) 决策边界（收入×负债率 标准化平面）
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(Xtr[:, 0].min() - 0.5, Xtr[:, 0].max() + 0.5, 200),
                         np.linspace(Xtr[:, 1].min() - 0.5, Xtr[:, 1].max() + 0.5, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.75)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.8)
    good = yte == 0
    ax.scatter(Xte[good, 0], Xte[good, 1], s=18, c="#2ca02c",
               edgecolors="w", linewidths=0.4, label="守约(测试)")
    ax.scatter(Xte[~good, 0], Xte[~good, 1], s=18, c="#8c564b",
               marker="^", edgecolors="w", linewidths=0.4, label="违约(测试)")
    ax.set_xlabel("收入（标准化）")
    ax.set_ylabel("负债率（标准化）")
    ax.set_title("(b) GBDT 违约概率场（黑线=0.5 判定）")
    ax.legend(fontsize=8, loc="upper right")

    # (c) 特征重要性
    ax = axes[2]
    names = ["收入", "负债率"]
    colors = ["#2ca02c", "#d62728"]
    ax.bar(names, imp, color=colors, alpha=0.85)
    for i, v in enumerate(imp):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(imp) * 1.25)
    ax.set_ylabel("分裂贡献占比")
    ax.set_title("(c) 特征重要性：谁在主导判别")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Day12 · GBDT 信贷风控：串行接力降偏差（Boosting 对 Bagging）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "gbdt_credit_boosting.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[图] 已保存 {out}")

    print("\n完成 ✓")


if __name__ == "__main__":
    main()
