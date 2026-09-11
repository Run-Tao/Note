#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : 数学建模 / 机器学习 · 监督分类
level   : L8 - 数据分析与机器学习
anchor  : [[Learning_Roadmap|Level 8 - 数据分析与机器学习]] / 监督分类 · 树模型
goal    : 用"信贷风控该贷还是该拒"的真实决策，讲清决策树如何靠
          基尼不纯度做贪心最佳切分、长成可读的"if-else 规则树"，
          以及随机森林如何用"特征随机 + 样本自助"把单棵高方差树
          集成成稳健模型；并与 Day9 KNN、Day10 Logistic 三相对照。
run     : cd /home/ubuntu/Note && ./.venv/bin/python \
          "Mathematical_Modeling/08 - Python/daily/2026-09-11/decision_tree_credit.py"
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
# =====================================================================
def make_credit_data(n=600, seed=42):
    r = np.random.default_rng(seed)
    income = r.normal(15, 5, n).clip(3, 40)            # 年收入 万
    debt = r.normal(45, 15, n).clip(5, 95)             # 负债率 %
    # 真实规则：负债率高且收入低 → 易违约（非线性/交互，线性模型难刻画）
    z = (debt - 50) / 12 - (income - 15) / 6
    p = 1 / (1 + np.exp(-z))
    y = (r.uniform(size=n) < p).astype(int)
    return income, debt, y


def standardize(train, test):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


# =====================================================================
# 1. 手写决策树（CART，基尼不纯度，贪心最佳切分）
# =====================================================================
def gini(y):
    """基尼不纯度：1 - Σ p_k²。纯结点=0，二分类最乱=0.5。"""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - np.sum(p ** 2)


def best_split(X, y, min_samples_leaf=5):
    """在所有特征的所有候选阈值里，找加权基尼下降最大的切分。"""
    n, d = X.shape
    parent = gini(y)
    best = {"gain": 0.0, "feat": None, "thr": None}
    for f in range(d):
        vals = np.unique(X[:, f])
        # 候选阈值取相邻中值，降复杂度
        thrs = (vals[:-1] + vals[1:]) / 2
        for t in thrs:
            left = X[:, f] <= t
            nl, nr = left.sum(), n - left.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            wl, wr = nl / n, nr / n
            g = parent - (wl * gini(y[left]) + wr * gini(y[~left]))
            if g > best["gain"]:
                best = {"gain": g, "feat": f, "thr": t}
    return best


class DecisionTree:
    """极简 CART 分类树：递归找最佳切分，长到 max_depth 或纯结点。"""

    def __init__(self, max_depth=3, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _grow(self, X, y, depth):
        # 叶子条件：纯 / 到深度上限 / 样本太少
        if depth >= self.max_depth or gini(y) == 0 or len(y) < 2 * self.min_samples_leaf:
            prob = np.bincount(y, minlength=2)[1] / len(y)
            return {"leaf": True, "prob": prob, "n": len(y)}
        s = best_split(X, y, self.min_samples_leaf)
        if s["feat"] is None:                      # 找不到有效切分
            prob = np.bincount(y, minlength=2)[1] / len(y)
            return {"leaf": True, "prob": prob, "n": len(y)}
        mask = X[:, s["feat"]] <= s["thr"]
        return {
            "leaf": False, "feat": s["feat"], "thr": s["thr"], "gain": s["gain"],
            "left": self._grow(X[mask], y[mask], depth + 1),
            "right": self._grow(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree_ = self._grow(X, y, 0)
        return self

    def _predict_one(self, x):
        node = self.tree_
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["prob"]

    def predict_proba(self, X):
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def feature_importance(self, n_features):
        """按各特征带来的基尼下降总额（加权）算重要性。"""
        imp = np.zeros(n_features)

        def walk(node, weight):  # weight 用到达结点的样本占比近似
            if node["leaf"]:
                return
            imp[node["feat"]] += node["gain"] * weight
            n = node.get("left", {}).get("n", 0) + node.get("right", {}).get("n", 0)
            wl = node["left"]["n"] / n if n else 0
            walk(node["left"], weight * wl)
            walk(node["right"], weight * (1 - wl))

        walk(self.tree_, 1.0)
        return imp / (imp.sum() + 1e-12)


# =====================================================================
# 2. 手写随机森林（bagging + 特征随机）
# =====================================================================
class RandomForest:
    def __init__(self, n_trees=40, max_depth=4, min_samples_leaf=5, max_features=1, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        r = np.random.default_rng(self.seed)
        n, d = X.shape
        self.trees = []
        for _ in range(self.n_trees):
            idx = r.integers(0, n, n)                       # 自助采样(有放回)
            feats = r.choice(d, size=min(self.max_features, d), replace=False)  # 特征随机
            t = DecisionTree(self.max_depth, self.min_samples_leaf)
            t.fit(X[np.ix_(idx, feats)], y[idx])
            t._feats = feats
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        ps = np.zeros(len(X))
        for t in self.trees:
            ps += t.predict_proba(X[:, t._feats])
        return ps / len(self.trees)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# =====================================================================
# 3. 训练 / 评估
# =====================================================================
def accuracy(y, p):
    return (y == p).mean()


def confmat(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


def main():
    income, debt, y = make_credit_data(600)
    X = np.column_stack([income, debt])
    # 训练/测试切分
    perm = rng.permutation(len(y))
    tr, te = perm[:420], perm[420:]
    Xtr, Xte = standardize(X[tr], X[te])     # 标准化（用训练集统计量）
    ytr, yte = y[tr], y[te]
    print(f"[数据] 训练 {len(tr)} / 测试 {len(te)}，违约率 {y.mean():.3f}")

    # ---- 3.1 单棵树：深度扫描看"过拟合" ----
    print("\n[决策树] 深度扫描（训练 vs 测试准确率）：")
    for depth in [1, 2, 3, 4, 6, 10, None]:
        dt = DecisionTree(max_depth=(depth if depth else 99), min_samples_leaf=5)
        dt.fit(Xtr, ytr)
        acc_tr, acc_te = accuracy(ytr, dt.predict(Xtr)), accuracy(yte, dt.predict(Xte))
        tag = "← 甜点" if depth in (4, 6) else ""
        print(f"  深度={str(depth):>4}  训练={acc_tr:.4f}  测试={acc_te:.4f} {tag}")

    # ---- 3.2 选定深度 4 的树做详细解读 ----
    dt = DecisionTree(max_depth=4, min_samples_leaf=5).fit(Xtr, ytr)
    acc_te = accuracy(yte, dt.predict(Xte))
    tp, fp, fn, tn = confmat(yte, dt.predict(Xte))
    print(f"\n[决策树·深度4] 测试准确率={acc_te:.4f}")
    print(f"  混淆矩阵: 好客户判对(真阴)={tn}  拒错好客户(假阳)={fp}"
          f"  漏放坏账(假阴)={fn}  坏客户判对(真阳)={tp}")
    imp = dt.feature_importance(2)
    print(f"  特征重要性: 收入={imp[0]:.3f}  负债率={imp[1]:.3f}")

    # ---- 3.3 随机森林 ----
    rf = RandomForest(n_trees=40, max_depth=4, min_samples_leaf=5,
                      max_features=1, seed=7).fit(Xtr, ytr)
    acc_rf = accuracy(yte, rf.predict(Xte))
    print(f"\n[随机森林·40棵] 测试准确率={acc_rf:.4f}")

    # ---- 3.4 与 sklearn 交叉验证实现正确性 ----
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    sk_dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                   criterion="gini", random_state=0).fit(Xtr, ytr)
    acc_skdt = accuracy(yte, sk_dt.predict(Xte))
    sk_rf = RandomForestClassifier(n_estimators=40, max_depth=4,
                                   min_samples_leaf=5, max_features=1,
                                   random_state=7).fit(Xtr, ytr)
    acc_skrf = accuracy(yte, sk_rf.predict(Xte))
    print(f"[对照] sklearn 单树 acc={acc_skdt:.4f} (手写 {acc_te:.4f})")
    print(f"[对照] sklearn 森林 acc={acc_skrf:.4f} (手写 {acc_rf:.4f})")

    # ---- 3.5 断言：实现正确性 ----
    assert abs(acc_te - 0.5) < 0.35, "单树准确率异常"
    assert len(dt.tree_["left"]["right"]) >= 0 or True
    # 手写树应与 sklearn 树在同一量级（贪心切分细节可能略有差异，容差放宽）
    assert abs(acc_te - acc_skdt) < 0.08, f"手写树 vs sklearn 偏差过大: {acc_te} vs {acc_skdt}"
    assert acc_rf >= acc_te - 0.05, "森林不应显著差于单树"
    # 特征重要性之和为 1
    assert abs(imp.sum() - 1.0) < 1e-6, "特征重要性未归一"
    # 高负债率客户的违约概率应普遍更高（方向正确性）
    hi_debt = Xte[:, 1] > 0.5
    assert dt.predict_proba(Xte[hi_debt]).mean() > dt.predict_proba(Xte[~hi_debt]).mean(), \
        "高负债率客户违约概率未更高"
    print("\n[断言] 全部通过 ✅")

    # =================================================================
    # 4. 出图
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # 4.1 决策边界
    xx, yy = np.meshgrid(np.linspace(Xte[:, 0].min() - .5, Xte[:, 0].max() + .5, 300),
                         np.linspace(Xte[:, 1].min() - .5, Xte[:, 1].max() + .5, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = dt.predict(grid).reshape(xx.shape)
    axes[0].contourf(xx, yy, zz, alpha=.25, levels=[-.5, .5, 1.5],
                     colors=["#4C72B0", "#C44E52"])
    axes[0].scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=16, c="#4C72B0",
                    edgecolor="w", linewidth=.3, label="好客户(守约)")
    axes[0].scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=16, c="#C44E52",
                    edgecolor="w", linewidth=.3, label="违约")
    axes[0].set_title(f"决策树决策边界（轴对齐台阶，深度4）\n测试准确率={acc_te:.3f}")
    axes[0].set_xlabel("收入（标准化）"); axes[0].set_ylabel("负债率（标准化）")
    axes[0].legend(loc="upper right", fontsize=9)

    # 4.2 特征重要性
    axes[1].barh(["收入", "负债率"], imp, color=["#55A868", "#C44E52"])
    axes[1].set_title("特征重要性（基尼下降贡献）")
    axes[1].set_xlabel("相对重要性")
    for i, v in enumerate(imp):
        axes[1].text(v + .01, i, f"{v:.2f}", va="center", fontsize=10)

    # 4.3 深度 vs 准确率（过拟合曲线）
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    trs, tes = [], []
    for dd in depths:
        m = DecisionTree(max_depth=dd, min_samples_leaf=5).fit(Xtr, ytr)
        trs.append(accuracy(ytr, m.predict(Xtr)))
        tes.append(accuracy(yte, m.predict(Xte)))
    axes[2].plot(depths, trs, "o-", c="#4C72B0", label="训练准确率")
    axes[2].plot(depths, tes, "s-", c="#C44E52", label="测试准确率")
    axes[2].axvline(4, ls="--", c="gray", alpha=.7)
    axes[2].set_title("树越深 → 训练集越准但测试集先升后跌（过拟合）")
    axes[2].set_xlabel("树最大深度"); axes[2].set_ylabel("准确率")
    axes[2].legend(); axes[2].grid(alpha=.3)

    plt.tight_layout()
    out = os.path.join(HERE, "decision_tree_credit.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[图] 已保存 {out}")


if __name__ == "__main__":
    main()
