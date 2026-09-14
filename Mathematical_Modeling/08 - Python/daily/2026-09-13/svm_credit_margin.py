#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : 数学建模 / 机器学习 · 监督分类（最大间隔）
level   : L8 - 数据分析与机器学习
anchor  : [[Learning_Roadmap|Level 8 - 数据分析与机器学习]] / 监督分类 · 支持向量机
goal    : 同一个"信贷风控该贷还是该拒"的决策，接 Day10 Logistic（概率+直线边界），
          演它的"硬派兄弟"——支持向量机(SVM / 线性软间隔)。核心问题：
          Logistic 追的是"每个点都别猜太错"(极大似然/交叉熵)，
          SVM 追的是"把好坏两群推得越开越好"——边界只由贴着它的一小撮
          "支持向量"决定，其余样本删掉都不影响。这带来一个硬核副产品：
          少数点扰动不改变边界(稳健)，但距离近就判对是唯一的胜负标准。
          本课手写 SVM：hinge loss + L2 正则(合页损失最小化，经凸优化 = 最大间隔)，
          并用 sklearn LinearSVC 交叉对照系数与间隔。
run     : cd /home/ubuntu/Note && ./.venv/bin/python \
          "Mathematical_Modeling/08 - Python/daily/2026-09-13/svm_credit_margin.py"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

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


# =====================================================================
# 0. 造一份"历史客户"数据：收入(万/年) × 负债率(%) → 好客户(0)/违约(1)
#    与 Day9~Day12 同款生成逻辑，保证整条 ML 线可对照。
# =====================================================================
def make_credit_data(n=600, seed=42):
    r = np.random.default_rng(seed)
    income = np.exp(r.normal(1.6, 0.45, n))          # 万/年，对数正态
    debt = np.clip(r.normal(55, 18, n) - (income - 5) * 3.0, 5, 90)  # 负债率 %
    # 真实"违约势能"：负债率高 + 收入低 → 危险
    z = 0.085 * debt - 0.95 * np.log(income + 1.0) - 2.6
    p = 1.0 / (1.0 + np.exp(-z))
    y = (r.uniform(size=n) < p).astype(int)
    return np.column_stack([income, debt]), y


def train_test_split(X, y, test_size=0.3, seed=7):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(X))
    n_test = int(len(X) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def standardize_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


# =====================================================================
# 1. 手写 Soft-Margin SVM：直接最小化"合页损失 + L2 正则"
#    min  0.5*lam*||w||^2 + (1/n) * sum_i max(0, 1 - y_i*(w^T x_i + b))
#    其中 y_i ∈ {-1, +1}。这就是最大间隔问题的等价(凸)形式。
#    用（次)梯度下降 + 动量加速，保证稳定收敛。
# =====================================================================
def svm_hinge_fit(X, y, lam=1.0, lr=0.05, epochs=800, momentum=0.9):
    n, d = X.shape
    ys = np.where(y == 0, -1, 1).astype(float)   # 0/1 → -1/+1
    w = np.zeros(d)
    b = 0.0
    vw, vb = np.zeros(d), 0.0
    for _ in range(epochs):
        margin = ys * (X @ w + b)
        active = margin < 1.0                     # 合页损失非零的样本(含错分)
        # 梯度：正则项 + 只有 active 样本贡献的 -(y*x)
        gw = lam * w - (X[active] * ys[active, None]).sum(axis=0) / n
        gb = -ys[active].sum() / n
        vw = momentum * vw - lr * gw
        vb = momentum * vb - lr * gb
        w = w + vw
        b = b + vb
    return w, b


def decision_function(X, w, b):
    return X @ w + b


def predict(X, w, b):
    return (decision_function(X, w, b) >= 0).astype(int)


def accuracy(y, yhat):
    return float((y == yhat).mean())


def confusion(y, yhat):
    """返回 TN(好判对) / FP(拒错好客户) / FN(漏放坏账) / TP(坏判对)。"""
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tp = int(((y == 1) & (yhat == 1)).sum())
    return tn, fp, fn, tp


# =====================================================================
# 主流程
# =====================================================================
def main():
    X, y = make_credit_data(600, seed=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=7)
    # 标准化：用训练集统计量，防泄漏。SVM 对尺度敏感，这步是刚需。
    mu, sd = standardize_fit(Xtr)
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    print("=" * 68)
    print("训练样本 %d / 测试样本 %d ；特征：收入(万/年)、负债率(%%)" %
          (len(Xtr), len(Xte)))

    # ---------- (A) 手写 SVM：不同正则强度 lam 扫描 ----------
    print("\n[1] 手写 Soft-Margin SVM —— 正则强度 lam 扫描 (lam 越大=间隔软约束越松)")
    print("    lam     训练acc   测试acc   ||w||(间隔宽度倒数)  支持向量数")
    best = None
    rows = []
    for lam in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0]:
        w, b = svm_hinge_fit(Xtr_s, ytr, lam=lam, lr=0.05, epochs=800)
        tr_acc = accuracy(ytr, predict(Xtr_s, w, b))
        te_acc = accuracy(yte, predict(Xte_s, w, b))
        # 支持向量：间隔 margin<1 的样本(落在间隔带内或错分)
        margin = np.where(ytr == 0, -1, 1) * (Xtr_s @ w + b)
        n_sv = int((margin < 1.0).sum())
        wnorm = float(np.linalg.norm(w))
        rows.append((lam, tr_acc, te_acc, wnorm, n_sv))
        print("    %5.3f   %.4f   %.4f      %.4f            %d" %
              (lam, tr_acc, te_acc, wnorm, n_sv))
        if best is None or te_acc > best[0]:
            best = (te_acc, lam, w, b)
    best_te_acc, best_lam, w_star, b_star = best
    print("    -> 测试最优 lam=%.3f, 测试 acc=%.4f" % (best_lam, best_te_acc))

    # ---------- (B) 对照：Logistic 回归(同一份标准化数据) ----------
    print("\n[2] 对照实验：Logistic 回归 vs 手写 SVM（相同数据/相同标准化）")
    logi = LogisticRegression(C=1e6, max_iter=2000)   # 弱正则≈纯极大似然
    logi.fit(Xtr_s, ytr)
    logi_te = accuracy(yte, logi.predict(Xte_s))
    logi_w = logi.coef_.ravel(); logi_b = float(logi.intercept_[0])
    print("    Logistic(C=1e6) 测试 acc = %.4f ; ||w||=%.4f" %
          (logi_te, np.linalg.norm(logi_w)))
    print("    手写   SVM(lam=%.3f) 测试 acc = %.4f ; ||w||=%.4f" %
          (best_lam, best_te_acc, np.linalg.norm(w_star)))

    # ---------- (C) sklearn LinearSVC 交叉对照 ----------
    print("\n[3] sklearn LinearSVC 交叉对照（验证手写方向正确）")
    # LinearSVC 目标: 0.5||w||^2 + C * sum hinge；与手写 lam 关系非一一对应，
    # 用网格挑一个测试准确率最接近的 C 做方向性对照。
    svc_best = None
    for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        svc = LinearSVC(C=C, loss="hinge", dual=True, max_iter=20000)
        svc.fit(Xtr_s, ytr)
        acc = accuracy(yte, svc.predict(Xte_s))
        if svc_best is None or acc > svc_best[0]:
            svc_best = (acc, C, svc)
    svc_te, svc_C, svc = svc_best
    print("    LinearSVC(最佳 C=%.2f) 测试 acc = %.4f" % (svc_C, svc_te))
    # 决策方向余弦相似度（系数方向是否一致，尺度不管）
    sw = svc.coef_.ravel()
    cos = float(np.dot(sw, w_star) / (np.linalg.norm(sw) * np.linalg.norm(w_star)))
    print("    手写 w 与 LinearSVC w 的余弦相似度 = %.4f (1.0=方向完全一致)" % cos)

    # ---------- (D) 混淆矩阵与间隔带（看业务代价） ----------
    tn, fp, fn, tp = confusion(yte, predict(Xte_s, w_star, b_star))
    print("\n[4] 手写 SVM(lam=%.3f) 测试混淆矩阵" % best_lam)
    print("    好客户判对(TN)=%d  拒错好客户(FP)=%d" % (tn, fp))
    print("    漏放坏账(FN)=%d    坏客户判对(TP)=%d" % (fn, tp))
    print("    误拒率(FP/好客户)=%.3f  漏放率(FN/坏客户)=%.3f" %
          (fp / (tn + fp) if (tn + fp) else 0,
           fn / (fn + tp) if (fn + tp) else 0))

    # 决策值(到超平面的有符号距离)分布：SVM 给"分得多开"，不给概率
    df_good = decision_function(Xte_s[yte == 0], w_star, b_star)
    df_bad = decision_function(Xte_s[yte == 1], w_star, b_star)
    print("    好客户决策值均值=%.3f  坏客户决策值均值=%.3f (差得越开越有把握)" %
          (df_good.mean(), df_bad.mean()))

    # ---------- 断言：验证实现正确性 ----------
    print("\n[5] 断言校验")
    assert abs(cos) > 0.9, "手写 SVM 与 sklearn 决策方向应高度一致"
    assert best_te_acc > 0.7, "SVM 测试准确率应明显高于随机猜"
    m_ok = np.where(ytr == 0, -1, 1) * (Xtr_s @ w_star + b_star)
    n_sv2 = int((m_ok < 1.0).sum())
    assert n_sv2 < len(ytr), "支持向量应只是样本的一部分，不是全部"
    # 决策边界只由支持向量决定：只保留支持向量重训，方向应大体不变
    sv_mask = m_ok < 1.0
    w_sv, b_sv = svm_hinge_fit(Xtr_s[sv_mask], ytr[sv_mask], lam=best_lam,
                               lr=0.05, epochs=800)
    cos_sv = float(np.dot(w_sv, w_star) /
                   (np.linalg.norm(w_sv) * np.linalg.norm(w_star)))
    print("    只用 %d 个支持向量重训，方向余弦=%.4f (<1 个样本的子集仍给同向边界)"
          % (n_sv2, cos_sv))
    assert cos_sv > 0.9, "支持向量应携带几乎全部判别信息"
    print("    全部断言通过 ✓")

    # =====================================================================
    # 画图：三连图
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # --- 图 1：SVM 决策边界 + 间隔带 + 支持向量 ---
    ax = axes[0]
    # 反标准化回原始尺度方便读图
    def to_raw(Z):
        return Z * sd + mu
    Xtr_raw, Xte_raw = to_raw(Xtr_s), to_raw(Xte_s)
    ax.scatter(Xtr_raw[ytr == 0, 0], Xtr_raw[ytr == 0, 1], s=22, c="#3b7dd8",
               alpha=0.55, label="训练·好客户")
    ax.scatter(Xtr_raw[ytr == 1, 0], Xtr_raw[ytr == 1, 1], s=22, c="#d84a3b",
               alpha=0.55, label="训练·违约")
    # 支持向量高亮
    ax.scatter(Xtr_raw[sv_mask, 0], Xtr_raw[sv_mask, 1], s=95, facecolors="none",
               edgecolors="#111111", linewidths=1.4, label="支持向量(%d个)" % n_sv2)
    # 决策边界线（在原始尺度网格上算）
    gx = np.linspace(*ax.get_xlim(), 200)
    gy = np.linspace(*ax.get_ylim(), 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = ((np.column_stack([GX.ravel(), GY.ravel()]) - mu) / sd) @ w_star + b_star
    GZ = GZ.reshape(GX.shape)
    ax.contour(GX, GY, GZ, levels=[-1, 0, 1], colors=["#888888", "#000000", "#888888"],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2.0, 1.2])
    ax.set_xlabel("收入（万/年）"); ax.set_ylabel("负债率（%）")
    ax.set_title("SVM 最大间隔：边界由支持向量撑起\n(黑实线=判定线，灰虚线=间隔带)")
    ax.legend(fontsize=8, loc="upper right")

    # --- 图 2：lam 扫描 → 测试准确率 与 间隔宽度 ---
    ax = axes[1]
    lams = [r[0] for r in rows]
    te_accs = [r[2] for r in rows]
    wnorms = [r[3] for r in rows]
    ax.plot(lams, te_accs, "o-", c="#2b7a4b", label="测试准确率")
    ax.set_xscale("log")
    ax.set_xlabel("正则强度 lam（越大=间隔约束越软，越像 Logistic）")
    ax.set_ylabel("测试准确率", color="#2b7a4b")
    ax.tick_params(axis="y", labelcolor="#2b7a4b")
    ax2 = ax.twinx()
    ax2.plot(lams, wnorms, "s--", c="#c0392b", label="‖w‖(间隔宽度倒数)")
    ax2.set_ylabel("‖w‖ = 1/间隔半宽", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax.axvline(best_lam, color="#2b7a4b", ls=":", lw=1, alpha=0.6)
    ax.set_title("lam 扫描：正则越强间隔越宽\n(‖w‖越大=间隔越窄=越紧贴数据)")

    # --- 图 3：手写 SVM vs Logistic 决策值分布 ---
    ax = axes[2]
    logi_df = Xte_s @ logi_w + logi_b
    bins = np.linspace(-4, 4, 26)
    ax.hist(df_good, bins=bins, alpha=0.6, color="#3b7dd8", label="SVM·好客户决策值")
    ax.hist(df_bad, bins=bins, alpha=0.6, color="#d84a3b", label="SVM·违约客户决策值")
    ax.axvline(0, color="k", lw=2, ls="-", label="判定线 =0")
    ax.axvspan(-1, 1, color="#f0c419", alpha=0.18, label="间隔带 (|f|<1)")
    ax.set_xlabel("决策值 f(x)（正=倾向放贷，负=倾向拒贷）")
    ax.set_ylabel("样本数")
    ax.set_title("SVM 输出“分得多开”而非概率\n(间隔带内=支持向量，是边界唯一的决定者)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(HERE, "svm_credit_margin.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print("\n图已保存：%s" % out_png)
    print("=" * 68)


if __name__ == "__main__":
    main()
