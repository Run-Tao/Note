#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : 数学建模 / 评价与决策模型（多准则决策 MCDM）
level   : L5 - 评价与决策模型
anchor  : [[Learning_Roadmap|Level 5 - 评价与决策模型]] / AHP · 熵权法 · TOPSIS
goal    : 同一个"该选哪个方案/城市/供应商"的决策，用三种经典评价模型一次讲清：
          (1) AHP 层次分析法 —— 主观赋权：把"两两谁更重要"变成可检验的权重；
          (2) 熵权法 —— 客观赋权：让数据自己说话，信息量大的指标权重大；
          (3) TOPSIS 逼近理想解 —— 综合排序：离"最优解"越近、离"最差解"越远越好。
          并以"4 座候选城市落户选址"为真实决策场景，三法互相印证 + 做敏感性分析。
          全部手写（numpy），不依赖任何专用评价库；结果可读、可复现。
run     : cd /home/ubuntu/Note && ./.venv/bin/python \
          "Mathematical_Modeling/08 - Python/daily/2026-09-15/multi_criteria_decision_ahp_topsis.py"
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
np.set_printoptions(precision=4, suppress=True)


# =====================================================================
# 场景数据：某互联网公司要在 4 座候选城市设区域总部，按 5 个指标打分
# ---------------------------------------------------------------------
# 指标        类型        含义（越大越好 or 越小越好）
#   C1 人才供给   效益(+)   高校/工程师储备（0-10 分）
#   C2 生活成本   成本(-)   房租+薪酬折算（越低越好，0-10 分）
#   C3 政策补贴   效益(+)   税收返还/补贴力度（0-10 分）
#   C4 交通枢纽   效益(+)   机场/高铁/物流通达度（0-10 分）
#   C5 市场潜力   效益(+)   目标客户密度与增长（0-10 分）
# =====================================================================
cities = ["城市甲", "城市乙", "城市丙", "城市丁"]
criteria = ["人才供给", "生活成本", "政策补贴", "交通枢纽", "市场潜力"]
benefit = np.array([True, False, True, True, True])   # False = 成本型（越小越好）

# 决策矩阵 X：行=方案(城市)，列=指标
X = np.array([
    [9.0, 4.0, 7.0, 8.0, 9.5],   # 城市甲：人才/交通/市场强，但生活成本高
    [7.5, 7.5, 6.0, 7.0, 7.0],   # 城市乙：各指标均衡
    [6.0, 8.5, 9.5, 5.5, 6.5],   # 城市丙：成本低、补贴猛，但交通/市场弱
    [8.0, 6.0, 5.0, 9.0, 8.0],   # 城市丁：交通/人才不错
])


# =====================================================================
# 1) AHP：主观赋权 —— 特征值法 + 一致性检验
# =====================================================================
# 判断矩阵 A：a_ij = 指标 i 相对指标 j 的重要度（1-9 标度），互反 a_ji=1/a_ij
# 专家判断（管理层视角）：市场潜力最重要，其次人才，生活成本相对最不重要
A = np.array([
    [1,   2,   3,   4,   1/2],   # 人才供给
    [1/2, 1,   2,   3,   1/3],   # 生活成本
    [1/3, 1/2, 1,   2,   1/4],   # 政策补贴
    [1/4, 1/3, 1/2, 1,   1/5],   # 交通枢纽
    [2,   3,   4,   5,   1  ],   # 市场潜力
], dtype=float)

# RI 随机一致性指标（n=1..10）
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(A, ri_table=RI_TABLE):
    """特征值法求 AHP 权重，并做一致性检验。
    返回 (权重向量 w, 最大特征值 lambda_max, CI, CR, 是否通过)。"""
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))          # 最大特征值索引
    lam_max = eigvals.real[k]
    w = eigvecs.real[:, k]
    w = np.abs(w)
    w = w / w.sum()                            # 归一化
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = ri_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return w, lam_max, CI, CR, (CR < 0.1)


def ahp_geometric_mean(A):
    """几何平均（方根）法作为稳健对照，一阶近似无需特征值。"""
    g = np.prod(A, axis=1) ** (1.0 / A.shape[0])
    return g / g.sum()


# =====================================================================
# 2) 熵权法：客观赋权 —— 让数据自己说话
# =====================================================================
def entropy_weights(X, benefit):
    """熵权法：先极差标准化(成本型取负向)，再由信息熵定权。
    指标越分散(信息熵越小) -> 区分度越大 -> 权重越大。"""
    n, m = X.shape
    Z = np.zeros_like(X, dtype=float)
    for j in range(m):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if hi == lo:
            Z[:, j] = 1.0
        elif benefit[j]:
            Z[:, j] = (col - lo) / (hi - lo)
        else:
            Z[:, j] = (hi - col) / (hi - lo)   # 成本型反向
    # 归一化到概率（平移避免 0 导致 log 未定义）
    P = Z / Z.sum(axis=0, keepdims=True)
    eps = 1e-12
    E = -(P * np.log(P + eps)).sum(axis=0) / np.log(n)   # 各指标信息熵 [0,1]
    d = 1.0 - E                                            # 差异系数
    w = d / d.sum()
    return w, Z, E


# =====================================================================
# 3) TOPSIS：逼近理想解 —— 综合排序
# =====================================================================
def topsis(X, w, benefit):
    """TOPSIS：向量归一化 -> 加权 -> 到正/负理想解的距离 -> 贴近度排序。"""
    n, m = X.shape
    # 3.1 向量归一化（消除量纲）
    norm = np.sqrt((X ** 2).sum(axis=0, keepdims=True))
    R = X / norm
    # 3.2 加权
    V = R * w
    # 3.3 理想解：效益型取最大、成本型取最小
    ideal_best = np.where(benefit, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(benefit, V.min(axis=0), V.max(axis=0))
    # 3.4 欧氏距离
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    # 3.5 贴近度 C = d_worst / (d_best + d_worst)，越接近 1 越好
    C = d_worst / (d_best + d_worst)
    rank = (-C).argsort().argsort() + 1     # 1 = 最优
    return C, rank, V, d_best, d_worst, ideal_best, ideal_worst


def main():
    print("=" * 68)
    print("多准则决策：4 座候选城市落户选址（AHP / 熵权法 / TOPSIS 三件套）")
    print("=" * 68)
    print("\n决策矩阵 X（行=城市，列=指标）:")
    print(f"{'':8s}" + "".join(f"{c:>10s}" for c in criteria))
    for name, row in zip(cities, X):
        print(f"{name:8s}" + "".join(f"{v:>10.1f}" for v in row))
    print("指标方向:", {c: ("效益+" if b else "成本-") for c, b in zip(criteria, benefit)})

    # ---------------- AHP ----------------
    print("\n" + "-" * 68)
    print("【1】AHP 层次分析法（主观赋权）")
    w_ahp, lam, CI, CR, ok = ahp_weights(A)
    w_root = ahp_geometric_mean(A)
    print(f"  最大特征值 λmax = {lam:.4f}（理论值 n = {len(criteria)}）")
    print(f"  CI = {CI:.4f}, RI = {RI_TABLE[len(criteria)]}, CR = {CR:.4f} -> "
          f"{'通过一致性检验(<0.1)' if ok else '未通过，需重调判断矩阵'}")
    print("  AHP 权重(特征值法):")
    for c, v in zip(criteria, w_ahp):
        print(f"    {c:>6s}: {v:.4f}")
    print("  几何平均法对照:", {c: round(float(v), 4) for c, v in zip(criteria, w_root)})
    assert ok, "AHP 一致性检验未通过"
    assert abs(w_ahp.sum() - 1) < 1e-9

    # ---------------- 熵权法 ----------------
    print("\n" + "-" * 68)
    print("【2】熵权法（客观赋权，让数据自己说话）")
    w_ent, Z, E = entropy_weights(X, benefit)
    print("  各指标信息熵 E 与差异系数 d=1-E:")
    for c, e, w_ in zip(criteria, E, w_ent):
        print(f"    {c:>6s}: E={e:.4f}  d={1-e:.4f}  weight={w_:.4f}")
    print("  熵权(越大说明该指标在方案间区分度越高):")
    assert abs(w_ent.sum() - 1) < 1e-9

    # ---------------- 权重对比 ----------------
    print("\n" + "-" * 68)
    print("【3】两种权重来源对比（主客观对照）")
    print(f"  {'指标':<8s}{'AHP(主观)':>12s}{'熵权(客观)':>12s}{'差异':>10s}")
    for c, wa, we in zip(criteria, w_ahp, w_ent):
        print(f"  {c:<8s}{wa:>12.4f}{we:>12.4f}{abs(wa-we):>10.4f}")

    # ---------------- TOPSIS ----------------
    print("\n" + "-" * 68)
    print("【4】TOPSIS 综合排序")
    report = {}
    for tag, w in [("AHP 权重", w_ahp), ("熵权 权重", w_ent)]:
        C, rank, V, db, dw, ib, iw = topsis(X, w, benefit)
        report[tag] = (C, rank)
        print(f"\n  ── 用 {tag} 的 TOPSIS 结果 ──")
        print(f"    {'城市':<8s}{'d+':>8s}{'d-':>8s}{'贴近度C':>10s}{'排名':>6s}")
        for i, name in enumerate(cities):
            print(f"    {name:<8s}{db[i]:>8.4f}{dw[i]:>8.4f}{C[i]:>10.4f}{rank[i]:>6d}")
        best = cities[int(np.argmax(C))]
        print(f"    最优方案: {best}")

    # AHP / 熵权 是否给出相同冠军
    champ_ahp = cities[int(np.argmax(report['AHP 权重'][0]))]
    champ_ent = cities[int(np.argmax(report['熵权 权重'][0]))]
    print(f"\n  结论一致性: AHP 选出【{champ_ahp}】，熵权选出【{champ_ent}】——"
          f"{'一致 ✅' if champ_ahp == champ_ent else '不一致 ⚠️（需讨论权重来源）'}")

    # ---------------- 敏感性分析 ----------------
    print("\n" + "-" * 68)
    print("【5】敏感性分析：扰动权重，看冠军是否稳健")
    rng = np.random.default_rng(42)
    base = w_ahp
    n_trials = 2000
    counts = {c: 0 for c in cities}
    for _ in range(n_trials):
        noise = rng.uniform(0.5, 1.5, size=len(criteria))   # ±50% 乘性扰动
        w = base * noise
        w = w / w.sum()
        C, _, *_ = topsis(X, w, benefit)
        counts[cities[int(np.argmax(C))]] += 1
    print(f"  在 AHP 权重 ±50% 随机扰动 {n_trials} 次下，各城市夺冠频率:")
    for c in sorted(counts, key=lambda k: -counts[k]):
        print(f"    {c:<8s}: {counts[c]/n_trials:6.1%}")
    win_rate = max(counts.values()) / n_trials
    print(f"  最高夺冠频率 = {win_rate:.1%} -> 结论"
          f"{'稳健' if win_rate > 0.6 else '对权重较敏感，需谨慎'}")

    # ---------------- 断言 ----------------
    assert abs(w_ahp.sum() - 1) < 1e-9
    assert CR < 0.1
    assert abs(w_ent.sum() - 1) < 1e-9
    C_ahp, rank_ahp = report['AHP 权重']
    # rank 为名次(1=最优)：验证它等价于按贴近度降序的名次
    expected_rank = np.empty_like(C_ahp, dtype=int)
    expected_rank[np.argsort(-C_ahp)] = np.arange(1, len(C_ahp) + 1)
    assert list(rank_ahp) == list(expected_rank)
    assert max(C_ahp) <= 1.0 + 1e-9 and min(C_ahp) >= -1e-9
    print("\n[断言] AHP 权重和为1 ✅  一致性通过 ✅  熵权和为1 ✅  "
          "排名与贴近度一致 ✅  贴近度∈[0,1] ✅")

    # ================= 画图 =================
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # 图1 权重对比
    ax = axes[0]
    y = np.arange(len(criteria))
    h = 0.38
    ax.barh(y - h/2, w_ahp, height=h, color="#4C78A8", label="AHP(主观)")
    ax.barh(y + h/2, w_ent, height=h, color="#F58518", label="熵权(客观)")
    ax.set_yticks(y); ax.set_yticklabels(criteria)
    ax.invert_yaxis()
    ax.set_xlabel("权重"); ax.set_title("图1  主客观两类权重对比")
    ax.legend(); ax.grid(axis="x", alpha=0.3)

    # 图2 TOPSIS 贴近度排序（AHP权重）
    ax = axes[1]
    order = np.argsort(-C_ahp)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(cities)))
    ax.bar(range(len(cities)), C_ahp[order], color=colors)
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels([cities[i] for i in order])
    ax.set_ylabel("贴近度 C（越接近1越好）")
    ax.set_title("图2  TOPSIS 综合排名（AHP权重）")
    for k, i in enumerate(order):
        ax.text(k, C_ahp[i] + 0.005, f"{C_ahp[i]:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(C_ahp) * 1.18)
    ax.grid(axis="y", alpha=0.3)

    # 图3 敏感性：夺冠频率
    ax = axes[2]
    names = sorted(counts, key=lambda k: -counts[k])
    vals = [counts[c] / n_trials for c in names]
    ax.bar(names, vals, color="#E45756")
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("夺冠频率")
    ax.set_title("图3  敏感性分析（权重±50%扰动2000次）")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(HERE, "multi_criteria_decision_ahp_topsis.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[图已保存] {out}")


if __name__ == "__main__":
    main()
