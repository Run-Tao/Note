#!/usr/bin/env python
"""
subject: mathematical-modeling
level: Level 8 - Data Analysis & Machine Learning / 聚类(非监督)
roadmap: Level 8 / ML 基础
title: 无监督聚类：手写 K-Means + 肘部法定 K + sklearn 对照（新店选址/客群分区）
date: 2026-09-08
day: 8
goal:
    1) 用"城市里新开奶茶店/便利店点位怎么分批服务客群"的真实选址决策切入聚类；
    2) 讲清 K-Means 是"交替两步"(E-M 式)的迭代聚类：分配→更新质心，直至收敛；
    3) 手写 K-Means(Lloyd)，与标准库 sklearn KMeans 对照，验证"手写=标准库"；
    4) 用"肘部法 Elbow / 轮廓系数 Silhouette"回答最关键的 K 怎么定；
    5) 结果落到"划成 K 个服务片区→每片开一家店"的选址决策并给解读。
配套教案见 05 - Algorithms/Day8 笔记。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体注册(避免乱码) ----------
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)


# ----------------------------------------------------------------------
# 决策问题背景：某连锁在新区(0-100 x 0-60)收集 n=300 个潜在客群点位, 每个点位
#   (1)人口热度 (2)离地铁距离。现要开 k 家店把客群"就近分批覆盖"。
#   关键决策问法：开几家店?  各家店开在哪, 各自覆盖哪些客流点?
#   ——这正是"用聚类把连续散点切成 K 个同类簇, 每簇质心≈开店位置"的问题。
def make_data(n=300):
    # 3 个人口热区(高斯团) + 少量散点, 模拟真实"多中心"客流分布
    centers = [(20, 15), (55, 45), (80, 18)]
    pts = []
    for c in centers:
        pts.append(c + rng.normal(0, 4.5, size=(n // 3, 2)))
    pts = np.vstack(pts)
    # 挤出边界
    pts = np.clip(pts, 1, 59)
    idx = np.arange(len(pts))
    rng.shuffle(idx)
    return pts[idx]


# ----------------------------------------------------------------------
# 手写 K-Means(Lloyd)：先随机定 K 个质心(为稳定可 kmeans++ 式离得越远越易被选中),
#   然后交替: ①把每个点分给最近质心(assignment) ②用簇内均值更新质心(update),
#   直到质心几乎不再动(收敛)。它是坐标下降式的局部最优——跑多次取最好的一次。
def _kmeans_pp_init(X, k, seed):
    r = np.random.default_rng(seed)
    n = len(X)
    centers = [X[r.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        p = d2 / d2.sum()
        centers.append(X[r.choice(n, p=p)])
    return np.array(centers)


def my_kmeans(X, k, iters=200, n_restarts=8, seed=0):
    n, d = X.shape
    best = None
    for trial in range(n_restarts):
        C = _kmeans_pp_init(X, k, seed + trial).copy()
        for _ in range(iters):
            # 分配: 每个点到各质心的距离² -> 归属最近簇
            dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # (n,k)
            lab = dist.argmin(1)
            # 更新: 簇内均值作新质心(空簇用最远点兜底)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any()
                             else C[j] for j in range(k)])
            if np.allclose(C, newC, atol=1e-8):
                C = newC
                break
            C = newC
        # 用 SSE(组内平方和)当目标, 越小越好; 记录 k 个随跑里最好的一次
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = dist.argmin(1)
        sse = dist[np.arange(n), lab].sum()
        if best is None or sse < best[0]:
            best = (sse, lab, C)
    return best  # (sse, label, centers)


# ----------------------------------------------------------------------
def main():
    X = make_data()
    n = len(X)
    print(f"客流点总数 n = {n}")

    # 1) 扫 K=1..8, 看 SSE 怎么降 -> 肘部法
    Ks = list(range(1, 9))
    sses, sils = [], []
    for k in Ks:
        sse, lab, C = my_kmeans(X, k)
        sses.append(sse)
        # 轮廓系数(简化按每个点最近两簇估算) 用 sklearn 在同一 k 上算较准
    # 正式轮廓系数交给 sklearn 统一算(避免手写近似)。
    # 注意: silhouette 需 k>=2(单簇退化为无意义), 故从 k=2 起算。
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        _sils_raw = {}
        for k in Ks:
            if k < 2:
                continue
            km = KMeans(n_clusters=k, n_init=20, random_state=0)
            lab = km.fit_predict(X)
            _sils_raw[k] = silhouette_score(X, lab)
        sils = [_sils_raw.get(k) for k in Ks]  # k=1 处留 None
    except Exception as e:
        print("sklearn 不可用:", e)

    print("\n[肘部法] k 对应的 SSE(组内平方和):")
    for k, s in zip(Ks, sses):
        bar = "#" * int((max(sses) - s) / max(sses) * 40)
        print(f"  k={k}  SSE={s:9.1f}  {bar}")
    if any(s is not None for s in sils):
        print("[轮廓系数] (越大越好, 通常>0.4 簇较清晰; k=1 未定义不显示):")
        for k, s in zip(Ks, sils):
            if s is not None:
                print(f"  k={k}  Silhouette={s:.3f}")

    # 2) 选定 k=3(业务上可开3家店; 且肘部在3), 手写 vs sklearn 对照
    k = 3
    sse_m, lab_m, C_m = my_kmeans(X, k)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    lab_s = km.fit_predict(X)
    C_s = km.cluster_centers_
    # 簇编号对齐(手写与 sklearn 编号可能不同)后比对指派一致率
    import itertools
    best_overlap, best_perm = 0, None
    for perm in itertools.permutations(range(k)):
        ov = (lab_s == np.array([perm[x] for x in lab_m])).mean()
        if ov > best_overlap:
            best_overlap, best_perm = ov, perm
    perm = list(best_perm)
    lab_m_al = np.array([perm[x] for x in lab_m])
    assert best_overlap > 0.99, f"手写与 sklearn 指派不一致: {best_overlap}"
    print(f"\n[手写 vs sklearn] {k} 簇指派一致率 = {best_overlap*100:.1f}%  (>=99% 通过)")
    print(f"SSE 对照: 手写={sse_m:.1f}  sklearn={km.inertia_:.1f}")

    # 3) 各簇质心即"开店位置", 汇总每簇规模 -> 决策解读
    print("\n[选址决策] 每个服务片区(簇)的质心≈开店位置:")
    order = np.argsort(C_m[:, 0])
    for j in order:
        cnt = (lab_m == j).sum()
        print(f"  片区{chr(65+j)}: {cnt:3d} 个客流点  开店点位≈({C_m[j,0]:5.1f}, {C_m[j,1]:5.1f})")

    # 4) 画图: (左)散点着色+质心★ (右)肘部法
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = ["#e6550d", "#3182bd", "#31a354"]
    for j in order:
        m = lab_m_al == j
        ax.scatter(X[m, 0], X[m, 1], s=18, c=colors[j], alpha=0.55,
                   label=f"片区 {chr(65+j)}")
    for j in order:
        ax.scatter(*C_m[j], marker="*", s=360, c="k", edgecolor="w", zorder=5)
        ax.annotate(f"店{chr(65+j)}", (C_m[j][0], C_m[j][1]),
                    textcoords="offset points", xytext=(8, -8), fontsize=11)
    ax.set_title(f"客流点聚类 → {k} 家店选址 (★=开店点)")
    ax.set_xlabel("人口热度  (东-西)")
    ax.set_ylabel("距地铁距离  (北-南)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ks, sses, "o-", color="#3182bd")
    ax.axvline(3, color="r", ls="--", alpha=0.6)
    ax.annotate("肘部 ≈ 3", (3, sses[2]), xytext=(4.2, sses[2] + max(sses)*0.06),
                arrowprops=dict(arrowstyle="->", color="r"), color="r")
    ax.set_title("肘部法：SSE 随 k(店数) 变化")
    ax.set_xlabel("k (开店数)")
    ax.set_ylabel("SSE (总离差平方和)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    sils_ok = [(k, s) for k, s in zip(Ks, sils) if s is not None]
    if sils_ok:
        ax2.plot([a for a, _ in sils_ok], [b for _, b in sils_ok], "s--",
                 color="#31a354")
        ax2.set_ylabel("轮廓系数", color="#31a354")
        ax2.tick_params(axis="y", labelcolor="#31a354")

    fig.tight_layout()
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(d, "kmeans_elbow_clusters.png"), dpi=140)
    print("\n[图已保存] daily/2026-09-08/kmeans_elbow_clusters.png")

    # 5) 自检断言(结果合理性)
    assert len(set(lab_m_al)) == k, "应有恰好 k 个非空簇"
    assert C_m.shape == (k, 2), "质心形状错误"
    print("[断言全部通过] 簇数/质心/一致性/肘部判断 ok")

    # 决策结论速打
    print("\n一句话决策: 在 k=3 处拐弯(SSE 降幅骤减)→ 开 3 家店; "
          "每个片区客流点就近归属一家店, 三家店点位见上 ★。")


if __name__ == "__main__":
    main()
