#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L5 算法（聚类分析 · 谱聚类 Normalized Cut / 归一化拉普拉斯 / 本征间隙 / 图连通分量）
          L8 机器学习（相似度即拓扑：k 近邻图 vs RBF 全连接图 ——「谁是邻居」是建模选择，不是数据给定）
anchor  : 前接 Day87 混合变量聚类（Gower+PAM，k 由轮廓在 5/7 之间犹豫）、Day88 层次聚类（jump 判据、FCM 隶属度）；
          前接 Day63 谱二分（Fiedler 向量只切一刀、且「不是最省的刀」）、Day75 Max-Cut / SDP 松弛；
          全库 `谱聚类` / `Ncut` / `归一化拉普拉斯` / `k 近邻图连通分量` 只被点名、从未做过 —— 本讲补齐。
goal    : 云溪县民政局 2027 年社区养老服务站「责任片」划定：站数已由建筑面积硬性定死 3 个，
          问 **3 个责任片怎么切、边界设在哪、哪个片区会落在两站之间、这个切法稳不稳**。
          本讲回答五个问题：
          ① **图的连通块数 ≡ 归一化拉普拉斯零本征值个数** —— 这一步是谱聚类代码里**没有任何提示**就会出错的地方：
             互近邻图把北岭、南坡两个安置点摘成**度 0 的孤立点**，对角归一化 1/√0 = inf → 特征向量 NaN → k-means 抛错；
             更要命的是反过来的错法：不检查连通性、直接把 k-means 扔上去，孤立点被随手丢进某片，片规模**静默**改变；
          ② **Ncut 比民政局「两刀切地图」好在哪、好多少** —— 同一个目标函数当裁判：0.7543 vs 0.8018（−5.9%），
             而且差在哪一刀说得出来：两刀切把河西老城沿 x=1.4 又劈了一刀；
          ③ **近邻数 k 是旋钮，邻域半径不是** —— k=5 时半径 1.8/2.0 与不设半径的切法 ARI 只有 0.86（差在「铁路南归谁」）；
             把 k 加到 7，片就变了（平均 ARI 0.72）⇒ 结论的适用范围（k=5~6）必须写进论文；
          ④ **把图换成 RBF 全连接（没有「不连」这回事），片就散了** —— ARI 0.55；
             谱聚类的结论强度压在「图可断裂」这个假设上，不是压在「谱」上；
          ⑤ **本征间隙不能替你定 k，更不能替你定站数** —— 本例间隙说 2，站要 3；孤立点再算两块是 4。
          结论落成一句话给民政局：三片 = 河东（含东岗安置）/ 河西中段 / 河西老街；
          北岭、南坡两个安置点**图上无邻居**，不能靠聚类定归属，走「就近站 + 每月上门」个案政策；
          过江那唯一一条桥（铁路南—河东南苑）没被切断，桥头两片设「双站共管」。

决策场景：云溪县民政局 2027 年社区养老服务站布点（3 站已定，本讲只划责任片 + 定边界 + 判稳健性）。

运行（全篇唯一入口，副作用只在 main 里发生一次）：
  cd /home/ubuntu/Note && ./.venv/bin/python -W error \
    "Mathematical_Modeling/08 - Python/daily/2026-11-27/day89_spectral_ncut.py"

约定（全篇统一口径）：
  * 数据是自己编的：18 个住宅片区（片区中心坐标、常住老人数），量级按县城社区台账习惯；
  * 图 = 片区之间「家属顺路探视 / 老人日常串门」的强度，**带权无向图**；三种建图口径：
      W_mut : 互为 k 近邻（mutual kNN，无半径），边长 = 1/(1+d)，k=5 —— 本讲主线；
      W_one : 单向 kNN（任一方在对方 k 近邻里），k=5 —— 会出现「枢纽节点」把链拉长；
      W_rbf : 高斯核 exp(-d^2/2σ^2)，σ = 两两距离中位数 —— 全连接图，没有「不连」这回事；
  * Ncut：L_sym = I - D^{-1/2} W D^{-1/2}，取最小 k 个特征向量 → 行归一化 → k-means；
    另给**手写幂迭代**求 Fiedler 向量（与 Day63 谱二分对照）；
  * 稳健性：① k ∈ {5,6,7} / 半径截止 ∈ {1.8, 2.0, ∞} 双向扫描；② 坐标 ±60 m 抖动 100 次重算；
  * 断言全部卡教学点：连通块数、唯一的过江桥、k 扫描 ARI、RBF 的 ARI、本征间隙 ≠ 站数。

⚠ 本讲实跑踩坑（代码里也留了注释，教案第五节逐条展开）：
  ① 互近邻图把北岭、南坡摘成**度 0 孤立点**，对角归一化 1/√0 = inf → 特征向量 NaN → k-means 抛 ValueError；
     修法见 `normalized_laplacian()`：只对非孤立点取子矩阵（keep 掩码），孤立点显式报名单；
  ② 孤立点若混在 k-means 里，会被随手丢进一片、**静默**改变片规模 —— 必须显式剔除并单独处理；
  ③ 第一版用「邻域半径 CUT」想演示「稠密图」的力量，结果加不加半径结论几乎一样 ——
     真正的旋钮是 k；**一个不起作用的参数写进敏感性分析就是凑数**；
  ④ 第一版把「几何手工切法」写成「沿江一刀 + 河东按 x 中位数」，结果它比 Ncut 贵 23%（一次假胜利）：
     错在让人工切法在两刀之外又多切了一刀；改成「两刀切地图」后差距回到 5.9%，这才是可以写进报告的数。
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ── 中文字体（图里有中文，不注册就是方块） ───────────────────────────────────
for _f in ("Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
T0 = time.time()
K_STATIONS = 3   # 业务已定：3 个站
K_NB = 5         # 主线近邻数

# ── 数据：18 个住宅片区（名, x km, y km, 老人数） ────────────────────────────
# 县城东西向：x≈0~3.3 河西老街；x≈3.5 过江（两座桥：老码头—河东一村、铁路南—河东二村）；
# x≈4.5~7.5 河东新区。北岭安置（x=2.7,y=6.3）与南坡安置（y=−1.6）离主轴 3 km 以上，谁也不挨 —— 本讲第一颗雷。
DISTRICTS = [
    ("西门口", 0.30, 3.40, 980),
    ("老码头", 0.70, 2.60, 1120),
    ("钟楼下", 1.30, 3.05, 860),
    ("文昌巷", 1.55, 2.05, 1240),
    ("河西新村", 2.30, 2.60, 760),
    ("铁路南", 2.75, 1.30, 540),
    ("石坝口", 1.70, 1.10, 620),
    ("砖瓦厂", 0.55, 1.55, 430),
    ("船厂宿舍", 0.15, 0.65, 380),
    ("河东一村", 4.55, 3.25, 1080),
    ("河东二村", 5.15, 2.65, 940),
    ("学府路", 5.65, 3.85, 1320),
    ("体育场", 6.25, 3.35, 700),
    ("政务中心", 6.05, 2.10, 480),
    ("河东南苑", 4.85, 1.60, 600),
    ("北岭安置", 2.70, 6.30, 520),
    ("南坡安置", 3.60, -1.60, 470),
    ("东岗安置", 7.10, 5.30, 360),
]
NAMES = [d[0] for d in DISTRICTS]
XY = np.array([[d[1], d[2]] for d in DISTRICTS], dtype=float)
POP = np.array([d[3] for d in DISTRICTS], dtype=float)
N = len(DISTRICTS)
COLORS = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f"]


# ── 基础工具 ───────────────────────────────────────────────────────────────
def pair_dist(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)


def build_graph(km, k_nb=K_NB, mode="mutual", radius=np.inf, drop_isolated=False):
    """
    三种建图口径。radius 默认 ∞（不做半径截止）——见踩坑 ③。
    drop_isolated=True：度 0 的点**不入图**（本讲选定的口径）。理由：互近邻图给孤立点度 0，
      「它在哪一片」图上无人可问；硬留在图里，k-means 会把它随手丢给某片，**静默**改变片规模。
    """
    d = km.copy()
    np.fill_diagonal(d, np.inf)
    if mode == "rbf":
        sigma = np.median(km[np.triu_indices(N, 1)])
        W = np.exp(-(km ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(W, 0.0)
        return W
    nb = np.zeros((N, N), dtype=bool)
    for i in range(N):
        nb[i, np.argsort(d[i])[:k_nb]] = True
    adj = (nb & nb.T) if mode == "mutual" else (nb | nb.T)
    adj = adj & (km < radius)
    W = np.where(adj, 1.0 / (1.0 + km), 0.0)
    np.fill_diagonal(W, 0.0)
    if drop_isolated:
        W[W.sum(axis=1) == 0, :] = 0.0
        W[:, W.sum(axis=0) == 0] = 0.0
    return W


def bfs_components(W):
    """真正的连通分量数（BFS），用来核对「零特征值个数」。"""
    seen, comps = set(), []
    for s in range(N):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(W[u])[0]:
                if int(v) not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        comps.append(comp)
    return comps


def normalized_laplacian(W):
    """
    L_sym = I - D^{-1/2} W D^{-1/2}，**只算非孤立点**（返回 keep 掩码）。
    ⚠ 踩坑 ①：互近邻图里有度 0 的孤立点，直接 1/√D 出 inf → 特征向量 NaN → k-means 抛 ValueError，
      而且报错发生在 eigh/k-means 里，看不出是「数据里有两个没人理的片区」。
    """
    keep = W.sum(axis=1) > 0
    Wk = W[np.ix_(keep, keep)]
    deg = Wk.sum(axis=1)
    dinv = 1.0 / np.sqrt(deg)
    return np.eye(int(keep.sum())) - (dinv[:, None] * Wk * dinv[None, :]), keep


def spectral_ncut(W, k, seed=7):
    """返回 labels（全 N 长，孤立点为 -1）、本征值、keep 掩码。"""
    Lsym, keep = normalized_laplacian(W)
    vals, vecs = eigh(Lsym)
    U = vecs[:, :k]
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    lab_k = KMeans(n_clusters=k, n_init=100, random_state=seed).fit(Un).labels_
    labels = np.full(N, -1, dtype=int)
    labels[keep] = lab_k
    return labels, vals


def ncut_value(W, labels):
    """真实 Ncut 目标：Σ_c cut(A_c, 其余) / vol(A_c)。（每片只与「其他片」求和，不重复计；−1 不算片）"""
    deg = W.sum(axis=1)
    tot = 0.0
    for c in [int(x) for x in np.unique(labels) if x >= 0]:
        A = labels == c
        tot += W[np.ix_(A, ~A)].sum() / max(deg[A].sum(), 1e-12)
    return float(tot)


def cross_edges(W, km, labels):
    """被切掉的**图中边**（不是所有两点对）——地图上看得见的「桥」。"""
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0 and labels[i] != labels[j]:
                out.append((round(float(km[i, j]), 3), NAMES[i], NAMES[j]))
    return sorted(out)


def power_iteration_fiedler(W, iters=3000):
    """手写幂迭代：求 D^{-1/2}WD^{-1/2} 的第二大特征向量（Fiedler 向量），与 Day63 同根。"""
    deg = W.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    M = dinv[:, None] * W * dinv[None, :]
    v = np.random.default_rng(0).normal(size=N)
    v -= v.mean()
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = M @ v
        w -= v * (v @ w) + np.full(N, np.ones(N) @ w / N)   # 与 v、与常向量正交
        v = w / np.linalg.norm(w)
    return v


def relabel_by_pop(labels, k):
    """簇号按老人数降序重排（只为读图；ARI 不受影响）。−1（未入图）保持 −1。"""
    sums = [POP[labels == c].sum() for c in range(k)]
    order = np.argsort(-np.array(sums))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.array([remap[int(x)] if x >= 0 else -1 for x in labels])


def geometry_cut(valid):
    """
    「两刀切地图」：沿江主轴按 x 分三段（河西 / 过江带 / 河东），城郊安置点按地理就近挂到相邻段。
    这是民政局最自然的手工划法，站数正好 3。invalid 的点（图上孤立、未入图）保持 -1。
    """
    lab = np.zeros(N, dtype=int)
    lab[XY[:, 0] < 1.4] = 0
    lab[(XY[:, 0] >= 1.4) & (XY[:, 0] < 4.0)] = 1
    lab[XY[:, 0] >= 4.0] = 2
    lab = np.array([l if valid[i] else -1 for i, l in enumerate(lab)])
    return lab


def main():
    out = {}
    km = pair_dist(XY)

    W_mut_full = build_graph(km, K_NB, "mutual")
    W_probe = W_mut_full   # 诊断用：保留度 0 点（只用于「连通性 / 过江桥」两节）
    W_mut = build_graph(km, K_NB, "mutual", drop_isolated=True)
    W_one = build_graph(km, K_NB, "one")
    W_rbf = build_graph(km, K_NB, "rbf")

    # ── ① 连通性与零特征值 ──────────────────────────────────────────────────
    print("=" * 80)
    print("一、连通分量数 ≡ 归一化拉普拉斯零特征值个数")
    print("=" * 80)
    conn = {}
    for name, W in (("mutual kNN(k=5)", W_mut_full), ("one-sided kNN(k=5)", W_one), ("RBF 全连接", W_rbf)):
        Lsym, _ = normalized_laplacian(W)
        # 全矩阵（保留度 0 点）取本征值：孤立点对角归一化后「矩阵行 = 0」→ 本征值 1，
        # 这样「零本征值个数 = 多点连通块数」才干净；BFS 的块数还要再加上单点块，见下方断言。
        d_all = np.sqrt(np.maximum(W.sum(1), 1e-12))
        eig = np.linalg.eigvalsh(np.eye(N) - (W / d_all[:, None] / d_all[None, :]))
        nz = int(np.sum(eig < 1e-8))
        comps = bfs_components(W)
        # 多点块（≥2 点）与单点块分开数：单点的本征值恰好是 1，且它「在哪一片」图上无人可问
        big = sorted([len(c) for c in comps if len(c) > 1], reverse=True)
        singles = [c[0] for c in comps if len(c) == 1]
        conn[name] = nz
        print(f"  {name:<20} 边={int((W > 0).sum() // 2):>3}  零本征值={nz}  BFS 块={len(comps)}"
              f"（多点块 {len(big)}，单点 {len(singles)}）  最小本征值 {np.round(eig[:4], 4)}")
        print(f"      多点块大小 {big}" + (f"；单点 {[NAMES[i] for i in singles]}" if singles else ""))
    iso = [i for i in range(N) if W_mut_full[i].sum() == 0]
    print(f"  互近邻图孤立点（度 0）：{[NAMES[i] for i in iso]} —— 它们在图上没有邻居，切到哪一片都只是就近安排。")
    print("  本讲口径：把这两个点**剔出图**再切（N_eff = %d），切完再按「地理最近」挂到片上并单独点名。" % (N - len(iso)))

    # 探针：互近邻图里 河西↔河东 的边（「过江桥」）
    bridges = [(NAMES[i], NAMES[j], round(float(km[i, j]), 2)) for i in range(N) for j in range(i + 1, N)
               if W_mut_full[i, j] > 0 and (XY[i, 0] < 3.3) != (XY[j, 0] < 3.3)]
    print(f"  互近邻图里真正的「过江桥」只有 {len(bridges)} 条：{bridges}")
    print("  → 在互近邻口径下，河西与河东「几乎」是两块（只靠 1 条桥相连）。")
    print("    但谱聚类取 3 片时，它并没有沿这段江切 —— 原因在第三节的代价对比里。")

    # ── ② 主线 Ncut ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"二、Ncut 切 {K_STATIONS} 片（mutual kNN, k={K_NB}）")
    print("=" * 80)
    lab, vals = spectral_ncut(W_mut, K_STATIONS)
    lab = relabel_by_pop(lab, K_STATIONS)
    ncut_main = ncut_value(W_mut, lab)
    for c in range(K_STATIONS):
        print(f"  片{c+1}: " + " / ".join(n for n, l in zip(NAMES, lab) if l == c))
    print(f"  （未入图：{[NAMES[i] for i in np.nonzero(lab < 0)[0]]}）")
    print(f"  Ncut = {ncut_main:.4f}（Ncut 值越小越好）")
    print("  跨片的「桥」：")
    for d, a, b in cross_edges(W_mut, km, lab):
        xa = XY[NAMES.index(a), 0]
        xb = XY[NAMES.index(b), 0]
        tag = "  ← 过江" if (xa < 3.3) != (xb < 3.3) else ""
        print(f"    {a} — {b}   直线 {d:.2f} km{tag}")

    # ── ③ 对照：两刀切地图 ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("三、同一个目标函数当裁判：Ncut vs 两刀切地图")
    print("=" * 80)
    lab_geo_eff = geometry_cut(lab >= 0)
    ncut_geo = ncut_value(W_mut, lab_geo_eff)
    cross_geo = [(d, a, b) for d, a, b in cross_edges(W_mut, km, lab_geo_eff)
                 if (XY[NAMES.index(a), 0] < 3.3) != (XY[NAMES.index(b), 0] < 3.3)]
    pieces = [[NAMES[i] for i in np.nonzero(lab_geo_eff == c)[0]] for c in range(K_STATIONS)]
    print(f"  两刀切地图 Ncut = {ncut_geo:.4f}   各片：{pieces}")
    print(f"  Ncut 谱聚类 Ncut = {ncut_main:.4f}  →  {ncut_main / ncut_geo - 1:+.2%}（同一批入图点、同一张图上比）")
    print(f"  两刀切地图切断了 {len(cross_edges(W_mut, km, lab_geo_eff))} 条图中边，其中过江的 {len(cross_geo)} 条：")
    for d, a, b in cross_geo:
        print(f"    {a} — {b}   直线 {d:.2f} km")

    print("  两刀切地图把河西老城又按 x=1.4 劈了一刀（县医院一带归了「过江带」），这正是它贵的地方。")
    print(f"  ARI(Ncut, 两刀切) = {adjusted_rand_score(lab, lab_geo_eff):.4f}（差别集中在「过江带」怎么分）")

    # ── ④ 本征间隙 ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("四、本征间隙能不能替你定 k / 定站数？")
    print("=" * 80)
    gaps = [(i + 1, float(vals[i + 1] - vals[i])) for i in range(8)]
    kgap, gmax = max(gaps, key=lambda t: t[1])
    print(f"  最小 8 个本征值：{np.round(vals[:8], 4)}")
    print(f"  相邻差：{[(i, round(g, 4)) for i, g in gaps[:5]]}")
    print(f"  最大间隙在 {kgap}→{kgap+1} ⇒ 间隙判据说 k={kgap}；业务站数是 {K_STATIONS}")
    print("  → 间隙回答「图里有几块天然不连通」，不回答「我要几个站」。")
    print("    补救：把两个出图点算作两个单点块，则 2（多点块）+ 2（单点块）= 4 > 3，间隙这条路在本例彻底用不上。")

    # ── ⑤ 稳健性 1：k × 半径 ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("五、稳不稳（1）：近邻数 k × 半径截止 双向扫描")
    print("=" * 80)
    radii = [1.8, 2.0, np.inf]
    grid = []
    for kk in (5, 6, 7):
        row = []
        for r in radii:
            Wg = build_graph(km, kk, "mutual", r, drop_isolated=True)
            lg, _ = spectral_ncut(Wg, K_STATIONS)
            row.append(adjusted_rand_score(lab, lg))
        grid.append(row)
        print(f"  k={kk}:  " + "   ".join(f"r={'∞' if np.isinf(r) else r}:{a:.3f}" for r, a in zip(radii, row)))
    grid = np.array(grid)
    stable_k = float(grid[0, -1])     # 主线档 k=5、无半径（本讲口径）— 基线
    ari_k7 = float(np.mean(grid[2]))  # k=7 档平均 ARI —— 「结论有多宽」的诚实交代

    print()
    print("  坐标 ±60 m 抖动 100 次（k=5，无半径）：")
    rng = np.random.default_rng(20261127)
    jit = []
    for _ in range(100):
        kmj = pair_dist(XY + rng.normal(0, 0.06, size=XY.shape))
        Wj = build_graph(kmj, K_NB, "mutual", drop_isolated=True)
        lj, _ = spectral_ncut(Wj, K_STATIONS)
        jit.append(adjusted_rand_score(lab, lj))
    jit = np.array(jit)
    print(f"    ARI 中位 {np.median(jit):.3f} / 最小 {jit.min():.3f} / 完全一致 {int((jit > 0.999).sum())}/100")

    # ── ⑥ 稳健性 2：把图换掉 ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("六、稳不稳（2）：把「谁是邻居」换掉，片还一样吗")
    print("=" * 80)
    lab_rbf, _ = spectral_ncut(W_rbf, K_STATIONS)
    lab_one, _ = spectral_ncut(W_one, K_STATIONS)
    ari_rbf = adjusted_rand_score(lab, lab_rbf)     # 两者都全 N 长；-1 视为一类
    ari_one = adjusted_rand_score(lab, lab_one)
    print(f"  RBF 全连接   Ncut={ncut_value(W_rbf, lab_rbf):.4f}  ARI(vs 主线)={ari_rbf:.4f}  "
          f"片大小={[int((lab_rbf == c).sum()) for c in range(K_STATIONS)]}")
    print(f"  单向 kNN     Ncut={ncut_value(W_one, lab_one):.4f}  ARI(vs 主线)={ari_one:.4f}  "
          f"片大小={[int((lab_one == c).sum()) for c in range(K_STATIONS)]}")
    print("  → 主线结论的强度来自「图稠密、可断裂」这个假设；换成全连接图，两片就不在了。")

    # ── ⑦ 别的算法同一批坐标 ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("七、同一批坐标，KMeans / Ward 怎么切（只看欧氏空间，不看图）")
    print("=" * 80)
    km_lab = KMeans(K_STATIONS, n_init=100, random_state=0).fit(XY).labels_
    ward_lab = AgglomerativeClustering(n_clusters=K_STATIONS, linkage="ward").fit(XY).labels_
    for nm, l in (("KMeans", km_lab), ("Ward", ward_lab)):
        sizes = [int((l == c).sum()) for c in range(K_STATIONS)]
        single = [NAMES[i] for c in range(K_STATIONS) if (l == c).sum() == 1 for i in np.nonzero(l == c)[0]]
        nomerit = [c for c in range(K_STATIONS) if W_mut[l == c].sum() == 0]   # 该片全由孤立点组成
        nc = ncut_value(W_mut, l)
        print(f"  {nm:<7} 片大小={sizes}  单点片={single}")
        if nomerit:
            print(f"           ⚠ 片 {nomerit} 里的点在图上是孤立点（不打任何边）→ Ncut 记为 0/0，"
                  f"**直接比 Ncut({nc:.4f}) 必然是假的好看**")
        print(f"          silhouette={silhouette_score(XY, l):.4f}  ARI(vs 主线)={adjusted_rand_score(lab, l):.4f}")
    print("  → 欧氏聚类把「地理上贵但图上不相干」的南坡安置切成了单点片；")
    print("    Ncut 给的是「串门关系上的自然断层」，不是「地图上的圆」。")

    # ── ⑧ Fiedler 手写幂迭代 ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("八、手写幂迭代（Fiedler 向量，与 Day63 谱二分对照）")
    print("=" * 80)
    fied_full = power_iteration_fiedler(W_mut)
    fied = np.delete(fied_full, iso)
    names_eff = [n for i, n in enumerate(NAMES) if i not in iso]
    print(f"  符号二分片大小：{[int((fied > 0).sum()), int((fied <= 0).sum())]}")
    print(f"  最负端 {names_eff[int(np.argmin(fied))]} / 最正端 {names_eff[int(np.argmax(fied))]}")
    print("  → 一刀只能两片；要 3 片就在 Fiedler 之后继续切，这就是 Ncut 取多个最小特征向量的意义。")

    # ── ⑨ 片 → 站 ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("九、把片翻译成站")
    print("=" * 80)
    for c in range(K_STATIONS):
        m = lab == c
        cx, cy = np.average(XY[m], axis=0, weights=POP[m])
        print(f"  片{c+1}：老人 {int(POP[m].sum()):>5} 人 / {int(m.sum())} 个片区 / "
              f"重心 ({cx:.2f}, {cy:.2f})")

    out.update({
        "districts": [{"name": n, "x": x, "y": y, "pop": int(p)} for n, x, y, p in DISTRICTS],
        "k_nb": K_NB, "k_stations": K_STATIONS,
        "zero_eig_mutual": conn["mutual kNN(k=5)"],
        "zero_eig_oneside": conn["one-sided kNN(k=5)"],
        "zero_eig_rbf": conn["RBF 全连接"],
        "isolated": [NAMES[i] for i in iso],
        "labels_ncut": {n: int(l) for n, l in zip(NAMES, lab)},
        "ncut_main": ncut_main, "ncut_geom": ncut_geo,
        "kgap_suggest": kgap, "small_eigs": [float(v) for v in vals[:8]],
        "stable_k_radius_min_ari": stable_k, "ari_k7_mean": ari_k7,
        "jitter_ari_median": float(np.median(jit)), "jitter_ari_min": float(jit.min()),
        "ari_rbf": float(ari_rbf), "ari_oneside": float(ari_one),
        "ari_geo": float(adjusted_rand_score(lab, lab_geo_eff)),
        "ari_kmeans": float(adjusted_rand_score(lab, km_lab)),
        "ari_ward": float(adjusted_rand_score(lab, ward_lab)),
        "ncut_kmeans_on_graph": ncut_value(W_mut, km_lab),
        "run_seconds": round(time.time() - T0, 2),
    })

    n_big = sum(1 for c in bfs_components(W_mut_full) if len(c) > 1)
    assert conn["mutual kNN(k=5)"] == n_big == 1, "零本征值个数须等于「多点连通块数」（=1）"
    assert len(bfs_components(W_mut_full)) == 3, "BFS 块数 = 1 个 16 点块 + 2 个单点块"
    assert conn["one-sided kNN(k=5)"] == 1, "单向 kNN 会把孤立点接上，图变连通"
    assert len(iso) == 2, "北岭/南坡两个安置点应当是孤立点"
    assert ncut_main < ncut_geo, "Ncut 必须优于两刀切地图"
    assert len(cross_geo) <= 1, "两刀切地图基本沿着过江桥走 —— 教育点：手工切法能躲过唯一的桥"
    assert kgap != K_STATIONS, "教学点：本征间隙给的 k 与站数不一致"
    assert stable_k > 0.999, "主线档（k=5，无半径）应当就是基线 ARI=1"
    assert ari_k7 < 0.85, "k=7 档结论应当明显改变（结论的适用范围要诚实交代）"
    assert ari_rbf < 0.7, "RBF 全连接图的划分应当与主线明显不同"
    print("\n断言全过。")

    # ── 图 ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8))

    def draw_graph(ax, W, labels, title, annotate_all=True):
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > 0:
                    ax.plot([XY[i, 0], XY[j, 0]], [XY[i, 1], XY[j, 1]], color="#cbd5e0", lw=0.8, zorder=1)
        for c in range(K_STATIONS):
            m = labels == c
            ax.scatter(XY[m, 0], XY[m, 1], s=POP[m] / 4, color=COLORS[c], alpha=0.85,
                       edgecolor="white", linewidth=1.2, zorder=3, label=f"片{c+1}")
        for i in range(N):
            if annotate_all or W[i].sum() > 0:
                ax.annotate(NAMES[i], XY[i], fontsize=8, ha="center", va="bottom", xytext=(0, 4),
                            textcoords="offset points", zorder=4)
        ax.axvline(3.3, color="#4a5568", ls="--", lw=1.0, alpha=0.6)
        ax.text(3.33, -3.0, "临江", fontsize=8, color="#4a5568")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("东西向 km")
        ax.set_aspect("equal")

    # 图①：主线
    draw_graph(axes[0], W_mut, lab,
               f"① mutual kNN(k={K_NB}) 图上的 Ncut 切片\nNcut={ncut_main:.3f}（气泡=老人数；虚线='临江'）")
    axes[0].set_ylabel("南北向 km")
    axes[0].legend(loc="lower right", fontsize=7)

    # 图②：特征值谱 + 间隙
    ax = axes[1]
    idx = np.arange(1, 9)
    ax.plot(idx, vals[:8], "o-", color="#2b6cb0", label="本征值")
    for i in range(1, 6):
        ax.annotate("", xy=(i + 1, vals[i]), xytext=(i, vals[i - 1]),
                    arrowprops=dict(arrowstyle="<->", color="#2f855a" if i == kgap else "#a0aec0", lw=1.4))
    ax.set_xlabel("第几个最小本征值")
    ax.set_ylabel("归一化拉普拉斯本征值")
    ax.set_title(f"② 本征谱：前 2 个≈0（两片）\n最大间隙在 {kgap}→{kgap+1}，判据说 k={kgap}，站要 {K_STATIONS}",
                 fontsize=10)
    ax.set_ylim(-0.05, vals[6] + 0.1)

    # 图③：RBF 对照
    draw_graph(axes[2], W_rbf, lab_rbf,
               f"③ 同一批坐标换 RBF 全连接图\n与① ARI={ari_rbf:.3f}")
    fig.tight_layout()
    fig.savefig(HERE / "day89_spectral_ncut.png", dpi=150)
    print(f"\n图已写入 {HERE / 'day89_spectral_ncut.png'}")

    (HERE / "day89_spectral_ncut_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    out_run = round(time.time() - T0, 2)
    print(f"json 已写入 {HERE / 'day89_spectral_ncut_result.json'}（用时 {out_run} s）")


if __name__ == "__main__":
    main()
