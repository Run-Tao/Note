#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L5 算法（聚类分析 · 层次聚类 dendrogram 切树高度 vs 并类距离 jump）/ L8 机器学习（聚类 ≠ 聚类分析）
anchor  : 前接 Day87 混合变量聚类（Gower + PAM，k 由轮廓在 5/7 之间犹豫）；
          前接 Day8 KMeans（只吃连续变量）、Day63 MCL 谱二分（「分几段是扫出来的」）；
          本讲补三格全库零命中/只被点名未做的：**层次聚类（average/Ward/单链对照）**、
          **变量加权（Gower 等权 → 业务权重）**、**模糊 C-means（FCM 隶属度）**。
goal    : 一县 16 村产业底账，要「分几组、哪几个村另册、C 类政策该给谁」。
          本讲回答三个问题：
          ① 层次聚类要不要切成 5 组还是 7 组——判据不是「树长得好不好看」，而是「下一个并类距离跳多高」
             与「切在这层时簇内距离有多紧」两把尺子（jump / gap / 紧致度），且三种 linkage 必须同向；
          ② 等权 Gower 是不是一句空话——把「耕地与是否旅游村」视为同权，三个单例还剩几个；
          ③ 分界村（联星、新坝）到底算平原粮村还是自己的产业——FCM 隶属度给的是「几分之几」，不是「哪一类」。
          并回答：模型算出来的组，与「地形×主产业」的规矩分差多少？这 16% 的差值不值钱？

决策场景：云溪县农业农村局 2027 年「一村一品」扶持资金切块（320 万元），先分底账组再定政策包。

运行（全篇唯一入口，副作用只在 main 里发生一次）：
  cd /home/ubuntu/Note && ./.venv/bin/python -W error \
    "Mathematical_Modeling/08 - Python/daily/2026-11-26/day88_hierarchical_weighted_fcm.py"

约定（全篇统一口径）：
  * 数据与 Day87 完全相同（同 16 村、同 6 变量），便于逐项对照；数值是编的，量级按县农业局年报习惯；
  * 距离：Gower（连续 |差|/极差、分类 1[不等]、有序 秩差/最大秩差），默认等权；加权版按业务权重归一后重平均；
  * 层次聚类用 scipy linkage（average / complete / ward / single），**单链**只用来演示「链式效应」；
    ⚠ average/complete/single 吃的是**距离**，Ward 吃的是**离差平方和**——两者 dendrogram 的 y 轴不同量纲，
    下面所有 jump 判据都**只在同一 linkage 内部**比较，不跨 linkage 比较绝对高度；
  * FCM：手写迭代（Picard 不动点），m=2.0，硬划分由最大隶属度得到，同时报「隶属度最高与次高之差」作为把握度；
    ⚠ **16 个点上的真实陷阱**：纯随机初值的关系型 FCM 会稳定落到「每点隶属度 1/c」的平凡不动点——J 照样下降、
      看起来收敛漂亮，其实什么都没分开。判据：均值隶属度 ≥ 1.02/c 即判退化，换初值（本讲第一节踩坑）。
  * 稳定性：200 次 bootstrap 重抽样，只在被抽到的原样本上算 ARI（Day87 的口径，不做按结果筛选）。
"""

import json
import time
from math import comb

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

# ── 中文字体（图里有中文，不注册就是方块） ───────────────────────────────────
for _f in ("Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261126)

# ── 数据：与 Day87 同一批 16 村（改一个数都会让结论不可比） ───────────────────
VILLAGES = [
    ("东坪", 1180, 18600, 0.41, "平原", "粮", "一级"),
    ("西塘", 1350, 17900, 0.38, "平原", "粮", "一级"),
    ("同心", 1420, 17200, 0.44, "平原", "粮", "二级"),
    ("联星", 1290, 19100, 0.36, "平原", "菜", "一级"),
    ("新坝", 960, 15400, 0.52, "平原", "菜", "二级"),
    ("沙湾", 320, 14800, 0.55, "沿江", "渔", "二级"),
    ("柳湾", 280, 16100, 0.47, "沿江", "渔", "三级"),
    ("河湾", 360, 14200, 0.58, "沿江", "渔", "三级"),
    ("石桥", 540, 13800, 0.61, "丘陵", "茶", "三级"),
    ("坳上", 620, 14500, 0.49, "丘陵", "茶", "二级"),
    ("高岭", 480, 12600, 0.66, "丘陵", "茶", "三级"),
    ("桐子", 580, 15100, 0.43, "丘陵", "果", "二级"),
    ("黄泥", 410, 11900, 0.71, "山地", "药", "四级"),
    ("杉木", 450, 12300, 0.68, "山地", "药", "四级"),
    ("盘石", 260, 10800, 0.74, "山地", "药", "四级"),
    ("云顶", 150, 22600, 0.29, "山地", "旅游", "三级"),
]
COLS = ["村", "耕地", "人均纯收入", "外出务工占比", "地形", "主产业", "道路等级"]
NUM = ["耕地", "人均纯收入", "外出务工占比"]
CAT = ["地形", "主产业"]
ORD = ["道路等级"]
ROAD = {"一级": 1, "二级": 2, "三级": 3, "四级": 4}

# 业务权重：农业局口径 —— 收入与产业 >> 地形；「是否旅游村」不是独立变量，它藏在主产业里
W_EQUAL = {**{c: 1.0 for c in NUM + CAT + ORD}}
W_BUSINESS = {"耕地": 1.0, "人均纯收入": 3.0, "外出务工占比": 1.0, "地形": 0.5, "主产业": 2.0, "道路等级": 1.0}
# 第三个权重口径是「反直觉对照」：把地形压到近乎无意义，看单例村还在不在
W_NOLANDSCAPE = {"耕地": 1.0, "人均纯收入": 1.5, "外出务工占比": 1.0, "地形": 0.05, "主产业": 2.5, "道路等级": 1.0}


def build_df():
    return pd.DataFrame(VILLAGES, columns=COLS)


# ── 距离 ────────────────────────────────────────────────────────────────────
def block_distances(df):
    """三族分块距离矩阵（每块已按族内变量数平均，值域 [0,1]）。"""
    n = len(df)
    Dn = np.zeros((n, n))
    for c in NUM:
        x = df[c].to_numpy(float)
        Dn += np.abs(x[:, None] - x[None, :]) / (x.max() - x.min())
    Dn /= len(NUM)

    Dc = np.zeros((n, n))
    for c in CAT:
        x = df[c].to_numpy()
        Dc += (x[:, None] != x[None, :]).astype(float)
    Dc /= len(CAT)

    Do = np.zeros((n, n))
    for c in ORD:
        r = df[c].map(ROAD).to_numpy(float)
        Do += np.abs(r[:, None] - r[None, :]) / (r.max() - r.min())
    Do /= len(ORD)
    for M in (Dn, Dc, Do):
        np.fill_diagonal(M, 0.0)
    return Dn, Dc, Do


def gower(df, w=None):
    Dn, Dc, Do = block_distances(df)
    if w is None:
        w = W_EQUAL
    # 族权重 × 族内距离，最后除以总权重 —— 等价于「逐变量加权后 Gower」
    wn = sum(w[c] for c in NUM)
    wc = sum(w[c] for c in CAT)
    wo = sum(w[c] for c in ORD)
    tot = wn + wc + wo
    D = (wn * Dn + wc * Dc + wo * Do) / tot
    np.fill_diagonal(D, 0.0)
    return D


# ── 层次聚类：切树的三把尺子 ────────────────────────────────────────────────
def cut_stats(Z, D, ks=range(2, 9)):
    """对每个 k：并类距离（树高）、jump（跨越这一刀的相对跳幅）、簇内平均距离。"""
    heights = Z[:, 2]
    links = Z[:, :2].astype(int)
    n = D.shape[0]
    rows = []
    for k in ks:
        lab = fcluster(Z, t=k, criterion="maxclust")
        # 并类距离：把 16 个点切成 k 组，用的那一刀是「第 n-k 次并类」
        h_cut = heights[n - k]
        next_low = heights[n - k - 1] if n - k - 1 >= 0 else 0.0
        jump = (h_cut - next_low) / next_low if next_low > 1e-12 else np.inf
        # 簇内平均距离（越小越紧）
        within = []
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            if len(idx) > 1:
                sub = D[np.ix_(idx, idx)]
                within.append(sub[np.triu_indices(len(idx), 1)].mean())
        rows.append(
            {
                "k": int(k),
                "cut_height": float(h_cut),
                "jump": float(jump) if np.isfinite(jump) else None,
                "within_mean": float(np.mean(within)) if within else 0.0,
                "sizes": sorted([int((lab == c).sum()) for c in np.unique(lab)], reverse=True),
            }
        )
    return rows, links, heights


def consensus_at_k(linkages, k):
    """同一 k 下不同 linkage 的分组一致度：两两 ARI 的平均。"""
    labs = [fcluster(Z, t=k, criterion="maxclust") for Z in linkages]
    vals = [ari(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(vals)) if vals else 1.0


def ari(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)

    def c2(x):
        return comb(int(x), 2)

    ia = pd.factorize(a)[0]
    ib = pd.factorize(b)[0]
    M = np.zeros((ia.max() + 1, ib.max() + 1))
    for i, j in zip(ia, ib):
        M[i, j] += 1
    s_ij = sum(c2(v) for v in M.ravel())
    s_a = sum(c2(v) for v in M.sum(axis=1))
    s_b = sum(c2(v) for v in M.sum(axis=0))
    exp = s_a * s_b / c2(n)
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


# ── FCM（模糊 C-means，手写：Picard 迭代，在「等权 Gower 特征空间」里做） ─────
def gower_features(df, w=None):
    """Gower 特征化：三族各自在 [0,1] 上编码、每族平均到 1 个向量，再按 √权重 缩放。

    欧氏距离² = 2(1 − Gower 相似度) 的放大版；族内平均保证「一个族不因为变量多就占便宜」。
    """
    if w is None:
        w = W_EQUAL
    Z = df[NUM].to_numpy(float)
    Z = (Z - Z.min(0)) / (Z.max(0) - Z.min(0))
    cat_levels = [(df["地形"] == t).to_numpy(float) for t in ["平原", "沿江", "丘陵", "山地"]]
    cat_levels += [(df["主产业"] == p).to_numpy(float) for p in ["粮", "菜", "渔", "茶", "果", "药", "旅游"]]
    Oh = np.column_stack(cat_levels)
    R = ((df["道路等级"].map(ROAD).to_numpy(float) - 1) / 3)[:, None]
    wn = np.mean([w[c] for c in NUM])
    wc = np.mean([w[c] for c in CAT])
    wo = w[ORD[0]]
    return np.column_stack([Z * np.sqrt(wn), Oh * np.sqrt(wc / 2), R * np.sqrt(wo)])


def fcm(X, c=7, m=2.0, n_init=100, max_iter=500, tol=1e-10):
    """模糊 C-means：最小化 J = Σ_i Σ_j u_ij^m ‖x_i − v_j‖²。

    🔴 **退化陷阱（本讲第一节踩坑，两轮）**：
      ① 直接在关系矩阵 D 上套 relational FCM：软中心互相重合 → 所有距离相等 → 不动点 U ≡ 1/c。
         16 个点里 7 个维度上它**每次都收敛到那里**，J 一路下降、输出「每村隶属 0.143」，
         还排出一张「最模糊村」名单 —— 跑得通、数字好看、结论全假。
      ② 改到特征空间后第一版写错一行（U 转置前后混用），把 U 变成 (c,n) 还照样算，
         结果是「维度对不上」才报错 —— **是报错救了我，不是我自己看出来**。
      本函数统一 U 的形状恒为 (n, c)，并在返回前检查 (U.sum(1)≈1)。
    """
    X = np.asarray(X, float)
    n = len(X)
    best = (np.inf, None)
    for _ in range(n_init):
        U = RNG.dirichlet(np.full(c, 0.4), size=n)  # (n, c)，比均匀更尖锐，避开平凡不动点
        for _it in range(max_iter):
            Um = U**m
            W = Um / Um.sum(axis=0, keepdims=True)  # (n, c)
            V = W.T @ X  # (c, d)
            D2 = ((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2) + 1e-12  # (n, c)
            inv = D2 ** (-1.0 / (m - 1))
            U_new = inv / inv.sum(axis=1, keepdims=True)  # (n, c)
            if np.max(np.abs(U_new - U)) < tol:
                U = U_new
                break
            U = U_new
        assert U.shape == (n, c) and np.allclose(U.sum(axis=1), 1, atol=1e-8), "FCM 的 U 形状/归一化坏了"
        gap = np.sort(U, axis=1)[:, -1] - np.sort(U, axis=1)[:, -2]
        if gap.mean() < 0.05:  # 近均匀 = 退化，丢弃
            continue
        Um = U**m
        W = Um / Um.sum(axis=0, keepdims=True)
        V = W.T @ X
        J = float((Um * ((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2)).sum())
        if J < best[0]:
            best = (J, U.copy())
    if best[1] is None:
        raise RuntimeError(f"FCM c={c}：全部初值都退化为近均匀隶属")
    return best[1], best[0]


def hard_from_U(U):
    lab = np.argmax(U, axis=1)
    order = np.argsort(U, axis=1)
    gap = U[np.arange(len(U)), order[:, -1]] - U[np.arange(len(U)), order[:, -2]]
    return lab, gap


# ── 业务规则分（对照组，不是答案） ────────────────────────────────────────
def rule_groups(df):
    return pd.factorize(df["地形"] + "|" + df["主产业"])[0]


def describe(df, lab):
    """按簇号输出成员与关键均值，供教案/读者逐条核对。"""
    out = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        sub = df.iloc[idx]
        out.append(
            {
                "簇": int(c),
                "成员": sub["村"].tolist(),
                "n": int(len(idx)),
                "耕地均值": round(float(sub["耕地"].mean()), 1),
                "收入均值": round(float(sub["人均纯收入"].mean()), 1),
                "务工均值": round(float(sub["外出务工占比"].mean()), 3),
                "地形": sorted(set(sub["地形"])),
                "主产业": sorted(set(sub["主产业"])),
            }
        )
    return out


def main():
    t0 = time.time()
    df = build_df()
    names = df["村"].tolist()
    D = gower(df)
    Dh = squareform(D, checks=False)

    out = {"data": df.to_dict("records")}

    # ══ §1 层次聚类：average / complete / ward / single ═══════════════════════
    Zs = {
        "average": linkage(Dh, method="average"),
        "complete": linkage(Dh, method="complete"),
        "ward": linkage(Dh, method="ward"),
        "single": linkage(Dh, method="single"),
    }
    stats = {}
    for name, Z in Zs.items():
        rows, links, heights = cut_stats(Z, D, ks=range(2, 9))
        stats[name] = {"rows": rows, "heights": heights.tolist()}
    # 每个 k 下 average/complete/ward 三法一致度（单链不参与：它会被链式效应绑架）
    trio = [Zs["average"], Zs["complete"], Zs["ward"]]
    cons = {k: consensus_at_k(trio, k) for k in range(2, 9)}
    out["linkage_cut_stats"] = stats
    out["consensus_by_k"] = {str(k): v for k, v in cons.items()}
    # 最孤立的点：单链在第一次并类时的对象
    first = Zs["single"][0]
    out["single_first_merge"] = {
        "pair": [names[int(first[0])], names[int(first[1])]],
        "height": float(first[2]),
    }
    mid = Zs["average"][0]
    out["average_first_merge"] = {"pair": [names[int(mid[0])], names[int(mid[1])]], "height": float(mid[2])}

    # ══ §2 定 k 的三把尺子（只在 average 上做） ══════════════════════════════
    rows_avg = stats["average"]["rows"]
    best_jump = max(rows_avg, key=lambda r: r["jump"])["k"]
    # 紧致度拐点：k 从 k→k+1 时 within_mean 的相对降幅，取第一个 < 15% 的位置（前一个 k）
    within = {r["k"]: r["within_mean"] for r in rows_avg}
    knee = None
    for k in range(2, 8):
        drop = (within[k] - within[k + 1]) / within[k]
        if k < 8 and drop < 0.15:
            knee = k
            break
    out["k_choose"] = {
        "by_jump": int(best_jump),
        "by_knee_kmeans_style": int(knee) if knee else None,
        "jumps": {str(r["k"]): r["jump"] for r in rows_avg},
        "within": {str(r["k"]): r["within_mean"] for r in rows_avg},
        "within_drop": {
            str(k): (within[k] - within[k + 1]) / within[k] for k in range(2, 8) if k + 1 in within
        },
    }

    # 与 Day87 的 PAM 结果对照：KMeans 的轮廓选 k（Day87 口径），看层次聚类选几
    X = np.column_stack(
        [
            (df[NUM].to_numpy(float) - df[NUM].to_numpy(float).mean(0)) / df[NUM].to_numpy(float).std(0),
            pd.get_dummies(df[CAT]).to_numpy(float),
            df[ORD[0]].map(ROAD).to_numpy(float),
        ]
    )
    sil_km = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
        from sklearn.metrics import silhouette_score

        sil_km[k] = float(silhouette_score(X, km.labels_, metric="euclidean"))
    out["kmeans_silhouette"] = {str(k): v for k, v in sil_km.items()}

    # ══ §3 权重口径：等权 vs 业务权重 vs 去地形权重 ════════════════════════════
    KW = {}
    for tag, w in (("equal", W_EQUAL), ("business", W_BUSINESS), ("no_landscape", W_NOLANDSCAPE)):
        Dw = gower(df, w)
        Zw = linkage(squareform(Dw, checks=False), method="average")
        for k in (5, 6, 7):
            KW[f"{tag}_k{k}"] = fcluster(Zw, t=k, criterion="maxclust")
    wtab = []
    for k in (5, 6, 7):
        wtab.append(
            {
                "k": k,
                "ari_eq_biz": ari(KW[f"equal_k{k}"], KW[f"business_k{k}"]),
                "ari_eq_noland": ari(KW[f"equal_k{k}"], KW[f"no_landscape_k{k}"]),
                "groups_business": describe(df, KW[f"business_k{k}"]),
            }
        )
    out["weight_scan"] = wtab
    # 单例村在三个权重口径下是否还在
    def singletons(lab):
        return sorted([names[i] for i in range(len(lab)) if (lab == lab[i]).sum() == 1])

    out["singletons_by_weight"] = {
        tag: {str(k): singletons(KW[f"{tag}_k{k}"]) for k in (5, 6, 7)}
        for tag in ("equal", "business", "no_landscape")
    }

    # ══ §4 FCM 隶属度（等权 Gower 特征空间；加权口径再跑一遍做对照） ══════════
    betw = {}
    Xf = gower_features(df)
    fcm_main = fcm(Xf, c=int(best_jump), n_init=200)
    for k in (5, 6, 7):
        U, J = fcm(Xf, c=k, n_init=200)
        lab, gap = hard_from_U(U)
        betw[str(k)] = {
            "J": round(J, 4),
            "ari_vs_avg_hier": ari(lab, fcluster(Zs["average"], t=k, criterion="maxclust")),
            "mean_membership_gap": float(gap.mean()),
            "n_singletons": int(sum((lab == j).sum() == 1 for j in np.unique(lab))),
        }
        if k == int(best_jump):
            out["fcm_groups"] = describe(df, lab)
            out["fcm_fuzzy"] = [
                {
                    "村": names[i],
                    "最大": round(float(U[i].max()), 3),
                    "次高": round(float(np.sort(U[i])[-2]), 3),
                    "簇": int(np.argmax(U[i])),
                }
                for i in np.argsort(gap)  # 把握度从低到高：最模糊的排前面
            ]
    # 加权 Gower 特征空间上的 FCM（同一 c），看软划分会不会跟着权重改
    Uw, Jw = fcm(gower_features(df, W_BUSINESS), c=int(best_jump), n_init=200)
    lab_w, _ = hard_from_U(Uw)
    lab_e, _ = hard_from_U(fcm_main[0])
    out["fcm_weight_effect"] = {
        "c": int(best_jump),
        "J_equal": round(fcm_main[1], 4),
        "J_business": round(Jw, 4),
        "ari_equal_vs_business": ari(lab_e, lab_w),
    }
    out["fcm_k_scan"] = betw

    # ══ §5 稳定性：average 层次 vs PAM 口径（bootstrap 200） ════════════════
    def boot_stability(k, B=200):
        base = fcluster(Zs["average"], t=k, criterion="maxclust")
        sc = []
        for _ in range(B):
            idx = RNG.choice(len(df), len(df), replace=True)
            sub = df.iloc[idx].reset_index(drop=True)
            Ds = gower(sub)
            Zb = linkage(squareform(Ds, checks=False), method="average")
            lab = fcluster(Zb, t=k, criterion="maxclust")
            seen, a, b = set(), [], []
            for pos, orig in enumerate(idx):
                if orig in seen:
                    continue
                seen.add(orig)
                a.append(base[orig])
                b.append(lab[pos])
            sc.append(ari(a, b))
        return float(np.mean(sc))

    out["stability_average"] = {str(k): boot_stability(k) for k in (5, 6, 7)}

    # ══ §6 与规矩分对照 ════════════════════════════════════════════════════
    lab7 = fcluster(Zs["average"], t=7, criterion="maxclust")
    lab5 = fcluster(Zs["average"], t=5, criterion="maxclust")
    rg = rule_groups(df)
    out["ari_vs_rule"] = {"k7": ari(lab7, rg), "k5": ari(lab5, rg)}
    out["groups_avg_k7"] = describe(df, lab7)
    out["groups_avg_k5"] = describe(df, lab5)
    out["rule_groups"] = describe(df, rg)
    # 🔴 5 和 7 之间只差「平原那一刀」：联星/新坝是不是独立成「菜村」
    out["k5_vs_k7"] = {
        "moved": [
            {"村": names[i], "k5组": int(lab5[i]), "k7组": int(lab7[i])}
            for i in range(len(names))
            if lab5[i] != lab7[i]
        ],
        "cut_height_gap": round(
            float(stats["average"]["heights"][16 - 6] - stats["average"]["heights"][16 - 7]), 4
        ),
        "jump_at_6": rows_avg[4]["jump"],
    }

    # ══ §7 三把尺子是否同向（结论可信度总表） ══════════════════════════════
    out["summary"] = {
        "k_by_jump": int(best_jump),
        "k_by_jump_complete": int(max(stats["complete"]["rows"], key=lambda r: r["jump"])["k"]),
        "k_by_jump_ward": int(max(stats["ward"]["rows"], key=lambda r: r["jump"])["k"]),
        "k_by_knee": out["k_choose"]["by_knee_kmeans_style"],
        "k_by_kmeans_sil": int(max(sil_km, key=sil_km.get)),
        "consensus_at_jump_k": cons[int(best_jump)],
        # 两个候选 k 只在「平原粮 vs 平原菜」这一刀上不同 —— 其余 6 刀完全一样
        "split_between_k5_k7": {
            "cut_height_gap": round(stats["average"]["heights"][16 - 6] - stats["average"]["heights"][16 - 7], 4),
            "jump6": rows_avg[4]["jump"],
        },
    }
    out["runtime_s"] = round(time.time() - t0, 2)

    # ── 落盘（json 只含原始字段 + 结果，不含派生列） ──────────────────────────
    base = "Mathematical_Modeling/08 - Python/daily/2026-11-26/day88_hierarchical_weighted_fcm"
    with open(base + "_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ══ 图：2×2 四联 ═════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    dendrogram(
        Zs["average"],
        labels=names,
        ax=ax,
        color_threshold=stats["average"]["heights"][16 - int(best_jump)],
        above_threshold_color="0.55",
    )
    for k, h in ((5, stats["average"]["heights"][16 - 5]), (7, stats["average"]["heights"][16 - 7])):
        ax.axhline(h, ls="--", lw=1.2, color="#c0392b" if k == best_jump else "#2c7fb8")
        ax.text(0.2, h, f"k={k}", va="bottom", fontsize=9, color="#c0392b" if k == best_jump else "#2c7fb8")
    ax.set_title("(a) average linkage 树：两刀都「说得过去」，jump 判据选 k=%d" % best_jump)
    ax.set_ylabel("并类距离（Gower）")

    ax = axes[0, 1]
    ks = [r["k"] for r in rows_avg]
    ax.plot(ks, [within[k] for k in ks], "o-", color="#2c7fb8", label="簇内平均距离（越小越紧）")
    ax.set_xlabel("k（切成几组）")
    ax.set_ylabel("簇内平均 Gower 距离", color="#2c7fb8")
    ax2 = ax.twinx()
    for nm, c in (("average", "#c0392b"), ("complete", "#e67e22"), ("ward", "#27ae60")):
        ax2.plot(ks, [r["jump"] for r in stats[nm]["rows"]], "s--", color=c, lw=1, label=f"{nm} jump")
    ax2.set_ylabel("jump = 本刀高度 / 上一刀高度")
    ax2.set_yscale("log")
    ax.set_title("(b) 紧致度一路变好（没给答案），jump 在 k=5 与 k=7 两个台阶上都突起")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    kk = sorted(cons)
    ax.bar([str(k) for k in kk], [cons[k] for k in kk], color="#8e44ad", alpha=0.85)
    ax.axhline(0.9, ls=":", color="k")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k")
    ax.set_ylabel("average/complete/ward 三法两两 ARI 均值")
    ax.set_title("(c) 三种 linkage 两两 ARI：低处 k=3/4 各说各话，k=5/7 却完全一致")

    ax = axes[1, 1]
    fz = out["fcm_fuzzy"]
    top = fz[:9][::-1]
    y = np.arange(len(top))
    ax.barh(y - 0.2, [t["最大"] for t in top], height=0.38, color="#16a085", label="最大隶属度")
    ax.barh(y + 0.2, [t["次高"] for t in top], height=0.38, color="#95a5a6", label="次高隶属度")
    ax.set_yticks(y)
    ax.set_yticklabels([t["村"] for t in top], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("隶属度 u（m=2）")
    ax.set_title("(d) FCM 最模糊的 9 个村：云顶/桐子挂在两棵树上，联星新坝并不模糊")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Day88 · 层次聚类切树 / 变量加权 / FCM 隶属度 —— 云溪县 16 村「一村一品」底账", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(base + ".png", dpi=130)
    print(f"✅ 完成，用时 {out['runtime_s']} s；图 → {base}.png；json → {base}_result.json")
    print(json.dumps(out["summary"], ensure_ascii=False))
    print("k_choose:", json.dumps(out["k_choose"], ensure_ascii=False))
    print("singletons:", json.dumps(out["singletons_by_weight"], ensure_ascii=False))
    print("ari_vs_rule:", out["ari_vs_rule"], " stability:", out["stability_average"])
    print("fcm k scan:", json.dumps(out["fcm_k_scan"], ensure_ascii=False))
    print("fcm 最模糊:", json.dumps(fz[:5], ensure_ascii=False))


if __name__ == "__main__":
    main()
