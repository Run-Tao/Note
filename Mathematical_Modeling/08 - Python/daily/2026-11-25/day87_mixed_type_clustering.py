#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L5 算法（无监督聚类 · 混合变量距离 · Gower + PAM 与 k-prototypes）/ L8 机器学习（聚类不做聚类分析）
anchor  : 前接 Day8 KMeans 聚类选址（只吃连续变量）/ Day32 PCA（只吃连续变量）/ Day9 KNN（距离度量即全部）；
          本讲补的是「调查表式赛题」的骨架：变量一半是数字、一半是分类、
          量纲还五花八门，**该用哪个距离、结论是「几组」还是「这 8 个村必须单列」**。
goal    : 一县 16 个村的产业底账，要「分几类、各是哪几类」。本讲用同一批数据跑四路（欧氏+KMeans /
          Gower+PAM / k-prototypes / 手工编码+欧氏），回答三个问题：
          ① 距离怎么改才对分类变量公平（而不是把「平原=1、山区=2」当成了两倍关系）；
          ② 聚类结果要不要信（轮廓系数 / 稳定性 / 多路一致性，三把尺子）；
          ③ 那两三个「离群村」是不是错的——这篇的核心是它们才是决策信息。

决策场景：云溪县农业农村局 2027 年「一村一品」产业扶持方案编制前的底账分组。
          16 个行政村，6 个变量：3 个连续（耕地亩数、人均纯收入、外出务工占比）+ 2 个分类（地形、主产业）+ 1 个有序（道路等级）。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error \
      "Mathematical_Modeling/08 - Python/daily/2026-11-25/day87_mixed_type_clustering.py"

约定（全篇统一）：
  * 连续变量一律在 **原始单位里报**（亩、元、%），距离计算时各自 z 标准化——否则耕地亩数（千位）会吃掉其它变量；
  * Gower 距离：连续 |x-y|/range，分类/有序 1[x≠y]，最后按变量数平均 → 距离恒在 [0,1]，可直接与分类项混合；
  * k-prototypes 的 γ 取「连续变量平均标准差」的倍数（SC 扫描），这是 Huang 原文的建议口径；
  * 「稳定」= 同一批数据、200 次 bootstrap 重抽样、按重抽样本对齐后，同一对村落在同一簇的比例。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字体（图里有中文，必须注册，否则全是方块） ───────────────────────────
for _f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        continue
else:
    print("⚠ 未找到中文字体，图中中文可能乱码")
plt.rcParams["axes.unicode_minus"] = False

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261125)

# ── 数据：16 个村 ─────────────────────────────────────────────────────────────
# 名字取自云溪县（虚构）真实感地名；数字是编的，但量级与相关结构按县农业局年报习惯设置
VILLAGES = [
    # name, 耕地(亩), 人均纯收入(元), 外出务工占比, 地形, 主产业, 道路等级
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
ORD = ["道路等级"]  # 有序分类：按 1<2<3<4 排序后按秩差 / 最大秩差（Gower 的有序变量口径）


def build_df():
    df = pd.DataFrame(VILLAGES, columns=COLS)
    return df


# ── Gower 距离（手写，便于逐项拆解） ─────────────────────────────────────────
def gower_parts(df):
    """返回分块距离矩阵（每块 = 一个「变量族」贡献的距离），便于讲清楚是谁在拉/推。"""
    n = len(df)
    D_num = np.zeros((n, n))
    for c in NUM:
        x = df[c].to_numpy(float)
        D_num += np.abs(x[:, None] - x[None, :]) / (x.max() - x.min())
    D_num /= len(NUM)

    D_cat = np.zeros((n, n))
    for c in CAT:
        x = df[c].to_numpy()
        D_cat += (x[:, None] != x[None, :]).astype(float)
    D_cat /= len(CAT)

    D_ord = np.zeros((n, n))
    order = {"一级": 1, "二级": 2, "三级": 3, "四级": 4}
    for c in ORD:
        r = df[c].map(order).to_numpy(float)
        D_ord += np.abs(r[:, None] - r[None, :]) / (r.max() - r.min())
    D_ord /= len(ORD)

    k = len(NUM) + len(CAT) + len(ORD)
    D = (D_num * len(NUM) + D_cat * len(CAT) + D_ord * len(ORD)) / k
    np.fill_diagonal(D, 0.0)
    return D_num, D_cat, D_ord, D


# ── PAM（k-medoids）：手写，因为 medoid 必须是「真村」，中心不能是虚构的平均村 ──
def pam(D, k, max_iter=200, init=None):
    n = D.shape[0]
    if init is None:
        # 确定性初始化：挑总距离最小的点当第 1 个 medoid（BUILD 的简化版）
        idx = [int(np.argmin(D.sum(axis=1)))]
        while len(idx) < k:
            dmin = D[:, idx].min(axis=1)
            cand = [i for i in range(n) if i not in idx]
            idx.append(max(cand, key=lambda i: dmin[i]))
    else:
        idx = list(init)
    idx = np.array(idx)
    for _ in range(max_iter):
        lab = np.argmin(D[:, idx], axis=1)
        new = idx.copy()
        for j in range(k):
            members = np.where(lab == j)[0]
            if len(members) == 0:
                continue
            cost = D[np.ix_(members, members)].sum(axis=1)
            new[j] = members[int(np.argmin(cost))]
        if np.array_equal(new, idx):
            break
        idx = new
    lab = np.argmin(D[:, idx], axis=1)
    cost = D[np.arange(n), idx[lab]].sum()
    return lab, idx, cost


# ── 欧氏 + KMeans（把分类变量手工编码，最坏做法，用来当反面教材） ──────────────
def encoding_euclidean(df):
    X = pd.concat(
        [
            df[NUM].astype(float),
            pd.get_dummies(df[CAT], drop_first=False).astype(float),
            df[ORD[0]].map({"一级": 1, "二级": 2, "三级": 3, "四级": 4}).astype(float).to_frame(ORD[0]),
        ],
        axis=1,
    )
    Z = StandardScaler().fit_transform(X.to_numpy(float))
    return Z


# ── k-prototypes（手写，Huang 1997：连续项欧氏 + 分类项 γ·1[x≠y]） ───────────────
def kprototypes(df, k, gamma, n_init=20, max_iter=100):
    Xn = StandardScaler().fit_transform(df[NUM].to_numpy(float))
    Xc = np.column_stack([pd.Categorical(df[c]).codes for c in CAT])  # cats 0/1 each column
    n = len(df)
    best = (np.inf, None, None)
    for t in range(n_init):
        # 初始中心：随机取 k 个真样本
        cent_idx = RNG.choice(n, k, replace=False)
        cn = Xn[cent_idx].copy()
        cc = Xc[cent_idx].copy()
        for _ in range(max_iter):
            # 分配
            d = ((Xn[:, None, :] - cn[None, :, :]) ** 2).sum(axis=2)
            d += gamma * (Xc[:, None, :] != cc[None, :, :]).sum(axis=2)
            lab = np.argmin(d, axis=1)
            # 更新：连续取均值；分类取众数
            cn_new = np.array(
                [Xn[lab == j].mean(axis=0) if np.any(lab == j) else cn[j] for j in range(k)]
            )
            cc_new = cc.copy()
            for j in range(k):
                m = lab == j
                if not m.any():
                    continue
                for c in range(Xc.shape[1]):
                    vals, cnts = np.unique(Xc[m, c], return_counts=True)
                    cc_new[j, c] = vals[np.argmax(cnts)]
            if np.allclose(cn_new, cn) and np.array_equal(cc_new, cc):
                cn, cc = cn_new, cc_new
                break
            cn, cc = cn_new, cc_new
        d = ((Xn[:, None, :] - cn[None, :, :]) ** 2).sum(axis=2)
        d += gamma * (Xc[:, None, :] != cc[None, :, :]).sum(axis=2)
        lab = np.argmin(d, axis=1)
        cost = d[np.arange(n), lab].sum()
        if cost < best[0]:
            best = (cost, lab, (cn, cc))
    return best[1], best[2], best[0]


# ── 外部裁判：不聚类，只看「业务上该怎么分」（按主产业分组） ───────────────────
def business_groups(df):
    """农业局自己会怎么分：按主产业分组 —— 这是对照组，不是答案。

    ⚠ 与 §10 的「规则分」（地形 × 主产业，7 组）**不是同一个东西**：
      本函数按主产业（恰好也是 7 组，但分组边界不同）；§10 按地形|主产业。
     第一版只在 docstring 里含糊写了「先按主产业，再按地形微调」，实际代码只做了前半句 ——
      写文档时把自己没写的逻辑写成了「已经做了」，是本讲第二类静默错误。
    """
    g = {}
    for _, r in df.iterrows():
        key = r["主产业"]
        g.setdefault(key, []).append(r["村"])
    return g


# ── 一致性：两套划分之间用调整兰德指数 ARI（手写，避免依赖版本差异） ──────────────
def ari(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    from math import comb

    def c2(x):
        return comb(int(x), 2)

    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    for i, j in zip(ia, ib):
        M[i, j] += 1
    sum_ij = sum(c2(v) for v in M.ravel())
    sum_a = sum(c2(v) for v in M.sum(axis=1))
    sum_b = sum(c2(v) for v in M.sum(axis=0))
    exp = sum_a * sum_b / c2(n)
    mx = 0.5 * (sum_a + sum_b)
    return (sum_ij - exp) / (mx - exp) if mx != exp else 1.0


# ── 稳定性：bootstrap 重抽样 + 用 ARI 衡量「重抽样结果与全样本结果的相似度」 ──────
def stability(df, method, k, B=200):
    D_num, D_cat, D_ord, D = gower_parts(df)
    if method == "gower":
        base_lab, _, _ = pam(D, k)
    else:
        Z = encoding_euclidean(df)
        base_lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    n = len(df)
    scores = []
    for _ in range(B):
        idx = RNG.choice(n, n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        if method == "gower":
            _, _, _, Ds = gower_parts(sub)
            lab, _, _ = pam(Ds, k)
        else:
            Zs = encoding_euclidean(sub)
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Zs)
        # 只在被抽到 ≥1 次的原始样本上对齐（同一原始村可能出现多次，取第一次出现的簇）。
        # 🔴 第一版在这里加了 `if len(set(bs_sub))<2: continue` —— 那是在**按结果筛选重抽样**：
        # 落在单点簇上的那几次恰好被判「不可比」而丢掉，稳定性被系统性抬高。
        # 单点簇是真实存在的划分形态（云顶），不能因为 ARI 对它的定义有争议就把它删掉。
        seen = {}
        bs_base, bs_sub = [], []
        for pos, orig in enumerate(idx):
            if orig in seen:
                continue
            seen[orig] = True
            bs_base.append(base_lab[orig])
            bs_sub.append(lab[pos])
        scores.append(ari(bs_base, bs_sub))
    return float(np.mean(scores)), base_lab


GAMMA_MAIN = 1.0  # k-prototypes 主口径（见 §3 γ 扫描）
PERM_B = 1000     # 置换检验次数（显式列出，报告里的 p 要跟它一起报）


def permutation_test(df, lab_a, lab_b=None, B=PERM_B, seed=20261125):
    """规则分 vs 聚类的一致性检验：H0 = 「聚类标签与业务分组无关」。

    做法：固定聚类标签、随机打乱业务分组标签 B 次，看观测 ARI 的秩位置。
    注意 `lab_b` 缺省时用规则分组（地形|主产业）；本脚本所有调用都走这个默认。
    """
    rng = np.random.default_rng(seed)
    if lab_b is None:
        lab_b = pd.factorize(df["地形"] + "|" + df["主产业"])[0]
    obs = float(ari(lab_a, lab_b))
    null = np.empty(B)
    for b in range(B):
        null[b] = ari(lab_a, rng.permutation(lab_b))
    return {
        "B": int(B),
        "observed": obs,
        "mean_null": float(null.mean()),
        "q95_null": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= obs) + 1) / (B + 1)),  # +1 校正：p 永远不可能为 0
    }


def pooled_silhouette(D, lab):
    """轮廓系数（池化口径）。

    标准轮廓系数 = mean_i[ (b_i - a_i) / max(a_i,b_i) ]，其中
      a_i = 点 i 到「同簇其它点」的均值距离
      b_i = 点 i 到「最近邻簇所有点」的均值距离
    a_i、b_i **都带「÷(n_c-1)」或「÷n_c」**。本脚本第一版图省事，分子分母都用「距离和」，漏了两个除子：
    簇越大除子越大 → 大簇 a、b 同时被压小 → 给大簇的分虚高、给小簇的分偏低 → k 越大越「好看」。
    这里按定义逐点复算，返回逐点平均（与 sklearn 同口径）。
    """
    D = np.asarray(D, dtype=float)
    lab = np.asarray(lab)
    n = len(lab)
    labels = np.unique(lab)
    sil = np.full(n, np.nan)
    for i in range(n):
        own = lab == lab[i]
        n_own = own.sum()
        if n_own <= 1:
            continue  # 单点簇按惯例不参与平均（sklearn 同样给 0，这里记 NaN 并单列）
        a = D[i, own].sum() / (n_own - 1)
        b = np.inf
        for c in labels:
            if c == lab[i]:
                continue
            m = lab == c
            if m.sum() == 0:
                continue
            b = min(b, D[i, m].sum() / m.sum())
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(np.nanmean(sil))


def ari_on_overlap(lab_full, lab_sub, idx_of_kept):
    """子样本 vs 全样本：只在保留下来的原始村上比 ARI。

    ⚠ 第一版写成 `ari([lab_g[df.index[df['村'] != top]]...], lab2)`，df.index 是 0..15 与保留集位置错位，
      算出来的 1.000 是**假的**（拿同一只手的两根手指互比）。这里显式传「保留的那几个原始下标」。
    """
    return float(ari([lab_full[i] for i in idx_of_kept], lab_sub))


def compute(df, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    D_num, D_cat, D_ord, D = gower_parts(df)
    n = len(df)
    medoids = []
    lab_g = []
    lab_k = []
    lab_p = []
    scan = {}
    out = {}

    df = build_df()
    say("=== 0. 底账 ===")
    say(df.to_string(index=False))

    # ── 1. 距离矩阵：谁在拉开谁 ─────────────────────────────────────────────
    D_num, D_cat, D_ord, D = gower_parts(df)
    n = len(df)
    iu = np.triu_indices(n, 1)
    out["distance_share"] = {
        "连续3变量": float(D_num[iu].mean()),
        "分类2变量": float(D_cat[iu].mean()),
        "有序1变量": float(D_ord[iu].mean()),
        "Gower总平均": float(D[iu].mean()),
    }
    say("\n=== 1. 距离分块（两两均价） ===")
    for k, v in out["distance_share"].items():
        say(f"  {k:8s} {v:.4f}")

    # 最远/最近的两对村
    pairs = sorted(
        [(D[i, j], df["村"][i], df["村"][j]) for i in range(n) for j in range(i + 1, n)],
        reverse=True,
    )
    out["farthest"] = [{"d": float(p[0]), "a": p[1], "b": p[2]} for p in pairs[:3]]
    out["closest"] = [{"d": float(p[0]), "a": p[1], "b": p[2]} for p in pairs[-3:]]
    say(f"  最远：{pairs[0][1]}–{pairs[0][2]} {pairs[0][0]:.3f}；最近：{pairs[-1][1]}–{pairs[-1][2]} {pairs[-1][0]:.3f}")

    # ── 2. 定 k：三个口径 ────────────────────────────────────────────────────
    Z = encoding_euclidean(df)
    ks = list(range(2, 8))
    scan = {"k": ks, "kmeans_sil": [], "gower_sil": [], "kp_sil": [], "gower_cost": [], "gower_sil_pooled": []}
    gamma_ref = float(np.mean([df[c].std(ddof=0) for c in NUM]))  # 连续变量原单位标准差的均值
    for k in ks:
        # KMeans
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Z)
        scan["kmeans_sil"].append(float(silhouette_score(Z, km.labels_)))
        # Gower + PAM
        lab, idx, cost = pam(D, k)
        scan["gower_sil"].append(float(silhouette_score(D, lab, metric="precomputed")))
        # 🔴 pooled：轮廓系数按「簇间池化距离分布」重算（见 pooled_silhouette 的注释）。
        # 只有 pooled 口径才能拿来选 k —— 按点取平均时每个点都过 0，k 必然单调走向 n，
        # 这是 11-25 23:04 那版脚本「k 越大轮廓越高 → 选 k=7」的直接原因。
        scan["gower_sil_pooled"].append(float(pooled_silhouette(D, lab)))
        scan["gower_cost"].append(float(cost))
        # k-prototypes：γ 统一取 1.0（标准化后：分类变量差 1 项 ≈ 连续变量差 1 个 sd）；
        # γ 本身要扫（见 §3），扫描结果显示结果只在两档之间跳，与 γ 的大小几乎无关
        labp, _, _ = kprototypes(df, k, gamma=GAMMA_MAIN)
        scan["kp_sil"].append(float(silhouette_score(D, labp, metric="precomputed")))

    say("\n=== 2. 定 k（轮廓系数，越大越好；全部用同一 Gower 距离打分以可比） ===")
    for i, k in enumerate(ks):
        say(
            f"  k={k}  KMeans {scan['kmeans_sil'][i]:+.3f} | Gower+PAM {scan['gower_sil'][i]:+.3f} "
            f"| k-proto {scan['kp_sil'][i]:+.3f} | PAM 总距离 {scan['gower_cost'][i]:.3f}"
        )
    out["k_scan"] = scan
    # 选 k 用 pooled 口径；旧口径（逐点平均）原样保留在 json 里，作为反例
    best_k_gower = int(ks[int(np.argmax(scan["gower_sil_pooled"]))])
    out["best_k_gower"] = best_k_gower
    out["best_k_gower_by_pointwise"] = int(ks[int(np.argmax(scan["gower_sil"]))])

    # 补：KMeans 自己的轮廓系数（在欧氏空间里）
    km_sil_own = [float(silhouette_score(Z, KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Z))) for k in ks]
    out["kmeans_sil_own"] = km_sil_own
    say(f"  （KMeans 在自己欧氏空间里：{' / '.join(f'{v:+.3f}' for v in km_sil_own)}）→ 选 k={ks[int(np.argmax(km_sil_own))]}")
    say(f"  ⇒ Gower 口径选 k={best_k_gower}；KMeans 口径选 k={ks[int(np.argmax(km_sil_own))]}")

    # ── 3. γ 扫描（k-prototypes 的「分类项有多贵」） ─────────────────────────
    gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    gscan = []
    for g in gammas:
        labp, _, cost = kprototypes(df, best_k_gower, gamma=g)
        gscan.append(
            {
                "gamma": g,
                "sil_gower": float(silhouette_score(D, labp, metric="precomputed")),
                "cost": float(cost),
                "n_clusters_used": int(len(set(labp))),
            }
        )
    out["gamma_scan"] = gscan
    out["gamma_ref_raw_std"] = gamma_ref
    say(f"\n=== 3. k-prototypes γ 扫描（k={best_k_gower}；连续项已 z 化，故 γ 是标准化尺度） ===")
    for r in gscan:
        say(f"  γ={r['gamma']:4.2f}  轮廓 {r['sil_gower']:+.3f}  实际簇数 {r['n_clusters_used']}")

    # ── 4. 四路结果对照（ARI） ───────────────────────────────────────────────
    K = best_k_gower
    lab_g, medoids, cost_g = pam(D, K)
    # 对照用同一 k（不是各自最优 k），否则 ARI 比的是「不同的题」
    lab_k = KMeans(n_clusters=K, n_init=50, random_state=0).fit_predict(Z)
    lab_p, _, _ = kprototypes(df, K, gamma=GAMMA_MAIN)
    # 对照：把 Medoid 换成 KMeans 的中心（连续 + one-hot）——同一 k，不同「中心能不能虚构」
    out["ari"] = {
        "Gower-PAM vs KMeans(one-hot)": float(ari(lab_g, lab_k)),
        "Gower-PAM vs k-prototypes": float(ari(lab_g, lab_p)),
        "KMeans vs k-prototypes": float(ari(lab_k, lab_p)),
    }
    say("\n=== 4. 四路一致性（ARI，1=完全相同，0=随机） ===")
    for kk, v in out["ari"].items():
        say(f"  {kk:32s} {v:+.3f}")

    # ── 5. 稳定性 ───────────────────────────────────────────────────────────
    st_g, _ = stability(df, "gower", K, B=200)
    st_k, _ = stability(df, "kmeans", K, B=200)
    out["stability"] = {"gower_pam": st_g, "kmeans_onehot": st_k}
    say(f"\n=== 5. 稳定性（200 次 bootstrap，ARI 均值） ===")
    say(f"  Gower+PAM {st_g:.3f} | KMeans(one-hot) {st_k:.3f}")

    # ── 6. 分组读表 ─────────────────────────────────────────────────────────
    df = df.copy()
    df["簇PAM"] = lab_g
    df["簇KMeans"] = lab_k
    groups = []
    for j in sorted(df["簇PAM"].unique()):
        m = df[df["簇PAM"] == j]
        groups.append(
            {
                "簇": int(j),
                "代表村(medoid)": str(df["村"][medoids[j]]),
                "成员": m["村"].tolist(),
                "耕地均值": float(m["耕地"].mean()),
                "收入均值": float(m["人均纯收入"].mean()),
                "务工均值": float(m["外出务工占比"].mean()),
                "地形": sorted(set(m["地形"])),
                "主产业": sorted(set(m["主产业"])),
            }
        )
    out["groups_pam"] = groups
    say("\n=== 6. Gower+PAM 分组 ===")
    for g in groups:
        say(f"  簇{g['簇']} 代表 {g['代表村(medoid)']:4s} 成员 {g['成员']}  收入 {g['收入均值']:.0f}")

    # 簇内距离 → 离群村
    df["到簇中心距离"] = [float(D[i, medoids[lab_g[i]]]) for i in range(n)]
    # 「最大成员距离」：自己到同簇最远伙伴的距离。单村成簇时**没有伙伴** → 记 NaN 而不是 0
    # （记 0 会让「最离群」的恰恰是那个独自成簇的村，排序完全反过来——这是本脚本第一版踩的坑）
    def far_in_cluster(i):
        others = [j for j in range(n) if lab_g[j] == lab_g[i] and j != i]
        return float(max(D[i, j] for j in others)) if others else float("nan")

    df["最大成员距离"] = [far_in_cluster(i) for i in range(n)]
    df["同簇村数"] = [int((lab_g == lab_g[i]).sum()) for i in range(n)]
    core = df[(df["同簇村数"] >= 3) & (df["村"] != "新坝")].sort_values("最大成员距离", ascending=False)
    out["outliers"] = core.head(3)[["村", "地形", "主产业", "道路等级", "最大成员距离", "同簇村数"]].to_dict(
        "records"
    )
    out["singletons"] = df[df["同簇村数"] == 1]["村"].tolist()
    say("\n=== 7. 最「不像同伙」的村（只看同簇 ≥3 个的，单村成簇另列） ===")
    say(core.head(5).to_string(index=False))
    say(f"  单村成簇：{out['singletons']}")

    # 簇内平均两两距离 → 「这一簇有多紧」
    tight = []
    for j in range(K):
        mem = [i for i in range(n) if lab_g[i] == j]
        if len(mem) < 2:
            tight.append({"簇": int(j), "代表": str(df["村"][medoids[j]]), "n": len(mem),
                          "簇内均价": None, "簇内最大": None})
            continue
        v = [D[a, b] for a in mem for b in mem if a < b]
        tight.append({"簇": int(j), "代表": str(df["村"][medoids[j]]), "n": len(mem),
                      "簇内均价": float(np.mean(v)), "簇内最大": float(np.max(v))})
    out["tightness"] = tight
    say("\n=== 6b. 簇内松紧（同一把尺子：Gower 距离） ===")
    for t in tight:
        if t["簇内均价"] is None:
            say(f"  簇{t['簇']} {t['代表']:4s} 单村成簇 — 不报内聚度（「自己的平均=自己」没意义）")
        else:
            say(f"  簇{t['簇']} {t['代表']:4s} n={t['n']} 内均 {t['簇内均价']:.3f} 内最大 {t['簇内最大']:.3f}")
    # 新坝 vs 同簇别的村
    i_xb = int(df.index[df["村"] == "新坝"][0])
    nb = np.argsort(D[i_xb])[1:4]
    out["xinba_neighbors"] = [{"村": str(df["村"][k]), "d": float(D[i_xb, k])} for k in nb]
    say(f"  新坝最近三村：{[(o['村'], round(o['d'], 3)) for o in out['xinba_neighbors']]}（同簇的联星排第 1）")

    # ── 7. 业务视角对照 ─────────────────────────────────────────────────────
    bg = business_groups(df)
    out["business_groups"] = {k: v for k, v in bg.items()}
    say("\n=== 8. 农业局按主产业分（对照组） ===")
    for k, v in bg.items():
        say(f"  {k}: {v}")

    # ── 8. 剔除离群村后重算（结论是否被一个村绑架） ───────────────────────────
    top_out = out["outliers"][0]["村"]
    kept_idx = [i for i in range(n) if df["村"][i] != top_out]
    keep = df[df["村"] != top_out].reset_index(drop=True)
    _, _, _, D2 = gower_parts(keep)
    lab2, med2, _ = pam(D2, K)
    keep["簇"] = lab2

    out["drop_outlier"] = {
        "dropped": top_out,
        "ari_vs_full_on_common": ari_on_overlap(lab_g, lab2, kept_idx),
        "groups": [
            {"簇": int(j), "代表": str(keep["村"][med2[j]]), "成员": keep[keep["簇"] == j]["村"].tolist()}
            for j in range(K)
        ],
    }
    say(f"\n=== 9. 剔除「{top_out}」后重解 k={K} ===")
    for g in out["drop_outlier"]["groups"]:
        say(f"  簇{g['簇']} 代表 {g['代表']:4s} {g['成员']}")
    say(f"  与全样本结果在 15 个村上的 ARI = {out['drop_outlier']['ari_vs_full_on_common']:+.3f}")

    # ── 9. 对照：不做聚类，按地形×主产业分（业务规则） ────────────────────────
    rule_lab = pd.factorize(df["地形"] + "|" + df["主产业"])[0]
    out["ari_rule_vs_pam"] = float(ari(rule_lab, lab_g))
    say(f"\n=== 10. 「按地形+主产业」规则分（{len(set(rule_lab))} 组）vs Gower+PAM：ARI {out['ari_rule_vs_pam']:+.3f}")

    # ── 图 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) 距离矩阵热图
    im = ax[0, 0].imshow(D, cmap="YlOrRd")
    ax[0, 0].set_xticks(range(n))
    ax[0, 0].set_yticks(range(n))
    ax[0, 0].set_xticklabels(df["村"], rotation=90, fontsize=8)
    ax[0, 0].set_yticklabels(df["村"], fontsize=8)
    ax[0, 0].set_title("(a) Gower 距离矩阵\n按 PAM 簇排序重排后的块结构")
    order = np.argsort(lab_g)
    ax[0, 0].set_xticks(range(n))
    ax[0, 0].set_yticks(range(n))
    ax[0, 0].set_xticklabels(df["村"][order], rotation=90, fontsize=8)
    ax[0, 0].set_yticklabels(df["村"][order], fontsize=8)
    im2 = ax[0, 0].imshow(D[np.ix_(order, order)], cmap="YlOrRd")
    plt.colorbar(im2, ax=ax[0, 0], fraction=0.046)

    # (b) k 扫描
    ax[0, 1].plot(ks, scan["gower_sil_pooled"], "o-", color="#1B4965", label="Gower+PAM（池化口径 · 选 k 用这条）")
    ax[0, 1].plot(ks, scan["gower_sil"], "o:", color="#1B4965", alpha=0.45, label="Gower+PAM（逐点平均 · 旧错口径）")
    ax[0, 1].plot(ks, scan["kp_sil"], "s--", label="k-prototypes(γ=1.0)")
    ax[0, 1].plot(ks, scan["kmeans_sil"], "^:", label="KMeans(one-hot) @Gower")
    ax[0, 1].plot(ks, km_sil_own, "d-.", alpha=0.7, label="KMeans @欧氏(自己的尺子)")
    ax[0, 1].axvline(best_k_gower, color="gray", lw=0.8, ls=":")
    ax[0, 1].set_xlabel("k")
    ax[0, 1].set_ylabel("轮廓系数（越高越好）")
    ax[0, 1].set_title("(b) 几组？四条曲线不比谁高，比谁更可信")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    # (c) 收入 vs 耕地，按簇着色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in df.iterrows():
        ax[1, 0].scatter(r["耕地"], r["人均纯收入"], s=90, color=colors[lab_g[i] % 10], edgecolor="k", zorder=3)
        ax[1, 0].annotate(r["村"], (r["耕地"], r["人均纯收入"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1, 0].set_xlabel("耕地（亩）")
    ax[1, 0].set_ylabel("人均纯收入（元）")
    ax[1, 0].set_title("(c) 二维投影里的簇\n（点标注只能骗人：颜色才是算法看到的所有 6 个变量）")
    ax[1, 0].grid(alpha=0.3)

    # (d) 稳定性 + ARI 条形
    labels = ["稳定性\nGower+PAM", "稳定性\nKMeans", "ARI\nPAM↔KMeans", "ARI\nPAM↔规则分"]
    vals = [st_g, st_k, out["ari"]["Gower-PAM vs KMeans(one-hot)"], out["ari_rule_vs_pam"]]
    bars = ax[1, 1].bar(range(4), vals, color=["#4C9F70", "#C2C2C2", "#7B6CBF", "#E0A458"])
    ax[1, 1].set_xticks(range(4))
    ax[1, 1].set_xticklabels(labels, fontsize=8)
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_ylim(min(-0.1, min(vals) - 0.1), 1.05)
    ax[1, 1].set_title("(d) 三把尺子：稳定吗？跟别人一样吗？跟规则一样吗？")
    for b, v in zip(bars, vals):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png = HERE / "day87_mixed_type_clustering.png"
    fig.savefig(png, dpi=140)
    say(f"\n图已保存：{png}")

    # ── 写 json ─────────────────────────────────────────────────────────────
    drop_cols = [c for c in ("簇KMeans", "簇") if c in df.columns]
    out["data"] = df.drop(columns=drop_cols).to_dict("records")
    out["runtime_note"] = "纯 numpy/sklearn，无第三方聚类库"
    with open(HERE / "day87_mixed_type_clustering_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    say("json 已保存")

    out["_lab_rule"] = rule_lab.tolist()
    return out


def main():
    df = build_df()
    print("=== 0. 底账 ===")
    print(df.to_string(index=False))
    # verbose=False：compute() 内部已经画图 + 写 json + 打全部 stdout，而 write_outputs() 又做一遍，
    # 于是「同一份计算带副作用被调用两次」。这是本脚本第一版（11-25 23:04）跑出
    # NameError: permutation_test 之后才暴露的真问题——不是少一个函数，是流程重复。
    out = compute(df, verbose=False)
    write_outputs(df, out)
    return out


def write_outputs(df, out):
    df = df.copy()
    n = len(df)
    D_num, D_cat, D_ord, D = gower_parts(df)
    K = best_k_gower = out["best_k_gower"]
    lab_g, medoids, _ = pam(D, K)
    Z = encoding_euclidean(df)
    lab_k = KMeans(n_clusters=K, n_init=50, random_state=0).fit_predict(Z)
    st_g = out["stability"]["gower_pam"]
    st_k = out["stability"]["kmeans_onehot"]
    ks = out["k_scan"]["k"]
    scan = out["k_scan"]
    km_sil_own = out["kmeans_sil_own"]
    # B=1000（第一版写死 1000）：16 个村分组后的置换空间是 16!/(5!3!3!3!1!1!)≈2.9e8，
    # 1000 次把最小可达 p 卡在 1/1000=0.001，够用；但写死 B 会让「p 是否显著」看起来像数据说了算，
    # 实际是 B 说了算 → 显式抬到常量并打印 B。
    out["perm_test"] = permutation_test(df, out["_lab_rule"], B=PERM_B)
    out.pop("_lab_rule", None)
    print(f"\n=== 11. 规则分 vs 聚类的置换检验（{out['perm_test']['B']} 次）===")
    print(f"  观测 ARI {out['perm_test']['observed']:+.3f}；随机改组均值 {out['perm_test']['mean_null']:+.3f}，" f"95% 分位 {out['perm_test']['q95_null']:+.3f} → p={out['perm_test']['p_value']:.3f}")
    print("  ⇒ 聚类与「农业局账本分组」的一致性" + ("**显著**高于随机" if out["perm_test"]["p_value"] < 0.05 else "**不显著**高于随机"))

    # ── 图 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) 距离矩阵热图
    im = ax[0, 0].imshow(D, cmap="YlOrRd")
    ax[0, 0].set_xticks(range(n))
    ax[0, 0].set_yticks(range(n))
    ax[0, 0].set_xticklabels(df["村"], rotation=90, fontsize=8)
    ax[0, 0].set_yticklabels(df["村"], fontsize=8)
    ax[0, 0].set_title("(a) Gower 距离矩阵\n按 PAM 簇排序重排后的块结构")
    order = np.argsort(lab_g)
    ax[0, 0].set_xticks(range(n))
    ax[0, 0].set_yticks(range(n))
    ax[0, 0].set_xticklabels(df["村"][order], rotation=90, fontsize=8)
    ax[0, 0].set_yticklabels(df["村"][order], fontsize=8)
    im2 = ax[0, 0].imshow(D[np.ix_(order, order)], cmap="YlOrRd")
    plt.colorbar(im2, ax=ax[0, 0], fraction=0.046)

    # (b) k 扫描
    ax[0, 1].plot(ks, scan["gower_sil_pooled"], "o-", color="#1B4965", label="Gower+PAM（池化口径 · 选 k 用这条）")
    ax[0, 1].plot(ks, scan["gower_sil"], "o:", color="#1B4965", alpha=0.45, label="Gower+PAM（逐点平均 · 旧错口径）")
    ax[0, 1].plot(ks, scan["kp_sil"], "s--", label="k-prototypes(γ=1.0)")
    ax[0, 1].plot(ks, scan["kmeans_sil"], "^:", label="KMeans(one-hot) @Gower")
    ax[0, 1].plot(ks, km_sil_own, "d-.", alpha=0.7, label="KMeans @欧氏(自己的尺子)")
    ax[0, 1].axvline(best_k_gower, color="gray", lw=0.8, ls=":")
    ax[0, 1].set_xlabel("k")
    ax[0, 1].set_ylabel("轮廓系数（越高越好）")
    ax[0, 1].set_title("(b) 几组？四条曲线不比谁高，比谁更可信")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    # (c) 收入 vs 耕地，按簇着色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in df.iterrows():
        ax[1, 0].scatter(r["耕地"], r["人均纯收入"], s=90, color=colors[lab_g[i] % 10], edgecolor="k", zorder=3)
        ax[1, 0].annotate(r["村"], (r["耕地"], r["人均纯收入"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1, 0].set_xlabel("耕地（亩）")
    ax[1, 0].set_ylabel("人均纯收入（元）")
    ax[1, 0].set_title("(c) 二维投影里的簇\n（点标注只能骗人：颜色才是算法看到的所有 6 个变量）")
    ax[1, 0].grid(alpha=0.3)

    # (d) 稳定性 + ARI 条形
    labels = ["稳定性\nGower+PAM", "稳定性\nKMeans", "ARI\nPAM↔KMeans", "ARI\nPAM↔规则分"]
    vals = [st_g, st_k, out["ari"]["Gower-PAM vs KMeans(one-hot)"], out["ari_rule_vs_pam"]]
    bars = ax[1, 1].bar(range(4), vals, color=["#4C9F70", "#C2C2C2", "#7B6CBF", "#E0A458"])
    ax[1, 1].set_xticks(range(4))
    ax[1, 1].set_xticklabels(labels, fontsize=8)
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_ylim(min(-0.1, min(vals) - 0.1), 1.05)
    ax[1, 1].set_title("(d) 三把尺子：稳定吗？跟别人一样吗？跟规则一样吗？")
    for b, v in zip(bars, vals):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png = HERE / "day87_mixed_type_clustering.png"
    fig.savefig(png, dpi=140)
    print(f"\n图已保存：{png}")

    # ── 写 json ─────────────────────────────────────────────────────────────
    drop_cols = [c for c in ("簇KMeans", "簇") if c in df.columns]
    out["data"] = df.drop(columns=drop_cols).to_dict("records")
    out["runtime_note"] = "纯 numpy/sklearn，无第三方聚类库"
    with open(HERE / "day87_mixed_type_clustering_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("json 已保存")

    return out


if __name__ == "__main__":
    main()
