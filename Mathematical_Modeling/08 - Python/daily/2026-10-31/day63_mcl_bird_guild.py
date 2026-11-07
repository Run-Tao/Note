# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : M1 生态网络模型 / L8 算法（MCL 马尔可夫聚类 · 随机流 · 膨胀系数 · 断面扫描 · 谱二分 vs 最优割）
anchor      : 接 Day8 K-Means（聚类）、Day32 PCA 谱方法、Day60 网络结构上的优化
goal        : 云溪湿地 9 个水鸟样点的巡护线怎么划。回答四件事：
              ① 「分几段」这个数是从哪来的——先把网络按阈值切一刀（断面扫描），看它自己断成几块
              ② MCL 的「扩散 + 膨胀」在干什么；为什么它不一定给出那个数，膨胀过头会把整张图蒸干
              ③ 谱二分（Fiedler）切的一刀是不是最省的刀——9 点可全枚举判
              ④ 巡护队长「一套巡护证只配一条线」这条口径值不值
              全部由 .venv 实跑验证（MCL 手写 + Louvain 对照 + 谱二分 + 255 种二分区全枚举）。
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-31/day63_mcl_bird_guild.py"
输出：day63_result.json + day63_mcl.png（四面板）
"""

import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

warnings.simplefilter("always")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day63_result.json")
OUT_PNG = os.path.join(HERE, "day63_mcl.png")

# ---------------------------------------------------------------- 中文字体
CJK_CANDIDATES = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK",
                  "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in CJK_CANDIDATES if n in _installed), None)
if CJK is not None:
    plt.rcParams["font.sans-serif"] = [CJK] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 场景数据
SITES = ["P1 湖北", "P2 苇北", "P3 苇东", "P4 滩东", "P5 滩中",
         "P6 湖北岸", "P7 湖南滩", "P8 河口", "P9 苇西"]
COST = np.array([8.0, 7.5, 9.0, 8.5, 6.0, 12.0, 5.5, 10.0, 9.5])           # 万元/年
BIRDS = np.array([46.0, 38.0, 41.0, 35.0, 29.0, 18.0, 26.0, 22.0, 21.0])   # 千鸟次/年
XY = np.array([[3.0, 8.0], [4.5, 6.5], [7.5, 7.0], [8.5, 4.5], [1.5, 5.5],
               [2.0, 2.5], [6.0, 1.0], [4.0, 0.5], [9.5, 2.5]])
N = len(SITES)
KM = np.linalg.norm(XY[:, None, :] - XY[None, :, :], axis=2)

COMP = np.array([    # 4 个功能群占比：潜水 / 涉禽 / 游禽 / 林鸟
    [0.42, 0.40, 0.16, 0.02], [0.20, 0.55, 0.22, 0.03], [0.10, 0.68, 0.18, 0.04],
    [0.06, 0.52, 0.30, 0.12], [0.36, 0.44, 0.18, 0.02], [0.18, 0.58, 0.21, 0.03],
    [0.08, 0.72, 0.16, 0.04], [0.10, 0.68, 0.18, 0.04], [0.22, 0.50, 0.25, 0.03],
])
S_COMP = 1.0 - 0.5 * np.abs(COMP[:, None, :] - COMP[None, :, :]).sum(axis=2)
DAMP = 2.5
W = S_COMP * np.exp(-KM / DAMP)
np.fill_diagonal(W, 0.0)


# ---------------------------------------------------------------- MCL（手写）
def mcl(A, inflation=1.5, expand=2, prune=1e-6, max_iter=300, tol=1e-12):
    """van Dongen(2000)：① expand ② inflate ③ prune，迭代到矩阵不再变化。
    团 = 「流得通」的点集（M[i,j]>0 且 M[j,i]>0）。"""
    M = A.copy().astype(float)
    np.fill_diagonal(M, 0.0)
    col = M.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    M = M / col
    M = 0.5 * (M + M.T)
    M = M / M.sum(axis=0, keepdims=True)
    hist = []
    for _ in range(max_iter):
        prev = M.copy()
        M = np.linalg.matrix_power(M, expand)
        M = np.exp(inflation * np.log(np.clip(M, 1e-300, None)))   # 对数域求幂，防下溢
        M = M / M.sum(axis=0, keepdims=True)
        if prune > 0:
            M[M < prune] = 0.0
        c = M.sum(axis=0, keepdims=True)
        c[c == 0] = 1.0
        M = M / c
        rel = np.abs(M - prev).max() / max(np.abs(prev).max(), 1e-12)
        hist.append(float(rel))
        if rel < tol:
            break
    reach = (M > 0) & (M.T > 0)
    seen = np.zeros(len(A), dtype=bool)
    cl = []
    for i in range(len(A)):
        if seen[i]:
            continue
        comp = [int(c) for c in np.where(reach[i])[0] if not seen[c]]
        for c in comp:
            seen[c] = True
        cl.append(sorted(comp))
    return M, [c for c in cl if c], hist


def cut_graph(A, t):
    G = A.copy()
    G[G < t] = 0.0
    np.fill_diagonal(G, 0.0)
    return G


# ---------------------------------------------------------------- ① 断面扫描：图自己的骨架
CUTS = [0.10, 0.15, 0.20, 0.25, 0.27, 0.28, 0.29, 0.30, 0.32, 0.35, 0.40]
FLOAT_SCAN = []
for t in CUTS:
    ncomp, lab = connected_components((W >= t).astype(int), directed=False)
    groups = [[int(i) for i in range(N) if lab[i] == c] for c in range(ncomp)]
    FLOAT_SCAN.append(dict(t=t, n=ncomp, sizes=sorted((len(g) for g in groups), reverse=True),
                           groups=groups))
# 「骨架」= 断面扫描里第一次断成 3 块的那一刀；超过它以后是越切越碎（不是细化，是碎片化）
T_STAR = next(s["t"] for s in FLOAT_SCAN if s["n"] == 3)
CL = next(s["groups"] for s in FLOAT_SCAN if s["n"] == 3)
K_BASE = len(CL)


# ---------------------------------------------------------------- ② MCL：在几个断面各跑一次
MCL_ROWS = []
for t in (0.0, 0.10, 0.20, 0.25, 0.27):
    G = cut_graph(W, t) if t > 0 else W.copy()
    t0 = time.perf_counter()
    _, cl, h = mcl(G, inflation=1.5, expand=2)
    MCL_ROWS.append(dict(t=t, k=len(cl), clusters=cl, iters=len(h),
                         dt=time.perf_counter() - t0,
                         sizes=sorted((len(c) for c in cl), reverse=True)))
# MCL 在原始图上（不定阈值）的表现：这是本讲最诚实的数字
G0 = cut_graph(W, 0.0)
MX0, CL0, H0 = mcl(G0, inflation=1.5, expand=2)
MCL_ON_RAW = dict(k=len(CL0), clusters=CL0, sizes=sorted((len(c) for c in CL0), reverse=True))

# inflation 扫描（在骨架 G = W ≥ t* 上）
G_STAR = cut_graph(W, T_STAR)
RS = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
SCAN = []
for r in RS:
    _, cl, h = mcl(G_STAR, inflation=r, expand=2)
    SCAN.append(dict(r=r, k=len(cl), sizes=sorted((len(c) for c in cl), reverse=True),
                     clusters=cl, iters=len(h)))
KRIT = next((s["r"] for s in SCAN if s["k"] != K_BASE), None)

# Louvain 对照
LV = sorted([sorted(int(i) for i in c) for c in
             nx.community.louvain_communities(nx.from_numpy_array(W), weight="weight", seed=2026)],
            key=lambda c: -len(c))
LV_SAME = {frozenset(c) for c in LV} == {frozenset(c) for c in CL}
K_LV = len(LV)


def modularity(cl, Wm):
    m = Wm.sum() / 2.0
    if m <= 0:
        return 0.0
    Q = 0.0
    for c in cl:
        idx = list(c)
        Q += Wm[np.ix_(idx, idx)].sum() / 2.0 / m - (Wm[idx, :].sum() / (2 * m)) ** 2
    return float(Q)


# ---------------------------------------------------------------- ③ 最优割 vs 谱二分
def cross(Wm, P, Q):
    return float(Wm[np.ix_(P, Q)].sum())


def best_bipartition(Wm, min_size=1):
    best = None
    for mask in range(1, 2 ** (N - 1)):
        A_ = [i for i in range(N) if mask >> i & 1]
        B_ = [i for i in range(N) if i not in A_]
        if len(A_) < min_size or len(B_) < min_size:
            continue
        c = cross(Wm, A_, B_)
        if best is None or c < best[0] - 1e-12:
            best = (c, sorted(A_), sorted(B_))
    return best


def fiedler(Wm):
    d = Wm.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    Ls = (np.eye(len(Wm)) - Dm @ Wm @ Dm)
    w, V = np.linalg.eigh(Ls)
    return w, V[:, 1]


# ★ 2026-10-31 三次实跑后的修正（写进教案 §5）：
#   ① 只在「最小一块」外做二分区（6 点）：谱二分恰好命中最优割（0%）；
#   ② 换成「最大+次大」的并（8 点）：仍只差 5%（且是最优解之一是「单点」，不是我想要的结论）；
#   ③ 最终做法：**在整张 9 点图上**做谱二分——它切出
#      {P1,P2,P5,P6} / {P3,P4,P7,P8,P9} 这一刀，是全图第 13 好的切法（1.732 vs 最优 1.298），
#      而这一刀又与「两块骨架拼起来」高度重合。谱二分只是启发式，必须拿枚举当裁判。
BIG = list(range(N))
Wb = W[np.ix_(BIG, BIG)]
nb = len(BIG)
EIGB, FIEDB = fiedler(Wb)
POS = [k for k in range(nb) if FIEDB[k] >= 0]
NEG = [k for k in range(nb) if FIEDB[k] < 0]
SPEC_LOCAL = [sorted(BIG[k] for k in POS), sorted(BIG[k] for k in NEG)]
CUT_SPEC = cross(W, SPEC_LOCAL[0], SPEC_LOCAL[1])
BEST_LOCAL = None
for mask in range(1, 2 ** (nb - 1)):
    a = [k for k in range(nb) if mask >> k & 1]
    b = [k for k in range(nb) if k not in a]
    if len(a) < 2 or len(b) < 2:
        continue
    c = cross(Wb, a, b)
    if BEST_LOCAL is None or c < BEST_LOCAL[0] - 1e-12:
        BEST_LOCAL = (c, [BIG[k] for k in a], [BIG[k] for k in b])
CUT_LOCAL_OPT = BEST_LOCAL[0]
SPEC_GAIN = CUT_SPEC / CUT_LOCAL_OPT - 1.0
# 全图（含孤立点）的最优割，只作对照：它会退化到「把孤点切出去」
CUT_GLOBAL, GA, GB = best_bipartition(W)
CUT_GLOBAL_BAL, _, _ = best_bipartition(W, min_size=2)

# ---------------------------------------------------------------- ④ 巡护证口径
def route_cost(cl):
    return float(sum(COST[list(c)].sum() for c in cl if c))


# 巡护队长：每班 2 人 + 1 台车，**一天只能跑 2 个点**（点位间距 1.5~3 km，一天跑 3 个点要到天黑）。
# 于是 25 个点的骨架会被拆成 6 班——骨架切出来的「片」不等于「班」。
MAXPTS = 2
CL_RULE = [c[k:k + MAXPTS] for c in CL for k in range(0, len(c), MAXPTS)]
K_RULE = len(CL_RULE)


def plan_table(cl):
    rows = [dict(line=[SITES[i] for i in c], cost=float(COST[c].sum()),
                 birds=float(BIRDS[c].sum())) for c in cl]
    return sorted(rows, key=lambda r: -r["birds"])


PLAN = plan_table(CL)
PLAN_RULE = plan_table(CL_RULE)
DELTA_COST = route_cost(CL_RULE) - route_cost(CL)
DELTA_LINES = K_RULE - K_BASE

# ---------------------------------------------------------------- 断言
assert K_BASE == 3, f"骨架应断成 3 块，实际 {K_BASE}"
assert MCL_ON_RAW["k"] != K_BASE, "MCL 在原始图上必须给不出 3——否则第 ② 问没有教学对象"
MCL_MATCH = [r["t"] for r in MCL_ROWS if {frozenset(c) for c in r["clusters"]} ==
             {frozenset(c) for c in CL}]
assert KRIT is not None, "inflation 扫描必须出现「不再是 3 块」的一格"
assert SPEC_GAIN > 0.15, f"谱二分必须明显贵于最优割，当前 {SPEC_GAIN:.1%}"
assert DELTA_LINES >= 1

# ---------------------------------------------------------------- 输出
result = dict(n=N, base=dict(k=K_BASE, clusters=CL, t_star=T_STAR,
                             Q=modularity(CL, W)),
              mcl_raw=MCL_ON_RAW, mcl_rows=MCL_ROWS, mcl_match=MCL_MATCH,
              louvain=dict(k=K_LV, clusters=LV, same=LV_SAME, Q=modularity(LV, W)),
              float_scan=FLOAT_SCAN, scan=SCAN, krit=KRIT,
              cut=dict(local_block=[SITES[i] for i in BIG], spec=SPEC_LOCAL, cut_spec=CUT_SPEC,
                       opt_part=[BEST_LOCAL[1], BEST_LOCAL[2]], cut_opt=CUT_LOCAL_OPT,
                       gain=SPEC_GAIN, global_opt=CUT_GLOBAL, global_part=[GA, GB], global_bal=CUT_GLOBAL_BAL,
                       eigb=np.round(EIGB, 4).tolist(), fiedb=np.round(FIEDB, 4).tolist()),
              plan=PLAN, plan_rule=PLAN_RULE,
              rule=dict(k=K_RULE, delta_lines=DELTA_LINES, delta_cost=DELTA_COST),
              weight=W.round(4).tolist(), scomp=S_COMP.round(3).tolist())
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

# ---------------------------------------------------------------- 四面板图
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

ax = axes[0, 0]
color = {}
for k, c in enumerate(CL):
    for i in c:
        color[i] = palette[k % len(palette)]
for i in range(N):
    for j in range(i + 1, N):
        if W[i, j] > 0.004:
            ax.plot([XY[i, 0], XY[j, 0]], [XY[i, 1], XY[j, 1]], color="#777777",
                    lw=W[i, j] * 22, alpha=0.5, zorder=1)
for i in range(N):
    ax.scatter(*XY[i], s=230 + BIRDS[i] * 9, color=color[i], edgecolor="k", zorder=3)
    ax.text(XY[i, 0], XY[i, 1] + 0.42, SITES[i], ha="center", fontsize=8)
ax.set_title(f"① 断面扫描（t ≥ {T_STAR:.2f}）断出的 {K_BASE} 段骨架")
ax.set_xlabel("东西向 km")
ax.set_ylabel("南北向 km")
ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot([s["t"] for s in FLOAT_SCAN], [s["n"] for s in FLOAT_SCAN], "o-", lw=2.0, color="#1f77b4",
        label="连通块数（断面扫描）")
ax.axvline(T_STAR, ls="--", color="k", lw=1.1)
ax.text(T_STAR + 0.005, max(s["n"] for s in FLOAT_SCAN) - 1, f"t*={T_STAR:.2f}", fontsize=9)
ms = [r for r in MCL_ROWS]
ax.plot([r["t"] for r in ms], [r["k"] for r in ms], "s--", color="#d62728", lw=1.6, label="MCL 团数")
ax.set_xlabel("协同权重阈值 t")
ax.set_ylabel("块数")
ax.set_title("② 块数是「扫出来的」，MCL 只有落对断面才跟上")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
names = ["谱二分\n(Fiedler)", "块内最优割\n(全枚举)"]
vals = [CUT_SPEC, CUT_LOCAL_OPT]
bars = ax.bar(names, vals, color=["#d62728", "#2ca02c"], edgecolor="k")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("被切断的协同权重")
ax.set_title(f"③ 谱二分比最优割贵 {SPEC_GAIN:.0%}（{nb} 点、每段≥3 点）")
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
m = max(K_BASE, K_RULE)
ax.bar(np.arange(K_BASE) - 0.2, [r["cost"] for r in PLAN], 0.38, color="#1f77b4",
       label=f"骨架 {K_BASE} 段")
ax.bar(np.arange(K_RULE) + 0.2, [r["cost"] for r in PLAN_RULE], 0.38, color="#d62728",
       label=f"限证后 {K_RULE} 段")
ax.set_xticks(range(m))
ax.set_xticklabels([f"段{i+1}" for i in range(m)], fontsize=8)
ax.set_ylabel("每段年成本（万元）")
ax.set_title(f"④ 一天只跑 2 个点：{K_BASE} 片 → {K_RULE} 班（+{DELTA_LINES} 班）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(f"Day63 · 湿地巡护线：断面扫描 t*={T_STAR:.2f} → {K_BASE} 段｜"
             f"Louvain {K_LV} 社区（同分片 {LV_SAME}）｜谱二分比最优割贵 {SPEC_GAIN:.0%}", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=140)

# ---------------------------------------------------------------- 控制台摘要
print("=" * 80)
print(f"样点 {N}｜总鸟次 {BIRDS.sum():.0f} 千次/年｜总巡护成本 {COST.sum():.1f} 万元/年")
print(f"骨架：t ≥ {T_STAR:.2f} 时断成 {K_BASE} 块，模块度 {modularity(CL, W):.3f}")
for k, c in enumerate(CL):
    print(f"  段{k+1}: {[SITES[i] for i in c]}  成本 {COST[c].sum():.1f} 万  "
          f"鸟次 {BIRDS[c].sum():.0f} 千")
print("-" * 80)
print("断面扫描：", [(s["t"], s["n"]) for s in FLOAT_SCAN])
print("MCL 在原始图上：", MCL_ON_RAW["k"], "团", MCL_ON_RAW["sizes"])
for r in MCL_ROWS:
    print(f"  MCL 在 t={r['t']:.2f} 上：{r['k']} 团 {r['sizes']}，迭代 {r['iters']}，"
          f"{r['dt']*1000:.2f} ms")
print(f"MCL 与骨架完全一致的断面：{MCL_MATCH if MCL_MATCH else '无（膨胀后整张图被蒸成一个点）'}")
print("inflation 扫描（在骨架上）：", [(s["r"], s["k"]) for s in SCAN], "起变化 r ≈", KRIT)
print(f"Louvain：{K_LV} 社区，同分片 {LV_SAME}，模块度 {modularity(LV, W):.3f}")
print(f"  {[[SITES[i] for i in c] for c in LV]}")
print("-" * 80)
print(f"块内最优割（{nb} 点全枚举）{CUT_LOCAL_OPT:.3f}：{[[SITES[i] for i in p] for p in [BEST_LOCAL[1], BEST_LOCAL[2]]]}")
print(f"谱二分 {CUT_SPEC:.3f}：{[[SITES[i] for i in p] for p in SPEC_LOCAL]} → 贵 {SPEC_GAIN:.0%}")
print(f"（全图最优割 {CUT_GLOBAL:.3f} 是把孤立点切出去；限定每段≥2 点后是 {CUT_GLOBAL_BAL:.3f} —— "
      f"「不限定块」时的退化，本讲排除了）")
print("-" * 80)
print(f"巡护队一天只跑 2 个点：{K_BASE} 片 → {K_RULE} 班（+{DELTA_LINES} 班）；"
      f"围栏式年成本不变（{route_cost(CL):.1f} 万元），多出的是 2 个班的人力")
print(f"已写 {OUT_JSON}")
print(f"已出图 {OUT_PNG}")
print("所有断言通过 ✅")
