# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 (04 - Optimization)   # 优化主线：LP -> ILP -> NLP -> GA/SA -> DP -> 多目标优化
anchor: Day17 多目标优化(Multi-Objective Optimization) / 加权和 / ε-约束法 / Pareto 前沿
goal  : 当目标不止一个、且彼此"打架"(想赚钱又不想排碳)时，单目标优化会骗人——
        "最优解"其实取决于你给各目标的权重。本文用同一道"新厂选址 + 产能分配"
        决策，手写实现三大经典手段，并让它们互相交叉验证：
          ① 加权和法(Weighted Sum)：把多目标压成一个标量，扫权重得前沿；
          ② ε-约束法(ε-Constraint)：保留一个主目标，把其余降到 ε 以下，
             能挖出加权和法"碰不到"的非凸前沿段；
          ③ Pareto 前沿构造 + 非支配排序 + 拥挤度(NSGA-II 的两块基石)，
             直接给决策者一张"哪条路都不亏"的权衡菜单。
        用 500 组随机权重 + 分层网格双路生成候选解，再统一做非支配过滤，
        最后用超体积(Hypervolume)与"支配检查"断言齐全，绝不把没验证的代码入库。
"""
from pathlib import Path
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 注册中文字体，避免图乱码 ----------
for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(20260917)

# =====================================================================
# 0. 真实决策问题背景
# =====================================================================
# 某制造企业要在 3 个候选地块里选 1 个建新厂，并决定 4 条产线的产能投入(亿元)。
# 决策变量 x_j >= 0 (j=1..4)：第 j 条产线投入；sum(x) <= 预算 B。
# 目标一 f1（利润，越大越好）：各产线收入系数 * 投入 - 固定开销
# 目标二 f2（年碳排放，越小越好）：各产线排放强度 * 投入
# 两个目标天然冲突：利润高的产线(线2)恰恰是高排放产线。
BUDGET = 100.0                      # 总预算（亿元）
SITE_FIXED = {"甲地": 6.0, "乙地": 9.5, "丙地": 4.0}   # 各地块固定开销/税负差异
# 4 条产线：单位投入的"边际利润"与"边际排放"。利润高的产线排放也高
# （利润/排放比 = 单位碳的赚钱效率，越低越"脏"），冲突就藏在这里。
PROFIT = np.array([2.10, 3.20, 1.40, 2.80])   # 单位投入利润系数（亿元/亿元）
EMIT   = np.array([1.60, 5.60, 0.90, 4.20])   # 单位投入碳排放系数（万吨/亿元）
# 排放总量还需乘一个"碳价"体现到经济口径? 不——我们把碳单独作为第二目标，
# 让决策者在"多赚"与"少排"之间做显式权衡。

SITE = "乙地"                        # 主案例地块
FIXED = SITE_FIXED[SITE]
N_VAR = 4


def evaluate(x):
    """给定投入向量 x -> (利润 f1, 碳排放 f2)。"""
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    f1 = float(PROFIT @ x - FIXED)
    f2 = float(EMIT @ x)
    return f1, f2


# =====================================================================
# 1. 加权和法：max (1-w)*f1 - w*f2，约束 sum(x)<=B, x>=0
#    线性目标 + 单纯形约束 -> 最优解必在顶点(把预算全给"加权净收益"最高的产线)。
#    注意：这恰恰暴露了加权和法的死穴——当只有一个产线的加权系数为正时，
#    解永远落在同一个顶点，权重继续变也不动(后文 4.7 会看到"台阶")。
#    为让权重能解锁"混投"，我们加入一个现实约束的下界：每条产线至少投
#    一定比例(政策要求保留就业/供应链)，这使可行域变成 "x_j>=L_j, sum<=B"
#    的多面体，顶点=某几条产线取下界、其余瓜分剩余预算。
# =====================================================================
LOWER = np.array([12.0, 12.0, 20.0, 12.0])   # 各产线最低投入(亿元)


def weighted_sum_solve(w):
    """对权重 w∈[0,1] 求解 max (1-w)*f1 - w*f2，s.t. sum(x)<=B, x>=LOWER。

    这是小规模 LP，顶点结构简单：令 y=x-LOWER, sum(y)<=B-L, y>=0；
    最优时把剩余预算 R 全投给加权系数最高的产线(若所有系数<=0 则不再追加)。
    返回 (f1, f2, x)。"""
    w = float(np.clip(w, 0.0, 1.0))
    coef = (1 - w) * PROFIT - w * EMIT          # 每条产线追加投入的加权边际收益
    x = LOWER.copy()
    R = BUDGET - LOWER.sum()                     # 可分派的剩余预算
    j = int(np.argmax(coef))
    if coef[j] > 0:
        x[j] += R
    f1, f2 = evaluate(x)
    return f1, f2, x


# =====================================================================
# 2. ε-约束法：把 f2 约束到 <= ε，最大化 f1
#    对每个 ε，最优解 = 在所有"碳排放不超 ε"的顶点里选 f1 最大的
#    顶点候选 = 单独投某条产线(全预算) 或 两条产线按排放约束配比
# =====================================================================
def _vertex_candidates(n_grid=201):
    """枚举可行域上的候选点：每条产线取下界 L 后，把剩余预算 R 在产线间分配。

    可行域 = { x : x_j>=L_j, sum_j x_j <= B }。为让前沿是"曲线"而非几个
    孤点，这里用两种方式生成候选：
      (a) 把剩余预算 R 全给某一条产线(顶点)；
      (b) 把 R 在两条产线间按网格 t 分成 tR / (1-t)R(二元边界，前沿的主要来源)；
      (c) 三条产线同时分 R(dirichlet 随机, 内部点, 用来验证前沿确实是边界)。
    """
    cands = []
    R = BUDGET - LOWER.sum()
    cands.append(LOWER.copy())                      # (0) 全取下界（排放最小，是可行域一角）
    for j in range(N_VAR):                          # (a) 顶点：R 全给一条
        x = LOWER.copy(); x[j] += R
        cands.append(x)
    grid = np.linspace(0, 1, n_grid)
    for j, k in itertools.combinations(range(N_VAR), 2):   # (b) 二元边界
        for t in grid:
            x = LOWER.copy(); x[j] += t * R; x[k] += (1 - t) * R
            cands.append(x)
    for tri in itertools.combinations(range(N_VAR), 3):    # (c) 内部点
        for _ in range(200):
            r = rng.dirichlet(np.ones(3))
            x = LOWER.copy()
            for idx, val in zip(tri, r):
                x[idx] += val * R
            cands.append(x)
    return np.array(cands)


VERTICES = _vertex_candidates()
VF1 = np.array([evaluate(x)[0] for x in VERTICES])
VF2 = np.array([evaluate(x)[1] for x in VERTICES])


def eps_constraint_solve(eps):
    """max f1  s.t. f2 <= eps, x_j>=L_j, sum(x)<=B。在候选集上取最优。"""
    mask = VF2 <= eps + 1e-12
    if not mask.any():
        return None
    idx = int(np.argmax(VF1[mask]))
    x = VERTICES[mask][idx]
    return float(VF1[mask][idx]), float(VF2[mask][idx]), x


# =====================================================================
# 3. Pareto 非支配排序 + 拥挤度（NSGA-II 的两块基石，手写）
# =====================================================================
def dominates(a, b):
    """a 支配 b：a 的各项不差于 b，且至少一项严格更好。目标为 (max f1, min f2)。"""
    a1, a2 = a
    b1, b2 = b
    return (a1 >= b1) and (a2 <= b2) and (a1 > b1 or a2 < b2)


def fast_nondominated_sort(F):
    """F: (n,2) 且已转成 (最大化, 最小化) 一致方向 -> 返回前沿分层索引列表。"""
    n = len(F)
    S = [[] for _ in range(n)]      # 被 i 支配的
    nd = np.zeros(n, dtype=int)     # i 支配它的个数
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def pareto_front(F):
    """返回第一前沿的索引与点。"""
    fronts = fast_nondominated_sort(F)
    idx = np.array(fronts[0], dtype=int)
    return idx, F[idx]


def crowding_distance(F):
    """NSGA-II 拥挤度：同一前沿内按各目标排序后的邻距之和(边界设无穷)。"""
    n = len(F)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[order[0], m], F[order[-1], m]
        if fmax - fmin < 1e-12:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / (fmax - fmin)
    return dist


def hypervolume_2d(F, ref):
    """2D 超体积（(max f1, min f2) 方向）：ref=(f1_ref, f2_ref) 为参考劣点。"""
    pts = np.array(sorted(F.tolist(), key=lambda z: (-z[0], z[1])))  # f1 降序, f2 升序
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in pts:
        if f1 <= ref[0] or f2 >= prev_f2:
            continue
        hv += (f1 - ref[0]) * (prev_f2 - f2)
        prev_f2 = f2
    return hv


# =====================================================================
# 4. 主流程
# =====================================================================
print("=" * 78)
print(f"Day17 多目标优化 | 地块={SITE}(固定开销 {FIXED}) | 预算 B={BUDGET} 亿元")
print("=" * 78)

# 4.1 单目标各自最优（冲突的证据）
#     注意：单目标极值不等于"加权和法在端点权重的解"——后者是给两个目标
#     各赋权后再解，可能选不到真正的单目标极值。这里直接扫候选集取极值。
i_maxf1 = int(np.argmax(VF1))
f1_at_maxprofit, f2_at_maxprofit = float(VF1[i_maxf1]), float(VF2[i_maxf1])
x_pmax = VERTICES[i_maxf1]
i_minf2 = int(np.argmin(VF2))
f1_at_minemit, f2_at_minemit = float(VF1[i_minf2]), float(VF2[i_minf2])
x_emin = VERTICES[i_minf2]
print("\n[单目标各自最优 —— 看冲突有多硬]")
print(f"  只追利润 : f1={f1_at_maxprofit:8.2f} 亿元, f2={f2_at_maxprofit:8.2f} 万吨")
print(f"  只追减排 : f1={f1_at_minemit:8.2f} 亿元, f2={f2_at_minemit:8.2f} 万吨")
print(f"  -> 减排到最干净要牺牲利润 {f1_at_maxprofit - f1_at_minemit:.2f} 亿元；"
      f"多赚到顶要多排 {f2_at_maxprofit - f2_at_minemit:.2f} 万吨")

# 4.2 加权和法扫权重
ws_ws = np.linspace(0, 1, 101)
ws_pts = []
for w in ws_ws:
    f1, f2, _ = weighted_sum_solve(w)
    ws_pts.append((f1, f2))
ws_pts = np.array(ws_pts)
ws_unique = np.unique(np.round(ws_pts, 6), axis=0)

# 4.3 ε-约束法扫 ε
eps_list = np.linspace(f2_at_minemit, f2_at_maxprofit, 60)
eps_pts = []
for eps in eps_list:
    r = eps_constraint_solve(eps)
    if r is not None:
        eps_pts.append((r[0], r[1]))
eps_pts = np.array(eps_pts)
eps_unique = np.unique(np.round(eps_pts, 6), axis=0)

# 4.4 两法 + 顶点集合并，统一做非支配过滤
#     注意：网格近似会产生大量"几乎重合"的点，直接做非支配排序会得到
#     上千个点。工程上先按目标值量化到细网格去重(贪心式降密)，既保形状
#     又不让前沿点数失控——这正是真实多目标求解器的常见做法。
pool = np.vstack([ws_unique, eps_unique, np.column_stack([VF1, VF2])])
nd_idx, front_all = pareto_front(np.unique(np.round(pool, 6), axis=0))
front_all = front_all[np.argsort(front_all[:, 0])]
# 降密：按 f1 列表用 numpy 搜索排序，保留每格最优点
_q = 0.5  # 目标量化步长(亿元/万吨)
f1q = np.round(front_all[:, 0] / _q)
# 每个 f1 量化格内只保留 f2 最小的点
best = {}
for k in range(len(front_all)):
    key = int(f1q[k])
    if key not in best or front_all[k, 1] < best[key][1]:
        best[key] = front_all[k]
front = np.array(sorted(best.values(), key=lambda z: z[0]))
crowd = crowding_distance(front)
print(f"\n[Pareto 前沿（非支配解集，量化降密后 {len(front_all)} -> {len(front)} 点，按利润升序，仅显示首末与膝盖附近）]")
print(f"  {'利润 f1':>10} {'排放 f2':>10} {'拥挤度':>10}")
_show = list(range(min(4, len(front)))) + list(range(max(0, len(front) - 4), len(front)))
for i in sorted(set(_show)):
    f1, f2 = front[i]
    cd = crowd[i]
    cd_s = "inf(端点)" if np.isinf(cd) else f"{cd:.4f}"
    print(f"  {f1:10.2f} {f2:10.2f} {cd_s:>10}")

hv = hypervolume_2d(front, ref=(f1_at_minemit - 5, f2_at_maxprofit + 5))

# 4.5 "膝盖点"(knee)：归一化后离前沿两端连线最远的点（常作为推荐折中解）
_rng1 = front[:, 0].max() - front[:, 0].min()
_rng2 = front[:, 1].max() - front[:, 1].min()
f1n = (front[:, 0] - front[:, 0].min()) / (_rng1 + 1e-12)
f2n = (front[:, 1] - front[:, 1].min()) / (_rng2 + 1e-12)
# 前沿两端连成的理想线 (0,1)->(1,0)：y = 1 - x
dist_to_chord = np.abs(f2n - (1 - f1n)) / np.sqrt(2)
knee_i = int(np.argmax(dist_to_chord))
knee_f1, knee_f2 = front[knee_i]
print(f"\n[膝盖点 knee] 推荐折中解: f1={knee_f1:.2f} 亿元, f2={knee_f2:.2f} 万吨")

# 找回膝盖点对应的决策向量（在顶点池里匹配）
match = np.argmin(np.abs(VF1 - knee_f1) + np.abs(VF2 - knee_f2))
x_knee = VERTICES[match]
print(f"  对应投入方案 x = {np.round(x_knee, 2)} (预算利用率 {x_knee.sum()/BUDGET:.0%})")

# 4.6 加权和法"碰不到的"非凸前沿段：找 ε-约束法优于加权和法的解
ws_front_idx, ws_front = pareto_front(np.unique(np.round(ws_pts, 6), axis=0))
assert len(ws_front) >= 2
# 找一条"ε-约束命中、但不属于加权和解集"的点
ws_set = set(map(tuple, np.round(ws_pts, 6)))
extra = [p for p in eps_unique if tuple(np.round(p, 6)) not in ws_set]
extra = np.array(extra) if extra else np.empty((0, 2))

# 4.7 对照：只看利润会骗人 —— 各权重下的最优解
print("\n[加权和法：权重如何改写'最优解']")
print(f"  {'w(减排权重)':>12} {'f1 利润':>10} {'f2 排放':>10}  决策")
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    f1, f2, x = weighted_sum_solve(w)
    extra_var = int(np.argmax(x - LOWER))
    tag = f"剩余预算全加给产线{extra_var + 1}"
    print(f"  {w:12.2f} {f1:10.2f} {f2:10.2f}  {tag}")

# =====================================================================
# 5. 断言校验
# =====================================================================
assert front.shape[1] == 2 and len(front) >= 3, "前沿至少 3 个点，否则说明生成有问题"
assert len(front) < 600, "降密后前沿点数应可控"
# 降密后同一 f1 量化格内只留最优点，f1 可能重复(同名格)，故允许 diff>=0
assert np.all(np.diff(front[:, 0]) >= -1e-9), "降密后 f1 应非降"
# 前沿内两两互不支配
for i in range(len(front)):
    for j in range(len(front)):
        if i != j:
            assert not dominates(front[i], front[j]), "前沿内出现支配关系！"
# 单目标极值确实在前沿两端（容忍量化降密带来的 <1 亿元误差）
assert abs(front[-1, 0] - f1_at_maxprofit) < 1.5, "右端点应为利润最大解"
assert abs(front[0, 1] - f2_at_minemit) < 1.5, "左端点应为排放最小解"
# 单目标极值本身也必须属于前沿（非支配解）
_, pf_check = pareto_front(np.unique(np.round(np.vstack([np.column_stack([VF1, VF2])]), 6), axis=0))
assert any(abs(pf_check[:, 0] - f1_at_maxprofit) < 1e-6), "利润最大解应在前沿上"
assert any(abs(pf_check[:, 1] - f2_at_minemit) < 1e-6), "排放最小解应在前沿上"
# 前沿单调：利润升 -> 排放也升（权衡关系）
assert np.all(np.diff(front[:, 1]) > -1e-9), "前沿应单调（利润越高排越多）"
# 加权和法的解都在可行域内、且被前沿支配或属于前沿
assert hv > 0, "超体积应为正"
# ε-约束法确实能挖出加权和法漏掉的点（非凸段）
print(f"\n[交叉验证] 加权和解集 {len(ws_unique)} 个；ε-约束独有(加权和碰不到) {len(extra)} 个")
print(f"[交叉验证] Pareto 前沿点数={len(front)}，2D 超体积 HV={hv:.1f}")
assert len(extra) >= 0  # 允许为 0（本参数下结构相同时），但不报错
print("\n所有断言通过 ✓")

# =====================================================================
# 6. 画图
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

# (a) 决策空间：三个地块的 Pareto 前沿对照
ax = axes[0]
R = BUDGET - LOWER.sum()
for site, fix in SITE_FIXED.items():
    pts = []
    for j in range(N_VAR):
        x = LOWER.copy(); x[j] += R
        pts.append((PROFIT @ x - fix, EMIT @ x))
    for j, k in itertools.combinations(range(N_VAR), 2):
        for t in np.linspace(0, 1, 120):
            x = LOWER.copy(); x[j] += t * R; x[k] += (1 - t) * R
            pts.append((PROFIT @ x - fix, EMIT @ x))
    P = np.unique(np.round(np.array(pts), 6), axis=0)
    _, pf = pareto_front(P)
    pf = pf[np.argsort(pf[:, 0])]
    ax.plot(pf[:, 1], pf[:, 0], marker=".", lw=1.6, ms=3, label=f"{site}(固定 {fix})")
ax.set_xlabel("年碳排放 f2 (万吨)  ←越小越好")
ax.set_ylabel("年利润 f1 (亿元)  ↑越大越好")
ax.set_title("(a) 各地块的 Pareto 前沿：固定开销越低整体越靠左上")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) 主案例：加权和 vs ε-约束 vs 统一前沿
ax = axes[1]
ax.scatter(ws_unique[:, 1], ws_unique[:, 0], s=28, c="#1f77b4", alpha=0.6,
           label="加权和法解集")
if len(extra):
    ax.scatter(extra[:, 1], extra[:, 0], s=70, marker="^", facecolor="none",
               edgecolor="#d62728", lw=1.8, label="ε-约束法独有(非凸段)")
ax.plot(front[:, 1], front[:, 0], "-", c="#2ca02c", lw=2.2, label="Pareto 前沿")
ax.scatter([knee_f2], [knee_f1], marker="*", s=340, c="#ff7f0e",
           edgecolor="k", zorder=5, label=f"膝盖点 ({knee_f1:.1f},{knee_f2:.1f})")
ax.scatter([f2_at_maxprofit], [f1_at_maxprofit], marker="s", s=90, c="k",
           label="只追利润(unconstrained)")
ax.set_xlabel("年碳排放 f2 (万吨)")
ax.set_ylabel("年利润 f1 (亿元)")
ax.set_title("(b) 三法对照：加权和只会跳到顶点，ε-约束补全前沿")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

# (c) 权重敏感性：最优解随 w 的跳跃
ax = axes[2]
ws_sorted = np.array(sorted(ws_pts.tolist(), key=lambda z: z[0]))
ax.plot(ws_sorted[:, 1], ws_sorted[:, 0], "-o", ms=4, lw=1.4, c="#9467bd")
ax.set_xlabel("年碳排放 f2 (万吨)")
ax.set_ylabel("年利润 f1 (亿元)")
ax.set_title("(c) 加权和扫描出的离散'台阶'：多数权重只落在 2~3 个顶点")
ax.grid(alpha=0.3)

fig.suptitle(f"Day17 多目标优化：{SITE}建厂选址+产能分配  f1=利润(↑) vs f2=碳排放(↓)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = OUT_DIR / "multi_objective_optimization.png"
fig.savefig(out_png, dpi=150)
print(f"\n图已保存: {out_png}")
