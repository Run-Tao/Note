# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 算法（元启发式 · 模拟退火 / 禁忌搜索 · 邻域设计 · 接受准则）
anchor: L5 算法主线，接 Day6（GA/SA 求 TSP 入门）、Day21（TSP 精确解对照）、
        Day79（灵敏度分析：方案稳不稳）——本讲正面回答「解稳不稳」
goal: 云溪县 12 个村级冷链自提点的夜班配送 —— 一辆 7 吨车从中心库出发，每晚跑完 12 点回库。
      ① 三种构造法（最近邻 / 最省插入 / 扫描法）谁的开头更好；
      ② 模拟退火的温度到底在起作用还是在装样子（同链长度下与"永远接受"对照）；
      ③ 禁忌搜索的禁忌表到底救了什么（关掉禁忌 = 纯下降会卡在什么程度）；
      ④ 拒绝率诊断：接受率长期接近 1 → 温度太高；长期为 0 → 温度太低。
run: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-18/day80_sa_tabu_vrp.py"
"""

import json
import math
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体 ----------
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
SEED = 20261118
NV = None  # 路线长度（不含中心库），主流程里赋值

# ============================================================
# 0. 数据：云溪县 12 个村级冷链自提点（中心库在原点）
#    坐标单位 km；需求单位：筐（每筐 20 kg）
# ============================================================
NAMES = ["中心库", "梅溪", "双桥", "石坳", "南坪", "黄泥", "榕树脚",
         "白水", "太平桥", "冷水", "新塘", "竹山", "枫林坳"]
XY = np.array([
    [0.00, 0.00],   # 中心库
    [2.10, 6.40],   # 梅溪
    [5.80, 3.10],   # 双桥
    [7.40, 8.90],   # 石坳
    [4.20, -2.60],  # 南坪
    [-3.50, 3.80],  # 黄泥
    [-6.20, -1.40], # 榕树脚
    [-1.80, -6.90], # 白水
    [3.30, -6.10],  # 太平桥
    [9.60, 1.20],   # 冷水
    [11.20, -4.30], # 新塘
    [-8.40, 5.60],  # 竹山
    [-5.10, -7.20], # 枫林坳
])
DEMAND = np.array([0, 18, 24, 12, 20, 15, 10, 22, 16, 14, 19, 9, 11], dtype=float)
CAP_KG = 7000.0          # 7 吨车
KG_PER_BASKET = 20.0
CAP = CAP_KG / KG_PER_BASKET   # 350 筐 / 晚
N = len(XY)
D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))

# 夜班固定成本（用于把"距离"翻译成"钱"，让结论能对站长说话）
Y_PER_KM = 4.20          # 元/km（燃油+折旧+司机折算）
FIX_PER_RUN = 0.0        # 单车单趟，无固定发车成本


def tour_len(t):
    """闭合路线总长（km）。t 为不含中心库的访问序列（点编号 1..N-1）。"""
    s = 0.0
    prev = 0
    for v in t:
        s += D[prev, v]
        prev = v
    return s + D[prev, 0]


def check_tour(t):
    """路线合法性：恰好访问每个点一次、且不访问中心库。"""
    return sorted(t) == list(range(1, N))


def load_of(t):
    return float(DEMAND[list(t)].sum())


# ============================================================
# 1. 三种构造法：谁的起点更好
# ============================================================
def nn_tour(start):
    """最近邻构造，可指定起点。"""
    t = [start]
    left = set(range(1, N)) - {start}
    while left:
        last = t[-1]
        nxt = min(left, key=lambda v: D[last, v])
        t.append(nxt)
        left.remove(nxt)
    return tuple(t)


def cheapest_insertion():
    """最省插入：从最远点出发，每次插入增量最小的点。"""
    # 起始：离库最远的两个点组成一条链路
    far = max(range(1, N), key=lambda v: D[0, v])
    t = [far]
    while len(t) < N - 1:
        best = None
        for v in range(1, N):
            if v in t:
                continue
            for pos in range(len(t) + 1):
                a = 0 if pos == 0 else t[pos - 1]
                b = 0 if pos == len(t) else t[pos]
                delta = D[a, v] + D[v, b] - D[a, b]
                if best is None or delta < best[0]:
                    best = (delta, v, pos)
        _, v, pos = best
        t.insert(pos, v)
    return tuple(t)


def sweep_tour():
    """扫描法：按极角排序（从离库最远点起，避免库点角度无定义）。"""
    ang = np.arctan2(XY[1:, 1], XY[1:, 0])
    order = np.argsort(ang) + 1
    return tuple(int(v) for v in order)


# ============================================================
# 2. 邻域：2-opt 段反转 + or-opt 单点移动
# ============================================================
def delta_2opt(t, i, j):
    """把 t[i..j] 反转后的距离增量（t 不含中心库，首尾接库）。"""
    n = len(t)
    a = 0 if i == 0 else t[i - 1]
    b = t[i]
    c = t[j]
    d = 0 if j == n - 1 else t[j + 1]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def delta_oropt(t, i, k):
    """把元组下标 i 的点搬到【移除后的位置 k】（0..n-1，k 为插入后落在的槽位）。"""
    n = len(t)
    v = t[i]
    prev = 0 if i == 0 else t[i - 1]
    nxt = 0 if i == n - 1 else t[i + 1]
    remove = D[prev, nxt] - D[prev, v] - D[v, nxt]
    rest = t[:i] + t[i + 1:]
    if k < 0 or k > n - 1:
        return float("inf")
    if k == i:          # 插回原位
        return 0.0
    a = 0 if k == 0 else rest[k - 1]
    b = 0 if k == n - 1 else rest[k]
    add = D[a, v] + D[v, b] - D[a, b]
    return remove + add


def apply_2opt(t, i, j):
    """i/j 为元组下标（与 delta_2opt/apply_oropt 同一口径）。"""
    return t[:i] + t[i:j + 1][::-1] + t[j + 1:]


def apply_oropt(t, i, k):
    """i 为元组下标，k 为移除后的插入槽位（0..n-1）。"""
    v = t[i]
    rest = t[:i] + t[i + 1:]
    return rest[:k] + (v,) + rest[k:]


# ============================================================
# 3. 模拟退火
# ============================================================
def initial_temp(t0, rng, sample=300, p=0.85):
    """经验公式：采样邻域正增量，取使 P(accept)=0.85 的平均正增量。"""
    t = list(t0)
    pos = []
    for _ in range(sample):
        r = rng.random()
        if r < 0.7:
            i = int(rng.integers(0, NV - 1))
            j = int(rng.integers(i + 1, NV))
            dlt = delta_2opt(t, i, j)
        else:
            i = int(rng.integers(0, NV - 1))
            k = int(rng.integers(0, NV))
            dlt = delta_oropt(t, i, k)
        if dlt > 0:
            pos.append(dlt)
    return float(np.mean(pos) / -math.log(p)) if pos else 1.0


def run_sa(t0, iters, seed, always_accept=False, t_start=None, cool=None):
    rng = np.random.default_rng(seed)
    t = tuple(t0)
    cur = tour_len(t)
    best, best_t = cur, t
    temp = initial_temp(t, rng) if t_start is None else t_start
    cool = cool if cool is not None else (1e-4 / temp) ** (1.0 / iters)
    acc = 0
    trace = []
    for it in range(iters):
        r = rng.random()
        if r < 0.7:
            i = int(rng.integers(0, NV - 1))
            j = int(rng.integers(i + 1, NV))
            dlt = delta_2opt(t, i, j)
            cand = apply_2opt(t, i, j)
        else:
            i = int(rng.integers(0, NV - 1))
            k = int(rng.integers(0, NV))
            dlt = delta_oropt(t, i, k)
            cand = apply_oropt(t, i, k)
        if always_accept:
            take = True
        elif dlt <= 0:
            take = True
        else:
            take = rng.random() < math.exp(-dlt / temp)
        if take:
            t, cur = cand, cur + dlt
            acc += 1
            if cur < best - 1e-12:
                best, best_t = cur, t
        temp *= cool
        if it % 50 == 0:
            trace.append((it, cur, best))
    return dict(best=best, tour=best_t, trace=trace, acc_rate=acc / iters,
                final_acc=None)


# ============================================================
# 4. 禁忌搜索（2-opt 段反转 + 点移动，禁忌属性 = 被移动的点）
# ============================================================
def run_tabu(t0, iters, seed, tenure=7, use_tabu=True, cand_count=60):
    rng = np.random.default_rng(seed)
    t = tuple(t0)
    cur = tour_len(t)
    best, best_t = cur, t
    tabu = {}
    trace = []
    for it in range(iters):
        best_move = None
        best_d = np.inf
        for _ in range(cand_count):
            if rng.random() < 0.6:
                i = int(rng.integers(0, NV - 1))
                j = int(rng.integers(i + 1, NV))
                dlt = delta_2opt(t, i, j)
                key = ("2opt", i, j)
                mv = ("2opt", i, j)
            else:
                i = int(rng.integers(0, NV - 1))
                k = int(rng.integers(0, NV))
                dlt = delta_oropt(t, i, k)
                key = ("move", i, None)
                mv = ("move", i, k)
            is_tabu = use_tabu and tabu.get(key, -1) > it
            aspir = (cur + dlt) < best - 1e-12      # 特赦准则
            if is_tabu and not aspir:
                continue
            if dlt < best_d:
                best_d, best_move, best_key = dlt, mv, key
        if best_move is None:
            # 全被禁忌：强制取第一个候选（真实实现必须有这条兜底）
            i = int(rng.integers(0, NV - 1))
            j = int(rng.integers(i + 1, NV))
            best_move, best_key = ("2opt", i, j), ("2opt", i, j)
            best_d = delta_2opt(t, i, j)
        if best_move[0] == "2opt":
            _, i, j = best_move
            t = apply_2opt(t, i, j)
        else:
            _, i, k = best_move
            t = apply_oropt(t, i, k)
        cur = tour_len(t)
        tabu[best_key] = it + tenure
        if cur < best - 1e-12:
            best, best_t = cur, t
        if it % 20 == 0:
            trace.append((it, cur, best))
    return dict(best=best, tour=best_t, trace=trace)


# ============================================================
# 主流程
# ============================================================
print("=" * 78)
print("云溪县 12 个村级冷链自提点夜班配送（单车辆 TSP）")
print("=" * 78)
NV = N - 1
tot_load = load_of(range(1, N))
print(f"  总需求 {tot_load:.0f} 筐 = {tot_load*KG_PER_BASKET:.0f} kg，车容量 {CAP:.0f} 筐 "
      f"({CAP_KG:.0f} kg) → 装满率 {tot_load/CAP*100:.1f}%，**单趟可行**")
print(f"  里程成本 {Y_PER_KM:.2f} 元/km")

# ---------- [1] 构造法 ----------
print("\n[1] 三种构造法：起点差多少")
cands = {}
for s in range(1, N):
    t = nn_tour(s)
    assert check_tour(t), "最近邻路线不合法"
    cands[f"最近邻(起于{NAMES[s]})"] = (tour_len(t), t)
cands["最省插入"] = (tour_len(cheapest_insertion()), cheapest_insertion())
cands["扫描法"] = (tour_len(sweep_tour()), sweep_tour())
for k, (v, t) in sorted(cands.items(), key=lambda kv: kv[1][0]):
    print(f"  {k:20s} L = {v:7.2f} km")
best_cons_name = min(cands, key=lambda k: cands[k][0])
t_cons = cands[best_cons_name][1]
L_cons = cands[best_cons_name][0]
print(f"  → 最好的构造解：{best_cons_name}，L = {L_cons:.2f} km（"
      f"{L_cons*Y_PER_KM:,.1f} 元/晚）")

# ---------- [2] 模拟退火 ----------
print("\n[2] 模拟退火：温度在起作用吗（同样 20,000 步，三条链对照）")
ITERS = 20000
sa = run_sa(t_cons, ITERS, SEED)
aa = run_sa(t_cons, ITERS, SEED, always_accept=True)
# 冷启动：极低初温 = 几乎纯下降
sa_cold = run_sa(t_cons, ITERS, SEED, t_start=1e-6)
print(f"  正常退火        L_best = {sa['best']:7.2f} km  接受率 {sa['acc_rate']*100:5.1f}%")
print(f"  永远接受(纯随机) L_best = {aa['best']:7.2f} km  接受率 {aa['acc_rate']*100:5.1f}%")
print(f"  极低初温(准下降) L_best = {sa_cold['best']:7.2f} km  接受率 {sa_cold['acc_rate']*100:5.1f}%")

# 多随机种子看方差（单次跑赢不算数）
seeds = [1, 2, 3, 4, 5, 6, 7, 8]
mult = [run_sa(t_cons, 8000, 1000 + s)["best"] for s in seeds]
print(f"  8 个种子的 SA(8000 步)：均值 {np.mean(mult):.2f}，标准差 {np.std(mult):.2f}，"
      f"最好 {min(mult):.2f}，最差 {max(mult):.2f}")

# ---------- [3] 禁忌搜索 ----------
print("\n[3] 禁忌搜索：禁忌表到底救了什么（同样 20,000 步）")
tb = run_tabu(t_cons, 3000, SEED, use_tabu=True)
tb_off = run_tabu(t_cons, 3000, SEED, use_tabu=False)
print(f"  有禁忌(tenure=7) L_best = {tb['best']:7.2f} km")
print(f"  关禁忌(纯下降)   L_best = {tb_off['best']:7.2f} km")

# 禁忌长度敏感性
ten_scan = {}
for ten in [3, 5, 7, 11, 15, 23]:
    r = run_tabu(t_cons, 3000, SEED, tenure=ten, use_tabu=True)
    ten_scan[ten] = r["best"]
    print(f"  tenure={ten:2d} → L_best = {r['best']:7.2f} km")

# ---------- [4] 接受率诊断 ----------
print("\n[4] 拒绝率诊断：温度调坏的两种典型症状")
diag = {}
for name, tk in [("过高(初温×50)", None), ("过低(初温×0.01)", None)]:
    base_t = initial_temp(t_cons, np.random.default_rng(0))
    tt = base_t * (50 if "过高" in name else 0.01)
    r = run_sa(t_cons, 4000, SEED, t_start=tt)
    diag[name] = r["acc_rate"]
    print(f"  {name:16s} 接受率 {r['acc_rate']*100:5.1f}%  L_best = {r['best']:7.2f} km")
print("  读法：接受率≈1 → 温度太高在乱走；接受率≈0 → 温度太低等于纯下降；"
      "健康区间大致 10%~40%")

# ---------- [5] 下界估算（1-tree 松弛，给「离最优还有多远」一个锚） ----------
print("\n[5] 下界：最小 1-tree（放弃度数约束的松弛，仅作参考锚点）")


def mst_cost(nodes, exclude_edge=None):
    """Prim 最小生成树。"""
    nodes = list(nodes)
    if len(nodes) < 2:
        return 0.0
    in_tree = {nodes[0]}
    out = set(nodes[1:])
    cost = 0.0
    while out:
        best = None
        for a in in_tree:
            for b in out:
                if exclude_edge and ((a, b) == exclude_edge or (b, a) == exclude_edge):
                    continue
                w = D[a, b]
                if best is None or w < best[0]:
                    best = (w, b)
        cost += best[0]
        in_tree.add(best[1])
        out.remove(best[1])
    return cost


lower_bounds = []
for root in range(N):
    others = [v for v in range(N) if v != root]
    # 选两条以 root 为端点的最短边
    e = sorted(others, key=lambda v: D[root, v])[:2]
    lb = D[root, e[0]] + D[root, e[1]] + mst_cost(others, exclude_edge=(root, e[0]))
    lower_bounds.append(lb)
LB = max(lower_bounds)
print(f"  最紧的 1-tree 下界 LB = {LB:.2f} km")
print(f"  SA 最好解 {sa['best']:.2f} km，gap = {(sa['best']-LB)/LB*100:.2f}%（下界不等于最优，"
      "gap 是「最多还差这么多」）")

# ---------- [6] 方案稳定性（回应 Day79：方案本身会不会换人） ----------
print("\n[6] 稳定性：把 SA 跑 8 次，看最优路线是不是同一个（Day79 的教训：基一变 z 可能只掉一点）")
tours = [run_sa(t_cons, 8000, 2000 + s)["tour"] for s in seeds]
canon = set()
for t in tours:
    # 规范化：正向/反向取字典序小者
    f = tuple(t)
    r = tuple(reversed(t))
    canon.add(min(f, r))
print(f"  8 次跑出 {len(canon)} 条不同的最优路线（长度 {len(tours[0])} 点）")
for i, c in enumerate(canon):
    print(f"    路线 {i+1}: " + " → ".join(NAMES[v] for v in c))

# ============================================================
# 出图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

ax = axes[0, 0]
best_t_sa = sa["tour"]
route = [0] + list(best_t_sa) + [0]
ax.plot(XY[route, 0], XY[route, 1], "-o", lw=1.6, ms=6, color="#4C78A8")
ax.plot(XY[0, 0], XY[0, 1], "s", ms=13, color="crimson")
for v in range(N):
    ax.annotate(NAMES[v], (XY[v, 0], XY[v, 1]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
ax.set_title(f"(a) SA 最优路线 L = {sa['best']:.2f} km")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.grid(alpha=0.3)

ax = axes[0, 1]
for name, r, col in [("正常退火", sa, "#E45756"),
                     ("永远接受", aa, "#F58518"),
                     ("极低初温", sa_cold, "#54A24B")]:
    tr = np.array(r["trace"], dtype=float)
    ax.plot(tr[:, 0], tr[:, 2], lw=1.8, label=f"{name}（最优 {r['best']:.1f}）", color=col)
ax.axhline(L_cons, ls=":", lw=1.3, color="black", label=f"构造解 {L_cons:.1f}")
ax.set_xlabel("迭代步")
ax.set_ylabel("当前最优路线长度 (km)")
ax.set_title("(b) 三条链对照：温度不是装饰")
ax.legend(fontsize=8)

ax = axes[1, 0]
tr = np.array(sa["trace"], dtype=float)
ax.plot(tr[:, 0], tr[:, 1], lw=0.9, color="#B279A2", label="当前解")
ax.plot(tr[:, 0], tr[:, 2], lw=1.8, color="#E45756", label="历史最优")
ax.set_xlabel("迭代步")
ax.set_ylabel("路线长度 (km)")
ax.set_title(f"(c) SA 收敛轨迹（接受率 {sa['acc_rate']*100:.1f}%）")
ax.legend(fontsize=8)

ax = axes[1, 1]
ten = sorted(ten_scan)
ax.bar([str(v) for v in ten], [ten_scan[v] for v in ten], color="#72B7B2")
ax.axhline(tb_off["best"], ls="--", color="crimson", lw=1.5,
           label=f"关禁忌纯下降 {tb_off['best']:.1f}")
ax.set_xlabel("禁忌长度 tenure")
ax.set_ylabel("最优路线长度 (km)")
ax.set_title("(d) 禁忌长度敏感性")
ax.legend(fontsize=8)

fig.suptitle("Day80 · 模拟退火 vs 禁忌搜索 —— 云溪县冷链夜班配送 TSP", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "day80_sa_tabu_vrp.png"), dpi=130)

# ============================================================
# 断言（只检验设计公平性，不预设谁该赢）
# ============================================================
assert check_tour(sa["tour"]), "SA 返回的路线不合法"
assert check_tour(tb["tour"]), "禁忌搜索返回的路线不合法"
assert abs(tour_len(sa["tour"]) - sa["best"]) < 1e-9, "SA 声称的最优值与重算不一致"
assert abs(tour_len(tb["tour"]) - tb["best"]) < 1e-9, "禁忌声称的最优值与重算不一致"
assert all(v > 0 for v in mult), "多种子结果出现非正长度"
assert LB <= sa["best"] + 1e-9, "下界超过可行解 → 下界算错了"
print(f"\n断言全部通过；用时 {time.time()-T0:.1f}s")

payload = {
    "N_nodes": N,
    "total_load_baskets": round(tot_load, 1),
    "capacity_baskets": round(CAP, 1),
    "fill_rate": round(tot_load / CAP, 4),
    "construction": {k: round(v[0], 4) for k, v in cands.items()},
    "best_construction": best_cons_name,
    "best_construction_len": round(float(L_cons), 4),
    "sa": {"len": round(sa["best"], 4), "acc_rate": round(sa["acc_rate"], 4),
           "tour": [NAMES[v] for v in sa["tour"]]},
    "always_accept": {"len": round(aa["best"], 4), "acc_rate": round(aa["acc_rate"], 4)},
    "cold_start": {"len": round(sa_cold["best"], 4), "acc_rate": round(sa_cold["acc_rate"], 4)},
    "multi_seed": {"n": len(mult), "mean": round(float(np.mean(mult)), 4),
                   "std": round(float(np.std(mult)), 4),
                   "min": round(float(min(mult)), 4),
                   "max": round(float(max(mult)), 4)},
    "tabu": {"len": round(tb["best"], 4), "tour": [NAMES[v] for v in tb["tour"]]},
    "tabu_off": {"len": round(tb_off["best"], 4)},
    "tenure_scan": {str(k): round(v, 4) for k, v in ten_scan.items()},
    "accept_diag": {k: round(v, 4) for k, v in diag.items()},
    "lower_bound_1tree": round(float(LB), 4),
    "gap_pct": round((sa["best"] - LB) / LB * 100, 4),
    "distinct_optimal_tours": len(canon),
}
with open(os.path.join(OUT, "day80_result.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("json 已保存")
