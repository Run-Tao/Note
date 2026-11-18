# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 算法（多目标优化 · 加权和 / ε-约束 · Pareto 前沿 · 目标之间的支配关系）
anchor: 接 Day81（单目标 VRPTW → 拉开算法差距）——本讲把「里程 / 准时 / 司机工时」三个目标同时摆上台，
        问的不是「哪个解最好」，而是「哪一组解都值得考虑」
goal: 云溪县冷链夜班配送，沿用 Day81 的 12 个点 + 3 辆车 + 时间窗。三个目标：
      ① 总里程（钱）；② 迟到总分钟（口碑）；③ 司机总工时（工资/疲劳）。
      问：三个目标能不能同时变好？还是必然此消彼长？
      做法：加权和扫描 → ε-约束 → Pareto 前沿 → 各目标「影子价格」→ 支配关系检验
run: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-18/day82_pareto_multiobj.py"
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

for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
SEED = 20261118

# ============================================================
# 0. 数据：完全沿用 Day81（同一张地图、同样的窗），保证两讲可比
# ============================================================
NAMES = ["中心库", "梅溪", "双桥", "石坳", "南坪", "黄泥", "榕树脚",
         "白水", "太平桥", "冷水", "新塘", "竹山", "枫林坳"]
XY = np.array([
    [0.00, 0.00], [2.10, 6.40], [5.80, 3.10], [7.40, 8.90], [4.20, -2.60],
    [-3.50, 3.80], [-6.20, -1.40], [-1.80, -6.90], [3.30, -6.10],
    [9.60, 1.20], [11.20, -4.30], [-8.40, 5.60], [-5.10, -7.20],
])
DEMAND = np.array([0, 18, 24, 12, 20, 15, 10, 22, 16, 14, 19, 9, 11], dtype=float)
CAP_BASKETS = 150.0
N_VEH = 3
SPEED_KMH = 25.0
SERVICE_MIN = 6.0
T_HORIZON = 180.0
LATE_HARD = 120.0          # ε-约束：允许的最大总迟到分钟

TW = np.array([
    [0, 999], [0, 40], [0, 999], [60, 120], [0, 999], [30, 90], [0, 999],
    [0, 999], [0, 999], [0, 60], [70, 150], [40, 100], [0, 999],
], dtype=float)

N = len(XY)
D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))
TRAVEL = D / SPEED_KMH * 60.0

Y_PER_KM = 4.20
Y_PER_MIN_LATE = 18.0
Y_PER_HOUR_LABOR = 42.0     # 司机时薪折算


# ============================================================
# 1. 三目标评估（不做加权，分开报）
# ============================================================
def route_eval(r):
    if not r:
        return 0.0, 0.0, 0.0     # dist, end_time, late
    t = 0.0
    dist = 0.0
    late = 0.0
    prev = 0
    for v in r:
        t += TRAVEL[prev, v]
        dist += D[prev, v]
        lo, hi = TW[v]
        if t < lo:
            t = lo
        if t > hi:
            late += t - hi
        t += SERVICE_MIN
        prev = v
    dist += D[prev, 0]
    t += TRAVEL[prev, 0]
    return dist, t, late


def objectives(sol):
    """返回 (f1 里程 km, f2 迟到分钟, f3 司机总工时 h) 以及是否可行。"""
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    cap_viol = 0.0
    span_viol = 0.0
    for r in sol:
        if not r:
            continue
        ld = float(DEMAND[list(r)].sum())
        if ld > CAP_BASKETS:
            cap_viol += ld - CAP_BASKETS
        d, te, lt = route_eval(r)
        f1 += d
        f2 += lt
        f3 += te / 60.0
        if te > T_HORIZON:
            span_viol += te - T_HORIZON
    return dict(f1=f1, f2=f2, f3=f3, cap_viol=cap_viol, span_viol=span_viol,
                feasible=(cap_viol < 1e-9 and span_viol < 1e-9))


def cover_ok(sol):
    return sorted(v for r in sol for v in r) == list(range(1, N))


# ============================================================
# 2. 邻域（沿用 Day81 三件套）
# ============================================================
def neighbor(sol, rng):
    s = [list(r) for r in sol]
    r = rng.random()
    nonempty = [i for i in range(len(s)) if s[i]]
    if r < 0.45 and nonempty:
        i = rng.choice(nonempty)
        if len(s[i]) >= 2:
            a = int(rng.integers(0, len(s[i]) - 1))
            b = int(rng.integers(a + 1, len(s[i])))
            s[i][a:b + 1] = s[i][a:b + 1][::-1]
        return s
    if r < 0.80 and nonempty:
        i = rng.choice(nonempty)
        a = int(rng.integers(0, len(s[i])))
        v = s[i].pop(a)
        j = int(rng.integers(0, len(s)))
        s[j].insert(int(rng.integers(0, len(s[j]) + 1)), v)
        return s
    if len(nonempty) >= 2:
        i, j = rng.choice(nonempty, size=2, replace=False)
        a = int(rng.integers(0, len(s[i])))
        b = int(rng.integers(0, len(s[j])))
        s[i][a], s[j][b] = s[j][b], s[i][a]
    return s


def scalar_cost(o, w):
    """加权和：w=(w1,w2,w3)，全部归一到「元」再相加。"""
    return (w[0] * o["f1"] * Y_PER_KM
            + w[1] * o["f2"] * Y_PER_MIN_LATE
            + w[2] * o["f3"] * Y_PER_HOUR_LABOR
            + 1e4 * (o["cap_viol"] + o["span_viol"]))


def optimize(sol0, weights, iters=9000, seed=SEED, eps_late=None):
    """加权和 SA；eps_late 给定时叠加 ε-约束（超过 ε 的解重罚）。"""
    rng = np.random.default_rng(seed)
    cur = [list(r) for r in sol0]
    cur_o = objectives(cur)
    cur_c = scalar_cost(cur_o, weights)
    if eps_late is not None and cur_o["f2"] > eps_late:
        cur_c += 1e3 * (cur_o["f2"] - eps_late)
    best_c, best = cur_c, [list(r) for r in cur]
    best_o = cur_o
    temp = 60.0
    cool = 1e-4 ** (1.0 / iters)
    for _ in range(iters):
        cand = neighbor(cur, rng)
        if not cover_ok(cand):
            continue
        o = objectives(cand)
        c = scalar_cost(o, weights)
        if eps_late is not None and o["f2"] > eps_late:
            c += 1e3 * (o["f2"] - eps_late)
        d = c - cur_c
        if d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
            cur, cur_c, cur_o = cand, c, o
            if c < best_c - 1e-9:
                best_c, best, best_o = c, [list(r) for r in cand], o
        temp *= cool
    return dict(sol=best, o=best_o, cost=best_c)


def scan_split():
    ang = np.arctan2(XY[1:, 1], XY[1:, 0])
    order = [int(v) for v in (np.argsort(ang) + 1)]
    routes = [[] for _ in range(N_VEH)]
    k = 0
    for v in order:
        if float(DEMAND[routes[k]].sum()) + DEMAND[v] > CAP_BASKETS and k < N_VEH - 1:
            k += 1
        routes[k].append(v)
    return routes


# ============================================================
# 3. Pareto 工具
# ============================================================
def dominates(a, b):
    """a 支配 b：a 三项都不差，且至少一项严格更好。"""
    return (a[0] <= b[0] + 1e-9 and a[1] <= b[1] + 1e-9 and a[2] <= b[2] + 1e-9
            and (a[0] < b[0] - 1e-9 or a[1] < b[1] - 1e-9 or a[2] < b[2] - 1e-9))


def pareto_filter(pts):
    """pts: [(f1,f2,f3,payload)] → 非支配集。"""
    keep = []
    for p in pts:
        if any(dominates(q[:3], p[:3]) for q in pts):
            continue
        # 去重（三项都相同）
        if any(all(abs(p[i] - q[i]) < 1e-6 for i in range(3)) for q in keep):
            continue
        keep.append(p)
    return keep


# ============================================================
# 主流程
# ============================================================
print("=" * 78)
print("云溪县冷链夜班配送 · 三目标（里程 / 迟到 / 司机工时）")
print("=" * 78)
base = scan_split()
ob = objectives(base)
print(f"  起点：里程 {ob['f1']:.2f} km，迟到 {ob['f2']:.1f} 分，工时 {ob['f3']:.2f} h，"
      f"可行 {'✓' if ob['feasible'] else '✗'}")
print(f"  换算：{Y_PER_KM:.2f} 元/km，{Y_PER_MIN_LATE:.0f} 元/迟到分，{Y_PER_HOUR_LABOR:.0f} 元/司机时")

# ---------- [1] 加权和扫描：三个目标各偏向一次 ----------
print("\n[1] 加权和扫描（每项归一后加权，权重和 = 1）")
scan = []
for w1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    w2 = 1.0 - w1
    # 第三目标固定小权重，观察 f1 / f2 的此消彼长
    w = (w1, w2 * 0.8, w2 * 0.2)
    r = optimize(base, w, iters=7000)
    o = r["o"]
    scan.append((round(w1, 2), round(o["f1"], 3), round(o["f2"], 3), round(o["f3"], 3),
                 o["feasible"]))
    print(f"  w=(里程{w[0]:.2f}, 迟到{w[1]:.2f}, 工时{w[2]:.2f}) → "
          f"里程 {o['f1']:7.2f} km | 迟到 {o['f2']:6.1f} 分 | 工时 {o['f3']:6.2f} h | "
          f"{'✓' if o['feasible'] else '✗'}")

# ---------- [2] ε-约束：给迟到设上限 ----------
print("\n[2] ε-约束：硬性限制总迟到 ≤ ε，看里程最少付出多少")
eps_scan = []
for eps in [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, None]:
    w = (1.0, 0.0, 0.0)    # 纯里程优先
    r = optimize(base, w, iters=8000, eps_late=eps)
    o = r["o"]
    tag = "无约束" if eps is None else f"ε={eps:5.1f}"
    eps_scan.append(dict(eps=eps, f1=round(o["f1"], 3), f2=round(o["f2"], 3),
                         f3=round(o["f3"], 3), feasible=bool(o["feasible"])))
    print(f"  {tag} → 里程 {o['f1']:7.2f} km | 实迟到 {o['f2']:6.1f} 分 | 工时 {o['f3']:6.2f} h")

# ---------- [3] Pareto 前沿：多组权重汇总 + 非支配筛选 ----------
print("\n[3] Pareto 前沿筛选（把上面所有解 + 若干随机权重解一起丢进去）")
pool = []
for w1 in [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
    for w3 in [0.0, 0.15, 0.35]:
        rest = max(0.0, 1.0 - w1 - w3)
        w = (w1, rest, w3)
        r = optimize(base, w, iters=4000)
        o = r["o"]
        if o["feasible"]:
            pool.append([o["f1"], o["f2"], o["f3"], r["sol"]])
print(f"  可行解池 {len(pool)} 个")
front = pareto_filter(pool)
front.sort(key=lambda p: p[0])
print(f"  非支配解 {len(front)} 个：")
for i, p in enumerate(front):
    print(f"    #{i+1}  里程 {p[0]:7.2f} km | 迟到 {p[1]:6.1f} 分 | 工时 {p[2]:6.2f} h")

# ---------- [4] 各目标的「影子价格」：多要 1 分钟准时，要付多少里程 ----------
print("\n[4] 影子价格：从最省油的解走向最准时的解，每秒准时多少钱")
if len(front) >= 2:
    a, b = front[0], front[-1]
    d_km = b[0] - a[0]
    d_late = a[1] - b[1]
    if d_late > 1e-6:
        km_per_min = d_km / d_late
        print(f"  最省里程解（{a[0]:.2f} km, 迟到 {a[1]:.1f} 分）↔ 最准时解（{b[0]:.2f} km, 迟到 {b[1]:.1f} 分）")
        print(f"  → 每减少 1 分钟迟到，平均多跑 {km_per_min:.3f} km "
              f"（≈ {km_per_min*Y_PER_KM:.2f} 元）")
    else:
        km_per_min = None
        print("  两个端点迟到相同，无法算斜率")
else:
    km_per_min = None
    print("  前沿点不足 2 个")

# ---------- [5] 支配关系检验：Day81 的解是「被支配」的吗 ----------
print("\n[5] Day81 的单目标最优解，在三目标下是否被支配")
day81 = optimize(base, (0.0, 1.0, 0.0), iters=9000)   # 只求准时（Day81 的主目标）
o81 = day81["o"]
print(f"  Day81 口径解：里程 {o81['f1']:.2f} | 迟到 {o81['f2']:.1f} | 工时 {o81['f3']:.2f}")
dominated_by = [p for p in front if dominates(p[:3], [o81["f1"], o81["f2"], o81["f3"]])]
if dominated_by:
    q = min(dominated_by, key=lambda p: p[0])
    print(f"  → **被支配**：前沿上有解（里程 {q[0]:.2f}, 迟到 {q[1]:.1f}, 工时 {q[2]:.2f}）三项全不差")
else:
    print("  → 未被支配：Day81 的解本身就在 Pareto 前沿上")

# ---------- [6] 三目标两两散点：看相关性 ----------
print("\n[6] 目标之间的相关性（在可行解池上）")
arr = np.array([p[:3] for p in pool])
c12 = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
c13 = np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]
c23 = np.corrcoef(arr[:, 1], arr[:, 2])[0, 1]
print(f"  corr(里程, 迟到) = {c12:+.3f}")
print(f"  corr(里程, 工时) = {c13:+.3f}")
print(f"  corr(迟到, 工时) = {c23:+.3f}")
print("  读法：负相关 = 此消彼长（真冲突）；正相关 = 同向（可以一起改善，不是真冲突）")

# ============================================================
# 出图
# ============================================================
fig = plt.figure(figsize=(13.5, 10.0))
ax1 = fig.add_subplot(2, 2, 1)
allp = np.array([p[:3] for p in pool])
ax1.scatter(allp[:, 0], allp[:, 1], s=22, c="#B279A2", alpha=0.55, label=f"可行解池 {len(pool)}")
fp = np.array([p[:3] for p in front])
ax1.plot(fp[:, 0], fp[:, 1], "-o", lw=1.8, ms=7, color="#E45756", label=f"Pareto 前沿 {len(front)}")
ax1.set_xlabel("总里程 (km)")
ax1.set_ylabel("总迟到 (分)")
ax1.set_title("(a) 里程 vs 迟到：前沿上的此消彼长")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(allp[:, 0], allp[:, 2], s=22, c="#72B7B2", alpha=0.55)
ax2.scatter(fp[:, 0], fp[:, 2], s=70, c="#E45756", marker="D", zorder=5, label="前沿解")
ax2.set_xlabel("总里程 (km)")
ax2.set_ylabel("司机总工时 (h)")
ax2.set_title(f"(b) 里程 vs 工时（corr={c13:+.2f}）")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ep = [e for e in eps_scan]
xs = [e["eps"] if e["eps"] is not None else 200 for e in ep]
ax3.plot(xs, [e["f1"] for e in ep], "-o", color="#4C78A8", label="最优里程 km")
ax3.axvline(200, ls="--", color="gray", lw=1.2, label="无约束")
ax3.set_xlabel("允许多少迟到（ε，分）")
ax3.set_ylabel("最优里程 (km)")
ax3.set_title("(c) ε-约束：放宽迟到能省多少里程")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
if front:
    best = min(front, key=lambda p: p[0] + 0.5 * p[1] + 5.0 * p[2])
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    for i, r in enumerate(best[3]):
        if not r:
            continue
        path = [0] + list(r) + [0]
        ax4.plot(XY[path, 0], XY[path, 1], "-o", lw=1.8, ms=5, color=colors[i % 3],
                 label=f"车{i+1}")
    ax4.plot(0, 0, "s", ms=13, color="black")
    for v in range(N):
        ax4.annotate(NAMES[v], (XY[v, 0], XY[v, 1]), fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
    ax4.set_title(f"(d) 折中解 {best[0]:.1f}km / 迟{best[1]:.0f}分 / {best[2]:.1f}h")
    ax4.legend(fontsize=8)
ax4.set_xlabel("x (km)")
ax4.set_ylabel("y (km)")
ax4.grid(alpha=0.3)

fig.suptitle("Day82 · 三目标配送：里程 / 准时 / 工时 —— 谁和谁真的冲突", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "day82_pareto_multiobj.png"), dpi=130)

# ============================================================
# 断言（设计公平性）
# ============================================================
assert cover_ok(base), "起点不覆盖全部点"
assert len(front) >= 1, "Pareto 前沿为空"
for p in front:
    assert not any(dominates(q[:3], p[:3]) for q in front if q is not p), "前沿里有被支配点"
assert all(e["f1"] > 0 for e in eps_scan), "ε 扫描出现非正里程"
print(f"\n断言全部通过；用时 {time.time()-T0:.1f}s")

payload = {
    "params": {"Y_PER_KM": Y_PER_KM, "Y_PER_MIN_LATE": Y_PER_MIN_LATE,
               "Y_PER_HOUR_LABOR": Y_PER_HOUR_LABOR,
               "N_VEH": N_VEH, "CAP_BASKETS": CAP_BASKETS},
    "baseline": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                     round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in ob.items()},
    "weight_scan": [{"w1": s[0], "f1": s[1], "f2": s[2], "f3": s[3], "feasible": bool(s[4])}
                    for s in scan],
    "eps_scan": [{"eps": e["eps"], "f1": e["f1"], "f2": e["f2"], "f3": e["f3"],
                  "feasible": bool(e["feasible"])} for e in eps_scan],
    "pareto_front": [{"f1": round(p[0], 4), "f2": round(p[1], 4), "f3": round(p[2], 4)}
                     for p in front],
    "pool_size": len(pool),
    "km_per_min_of_punctuality": round(float(km_per_min), 4) if km_per_min else None,
    "day81_solution": {"f1": round(o81["f1"], 4), "f2": round(o81["f2"], 4),
                       "f3": round(o81["f3"], 4), "dominated": bool(len(dominated_by) > 0)},
    "correlations": {"f1_f2": round(float(c12), 4), "f1_f3": round(float(c13), 4),
                     "f2_f3": round(float(c23), 4)},
}
with open(os.path.join(OUT, "day82_result.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("json 已保存")
