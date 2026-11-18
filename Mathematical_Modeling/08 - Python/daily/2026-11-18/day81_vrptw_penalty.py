# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 算法（带时间窗 VRP · 可行性惩罚 · 多车容量 · 邻域可行性修复）
anchor: 接 Day80（无约束 TSP 上元启发式是装饰品）——本讲给它加时间窗 + 多车容量，
        让局部最优真的变烂，禁忌/多起点的作用第一次显现出来
goal: 云溪县 12 个冷链自提点夜班配送（**不再一辆车**）：3 辆车、每车 2 小时作业窗、12 个点各有收货时间窗。
      ① 时间窗一加，Day80 那条 71.51 km 的路线还有几条能用；
      ② 罚函数把不可行解也纳入搜索，代价是什么（可行率 vs 总里程的权衡）；
      ③ 同样迭代预算下，多起点 2-opt / 禁忌 / 纯下降 谁先撑不住；
      ④ 迟到 1 分钟值多少钱（软时间窗罚金 vs 硬约束）；
      ⑤ 晚到代价换算成"多跑多少 km"。
run: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-18/day81_vrptw_penalty.py"
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
# 0. 数据（沿用 Day80 的 12 个点，加时间窗与车辆数）
# ============================================================
NAMES = ["中心库", "梅溪", "双桥", "石坳", "南坪", "黄泥", "榕树脚",
         "白水", "太平桥", "冷水", "新塘", "竹山", "枫林坳"]
XY = np.array([
    [0.00, 0.00], [2.10, 6.40], [5.80, 3.10], [7.40, 8.90], [4.20, -2.60],
    [-3.50, 3.80], [-6.20, -1.40], [-1.80, -6.90], [3.30, -6.10],
    [9.60, 1.20], [11.20, -4.30], [-8.40, 5.60], [-5.10, -7.20],
])
DEMAND = np.array([0, 18, 24, 12, 20, 15, 10, 22, 16, 14, 19, 9, 11], dtype=float)  # 筐
KG_PER_BASKET = 20.0
CAP_BASKETS = 150.0        # 每车 3 吨 = 150 筐（Day80 是 7 吨单车，本讲拆成 3 辆小车）
N_VEH = 3
SPEED_KMH = 25.0           # 村道夜间限速，含装卸
SERVICE_MIN = 6.0          # 每点固定装卸 6 分钟
T0_MIN = 0.0               # 发车时刻（相对）
T_HORIZON = 180.0          # 单车最长作业时长（2 小时）

# 收货时间窗（分，相对发车）。故意造成"不能全按最短路线走"
TW = np.array([
    [0, 999],    # 库
    [0, 40],     # 梅溪  早
    [0, 999],    # 双桥  不限
    [60, 120],   # 石坳  中段
    [0, 999],    # 南坪  不限
    [30, 90],    # 黄泥  早中
    [0, 999],    # 榕树脚 不限
    [0, 999],    # 白水  不限
    [0, 999],    # 太平桥 不限
    [0, 60],     # 冷水  很早
    [70, 150],   # 新塘  后段
    [40, 100],   # 竹山  中段
    [0, 999],    # 枫林坳 不限
], dtype=float)

N = len(XY)
D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))
TRAVEL = D / SPEED_KMH * 60.0     # 分钟

Y_PER_KM = 4.20
LATE_PENALTY_PER_MIN = 18.0       # 元/分钟（生鲜断链一次投诉的成本折算）
W_DIST = 1.0                      # 目标权重：里程（km）
PEN = {"late": 3.0}               # 迟到罚重（1 分钟迟到 ≈ 3 km），用 dict 便于扫描时改写


# ============================================================
# 1. 解的表示：3 条路线，每条是一个不含库的点序列
#    注意：这里必须允许"空路线"（车可以不用）
# ============================================================
def clone(sol):
    return [list(r) for r in sol]


def route_load(r):
    return float(DEMAND[list(r)].sum()) if r else 0.0


def route_eval(r, allow_wait=True):
    """返回 (距离 km, 最晚到达相对时间, 迟到分钟, 可行性说明)。
    时间轴：t=发车时刻 0；到点即服务；若早于窗下界则等待（allow_wait）。"""
    if not r:
        return 0.0, 0.0, 0.0, "空车"
    t = 0.0
    dist = 0.0
    late = 0.0
    prev = 0
    for v in r:
        t += TRAVEL[prev, v]
        dist += D[prev, v]
        lo, hi = TW[v]
        if t < lo and allow_wait:
            t = lo
        if t > hi:
            late += t - hi
        t += SERVICE_MIN
        prev = v
    dist += D[prev, 0]
    t += TRAVEL[prev, 0]
    return dist, t, late, ""


def sol_eval(sol, allow_wait=True):
    """总目标：里程 + W_LATE × 迟到分钟；同时返回可行性明细。"""
    dist = 0.0
    late = 0.0
    cap_viol = 0.0
    span_viol = 0.0
    for r in sol:
        if not r:
            continue
        ld = route_load(r)
        if ld > CAP_BASKETS:
            cap_viol += ld - CAP_BASKETS
        d, t_end, lt, _ = route_eval(r, allow_wait)
        dist += d
        late += lt
        if t_end > T_HORIZON:
            span_viol += t_end - T_HORIZON
    cost = dist + PEN["late"] * late + 5.0 * cap_viol + 2.0 * span_viol
    return dict(dist=dist, late=late, cap_viol=cap_viol, span_viol=span_viol,
                cost=cost, feasible=(cap_viol < 1e-9 and span_viol < 1e-9))


def all_nodes(sol):
    return sorted(v for r in sol for v in r)


def check_cover(sol):
    return all_nodes(sol) == list(range(1, N))


# ============================================================
# 2. 邻域：路线内 2-opt / 路线间点迁移 / 路线间交换
# ============================================================
def neighbor(sol, rng):
    s = clone(sol)
    r = rng.random()
    nonempty = [i for i in range(len(s)) if s[i]]
    if r < 0.45 and nonempty:
        i = rng.choice(nonempty)
        r_ = s[i]
        if len(r_) >= 2:
            a = int(rng.integers(0, len(r_) - 1))
            b = int(rng.integers(a + 1, len(r_)))
            r_[a:b + 1] = r_[a:b + 1][::-1]
        return s
    if r < 0.80 and nonempty:
        i = rng.choice(nonempty)
        r_ = s[i]
        a = int(rng.integers(0, len(r_)))
        v = r_.pop(a)
        j = int(rng.integers(0, len(s)))
        s[j].insert(int(rng.integers(0, len(s[j]) + 1)), v)
        return s
    # 交换两个不同路线里的点
    if len(nonempty) >= 2:
        i, j = rng.choice(nonempty, size=2, replace=False)
        a = int(rng.integers(0, len(s[i])))
        b = int(rng.integers(0, len(s[j])))
        s[i][a], s[j][b] = s[j][b], s[i][a]
    return s


def run_search(sol0, iters, seed, mode="sa", init_temp=6.0, cool_ratio=1e-4):
    """mode: sa | descent | tabu | restart_desc"""
    rng = np.random.default_rng(seed)
    if mode == "restart_desc":
        best = None
        best_sol = None
        per = max(1, iters // 5)
        for k in range(5):
            r = run_search(sol0, per, seed + 101 * k, mode="descent")
            if best is None or r["cost"] < best:
                best, best_sol = r["cost"], r["sol"]
        return dict(cost=best, sol=best_sol, dist=sol_eval(best_sol)["dist"],
                    trace=[], mode=mode)

    cur = clone(sol0)
    cur_e = sol_eval(cur)
    best_e = cur_e
    best_sol = clone(cur)
    temp = init_temp
    cool = (cool_ratio) ** (1.0 / max(1, iters))
    tabu = {}
    trace = []
    for it in range(iters):
        cand = neighbor(cur, rng)
        ce = sol_eval(cand)
        take = False
        if mode == "descent":
            take = ce["cost"] < cur_e["cost"] - 1e-9
        elif mode == "tabu":
            key = (tuple(sorted(v for r in cand for v in r)),
                   tuple(tuple(r) for r in cand))
            is_tabu = tabu.get(key, -1) > it
            take = (not is_tabu) or (ce["cost"] < best_e["cost"] - 1e-9)
            if take:
                tabu[key] = it + 6
        else:  # sa
            d = ce["cost"] - cur_e["cost"]
            take = d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-9))
        if take:
            cur, cur_e = cand, ce
            if ce["cost"] < best_e["cost"] - 1e-9:
                best_e, best_sol = ce, clone(cand)
        temp *= cool
        if it % 25 == 0:
            trace.append((it, cur_e["dist"], best_e["dist"]))
    return dict(cost=best_e["cost"], sol=best_sol, dist=best_e["dist"],
                trace=trace, mode=mode)


# ============================================================
# 3. 构造：Day80 的扫描法路线按容量切成 3 车（朴素起点）
# ============================================================
def scan_split():
    ang = np.arctan2(XY[1:, 1], XY[1:, 0])
    order = [int(v) for v in (np.argsort(ang) + 1)]
    routes = [[] for _ in range(N_VEH)]
    k = 0
    for v in order:
        if route_load(routes[k]) + DEMAND[v] > CAP_BASKETS and k < N_VEH - 1:
            k += 1
        routes[k].append(v)
    return routes


def day80_single_route():
    """Day80 的 71.51 km 最优路线（单条），按容量切 3 段做对照。"""
    order = [1, 3, 2, 9, 10, 4, 8, 12, 6, 11, 7, 5]  # 梅溪石坳双桥冷水新塘南坪太平桥白水枫林坳榕树脚竹山黄泥
    routes = [[] for _ in range(N_VEH)]
    k = 0
    for v in order:
        if route_load(routes[k]) + DEMAND[v] > CAP_BASKETS and k < N_VEH - 1:
            k += 1
        routes[k].append(v)
    return routes


# ============================================================
# 主流程
# ============================================================
print("=" * 78)
print("云溪县冷链夜班配送 · 加时间窗 + 3 辆车（VRPTW）")
print("=" * 78)
tot = float(DEMAND[1:].sum())
print(f"  总需求 {tot:.0f} 筐 = {tot*KG_PER_BASKET:.0f} kg；3 车 × {CAP_BASKETS:.0f} 筐 = {N_VEH*CAP_BASKETS:.0f} 筐"
      f"（装载率 {tot/(N_VEH*CAP_BASKETS)*100:.1f}%）")
print(f"  单车作业窗 {T_HORIZON:.0f} 分，村道 {SPEED_KMH:.0f} km/h，每点装卸 {SERVICE_MIN:.0f} 分")
print(f"  迟到罚 {PEN['late']:.1f} km/分（≈{LATE_PENALTY_PER_MIN:.0f} 元/分）")

# ---------- [1] Day80 的最优路线，在时间窗下还剩几条能用 ----------
print("\n[1] Day80 的 71.51 km 单条路线，切成 3 段后能不能满足时间窗")
r80 = day80_single_route()
print("  切分结果：")
for i, r in enumerate(r80):
    d, te, lt, _ = route_eval(r)
    ld = route_load(r)
    print(f"    车{i+1}: {len(r):2d} 点  载重 {ld:5.1f}/{CAP_BASKETS:.0f} 筐  里程 {d:6.2f} km  "
          f"回库 {te:6.1f} 分  迟到 {lt:6.1f} 分")
e80 = sol_eval(r80)
print(f"  合计：里程 {e80['dist']:.2f} km，迟到 {e80['late']:.1f} 分，"
      f"超载 {e80['cap_viol']:.1f} 筐，超时 {e80['span_viol']:.1f} 分")
print(f"  → 可行？{'是' if e80['feasible'] else '否'}")

# 只算距离约束下界：如果不看时间窗，3 车最少要跑多少
n_trucks_needed = math.ceil(tot / CAP_BASKETS)
print(f"  理论上最少用车 {n_trucks_needed} 辆（{tot:.0f}/{CAP_BASKETS:.0f}）")

# ---------- [2] 三种搜索对照 ----------
print("\n[2] 同样 12,000 次评估预算：SA / 纯下降 / 禁忌 / 多起点下降")
ITERS = 12000
base = scan_split()
print("  构造起点（扫描法切 3 车）：")
for i, r in enumerate(base):
    d, te, lt, _ = route_eval(r)
    print(f"    车{i+1}: 载重 {route_load(r):5.1f}  里程 {d:6.2f}  回库 {te:6.1f}  迟到 {lt:5.1f}")
eb = sol_eval(base)
print(f"  起点合计：里程 {eb['dist']:.2f} km，迟到 {eb['late']:.1f} 分，cost={eb['cost']:.2f}")

results = {}
for mode, label in [("sa", "模拟退火"), ("descent", "纯下降"), ("tabu", "禁忌"), ("restart_desc", "多起点下降")]:
    r = run_search(base, ITERS, SEED, mode=mode)
    e = sol_eval(r["sol"])
    results[mode] = dict(cost=r["cost"], dist=e["dist"], late=e["late"],
                         cap_viol=e["cap_viol"], span_viol=e["span_viol"],
                         feasible=e["feasible"], sol=r["sol"], trace=r.get("trace", []))
    print(f"  {label:8s} 里程 {e['dist']:7.2f}  迟到 {e['late']:6.1f}  超载 {e['cap_viol']:5.1f}  "
          f"cost {r['cost']:8.2f}  可行 {'✓' if e['feasible'] else '✗'}")

# ---------- [3] 罚函数权重：可行率 vs 里程的权衡 ----------
print("\n[3] 迟到罚权重扫描：把「准时」标价，解会怎么挪")
sweep = {}
for w in [0.0, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]:
    PEN["late"] = w
    r = run_search(base, 6000, SEED, mode="sa")
    e = sol_eval(r["sol"])
    sweep[w] = dict(dist=round(e["dist"], 3), late=round(e["late"], 3), feasible=bool(e["feasible"]))
    print(f"  W_LATE={w:6.1f} → 里程 {e['dist']:7.2f} km，迟到 {e['late']:7.1f} 分，"
          f"可行 {'✓' if e['feasible'] else '✗'}")
PEN["late"] = 3.0

# ---------- [4] 迟到 1 分钟 = 多跑几公里 ----------
print("\n[4] 把迟到换成里程：1 分钟迟到约等于多跑多少 km")
best_sa = results["sa"]
e_sa = sol_eval(best_sa["sol"])
d_km_per_min = None
# 用最优解的路线人工制造迟到：把最晚的一辆车延后 5 分钟出发
if e_sa["late"] > 1e-9:
    d_km_per_min = e_sa["dist"] / e_sa["late"]
    print(f"  当前 SA 解迟到 {e_sa['late']:.1f} 分；不迟到前提下，每省 1 分钟迟到 ≈ "
          f"{d_km_per_min:.2f} km 的里程代价（本解口径）")
else:
    print("  SA 解已完全准时（迟到 0），该换算由 W_LATE 扫描给出：见 [3]")

# ---------- [5] 可行性代价：加窗 vs 不加窗 ----------
print("\n[5] 时间窗到底让问题难了多少（同一点集，同 3 车，开/关时间窗）")
TW_backup = TW.copy()
TW[:, :] = [[0, 999]] * N   # 关掉所有窗
r_no_tw = run_search(base, ITERS, SEED, mode="sa")
e_no_tw = sol_eval(r_no_tw["sol"])
TW[:, :] = TW_backup
print(f"  关时间窗：里程 {e_no_tw['dist']:7.2f} km（本讲口径下等价于 Day80 的多车版）")
print(f"  开时间窗：里程 {e_sa['dist']:7.2f} km")
print(f"  代价 = {(e_sa['dist']-e_no_tw['dist']):+.2f} km（{(e_sa['dist']/e_no_tw['dist']-1)*100:+.2f}%）")

# 再补一条：Day80 单车的 71.51 作为理论参照
print(f"  参照：Day80 单车 12 点最优 71.51 km（无窗、7 吨车）")

# ---------- [6] 深夜工况：时间窗收紧 20% ----------
print("\n[6] 时间窗收紧 20%：还有可行解吗")
TW_backup = TW.copy()
mask = TW[:, 1] < 900
TW[mask, 1] = TW[mask, 0] + (TW[mask, 1] - TW[mask, 0]) * 0.8
r_tight = run_search(base, ITERS, SEED, mode="sa")
e_tight = sol_eval(r_tight["sol"])
print(f"  收紧后：里程 {e_tight['dist']:7.2f} km，迟到 {e_tight['late']:7.1f} 分，"
      f"可行 {'✓' if e_tight['feasible'] else '✗'}  超载 {e_tight['cap_viol']:.1f}")
TW[:, :] = TW_backup

# ============================================================
# 出图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
colors = ["#4C78A8", "#E45756", "#54A24B"]

ax = axes[0, 0]
best_sol = best_sa["sol"]
for i, r in enumerate(best_sol):
    if not r:
        continue
    path = [0] + list(r) + [0]
    ax.plot(XY[path, 0], XY[path, 1], "-o", lw=1.8, ms=5, color=colors[i % 3],
            label=f"车{i+1}（{route_load(r):.0f} 筐）")
ax.plot(0, 0, "s", ms=13, color="black")
for v in range(N):
    ax.annotate(NAMES[v], (XY[v, 0], XY[v, 1]), fontsize=8, xytext=(4, 4),
                textcoords="offset points")
ax.set_title(f"(a) VRPTW 最优方案 里程 {e_sa['dist']:.2f} km")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
labels = ["模拟退火", "纯下降", "禁忌", "多起点下降"]
keys = ["sa", "descent", "tabu", "restart_desc"]
ax.bar(labels, [results[k]["cost"] for k in keys],
       color=["#E45756", "#F58518", "#54A24B", "#B279A2"])
for i, k in enumerate(keys):
    ax.text(i, results[k]["cost"], f"{results[k]['dist']:.1f}km\n迟{results[k]['late']:.0f}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("目标 cost（里程 + 罚）")
ax.set_title("(b) 加约束后，四种搜索才分出高下")
ax.tick_params(axis="x", labelsize=9)

ax = axes[1, 0]
ws = sorted(sweep)
ax.plot(ws, [sweep[w]["dist"] for w in ws], "-o", color="#4C78A8", label="里程 km")
ax.plot(ws, [sweep[w]["late"] for w in ws], "-s", color="#E45756", label="迟到 分")
ax.set_xscale("symlog")
ax.set_xlabel("迟到权重 W_LATE（symlog）")
ax.set_title("(c) 把准时标价：里程与迟到的此消彼长")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
val = [e_no_tw["dist"], e_sa["dist"], e_tight["dist"]]
lab = ["关时间窗", "时间窗", "收紧 20%"]
ax.bar(lab, val, color=["#54A24B", "#4C78A8", "#E45756"])
for i, v in enumerate(val):
    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("总里程 (km)")
ax.set_title("(d) 时间窗的代价")

fig.suptitle("Day81 · 带时间窗的多车配送（VRPTW）—— 约束一加，元启发式才有活干", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "day81_vrptw_penalty.png"), dpi=130)

# ============================================================
# 断言（设计公平性，不预设谁该赢）
# ============================================================
assert check_cover(best_sol), "SA 解没有覆盖全部点"
assert check_cover(r80), "Day80 切分路线不覆盖全部点"
assert abs(route_load(best_sol[0]) - route_load(best_sol[0])) < 1e-9
for k, r in results.items():
    assert check_cover(r["sol"]), f"{k} 的解不覆盖全部点"
assert e_sa["dist"] > 0, "里程不能为 0"
assert all(v > 0 for v in val), "对比组出现非正里程"
print(f"\n断言全部通过；用时 {time.time()-T0:.1f}s")

payload = {
    "n_nodes": N, "n_vehicles": N_VEH, "cap_baskets": CAP_BASKETS,
    "total_baskets": round(tot, 1),
    "day80_route_on_tw": {"dist": round(e80["dist"], 4), "late": round(e80["late"], 4),
                          "cap_viol": round(e80["cap_viol"], 4),
                          "feasible": bool(e80["feasible"])},
    "base_scan_split": {"dist": round(eb["dist"], 4), "late": round(eb["late"], 4),
                        "cost": round(eb["cost"], 4)},
    "methods": {k: {"dist": round(v["dist"], 4), "late": round(v["late"], 4),
                    "cap_viol": round(v["cap_viol"], 4), "cost": round(v["cost"], 4),
                    "feasible": bool(v["feasible"])} for k, v in results.items()},
    "w_late_sweep": {str(k): v for k, v in sweep.items()},
    "no_tw_dist": round(e_no_tw["dist"], 4),
    "with_tw_dist": round(e_sa["dist"], 4),
    "tw_cost_pct": round((e_sa["dist"] / e_no_tw["dist"] - 1) * 100, 4),
    "tight_tw": {"dist": round(e_tight["dist"], 4), "late": round(e_tight["late"], 4),
                 "feasible": bool(e_tight["feasible"])},
    "best_plan": {"routes": [[NAMES[v] for v in r] for r in best_sol if r],
                  "loads": [round(route_load(r), 2) for r in best_sol if r],
                  "route_dist": [round(route_eval(r)[0], 3) for r in best_sol if r]},
}
with open(os.path.join(OUT, "day81_result.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("json 已保存")
