# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L6 优化模型 / L8 算法（CVRP · VRPTW · 车辆装箱下界 · 局部搜索 · 时间窗诊断）
anchor      : Roadmap L6 路径优化（接 Day59 选址+车次 ILP、Day60 覆盖/中值、Day14 最短路）
goal        : 云溪县集配中心凌晨配送（奶+菜）。18 家门店、5:30 发车。回答四件事：
              ① 「最少几辆车」——容量算式说不小于 6，装箱下界却卡在 7，这 1 辆卡在哪
              ② 「按片区各跑各的」这条最像常识的做法，贵的是车还是路
              ③ 门店提出「7:30 前必须送到」——不动车能答应吗？哪些店的窗口让其他店的货根本装不进来
              ④ 「再买一台 40 箱的大车」值不值——车少 1 台但路多了，到底省多少
              单路线里程用 Held-Karp 精确 DP 核对；结论全部由 .venv 实跑。
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-30/day61_vrptw_milkrun.py"
输出：day61_result.json + day61_vrp.png（四面板）
"""

import itertools
import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.simplefilter("always")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day61_result.json")
OUT_PNG = os.path.join(HERE, "day61_vrp.png")

# ---------------------------------------------------------------- 中文字体
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
]
_installed = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in CJK_CANDIDATES if n in _installed), None)
if CJK is not None:
    plt.rcParams["font.sans-serif"] = [CJK] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 场景数据
# 节点 0 = 集配中心；1..19 = 门店。坐标 km。窗口都是「最早 / 最晚」卸货时刻（小时）。
STORES = [
    # 名称,          x,     y,  需求, 最早,  最晚
    ("老城菜场",      2.0,   1.2,   9, 7.0, 8.0),   # 早市 7 点后摆摊，8 点前上齐
    ("城关小学",      1.4,   2.6,   8, 5.5, 7.5),   # 早餐窗口 7:30 关
    ("县医院食堂",    3.1,   2.2,  14, 5.5, 7.5),   # 早餐窗口 7:30 关
    ("河东超市",      5.6,   3.4,  10, 5.5, 8.0),
    ("工业园食堂",    7.8,   1.6,  16, 5.5, 8.5),
    ("高铁站店",      9.4,   3.8,   9, 5.5, 9.0),
    ("石桥中学",     -5.2,   4.6,  10, 5.5, 6.7),   # 食堂 6:40 前必须到岗
    ("石桥超市",     -6.4,   3.2,  13, 5.5, 7.5),
    ("柳湾小学",     -8.6,   6.4,   7, 5.5, 8.0),
    ("柳湾饭店",     -9.8,   5.0,   9, 5.5, 8.5),
    ("云岭乡店",     -4.0,   8.2,   6, 5.5, 9.0),
    ("坳上小学",     -9.0,   9.6,   8, 5.5, 9.0),
    ("河湾村店",     -2.4,  10.6,   5, 5.5, 9.5),
    ("城南市场",      1.2,  -3.4,  15, 5.5, 8.0),
    ("南站食堂",      3.6,  -4.8,  12, 5.5, 8.5),
    ("云岭茶园",     -6.8,  11.4,   6, 5.5, 9.5),
    ("高速服务区",    9.6,  -2.6,  10, 7.5, 8.0),   # 卡口 7:30 才放行重型车，8:00 后不收
    ("河湾码头",     -2.4,  10.6,   5, 8.0, 8.8),   # 渡船 8:00 才靠岸
    ("坳上饭店",    -11.2,   8.0,   7, 5.5, 8.5),
]
NAMES = [s[0] for s in STORES]
N = len(STORES)
DEMAND = np.array([s[3] for s in STORES], float)
READY = np.array([s[4] for s in STORES], float)
DUE = np.array([s[5] for s in STORES], float)
XY = np.vstack([[[0.0, 0.0]], [[s[1], s[2]] for s in STORES]])
DIST = np.linalg.norm(XY[:, None, :] - XY[None, :, :], axis=2) * 1.25   # 路网绕行系数
SPEED = 35.0
TIME = DIST / SPEED

CAP = 30.0           # 单车装载上限（箱）：一台冷链轻卡
EXTRA_CAP = 32.0     # 「只买大 2 箱的同款车」的对照容量
SERVICE = 9.0 / 60.0
FIXED = 400.0        # 每次出车固定成本（司机凌晨补贴 + 折旧 + 预冷）
PER_KM = 4.6         # 元/km
LATE_RATE = 300.0    # 违约金 元/小时/店
HARD = 1e4           # 字典序罚：1 分钟迟到 = 10000 元，保证「先合规、再省钱」
DEPOT_OPEN = 5.5
TOTAL = DEMAND.sum()


# ---------------------------------------------------------------- 评价器
def eval_route(route, due=None, ready=None, cap=None):
    cap = CAP if cap is None else cap
    if not route:
        return dict(km=0.0, cost=0.0, cash=0.0, late_min=0.0, load=0.0, seq=[], finish=DEPOT_OPEN)
    due = DUE if due is None else due
    ready = READY if ready is None else ready
    load = DEMAND[np.array(route) - 1].sum()
    if load > cap + 1e-9:
        return dict(km=np.inf, cost=np.inf, cash=np.inf, late_min=np.inf,
                    load=load, seq=[], finish=np.inf)
    t, prev, km, late = DEPOT_OPEN, 0, 0.0, 0.0
    seq = []
    for j in route:
        arrive = t + TIME[prev, j]
        l_min = max(0.0, arrive - due[j - 1]) * 60.0
        late += l_min
        t = max(arrive, ready[j - 1]) + SERVICE
        seq.append(dict(name=NAMES[j - 1], arrive=float(arrive), done=float(t), late_min=l_min))
        km += DIST[prev, j]
        prev = j
    km += DIST[prev, 0]
    return dict(km=km, cost=FIXED + PER_KM * km + HARD * late,
                cash=FIXED + PER_KM * km + LATE_RATE * late / 60.0,
                late_min=late, load=load, seq=seq, finish=t)


def eval_solution(routes, due=None, cap=None, ready=None):
    parts = [eval_route(r, due, ready=ready, cap=cap) for r in routes if r]
    return dict(km=sum(p["km"] for p in parts),
                cost=sum(p["cost"] for p in parts),
                cash=sum(p["cash"] for p in parts),
                late_min=sum(p["late_min"] for p in parts),
                n=len(parts), parts=parts,
                max_finish=max([p["finish"] for p in parts], default=DEPOT_OPEN))


def held_karp(nodes):
    """单路线精确最短时间（h）+ 顺序。O(2^n·n^2)，n≤12 秒出。"""
    nodes = list(nodes)
    m = len(nodes)
    ids = [0] + nodes
    tt = TIME[np.ix_(ids, ids)]
    inf = 1e18
    dp = np.full((1 << m, m), inf)
    par = np.full((1 << m, m), -1, dtype=int)
    for k in range(m):
        dp[1 << k, k] = tt[0, k + 1]
    for mask in range(1 << m):
        row = dp[mask]
        for k in range(m):
            if row[k] >= inf:
                continue
            base = row[k]
            for nx in range(m):
                if mask >> nx & 1:
                    continue
                v = base + tt[k + 1, nx + 1]
                nm = mask | (1 << nx)
                if v < dp[nm, nx]:
                    dp[nm, nx] = v
                    par[nm, nx] = k
    full = (1 << m) - 1
    best, arg = inf, -1
    for k in range(m):
        v = dp[full, k] + tt[k + 1, 0]
        if v < best:
            best, arg = v, k
    path, mask, k = [], full, arg
    while k != -1:
        path.append(nodes[k])
        pk = par[mask, k]
        mask ^= 1 << k
        k = pk
    return best, path[::-1]


# ---------------------------------------------------------------- 构造
def zone_routes():
    quad = []
    for x, y in XY[1:, :]:
        quad.append(0 if (x >= 0 and y >= 0) else (1 if x < 0 and y >= 0 else (2 if x < 0 else 3)))
    routes = []
    for z in range(4):
        idx = [i + 1 for i, q in enumerate(quad) if q == z]
        idx.sort(key=lambda j: np.arctan2(XY[j, 1], XY[j, 0]))
        cur, load = [], 0.0
        for j in idx:
            if load + DEMAND[j - 1] > CAP + 1e-9:
                routes.append(cur)
                cur, load = [], 0.0
            cur.append(j)
            load += DEMAND[j - 1]
        if cur:
            routes.append(cur)
    return routes


def clarke_wright():
    routes = {j: [j] for j in range(1, N + 1)}
    pairs = sorted(((DIST[0, i] + DIST[0, j] - DIST[i, j], i, j)
                    for i, j in itertools.combinations(range(1, N + 1), 2)), reverse=True)
    for s, i, j in pairs:
        ri = next((k for k, v in routes.items() if i in v), None)
        rj = next((k for k, v in routes.items() if j in v), None)
        if ri is None or rj is None or ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        else:
            continue
        if DEMAND[np.array(new) - 1].sum() > CAP + 1e-9:
            continue
        routes[ri] = new
        del routes[rj]
    return list(routes.values())


def route_neighborhoods(r):
    """反转一段 + 把一段整段挪位。**只用 2-opt 会卡在「远点要挪到队尾」的形状上**。"""
    n = len(r)
    for i in range(n - 1):
        for k in range(i + 1, n):
            yield r[:i] + r[i:k + 1][::-1] + r[k + 1:]
    for i in range(n):
        for k in range(i, n):
            seg = r[i:k + 1]
            rest = r[:i] + r[k + 1:]
            for p in range(len(rest) + 1):
                if p == i:
                    continue
                yield rest[:p] + seg + rest[p:]


def two_opt(routes, due=None, ready=None):
    best = [list(r) for r in routes if r]
    improved = True
    while improved:
        improved = False
        for r in best:
            base = eval_route(r, due, ready=ready)["cost"]
            for cand in route_neighborhoods(r):
                c = eval_route(cand, due, ready=ready)["cost"]
                if c < base - 1e-9:
                    r[:] = list(cand)
                    base = c
                    improved = True
    return best


def relocate_improve(routes, due=None, ready=None, max_rounds=400):
    """relocate：把一家门店挪走（去别的线，或单独立一辆车）。这是**唯一能改变车数**的邻域。"""
    best = [list(r) for r in routes if r]
    cur = eval_solution(best, due, ready=ready)["cost"]
    for _ in range(max_rounds):
        moved = False
        for i in range(len(best)):
            for p in range(len(best[i])):
                for j in range(len(best) + 1):
                    if j == i:
                        continue
                    for q in range((len(best[j]) + 1) if j < len(best) else 1):
                        cand = [list(x) for x in best]
                        node = cand[i].pop(p)
                        if j == len(best):
                            cand.append([node])
                        else:
                            cand[j].insert(q, node)
                        cand = [x for x in cand if x]
                        c = eval_solution(cand, due, ready=ready)["cost"]
                        if c < cur - 1e-9:
                            best, cur, moved = cand, c, True
                            break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break
    return best, cur


def solve(due=None, cap=None, ready=None):
    cap = CAP if cap is None else cap
    old = globals()["CAP"]
    globals()["CAP"] = cap
    best = None
    for fn in (zone_routes, clarke_wright):
        rt = two_opt(fn(), due, ready=ready)
        rt, _ = relocate_improve(rt, due, ready=ready)
        c = eval_solution(rt, due, cap=cap, ready=ready)["cost"]
        if best is None or c < best[1]:
            best = (rt, c)
    globals()["CAP"] = old
    return best


# ---------------------------------------------------------------- 三个「车数下界」
def lb_naive():
    """① 容量算式：ceil(总需求 / 单车容量)。只看「货够不够撑满」，不看货能不能凑齐。"""
    return int(np.ceil(TOTAL / CAP))


def lb_time():
    """② 只看时间：每条线最多跑 3 小时（5:30–8:30），按「单点往返 + 卸货」估。"""
    t_each = 2 * TIME[0, 1:] + SERVICE
    return int(np.ceil(t_each.sum() / 3.0))


def min_bins(weights, cap):
    """③ 精确最小装箱数（分支定界）：货**到底能不能**装进 k 个箱子。

    这就是本讲的核心事实：装箱下界比容量算式更强——
    178 箱 / 32 箱 = 5.56，算式说「6 辆够」，但把这些**整箱**分进 6 个 32 箱的车，装不进去。
    """
    w = sorted([float(x) for x in weights], reverse=True)
    best = [len(w) + 1]

    def dfs(idx, bins):
        if len(bins) >= best[0]:
            return
        if idx == len(w):
            best[0] = len(bins)
            return
        x = w[idx]
        seen = set()
        for b in range(len(bins)):
            if bins[b] + x <= cap + 1e-9 and bins[b] not in seen:
                seen.add(bins[b])
                bins[b] += x
                dfs(idx + 1, bins)
                bins[b] -= x
        if len(bins) + 1 < best[0]:
            bins.append(x)
            dfs(idx + 1, bins)
            bins.pop()

    dfs(0, [])
    return best[0]


# ---------------------------------------------------------------- 窗口诊断
def window_pressure():
    """每个店的「窗口紧度」：网点出门直奔它，也只剩多少分钟余量。

    余量极小的店 = 时间窗把它变成了「必须先服务」的硬约束，
    它会把同车其他店的货挤出车外——这就是装箱下界卡在 7 的物理原因。
    """
    rows = []
    for j in range(1, N + 1):
        arrive = DEPOT_OPEN + TIME[0, j]
        slack = (DUE[j - 1] - arrive) * 60.0
        rows.append(dict(name=NAMES[j - 1], demand=float(DEMAND[j - 1]),
                         ready=float(READY[j - 1]), due=float(DUE[j - 1]),
                         direct_arrive=float(arrive), slack_min=float(slack)))
    return sorted(rows, key=lambda d: d["slack_min"])


def main():
    t0 = time.time()
    out = {}
    print("=" * 82)
    print("Day61 · 云溪县集配中心凌晨配送（CVRP / VRPTW）")
    print("=" * 82)
    print(f"门店 {N} 家 / 总需求 {TOTAL:.0f} 箱 / 单车 {CAP:.0f} 箱 / "
          f"网点 {DEPOT_OPEN:.1f} 点开门 / 车速 {SPEED:.0f} km/h")

    lb1 = lb_naive()
    lb2 = lb_time()
    lb3 = min_bins(DEMAND.tolist(), CAP)
    print(f"\n[车数下界①] 容量算式 ceil({TOTAL:.0f}/{CAP:.0f}) = {lb1}")
    print(f"[车数下界②] 只看时间 = {lb2}")
    print(f"[车数下界③] 精确最小装箱（整箱不可拆） = {lb3}")

    zone = zone_routes()
    ez = eval_solution(zone)
    print(f"\n[1 片区直发] {ez['n']} 车 {ez['km']:.1f} km 现金 {ez['cash']:.0f} 元 "
          f"迟到 {ez['late_min']:.0f} 分 最晚收车 {ez['max_finish']:.2f} 点")

    cw = clarke_wright()
    ecw = eval_solution(cw)
    print(f"[2 节省法]   {ecw['n']} 车 {ecw['km']:.1f} km 现金 {ecw['cash']:.0f} 元 "
          f"迟到 {ecw['late_min']:.0f} 分")

    fin, fc = solve()
    ef = eval_solution(fin)
    print(f"[3 局部搜索] {ef['n']} 车 {ef['km']:.1f} km 现金 {ef['cash']:.0f} 元 "
          f"迟到 {ef['late_min']:.0f} 分")
    for r in fin:
        p = eval_route(r)
        seq = " → ".join(f"{s['name']}({s['arrive']:.2f})" for s in p["seq"])
        print(f"    {str(r):26s} 装载 {p['load']:5.1f}  里程 {p['km']:6.1f}  "
              f"收车 {p['finish']:.2f} 点  {seq}")

    # Held-Karp 核对：每条线在「不考虑窗口」下的最优顺序与里程
    hk_rows = []
    hk_gain = 0.0
    viol = []
    for r in fin:
        opt_t, order = held_karp(r)
        meas = eval_route(r)["km"]
        hk_rows.append(dict(route=list(r), meas=float(meas), opt=float(opt_t * SPEED), order=order))
        hk_gain += meas - opt_t * SPEED
        # 纯 CVRP 对照：把这条线的节点重新排成 Held-Karp 最优顺序（不检查时间窗），看它会把谁弄迟到
        pure = []
        for j in order:
            pure.append(j)
        check = eval_route(pure)
        if check["late_min"] > 1e-6:
            viol.append(dict(route=list(r), order=order, pure_late=float(check["late_min"]),
                             detail=[(s2["name"], round(s2["arrive"], 2), round(s2["late_min"], 1))
                                     for s2 in check["seq"] if s2["late_min"] > 1e-6]))
    print(f"\n[核对] Held-Karp 逐线里程：现总 {ef['km']:.1f} km / 理论最好 "
          f"{ef['km'] - hk_gain:.1f} km（差 {hk_gain:.1f} km = {hk_gain * PER_KM:.0f} 元）")
    for h in hk_rows:
        if h["opt"] < h["meas"] - 0.2:
            print(f"    {h['route']}: 现 {h['meas']:.1f} / 无窗最优 {h['opt']:.1f} "
                  f"（最优顺序 {h['order']}）→ 被时间窗锁住了")
    print(f"[核对] 把每条线改成「纯 CVRP 最短顺序」，有 {len(viol)} 条线会违约：")
    for v in viol:
        det = "；".join(f"{a} {b:.2f} 到 / 迟 {c:.0f} 分" for a, b, c in v["detail"])
        print(f"    {v['route']} → 顺序 {v['order']}，{det} "
              f"（迟到 {v['pure_late']:.0f} 分钟）")
    out["pure_violation"] = viol

    out.update(lb=dict(naive=lb1, time=lb2, bins=lb3),
               zone=dict(n=ez["n"], km=float(ez["km"]), cash=float(ez["cash"]),
                         late=float(ez["late_min"]), routes=zone,
                         finish=float(ez["max_finish"])),
               cw=dict(n=ecw["n"], km=float(ecw["km"]), cash=float(ecw["cash"]),
                       late=float(ecw["late_min"]), routes=cw),
               final=dict(n=ef["n"], km=float(ef["km"]), cash=float(ef["cash"]),
                          late=float(ef["late_min"]), routes=fin,
                          loads=[float(eval_route(r)["load"]) for r in fin],
                          kms=[float(eval_route(r)["km"]) for r in fin],
                          finishes=[float(eval_route(r)["finish"]) for r in fin],
                          seqs=[eval_route(r)["seq"] for r in fin]),
               hk=hk_rows)

    print(f"\n[对比] 片区直发 {ez['cash']:.0f} 元 → 局部搜索 {ef['cash']:.0f} 元，"
          f"省 {ez['cash'] - ef['cash']:.0f} 元（{(ez['cash'] / ef['cash'] - 1) * 100:.1f}%）")
    print(f"       车 {ez['n']} → {ef['n']}，里程 {ez['km']:.1f} → {ef['km']:.1f} km")

    # 窗口压力
    press = window_pressure()
    print("\n[4 窗口压力] 网点直奔也只剩 < 90 分钟的店：")
    for d in press:
        if d["slack_min"] < 90:
            print(f"    {d['name']:10s} 货 {d['demand']:4.0f} 箱  窗口 "
                  f"{d['ready']:.1f}-{d['due']:.1f}  直奔 {d['direct_arrive']:.2f} 到  "
                  f"余 {d['slack_min']:5.1f} 分钟")
    out["pressure"] = press[:6]

    # 放宽窗口（推后最晚卸货）
    print("\n[5 如果要求放软：最晚卸货整体 +X 小时（最早卸货不动）]")
    scan = []
    for shift in (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        rt, _ = solve(due=DUE + shift)
        e = eval_solution(rt, due=DUE + shift)
        scan.append(dict(shift=shift, n=e["n"], km=float(e["km"]), cash=float(e["cash"]),
                         late=float(e["late_min"]), routes=rt))
        print(f"    +{shift:.1f} h: {e['n']} 车 {e['km']:.1f} km {e['cash']:.0f} 元")
    out["window_scan"] = scan

    # 容量扫描：换成更大的车
    print("\n[6 如果换车：单车容量扫描]")
    capscan = []
    for cap in (30.0, 32.0, 34.0, 36.0, 40.0):
        rt, _ = solve(cap=cap)
        e = eval_solution(rt, cap=cap)
        capscan.append(dict(cap=cap, n=e["n"], km=float(e["km"]), cash=float(e["cash"]),
                            loads=[float(eval_route(r, cap=cap)["load"]) for r in rt], routes=rt))
        print(f"    单车 {cap:4.0f} 箱: {e['n']} 车 {e['km']:.1f} km {e['cash']:.0f} 元 "
              f"装载 {[round(x) for x in capscan[-1]['loads']]}")
    out["cap_scan"] = capscan

    # 验收断言：卡「教学点真的发生」，不是卡我脑子里的数字
    assert lb3 >= lb1, "装箱下界不应弱于容量下界"
    assert ef["n"] >= lb3, "解的车数不可能低于装箱下界"
    assert ef["late_min"] == 0, "局部搜索的解必须零违约"
    assert ez["n"] > ef["n"], "片区直发必须比局部搜索多用车"
    assert ez["cash"] > ef["cash"], "片区直发必须更贵"
    assert any(row["n"] < ef["n"] for row in capscan), "换大车必须能减少车数"
    assert capscan[0]["cash"] - capscan[-1]["cash"] > 500, "换车必须真的省钱，否则教学点不成立"
    assert capscan[1]["n"] < capscan[0]["n"], "只加 2 箱容量就要能省一台车（本讲的核心反差）"
    assert hk_gain < 5.0, "顺序层面应该已接近最优，教学点应落在装载而非顺序"
    tight = [d for d in press if d["slack_min"] < 60]
    assert tight, "必须存在窗口极紧的门店，否则第 4 问立不住"
    assert all(abs(p["load"] - round(p["load"])) < 1e-6 for p in ef["parts"]), "箱数必须整"

    plot(out)

    out["elapsed"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n耗时 {time.time() - t0:.2f} s → {OUT_JSON}")
    return out


def plot(out):
    fig = plt.figure(figsize=(15.5, 11.0))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    # 1) 路线图
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, (r, seq) in enumerate(zip(out["final"]["routes"], out["final"]["seqs"])):
        path = [0] + r + [0]
        ax.plot(XY[path, 0], XY[path, 1], "-o", lw=1.6, ms=4, color=colors[k % 10],
                label=f"车{k + 1}（{out['final']['loads'][k]:.0f}箱/{out['final']['kms'][k]:.0f}km）")
    ax.scatter([0], [0], marker="*", s=320, c="k", zorder=5, label="集配中心")
    for j in range(1, N + 1):
        ax.annotate(NAMES[j - 1], (XY[j, 0], XY[j, 1]), fontsize=7.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"最优方案：{out['final']['n']} 车 / {out['final']['km']:.1f} km"
                 f"（片区直发 {out['zone']['n']} 车 / {out['zone']['km']:.1f} km）", fontsize=11)
    ax.set_xlabel("东西向 km")
    ax.set_ylabel("南北向 km")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)

    # 2) 三种下界 vs 三种方案
    ax = fig.add_subplot(gs[0, 1])
    labels = ["车数下界\n容量算式", "车数下界\n只看时间", "车数下界\n装箱精确",
              "片区直发", "节省法", "局部搜索"]
    vals = [out["lb"]["naive"], out["lb"]["time"], out["lb"]["bins"],
            out["zone"]["n"], out["cw"]["n"], out["final"]["n"]]
    cols = ["#9ecae1", "#9ecae1", "#6baed6", "#fdae6b", "#fdae6b", "#e6550d"]
    bars = ax.bar(range(6), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v}", ha="center", fontsize=10)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("车辆数（辆）")
    ax.set_ylim(0, max(vals) + 1.2)
    ax.set_title("车数：下界说 6，实际要 7 —— 卡在装载不在路线", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    # 3) 容量扫描：车数 & 现金成本
    ax = fig.add_subplot(gs[1, 0])
    caps = [r["cap"] for r in out["cap_scan"]]
    ns = [r["n"] for r in out["cap_scan"]]
    cash = [r["cash"] for r in out["cap_scan"]]
    ax.plot(caps, ns, "-o", color="#3182bd", lw=2, ms=7, label="车辆数（辆）")
    ax.set_xlabel("单车装载上限（箱）")
    ax.set_ylabel("车辆数", color="#3182bd")
    ax.tick_params(axis="y", labelcolor="#3182bd")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(caps, cash, "-s", color="#e6550d", lw=2, ms=7, label="现金成本（元）")
    ax2.set_ylabel("现金成本（元）", color="#e6550d")
    ax2.tick_params(axis="y", labelcolor="#e6550d")
    for c, n_, k in zip(caps, ns, cash):
        ax2.annotate(f"{n_}辆 / {k:.0f}元", (c, k), fontsize=8, xytext=(0, 8),
                     textcoords="offset points", ha="center")
    ax.set_title(f"换车：{caps[0]:.0f}→{caps[1]:.0f} 箱省 1 台车 + "
                 f"{cash[0] - cash[1]:.0f} 元；{caps[-1]:.0f} 箱降到 {ns[-1]} 台", fontsize=11)

    # 4) 各车装载与窗口
    ax = fig.add_subplot(gs[1, 1])
    loads = out["final"]["loads"]
    idx = np.arange(len(loads))
    ax.bar(idx, loads, color="#74c476", alpha=0.85)
    ax.axhline(CAP, color="r", ls="--", lw=1.5, label=f"单车容量 {CAP:.0f} 箱")
    ax.axhline(TOTAL / len(loads), color="gray", ls=":", lw=1.2,
               label=f"平均装载 {TOTAL / len(loads):.1f} 箱")
    for i, v in enumerate(loads):
        ax.text(i, v + 0.4, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"车{i + 1}\n收{out['final']['finishes'][i]:.1f}点" for i in idx], fontsize=8)
    ax.set_ylabel("装载（箱）")
    ax.set_ylim(0, CAP * 1.18)
    ax.set_title(f"{len(loads)} 辆车的装载：最满 {max(loads):.0f} 箱，最空的只装 {min(loads):.0f} 箱", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("云溪县集配中心凌晨配送：车数由「装载」定，成本由「里程」定", fontsize=13.5)
    fig.savefig(OUT_PNG, dpi=135, bbox_inches="tight")
    print(f"图 → {OUT_PNG}")


if __name__ == "__main__":
    main()
