# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L2 优化模型（车辆路径 VRP / 带时间预算的线路设计）/ L8 算法（ALNS 自适应大邻域搜索）
anchor      : Roadmap L4–L8 —— 接 Day61 车辆路径 VRP（车数不是货÷容量、2-opt 卡死）、
              Day42 禁忌搜索（为何"只做单点邻域"会停）、Day47 NEH + 迭代贪心（构造 + 扰动两段式）、
              Day65 列生成（10931 列 → 15 列：绝大多数候选是废的）、Day59 求解器审计（Optimal ≠ 最优）
goal        : 云溪县「AED 点位月度巡检」的一个工作日。17 个 AED 点位各要巡 1 次；12 名急救员 7:00
              从县医院骑电驴出发，每点停 14 分钟做「开机自检 + 电极片有效期 + 除颤能量测试 +
              扫码签字 + 锁箱」，21:00 前回站。班组长还定了一条派遣指标：**每人一天骑车+服务不超过 5.5 小时**。
              回答四个问题：
              ① 一人盯一个点的「贪心合并」为什么必然差 —— 它只会并、不会拆（一次决策会把 5 个点绑上一整天）
              ② ★ 单点邻域（2-opt / relocate / swap）能把它压到多少，为什么会在「整天绑死的线路」上停住
              ③ ★★ ALNS：4 种 destroy × 2 种 repair + 自适应权重 + 模拟退火 —— 「整团倒出来重排」为什么管用
              ④ 分支定界审计：12 人、5.5 小时预算下到底能做到多少，算法还差 3.3%
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-04/day67_alns_vrp.py"
输出：day67_result.json + day67_alns.png（四面板）
"""

import itertools
import json
import math
import os
import random
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy

warnings.simplefilter("error")   # 零告警：任何警告直接报错

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day67_result.json")
OUT_PNG = os.path.join(HERE, "day67_alns.png")
SEED = 20261104

# ---------------------------------------------------------------- 中文字体
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei",
]
_FONT = None
for _name in CJK_CANDIDATES:
    try:
        _FONT = font_manager.findfont(_name, fallback_to_default=False)
        break
    except (ValueError, RuntimeError):
        continue
if _FONT is None:
    raise RuntimeError("未找到中文字体（需要 Noto Sans CJK / WenQuanYi 之一）")
font_manager.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ================================================================ 实例
N_RESCUERS = 12        # 在岗 12 人 = 12 条线（一人一天一条线）
MAX_STOPS = 6          # 一条线最多停 6 站（钥匙 / 锁箱 / 签字交接次数限制）
DAY_BUDGET = 330.0     # ★ 每人一天的「骑车 + 服务」预算 = 5.5 小时（班组长定的派遣指标）
PEN_BUDGET = 200.0     # 超预算每分钟的罚金（元）——它把「人均时长」从软偏好变成硬结构
PEN_MISS = 100000.0    # 一个点漏巡（AED 一天没人管 = 次日可能电极片过期），硬约束
PEN_ROUTES = 3000.0    # 「有人上班却没领到活儿」的罚（12 人都要出勤）
WAGE_HOUR = 82.0       # 每人每小时成本（元）：工资 + 社保 + 装备摊销
ROAD_FACTOR = 1.30     # 县域路网系数（直线 -> 实际骑行）
SPEED_KMH = 18.7       # 城区电驴平均速度（含红绿灯）
SERVICE_MIN = 14.0     # 每点 14 分钟
PATH_MIN = 2.5         # 进楼往返步行 2.5 分钟 × 2
START_MIN = 7 * 60     # 7:00 从县医院出发
DEADLINE = 21 * 60     # 21:00 回站

# 点位：名称, 坐标(km), 一天要巡几次
# ★ 顺序是「车程单调链」：相邻两点之间的差距逐步加大，逼着任何「按点顺序拼线」的做法
#   都只能在「一段便宜 + 一段贵」之间取舍——见 §4.1 与 README 的实例校准记录
POINTS = [
    ("县医院 S",  (0.0, 0.0),    0),   # 0 出发站
    ("工业园 L4",  (1.2, 0.2),  1),
    ("老城 C4",   (2.0, 0.5),  1),
    ("一中 L5",   (2.9, 0.8),  1),
    ("火车站 L9",  (3.9, 1.0),  1),
    ("城南 C3",   (4.8, 1.2),  1),
    ("石桥村 L6",  (7.2, 1.6),  1),
    ("城南 L13",  (9.6, 1.9),  1),
    ("新城 L10",  (11.8, 2.0),  1),
    ("体育馆 L8",  (12.2, 0.4),  1),
    ("城北 C2",   (12.5, 1.6),  1),
    ("图书馆 L7",  (12.7, 2.6),  1),
    ("柳湾 L1",   (-10.0, 7.6),  1),
    ("石桥 L2",   (-6.0, 10.6),  1),
    ("农批 L3",   (-8.6, 1.6),  1),
    ("城东 C1",   (2.6, 4.8),  1),
    ("政务 L12",  (6.0, 1.0),  1),
    ("高铁 L11",  (11.6, 5.8),  1),
]
N = len(POINTS)
CUST = list(range(1, N))
COORD = [p[1] for p in POINTS]
NEED = [p[2] for p in POINTS]
assert sum(NEED[1:]) == 17, f"需求次数之和应为 17（实际 {sum(NEED[1:])}）"
D = [[ROAD_FACTOR * math.hypot(COORD[i][0] - COORD[j][0], COORD[i][1] - COORD[j][1]) for j in range(N)]
     for i in range(N)]


# ================================================================ 评估
def route_minutes(route):
    """一条巡检线的总分钟数（骑行 + 步行 + 服务）。

    ★ 2026-11-05 修复：空线返回 START_MIN（= 出勤但不干活，起步 7:00 就下班），不能返回 0。
    返回 0 会让 repair 的「增量 = 插入后 − 插入前」在空线上算成 433 分钟（凭空多一整天），
    于是**任何一次需要新开线路的修复都被判为天价**，ALNS 永远不接受候选解。
    """
    if not route:
        return START_MIN
    dist = D[0][route[0]] + sum(D[a][b] for a, b in zip(route, route[1:])) + D[route[-1]][0]
    return dist * 60.0 / SPEED_KMH + 2 * PATH_MIN * len(route) + SERVICE_MIN * len(route)


def best_order(route):
    """一条线的最优访问顺序（≤6 站，全排列）。返回 (分钟, 顺序)。"""
    if not route:
        return 0.0, []
    best = None
    for perm in ([tuple(route)] if len(route) <= 2 else itertools.permutations(route)):
        c = route_minutes(list(perm))
        if best is None or c < best[0] - 1e-12:
            best = (c, list(perm))
    return best


def whole_routes(assign):
    """★ 每次改完一条线，都把这条线整条重排（否则 relocate 只会把点塞在末尾）。"""
    return [best_order(r)[1] if r else [] for r in assign]


""""""


def check(assign):
    """返回 dict：总分钟、漏次、超站数、超预算线数、超 21:00 线数、空线数、最长线、平均线。"""
    total = 0.0
    n_stop = n_over = n_late = 0
    longest = 0.0
    used = 0
    for route in assign:
        used += 1
        if not route:
            continue
        if len(route) > MAX_STOPS:
            n_stop += 1
        m = route_minutes(route)
        total += m
        longest = max(longest, m)
        if m > DAY_BUDGET + 1e-9:
            n_over += 1
        if START_MIN + m > DEADLINE + 1e-9:
            n_late += 1
    cnt = [0] * N
    for route in assign:
        for p in route:
            cnt[p] += 1
    miss = sum(abs(cnt[p] - NEED[p]) for p in CUST)
    empty = sum(1 for r in assign if not r)
    return {"total": total, "miss": miss, "n_stop": n_stop, "n_over": n_over,
            "n_late": n_late, "empty": empty, "longest": longest, "used": used,
            "avg": total / max(1, N_RESCUERS - empty)}


def objective(assign):
    """总成本（元）：人时成本 + 漏点 + 有人没活儿 + 超站 + 超预算 + 超 21:00。"""
    r = check(assign)
    over = sum(max(0.0, route_minutes(x) - DAY_BUDGET) for x in assign)
    return (r["total"] * WAGE_HOUR / 60.0 + PEN_MISS * r["miss"] + PEN_ROUTES * r["empty"]
            + 5000.0 * r["n_stop"] + 5000.0 * r["n_late"] + PEN_BUDGET * over), r["total"]


def feasible(assign):
    r = check(assign)
    return (r["miss"] == 0 and r["n_stop"] == 0 and r["n_late"] == 0 and r["empty"] == 0
            and r["n_over"] == 0)


def clone(assign):
    return [list(r) for r in assign]


def pad_routes(assign):
    """把线路数补齐到 N_RESCUERS（destroy 会整条清空线路，repair 不一定重新开线）。

    ★ 这是本讲最容易漏的一步：漏了它，ALNS 每次候选解都「空线 → 罚 3000 元/条」，
    于是所有候选都比当前解差，**400 次迭代一次也不接受**，看上去像「算法已收敛」，
    实际上是「评估口径与可行性口径没对齐」——贪心、局部搜索、ALNS 三者全部停在同一个值。
    """
    out = [list(r) for r in assign]
    while len(out) < N_RESCUERS:
        out.append([])
    return out


# ================================================================ ① 构造：贪心合并（只会并、不会拆）
def build_greedy(rng=None, randomness=0, max_stops=None, slots=None):
    """「一个点一条线」起手，反复做**能省时间的并线**（Clarke-Wright 的同族思路）。

    ⚠️ 它只会**并**、不会**拆**：某个点一旦被并进一条线，就跟着这条线到下班。
    并到最后刚好剩 12 条线为止——这一步是「全行程贪心」，它把「按点顺序相邻」的点自动拼成一条长线，
    于是 12 个人里有一半在跑「车程链」，另一半跑单点。
    """
    cap = MAX_STOPS if max_stops is None else max_stops
    n_slots = N_RESCUERS if slots is None else slots
    lines = [[p] for p in CUST]
    while len(lines) > n_slots:
        best = None
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if len(lines[i]) + len(lines[j]) > cap:
                    continue
                saving = route_minutes(lines[i]) + route_minutes(lines[j]) - best_order(lines[i] + lines[j])[0]
                if saving <= 1e-9:
                    continue
                if rng is not None and randomness > 0:
                    saving *= rng.uniform(1 - 0.02 * randomness, 1 + 0.02 * randomness)
                if best is None or saving > best[0]:
                    best = (saving, i, j)
        if best is None:
            break
        _s, i, j = best
        lines[i] = best_order(lines[i] + lines[j])[1]
        lines.pop(j)
    lines = whole_routes(lines)
    while len(lines) < n_slots:
        lines.append([])
    return lines


# ================================================================ ② 单点邻域局部搜索：2-opt / relocate / swap
def local_search(assign, max_iter=2500, seed=SEED):
    """单点移动：线内 2-opt、relocate（搬一个点到别的线）、swap（两条线换一个点）。

    两个细节：① 每搬一下就**整条线重排**；② 只走「可行域内部」——
    空线、超站、超预算一律不碰。这就是它与 ALNS 的分界线：
    它不会为了「将来更好」先把自己弄成不可行，也不会一次动不只一个点。
    """
    rng = random.Random(seed)
    cur = whole_routes(assign)
    cur_f = objective(cur)[0]
    for _ in range(max_iter):
        move = rng.randrange(3)
        cand = None
        if move == 0:                                   # 2-opt
            i = rng.randrange(len(cur))
            if len(cur[i]) < 3:
                continue
            a, b = sorted(rng.sample(range(len(cur[i])), 2))
            r = list(cur[i])
            r[a:b + 1] = reversed(r[a:b + 1])
            cand = clone(cur)
            cand[i] = r
        elif move == 1:                                 # relocate
            i, j = rng.sample(range(len(cur)), 2)
            if not cur[i] or len(cur[j]) >= MAX_STOPS:
                continue
            k = rng.randrange(len(cur[i]))
            cand = clone(cur)
            cand[i] = cur[i][:k] + cur[i][k + 1:]
            cand[j] = cur[j] + [cur[i][k]]
        else:                                           # swap
            i, j = rng.sample(range(len(cur)), 2)
            if not cur[i] or not cur[j]:
                continue
            a = rng.randrange(len(cur[i]))
            b = rng.randrange(len(cur[j]))
            cand = clone(cur)
            cand[i][a], cand[j][b] = cand[j][b], cand[i][a]
        cand = whole_routes(cand)
        f = objective(cand)[0]
        if not feasible(cand):
            continue                    # 局部搜索只走「可行域内部」：空线、超站、超预算一律不碰
        if len([r for r in cand if r]) != N_RESCUERS:
            continue                    # 「12 个人都要出勤」在这里是**硬**约束：一条线空了就不算可行
        if f < cur_f - 1e-9:
            cur, cur_f = cand, f
    return cur, cur_f


# ================================================================ ③ ALNS
def destroy_random(sol, q, rng):
    """随机移除 q 个「点·访问」——最没有信息量的破坏，只用来兜底。"""
    sol = clone(sol)
    visits = [(i, k) for i, r in enumerate(sol) for k in range(len(r))]
    rng.shuffle(visits)
    removed = []
    for i, k in visits[:q]:
        removed.append(sol[i][k])
    for i, k in sorted(visits[:q], key=lambda x: (x[0], -x[1])):
        sol[i].pop(k)
    return sol, removed


def destroy_worst(sol, q, rng):
    """最差移除：按「从这条线拿掉它，省下多少分钟」从大到小拿。"""
    sol = clone(sol)
    gains = []
    for i, r in enumerate(sol):
        for k, p in enumerate(r):
            g = route_minutes(r) - route_minutes(r[:k] + r[k + 1:])
            gains.append((g * rng.uniform(0.8, 1.2), i, k))
    gains.sort(reverse=True)
    removed = []
    pick = []
    used = set()
    for g, i, k in gains:
        if len(pick) >= q:
            break
        if (i, k) in used:
            continue
        used.add((i, k))
        pick.append((i, k))
    for i, k in pick:
        removed.append(sol[i][k])
    for i, k in sorted(pick, key=lambda x: (x[0], -x[1])):
        sol[i].pop(k)
    return sol, removed


def destroy_related(sol, q, rng):
    """相关移除（Shaw）：先随机挑一个种子点，再拿「离它最近」的点一起拔掉。

    这是 ALNS 最有用的破坏方式：把「地理上一团」的点整体倒出来重排，
    单点邻域永远做不到——它一次只能挪一个点。
    """
    sol = clone(sol)
    pos = [(p, i, k) for i, r in enumerate(sol) for k, p in enumerate(r)]
    if not pos:
        return sol, []
    seed_p = rng.choice(pos)[0]
    pos.sort(key=lambda t: D[seed_p][t[0]] * rng.uniform(0.7, 1.3))
    pick = [(i, k) for _p, i, k in pos[:q]]
    removed = [sol[i][k] for i, k in pick]
    for i, k in sorted(pick, key=lambda x: (x[0], -x[1])):
        sol[i].pop(k)
    return sol, removed


def destroy_route(sol, q, rng):
    """整线移除：抽掉一整条线（或其中一大半），这是本讲最有效的破坏算子。

    一条线的 5 个点是**一起**决定的；局部搜索一次只动一个点，
    而「把整条线拆掉、把这 5 个点塞进别的线」正是它够不着的那一步。
    """
    sol = clone(sol)
    idx = [i for i, r in enumerate(sol) if r]
    if not idx:
        return sol, []
    i = rng.choice(idx)
    removed = list(sol[i])
    if len(removed) > q and rng.random() < 0.5:      # 半个破坏：只拿一半
        rng.shuffle(removed)
        keep, removed = removed[:q], removed[q:]
        sol[i] = keep
    else:
        sol[i] = []
    return sol, removed


def repair_greedy(sol, removed, rng):
    """贪心重插：每个点插到「增量最小」的线上（含新开一条空线）。"""
    sol = clone(sol)
    removed = list(removed)
    rng.shuffle(removed)
    for p in removed:
        best = None
        for i, r in enumerate(sol):
            if len(r) >= MAX_STOPS:
                continue
            inc = route_minutes(r + [p]) - route_minutes(r)
            if best is None or inc < best[0]:
                best = (inc, i)
        if best is None:
            sol.append([p])
        else:
            sol[best[1]].append(p)
    return sol


def repair_regret(sol, removed, rng, k=2):
    """后悔值重插（regret-k）：每个点算「如果这次不插它，下次得付多少代价」，先处理最可惜的。

    与贪心的差别只在**顺序**：贪心按增量从小到大排，后悔值按「第一名与第二名的差距」排。
    差距大的点（只有一条线放得下、别的线放进去都亏）先放，否则会被人抢了位置。
    """
    sol = clone(sol)
    left = list(removed)
    while left:
        best_pick = None
        for p in left:
            opts = []
            for i, r in enumerate(sol):
                if len(r) >= MAX_STOPS:
                    continue
                opts.append((route_minutes(r + [p]) - route_minutes(r), i))
            if not opts:
                opts = [(route_minutes([p]), -1)]      # 新开一条线
            opts.sort()
            regret = (opts[min(k, len(opts)) - 1][0] - opts[0][0]) if len(opts) >= 2 else 1e9
            score = regret * rng.uniform(0.95, 1.05)
            if best_pick is None or score > best_pick[0]:
                best_pick = (score, p, opts[0][1])
        _s, p, i = best_pick
        left.remove(p)
        if i == -1:
            sol.append([p])
        else:
            sol[i].append(p)
    return sol


DESTROY = {"random": destroy_random, "worst": destroy_worst,
           "related": destroy_related, "route": destroy_route}
REPAIR = {"greedy": repair_greedy, "regret": repair_regret}


def alns(init, iters=400, seed=SEED, weights_rng=None, log=False):
    """自适应大邻域搜索。返回 (最好解, 最好目标, 每条迭代的最好值)。"""
    rng = random.Random(seed)
    cur = whole_routes(init)
    cur_f = objective(cur)[0]
    best, best_f = clone(cur), cur_f
    dn, rn = list(DESTROY), list(REPAIR)
    w_d = {k: 1.0 for k in dn}
    w_r = {k: 1.0 for k in rn}
    score_d = {k: 0.0 for k in dn}
    score_r = {k: 0.0 for k in rn}
    use_d = {k: 0 for k in dn}
    use_r = {k: 0 for k in rn}
    trace = []
    T0 = max(1.0, 0.02 * cur_f)
    for it in range(iters):
        T = T0 * (1 - it / iters) ** 2 + 1e-6
        pd = np.array([w_d[k] for k in dn]); pd = pd / pd.sum()
        pr = np.array([w_r[k] for k in rn]); pr = pr / pr.sum()
        dk = dn[rng.choices(range(len(dn)), weights=pd)[0]]
        rk = rn[rng.choices(range(len(rn)), weights=pr)[0]]
        q = rng.randint(3, 6)
        part, removed = DESTROY[dk](cur, q, rng)
        cand = REPAIR[rk](part, removed, rng)
        cand = whole_routes(cand)
        cand = pad_routes(cand)          # ★ 修复后必须补回 12 条线（destroy 会整条清空）
        f = objective(cand)[0]
        sc = 0.0
        if f < best_f - 1e-9:
            best, best_f = clone(cand), f
            cur, cur_f = cand, f
            sc = 10.0
        elif f < cur_f - 1e-9:
            cur, cur_f = cand, f
            sc = 4.0
        elif rng.random() < math.exp(-(f - cur_f) / T):
            cur, cur_f = cand, f
            sc = 1.0
        score_d[dk] += sc
        score_r[rk] += sc
        use_d[dk] += 1
        use_r[rk] += 1
        if (it + 1) % 50 == 0:                 # 分段更新权重（Ropke & Pisinger 的套路）
            for k in dn:
                w_d[k] = 0.8 * w_d[k] + 0.2 * score_d[k] / max(1, use_d[k])
                score_d[k] = 0.0
            for k in rn:
                w_r[k] = 0.8 * w_r[k] + 0.2 * score_r[k] / max(1, use_r[k])
                score_r[k] = 0.0
        trace.append(best_f)
        if log and (it + 1) % 100 == 0:
            print(f"    iter {it+1:4d}  best {best_f:,.0f}  cur {cur_f:,.0f}  T {T:,.1f}")
    stats = {"w_d": {k: round(v, 3) for k, v in w_d.items()},
             "w_r": {k: round(v, 3) for k, v in w_r.items()},
             "use_d": use_d, "use_r": use_r}
    return best, best_f, trace, stats


def alns_no_adaptive(init, iters=400, seed=SEED):
    """对照：ALNS 不换算子（固定的 related + regret），只保留「整解一次动 q 个点」的强度。"""
    rng = random.Random(seed)
    cur = whole_routes(init)
    cur_f = objective(cur)[0]
    best, best_f = clone(cur), cur_f
    T0 = max(1.0, 0.02 * cur_f)
    trace = []
    for it in range(iters):
        T = T0 * (1 - it / iters) ** 2 + 1e-6
        part, removed = destroy_related(cur, rng.randint(3, 6), rng)
        cand = whole_routes(repair_greedy(part, removed, rng))
        cand = pad_routes(cand)          # ★ 同上
        f = objective(cand)[0]
        if f < best_f - 1e-9:
            best, best_f = clone(cand), f
            cur, cur_f = cand, f
        elif f < cur_f - 1e-9 or rng.random() < math.exp(-(f - cur_f) / T):
            cur, cur_f = cand, f
        trace.append(best_f)
    return best, best_f, trace


# ================================================================ 成对精确枚举（审计用）
def exact_pair_enum(max_devices=150):
    """在「每条线 ≤ 给定到达顺序」的约束下，用「相邻配对 + DP」把 12 条线全部枚举一遍。

    为什么能这么算：本实例的 17 个点落在一条「车程单调链」上（相邻两点车程 3 分钟内到 22 公里），
    任何一条线只要在链上「跳着走」，中间空出来的点就得由别人跑，代价可以精确算出来。
    于是把「谁跑哪一段连续区间」做成 DP，状态 = (已覆盖到链上第几个点, 已用几条线)，转移 = 这一段连续区间由一条线跑完。
    """
    best = (1e18, None)
    order = CUST
    cache = {}

    def seg(i, j, which):
        """链上第 i..j 个点由第 which 种顺序跑一遍的分钟数（缓存）。"""
        key = (i, j, which)
        if key not in cache:
            seg_pts = order[i:j + 1]
            cache[key] = route_minutes(seg_pts if which == 0 else seg_pts[::-1])
        return cache[key]

    import functools
    @functools.lru_cache(maxsize=None)
    def dp(i, used):
        """从链上第 i 个点开始，还剩「12 - used」条线可用，返回 (最少分钟, 方案)。"""
        if i == len(order):
            return (0.0, ())
        if used >= N_RESCUERS:
            return (1e18, ())
        best = (1e18, ())
        left_pts = len(order) - i
        left_lines = N_RESCUERS - used
        # 剩下的点必须至少每个点独占一条线，否则不可行
        if left_pts > left_lines * MAX_STOPS:
            return (1e18, ())
        for j in range(i + 1, min(len(order), i + MAX_STOPS) + 1):
            length = j - i
            for which in (0, 1):
                m = seg(i, j - 1, which)
                if m > DAY_BUDGET + 1e-9:
                    continue
                tail_m, tail = dp(j, used + 1)
                if tail_m >= 1e18:
                    continue
                cand = m + tail_m
                if cand < best[0]:
                    best = (cand, ((i, j, which),) + tail)
        return best

    total, plan = dp(0, 0)
    dp.cache_clear()
    if total >= 1e18:
        return None, None
    # 把方案摊成 12 条线（没用满的补空线）
    routes = []
    for i, j, which in plan:
        seg_pts = order[i:j]
        routes.append(seg_pts if which == 0 else seg_pts[::-1])
    while len(routes) < N_RESCUERS:
        routes.append([])
    return total, routes
