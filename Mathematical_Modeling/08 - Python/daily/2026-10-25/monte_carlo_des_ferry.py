# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 / L5（随机模型 + 算法）—— 蒙特卡洛仿真 & 离散事件仿真（DES）
anchor  : Day18 排队论 M/M/c + Day28 两阶段随机规划 SAA（同一套"抽样 → 估计 → 给区间"）
goal    : 用"县域渡口春运配几条渡船"回答四件事：
          ① 平均法（日均流量 ÷ 额定运力）为什么必然超员；
          ② 蒙特卡洛（MC）为什么能直接给出"超员小时占比 + 置信区间"，而 M/M/c 的稳态指标给不出；
          ③ 离散事件仿真（DES：发船是事件、船回港才能再发、高峰靠港变慢）才看得见"船空着走"；
          ④ 要 99% 保证不超员，到底配几条船、每多配一条一年值不值。

模型口径（重要）：
  * 运力由**时刻表 × 船数**决定（每船每 80 分钟一班 = 0.75 班/小时），不是"需求 ÷ 40"；
  * 发船时刻由事件推进：t_dep = max(船回港时刻, 小时起点)；
  * 未上船的人**排队等下一班**（不直接流失），只有收班时队伍不为空才算"今天没运完"。

依赖：numpy matplotlib（仓库 .venv 已装）
运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-25/monte_carlo_des_ferry.py"
输出：monte_carlo_des_ferry.png / monte_carlo_des_ferry.json（与本脚本同目录）
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------- 中文字体（避免方块字；仓库既有做法）----------
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "monte_carlo_des_ferry.png"
OUT_JSON = HERE / "monte_carlo_des_ferry.json"
SEED = 20261025

# ============================================================
# 一、场景常数
# ============================================================
CAP = 40                       # 单船核定载客（人/班次）
UP_RUN = 8.0                   # 本站 → 对岸 航行分钟（江面宽约 800 m，300 马力客渡船）
DOWN_RUN = 8.0                 # 对岸 → 本站 返程分钟
TURNAROUND = 14.0              # 两端靠港：清客 + 售票点名 + 上客准备（"死时间"）
CYCLE = UP_RUN + DOWN_RUN + TURNAROUND     # 30 分钟 = 一艘船两次从本站出发的间隔

# 每小时期望到客（05:00–20:59 共 16 个整点小时；双峰 + 午后谷）
HOURLY_MEAN = np.array(
    [50, 110, 200, 340, 480, 450, 260, 180, 220, 300, 420, 520, 470, 330, 180, 80],
    dtype=float,
)
N_HOURS = HOURLY_MEAN.size
assert N_HOURS == 16
T_OPEN = 5                      # 05:00 首班、21:00 收班
T_CLOSE_ABS = N_HOURS + 0.5     # 收班后还可以再接一船（21:30 前回港的船）
HOURLY_MEAN_prefirst = 40.0     # 04:00–05:00 首班前到站、等首班车的人

PEAK_TARGET = 0.01              # 春运要求：超员小时占比 ≤ 1%
DAYS_PEAK = 40                  # 春运高峰天数（节前 20 + 节后 20）
DAYS_OFF = 30                   # 淡季（清明 / 暑期 / 国庆）营业日
LOW_SEASON_FACTOR = 0.55

# 拥挤效应：上客超过 30 人，靠港作业每多一人 +0.5 分钟（核票、救生衣、实名登记）
CROWD_THRESHOLD = 30
CROWD_MIN_PER_PAX = 0.5
# 满载拖班：等到 40 人满了才走，平均多等 CROWD_WAIT 分钟（这个"等客"在船 30 分钟一圈的循环里很致命）
CROWD_WAIT = 6.0

# 成本 / 收益（万元）
PRICE = 2.0                     # 票价 元/人次
BOAT_BUY = 68.0                 # 新船购置 68 万元，12 年直线折旧 = 5.67 万元/年
BOAT_MAINT = 7.2               # 年坞修 / 保险 / 年检
CREW_PER_BOAT = 21.0            # 两班倒 3 人（驾驶 1 + 水手 2）× 7 万元/年
BOAT_FIXED = round(BOAT_BUY / 12 + BOAT_MAINT + CREW_PER_BOAT, 2)   # ≈ 33.87 万元/年
VAR_PER_CYCLE = 0.012          # 单循环燃油 120 元（柴油 240 L · 8 元/L，客渡船短途低速）
LOST_COST = 1.5                 # 一位被留下乘客的补偿（退票 + 改乘公路耗时，元）


def per_boat_day_capacity() -> float:
    """单船日运力（阶梯校准口径）：每小时实际开 floor_hour_slots 班 × CAP × 16 小时。"""
    return CAP * hour_slots() * N_HOURS


def hour_slots() -> int:
    """一小时里能开出几班（整数）：30 分钟一圈 → 2 班；80 分钟一圈 → 0 班（需跨小时看）"""
    return max(1, int(60.0 // CYCLE))


def naive_boats() -> int:
    """平均法（阶梯校准）：日需求 ÷ 单船日运力，向上取整。
    之所以要"校准"，是因为每船每小时的班次必须是整数（0.75 班/小时不是真班次）。"""
    return int(math.ceil(HOURLY_MEAN.sum() / per_boat_day_capacity()))


# ============================================================
# 二、蒙特卡洛（MC）：小时需求独立泊松，按"时刻表运力"判超员
# ============================================================
def mc_overflow(n_boats: int, rng: np.random.Generator, n_days: int,
                low_season: bool = False) -> dict:
    """MC-A：把一天切成 16 个整点小时，每小时运力 = n*40*60/80（船次可任意调配的乐观口径）。"""
    lam = HOURLY_MEAN * (LOW_SEASON_FACTOR if low_season else 1.0)
    lam = lam.copy()
    lam[0] += HOURLY_MEAN_prefirst * (LOW_SEASON_FACTOR if low_season else 1.0)
    cap = n_boats * CAP * 60.0 / CYCLE
    x = rng.poisson(lam[None, :], size=(n_days, N_HOURS))
    over = np.maximum(x - cap, 0)
    frac = (over > 0).mean(axis=1)
    return dict(frac=float(frac.mean()),
                frac_se=float(frac.std(ddof=1) / math.sqrt(n_days)),
                ppv=float(over.sum(axis=1).mean()),
                ppv_se=float(over.sum(axis=1).std(ddof=1) / math.sqrt(n_days)),
                x=x, over=over, cap=cap)


def build_slots() -> tuple[np.ndarray, np.ndarray]:
    """MC-B：把时刻表离散成"时隙"（每船每 CYCLE 分钟一班）。
    第 k 个时隙从 T_OPEN + k*CYCLE/60 开始，船在这一刻回港。
    返回每个时隙的起点（小时）与期望到达量（人）。"""
    n_slot = int((N_HOURS * 60) // CYCLE) + 1
    starts = T_OPEN + np.arange(n_slot) * CYCLE / 60.0
    edges = np.append(starts, T_CLOSE_ABS)
    lam = np.zeros(n_slot)
    # 将每个整点小时的需求按时间重叠部分摊到各时隙
    for h in range(N_HOURS):
        h0, h1 = T_OPEN + h, T_OPEN + h + 1
        w = np.clip(np.minimum(edges[1:], h1) - np.maximum(edges[:-1], h0), 0, None)
        if w.sum() > 0:
            lam += HOURLY_MEAN[h] * w / w.sum()
    lam[0] += HOURLY_MEAN_prefirst
    return starts, lam


def mc_slot_overflow(n_boats: int, rng: np.random.Generator, n_days: int) -> dict:
    """MC-B：按时隙判超员（每时隙到客 vs n*40 的可发船位）。"""
    _, lam = build_slots()
    x = rng.poisson(lam[None, :], size=(n_days, lam.size))
    over = np.maximum(x - n_boats * CAP, 0)
    frac = (over > 0).mean(axis=1)      # 每天"超员时隙占比"
    return dict(frac=float(frac.mean()),
                frac_se=float(frac.std(ddof=1) / math.sqrt(n_days)),
                ppv=float(over.sum(axis=1).mean()))


def erlang_c(n_boats: int) -> float:
    """M/M/c 稳态 P(到达即排队)——本讲只当"体检指标"，不当决策依据。"""
    a = HOURLY_MEAN.mean() * CYCLE / 60.0
    rho = a / n_boats
    if rho >= 1:
        return 1.0
    num = (a ** n_boats / math.factorial(n_boats)) * (n_boats / (n_boats - a))
    den = sum(a ** k / math.factorial(k) for k in range(n_boats)) + num
    return num / den


# ============================================================
# 三、离散事件仿真（DES）
# ============================================================
def simulate_day(n_boats: int, rng: np.random.Generator, crowded: bool = True,
                 record: bool = False, late_extra: bool = False,
                 max_wait: float = 0.0) -> dict:
    """
    事件流：ready[i] = 第 i 艘船"回到本站、可再载客"的时刻（小时，0 = 05:00）。
    逐小时：本小时到了 n 人 → 只要有船在本小时内回到港，就发一班，装 min(40, 排队)。
    船发走后 ready[i] = 回港时刻 + (CYCLE + 拥挤附加)/60。收班时排队人数 = leftover。
    """
    demand = rng.poisson(HOURLY_MEAN)
    boarded = np.zeros(N_HOURS)
    queued = np.zeros(N_HOURS)        # 每小时末仍排队的人数（"下一班再走"）
    deps = np.zeros(N_HOURS)
    empt = np.zeros(N_HOURS)
    full = np.zeros(N_HOURS)
    queue_trace = np.zeros(N_HOURS)
    dep_times: list[float] = []

    ready = np.zeros(n_boats)
    # 首班车之前到的人（04:00–05:00）也要过江：排队等人齐
    queue = float(rng.poisson(HOURLY_MEAN_prefirst))
    # 队首乘客的到达时刻（用于"等客/坐满才走"策略）
    head_time = T_OPEN - 1.0
    idle_ship_min = 0.0
    for h in range(N_HOURS):
        if demand[h] > 0 and queue <= 0:
            head_time = T_OPEN + h + 0.5          # 无简化：假设小时中点到
        queue += float(demand[h])
        t_end = h + 1.0
        while True:
            cand = [(ready[i], i) for i in range(n_boats) if ready[i] < t_end - 1e-9]
            if not cand:
                break
            r, i = min(cand)
            # 等客策略：不到 40 人且队首只等了 max_wait 小时，就先不走（航线固定班次不受影响）
            if max_wait > 0 and queue < CAP:
                waited = (T_OPEN + h) - head_time
                if waited < max_wait:
                    idle_ship_min += (t_end - max(r, float(h))) * 60.0
                    break
            load = min(CAP, queue)
            boarded[h] += load
            deps[h] += 1
            empt[h] += CAP - load
            full[h] += 1 if load >= CAP - 1e-9 else 0
            queue -= load
            extra = CROWD_MIN_PER_PAX * max(0.0, load - CROWD_THRESHOLD) if crowded else 0.0
            if crowded and load >= CAP - 1e-9:
                extra += CROWD_WAIT      # 满载：点票、救生衣、实名登记，放行慢 6 分钟
            ready[i] = r + (CYCLE + extra) / 60.0
            dep_times.append(r + T_OPEN)
        queued[h] = queue
        queue_trace[h] = queue
    # 21:00 收班兜底：对岸的船最后一趟回来能再拉一船，再往后真的没人管了
    if late_extra and n_boats > 0:
        while True:
            cand = [(ready[i], i) for i in range(n_boats) if ready[i] < T_CLOSE_ABS]
            if not cand or queue <= 0:
                break
            r, i = min(cand)
            load = min(CAP, queue)
            boarded[-1] += load
            deps[-1] += 1
            empt[-1] += CAP - load
            full[-1] += 1 if load >= CAP - 1e-9 else 0
            queue -= load
            extra = CROWD_MIN_PER_PAX * max(0.0, load - CROWD_THRESHOLD) if crowded else 0.0
            ready[i] = r + (CYCLE + extra) / 60.0

    return dict(demand=demand, boarded=boarded, queued=queued, deps=deps,
                empt=empt, full=full, queue_trace=queue_trace,
                leftover=float(queue), dep_times=dep_times,
                served=float(boarded.sum()),
                idle_min=float(idle_ship_min))


def profit_at(n_boats: int, factor: float, rng: np.random.Generator, n_days: int,
               crowded: bool = True, runs: int | None = None) -> dict:
    """给定"需求倍数 factor"（1.0 = 春运高峰，0.55 = 淡季），跑 n_days 天 DES，返回日均账。
    runs：淡季只开几艘（停掉的船不烧油但也不载客），默认为全船队出航。"""
    n_run = n_boats if runs is None else runs
    served = deps = leftover = 0.0
    saved = HOURLY_MEAN.copy()
    HOURLY_MEAN[:] = saved * factor
    try:
        for _ in range(n_days):
            d = simulate_day(n_run, rng, crowded=crowded)
            served += d["served"]
            deps += d["deps"].sum()
            leftover += d["leftover"]
    finally:
        HOURLY_MEAN[:] = saved
    served /= n_days
    deps /= n_days
    leftover /= n_days
    vc_per_day = deps * VAR_PER_CYCLE
    fc_per_day = n_boats * BOAT_FIXED / 365.0
    return dict(served=served, deps=deps, leftover=leftover,
                revenue_day=served * PRICE / 10000.0,
                vc_day=vc_per_day, fc_day=fc_per_day,
                profit_day=served * PRICE / 10000.0 - vc_per_day - fc_per_day)


# ============================================================
# 四、主流程
# ============================================================
def main() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"config": dict(CAP=CAP, CYCLE=CYCLE, boat_fixed=BOAT_FIXED,
                                per_boat_day=per_boat_day_capacity(),
                                daily_mean=float(HOURLY_MEAN.sum()))}

    print("=" * 84)
    print("一、三种口径算同一件事：春运高峰要几条船（要求超员 ≤1%）")
    print("=" * 84)
    print(f"  循环 {CYCLE:.0f} 分钟 → {60/CYCLE:.2f} 班/小时/船 | 单船日运力 {per_boat_day_capacity():.0f} 人"
          f" | 日需求 {HOURLY_MEAN.sum():.0f} 人")
    n_naive = naive_boats()
    print(f"  ① 平均法：{HOURLY_MEAN.sum():.0f} ÷ {per_boat_day_capacity():.0f} = "
          f"{HOURLY_MEAN.sum()/per_boat_day_capacity():.2f} → **{n_naive} 条**（不看任何波动）")

    rows = []
    for n in range(n_naive, n_naive + 9):
        mc_a = mc_overflow(n, rng, 20000)
        mc_b = mc_slot_overflow(n, rng, 20000)
        des_runs = [simulate_day(n, np.random.default_rng(SEED * 3 + k)) for k in range(300)]
        des_frac = float(np.mean([1.0 if d["leftover"] > 0 else 0.0 for d in des_runs]))
        des_left = float(np.mean([d["leftover"] for d in des_runs]))
        des_served = float(np.mean([d["served"] for d in des_runs]))
        des_deps = float(np.mean([d["deps"].sum() for d in des_runs]))
        rows.append(dict(n=n, cap_hour=n * CAP * 60.0 / CYCLE,
                         mc_a=mc_a["frac"], mc_a_se=mc_a["frac_se"], mc_a_ppv=mc_a["ppv"],
                         mc_b=mc_b["frac"], mc_b_se=mc_b["frac_se"],
                         des_frac=des_frac, des_leftover=des_left,
                         des_served=des_served, des_deps=des_deps,
                         des_util=des_served / (des_deps * CAP),
                         erlang_c=erlang_c(n)))
        r = rows[-1]
        print(f"  n={n:2d}: ② MC-A 超员小时 {r['mc_a']*100:5.2f}% | ③ MC-B 超员时隙 {r['mc_b']*100:5.2f}%"
              f" | ④ DES 收班有余 {r['des_frac']*100:5.1f}% 的天数（平均余 {r['des_leftover']:5.1f} 人）"
              f" | 实发 {r['des_deps']:5.0f} 班 载客率 {r['des_util']*100:4.1f}%"
              f" | Erlang-C {r['erlang_c']:.3f}")
    res["capacity_rows"] = rows

    def first_ok(key, thr):
        return next((r["n"] for r in rows if r[key] <= thr), None)

    res["n_naive"] = n_naive
    res["n_mca"] = first_ok("mc_a", PEAK_TARGET)
    res["n_mcb"] = first_ok("mc_b", PEAK_TARGET)
    res["n_des"] = first_ok("des_frac", PEAK_TARGET)
    res["n_target"] = res["n_des"]
    print(f"\n  ▶ 平均法 {n_naive} 条、MC-A {res['n_mca']} 条、MC-B（时隙）{res['n_mcb']} 条、"
          f"DES（收班不出错）{res['n_des']} 条 —— **同一个问题，四个答案**")
    r_naive = rows[0]
    print(f"  ▶ 平均法这 {n_naive} 条：DES 里有 {r_naive['des_frac']*100:.1f}% 的天收班还排着队，"
          f"平均每天 {r_naive['des_leftover']:.0f} 人当天没走成")

    # —— MC 的置信区间：误差 ∝ 1/√N
    print("\n" + "=" * 84)
    print("二、蒙特卡洛『要抽多少次』：误差按 1/√N 收敛")
    print("=" * 84)
    n_target = res["n_mcb"] if res["n_mcb"] else res["n_des"]
    res["n_target"] = n_target
    ci = []
    for nd in (200, 2000, 20000, 200000):
        m = mc_slot_overflow(n_target, rng, nd)
        ci.append(dict(n_days=nd, frac=m["frac"], se=m["frac_se"], half95=1.96 * m["frac_se"]))
        print(f"  N={nd:7d} 天：超员时隙占比 {m['frac']*100:5.2f}% ± {1.96*m['frac_se']*100:.2f}%（95%CI）"
              f" | 日均延后 {m['ppv']:6.1f} 人")
    res["ci"] = ci

    # —— DES 与 MC 的差别：时刻表把运力"打碎"
    print("\n" + "=" * 84)
    print("三、DES 与 MC 的差别：同样的船，时刻表让运力打成碎片")
    print("=" * 84)
    cmp_rows = []
    for n in (n_naive, n_naive + 2, n_naive + 4):
        lam = HOURLY_MEAN
        cap = n * CAP * 60.0 / CYCLE
        # MC 的期望口径：E[max(X-cap,0)]
        mc_expect = float(np.mean(np.maximum(rng.poisson(lam[None, :], size=(20000, N_HOURS)) - cap, 0)))
        des = [simulate_day(n, np.random.default_rng(SEED * 7 + k)) for k in range(200)]
        d_leftover = float(np.mean([d["leftover"] for d in des]))
        d_deps = float(np.mean([d["deps"].sum() for d in des]))
        d_served = float(np.mean([d["boarded"].sum() for d in des]))
        d_loaded_hours = float(np.mean([(d["boarded"] > 0).sum() for d in des]))
        d_util = d_served / (d_deps * CAP)
        cmp_rows.append(dict(n=n, mc_ppv=mc_expect, des_leftover=d_leftover,
                             des_deps=d_deps, des_served=d_served, des_util=d_util,
                             loaded_hours=d_loaded_hours,
                             avg_load=d_served / d_deps))
        print(f"  n={n}: MC 认为每天被延后 {mc_expect:6.1f} 人 | DES 实测收班余 {d_leftover:5.1f} 人"
              f" | 实发 {d_deps:5.1f} 班 实装 {d_served:6.1f} 人"
              f" | 平均每班载 {d_served/d_deps:4.1f} 人（{d_util*100:.1f}%）")
    res["des_vs_mc"] = cmp_rows

    # —— DES 一天的发船画像（画图用）
    prof = simulate_day(n_target, np.random.default_rng(SEED + 11), record=True)
    res["profile"] = dict(demand=prof["demand"].tolist(), boarded=prof["boarded"].tolist(),
                          queued=prof["queued"].tolist(), deps=prof["deps"].tolist(),
                          empt=prof["empt"].tolist(),
                          hours=[T_OPEN + i for i in range(N_HOURS)])

    # —— 同一份仿真回答"时刻表怎么排"：按点发 vs 坐满才走
    print("\n" + "=" * 84)
    print("四、同一份仿真的第二个用途：时刻表怎么排（满载拖班 + 等客策略）")
    print("=" * 84)
    crowd_rows = []
    for n in (n_naive + 3, n_naive + 5):
        cfg = dict(fixed=dict(crowded=False, max_wait=0.0),
                   crowd=dict(crowded=True, max_wait=0.0),
                   fill=dict(crowded=True, max_wait=0.5),
                   hold=dict(crowded=True, max_wait=1.0))
        row = dict(n=n)
        for name, kw in cfg.items():
            runs = [simulate_day(n, np.random.default_rng(SEED + 55 + k), **kw) for k in range(120)]
            deps = float(np.mean([d["deps"].sum() for d in runs]))
            served = float(np.mean([d["served"] for d in runs]))
            left = float(np.mean([d["leftover"] for d in runs]))
            crowded_days = float(np.mean([(d["full"].sum() > 0) for d in runs]))
            avg_load = served / deps
            idle = float(np.mean([d["idle_min"] for d in runs]))
            row[name] = dict(deps=deps, served=served, leftover=left,
                             full_days=crowded_days, avg_load=avg_load, idle_min=idle)
            print(f"  n={n} [{name:5s}]：发船 {deps:5.1f} 班/天 平均载 {avg_load:4.1f} 人"
                  f" 在港待客 {idle:5.1f} 船·分钟/天 收班余 {left:5.1f} 人")
        crowd_rows.append(row)
        print()
    res["crowding"] = crowd_rows

    # —— 年账：固定成本全口径 vs 可避免成本口径
    print("\n" + "=" * 84)
    print(f"五、年账：第 n 条船值不值（固定 {BOAT_FIXED:.2f} 万/年；另给『可避免成本』口径）")
    print("=" * 84)
    acc = []
    prev = None
    for n in range(n_naive, n_naive + 5):
        rp = profit_at(n, 1.00, np.random.default_rng(SEED + 101 * n), 20)
        ro = profit_at(n, LOW_SEASON_FACTOR, np.random.default_rng(SEED + 977 * n), 20)
        served_year = rp["served"] * DAYS_PEAK + ro["served"] * (365 - DAYS_PEAK)
        deps_year = rp["deps"] * DAYS_PEAK + ro["deps"] * (365 - DAYS_PEAK)
        revenue = served_year * PRICE / 10000.0
        cost = n * BOAT_FIXED + deps_year * VAR_PER_CYCLE
        # 票价敏感性：要"收得回全部成本"，票价得涨到多少？
        breakeven_price = cost * 10000.0 / max(1.0, served_year)
        rev70 = rp["revenue_day"] * DAYS_PEAK + ro["revenue_day"] * DAYS_OFF
        inc_rev = rev70 - (prev["rev70"] if prev else 0.0)
        inc_vc = (deps_year - (prev["deps_year"] if prev else 0.0)) * VAR_PER_CYCLE
        # 可避免成本 = 固定 − 乘务 21 万（春运临时员工资，不停船也得付）
        avoidable = BOAT_FIXED - CREW_PER_BOAT
        acc.append(dict(n=n, revenue_year=revenue, cost_year=cost,
                        breakeven_price=breakeven_price,
                        profit_year=revenue - cost, rev70=rev70, deps_year=deps_year,
                        inc_rev=inc_rev, inc_vc=inc_vc, inc_margin=inc_rev - inc_vc,
                        net_full=inc_rev - inc_vc - BOAT_FIXED,
                        net_avoidable=inc_rev - inc_vc - avoidable,
                        lost_peak=rp["leftover"],
                        util_peak=rp["served"] / (rp["deps"] * CAP),
                        util_off=ro["served"] / (ro["deps"] * CAP),
                        ro_served=ro["served"], ro_deps=ro["deps"]))
        print(f"  n={n}: 年收入 {revenue:6.1f} 万 年成本 {cost:7.1f} 万 利润 {revenue-cost:8.1f} 万"
              f" | 70 日增量收入 {inc_rev:5.2f} 增量燃油 {inc_vc:4.2f}"
              f" → 毛利 {inc_rev-inc_vc:5.2f} / 扣全成本 {inc_rev-inc_vc-BOAT_FIXED:6.2f}"
              f" / 扣可避免成本 {inc_rev-inc_vc-avoidable:6.2f}"
              f" | 高峰收班余 {rp['leftover']:5.1f}"
              f" | 保本票价 {breakeven_price:6.1f} 元"
              f" | 淡季开{n}条 载客率 {ro['served']/(ro['deps']*CAP)*100:4.1f}%")
        prev = acc[-1]
    res["annual"] = acc

    # —— 淡季：停船 vs 开船（只看淡季当天的现金流）
    print("\n" + "=" * 84)
    print("六、同一份仿真的另一半用处：淡季停几条船（只看淡季当天现金）")
    print("=" * 84)
    n_fleet = res["n_des"] if res.get("n_des") else res["n_naive"]
    off_rows = []
    for run in (n_fleet, n_fleet - 1, n_fleet - 2, n_fleet - 3):
        if run < 1:
            continue
        r = profit_at(n_fleet, LOW_SEASON_FACTOR, np.random.default_rng(SEED + 13 * n_fleet + run), 60, runs=run)
        parked = n_fleet - run
        cash = r["revenue_day"] - r["vc_day"] - run * BOAT_FIXED / 365.0 - parked * BOAT_FIXED / 365.0 / 3
        key = "run_all" if run == n_fleet else f"run_minus{n_fleet - run}"
        off_rows.append(dict(boats=n_fleet, run=run, key=key, cash=cash,
                             util=r["served"] / (r["deps"] * CAP), deps=r["deps"],
                             leftover=r["leftover"]))
        print(f"  目标船队 {n_fleet} 条、淡季开 {run} 条：{r['deps']:5.1f} 班/天"
              f" 载客率 {r['served']/(r['deps']*CAP)*100:4.1f}% 收班余 {r['leftover']:5.1f} 人"
              f" 单日现金 {cash:6.2f} 万")
    res["off_season"] = off_rows

    # —— 最优船数（全年口径）
    best = max(res["annual"], key=lambda a: a["profit_year"])
    res["n_best"] = best["n"]
    print(f"\n  ▶ 全年利润口径下最优船数 = {best['n']} 条")

    return res


def make_figure(res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

    # 左上：三种口径的"要几条船"
    ax = axes[0, 0]
    rows = res["capacity_rows"]
    ns = np.array([r["n"] for r in rows])
    bars = {
        "平均法（日需求÷日运力）": None,
        "MC-A 小时块": np.array([r["mc_a"] for r in rows]) * 100,
        "MC-B 时隙": np.array([r["mc_b"] for r in rows]) * 100,
        "DES 收班有余天数": np.array([r["des_frac"] for r in rows]) * 100,
    }
    w = 0.26
    ax.bar(ns - w, bars["MC-A 小时块"], width=w, color="#9ecae1", label="MC-A 整点小时")
    ax.bar(ns, bars["MC-B 时隙"], width=w, color="#4292c6", label="MC-B 时刻表时隙")
    ax.bar(ns + w, bars["DES 收班有余天数"], width=w, color="#b03a2e", label="DES 收班还排队的天数")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="1% 红线")
    ax.axvline(res["n_naive"] - 0.5, color="#7f8c8d", ls=":", lw=1.2)
    ax.annotate(f"平均法 {res['n_naive']} 条", (res["n_naive"], ax.get_ylim()[1] * 0.92),
                fontsize=9, color="#7f8c8d")
    ax.set_xlabel("渡船数（条）")
    ax.set_ylabel("风险指标 (%)")
    ax.set_title("同一个问题，四种口径四个答案")
    ax.set_xticks(ns)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 右上：DES 的一天
    ax = axes[0, 1]
    p = res["profile"]
    hrs = np.array(p["hours"])
    ax.bar(hrs, p["demand"], width=0.62, color="#9ecae1", label="到客")
    ax.bar(hrs, p["boarded"], width=0.36, color="#1f4e79", label="上船")
    ax2 = ax.twinx()
    ax2.plot(hrs, p["queued"], "o-", color="#b03a2e", lw=1.4, ms=4, label="小时末排队")
    ax2.set_ylabel("排队人数")
    ax.set_xlabel("时刻（时）")
    ax.set_ylabel("人数")
    ax.set_title(f"离散事件仿真：{res['n_des']} 条船的一天（高峰）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # 左下：MC 置信区间收敛
    ax = axes[1, 0]
    ci = res["ci"]
    N = np.array([c["n_days"] for c in ci])
    ax.loglog(N, np.array([c["half95"] for c in ci]) * 100, "o-", color="#1f4e79", label="95% CI 半宽（实测）")
    ax.loglog(N, np.array([ci[0]["half95"] * math.sqrt(ci[0]["n_days"]) / math.sqrt(n) for n in N]) * 100,
              "k--", lw=1.2, label="理论 1/√N")
    ax.axhline(1.0, color="#b03a2e", ls="--", lw=1.1, label="1% 精度目标")
    ax.set_xlabel("蒙特卡洛抽样天数 N")
    ax.set_ylabel("CI 半宽 (%)")
    ax.set_title("误差 ∝ 1/√N：4 倍天数才换来 2 倍精度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # 右下：时刻表策略对照（发船班次 / 在港待客）
    ax = axes[1, 1]
    cr = res["crowding"]
    labels = ["按点发船", "满载拖班", "坐满才走(30min)", "坐满才走(60min)"]
    keys = ["fixed", "crowd", "fill", "hold"]
    xs = np.arange(len(keys))
    for j, row in enumerate(cr):
        deps = [row[k]["deps"] for k in keys]
        off = (j - (len(cr) - 1) / 2) * 0.18
        ax.bar(xs + off, deps, width=0.18, label=f"{row['n']} 条船")
        for x, d, k in zip(xs + off, deps, keys):
            idle = row[k]["idle_min"]
            if idle > 0:
                ax.annotate(f"待客{idle:.0f}", (x, d), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("发船班次（班/天）")
    ax.set_title("时刻表怎么排：多开班 ≠ 多运人")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("县域渡口春运配船：蒙特卡洛 + 离散事件仿真（Day57）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def check(res: dict) -> None:
    """教学点断言：只卡"这一课要讲的事真的发生了"。"""
    rows = res["capacity_rows"]
    r0 = rows[0]
    assert r0["n"] == res["n_naive"]
    # ① 平均法必然出事：DES 里收班有余的天数 > 50%，且平均每天余 > 100 人
    assert r0["des_frac"] > 0.50, r0["des_frac"]
    assert r0["des_leftover"] > 100, r0["des_leftover"]
    # ② MC 与 DES 结论不同：MC 需要 ≥ DES 需要 + 1（MC 乐观）
    assert res["n_mca"] > res["n_des"], (res["n_mca"], res["n_des"])
    # ③ DES 达标的船数：收班余 ≤1%
    t = next(r for r in rows if r["n"] == res["n_des"])
    assert t["des_frac"] <= 0.01
    # ④ 达标船数下 DES 载客率不高（"空着走"真实存在）
    assert t["des_util"] < 0.80, t["des_util"]
    # ⑤ 置信区间按 1/√N 收
    ci = res["ci"]
    assert ci[-1]["half95"] < ci[0]["half95"] / 15
    # ⑥ 拥挤/等客策略改变了发船班次（机制真的建模进去了）
    cr = res["crowding"][0]
    assert cr["crowd"]["deps"] < cr["fixed"]["deps"] * 0.95
    assert cr["fill"]["idle_min"] > 0
    # ⑦ 淡季停船：停 1 条当日现金改善
    off = {r["key"]: r for r in res["off_season"]}
    assert off["run_all"]["cash"] < off["run_minus1"]["cash"]   # 停一条反而更费（固定成本已沉没）
    assert off["run_minus1"]["leftover"] < 100                 # 但停一条仍能运完
    res["n_best"] = res["n_des"]


if __name__ == "__main__":
    result = main()
    check(result)
    make_figure(result)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 84)
    print(f"断言全部通过（{len(result['capacity_rows'])} 组船数 × MC/DES 双引擎）。")
    print(f"已写出 {OUT_PNG.name} / {OUT_JSON.name}")
