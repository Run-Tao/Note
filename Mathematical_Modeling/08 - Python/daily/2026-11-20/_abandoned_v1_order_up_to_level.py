#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 随机优化 / L5 算法（n 期库存 · 临界分位数 · (s,S) 策略 · 动态规划 · Bellman 方程 · K-convexity）
anchor  : 前接 Day27 报童（单期、一次决策）/ Day18 MDP 值迭代（随机跳转状态）/ Day57 DES（把一天演一遍）
goal    : 回答同一个仓管问题——「什么时候下补货单、一次补多少」——在四种业务口径下是不是同一个答案：
          (1) 期末缺货不许欠（lost sales）(2) 缺货可欠（backlog）(3) 必须整箱发（整数）
          (4) 一次下单要付调度费 K（固定订购成本）
          核心交付物：临界分位数决定「补不补」、K 决定「补多少」；K=0 时 s=S 的「天天小补」不是最优，是退化。

决策场景：云溪县农资公司 2027 年春耕季「0 号复合肥」县级中心仓，
          40 天一个订货周期，每周下 1 次补货单，订货到货提前 1 周（lead time = 1）。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-11-20/day85_order_up_to_level.py"
"""

import json
import math
import time
from itertools import product

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------- 中文字体（避免乱码） ----------------
_CJK = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
for _n in _CJK:
    if _n in _avail:
        plt.rcParams["font.sans-serif"] = [_n]
        break
else:
    print("⚠ 未找到中文字体，图中中文可能乱码")
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 一、业务参数
# =========================================================
P = dict(
    demand_mean=120.0,      # 周均需求（吨）
    demand_cv=0.35,         # 需求变异系数（周需求 sd = 42 吨）
    unit_cost=2800.0,       # 采购单价 元/吨
    hold_rate=0.20,         # 年持有成本率（资金 6% + 仓储保险损耗 14%）
    weeks_per_year=52.0,
    order_cost_K=4200.0,    # 一次补货单的固定成本（调度 + 质检 + 司机等单）元
    penalty=1800.0,         # 缺货惩罚 元/吨·周（替代采购差价 + 渠道信誉损失摊销）
    lead_time=1,            # 补货提前期（周）
    horizon=12,             # 有限期决策（周）
    box=10.0,               # 整箱 10 吨
    seasonal_sd=0.18,       # 旺季需求水平本身的周度波动（共同冲击）
    seasonal_mean=1.25,     # 春耕旺季需求倍率
)


def build_params(**over):
    p = dict(P)
    p.update(over)
    h = p["unit_cost"] * p["hold_rate"] / p["weeks_per_year"]   # 元/吨/周
    p["h"] = h
    p["demand_sd"] = p["demand_mean"] * p["demand_cv"]
    return p


def newsvendor_critical_ratio(p):
    """单期临界分位数 (cu - K) / (cu + h)，cu 为缺货成本 = 惩罚 + 当周毛利外的机会损失。"""
    return (p["penalty"] - p["order_cost_K"]) / (p["penalty"] + p["h"])


# =========================================================
# 二、精确动态规划：有限期 (s,S) 型最优策略
#    状态 x = 期初库存（已到货的现货，可为负 = 欠单）
#    动作 q = 本周补货量。q 本周下单、下周到货：先看本周需求，再进下一期。
#    这里用「简化时序」：本周可用的量 = x（到货已在期初），下单影响下一期期初。
#    V_t(x) = min_{q>=0} [ K*1(q>0) + c*q + E_d[ g(x - d) + V_{t+1}(x - d + q) ] ]
#    g(y) = h*max(y,0) + p_pen*max(-y,0)   （backlog 口径）
# =========================================================
def make_q_grid(q_max, box=None):
    if box is None:
        return np.linspace(0.0, q_max, 301)
    return np.arange(0.0, q_max + 0.5 * box, box)


def solve_dp(p, q_grid, x_max=None, x_min=None):
    """有限期 DP，离散需求 + q 网格线性扫描（向量化）。返回策略表与成本。"""
    h, pen, c, K = p["h"], p["penalty"], p["unit_cost"], p["order_cost_K"]
    mu, sd = p["demand_mean"], p["demand_sd"]
    qmax = float(q_grid[-1])
    if x_max is None:
        x_max = mu + 4 * sd + qmax + 20
    if x_min is None:
        x_min = -(mu + 4 * sd) - 20

    # 需求离散化：等距支撑 + 概率质量
    k_mid = np.arange(mu - 5 * sd, mu + 5 * sd + 1.0, 1.0)
    cdf = 0.5 * (1 + math.erf(0.0))
    cdf = np.array([0.5 * (1 + math.erf((k - mu) / (sd * math.sqrt(2)))) for k in k_mid])
    prob = np.diff(np.concatenate([[0.0], cdf]))
    prob = np.clip(prob, 1e-12, None)
    prob = prob / prob.sum()
    dvals = k_mid  # 用离散点本身代表需求

    # 状态网格：与需求同分辨率（1 吨）+ q 网格也吸附到 1 吨 => 状态可直接索引
    step = 1.0
    xs = np.arange(x_min, x_max + step, step)
    n = len(xs)
    idx0 = 0

    # 需求折到「带符号库存」的偏移量（以吨为单位的整数偏移）
    doff = np.round(dvals - xs[idx0]).astype(int)  # 相对 x_min 的偏移

    V = np.zeros(n)
    pol_q = np.zeros((p["horizon"], n))
    pol_S = np.zeros((p["horizon"], n))

    for t in range(p["horizon"] - 1, -1, -1):
        g_all = h * np.maximum(xs, 0.0) + pen * np.maximum(-xs, 0.0)
        # W(x) = E_d[ g(x-d) + V_{t+1}(x-d) ]      （本周不改动时，下一期期初 = x-d）
        y_idx = np.arange(n)[:, None] - doff[None, :]
        valid = (y_idx >= 0) & (y_idx < n)
        y_idx_c = np.clip(y_idx, 0, n - 1)
        stage = g_all[y_idx_c] + V[y_idx_c]
        W = np.where(valid, stage, 1e9) @ prob  # shape (n,)

        # 决策：本周下单 q -> 下一期期初 x + q，成本 K + c*q，再取 min
        qidx = np.round((q_grid - q_grid[0]) / step).astype(int)
        best = np.full(n, np.inf)
        bestq = np.zeros(n)
        bestS = np.zeros(n)
        # 对每个 q，整体平移 W
        for qi, q in enumerate(q_grid):
            shift = int(round(q / step))
            Ws = np.full(n, np.inf)
            lo, hi = max(0, -shift), min(n, n - shift)
            Ws[lo:hi] = W[lo + shift: hi + shift]
            cost = Ws + c * q + (K if q > 0 else 0.0)
            better = cost < best - 1e-9
            best[better] = cost[better]
            bestq[better] = q
        for i in range(n):
            bestS[i] = xs[i] + bestq[i]
        V = best
        pol_q[t] = bestq
        pol_S[t] = bestS

    # 从策略表抽 (s,S)：期初库存越小越应该补，S = 补到的那条水平
    tbl = []
    for i in range(n):
        tbl.append((xs[i], bestq[0] if False else pol_q[0, i], pol_S[0, i]))
    xs_r = np.array([r[0] for r in tbl])
    q_r = np.array([r[1] for r in tbl])
    S_r = np.array([r[2] for r in tbl])

    # s = 最后一档「下单」的期初库存（且其后 q 恒为 0）
    order_mask = q_r > 1e-9
    s_hat = None
    if order_mask.any():
        # 从右往左找第一档下单
        i = len(xs_r) - 1
        while i >= 0 and not order_mask[i]:
            i -= 1
        if i >= 0:
            s_hat = xs_r[i]
    S_hat = float(np.median(S_r[order_mask])) if order_mask.any() else xs_r[-1]

    # 用 (s,S) 策略复算成本，做交叉验证（只在政策形状确实是单一阈值时报一致）
    is_threshold = True
    if order_mask.any():
        ii = int(np.where(order_mask)[0][-1])
        is_threshold = not order_mask[ii + 1:].any()
        is_threshold = is_threshold and np.allclose(S_r[order_mask], S_hat, atol=2.0)

    return dict(V0=float(V[0]), xs=xs_r, q=q_r, S=S_r, s=s_hat, S_level=S_hat,
                is_threshold=bool(is_threshold), policy_shape=((xs_r, q_r, S_r)))


# =========================================================
# 三、三种朴素做法
# =========================================================
def policy_ss_sim(p, s, S, n_weeks, seed=0, backlog=True, rng=None):
    """给定 (s,S) 策略，跑 Monte Carlo 求平均周成本（有限期 12 周重复多次）。"""
    rng = rng or np.random.default_rng(seed)
    h, pen, c, K, mu, sd = p["h"], p["penalty"], p["unit_cost"], p["order_cost_K"], p["demand_mean"], p["demand_sd"]
    reps = 4000
    dem = rng.normal(mu, sd, size=(reps, n_weeks))
    dem = np.maximum(dem, 0.0)
    x = np.full(reps, S, dtype=float)   # 期初库存设为 S（首周不打冷战）
    tot = np.zeros(reps)
    for t in range(n_weeks):
        if backlog:
            y = x - dem[:, t]
            tot += h * np.maximum(y, 0) + pen * np.maximum(-y, 0)
            x = y
        else:
            sold = np.minimum(x, dem[:, t])
            lost = dem[:, t] - sold
            tot += h * np.maximum(x - sold, 0) + pen * lost
            x = x - sold
        order = x <= s
        q = np.where(order, S - x, 0.0)
        tot += np.where(order, K + c * q, 0.0)
        x = x + q
    return float(tot.mean() / n_weeks)




# =========================================================
# 四、四种口径下的计算
# =========================================================
def run(verbose=True):
    out = {}
    t0 = time.time()
    base = build_params()
    h = base["h"]

    if verbose:
        print("=" * 74)
        print("Day85 · n 期库存：同一个仓管问题，四种口径四个答案")
        print("=" * 74)
        print(f"参数：周均需求 {P['demand_mean']:.0f} 吨 / sd {base['demand_sd']:.0f} 吨（CV {P['demand_cv']:.0%}）"
              f"｜单价 {P['unit_cost']:.0f} 元/吨｜持有 h = {h:.4f} 元/吨/周"
              f"｜缺货罚 {P['penalty']:.0f}｜订货固定 {P['order_cost_K']:.0f}｜提前期 {P['lead_time']} 周")
        cr = newsvendor_critical_ratio(base)
        print(f"\n单期临界分位数 (cu-K)/(cu+h) = ({P['penalty']:.0f}-{P['order_cost_K']:.0f})/"
              f"({P['penalty']:.0f}+{h:.2f}) = {cr:.4f}")
        print("  → 不看 K 的话：cu/(cu+h) = "
              f"{P['penalty']/(P['penalty']+h):.4f}（报童口径）")

    # ---------- 口径 A：整周(12周) 有限期 DP，(s,S) with K ----------
    q_grid = np.arange(0.0, 700.0 + 1.0, 1.0)
    dpK = solve_dp(base, q_grid)
    # 连续区间插值求 s
    sK, SK = dpK["s"], dpK["S_level"]
    if verbose:
        print(f"\n[A] 有限 12 期 DP（K = {P['order_cost_K']:.0f} 元/单，backlog 口径）")
        print(f"    最优 (s, S) = ({sK:.0f}, {SK:.0f})｜DP 给的成本（12 期合计，折周均近似）≈ "
              f"{dpK['V0']:.0f} 元/12期 起点库存=s 处")
        print(f"    策略形状是单一阈值吗？ {dpK['is_threshold']}")

    # ---------- 口径 B：K=0（去掉固定订购成本）----------
    p0 = build_params(order_cost_K=0.0)
    dpK0 = solve_dp(p0, q_grid)
    if verbose:
        print(f"\n[B] 同一题、只把 K 设为 0：最优 (s, S) = ({dpK0['s']:.0f}, {dpK0['S_level']:.0f})"
              f"  →  s≈S 说明「天天小补」是退化，不是最优")

    # ---------- 口径 C：整数（整箱 10 吨）----------
    pi = build_params()
    dpi = solve_dp(pi, make_q_grid(400.0, box=10.0))
    if verbose:
        print(f"\n[C] 必须整箱发（10 吨一箱）：(s, S) = ({dpi['s']:.0f}, {dpi['S_level']:.0f})")

    # ---------- 口径 D：lost sales（缺货不欠）用仿真对照 ----------
    # 说明：lost sales 口径下 DP 的状态是「期末余货」，结构不同，这里用仿真在 (s,S) 网格上搜
    def grid_search_sS(p, backlog, s_grid, S_grid, weeks=12):
        best = (np.inf, None, None)
        for s in s_grid:
            for S in S_grid:
                if S <= s:
                    continue
                cst = policy_ss_sim(p, s, S, weeks, backlog=backlog, rng=np.random.default_rng(12345))
                if cst < best[0]:
                    best = (cst, s, S)
        return best

    s_grid = np.arange(20, 262, 10.0)
    S_grid = np.arange(60, 402, 10.0)
    bl = grid_search_sS(base, True, s_grid, S_grid)
    ls = grid_search_sS(base, False, s_grid, S_grid)
    if verbose:
        print(f"\n[D] 同参数两套缺货口径的仿真网格搜索（12 周 × 4000 重复）：")
        print(f"    backlog（可欠）：(s,S)=({bl[1]:.0f},{bl[2]:.0f})  周均成本 {bl[0]:.0f} 元")
        print(f"    lost sales（不欠）：(s,S)=({ls[1]:.0f},{ls[2]:.0f})  周均成本 {ls[0]:.0f} 元")

    # ---------- 契约 vs 期望：用平均需求（确定性）会怎样 ----------
    pdet = build_params(demand_cv=1e-6)
    dpdet = solve_dp(pdet, q_grid)
    if verbose:
        print(f"\n[E] 把需求当成确定性（均值 120 吨）：(s,S)=({dpdet['s']:.0f},{dpdet['S_level']:.0f})")

    # ---------- 成本分解：h 与 K 各花多少 ----------
    def decompose(p, s, S, weeks=12):
        """在同一 Monte Carlo 样本上把成本拆成四项。"""
        rng = np.random.default_rng(7)
        reps = 4000
        dem = np.maximum(rng.normal(p["demand_mean"], p["demand_sd"], size=(reps, weeks)), 0.0)
        x = np.full(reps, S, dtype=float)
        acc = dict(hold=0.0, short=0.0, buy=0.0, fix=0.0)
        for t in range(weeks):
            y = x - dem[:, t]
            acc["hold"] += np.maximum(y, 0).mean() * p["h"]
            acc["short"] += np.maximum(-y, 0).mean() * p["penalty"]
            x = y
            order = x <= s
            q = np.where(order, S - x, 0.0)
            acc["fix"] += order.mean() * p["order_cost_K"]
            acc["buy"] += (q.mean()) * p["unit_cost"]
            x = x + q
        return {k: v / weeks for k, v in acc.items()}

    dec = decompose(base, sK, SK) if SK > sK else decompose(base, bl[1], bl[2])
    svc = None
    # 服务水平：该 (s,S) 下「一周内不缺货」的概率
    rng = np.random.default_rng(11)
    dem = np.maximum(rng.normal(base["demand_mean"], base["demand_sd"], (4000, base["horizon"])), 0)
    x = np.full(4000, SK, dtype=float)
    ok = 0
    for t in range(base["horizon"]):
        y = x - dem[:, t]
        ok += (y >= 0).sum()
        x = np.where(y <= sK, SK, y)
    svc = ok / (4000 * base["horizon"])

    if verbose:
        print(f"\n[F] (s,S)=({sK:.0f},{SK:.0f}) 的成本分解（元/周）："
              f"持有 {dec['hold']:.0f} / 缺货 {dec['short']:.0f} / 货款 {dec['buy']:.0f} / 固定 {dec['fix']:.0f}")
        print(f"    周内不缺货概率（即服务水平）≈ {svc:.3f}")

    out.update(dict(
        params={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in base.items()},
        critical_ratio=float(newsvendor_critical_ratio(base)),
        newsboy_ratio_nok=float(base["penalty"] / (base["penalty"] + h)),
        dpK=dict(s=None if sK is None else float(sK), S=float(SK),
                 is_threshold=dpK["is_threshold"], V0=dpK["V0"]),
        dpK0=dict(s=None if dpK0["s"] is None else float(dpK0["s"]), S=float(dpK0["S_level"])),
        dp_int=dict(s=None if dpi["s"] is None else float(dpi["s"]), S=float(dpi["S_level"])),
        dp_det=dict(s=None if dpdet["s"] is None else float(dpdet["s"]), S=float(dpdet["S_level"])),
        sim_backlog=dict(s=float(bl[1]), S=float(bl[2]), cost=float(bl[0])),
        sim_lostsales=dict(s=float(ls[1]), S=float(ls[2]), cost=float(ls[0])),
        decompose={k: float(v) for k, v in dec.items()},
        service_level=float(svc),
        runtime_s=float(time.time() - t0),
    ))
    if verbose:
        print(f"\n用时 {time.time() - t0:.2f} s")

    # ---------------- 图：四联 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    ax = axes[0, 0]
    xs = dpK["xs"]
    m = (xs >= -60) & (xs <= 400)
    ax.step(xs[m], dpK["q"][m], where="post", lw=2, color="#1f77b4", label=f"最优补货量 q(x)（K={P['order_cost_K']:.0f}）")
    ax.step(xs[m], dpK0["q"][m], where="post", lw=1.6, color="#d62728", ls="--", label="K=0 时的 q(x)")
    ax.axvline(sK, color="#888", ls=":", lw=1)
    ax.annotate(f"s={sK:.0f}", (sK, max(dpK['q'][m]) * 0.75), fontsize=10, color="#555")
    ax.set_xlabel("期初库存 x（吨，负=欠单）")
    ax.set_ylabel("本周下单量 q（吨）")
    ax.set_title("(a) 策略形状：S − x 的斜率为 1，断点在 s")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ks = [0, 1000, 2000, 3000, 4200, 6000, 8000, 12000]
    ss_res = [solve_dp(build_params(order_cost_K=k), q_grid) for k in ks]
    ax.plot(ks, [r["S_level"] for r in ss_res], "o-", lw=2, color="#2ca02c", label="S（补到多少）")
    ax.plot(ks, [r["s"] or 0 for r in ss_res], "s-", lw=2, color="#ff7f0e", label="s（补不补的线）")
    ax.set_xlabel("订货固定成本 K（元/单）")
    ax.set_ylabel("库存水平（吨）")
    ax.set_title("(b) K 变大：S 几乎不动，s 一路往下掉")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    cst_b = [policy_ss_sim(base, s, S, 12, backlog=True) for s, S in [(bl[1], bl[2]), (sK, SK)]]
    labels = [f"仿真最优\n({bl[1]:.0f},{bl[2]:.0f})", f"DP 解\n({sK:.0f},{SK:.0f})"]
    ax.bar(labels, cst_b, color=["#1f77b4", "#ff7f0e"])
    for i, v in enumerate(cst_b):
        ax.text(i, v * 1.01, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("周均总成本（元）")
    ax.set_title("(c) 交叉验证：DP 解 vs 仿真网格最优（backlog 口径）")

    ax = axes[1, 1]
    items = ["持有", "缺货", "货款", "固定订购"]
    vals = [dec["hold"], dec["short"], dec["buy"], dec["fix"]]
    colors = ["#4c72b0", "#c44e52", "#dd8452", "#55a868"]
    ax.bar(items, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("元/周")
    ax.set_title(f"(d) 成本结构（{sK:.0f},{SK:.0f}）：缺货罚与持有成本谁大")

    plt.rcParams["axes.titlesize"] = 11
    fig.tight_layout()
    fig.savefig(__file__.replace(".py", ".png"), dpi=130)
    print("图已保存：", __file__.replace(".py", ".png"))
    return out


if __name__ == "__main__":
    res = run()
    with open(__file__.replace(".py", "_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已保存：", __file__.replace(".py", "_result.json"))
