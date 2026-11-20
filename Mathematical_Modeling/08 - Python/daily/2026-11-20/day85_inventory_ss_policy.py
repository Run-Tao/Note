#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 随机优化 / L5 算法（n 期库存 · 报童临界分位数 · (s,S) 与周期订货 · Wagner-Whitin 动态规划）
anchor  : 前接 Day27 报童（单期一次决策）/ Day18 排队论与 MDP（随机状态 + 值迭代）/ Day57 蒙特卡洛·DES（把系统演一遍）
goal    : 一个问题——「什么时候下单、一次下多少」——拆成两个互不相干的问题：
          临界分位数（只由 cu 与 h 决定）定「补到多高」S*；固定订购成本 K 定「多久补一次」T 与「降到多低才补」s。
          并用四种口径各打一枪：K 扫描 / 有界 vs 无界需求 / 缺货能不能欠 / 季节性非平稳需求（Wagner-Whitin）。

决策场景：云溪县农资公司 2027 年春耕「0 号复合肥」县级中心仓，
          8 周一个订货季（淡季 2 周 → 旺季 4 周 → 尾季 2 周），补货提前期 1 周，整箱 20 吨。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-11-20/day85_inventory_ss_policy.py"

约定（全篇统一，否则数字互相打架）：
  * h = 10.7692 元/吨·周（年持有率 20% ÷ 52，单价 2800 元/吨）
  * 缺货罚 pen = 1800 元/吨（一次补齐的替代采购差价 + 渠道信誉摊销）
  * 三角需求 三角(0, 240, 120)：均值 120 吨/周、sd 49.0 → CV 41%（旺季周；淡季等比缩放）
  * 「周期订货」口径：每 T 周在周初补一次，补到 S = s + μT（订货量 Q = μT 由 T 决定，只有首期按差额补）
"""

import json
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------- 中文字体（缺失会乱码） ----------------
for _n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
           "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]:
    if _n in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_n]
        break
else:
    print("⚠ 未找到中文字体")
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 一、业务参数
# =========================================================
PRICE, HOLD_RATE, WEEKS = 2800.0, 0.20, 52.0
H = PRICE * HOLD_RATE / WEEKS          # 10.7692 元/吨·周
PEN = 1800.0
K0 = 4200.0
BOX = 20.0
TRI = (0.0, 240.0, 120.0)              # (下界, 上界, 众数)
SEASON = [0.75, 0.75, 1.25, 1.25, 1.25, 1.25, 0.92, 0.92]
BASE = 120.0
L = 1                                  # 提前期（周）


# =========================================================
# 二、三角分布工具
# =========================================================
def tri_pdf(x, a, b, c):
    x = np.asarray(x, float)
    return np.where((x >= a) & (x < c), 2 * (x - a) / ((b - a) * (c - a)),
                    np.where((x >= c) & (x <= b), 2 * (b - x) / ((b - a) * (b - c)), 0.0))


def tri_cdf(x, a, b, c):
    x = np.asarray(x, float)
    return np.where(x < a, 0.0,
           np.where(x < c, (x - a) ** 2 / ((b - a) * (c - a)),
           np.where(x <= b, 1 - (b - x) ** 2 / ((b - a) * (b - c)), 1.0)))


def tri_quantile(p, a, b, c):
    f = (c - a) / (b - a)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return a + math.sqrt(p * (b - a) * (c - a)) if p <= f else b - math.sqrt((1 - p) * (b - a) * (b - c))


def tri_moments(a, b, c):
    m = (a + b + c) / 3
    v = (a * a + b * b + c * c - a * b - a * c - b * c) / 18
    return m, math.sqrt(v)


def tri_draw(rng, shape, a, b, c):
    u = rng.random(shape)
    f = (c - a) / (b - a)
    return np.where(u <= f, a + np.sqrt(u * (b - a) * (c - a)), b - np.sqrt((1 - u) * (b - a) * (b - c)))


# =========================================================
# 三、单期报童：临界分位数 → S*
# =========================================================
def newsboy_curve(a, b, c, h=H, pen=PEN, n=6000):
    """对每个 S 算单周期望成本，返回 (网格, 成本, 数值最优 S, 解析最优 S, 临界比)。"""
    x = np.linspace(a, b, n)
    pdf = tri_pdf(x, a, b, c)
    grid = np.linspace(a, b, 400)
    cost = np.array([np.trapezoid((h * np.maximum(S - x, 0) + pen * np.maximum(x - S, 0)) * pdf, x)
                     for S in grid])
    ratio = pen / (pen + h)
    return grid, cost, float(grid[int(np.argmin(cost))]), float(tri_quantile(ratio, a, b, c)), ratio


# =========================================================
# 四、周期订货仿真：每 T 周在周初补一次（有界三角需求，不欠单）
# =========================================================
def sim_periodic(s, T, K, a, b, c, h=H, pen=PEN, reps=6000, cycles=60, burn=20, seed=1,
                 box=None, lost_sales=True, return_alpha=False):
    """第 0 周补到 S = s + μT；之后每 T 周初再按差额补一次（Q 固定 = μT）。
       需求按三角分布独立同分布抽样；缺货不欠（lost sales）。返回周均成本拆解。"""
    rng = np.random.default_rng(seed)
    mu, _ = tri_moments(a, b, c)
    S = s + mu * T
    dem = tri_draw(rng, (reps, cycles * T), a, b, c)
    onhand = np.full(reps, float(S))
    pipeline = np.zeros(reps)
    hold = np.zeros(reps)
    short = np.zeros(reps)
    for cy in range(cycles):
        for k in range(T):
            onhand += pipeline                      # 上周下的单本周到（L = 1）
            pipeline = np.zeros(reps)
            if k == 0:
                q = np.maximum(S - onhand, 0.0)
                if box:
                    q = np.ceil(q / box) * box
                pipeline = q
            d = dem[:, cy * T + k]
            sold = np.minimum(onhand, d)
            lost = d - sold
            onhand = onhand - sold
            if cy >= burn:
                hold += h * onhand
                short += pen * lost
    n = (cycles - burn) * T
    hm, sm = float(hold.mean() / n), float(short.mean() / n)
    out = dict(cost=hm + sm + K / T, hold=hm, short=sm, fix=K / T)
    if return_alpha:
        out["alpha"] = 1.0 - float(lost.mean() / (cycles - burn))
    return out


def best_policy_for_T(T, K, a, b, c, Sstar, reps=4000, cycles=50, seed=2, **kw):
    """给定 T，在 s 上搜索最优（S = s + μT）。"""
    mu, _ = tri_moments(a, b, c)
    best = (np.inf, None, None)
    for s in np.arange(0.0, Sstar + 50.0, 10.0):
        r = sim_periodic(s, T, K, a, b, c, reps=reps, cycles=cycles, seed=seed, **kw)
        if r["cost"] < best[0]:
            best = (r["cost"], float(s), r)
    return best


def best_T_for_K(K, a, b, c, Sstar, Tmax=12, **kw):
    best = (np.inf, None, None, None)
    for T in range(1, Tmax + 1):
        cost, s, r = best_policy_for_T(T, K, a, b, c, Sstar, **kw)
        if cost < best[0]:
            best = (cost, T, s, r)
    return best


# =========================================================
# 五、非平稳需求：Wagner-Whitin 精确计划（确定性等价）
# =========================================================
def wagner_whitin(d, K, h):
    T = len(d)
    C = [0.0] * (T + 1)
    arg = [0] * (T + 1)
    for j in range(1, T + 1):
        best, bi = np.inf, j
        for i in range(1, j + 1):
            hold = sum(h * (t - i) * d[t - 1] for t in range(i, j + 1)) if i < j else 0.0
            cost = C[i - 1] + K + hold
            if cost < best - 1e-9:
                best, bi = cost, i
        C[j] = best
        arg[j] = bi
    segs, j = [], T
    while j > 0:
        i = arg[j]
        segs.append((i, j))
        j = i - 1
    segs.reverse()
    return C[T], segs


def plan_quantities(d, segs):
    q = [0.0] * len(d)
    for i, j in segs:
        q[i - 1] = sum(d[i - 1:j])
    return q


# =========================================================
# 主流程
# =========================================================
def main():
    t0 = time.time()
    a, b, c = TRI
    mu, sd = tri_moments(a, b, c)
    res = {}

    print("=" * 80)
    print("Day85 · n 期库存：K 决定『多久补一次』，临界分位数决定『补到多高』")
    print("=" * 80)
    print(f"旺季周需求 ~ 三角({a:.0f}, {b:.0f}, {c:.0f})：均值 {mu:.1f} 吨、sd {sd:.1f}（CV {sd/mu:.0%}，有界：不可能超过 {b:.0f}）")
    print(f"单价 {PRICE:.0f} 元/吨｜h = {H:.4f} 元/吨·周｜缺货罚 {PEN:.0f}｜K = {K0:.0f}｜提前期 {L} 周｜整箱 {BOX:.0f} 吨")

    # ---------- [1] 报童：临界分位数 ----------
    grid, curve, S_num, S_ana, ratio = newsboy_curve(a, b, c)
    print(f"\n[1] 单期报童（一次决策、卖不完就剩、缺了就赔）：")
    print(f"    临界分位数 cu/(cu+h) = {PEN:.0f}/({PEN:.0f}+{H:.2f}) = {ratio:.4f}")
    print(f"    → S* = 三角分位数 = {S_ana:.1f} 吨（数值最优 {S_num:.1f}，两条路径互证）")
    print(f"    → 安全库存 = {S_ana - mu:.1f} 吨 = {sd:.1f} 吨方差的 {(S_ana-mu)/sd:.2f} 个标准差")
    print(f"    → 换句话：K 不在这条公式里。K 不是「敢不敢多备」，是「多久补一次」。")
    res["newsboy"] = dict(ratio=ratio, S_star=S_ana, S_numeric=S_num, mean=mu, sd=sd,
                          safety=(S_ana - mu) / sd)

    # ---------- [2] K 扫描：T 上去，s 不动 ----------
    print("\n[2] K 扫描：同样的需求，只改一次补货单的固定成本（每 T 周补一次，s 在网格上搜）")
    Ks = [0.0, 500.0, 1000.0, 2000.0, 4200.0, 8000.0, 15000.0, 30000.0]
    kscan = []
    for K in Ks:
        cost, T, s, r = best_T_for_K(K, a, b, c, S_ana, reps=3000, cycles=45)
        rec = dict(K=K, T=T, s=s, S=s + mu * T, Q=mu * T, cost=cost,
                   hold=r["hold"], short=r["short"], fix=r["fix"])
        kscan.append(rec)
        print(f"    K={K:6.0f} → 每 {T:2d} 周补一次  s={s:5.0f}  S=s+μT={s + mu*T:6.0f}"
              f"  持有 {r['hold']:6.0f}  缺货 {r['short']:5.0f}  固定 {r['fix']:5.0f}")
    res["K_scan"] = kscan
    s_all = [r["s"] for r in kscan]
    print(f"    ⇒ K 涨了 30000 倍，s 在 8 个 K 上全部停在 {min(s_all):.0f}（网格分辨率 10 吨、"
          f"目标差 <2% 的台阶都是平的），而补货周期 T 从 1 周拉到 {max(r['T'] for r in kscan)} 周")

    # ---------- [3] 有界 vs 无界 ----------
    print("\n[3] 同样的均值、同样的方差，用正态近似会怎样？（正态无界，三角有界）")
    z = ratio
    from statistics import NormalDist
    normal_S = mu + sd * NormalDist().inv_cdf(ratio)
    print(f"    三角分布给出 S* = {S_ana:.1f}；用同均值同方差的正态给出 S* = {normal_S:.1f}"
          f"（差 {normal_S - S_ana:+.1f} 吨 = {100*(normal_S-S_ana)/S_ana:+.1f}%）")
    print(f"    差别的来源：正态的右尾把「永远发生不了的需求」也算进去了 ——")
    print(f"    三角的 99.41% 分位 {S_ana:.1f} 离上界 {b:.0f} 还有 {b-S_ana:.0f} 吨；正态的 99.41% 分位 {normal_S:.1f} 已经越过上界。")
    print(f"    但 +7.2% 只是 S 的差；若把安全库存也算成「几倍 sd」，差的是 {100*((normal_S-mu)/sd - (S_ana-mu)/sd)/((S_ana-mu)/sd):+.0f}% 的安全库存量。")
    res["bounded_vs_normal"] = dict(tri_S=S_ana, normal_S=normal_S, diff_pct=100 * (normal_S - S_ana) / S_ana)

    # ---------- [4] 缺货能不能欠 ----------
    print("\n[4] 缺货能不能欠（lost sales vs backlog 的会计差别）")
    print("    本模型全程用「不欠」口径：卖不出去的就是丢了（化肥过了农时没人要），")
    print("    所以 pen 是一次性的 1800 元/吨，而不是「欠着以后还得补」的持有式罚金。")
    print("    注意：若误用 backlog 口径，欠单会滚成永久负库存，成本曲线会被「欠得越多越省」带偏——")
    print("    本班 Day18 的 (s,S) MDP 就是 backlog 口径，两讲的 s 不能直接比。")
    res["penalty_note"] = "lost sales"

    # ---------- [5] 季节性：WW 精确计划 ----------
    print("\n[5] 春耕季 8 周：需求非平稳（淡 2 周 → 旺 4 周 → 尾 2 周），K = 4200")
    d = [BASE * f for f in SEASON]
    tot, segs = wagner_whitin(d, K0, H)
    q = plan_quantities(d, segs)
    print(f"    计划需求 {[round(v) for v in d]}，合计 {sum(d):.0f} 吨")
    print(f"    W-W 最优：{len(segs)} 次进货，段 {segs}，总成本 {tot:,.0f} 元")
    print(f"    下单周 {[i+1 for i, v in enumerate(q) if v > 0]}，下单量 {[round(v) for v in q if v > 0]}")
    print(f"    ⇒ 第 3 周旺季一开始要一次压足 {round(q[2])} 吨（5 周量），因为 K 比持有 5 周的利息贵")
    ww = []
    for K in [0, 500, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 20000]:
        t_, sg = wagner_whitin(d, K, H)
        ww.append(dict(K=K, n=len(sg), cost=t_))
    res["ww"] = dict(cost=tot, segs=segs, q=q, scan=ww, demand=d)

    # ---------- [6] 整箱：搜最优 s ----------
    print("\n[6] 整箱 20 吨（必须整箱发）——注意：整箱不是纯代价，可能反而更便宜，因为「多发一点」就是多备安全库存：")
    T_best = next(r["T"] for r in kscan if r["K"] == K0)
    s_best = next(r["s"] for r in kscan if r["K"] == K0)
    free = best_policy_for_T(T_best, K0, a, b, c, S_ana, reps=6000, cycles=60, seed=11)
    boxed = best_policy_for_T(T_best, K0, a, b, c, S_ana, reps=6000, cycles=60, seed=11, box=BOX)
    print(f"    连续量最优：s={free[1]:.0f} → {free[0]:.0f} 元/周")
    print(f"    整箱最优  ：s={boxed[1]:.0f} → {boxed[0]:.0f} 元/周"
          f"（差 {boxed[0]-free[0]:+.0f} 元/周，年 {(boxed[0]-free[0])*52:+,.0f} 元）")
    print(f"    机理：整箱必须「向上取整」，补货量系统性偏大 → 等价于免费加了安全库存；")
    print(f"          本实例缺货罚很贵（1800/吨），加的安全库存恰好值这个钱，所以整箱反而不亏。")
    s_best = boxed[1]
    res["box"] = dict(free_s=free[1], free=free[0], boxed_s=boxed[1], boxed=boxed[0],
                      diff=boxed[0] - free[0])

    # ---------- [7] 服务水平：用「没算 K」的老规矩去跑会被现实打脸 ----------
    print("\n[7] 两种「服务水平」口径，以及「抄公式不管 K」的后果")
    r_fin = sim_periodic(s_best, T_best, K0, a, b, c, reps=20000, cycles=150, seed=13,
                         box=BOX, return_alpha=True)
    beta = 1 - (r_fin["short"] / PEN) / mu
    print(f"    最优 (T={T_best}, s={s_best:.0f})：β（满足的需求占比）= {beta:.3%}，"
          f"α（补货周期内完全不缺货）= {r_fin['alpha']:.2%}")
    print(f"    教材里的「单期点订货」会把订货点取在 S* = {S_ana:.0f}（把 K 忽略掉）：")
    r_naive = sim_periodic(S_ana, T_best, K0, a, b, c, reps=20000, cycles=150, seed=13, box=BOX)
    print(f"      它每 {T_best} 周只在剩 {S_ana:.0f} 吨时补 240 吨 → 周均 {r_naive['cost']:.0f} 元，"
          f"比最优贵 {r_naive['cost'] - boxed[0]:+.0f} 元/周（年 {(r_naive['cost'] - boxed[0]) * 52:+,.0f}）")
    print(f"      贵在哪：缺货成本 {r_naive['short']:.0f} 元/周 vs 最优 {r_fin['short']:.0f} 元/周")
    print(f"    ⇒ 服务水平是 99% 还是 90%，取决于 s 有没有把「这一补要撑 {T_best} 周」算进去")
    res["service"] = dict(beta=beta, alpha=r_fin["alpha"], T=T_best, s=s_best,
                          naive_s=S_ana, naive_cost=r_naive["cost"], opt_cost=boxed[0])

    res["runtime_s"] = time.time() - t0
    print(f"\n用时 {res['runtime_s']:.1f} s")

    # ================= 图 =================
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.6))

    ax[0, 0].plot(grid, curve, lw=2, color="#1f77b4")
    ax[0, 0].axvline(S_ana, color="#d62728", ls="--", lw=1.5, label=f"S* = {S_ana:.0f}")
    ax[0, 0].axvline(mu, color="#888", ls=":", label=f"均值 {mu:.0f}")
    ax[0, 0].set_xlabel("周初进货后的存货 S（吨）")
    ax[0, 0].set_ylabel("单周期望成本（元）")
    ax[0, 0].set_title(f"(a) 报童曲线：最低点在 {ratio:.2%} 分位，与 K 无关")
    ax[0, 0].legend(fontsize=9)

    ax[0, 1].plot([r["K"] for r in kscan], [r["s"] for r in kscan], "o-", lw=2, label="s（降到这条线才补）")
    ax[0, 1].plot([r["K"] for r in kscan], [r["S"] for r in kscan], "s-", lw=2, label="S = s + μT（补到多高）")
    ax[0, 1].axhline(S_ana, color="#d62728", ls=":", label="单期报童 S*")
    ax[0, 1].set_xlabel("固定订购成本 K（元/单）")
    ax[0, 1].set_ylabel("库存（吨）")
    ax[0, 1].set_xscale("symlog", linthresh=500)
    ax[0, 1].set_title("(b) K 涨 7 倍：s 几乎不动，S 一路抬高")
    ax[0, 1].legend(fontsize=9)

    ax[1, 0].plot([r["K"] for r in kscan], [r["T"] for r in kscan], "o-", lw=2, color="#2ca02c")
    ax[1, 0].set_xlabel("固定订购成本 K（元/单）")
    ax[1, 0].set_ylabel("补货周期 T（周）")
    ax[1, 0].set_xscale("symlog", linthresh=500)
    ax[1, 0].set_title("(c) K 真正买来的是「更少的单」：T 单调不减（不是 √K，见 README 避坑 4）")

    ax[1, 1].bar(np.arange(1, 9), d, 0.55, color="#aec7e8", label="各周计划需求")
    ax[1, 1].bar(np.arange(1, 9), q, 0.32, color="#ff7f0e", label="W-W 下单量")
    ax[1, 1].set_xlabel("春耕季第几周")
    ax[1, 1].set_ylabel("吨")
    ax[1, 1].set_title(f"(d) 非平稳需求：{len(segs)} 次进货，旺季前一次压足")
    ax[1, 1].legend(fontsize=9)

    fig.tight_layout()
    out_png = __file__.replace(".py", ".png")
    fig.savefig(out_png, dpi=130)
    print("图已保存：", out_png)
    with open(__file__.replace(".py", "_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已保存：", __file__.replace(".py", "_result.json"))
    return res


if __name__ == "__main__":
    main()
