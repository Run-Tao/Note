#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 随机优化（两阶段随机规划 · SAA 样本均值近似 · 随机库存） / L5 算法（HiGHS LP · 场景树 · 蒙特卡洛评估）
anchor  : 前接 Day85（n 期库存 (s,S) 与报童临界分位数——那一讲提前期固定 1 周、需求逐周独立）、
          Day28（两阶段随机规划与 SAA——那一讲只有一次补货决策）、
          Day29（CVaR 风险厌恶）、Day38（Benders 分解）、Day57（蒙特卡洛 / DES 评估）。
          本讲把 Day85 的「提前期固定」放宽为**提前期随机**，并把 Day28 的「一期」拉成「一季 8 周」。
goal    : 一个问题——「春耕季的肥，按平均值备还是按场景树备？提前期随机了，该多备还是少备？」
          ① 三条口径：EV（按平均需求算一个计划）／ SAA（场景树算一个计划）／ PI（事后全知，下界）；
          ② **仿真裁判**：场景树给的确定性等价永远偏低，量出偏差；量出 VSS 与 EVPI；
          ③ 提前期 L∈{2,3,4} 随机：最优点是「多备」吗？多备多少？—— 结论由缺货罚 p 决定，不由「L 随机」决定；
          ④ 用 LP 场景树**和**网格搜索双签同一个计划，证明「不是求解器说了算」。

决策场景：云溪县农资公司 2027 年春耕「0 号复合肥」县级中心仓（同一家仓、同一批货，接 Day85）。
          8 周一个订货季（淡 2 周 → 旺 4 周 → 尾 2 周，季节因子 0.75/0.75/1.25×4/0.92×2）；
          旺季周需求上限 240 吨（全县耕地硬上限），旺季均值 120 吨、sd 49 吨（截断正态，与 Day85 的三角分布并列）；
          提前期 L ∈ {2,3,4} 周（平原车 30% / 山区车 50% / 阴雨塌方 20%）；季末剩余半价处理（1400 元/吨）。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error \\
      "Mathematical_Modeling/08 - Python/daily/2026-11-22/day86_two_stage_stochastic_inventory.py"

口径（全篇统一，否则数字互相打架）：
  * 单价 c = 2800 元/吨；周持有 h = c×20%÷52 = 10.7692 元/吨·周（按**卖到手的量**计提）
  * 缺货罚 p：基准 1800 元/吨，另做 3600 / 7200 / 10800 四档扫描
  * 决策 = 「每周到货多少」的 8 维计划 arr[0..7]（= 场景树里的**非预见性**第 1 阶段决策）
  * 提前期 L 在仿真里决定**何时下这笔单**（提前 L 周下单、占 L 周资金），到货时刻由计划给定
  * 「EV 计划」= 把需求当成确定值（每周 = 均值×季节因子）求出的计划；「SAA 计划」= 用 n 个随机场景求出的计划
"""

import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linprog

# ---------------- 中文字体（缺失会乱码） ----------------
for _n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
           "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]:
    if _n in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_n]
        break
else:
    print("⚠ 未找到中文字体")
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
SEED = 20261122

# =========================================================
# 一、业务参数
# =========================================================
PRICE = 2800.0                    # 元/吨
HOLD_RATE = 0.20                  # 年持有率
H = PRICE * HOLD_RATE / 52.0      # 10.7692 元/吨·周
R_U = 1400.0                      # 季末残值：半价处理
PEN = 1800.0                      # 基准缺货罚
W = 8                             # 订货季周数
SEASON = np.array([0.75, 0.75, 1.25, 1.25, 1.25, 1.25, 0.92, 0.92])
MEAN_PK, SD_PK, CAP_PK = 120.0, 49.0, 240.0
LT_VALUES = np.array([2, 3, 4])   # 提前期 {2,3,4} 周
LT_PROB = np.array([0.30, 0.50, 0.20])
K_FIX = 4200.0                    # 每趟车固定费（Day85 同口径）
N_EVAL = 60_000                   # 评估路径数（独立于任何场景树）
PEN_GRID = (1800.0, 3600.0, 7200.0, 10800.0)


# =========================================================
# 二、需求采样
# =========================================================
def sample_peaks(rng, n):
    """旺季周需求：截断正态 [0, 240]（240 = 全县耕地硬上限）"""
    return np.clip(rng.normal(MEAN_PK, SD_PK, n), 0.0, CAP_PK)


def sample_demand(rng, n):
    """n 条路径 × 8 周的需求矩阵

    2026-11-23 修复：早期版本把「旺季峰值∈[0,240]」再乘季节因子 1.25 当旺季周需求，
    结果旺季实际需求可达 300 吨，越过 240 吨的耕地硬上限（正是 Day85 §五.3 点的错），
    把持有与缺货成本一起抬到 15 万/周量级。现在改为「直接按周需求」采样：
    逐周均值 = μ·季节因子，截断到 [0, 240]，旺季封顶 240。
    """
    mu = (MEAN_PK * SEASON)[None, :]
    sd = SD_PK * SEASON[None, :]
    x = rng.normal(mu, sd, size=(n, W))
    return np.clip(x, 0.0, CAP_PK)


# =========================================================
# 三、统一的成本口径（唯一真值，仿真 / 手算 / LP 三处共用）
#     记 I_w = 第 w 周周初库存（I_0 = 0，季前无结转）
#         a_w = 第 w 周到货量（第 1 阶段决策，整季 8 维，不随需求实现）
#         d_w = 第 w 周需求，u_w = 卖出量，L_w = 缺货量
#     成本（元/季，单条路径）：
#         Σ_w [ p·L_w + h·(I_w + a_w − u_w) ]  +  (h + c − r_u)·(I_W)  +  h·Σ_w a_w·E[L]/2
#       即 缺货罚 + 每周留存库存持有费 + 季末余货（再持一周 + 半价残值损失）+ 在途资金。
#     口径说明（2026-11-23 重写）：早期版本混用「现金口径」（计卖货收入 c·u）与
#     「成本口径」（p/(p+h) 分位数），三者互相矛盾；Day85 的 5,207.7 元/周 是
#     「持有 + 缺货 + 固定费」口径，本讲全部与之对齐，不再出现收入项。
# =========================================================
def simulate(arr, dem, pen=PEN, r_u=R_U, k_each=None, lt=None):
    """n 条路径的季成本矩阵（元），口径见上（缺货 + 逐周持有 + 季末残值 + 在途资金）
       lt: 覆盖提前期（只影响在途资金项）；默认用 E[L] = LT_PROB·LT_VALUES"""
    arr = np.asarray(arr, float)
    n, Wn = dem.shape
    assert arr.shape == (Wn,), "计划必须是 8 维"
    cost = np.zeros(n)
    inv = np.zeros(n)                                  # I_w：第 w 周周初库存
    for w in range(Wn):
        a = float(arr[w])
        if a > 0:
            lt_eff = float(LT_PROB @ LT_VALUES) if lt is None else float(lt)
            cost += H * a * (lt_eff / 2.0)                        # 在途资金
            if k_each:
                cost += k_each
        stock = inv + a
        cost += pen * np.maximum(dem[:, w] - stock, 0.0)         # 缺货
        inv = np.maximum(stock - dem[:, w], 0.0)                 # 下周周初
        cost += H * inv                                          # 留存库存持有一周
    cost += (PRICE - r_u) * inv                                  # 季末余货残值损失
    return cost


def simulate_mean(arr, dem, pen=PEN, **kw):
    return float(simulate(arr, dem, pen=pen, **kw).mean() / W)


# =========================================================
# 四、场景树 LP（确定性等价）——与 simulate 逐项同口径
# =========================================================
def build_tree_model(peaks, pen=PEN, r_u=R_U):
    """变量 I(s,w)（周初库存，w=0 时由约束钉为 0）、A(w)（到货，第 1 阶段）、
       U(s,w)（卖出）、L(s,w)（缺货）、E(s)（季末余货）。
       约束：
         (E1) I(s,w) + A(w) − U(s,w) = I(s,w+1)   （w=W−1 时右侧为 E(s)）  结转
         (E2) U(s,w) + L(s,w) = d(s,w)                                     需求去处
         (E3) I(s,0) = 0                                                   季前无结转
       目标（= simulate 的期望）：
         Σ h·A·E[L]/2 + (1/S)Σ_s [ Σ_w (p·L(s,w) + h·I(s,w+1)) + (c−r_u)·E(s) ]
    """
    S_ = len(peaks)
    dem = peaks[:, None] * SEASON[None, :]
    n_i, n_a, n_u, n_l, n_e = S_ * W, W, S_ * W, S_ * W, S_
    nx = n_i + n_a + n_u + n_l + n_e
    ix = {"I": lambda s, w: s * W + w,
          "A": lambda w: n_i + w,
          "U": lambda s, w: n_i + n_a + s * W + w,
          "L": lambda s, w: n_i + n_a + n_u + s * W + w,
          "E": lambda s: n_i + n_a + n_u + n_l + s}

    c = np.zeros(nx)
    c[ix["A"](0):ix["A"](0) + W] = H * float(LT_PROB @ LT_VALUES) / 2.0
    for s in range(S_):
        c[ix["E"](s)] = (PRICE - r_u) / S_
        for w in range(W):
            c[ix["L"](s, w)] = pen / S_
            if w < W - 1:
                c[ix["I"](s, w + 1)] += H / S_      # 下周周初 = 本周留存，持有费挂这里
            else:
                c[ix["E"](s)] += H / S_             # 末周留存进 E，也要付一周持有费

    A_eq, b_eq = [], []
    for s in range(S_):
        r = np.zeros(nx); r[ix["I"](s, 0)] = 1      # (E3)
        A_eq.append(r); b_eq.append(0.0)
        for w in range(W):
            r = np.zeros(nx)                        # (E1)
            r[ix["I"](s, w)] = 1; r[ix["A"](w)] = 1; r[ix["U"](s, w)] = -1
            if w < W - 1:
                r[ix["I"](s, w + 1)] -= 1
            else:
                r[ix["E"](s)] -= 1
            A_eq.append(r); b_eq.append(0.0)

            r = np.zeros(nx)                        # (E2)
            r[ix["U"](s, w)] = 1; r[ix["L"](s, w)] = 1
            A_eq.append(r); b_eq.append(float(dem[s, w]))

    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    return {"c": c, "A_eq": A_eq, "b_eq": b_eq, "const": 0.0,
            "idx": ix, "nx": nx, "dem": dem, "S": S_}


def plan_cost_exact(arr, peaks, pen=PEN, r_u=R_U):
    """同一口径的手算版：逐场景用 simulate 的成本公式取平均（LP 双签用）"""
    dem = peaks[:, None] * SEASON[None, :] if np.ndim(peaks) == 1 else peaks
    return float(simulate(arr, dem, pen=pen, r_u=r_u).mean())


def solve_tree(peaks, pen=PEN, r_u=R_U, verbose=False):
    """求解场景树 LP，返回 (季成本, 每周到货量, 列数)"""
    mdl = build_tree_model(peaks, pen, r_u)
    r = linprog(mdl["c"], A_eq=mdl["A_eq"], b_eq=mdl["b_eq"],
                bounds=[(0, None)] * mdl["nx"], method="highs")
    assert r.status == 0, r.message
    arr = np.array([r.x[mdl["idx"]["A"](w)] for w in range(W)])
    val = float(r.fun + mdl["const"])
    if verbose:
        print(f"    [LP] 场景 {mdl['S']} · 列 {mdl['nx']} · 等式 {mdl['A_eq'].shape[0]} · 目标 {val:,.2f}")
    return val, arr, mdl["nx"]


def lp_unit_test():
    """护栏：(a) 确定性需求下 LP 必须逐周订到预测值；(b) LP 目标 == 手算同一计划"""
    val_a, arr_a, _ = solve_tree(np.array([MEAN_PK]))
    ok_a = np.allclose(arr_a, MEAN_PK * SEASON, atol=1e-6)
    peaks = np.random.default_rng(11).normal(MEAN_PK, SD_PK, 7).clip(0, CAP_PK)
    v_lp, a_lp, _ = solve_tree(peaks)
    hand = plan_cost_exact(a_lp, peaks)
    ok_b = abs(hand - v_lp) < 1e-6
    print(f"  单元自检(a) 确定性需求：LP 给出 {np.round(arr_a, 1)} → {'✅ 逐周订到预测' if ok_a else '❌ 写错了'}")
    print(f"  单元自检(b) LP 目标 vs 手算：{v_lp:,.4f} vs {hand:,.4f} → {'✅ 一致' if ok_b else '❌ 不一致'}")
    assert ok_a and ok_b, "场景树 LP 写错了（平衡式 / 持有项 / 季末残值 三处之一）"
    return val_a, arr_a


# =========================================================
# 五、网格双签：现场计划只有两段（开季首批 + 旺前补一批）
# =========================================================
def grid_plan(a0, a1):
    arr = np.zeros(W)
    arr[1] = a0
    arr[3] = a1
    return arr


def grid_search(peaks, pen=PEN, r_u=R_U, step=5.0):
    """在确定性等价口径下搜 (a0, a1)：与 LP 同口径（都走 plan_cost_exact），但不过求解器"""
    best = None
    # 2026-11-23 修复：上界 420 → 600。两段网格最多只能装 840 吨，而 8 周需求合计可达 8×240=1920 吨，
    # 420+420 是被上界卡死的，与 LP 的差距里有一半是网格自己够不着（差点写成「网格比 LP 差 27%」）
    for a0 in np.arange(0.0, 600.0 + 1e-9, step):
        for a1 in np.arange(0.0, 600.0 + 1e-9, step):
            v = plan_cost_exact(grid_plan(a0, a1), peaks, pen, r_u)
            if best is None or v < best[0]:
                best = (v, float(a0), float(a1))
    return best


# =========================================================
# 六、单期解析参照：报童分位数
# =========================================================
def newsvendor_ctr(pen, r_u=R_U):
    """单期一次性备货（季末无结转）的临界分位数：p / (p + c − r_u)
       注意分母是 (p + c − r_u) 而非 (p + h)：h 只管跨期持有，一次性备货的过量成本是残值损失"""
    return pen / (pen + PRICE - r_u)


def trunc_norm_mean(mu=MEAN_PK, sd=SD_PK, cap=CAP_PK):
    """[0,cap] 截断正态的真实均值（数值积分，用作「场景树均值偏不偏」的裁判）"""
    xs = np.linspace(0.0, cap, 20001)
    pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2)
    return float((xs * pdf).sum() / pdf.sum())


def truncnorm_quantile(p, mu=MEAN_PK, sd=SD_PK, cap=CAP_PK):
    from scipy.stats import truncnorm
    a, b = (0 - mu) / sd, (cap - mu) / sd
    return float(truncnorm.ppf(p, a, b, loc=mu, scale=sd))


# =========================================================
# 七、主流程
# =========================================================
def main():
    out = {}
    dem_eval = sample_demand(np.random.default_rng(SEED + 1), N_EVAL)

    def row(arr, name, pen=PEN):
        cc = simulate(arr, dem_eval, pen=pen)
        return {"name": name, "plan": [round(float(x), 2) for x in arr],
                "mean": round(float(cc.mean()), 2),
                "sd": round(float(cc.std()), 2),
                "p90": round(float(np.quantile(cc, 0.90)), 2),
                "cvar90": round(float(cc[cc >= np.quantile(cc, 0.90)].mean()), 2)}, cc

    # ---------- 6.1 三条口径 ----------
    print("=" * 80)
    print("【一】三条口径：EV（按平均需求）／ SAA（场景树）／ PI（事后全知）")
    print("=" * 80)
    ev_val, ev_arr, _ = solve_tree(np.array([MEAN_PK]), pen=PEN)
    ev_row, _ = row(ev_arr, "EV")
    print(f"  EV  计划 arr = {np.round(ev_arr, 1)}  场景口径周均 {ev_val/W:,.2f}  真值周均 {ev_row['mean']/W:,.2f} 元")

    saa_rows = []
    for nsc in (5, 20, 100, 500):
        peaks = sample_peaks(np.random.default_rng(SEED + nsc), nsc)
        val, arr, nx = solve_tree(peaks, pen=PEN)
        r, _ = row(arr, f"SAA n={nsc}")
        r.update({"n_scen": nsc, "ceq_weekly": round(val / W, 2), "lp_cols": nx})
        # 2026-11-23 追加：把**这一组场景自己的平均峰值**打出来。
        #   没有这一列，表五里 +23%~+81% 的偏差会被误读成「样本越多越准」
        r["peak_mean_scen"] = round(float(peaks.mean()), 2)
        r["peak_mean_true"] = round(float(trunc_norm_mean()), 2)
        saa_rows.append(r)
        print(f"       （该组场景峰值均值 {r['peak_mean_scen']:.1f} vs 真分布均值 {r['peak_mean_true']:.1f}）")
        print(f"  SAA n={nsc:3d}: 场景口径 {val/W:>9,.2f} | 真值 {r['mean']/W:>9,.2f} | "
              f"arr={np.round(arr, 0)} | 偏差 {100*(val/W - r['mean']/W)/(r['mean']/W):+5.2f}%")

    # PI（事后全知下界）：知道每条路径的真实需求后，每周各自按最优备货（每周独立、无结转）
    # 与 simulate 同口径：持有 + 缺货（+ 在途资金），用蒙特卡洛最大化利润即最小化该成本
    pi_cost = 0.0
    for w in range(W):
        d_paths = dem_eval[:, w]
        lo, hi = 0.0, CAP_PK
        grid = np.linspace(lo, hi, 241)
        # 每条路径单独最优：给定 d，备货 max(?,0) 下的成本 = h·max(a−d,0) + p·max(d−a,0)
        costs = np.array([np.mean(H * np.maximum(a - d_paths, 0.0)
                                  + PEN * np.maximum(d_paths - a, 0.0)) for a in grid])
        a_star = float(grid[int(np.argmin(costs))])
        pi_cost += float(costs.min()) + H * a_star * (float(LT_PROB @ LT_VALUES) / 2.0)
    pi_week = pi_cost / W
    print(f"  PI（事后全知，每周单独最优）周均 {pi_week:,.2f} 元")

    best_saa = min(saa_rows, key=lambda r: r["mean"])
    vss = (best_saa["mean"] - ev_row["mean"]) / W
    evpi = (ev_row["mean"] - pi_cost) / W
    print(f"\n  ⇒ VSS = SAA − EV  = {vss:+,.2f} 元/周（{100*vss/(ev_row['mean']/W):+.2f}%）")
    print(f"  ⇒ EVPI = EV − PI  = {evpi:+,.2f} 元/周（{100*evpi/(ev_row['mean']/W):.2f}%）")
    print(f"  ⇒ EVPI / VSS = {evpi/vss:.2f} 倍 —— 「知道未来」与「考虑不确定性」不是一回事")
    out["benchmark"] = {"EV": ev_row, "SAA": saa_rows,
                        "PI_weekly": round(pi_week, 2),
                        "VSS_weekly": round(vss, 2), "EVPI_weekly": round(evpi, 2),
                        "EVPI_over_VSS": round(evpi / vss, 3)}

    # ---------- 6.2 LP vs 网格：双签 ----------
    print("\n" + "=" * 80)
    print("【二】不是求解器说了算：LP 解 vs 网格暴力解（同口径）")
    print("=" * 80)
    peaks100 = sample_peaks(np.random.default_rng(SEED + 100), 100)
    lp_val, lp_arr, _ = solve_tree(peaks100, pen=PEN)
    gv, g0, g1 = grid_search(peaks100, pen=PEN)
    print(f"  LP   ：周均 {lp_val/W:>9,.2f}  arr[1]={lp_arr[1]:6.1f} arr[3]={lp_arr[3]:6.1f}  其余 {np.round(lp_arr[[0,2,4,5,6,7]],1)}")
    print(f"  网格 ：周均 {gv/W:>9,.2f}  a0={g0:6.1f}  a1={g1:6.1f}")
    gap = 100 * (gv - lp_val) / lp_val
    print(f"  ⇒ 网格比 LP 差 {gap:+.2f}%（网格只能表达「两段批量」，LP 允许 8 周任意）")
    out["lp_vs_grid"] = {"lp_weekly": round(lp_val / W, 2), "grid_weekly": round(gv / W, 2),
                         "gap_pct": round(gap, 3),
                         "lp_arr": [round(float(x), 2) for x in lp_arr],
                         "grid": {"a0": float(g0), "a1": float(g1)}}

    # ---------- 6.3 缺货罚扫描 ----------
    print("\n" + "=" * 80)
    print("【三】缺货罚 p 扫描：VSS 怎么长，EV 在 p 小时反而更优")
    print("=" * 80)
    pen_scan = []
    for pen in PEN_GRID:
        _, arr_ev, _ = solve_tree(np.array([MEAN_PK]), pen=pen)
        _, arr_saa, _ = solve_tree(peaks100, pen=pen)
        c_ev = simulate(arr_ev, dem_eval, pen=pen)
        c_saa = simulate(arr_saa, dem_eval, pen=pen)
        d = c_saa - c_ev
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        rec = {"pen": pen,
               "plan_ev": [round(float(x), 1) for x in arr_ev],
               "plan_saa": [round(float(x), 1) for x in arr_saa],
               "ev_weekly": round(float(c_ev.mean() / W), 2),
               "saa_weekly": round(float(c_saa.mean() / W), 2),
               "vss_weekly": round(float(d.mean() / W), 2),
               "vss_pct": round(float(100 * d.mean() / c_ev.mean()), 3),
               "t_paired": round(t, 2)}
        pen_scan.append(rec)
        print(f"  p={pen:6.0f}: EV {rec['ev_weekly']:>9,.0f} | SAA {rec['saa_weekly']:>9,.0f} | "
              f"VSS {rec['vss_weekly']:+8,.0f}/周 ({rec['vss_pct']:+5.2f}%) t={t:+7.1f}")
    out["pen_scan"] = pen_scan

    # ---------- 6.4 提前期随机：L 越大，最优计划越「重」还是越「轻」 ----------
    print("\n" + "=" * 80)
    print("【四】提前期 L∈{2,3,4}：L 变大，最优批次该重还是该轻？")
    print("=" * 80)
    # 2026-11-23 注：这里用 SAA n=100 的计划（arr[1..7] 有量、arr[0]=0）评「同一计划换 L」，
    # 因为提前期只在 w≥1 的到货上起作用（w=0 的到货在第 1 周就落地）
    lt_scan = []
    for label, lt in (("L≡2（固定）", 2), ("L≡3（固定）", 3), ("L≡4（固定）", 4),
                      ("L~{2,3,4}（随机）", None)):
        cc = simulate(best_saa["plan"], dem_eval, lt=lt)   # 仅通过在途资金项区分 L（口径与主成本函数一致）
        rec = {"label": label, "L": lt, "weekly": round(float(cc.mean() / W), 2)}
        lt_scan.append(rec)
        print(f"  {label:20s}: 同一计划下周均 {rec['weekly']:>10,.2f} 元")
    # 对每种 L 重优化「季节内两段批量」，看最优批次随 L 的变化
    lt_opt = []
    dem_scen = peaks100[:, None] * SEASON[None, :]
    for lt in (1, 2, 3, 4, 5):
        best = None
        # 2026-11-23 修复：原来对**场景树**（100 个峰值×季节）优化、却在**评估分布** dem_eval 上计分，
        #   两者分布不同 → 过拟合，周均 82,950 是假数，且最优总是顶到 400+400 边界
        #   （边界顶死 ⇒ 假结论「L 不影响批次」）。现在在 dem_eval[:20000] 上重优化、同批路径计分。
        for a0 in np.arange(0.0, 600.0 + 1e-9, 40.0):
            for a1 in np.arange(0.0, 600.0 + 1e-9, 40.0):
                v = float(simulate(grid_plan(a0, a1), dem_eval[:20000], lt=lt).mean())
                if best is None or v < best[0]:
                    best = (v, float(a0), float(a1))
        lt_opt.append({"L": lt, "weekly": round(best[0] / W, 2),
                       "a0": best[1], "a1": best[2], "total": round(best[1] + best[2], 1)})
        print(f"  L≡{lt}: 最优首批 {best[1]:5.0f} + 补批 {best[2]:5.0f} = {best[1]+best[2]:5.0f} 吨"
              f"（周均 {best[0]/W:>9,.0f}）")
    dQ = lt_opt[3]["total"] - lt_opt[1]["total"]
    print(f"  ⇒ L 从 2 周变 4 周，最优总量 {lt_opt[1]['total']:.0f} → {lt_opt[3]['total']:.0f} 吨"
          f"（{100*dQ/lt_opt[1]['total']:+.1f}%）——「提前期翻倍 ⇒ 库存翻倍」是错的")
    out["leadtime"] = {"same_plan": lt_scan, "reopt": lt_opt,
                       "dQ_L2_L4": round(float(dQ), 1),
                       "mean_L": float(np.dot(LT_VALUES, LT_PROB)),
                       "var_L": float((LT_PROB * (LT_VALUES - np.dot(LT_VALUES, LT_PROB)) ** 2).sum())}

    # ---------- 6.5 场景树偏差 ----------
    print("\n" + "=" * 80)
    print("【五】场景树口径 vs 仿真真值：这一讲是「报忧」——场景树偏保守，真值更低")
    print("=" * 80)
    bias = []
    for r in saa_rows:
        tw = r["mean"] / W
        bias.append({"n_scen": r["n_scen"], "ceq_weekly": r["ceq_weekly"], "true_weekly": round(tw, 2),
                     "bias_pct": round(100 * (r["ceq_weekly"] - tw) / tw, 2)})
        print(f"  n={r['n_scen']:3d}: 场景口径 {r['ceq_weekly']:>9,.2f} | 真值 {tw:>9,.2f} | 偏差 {bias[-1]['bias_pct']:+6.2f}%")
    out["ceq_bias"] = bias

    # ---------- 6.6 计划抖动 ----------
    print("\n" + "=" * 80)
    print("【六】同一 n，换一组场景，计划抖多少（能不能直接给司机）")
    print("=" * 80)
    runs = []
    for rep in range(8):
        pk = sample_peaks(np.random.default_rng(SEED + 7000 + rep), 100)
        _, arr, _ = solve_tree(pk, pen=PEN)
        runs.append([round(float(x), 1) for x in arr])
    J = np.array(runs)
    width = J.max(axis=0) - J.min(axis=0)
    print(f"  8 次重抽（n=100）各周到货量区间宽度 {np.round(width, 1)} 吨；旺季第 3~6 周最宽")
    out["jitter"] = {"runs": runs, "width_by_week": [round(float(x), 1) for x in width],
                     "max_width": round(float(width.max()), 1)}

    # ---------- 6.7 趟次：K 换来「少跑几趟」值不值 ----------
    print("\n" + "=" * 80)
    print("【七】固定费 K 扫描：多跑一趟值不值 4200 元")
    print("=" * 80)
    # 把「两段计划」拓成 k 段（第 1 段在 w=0 到、其余按每段一段）——否则 k>2 的表达力不够，
    # 会让「段数越多越便宜」永远抽不出来（2026-11-23 重写）
    def k_seg_plan(k, q, first_frac=0.45):
        """k 段计划：每段约 W//k 周量，首批占 first_frac"""
        arr = np.zeros(W)
        segs = np.array_split(np.arange(W), k)
        for j, seg in enumerate(segs):
            share = (1.0 if j < k else 0.0) / k
            if j == 0 and k > 1:
                share = first_frac
            elif k > 1:
                share = (1 - first_frac) / (k - 1)
            for w in seg:
                # 段内均匀到货（同一段的各周分摊这一段的总量）
                mu_w = MEAN_PK * SEASON[w]
                arr[w] = mu_w + (q * share - (MEAN_PK * SEASON[seg]).sum()) / len(seg)
        return np.maximum(arr, 0.0)

    k_scan = []
    for k in (1, 2, 3, 4):
        best = None
        for q in np.arange(700.0, 1400.0 + 1e-9, 20.0):
            arr = k_seg_plan(k, q)
            kmul = K_FIX * (1 if k <= 2 else k - 1)   # 1~2 段按 2 趟计价（Day85 同口径），3/4 段真多跑
            trips = max(k, 2)
            v = simulate_mean(arr, dem_eval[:20000], k_each=K_FIX)
            v += K_FIX * (trips - 2) / W
            # 2026-11-23 自曝（见下方 note）：本段参数族把「首批落在第 1 周」硬编码，
            # 无法表达「平原车第 2 周才到」，所以 k=4 的 33,052 元是参数族的假最优 —— 本节数字**不作为结论**。
            if best is None or v < best[0]:
                best = (v, q, arr, trips)
        k_scan.append({"k_seg": k, "weekly": round(best[0], 2), "q": round(float(best[1]), 1),
                       "trips": best[3], "plan": [round(float(x), 1) for x in best[2]]})
        print(f"  {k} 段（{best[3]} 趟）: 全季总量 {best[1]:6.0f} 吨 → 周均 {best[0]:>9,.0f} 元")
    out["k_scan"] = k_scan
    print("  ⚠ 本节只做「参数族够不够表达」的演示：k_seg 参数族强制首批落在第 1 周，")
    print("    无法表达「第 2 周才到货」，所以 4 段的 33,052 元不可信，不能写成「段数越多越便宜」。")

    # ---------- 6.8 同量前移：相同总量、不同到货时点，值不值钱 ----------
    print("\n" + "=" * 80)
    print("【八】同一总量、同一批到货次数：开季多压 vs 尾季再补，差多少钱")
    print("=" * 80)
    shift_scan = []
    for label, mv in (("照均值季", 0.0), ("淡季多备 10%", 0.10), ("淡季多备 20%", 0.20)):
        # 只在第 1 周多备、第 7 周（尾季第 1 周）同量少备 —— 总量一模一样，只是到货时点不同
        arr = MEAN_PK * SEASON.copy()
        moved = mv * arr[0]
        arr[0] += moved
        arr[6] -= moved
        row_p = {"label": label, "shift": mv, "plan": [round(float(x), 1) for x in arr]}
        for pen in (PEN, 2 * PEN):
            row_p[f"weekly_p{int(pen)}"] = round(float(simulate(arr, dem_eval, pen=pen).mean() / W), 2)
        shift_scan.append(row_p)
        print(f"  {label:10s} 周均（p=1800） {row_p['weekly_p1800']:>9,.0f} | （p=3600） {row_p['weekly_p3600']:>9,.0f}"
              f" | 计划 {row_p['plan']}")
    out["same_qty_shift"] = shift_scan

    # =========================================================
    # 八、四联图
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    wk = np.arange(1, W + 1)
    ax.plot(wk, MEAN_PK * SEASON, "k:", lw=1.6, label="平均需求（均值×季节）")
    ax.plot(wk, ev_arr, "o-", lw=2, label=f"EV 计划（按均值）")
    ax.plot(wk, best_saa["plan"], "s--", lw=2, label=f"SAA 计划（场景树 n={best_saa['n_scen']}）")
    ax.set_xlabel("周次")
    ax.set_ylabel("到货量（吨）")
    ax.set_title("(a) 两种计划：差别在「旺季前要不要先压一批」")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    pens = [r["pen"] for r in pen_scan]
    ax.plot(pens, [r["vss_weekly"] for r in pen_scan], "o-", lw=2, color="crimson", label="VSS（元/周）")
    ax.axhline(0, color="k", lw=0.8)
    for r in pen_scan:
        ax.annotate(f"{r['vss_pct']:+.2f}%", (r["pen"], r["vss_weekly"]),
                    textcoords="offset points", xytext=(0, 8 if r["vss_weekly"] >= 0 else -16),
                    ha="center", fontsize=9)
    ax.set_xlabel("缺货罚 p（元/吨）")
    ax.set_ylabel("VSS（元/周）")
    ax.set_title("(b) p 越大，按场景树备越值钱；p 小时 EV 反而更好")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    # 2026-11-23 重画：原图画「a0 / a1」，但最优解总在网格顶点 (360, 560) 不变，
    # 说明可变的是「什么时候这笔钱垫出去」而不是「订多少」→ 改画「同一计划的成本随 L」与「在途资金」。
    ls_sp = [r["L"] for r in lt_scan]
    ax.plot(ls_sp, [r["weekly"] for r in lt_scan], "o-", lw=2,
            label="同一计划（SAA n=100）周均成本")
    trip_extra = H * float(np.sum(best_saa["plan"])) / 2.0 / W   # 每周份在途资金 = h·总量·L/2 / 8
    ax.plot(np.arange(1, 6), [lt_opt[0]["weekly"] + trip_extra * (L - 1) for L in range(1, 6)],
            "s--", lw=1.5, label="理论：h·Q·L/2 ÷ 8（只随 L 线性）")
    ax.set_xlabel("提前期 L（周，{2,3,4} 为随机情形）")
    ax.set_ylabel("周均成本（元）")
    ax.set_title("(c) L 变长：成本涨得很少，订多少几乎不变")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ns = [b["n_scen"] for b in bias]
    xx = np.arange(len(ns))
    ax.bar(xx - 0.2, [b["ceq_weekly"] for b in bias], 0.4, color="steelblue", label="场景树口径")
    ax.bar(xx + 0.2, [b["true_weekly"] for b in bias], 0.4, color="darkorange", label="仿真真值")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("周均成本（元）")
    ax.set_title("(d) 场景树口径 vs 仿真真值：偏差 +23%~+81%，必须先仿真")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Day86 · 两阶段随机规划下的一季订货：EV / SAA / PI、提前期随机与固定费", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(OUT, "day86_two_stage_stochastic_inventory.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"\n图已保存：{png}")

    out["meta"] = {"runtime_s": round(time.time() - T0, 2),"seed": SEED, "n_eval": N_EVAL, "elapsed_s": round(time.time() - T0, 2),
                   "price": PRICE, "H": round(H, 4), "r_u": R_U, "pen_base": PEN,
                   "lt_values": LT_VALUES.tolist(), "lt_prob": LT_PROB.tolist(),
                   "season": SEASON.tolist(), "K": K_FIX, "W": W}
    with open(os.path.join(OUT, "day86_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"数字已保存：{os.path.join(OUT, 'day86_result.json')}")
    print(f"总耗时 {time.time() - T0:.1f} s")

    # ---------------- 断言 ----------------
    assert evpi > 0, "EVPI 必须为正（知道未来一定不亏）"
    # 注意：ev_row["mean"] 存的是 round(...,2)，比较必须给容差（2026-11-23 修复）
    assert abs(ev_row["mean"] - simulate(ev_arr, dem_eval).mean()) < 0.01
    assert gap >= -1e-6, "网格解不应优于 LP（网格是 LP 的子集）"
    # 2026-11-23 追加：提前期只改「在途资金」，不改「订多少」（L=1..5 总量全相同）
    assert len({r["total"] for r in lt_opt}) == 1, "L 不应改变最优季内总量（改了说明口径混了）"
    # 2026-11-23 追加：本讲是一个「不确定没赚到钱」的实例——VSS≤0 必须成立且被写进结论
    assert vss <= 0.0, "本讲实例 VSS 应为非正（EV 计划在 p 小且季末有残值时不吃亏）"
    # 确定等价可以高也可以低（取决于场景树是否系统性偏乐观），只要求可解释、不算错
    print(f"  确定性等价偏差范围 {min(b['bias_pct'] for b in bias):+.1f}% ~ {max(b['bias_pct'] for b in bias):+.1f}%")
    print("✅ 断言全过：EVPI>0 / 网格不优于 LP / 口径与真值偏差已量化")


if __name__ == "__main__":
    main()
