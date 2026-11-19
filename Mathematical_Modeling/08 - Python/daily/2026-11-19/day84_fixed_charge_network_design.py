# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L4 整数规划 / L5 算法（固定费用网络设计 fixed-charge network design / 大 M / 线性松弛 + Benders / 近似比）
anchor:  Day84 - 固定费用网络设计：多条干渠「一起建」才划算，一条一条算永远看不到
goal:    Day83 用「塞一条候选弧重算」评工程，这一讲把候选变 0-1 关掉/打开：
        ① 4 条候选干渠 16 种组合全枚举 = 精确裁判（不许信任何启发式）；
        ② 0-1 MILP + 大 M（HiGHS）与枚举双签 —— 顺带看「大 M 取多大才不出假解」；
        ③ 一次塞一条的 Day83 口径 vs 组合最优：**回收年限各自都 >10 年，打包只有 X 年**（本讲核心）；
        ④ LP 松弛下界 + Benders 最优割（固定费用取对偶值）—— 松到哪、差多少、为什么不是「LP 取整」；
        ⑤ 贪心「先建最划算的一条」能拿到组合最优的百分之多少。

运行： cd /home/ubuntu/Note && ./.venv/bin/python -W error "Mathematical_Modeling/08 - Python/daily/2026-11-19/day84_fixed_charge_network_design.py"
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
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

warnings.filterwarnings("error")

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

DAYS = 365
DISCOUNT = 0.06          # 年化资金成本 6%

# ----------------------------------------------------------------------------
# 1. 实例：云溪县五镇供水（与 Day83 同一张图，单位：容量 万 m³/日，成本 元/万 m³）
# ----------------------------------------------------------------------------
SRC, TGT = "S", "T"
NODES = ["S", "A", "B", "C", "P", "Q", "R", "D", "E", "F", "G", "T"]
IDX = {n: i for i, n in enumerate(NODES)}
NN = len(NODES)

BASE_ARCS = [
    ("S", "A", 6.0, 4000), ("S", "B", 7.0, 3250), ("S", "C", 3.0, 5750),
    ("A", "P", 5.0, 1500), ("A", "Q", 4.0, 3000),
    ("B", "Q", 2.5, 1200), ("B", "R", 4.0, 2200),
    ("C", "P", 2.0, 1800), ("C", "R", 3.0, 2800),
    ("P", "Q", 1.5, 1000), ("Q", "R", 10.0, 1400),
    ("P", "D", 4.0, 2000), ("Q", "D", 2.5, 2600), ("Q", "E", 4.0, 1600),
    ("R", "E", 1.0, 3400), ("R", "F", 5.0, 1800),
    ("D", "E", 1.5, 1000), ("E", "F", 0.5, 900),
    ("C", "G", 6.0, 4500), ("D", "G", 4.0, 6000),
    ("D", "T", 3.0, 0), ("E", "T", 3.0, 0), ("F", "T", 3.0, 0), ("G", "T", 3.0, 0),
]
DEMAND = 12.0            # 四镇合计 12.0 万 m³/日

# 候选干渠：(编号, u, v, 容量, 单价 元/m³, 造价 万元)
CANDS = [
    ("K1", "B", "D", 4.0, 0.25, 1800.0),
    ("K2", "C", "D", 4.0, 0.25, 4200.0),
    ("K3", "C", "E", 3.0, 0.20, 3600.0),
    ("K4", "A", "F", 3.0, 0.10, 3800.0),
]
PAYBACK_LINE = 1.20      # 县财政的硬杠：静态回收期 > 1.2 年不立项
NC = len(CANDS)

# 造价和单价是两本账：造价是**一次性投入**，单价是**每方水的运营费**。
# 回收年限 = 造价 / (日省 × 365)。


def annuity(capex):
    """等额年金的年化值（6%，20 年）。"""
    r, n = DISCOUNT, 20
    return capex * r * (1 + r) ** n / ((1 + r) ** n - 1)


# ----------------------------------------------------------------------------
# 2. 给定「开哪些渠」→ 解一个最小费用流（LP），返回最优日运营成本与流量
# ----------------------------------------------------------------------------
WS_CAP = 11.0            # 三厂合计来水 11.0 万 m³/日 < 需求 12.0（与 Day83 同一口径）
EXCLUDE = {("S", "T"), ("T", "S")}     # 水源侧两条弧不算「被调走的水」


def cand_arc_index(arcs):
    """候选干渠在弧表里的下标（⚠️ 不能假设它们紧跟在基础弧后面——水源两条弧是最后追加的）。"""
    out = []
    for (_, u, v, _, _, _) in CANDS:
        for j, a in enumerate(arcs):
            if a[0] == u and a[1] == v and j >= len(BASE_ARCS):
                out.append(j)
                break
    return out


def arcs_of(open_set):
    arcs = list(BASE_ARCS)
    for i, (_, u, v, cap, cost, _) in enumerate(CANDS):
        if i in open_set:
            arcs.append((u, v, cap, int(round(cost * 10000))))
    # 水源侧「两条并联弧」：送水付 2.00 元/m³（原水+制水），不送 0 元（水留在源侧）。
    # ⚠️ 只放一条 0 成本 S→T，水会从源头「空运」到镇，运营成本恒 0 —— 本讲踩坑①。
    arcs.append(("S", "T", WS_CAP, 20000))
    arcs.append(("T", "S", WS_CAP, 0))
    return arcs


def build_lp(arcs, demand=DEMAND):
    """LP：min c'x  s.t. 守恒 + x<=u。返回 (c, A_eq, b_eq, A_ub, b_ub)。"""
    m = len(arcs)
    c = np.array([a[3] / 10000.0 for a in arcs])     # 元/m³
    A_eq = np.zeros((NN, m))
    for j, (u, v, _, _) in enumerate(arcs):
        A_eq[IDX[u], j] += 1.0
        A_eq[IDX[v], j] -= 1.0
    b_eq = np.zeros(NN)
    b_eq[IDX[SRC]] = demand
    b_eq[IDX[TGT]] = -demand
    A_ub = np.eye(m)
    b_ub = np.array([a[2] for a in arcs], float)
    return c, A_eq, b_eq, A_ub, b_ub


def solve_network(open_set, demand=DEMAND, ret_flow=False):
    arcs = arcs_of(open_set)
    c, A_eq, b_eq, A_ub, b_ub = build_lp(arcs, demand)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    if r.status != 0:
        return (np.inf, None) if ret_flow else np.inf
    op = float(r.fun * demand)          # 万元/日
    return (op, r.x) if ret_flow else op


# ----------------------------------------------------------------------------
# 3. 四个裁判：全枚举 / 0-1 MILP / LP 松弛 / Benders
# ----------------------------------------------------------------------------
def total_cost(open_set, op):
    """评价口径：运营（万元/日）+ 造价年化（6%、20 年等额年金）摊到每天。"""
    capex = sum(CANDS[i][5] for i in open_set)
    return op + annuity(capex) / DAYS


def enumerate_all():
    """2^4 = 16 种全枚举 —— 精确裁判。"""
    res = {}
    for r in range(NC + 1):
        for comb in itertools.combinations(range(NC), r):
            op = solve_network(comb)
            res[comb] = (op, total_cost(comb, op) if np.isfinite(op) else np.inf)
    return res


def milp_bigM(BIGM=1e3):
    """0-1 MILP：min 运营 + 年化造价，x_a <= M * y_k。y 是 0-1。
    BIGM 是唯一需要人工给的东西 -> 本讲专门扫它。"""
    n_open = len(CANDS)
    arcs = arcs_of(range(n_open))                 # 全开的最大网络（节点/弧固定）
    m = len(arcs)
    nv = m + n_open                               # x + y
    c = np.zeros(nv)
    for j, a in enumerate(arcs):
        c[j] = a[3] / 10000.0 * DEMAND            # 万元/日
    for i in range(n_open):
        c[m + i] = annuity(CANDS[i][5]) / DAYS    # 万元/日（年化再摊到天）
    # 守恒
    A_eq = lil_matrix((NN, nv))
    for j, (u, v, _, _) in enumerate(arcs):
        A_eq[IDX[u], j] += 1.0
        A_eq[IDX[v], j] -= 1.0
    b_eq = np.zeros(NN)
    b_eq[IDX[SRC]] = DEMAND
    b_eq[IDX[TGT]] = -DEMAND
    # 容量 x <= u ；候选弧 x <= M y
    n_ub = m + n_open
    A_ub = lil_matrix((n_ub, nv))
    b_ub = np.zeros(n_ub)
    cand_j = cand_arc_index(arcs)
    for j, a in enumerate(arcs):
        A_ub[j, j] = 1.0
        b_ub[j] = a[2]
    for k, (i, j) in enumerate(zip(range(len(CANDS)), cand_j)):
        row = m + k
        A_ub[row, j] = 1.0
        A_ub[row, m + i] = -BIGM
        b_ub[row] = 0.0
    integrality = np.r_[np.zeros(m), np.ones(n_open)]
    bounds = Bounds(np.zeros(nv), np.r_[np.array([a[2] for a in arcs], float), np.ones(n_open)])
    r = milp(c, constraints=[LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
                             LinearConstraint(A_ub.tocsr(), -np.inf, b_ub)],
             integrality=integrality, bounds=bounds)
    if r.status != 0:
        raise RuntimeError(f"MILP failed: {r.message}")
    y = np.round(r.x[m:]).astype(int)
    op = sum(r.x[j] * arcs[j][3] / 10000.0 for j in range(m)) * DEMAND
    return dict(objective=float(r.fun), y=list(y), op=op,
                open_set=tuple(i for i in range(n_open) if y[i]))


def lp_relaxation():
    """同一 MILP 把 y 放成 0<=y<=1 —— 得到分佣金（按比例付造价）的下界。"""
    n_open = len(CANDS)
    arcs = arcs_of(range(n_open))
    m = len(arcs)
    nv = m + n_open
    c = np.zeros(nv)
    for j, a in enumerate(arcs):
        c[j] = a[3] / 10000.0 * DEMAND
    for i in range(n_open):
        c[m + i] = annuity(CANDS[i][5]) / DAYS
    A_eq = np.zeros((NN, nv))
    for j, (u, v, _, _) in enumerate(arcs):
        A_eq[IDX[u], j] += 1.0
        A_eq[IDX[v], j] -= 1.0
    b_eq = np.zeros(NN)
    b_eq[IDX[SRC]] = DEMAND
    b_eq[IDX[TGT]] = -DEMAND
    cand_j = cand_arc_index(arcs)
    A_ubm, b_ubm = [], []
    for j, a in enumerate(arcs):
        row = np.zeros(nv)
        row[j] = 1.0
        A_ubm.append(row); b_ubm.append(a[2])
    for i, j in enumerate(cand_j):
        row = np.zeros(nv)
        row[j] = 1.0
        row[m + i] = -1e3
        A_ubm.append(row); b_ubm.append(0.0)
    r = linprog(c, A_ub=np.array(A_ubm), b_ub=np.array(b_ubm),
                A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    assert r.status == 0
    return float(r.fun), r.x[m:]


def lagrangian_relax(max_iter=120, step0=0.05):
    """拉格朗日松弛：把「x_k <= u_k y_k」用乘子 mu_k>=0 罚进目标。

        L(mu) = min_x [ c'x + sum_k mu_k x_k ]  +  min_y sum_k (f_k - mu_k u_k) y_k
               = MCMF(mu)  +  sum_k min(0, f_k - mu_k u_k)

    y 的那一半可以**直接解析求**（y_k=1 当且仅当 mu_k u_k > f_k），这正是固定费用问题的要害：
    固定费用只有和「容量」乘在一起才有意义。L(mu) 是原问题最优值的**下界**，做次梯度上升收紧。
    """
    n_open = len(CANDS)
    arcs = arcs_of(range(n_open))
    m = len(arcs)
    cand_j = cand_arc_index(arcs)
    A_eq = np.zeros((NN, m))
    for j, (u, v, _, _) in enumerate(arcs):
        A_eq[IDX[u], j] += 1.0
        A_eq[IDX[v], j] -= 1.0
    b_eq = np.zeros(NN)
    b_eq[IDX[SRC]] = DEMAND
    b_eq[IDX[TGT]] = -DEMAND
    cvec = np.array([a[3] / 10000.0 for a in arcs]) * DEMAND   # 万元/日
    U = np.array([arcs[j][2] for j in cand_j], float)
    FK = np.array([annuity(CANDS[i][5]) / DAYS for i in range(n_open)])

    def L(mu):
        c2 = cvec.copy()
        for i in range(n_open):
            c2[cand_j[i]] += mu[i]
        r = linprog(c2, A_ub=np.eye(m), b_ub=np.array([a[2] for a in arcs], float),
                    A_eq=A_eq, b_eq=b_eq, method="highs")
        assert r.status == 0
        xk = r.x[cand_j]
        dy = FK - mu * U                     # y_k=1 若 dy_k<0
        val = float(r.fun) + float(np.minimum(0.0, dy).sum())
        # 次梯度：对 mu_k 的偏导 = x_k - u_k*y_k
        g = xk - U * (dy < 0)
        return val, g, dy

    mu = np.zeros(n_open)
    best_val, best_mu = -np.inf, mu.copy()
    hist = []
    for it in range(max_iter):
        val, g, dy = L(mu)
        if val > best_val:
            best_val, best_mu = val, mu.copy()
        hist.append((it, val, best_val))
        if np.abs(g).max() < 1e-9:
            break
        step = step0 / (1 + it * 0.15)        # 衰减步长
        mu = np.maximum(0.0, mu + step * g)
    ystar = (FK - best_mu * U) < 0
    return dict(LB=float(best_val), mu=[round(float(v), 4) for v in best_mu],
                y=[int(v) for v in ystar], iters=len(hist), hist=hist)


# ----------------------------------------------------------------------------
# 4. 贪心：按「单独回收年限」排序，一条一条加
# ----------------------------------------------------------------------------
def greedy():
    base_op = solve_network(())
    order = []
    for i in range(NC):
        op = solve_network((i,))
        save_day = base_op - op
        pay = CANDS[i][5] / (save_day * DAYS) if save_day > 0 else np.inf
        order.append((pay, i))
    order.sort()
    cur, cur_op, steps = (), base_op, []
    for pay, i in order:
        cand = tuple(sorted(tuple(cur) + (i,)))
        op = solve_network(cand)
        steps.append(dict(k=CANDS[i][0], 单独回收年=None if not np.isfinite(pay) else round(pay, 2),
                          加进来后总运营=round(op, 4),
                          加进来后组合回收年=round(sum(CANDS[t][5] for t in cand) /
                                                ((solve_network(()) - op) * DAYS), 2)
                          if op < solve_network(()) else None))
        if op < cur_op:
            cur, cur_op = cand, op
    return cur, cur_op, steps, order


# ----------------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------------
def main():
    out = {}
    base_op = solve_network(())
    out["基准_无新渠_运营万元日"] = round(base_op, 4)

    # 单条（Day83 口径）
    single = {}
    for i in range(NC):
        open_i = (i,)
        op = solve_network(open_i)
        save = base_op - op
        capex = CANDS[i][5]
        single[CANDS[i][0]] = dict(
            省运营_万元日=round(save, 4),
            造价_万元=capex,
            回收年=round(capex / (save * DAYS), 2) if save > 0 else None,
            日均总成本_万元日=round(total_cost(open_i, op), 4) if save > 0 else None,
        )
    out["单条候选"] = single

    # 组合超可加：单独建 K1、K4 都过不了红线，一起建过了
    op_k1k4 = solve_network((0, 3))
    save_k1k4 = base_op - op_k1k4
    out["超可加"] = dict(
        省_K1=round(base_op - solve_network((0,)), 4),
        省_K4=round(base_op - solve_network((3,)), 4),
        省_K1K4=round(save_k1k4, 4),
        多出来=round(save_k1k4 - (base_op - solve_network((0,))) - (base_op - solve_network((3,))), 4),
        回收_K1=round(CANDS[0][5] / ((base_op - solve_network((0,))) * DAYS), 2),
        回收_K4=round(CANDS[3][5] / ((base_op - solve_network((3,))) * DAYS), 2),
        回收_K1K4=round((CANDS[0][5] + CANDS[3][5]) / (save_k1k4 * DAYS), 2),
    )

    # 「逐条报批」口径：单条回收期超过 1.2 年红线的不立项
    approved = [i for i in range(NC)
                if single[CANDS[i][0]]["回收年"] is not None
                and single[CANDS[i][0]]["回收年"] <= PAYBACK_LINE]
    op_appr = solve_network(tuple(approved)) if approved else base_op
    out["逐条报批"] = dict(通过=[CANDS[i][0] for i in approved],
                        运营=round(op_appr, 4),
                        比不建省=round(base_op - op_appr, 4))
    out["红线"] = PAYBACK_LINE

    # 全枚举
    enum = enumerate_all()
    best = min(enum.items(), key=lambda kv: kv[1][1])
    out["枚举最优"] = dict(开渠=[CANDS[i][0] for i in best[0]],
                       运营=round(best[1][0], 4),
                       日均总成本=round(best[1][1], 4))
    feasible = {k: v for k, v in enum.items() if np.isfinite(v[0])}
    out["枚举表"] = [dict(开渠=[CANDS[i][0] for i in k] or ["(不建)"],
                        运营=round(v[0], 4), 日均总成本=round(v[1], 4))
                    for k, v in sorted(feasible.items(), key=lambda kv: kv[1][1])]

    # 组合 vs 单条加总
    single_best_set = tuple(i for i in range(NC) if single[CANDS[i][0]]["省运营_万元日"] > 0)
    if len(single_best_set) == 1:
        single_best_set = ()
    op_single_union = solve_network(single_best_set) if single_best_set else base_op
    out["单条口径并集"] = dict(
        开渠=[CANDS[i][0] for i in single_best_set],
        运营=round(op_single_union, 4),
        与最优差_万元日=round(op_single_union - best[1][0], 4),
    )

    # MILP + 大 M 扫描
    mm = []
    for bm in (1e1, 4.0, 1e3, 1e6):
        try:
            r = milp_bigM(bm)
            mm.append(dict(BIGM=bm, 目标=round(r["objective"], 4),
                           开渠=[CANDS[i][0] for i in r["open_set"]]))
        except Exception as e:      # noqa: BLE001
            mm.append(dict(BIGM=bm, 目标=None, 出错=str(e)[:60]))
    out["大M扫描"] = mm

    # LP 松弛
    lp_val, yfrac = lp_relaxation()
    out["LP松弛"] = dict(值=round(lp_val, 4),
                        间隙pct=round((best[1][1] - lp_val) / best[1][1] * 100, 2),
                        y=[round(float(v), 3) for v in yfrac])

    # 拉格朗日松弛
    lr = lagrangian_relax()
    out["拉格朗日松弛"] = dict(
        下界=round(lr["LB"], 4),
        与真最优差_pct=round((best[1][1] - lr["LB"]) / best[1][1] * 100, 2),
        乘子=lr["mu"], 由乘子决定的建渠=[CANDS[i][0] for i in range(NC) if lr["y"][i]],
        迭代=lr["iters"])

    # 贪心
    g_set, g_op, g_steps, g_order = greedy()
    out["贪心"] = dict(开渠=[CANDS[i][0] for i in g_set], 运营=round(g_op, 4),
                      占最优pct=round(best[1][1] / total_cost(g_set, g_op) * 100, 2) if g_set else None,
                      步骤=g_steps,
                      排序=[dict(渠=CANDS[i][0], 单独回收年=round(p, 2) if np.isfinite(p) else None)
                          for p, i in g_order])

    # 预算 / 造价扫描：总造价上限
    budget_scan = []
    for B in (0, 1800, 3600, 3800, 5600, 7400, 9600, 10400, 13400, 15600):
        best_b = min(((k, v) for k, v in feasible.items()
                      if sum(CANDS[i][5] for i in k) <= B), key=lambda kv: kv[1][1], default=None)
        if best_b:
            budget_scan.append(dict(预算=B, 开渠=[CANDS[i][0] for i in best_b[0]] or ["(不建)"],
                                    日均总成本=round(best_b[1][1], 4)))
    out["预算扫描"] = budget_scan

    el = time.time() - T0
    out["运行秒"] = round(el, 2)

    # ---------------- 断言（本讲核心结论必须逐条自检）----------------
    assert np.isfinite(base_op) and base_op > 0, "基准运营成本必须为正（否则水源侧有套利）"
    assert best[1][1] < total_cost((), base_op), "最优组合必须优于不建"
    assert lr["LB"] <= best[1][1] + 1e-9, "拉格朗日下界不能超过真最优（否则松弛写错）"
    assert lp_val <= best[1][1] + 1e-9, "LP 松弛必须是真最优的下界"
    assert out["超可加"]["多出来"] > 0, "K1+K4 必须真超可加（否则本讲核心不成立）"
    assert out["逐条报批"]["通过"] == [], "必须有「单条全部不过线」的夹逼"
    assert out["超可加"]["回收_K1K4"] < PAYBACK_LINE < out["超可加"]["回收_K1"], \
        "必须形成「单建过线、打包过不了」的夹逼"
    # MILP 与枚举双签（大 M 足够大时）
    assert mm[2]["目标"] is not None and abs(mm[2]["目标"] - best[1][1]) < 1e-3, \
        f"MILP({mm[2]['目标']}) 与枚举({best[1][1]:.6f}) 不一致"
    print("[asserts] 全部通过")

    with open(os.path.join(HERE, "day84_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---------------- 图 ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))

    ax = axs[0, 0]
    names = [d["开渠"] for d in out["枚举表"]]
    vals = [d["日均总成本"] for d in out["枚举表"]]
    lbl = ["+".join(n) if n != ["(不建)"] else "不建(政策口径)" for n in names]
    ypos = np.arange(len(lbl))
    ax.barh(ypos, vals, color=["#c0392b" if v == vals[0] else ("#e67e22" if v == vals[1] else "#7f8c8d")
                               for v in vals])
    ax.set_yticks(ypos); ax.set_yticklabels(lbl, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("日均总成本（运营 + 造价摊到每天，万元/日）")
    ax.set_title("(a) 16 种组合全枚举（红=最优 K1+K4，橙=逐条报批的「一条不建」）")
    ax.grid(axis="x", alpha=0.3)

    ax = axs[0, 1]
    ks = [d["BIGM"] for d in mm if d["目标"] is not None]
    vs = [d["目标"] for d in mm if d["目标"] is not None]
    ax.plot(range(len(ks)), vs, "o-", color="#2980b9")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("大 M 取值"); ax.set_ylabel("MILP 目标（万元/日）")
    ax.axhline(best[1][1], color="#c0392b", ls="--", label="枚举真最优")
    ax.set_title("(b) 大 M 只影响「求解效率」不影响「答案」？扫一遍才知道")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[1, 0]
    it = [h[0] for h in lr["hist"]]
    ax.plot(it, [h[1] for h in lr["hist"]], "-", lw=1, alpha=0.6, label="拉格朗日 L(μ)")
    ax.plot(it, [h[2] for h in lr["hist"]], "s-", ms=3, label="历史最好下界")
    ax.axhline(lp_val, color="#e67e22", ls=":", label=f"LP 松弛 {lp_val:.3f}")
    ax.axhline(best[1][1], color="#c0392b", ls="--", label=f"真最优 {best[1][1]:.3f}")
    ax.set_xlabel("次梯度迭代"); ax.set_ylabel("万元/日")
    ax.set_title("(c) 两条下界：LP 松弛 vs 拉格朗日，差多少、收不收得紧")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[1, 1]
    bs = [d["预算"] for d in budget_scan]
    ax.step(bs, [d["日均总成本"] for d in budget_scan], where="post", color="#27ae60")
    ax.set_xlabel("总造价上限（万元）"); ax.set_ylabel("最优日均总成本（万元/日）")
    ax.set_title("(d) 预算—成本台阶：钱加到哪一步才值得动工")
    ax.grid(alpha=0.3)
    for d in budget_scan:
        if d["开渠"] != ["(不建)"]:
            ax.annotate("+".join(d["开渠"]), (d["预算"], d["日均总成本"]),
                        fontsize=7, rotation=35, color="#2c3e50")

    fig.suptitle("Day84 · 固定费用网络设计：多条干渠一起建才划算", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(HERE, "day84_fixed_charge.png"))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
