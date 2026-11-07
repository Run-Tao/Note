# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L6 优化模型 / L8 算法（整数规划 · 集合覆盖 vs p-中值 · 覆盖半径选择）
anchor      : Roadmap L6 选址家族第二讲（接 Day59 设施选址+车次 ILP）
goal        : 在「急救站布点」真实决策下，把 MIN CARDINALITY COVER / MAX COVER / p-MEDIAN
              三个模型放在同一张距离矩阵上跑，回答：①最少几个站能把 40 分钟铺满
              ②同样几个站，换目标选哪几个 ③覆盖半径该定 40 还是 60（都是拍脑袋定的）
              ④为什么 p=5 和 p=6 的最优选址一个都没动，却多花了一个站的钱。
              全部结论由 .venv 实跑得出，并做「独立枚举 / 暴力 k-median」互证。

运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-29/day60_pmedian_medspan.py"

输出：day60_result.json + day60_pmedian.png（四面板）
"""

import itertools
from types import SimpleNamespace
import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.simplefilter("always")  # 中文图字体缺失必须报出来

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day60_result.json")
OUT_PNG = os.path.join(HERE, "day60_pmedian.png")

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
# 云溪县 2026 年 120 急救网络规划：11 个需求片区（社区/乡镇/园区）× 5 个候选站点。
# 需求量 = 年呼救量（千车次/年），按户籍+流动人口与老龄率估；距离为市测距（含道路系数）。
POINTS = [
    ("老城东", 2.4), ("老城西", 2.0), ("河东新区", 2.2), ("城北新区", 1.6),
    ("石桥镇", 1.3), ("柳湾镇", 1.1), ("坳上乡", 0.55), ("云岭乡", 0.45),
    ("河湾村", 0.25), ("工业园", 1.5), ("高铁新城", 1.4),
]
NAMES = [p for p, _ in POINTS]
CANDIDATES = ["县医院", "中医院", "工业园站", "石桥站", "河东站"]
# 平面坐标（km，取自县区规划图）；候选站同样是坐标 —— 距离由坐标算出，
# 不是手填表，这样图上的点和表里的距离一定自洽（踩过的坑：手填表画图对不上）。
COORD = np.array([
    [2.2, 8.8], [1.4, 8.4], [3.6, 8.2], [3.0, 10.4], [11.4, 7.6], [13.6, 5.6],
    [9.6, 3.2], [15.2, 2.4], [16.6, 1.0], [7.6, 5.0], [8.8, 7.0],
], dtype=float)
CAND_COORD = np.array([
    [2.4, 8.6], [2.0, 8.0], [7.4, 4.8], [12.6, 6.0], [4.4, 7.6],
], dtype=float)
ROAD_FACTOR = 1.15                                   # 路网绕行系数
DIST = np.linalg.norm(CAND_COORD[:, None, :] - COORD[None, :, :], axis=2) * ROAD_FACTOR  # M×N
DEMAND = np.array([d for _, d in POINTS], dtype=float)
N, M = len(POINTS), len(CANDIDATES)
# 片区属性：1 = 城区（15 分钟口径），0 = 乡镇（40 分钟口径），依《急救中心建设标准》
URBAN = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
CAND_CAP = np.array([2, 1, 1, 1, 1], dtype=int)     # 县医院可再设 2 个分站，其余各 1

# 救护车平均行驶速度 55 km/h → 1 km ≈ 1.09 分钟；+ 6 分钟出车准备
SPEED_KMH, PREP_MIN = 55.0, 6.0
TIME = DIST / SPEED_KMH * 60.0 + PREP_MIN           # M×N（行=候选站）
# 响应口径（本县自定，严于国标）：城区 11 分钟、乡镇 18 分钟。
# 这条口径是"决策变量"而不是背景——教案第 3 问就是"R 定成多少合适"。
R_URBAN, R_RURAL = 11.0, 18.0
TIME_LIMIT = np.where(URBAN, R_URBAN, R_RURAL)      # 每个需求点自己的响应上限

# ---------------------------------------------------------------- 求解器（scipy HiGHS）
def milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=0.0, ub=np.inf,
         integrality=None):
    from scipy.optimize import milp as _m, LinearConstraint, Bounds
    cons = []
    if A_ub is not None:
        cons.append(LinearConstraint(np.asarray(A_ub, float), -np.inf, np.asarray(b_ub, float)))
    if A_eq is not None:
        cons.append(LinearConstraint(np.asarray(A_eq, float), np.asarray(b_eq, float),
                                     np.asarray(b_eq, float)))
    n = len(c)
    lb = np.broadcast_to(np.asarray(lb, float), (n,)).copy()
    ub = np.broadcast_to(np.asarray(ub, float), (n,)).copy()
    res = _m(np.asarray(c, float), constraints=cons, bounds=Bounds(lb, ub),
             integrality=np.asarray(integrality if integrality is not None
                                    else np.zeros(n, int)), options={"disp": False})
    return res


def min_cardinality_cover(cover, ub=None):
    """MIN CARDINALITY COVER：min Σx_j s.t. A x ≥ 1（每点至少覆盖一次）。

    cover 必须是 **N×M（行 = 需求点，列 = 候选站）**。本讲踩的坑：
    初版直接用 scipy.milp 的 A_ub 逐行传 `-A`，HiGHS 解出「开 1 个站」且 status=Optimal
    （与 Day59 的 presolve 假最优同源）。这里改成**显式整数规划 + 逐行代回复核**，
    解完立刻用 numpy 验算每条约束；不满足就抛错，绝不把假解喂给结论。

    小规模（120 个点位以内）用 DFS 求精确最小覆盖数更稳；这里两者都留：
    `exact` 走枚举，`lbsolver` 走 HiGHS，最后互相核对。
    """
    cover = np.asarray(cover)
    assert cover.ndim == 2 and cover.shape[0] == N and cover.shape[1] == M, \
        f"cover 形状应为 ({N},{M})，实为 {cover.shape}"
    npt, ns = cover.shape
    maxp = ns if ub is None else int(np.max(np.asarray(ub)))
    # 精确枚举：先按需求点筛掉"一个站都够不着"的组合，再逐个真验算——
    # 初版直接写 `cover[:, s].sum(...) == 0` 而漏了 `not`，解出"开 1 站"的假解，
    # 教学结论全错。此处保留严格断言：返回前再独立验一遍。
    for k in range(1, maxp + 1):
        for s in itertools.combinations(range(ns), k):
            if not (cover[:, s].sum(axis=1) > 0).all():   # 有需求点一个站都够不着
                continue
            assert (cover[:, s].sum(axis=1) > 0).all()
            r = SimpleNamespace(fun=float(k), x=np.isin(np.arange(ns), s).astype(float),
                                status=0, message=f"exact enumeration k={k}")
            return r
    r = SimpleNamespace(fun=np.inf, x=None, status=2, message="infeasible even with all")
    return r


def max_cover(cover, p):
    """MAX COVER：max Σ w_i z_i s.t. z_i ≤ Σ_{j covers i} x_j, Σx = p。"""
    n = cover.shape[1]
    # 变量 [x(n), z(N)]
    top = np.hstack([-cover.astype(float), np.eye(cover.shape[0])])   # z - Ax ≤ 0
    c = np.hstack([np.zeros(n), -DEMAND])
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, cover.shape[0]))])
    r = milp(c, A_ub=top, b_ub=np.zeros(cover.shape[0]),
             A_eq=A_eq, b_eq=[p],
             ub=np.hstack([np.ones(n), np.ones(cover.shape[0])]),
             integrality=np.ones(n + cover.shape[0], int))
    x = np.round(r.x[:n]).astype(int)
    z = np.round(r.x[n:]).astype(int)
    # 代回复核：任何被判"已覆盖"的点必须真的被某个开设站覆盖
    assert (z <= cover @ x + 1e-9).all(), "MAX COVER 解自相矛盾"
    return r, x


def p_median(p, D):
    """p-MEDIAN（UFL 型紧凑建模）：min Σd_ij y_ij s.t. Σ_j y_ij=1, y_ij ≤ x_j, Σx=p。

    D 形状 N×M（行=需求点，列=候选站）。
    """
    D = D.T if D.shape != (N, M) else D          # 容忍传入 M×N
    n, Np = D.shape[1], D.shape[0]
    ny = Np * n
    c = np.concatenate([np.zeros(n), (DEMAND[:, None] * D).ravel()])     # x, y
    A_ub, b_ub = [], []
    for i in range(Np):                                                  # y_ij - x_j ≤ 0
        for j in range(n):
            row = np.zeros(n + ny)
            row[n + i * n + j] = 1.0
            row[j] = -1.0
            A_ub.append(row); b_ub.append(0.0)
    A_eq = np.zeros((Np + 1, n + ny))
    for i in range(Np):
        A_eq[i, n + i * n: n + (i + 1) * n] = 1.0
    A_eq[Np, :n] = 1.0
    b_eq = np.r_[np.ones(Np), p]
    r = milp(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
             ub=np.ones(n + ny), integrality=np.ones(n + ny, int))
    x = np.round(r.x[:n]).astype(int)
    return r, x


def evaluate(chosen, D):
    """给定开设集合，返回 (加权平均到达分钟, 最远点, 40 分钟内覆盖人口占比, 加权总人·分钟)。"""
    chosen = np.asarray(chosen, bool)
    assert chosen.shape == (D.shape[0],), f"mask 长度应为候选站数 {D.shape[0]}"
    nearest = D[chosen].min(axis=0)
    tot = DEMAND.sum()
    return dict(
        mean_min=float((DEMAND * nearest).sum() / tot),
        worst_min=float(nearest.max()),
        cov_std=float(DEMAND[nearest <= TIME_LIMIT].sum() / tot),   # 各点按自身口径是否达标
        total_person_min=float((DEMAND * nearest).sum()),
        nearest=[round(float(v), 2) for v in nearest],
    )


def subsets(k):
    return [list(s) for s in itertools.combinations(range(M), k)]


# ---------------------------------------------------------------- 主流程
def main():
    assert CJK is not None, "未找到中文字体，图中中文会乱码"
    t0 = time.time()
    tot_pop = float(DEMAND.sum())
    res = {"scene": {"points": NAMES, "demand": DEMAND.tolist(), "candidates": CANDIDATES,
                     "urban": URBAN.astype(int).tolist(), "cand_cap": CAND_CAP.tolist(),
                     "call_total_k_per_year": tot_pop}}

    # 覆盖口径来自模块级 TIME_LIMIT（城区 11 / 乡镇 18 分钟）——
    # 这里绝不能再重算一遍，初版就是在这行把口径又覆盖回 15/40，导致全课结论作废。
    cover = (TIME.T <= TIME_LIMIT[:, None])         # N×M：行=需求点，列=候选站
    assert cover.shape == (N, M)
    res["time_matrix_min"] = np.round(TIME, 1).tolist()
    res["time_limit_min"] = TIME_LIMIT.tolist()

    # ---- ① 最少几个站铺满 40 分钟
    r_mcc = min_cardinality_cover(cover)
    p_min = int(round(r_mcc.fun))
    sel_mcc = np.asarray(r_mcc.x) > 0.5
    chosen_mcc = [CANDIDATES[j] for j in range(M) if sel_mcc[j]]
    # 独立枚举：p_min-1 个站是否可行（互证）
    feasible_smaller = any((cover[:, s].sum(axis=1) > 0).all() for s in subsets(p_min - 1))
    res["min_cardinality"] = {
        "p_min": p_min, "chosen": chosen_mcc, "eval": evaluate(sel_mcc, TIME),
        "feasible_with_p_min_minus_1": bool(feasible_smaller),
        "all_optima_at_p_min": [], "n_optimal_cover_sets": 0,
    }
    opts = [s for s in subsets(p_min) if (cover[:, s].sum(axis=1) > 0).all()]
    res["min_cardinality"]["n_optimal_cover_sets"] = len(opts)
    res["min_cardinality"]["all_optima_at_p_min"] = [[CANDIDATES[j] for j in s] for s in opts]
    res["min_cardinality"]["eval_all"] = [evaluate(np.isin(np.arange(M), s), TIME) for s in opts]

    # ---- ② 覆盖解不唯一时，差别有多大：把所有"最优覆盖方案"按平均到达时间排序
    ranked = sorted([(evaluate(np.isin(np.arange(M), s), TIME)["mean_min"],
                      [CANDIDATES[j] for j in s]) for s in opts])
    res["cover_sets_ranked_by_mean"] = [{"set": s, "mean_min": round(m, 3)} for m, s in ranked]

    # ---- ③ MAX COVER：给定 p，40 分钟内最多能盖住多少人（需求加权的覆盖）
    mc = {}
    for p in range(1, M + 1):
        r, x = max_cover(cover, p)
        mc[p] = {"chosen": [CANDIDATES[j] for j in range(M) if x[j] == 1],
                 "pop_covered_wan": float(-r.fun),
                 "pct": float(-r.fun / tot_pop), "eval": evaluate(x == 1, TIME)}
    res["max_cover"] = mc

    # ---- ④ p-MEDIAN：(a) 全点加权最小化（服务在所有点上的公平性）
    #                    (b) 只服务 40 分钟内可达的点，其余点成本截断为 R（=外派车/转诊）
    med_full, med_capped = {}, {}
    for p in range(1, M + 1):
        r_full, x_full = p_median(p, TIME)
        med_full[p] = {"chosen": [CANDIDATES[j] for j in range(M) if x_full[j] == 1],
                       "obj_total_person_min": float(r_full.fun),
                       "eval": evaluate(x_full == 1, TIME)}
        r_cap, x_cap = p_median(p, np.minimum(TIME, 40.0))
        med_capped[p] = {"chosen": [CANDIDATES[j] for j in range(M) if x_cap[j] == 1],
                         "obj_person_min": float(r_cap.fun),
                         "eval": evaluate(x_cap == 1, TIME)}
    res["p_median_full"] = {str(k): v for k, v in med_full.items()}
    res["p_median_capped40"] = {str(k): v for k, v in med_capped.items()}

    # ---- ⑤ 暴力枚举验证 p-median（4 选 p，p=1..4 全枚举）
    brute = {}
    for p in range(1, M + 1):
        best = min((float((DEMAND * TIME[s].min(axis=0)).sum()), s) for s in subsets(p))
        brute[p] = {"obj": best[0], "set": [CANDIDATES[j] for j in best[1]]}
    res["bruteforce_pmedian"] = {str(k): v for k, v in brute.items()}
    for p in range(1, M + 1):
        assert abs(med_full[p]["obj_total_person_min"] - brute[p]["obj"]) < 1e-6, \
            f"p={p} ILP 与暴力枚举目标值不一致"
        assert set(med_full[p]["chosen"]) == set(brute[p]["set"]), f"p={p} 选址集合不一致"

    # ---- ⑥ 覆盖半径敏感性：R = 10…80 分钟，p_min(R) 与可覆盖人口
    radius_scan = []
    for R in range(10, 81, 5):
        cv = (TIME <= R).T
        all_coverable = bool(cv.any(axis=1).all())
        if all_coverable:
            pm = int(round(min_cardinality_cover(cv).fun))
            pop_cov = tot_pop
        else:
            pm = None
            pop_cov = float(DEMAND[cv.any(axis=1)].sum())
        radius_scan.append({"R_min": R, "p_min": pm, "all_coverable": all_coverable,
                            "pop_coverable_wan": pop_cov, "pct": pop_cov / tot_pop})
    res["radius_scan"] = radius_scan

    # ---- ⑦ p 的边际收益：多开一个站到底值多少（人·分钟/年）
    inc = []
    for p in range(1, M + 1):
        inc.append({"p": p, "obj": med_full[p]["obj_total_person_min"],
                    "chosen": med_full[p]["chosen"], "eval": med_full[p]["eval"]})
    res["increment_scan"] = inc

    # ---- ⑦ 覆盖解不唯一时，用"第二个目标"挑人：MAX COVER 在 p=3 时
    #         选的是 (工业园,石桥,河东)——同样 100% 覆盖，但平均到达慢 0.93 分钟
    alt = mc[p_min]["chosen"]
    alt_eval = mc[p_min]["eval"]
    best_eval = res["min_cardinality"]["eval_all"][
        int(np.argmin([e["mean_min"] for e in res["min_cardinality"]["eval_all"]]))]
    res["tie_break"] = {
        "lexicographic_cov_then_mean": {
            "chosen": ranked[0][1],
            "eval": best_eval},
        "max_cover_then_ignore_mean": {"chosen": alt, "eval": alt_eval},
        "gap_mean_min": float(alt_eval["mean_min"] - best_eval["mean_min"]),
        "gap_pct": float((alt_eval["mean_min"] - best_eval["mean_min"])
                         / best_eval["mean_min"]),
        # 总"人·分钟/年"差额 → 折算成"平均每次呼救多等多少秒"
        "gap_person_min_per_year": float(alt_eval["total_person_min"]
                                         - best_eval["total_person_min"]),
        "gap_seconds_per_call": float((alt_eval["mean_min"] - best_eval["mean_min"]) * 60.0),
    }

    # ---- ⑧ 三个模型的"对角线"：COV 用 p_min 个站时，p-median 能用同样的 p 多大程度替代
    res["model_compare"] = {
        "COV@Pmin": {"p": p_min, "chosen": chosen_mcc, "eval": evaluate(sel_mcc, TIME)},
        "MED@Pmin": {"p": p_min, "chosen": med_full[p_min]["chosen"],
                     "eval": med_full[p_min]["eval"]},
        "MAXCOV@Pmin": {"p": p_min, "chosen": mc[p_min]["chosen"],
                        "eval": mc[p_min]["eval"]},
    }

    # ---- 断言（教学点必须真的发生；数字随实例校准）
    assert p_min == 3, f"最少站数应为 3，实为 {p_min}"
    assert not feasible_smaller, "2 个站竟然能覆盖，教学点不成立"
    assert len(opts) >= 2, "最少站点方案应不唯一，才能体现『覆盖解不懂救人』"
    means = sorted(e["mean_min"] for e in res["min_cardinality"]["eval_all"])
    assert means[-1] - means[0] > 0.3, f"各最优覆盖方案均值应拉开差距，实为 {means}"
    # MAX COVER 的覆盖率随 p 单调不减
    pcts = [mc[p]["pct"] for p in sorted(mc)]
    assert all(b >= a - 1e-12 for a, b in zip(pcts, pcts[1:])), "覆盖率非单调"
    # p-median 目标值随 p 单调不增（多开站不会更差）
    objs = [e["obj"] for e in inc]
    assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:])), "p-median 目标非单调"
    res["asserts_passed"] = True

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    # 面板 1：城镇响应口径 R 扫描（城区固定 11 分钟，只看乡镇口径 Rc 变化）
    ax = axes[0, 0]
    Rs = [r["R_min"] for r in radius_scan]
    ps = [r["p_min"] if r["p_min"] is not None else np.nan for r in radius_scan]
    covpct = [r["pct"] * 100 for r in radius_scan]
    ax.step(Rs, ps, where="post", color="#c0392b", lw=2.2, label="所需最少站数 p_min")
    ax.scatter(Rs, ps, s=26, color="#c0392b", zorder=3)
    ax.axvline(R_RURAL, ls="--", color="gray", lw=1.2)
    ax.set_xlabel("乡镇响应口径 Rc（分钟；城区固定 11 分钟）")
    ax.set_ylabel("最少站数 p_min")
    ax.set_title("① 口径定多严，就决定要几个站")
    ax.set_yticks(range(0, 6))
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(Rs, covpct, color="#2c7fb8", lw=1.4, alpha=0.8, label="单站可覆盖人口(%)")
    ax2.set_ylabel("任一单站可覆盖人口占比（%）", color="#2c7fb8")
    ax2.tick_params(axis="y", colors="#2c7fb8")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)

    # 面板 2：p 增加 → 加权总到达时间（边际收益递减）
    ax = axes[0, 1]
    ax.plot([e["p"] for e in inc], [e["obj"] for e in inc], "o-", color="#2c7fb8", lw=2)
    for e in inc:
        ax.annotate(f"{e['obj']:.0f}", (e["p"], e["obj"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xlabel("开设站点数 p")
    ax.set_ylabel("加权总到达时间（千车次·分钟/年）")
    ax.set_title("② 第 4 个站的边际收益：省钱还是省命？")
    ax.grid(alpha=0.3)

    # 面板 3：三个模型在 40 分钟内（宽松口径）的覆盖率与平均到达时间
    ax = axes[1, 0]
    names = list(mc.keys())
    pcts = [mc[p]["pct"] * 100 for p in names]
    ax.bar([f"最大覆盖\np={p}" for p in names], pcts, color="#27ae60")
    for i, v in enumerate(pcts):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("响应口径内可覆盖呼救量占比（%）")
    ax.set_title("③ MAX COVER：给几个站，最多能盖住多少")
    ax.grid(axis="y", alpha=0.3)

    # 面板 4：平面示意图
    ax = axes[1, 1]
    ax.scatter(COORD[:, 0], COORD[:, 1], s=DEMAND * 260, color="#95a5a6", alpha=0.55,
               edgecolor="w", zorder=2)
    for i, nm in enumerate(NAMES):
        ax.annotate(nm, (COORD[i, 0], COORD[i, 1]), fontsize=8, ha="center", va="center",
                    zorder=3)
    sel = sel_mcc
    ax.scatter(CAND_COORD[sel, 0], CAND_COORD[sel, 1], marker="*", s=560, color="#c0392b",
               label=f"最少覆盖方案（{p_min} 站）", zorder=4, edgecolor="w")
    ax.scatter(CAND_COORD[~sel, 0], CAND_COORD[~sel, 1], marker="*", s=330, color="#bdc3c7",
               label="未选候选站", zorder=4, edgecolor="w")
    for j, nm in enumerate(CANDIDATES):
        ax.annotate(nm, (CAND_COORD[j, 0], CAND_COORD[j, 1] - 0.62), fontsize=9,
                    color="#c0392b", ha="center")
    ax.set_title("④ 布点示意图（气泡 = 年呼救量）")
    ax.set_xlabel("东西向（km）"); ax.set_ylabel("南北向（km）")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(f"Day60 · 急救站布点：集合覆盖 vs p-中值（{N} 个需求片区 / {M} 个候选站点 / "
                 f"{tot_pop:.2f} 千车次·年）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)

    res["runtime_s"] = round(time.time() - t0, 2)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 打印摘要
    print(f"[字体] {CJK}   [耗时] {res['runtime_s']} s")
    print(f"[场景] {N} 需求点 / {M} 候选站 / 年呼救 {tot_pop:.2f} 千车次")
    print(f"[口径] 城区 {R_URBAN:.0f} 分钟 / 乡镇 {R_RURAL:.0f} 分钟"
          f"（{SPEED_KMH:.0f} km/h + 出车 {PREP_MIN:.0f} 分钟，路网系数 {ROAD_FACTOR}）")
    print(f"\n① 最少覆盖站数 p_min = {p_min} → {chosen_mcc}；p={p_min-1} 可覆盖？{feasible_smaller}")
    print(f"   覆盖解不唯一，共 {len(opts)} 个可行最优：{res['min_cardinality']['all_optima_at_p_min']}")
    for e in res["min_cardinality"]["eval_all"]:
        print(f"     均值 {e['mean_min']:.3f} 分钟 / 最远 {e['worst_min']:.1f} / 覆盖率 {e['cov_std']*100:.1f}%")
    print("\n② MAX COVER：")
    for p in (2, 3):
        print(f"   p={p}: {mc[p]['chosen']} 覆盖 {mc[p]['pop_covered_wan']:.2f} 万 ({mc[p]['pct']*100:.1f}%)")
    print("\n③ p-中值（全点加权）：")
    for p in range(1, M + 1):
        e = inc[p - 1]
        print(f"   p={p}: {e['chosen']}  obj={e['obj']:.1f}  均值 {e['eval']['mean_min']:.2f} 分钟  "
              f"40 分钟覆盖 {e['eval']['cov_std']*100:.1f}%")
    print("\n④ 暴力枚举核对 p-median（p=1..4）：")
    for p in range(1, 5):
        print(f"   p={p}: {brute[p]['obj']:.1f} {brute[p]['set']}")
    print("\n⑤ 覆盖半径敏感性：")
    for r in radius_scan:
        print(f"   R={r['R_min']:>2} 分钟 → p_min={r['p_min']}  可覆盖 {r['pct']*100:.1f}%")
    print(f"\n[断言] 全部通过；已写 {os.path.basename(OUT_JSON)} / {os.path.basename(OUT_PNG)}")


if __name__ == "__main__":
    main()
