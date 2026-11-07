"""B1 参考解（PRIVATE）—— Evaluator 对照用。

求：确定性 MILP 最优、Pareto 前沿（ε-约束）、随机需求 95% 分位成本。
运行：cd ~/Note && ./.venv/bin/python "Mathematical_Modeling/System/benchmarks/2026-11_B1_coldchain/private/reference.py"
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

HERE = Path(__file__).resolve().parent

MU = np.array([3.2, 4.8, 2.1, 5.6, 3.9, 1.7, 4.4])
SIG = np.array([0.9, 1.5, 0.6, 1.8, 1.1, 0.5, 1.3])
FIX_YEAR = np.array([42, 35, 28], float)      # 万元/年
FIX_OPS = np.array([6, 5, 4], float)          # 万元/年
CAP = np.array([18, 14, 10], float)
D = np.array([[12, 18, 25, 9, 15, 30, 22],
              [20, 10, 14, 22, 8, 26, 12],
              [28, 22, 8, 30, 20, 12, 18]], float)
TRANS = 0.85      # 元/(吨·km)
RUSH = 200.0      # 元/吨
RUSH_KM = 20.0

NEIGHBORS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 3), (1, 4)]  # 地理相邻


def annual_to_week(x):
    return x * 1e4 / 52


def solve(I: int = 7, J: int = 3, max_avg_dist: float | None = None, k: float = 1.0):
    """I 乡镇、J 站。变量 z: [y(3), x(3*7), e(7)]"""
    nx = J * I
    n = J + nx + I
    c = np.zeros(n)
    c[:J] = annual_to_week(FIX_YEAR[:J] + FIX_OPS[:J]) * k
    for j in range(J):
        for i in range(I):
            c[J + j * I + i] = TRANS * D[j, i] * MU[i]
    for i in range(I):
        c[J + nx + i] = RUSH * MU[i]

    cons = []
    # 1 每个乡镇恰好一个站
    for i in range(I):
        row = np.zeros(n)
        for j in range(J):
            row[J + j * I + i] = 1
        cons.append(LinearConstraint(row, 1, 1))
    # 2 x <= y
    for j in range(J):
        for i in range(I):
            row = np.zeros(n)
            row[J + j * I + i] = 1
            row[j] = -1
            cons.append(LinearConstraint(row, -np.inf, 0))
    # 3 能力
    for j in range(J):
        row = np.zeros(n)
        for i in range(I):
            row[J + j * I + i] = MU[i]
        row[j] = -CAP[j]
        cons.append(LinearConstraint(row, -np.inf, 0))
    # 4 加急费联动
    for i in range(I):
        row = np.zeros(n)
        for j in range(J):
            if D[j, i] > RUSH_KM:
                row[J + j * I + i] = 1
        row[J + nx + i] = -1
        cons.append(LinearConstraint(row, -np.inf, 0))
    # 6 至少开 2 站
    row = np.zeros(n); row[:J] = 1
    cons.append(LinearConstraint(row, 2, np.inf))
    # 7 平均距离上限（ε-约束）
    if max_avg_dist is not None:
        row = np.zeros(n)
        for j in range(J):
            for i in range(I):
                row[J + j * I + i] = D[j, i] * MU[i]
        cons.append(LinearConstraint(row, 0, max_avg_dist * MU.sum()))

    integ = np.zeros(n); integ[:J] = 1; integ[J:J + nx] = 1
    bnd = Bounds(np.zeros(n), np.ones(n))
    r = milp(c, constraints=cons, integrality=integ, bounds=bnd, options={"mip_rel_gap": 0})
    if r.status != 0:
        return None
    y = np.round(r.x[:J]).astype(int)
    x = np.round(r.x[J:J + nx]).reshape(J, I).astype(int)
    e = np.round(r.x[J + nx:]).astype(int)
    avg_d = float((D * x * MU).sum() / MU.sum())
    return {"cost": float(r.fun), "y": y.tolist(), "x": x.tolist(), "e": e.tolist(),
            "avg_dist": round(avg_d, 4)}


def corr_matrix():
    R = np.eye(7)
    for a, b in NEIGHBORS:
        R[a, b] = R[b, a] = 0.4
    # 环形+弦集导致矩阵非正定（最小特征值 < 0）→ 特征值截断投影，再归一化
    ev, V = np.linalg.eigh(R)
    ev = np.clip(ev, 1e-6, None)
    R = V @ np.diag(ev) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def simulate(sol, n_sim=200_000, seed=2026):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix())
    x = np.array(sol["x"])
    cost = np.zeros(n_sim)
    overload_cost = np.zeros(n_sim)
    extra_rate = 1.6  # 超载部分外部调车单价倍数
    yv = np.array(sol["y"])
    fixed = (FIX_YEAR + FIX_OPS)[yv == 1].sum() * 1e4 / 52
    for b in range(0, n_sim, 10_000):
        m = min(10_000, n_sim - b)
        z = rng.standard_normal((m, 7)) @ L.T
        dem = MU + SIG * z
        dem = np.clip(dem, 0, None)
        for j in range(3):
            load = (x[j] * dem).sum(axis=1)
            over = np.clip(load - CAP[j], 0, None)
            overload_cost[b:b + m] += over * extra_rate * TRANS * 40  # 备用车按 40km 计
        base_transport = (x * MU * D).sum() * TRANS
        unit_transport = TRANS * (D * x).sum(axis=0)          # shape (I,)
        unit_rush = (np.where(D > RUSH_KM, RUSH, 0) * x).sum(axis=0)  # shape (I,)
        cost[b:b + m] = (fixed + base_transport
                         + (dem - MU) @ unit_transport
                         + dem @ unit_rush)
    cost = cost + overload_cost
    return {"mean": float(cost.mean()), "p95": float(np.quantile(cost, 0.95)),
            "p50": float(np.quantile(cost, 0.50)), "seed": seed, "n_sim": n_sim}


if __name__ == "__main__":
    det = solve()
    print("确定性最优:", json.dumps(det, ensure_ascii=False))
    # ε-约束 Pareto 前沿
    front = []
    for eps in np.linspace(det["avg_dist"] * 1.02, det["avg_dist"] * 2.5, 15):
        s = solve(max_avg_dist=eps)
        if s:
            front.append((round(s["cost"], 1), s["avg_dist"], tuple(s["y"])))
    front = sorted(set(front))
    print("Pareto 候选:", front)
    print("随机 95%:", json.dumps(simulate(det), ensure_ascii=False))
    for kk in [1.0, 1.15, 1.3, 1.5, 2.0]:
        s = solve(k=kk)
        print(f"建设成本×{kk}: y={s['y'] if s else None} cost={s['cost'] if s else None}")
    (HERE / "reference_result.json").write_text(json.dumps({
        "deterministic": det, "pareto": front, "stochastic": simulate(det),
        "multiplier_scan": {str(k): solve(k=k) for k in [1.0, 1.15, 1.3, 1.5, 2.0]},
        "corr_neighbors": NEIGHBORS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
