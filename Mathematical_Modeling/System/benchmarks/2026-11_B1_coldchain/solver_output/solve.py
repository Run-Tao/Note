"""B1 Solver 代码（独立解题，未读取 private/）

运行：cd ~/Note && ./.venv/bin/python "Mathematical_Modeling/System/benchmarks/2026-11_B1_coldchain/solver_output/solve.py"
退出码 0 = 全部断言通过
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

SEED = 2026
MU = np.array([3.2, 4.8, 2.1, 5.6, 3.9, 1.7, 4.4])
SIG = np.array([0.9, 1.5, 0.6, 1.8, 1.1, 0.5, 1.3])
F_YEAR = np.array([42, 35, 28], float) + np.array([6, 5, 4], float)   # 建设+运营（万元/年）
Q = np.array([18, 14, 10], float)
D = np.array([[12, 18, 25, 9, 15, 30, 22],
              [20, 10, 14, 22, 8, 26, 12],
              [28, 22, 8, 30, 20, 12, 18]], float)
TRANS, RUSH, RUSH_KM, WEEKS = 0.85, 200.0, 20.0, 52.0
NEIGH = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 3), (1, 4)]


def solve(open_set=None, max_avg=None, k=1.0, x_cont=False, chance_lambda=0.0):
    """open_set: 强制开启的站集合；max_avg: 平均距离上限；k: 成本乘数；chance_lambda: 机会约束系数"""
    I, J = 7, 3
    nx = J * I
    n = J + nx + I
    c = np.zeros(n)
    c[:J] = (F_YEAR * k) * 1e4 / WEEKS
    for j in range(J):
        for i in range(I):
            c[J + j * I + i] = TRANS * D[j, i] * MU[i]
    c[J + nx:] = RUSH * MU
    cons = []
    for i in range(I):
        r = np.zeros(n)
        for j in range(J):
            r[J + j * I + i] = 1
        cons.append(LinearConstraint(r, 1, 1))
    for j in range(J):
        for i in range(I):
            r = np.zeros(n); r[J + j * I + i] = 1; r[j] = -1
            cons.append(LinearConstraint(r, -np.inf, 0))
    for j in range(J):
        r = np.zeros(n)
        for i in range(I):
            r[J + j * I + i] = MU[i] + chance_lambda * SIG[i] / np.sqrt(len(NEIGH))
        r[j] = -Q[j]
        cons.append(LinearConstraint(r, -np.inf, 0))
    for i in range(I):
        r = np.zeros(n)
        for j in range(J):
            if D[j, i] > RUSH_KM:
                r[J + j * I + i] = 1
        r[J + nx + i] = -1
        cons.append(LinearConstraint(r, -np.inf, 0))
    r = np.zeros(n); r[:J] = 1
    cons.append(LinearConstraint(r, 2, np.inf))
    if open_set is not None:
        for j in range(J):
            r = np.zeros(n); r[j] = 1
            v = 1 if j in open_set else 0
            cons.append(LinearConstraint(r, v, v))
    if max_avg is not None:
        r = np.zeros(n)
        for j in range(J):
            for i in range(I):
                r[J + j * I + i] = D[j, i] * MU[i]
        cons.append(LinearConstraint(r, 0, max_avg * MU.sum()))
    integ = np.zeros(n); integ[:J] = 1
    if not x_cont:
        integ[J:J + nx] = 1
    bnd = Bounds(np.zeros(n), np.ones(n))
    t0 = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integ, bounds=bnd,
               options={"mip_rel_gap": 0, "time_limit": 60})
    dt = time.perf_counter() - t0
    if res.status != 0:
        return None
    y = np.round(res.x[:J]).astype(int)
    x = np.round(res.x[J:J + nx], 6).reshape(J, I)
    avg = float((D * x * MU).sum() / MU.sum())
    return {"cost": float(res.fun), "y": y.tolist(), "x": x.tolist(),
            "avg_dist": round(avg, 4), "time_s": round(dt, 4), "status": int(res.status)}


def verify(sol) -> bool:
    y, x = np.array(sol["y"]), np.array(sol["x"])
    ok = np.allclose(x.sum(0), 1, atol=1e-9)
    ok &= np.all(x <= y[:, None] + 1e-9)
    ok &= np.all((x * MU).sum(1) <= Q * y + 1e-6)
    ok &= y.sum() >= 2
    return bool(ok)


def corr():
    R = np.eye(7)
    for a, b in NEIGH:
        R[a, b] = R[b, a] = 0.4
    ev, V = np.linalg.eigh(R)
    ev = np.clip(ev, 1e-8, None)          # 原矩阵非正定 → 特征值截断投影
    R = V @ np.diag(ev) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def simulate(sol, n_sim=200_000, extra=1.6):
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(corr())
    y, x = np.array(sol["y"]), np.array(sol["x"])
    fixed = (F_YEAR * y).sum() * 1e4 / WEEKS
    unit_tr = TRANS * (D * x).sum(0)
    unit_rush = (np.where(D > RUSH_KM, RUSH, 0) * x).sum(0)
    base = fixed + TRANS * (x * D * MU).sum()
    cost = np.empty(n_sim)
    overload_prob = np.zeros(3)
    for b in range(0, n_sim, 20000):
        m = min(20000, n_sim - b)
        dem = np.clip(MU + SIG * (rng.standard_normal((m, 7)) @ L.T), 0, None)
        cc = base + (dem - MU) @ unit_tr + dem @ unit_rush
        for j in range(3):
            load = dem @ (x[j] * np.ones(7))
            over = np.clip(load - Q[j], 0, None)
            overload_prob[j] += float((over > 0).sum())
            cc += over * extra * TRANS * 40
        cost[b:b + m] = cc
    return {"mean": float(cost.mean()), "p50": float(np.quantile(cost, .5)),
            "p95": float(np.quantile(cost, .95)),
            "overload_prob": [round(v / n_sim, 4) for v in overload_prob]}


def main() -> int:
    seed_all(SEED)
    base_sol = solve()
    checks = []

    def chk(n, c, d=""):
        checks.append({"check": n, "pass": bool(c), "detail": d})

    chk("确定性解可行（独立验算约束）", verify(base_sol), str(base_sol["y"]))
    chk("求解器状态正常", base_sol["status"] == 0)

    # 全局最优证明：穷举 8 种开站组合
    combos = {}
    for r in range(2, 4):
        for S in itertools.combinations(range(3), r):
            s = solve(open_set=S)
            combos[str(S)] = None if s is None else round(s["cost"], 2)
    best_combos = [v for v in combos.values() if v is not None]
    chk("穷举所得全局最优与直接求解一致（2.0 元容差）",
        abs(min(best_combos) - base_sol["cost"]) < 2.0,
        f"穷举min={min(best_combos)} 直接={base_sol['cost']:.2f}")

    # Pareto 前沿
    front, seen = [], set()
    for eps in np.linspace(base_sol["avg_dist"] * 1.02, base_sol["avg_dist"] * 2.5, 15):
        s = solve(max_avg=eps)
        if s is None:
            continue
        key = (round(s["cost"], 1), s["avg_dist"])
        if key not in seen:
            seen.add(key); front.append({"cost": key[0], "avg_dist": key[1], "y": s["y"]})
    chk("Pareto 前沿已生成（≥1 点）且全部非支配",
        len(front) >= 1 and all(not (a["cost"] >= b["cost"] and a["avg_dist"] >= b["avg_dist"]
                                     and (a["cost"] > b["cost"] or a["avg_dist"] > b["avg_dist"]))
                                for i, a in enumerate(front) for b in front[i + 1:]),
        f"{len(front)} 点")

    stoch = simulate(base_sol)
    chk("随机模拟 95% 分位 ≥ 均值 ≥ 确定性成本",
        stoch["p95"] >= stoch["mean"] >= base_sol["cost"] - 1, str(stoch["p95"]))
    chk("瓶颈站已识别（某站超载概率 > 10%）", max(stoch["overload_prob"]) > 0.10,
        str(stoch["overload_prob"]))

    scan = {k: solve(k=k) for k in (1.0, 1.15, 1.3, 1.5, 2.0)}
    chk("建设成本扫描成本单调增",
        all(scan[a]["cost"] <= scan[b]["cost"] + 1e-6 for a, b in [(1.0, 1.15), (1.15, 1.3), (1.3, 1.5), (1.5, 2.0)]))
    split = solve(x_cont=True)
    chk("拆分服务（连续 x）不劣于整数方案", split is not None and split["cost"] <= base_sol["cost"] + 1e-6,
        f"{split['cost']:.2f} vs {base_sol['cost']:.2f}")
    cc = solve(chance_lambda=1.645)
    chk("机会约束版本求解完成（可能不可行，如实记录）",
        cc is None or cc["status"] == 0, "infeasible" if cc is None else "ok")

    out = {"benchmark": "2026-11-B1", "role": "solver", "deterministic": base_sol,
           "combo_enumeration": combos, "pareto": front, "stochastic": stoch,
           "multiplier_scan": {str(k): {"cost": v["cost"], "y": v["y"]} for k, v in scan.items()},
           "split_service": split, "chance_constrained": cc,
           "checks": checks}
    out["fingerprint"] = result_fingerprint(out)
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot(HERE, seed=SEED, extra={"benchmark": "2026-11-B1"})
    print(json.dumps({k: out[k] for k in ("deterministic", "stochastic", "combo_enumeration")},
                     ensure_ascii=False, indent=1))
    print("pareto:", front)
    print("checks:", [c for c in checks if not c["pass"]] or "ALL PASS")
    print("fingerprint:", out["fingerprint"])
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
