"""
W46 每周模型比较 #3：优化模型 — LP vs ILP（同现实问题的松弛 vs 整数化）

同一现实问题：县域 4 类运输方式的运力配置（车辆/班次采购）
  - LP 松弛：允许买 2.7 辆车（数学可解，现实不可行）
  - ILP：数量必须为整数
比较：目标值（费用）、整性间隙、求解时间、可行性、解稳定性（需求扰动）

求解器：scipy.optimize.linprog(HiGHS) 与 scipy.optimize.milp(HiGHS)
不使用随机算法，因此"解稳定性"通过需求参数扫描实现。

运行：cd ~/Note && ./.venv/bin/python "Mathematical_Modeling/System/experiments/2026-W46_optimization_lp_ilp/experiment.py"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = CFG["seed"]

# ----------------------------------------------------------------------------
# 问题设定（真实业务逻辑，模拟参数明确标记）
# 4 类运输方式：小型货车 / 中型货车 / 大型货车 / 夜间专线
# 参数：单车日成本(元)、日运力(吨)、最多可采购数量、最低服务保障
# 需求：日运输需求 640 吨；每类方式有最低使用下限（保底合同）
# ----------------------------------------------------------------------------
NAMES = ["小型货车", "中型货车", "大型货车", "夜间专线"]
COST = np.array(CFG["problem"]["cost"], float)       # 元/车·日
CAP = np.array(CFG["problem"]["capacity"], float)    # 吨/车·日
UB = np.array(CFG["problem"]["max_units"], float)    # 最多采购
LB_CONTRACT = np.array(CFG["problem"]["min_units"], float)  # 保底合同车辆数
DEMAND = float(CFG["problem"]["demand_ton"])


def solve_lp(D: float):
    """连续松弛：min c'x  s.t. cap'x >= D, 0<=x<=UB, x>=LB"""
    c = COST
    A_ub = -CAP.reshape(1, -1)          # -cap'x <= -D
    b_ub = np.array([-D])
    bounds = [(LB_CONTRACT[i], UB[i]) for i in range(len(NAMES))]
    t0 = time.perf_counter()
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    dt = time.perf_counter() - t0
    return r, dt


def solve_ilp(D: float):
    c = COST
    cons = LinearConstraint(CAP.reshape(1, -1), lb=D, ub=np.inf)
    bounds = Bounds(LB_CONTRACT, UB)
    t0 = time.perf_counter()
    r = milp(c, constraints=[cons], integrality=np.ones(len(NAMES)), bounds=bounds,
             options={"time_limit": 30, "mip_rel_gap": 0.0})
    dt = time.perf_counter() - t0
    return r, dt


def main() -> int:
    seed_all(SEED)
    lp, t_lp = solve_lp(DEMAND)
    il, t_il = solve_ilp(DEMAND)

    lp_x, il_x = lp.x, il.x
    lp_cost, il_cost = float(lp.fun), float(il.fun)
    gap = (il_cost - lp_cost) / lp_cost * 100

    # 可行性检查（不能信任求解器状态，自己验一遍）
    def feasible(x, D):
        return bool(CAP @ x >= D - 1e-9 and (x >= LB_CONTRACT - 1e-9).all()
                    and (x <= UB + 1e-9).all())

    # 稳定性：需求在 520~760 吨扫描，看整性间隙与方案结构
    scan = []
    for D in range(520, 761, 20):
        l, _ = solve_lp(float(D)); i, _ = solve_ilp(float(D))
        scan.append({
            "demand": D,
            "lp_cost": round(float(l.fun), 2), "ilp_cost": round(float(i.fun), 2),
            "gap_pct": round((float(i.fun) - float(l.fun)) / float(l.fun) * 100, 3),
            "lp_x": [round(float(v), 3) for v in l.x],
            "ilp_x": [round(float(v), 3) for v in i.x],
            "lp_frac": [k for k, v in enumerate(l.x) if abs(v - round(v)) > 1e-6],
        })

    out = {
        "experiment": "2026-W46_optimization_lp_ilp",
        "task_id": "weekly-2026-W46b",
        "data_note": "SIMULATED 运力配置参数（业务逻辑真实，数值为示意，不伪装现实观测）",
        "models": NAMES,
        "demand": DEMAND,
        "lp": {"cost": round(lp_cost, 2), "x": [round(float(v), 4) for v in lp_x],
               "status": int(lp.status), "time_s": round(t_lp, 4),
               "feasible": feasible(lp_x, DEMAND),
               "fractional_vars": [NAMES[k] for k, v in enumerate(lp_x) if abs(v - round(v)) > 1e-6]},
        "ilp": {"cost": round(il_cost, 2), "x": [round(float(v), 4) for v in il_x],
                "status": int(il.status), "mip_gap": getattr(il, "mip_gap", None),
                "time_s": round(t_il, 4), "feasible": feasible(il_x, DEMAND)},
        "integrality_gap_pct": round(gap, 3),
        "demand_scan": scan,
    }

    checks = []

    def chk(n, c, d=""):
        checks.append({"check": n, "pass": bool(c), "detail": d})

    chk("LP 求解成功(status=0)", lp.status == 0, str(lp.status))
    chk("ILP 求解成功(status=0)", il.status == 0, str(il.status))
    chk("两解均满足容量/上下限约束（独立验算）",
        out["lp"]["feasible"] and out["ilp"]["feasible"])
    chk("ILP 解全为整数", all(abs(v - round(v)) < 1e-6 for v in il_x), str([round(v, 3) for v in il_x]))
    chk("LP 松弛确实出现分数解（否则本次比较无意义）",
        len(out["lp"]["fractional_vars"]) >= 1, str(out["lp"]["fractional_vars"]))
    chk("ILP 目标值 ≥ LP 目标值（松弛是下界）", il_cost >= lp_cost - 1e-6,
        f"{il_cost:.2f} vs {lp_cost:.2f}")
    chk("整性间隙在合理区间 [0, 30%]", 0 <= gap <= 30, f"{gap:.3f}%")
    chk("需求扫描覆盖 ≥10 个点且全部可行",
        len(scan) >= 10 and all(s["ilp_cost"] > 0 for s in scan), f"{len(scan)} 点")
    chk("扫描中 ILP 成本随需求单调不减（更多需求不能更便宜）",
        all(scan[i]["ilp_cost"] <= scan[i + 1]["ilp_cost"] + 1e-6 for i in range(len(scan) - 1)))

    out["checks"] = checks
    out["fingerprint"] = result_fingerprint(out)
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot(HERE, seed=SEED, config=CFG, extra={"data_note": out["data_note"]})

    print("LP :", out["lp"])
    print("ILP:", out["ilp"])
    print("gap:", f"{gap:.3f}%")
    print("scan:", [(s["demand"], s["lp_cost"], s["ilp_cost"], s["gap_pct"]) for s in scan])
    print("checks:", [c for c in checks if not c["pass"]] or "ALL PASS")
    print("fingerprint:", out["fingerprint"])
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
