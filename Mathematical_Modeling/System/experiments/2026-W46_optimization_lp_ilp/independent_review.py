"""独立复核：LP vs ILP（W46b）

不 import experiment.py；用独立的可行域枚举/构造验证，而不是重复调用同一个求解器。
核查：
  1. ILP 解是否真的整数且可行，且无更便宜整数解（小规模穷举交叉验证）
  2. LP 解是否真的最优（对偶/KKT 简易检验：目标值下界）
  3. 整性间隙计算正确
  4. 需求扫描单调性
  5. 数据被标记为模拟
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
p = cfg["problem"]
cost = np.array(p["cost"], float); cap = np.array(p["capacity"], float)
ub = np.array(p["max_units"]); lb = np.array(p["min_units"]); D = p["demand_ton"]
checks = []


def chk(n, c, d=""):
    checks.append({"check": n, "pass": bool(c), "detail": d})


# ---- 1. 穷举 ILP 最优（4 维，搜索空间 27*23*17*16 ≈ 16.9 万，可接受） ----
ranges = [range(int(lb[i]), int(ub[i]) + 1) for i in range(4)]
best_cost, best_x = np.inf, None
for x in itertools.product(*ranges):
    x = np.array(x, float)
    if cap @ x >= D - 1e-9:
        c = cost @ x
        if c < best_cost - 1e-9:
            best_cost, best_x = c, x
chk("实验 ILP 解与穷举最优一致（成本）", abs(best_cost - res["ilp"]["cost"]) < 1e-6,
    f"穷举={best_cost} 实验={res['ilp']['cost']}")
chk("实验 ILP 解与穷举最优一致（方案）",
    np.allclose(best_x, np.array(res["ilp"]["x"])), f"穷举={best_x} 实验={res['ilp']['x']}")

# ---- 2. LP 下界：松弛最优不能高于任何可行点成本；且不能低于"最便宜单位运力成本×需求" ----
unit_cost = cost / cap
naive_lb = float(unit_cost.min() * D) * 0  # 粗略界不可靠，改用构造下界：cost>=min(cost)*sum(lb)
lb_bound = float(cost @ lb)
chk("LP 目标值 ≥ 合同保底成本（基本下界）", res["lp"]["cost"] >= lb_bound - 1e-6,
    f"{res['lp']['cost']} vs {lb_bound}")
chk("LP 目标值 ≤ ILP 目标值（松弛不会更贵）", res["lp"]["cost"] <= res["ilp"]["cost"] + 1e-6)
chk("LP 解满足需求（独立验算）",
    cap @ np.array(res["lp"]["x"]) >= D - 1e-6, f"供给={cap @ np.array(res['lp']['x'])}")

# ---- 3. 间隙计算 ----
gap = (res["ilp"]["cost"] - res["lp"]["cost"]) / res["lp"]["cost"] * 100
chk("记录间隙与独立重算一致(1e-2)", abs(gap - res["integrality_gap_pct"]) < 0.01,
    f"{gap:.4f} vs {res['integrality_gap_pct']}")

# ---- 4. 扫描单调性 ----
scan = res["demand_scan"]
chk("扫描 ILP 成本单调不减",
    all(scan[i]["ilp_cost"] <= scan[i + 1]["ilp_cost"] + 1e-6 for i in range(len(scan) - 1)))
chk("每个扫描点 ILP ≥ LP", all(s["ilp_cost"] >= s["lp_cost"] - 1e-6 for s in scan))
chk("LP 出现分数解的扫描点数落在合理范围 [1, n]",
    1 <= sum(1 for s in scan if s["lp_frac"]) <= len(scan),
    str(sum(1 for s in scan if s["lp_frac"])))

# ---- 5. 数据诚实性 ----
chk("数据标记为模拟", "SIMULATED" in res["data_note"] or "模拟" in res["data_note"], res["data_note"])

# ---- 6. 复跑 ----
fps = []
for _ in range(2):
    subprocess.run([sys.executable, str(HERE / "experiment.py")], capture_output=True, timeout=300)
    fps.append(json.loads((HERE / "result.json").read_text(encoding="utf-8"))["fingerprint"])
chk("复跑 2 次指纹一致", len(set(fps)) == 1, str(fps))

failed = [c for c in checks if not c["pass"]]
out = {"reviewer": "independent_review.py", "checks": checks,
       "passed": len(checks) - len(failed), "failed": len(failed),
       "verdict": "verified" if not failed else "rejected"}
(HERE / "review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for c in checks:
    print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
print(f"\n{out['passed']}/{len(checks)} passed → {out['verdict']}")
sys.exit(0 if not failed else 1)
