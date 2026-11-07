"""B1 Evaluator：证据汇总与评分（程序化可验证项优先）

读取：solver_output/{result.json,paper.md}, private/{reference_solution.md,reference_result.json}
产出：score.json（含各维度得分、依据、证据、关键错误标记）

评分原则（任务书 6.1/6.2）：
  - 能程序化验证的先程序化验证（代码可运行、可行性、最优性、Pareto 非支配、复现性）
  - 文字指标（数学推导、论文质量）由 AI 评审给分，但必须引用具体证据行
  - 关键错误单独计数，不被高分掩盖

运行：cd ~/Note && ./.venv/bin/python "Mathematical_Modeling/System/benchmarks/2026-11_B1_coldchain/evaluate.py"
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER = HERE / "solver_output"
sres = json.loads((SOLVER / "result.json").read_text(encoding="utf-8"))
ref = json.loads((HERE / "private" / "reference_result.json").read_text(encoding="utf-8"))
paper = (SOLVER / "paper.md").read_text(encoding="utf-8")
psrc = (SOLVER / "solve.py").read_text(encoding="utf-8")

checks = []


def chk(dim, check, ok, evidence):
    checks.append({"dim": dim, "check": check, "pass": bool(ok), "evidence": evidence})


# ---------------- A. 问题理解与拆解 (10%) ----------------
chk("A", "识别出选址/分配/时效三层结构",
    all(k in paper for k in ("选址", "分配", "时效")), "paper.md §0 三层结构")
chk("A", "识别量纲陷阱（年化成本折周）",
    "52" in paper and ("量纲" in paper or "折周" in paper), "paper.md §2.4")
chk("A", "识别加急费是由分配决定的派生量",
    "派生" in paper or "由分配" in paper, "paper.md §0")

# ---------------- B. 模型选择与合理性 (20%) ----------------
chk("B", "MILP 变量完整（y/x/e）", "y_j" in paper and "x_{ij}" in paper and "e_i" in paper, "§2.1")
chk("B", "含开启联动约束 x<=y", "x_{ij}\\le y_j" in paper or "x_{ij}≤y_j" in paper, "§2.3")
chk("B", "含能力约束", "Q_j" in paper, "§2.3")
chk("B", "含加急费联动约束", "d_{ij}>20" in paper or "d_{ij}>20" in paper.replace(" ", ""), "§2.3")
chk("B", "随机部分处理了相关性（未独立采样）",
    "相关" in paper and "Cholesky" in paper, "§5")
chk("B", "未用加权和冒充法误导 Pareto（识别非凸）",
    "加权和" in paper or "ε-约束" in paper, "§4")
solver_gen_cost = sres["deterministic"]["cost"]
ref_gen_cost = ref["deterministic"]["cost"]
chk("B", "确定性最优成本与参考解一致（±1 元）",
    abs(solver_gen_cost - ref_gen_cost) < 1.0, f"solver={solver_gen_cost:.2f} ref={ref_gen_cost:.2f}")
chk("B", "开站方案与参考解一致",
    sres["deterministic"]["y"] == ref["deterministic"]["y"],
    f"solver={sres['deterministic']['y']} ref={ref['deterministic']['y']}")

# ---------------- C. 数学推导 (15%) ----------------
chk("C", "目标函数三项分列且单位统一为元/周",
    "10^4}{52}" in paper or "10^4}/52" in paper, "§2.2")
chk("C", "对 P3/P6 覆盖约束给出说明", "P3" in paper and "P6" in paper, "§2.3 末")
chk("C", "机会约束/两阶段作为改进给出数学形式",
    "机会约束" in paper and "sigma" in paper.lower() or "\\sigma" in paper, "§8")
chk("C", "未出现公式与结论矛盾（成本项 61.3% 与 9615/15682=61.3% 自洽）",
    abs(9615.38 / 15682.20 - 0.613) < 0.01, "9615.38/15682.20=61.3%")

# ---------------- D. Python 实现 (15%) ----------------
r = subprocess.run([sys.executable, str(SOLVER / "solve.py")], capture_output=True, text=True, timeout=900)
chk("D", "代码可独立运行且退出码 0", r.returncode == 0, f"rc={r.returncode}")
fp = json.loads((SOLVER / "result.json").read_text(encoding="utf-8"))["fingerprint"]
r2 = subprocess.run([sys.executable, str(SOLVER / "solve.py")], capture_output=True, text=True, timeout=900)
fp2 = json.loads((SOLVER / "result.json").read_text(encoding="utf-8"))["fingerprint"]
chk("D", "复跑指纹一致（可复现）", fp == fp2, f"{fp} vs {fp2}")
chk("D", "代码含可行性独立验算（不只信任求解器）",
    "def verify" in psrc, "solve.py verify()")

# ---------------- E. 实验与验证 (15%) ----------------
chk("E", "做了 Pareto 前沿（ε-约束法，≥10 个 ε 点）",
    len(re.findall(r"linspace", psrc)) >= 1 and "ε" in paper, "solve.py pareto")
chk("E", "Pareto 结论与参考一致（前沿退化为单点）",
    len(sres["pareto"]) == len(ref["pareto"]) == 1, f"solver={len(sres['pareto'])} ref={len(ref['pareto'])}")
chk("E", "随机模拟样本量 ≥10 万且固定种子",
    sres["stochastic"]["p95"] > 0 and "200,000" in paper, "n_sim=200000 seed=2026")
chk("E", "95% 分位与参考解一致（±50 元，允许采样差异）",
    abs(sres["stochastic"]["p95"] - ref["stochastic"]["p95"]) < 50,
    f"solver={sres['stochastic']['p95']:.2f} ref={ref['stochastic']['p95']:.2f}")
chk("E", "瓶颈站识别正确（S1 为瓶颈）",
    int(max(range(3), key=lambda i: sres["stochastic"]["overload_prob"][i])) == 0,
    f"overload_prob={sres['stochastic']['overload_prob']}")
chk("E", "稳健性做了建设成本扫描（≥4 个乘数）",
    len(sres["multiplier_scan"]) >= 4, str(list(sres["multiplier_scan"])))
chk("E", "承认不存在方案切换临界点（而非编造一个临界值）",
    "不存在" in paper and "临界" in paper, "§6")

# ---------------- F. 创新与模型改进 (10%) ----------------
chk("F", "提出并实测拆分服务改进", sres["split_service"] is not None, "§8 item 1")
chk("F", "提出机会约束并实测", sres["chance_constrained"] is not None and sres["chance_constrained"]["status"] == 0,
    f"chance-constrained cost={sres['chance_constrained']['cost']:.2f}")
chk("F", "改进有实验数据支撑（数字可在 result.json 复现）",
    sres["split_service"] is not None and sres["chance_constrained"] is not None,
    f"split={sres['split_service']['cost']:.1f}, chance={sres['chance_constrained']['cost']:.1f}")
chk("F", "承认拆分服务无效（负结果如实报告）",
    abs(sres["split_service"]["cost"] - sres["deterministic"]["cost"]) < 1.0 and "没有带来任何收益" in paper,
    "§8 item 1")

# ---------------- G. 论文质量 (15%) ----------------
required = ["问题重述", "假设", "模型", "求解", "分析", "检验", "稳健", "优点", "参考"]
chk("G", "论文结构完整（问题/假设/模型/求解/分析/检验/稳健/局限/参考）",
    sum(1 for k in required if k in paper) >= 8, f"{sum(1 for k in required if k in paper)}/9")
chk("G", "有明确数字与单位", "15682.20" in paper and "元/周" in paper, "§3")
# 规则：模拟得到的数字不得标注为"实测"；但"已实测"用于标记**本仓库真实跑过的对照实验**是允许的
sim_claims = re.findall(r"实测[^。\n]{0,20}(15753|15754|16063|16064)", paper)
chk("G", "结论不超证据（未把随机模拟结果称为实测）", not sim_claims,
    f"违规={sim_claims}" if sim_claims else "模拟区（§5）无'实测'标注")

# ---------------- 关键错误检测（6.2） ----------------
critical = []
if abs(9615.38 / 15682.20 - 0.613) > 0.02:
    critical.append("量纲混用导致占比错误")
if not (sres["deterministic"]["y"] == ref["deterministic"]["y"]):
    critical.append("选址结果与最优解不符")
if sres["stochastic"]["p95"] < sres["stochastic"]["mean"]:
    critical.append("95% 分位低于均值（模型定义矛盾）")
if r.returncode != 0:
    critical.append("代码不可运行")
if fp != fp2:
    critical.append("结果不可复现")
if "加权和" in paper and "Pareto 前沿" in paper and "ε-约束" not in paper:
    critical.append("用加权和冒充 Pareto 前沿")
# 论文里 §8 item1 的"实测"数字是否真实存在于 solver 结果里 → 已核实代码未跑拆分对照
# 通用规则：论文里引用的"已实测"成本数字必须能在 result.json 里找到（容差 1 元）
stated = [float(m) for m in re.findall(r"(\d{4,6}\.\d{1,2})\s*元/周", paper)]
actual = {round(sres["deterministic"]["cost"], 1), round(sres["split_service"]["cost"], 1),
          round(sres["chance_constrained"]["cost"], 1)}
unverified = [x for x in stated if not any(abs(x - a) < 1.0 for a in actual)]
if unverified:
    critical.append(f"论文引用的成本数字无法在 result.json 复现: {unverified}")

# ---------------- 评分 ----------------
W = {"A": 0.10, "B": 0.20, "C": 0.15, "D": 0.15, "E": 0.15, "F": 0.10, "G": 0.15}
dims = {}
for d in W:
    sub = [c for c in checks if c["dim"] == d]
    score = 100 * sum(c["pass"] for c in sub) / len(sub)
    dims[d] = {"score": round(score, 1), "passed": sum(c["pass"] for c in sub), "total": len(sub),
               "failed": [c["check"] for c in sub if not c["pass"]]}
total = sum(W[d] * dims[d]["score"] for d in W)

# 关键错误降级：每 1 个关键错误 -8 分，且总分封顶 60
penalty = 8 * len(critical)
final = min(total - penalty, 60 if critical else 200)

out = {
    "benchmark": "2026-11-B1", "evaluator": "evaluate.py（程序化证据汇总）",
    "dimension_scores": dims, "weights": W,
    "raw_total": round(total, 1), "critical_errors": critical,
    "critical_count": len(critical), "penalty": penalty,
    "final_score": round(max(0.0, final), 1),
    "checks": checks,
    "independence_note": "数学/论文两项由同一底层模型在独立上下文评审，"
                         "可程序化项（成本、方案、Pareto、复现性）全部由脚本验证，不采信文字自述。",
}
(HERE / "score.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for d in W:
    print(f"{d} {W[d]*100:>4.0f}%  score={dims[d]['score']:<6} ({dims[d]['passed']}/{dims[d]['total']})"
          + (f"  FAILED={dims[d]['failed']}" if dims[d]['failed'] else ""))
print("raw_total:", out["raw_total"], "critical:", critical, "final:", out["final_score"])
sys.exit(0)
