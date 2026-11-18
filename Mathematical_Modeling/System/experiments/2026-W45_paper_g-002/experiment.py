#!/usr/bin/env python3
"""2026-W45 每周比较：证据纪律 — 论文中每个数字必须可追溯到 result.json（先跑后写）

攻击短板 G-002（priority P1, category paper）。

【这个实验到底在测什么】
不是测"某个模型好不好"，而是测**证据纪律这个工具本身管不管用**：
  - B1 基准的 `evaluate.py` 里有一条 critical 规则 ——「论文引用的成本数字必须能在 result.json 里找到（容差 1 元）」。
  - 这条规则是 G-002 的"工具化"落地物。问题是：**它真的能抓到违规吗？会不会误伤正确数字？**
  - 做法：把论文里的成本数字**逐个篡改/保留**，看 evaluate.py 的 critical 列表是否符合预期。

【为什么不预设"谁该赢"（纪律 §3，G-003 的教训）】
断言只检验三件事：
  ① **检测器有效性**：篡改过的数字必须被抓到（否则工具是摆设）；
  ② **零误伤**：未经篡改的正确数字必须全部放行（否则工具在制造噪声）；
  ③ **边界准确**：容差 1.0 元的边界行为必须与规则声明一致（0.99 放行 / 1.01 拦截）。
**断言里没有出现"应该抓几个"这种数字** —— 抓几个由规则本身决定，不由我们的期望决定。

【数据来源】
不是模拟数据：直接读 `benchmarks/2026-11_B1_coldchain/solver_output/result.json`（真实跑出来的求解结果）
与 `paper.md`（当时写的论文）。**先跑后写** —— 本实验不产生新的成本数字，只审计已有的。

run: cd /home/ubuntu/Note && ./.venv/bin/python \
       "Mathematical_Modeling/System/experiments/2026-W45_paper_g-002/experiment.py"
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # System/
sys.path.insert(0, str(ROOT))
from lib.env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
B1 = ROOT / "benchmarks" / "2026-11_B1_coldchain"
TOL = 1.0                                   # evaluate.py 里的容差，硬编码在规则里
NUM_RE = re.compile(r"(\d{4,6}\.\d{1,2})\s*元/周")


# ============================================================
# 1. 数据：读真实求解结果 + 真实论文
# ============================================================
def build_data():
    sres = json.loads((B1 / "solver_output" / "result.json").read_text(encoding="utf-8"))
    paper = (B1 / "solver_output" / "paper.md").read_text(encoding="utf-8")

    # evaluate.py 认定的"合法成本集合"（三个口径）
    legal = {
        "deterministic": round(sres["deterministic"]["cost"], 1),
        "split_service": round(sres["split_service"]["cost"], 1),
        "chance_constrained": round(sres["chance_constrained"]["cost"], 1),
    }
    stated = [float(m) for m in NUM_RE.findall(paper)]
    return {"sres": sres, "paper": paper, "legal": legal, "stated": stated}


# ============================================================
# 2. 复刻 evaluate.py 的那条规则（独立实现，不 import evaluate.py 的函数）
# ============================================================
def rule_unverified(paper_text, legal_costs, tol=TOL):
    """返回论文中无法在合法集合里复现的成本数字列表。"""
    stated = [float(m) for m in NUM_RE.findall(paper_text)]
    return [x for x in stated if not any(abs(x - a) < tol for a in legal_costs.values())]


# ============================================================
# 3. 变异测试：篡改每一个数字，看规则的反应
# ============================================================
def run_experiment(data):
    paper, legal, stated = data["paper"], data["legal"], data["stated"]
    base_hits = rule_unverified(paper, legal)

    # 3.1 基线：未篡改的论文
    baseline = {"n_stated": len(stated), "n_hits": len(base_hits), "hits": base_hits}

    # 3.2 单点篡改：把论文里第 i 个成本数字改成「不相交的随机值」
    mutations = []
    for i, v in enumerate(stated):
        # 替换成一个离所有合法值都 >= 50 元的值（保证一定越界）
        bad = round(v + 50.0, 2)
        # 注意：只替换第 i 次出现，避免误改别的数字
        nth = [0]

        def _repl(m, nth=nth, i=i, bad=bad):
            cur = nth[0]
            nth[0] += 1
            return f"{bad:.2f}元/周" if cur == i else m.group(0)

        mutated = NUM_RE.sub(_repl, paper)
        hits = rule_unverified(mutated, legal)
        mutations.append({
            "index": i, "original": v, "mutated": bad,
            "detected": bool(bad in hits),
            "n_hits": len(hits),
        })

    # 3.3 边界测试：容差 1.0 元的精确行为
    ref = legal["deterministic"]
    boundary = []
    for delta, label in [(0.0, "完全相同"), (0.99, "容差内"), (1.0, "恰好=容差"), (1.01, "刚越界")]:
        probe = f"总成本 {ref + delta:.2f}元/周"
        hits = rule_unverified(probe, legal)
        boundary.append({
            "delta": delta, "label": label, "probe_value": round(ref + delta, 2),
            "flagged": len(hits) > 0,
            # 规则代码写的是 abs(x-a) < tol，所以 delta 恰好 = 1.0 时**不算命中**
            "expected_flagged_by_rule": not (abs(delta) < TOL),
        })

    # 3.4 三个口径是否都被认定为合法
    coverage = []
    for name, val in legal.items():
        probe = f"该口径总成本 {val:.2f}元/周"
        coverage.append({"scope": name, "value": val, "accepted": len(rule_unverified(probe, legal)) == 0})

    # 3.5 用一个完全编造的数字（B1 之外的口径）测试能否拦住
    fabricated = "编造成本 9999.99元/周"
    fab_hits = rule_unverified(fabricated, legal)

    return {
        "data": {
            "source": "benchmarks/2026-11_B1_coldchain/solver_output/{result.json,paper.md}",
            "is_simulated": False,
            "n_stated_in_paper": len(stated),
            "stated_values": stated,
            "legal_costs": legal,
        },
        "rule": {"tol": TOL, "regex": NUM_RE.pattern,
                 "note": "复刻 evaluate.py 第 137-141 行的 critical 规则，独立实现"},
        "baseline": baseline,
        "mutation_results": mutations,
        "boundary_results": boundary,
        "scope_coverage": coverage,
        "fabricated_test": {"value": 9999.99, "flagged": len(fab_hits) > 0},
        "checks": [
            # ① 检测器有效性 —— 每一次单点篡改都必须被抓到
            {"check": "篡改后的数字 100% 被 detect（工具不是摆设）",
             "pass": all(m["detected"] for m in mutations),
             "detail": f"{sum(m['detected'] for m in mutations)}/{len(mutations)} detected"},
            # ② 零误伤 —— 未篡改论文必须零命中
            {"check": "未篡改论文零误伤（工具不制造噪声）",
             "pass": len(base_hits) == 0,
             "detail": f"hits={base_hits}"},
            # ③ 边界与规则声明一致
            {"check": "容差边界行为与规则声明一致（|Δ|<1.0 放行，≥1.0 拦截）",
             "pass": all(b["flagged"] == b["expected_flagged_by_rule"] for b in boundary),
             "detail": str([(b["delta"], b["flagged"]) for b in boundary])},
            # ④ 三个合法口径都被接受
            {"check": "三个口径（deterministic / split / chance）全部被认定为合法",
             "pass": all(c["accepted"] for c in coverage),
             "detail": str([(c["scope"], c["accepted"]) for c in coverage])},
            # ⑤ 编造数字必须拦住
            {"check": "B1 之外的编造数字被拦截",
             "pass": fab_hits == [9999.99],
             "detail": f"hits={fab_hits}"},
        ],
    }


def main() -> int:
    seed_all(CONFIG["experiment"]["seed"])
    data = build_data()
    result = run_experiment(data)
    result["task_id"] = CONFIG["task_id"]
    result["target_gap"] = CONFIG["target_gap"]
    result["config"] = CONFIG["experiment"]
    result["env"] = snapshot(
        HERE, seed=CONFIG["experiment"]["seed"],
        data_files=[B1 / "solver_output" / "result.json",
                    B1 / "solver_output" / "paper.md"],
        config=CONFIG["experiment"],
        extra={"audited_scope": "evaluate.py critical 规则（成本数字可追溯）"})
    result["fingerprint"] = result_fingerprint(result)

    (HERE / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"论文里出现的成本数字：{result['data']['stated_values']}")
    print(f"合法集合：{result['data']['legal_costs']}")
    print(f"基线命中：{result['baseline']['n_hits']} 个")
    print(f"单点篡改：{sum(m['detected'] for m in result['mutation_results'])}/"
          f"{len(result['mutation_results'])} 被抓到")
    print("边界：", [(b["delta"], b["flagged"]) for b in result["boundary_results"]])
    print("口径覆盖：", [(c["scope"], c["accepted"]) for c in result["scope_coverage"]])
    print("编造测试：", result["fabricated_test"])
    bad = [c for c in result["checks"] if not c["pass"]]
    print("checks:", bad if bad else "ALL PASS")
    print("fingerprint:", result["fingerprint"])
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
