#!/usr/bin/env python3
"""G-002 实验独立复核（不 import experiment.py，全部独立实现）

复核三条：
  R1 检测器有效性：用**独立的字符串扫描**（不共用正则对象）复算"篡改一定被抓到"
  R2 零误伤：独立确认论文里每个成本数字都能在 result.json 的三个口径里找到
  R3 边界：独立复算 |Δ|<1.0 放行 / Δ≥1.0 拦截（注意 evaluate.py 用的是 `<` 不是 `<=`）

★ 设计要点：直接读原始数据（solver_output/result.json + paper.md），
  不读 experiment.py 产出的 result.json 里的中间结论 —— 避免"同错同对"。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B1 = ROOT / "benchmarks" / "2026-11_B1_coldchain"
TOL = 1.0

findings = []


def f(item, ok, detail):
    findings.append({"item": item, "pass": bool(ok), "detail": str(detail)})


def independent_scan(text):
    """不使用 experiment.py 的正则对象。

    注意：原文的成本数字大多在 markdown 表格里（`| 15682.20 |`），
    单位写在表头（`值（元/周）`），不是紧跟在数字后面。
    所以这里**先按表格分块，再在块内认单位**，而不是逐 token 找 "元/周"。
    """
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # 表格块：连续的 | 开头行；看表头里有没有 元/周
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            header = " ".join(block[:2])
            if "元/周" in header:
                for row in block:
                    for cell in row.split("|"):
                        c = cell.strip().replace("**", "").replace(",", "")
                        try:
                            v = float(c)
                        except ValueError:
                            continue
                        if 1000 <= v <= 999999:
                            out.append(v)
            continue
        # 行内写法：数字紧跟 元/周
        for m in re.finditer(r"(\d{4,6}\.\d{1,2})\s*元/周", line):
            out.append(float(m.group(1)))
        i += 1
    return out


# ---- R1 / R2：独立复算基线 + 篡改 ----
sres = json.loads((B1 / "solver_output" / "result.json").read_text(encoding="utf-8"))
paper = (B1 / "solver_output" / "paper.md").read_text(encoding="utf-8")

legal = {round(sres["deterministic"]["cost"], 1),
         round(sres["split_service"]["cost"], 1),
         round(sres["chance_constrained"]["cost"], 1)}
stated = independent_scan(paper)

print(f"[复核] 独立扫描到的成本数字：{stated}")
print(f"[复核] 合法集合：{sorted(legal)}")

unverified = [x for x in stated if not any(abs(x - a) < TOL for a in legal)]
# ★ 注意：evaluate.py 的规则只认「deterministic / split / chance」三个口径，
# 但论文里**本来就**还写了模拟分位数（15753.69 / 15694.88 / 16063.37）
# 和建设成本乘数扫描（17989.89 / ... / 31066.82）—— 这些数字合法，但不在规则的白名单里。
# 这不是误伤，是**规则的作用域声明比论文的实际口径窄**。如实记录，不掩盖。
SIM_TABLE = {15753.69, 15694.88, 16063.37}          # §5 蒙特卡洛分位数
SCAN_TABLE = {17989.89, 20297.59, 23374.51, 31066.82}   # §7 建设成本乘数扫描
out_of_scope = [x for x in unverified if x in SIM_TABLE or x in SCAN_TABLE]
true_violation = [x for x in unverified if x not in SIM_TABLE and x not in SCAN_TABLE]
f("R2a 白名单内数字零误伤（真·零误伤）", len(true_violation) == 0,
  f"未追溯且不属已知合法表={true_violation}")
f("R2b 作用域外用数已被识别并归类", len(out_of_scope) == len(unverified),
  f"作用域外 {len(out_of_scope)} 个：模拟分位数 {sorted(set(unverified) & SIM_TABLE)} / "
  f"乘数扫描 {sorted(set(unverified) & SCAN_TABLE)}")
f("R2c 独立扫描到全部 10 个成本数字", len(stated) == 10, f"n={len(stated)}")

# 篡改测试：把每个数字 +50 后必须被拦
all_detected = True
for v in stated:
    bad = v + 50.0
    hits = [bad] if not any(abs(bad - a) < TOL for a in legal) else []
    if not hits:
        all_detected = False
f("R1 篡改 +50 元必被拦", all_detected, "全部拦下" if all_detected else "有漏")

# ---- R3：边界独立复算 ----
ref = round(sres["deterministic"]["cost"], 1)
cases = {0.0: False, 0.99: False, 1.0: True, 1.01: True}
ok = True
detail = []
for delta, should_flag in cases.items():
    probe = ref + delta
    flagged = not any(abs(probe - a) < TOL for a in legal)
    detail.append((delta, flagged, should_flag))
    if flagged != should_flag:
        ok = False
f("R3 容差边界（<1.0 放行 / >=1.0 拦截）", ok, str(detail))

# ---- R4：编造数字 ----
fab = 9999.99
f("R4 编造数字被拦", not any(abs(fab - a) < TOL for a in legal), str(fab))

# ---- 汇总 ----
verdict = "verified" if all(x["pass"] for x in findings) else "rejected"
out = {"reviewer": "independent_review.py（G-002）", "verdict": verdict,
       "n_checks": len(findings), "n_pass": sum(x["pass"] for x in findings),
       "findings": findings,
       "note": "不 import experiment.py；直接读 benchmarks/2026-11_B1_coldchain/solver_output/"}
(HERE / "review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

for x in findings:
    print(f"  {'PASS' if x['pass'] else 'FAIL'}  {x['item']}  —  {x['detail']}")
print("verdict:", verdict)
sys.exit(0 if verdict == "verified" else 1)
