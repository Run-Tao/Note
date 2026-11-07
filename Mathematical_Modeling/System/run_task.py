#!/usr/bin/env python3
"""统一任务入口：把"跑实验 → 独立复核 → 记账 → 汇总"变成一条命令。

用法：
  ./.venv/bin/python run_task.py weekly  2026-W45_prediction_lr_arima
  ./.venv/bin/python run_task.py weekly  2026-W46_optimization_lp_ilp --new-task-id weekly-2026-W46c
  ./.venv/bin/python run_task.py monthly 2026-11_B1_coldchain

行为：
  1. 若未指定 --new-task-id，从实验目录的 result.json 读取 task_id
  2. 幂等：同 task_id 已 verified → 直接退出 0（不重复跑）
  3. 加锁：同 kind 正在 running → 退出 2
  4. 执行退出码必须为 0，否则记 failed 并保留日志
  5. 独立复核退出码必须为 0，否则记 failed
  6. 全部通过 → 记 verified
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLI = ROOT / "cli.py"
PY = sys.executable


def sh(args: list[str], cwd: Path, log: Path) -> int:
    t0 = time.perf_counter()
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=1800)
    dt = time.perf_counter() - t0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(a) for a in args)}\n[rc={r.returncode} {dt:.1f}s]\n")
        f.write(r.stdout[-8000:])
        f.write(r.stderr[-4000:])
    print(f"  {'OK ' if r.returncode == 0 else 'FAIL'} {args[1].split('/')[-1]} {dt:.1f}s")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["weekly", "monthly"])
    ap.add_argument("name")
    ap.add_argument("--new-task-id", default=None)
    a = ap.parse_args()

    base = ROOT / ("experiments" if a.kind == "weekly" else "benchmarks") / a.name
    if not base.exists():
        print(f"目录不存在: {base}")
        return 1

    res_file = base / "result.json" if a.kind == "weekly" else base / "solver_output" / "result.json"
    tid = a.new_task_id
    if not tid:
        if res_file.exists():
            tid = json.loads(res_file.read_text(encoding="utf-8")).get("task_id")
        if not tid:
            tid = f"{a.kind}-{a.name}"

    lock_kind = f"{a.kind}-run"
    start = subprocess.run([PY, str(CLI), "run", "start", lock_kind, tid, "--note", a.name],
                           capture_output=True, text=True)
    print(start.stdout.strip() or start.stderr.strip())
    if start.returncode != 0:
        return start.returncode

    log = ROOT / "runs" / f"{tid}.log"
    ok = True

    # 1) 实验
    exp = (base / "experiment.py") if a.kind == "weekly" else (base / "solver_output" / "solve.py")
    if exp.exists():
        ok &= sh([PY, str(exp)], base, log) == 0
    # 2) 独立复核
    rev = base / ("independent_review.py" if a.kind == "weekly" else "cross_review.py")
    if ok and rev.exists():
        ok &= sh([PY, str(rev)], base, log) == 0
    # 3) 月度额外：评分
    if a.kind == "monthly" and ok and (base / "evaluate.py").exists():
        ok &= sh([PY, str(base / "evaluate.py")], base, log) == 0

    state = "verified" if ok else "failed"
    msg = "run_task.py 全流程通过" if ok else f"失败，见日志 {log}"
    subprocess.run([PY, str(CLI), "run", "finish", tid, state, "--msg", msg], capture_output=True)
    print(f"{tid} -> {state}  (log: {log})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
