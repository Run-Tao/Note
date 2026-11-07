#!/usr/bin/env python3
"""pipeline 执行器：只执行 status=approved 的实验目录。

设计意图（Phase 6）：
  generate_weekly.py  生成骨架 → status=pending  （需人工确认）
  generate_weekly.py --approve <dir> → status=approved
  run_pipeline.py     遍历 approved → run_task.py weekly → 成功则标 done

cron 每周一 06:10 先 generate（幂等，已有 pending 则跳过），再跑本脚本。
→ 效果：**没有人工批准就什么也不执行**，但每周会准备好下一个待确认的骨架。

用法：
  ./.venv/bin/python run_pipeline.py
  ./.venv/bin/python run_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline" / "pending.yaml"
PY = sys.executable


def read_rows() -> list[dict]:
    if not PIPE.exists():
        return []
    rows, cur = [], None
    for line in PIPE.read_text(encoding="utf-8").splitlines():
        if line.startswith("  - "):
            if cur:
                rows.append(cur)
            cur = {}
            m = re.match(r"^  -\s+(\w+):\s*(.*)$", line)
            if m:
                cur[m.group(1)] = m.group(2).strip()
        elif cur is not None:
            m = re.match(r"^\s+(\w+):\s*(.*)$", line)
            if m:
                cur[m.group(1)] = m.group(2).strip()
    if cur:
        rows.append(cur)
    return rows


def write_rows(rows: list[dict]) -> None:
    out = "version: 1\nupdated: %s\nitems:\n" % time.strftime("%Y-%m-%d")
    for r in rows:
        out += "  - dir: %s\n    status: %s\n    gap: %s\n    created: %s\n    approved_at: %s\n" % (
            r.get("dir", ""), r.get("status", ""), r.get("gap", ""),
            r.get("created", ""), r.get("approved_at", ""))
    PIPE.parent.mkdir(parents=True, exist_ok=True)
    PIPE.write_text(out, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = read_rows()
    todo = [r for r in rows if r.get("status") == "approved"]
    if not todo:
        print("pipeline: 没有已批准的实验（pending 需 generate_weekly.py --approve）")
        return 0
    dry = "  [dry-run]" if a.dry_run else ""
    print(f"pipeline: {len(todo)} 个已批准实验{dry}")
    rc_all = 0
    for r in todo:
        d = r["dir"]
        # 骨架未补全（experiment.py 仍抛 NotImplementedError）→ 不执行，保持 approved
        ex = ROOT / "experiments" / d / "experiment.py"
        if ex.exists() and "NotImplementedError" in ex.read_text(encoding="utf-8"):
            print(f"  跳过 {d}（骨架未补全）")
            continue
        if a.dry_run:
            print(f"  将执行 {d}")
            continue
        rc = subprocess.run([PY, str(ROOT / "run_task.py"), "weekly", d],
                            cwd=str(ROOT)).returncode
        r["status"] = "done" if rc == 0 else "failed"
        rc_all |= rc
        print(f"  {d} -> {r['status']}")
    if not a.dry_run:
        write_rows(rows)
    return 0 if rc_all == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
