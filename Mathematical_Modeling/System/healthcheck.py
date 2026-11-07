#!/usr/bin/env python3
"""系统健康检查（System Health Check）

检查项：
  1. 目录/文件完整性（registry、runs、experiments、benchmarks）
  2. 任务台账：卡在 running 超过 N 小时的僵尸任务
  3. verified 任务是否有对应证据文件（report/review/result）
  4. 未完成/失败任务统计
  5. 磁盘/内存
  6. 数据诚实性抽查：result.json 中是否有未标记来源的模拟数据

输出：reports/health_<date>.json + 控制台摘要
退出码 0 = 健康, 1 = 有告警
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTE = ROOT.parents[1]


def load_ledger():
    p = ROOT / "runs" / "ledger.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def merged(rows):
    runs = {}
    for r in rows:
        k = r.get("id")
        runs.setdefault(k, {}).update({f: r[f] for f in ("kind", "state", "note", "msg", "ts") if r.get(f)})
    return runs


def main() -> int:
    warns, info = [], []
    # 1 目录
    for d in ("lib", "registry", "experiments", "benchmarks", "runs", "reports"):
        if not (ROOT / d).exists():
            warns.append(f"缺少目录 {d}")
    for f in ("registry/models.yaml", "registry/gaps.yaml"):
        if not (ROOT / f).exists():
            warns.append(f"缺少台账 {f}")

    # 2/3/4 台账
    rows = load_ledger()
    runs = merged(rows)
    now = time.time()
    for tid, r in runs.items():
        if r.get("state") == "running":
            ts = r.get("ts", "")
            try:
                t = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
                if now - t > 3 * 3600:
                    warns.append(f"疑似僵尸任务 {tid}（running 超过 3 小时）")
            except Exception:
                warns.append(f"任务 {tid} 时间戳无法解析: {ts}")
        if r.get("state") == "failed":
            warns.append(f"失败任务 {tid}: {r.get('msg', '')[:80]}")
    verified = [t for t, r in runs.items() if r.get("state") == "verified"]

    # 4b 实验/测试证据完整性（骨架目录 status=pending_review 不计入）
    for base, need in (("experiments", ["result.json", "report.md", "review.json", "env.json"]),
                       ("benchmarks", ["score.json", "review.json"])):
        d = ROOT / base
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            cfg = sub / "config.json"
            skeleton = False
            if cfg.exists():
                try:
                    skeleton = json.loads(cfg.read_text(encoding="utf-8")).get("status") == "pending_review"
                except Exception:
                    pass
            if skeleton:
                info.append(f"{base}/{sub.name} 骨架待补全（跳过证据检查）")
                continue
            have = {f.name for f in sub.iterdir() if f.is_file()}
            so = sub / "solver_output"
            if so.exists():
                have |= {f.name for f in so.iterdir() if f.is_file()}
            missing = [n for n in need if n not in have]
            if missing:
                warns.append(f"{base}/{sub.name} 缺少 {missing}")
            else:
                info.append(f"{base}/{sub.name} 证据齐备")

    # 5 资源
    total, used, free = shutil.disk_usage("/")
    disk_pct = used / total * 100
    if disk_pct > 85:
        warns.append(f"磁盘使用率 {disk_pct:.1f}%")
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":")[0], line.split()[1]
            mem[k] = int(v)
        avail_pct = mem["MemAvailable"] / mem["MemTotal"] * 100
        if avail_pct < 10:
            warns.append(f"可用内存仅 {avail_pct:.1f}%")
    except Exception:
        avail_pct = None

    # 6 数据诚实性抽查
    for base in ("experiments", "benchmarks"):
        for rj in (ROOT / base).rglob("result.json"):
            try:
                d = json.loads(rj.read_text())
                note = json.dumps(d.get("data", {}), ensure_ascii=False) + str(d.get("data_note", ""))
                low = note.lower()
                if any(t in low for t in ("模拟", "simulated", "synthetic", "mock")):
                    info.append(f"{rj.parent.name}: 数据来源已标注（模拟/合成）")
                elif "real" in low:
                    info.append(f"{rj.parent.name}: 数据来源已标注（真实）")
                else:
                    warns.append(f"{rj.parent.name}: 数据来源标注不清")
            except Exception:
                pass

    out = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs_total": len(runs), "verified": len(verified),
        "running": sum(1 for r in runs.values() if r.get("state") == "running"),
        "failed": sum(1 for r in runs.values() if r.get("state") == "failed"),
        "disk_used_pct": round(disk_pct, 1),
        "mem_available_pct": None if avail_pct is None else round(avail_pct, 1),
        "warnings": warns, "info": info,
        "status": "healthy" if not warns else "warning",
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"health_{time.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "runs_total", "verified", "running", "failed",
                                          "disk_used_pct", "mem_available_pct")}, ensure_ascii=False))
    for w in warns:
        print("WARN:", w)
    return 0 if not warns else 1


if __name__ == "__main__":
    sys.exit(main())
