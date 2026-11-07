#!/usr/bin/env python3
"""数学建模能力进化系统 — 轻量 CLI（无第三方依赖，仅标准库）。

用法：
  python3 cli.py status                          # 系统总览
  python3 cli.py run start <kind> <id> [--note]  # 登记任务开始（加锁）
  python3 cli.py run finish <id> <state> [--msg] # 登记结束（completed/verified/failed/skipped）
  python3 cli.py run list [--last N]             # 历史任务
  python3 cli.py run show <id>                   # 单任务详情
  python3 cli.py models                          # 成熟度台账
  python3 cli.py gaps                            # 知识短板台账
  python3 cli.py history experiments|benchmarks  # 历史报告
复用既有 Python 环境可直接用 .venv/bin/python 运行。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # System/
NOTE = ROOT.parents[1]                          # Note/
RUNS = ROOT / "runs"
LEDGER = RUNS / "ledger.jsonl"
LOCK = RUNS / ".lock"

VALID_STATES = ["planned", "running", "completed", "verified", "failed", "skipped"]


# ---------- 最小 YAML 读取（只支持本系统 registry 用到的子集） ----------
def load_yaml_subset(path: Path) -> dict:
    """解析 registry 里的简单 YAML（顶层 key、列表、标量、行内 [..]）。"""
    data, cur_list, cur_item = {}, None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("  #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0 and s.endswith(":"):
            key = s[:-1]
            cur_list = [] if key in ("models", "gaps", "items") else None
            data[key] = cur_list if cur_list is not None else {}
            cur_item = None
        elif s.startswith("- ") and cur_list is not None:
            cur_item = {}
            cur_list.append(cur_item)
            kv = s[2:]
            if ":" in kv:
                k, v = kv.split(":", 1)
                cur_item[k.strip()] = _conv(v.strip())
        elif cur_item is not None and ":" in s:
            k, v = s.split(":", 1)
            cur_item[k.strip()] = _conv(v.strip())
    return data


def _conv(v: str):
    if v == "" or v == "null":
        return None
    if v == "{}":
        return {}
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [] if not inner else [_conv(x) for x in inner.split(",")]
    if v.isdigit():
        return int(v)
    return v.strip('"').strip("'")


# ---------- 任务台账 ----------
def _lock_alive(kind: str) -> dict | None:
    if not LOCK.exists():
        return None
    try:
        info = json.loads(LOCK.read_text())
        pid = info.get("pid")
        if pid and not _pid_alive(pid):
            return None  # 僵尸锁
        return info
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _append(rec: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_start(kind: str, ident: str, note: str = "") -> int:
    hold = _lock_alive(kind)
    if hold:
        print(f"[LOCKED] {kind} 正由 run={hold['id']} pid={hold['pid']} 执行中，拒绝重复启动", file=sys.stderr)
        return 2
    if ident == "auto":
        ident = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    # 幂等：同 kind+id 已 verified/completed 则不重复
    for rec in _read_all():
        if rec.get("kind") == kind and rec.get("id") == ident and rec.get("state") in ("completed", "verified"):
            print(f"[SKIP] {kind}/{ident} 已完成({rec['state']})，如需重跑请换 id 或 run finish 复位", file=sys.stderr)
            return 3
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, "id": ident,
           "state": "running", "note": note, "owner_pid": os.getpid()}
    LOCK.write_text(json.dumps({"kind": kind, "id": ident, "pid": os.getpid(),
                                "at": rec["ts"]}, ensure_ascii=False))
    _append(rec)
    print(ident)
    return 0


def run_finish(ident: str, state: str, msg: str = "") -> int:
    if state not in VALID_STATES:
        print(f"非法状态 {state}", file=sys.stderr)
        return 1
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "id": ident,
           "state": state, "msg": msg, "owner_pid": os.getpid()}
    # 回填 kind：finish 记录只有 id，合并时不能把 kind 覆盖成 None
    for prev in reversed(_read_all()):
        if prev.get("id") == ident and prev.get("kind"):
            rec["kind"] = prev["kind"]
            break
    _append(rec)
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            if info.get("id") == ident:
                LOCK.unlink()
        except Exception:
            pass
    print(f"{ident} -> {state}")
    return 0


def _read_all() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def merged_runs() -> list[dict]:
    """把 record 流水合并成每个 run 的最新状态。"""
    runs: dict[str, dict] = {}
    for r in _read_all():
        k = f"{r.get('kind', '?')}/{r['id']}"
        runs.setdefault(k, {"kind": r.get("kind"), "id": r["id"], "history": []})
        runs[k]["history"].append(r)
        for f in ("state", "note", "msg", "ts", "kind"):
            if r.get(f):
                runs[k][f] = r[f]
    return sorted(runs.values(), key=lambda x: x.get("ts", ""))


# ---------- 查询 ----------
def cmd_status():
    m = load_yaml_subset(ROOT / "registry" / "models.yaml")
    g = load_yaml_subset(ROOT / "registry" / "gaps.yaml") if (ROOT / "registry" / "gaps.yaml").exists() else {}
    runs = merged_runs()
    print("== 模型成熟度 ==")
    lv = {}
    for it in m.get("models") or []:
        lv[it["maturity"]] = lv.get(it["maturity"], 0) + 1
        print(f"  L{it['maturity']} {it['id']:<6} {it['name']}")
    print(f"  合计 {sum(lv.values())} 个模型，分布 {dict(sorted(lv.items()))}")
    gaps = g.get("items") or []
    print(f"== 知识短板：{len(gaps)} 条（P0={sum(1 for x in gaps if x.get('priority')=='P0')}）")
    print(f"== 任务台账：{len(runs)} 次运行")
    for r in runs[-8:]:
        print(f"  {r.get('ts','')[:19]}  {(r.get('kind') or '?'):<16} {r['id']:<28} {r.get('state')}")


def cmd_history(which: str):
    base = ROOT / ("experiments" if which == "experiments" else "benchmarks")
    if not base.exists():
        print("（暂无）")
        return
    dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not dirs:
        print("（暂无）")
        return
    for d in dirs:
        files = [f.name for f in d.iterdir() if f.is_file()]
        print(f"{d.name}  [{', '.join(sorted(files))}]\n  -> {d}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("models")
    m.add_argument("--level", type=int)
    sub.add_parser("gaps")
    r = sub.add_parser("run")
    rs = r.add_subparsers(dest="action", required=True)
    st = rs.add_parser("start"); st.add_argument("kind"); st.add_argument("id"); st.add_argument("--note", default="")
    fi = rs.add_parser("finish"); fi.add_argument("id"); fi.add_argument("state"); fi.add_argument("--msg", default="")
    li = rs.add_parser("list"); li.add_argument("--last", type=int, default=20)
    sh = rs.add_parser("show"); sh.add_argument("id")
    h = sub.add_parser("history"); h.add_argument("which", choices=["experiments", "benchmarks"])
    a = ap.parse_args()

    if a.cmd == "status":
        cmd_status()
    elif a.cmd == "models":
        for it in load_yaml_subset(ROOT / "registry" / "models.yaml").get("models") or []:
            if a.level and it["maturity"] != a.level:
                continue
            print(f"L{it['maturity']} {it['id']:<6} {it['name']:<10} {it.get('status')}  docs={len(it.get('docs') or [])} code={len(it.get('code') or [])}")
    elif a.cmd == "gaps":
        p = ROOT / "registry" / "gaps.yaml"
        items = load_yaml_subset(p).get("items") if p.exists() else []
        if not items:
            print("（暂无短板记录）")
        for x in items:
            print(f"{x.get('priority')} {x.get('id')} {x.get('knowledge_point')} | status={x.get('status')} | {x.get('evidence')}")
    elif a.cmd == "run":
        if a.action == "start":
            return run_start(a.kind, a.id, a.note)
        if a.action == "finish":
            return run_finish(a.id, a.state, a.msg)
        if a.action == "list":
            for x in merged_runs()[-a.last:]:
                print(f"{x.get('ts','')[:19]}  {x.get('kind') or '?':<16} {x['id']:<28} {x.get('state')}  {x.get('msg','') or x.get('note','')}")
        if a.action == "show":
            for x in merged_runs():
                if x["id"] == a.id:
                    print(json.dumps(x, ensure_ascii=False, indent=2))
    elif a.cmd == "history":
        cmd_history(a.which)
    return 0


if __name__ == "__main__":
    sys.exit(main())
