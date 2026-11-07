#!/usr/bin/env python3
"""每周选题自动生成（Phase 6 决策：由 Agent 从短板台账生成骨架，交人工确认）

流程：
  1. 读 registry/gaps.yaml + registry/models.yaml + 已有 experiments/*/result.json
  2. 选出一个"本期要攻的短板"：优先级排序 P0>P1>P2>P3，同级选最久未碰的知识域
  3. 选出一个"要横向比较的模型族"：成熟度 L2/L3 且近 3 期未出现的
  4. 生成本周实验目录骨架：config.json + data/README.md + experiment.py（TODO 桩）
  5. 写 `pipeline/pending.yaml`，run_task 只执行 status=pending 的目录
  6. **不自动执行**：需人工（或下一轮心跳）确认后把 status 改成 approved

用法：
  ./.venv/bin/python generate_weekly.py            # 生成下一个 pending
  ./.venv/bin/python generate_weekly.py --list     # 只看 pending / approved
  ./.venv/bin/python generate_weekly.py --approve <dir>   # 批准
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAPS = ROOT / "registry" / "gaps.yaml"
MODELS = ROOT / "registry" / "models.yaml"
EXP = ROOT / "experiments"
PIPE = ROOT / "pipeline" / "pending.yaml"


# ---------- 极简 YAML 读取（避免依赖 pyyaml） ----------
def read_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^-\s+(\w+):\s*(.*)$", line.strip())
        if m and not line.startswith("    "):
            if cur:
                items.append(cur)
            cur = {m.group(1): m.group(2).strip('"')}
        elif cur is not None:
            m2 = re.match(r"^(\w+):\s*(.*)$", s)
            if m2:
                cur[m2.group(1)] = m2.group(2).strip('"')
    if cur:
        items.append(cur)
    return items


def iso_week(d: datetime) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def already_used(domain: str) -> bool:
    """该知识域是否已有实验目录"""
    for d in EXP.iterdir() if EXP.exists() else []:
        if d.is_dir() and domain.lower() in d.name.lower():
            return True
    return False


def pick_gap(gaps: list[dict]) -> dict | None:
    prio = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    cand = [g for g in gaps if g.get("status") in ("open", "in_progress")]
    if not cand:
        return None
    cand.sort(key=lambda g: (prio.get(g.get("priority", "P3"), 9), g.get("found", "")))
    for g in cand:
        # 已被实验覆盖过的知识域跳过，避免重复
        if g.get("category") and already_used(g["category"]):
            continue
        return g
    return cand[0]


def pick_models(models: list[dict]) -> list[str]:
    recent = set()
    for d in sorted(EXP.iterdir()) if EXP.exists() else []:
        if d.is_dir():
            rj = d / "result.json"
            if rj.exists():
                try:
                    recent |= set(json.loads(rj.read_text()).get("models_tested", []))
                except Exception:
                    pass
    pool = [m["id"] for m in models if str(m.get("maturity")) in ("2", "3")]
    fresh = [m for m in pool if m not in recent]
    return (fresh or pool)[:3]


TEMPLATE = '''#!/usr/bin/env python3
"""{title}

由 generate_weekly.py 于 {ts} 自动生成骨架，攻击短板 {gap_id}（{gap_point}）。
★ 必须人工/Agent 补全后再运行。generate 阶段只保证"可运行"，不保证"有意义"。

纪律（System/README.md §3）：
  - 数据必须有明确来源；模拟数据必须写 data_note
  - 断言只检验设计公平性，不得预设"谁该赢"
  - 任何论文/报告里的数字必须来自本次 result.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lib.env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def build_data():
    """TODO: 构造或读取数据。真实数据优先；模拟数据必须注明。"""
    raise NotImplementedError("请先确定数据来源")


def run_experiment(data):
    """TODO: 完成横向比较实验，返回 result dict。"""
    raise NotImplementedError("请先设计实验")


def main() -> int:
    seed_all(CONFIG["experiment"]["seed"])
    data = build_data()
    result = run_experiment(data)
    result["task_id"] = CONFIG["task_id"]
    result["config"] = CONFIG["experiment"]
    result["env"] = snapshot(seed=CONFIG["experiment"]["seed"], data=data)
    result["fingerprint"] = result_fingerprint(result)
    (HERE / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (HERE / "env.json").write_text(
        json.dumps(result["env"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("fingerprint:", result["fingerprint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--approve", default=None)
    a = ap.parse_args()

    PIPE.parent.mkdir(parents=True, exist_ok=True)

    if a.approve:
        target = a.approve
        rows = read_items(PIPE)
        hit = False
        for r in rows:
            if r.get("dir") == target:
                r["status"] = "approved"
                r["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                hit = True
        if not hit:
            print(f"pipeline 中没有 {target}")
            return 1
        out = "version: 1\nupdated: %s\nitems:\n" % time.strftime("%Y-%m-%d")
        for r in rows:
            out += "  - dir: %s\n    status: %s\n    gap: %s\n    created: %s\n    approved_at: %s\n" % (
                r["dir"], r["status"], r.get("gap", ""), r.get("created", ""), r.get("approved_at", ""))
        PIPE.write_text(out, encoding="utf-8")
        print(f"approved: {target}")
        return 0

    rows = read_items(PIPE)
    if a.list:
        if not rows:
            print("pipeline 为空")
        for r in rows:
            print(f"{r['status']:<10} {r['dir']:<45} gap={r.get('gap')}")
        return 0

    # 幂等：已有 pending 就不再生
    pend = [r for r in rows if r.get("status") == "pending"]
    if pend:
        print(f"已有待确认骨架: {pend[0]['dir']}（已跳过生成）")
        return 0

    gap = pick_gap(read_items(GAPS))
    models = pick_models(read_items(MODELS))
    if not gap:
        print("短板台账为空，无需生成")
        return 0

    wk = iso_week(datetime.now())
    domain = (gap.get("category") or "misc").strip()
    name = f"{wk}_{domain}_{gap['id'].lower()}"
    d = EXP / name
    if d.exists():
        print(f"目录已存在: {d}")
        return 1
    d.mkdir(parents=True)

    cfg = {
        "task_id": f"weekly-{wk}-{gap['id'].lower()}",
        "title": f"{wk} 每周比较：{gap.get('knowledge_point', domain)}",
        "created_by": "generate_weekly.py",
        "target_gap": gap.get("id"),
        "gap_knowledge_point": gap.get("knowledge_point"),
        "experiment": {
            "seed": 20261107,
            "models_to_compare": models,
            "metrics": ["TODO"],
            "notes": "骨架，需人工补全 data/metrics/断言",
        },
        "status": "pending_review",
    }
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "experiment.py").write_text(TEMPLATE.format(
        title=cfg["title"], ts=time.strftime("%Y-%m-%d %H:%M"), gap_id=gap.get("id"),
        gap_point=gap.get("knowledge_point", "")), encoding="utf-8")
    (d / "data").mkdir(exist_ok=True)
    (d / "data" / "README.md").write_text(
        "# 数据说明（必须填写）\n\n- 来源：\n- 是否为模拟数据：\n- 若模拟，参数如何设定：\n- sha256（由 env_record.snapshot 自动记录）\n",
        encoding="utf-8")
    (d / "TODO.md").write_text(
        f"# 本周实验待办（自动生成）\n\n攻击短板：**{gap.get('id')}** {gap.get('knowledge_point')}\n\n"
        f"- [ ] 确定数据来源（真实优先）\n- [ ] 实现 experiment.py 的 build_data / run_experiment\n"
        f"- [ ] 写 independent_review.py（不同实现路径复核）\n- [ ] 断言只检验设计公平性\n"
        f"- [ ] 跑 run_task.py weekly {name}\n", encoding="utf-8")

    rows.append({"dir": name, "status": "pending", "gap": gap.get("id", ""),
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "approved_at": ""})
    out = "version: 1\nupdated: %s\nitems:\n" % time.strftime("%Y-%m-%d")
    for r in rows:
        out += "  - dir: %s\n    status: %s\n    gap: %s\n    created: %s\n    approved_at: %s\n" % (
            r["dir"], r["status"], r.get("gap", ""), r.get("created", ""), r.get("approved_at", ""))
    PIPE.write_text(out, encoding="utf-8")
    print(f"生成骨架: {name}\n  攻击短板 {gap.get('id')}: {gap.get('knowledge_point')}\n"
          f"  待比较模型: {models}\n  确认后执行: cli/generate_weekly.py --approve {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
