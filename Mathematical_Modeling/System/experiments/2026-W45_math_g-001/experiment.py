#!/usr/bin/env python3
"""2026-W45 每周比较：相关矩阵的正定性与可实现性(Cholesky 可行性 / 最近正定投影 / rho 可行上限)

由 generate_weekly.py 于 2026-11-07 21:35 自动生成骨架，攻击短板 G-001（相关矩阵的正定性与可实现性(Cholesky 可行性 / 最近正定投影 / rho 可行上限)）。
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
