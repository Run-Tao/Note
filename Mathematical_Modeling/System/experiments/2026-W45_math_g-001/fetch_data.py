#!/usr/bin/env python3
"""真实数据获取：多标的日频收益率（用于相关矩阵研究）

数据源优先级（全部为公开可访问的静态 CSV，不依赖登录/不被限流）：
  1. GitHub raw：datasets/s-and-p-500（Shiller 月度，1871-）→ 用月度收益构造低相关资产
  2. 若可用：其他公开 GitHub 数据集

★ 诚实声明：Yahoo Finance / Stooq 在本机被限流或需 JS 验证（实测 HTTP 429 / 挑战页），
   因此本实验**使用月度而非日频**数据。这不影响核心结论（相关矩阵可实现性与频率无关），
   但会改变具体数值，报告中必须写明。

产出：data/raw.csv（宽表：date + 各资产收益率）
      data/META.json（来源、行数、时间范围、sha256）
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0"}


def fetch() -> str:
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> int:
    raw = fetch()
    sha = hashlib.sha256(raw.encode()).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw)))

    # 构造 4 个"资产"：股票价格 / 股息 / 盈利 / 长期利率
    # 均为同一市场不同维度 —— 天然高度相关，正好是相关矩阵最容易非正定的场景
    series = {"date": [], "price": [], "dividend": [], "earnings": [], "rate": []}
    prev = {}
    for r in rows:
        try:
            d = r["Date"]
            vals = {"price": float(r["SP500"]), "dividend": float(r["Dividend"]),
                    "earnings": float(r["Earnings"]), "rate": float(r["Long Interest Rate"])}
        except (ValueError, KeyError):
            continue
        # 收益率（股息/盈利有 0 值，用 log1p 差分避免 -inf）
        rets = {}
        ok = True
        for k, v in vals.items():
            if v <= 0:
                ok = False
                break
            p = prev.get(k)
            rets[k] = None if p is None else (v / p - 1.0)
            prev[k] = v
        if not ok or any(v is None for v in rets.values()):
            continue
        series["date"].append(d)
        for k in rets:
            series[k].append(rets[k])

    n = len(series["date"])
    if n < 240:
        print(f"数据不足: {n} 行")
        return 1

    out = DATA / "raw.csv"
    keys = ["price", "dividend", "earnings", "rate"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date"] + keys)
        for i in range(n):
            w.writerow([series["date"][i]] + [f"{series[k][i]:.10f}" for k in keys])

    meta = {
        "source": URL,
        "source_note": "Yahoo Finance/Stooq 在本机被限流(HTTP 429)/需 JS 挑战，故改用公开 GitHub 数据集",
        "frequency": "monthly（非日频，报告中须说明）",
        "assets": keys,
        "rows": n,
        "date_range": [series["date"][0], series["date"][-1]],
        "raw_sha256": sha,
        "csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "type": "real",
    }
    (DATA / "META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK rows={n} range={meta['date_range'][0]}~{meta['date_range'][1]} sha={meta['csv_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
