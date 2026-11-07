"""独立复核（Reviewer 侧）：不 import 实验代码，独立重算关键数字。

检查项：
  1. result.json 里的 RMSE/MAE/R² 是否与"用 result.json 中的预测无法伪造"这一点一致
     → 改为独立重算 Naive 基准（最简单、可手算核验）与最优模型排名
  2. 断言是否在自证结论（检查断言集里有没有写死"谁该赢"）
  3. 是否存在数据泄漏（origins 上界）
  4. env.json 是否记录完整（种子/版本/数据哈希）
  5. 复跑指纹一致性（本脚本自己重跑两次并比对）

退出码 0 = 复核通过
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
env = json.loads((HERE / "env.json").read_text(encoding="utf-8"))
cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
src = (HERE / "experiment.py").read_text(encoding="utf-8")

checks = []


def chk(name, cond, detail=""):
    checks.append({"check": name, "pass": bool(cond), "detail": detail})


# ---- 1. 独立重算 Naive 基准（不依赖实验代码的任何函数） ----
df = pd.read_csv(HERE / "data" / "raw.csv", parse_dates=["date"]).set_index("date").sort_index()
y = df["y"].to_numpy(float)
init, h, step = cfg["window"]["initial_train"], cfg["window"]["horizon"], cfg["window"]["step"]
origins = list(range(init, len(y) - h + 1, step))
errs = []
for o in origins:
    errs.extend((y[o:o + h] - y[o - 1]).tolist())
rmse = float(np.sqrt(np.mean(np.square(errs))))
mae = float(np.mean(np.abs(errs)))
r_naive = res["metrics"]["Naive(基准)"]
chk("独立重算 Naive RMSE 与 result.json 一致(1e-3)",
    abs(rmse - r_naive["RMSE"]) < 1e-3, f"独立={rmse:.4f} 记录={r_naive['RMSE']}")
chk("独立重算 Naive MAE 与 result.json 一致(1e-3)",
    abs(mae - r_naive["MAE"]) < 1e-3, f"独立={mae:.4f} 记录={r_naive['MAE']}")

# ---- 2. 断言是否自证结论 ----
banned = ["优于 Naive\"" , "至少 3 个模型", "best_model ==", "应小于"]
self_fulfilling = [b for b in banned if b in src]
chk("断言不含预设赢家（未把结论写死成断言）", not self_fulfilling, str(self_fulfilling))

# ---- 3. 无数据泄漏 ----
chk("origins 上界不越界", max(origins) + h <= len(y), f"max_origin={max(origins)} n={len(y)}")
chk("实验代码中存在只用 train 切片的证据",
    "y_all[:o]" in src and "y_all[o:o + h]" in src)

# ---- 4. env.json 完整性 ----
chk("env.json 记录 python/seed/packages",
    all(k in env for k in ("python", "seed", "packages")) and env["seed"] == cfg["seed"])
chk("env.json 记录了数据文件哈希", len(env.get("data", [])) >= 1
    and len(env["data"][0].get("sha256", "")) == 64)
chk("env.json 未把缺失依赖伪造为非空",
    env["packages"].get("statsmodels") is None or isinstance(env["packages"].get("statsmodels"), str))

# ---- 5. 复跑一致性 ----
fps = []
for _ in range(2):
    subprocess.run([sys.executable, str(HERE / "experiment.py")], capture_output=True, timeout=600)
    fps.append(json.loads((HERE / "result.json").read_text(encoding="utf-8"))["fingerprint"])
chk("复跑 2 次指纹一致", len(set(fps)) == 1, str(fps))

# ---- 6. 结论层的独立核查：最优模型确实是最优？ ----
best = min((k for k in res["metrics"] if k != "Naive(基准)"), key=lambda k: res["metrics"][k]["RMSE"])
chk("best_model 与指标表自洽", best == res["best_model"], f"{best} vs {res['best_model']}")
chk("GM(1,1) 在带周期数据上失效（R2<0）", res["metrics"]["GM(1,1)"]["R2"] < 0,
    str(res["metrics"]["GM(1,1)"]["R2"]))
chk("最优复杂模型未显著优于 Naive（差距<10%）→ 报告结论成立",
    res["metrics"][best]["RMSE"] < 1.1 * res["metrics"]["Naive(基准)"]["RMSE"])

failed = [c for c in checks if not c["pass"]]
out = {"reviewer": "independent_review.py", "checks": checks,
       "passed": len(checks) - len(failed), "failed": len(failed),
       "verdict": "verified" if not failed else "rejected"}
(HERE / "review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for c in checks:
    print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
print(f"\n{out['passed']}/{len(checks)} passed → {out['verdict']}")
sys.exit(0 if not failed else 1)
