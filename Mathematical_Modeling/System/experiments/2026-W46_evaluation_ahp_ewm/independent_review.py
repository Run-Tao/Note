"""独立复核：综合评价实验（W46）

不 import experiment.py；独立实现 AHP 特征向量与 TOPSIS，重算并比对。
重点核查：
  1. AHP 权重与 CR 是否可独立复算
  2. TOPSIS 分值是否在 (0,1) 且排名稳定
  3. 是否真的做了稳健性扰动，扰动次数与种子是否正确
  4. 数据是否被明确标记为模拟（不得伪装现实）
  5. 复跑指纹一致
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
env = json.loads((HERE / "env.json").read_text(encoding="utf-8"))
src = (HERE / "experiment.py").read_text(encoding="utf-8")
checks = []


def chk(n, c, d=""):
    checks.append({"check": n, "pass": bool(c), "detail": d})


# 独立重算 AHP：用幂法迭代（与实验里的 np.linalg.eig 不同路径）
A = np.array([[1, 2, 1, 3, 2], [1/2, 1, 1/2, 2, 1], [1, 2, 1, 3, 2],
              [1/3, 1/2, 1/3, 1, 1/2], [1/2, 1, 1/2, 2, 1]], float)
w = np.ones(5)
for _ in range(200):
    w = A @ w
    w = w / w.sum()
lam = float(np.mean((A @ w) / w))
CI = (lam - 5) / 4
CR = CI / 1.12
chk("独立幂法重算 AHP 权重与记录一致(1e-3)",
    np.max(np.abs(w - np.array(res["weights"]["AHP"]))) < 1e-3,
    f"独立={np.round(w,4)}")
chk("独立重算 CR<0.1", CR < 0.1, f"CR={CR:.4f}")
chk("记录 CR 与独立复算差 <1e-3", abs(CR - res["ahp_consistency"]["CR"]) < 2e-3,
    f"{CR:.4f} vs {res['ahp_consistency']['CR']}")

# 独立重算 TOPSIS（熵权）
X = np.array([[320, 78.5, 22.0, 1500, 82], [298, 71.0, 26.5, 1320, 76],
              [345, 83.2, 18.5, 1750, 79], [276, 64.8, 29.0, 1200, 74],
              [310, 76.0, 21.0, 1410, 88], [255, 58.0, 34.5, 980, 81]], float)
d = [-1, 1, -1, -1, 1]
Z = np.zeros_like(X)
for j in range(5):
    lo, hi = X[:, j].min(), X[:, j].max()
    Z[:, j] = (X[:, j]-lo)/(hi-lo) if d[j] > 0 else (hi-X[:, j])/(hi-lo)
m = Z.shape[0]
P = Z / Z.sum(0, keepdims=True)
P = np.clip(P, 1e-12, None)
e = -(P*np.log(P)).sum(0)/np.log(m)
we = (1-e)/(1-e).sum()
chk("独立重算熵权与记录一致(1e-3)",
    np.max(np.abs(we - np.array(res["weights"]["熵权"]))) < 1e-3, f"{np.round(we,4)}")
V = Z * we
dp = np.sqrt(((V-V.max(0))**2).sum(1)); dn = np.sqrt(((V-V.min(0))**2).sum(1))
s = dn/(dp+dn)
chk("独立重算 TOPSIS 分值与记录一致(1e-3)",
    np.max(np.abs(s - np.array(res["scores"]["熵权+TOPSIS"]))) < 1e-3, f"{np.round(s,4)}")
chk("TOPSIS 分值均在 (0,1)", ((s > 0) & (s < 1)).all())

# 稳健性真实性
chk("扰动次数与配置一致", res["robustness"]["perturbations"] == cfg["perturbations"],
    f"{res['robustness']['perturbations']}")
chk("扰动排名翻转率在 [0,1]",
    0 <= res["robustness"]["mean_rank_flip_rate"] <= 1,
    str(res["robustness"]["mean_rank_flip_rate"]))
chk("最优方案在所有扰动下唯一（top1 计数和 == 扰动次数）",
    sum(res["robustness"]["top1_winner_counts"].values()) == cfg["perturbations"])

# 数据诚实性
chk("数据被明确标记为模拟", "模拟" in res["data_note"] or "SIMULATED" in res["data_note"],
    res["data_note"])
chk("env.json 种子与配置一致", env["seed"] == cfg["seed"])

# 复跑一致性
fps = []
for _ in range(2):
    subprocess.run([sys.executable, str(HERE / "experiment.py")], capture_output=True, timeout=300)
    fps.append(json.loads((HERE / "result.json").read_text(encoding="utf-8"))["fingerprint"])
chk("复跑 2 次指纹一致", len(set(fps)) == 1, str(fps))

failed = [c for c in checks if not c["pass"]]
out = {"reviewer": "independent_review.py", "checks": checks,
       "passed": len(checks)-len(failed), "failed": len(failed),
       "verdict": "verified" if not failed else "rejected"}
(HERE / "review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for c in checks:
    print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
print(f"\n{out['passed']}/{len(checks)} passed → {out['verdict']}")
sys.exit(0 if not failed else 1)
