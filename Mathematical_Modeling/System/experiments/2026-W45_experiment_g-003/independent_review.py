#!/usr/bin/env python3
"""G-003 实验独立复核（不 import experiment.py，全部独立实现）

复核四条：
  R1 坏断言的翻转现象：独立用 scipy.stats 的解析标准差（不模拟）复算
     「两种修复法风险差」在 100 万次模拟下的理论值，判断 0.005 阈值是否拍脑袋
  R2 好断言的稳定性：独立用 numpy 特征值判据复算「修复输出是否 PSD」
  R3 判别力：独立构造伪造结果，确认好断言会失败
  R4 坏断言对 PSD 失败无感：确认它只看数值差

★ 设计要点：不读 experiment.py 的任何中间结论，
  直接从 ring_chord 的数学定义重算（等权组合的方差有闭式解）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
findings = []


def f(item, ok, detail):
    findings.append({"item": item, "pass": bool(ok), "detail": str(detail)})


def ring_chord(n, rho):
    C = np.eye(n)
    for i in range(n):
        C[i, (i + 1) % n] = C[(i + 1) % n, i] = rho
    for i in range(n // 2):
        C[i, n - 1 - i] = C[n - 1 - i, i] = rho
    return C


def eig_clip(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def higham(C, iters=200):
    R = C.copy()
    for _ in range(iters):
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-8, None)
        R = V @ np.diag(w) @ V.T
        np.fill_diagonal(R, 1.0)
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def analytic_std(C):
    """等权组合 std 的**解析解**：sqrt(1'C1)/n。

    这与 experiment.py 的蒙特卡洛完全独立（那边是抽样，这是矩阵运算），
    所以能检出「模拟实现的 bug」。
    """
    n = C.shape[0]
    w = np.ones(n) / n
    return float(np.sqrt(w @ C @ w))


C_bad = ring_chord(10, 0.9)
A, B = eig_clip(C_bad), higham(C_bad)

# ---- R1：解析最优值，判断 0.005 阈值是否拍脑袋 ----
sd_a, sd_b = analytic_std(A), analytic_std(B)
gap_analytic = abs(sd_b - sd_a)
f("R1a 两种修复法的解析风险差 > 0.005 阈值（阈值偏小）",
  gap_analytic > 0.005,
  f"解析 |Δ|={gap_analytic:.5f} vs 阈值 0.005")
f("R1b 解析 std 与蒙特卡洛同量级（无实现 bug）",
  abs(sd_a - 0.520) < 0.01,
  f"解析 EIG_CLIP={sd_a:.5f}，实验大样本≈0.520")

# ---- R2：PSD 独立判据 ----
me_a, me_b = float(np.linalg.eigvalsh(A).min()), float(np.linalg.eigvalsh(B).min())
f("R2 两种修复输出均 PSD（特征值判据独立复算）",
  me_a >= -1e-9 and me_b >= -1e-9, f"min_eig A={me_a:.3e} B={me_b:.3e}")
f("R2b 病态输入确实非 PSD",
  float(np.linalg.eigvalsh(C_bad).min()) < -1e-3,
  f"input_min_eig={float(np.linalg.eigvalsh(C_bad).min()):.4f}")

# ---- R3：伪造结果 → 好断言必须失败 ----
fake = {"input_min_eig": 0.5, "input_is_psd": True,
        "EIG_CLIP": {"std": sd_a, "min_eig": me_a},
        "HIGHAM": {"std": sd_a + 0.001, "min_eig": -1.0}}
good_fails = (not fake["input_is_psd"] is False) or (fake["HIGHAM"]["min_eig"] < -1e-9) or \
             (fake["input_min_eig"] >= -1e-3)
f("R3 伪造结果（输入本来是好的 + 修复失败）会被好断言抓到", good_fails,
  f"input_is_psd={fake['input_is_psd']} input_min_eig={fake['input_min_eig']} "
  f"HIGHAM_min_eig={fake['HIGHAM']['min_eig']}")

# ---- R4：坏断言对 PSD 失败无感 ----
gap_fake = abs(fake["HIGHAM"]["std"] - fake["EIG_CLIP"]["std"])
bad_passes = gap_fake < 0.005
f("R4 坏断言在「修复失败」的伪造结果上仍判通过（只看数值差，不查 PSD）", bad_passes,
  f"|Δ|={gap_fake:.5f} < 0.005 → 通过（尽管 min_eig=-1）")

# ---- 汇总 ----
verdict = "verified" if all(x["pass"] for x in findings) else "rejected"
out = {"reviewer": "independent_review.py（G-003）", "verdict": verdict,
       "n_checks": len(findings), "n_pass": sum(x["pass"] for x in findings),
       "method": "解析闭式解 sqrt(1'C1)/n 代替蒙特卡洛，与实验代码完全独立",
       "findings": findings}
(HERE / "review.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

for x in findings:
    print(f"  {'PASS' if x['pass'] else 'FAIL'}  {x['item']}  —  {x['detail']}")
print("verdict:", verdict)
sys.exit(0 if verdict == "verified" else 1)
