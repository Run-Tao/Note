"""
W46 每周模型比较 #2（提前实跑）：综合评价模型 — AHP vs 熵权法 vs TOPSIS vs PCA

实验设计（任务书 4.4 对综合评价模型的要求）：
  - 同一指标体系、同一原始数据
  - 明确权重来源（主观 AHP / 客观熵权 / 半客观组合）
  - 分析权重变化对排名的影响（权重敏感性）
  - 检查评价结果的稳健性（Spearman 秩相关 + 扰动实验）

数据说明：使用**明确标记的模拟数据**（6 个候选方案 × 5 项指标），
  因为本实验的核心是"方法之间的差异"，而非某个现实结论；
  指标方向/量纲按真实场景设定（成本、时效为负向）。不伪装为现实观测。

运行：cd ~/Note && ./.venv/bin/python "Mathematical_Modeling/System/experiments/2026-W46_evaluation_ahp_ewm/experiment.py"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = CFG["seed"]

# ----------------------------------------------------------------------------
# 数据：6 方案 × 5 指标（模拟，明确标记）
# 指标：C1 年运营成本(万元, 负向) C2 覆盖率(%, 正向) C3 响应时效(分钟, 负向)
#       C4 投资额(万元, 负向) C5 满意度(分, 正向)
# ----------------------------------------------------------------------------
ALTERNATIVES = ["A-城东枢纽", "B-城西枢纽", "C-南站扩建", "D-北区新站", "E-老城改造", "F-郊区分站"]
CRITERIA = ["C1成本", "C2覆盖率", "C3时效", "C4投资", "C5满意度"]
DIRECTION = [-1, +1, -1, -1, +1]  # +1 越大越好, -1 越小越好
X = np.array([
    [320, 78.5, 22.0, 1500, 82],   # 方案 A：可能被"补偿性"方法误判为最优
    [298, 71.0, 26.5, 1320, 76],
    [345, 83.2, 18.5, 1750, 79],
    [276, 64.8, 29.0, 1200, 74],
    [310, 76.0, 21.0, 1410, 88],
    [255, 58.0, 34.5, 980, 81],
], dtype=float)

# AHP 主观判断矩阵（1-9 标度，按 CRITERIA 顺序）
AHP_PAIRWISE = np.array([
    [1,   2,   1,   3,   2],
    [1/2, 1,   1/2, 2,   1],
    [1,   2,   1,   3,   2],
    [1/3, 1/2, 1/3, 1,   1/2],
    [1/2, 1,   1/2, 2,   1],
])


# ----------------------------------------------------------------------------
# 方法实现
# ----------------------------------------------------------------------------
def ahp_weights(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    """特征向量法求权重；返回 (w, lambda_max, CR)。RI 查表（n=5 → 1.12）。"""
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    n = A.shape[0]
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lam = float(vals.real[k])
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI[n] if RI.get(n) else 0.0
    return w, lam, CR


def entropy_weights(Xp: np.ndarray) -> np.ndarray:
    """熵权法：正向化+归一化后的数据，p_ij=x_ij/sum_i x_ij, d_j=1-e_j/ln(m)。"""
    m, n = Xp.shape
    P = Xp / Xp.sum(axis=0, keepdims=True)
    P = np.clip(P, 1e-12, None)
    e = -(P * np.log(P)).sum(axis=0) / np.log(m)
    d = 1 - e
    return d / d.sum()


def normalize(X: np.ndarray, direction) -> np.ndarray:
    """极差归一化到 [0,1]（负向指标取反），避免量纲影响。"""
    m, n = X.shape
    Z = np.zeros_like(X, dtype=float)
    for j in range(n):
        lo, hi = X[:, j].min(), X[:, j].max()
        rng = (hi - lo) or 1.0
        Z[:, j] = (X[:, j] - lo) / rng if direction[j] > 0 else (hi - X[:, j]) / rng
    return Z


def topsis(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    V = Z * w
    best, worst = V.max(axis=0), V.min(axis=0)
    dp = np.sqrt(((V - best) ** 2).sum(axis=1))
    dn = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return dn / (dp + dn)


def topsis_weights(Z, w):
    return w


def pca_weights(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA 评价：取第一主成分载荷绝对值归一化为权重；返回 (w, 第一主成分方差占比)。"""
    Zc = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    load = np.abs(Vt[0])
    share = float(S[0] ** 2 / np.sum(S ** 2))
    return load / load.sum(), share


def rank(scores: np.ndarray) -> np.ndarray:
    """降序排名（1 = 最好）；并列取平均名次。"""
    order = np.argsort(-scores, kind="mergesort")
    r = np.empty(len(scores))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(scores[order[j + 1]] - scores[order[i]]) < 1e-12:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def main() -> int:
    seed_all(SEED)
    Z = normalize(X, DIRECTION)

    w_ahp, lam, cr = ahp_weights(AHP_PAIRWISE)
    w_ewm = entropy_weights(Z)
    w_pca, pca_share = pca_weights(Z)
    w_comb = 0.5 * w_ahp + 0.5 * w_ewm   # 组合权重（主观+客观）

    methods = {
        "AHP+TOPSIS": (w_ahp, topsis(Z, w_ahp)),
        "熵权+TOPSIS": (w_ewm, topsis(Z, w_ewm)),
        "组合权重+TOPSIS": (w_comb, topsis(Z, w_comb)),
        "PCA+加权和": (w_pca, Z @ w_pca),
        "等权+TOPSIS(基准)": (np.ones(len(CRITERIA)) / len(CRITERIA),
                          topsis(Z, np.ones(len(CRITERIA)) / len(CRITERIA))),
    }
    ranks = {k: rank(v[1]) for k, v in methods.items()}

    # ---------- 稳健性：权重扰动（±20% 乘性扰动，500 次，固定种子） ----------
    rng = np.random.default_rng(SEED)
    base_rank = ranks["熵权+TOPSIS"]
    flips, top1_stability = [], {}
    for _ in range(CFG["perturbations"]):
        w = w_ewm * rng.uniform(0.8, 1.2, size=len(CRITERIA))
        w = w / w.sum()
        flips.append(float(np.mean(rank(topsis(Z, w)) != base_rank)))
        top1_stability.setdefault(ALTERNATIVES[int(np.argmax(topsis(Z, w)))], 0)
        top1_stability[ALTERNATIVES[int(np.argmax(topsis(Z, w)))]] += 1

    # ---------- 方法之间的秩相关 ----------
    keys = list(methods)
    spearman = {f"{a} vs {b}": round(float(spearmanr(ranks[a], ranks[b]).statistic), 4)
                for i, a in enumerate(keys) for b in keys[i + 1:]}

    # ---------- 一票否决式检查：最差指标 vs 综合排名（补偿性陷阱） ----------
    worst_criterion = [CRITERIA[int(np.argmin(z))] for z in Z]

    out = {
        "experiment": "2026-W46_evaluation_ahp_ewm",
        "task_id": "weekly-2026-W46",
        "data_note": "SIMULATED 6 方案 × 5 指标（明确标记为模拟，非现实观测）",
        "we_criteria": CRITERIA,
        "weights": {k: [round(float(x), 4) for x in v] for k, v in
                    [("AHP", w_ahp), ("熵权", w_ewm), ("PCA", w_pca), ("组合", w_comb)]},
        "ahp_consistency": {"lambda_max": round(lam, 4), "CR": round(cr, 4),
                            "pass": bool(cr < 0.10)},
        "pca_first_pc_share": round(pca_share, 4),
        "scores": {k: [round(float(x), 4) for x in v[1]] for k, v in methods.items()},
        "ranks": {k: [float(x) for x in v] for k, v in ranks.items()},
        "spearman_between_methods": spearman,
        "robustness": {
            "base": "熵权+TOPSIS",
            "perturbations": CFG["perturbations"],
            "mean_rank_flip_rate": round(float(np.mean(flips)), 4),
            "max_rank_flip_rate": round(float(np.max(flips)), 4),
            "top1_winner_counts": top1_stability,
        },
        "worst_criterion_per_alt": dict(zip(ALTERNATIVES, worst_criterion)),
        "alternatives": ALTERNATIVES,
    }

    # ---------- 断言：检查数学正确性与设计公平性 ----------
    checks = []

    def chk(n, c, d=""):
        checks.append({"check": n, "pass": bool(c), "detail": d})

    chk("AHP 一致性比率 CR<0.1（判断矩阵可接受）", cr < 0.10, f"CR={cr:.4f} λmax={lam:.4f}")
    chk("所有权重和为 1", all(abs(np.sum(w) - 1) < 1e-9 for w in
                          (w_ahp, w_ewm, w_pca, w_comb)))
    chk("所有分值为有限值", all(np.isfinite(v[1]).all() for v in methods.values()))
    chk("TOPSIS 分值在 (0,1)", all((v[1] > 0).all() and (v[1] < 1).all()
                                for k, v in methods.items() if "TOPSIS" in k))
    chk("排名是 1..6 的排列（允许并列取半）",
        all(sorted(r) == sorted(ranks["等权+TOPSIS(基准)"]) for r in ranks.values()))
    chk("熵权法与 AHP 权重确实不同（否则本比较无意义）",
        float(np.abs(w_ewm - w_ahp).max()) > 0.03, f"max_diff={np.abs(w_ewm-w_ahp).max():.4f}")
    chk("PCA 第一主成分解释力已记录（>0）", pca_share > 0)
    chk("扰动实验次数 ≥100", CFG["perturbations"] >= 100)

    out["checks"] = checks
    out["fingerprint"] = result_fingerprint(out)
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot(HERE, seed=SEED, config=CFG, extra={"data_note": out["data_note"]})

    print("weights:", json.dumps(out["weights"], ensure_ascii=False))
    print("CR:", out["ahp_consistency"], "PCA share:", pca_share)
    for k in methods:
        print(f"{k:<20} scores={[round(x,3) for x in methods[k][1]]} ranks={list(ranks[k])}")
    print("perturb flip rate:", out["robustness"]["mean_rank_flip_rate"],
          "top1:", out["robustness"]["top1_winner_counts"])
    print("spearman:", json.dumps(spearman, ensure_ascii=False))
    print("checks:", [c for c in checks if not c["pass"]] or "ALL PASS")
    print("fingerprint:", out["fingerprint"])
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
