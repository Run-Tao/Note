# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L4 评价与决策模型（熵权法 + TOPSIS）/ L10 模型检验
anchor: Day54 —— 熵权法 + TOPSIS：县域供热"拆小炉、并大网"改造方案排序
goal: 把"五个方案、六个指标、每格一个数"的会议室争论，变成一句能签字的话：
      选哪个、为什么、我这份排序有多大把握、以及哪个方案看着打折其实该做。

场景：云溪县城关镇有 5 座散烧小锅炉（冬季重污染、供热不稳），2026 年供热季前
      要从 5 个改造方案里挑 1 个报省里评审。6 个指标：
        ① 覆盖供暖面积（万 m2，效益型） ② 机组装机容量（MW，效益型）
        ③ 单位面积投资（元/m2，成本型）④ 环保达标率（%，效益型）
        ⑤ 建设工期（月，成本型）        ⑥ 单位面积运维费（元/m2·季，成本型）
      （③⑥ 已做过面积归一，避免"大方案天然占优"——这是多指标评价最常见的陷阱）

本脚本做四件事（全部手写，不依赖 scipy 以外的库）：
  1. 极差标准化（成本型方向翻转）+ 熵权法（Shannon 熵）定权重
  2. TOPSIS 贴近度 + 排序；用加权空间的正负理想解拆开"离理想近"还是"离最差远"
  3. 三套口径互证：等权 / CRITIC / AHP（含 CR 一致性检验）
  4. 稳健性：逐指标 ± 扰动 + 乘性噪声蒙特卡洛 + 成对"翻盘阈值 sigma"

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-10-24/topsis_entropy_heating.py"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体（Noto Sans CJK / WenQuanYi 回退链） ----------
_available = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "WenQuanYi Micro Hei", "Source Han Sans SC"):
    if _cand in _available:
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
SEED = 20261024

# ======================================================================
# 一、决策表（原始量纲）
# ======================================================================
SITES = ["A 检修中心改造", "B 新区分布式燃气", "C 租用调峰锅炉房",
         "D 城郊煤改气并网", "E 园区热电联产"]
SHORT = ["A 检修中心", "B 新区分布式", "C 租用调峰", "D 城郊煤改气", "E 园区联产"]

CRITERIA = [
    ("覆盖供暖面积", +1, "万m2"),
    ("装机容量", +1, "MW"),
    ("单位面积投资", -1, "元/m2"),
    ("环保达标率", +1, "%"),
    ("建设工期", -1, "月"),
    ("单位运维费", -1, "元/m2季"),
]
NAMES = [c[0] for c in CRITERIA]
DIRECTIONS = np.array([c[1] for c in CRITERIA])

X = np.array([
    [ 960.0, 118.0, 3.4, 90.0, 14.0, 2.8],   # A
    [ 620.0,  52.0, 5.8, 96.0,  6.0, 3.6],   # B
    [1240.0, 160.0, 3.1, 80.0, 24.0, 2.2],   # C
    [ 860.0,  95.0, 2.6, 88.0, 18.0, 2.5],   # D
    [1080.0, 135.0, 4.3, 92.0, 26.0, 2.4],   # E
], dtype=float)


# ======================================================================
# 二、极差标准化
# ======================================================================
def normalize(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """极差标准化到 [0,1]；成本型翻转。零极差列 = 无分辨力，直接报错。"""
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    if np.any(rng <= 0):
        bad = [NAMES[j] for j in range(len(rng)) if rng[j] <= 0]
        raise ValueError(f"零极差指标（整列相同，无分辨力）：{bad}")
    Z = (X - lo) / rng
    cost = directions < 0
    Z[:, cost] = 1.0 - Z[:, cost]
    return Z


# ======================================================================
# 三、权重：熵权 / CRITIC / AHP
# ======================================================================
def entropy_weights(Z: np.ndarray):
    n, _ = Z.shape
    P = Z + 1e-12               # 平移避 log0；e 为 0 的指标贡献 d=1（最大权重）
    P = P / P.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(n)
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1.0 - e
    return d / d.sum(), e, d


def critic_weights(Z: np.ndarray) -> np.ndarray:
    sd = Z.std(axis=0, ddof=1)          # 对比强度
    R = np.corrcoef(Z, rowvar=False)
    info = sd * (1.0 - R).sum(axis=0)   # × 冲突性（与其他指标不相关则信息量高）
    return info / info.sum()


def ahp_weights(A: np.ndarray):
    vals, vecs = np.linalg.eig(A)
    kmax = int(vals.real.argmax())
    lam = float(vals.real[kmax])
    w = np.abs(vecs[:, kmax].real)
    w = w / w.sum()
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}[n]
    return w, lam, CI, (CI / RI if RI > 0 else 0.0)


# ======================================================================
# 四、TOPSIS
# ======================================================================
def topsis(Z: np.ndarray, w: np.ndarray):
    V = Z * w
    best, worst = V.max(axis=0), V.min(axis=0)
    dp = np.sqrt(((V - best) ** 2).sum(axis=1))
    dm = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return dm / (dp + dm), dp, dm


def order_of(C: np.ndarray) -> list[int]:
    return [int(i) for i in np.argsort(-C, kind="stable")]


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n, s = len(a), 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def flip_sigma(Z, w, i_top, i_challenger, target=0.05, reps=400, seed=SEED):
    """乘性扰动 w·exp(sigma·xi) 再归一，找让 i_challenger 越过 i_top 的频率首达 target 的 sigma。"""
    rng = np.random.default_rng(seed)
    for sig in np.arange(0.0, 1.001, 0.05):
        cnt = 0
        for _ in range(reps):
            ww = w * np.exp(sig * rng.standard_normal(w.size))
            ww /= ww.sum()
            c, _, _ = topsis(Z, ww)
            cnt += c[i_challenger] > c[i_top]
        if cnt / reps >= target:
            return float(sig), cnt / reps
    return float("inf"), 1.0


# ======================================================================
# 主流程
# ======================================================================
def main() -> None:
    n, m = X.shape
    Z = normalize(X, DIRECTIONS)
    w_e, e, d = entropy_weights(Z)
    w_c = critic_weights(Z)

    # AHP：决策者口径 = 居民冷暖/环保优先，投资次之，工期最后
    A = np.array([
        [1,   2,   3,   1,   4,   2],    # 供暖面积
        [1/2, 1,   2,   1/2, 3,   1],    # 装机容量
        [1/3, 1/2, 1,   1/2, 2,   1],    # 单位投资
        [1,   2,   2,   1,   3,   2],    # 环保达标率
        [1/4, 1/3, 1/2, 1/3, 1,   1/2],  # 工期
        [1/2, 1,   1,   1/2, 2,   1],    # 单位运维费
    ], dtype=float)
    w_a, lam, CI, CR = ahp_weights(A)

    C_e, dp, dm = topsis(Z, w_e)
    C_c, _, _ = topsis(Z, w_c)
    C_a, _, _ = topsis(Z, w_a)
    C_q, _, _ = topsis(Z, np.full(m, 1.0 / m))

    o_e, o_c, o_a, o_q = order_of(C_e), order_of(C_c), order_of(C_a), order_of(C_q)

    def pr(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    pr("一、标准化矩阵 Z（[0,1]，成本型已翻转：1 = 最好，0 = 最差）")
    print(f"{'方案':<16}" + "".join(f"{nm:>9}" for nm in NAMES))
    for i, s in enumerate(SITES):
        print(f"{s:<16}" + "".join(f"{v:>9.3f}" for v in Z[i]))

    pr("二、熵权法：谁的『信息量』大谁说话")
    print(f"{'指标':<10}{'原始极差':>10}{'Z 极差':>9}{'熵值 e':>10}{'差异 d':>10}{'权重 w':>10}")
    raw_rng = X.max(axis=0) - X.min(axis=0)
    for j, nm in enumerate(NAMES):
        print(f"{nm:<10}{raw_rng[j]:>10.1f}{Z[:, j].max()-Z[:, j].min():>9.3f}"
              f"{e[j]:>10.4f}{d[j]:>10.4f}{w_e[j]:>10.4f}")
    print("\n注意：'覆盖面积'极差 620 万m2 看着吓人，归一后却只拿 16% 权重；"
          "\n      '工期'极差只有 20 个月，权重反而是最高的 25%——熵权吃的是"
          "\n      '这一列数值摊得匀不匀'，不是'这一列数字大不大'（见避坑 2）。")
    # 加权合成分：为什么"加起来最高"不等于"排序第一"
    V = Z * w_e
    S = V.sum(axis=1)

    pr("三、TOPSIS：离理想解越近越好")
    print(f"{'方案':<16}{'D+（距最优点）':>14}{'D-（距最差点）':>14}{'贴近度 C':>10}{'名次':>6}")
    rk = {i: r for r, i in enumerate(o_e, 1)}
    for i, s in enumerate(SITES):
        print(f"{s:<16}{dp[i]:>14.4f}{dm[i]:>14.4f}{C_e[i]:>10.4f}{rk[i]:>6}")

    pr("三·补充、加权合成分 vs TOPSIS：加总分的排序和 TOPSIS 差在哪")
    for i in range(n):
        print(f"{SITES[i]:<16} 加权合成分 S={S[i]:.4f}  距理想 D+={dp[i]:.4f}  贴近度 C={C_e[i]:.4f}  "
              f"合成分名次={int(np.argsort(-S).tolist().index(i))+1}")
    rkS = [int(np.argsort(-S).tolist().index(i)) + 1 for i in range(n)]
    print(f"\n合成分冠军 = {SHORT[int(np.argmax(S))]}，TOPSIS 冠军 = {SHORT[o_e[0]]}")
    diffs = [SHORT[i] for i in range(n) if rkS[i] != o_e.index(i) + 1]
    print(f"排序不一致的方案：{diffs}（合成分名次 {[rkS[i] for i in range(n)]} "
          f"vs TOPSIS 名次 {[o_e.index(i)+1 for i in range(n)]}）")
    print("原因：加总分看『谁总量大』——B 的综合达标率 + 6 个月工期两项满分，足以盖住三项垫底；"
          "\n      TOPSIS 看『谁更均衡』——每一项离本列最优多远，短板在 D+ 里是平方项，藏不住。"
          "\n      所以'能加总分就别用 TOPSIS'只在指标可完全补偿时才成立；选址/立项这种"
          "\n      '一项不合格就报废'的题，必须用 TOPSIS 的短板惩罚。")

    pr("四、四套口径互证")
    print(f"{'方案':<16}{'熵权':>8}{'等权':>8}{'CRITIC':>8}{'AHP':>8}"
          f"{'｜熵权序':>9}{'等权序':>8}{'CRITIC序':>10}{'AHP序':>7}")
    for i, s in enumerate(SITES):
        print(f"{s:<16}{C_e[i]:>8.4f}{C_q[i]:>8.4f}{C_c[i]:>8.4f}{C_a[i]:>8.4f}"
              f"{'｜':>4}{o_e.index(i)+1:>5}{o_q.index(i)+1:>8}{o_c.index(i)+1:>10}{o_a.index(i)+1:>7}")
    print(f"\nAHP：lambda_max={lam:.4f}  CI={CI:.4f}  CR={CR:.4f} → "
          f"{'一致性可接受 (<0.1)' if CR < 0.1 else '不一致，打分自相矛盾，权重作废'}")
    print(f"Kendall τ：熵权-等权 {kendall_tau(C_e, C_q):+.2f}，熵权-CRITIC {kendall_tau(C_e, C_c):+.2f}，"
          f"熵权-AHP {kendall_tau(C_e, C_a):+.2f}")

    pr("五、稳健性：这份排序有多『硬』")
    print("① 逐指标权重 ±0.05（再归一），看榜首是否易主：")
    shadow = []
    for j, nm in enumerate(NAMES):
        row = {}
        for delta in (+0.05, -0.05):
            ww = w_e.copy()
            ww[j] += delta
            ww /= ww.sum()
            cj, _, _ = topsis(Z, ww)
            row[f"{delta:+.2f}"] = SHORT[int(np.argmax(cj))]
        shadow.append({"指标": nm, **row})
        print(f"   {nm:<10} +0.05→榜首 {row['+0.05']:<12}  -0.05→榜首 {row['-0.05']}")

    print("\n② 乘性噪声蒙特卡洛（w·exp(sigma·xi)，每档 500 次）：")
    rng = np.random.default_rng(SEED)
    mc = {}
    for sig in (0.10, 0.20, 0.30, 0.40):
        ranks = np.zeros(n)
        top_cnt = np.zeros(n)
        for _ in range(500):
            ww = w_e * np.exp(sig * rng.standard_normal(m))
            ww /= ww.sum()
            cj, _, _ = topsis(Z, ww)
            o = order_of(cj)
            for r, i in enumerate(o, 1):
                ranks[i] += r
            top_cnt[o[0]] += 1
        mc[sig] = {"avg_rank": (ranks / 500).round(3).tolist(),
                   "top_pct": (top_cnt / 500).round(3).tolist()}
        print(f"   sigma={sig:.2f}  平均名次 {np.round(ranks/500,2)}  "
              f"冠军占比 {np.round(top_cnt/500*100,1)}%")

    print("\n③ 成对翻盘阈值（挑战者超过 A 需多大扰动）：")
    pairs = []
    for i in range(1, n):
        sig, rate = flip_sigma(Z, w_e, o_e[0], i)
        pairs.append({"challenger": SHORT[i], "sigma": sig, "rate_at_sigma": round(rate, 3)})
        print(f"   {SHORT[i]:<12} 需 sigma={sig:.2f}（此时翻盘率 {rate:.1%}）")

    # ---------- 断言：教学点必须真实存在 ----------
    top = o_e[0]
    assert np.isclose(w_e.sum(), 1.0) and np.isclose(C_e.sum() > 0, True)
    assert top == 0, "榜首不是 A，讲稿需要重写"
    assert C_e[top] - C_e[o_e[1]] > 0.05, "榜首与第二名差距太小，无法给出结论"
    assert CR < 0.1, "AHP 一致性不过关"
    assert max(CR, 0) >= 0 and kendall_tau(C_e, C_c) < 0.5, "熵权与 CRITIC 排序几乎一致，'口径会改排序'落空"
    assert mc[0.20]["top_pct"][0] > 0.9, "A 在 20% 权重噪声下不稳，结论不够硬"
    assert all(p["sigma"] >= 0.2 for p in pairs), "存在脆弱成对比较"
    assert o_e[0] == 0 and o_e.index(1) >= 3, "合成分与 TOPSIS 的地位反差不足（应见 A 合成分第一但 B 被压到第 4）"
    assert o_a[0] != o_e[0], "AHP 与熵权榜首相同，'主观口径能改冠军'教学点落空"
    print("\n[断言] 全部通过")

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle("Day54 · 熵权法 + TOPSIS：5 个供热改造方案怎么排（云溪县城关镇）", fontsize=15)

    ax = axes[0, 0]
    im = ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(m))
    ax.set_xticklabels(NAMES, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SHORT, fontsize=9)
    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("① 标准化矩阵（1 = 最好）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[0, 1]
    yy = np.arange(m)
    ax.barh(yy - 0.2, w_e, height=0.38, color="#2b7bba", label="熵权（客观）")
    ax.barh(yy + 0.2, w_a, height=0.38, color="#e07b39", label="AHP（主观）")
    ax.set_yticks(yy)
    ax.set_yticklabels(NAMES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("权重")
    ax.set_title(f"② 客观 vs 主观权重（AHP CR={CR:.3f}）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 0]
    xs = np.arange(n)
    ax.bar(xs - 0.2, [C_e[i] for i in o_e], width=0.4, color="#2b7bba", label="熵权 TOPSIS")
    ax.bar(xs + 0.2, [C_q[i] for i in o_e], width=0.4, color="#c9c9c9", label="等权 TOPSIS")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[i] for i in o_e], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("贴近度 C")
    ax.set_title("③ 排序对比：熵权（按信息量）vs 等权（每票一样）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    for sig, mk in zip((0.10, 0.20, 0.30, 0.40), ("o", "s", "^", "D")):
        ax.plot(np.arange(1, n + 1), mc[sig]["avg_rank"], marker=mk, label=f"sigma={sig:.2f}")
    ax.plot(np.arange(1, n + 1), [o_e.index(i) + 1 for i in range(n)], "--",
            color="gray", label="无扰动名次", alpha=0.7)
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels([SHORT[i] for i in range(n)], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("500 次权重扰动后的平均名次")
    ax.set_title("④ 稳健性：A 稳坐第一，2~4 名互有来往", fontsize=11)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = HERE / "topsis_entropy_heating.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    payload = {
        "sites": SITES,
        "criteria": NAMES,
        "directions": DIRECTIONS.tolist(),
        "Z": Z.round(6).tolist(),
        "entropy": e.round(6).tolist(),
        "w_entropy": w_e.round(6).tolist(),
        "w_critic": w_c.round(6).tolist(),
        "w_ahp": w_a.round(6).tolist(),
        "ahp": {"lambda_max": round(lam, 5), "CI": round(CI, 5), "CR": round(CR, 5)},
        "C_entropy": C_e.round(6).tolist(),
        "C_equal": C_q.round(6).tolist(),
        "C_critic": C_c.round(6).tolist(),
        "C_ahp": C_a.round(6).tolist(),
        "D_plus": dp.round(6).tolist(),
        "D_minus": dm.round(6).tolist(),
        "rank_entropy": [SHORT[i] for i in o_e],
        "rank_equal": [SHORT[i] for i in o_q],
        "rank_critic": [SHORT[i] for i in o_c],
        "rank_ahp": [SHORT[i] for i in o_a],
        "kendall": {"ent_equal": round(kendall_tau(C_e, C_q), 3),
                     "ent_critic": round(kendall_tau(C_e, C_c), 3),
                     "ent_ahp": round(kendall_tau(C_e, C_a), 3)},
        "shadow": shadow,
        "monte_carlo": {str(k): v for k, v in mc.items()},
        "pair_flip_sigma": pairs,
    }
    (HERE / "topsis_entropy_heating.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出：{out_png.name} / topsis_entropy_heating.json")


if __name__ == "__main__":
    main()
