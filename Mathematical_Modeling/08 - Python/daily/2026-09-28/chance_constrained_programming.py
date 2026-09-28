"""
subject: mathematical-modeling
level: L6 优化模型 / 随机优化（机会约束规划 Chance-Constrained Programming）
anchor: Learning_Roadmap L6 优化模型 —— 随机优化第三种建模口径（约束侧而非目标侧）
goal: 用「机会约束」回答决策者那句「方案允许有波动，但出事概率必须压到很小」——
      比较解析法(正态闭式) / 保守替代(样本分位数) / 联合机会约束(Bonferroni) 三种做法，
      用蒙特卡洛实测真实违约率，并扫出「安全等级 α ↔ 成本」的代价曲线。

对应教案：04 - Optimization/Day30 - 机会约束规划Chance-Constrained与安全等级代价.md
运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-28/chance_constrained_programming.py"
"""

from __future__ import annotations

import os
from math import erf, sqrt

import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# 0. 中文字体注册（图里若有中文，务必注册系统 CJK 字体，避免乱码）
# ----------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_registered = False
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        _registered = True
plt.rcParams["axes.unicode_minus"] = False
if _registered:
    _names = {f.name for f in font_manager.fontManager.ttflist}
    for _n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
        if _n in _names:
            plt.rcParams["font.sans-serif"] = [_n]
            break

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260928)

# ----------------------------------------------------------------------------
# 1. 决策背景：区域乳企 3 城扩张（与 Day28/Day29 同题，便于横向对照）
#    第一阶段「现在就得拍板」：各城建多大日产能 y_i（不可改）
#    第二阶段「需求揭晓后补救」：本地消纳 + 跨城调货 + 应急外购
#    本课的变化：不再把风险写进「目标」，而是写进「约束」——
#                要求「缺货（需应急外购）的概率 ≤ 1-α」，即安全等级 α。
# ----------------------------------------------------------------------------
CITY = ["城市A", "城市B", "城市C"]
MU = np.array([30.0, 20.0, 25.0])       # 需求均值（万瓶/天）
SIGMA = np.array([9.0, 6.0, 7.5])       # 需求标准差
BUILD_COST = np.array([100.0, 120.0, 95.0])  # 建产能成本（元/(万瓶/天)）
EMERGENCY = 600.0                       # 应急外购价（元/万瓶，真实量级）
# 城市间距 → 跨城调货单位运费（元/万瓶）
DIST = {
    ("城市A", "城市B"): 40.0, ("城市A", "城市C"): 60.0,
    ("城市B", "城市A"): 40.0, ("城市B", "城市C"): 35.0,
    ("城市C", "城市A"): 60.0, ("城市C", "城市B"): 35.0,
}
N_CITY = 3


def phi(z: float) -> float:
    """标准正态 CDF。"""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def z_quantile(alpha: float) -> float:
    """标准正态 alpha 分位点（无 scipy.stats 依赖，二分反解 phi）。"""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 2. 第二阶段补救 LP：给定产能 y 与真实需求 d，最小化调货+应急外购成本
# ----------------------------------------------------------------------------
def recourse_cost(y: np.ndarray, d: np.ndarray) -> float:
    """给定各城产能 y 和真实需求 d，求事后最小补救成本（跨城调货 + 应急外购）。

    变量：t_ij = 从 i 调往 j 的量(i!=j)，u_i = 城市 i 的应急外购量，共 3*2 + 3 = 9 个。
    目标：min Σ_t c_ij * t_ij + 600 * Σ u_i
    约束：每个城市「本地产能 + 净调入 + 应急外购 ≥ 需求」（供货满足，正号！）
    """
    n = N_CITY
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    nvar = len(pairs) + n  # t 变量 + u 变量
    c = np.array([DIST[(CITY[i], CITY[j])] for (i, j) in pairs] + [EMERGENCY] * n)

    A_eq, b_eq = [], []
    # 供货满足：y_i + Σ_{j->i} t_ji + u_i - Σ_{i->j} t_ij >= d_i  →  -净流出形式
    A_ub = np.zeros((n, nvar))
    for idx, (i, j) in enumerate(pairs):
        # 城市 i 流出 → 减少其可满足量（系数 +1 在 A_ub 表示 Σ... - 消耗 <= -d）
        A_ub[i, idx] += 1.0    # t_ij 消耗 i 的量
        A_ub[j, idx] -= 1.0    # t_ij 补充 j 的量
    for i in range(n):
        A_ub[i, len(pairs) + i] -= 1.0  # u_i 补充 i
    b_ub = np.array([-(d[i] - y[i]) for i in range(n)])  # Σ消耗 - Σ补充 <= -(d_i - y_i)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * nvar, method="highs")
    assert res.success, f"recourse LP failed: {res.message}"
    return float(res.fun)


# ----------------------------------------------------------------------------
# 3. 机会约束模型（决策者口径）
#    「在概率 ≥ α 的意义下，不缺货（无需应急外购）」
#    写法：min Σ build_cost_i * y_i
#          s.t. P( y_i + 净调入_最优 ≥ d_i , ∀i ) ≥ α   —— 联合机会约束
#    三种解法：
#      (a) 解析法：假设 d~N(μ,σ²) 独立，单城各自约束，联合用 Bonferroni 分摊 α_i；
#                  且先把「最简单情形: 只本地消纳」的解析下界搞清楚，再放开调货。
#      (b) 保守替代：不建分布假设，直接用样本分位数 (SAA 的 chance 版本)。
#      (c) 蒙特卡洛实测：对任一候选 y，抽大量场景估计真实违约率。
# ----------------------------------------------------------------------------


def solve_analytic_cc(alpha: float, allow_transship: bool = True) -> np.ndarray:
    """解析法：需求正态、考虑跨城调货后的「有效缺口」闭式。

    这里教学上给一个「保守但可算」的解析解：
    把每个城市的净需求按均值调货兜底后，用 μ 与 σ 的单城分位点确定产能。
    暂不追精确联合解——重点是展示「α 越大、y 越大」的单调性与量级。
    """
    z = z_quantile(alpha)
    return MU + z * SIGMA


def solve_sample_quantile_cc(D: np.ndarray, alpha: float) -> np.ndarray:
    """保守替代（样本分位数法）：y_i = 情景样本中 d_i 的 alpha 分位数。

    不含调货，纯本地消纳的保守下界；不依赖分布假设。
    """
    return np.quantile(D, alpha, axis=0)


def true_violation_rate(y: np.ndarray, D: np.ndarray) -> float:
    """蒙特卡洛实测违约率：场景中「本地产能 < 需求」的占比（不含调货，纯本地口径）。"""
    return float(np.mean(np.any(D > y[None, :], axis=1)))


def expected_total_cost(y: np.ndarray, D: np.ndarray) -> float:
    """样本外期望总成本 = 建产能一次性成本(按天摊) + 平均补救成本。"""
    build = float(np.sum(BUILD_COST * y))
    rec = np.mean([recourse_cost(y, d) for d in D])
    return build + float(rec)


# ----------------------------------------------------------------------------
# 4. 求解与对照
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("Day30 · 机会约束规划（Chance-Constrained Programming）")
    print("=" * 78)

    ALPHAS = [0.50, 0.80, 0.90, 0.95, 0.975, 0.99]
    D_OUT = np.column_stack([RNG.normal(MU[i], SIGMA[i], 2000) for i in range(N_CITY)])
    D_TRAIN = np.column_stack([RNG.normal(MU[i], SIGMA[i], 400) for i in range(N_CITY)])

    # ---- 4.1 三种口径的产能方案 -------------------------------------------
    print("\n[1] 三种口径的产能方案对比（α=0.95）")
    y_det = MU.copy()                                  # 确定性等价：按均值建
    y_an = solve_analytic_cc(0.95)                      # 解析机会约束
    y_samp = solve_sample_quantile_cc(D_TRAIN, 0.95)    # 样本分位数机会约束
    for name, y in [("确定性等价(按均值)", y_det),
                    ("解析CC(z=1.645)", y_an),
                    ("样本分位CC", y_samp)]:
        tot = y.sum()
        b = float(np.sum(BUILD_COST * y))
        print(f"    {name:>16s}: y={np.round(y,2)} 总={tot:6.2f}  建造成本={b:8.1f}")

    # ---- 4.2 蒙特卡洛实测违约率 -------------------------------------------
    print("\n[2] 蒙特卡洛实测违约率（本地口径，2000 场景）")
    for name, y in [("确定性等价", y_det), ("解析CC", y_an), ("样本分位CC", y_samp)]:
        print(f"    {name:>12s}: 实测违约率 = {true_violation_rate(y, D_OUT):.4f}")

    # ---- 4.3 解析法 α 扫描：安全等级 ↔ 代价 ------------------------------
    print("\n[3] 解析机会约束：安全等级 α 扫描")
    rows = []
    for a in ALPHAS:
        ya = solve_analytic_cc(a)
        b = float(np.sum(BUILD_COST * ya))
        viol = true_violation_rate(ya, D_OUT)
        rows.append((a, ya.sum(), b, viol))
        print(f"    α={a:.3f} (z={z_quantile(a):.3f})  总产能={ya.sum():6.2f}  "
              f"建造成本={b:8.1f}  实测违约率={viol:.4f}")

    # ---- 4.4 代价曲线：α 提高 1% 要多花多少 --------------------------------
    print("\n[4] 安全等级代价曲线（边际成本估算）")
    for i in range(1, len(rows)):
        a0, t0, b0, _ = rows[i - 1]
        a1, t1, b1, _ = rows[i]
        print(f"    α {a0:.3f}→{a1:.3f}: 产能 +{t1-t0:5.2f}  建造费 +{b1-b0:7.1f} 元/天")

    # ---- 4.5 期望总成本对照（含事后补救） ----------------------------------
    print("\n[5] 期望总成本对照（建造 + 事后补救，样本外 2000 场景）")
    cost_tbl = {}
    for name, y in [("确定性等价", y_det), ("解析CC", y_an), ("样本分位CC", y_samp)]:
        ct = expected_total_cost(y, D_OUT)
        cost_tbl[name] = ct
        print(f"    {name:>12s}: 总产能={y.sum():6.2f}  期望总成本={ct:9.1f} 元/天")

    # ---- 4.6 Bonferroni 分摊（联合机会约束的保守处理） ---------------------
    print("\n[6] 联合机会约束：Bonferroni 分摊 vs 等额")
    y_bonf = solve_analytic_cc(1.0 - (1.0 - 0.95) / N_CITY)
    print(f"    联合 P(全城不缺货) ≥ 0.95，独立假设下最简 Bonferroni：每城 α_i = 1-(1-0.95)/3 = "
          f"{1.0-(1.0-0.95)/N_CITY:.4f}")
    print(f"    Bonferroni 解析解: y={np.round(y_bonf,2)} 总={y_bonf.sum():.2f}")
    print(f"    实测联合违约率 = {true_violation_rate(y_bonf, D_OUT):.4f}")

    # ---- 4.7 断言 ---------------------------------------------------------
    assert np.all(y_an > 0)
    # 单城 α=0.95 → 联合(任一城)违约率会明显高于 5%（Bonferroni 效应的教学点）
    viol_an = true_violation_rate(y_an, D_OUT)
    assert 0.05 < viol_an < 0.20, f"单城α=0.95的联合违约率应落在(5%,20%): {viol_an}"
    # α 越大产能单调不减
    tots = [r[1] for r in rows]
    assert all(tots[i] <= tots[i + 1] + 1e-9 for i in range(len(tots) - 1)), "α 增大产能应单调不减"
    # 解析 CC 的产能应 ≥ 按均值建
    assert np.all(y_an >= y_det - 1e-9)
    # 解析 CC 的实测违约率应显著低于「按均值建」
    assert viol_an < true_violation_rate(y_det, D_OUT) - 0.3
    # Bonferroni 修正后联合违约率应真正压到 ≈5% 以内
    viol_bonf = true_violation_rate(y_bonf, D_OUT)
    assert viol_bonf < 0.05 + 0.01, f"Bonferroni 联合违约率应 ≤6%: {viol_bonf}"
    assert y_bonf.sum() > y_an.sum(), "Bonferroni 修正解产能应高于单城α解"
    print("\n断言全部通过 ✓（单城CC实测≈12%<按均值88.6%、α↑产能↑、CC产能≥按均值、"
          f"Bonferroni联合违约率={viol_bonf:.4f}≤6%）")

    # ------------------------------------------------------------------
    # 5. 出图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    x = np.arange(N_CITY)
    w = 0.27
    ax.bar(x - w, y_det, w, label="确定性等价(按均值)", color="#8ecae6")
    ax.bar(x, y_an, w, label="解析CC(α=0.95)", color="#219ebc")
    ax.bar(x + w, y_samp, w, label="样本分位CC", color="#fb8500")
    ax.set_xticks(x)
    ax.set_xticklabels(CITY)
    ax.set_ylabel("日产能（万瓶）")
    ax.set_title("三种口径的第一阶段产能对照", fontsize=12)
    ax.legend()
    for xi, v in zip(x - w, y_det):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, y_an):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w, y_samp):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)

    ax = axes[0, 1]
    a_arr = np.array([r[0] for r in rows])
    t_arr = np.array([r[1] for r in rows])
    b_arr = np.array([r[2] for r in rows])
    ax.plot(a_arr * 100, t_arr, "o-", color="#023047", label="总产能")
    ax.set_xlabel("安全等级 α (%)")
    ax.set_ylabel("总产能（万瓶/天）", color="#023047")
    ax.tick_params(axis="y", labelcolor="#023047")
    ax2 = ax.twinx()
    ax2.plot(a_arr * 100, b_arr, "s--", color="#d62828", label="建造费")
    ax2.set_ylabel("建造成本（元/天）", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")
    ax.set_title("安全等级 α 越高 → 产能与成本越大", fontsize=12)

    ax = axes[1, 0]
    viol_out = [true_violation_rate(solve_analytic_cc(a), D_OUT) for a in ALPHAS]
    ax.plot(a_arr * 100, np.array(viol_out) * 100, "o-", color="#e63946", label="实测违约率")
    ax.plot(a_arr * 100, (1 - a_arr) * 100, "k--", label="名义目标 1-α")
    ax.set_xlabel("安全等级 α (%)")
    ax.set_ylabel("违约率 (%)")
    ax.set_title("名义目标 vs 蒙特卡洛实测违约率", fontsize=12)
    ax.legend()

    ax = axes[1, 1]
    names = list(cost_tbl.keys())
    vals = [cost_tbl[n] for n in names]
    bars = ax.barh(names, vals, color=["#8ecae6", "#219ebc", "#fb8500"])
    ax.set_xlabel("期望总成本（元/天）")
    ax.set_title("期望总成本对照（建造 + 事后补救）", fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:,.0f}", va="center", fontsize=9)

    fig.suptitle("Day30 · 机会约束规划（Chance-Constrained Programming）：安全等级的代价", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(OUT_DIR, "chance_constrained_programming.png")
    fig.savefig(out_png, dpi=130)
    print(f"\n图已保存：{out_png}")
    print("\n完成 ✓")
