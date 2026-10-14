#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day45 - 差分进化算法 DE 求解带约束非凸连续优化
subject : mathematical-modeling
level   : L5 算法（元启发式 / 进化算法）/ L6 优化模型（约束非线性优化）
anchor  : 04 - Optimization / 05 - Algorithms
goal    : 用「差分进化 DE」求解一道「县域农产品加工厂多产品产量决策」的
          带约束、非凸、多峰连续优化问题——
          ① 讲清 DE 凭「种群内两个体之差」当扰动、无需梯度、自适应步长；
          ② 与 SLSQP（梯度类）从天真起点卡在局部最优形成对照（本例差 6.5%+）；
          ③ 用「多起点 SLSQP + 密集网格」当独立裁判交叉验证；
          ④ 用惩罚法处理约束，并把「解必须可行」作为硬验收。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "<本文件路径>"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 中文字体注册（Noto Sans CJK SC / 文泉驿），避免图中中文乱码
# ----------------------------------------------------------------------
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_registered = []
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            _registered.append(font_manager.FontProperties(fname=_p).get_name())
        except Exception:
            pass
if _registered:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _registered + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
print("CJK 字体已注册：", _registered if _registered else "（无，可能乱码）")

RNG_SEED = 20261014

# ======================================================================
# 一、真实决策问题：县域农产品加工厂的「多产品产量决策」
# ======================================================================
# 背景：一家县域农产品加工厂用同一条产线生产 3 种产品（x1 果干、x2 果汁、
#       x3 冻干蔬菜，单位：吨/周）。决策者要定每周各产品产量，使**周利润最大**。
#
# 目标（最大化；本文按最小化取负）：
#   profit(x) = Σ_i p_i·x_i                                    （销售收入）
#               + A·Σ_i sin(ω_i·x_i)                           （价格的季节/批次波动）
#               - Σ_i c_i·x_i^2/(1+a_i·x_i)                    （边际成本先降后升）
#               + syn·sqrt(x1·x2)                              （果干果渣共用的协同收益）
#
#   ★ 关键：正弦「批次波动」项让利润面出现**大量局部峰谷**（本例梯度法可达 23 个局部最优），
#     而真正的全局最优藏在一个「梯度法从零产量出发根本走不到」的角落里。
#
# 约束（现实硬约束）：
#   (1) 总产能：x1 + x2 + x3 <= CAP               （产线工时）
#   (2) 冷藏库：0.5·x1 + 1.5·x2 + 0.2·x3 <= COLD  （冷链仓储）
#   (3) 排放：  0.8·x1 + 0.4·x2 + 1.2·x3 <= EMAX  （污水/能耗上限）
#   (4) 单品区间：0 <= x_i <= U_i

DIM = 3
LB = np.array([0.0, 0.0, 0.0])          # 产量非负
UB = np.array([80.0, 60.0, 50.0])       # 单品产能上限（吨/周）

P_PRICE = np.array([42.0, 30.0, 55.0])  # 单位售价（千元/吨）
C_COST = np.array([0.55, 0.62, 0.90])   # 二次成本系数
A_SCALE = np.array([0.09, 0.11, 0.07])  # 规模经济尺度
SYNERGY = 6.0                           # 协同收益系数（x1 与 x2 共用果渣原料）
RIPPLE_AMP = 45.0                       # 批次波动幅度
RIPPLE_W = np.array([0.50, 0.55, 0.45]) # 各产品的价格波动频率

CAP = 110.0                             # 产能上限（等效工时）
COLD = 95.0                             # 冷藏库上限
EMAX = 78.0                             # 排放上限
EMIT = np.array([0.8, 0.4, 1.2])        # 单位排放系数
COLDCOEF = np.array([0.5, 1.5, 0.2])    # 单位冷藏占用


def profit(x):
    """周利润（千元），向量化：x 形状 (..., DIM)。"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    revenue = (np.sum(P_PRICE * x, axis=1)
               + RIPPLE_AMP * np.sum(np.sin(RIPPLE_W * x), axis=1))
    cost = np.sum(C_COST * x**2 / (1.0 + A_SCALE * x), axis=1)
    synergy = SYNERGY * np.sqrt(np.clip(x[:, 0] * x[:, 1], 0, None))
    return revenue - cost + synergy


def negation(x):
    """给 scipy 最小化用：-profit。"""
    return -profit(x)


def constraints_violation(x):
    """返回各约束的违反量（<=0 为可行）。x 形状 (..., DIM)。"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    v1 = np.sum(x, axis=1) - CAP
    v2 = x @ COLDCOEF - COLD
    v3 = x @ EMIT - EMAX
    return np.stack([v1, v2, v3], axis=1)


def is_feasible(x, tol=1e-8):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if np.any(x < LB - 1e-9) or np.any(x > UB + 1e-9):
        return np.zeros(x.shape[0], dtype=bool)
    return np.all(constraints_violation(x) <= tol, axis=1)


# ======================================================================
# 二、主角：差分进化 DE（Storn & Price 1997）—— DE/rand/1/bin + 惩罚
# ======================================================================
def differential_evolution(objective, lb, ub, pop_size=60, max_gen=800,
                           F=0.6, CR=0.9, penalty=1000.0, seed=0,
                           record_history=False):
    """
    带惩罚的差分进化（最小化 objective + 约束惩罚）。
    机制：
      mutation  —— v = x_r1 + F·(x_r2 - x_r3)：用「种群内两个体之差」当自适应步长
      crossover —— 二项交叉，按 CR 逐维继承
      selection —— 贪心：子代不优于父代则不替换
    F  : 差分权重（缩放「群体自身差异」，即自适应步长倍率）
    CR : 交叉概率
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    dim = len(lb)
    scale = ub - lb

    pop = lb + rng.random((pop_size, dim)) * scale

    def fit(individuals):
        obj = objective(individuals)
        viol = np.clip(constraints_violation(individuals), 0, None).sum(axis=1)
        return obj + penalty * viol

    pop_fit = fit(pop)
    best_idx = int(np.argmin(pop_fit))
    best_x = pop[best_idx].copy()
    best_fit = pop_fit[best_idx]
    history = [best_fit] if record_history else None

    for _ in range(max_gen):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)

            cross_mask = rng.random(dim) < CR
            cross_mask[rng.integers(dim)] = True   # 至少继承一维
            trial = np.where(cross_mask, mutant, pop[i])

            trial_fit = float(fit(trial[None, :])[0])
            if trial_fit <= pop_fit[i]:
                pop[i] = trial
                pop_fit[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_x = trial.copy()
        if record_history:
            history.append(best_fit)

    return {
        "x": best_x,
        "fitness": best_fit,
        "success": bool(is_feasible(best_x[None, :])[0]),
        "history": np.array(history) if record_history else None,
    }


# ======================================================================
# 三、裁判 / 对照
# ======================================================================
def _scipy_cons():
    return [
        {"type": "ineq", "fun": lambda x: CAP - np.sum(x)},
        {"type": "ineq", "fun": lambda x: COLD - x @ COLDCOEF},
        {"type": "ineq", "fun": lambda x: EMAX - x @ EMIT},
    ]


def slsqp_from(x0, maxiter=800):
    """梯度类（SLSQP）从给定起点求解（带约束）。"""
    res = minimize(negation, np.asarray(x0, float), method="SLSQP",
                   bounds=list(zip(LB, UB)), constraints=_scipy_cons(),
                   options={"maxiter": maxiter, "ftol": 1e-11})
    ok = bool(is_feasible(res.x[None, :])[0])
    return {"x": res.x, "profit": float(profit(res.x)[0]), "success": ok}


def multistart_slsqp(n_starts=200, seed=1):
    """多起点 SLSQP：不依赖 DE 的独立强基线 / 裁判之一。"""
    rng = np.random.default_rng(seed)
    best = None
    n_local = 0
    seen = []
    for _ in range(n_starts):
        x0 = LB + rng.random(DIM) * (UB - LB)
        r = slsqp_from(x0)
        if r["success"]:
            if not any(np.allclose(r["x"], s, atol=1e-2) for s in seen):
                seen.append(r["x"].copy())
            if best is None or r["profit"] > best["profit"]:
                best = r
    n_local = len(seen)
    return best, n_local


def grid_search(n_per_dim=70):
    """密集网格暴力（可行域内），独立裁判之二。"""
    grids = [np.linspace(LB[i], UB[i], n_per_dim) for i in range(DIM)]
    g = np.meshgrid(*grids, indexing="ij")
    pts = np.stack([t.ravel() for t in g], axis=1)
    feas = is_feasible(pts)
    if not feas.any():
        return {"x": None, "profit": -np.inf, "n_feasible": 0}
    f = profit(pts[feas])
    best = int(np.argmax(f))
    return {"x": pts[feas][best], "profit": float(f[best]),
            "n_feasible": int(feas.sum())}


def naive_profit(naive_x):
    return float(profit(naive_x)[0])


# ======================================================================
# 四、主流程
# ======================================================================
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 68)
    print("Day45 · 差分进化 DE 求解带约束非凸连续优化")
    print("=" * 68)

    print(f"\n[问题体检] 单品上限 UB={UB}，产能<= {CAP}，冷藏<= {COLD}，排放<= {EMAX}")
    print(f"  零产量（天真起点）利润 = {float(profit(np.zeros(DIM))[0]):.4f}")

    # --- 天真方案：只产最贵的冻干蔬菜 x3 ---
    naive_x = np.array([LB[0], LB[1], UB[2]])
    if not is_feasible(naive_x[None, :])[0]:
        s = np.clip(min(CAP / naive_x.sum(), COLD / (naive_x @ COLDCOEF),
                        EMAX / (naive_x @ EMIT)), 0, 1)
        naive_x = naive_x * s
    print(f"  [天真-只产冻干蔬菜 x3={naive_x[2]:.1f}] 利润 = "
          f"{naive_profit(naive_x):.4f}  可行={bool(is_feasible(naive_x[None,:])[0])}")

    # --- 主角：DE ---
    print("\n[DE 差分进化] 种群 60 × 800 代，F=0.6, CR=0.9, 惩罚=1000")
    de = differential_evolution(negation, LB, UB, pop_size=60, max_gen=800,
                                F=0.6, CR=0.9, penalty=1000.0,
                                seed=RNG_SEED, record_history=True)
    de_profit = float(profit(de["x"])[0])
    print(f"  DE 最优产量 x = {np.round(de['x'], 4)}")
    print(f"  DE 最优利润 = {de_profit:.4f}   可行 = {de['success']}")

    # --- 对照①：SLSQP 从零产量起步（工程师最常做的天真做法）---
    s_zero = slsqp_from(np.zeros(DIM))
    print(f"\n[对照① SLSQP 从零产量起步] 利润 = {s_zero['profit']:.4f}  "
          f"x = {np.round(s_zero['x'], 4)}  可行={s_zero['success']}")

    # --- 对照②：SLSQP 从「40% 上限」起步 ---
    s_mid = slsqp_from(UB * 0.4)
    print(f"[对照② SLSQP 从 40%上限起步] 利润 = {s_mid['profit']:.4f}  "
          f"x = {np.round(s_mid['x'], 4)}  可行={s_mid['success']}")

    # --- 裁判①：多起点 SLSQP ---
    ms, n_local = multistart_slsqp(n_starts=200, seed=7)
    print(f"[裁判① 多起点 SLSQP×200] 利润 = {ms['profit']:.4f}  "
          f"x = {np.round(ms['x'], 4)}  发现局部最优 {n_local} 个")

    # --- 裁判②：密集网格暴力 ---
    gs = grid_search(n_per_dim=70)
    print(f"[裁判② 网格暴力 70^3={70**3} 点] 利润 = {gs['profit']:.4f}  "
          f"x = {np.round(gs['x'], 4)}  可行点数={gs['n_feasible']}")

    # --- 鲁棒性：DE 独立跑 20 次 ---
    print("\n[鲁棒性] DE 独立运行 20 次（不同种子，400 代）")
    runs = []
    for k in range(20):
        r = differential_evolution(negation, LB, UB, pop_size=60, max_gen=400,
                                   F=0.6, CR=0.9, penalty=1000.0, seed=100 + k)
        runs.append(float(profit(r["x"])[0]))
    runs = np.array(runs)
    spread = (runs.max() - runs.min()) / abs(runs.mean()) * 100
    print(f"  最好 {runs.max():.4f} / 最差 {runs.min():.4f} / "
          f"均值 {runs.mean():.4f} / 标准差 {runs.std():.4f}  自身浮动 {spread:.3f}%")

    # --- 断言 ---
    best_ref = max(ms["profit"], gs["profit"])
    gap_zero = (de_profit - s_zero["profit"]) / abs(s_zero["profit"]) * 100
    rel = abs(de_profit - best_ref) / abs(best_ref)
    assert de["success"], "DE 解必须可行！"
    assert gap_zero > 3.0, f"教学点落空：零起点 SLSQP 只差 {gap_zero:.2f}%（应 >3%）"
    assert de_profit >= s_zero["profit"] - 1e-6, "DE 竟差于零起点 SLSQP？"
    assert de_profit >= naive_profit(naive_x) - 1e-6, "DE 竟差于天真方案？"
    assert rel < 0.01, f"DE 与最强裁判差距 {rel*100:.4f}% 过大"
    assert (runs >= runs.mean() * 0.98).mean() >= 0.95, "DE 多次运行不够稳定"
    print(f"\n[断言全过] DE 可行 ✓  比零起点 SLSQP 多赚 {gap_zero:.2f}% ✓  "
          f"不劣于天真方案 ✓  与最强裁判相对差 {rel*100:.3f}% ✓  鲁棒性达标 ✓")

    plot_results(out_dir, de, de_profit, s_zero, s_mid, ms, gs, runs, naive_x)
    print("\n完成。图已保存。")
    return de, de_profit


def plot_results(out_dir, de, de_profit, s_zero, s_mid, ms, gs, runs, naive_x):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (1) DE 收敛曲线（历史最优适应度）
    ax = axes[0, 0]
    hist = de["history"]
    ax.plot(hist, color="#1f77b4", lw=2, label="DE 历史最优适应度")
    ax.set_xlabel("迭代代数"); ax.set_ylabel("适应度（含约束惩罚，越小越好）")
    ax.set_title("① DE 收敛曲线（阶梯式下探）", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2) 各方案利润对比
    ax = axes[0, 1]
    labels = ["零产量\n(天真起点)", "只产冻干蔬菜\n(天真)", "SLSQP\n零起点(卡住)",
              "SLSQP\n40%起点", "多起点\nSLSQP×200", "网格\n70³", "DE\n(本讲)"]
    vals = [float(profit(np.zeros(DIM))[0]), naive_profit(naive_x),
            s_zero["profit"], s_mid["profit"], ms["profit"], gs["profit"], de_profit]
    colors = ["#bbbbbb", "#bbbbbb", "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#2ca02c"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("周利润（千元）")
    ax.set_title("② 各方案利润对比（红=被局部最优困住）", fontsize=13)
    ax.tick_params(axis="x", labelsize=8); ax.grid(alpha=0.3, axis="y")

    # (3) x1-x2 切片热力图（x3 固定为 DE 的 x3*）
    ax = axes[1, 0]
    xs = np.linspace(0, UB[0], 240)
    ys = np.linspace(0, UB[1], 240)
    XX, YY = np.meshgrid(xs, ys)
    x3fix = de["x"][2]
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, x3fix)], axis=1)
    Z = profit(pts).reshape(XX.shape)
    feas = is_feasible(pts).reshape(XX.shape)
    Zmask = np.where(feas, Z, np.nan)
    im = ax.contourf(XX, YY, Zmask, levels=25, cmap="viridis")
    plt.colorbar(im, ax=ax, label="利润（千元）")
    ax.contour(XX, YY, Zmask, levels=14, colors="white", linewidths=0.4, alpha=0.5)
    ax.plot(de["x"][0], de["x"][1], "r*", ms=24, mec="white", mew=1.6,
            label="DE 最优（全局）")
    ax.plot(s_zero["x"][0], s_zero["x"][1], "o", color="orange", ms=12,
            mec="black", label="SLSQP 零起点（卡局部最优）")
    ax.plot(naive_x[0], naive_x[1], "s", color="lightgrey", ms=11, mec="black",
            label="天真-只产冻干蔬菜")
    ax.set_xlabel("x1 果干（吨/周）"); ax.set_ylabel("x2 果汁（吨/周）")
    ax.set_title(f"③ x1-x2 利润地形切片（x3={x3fix:.1f}；白=不可行；多峰明显）", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")

    # (4) 鲁棒性直方图
    ax = axes[1, 1]
    if runs.max() - runs.min() < 1e-6:
        # 全部一致，画点线
        ax.plot(np.arange(1, len(runs) + 1), runs, "o-", color="#2ca02c")
        ax.set_ylim(runs.mean() - 1, runs.mean() + 1)
        ax.set_title("④ DE 鲁棒性：20 次完全一致（零浮动）", fontsize=13)
        ax.set_xlabel("独立运行序号"); ax.set_ylabel("最终利润（千元）")
    else:
        ax.hist(runs, bins=10, color="#2ca02c", edgecolor="black", alpha=0.75)
        ax.axvline(runs.mean(), color="red", ls="--", lw=1.5,
                   label=f"均值 {runs.mean():.1f}")
        ax.axvline(runs.max(), color="navy", ls=":", lw=1.5,
                   label=f"最好 {runs.max():.1f}")
        ax.set_xlabel("20 次独立运行最终利润（千元）"); ax.set_ylabel("频数")
        ax.set_title(f"④ DE 鲁棒性：自身浮动 {((runs.max()-runs.min())/abs(runs.mean())*100):.3f}%",
                     fontsize=13)
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Day45 · 差分进化 DE 求解带约束非凸连续优化（县域农产品加工厂多产品产量决策）",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = os.path.join(out_dir, "differential_evolution_constrained.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"图已保存：{out_png}")


if __name__ == "__main__":
    main()
