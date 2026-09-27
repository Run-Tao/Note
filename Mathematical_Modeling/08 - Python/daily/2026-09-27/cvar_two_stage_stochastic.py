# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L6 优化模型 / 随机优化（风险厌恶两阶段随机规划 + CVaR）
anchor:  Day29 —— 接 Day28 两阶段随机规划(Two-Stage SP + SAA)、Day27 报童、Day18 MDP
goal:    回答"既要省钱，又要避免最坏局面把公司打穿"这个决策——
        用 CVaR (Conditional Value-at-Risk) 把"尾部风险"显式写进目标，
        在"期望成本最小(风险中性)"与"95% 最坏场景平均也不许太惨(风险厌恶)"之间
        扫出一条有效前沿，并回答：多花多少钱能买来多少尾部安全？

场景：一家区域乳企在 3 城扩张，先建日产能（建完不改，here-and-now），
      需求揭晓后再跨城调货/应急外购（recourse）。
      与 Day28 同一套数据，但目标从"期望成本最小"升级为"CVaR 约束下的期望成本最小"。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-09-27/cvar_two_stage_stochastic.py"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linprog
from scipy.stats import norm
import os
import time

# ---------- 中文字体（避免图中乱码） ----------
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260927)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 0. 问题数据（与 Day28 同题，便于横向对照）
# ============================================================
CITIES = ["城市A", "城市B", "城市C"]
N = 3
MU = np.array([30.0, 20.0, 25.0])          # 需求均值（万瓶/天）
SIGMA = np.array([9.0, 6.0, 7.5])          # 需求标准差
CAP_COST = np.array([100.0, 120.0, 95.0])  # 建产能成本（元/万瓶）

# 跨城调货成本 = 城间距 × 单位运费（A-B 40, A-C 60, B-C 35）
SHIP = np.array([
    [0.0, 40.0, 60.0],
    [40.0, 0.0, 35.0],
    [60.0, 35.0, 0.0],
])
P_EMERG = 600.0   # 应急外购单价（元/万瓶），真实量级的缺货代价


def sample_demand(n, sigma=None):
    """生成 n 个需求场景，shape (n, N)。sigma 缺省用全局 SIGMA。"""
    s = SIGMA if sigma is None else sigma
    return rng.normal(MU, s, size=(n, N))


# ============================================================
# 1. 第二阶段：给定产能 y 与需求 d，求最小补救成本 Q(y, d)
#    变量: t[j,i] 从 j 调到 i 的量(j≠i), u[i] 应急外购量
# ============================================================
def recourse_cost(y, d, return_plan=False):
    """第二阶段调货 LP。约束：本地产能 + 调入 − 外调 + 应急 >= 需求。"""
    nv = N * N + N          # t 占 N*N, u 占 N
    c = np.zeros(nv)
    for j in range(N):
        for i in range(N):
            if i != j:
                c[j * N + i] = SHIP[j, i]
    for i in range(N):
        c[N * N + i] = P_EMERG

    # 供货满足约束 (N 条): -sum_j t[j,i] + sum_j t[i,j] - u[i] <= -d[i] + y[i]
    A_ub = np.zeros((N, nv))
    b_ub = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                A_ub[i, j * N + i] = -1.0   # 从 j 调入 i（增加 i 供给）
                A_ub[i, i * N + j] = +1.0   # 从 i 调出 j（减少 i 供给）
        A_ub[i, N * N + i] = -1.0           # 应急外购增加供给
        b_ub[i] = -d[i] + y[i]              # 补给缺口 = 需求 - 本地产能

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        raise RuntimeError("第二阶段 LP 求解失败: " + res.message)
    if return_plan:
        return res.fun, res.x
    return res.fun


def evaluate_policy(y, D):
    """在场景集 D 上评估第一阶段决策 y 的每个场景总成本（建产能 + 补救）。"""
    build = float(CAP_COST @ y)
    rec = np.array([recourse_cost(y, D[k]) for k in range(D.shape[0])])
    return build + rec          # 每个场景的总成本向量


# ============================================================
# 2. SAA 场景展开：把两阶段并成一个大的确定性等价 LP
#    变量 = y(N) + 每场景的 t(N*N) + u(N)
# ============================================================
def solve_saa(D, risk=None, alpha=0.95, lam=0.0):
    """
    在场景集 D 上求第一阶段产能 y。

    risk=None          -> 风险中性：min 期望总成本
    risk="cvar_lp"     -> 风险厌恶：min 期望成本 + lam * CVaR_alpha(总成本)
                          （Rockafellar-Uryasev 线性化：CVaR = min_eta eta +
                           (1/((1-alpha)S)) sum s_k,  s_k >= 成本_k - eta, s_k>=0）

    返回 (y, info_dict)
    """
    S = D.shape[0]
    ny, nt = N, N * N
    # 变量布局: [y(N), t0(N*N), u0(N), t1,..., eta, s0..s_{S-1}]
    n_main = N + S * (nt + N)
    use_cvar = risk == "cvar_lp"
    nv = n_main + (1 + S if use_cvar else 0)
    c = np.zeros(nv)

    # 目标：建产能成本 + 场景平均补救成本
    c[:N] = CAP_COST
    for k in range(S):
        base = N + k * (nt + N)
        for j in range(N):
            for i in range(N):
                if i != j:
                    c[base + j * N + i] = SHIP[j, i] / S
        for i in range(N):
            c[base + nt + i] = P_EMERG / S

    if use_cvar:
        eta_idx = n_main
        s_idx0 = n_main + 1
        c[eta_idx] = lam
        c[s_idx0:s_idx0 + S] = lam / ((1 - alpha) * S)

    A_ub, b_ub = [], []
    # 每场景的供货满足约束
    for k in range(S):
        base = N + k * (nt + N)
        A = np.zeros((N, nv))
        for i in range(N):
            for j in range(N):
                if i != j:
                    A[i, base + j * N + i] = -1.0   # 从 j 调入 i（增加 i 供给）
                    A[i, base + i * N + j] = +1.0   # 从 i 调出 j（减少 i 供给）
            A[i, base + nt + i] = -1.0              # 应急外购增加供给
            A[i, i] = -1.0                          # 本地产能 y_i 提供供给
            b_ub.append(-D[k, i])                   # 供给 >= 需求
        A_ub.append(A)

    if use_cvar:
        # s_k >= 总成本_k - eta  =>  (成本_k - eta) - s_k <= 0
        for k in range(S):
            base = N + k * (nt + N)
            A = np.zeros((1, nv))
            A[0, :N] = CAP_COST                 # 建产能成本 (每场景相同)
            for j in range(N):
                for i in range(N):
                    if i != j:
                        A[0, base + j * N + i] = SHIP[j, i]
            for i in range(N):
                A[0, base + nt + i] = P_EMERG
            A[0, eta_idx] = -1.0
            A[0, s_idx0 + k] = -1.0
            A_ub.append(A)
            b_ub.append(0.0)

    A_ub = np.vstack(A_ub)
    bounds = [(0, None)] * nv
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("SAA 主问题求解失败: " + res.message)

    y = res.x[:N]
    info = {"obj": res.fun, "success": res.success}
    if use_cvar:
        info["eta"] = res.x[n_main]
        info["cvar_in_sample"] = res.x[n_main] + res.x[n_main + 1: n_main + 1 + S].sum() / ((1 - alpha) * S)
    return y, info


# ============================================================
# 3. 风险度量：样本 CVaR
# ============================================================
def cvar(costs, alpha=0.95):
    """样本 CVaR_alpha（最差 (1-alpha) 尾部的平均成本）。"""
    q = np.quantile(costs, alpha)          # VaR 分位点
    tail = costs[costs >= q]
    return float(tail.mean()), float(q)


# ============================================================
# 4. 主流程
# ============================================================
print("=" * 68)
print("Day29 · CVaR 风险厌恶两阶段随机规划")
print("=" * 68)

S_TRAIN, S_TEST = 300, 2000
D_train = sample_demand(S_TRAIN)
D_test = sample_demand(S_TEST)

# --- 4.1 风险中性 SAA（Day28 的复现，作为基线）---
t0 = time.time()
y_risk_neutral, _ = solve_saa(D_train, risk=None)
cost_rn = evaluate_policy(y_risk_neutral, D_test)
mean_rn, var_rn = cost_rn.mean(), np.quantile(cost_rn, 0.95)
cvar_rn, _ = cvar(cost_rn, 0.95)
print(f"\n[风险中性 SAA] y* = {np.round(y_risk_neutral, 2)}  总产能 {y_risk_neutral.sum():.2f}")
print(f"  样本外期望成本 {mean_rn:,.2f} 元/天 | VaR95 {var_rn:,.2f} | CVaR95 {cvar_rn:,.2f}")

# --- 4.2 确定性等价：按需求均值建（DET 基线）---
y_det = MU.copy()
cost_det = evaluate_policy(y_det, D_test)
mean_det, var_det = cost_det.mean(), np.quantile(cost_det, 0.95)
cvar_det, _ = cvar(cost_det, 0.95)
print(f"\n[确定性等价 DET(按均值)] y = {y_det}  总产能 {y_det.sum():.2f}")
print(f"  样本外期望成本 {mean_det:,.2f} | VaR95 {var_det:,.2f} | CVaR95 {cvar_det:,.2f}")

# --- 4.3 扫 lam，生成"期望成本 vs CVaR"有效前沿 ---
lams = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
frontier = []
for lam in lams:
    if lam == 0.0:
        y = solve_saa(D_train, risk=None)[0]
    else:
        y, _ = solve_saa(D_train, risk="cvar_lp", alpha=0.95, lam=lam)
    costs = evaluate_policy(y, D_test)
    m, cv = costs.mean(), cvar(costs, 0.95)[0]
    frontier.append((lam, y.copy(), m, cv))
    print(f"  lam={lam:5.1f}  y={np.round(y,2)}  总产能{y.sum():5.2f}  "
          f"期望{m:,.1f}  CVaR95 {cv:,.1f}")

# --- 4.4 最优风险厌恶解（选取 CVaR 收益最大的一个 lam）---
# 以"每亿元期望成本的 CVaR 削减"为准，人工选一个折中点 lam=2.0 作展示
y_cvar, info_cvar = solve_saa(D_train, risk="cvar_lp", alpha=0.95, lam=2.0)
cost_cvar = evaluate_policy(y_cvar, D_test)
mean_cvar, var_cvar = cost_cvar.mean(), np.quantile(cost_cvar, 0.95)
cvar_cvar, _ = cvar(cost_cvar, 0.95)
print(f"\n[风险厌恶 SAA (lam=2, alpha=0.95)] y* = {np.round(y_cvar,2)}  总产能 {y_cvar.sum():.2f}")
print(f"  样本外期望成本 {mean_cvar:,.2f} | VaR95 {var_cvar:,.2f} | CVaR95 {cvar_cvar:,.2f}")

# --- 4.5 尾部对照：三个方案在"最惨 5% 场景"的表现 ---
print("\n" + "-" * 68)
print(f"{'方案':<22}{'总产能':>8}{'期望成本':>12}{'VaR95':>11}{'CVaR95':>11}")
for name, y in [("确定性等价 DET", y_det),
                ("风险中性 SAA", y_risk_neutral),
                ("CVaR 风险厌恶 SAA", y_cvar)]:
    cs = evaluate_policy(y, D_test)
    print(f"{name:<22}{y.sum():>8.2f}{cs.mean():>12,.1f}"
          f"{np.quantile(cs,0.95):>11,.1f}{cvar(cs,0.95)[0]:>11,.1f}")

d_mean = mean_cvar - mean_rn
d_cvar = cvar_rn - cvar_cvar
print(f"\n风险厌恶 vs 风险中性：期望成本 +{d_mean:,.1f} 元/天，"
      f"但 CVaR95 降 {d_cvar:,.1f} 元/天")
ratio = cvar_det - cvar_cvar
print(f"风险厌恶 vs 按均值建：CVaR95 降 {ratio:,.1f} 元/天，"
      f"尾部风险 {(cvar_det-cvar_cvar)/cvar_det*100:.1f}% 被消掉")

# --- 4.6 敏感性 A：alpha 变大（越怕尾部）产能怎么变 ---
print("\n[敏感性A] 置信水平 alpha 对决策的影响")
alpha_rows = []
for a in [0.50, 0.75, 0.90, 0.95, 0.99]:
    y, _ = solve_saa(D_train, risk="cvar_lp", alpha=a, lam=2.0)
    cs = evaluate_policy(y, D_test)
    alpha_rows.append((a, y.copy(), y.sum(), cs.mean(), cvar(cs, a)[0]))
    print(f"  alpha={a:.2f}  总产能{y.sum():5.2f}  期望{cs.mean():,.1f}  "
          f"CVaR_{a:.2f} {cvar(cs,a)[0]:,.1f}")

# --- 4.7 敏感性 B：需求波动加大 ---
print("\n[敏感性B] 需求波动 σ 放大对总产能与 CVaR 的影响")
sig_rows = []
for scale in [0.5, 1.0, 1.5, 2.0]:
    sig_s = np.array([9.0, 6.0, 7.5]) * scale
    Dtr = sample_demand(S_TRAIN, sigma=sig_s)
    Dte = sample_demand(S_TEST, sigma=sig_s)
    y, _ = solve_saa(Dtr, risk="cvar_lp", alpha=0.95, lam=2.0)
    cs = evaluate_policy(y, Dte)
    sig_rows.append((scale, y.sum(), cs.mean(), cvar(cs, 0.95)[0]))
    print(f"  σ×{scale:.1f}  总产能{y.sum():5.2f}  期望{cs.mean():,.1f}  "
          f"CVaR95 {cvar(cs,0.95)[0]:,.1f}")

# ============================================================
# 5. 断言（自检）
# ============================================================
assert np.all(y_risk_neutral >= -1e-6), "产能必须非负"
assert np.all(y_cvar >= -1e-6), "产能必须非负"
# 风险厌恶：CVaR 更低（尾部更安全）
assert cvar_cvar <= cvar_rn + 1e-6, "风险厌恶解的 CVaR 不应高于风险中性"
# 风险厌恶通常要付出更高期望成本（或至少不低太多）
assert mean_cvar >= mean_rn - 1e-6, "风险厌恶不应比风险中性更省期望成本"
# 最怕尾巴的方案（alpha 更大）总产能不减少
assert alpha_rows[-1][2] >= alpha_rows[0][2] - 1e-6, "alpha 越大产能应越多"
# 波动越大产能越多
assert sig_rows[-1][1] >= sig_rows[0][1] - 1e-6, "波动越大产能应越多"
# CVaR >= 期望成本（尾部一定不比平均乐观）
assert cvar_rn >= mean_rn - 1e-6 and cvar_cvar >= mean_cvar - 1e-6, "CVaR>=均值"
# 有效前沿：lam 越大 CVaR 越低（允许极小数值波动）
assert frontier[-1][3] <= frontier[0][3] + 1e-6, "lam 越大尾部风险应越低"
print("\n[断言] 全部通过 ✓")

# ============================================================
# 6. 画图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上：有效前沿（期望成本 vs CVaR）
ax = axes[0, 0]
ms = [f[2] for f in frontier]
cvs = [f[3] for f in frontier]
ax.plot(ms, cvs, "o-", color="#2c7fb8", lw=2, ms=7)
for lam, yy, m, cv in frontier:
    ax.annotate(f"λ={lam:g}", (m, cv), textcoords="offset points",
                xytext=(6, 4), fontsize=8)
ax.scatter([mean_cvar], [cvar_cvar], s=180, marker="*",
           color="#d7191c", zorder=5, label="推荐折中解(λ=2)")
ax.set_xlabel("样本外期望成本（元/天）")
ax.set_ylabel("样本外 CVaR95（元/天，尾部风险）")
ax.set_title("有效前沿：多花期望成本买尾部安全")
ax.legend()
ax.grid(alpha=0.3)

# 右上：三方案 CVaR 对照柱
ax = axes[0, 1]
names = ["确定性\n等价DET", "风险中性\nSAA", "CVaR\n风险厌恶SAA"]
vals_m = [mean_det, mean_rn, mean_cvar]
vals_c = [cvar_det, cvar_rn, cvar_cvar]
x = np.arange(3)
ax.bar(x - 0.2, vals_m, 0.4, label="期望成本", color="#4daf4a")
ax.bar(x + 0.2, vals_c, 0.4, label="CVaR95", color="#e41a1c")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("元/天")
ax.set_title("三个方案：期望成本 vs 尾部风险")
ax.legend(); ax.grid(alpha=0.3, axis="y")

# 左下：成本分布（三方案直方图）
ax = axes[1, 0]
for name, cs, col in [("按均值建 DET", cost_det, "#999999"),
                      ("风险中性 SAA", cost_rn, "#4daf4a"),
                      ("CVaR 风险厌恶", cost_cvar, "#e41a1c")]:
    ax.hist(cs, bins=60, alpha=0.5, label=name, color=col, density=True)
ax.axvline(cvar_det, color="#999999", ls="--", lw=1)
ax.axvline(cvar_cvar, color="#e41a1c", ls="--", lw=1)
ax.set_xlabel("单日总成本（元/天）")
ax.set_ylabel("概率密度")
ax.set_title("成本分布对照：风险厌恶把右尾收窄")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 右下：敏感性（alpha 与 σ 对总产能）
ax = axes[1, 1]
alphas = [r[0] for r in alpha_rows]
cap_a = [r[2] for r in alpha_rows]
ax.plot(alphas, cap_a, "o-", color="#984ea3", lw=2, label="α 对总产能")
ax2 = ax.twinx()
scales = [r[0] for r in sig_rows]
cap_s = [r[1] for r in sig_rows]
ax2.plot(scales, cap_s, "s--", color="#ff7f00", lw=2, label="σ 倍数对总产能")
ax.set_xlabel("α（紫色）/ σ 放大倍数（橙色，右轴）")
ax.set_ylabel("总产能（万瓶/天）", color="#984ea3")
ax2.set_ylabel("总产能（万瓶/天）", color="#ff7f00")
ax.set_title("越怕尾部 / 波动越大 → 产能越足")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
png = os.path.join(OUT_DIR, "cvar_two_stage_stochastic.png")
plt.savefig(png, dpi=130, bbox_inches="tight")
print(f"\n图已保存: {png}")
print(f"总耗时 {time.time()-t0:.1f}s")
print("=" * 68)
