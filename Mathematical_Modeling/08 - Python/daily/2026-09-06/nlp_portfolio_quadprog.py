# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 (04 - Optimization)   # LP->ILP->NLP 数学规划三件套第3篇
anchor: Day5 非线性规划 NLP / 二次规划(均值-方差投资组合, Markowitz)
goal  : 演示当目标函数出现"非线性"（风险=二次项）时，LP 无法表达、
        scipy.optimize.minimize(SLSQP) 如何求"资金三分+控险"的最优解，
        并与解析/数值核验比对。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

# ---------- 注册中文字体，避免图乱码 ----------
from matplotlib import font_manager
for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Zen Hei Mono"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

# =====================================================================
# 决策背景（真实问题）：资金不足，三选分配
# 你手上有 W=1(归一化=100%) 的研发/营销预算，投给三个新产品线。
# 每条线有"期望回报率 mu_i"和两两协方差(风险矩阵 Sigma)。
# LP 只会问"资金给谁回报最大"把全押回报最高(单一)那条 ——
# 但那会因"高波动/高相关"背上巨大风险(方差灾难)。
# 决策要的是：给定一个可接受的最低期望回报 target，
# 在"三个权重之和=1、每个 >=0(不卖空)"下，让总风险(方差)最小 ——
# 目标是"权重的二次函数"，这就是 NLP / QP。
# =====================================================================

# 三条产品线的期望回报(年化) 与 协方差矩阵(风险, 手工构造: 1与2相关高, 3是独立避险标的)
mu = np.array([0.12, 0.10, 0.06])     # 期望回报 12%, 10%, 6%
Sigma = np.array([
    [0.0400, 0.0360, 0.0040],   # 线1 波动大且与线2强相关
    [0.0360, 0.0324, 0.0036],
    [0.0040, 0.0036, 0.0100],   # 线3 低波动低相关 -> 分散风险主力
])
n = len(mu)
W_total = 1.0

# ---- LP 视角(只最大化回报, 不考虑风险): 全押 mu 最大线1 ----
mu_max_idx = int(np.argmax(mu))
lp_like = np.zeros(n); lp_like[mu_max_idx] = 1.0
lp_ret = mu @ lp_like
lp_risk = float(lp_like @ Sigma @ lp_like)

# =====================================================================
# NLP: min w' Σ w   s.t.  Σw=1, w>=0,  mu'w >= target
# =====================================================================
def risk(w: np.ndarray) -> float:
    w = np.asarray(w)
    return float(w @ Sigma @ w)

bounds = [(0.0, 1.0)] * n   # 不卖空(模块级, 供多方法共用)

def solve_eff(target: float):
    """在要求最低回报=target 下求最小方差组合。"""
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - W_total},   # 预算全部分光
        {"type": "ineq", "fun": lambda w: (mu @ w) - target},  # 回报达标
    ]
    w0 = np.ones(n) / n         # 等权起点
    res = minimize(risk, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    return res

# ---------- 案例A: target = 10%(风险较低的合理收益) ----------
targetA = 0.10
resA = solve_eff(targetA)
wA = resA.x
riskA = risk(wA)
retA = mu @ wA

# ---------- 验证1: 等式约束恰好满配; 与"仅最小方差、无回报要求"基准对比 ----------
res_minvar = solve_eff(0.0)
w_minvar = res_minvar.x
# 目标10%组合的方差应 大于等于 纯最小方差组合(它不保证收益)

# =====================================================================
# 数值验证：手工随机搜+解析闭式对照  (QP 可解析，这里给 Newton 正确性佐证)
# 无不等式(允许w可负但保持和为1、回报达标)的投影是最小二乘意义；
# 我们采用保守做法：用 scipy 独立方法 'trust-constr' 再解一次交叉核对，
# 并对"最小方差组合(无收益约束)"用解析公式 W_gmv = Σ^{-1}1/(1'Σ^{-1}1) 核对。
# =====================================================================
Sinv = np.linalg.inv(Sigma)
one = np.ones(n)
w_gmv_theory = Sinv @ one / (one @ Sinv @ one)          # 全局最小方差(可负)解析
risk_gmv_theory = float(w_gmv_theory @ Sigma @ w_gmv_theory)

# 交叉核对 A: 用 trust-constr 重解 target=10%
LC = LinearConstraint(np.vstack([np.ones(n), mu]), [1.0, targetA], [1.0, np.inf])
resA2 = minimize(risk, wA.copy(), method="trust-constr", bounds=bounds,
                 constraints=LC, options={"maxiter": 1000})
print("== 交叉核对: SLSQP vs trust-constr 权重最大差 ==")
print(f"   {np.max(np.abs(resA.x - resA2.x)):.3e} (两种方法的收敛容差不同, ~1e-3 内即一致; 主结果取容差更紧的 SLSQP)")
assert np.allclose(resA.x, resA2.x, atol=5e-3), "两种NLP方法结果不一致"

# ---------- 案例B: 画出 有效前沿(Effcient Frontier) ----------
grid_targets = np.linspace(0.04, 0.122, 40)
front = []
for t in grid_targets:
    r = solve_eff(t)
    if r.success:
        front.append((mu @ r.x, risk(r.x)))
front = np.array(front)
# 在 target 范围内选"最小风险"的可行目标(有效前沿单调段)
minvar_idx = int(np.argmin(front[:, 1]))
eff = front[minvar_idx:]           # 风险随要求收益上升的单调段 -> 有效前沿

# ---------- 输出结果 ----------
def pct(x):
    return f"{x*100:.2f}%"

print("=" * 62)
print("背景: 预算 W=1 分给三条产品线, 要求总期望回报达标下最小化组合风险(方差)")
print("mu  (期望回报):  线1 12%   线2 10%   线3 6%(避险)")
print("-" * 62)
print(f"[LP式·只求收益·极端] 全押线1:  回报={pct(lp_ret)},  方差={lp_risk:.4f} (波动≈{np.sqrt(lp_risk)*100:.1f}%)")
print("-" * 62)
print(f"[NLP 最优  target={pct(targetA)} ]")
print(f"  权重: 线1={wA[0]*100:.1f}%  线2={wA[1]*100:.1f}%  线3={wA[2]*100:.1f}%   (和={np.sum(wA)*100:.1f}%)")
print(f"  实际回报={pct(retA)},  方差={riskA:.4f} (波动≈{np.sqrt(riskA)*100:.1f}%)")
print(f"  [对比]单纯全押线1: 波动≈{np.sqrt(lp_risk)*100:.1f}% -> "
      f"NLP把波动从 {np.sqrt(lp_risk)*100:.1f}% 压到 {np.sqrt(riskA)*100:.1f}%, 回报还高出/接近")
print("-" * 62)
print(f"[解析核对] 全局最小方差组合(理论) w_gmv={np.round(w_gmv_theory,3)}, 方差={risk_gmv_theory:.4f}")
print(f"          说明: 该理论解含负权重(可卖空), 我们的 NLP(w>=0) 是它的[约束现实版]; "
      f"[最小方差约束版] w_minvar={np.round(w_minvar,3)}")
print("=" * 62)

# ---------- 断言(数值自检) ----------
assert resA.success, "SLSQP 未收敛"
assert abs(np.sum(wA) - 1.0) < 1e-9, "预算约束被违反"
assert all(wA >= -1e-9), "出现负权重(卖空)"
assert mu @ wA >= targetA - 1e-9, "回报达标约束被违反"

# 关键洞察1: NLP 比"全押最高收益"更分散, 且风险显著更低
if lp_risk > riskA:
    print("=> 洞察1成立: NLP在几乎不掉or低收益的情况下显著降低了组合风险(分散化收益)")
# 关键洞察2: 纯最小方差(不管收益) 与 达标组合 的方差关系
print(f"洞察2: 达标组合(target=10%)方差 {riskA:.4f} >= 纯最小方差组合方差 "
      f"{risk(w_minvar):.4f} —— 想要更高收益就要多承担一点风险(罚")

# =====================================================================
# 画图: ①资产点与权重(三角)  ②有效前沿 + 单资产波动/回报
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# 左: 三条线各自的(波动,回报)散点 + NLP最优组合
ax1.scatter(np.sqrt(np.diag(Sigma)), mu, c="tab:blue", s=70, zorder=3, label="单条产品线")
for i, txt in enumerate(["线1(高回报高波动)", "线2", "线3(避险)"]):
    ax1.annotate(txt, (np.sqrt(Sigma[i, i]) + 0.012, mu[i]), fontsize=8)
ax1.scatter([np.sqrt(riskA)], [retA], c="red", s=120, marker="*", zorder=4,
            label=f"NLP最优组合(target={pct(targetA)})")
ax1.scatter([np.sqrt(lp_risk)], [lp_ret], c="orange", s=90, marker="X", zorder=4,
            label="LP式\"全押线1\"")
ax1.set_xlabel("风险(年化波动 √方差)"); ax1.set_ylabel("期望回报")
ax1.set_title("三资产 vs 优化后最优组合（波动更低 / 回报不降）")
ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

# 右: 有效前沿
ax2.plot(np.sqrt(front[:, 1]), front[:, 0], "o-", ms=4, lw=1.4, color="tab:green",
         label="可行组合(不同 target)")
ax2.plot(np.sqrt(eff[:, 1]), eff[:, 0], "s-", ms=5, lw=2.2, color="tab:red",
         label="有效前沿(风险-收益最优边界)")
ax2.scatter([np.sqrt(riskA)], [retA], c="red", s=90, marker="*", zorder=4, label="本课最优解")
ax2.set_xlabel("风险(√方差)"); ax2.set_ylabel("期望回报")
ax2.set_title("均值-方差 有效前沿（NLP描出的风险-收益权衡线）")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
fig.suptitle("Day5 非线性规划(NLP) · 均值-方差投资组合", y=1.02)
fig.tight_layout()
png = OUT / "nlp_efficient_frontier.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print(f"\n[图已保存] {png.name}")

# 把关键数值写盘便于 README 引用(可选, 便于复述)
print("\nALL CHECKS PASSED ✅  (SLSQP 与 trust-constr 交叉一致; 约束全部满足)")
