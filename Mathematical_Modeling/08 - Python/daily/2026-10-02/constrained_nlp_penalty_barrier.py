#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L6 优化模型 / L8 算法（约束非线性优化：KKT / 罚函数 / 对数障碍内点法）
anchor:  Day34 <- Day33(无约束 GD/拟牛顿) / Day5(NLP 建模) / Day26(LP 对偶) / Day4(ILP)
goal:    用一道「三作物种植面积」决策，手写罚函数法与对数障碍内点法求解带非线性
         约束的 NLP，并用 KKT 条件、影子价格、独立求解器(SLSQP)三重交叉验证；
         回答：有约束时最优解长什么样、为什么卡边界、约束值多少钱、机器怎么逼出解。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
        "Mathematical_Modeling/08 - Python/daily/2026-10-02/constrained_nlp_penalty_barrier.py"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 注册系统中文字体，避免图中中文乱码 ----
import matplotlib.font_manager as fm
for _cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
              "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei"]:
    try:
        fm.findfont(fm.FontProperties(family=_cand), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261002)

# =====================================================================
# 1. 真实决策问题：三作物种植面积（带非线性约束的 NLP）
#    变量 x = (x1, x2, x3) 为三种作物面积（亩）
#    目标：最小化「机会成本」= (各作物理想利润 - 实际利润)
#          利润含规模效应二次项（种得越多、单位利润递减）
#    约束：
#      g1(x) = 土地约束     sum(x) - 50 <= 0   （线性）
#      g2(x) = 用水约束     sum(a_i x_i^2) - 1000 <= 0 （非线性，边际耗水递增）
#      x >= 0 （下界，用 -x_i <= 0 表示）
# =====================================================================

TOTAL_LAND = 50.0          # 总土地（亩）
WATER_CAP = 450.0          # 总水量上限（设计为使土地与用水在最优处同时卡脖子）
WATER_COEF = np.array([0.35, 0.6, 0.45])   # 各作物单位面积耗水系数（二次）

# 规模效应二次成本（凸），保证 f 凸且有内部最优
SCALE = np.array([1.1, 1.5, 0.9])
PRICE = np.array([120.0, 95.0, 140.0])     # 基准单位利润


def f(x):
    """目标：最小化机会成本（凸）。"""
    x = np.asarray(x, dtype=float)
    profit = PRICE * x - SCALE * x ** 2
    return float(-np.sum(profit))


def grad_f(x):
    x = np.asarray(x, dtype=float)
    return -(PRICE - 2.0 * SCALE * x)


def g(x):
    """不等式约束向量 g(x) <= 0。"""
    x = np.asarray(x, dtype=float)
    g1 = np.sum(x) - TOTAL_LAND
    g2 = np.sum(WATER_COEF * x ** 2) - WATER_CAP
    g3 = -x  # 非负：-x_i <= 0
    return np.concatenate([[g1, g2], g3])


def grad_g(x):
    """约束梯度，返回 shape (5, 3)。"""
    x = np.asarray(x, dtype=float)
    J = np.zeros((5, 3))
    J[0, :] = 1.0                        # 土地
    J[1, :] = 2.0 * WATER_COEF * x       # 用水
    J[2:, :] = -np.eye(3)                # -x_i
    return J


def numeric_grad(func, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    gnum = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        gnum[i] = (func(x + e) - func(x - e)) / (2 * h)
    return gnum


# 解析梯度自检（优化课铁律：先验梯度）
_x0 = np.array([10.0, 10.0, 10.0])
assert np.allclose(grad_f(_x0), numeric_grad(f, _x0), atol=1e-6), "解析梯度与数值梯度不一致！"
J_num = np.vstack([numeric_grad(lambda x, i=i: g(x)[i], _x0) for i in range(5)])
assert np.allclose(grad_g(_x0), J_num, atol=1e-6), "约束雅可比与数值不一致！"
print("[自检] 目标解析梯度 == 数值梯度，约束雅可比 == 数值雅可比 ✓")


# =====================================================================
# 2. 无约束最优（贪心理想解）：无视约束，看它有多"不可行"
# =====================================================================
def unconstrained_min():
    # f 为凸二次，驻点解析：x_i = (PRICE_i - 2*? ) 直接解 PRICE_i*x - SCALE_i*x^2 的极值
    # d(profit)/dx = PRICE - 2*SCALE*x = 0  => x = PRICE/(2*SCALE)
    return PRICE / (2.0 * SCALE)


x_unc = unconstrained_min()
print("\n[实验1] 无约束（贪心）最优:", np.round(x_unc, 3))
print("        无约束解的约束值 g(x) =", np.round(g(x_unc), 3),
      "  ->", "违反约束!" if np.any(g(x_unc) > 1e-9) else "可行")


# =====================================================================
# 3. 罚函数法（外点）：min f(x) + rho * sum(max(0,g)^2)，rho 倍增
#    每轮用「投影梯度 + 回溯」在罚目标上下降，并保持可行投影
# =====================================================================
def project_to_feasible(x):
    """把点投影回可行域（近似：先截断非负，再等比缩放土地/用水）。"""
    x = np.maximum(x, 0.0)
    s = np.sum(x)
    if s > TOTAL_LAND:
        x = x * (TOTAL_LAND / s)
    # 用水约束：若超限，按比例缩小到刚好满足
    w = np.sum(WATER_COEF * x ** 2)
    if w > WATER_CAP:
        x = x * np.sqrt(WATER_CAP / w)
    return np.maximum(x, 0.0)


def _bfgs_inner(obj, grad_obj, x0, iters=200, tol=1e-10):
    """手写 BFGS（同 Day33）求无约束极小——用作罚函数/障碍子问题的内层求解器。"""
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    H = np.eye(n)
    gk = grad_obj(x)
    for _ in range(iters):
        if np.linalg.norm(gk) < tol:
            break
        p = -H @ gk
        # Armijo 回溯线搜索
        step, c1 = 1.0, 1e-4
        fk = obj(x)
        for _ls in range(50):
            if obj(x + step * p) <= fk + c1 * step * (gk @ p):
                break
            step *= 0.5
        xn = x + step * p
        gn = grad_obj(xn)
        s = xn - x
        y = gn - gk
        sy = s @ y
        if sy > 1e-12:                     # 保正定
            rho = 1.0 / sy
            I = np.eye(n)
            M = I - rho * np.outer(s, y)
            H = M @ H @ M.T + rho * np.outer(s, s)
        if np.linalg.norm(xn - x) < 1e-12:
            x = xn; break
        x, gk = xn, gn
    return x


def penalty_method(x_init, rhos=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)):
    """外点罚函数：min f(x) + rho*sum(max(0,g)^2)，rho 倍增；内层手写 BFGS。"""
    x = np.asarray(x_init, dtype=float).copy()
    hist = []
    for rho in rhos:
        def pen(xx, rho=rho):
            return f(xx) + rho * np.sum(np.maximum(0.0, g(xx)) ** 2)

        def gpen(xx, rho=rho):
            gg = np.maximum(0.0, g(xx))
            return grad_f(xx) + rho * 2.0 * (gg[:, None] * grad_g(xx)).sum(axis=0)

        x = _bfgs_inner(pen, gpen, x)
        hist.append((rho, x.copy(), f(x), np.max(g(x))))
    return x, hist


x_pen, pen_hist = penalty_method(np.array([6.0, 3.0, 6.0]))
print("\n[实验3a] 罚函数法（外点）最优:", np.round(x_pen, 4),
      " 目标:", round(f(x_pen), 4))
print("         约束违反 max(g) =", f"{np.max(g(x_pen)):.3e}")


# =====================================================================
# 4. 对数障碍法（内点）：min f(x) - t * sum(log(-g))，t 缩减
#    从严格内点出发，内层手写 BFGS；每轮只接受严格可行解
# =====================================================================
def strict_interior_point():
    return np.array([6.0, 3.0, 6.0])  # 已验证严格可行（土地<50、用水<450、全正）


def log_barrier_method(x_init, ts=(10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), iters=4000):
    """内点对数障碍：min f(x) - t*sum(log(-g))，t 缩减。
    手写「可行起点 + 自适应步长梯度下降」：只接受严格可行且目标下降的步，
    因而全程留在可行域内部（这是内点法的核心保证）。"""
    x = np.asarray(x_init, dtype=float).copy()
    hist = []
    for t in ts:
        def obj(xx, t=t):
            return f(xx) - t * np.sum(np.log(-g(xx)))

        def gob(xx, t=t):
            gv = g(xx)
            return grad_f(xx) - t * np.sum(grad_g(xx) / gv[:, None], axis=0)

        step = 1.0
        for _ in range(iters):
            d = gob(x)
            s = step
            for _ls in range(60):
                xn = x - s * d
                if np.all(g(xn) < -1e-12) and obj(xn) < obj(x):
                    break
                s *= 0.5
            else:
                break
            if np.linalg.norm(xn - x) < 1e-13:
                x = xn; break
            x = xn
            step = min(s * 1.5, 10.0)
        hist.append((t, x.copy(), f(x), np.max(g(x))))
    return x, hist


x_bar, bar_hist = log_barrier_method(strict_interior_point())
print("[实验3b] 对数障碍法（内点）最优:", np.round(x_bar, 4),
      " 目标:", round(f(x_bar), 4))
print("         约束违反 max(g) =", f"{np.max(g(x_bar)):.3e}")


# =====================================================================
# 5. 独立对照：scipy SLSQP（仅用于验证，不作教学主体）
# =====================================================================
try:
    from scipy.optimize import minimize as sp_minimize
    cons = [
        {"type": "ineq", "fun": lambda x: -g(x)[0]},   # 土地 g1 <= 0  -> -g1 >= 0
        {"type": "ineq", "fun": lambda x: -g(x)[1]},   # 用水 g2 <= 0
        {"type": "ineq", "fun": lambda x: x[0]},       # x >= 0
        {"type": "ineq", "fun": lambda x: x[1]},
        {"type": "ineq", "fun": lambda x: x[2]},
    ]
    res = sp_minimize(f, x0=np.array([6.0, 3.0, 6.0]), jac=grad_f,
                      constraints=cons, bounds=[(0, None)] * 3, method="SLSQP",
                      options={"maxiter": 500, "ftol": 1e-12})
    x_ref = res.x
    print("\n[对照] scipy SLSQP 最优:", np.round(x_ref, 4), " 目标:", round(f(x_ref), 6))
except Exception as e:
    x_ref = None
    print("\n[对照] scipy SLSQP 不可用:", e)


# =====================================================================
# 6. KKT 条件核验（在最优解处）
# =====================================================================
def kkt_check(x_star, x_ref=None):
    gs = g(x_star)
    active = gs > -1e-6          # 活动（紧）约束集合
    J = grad_g(x_star)
    # 用最小二乘解活动约束的乘子：grad_f + sum mu_i grad_g_i = 0
    A = J[active].T
    rhs = -grad_f(x_star)
    mu_active, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    resid = np.linalg.norm(A @ mu_active - rhs)
    names = ["土地", "用水", "x1>=0", "x2>=0", "x3>=0"]
    print("\n[实验2/KKT] 在罚函数解处核验 KKT：")
    for i, nm in enumerate(names):
        tag = "紧" if active[i] else "松"
        mu = mu_active[list(np.where(active)[0]).index(i)] if active[i] else 0.0
        print(f"    {nm:6s} g={gs[i]:+8.4f}  {tag}  影子价格 mu≈{mu:8.4f}")
    print(f"    平稳性残差 ||grad_f + sum mu grad_g|| = {resid:.3e}")
    return names, gs, active, mu_active, resid


names, gs_pen, active_pen, mu_pen, resid_pen = kkt_check(x_pen)


# =====================================================================
# 7. 影子价格实测：把土地/用水约束各放宽 1 单位，看目标改善
# =====================================================================
def solve_with_relax(land, water):
    def g_rel(x):
        return np.concatenate([[np.sum(x) - land], [np.sum(WATER_COEF * x ** 2) - water], -x])
    cons = [
        {"type": "ineq", "fun": lambda x: -g_rel(x)[0]},
        {"type": "ineq", "fun": lambda x: -g_rel(x)[1]},
    ] + [{"type": "ineq", "fun": lambda x, i=i: x[i]} for i in range(3)]
    r = sp_minimize(f, x0=np.array([6.0, 3.0, 6.0]), jac=grad_f,
                    constraints=cons, bounds=[(0, None)] * 3, method="SLSQP",
                    options={"maxiter": 500, "ftol": 1e-12})
    return r.x, f(r.x)


base_land, base_water = TOTAL_LAND, WATER_CAP
x_b, f_b = solve_with_relax(base_land, base_water)
x_l, f_l = solve_with_relax(base_land + 1.0, base_water)
x_w, f_w = solve_with_relax(base_land, base_water + 1.0)
# 影子价格 = -d(最小目标)/d(资源上限) = (f_b - f_l)
mu_land_real = f_b - f_l
mu_water_real = f_b - f_w
print("\n[实验2] 影子价格实测（放宽 1 单位，目标改善量）：")
print(f"    土地 +1 亩 -> 目标改善 {mu_land_real:+.6f}")
print(f"    用水 +1 吨 -> 目标改善 {mu_water_real:+.6f}")


# =====================================================================
# 8. 收敛证据与交叉验证
# =====================================================================
print("\n[收敛证据] 罚函数 rho 序列：")
for rho, xh, fh, vh in pen_hist:
    print(f"    rho={rho:>9.1f}  x={np.round(xh,3)}  f={fh:9.4f}  max_g={vh:+.2e}")
print("[收敛证据] 对数障碍 t 序列：")
for t, xh, fh, vh in bar_hist:
    print(f"    t={t:>6.3f}  x={np.round(xh,3)}  f={fh:9.4f}  max_g={vh:+.2e}")

f_pen, f_bar = f(x_pen), f(x_bar)
print("\n[交叉验证] 罚函数解 vs 对数障碍解 最大分量差 =",
      f"{np.max(np.abs(x_pen - x_bar)):.3e}")
if x_ref is not None:
    print("[交叉验证] 两解 vs SLSQP 参考解 最大分量差 =",
          f"pen {np.max(np.abs(x_pen - x_ref)):.3e}, bar {np.max(np.abs(x_bar - x_ref)):.3e}")

# ---- 断言 ----
assert np.all(g(x_pen) <= 1e-4), "罚函数解不满足约束"
assert abs(np.sum(x_bar) - TOTAL_LAND) < 1.0, "对数障碍解未逼近土地边界"
assert abs(f_pen - f_bar) < 0.1, "两法目标不一致（应收敛到同一 KKT 点）"
if x_ref is not None:
    assert np.allclose(x_pen, x_ref, atol=1e-3), "罚函数解与参考解偏差过大"
    assert np.allclose(x_bar, x_ref, atol=1e-2), "对数障碍解与参考解偏差过大"
assert resid_pen < 1e-4, "KKT 平稳性残差过大"
assert np.any(g(x_unc) > 1e-9), "无约束解居然可行？请检查数据设计"
print("\n[断言] 全部通过 ✓")


# =====================================================================
# 9. 绘图
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

# (1) 无约束 vs 带约束 vs 两法 面积对照
ax = axes[0, 0]
labels = ["x1", "x2", "x3"]
w = 0.2
idx = np.arange(3)
ax.bar(idx - 1.5 * w, x_unc, w, label="无约束(贪心)", color="#d62728", alpha=0.85)
ax.bar(idx - 0.5 * w, x_pen, w, label="罚函数(外点)", color="#1f77b4", alpha=0.85)
ax.bar(idx + 0.5 * w, x_bar, w, label="对数障碍(内点)", color="#2ca02c", alpha=0.85)
if x_ref is not None:
    ax.bar(idx + 1.5 * w, x_ref, w, label="SLSQP 参考", color="#ff7f0e", alpha=0.85)
ax.set_xticks(idx); ax.set_xticklabels(labels)
ax.set_ylabel("面积（亩）")
ax.set_title("① 无约束贪心 vs 带约束最优：约束把解顶回可行域")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# (2) 影子价格
ax = axes[0, 1]
ax.bar(["土地\n(+1亩)", "用水\n(+1吨)"], [mu_land_real, mu_water_real],
       color=["#8c564b", "#17becf"], alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
for i, v in enumerate([mu_land_real, mu_water_real]):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
ax.set_ylabel("目标改善量（影子价格）")
ax.set_title("② 影子价格：该先松哪个约束")
ax.grid(alpha=0.3, axis="y")

# (3) 罚函数收敛：rho↑ -> 违反量↓
ax = axes[1, 0]
rhos = [h[0] for h in pen_hist]
viol = [max(0.0, h[3]) for h in pen_hist]
ax.plot(range(len(rhos)), viol, "o-", color="#1f77b4", label="约束违反 max(g)")
ax.set_xticks(range(len(rhos)))
ax.set_xticklabels([f"{r:g}" for r in rhos], rotation=45)
ax.set_xlabel("罚参数 rho（倍增）"); ax.set_ylabel("最大约束违反")
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_title("③a 罚函数：rho 越大越守规矩")
ax.grid(alpha=0.3, which="both")

# (4) 对数障碍收敛：t↓ -> 逼近边界
ax = axes[1, 1]
ts = [h[0] for h in bar_hist]
feas_gap = np.abs([h[3] for h in bar_hist])
ax.plot(range(len(ts)), feas_gap, "s-", color="#2ca02c", label="到边界的距离 |max(g)|")
ax.set_xticks(range(len(ts)))
ax.set_xticklabels([f"{t:g}" for t in ts], rotation=45)
ax.set_xlabel("障碍参数 t（缩减）"); ax.set_ylabel("到边界距离")
ax.set_yscale("log")
ax.set_title("③b 对数障碍：t 越小越贴边")
ax.grid(alpha=0.3, which="both")

fig.suptitle("Day34 约束非线性优化：KKT / 罚函数 / 对数障碍内点法", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = OUT / "constrained_nlp_penalty_barrier.png"
fig.savefig(out_png, dpi=130)
print(f"\n[图] 已保存: {out_png}")
print("\n完成。")
