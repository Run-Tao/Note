# -*- coding: utf-8 -*-
r"""
subject: mathematical-modeling
level:   L5 非线性规划 / 凸优化（二次规划 QP：投影梯度 PGD vs Frank-Wolfe 条件梯度）/ L7 数值算法
anchor:  Roadmap L5 → 非线性规划 → 「约束下怎么走一步」：投影梯度 vs Frank-Wolfe（LMO 只解一个 LP）
goal:    用一个真实的「四所村小撤并后巡回排课」决策，讲清「同一道 QP，换一种走法」差在哪里：
         ① 目标写「理想课时」而不是「课时越小越好」——否则最优解全贴底线，公平约束根本不起作用（本课返工最深的一处）；
         ② 投影梯度每步都要「投影到约束集」= 每步再解一个 QP；Frank-Wolfe 每步只解一个**线性**子问题（LMO），
            本次的 LMO 直接是 linprog，规模大时还能换成最短路 / 匹配；
         ③ FW 的解天然是**顶点的凸组合**（本实例 6 个顶点里只用到 3 个）；代价是 O(1/k) 而 PGD 线性收敛；
         ④ FW 的**方向间隙 g_k = ∇f·(x_k − s_k) ≥ f(x_k) − f***，是一个不需要知道最优值就能报的上界 —— 这是它真正值钱的地方。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error \
        "Mathematical_Modeling/08 - Python/daily/2026-11-13/day76_frank_wolfe_equity.py"
"""
import json
import os
import time
from itertools import combinations

import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()


def setup_cjk_font():
    cands = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in cands:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK = setup_cjk_font()

# ============================================================================
# 一、场景：云溪县四所村小撤并，中心校每天派「巡回车 + 2 名流动教师」跑四个点
#     决策：这一天在四个点各上几节课 x_i（连续变量，半节可以合并/拆开）。
#     约束：
#       ① 一天总课时 ≤ S = 76 节（88 节里匀出 12 节做车程与课间）
#       ② 每个点有「国家课程底线」D_i（低于它教学点挂黑牌）
#       ③ 镇里加的一条公平口径：东庄 + 北岭合计 = 30 节（撤并前两所大校的体面）
#     目标：四个点尽量都靠近「理想日课量」18 节 →  minimize ½ Σ (x_i − 18)²
# ============================================================================
NAMES = ["东庄", "南坪", "西庄", "北岭"]
STUDENTS = np.array([26, 10, 8, 6])
D = np.array([10.0, 14.0, 20.0, 8.0])        # 各点底线（节/天）
S = 76.0                                       # 一天总课时上限
G = np.array([1.0, 0.0, 0.0, 1.0])            # 公平线作用在东庄、北岭
H_CONST = 30.0                                 # 东庄 + 北岭 = 30
M_TARGET = np.array([18.0, 18.0, 18.0, 18.0])  # 四个点的理想日课量


def f(x):
    """½ Σ (x_i − m_i)²：离「理想日课量」越近越好（严格凸）。"""
    return float(0.5 * (((x - M_TARGET) ** 2).sum()))


def grad_f(x):
    return x - M_TARGET


# ============================================================================
# 二、LMO：线性最小化 oracle —— FW 的「全部知识」就藏在这里
#     🔴 返工最久的一处：LMO 必须在 **x 空间**建模。
#     一开始在 z = x − D 空间里写「z >= 0, Σz ≤ S − ΣD」，把「各自底线」误当成了
#     一个可以合并进总和的上界，可行域被削到只剩一小块 —— FW 收敛到一个
#     「看起来收敛、其实错了 3 倍」的点（f = 202 vs 68）。断言抓出来的。
# ============================================================================
def LMO(c):
    """min c·x s.t. x >= D, Σx <= S, x0 + x3 = H"""
    r = linprog(c, A_ub=np.ones((1, 4)), b_ub=np.array([S]),
                A_eq=G.reshape(1, -1), b_eq=np.array([H_CONST]),
                bounds=[(float(D[k]), None) for k in range(4)], method="highs")
    return None if not r.success else r.x.copy()


# ============================================================================
# 三、可行域的顶点（FW 的解一定是顶点的凸组合，本课要用它来讲「稀疏」）
# ============================================================================
def vertices():
    """顶点 = 4 个活跃约束的交点：从 {x0>=D0,...,x3>=D3, Σ=S, x0+x3=H} 里挑 4 个取等。"""
    verts = []
    for m in range(0, 5):
        for idx in combinations(range(4), m):
            for use_sum in (0, 1):
                for use_g in (0, 1):
                    rows, rhs = [], []
                    for i in idx:
                        r = np.zeros(4); r[i] = 1.0
                        rows.append(r); rhs.append(D[i])
                        # 用 x_i = D_i 表示该下界取等
                    if use_sum:
                        rows.append(np.ones(4)); rhs.append(S)
                    if use_g:
                        rows.append(G.copy()); rhs.append(H_CONST)
                    if len(rows) != 4:
                        continue
                    try:
                        sol = np.linalg.solve(np.array(rows), np.array(rhs))
                    except np.linalg.LinAlgError:
                        continue
                    if np.all(sol >= D - 1e-9) and sol.sum() <= S + 1e-9 \
                            and abs(G @ sol - H_CONST) < 1e-9:
                        verts.append(sol)
    uniq = []
    for v in verts:
        if not any(np.allclose(v, u, atol=1e-7) for u in uniq):
            uniq.append(v)
    return np.array(uniq)


VERTS = vertices()

# ============================================================================
# 四、算法一：Frank-Wolfe（开环步长 2/(k+2)）+ 方向间隙作为停止准则
# ============================================================================
def frank_wolfe(x0, max_iter=400, tol=1e-12):
    x = np.array(x0, dtype=float)
    hist = []
    for k in range(max_iter):
        g = grad_f(x)
        s = LMO(g)
        d = s - x
        gap = float(-g @ d)                  # g·(x − s) ≥ f(x) − f*
        hist.append({"k": k, "f": f(x), "gap": gap})
        if gap <= tol:
            break
        x = x + (2.0 / (k + 2.0)) * d
    return x, hist


X0 = np.array([D[0], D[1], D[2], H_CONST - D[0]])      # 东庄贴底、北岭补公平线的烂起点
X_FW, HIST_FW = frank_wolfe(X0)

# ============================================================================
# 五、算法二：投影梯度 —— 每一步都要「投影到 P 上」，这个投影自己就是一个 QP
# ============================================================================
def project(y):
    r = minimize(lambda v: float((v - y) @ (v - y)), np.clip(y, D, None),
                 jac=lambda v: 2 * (v - y), method="SLSQP",
                 bounds=[(float(D[k]), None) for k in range(4)],
                 constraints=[{"type": "ineq", "fun": lambda v: S - v.sum()},
                              {"type": "eq", "fun": lambda v: G @ v - H_CONST}],
                 options={"maxiter": 400, "ftol": 1e-14})
    return r.x


def projected_gradient(x0, L=1.0, max_iter=400):
    """L = 1（f 的 Hessian 是单位阵），步长 1/L = 1 → 线性收敛。"""
    x = np.array(x0, dtype=float)
    hist = []
    for k in range(max_iter):
        x_new = project(x - (1.0 / L) * grad_f(x))
        hist.append({"k": k, "f": f(x_new)})
        if np.linalg.norm(x_new - x) < 1e-13:
            x = x_new
            break
        x = x_new
    return x, hist


X_PG, HIST_PG = projected_gradient(X0)

# ============================================================================
# 六、裁判：精确解（高精度 SLSQP）+ 手工 KKT 复核
# ============================================================================
X_STAR = minimize(f, np.array([12.0, 18.0, 18.0, 18.0]), jac=grad_f, method="SLSQP",
                  bounds=[(float(D[k]), None) for k in range(4)],
                  constraints=[{"type": "ineq", "fun": lambda x: S - x.sum(),
                                "jac": lambda x: -np.ones(4)},
                               {"type": "eq", "fun": lambda x: G @ x - H_CONST, "jac": lambda x: G}],
                  options={"maxiter": 2000, "ftol": 1e-16}).x.copy()
F_STAR = f(X_STAR)


def check_kkt(x):
    """约束 x>=D、Σx<=S、x0+x3=H。KKT：∇f + λ_S·1 + λ_G·G − μ = 0，μ>=0，μ_i(x_i−D_i)=0。"""
    inactive = np.where(x > D + 1e-6)[0]
    if inactive.size == 0:
        return 0.0, (0.0, 0.0), True
    A_mat = -np.vstack([np.ones(inactive.size), G[inactive]]).T
    sol, *_ = np.linalg.lstsq(A_mat, grad_f(x)[inactive], rcond=None)
    lamS, lamG = float(sol[0]), float(sol[1])
    mu = grad_f(x) + lamS * np.ones(4) + lamG * G
    return float(mu.min()), (lamS, lamG), bool(np.all(mu > -1e-6))


KKT_MIN_MU, (LAM_S, LAM_G), KKT_OK = check_kkt(X_STAR)

# ============================================================================
# 七、FW 解 = 顶点的凸组合（讲「稀疏」的那个数）
# ============================================================================
def decompose(x, verts):
    A_eq = np.vstack([verts.T, np.ones(len(verts))])
    b_eq = np.concatenate([x, [1.0]])
    r = linprog(np.zeros(len(verts)), A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * len(verts), method="highs")
    return None if not r.success else r.x


LAM = decompose(X_FW, VERTS)
N_USED = int((LAM > 1e-6).sum()) if LAM is not None else None

# ============================================================================
# 八、公平线从 30 放到多少才开始「咬人」？—— 用 FW 的间隙扫一遍（本课增量）
# ============================================================================
def solve_for_H(h):
    """给定公平线 h，重解（x 空间），返回 (f*, x*)。"""
    r = minimize(f, np.array([12.0, 18.0, 18.0, 18.0]), jac=grad_f, method="SLSQP",
                 bounds=[(float(D[k]), None) for k in range(4)],
                 constraints=[{"type": "ineq", "fun": lambda x: S - x.sum()},
                              {"type": "eq", "fun": lambda x: G @ x - h}],
                 options={"maxiter": 2000, "ftol": 1e-16})
    ok = r.success and np.all(r.x >= D - 1e-6) and r.x.sum() <= S + 1e-6 and abs(G @ r.x - h) < 1e-6
    return (f(r.x), r.x.copy()) if ok else (np.nan, None)


# 手工网格枚举：h 从 18 到 46，步长 1。每个 h 下，x1=D1, x2=D2 固定（北岭/西庄贴底），
# x0 + x3 = h 且 Σx <= S → x0 ∈ [D0, h - D3]，f 是关于 x0 的一元二次 → 闭式解：
#   x0* = clip((h - (m3 - m0) ... ) 简化为：x0 = (m0 - m3 + h)/2，再夹到区间里
# 这里仍然用高精度 SLSQP（带多重起点），因为要的只是「f*(h) 这条曲线」。
def solve_for_H(h):
    best_f, best_x = np.inf, None
    for x0 in (D[0], (h - D[3]), h / 2.0, (M_TARGET[0] - M_TARGET[3] + h) / 2.0):
        x0 = float(np.clip(x0, D[0], h - D[3]))
        x = np.array([x0, D[1], D[2], h - x0])
        if x.sum() > S + 1e-9 or np.any(x < D - 1e-9):
            continue
        r = minimize(f, x, jac=grad_f, method="SLSQP",
                     bounds=[(float(D[k]), None) for k in range(4)],
                     constraints=[{"type": "ineq", "fun": lambda v: S - v.sum(),
                                   "jac": lambda v: -np.ones(4)},
                                  {"type": "eq", "fun": lambda v, h=h: G @ v - h,
                                   "jac": lambda v: G}],
                     options={"maxiter": 3000, "ftol": 1e-16})
        v = r.x
        ok = (r.success and np.all(v >= D - 1e-6) and v.sum() <= S + 1e-6 and abs(G @ v - h) < 1e-6)
        if ok and f(v) < best_f:
            best_f, best_x = f(v), v.copy()
    return (best_f, best_x) if best_x is not None else (np.nan, None)


FAIR_SWEEP = []
for h in np.arange(18.0, 46.1, 1.0):
    fv, xv = solve_for_H(float(h))
    if xv is not None and np.all(xv >= D - 1e-6) and xv.sum() <= S + 1e-6:
        FAIR_SWEEP.append({"h": float(h), "f": round(float(fv), 4),
                           "x": [round(float(t), 3) for t in xv]})
    else:
        FAIR_SWEEP.append({"h": float(h), "f": None, "x": None})
# 「不咬人」= f*(h) 与 f*(30) 平（差 < 1e-3）；本实例其实**没有没代价的区间**，
# 所以这里改成找「最便宜的一段」—— 这才是真实答案。
F_AT_H = {s["h"]: s["f"] for s in FAIR_SWEEP if s["f"] is not None}
H_BITE = min((s["h"] for s in FAIR_SWEEP if s["f"] is not None), key=lambda h: F_AT_H[h])

# ============================================================================
# 九、断言
# ============================================================================
F_FW = f(X_FW)
F_PG = f(X_PG)
assert abs(F_FW - F_STAR) < 1e-6, f"FW 必须收敛到精确最优（{F_FW} vs {F_STAR}）"
assert abs(F_PG - F_STAR) < 1e-6, f"PG 必须收敛到精确最优（{F_PG} vs {F_STAR}）"
assert KKT_OK, "精确解必须满足 KKT"
assert len(VERTS) >= 4, "可行域必须有多个顶点"
assert N_USED is not None and N_USED <= len(VERTS), "FW 解必须能写成顶点组合"
assert HIST_FW[-1]["gap"] >= F_FW - F_STAR - 1e-9, "FW 的间隙必须是 f−f* 的合法上界"
assert H_BITE is not None and np.isfinite(F_AT_H[H_BITE]), "公平线扫描必须给出可读曲线"
H_FEAS_MAX = max(s["h"] for s in FAIR_SWEEP if s["f"] is not None)
# 「逐点更贵」的起点：f*(h) 第一次不再下降（序列最小值的位置）
_H_ORDER = [s["h"] for s in FAIR_SWEEP if s["f"] is not None]
H_HALF = None
for _i in range(1, len(_H_ORDER)):
    if F_AT_H[_H_ORDER[_i]] > F_AT_H[_H_ORDER[_i - 1]] + 1e-6:
        H_HALF = _H_ORDER[_i]
        break
assert H_HALF is not None, "公平线曲线必须有拐点（否则说明 f 不是凸的或扫描坏了）"
assert abs(F_AT_H[H_CONST] - F_STAR) < 1e-4, "扫描在 h=30 处必须复现基准最优"

# ============================================================================
# 十、图
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
fig.suptitle("Day76b · 四校巡回排课：约束下每一步「怎么走」", fontsize=15)

ax = axes[0, 0]
kf = [h["k"] for h in HIST_FW]
ax.semilogy(kf, np.maximum([h["f"] - F_STAR for h in HIST_FW], 1e-16), "-", color="#1f77b4",
            label="Frank-Wolfe（2/(k+2)）")
ax.semilogy([h["k"] for h in HIST_PG], np.maximum([h["f"] - F_STAR for h in HIST_PG], 1e-16),
            "-", color="#d62728", label="投影梯度（1/L）")
ax.semilogy(kf, np.maximum([h["gap"] for h in HIST_FW], 1e-16), "--", color="#2ca02c", lw=1.2,
            label="FW 方向间隙（可报的上界）")
ax.set_xlabel("迭代 k")
ax.set_ylabel("f(x_k) − f*（对数）")
ax.set_title("PGD 线性收敛；FW 是 O(1/k)，但每步更便宜")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[0, 1]
idx = np.arange(4)
ax.bar(idx - 0.27, X_STAR, 0.27, color="#2ca02c", label=f"精确最优 f*={F_STAR:.2f}")
ax.bar(idx, X_FW, 0.27, color="#1f77b4", label=f"FW 收敛解 f={F_FW:.2f}")
ax.bar(idx + 0.27, X0, 0.27, color="#999999", label="起点（贴底线的烂方案）")
ax.plot(idx, D, "_", ms=22, color="k", label="底线 D")
ax.set_xticks(idx); ax.set_xticklabels(NAMES)
ax.set_ylabel("节 / 天")
ax.set_title("FW 从烂起点收敛到唯一最优（东庄15 南坪18 西庄20 北岭15）")
ax.legend(fontsize=8)

ax = axes[1, 0]
hs = [s["h"] for s in FAIR_SWEEP]
fs = [s["f"] if s["f"] is not None else np.nan for s in FAIR_SWEEP]
ax.plot(hs, fs, "o-", color="#1f77b4")
ax.axvline(H_CONST, color="#d62728", ls="--", lw=1.2)
ax.text(H_CONST + 0.4, float(np.nanmax(fs)) * 0.72, f"本次公平线 h={H_CONST:.0f}", color="#d62728",
        fontsize=8.5)
if H_BITE is not None:
    ax.plot([H_BITE], [F_AT_H[H_BITE]], "*", ms=18, color="#2ca02c")
    ax.annotate(f"最便宜 h={H_BITE:.0f}（f*={F_AT_H[H_BITE]:.2f}）", xy=(H_BITE, F_AT_H[H_BITE]),
                xytext=(H_BITE + 1.0, F_AT_H[H_BITE] + 8), fontsize=8.5, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
ax.set_ylim(0, float(np.nanmax(fs)) * 1.08)
ax.set_xlabel("公平线 x0+x3 = h（节）")
ax.set_ylabel("最优目标 f*")
ax.set_title("公平线越偏越贵：h=30 时 f*=11，h=36 最便宜（两个大校都满意），h≥42 开始罚北岭")
ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.scatter([h["k"] for h in HIST_PG], [h["f"] for h in HIST_PG], s=14, color="#d62728", label="PG 轨迹")
ax.scatter([h["k"] for h in HIST_FW], [h["f"] for h in HIST_FW], s=14, color="#1f77b4", label="FW 轨迹")
ax.axhline(F_STAR, color="k", ls="--", lw=1)
ax.set_xlim(0, 60); ax.set_ylim(F_STAR - 0.5, F_STAR + 12)
ax.set_xlabel("迭代 k")
ax.set_ylabel("f(x_k)")
ax.set_title("前 60 步放大：FW 前期猛、后期拖；PG 稳")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "day76_frank_wolfe_equity.png"), dpi=140)
plt.close(fig)

# ============================================================================
# 十一、落盘
# ============================================================================
out = {
    "instance": {"names": NAMES, "students": STUDENTS.tolist(), "D": D.tolist(), "S": S,
                 "fair": "x0 + x3 = 30", "ideal": M_TARGET.tolist()},
    "exact": {"x": [round(float(v), 4) for v in X_STAR], "f": round(F_STAR, 6),
              "kkt_ok": bool(KKT_OK), "lam_S": round(float(LAM_S), 6),
              "lam_G": round(float(LAM_G), 6), "min_mu": round(float(KKT_MIN_MU), 8)},
    "fw": {"x": [round(float(v), 4) for v in X_FW], "f": round(F_FW, 8), "iters": len(HIST_FW),
           "final_gap": round(float(HIST_FW[-1]["gap"]), 10),
           "n_vertices_total": int(len(VERTS)), "n_vertices_used": N_USED,
           "vertex_mix": (None if LAM is None else
                          [[round(float(l), 4), [round(float(t), 3) for t in VERTS[i]]]
                           for i, l in enumerate(LAM) if l > 1e-6])},
    "pg": {"x": [round(float(v), 4) for v in X_PG], "f": round(F_PG, 8), "iters": len(HIST_PG)},
    "iter_to_tol": {
        "targets": [1e-1, 1e-2, 1e-3],
        "fw": [next((h["k"] for h in HIST_FW if h["f"] - F_STAR <= t), None) for t in (1e-1, 1e-2, 1e-3)],
        "pg": [next((h["k"] for h in HIST_PG if h["f"] - F_STAR <= t), None) for t in (1e-1, 1e-2, 1e-3)],
    },
    "fair_sweep": FAIR_SWEEP,
    "fair_bite_h": H_BITE,
    "fair_turn_h": H_HALF,
    "fair_feasible_max_h": H_FEAS_MAX,
    "runtime_s": round(time.time() - T0, 2),
}
with open(os.path.join(HERE, "day76_frank_wolfe_result.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

print("=== Day76b 四校巡回排课：QP 的两种走法 ===")
print(f"精确最优 x* = {np.round(X_STAR, 3)}  f* = {F_STAR:.6f}  KKT={KKT_OK}")
print(f"FW  x = {np.round(X_FW, 4)}  f = {F_FW:.8f}  迭代 {len(HIST_FW)}  最终间隙 {HIST_FW[-1]['gap']:.2e}")
print(f"PG  x = {np.round(X_PG, 4)}  f = {F_PG:.8f}  迭代 {len(HIST_PG)}")
print(f"顶点 {len(VERTS)} 个，FW 解用到 {N_USED} 个：{out['fw']['vertex_mix']}")
print(f"到 f−f*≤1e-2：FW {out['iter_to_tol']['fw'][1]} 步 / PG {out['iter_to_tol']['pg'][1]} 步")
print(f"公平线扫 18~46：最便宜 h = {H_BITE}（f*={F_AT_H[H_BITE]}），拐点 h = {H_HALF}，"
      f"h > {H_FEAS_MAX} 不可行")
print(f"耗时 {out['runtime_s']} s")
