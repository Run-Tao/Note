# -*- coding: utf-8 -*-
r"""
subject: mathematical-modeling
level:   L4 整数规划（ILP：LP 松弛 · 整数性间隙 Gomory gap · 割平面 cutting plane）/ L7 算法（「先松后紧」）
anchor:  Roadmap L4 → 整数规划 ILP → LP 松弛给界、舍入会骗人、割平面抬高下界
goal:    用「酱菜厂整箱接单」这个真实决策，把「怎么解 ILP」倒过来讲：**你怎么知道自己解得对不对**。
         ① LP 松弛是一个**合法的界**：z_LP >= z_IP 永远成立，但它给的不是方案，是「能有多好」；
         ② 「四舍五入」是最自然、也最危险的解法 —— 可能**违反约束**（求解器不报错，它只给你一个漂亮的数），
            也可能「砍错人」，比最优少一大截；
         ③ 整数性间隙（Gomory gap）：本例 z_LP=277,393.94 vs 整数最优 277,330 —— 只有 0.023%，
            但**车间排班表上就是 63.94 元的差距 + 一组不同的箱子数**；
         ④ 割平面：**一条在所有整数解上成立、却把 LP 最优点切掉的超平面**。
            加进去以后 LP 界从 277,393.94 降到 277,342.28（间隙从 63.94 缩到 12.28，缩小 81%），
            而整数最优一动不动 —— 这就是「割」的全部意义：**不动答案，只动界**。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error \
        "Mathematical_Modeling/08 - Python/daily/2026-11-13/day76_integer_lp_gap.py"
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()


def _dbg(msg):
    print(f"[dbg {time.time() - T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


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
# 一、实例：云溪县「味源」酱菜厂 —— 整箱外协接单
#     三种酱菜（爽口萝卜 / 香辣豆腐乳 / 老坛酸菜）抢三道工序（沿用 Day52 同一家厂），
#     但订单**只收整箱**：半箱要拆托盘、人工分装、质检不过 → 决策变量必须是整数，这是纯 ILP。
#     （Day52 把它当连续 LP 解，得到 638.55 箱这种「半箱订单」—— 现实里不存在。）
# ============================================================================
NAMES = ["爽口萝卜", "香辣豆腐乳", "老坛酸菜"]
PROFIT = np.array([170.0, 210.0, 260.0])            # 元/箱
A = np.array([[0.60, 1.00, 1.40],                    # 腌渍 工分/箱
              [1.50, 0.80, 0.70],                    # 灌装
              [0.80, 1.10, 0.90]])                   # 杀菌
B = np.array([1210.0, 1560.0, 1330.0])               # 本周可用工分

# 单变量上界（只受「某一道工序」卡时最多能做几箱）：取三道工序里最松的那一刀
VMAX = [int(np.floor(max(B[j] / A[j, k] for j in range(3)))) for k in range(3)]


def solve_lp(extra_rows=None):
    """解 LP 松弛；extra_rows 用来加割。返回 (z, x) 或 (−inf, None)。"""
    A2, B2 = A, B
    if extra_rows:
        for (w, b) in extra_rows:
            A2 = np.vstack([A2, np.asarray(w, dtype=float).reshape(1, -1)])
            B2 = np.concatenate([B2, [b]])
    res = linprog(-PROFIT, A_ub=A2, b_ub=B2, bounds=[(0.0, None)] * 3, method="highs")
    if not res.success:
        return -np.inf, None
    return float(-res.fun), res.x.copy()


Z_LP, X_LP = solve_lp()
_dbg(f"LP {Z_LP:.4f} x={np.round(X_LP, 4)}")

# ============================================================================
# 二、裁判：整数最优
#    ① 切片枚举：固定 x0、x1，x2 取剩余产能允许的最大整数（目标系数全正 → floor 不丢最优）。
#       范围只取「腌渍」「灌装」两道工序允许的前两批（1041 × 1211），比 1.1e9 的全网格小 4 个数量级。
#    ② 邻域全枚举（±1 箱）做第二裁判；③ HiGHS 的 MILP 求解器（完全不同的代码路径）。
# ============================================================================
def enum_int_smart():
    """切片枚举（向量化）：固定 x0、x1，x2 取剩余产能允许的最大整数（系数全正 → floor 不丢最优）。

    🔴 踩坑（2026-11-13）：最初写的是 Python 双层 for（1041×1211 ≈ 126 万次），单跑 34 秒 ——
    光这一项就吃掉了整节课一半的运行时间。改成 numpy 网格一次算完 → 0.5 秒。
    「126 万次循环」在建模课里必须当成警告：**规模先量，写法再选**。
    """
    n0 = int(np.floor(B[0] / A[0, 0])) + 1
    n1 = int(np.floor(B[1] / A[1, 1])) + 1
    i0 = np.arange(n0, dtype=np.float64)
    i1 = np.arange(n1, dtype=np.float64)
    X0, X1 = np.meshgrid(i0, i1, indexing="ij")
    rem = np.full(X0.shape, np.inf)
    for j in range(3):
        rem = np.minimum(rem, (B[j] - A[j, 0] * X0 - A[j, 1] * X1) / A[j, 2])
    ok = rem >= -1e-9
    X2 = np.where(ok, np.floor(np.maximum(rem, 0) + 1e-9), 0.0)
    Z = PROFIT[0] * X0 + PROFIT[1] * X1 + PROFIT[2] * X2
    Z = np.where(ok, Z, -np.inf)
    k = int(np.argmax(Z))
    return float(Z.flat[k]), np.array([X0.flat[k], X1.flat[k], X2.flat[k]], dtype=float)


def local_bruteforce(x0v, rad=1):
    best, best_x, cnt = -np.inf, None, 0
    ranges = [range(max(0, int(x0v[k]) - rad), int(x0v[k]) + rad + 1) for k in range(3)]
    for x in itertools.product(*ranges):
        xv = np.array(x, dtype=float)
        if np.all(A @ xv <= B + 1e-9):
            cnt += 1
            z = float(PROFIT @ xv)
            if z > best:
                best, best_x = z, xv
    return best, best_x, cnt


Z_ENUM, X_ENUM = enum_int_smart()
Z_BF, X_BF = Z_ENUM, X_ENUM
Z_LOCAL, X_LOCAL, N_FEAS = local_bruteforce(X_ENUM, rad=1)
_dbg(f"enum {Z_BF:.2f} x={X_BF.astype(int)}  local {Z_LOCAL:.2f} ({N_FEAS} 个邻域可行点)")


def solve_milp_highs():
    res = milp(c=-PROFIT, constraints=LinearConstraint(A, -np.inf, B),
               integrality=np.ones(3), bounds=Bounds(0, np.inf), options={"time_limit": 30})
    return float(-res.fun), np.round(res.x).copy()


Z_MILP, X_MILP = solve_milp_highs()
_dbg(f"MILP {Z_MILP:.2f} x={X_MILP.astype(int)}")

# ============================================================================
# 三、三种「最自然的错误解法」
# ============================================================================
def err_naive_round(x):
    """错解 A：直接四舍五入 → 可能不可行（最常见、最危险）。"""
    xr = np.round(x)
    return xr, bool(np.all(A @ xr <= B + 1e-9)), float(PROFIT @ xr)


def err_floor(x):
    """错解 B：全部向下取整 → 一定可行，但「每个产品都白干半箱」。"""
    xf = np.floor(x)
    return xf, True, float(PROFIT @ xf)


def err_round_then_fix(x):
    """错解 C：四舍五入后超限就「按单位工分利润最低的砍」→ 常常砍错人。"""
    xr = np.round(x).copy()
    guard = 0
    while np.any(A @ xr > B + 1e-9) and guard < 50:
        over = A @ xr - B
        j = int(np.argmax(over))
        costs = [PROFIT[k] / max(A[j, k], 1e-9) for k in range(3)]
        k = int(np.argmin(costs))
        xr[k] = max(0.0, xr[k] - 1)
        guard += 1
    return xr, bool(np.all(A @ xr <= B + 1e-9)), float(PROFIT @ xr)


ERR_A = err_naive_round(X_LP)
ERR_B = err_floor(X_LP)
ERR_C = err_round_then_fix(X_LP)
_dbg(f"errA {ERR_A}  errB {ERR_B}  errC {ERR_C}")

# ============================================================================
# 四、割：一条「在所有整数解上成立、却把 LP 点切掉」的平面
#     做法（支撑函数版，几秒钟跑完）：
#       · 所有整数可行点都在盒子 [0, VMAX] 里；对 w >= 0，max_{整数可行点} w·x
#         只需要扫一遍「边界点」——固定 x0,x1，x2 取最大；
#       · 每条约束的线性组合仍然有效：任取 u >= 0，则 (u·A)·x <= u·B 对所有可行解成立；
#       · 若 (u·A)·x_LP > max_{整数点} (u·A)·x，则这条平面**只切分数点** → 合法整数割。
# ============================================================================
def boundary_integer_points():
    """整数可行点的「上边界」：固定 x0,x1，x2 取剩余产能允许的最大整数。
    对 w >= 0，max w·x 一定在边界上取到 → 只需这些点（同一套向量化写法）。"""
    i0 = np.arange(VMAX[0] + 1, dtype=np.float64)
    i1 = np.arange(VMAX[1] + 1, dtype=np.float64)
    X0, X1 = np.meshgrid(i0, i1, indexing="ij")
    rem = np.full(X0.shape, np.inf)
    for j in range(3):
        rem = np.minimum(rem, (B[j] - A[j, 0] * X0 - A[j, 1] * X1) / A[j, 2])
    ok = rem >= -1e-9
    X2 = np.where(ok, np.floor(np.maximum(rem, 0) + 1e-9), 0.0)
    return np.stack([X0[ok], X1[ok], X2[ok]], axis=1)


P_BND = boundary_integer_points()
_dbg(f"boundary integer points {P_BND.shape[0]}")


def find_cut(P, wmax=6):
    cuts = []
    for coef in itertools.product(range(0, wmax), repeat=3):
        w = np.array(coef, dtype=float)
        if not np.any(w):
            continue
        b_max = float((P @ w).max())        # 所有整数点都满足 w·x <= b_max
        v_lp = float(w @ X_LP)
        if v_lp > b_max + 1e-9:             # LP 点被切掉
            cuts.append((w, b_max, v_lp))
    cuts.sort(key=lambda t: t[2] - t[1], reverse=True)   # 切得最狠的排前面
    return cuts


CUT_CANDS = find_cut(P_BND)
CUT = CUT_CANDS[0] if CUT_CANDS else None
_dbg(f"cuts {len(CUT_CANDS)}: "
     f"{[(w.tolist(), b, round(v, 2)) for w, b, v in CUT_CANDS[:4]]}")

CUT_SWEEP = []
for w, b, v in CUT_CANDS[:40]:
    z2, x2 = solve_lp([(w, b)])
    if x2 is not None:
        CUT_SWEEP.append({"w": w.astype(int).tolist(), "b": round(b, 4), "lp_after": round(z2, 3),
                          "gap_after": round(z2 - Z_BF, 4),
                          "x_after": [round(float(t), 4) for t in x2]})

if CUT is not None:
    Z_CUT, X_CUT = solve_lp([(CUT[0], CUT[1])])
else:
    Z_CUT, X_CUT = Z_LP, X_LP
_dbg(f"cut applied → LP {Z_CUT:.3f} x={None if X_CUT is None else np.round(X_CUT, 3)}")

# ============================================================================
# 五、敏感性：产能放大 10% 时，界与最优同时涨，但间隙不是常数
# ============================================================================
GAP = Z_LP - Z_BF
SENS = []
for rho in [0.90, 0.95, 1.00, 1.05, 1.10]:
    B2 = B * rho
    r = linprog(-PROFIT, A_ub=A, b_ub=B2, bounds=[(0, None)] * 3, method="highs")
    z_lp = float(-r.fun) if r.success else np.nan
    m = milp(c=-PROFIT, constraints=LinearConstraint(A, -np.inf, B2),
             integrality=np.ones(3), bounds=Bounds(0, np.inf), options={"time_limit": 30})
    z_ip = float(-m.fun) if m.success else np.nan
    SENS.append({"rho": rho, "z_lp": round(z_lp, 3), "z_ip": round(z_ip, 3),
                 "gap": round(z_lp - z_ip, 3)})

# ============================================================================
# 六、断言
# ============================================================================
assert Z_LP >= Z_BF - 1e-6, "LP 松弛必须不差于整数最优"
assert abs(Z_LOCAL - Z_BF) < 1e-6, "±1 邻域全枚举不能找到更好的整数解"
assert abs(Z_BF - Z_MILP) < 1e-6, "HiGHS MILP 与切片枚举必须一致"
assert GAP > 1e-6, "本实例必须有正间隙（否则教学点消失）"
assert (not ERR_A[1]) or ERR_A[2] <= Z_BF + 1e-9, "四舍五入不能给出比最优更好的可行解"
assert ERR_B[2] <= Z_BF + 1e-9, "向下取整必须可行且不超最优"
assert CUT is not None, "必须找到一条有效整数割"
assert Z_CUT >= Z_BF - 1e-6, "加割后的 LP 界不能低于整数最优（否则割切掉了整数点 → 非法）"
assert Z_CUT <= Z_LP + 1e-9, "加割后的 LP 界不能变大"
assert Z_CUT < Z_LP - 1e-9, "加割后 LP 界必须严格下降（否则割没切到 LP 点）"

# ============================================================================
# 七、图：四联图
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4))
fig.suptitle("Day76 · 整箱接单：LP 松弛给界，舍入会骗人，割平面只抬界不改答案", fontsize=14.5)

# 左上：各方案目标值
ax = axes[0, 0]
labels = ["LP 松弛\n(上界)", "四舍五入\n(错解A)", "全部取整\n(错解B)", "砍一刀\n(错解C)",
          "加割后\nLP 界", "整数最优\n(枚举+MILP)"]
vals = [Z_LP, ERR_A[2], ERR_B[2], ERR_C[2], Z_CUT, Z_BF]
cols = ["#888888", "#d62728", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
ax.bar(range(len(vals)), vals, color=cols)
ax.axhline(Z_BF, color="#2ca02c", ls="--", lw=1.2)
for i, v in enumerate(vals):
    note = ""
    if i == 1 and not ERR_A[1]:
        note = "\n(不可行)"
    ax.text(i, v + 1800, f"{v:,.0f}{note}", ha="center", fontsize=8.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("本周利润（元）")
ax.set_ylim(min(vals) * 0.96, Z_LP * 1.04)
ax.set_title(f"LP 界 {Z_LP:,.2f} → 整数 {Z_BF:,.0f}：间隙只有 {GAP:.2f} 元")

# 右上：割的几何（x0–x1 切片，x2 固定为 LP 值）
ax = axes[0, 1]
n = 300
x1g = np.linspace(560, 700, n)
y1g = np.linspace(560, 700, n)
XX, YY = np.meshgrid(x1g, y1g)
x2fix = X_LP[2]
mask = np.ones_like(XX, dtype=bool)
for j in range(3):
    mask &= (A[j, 0] * XX + A[j, 1] * YY + A[j, 2] * x2fix) <= B[j]
ax.contourf(XX, YY, mask.astype(float), levels=[0.5, 1.5], colors=["#d9ead9"])
for j, cc in zip(range(3), ["#9467bd", "#8c564b", "#17becf"]):
    ax.contour(XX, YY, A[j, 0] * XX + A[j, 1] * YY + A[j, 2] * x2fix - B[j], [0], colors=cc, linewidths=1)
if CUT is not None:
    w, b, _v = CUT
    if abs(w[1]) > 1e-9:
        xl = np.linspace(560, 700, 60)
        yl = (b - w[0] * xl - w[2] * x2fix) / w[1]
        ax.plot(xl, yl, "-", color="#d62728", lw=1.6,
                label=f"割 {int(w[0])}x0+{int(w[1])}x1+{int(w[2])}x2 ≤ {b:.0f}")
ax.plot(X_LP[0], X_LP[1], "*", ms=17, color="red", label="LP 最优（被割切掉）")
ax.plot(X_BF[0], X_BF[1], "o", ms=9, color="green", label="整数最优（仍满足割）")
ax.set_xlim(560, 700); ax.set_ylim(560, 700)
ax.set_xlabel("爽口萝卜 x0（箱）")
ax.set_ylabel("香辣豆腐乳 x1（箱）")
ax.set_title(f"切片 x2={x2fix:.1f}：割只切分数点，不切整数点")
ax.legend(fontsize=7.5, loc="lower left")

# 左下：从「界」到「真答案」的距离（gap 瀑布）
ax = axes[1, 0]
names = ["LP 松弛\n间隙", "加一条割\n后剩余", "整数最优\n(间隙=0)"]
heights = [GAP, Z_CUT - Z_BF, 0.0]
bottom = [Z_BF, Z_BF, Z_BF]
ax.bar([0], [GAP], bottom=[Z_BF], color="#d62728", width=0.5)
ax.bar([1], [Z_CUT - Z_BF], bottom=[Z_BF], color="#9467bd", width=0.5)
ax.bar([2], [0.05], bottom=[Z_BF], color="#2ca02c", width=0.5)
ax.axhline(Z_BF, color="#2ca02c", ls="--", lw=1)
ax.text(0, Z_BF + GAP / 2, f"{GAP:.2f} 元\n（{100*GAP/Z_BF:.4f}%）", ha="center", fontsize=9)
ax.text(1, Z_BF + (Z_CUT - Z_BF) / 2, f"{Z_CUT-Z_BF:.2f} 元", ha="center", fontsize=9,
        color="white")
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("利润（元）")
ax.set_ylim(Z_BF - 5, Z_LP + 5)
ax.set_title("一条割把「能有多好」的上界抬下了 81%")

# 右下：产能 × ρ 敏感性
ax = axes[1, 1]
rhos = [s["rho"] for s in SENS]
ax.plot(rhos, [s["z_lp"] for s in SENS], "o-", color="#d62728", label="LP 上界")
ax.plot(rhos, [s["z_ip"] for s in SENS], "s-", color="#2ca02c", label="整数最优")
for s in SENS:
    ax.annotate(f'{s["gap"]:.1f}', xy=(s["rho"], (s["z_lp"] + s["z_ip"]) / 2), fontsize=8, color="#555")
ax.set_xlabel("本周工分上限 × ρ")
ax.set_ylabel("利润（元）")
ax.set_title("产能放 10%：界与最优都涨，间隙跟着涨")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "day76_integer_lp_gap.png"), dpi=140)
plt.close(fig)

# ============================================================================
# 八、落盘
# ============================================================================
out = {
    "instance": {"names": NAMES, "profit": PROFIT.tolist(), "A": A.tolist(), "B": B.tolist(),
                 "vmax": VMAX},
    "lp": {"z": round(Z_LP, 4), "x": [round(float(v), 5) for v in X_LP]},
    "integer_best": {"z": round(Z_BF, 3), "x": [int(round(float(v))) for v in X_BF],
                     "local_check_z": round(Z_LOCAL, 3), "local_feasible_points": N_FEAS},
    "milp_highs": {"z": round(Z_MILP, 3), "x": [int(round(float(v))) for v in X_MILP]},
    "gomory_gap": {"abs": round(GAP, 4), "pct": round(100 * GAP / Z_BF, 6)},
    "wrong": {
        "A_naive_round": {"x": [float(round(v)) for v in ERR_A[0]], "feasible": ERR_A[1],
                          "z": round(ERR_A[2], 3),
                          "overrun": [round(float(t), 3) for t in (A @ ERR_A[0] - B)]},
        "B_floor": {"x": [float(v) for v in ERR_B[0]], "feasible": ERR_B[1], "z": round(ERR_B[2], 3)},
        "C_round_fix": {"x": [float(v) for v in ERR_C[0]], "feasible": ERR_C[1],
                        "z": round(ERR_C[2], 3)},
    },
    "cut": {"w": None if CUT is None else CUT[0].astype(int).tolist(),
            "b": None if CUT is None else round(CUT[1], 4),
            "lp_at_cut": None if CUT is None else round(CUT[2], 4),
            "lp_after": round(Z_CUT, 4),
            "x_after": None if X_CUT is None else [round(float(v), 4) for v in X_CUT],
            "gap_after": round(Z_CUT - Z_BF, 4),
            "gap_reduction_pct": round(100 * (1 - (Z_CUT - Z_BF) / GAP), 2),
            "n_candidates": len(CUT_CANDS),
            "sweep_top": CUT_SWEEP[:8],
            "boundary_points": int(P_BND.shape[0])},
    "sensitivity": SENS,
    "runtime_s": round(time.time() - T0, 2),
}
with open(os.path.join(HERE, "day76_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("=== Day76 整箱接单：LP 松弛 / 整数性间隙 / 割平面 ===")
print(f"LP 松弛        : z={Z_LP:.4f}  x={np.round(X_LP, 4)}")
print(f"整数最优        : z={Z_BF:.2f}  x={X_BF.astype(int)}  （邻域裁判 {Z_LOCAL:.2f} / MILP {Z_MILP:.2f}）")
print(f"整数性间隙      : {GAP:.4f} 元（{100*GAP/Z_BF:.4f}%）")
print(f"错解A 四舍五入 : x={ERR_A[0]}  可行={ERR_A[1]}  z={ERR_A[2]:.2f}")
print(f"错解B 全取整   : x={ERR_B[0]}  z={ERR_B[2]:.2f}（比最优少 {Z_BF-ERR_B[2]:.2f}）")
print(f"错解C 砍一刀   : x={ERR_C[0]}  可行={ERR_C[1]}  z={ERR_C[2]:.2f}")
if CUT is not None:
    print(f"割 {CUT[0].astype(int)}·x <= {CUT[1]:.0f}：LP 界 {Z_LP:.2f} → {Z_CUT:.2f}，"
          f"剩余间隙 {Z_CUT-Z_BF:.2f}（缩小 {100*(1-(Z_CUT-Z_BF)/GAP):.1f}%），"
          f"候选割 {len(CUT_CANDS)} 条")
print(f"敏感性 : {[(s['rho'], s['z_lp'], s['z_ip'], s['gap']) for s in SENS]}")
print(f"耗时 {out['runtime_s']} s")
