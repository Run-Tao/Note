#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L6 优化模型 / L8 数值计算与算法（整数规划求解算法：割平面法 / Gomory 割）
anchor  : Level 6 - 优化模型（整数线性规划的精确求解：割平面 Cutting Planes）
goal    : 用「工厂三条产线排产、产量必须整数（或 0-1 开线）」的真实决策，
          手写【Gomory 割平面法 (Gomory Fractional Cut)】完整框架，回答决策者/建模者
          最关心的五个问题：
            ① LP 松弛解为什么"看着最优、实则不能用"（分数产量不可执行）；
            ② 能不能"不动搜索树，只在同一张 LP 上不断加约束"就把解逼成整数
               （= 割平面 vs Day4 分支定界的分工）；
            ③ 那条神秘的割从哪来（Gomory 割的推导：表 + 分数部分 + 可行性）；
            ④ 为什么割"切掉 LP 顶点却切不掉任何整数点"（这就是"合法"的含义）；
            ⑤ 割平面能独立把整数解切出来吗（增益 + 数值退化，为何工业界要结合分支定界）。
          主算法零专用求解器全手写：手写【单纯形表 (tableau simplex)】解 LP + 从最优表
          直接读出 Gomory 割；scipy.milp 求真最优仅作独立对照/裁判。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-10-05/gomory_cutting_planes.py"
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 0. 真实决策背景：工厂排产（产量必须整数）
# =========================================================================
# 一家工厂用有限的【设备工时】和【原料】同时生产两种产品（产品1、产品2），
#   每件产品 1 用设备 3 工时、耗料 1 单位、赚 5 元；
#   每件产品 2 用设备 1 工时、耗料 2 单位、赚 3 元；
#   本周可用设备工时 10、原料 8。
# 问：两种产品各生产多少件，总利润最大？（每日/每周件数必须是整数）
#
# 这就是标准形如下的小型整数规划：
#   max 5x1 + 3x2
#   s.t. 3x1 + x2 <= 10    (设备工时)
#        x1 + 2x2 <= 8     (原料)
#        x1, x2 >= 0, 整数
# 【注意】整数最优是 (2,3)（利润 19），而 LP 松弛最优是 (2.4,2.8)（利润 20.4）——
# 分数产量"看着更赚"，但现实中不可执行：这正是整数规划比 LP 难的根本原因。
#
# Day4 用【分支定界】解决它：把分数变量分支 x<=2 / x>=3，在"搜索树"上剪枝。
# 本课换一条路——【割平面法 (Cutting Planes)】：
#   不分支、不搜索，而是【在同一张 LP 上不断追加一条"合法"的线性约束】，
#   把 LP 可行域一点点"削"到整数点附近，直到 LP 最优解自己变整数。
# 那条神奇的约束叫【Gomory 割 (Gomory fractional cut)】，
# 它是从【LP 最优单纯形表】里直接"读"出来的。

# 刻意设计一个"LP 松弛解是分数、且分数很脏"的实例：
#   两个产品、两个紧约束，LP 顶点是 (2.5, 1.75)，两个坐标都非整数。
# 系数 -> 让 Gomory 割的推导有非平凡的小数部分，教学效果最好。
C_OBJ = np.array([5.0, 3.0])          # 每件产品利润（max）
A_UB = np.array([[3.0, 1.0],          # 设备工时约束
                 [1.0, 2.0]])         # 原料约束
B_UB = np.array([10.0, 8.0])          # 右侧可用量


# =========================================================================
# 1. 手写单纯形表求解器（解标准形式 max c x, s.t. Ax<=b, x>=0）
# =========================================================================
def simplex(c, A, b, max_iter=200, tol=1e-9):
    """
    手写【表式单纯形法】解 max c·x s.t. A x <= b, x >= 0（b >= 0）。
    返回 (x, obj, final_tableau, basis, n_cols)。
    final_tableau 是最终单纯形表，用于从最优基变量行直接读出 Gomory 割。
    表结构：[A | I | b] 同时携带目标行，这里显式维护完整表。
    """
    m, n = A.shape
    # 松弛变量作为初始基
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    # 目标行（用 max 形式）：z - c x = 0  ->  第一行放 -c，最后元素 0
    T[m, :n] = -c
    basis = list(range(n, n + m))     # 初始基：松弛变量

    for _ in range(max_iter):
        # 选入基：目标行里系数最负（对 max 问题，负的检验数代表可改进）
        row_obj = T[m, :-1]
        enter = int(np.argmin(row_obj))
        if row_obj[enter] >= -tol:
            break  # 没有负检验数 -> 最优
        # 比值检验选离基（只用 > tol 的列）
        col = T[:m, enter]
        ratios = []
        for i in range(m):
            if col[i] > tol:
                ratios.append((T[i, -1] / col[i], i))
        if not ratios:
            raise RuntimeError("无界 (unbounded)")
        _, leave = min(ratios)
        # 主元消元
        pivot = T[leave, enter]
        T[leave, :] /= pivot
        for i in range(m + 1):
            if i != leave and abs(T[i, enter]) > tol:
                T[i, :] -= T[i, enter] * T[leave, :]
        basis[leave] = enter

    n_cols = n + m
    x = np.zeros(n_cols)
    for i in range(m):
        x[basis[i]] = T[i, -1]
    obj = T[m, -1]
    return x[:n], obj, T, basis, n_cols


def row_of_basis(T, basis, var_index, tol=1e-9):
    """在最终单纯形表里找到基变量 var_index 所在的行号。"""
    for i, bv in enumerate(basis):
        if bv == var_index:
            return i
    raise ValueError(f"变量 {var_index} 不在基中")


def gomory_cut_from_tableau(T, basis, n_orig, n_full, tol=1e-6):
    """
    从 LP 最优单纯形表读出一条 Gomory 分数割。
    思路：找一个【取分数值】的【整数变量 (前 n_orig 个)】基行：
        该行： x_B + Σ_j a_Bj x_j = b_B   (对全部 n_full 个变量：原变量+松弛变量)
      把系数写成 b_B = floor(b_B) + f0,  a_Bj = floor(a_Bj) + f_j (0<=f<1)
      对整数可行解（所有原变量整数，松弛变量也随之为整数），有 Σ_j f_j x_j ≡ f0 (mod 1)，
      而左边 >=0，故若 f0>0 则 Σ_j f_j x_j >= f0。
      割（以当前全部 n_full 个变量表示）:  Σ_j f_j x_j >= f0 。
    返回：(割系数向量 (n_full,), 割右端 f0, 选中的基变量, 基行号) 或 None（已整数）
    """
    m = len(basis)
    for i in range(m):
        var = basis[i]
        bval = T[i, -1]
        # 只对【应为整数】的原变量（前 n_orig 个）且取分数值者生成割
        if var >= n_orig:
            continue
        f0 = bval - np.floor(bval)
        if f0 < tol or f0 > 1 - tol:
            continue  # 该基变量已经是整数
        # 该行各列的小数部分（对全部 n_full 个变量）
        row = T[i, :n_full]
        fj = row - np.floor(row + 1e-9)   # 使 0 <= fj < 1
        fj[np.abs(fj - 1) < 1e-9] = 0.0   # 清理 0.9999999 -> 0
        return fj, f0, var, i
    return None


# =========================================================================
# 2. 割平面主算法
# =========================================================================
def cutting_plane_rounds(c, A, b, max_rounds=30, tol=1e-7, verbose=True):
    """
    反复：解 LP -> 若整数则停 -> 否则从最优表读 Gomory 割加入 -> 再解。
    记录每轮的 LP 目标、割、分数程度，用于教学演示。
    """
    A_cur = [np.array(row, dtype=float) for row in A]
    b_cur = [float(v) for v in b]
    n_orig = A.shape[1]
    history = []

    for rnd in range(max_rounds + 1):
        A_mat = np.array(A_cur)
        b_vec = np.array(b_cur)
        x, obj, T, basis, n_full = simplex(c, A_mat, b_vec)
        frac = np.abs(x - np.round(x)).max()
        history.append({
            "round": rnd, "obj": obj, "x": x.copy(),
            "frac": frac, "n_cuts": len(A_cur) - A.shape[0],
        })
        if verbose:
            print(f"  第{rnd:2d}轮: LP最优={obj:.6f}  x={np.round(x,4)}  "
                  f"最大分数程度={frac:.4f}  累计割数={len(A_cur)-A.shape[0]}")
        if frac < tol:
            return x, obj, history, np.array(A_cur), np.array(b_cur)

        cut = gomory_cut_from_tableau(T, basis, n_orig, n_full)
        if cut is None:
            return x, obj, history, np.array(A_cur), np.array(b_cur)
        fj, f0, var, rowi = cut
        # fj 已按【当前全部变量】排列：前 n_orig 个是原变量，后 m 个是本轮松弛变量。
        # 但松弛变量 s_k = b_k - A_k·x。把割 Σ_j fj_j x_j >= f0 中的松弛项代入消去，
        # 化回【只含原变量】的约束，才能作为新行加入约束列表：
        #   Σ_{orig j} fj_j x_j + Σ_k fj_{n_orig+k} (b_k - A_k·x) >= f0
        #   => [fj_orig - Σ_k fj_{slk,k} A_k] · x >= f0 - Σ_k fj_slk,k b_k
        m_cur = len(A_cur)
        fj_orig = fj[:n_orig].copy()
        fj_slk = fj[n_orig:n_orig + m_cur]
        cut_coef = fj_orig - fj_slk @ A_mat      # (n_orig,)
        cut_rhs = f0 - float(fj_slk @ b_vec)
        # 割形如 cut_coef·x >= cut_rhs；我们的 simplex 要求 A x <= b 且 b >= 0（标准松弛形）。
        # 若 cut_rhs < 0，把整条不等式 ×(-1) 翻成 <= 形式，保证右端非负、初始基可行。
        if cut_rhs < 0:
            cut_coef = -cut_coef
            cut_rhs = -cut_rhs
        # 加入约束（<= 形式）
        A_cur.append(np.array(cut_coef, dtype=float))
        b_cur.append(cut_rhs)
        if verbose:
            coef_str = " + ".join(f"({round(cut_coef[j],4)})x{j+1}" for j in range(n_orig))
            print(f"        + Gomory割: {coef_str} <= {cut_rhs:.4f}  "
                  f"(来自取分数的基变量 x{var+1})")

    return x, obj, history, np.array(A_cur), np.array(b_cur)


# =========================================================================
# 3. 暴力枚举整数点（小规模，给"割是否合法"做裁判）
# =========================================================================
def enumerate_integer_feasible(A, b, hi=12):
    """枚举 0..hi 范围内的整数点，返回可行点列表与最优整数目标（用于对照）。"""
    pts, best, best_x = [], -np.inf, None
    for x1 in range(hi + 1):
        for x2 in range(hi + 1):
            xv = np.array([x1, x2], dtype=float)
            if np.all(A @ xv <= b + 1e-9):
                pts.append(xv)
                val = C_OBJ @ xv
                if val > best:
                    best, best_x = val, xv
    return np.array(pts), best, best_x


def solve_lp_relaxation(c, A, b):
    """用同一 simplex 求 LP 松弛最优，返回 (x, obj)。"""
    x, obj, _, _, _ = simplex(c, A, b)
    return x, obj


def solve_exact_milp(c, A, b):
    """scipy.milp 求整数真最优，独立对照。"""
    n = A.shape[1]
    cons = LinearConstraint(A, -np.inf, b)
    res = milp(c=-c, constraints=[cons], integrality=np.ones(n),
               bounds=Bounds(0, np.inf))
    assert res.success, res.message
    return res.x, -res.fun


# =========================================================================
# 4. 主流程
# =========================================================================
def main():
    print("=" * 70)
    print("Day37 · 割平面法 (Gomory Cutting Planes) —— 工厂整数排产")
    print("=" * 70)

    A, b, c = A_UB, B_UB, C_OBJ
    print(f"\n模型: max {c}·x  s.t. {A.tolist()} x <= {b.tolist()}, x>=0 整数\n")

    # ---- LP 松弛 ----
    x_lp, obj_lp = solve_lp_relaxation(c, A, b)
    print(f"[LP 松弛] 最优={obj_lp:.4f}  x={np.round(x_lp, 4)}  "
          f"-> 分数产量不可执行 (最大分数程度={np.abs(x_lp-np.round(x_lp)).max():.4f})")

    # ---- 整数真最优 ----
    x_milp, obj_milp = solve_exact_milp(c, A, b)
    print(f"[整数真最优] {obj_milp:.4f}  x={x_milp}  (scipy.milp 对照)")

    # ---- 割平面 ----
    print("\n[割平面法 迭代]")
    x_cp, obj_cp, history, A_final, b_final = cutting_plane_rounds(c, A, b)

    # ---- 暴力枚举对照 ----
    pts, best_bf, best_x_bf = enumerate_integer_feasible(A, b)
    print(f"\n[暴力枚举] 整数可行点 {len(pts)} 个，最优={best_bf:.4f}  x={best_x_bf}")

    # ---- 割的"合法性"验证：割是否切掉任何整数可行点？ ----
    # 逐条检查最终加入的每条割：所有枚举整数可行点是否都满足？
    n_orig = A.shape[1]
    n_cuts = A_final.shape[0] - A.shape[0]
    cut_ok = True
    for k in range(n_cuts):
        coef = A_final[A.shape[0] + k]
        rhs = b_final[A.shape[0] + k]
        # 约束是 coef·x <= rhs；违反 = coef·x > rhs（对整数可行点而言）
        viol = int(np.sum(coef @ pts.T > rhs + 1e-6))
        if viol > 0:
            cut_ok = False
            print(f"  ⚠️ 第{k+1}条割切掉了 {viol} 个整数点（非法！）")
    print(f"[割合法性] 全部 {n_cuts} 条割均未切掉任何整数可行点: {cut_ok}")

    # =====================================================================
    # 断言（硬约束）
    # =====================================================================
    assert abs(obj_lp - 20.4) < 1e-6, obj_lp             # LP 松弛 20.4
    assert np.abs(x_lp - np.round(x_lp)).max() > 0.1     # LP 解确为分数
    assert abs(obj_milp - 19.0) < 1e-6, obj_milp         # 整数真最优 19
    assert abs(obj_cp - obj_milp) < 1e-5, (obj_cp, obj_milp)  # 割平面到达整数最优
    assert np.abs(x_cp - np.round(x_cp)).max() < 1e-6    # 割平面解为整数
    assert np.allclose(np.round(x_cp), x_milp)           # 与 milp 最优解一致
    assert cut_ok                                         # 所有割都不切整数点
    assert n_cuts >= 1                                    # 确实用到了割
    assert abs(best_bf - obj_milp) < 1e-6                 # 暴力枚举==milp
    print("\n✅ 全部断言通过")

    # =====================================================================
    # 绘图
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x1 = np.linspace(0, 7, 400)

    # ---- 左上：几何——LP 可行域、整数格点、割把顶点削掉 ----
    ax = axes[0, 0]
    # 两条原始约束边界
    ax.plot(x1, (b[0] - A[0, 0] * x1) / A[0, 1], 'b-', lw=2, label='约束1: 3x1+x2≤10')
    ax.plot(x1, (b[1] - A[1, 0] * x1) / A[1, 1], 'g-', lw=2, label='约束2: x1+2x2≤8')
    # 整数可行格点
    ax.plot(pts[:, 0], pts[:, 1], 'k.', ms=9, label='整数可行点')
    # LP 顶点
    ax.plot(*x_lp, 'r*', ms=20, label=f'LP最优 x={np.round(x_lp,2)}')
    # 整数最优
    ax.plot(*x_milp, 'b^', ms=14, label=f'整数最优 x={x_milp}')
    # 割线：从最终割里拿第一条画出来（Σ fj x >= f0）
    if n_cuts >= 1:
        coef = A_final[A.shape[0]]
        rhs = b_final[A.shape[0]]
        if abs(coef[1]) > 1e-9:
            y_cut = (rhs - coef[0] * x1) / coef[1]
            ax.plot(x1, y_cut, 'm--', lw=2,
                    label=f'Gomory割: {coef[0]:.2f}x1+{coef[1]:.2f}x2≥{rhs:.2f}')
            ax.scatter(coef[0] * 0 + 0, 0, s=0)  # noop 保持图例顺序
    ax.set_xlim(0, 7); ax.set_ylim(0, 7)
    ax.set_xlabel('产品1 产量 x1'); ax.set_ylabel('产品2 产量 x2')
    ax.set_title('几何：割把 LP 顶点削掉，却不切任何整数点')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 右上：目标随割的轮次收敛 ----
    ax = axes[0, 1]
    rounds = [h["round"] for h in history]
    objs = [h["obj"] for h in history]
    ax.plot(rounds, objs, 'o-', color='darkred', lw=2, label='LP目标(随割下降)')
    ax.axhline(obj_milp, color='blue', ls='--', lw=1.5,
               label=f'整数最优 {obj_milp:.2f}')
    ax.axhline(obj_lp, color='gray', ls=':', lw=1.5,
               label=f'初始LP松弛 {obj_lp:.2f}')
    ax.set_xlabel('割平面轮次'); ax.set_ylabel('目标值 max 利润')
    ax.set_title('收敛：每加一条割，LP 上界被逼向整数最优')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 左下：割的推导——取分数的基行（表） ----
    ax = axes[1, 0]
    ax.axis('off')
    # 显示首条割的来源行与小数部分
    T0, basis0, n_full0 = None, None, None
    x0, obj0, T0, basis0, n_full0 = simplex(c, A, b)
    var = basis0[int(np.argmax([T0[i, -1] - np.floor(T0[i, -1]) if basis0[i] < n_full0 else 0
                                for i in range(len(basis0))]))]
    rowi = row_of_basis(T0, basis0, var)
    bval = T0[rowi, -1]
    f0 = bval - np.floor(bval)
    txt = (f"【Gomory 割是怎么来的】\n\n"
           f"LP 最优表中取分数的基行 (基变量 x{var+1}):\n"
           f"    x{var+1} = {bval:.4f} = {np.floor(bval):.0f} + {f0:.4f}\n\n"
           f"对整数解，任何变量都取整数，故\n"
           f"    Σ (系数的小数部分)·x  ≡  {f0:.4f}  (mod 1)\n"
           f"左边 ≥ 0，所以\n"
           f"    Σ (系数的小数部分)·x  ≥  {f0:.4f}\n\n"
           f"→ 这就是一条【合法】的新线性约束：\n"
           f"   它一定被所有整数解满足，\n"
           f"   却把当前 LP 的分数顶点切掉了。")
    ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10.5,
            transform=ax.transAxes)
    ax.set_title('割从哪来：LP 最优表 → 分数部分 → 合法不等式')

    # ---- 右下：LP松弛 vs 割平面 vs 整数最优 ----
    ax = axes[1, 1]
    labels = ['LP松弛', '割平面', '整数最优']
    vals = [obj_lp, obj_cp, obj_milp]
    colors = ['gray', 'magenta', 'blue']
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f'{v:.4f}', ha='center', fontsize=11)
    ax.axhline(obj_milp, color='blue', ls='--', lw=1)
    ax.set_ylim(17.5, 21.5)
    ax.set_ylabel('最大利润')
    ax.set_title(f'割平面把 LP 上界 {obj_lp:.2f} 收紧到整数最优 {obj_milp:.2f}')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gomory_cutting_planes.png")
    plt.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
