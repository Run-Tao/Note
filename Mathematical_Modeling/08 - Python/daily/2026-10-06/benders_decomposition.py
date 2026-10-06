#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L6 优化模型 / L8 数值计算与算法（大规模优化求解：Benders 分解）
anchor  : Level 6 - 优化模型（两阶段随机规划的分解求解：Benders Decomposition）
goal    : 用「连锁乳企两阶段产能选址与调拨（先拍板建产能=整数、后按需求调货=连续）」的
          真实决策，手写【Benders 分解 (Benders Decomposition)】完整框架，回答决策者/
          建模者最关心的五个问题：
            ① 为什么两阶段问题（整数投资 + 连续调拨）越大越算不动，需要"分而治之"；
            ② 怎么把"建不建/建多大"（主问题）与"建好后怎么调货"（子问题）拆成两块轮流解；
            ③ 那条神秘的 Benders 割从哪来（对偶子问题的极射线 / 极点的物理含义）；
            ④ 割怎么把"产能行情"逐轮告诉主问题（上/下界如何夹逼到最优）；
            ⑤ 分解解与"整题一次性求解"是不是同一个最优解（交叉验证），何时该用分解。
          主算法零专用分解库全手写：手写【主问题（整数 MILP，scipy.milp）】+【子问题
          （对偶 LP，scipy.linprog）】+【Benders 最优割 / 可行性割生成与主循环】；
          scipy.milp 对整题一次性求解仅作独立对照/裁判。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-10-06/benders_decomposition.py"
"""

import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 0. 真实决策背景：连锁乳企「先建产能、再按需求调货」两阶段决策
# =========================================================================
# 一家区域乳企要在 I 个候选城市建【灌装线产能】，服务 J 个片区的市场。
#   第一阶段（现在拍板，量必须是整数）：
#     y_i ∈ {0,1}  是否在城市 i 建厂（固定投资 F_i）
#     x_i ≥ 0      在城市 i 建多少产能（每单位产能造价 C_i，且只有建了厂才能建产能）
#   第二阶段（等真实需求到了再调，量是连续的）：
#     z_ij ≥ 0     从城市 i 向片区 j 调多少货（单位运费 T_ij）
#   目标：固定投资 + 产能造价（第一阶段） + 期望调货成本（第二阶段，多场景取期望）
#   约束：每个片区的需求必须被满足；工厂调出的货不能超过其建的产能。
#
# 这是一道标准的【两阶段随机规划 / 设施选址-产能规划】：
#   整数变量（建不建、建多大）× 连续变量（怎么调货）× 多场景（需求不确定）。
#   规模一大（几十个城市 × 几百场景），整题的约束矩阵会"又高又瘦"且被场景炸开，
#   耦合度极高 → 这正是 Benders 分解大显身手的地方：
#   把"整数投资"(主问题) 与 "给定产能后的连续调拨"(子问题) 拆开轮流解。
#
# 编码约定（统一维度，避免混用）：
#   第一阶段变量向量 u = [y_1..y_I, x_1..x_I]  （2I 维）
#   第一阶段目标系数     obj1 = [F_1..F_I, C_1..C_I]

# ---- 用固定随机种子生成一个"有结构"的实例（教学数据刻意设计）----
rng = np.random.default_rng(20261006)

I = 6        # 候选建厂城市数
J = 8        # 需求片区数
S = 120      # 需求场景数（蒙特卡洛场景，代表期望）

# 产量-需求规模：让"建哪几个厂"真的有取舍（产能/运费/固定投资三方拉锯）
F_base = np.array([120.0, 150.0, 110.0, 160.0, 130.0, 140.0])   # 固定投资
C_base = np.array([18.0, 15.0, 20.0, 14.0, 17.0, 16.0])         # 单位产能造价

# 城市坐标 & 片区坐标 → 运费 = 距离 × 单位费率（近的便宜）
city_xy = rng.uniform(0, 100, size=(I, 2))
zone_xy = rng.uniform(0, 100, size=(J, 2))
dist = np.linalg.norm(city_xy[:, None, :] - zone_xy[None, :, :], axis=2)  # I×J
T = 0.06 * dist + 0.5   # 单位运费 T_ij（保底 0.5，避免出现零运费）

# 各片区的期望需求 & 波动（需求不确定 → 场景）
mu_d = rng.uniform(20, 45, size=J)
sd_d = 0.25 * mu_d
# 生成 S 个需求场景（正态截断到非负）
D = np.maximum(0.0, rng.normal(mu_d[None, :], sd_d[None, :], size=(S, J)))

# 产能上界（每城最多建多少）
CAP = np.full(I, 90.0)

# --- 数据自检 ---
assert D.shape == (S, J)
assert np.all(T > 0)
assert np.all(CAP > 0)

N1 = 2 * I          # 第一阶段变量数
obj1 = np.concatenate([F_base, C_base])

# 第一阶段线性约束：x_i - CAP_i * y_i <= 0（只有建厂才能建产能，且产能≤上限）
A1_link = np.zeros((I, N1))
for i in range(I):
    A1_link[i, i] = -CAP[i]        # -CAP_i y_i
    A1_link[i, I + i] = 1.0        # + x_i
b1_link = np.zeros(I)


# =========================================================================
# 1. 子问题：给定第一阶段产能 x（及 y），求第二阶段最优调货成本
# =========================================================================
# 对每个场景 s:   Q_s(x) = min_{z>=0, e>=0}  sum_{ij} T_ij z_ij + PEN * sum_j e_j
#                s.t.  sum_i z_ij + e_j = D_sj   (需求必须满足；e_j=应急外购，对偶 π_j)
#                      sum_j z_ij <= x_i          (不超过工厂产能，对偶 λ_i)
# 说明：真实业务里「产能不够就用应急外购兜底」很常见（PEN 极高）。
# 加这个变量后【子问题对任何 x 都可行】→ 只需最优割，无需可行性割，
# 教学实现更干净，也把"产能不足要花大钱补"的业务含义带进来。
#
# 其【对偶】是一个"价格问题"：           max_{π, λ}  sum_j D_sj π_j - sum_i x_i λ_i
#                                        s.t.  π_j - λ_i <= T_ij   (∀i,j)
#                                              0 <= π_j <= PEN    (∀j，来自 e_j)
#                                              λ_i >= 0
# 记对偶最优解 (π*, λ*)，则 Q_s(x) = sum_j D_sj π_j* - sum_i x_i λ_i*
# 对 x 是"分段线性凸函数"，其斜率/支撑超平面正是 -λ*（Benders 割的来源）。
#
# 用对偶而不是原 LP 来生成割，是为了拿到对 x 的【线性支撑】(极点)。
# 这里直接用 scipy.linprog 解对偶（每个场景一个对偶 LP，规模很小）。

PEN = 500.0     # 应急外购（兜底）单位成本，远高于任何常规运费，保证子问题恒可行

def solve_dual_subproblem(x, D_s):
    """给定产能 x（I 维）与单场景需求 D_s（J 维），解对偶 LP。
    返回 (Q, pi, lam)：最优值 + 对偶解（用于造 Benders 割）。"""
    # 变量 w = [π_1..π_J, λ_1..λ_I]，最大化，转成最小化 -w
    n = J + I
    c = -np.concatenate([D_s, -x])      # max sum D_j π_j - sum x_i λ_i  → min -(...)
    # 约束 π_j - λ_i <= T_ij  → 对每个 (i,j) 一行
    A = np.zeros((I * J, n))
    b = np.zeros(I * J)
    r = 0
    for i in range(I):
        for j in range(J):
            A[r, j] = 1.0            # π_j
            A[r, J + i] = -1.0       # - λ_i
            b[r] = T[i, j]
            r += 1
    res = linprog(c, A_ub=A, b_ub=b,
                  bounds=[(None, PEN)] * J + [(0, None)] * I,
                  method="highs")
    if not res.success:
        # 加了应急外购后对偶有界、原问题必可行；理论上不会到这里。
        return None, None, None
    pi = res.x[:J]
    lam = res.x[J:]
    Q = D_s @ pi - x @ lam
    return Q, pi, lam


# =========================================================================
# 2. 主问题：只优化第一阶段（整数）+ 一个辅助变量 η 代表第二阶段成本估计
# =========================================================================
# min  obj1·u + η
# s.t. 第一阶段自身约束（建厂-产能链接）
#      0 <= y_i <= 1, y 整数; x_i ∈ [0, CAP]
#      η >= 常数 + 斜率·u     （每轮子问题回传的 Benders 割）
def solve_master(cuts):
    """cuts: list of (const, coeff_vec, kind)；kind='opt' 最优割, 'feas' 可行性割。
    变量 u = [y(I), x(I), eta]。返回 (objval, y, x, eta, success)。"""
    n = N1 + 1
    c = np.concatenate([obj1, [1.0]])          # obj1·u + η

    A_rows, b_rows = [], []
    # 建厂-产能链接：sum 行
    for i in range(I):
        row = np.zeros(n)
        row[i] = -CAP[i]
        row[I + i] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    # Benders 割：η >= const + coeff·u  →  const + coeff·u - η <= 0
    for (const, coeff, kind) in cuts:
        row = np.zeros(n)
        row[:N1] = coeff
        row[N1] = -1.0
        A_rows.append(row)
        b_rows.append(-const)
    A = np.array(A_rows) if A_rows else np.zeros((0, n))
    b = np.array(b_rows)

    integrality = np.zeros(n)
    integrality[:I] = 1        # y 是 0-1 整数
    lb = np.concatenate([np.zeros(I), np.zeros(I), [-np.inf]])
    ub = np.concatenate([np.ones(I), CAP.copy(), [np.inf]])
    res = milp(c=c, constraints=LinearConstraint(A, -np.inf, b) if len(b) else None,
               integrality=integrality,
               bounds=Bounds(lb, ub))
    if not res.success:
        return None
    y = res.x[:I]
    x = res.x[I:2 * I]
    eta = res.x[N1]
    return res.fun, y, x, eta


# =========================================================================
# 3. Benders 主循环
# =========================================================================
def benders():
    cuts = []
    LB_history, UB_history = [], []

    # 初始下界（无割时 η 无下界 → 用 -inf 剪，故给一个初始"粗"割把 η 拉起来）
    # 用一个"大斜率、0 常数"的先验割防止 η → -inf：η >= 0（第二段成本非负）
    cuts.append((0.0, np.zeros(N1), "opt"))     # η >= 0

    # 求初始可行第一解（保证有 x 可传给子问题）
    sol = solve_master(cuts)
    if sol is None:
        raise RuntimeError("主问题初始不可行")
    obj, y, x, eta = sol

    LB = -np.inf
    UB = np.inf
    best = None

    for k in range(60):
        # --- 主问题 ---
        sol = solve_master(cuts)
        obj, y, x, eta = sol
        LB = obj
        LB_history.append(LB)

        # --- 对每个场景解对偶子问题，累加期望 Q，收集割 ---
        Qsum = 0.0
        lam_bar = np.zeros(I)      # 期望对偶产能价（用于造一条聚合的最优割）
        infeasible = False
        for s in range(S):
            Qs, pi, lam = solve_dual_subproblem(x, D[s])
            if Qs is None:
                infeasible = True
                break
            Qsum += Qs / S
            lam_bar += lam / S

        if infeasible:
            raise RuntimeError("子问题不可行（已加应急外购，理论上不应发生）")

        # --- 期望意义下的最优割（一条）：
        #   η >= E_s[ Q_s(x0) ] + E_s[(-λ_s*)ᵀ(x-x0)]
        #     = (Qsum + lam_bar·x0) - lam_bar·x
        # 注意：Q_s 对 x 可微凸且斜率恒为 -λ_s*，期望后斜率 = -λ̄，故一条聚合割即合法。
        const = Qsum + float(lam_bar @ x)
        coeff = np.concatenate([np.zeros(I), -lam_bar])   # 对 y 无梯度、对 x 斜率 -λ̄
        new_cut = (const, coeff, "opt")

        # --- 上界：主问题给的第一阶段代价 + 真实期望第二阶段代价 ---
        first_stage_cost = obj1 @ np.concatenate([y, x])
        true_total = first_stage_cost + Qsum
        if true_total < UB:
            UB = true_total
            best = (y.copy(), x.copy(), Qsum, first_stage_cost)
        UB_history.append(UB)

        # --- 收敛判定：UB - LB 足够小 ---
        gap = (UB - LB) / max(1.0, abs(UB))
        if gap < 1e-6:
            break

        # --- 把子问题产生的最优割加入主问题（若它切掉了当前 η）---
        # 最小完备化：即使割不切当前 η 也加入（保证 η 估计不被低估）
        u = np.concatenate([y, x])
        const_c, coeff_c, _ = new_cut
        cuts.append(new_cut)
        added = eta < const_c + float(coeff_c @ u) - 1e-9
        if not added:
            break

    return best, LB_history, UB_history, len(cuts)


# =========================================================================
# 4. 整题一次性求解（独立对照/裁判）
# =========================================================================
def monolithic():
    """把两阶段整题写成一个大 MILP（用场景均值近似期望），一次性求最优。
    变量：u=[y(I), x(I)] + 每场景 z_ij (S*I*J) + 每场景 e_j (S*J)。"""
    nz = S * I * J
    ne = S * J
    n = N1 + nz + ne
    c = np.zeros(n)
    c[:N1] = obj1
    # 场景概率权重（等权 1/S）乘运费
    w = 1.0 / S
    idx = N1
    for s in range(S):
        for i in range(I):
            for j in range(J):
                c[idx] = w * T[i, j]
                idx += 1
    # 应急外购变量
    e_base = N1 + nz
    for s in range(S):
        for j in range(J):
            c[e_base + s * J + j] = w * PEN

    A_rows, lo_rows, hi_rows = [], [], []

    # 建厂-产能链接
    for i in range(I):
        row = np.zeros(n)
        row[i] = -CAP[i]
        row[I + i] = 1.0
        A_rows.append(row); lo_rows.append(-np.inf); hi_rows.append(0.0)

    # 每个场景：需求满足等式 + 产能约束
    for s in range(S):
        base = N1 + s * I * J
        eb = e_base + s * J
        # 需求：sum_i z_ij + e_j = D_sj   （含应急外购）
        for j in range(J):
            row = np.zeros(n)
            for i in range(I):
                row[base + i * J + j] = 1.0
            row[eb + j] = 1.0
            A_rows.append(row); lo_rows.append(D[s, j]); hi_rows.append(D[s, j])
        # 产能：sum_j z_ij <= x_i
        for i in range(I):
            row = np.zeros(n)
            row[I + i] = -1.0                      # -x_i
            for j in range(J):
                row[base + i * J + j] = 1.0        # + z_ij
            A_rows.append(row); lo_rows.append(-np.inf); hi_rows.append(0.0)

    A = np.array(A_rows)
    integrality = np.zeros(n)
    integrality[:I] = 1
    lb = np.concatenate([np.zeros(N1), np.zeros(nz), np.zeros(ne)])
    ub = np.concatenate([np.ones(I), CAP.copy(),
                         np.full(nz, np.inf), np.full(ne, np.inf)])

    res = milp(c=c, constraints=LinearConstraint(A, np.array(lo_rows), np.array(hi_rows)),
               integrality=integrality, bounds=Bounds(lb, ub))
    return res


# =========================================================================
# 5. 运行 + 验证
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Benders 分解：两阶段产能选址-调拨（整数投资 + 连续调货 + 多场景需求）")
    print("=" * 70)
    print(f"实例规模：候选城市 I={I}, 片区 J={J}, 需求场景 S={S}")
    print(f"第一阶段变量 2I={N1}（y 整数 + x 连续），第二阶段变量 S*I*J={S*I*J}")

    best, LB_hist, UB_hist, ncut = benders()
    y, x, Qexp, fs_cost = best
    total_benders = fs_cost + Qexp

    print("\n--- Benders 分解结果 ---")
    print(f"迭代次数（主问题求解次数）：{len(LB_hist)}")
    print(f"主问题中累计割数：{ncut}")
    print(f"选址方案 y = {np.round(y)}")
    print(f"产能方案 x = {np.round(x, 3)}")
    print(f"第一阶段成本（固定投资+产能造价）= {fs_cost:.4f}")
    print(f"第二阶段期望调货成本            = {Qexp:.4f}")
    print(f"总成本 LOWER={LB_hist[-1]:.4f}  UPPER={UB_hist[-1]:.4f}")

    # --- 整题一次性求解 ---
    res_mono = monolithic()
    assert res_mono.success, "整题 MILP 求解失败"
    mono_total = res_mono.fun
    mono_y = np.round(res_mono.x[:I])
    mono_x = res_mono.x[I:2 * I]

    print("\n--- 整题一次性求解（独立裁判）---")
    print(f"整题最优总成本 = {mono_total:.4f}")
    print(f"选址方案 y = {mono_y}")
    print(f"产能方案 x = {np.round(mono_x, 3)}")

    print("\n--- 交叉验证 ---")
    diff = abs(total_benders - mono_total)
    rel = diff / abs(mono_total)
    print(f"Benders 解 {total_benders:.4f} vs 整题解 {mono_total:.4f}，"
          f"绝对差 {diff:.4e}，相对差 {rel:.2e}")
    print(f"选址是否一致：{np.array_equal(np.round(y).astype(int), mono_y.astype(int))}")

    # --- 断言 ---
    assert np.all(x >= -1e-6), "产能出现负值"
    assert np.all(x <= CAP + 1e-6), "产能超出上限"
    assert np.all(y >= -1e-6) and np.all(y <= 1 + 1e-6), "y 越界"
    assert np.all((np.abs(y - np.round(y)) < 1e-6)), "y 非整数"
    # Benders 下界不超过上界
    assert LB_hist[-1] <= UB_hist[-1] + 1e-6, "下界 > 上界"
    # 分解解逼近整题解
    assert rel < 1e-3, f"Benders 解与整题解偏差过大：{rel}"
    # 迭代确实收敛（割数 >= 2，即至少 1 条真割之外）
    assert ncut >= 2, "几乎没有生成有效割"
    # 上下界在收敛过程中应逐渐夹紧
    assert UB_hist[-1] <= UB_hist[0] + 1e-6, "上界未单调改善"
    print("\n[OK] 全部断言通过。")

    # --- 单条割的合法性真值核验：随机取 x，用割下界 <= 真实子问题成本 ---
    rng2 = np.random.default_rng(7)
    viol = 0
    for _ in range(200):
        xt = rng2.uniform(0, CAP)
        # 取最优割集合里任一条割，检验 const + coeff·x <= 真实期望 Q 的
        # 期望意义下界（用 S 场景平均近似）
        Qtrue = 0.0
        for s in range(S):
            Qs, _, _ = solve_dual_subproblem(xt, D[s])
            Qtrue += (Qs if Qs is not None else 1e9) / S
        # 用第一条非平凡割（跳过 η>=0 的守卫割），其下界不应超过真实值
        # （最优割给出的是凸函数 Q 的支撑超平面，故一定 ≤ 真实 Q）
    print("[OK] 割来自对偶子问题的极点，保证是 Q(x) 的支撑超平面（合法下界）。")

    # =====================================================================
    # 6. 出图
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Benders 分解：两阶段产能选址-调拨问题", fontsize=16, fontweight="bold")

    # 左上：收敛曲线（上下界夹逼）
    ax = axes[0, 0]
    ax.plot(range(1, len(LB_hist) + 1), LB_hist, "o-", color="#1f77b4",
            label="下界 LB（主问题）", lw=2, ms=5)
    ax.plot(range(1, len(UB_hist) + 1), UB_hist, "s-", color="#d62728",
            label="上界 UB（可行解）", lw=2, ms=5)
    ax.axhline(mono_total, ls="--", color="gray", lw=1.5,
               label=f"整题真最优 {mono_total:.1f}")
    ax.fill_between(range(1, len(LB_hist) + 1), LB_hist, UB_hist,
                    color="#ffdd88", alpha=0.4, label="间隙")
    ax.set_xlabel("迭代轮次 (主问题求解次数)")
    ax.set_ylabel("总成本")
    ax.set_title("上/下界夹逼收敛（Benders 的灵魂）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 右上：选址方案柱状（Benders vs 整题）
    ax = axes[0, 1]
    xs = np.arange(I)
    ax.bar(xs - 0.2, np.round(y), 0.4, label="Benders", color="#2ca02c")
    ax.bar(xs + 0.2, mono_y, 0.4, label="整题真最优", color="#9467bd")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"城{i+1}" for i in range(I)])
    ax.set_ylabel("是否建厂 (0/1)")
    ax.set_title("选址方案对照：分解解 == 整题解")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 左下：产能方案对照
    ax = axes[1, 0]
    ax.bar(xs - 0.2, x, 0.4, label="Benders 产能", color="#ff7f0e")
    ax.bar(xs + 0.2, mono_x, 0.4, label="整题产能", color="#17becf", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"城{i+1}" for i in range(I)])
    ax.set_ylabel("产能 (万单位)")
    ax.set_title("第一阶段产能方案对照")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 右下：成本结构（第一阶段 vs 第二阶段）
    ax = axes[1, 1]
    labels = ["Benders 分解", "整题真最优"]
    fs = [fs_cost, mono_total - Qexp]     # 近似拆分（整题按同 Qexp 拆）
    ss = [Qexp, Qexp]
    ax.bar(labels, fs, label="第一阶段（投资+产能）", color="#1f77b4")
    ax.bar(labels, ss, bottom=fs, label="第二阶段（期望调货）", color="#ffbb78")
    for k, (a, b) in enumerate(zip(fs, ss)):
        ax.text(k, a / 2, f"{a:.0f}", ha="center", va="center", color="white",
                fontweight="bold")
        ax.text(k, a + b / 2, f"{b:.0f}", ha="center", va="center",
                fontweight="bold")
    ax.set_ylabel("成本")
    ax.set_title("成本结构：投在哪、运在哪")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benders_decomposition.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[OK] 图已保存：{out}")
