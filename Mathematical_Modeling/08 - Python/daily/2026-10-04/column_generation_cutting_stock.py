#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L6 优化模型 / L8 数值计算与算法（列生成 + Dantzig-Wolfe 分解 + 割平面/最短路定价）
anchor  : Level 6 - 优化模型（大规模整数规划的分解求解：列生成）
goal    : 用「钢材厂接到一批不同长度钢管的订单，每根标准钢坯怎么切最省料」决策
          =【一维下料问题 Cutting Stock Problem】，手写【列生成法 Column Generation +
          受限主问题 RMP + 定价子问题(0-1 背包) + 整数化】完整框架，回答决策者/建模者
          最关心的五个问题：
            ① 当"每个可行切割方案"都变成一个变量、方案数指数爆炸时，怎么建模型还能解；
            ② 怎么"需要哪个方案就现场生成哪个方案"（列 = 切割模式，逐列加入）；
            ③ 为什么"先解 LP、再整数化"比"直接解整数规划"快几个数量级；
            ④ 定价子问题（最短路 / 背包）如何告诉我们"还缺哪种切割模式"（检验数/影子价格）；
            ⑤ 列生成解离真最优差多远（LP 下界 vs 整数解的间隙）；用分式解证明"要取整"。
          主算法零专用列生成库全手写；scipy.linprog 解 RMP、scipy.milp 求真最优仅作独立对照。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-10-04/column_generation_cutting_stock.py"
"""

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import itertools
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 0. 真实决策背景：钢材厂的一维下料（切割库存）问题
# =========================================================================
# 某钢材厂库存里全是【同一长度 L 的标准钢坯】（比如每根 20 米）。
# 客户下单要一批【不同长度、不同数量】的成品钢管：
#   - 成品规格 r（长度 len_r，数量 demand_r），比如 4m×120 根、7m×80 根 …
# 每根标准钢坯要按【某种切割模式（cutting pattern）】切成若干成品，
# 剩余的部分是【废料（trim loss）】。
#
# 问：怎么切，使【用掉的标准钢坯总根数】最少（= 总废料最少）。
#
# 【建模关键】两种等价建模方式：
#   (A) 按"每种规格切几根"（item-based，变量 = 第 i 根坯切出的各规格数量）——
#       坯的根数未知、且对称性极强，直接整数规划难解；
#   (B) 按"用每种切割模式几次"（pattern-based，变量 = 模式的使用次数）——
#       这就是【Gilmore–Gomory 下料模型】，也是列生成的经典载体：
#       每个「恰好不超长的一整根坯的切法」是一个变量（一列）。
#
# 难点：切割模式的数量【指数爆炸】——不能枚举全部列后再解 LP。
# 【列生成法】正是为此而生：不预先枚举所有列，而是「解一个只含部分列的
# 受限主问题 RMP → 用对偶价向"定价子问题"要一个最该加入的新列 → 加入 → 再解」，
# 如此迭代直到没有更优的列可加（即 LP 达到最优）。
#
# 定价子问题：给定 RMP 的对偶价 y_r（第 r 种规格的"边际价值"），
# 找一个"不超 L 长、使 1 - Σ y_r a_r 最小"的切割模式 a（a_r = 该模式切出规格 r 的根数）。
# 这正是一个【有界背包问题】（规格 r 的"利润"= y_r、"重量"= len_r，找利润最大的模式），
# 若该模式检验数 1 - 利润 < 0，说明列生成还没收敛，把它加入 RMP。

L = 25.0  # 每根标准钢坯长度（米）


def generate_instance():
    """一个真实感十足的下料实例：4 种规格，长度与需求量都不同。
    刻意选用「LP 松弛含分数解、整数最优比 LP 下界多 1 根」的实例，
    这样'必须取整'与'列生成只给下界'两个教学点才立得起来
    （教学数据要刻意设计以暴露算法行为，与 Day22/24/25/26/31/33/34/35 同一教训）。"""
    lengths = np.array([10.0, 8.0, 4.0, 3.0])   # 规格长度（米）
    demands = np.array([42, 145, 158, 91])       # 各规格需求根数
    return lengths, demands


# ---------------------------------------------------------------------
# 1. 把所有「极简切割模式」暴力枚举出来（仅用于小规模对照/初始列）
# ---------------------------------------------------------------------
def all_patterns(lengths):
    """枚举所有满足 Σ len_r * a_r <= L 的切割模式 a（a_r 为非负整数）。"""
    n = len(lengths)
    pats = []
    max_cnt = [int(np.floor(L / lengths[r])) for r in range(n)]
    for combo in itertools.product(*[range(m + 1) for m in max_cnt]):
        if sum(lengths[r] * combo[r] for r in range(n)) <= L + 1e-9:
            pats.append(combo)
    return pats


def initial_patterns(lengths, n_reg):
    """给一批"平凡初始列"，保证 RMP 可行（每种规格"独切一根"的模式）。"""
    pats = []
    for r in range(n_reg):
        a = [0] * n_reg
        a[r] = int(np.floor(L / lengths[r]))
        if a[r] > 0:
            pats.append(tuple(a))
    return pats


# ---------------------------------------------------------------------
# 2. 受限主问题 RMP：min Σ_p x_p s.t. Σ_p a_{rp} x_p >= demand_r, x>=0
# ---------------------------------------------------------------------
def solve_rmp(lengths, demands, patterns):
    """解 LP 松弛：用当前这批列（patterns）求最优 x 与对偶价 y。"""
    A = np.array(patterns, dtype=float).T        # (n_reg, n_pat)
    n_pat = A.shape[1]
    c = np.ones(n_pat)                           # 每用一次模式，成本 = 1 根坯
    # 约束：Σ_p a_{rp} x_p >= demand_r  <=>  -A x <= -demand
    res = linprog(c, A_ub=-A, b_ub=-demands, bounds=[(0, None)] * n_pat,
                  method="highs")
    assert res.success, res.message
    # 对偶价：linprog 的边际值；A_ub 形式下对偶价 y_r = -marginal（因为写成 -Ax<=-d）
    y = -res.ineqlin.marginals
    return res.fun, res.x, y


# ---------------------------------------------------------------------
# 3. 定价子问题：有界背包 —— 给定对偶价 y，找"利润最大"的切割模式
# ---------------------------------------------------------------------
def pricing_knapsack(lengths, y):
    """
    有界背包：max Σ y_r a_r  s.t. Σ len_r a_r <= L, a_r ∈ {0,1,...,floor(L/len_r)}。
    用 DP（按长度离散到 0.1 米）求"利润最大的切割模式"。
    返回 (最优利润, 模式 a, 检验数 reduced_cost = 1 - 利润)。
    """
    scale = 10  # 长度离散到 0.1 米，避免浮点
    cap = int(round(L * scale))
    # 尺寸向上取整（保守）：宁可少装也绝不超出真实长度，保证 DP 出的模式一定可用。
    # （若用 round 会因 25.04m 被舍成 250→25.0m 而把"超长模式"误判为可行，
    #   见下方避坑记录第 3 条。）
    sizes = [int(np.ceil(lengths[r] * scale - 1e-9)) for r in range(len(lengths))]
    max_cnt = [cap // s for s in sizes]
    # 多重有界背包：逐物品做 DP，记录选择（用二进制拆分的简化版：逐件枚举）
    # dp[w] = 最大利润；choice 记录该状态选了多少件规格 r（回溯用）
    # 规格数很少（4），直接用"分组有界背包"：对每种规格 r，枚举取 0..max_cnt[r] 件。
    n = len(lengths)
    dp = np.full(cap + 1, -np.inf)
    dp[0] = 0.0
    parent = [dict() for _ in range(cap + 1)]  # parent[w][r] = 用了多少件 r
    for r in range(n):
        new_dp = dp.copy()
        new_parent = [dict(p) for p in parent]
        for w in range(cap + 1):
            if dp[w] == -np.inf:
                continue
            for k in range(1, max_cnt[r] + 1):
                nw = w + k * sizes[r]
                if nw > cap:
                    break
                val = dp[w] + k * y[r]
                if val > new_dp[nw] + 1e-12:
                    new_dp[nw] = val
                    p = dict(parent[w])
                    p[r] = p.get(r, 0) + k
                    new_parent[nw] = p
        dp, parent = new_dp, new_parent
    best_w = int(np.argmax(dp))
    best_val = dp[best_w]
    if best_val == -np.inf:
        best_val = 0.0
        best_w = 0
    pat = tuple(parent[best_w].get(r, 0) for r in range(n))
    reduced_cost = 1.0 - best_val       # < 0 表示该列能改进 RMP
    return best_val, pat, reduced_cost


# ---------------------------------------------------------------------
# 4. 列生成主循环
# ---------------------------------------------------------------------
def column_generation(lengths, demands, seed_patterns, max_iter=60, verbose=True):
    patterns = list(seed_patterns)
    history = []           # (iter, LP_obj, n_cols, new_rc)
    for it in range(max_iter):
        obj, x, y = solve_rmp(lengths, demands, patterns)
        best_val, new_pat, rc = pricing_knapsack(lengths, y)
        history.append((it, obj, len(patterns), rc))
        if verbose:
            print(f"  [CG iter {it:2d}] LP下界={obj:8.4f}  列数={len(patterns):3d}  "
                  f"定价检验数={rc:+.6f}  新列={new_pat}")
        if rc >= -1e-9:
            if verbose:
                print("  收敛：无检验数为负的列可加，LP 已达最优。")
            break
        if new_pat in patterns:
            if verbose:
                print("  新列已存在，停止（数值收敛）。")
            break
        patterns.append(new_pat)
    return obj, x, y, patterns, history


# ---------------------------------------------------------------------
# 5. 整数化：把 LP 解取整成整数下料方案
# ---------------------------------------------------------------------
def round_up_integer(lengths, demands, patterns, x_lp):
    """简单稳妥的整数化：对 LP 解向上取整（保证可行），再给一个朴素对照。"""
    x_int = np.ceil(x_lp - 1e-9).astype(int)
    A = np.array(patterns, dtype=float).T
    used = int(x_int.sum())
    # 校验满足需求
    got = A @ x_int
    assert np.all(got >= demands - 1e-9), (got, demands)
    return x_int, used


def solve_exact_milp(lengths, demands, patterns):
    """用 MILP 求该下料问题的真最优（仅作对照）。整数变量 x_p >= 0。"""
    A = np.array(patterns, dtype=float).T
    n_pat = A.shape[1]
    c = np.ones(n_pat)
    constraints = LinearConstraint(A, lb=demands, ub=np.inf)
    res = milp(c=c, constraints=constraints,
               integrality=np.ones(n_pat), bounds=Bounds(0, np.inf))
    assert res.success, res.message
    return res.fun, res.x


def gomory_gap_demo(lengths, demands):
    """说明"必须取整"：报出 LP 最优里"分数根数"的列数（LP 解不可执行）。"""
    pats = all_patterns(lengths)
    obj, x, y = solve_rmp(lengths, demands, pats)
    frac = np.sum((x > 1e-6) & (x < 1 - 1e-6))
    return obj, frac, len(pats)


# =========================================================================
# 主流程
# =========================================================================
def main():
    lengths, demands = generate_instance()
    n_reg = len(lengths)
    print("=" * 74)
    print("一维下料问题（Cutting Stock）· 列生成法 Column Generation")
    print("=" * 74)
    print(f"标准钢坯长度 L = {L} 米")
    print(f"规格长度 = {lengths}, 需求 = {demands}")
    print(f"需求总长度 = {float(np.sum(lengths * demands)):.1f} 米 "
          f"→ 理论最少钢坯数(下界) = "
          f"{float(np.ceil(np.sum(lengths * demands) / L))} 根")
    print()

    # ---- 全枚举列（仅用于说明"列有多少"与做对照） ----
    all_pats = all_patterns(lengths)
    print(f"【全部极简切割模式数（暴力枚举）】= {len(all_pats)} 列")
    print()

    # ---- 列生成 ----
    seed = initial_patterns(lengths, n_reg)
    print(f"初始列（{len(seed)} 个平凡模式）：{seed}")
    print("-" * 74)
    cg_obj, cg_x, cg_y, cg_patterns, hist = column_generation(lengths, demands, seed)
    print("-" * 74)
    print(f"列生成结束：用了 {len(cg_patterns)} 列，LP 下界 = {cg_obj:.4f} 根")
    print()

    # ---- LP 全列对照（验证列生成没有得到更差/不同的 LP 下界） ----
    lp_full_obj, _, _ = solve_rmp(lengths, demands, all_pats)
    print(f"【对照1】全列 LP 下界 = {lp_full_obj:.4f} 根 "
          f"（列生成 LP 下界 {cg_obj:.4f}，差 {abs(lp_full_obj - cg_obj):.2e}）")
    print()

    # ---- 整数化 ----
    x_int, used_lp_round = round_up_integer(lengths, demands, cg_patterns, cg_x)
    print(f"【LP 解整数化（向上取整）】用钢坯 = {used_lp_round} 根")
    # "分数根数"= 用量不是整数的列（列生成 LP 解常含分数，故 LP 解不可直接执行）
    frac_cols = int(np.sum(np.abs(cg_x - np.round(cg_x)) > 1e-6))
    frac_detail = [(p, round(float(xi), 4)) for p, xi in zip(cg_patterns, cg_x)
                   if abs(float(xi) - round(float(xi))) > 1e-6 and xi > 1e-6]
    print(f"  → LP 最优解含 {frac_cols} 个'分数根数'的列（不可执行，必须取整）：{frac_detail}")
    print()

    # ---- MILP 真最优（对照） ----
    milp_obj, milp_x = solve_exact_milp(lengths, demands, cg_patterns)
    print(f"【对照2】用列生成得到的列做 MILP 整数最优 = {milp_obj:.0f} 根")
    milp_full_obj, _ = solve_exact_milp(lengths, demands, all_pats)
    print(f"【对照3】全列 MILP 整数最优 = {milp_full_obj:.0f} 根 "
          f"（与上一致说明列生成拿到的列已足够）")
    print()

    # ---- 整数下界：ceil(LP 下界) 是整数最优的合法下界 ----
    int_lb = int(np.ceil(cg_obj - 1e-9))
    print(f"【整数下界】ceil(LP下界) = {int_lb} 根 ≤ 整数最优 {int(milp_full_obj)} 根")
    print()

    # ---- 打印最终下料方案 ----
    print("最终下料方案（每行 = 一种切割模式，用量 = 根数）：")
    print(f"  模式(按规格 [" + ", ".join(f"{x:.0f}" for x in lengths) + "])")
    print(f"  {'':<24}{'用量':>6}")
    for p, cnt in sorted(zip(cg_patterns, x_int), key=lambda t: -t[1]):
        if cnt > 0:
            used_len = sum(lengths[r] * p[r] for r in range(n_reg))
            waste = L - used_len
            print(f"  {str(p):<40}{cnt:>6}    (用料{used_len:.0f}m, 废料{waste:.0f}m)")
    print()

    # =====================================================================
    # 断言验证
    # =====================================================================
    print("=" * 74)
    print("断言验证")
    print("=" * 74)

    # 1) 列生成 LP 下界 == 全列 LP 下界（列生成收敛到真 LP 最优）
    assert abs(cg_obj - lp_full_obj) < 1e-6, (cg_obj, lp_full_obj)
    print("✓ 1. 列生成 LP 下界 == 全列 LP 下界（列生成收敛到 LP 真最优）")

    # 2) LP 下界 <= 整数最优（下界性质）
    assert cg_obj <= milp_full_obj + 1e-6
    print("✓ 2. LP 下界 <= 整数最优（下界性质成立）")

    # 3) 用少量列做 MILP 的结果 == 用全部列做 MILP 的结果（列足够）
    assert abs(milp_obj - milp_full_obj) < 1e-6
    print("✓ 3. 列生成拿到的列做 MILP == 全列 MILP 最优（列已足够）")

    # 4) 整数化方案可行且 >= 整数最优
    assert used_lp_round >= milp_full_obj - 1e-9
    print("✓ 4. 向上取整方案可行，且 >= 整数最优")

    # 5) LP 解确实含分数（"必须取整"的证据）
    assert frac_cols >= 1
    print("✓ 5. LP 最优解含分数根数（LP 解不可执行，必须整数化）")

    # 6) 列生成只用了远少于全枚举的列
    assert len(cg_patterns) < 0.5 * len(all_pats)
    print(f"✓ 6. 列生成仅用 {len(cg_patterns)} 列 / 全枚举 {len(all_pats)} 列")

    # 7) 定价子问题 DP 正确性：与"同一离散尺度"的暴力枚举比（小规模随机背包）
    #    注意：DP 的容量/尺寸是向上取整离散的，故暴力枚举也用同样的整数尺度作对照，
    #    否则浮点误差会让"DP 合法而暴力以为超长"，得出假失败。
    rng = np.random.default_rng(0)
    scale = 10
    cap = int(round(L * scale))
    for _ in range(20):
        ls = rng.uniform(2.0, 8.0, 4)
        y = rng.uniform(0.05, 0.5, 4)
        best_val, pat, rc = pricing_knapsack(ls, y)
        sz = [int(np.ceil(ls[r] * scale - 1e-9)) for r in range(4)]
        mx = [cap // s for s in sz]
        bf = max(sum(y[r] * a[r] for r in range(4))
                 for a in itertools.product(*[range(mx[r] + 1) for r in range(4)])
                 if sum(sz[r] * a[r] for r in range(4)) <= cap)
        assert abs(best_val - bf) < 1e-6, (best_val, bf, ls, y)
    print("✓ 7. 定价背包 DP == 同尺度暴力枚举（20 个随机小实例）")

    print()
    print("=" * 74)
    print(f"结论：用 {len(cg_patterns)} 列（全枚举 {len(all_pats)} 列，省 "
          f"{100*(1-len(cg_patterns)/len(all_pats)):.0f}%）")
    print(f"      LP 下界 {cg_obj:.4f} → 整数最优 {int(milp_full_obj)} 根钢坯")
    print("=" * 74)

    # =====================================================================
    # 画图
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：列生成迭代收敛（LP 下界随列数增加而上升，逼近全列 LP 下界）
    ax = axes[0, 0]
    iters = [h[0] for h in hist]
    objs = [h[1] for h in hist]
    ncol = [h[2] for h in hist]
    ax.plot(iters, objs, "o-", color="#1f77b4", label="RMP 目标(LP下界)")
    ax.axhline(lp_full_obj, color="green", ls="--", lw=1.5,
               label=f"全列 LP 下界 {lp_full_obj:.3f}")
    ax.axhline(milp_full_obj, color="red", ls=":", lw=1.5,
               label=f"整数最优 {int(milp_full_obj)}")
    ax.set_xlabel("列生成迭代轮次")
    ax.set_ylabel("目标值（钢坯根数）")
    ax.set_title("列生成：LP 下界随新列加入而抬升，收敛到全列 LP 下界")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 右上：列数增长与检验数
    ax = axes[0, 1]
    rcs = [h[3] for h in hist]
    ax.plot(iters, rcs, "s-", color="#ff7f0e", label="定价检验数 $1-\\sum y_r a_r$")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("列生成迭代轮次")
    ax.set_ylabel("定价检验数")
    ax.set_title("定价子问题检验数：降到 0 以下即无列可加 → 收敛")
    ax2 = ax.twinx()
    ax2.plot(iters, ncol, "^:", color="#2ca02c", label="当前列数")
    ax2.set_ylabel("当前列数", color="#2ca02c")
    ax.legend(loc="upper right", fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # 左下：最终下料方案（各切割模式用量）柱状
    ax = axes[1, 0]
    labels, vals = [], []
    for p, cnt in sorted(zip(cg_patterns, x_int), key=lambda t: -t[1]):
        if cnt > 0:
            labels.append("×".join(f"{p[r]}" for r in range(n_reg)))
            vals.append(cnt)
    ax.bar(range(len(vals)), vals, color="#9467bd")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("切割模式（各界别切出根数）")
    ax.set_ylabel("使用根数")
    ax.set_title(f"最终下料方案：共 {used_lp_round} 根钢坯")
    ax.grid(alpha=0.3, axis="y")

    # 右下：下界-最优-上界夹逼 + 方法对照
    ax = axes[1, 1]
    names = ["LP下界\n(列生成)", "整数下界\nceil(LP)", "整数最优\n(MILP)", "取整方案\n(上界)"]
    values = [cg_obj, int_lb, milp_full_obj, used_lp_round]
    colors = ["#1f77b4", "#17becf", "#d62728", "#ff7f0e"]
    bars = ax.bar(names, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.3g}",
                ha="center", fontsize=10)
    ax.set_ylabel("钢坯根数")
    ax.set_title("下界—整数最优—上界 夹逼")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Day36 · 一维下料问题与列生成法（Column Generation）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(__file__).with_name("column_generation_cutting_stock.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
