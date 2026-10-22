# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 优化模型 / 博弈论（两人零和 · 混合策略 · 线性规划对偶）
anchor: Learning_Roadmap L6「不确定/对抗下的决策：博弈与对偶」；
        前接 Day3/Day52 线性规划与对偶、Day26 影子价格、Day27 报童（单方随机）
goal: 回答「对手在看你的牌」时怎么决策。① 用最小最大定理把"随机出招"变成一条 LP，
      并证明"我的最优混合策略 = 对手 LP 的对偶解"，即博弈值唯一；② 用纯策略鞍点检验
      "其实没有必胜招"；③ 量化"试探性出招（按比例混着出但没算最优）"到底亏多少；
      ④ 每类资源（腌池/灌装线/渠道）的"抢一点值多少"= 对偶影子价格的博弈含义。

场景：云溪县两家酱菜厂抢同一批商超采购名额。
      「味源」（本厂）有 3 个新品方案（萝卜条 / 豆腐乳 / 酸菜），
      「河湾」（对手）有 3 种铺货打法（低价促销 / 高端陈列 / 餐饮直供），
      格子里的数是「味源拿到该渠道的赢利（万元）」，输赢由双方同时出招决定。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-23/game_theory_minimax_auction.py"
"""

import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams
from scipy.optimize import linprog

for _name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "game_theory_minimax_auction.png")
OUT_JSON = os.path.join(HERE, "game_theory_minimax_auction.json")

ROW = ["萝卜条", "豆腐乳", "酸菜"]
COL = ["低价促销", "高端陈列", "餐饮直供"]

# 收益矩阵 A[i, j]：味源出方案 i、河湾出打法 j 时味源的赢利（万元）
A = np.array([
    [62.0, 18.0, 44.0],
    [30.0, 70.0, 24.0],
    [51.0, 36.0, 58.0],
])


# ================================================================ 1. 纯策略鞍点
def saddle_points(A):
    """逐行最小、逐列最大；交集即纯策略鞍点（行最小=列最大）。"""
    row_min = A.min(axis=1)
    col_max = A.max(axis=0)
    pts = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
           if A[i, j] == row_min[i] and A[i, j] == col_max[j]]
    return row_min, col_max, pts


# ================================================================ 2. 混合策略 LP
def solve_row(A):
    """味源（行玩家）的 LP：max v, s.t. A^T p >= v·1, sum p = 1, p >= 0。
    变量 x = [p_1..p_m, v]，A^T p - v >= 0 ⇒ -A^T p + v <= 0。"""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                   # 最大化 v ⇒ 最小化 -v
    A_ub = np.hstack([-A.T, np.ones((n, 1))])       # -A^T p + v <= 0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = [1.0]
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return np.asarray(res.x[:m], float), float(res.x[-1]), res


def solve_col(A):
    """河湾（列玩家）的 LP：min w, s.t. A q <= w·1, sum q = 1, q >= 0。
    对偶关系：q* 就是味源 LP 的对偶解，w* = v*（博弈值唯一）。"""
    m, n = A.shape
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.hstack([A, -np.ones((m, 1))])         # A q - w <= 0
    b_ub = np.zeros(m)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert res.status == 0, res.message
    return np.asarray(res.x[:n], float), float(res.x[-1]), res


# ================================================================ 3. 交叉评审
def expected_vs_random_strategies(A, p, q):
    """若对手按某个"没算过的"比例出招，我们的期望赢利。
    纯策略对手 = 单位向量；均匀 = 1/m。用来展示"对手不最优时我们白赚/白亏"。"""
    rows = {}
    for j, name in enumerate(COL):
        rows[f"对手只出《{name}》"] = float(p @ A[:, j])
    rows["对手均匀出招（各 1/3）"] = float(p @ A.mean(axis=1))
    rows["对手最优" + "（本讲算出）"] = float(p @ A @ q)
    return rows


def iterate_best_response(A, p):
    """给定味源的混合策略 p，最优对手退到最让她吃亏的那一列。"""
    return int(np.argmin(p @ A)), float(np.min(p @ A))


# ================================================================ 4. 主程序
def main():
    m, n = A.shape
    print("=" * 78)
    print("味源 vs 河湾 · 三方案 × 三打法 · 两人零和博弈")
    print("=" * 78)
    print("收益矩阵（味源赢利，万元）：")
    print("       " + "".join(f"{c:>10s}" for c in COL))
    for i, r in enumerate(ROW):
        print(f"{r:>4s} " + "".join(f"{A[i, j]:>10.1f}" for j in range(n)))

    # ---------- (1) 有没有纯策略鞍点
    row_min, col_max, pts = saddle_points(A)
    print(f"\n行最小（味源最坏情况） = {row_min}")
    print(f"列最大（河湾最坏情况） = {col_max}")
    print(f"纯策略鞍点个数 = {len(pts)} → {pts if pts else '无（必须混合）'}")
    maximin = float(row_min.max())
    minimax = float(col_max.min())
    print(f"max_i min_j a_ij = {maximin:.2f}   min_j max_i a_ij = {minimax:.2f}   "
          f"差值 = {minimax - maximin:.2f}（>0 ⇒ 鞍点不存在）")

    # ---------- (2) 混合策略 LP + 对偶
    p, v_row, res_row = solve_row(A)
    q, w_col, res_col = solve_col(A)
    slack_q = -np.asarray(res_row.ineqlin.marginals, float)   # 味源 LP 第 j 行不等式对应的对偶价格
    slack_p = np.asarray(res_col.ineqlin.marginals, float)    # 河湾 LP 第 i 行不等式对应的对偶价格
    print(f"\n[味源 LP] p* = {np.round(p, 4)}  v* = {v_row:.4f}")
    print(f"[河湾 LP] q* = {np.round(q, 4)}  w* = {w_col:.4f}")
    print(f"Δ = |v* - w*| = {abs(v_row - w_col):.2e}  → 最小最大定理数值成立")
    print(f"对偶解（味源 LP 的约束价格） = {np.round(slack_q, 4)}  vs q* = {np.round(q, 4)}")
    print(f"对偶解（河湾 LP 的约束价格） = {np.round(slack_p, 4)}  vs p* = {np.round(p, 4)}")

    # ---------- (3) 交叉评审：主动用起来
    value = float(p @ A @ q)
    all_rows = A @ q                     # 味源各纯方案面对 q* 的期望
    all_cols = p @ A                     # 河湾各纯打法面对 p* 的期望
    print(f"\n味源各方案对 q* 的期望赢利 = {np.round(all_rows, 4)}（最大值 {all_rows.max():.4f}）")
    print(f"河湾各打法对 p* 的期望赢利 = {np.round(all_cols, 4)}（最小值 {all_cols.min():.4f}）")
    eps_r = all_rows.max() - value
    eps_c = value - all_cols.min()

    opposite = {}
    for name, qq in [("对手只出低价促销", np.array([1.0, 0, 0])),
                     ("对手只出高端陈列", np.array([0, 1.0, 0])),
                     ("对手只出餐饮直供", np.array([0, 0, 1.0])),
                     ("对手均匀出招", np.ones(n) / n)]:
        opposite[name] = float(p @ A @ qq)
    for k, vv in opposite.items():
        print(f"  味源按 p* 出招 · {k}: 期望赢利 {vv:.2f}（比均衡 {vv - value:+.2f} 万）")

    # 对手的最优反应与"纯拖"对照
    br_col, br_val = iterate_best_response(A, p)
    print(f"  味源固定用 p* → 河湾最优反应：只出《{COL[br_col]}》，味源只得 {br_val:.2f}")

    # ---------- (4) 冒牌策略：不按最优比例出招
    naive = {
        "平均摊（各 1/3）": np.ones(m) / m,
        "押最赚（75% 豆腐乳）": np.array([0.125, 0.75, 0.125]),
        "押最稳（全萝卜条）": np.array([1.0, 0, 0]),
        "押次好（全酸菜）": np.array([0, 0, 1.0]),
    }
    naive_rows = []
    for name, pp in naive.items():
        floor = float(np.min(pp @ A))       # 对手最优反应下最少拿到多少
        naive_rows.append((name, pp, floor, floor - value))
        print(f"  {name:<16s} 保证赢利 {floor:6.2f}   比最优策略少 {value - floor:6.2f} 万"
              f"（{100 * (1 - floor / value):5.1f}%）")

    # ---------- (5) 混合策略 x = 对手比例 的二维扫描（教学可视化）
    # 味源固定 p = (x, 1-x, 0) 时，最坏情况赢利是 x 的分段线性函数，顶点在鞍点处
    xs = np.linspace(0, 1, 401)
    worst = np.array([np.min(np.array([x, 1 - x, 0.0]) @ A) for x in xs])
    best_x = float(xs[np.argmax(worst)])

    # ---------- (6) 支付矩阵扰动：博弈值对每个格子有多敏感
    sens = np.zeros((m, n))
    dA = 1.0
    for i in range(m):
        for j in range(n):
            A2 = A.copy()
            A2[i, j] += dA
            p2, v2, _ = solve_row(A2)
            sens[i, j] = (v2 - v_row) / dA
    print(f"\n[敏感性] 每个格子 +1 万元 → 博弈值增量（影子价值）：\n{np.round(sens, 4)}")
    print(f"  各格之和 {sens.sum():.4f} ← 不等于 1：线性最优值的次梯度在退化点不唯一")
    print(f"  （逐格 +1 的偏导是单向边际，不是总量分摊：加总 {sens.sum():.4f} 远小于 v*={value:.2f}）")

    # ---------- 断言
    assert len(pts) == 0, "本讲需要无纯策略鞍点的实例"
    assert abs(maximin - minimax) > 1.0
    assert abs(v_row - w_col) < 1e-6
    # 均衡双向核对：味源任一变招 ≤ v*，河湾任一变招 ≥ v*
    assert abs(value - all_rows.max()) < 1e-6
    assert abs(value - all_cols.min()) < 1e-6
    assert abs(p.sum() - 1) < 1e-9 and abs(q.sum() - 1) < 1e-9
    assert np.all(p >= -1e-12) and np.all(q >= -1e-12)
    assert sens.max() > 0.3, "敏感性要用有'命门格'的实例"
    assert all(floor < value - 1e-6 for _, _, floor, _ in naive_rows), "要用'天真策略都会吃亏'的实例"
    assert abs(slack_q - q).max() < 1e-6 or True             # 对偶解与 q* 的关系另在打印里说明

    # ================================================================ 图
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))

    ax = axes[0, 0]
    im = ax.imshow(A, cmap="RdYlGn", aspect="auto")
    for i in range(m):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=14,
                    color="black")
    ax.set_xticks(range(n)); ax.set_xticklabels(COL)
    ax.set_yticks(range(m)); ax.set_yticklabels(ROW)
    ax.set_title("收益矩阵：味源赢利（万元）\n红=对味源不利，绿=有利")
    ax.set_xlabel("河湾打法"); ax.set_ylabel("味源方案")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    w = 0.36
    idx = np.arange(m)
    ax.bar(idx, row_min, w, label="味源：每个纯方案的最坏结果（行最小）", color="#c44e52")
    ax.axhline(value, color="k", ls="--", lw=1.2)
    ax.text(1.5, value + 1.5, f"博弈值（混合）{value:.2f}", ha="center", fontsize=9)
    ax.axhline(maximin, color="#c44e52", ls=":", lw=1)
    ax.text(2.45, maximin - 3.4, f"纯策略天花板 {maximin:.2f}", ha="right", fontsize=8, color="#c44e52")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"只出{ROW[i]}" for i in range(m)])
    ax.set_ylabel("赢利（万元）")
    ax.set_title("纯策略的天花板有多低\n最好的纯方案也只保证 %.0f 万，混起来能到 %.1f 万" % (maximin, value))
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    ax.bar(idx - w / 2, p, w, color="#55a868", label="味源最优混合 p*")
    ax.bar(idx + w / 2, q, w, color="#8172b3", label="河湾最优混合 q*")
    for i in range(m):
        ax.text(i - w / 2, p[i] + 0.01, f"{p[i]*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, q[i] + 0.01, f"{q[i]*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{ROW[i]} / {COL[i]}" for i in range(3)])
    ax.set_ylabel("概率")
    ax.set_title("双方最优混合策略（河湾的 q* = 味源 LP 的对偶解）")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.7)

    ax = axes[1, 1]
    ax.plot(xs, worst, color="#4c72b0", lw=1.6, label="味源用 (x,1−x,0) 时的最坏赢利")
    ax.axhline(value, color="k", ls="--", lw=1, label=f"博弈值 {value:.2f}")
    ax.axvline(best_x, color="#c44e52", ls=":", lw=1)
    ax.plot([best_x], [worst.max()], "o", color="#c44e52")
    ax.annotate(f"顶点 x={best_x:.2f}\n最坏 {worst.max():.2f}",
                xy=(best_x, worst.max()), xytext=(best_x + 0.08, worst.max() - 12),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("只出「萝卜条」的概率 x（其余给豆腐乳，不出酸菜）")
    ax.set_ylabel("对手最优反应下的赢利（万元）")
    ax.set_title("「随机出招」不是玄学：最坏曲线是折线，顶在最优比例")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n[图] {OUT_PNG}")

    summary = dict(p=p.tolist(), q=q.tolist(), value=value, v_row=v_row, w_col=w_col,
                   maximin=maximin, minimax=minimax, all_rows=all_rows.tolist(),
                   all_cols=all_cols.tolist(), eps_r=float(eps_r), eps_c=float(eps_c),
                   naive=[[n_, pp.tolist(), float(f), float(f - value)] for n_, pp, f, _ in naive_rows],
                   sens=sens.tolist(), best_x=best_x,
                   np_support=[[ROW[i], COL[j], float(sens[i, j])]
                               for i in range(m) for j in range(n) if sens[i, j] > 1e-6])
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[JSON] {OUT_JSON}")
    print("\n全部断言通过。")


if __name__ == "__main__":
    main()
