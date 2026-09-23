# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L6 优化模型 / L7 图论、网络与动态模型
roadmap: Day23 · 运输问题（Transportation Problem）与最小费用最大流（Min-Cost Max-Flow）
ref:     Mathematical_Modeling/05 - Algorithms/Day23 - 运输问题最小费用流与西北角Vogel法.md

一道"工厂 -> 仓库"的运输决策，把三种解法一次跑通并交叉验证：
  ① 西北角法（NW）  —— 找一个可行初始解（不求优）
  ② Vogel 近似法（VAM）—— 价差启发式，初始解常常已接近最优
  ③ 最小费用流（MCMF）—— 图上逐步增广，求全局最优
再叠加两个现实变体：
  A. 供不应求（总供给 < 总需求）→ 虚设产地吸收缺口
  B. 某路线禁运 → 该弧置大 M

全程手写、零专用运筹库（不依赖 networkx / scipy.optimize.linprog）。
"""

import itertools
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 中文字体（系统已注册 Noto Sans CJK SC）----
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

BIG = 10**7  # 大 M


# =====================================================================
# 0. 问题数据
# =====================================================================
# 3 个工厂（产地）A1..A3 的产量，4 个仓库（销地）B1..B4 的需求
SUPPLY = [20, 25, 15]          # 合计 60
DEMAND = [18, 22, 12, 8]       # 合计 60（产销平衡）
# 单位运费矩阵 cost[i][j]：工厂 i -> 仓库 j
COST = np.array([
    [8, 6, 10, 9],
    [9, 7, 6, 12],
    [5, 11, 6, 7],
], dtype=float)


# =====================================================================
# 1. 西北角法：从左上开始，尽量满足，得到一个基本可行解
# =====================================================================
def northwest_corner(supply, demand):
    s = list(supply)
    d = list(demand)
    m, n = len(s), len(d)
    x = np.zeros((m, n))
    i = j = 0
    while i < m and j < n:
        q = min(s[i], d[j])
        x[i, j] = q
        s[i] -= q
        d[j] -= q
        if s[i] == 0 and j < n:
            i += 1
        elif d[j] == 0 and i < m:
            j += 1
        else:
            break
    return x


# =====================================================================
# 2. Vogel 近似法（VAM）：每行列取"次小-最小"价差，先满足价差最大者
# =====================================================================
def vogel(supply, demand, cost):
    s = list(supply)
    d = list(demand)
    m, n = len(s), len(d)
    x = np.zeros((m, n))
    rows = list(range(m))
    cols = list(range(n))

    def penalty(vals):
        v = sorted(v for v in vals if v < BIG)
        if len(v) >= 2:
            return v[1] - v[0]
        if len(v) == 1:
            return v[0]
        return -1

    while rows and cols:
        rp = {i: penalty([cost[i, j] for j in cols]) for i in rows}
        cp = {j: penalty([cost[i, j] for i in rows]) for j in cols}
        best_r = max(rp.values())
        best_c = max(cp.values())
        if best_r >= best_c:
            i = max(rows, key=lambda k: (rp[k], -k))
            j = min(cols, key=lambda k: cost[i, k])
        else:
            j = max(cols, key=lambda k: (cp[k], -k))
            i = min(rows, key=lambda k: cost[k, j])
        q = min(s[i], d[j])
        x[i, j] = q
        s[i] -= q
        d[j] -= q
        if s[i] == 0:
            rows.remove(i)
        if d[j] == 0:
            cols.remove(j)
    return x


def total_cost(x, cost):
    return float(np.sum(x * cost))


# =====================================================================
# 3. 最小费用最大流（MCMF）：SPFA 找最短增广路，逐次推流
#    把运输问题建成二分图：源 S -> 工厂 -> 仓库 -> 汇 T
# =====================================================================
class MCMF:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]  # 每条边 [to, cap, cost, rev_index]

    def add_edge(self, u, v, cap, cost):
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def _spfa(self, s, t):
        INF = float("inf")
        dist = [INF] * self.n
        inq = [False] * self.n
        pre_v = [-1] * self.n
        pre_e = [-1] * self.n
        dist[s] = 0
        inq[s] = True
        q = deque([s])
        while q:
            u = q.popleft()
            inq[u] = False
            for idx, e in enumerate(self.g[u]):
                v, cap, cost, _ = e
                if cap > 0 and dist[u] + cost < dist[v] - 1e-12:
                    dist[v] = dist[u] + cost
                    pre_v[v] = u
                    pre_e[v] = idx
                    if not inq[v]:
                        inq[v] = True
                        q.append(v)
        if dist[t] == INF:
            return None
        return dist[t], pre_v, pre_e

    def solve(self, s, t):
        flow = 0
        cost = 0.0
        while True:
            res = self._spfa(s, t)
            if res is None:
                break
            d, pre_v, pre_e = res
            # 找增广路上的最小残量
            f = float("inf")
            v = t
            while v != s:
                u, idx = pre_v[v], pre_e[v]
                f = min(f, self.g[u][idx][1])
                v = u
            v = t
            while v != s:
                u, idx = pre_v[v], pre_e[v]
                self.g[u][idx][1] -= f
                self.g[v][self.g[u][idx][3]][1] += f
                v = u
            flow += f
            cost += f * d
        return flow, cost


def solve_by_mcmf(supply, demand, cost, forbid=None):
    """forbid: set of (i,j) 禁运 """
    forbid = forbid or set()
    m, n = len(supply), len(demand)
    S, T = 0, m + n + 1
    node_plant = lambda i: 1 + i
    node_wh = lambda j: 1 + m + j
    net = MCMF(m + n + 2)
    for i in range(m):
        net.add_edge(S, node_plant(i), supply[i], 0)
    for j in range(n):
        net.add_edge(node_wh(j), T, demand[j], 0)
    for i in range(m):
        for j in range(n):
            c = BIG if (i, j) in forbid else cost[i, j]
            net.add_edge(node_plant(i), node_wh(j), min(supply[i], demand[j]), c)
    return net.solve(S, T)


# =====================================================================
# 4. 暴力穷举真值（小规模核对）：枚举满足行列和的整数解
# =====================================================================
def brute_force(supply, demand, cost):
    m, n = len(supply), len(demand)
    best = [float("inf"), None]

    def rec(i, j, s, d, x):
        if i == m:
            if all(abs(v) < 1e-9 for v in d):
                tot = sum(x[k][l] * cost[k][l] for k in range(m) for l in range(n))
                if tot < best[0] - 1e-9:
                    best[0] = tot
                    best[1] = [row[:] for row in x]
            return
        if j == n:
            if abs(s[i]) < 1e-9:
                rec(i + 1, 0, s, d, x)
            return
        up = min(s[i], d[j])
        for val in range(int(up) + 1):
            x[i][j] = val
            s[i] -= val
            d[j] -= val
            rec(i, j + 1, s, d, x)
            s[i] += val
            d[j] += val
            x[i][j] = 0
        return

    rec(0, 0, list(supply), list(demand), [[0] * n for _ in range(m)])
    return best[0], np.array(best[1])


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 68)
    print("Day23 · 运输问题与最小费用流")
    print("=" * 68)
    print("产量:", SUPPLY, " 需求:", DEMAND)
    print("运费矩阵:\n", COST)

    # --- 三种初始/最优解 ---
    x_nw = northwest_corner(SUPPLY, DEMAND)
    x_vam = vogel(SUPPLY, DEMAND, COST)
    flow, opt_cost = solve_by_mcmf(SUPPLY, DEMAND, COST)
    bf_cost, x_bf = brute_force(SUPPLY, DEMAND, COST)

    print("\n--- ① 西北角法（基本可行解）---")
    print(x_nw.astype(int), " 成本 =", total_cost(x_nw, COST))
    print("--- ② Vogel 近似法 ---")
    print(x_vam.astype(int), " 成本 =", total_cost(x_vam, COST))
    print("--- ③ 最小费用流（最优）---")
    print("最大流 =", flow, "（= 总需求 {}）".format(sum(DEMAND)))
    print("最优成本 =", opt_cost)
    print("--- ④ 暴力穷举真值 ---")
    print("穷举最优 =", bf_cost)
    print(x_bf.astype(int))

    assert abs(total_cost(x_nw, COST) - float(np.sum(x_nw * COST))) < 1e-6
    assert flow == sum(DEMAND), "最大流应等于总需求"
    assert abs(opt_cost - bf_cost) < 1e-6, "MCMF 必须等于穷举真值"
    assert total_cost(x_vam, COST) >= opt_cost - 1e-6, "VAM 不优于最优"
    # 行列和守恒
    assert np.allclose(x_nw.sum(axis=1), SUPPLY)
    assert np.allclose(x_nw.sum(axis=0), DEMAND)
    print("\n✅ 断言全过：NW 行列守恒、MCMF=穷举、VAM≥最优")

    # --- 变体 A：供不应求（总供给 50 < 总需求 60）---
    sup_a = [12, 22, 16]  # 合计 50
    flow_a, cost_a = solve_by_mcmf(sup_a, DEMAND, COST)
    print("\n--- 变体A 供不应求：总供给 50 < 总需求 60 ---")
    print("最大流 =", flow_a, "（缺口 = {}）".format(sum(DEMAND) - flow_a))

    # --- 变体 B：禁运 A2->B3（当前最优里该弧载重）---
    forbid = {(1, 2)}  # A2 -> B3
    flow_b, cost_b = solve_by_mcmf(SUPPLY, DEMAND, COST, forbid)
    print("\n--- 变体B 禁运 A2->B3 ---")
    print("最大流 =", flow_b, " 最优成本 =", cost_b,
          " （较原最优 +{}）".format(round(cost_b - opt_cost, 2)))

    # --- 敏感性：全部运费 ±15% 扰动，看最优成本分布 ---
    rng = np.random.default_rng(42)
    trials = 2000
    costs = []
    for _ in range(trials):
        perturbed = COST * (1 + rng.uniform(-0.15, 0.15, COST.shape))
        _, c = solve_by_mcmf(SUPPLY, DEMAND, perturbed)
        costs.append(c)
    costs = np.array(costs)
    print("\n--- 敏感性 ±15% × {} 次 ---".format(trials))
    print("均值 {:.2f} / 标准差 {:.2f} / 区间 [{:.2f}, {:.2f}]".format(
        costs.mean(), costs.std(), costs.min(), costs.max()))

    # =================================================================
    # 画图
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 左上：运费矩阵热力图
    ax = axes[0, 0]
    im = ax.imshow(COST, cmap="YlOrRd")
    ax.set_xticks(range(len(DEMAND)))
    ax.set_xticklabels(["B{}".format(j + 1) for j in range(len(DEMAND))])
    ax.set_yticks(range(len(SUPPLY)))
    ax.set_yticklabels(["A{}".format(i + 1) for i in range(len(SUPPLY))])
    for i in range(len(SUPPLY)):
        for j in range(len(DEMAND)):
            ax.text(j, i, int(COST[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("单位运费矩阵（元/件）")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 右上：三法成本对照
    ax = axes[0, 1]
    labels = ["西北角法", "Vogel法", "最小费用流", "暴力穷举"]
    vals = [total_cost(x_nw, COST), total_cost(x_vam, COST), opt_cost, bf_cost]
    colors = ["#c44e52", "#dd8452", "#4c72b0", "#55a868"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, "{:.0f}".format(v),
                ha="center", fontsize=10)
    ax.set_ylabel("总运费（元）")
    ax.set_title("三种初始解/最优解对照")
    ax.axhline(opt_cost, ls="--", color="gray", lw=1)

    # 左下：变体 B 的流向桑基式示意（工厂->仓库）
    ax = axes[1, 0]
    _, _ = solve_by_mcmf(SUPPLY, DEMAND, COST, forbid)
    # 用最优解画流向
    x_opt = x_bf
    ys_p = np.linspace(0.2, 0.8, len(SUPPLY))
    ys_w = np.linspace(0.2, 0.8, len(DEMAND))
    for i in range(len(SUPPLY)):
        for j in range(len(DEMAND)):
            if x_opt[i, j] > 0:
                ax.annotate("", xy=(0.85, ys_w[j]), xytext=(0.15, ys_p[i]),
                            arrowprops=dict(arrowstyle="->", color="#4c72b0",
                                            lw=1 + x_opt[i, j] / 8, alpha=0.75))
                ax.text(0.5, (ys_p[i] + ys_w[j]) / 2, "{:.0f}".format(x_opt[i, j]),
                        fontsize=9, color="#333")
    for i, y in enumerate(ys_p):
        ax.text(0.1, y, "A{} 产{}".format(i + 1, SUPPLY[i]), ha="right",
                va="center", fontsize=10)
    for j, y in enumerate(ys_w):
        ax.text(0.9, y, "B{} 需{}".format(j + 1, DEMAND[j]), ha="left",
                va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("最优运输方案流向（线宽∝运量）")

    # 右下：敏感性分布
    ax = axes[1, 1]
    ax.hist(costs, bins=40, color="#4c72b0", alpha=0.8)
    ax.axvline(opt_cost, color="red", ls="--", lw=2, label="基准最优 {:.0f}".format(opt_cost))
    ax.set_xlabel("总运费（元）")
    ax.set_ylabel("频次")
    ax.set_title("运费 ±15% 扰动 × {} 次".format(trials))
    ax.legend()

    plt.tight_layout()
    out = "transportation_min_cost_flow.png"
    plt.savefig(out, dpi=130)
    print("\n图已保存：", out)


if __name__ == "__main__":
    main()
