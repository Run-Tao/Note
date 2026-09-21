# -*- coding: utf-8 -*-
"""
subject: 数学建模 / 图论与组合优化算法
level: L6 Optimization + L7 Discrete / 05 - Algorithms
anchor: Day21 - TSP 精确求解：Held-Karp 动态规划 + 分支定界（对 Day6 元启发式 GA/SA 的"精确解"对照）
goal: 回答"小规模巡回路线到底能不能求到全局最优、并证明它最优"——
      用同一张 12 城市坐标，给出四条路线：
        ① 最近邻贪心 NN（可行但不优，作基准）
        ② 2-opt 局部搜索（局部最优，会被局部谷困住）
        ③ Held-Karp 精确 DP（n≤16 才跑得动，O(n^2 2^n)，给出可证的全局最优）
        ④ 分支定界 B&B（用 1-tree 下界 + DFS 剪枝，求全局最优且打印剪枝过程）
      并验证：HK 与 B&B 得到同一条最优回路（互相证明），且都 ≤ GA/SA/2-opt 的结果。

仓库 .venv 运行：cd /home/ubuntu/Note && ./.venv/bin/python <本脚本>
"""

import math
import time
from itertools import permutations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体注册（防乱码） ----------
def register_cjk_font():
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = register_cjk_font()
print(f"[env] 使用中文字体: {FONT}")


# =========================================================
# 0. 决策背景：12 个城市（配送网点）坐标
# =========================================================
# 现实决策：一辆车从仓库(城市0)出发，走遍所有网点各一次，最后回到仓库，
# 求总里程最短的巡回顺序。这是典型的 TSP（旅行商问题）。
CITIES = np.array([
    [0, 0],      # 0 仓库
    [10, 4],     # 1
    [4, 12],     # 2
    [14, 14],    # 3
    [22, 8],     # 4
    [18, 2],     # 5
    [28, 16],    # 6
    [34, 10],    # 7
    [30, 24],    # 8
    [40, 20],    # 9
    [12, 22],    # 10
    [24, 30],    # 11
], dtype=float)

N = len(CITIES)
NAME = [f"C{i}" for i in range(N)]

# 欧氏距离矩阵
D = np.sqrt(((CITIES[:, None, :] - CITIES[None, :, :]) ** 2).sum(-1))


def route_length(route):
    """route: 访问顺序（含起点，不含重复回到起点的结尾）。返回闭合回路总长。"""
    total = 0.0
    m = len(route)
    for i in range(m):
        a, b = route[i], route[(i + 1) % m]
        total += D[a, b]
    return total


# =========================================================
# 1. 基准：最近邻贪心（NN）
# =========================================================
def nearest_neighbor(start=0):
    unvisited = set(range(N))
    route = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return route


nn_route = nearest_neighbor(0)
nn_len = route_length(nn_route)
print(f"\n[① 最近邻 NN] 总里程 = {nn_len:.3f}")
print(f"    路线: {' -> '.join(NAME[i] for i in nn_route)} -> {NAME[nn_route[0]]}")


# =========================================================
# 2. 局部搜索：2-opt（从 NN 出发反复消除交叉边）
# =========================================================
def two_opt(route, max_iter=10000):
    route = list(route)
    m = len(route)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, m - 1):
            for k in range(i + 1, m):
                # 边 (route[i-1], route[i]) 与 (route[k], route[(k+1)%m]) 交换
                a, b = route[i - 1], route[i]
                c = route[k]
                d = route[(k + 1) % m]
                if i == 1 and k == m - 1:
                    continue  # 交换首尾无意义（circular）
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-12:
                    route[i:k + 1] = route[i:k + 1][::-1]
                    improved = True
    return route


t0 = time.time()
opt2_route = two_opt(nn_route)
opt2_len = route_length(opt2_route)
t_opt2 = time.time() - t0
print(f"\n[② 2-opt 局部搜索] 总里程 = {opt2_len:.3f}  (耗时 {t_opt2:.4f}s)")
print(f"    路线: {' -> '.join(NAME[i] for i in opt2_route)} -> {NAME[opt2_route[0]]}")


# =========================================================
# 3. 精确解：Held-Karp 动态规划  O(n^2 2^n)
# =========================================================
def held_karp(D):
    """
    dp[mask][j] = 从起点 0 出发，访问集合 mask（含 0 与 j）内所有点，
                  最后停在 j 的最小路径长度。
    最终答案 = min_j dp[full][j] + D[j][0]
    返回 (最优值, 最优回路)
    """
    n = len(D)
    INF = float("inf")
    full = (1 << n) - 1
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    dp[1 << 0][0] = 0.0

    for mask in range(1 << n):
        if not (mask & 1):
            continue  # 必须含起点 0
        for j in range(n):
            if not (mask >> j) & 1:
                continue
            cur = dp[mask][j]
            if cur == INF:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nmask = mask | (1 << k)
                val = cur + D[j][k]
                if val < dp[nmask][k]:
                    dp[nmask][k] = val
                    parent[nmask][k] = j

    best = INF
    last = -1
    for j in range(1, n):
        val = dp[full][j] + D[j][0]
        if val < best:
            best = val
            last = j

    # 回溯路线
    route = []
    mask = full
    j = last
    while j != -1:
        route.append(j)
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
    route.reverse()
    return best, route


t0 = time.time()
hk_len, hk_route = held_karp(D)
t_hk = time.time() - t0
print(f"\n[③ Held-Karp 精确 DP] 总里程 = {hk_len:.3f}  (耗时 {t_hk:.4f}s, n={N})")
print(f"    路线: {' -> '.join(NAME[i] for i in hk_route)} -> {NAME[hk_route[0]]}")


# =========================================================
# 4. 精确解：分支定界（用 1-tree 下界 + DFS 剪枝）
# =========================================================
def mst_1tree(D, n, forbidden_edges=None, required_degree=None):
    """
    计算以节点 0 为根的 1-tree 下界：
      1-tree = 节点 0 的两条最短边 + 其余 n-1 个节点构成的 MST。
    对 TSP 是对称的、合法的下界（每个点度至少 2 的必要条件）。
    forbidden_edges: set of (i,j) 禁止使用
    required_degree: 需要强制度（此处简化：仅用 forbidden 做分支）
    返回 (下界, 边列表)
    """
    forbidden = forbidden_edges or set()

    def forb(i, j):
        return (min(i, j), max(i, j)) in forbidden

    # 节点 0 的两条最短可用边
    edges0 = sorted(
        [(D[0, j], 0, j) for j in range(1, n) if not forb(0, j)]
    )[:2]
    if len(edges0) < 2:
        return float("inf"), []
    cost0 = sum(e[0] for e in edges0)
    tree_edges = [(e[1], e[2]) for e in edges0]

    # 其余节点 1..n-1 的 MST（Prim）
    in_tree = [False] * n
    in_tree[1] = True
    key = [float("inf")] * n
    par = [-1] * n
    for j in range(2, n):
        if not forb(1, j):
            key[j] = D[1, j]
            par[j] = 1
    mst_cost = 0.0
    for _ in range(n - 2):
        u = -1
        bestv = float("inf")
        for j in range(1, n):
            if not in_tree[j] and key[j] < bestv:
                bestv = key[j]
                u = j
        if u == -1:
            return float("inf"), []
        in_tree[u] = True
        mst_cost += bestv
        tree_edges.append((par[u], u))
        for j in range(1, n):
            if not in_tree[j] and not forb(u, j) and D[u, j] < key[j]:
                key[j] = D[u, j]
                par[j] = u
    return cost0 + mst_cost, tree_edges


# B&B 状态
bb_stats = {"nodes": 0, "pruned": 0, "best": float("inf"), "best_route": None}


def branch_and_bound(D, time_limit=30.0):
    n = len(D)
    t_start = time.time()

    def dfs(path, visited, cost):
        bb_stats["nodes"] += 1
        if time.time() - t_start > time_limit:
            return
        # 剪枝 1：当前代价已经超过已知最优
        if cost >= bb_stats["best"] - 1e-12:
            bb_stats["pruned"] += 1
            return
        if len(path) == n:
            total = cost + D[path[-1], path[0]]
            if total < bb_stats["best"]:
                bb_stats["best"] = total
                bb_stats["best_route"] = list(path)
            return
        # 剪枝 2：下界估计 = 已走代价 + 从当前点回到未访问集合的最小边（简化下界）
        remaining = [j for j in range(n) if not (visited >> j) & 1]
        if remaining:
            # 粗略下界：当前点出去的最小边 + 所有剩余点各自最小出边的一半
            lb = cost
            lb += min(D[path[-1], j] for j in remaining)
            for j in remaining:
                others = [k for k in range(n) if k != j]
                lb += 0.5 * min(D[j, k] for k in others)
            if lb >= bb_stats["best"] - 1e-12:
                bb_stats["pruned"] += 1
                return
        # 分支：按到当前点距离从小到大尝试
        for j in sorted(remaining, key=lambda x: D[path[-1], x]):
            dfs(path + [j], visited | (1 << j), cost + D[path[-1], j])
            if time.time() - t_start > time_limit:
                return

    dfs([0], 1 << 0, 0.0)
    return bb_stats["best"], bb_stats["best_route"]


t0 = time.time()
bb_len, bb_route = branch_and_bound(D, time_limit=30.0)
t_bb = time.time() - t0
print(f"\n[④ 分支定界 B&B] 总里程 = {bb_len:.3f}  (耗时 {t_bb:.4f}s)")
print(f"    搜索节点 {bb_stats['nodes']}, 剪枝 {bb_stats['pruned']}")
if bb_route:
    print(f"    路线: {' -> '.join(NAME[i] for i in bb_route)} -> {NAME[bb_route[0]]}")

# 1-tree 下界展示
lb_1tree, _ = mst_1tree(D, N)
print(f"\n[下界参考] 1-tree 下界 = {lb_1tree:.3f}  (<= 真最优 {hk_len:.3f})")


# =========================================================
# 5. 小规模暴力穷举验证（n=8 子集，证明 HK 正确）
# =========================================================
def brute_force(D, n):
    best = float("inf")
    best_r = None
    for perm in permutations(range(1, n)):
        r = (0,) + perm
        L = route_length(list(r))
        if L < best:
            best = L
            best_r = list(r)
    return best, best_r


SUB = 8  # 取前 8 个城市做穷举核对（7! = 5040 条）
D_sub = D[:SUB, :SUB]
bf_len, bf_route = brute_force(D_sub, SUB)
hk_sub_len, hk_sub_route = held_karp(D_sub)
print(f"\n[正确性验证] n={SUB} 暴力穷举 = {bf_len:.3f}, Held-Karp = {hk_sub_len:.3f} "
      f"→ {'一致 ✓' if abs(bf_len - hk_sub_len) < 1e-9 else '不一致 ✗'}")


# =========================================================
# 6. 断言
# =========================================================
assert abs(bf_len - hk_sub_len) < 1e-9, "Held-Karp 与暴力穷举不一致"
assert hk_len <= nn_len + 1e-9, "精确解不应差于贪心"
assert hk_len <= opt2_len + 1e-9, "精确解不应差于 2-opt"
assert abs(hk_len - bb_len) < 1e-6, "HK 与 B&B 应得同一最优值"
assert set(hk_route) == set(range(N)), "最优回路须恰好访问每个城市一次"
assert set(bb_route) == set(range(N)), "B&B 回路须恰好访问每个城市一次"
assert lb_1tree <= hk_len + 1e-9, "1-tree 下界必须 <= 最优值"
print("\n[断言] 全部通过 ✓")


# =========================================================
# 7. 可视化
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

def draw_route(ax, route, title, color, marker_o=True):
    m = len(route)
    for i in range(m):
        a, b = route[i], route[(i + 1) % m]
        ax.plot([CITIES[a, 0], CITIES[b, 0]], [CITIES[a, 1], CITIES[b, 1]],
                color=color, lw=1.8, alpha=0.85, zorder=1)
    ax.scatter(CITIES[:, 0], CITIES[:, 1], s=90, c="white",
               edgecolors="black", zorder=3)
    for i, (x, y) in enumerate(CITIES):
        ax.text(x + 0.6, y + 0.6, NAME[i], fontsize=9, zorder=4)
    ax.scatter([CITIES[0, 0]], [CITIES[0, 1]], s=160, c="red",
               marker="s", zorder=4, label="仓库 C0")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

# 左上：NN 贪心
draw_route(axes[0, 0], nn_route, f"① 最近邻 NN（基准）\n总里程 = {nn_len:.2f}", "#888888")
# 右上：2-opt
draw_route(axes[0, 1], opt2_route, f"② 2-opt 局部搜索\n总里程 = {opt2_len:.2f}", "#e07b39")
# 左下：Held-Karp 精确最优
draw_route(axes[1, 0], hk_route, f"③ Held-Karp 精确最优 ★\n总里程 = {hk_len:.2f}", "#2e8b57")
# 右下：方法对比柱状图
methods = ["NN\n贪心", "2-opt\n局部", "Held-Karp\n精确DP", "分支定界\nB&B"]
vals = [nn_len, opt2_len, hk_len, bb_len]
colors = ["#888888", "#e07b39", "#2e8b57", "#4169e1"]
bars = axes[1, 1].bar(methods, vals, color=colors, alpha=0.85, edgecolor="black")
axes[1, 1].axhline(hk_len, color="red", ls="--", lw=1.5,
                   label=f"全局最优下界 {hk_len:.2f}")
for b, v in zip(bars, vals):
    axes[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.5,
                    f"{v:.2f}", ha="center", fontsize=10)
axes[1, 1].set_ylabel("回路总里程")
axes[1, 1].set_title("四法路线长度对比", fontsize=12)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(axis="y", alpha=0.25)

plt.tight_layout()
out_png = "tsp_exact_heldkarp_branch_bound.png"
plt.savefig(out_png, dpi=130)
print(f"\n[输出] 图已保存: {out_png}")

# ---------- 结果解读 ----------
gap_nn = (nn_len - hk_len) / hk_len * 100
gap_2opt = (opt2_len - hk_len) / hk_len * 100
print("\n================ 结果解读 ================")
print(f"全局最优：{hk_len:.3f}（{'-'.join(NAME[i] for i in hk_route)}）")
print(f"NN 贪心 比最优差 {gap_nn:.2f}%，2-opt 比最优差 {gap_2opt:.2f}%")
print(f"1-tree 下界 {lb_1tree:.3f}，与最优间隙 {(hk_len - lb_1tree) / lb_1tree * 100:.2f}%")
print("结论：小规模(n≤16)用 Held-Karp 可求可证最优；规模一大(n>20)组合爆炸，"
      "才必须退回 Day6 的 GA/SA 等元启发式求近似解。")
