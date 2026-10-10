# -*- coding: utf-8 -*-
"""
subject:   数学建模 / 中国邮递员问题（Chinese Postman Problem）
level:     L6 优化模型 / L7 图论、网络与动态模型（最小权完美匹配 / 欧拉回路 / 路径规划）
anchor:    Roadmap L6-L7 —— 承接 Day14（最短路/最小生成树/最大流）、Day22（匈牙利指派）、
           Day26（LP 对偶）、Day39/40（一般图匹配 / 最大权匹配 KM）；本讲把「匹配」与「欧拉回路」
           合体，回答「怎么走一条路把每条街都覆盖且总路程最短」这一经典路线规划决策。
goal:      (1) 用最小权完美匹配（奇度点配对）+ 最短路（Floyd 全源）求中国邮递员最优路线；
           (2) 手写欧拉回路（Hierholzer）把「重边后的多重图」还原成一条闭合巡游；
           (3) 用「全枚举奇度点配对」暴力裁判验证最优性；与「不走回头路（贪心）」「重复所有边」
               两个对照方案比出差距；给出真实路线规划的结果解读。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-10-10/chinese_postman_matching_euler.py"
"""

import itertools
import os
from collections import defaultdict, deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
# 0. 中文字体注册（Noto Sans CJK / WenQuanYi），避免图中中文乱码
# ============================================================
def setup_chinese_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    # 优先注册并选用简体中文字体名
    for p in candidates:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            props = font_manager.FontProperties(fname=p)
            # .ttc 可能含多语言，强制指向 Simplified Chinese 字形
            try:
                name = props.get_name()
            except Exception:
                name = "Noto Sans CJK SC"
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                              "WenQuanYi Zen Hei", name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


FONT_NAME = setup_chinese_font()

# ============================================================
# 1. 决策场景：社区网格道路巡检（无向带权图）
#    环卫车要从车场出发，走遍每条街道至少一次，最后回到车场，使总行驶里程最短。
#    街道权重 = 里程（公里）。这是标准「中国邮递员问题」。
# ============================================================
# 顶点：0=车场(中心), 1..8=八个路口/小区入口
# 边：真实街道，带里程
EDGES = [
    (0, 1, 3), (0, 2, 7), (0, 3, 4),
    (1, 2, 2), (1, 4, 8),
    (2, 3, 5), (2, 5, 4),
    (3, 6, 3),
    (4, 5, 2), (4, 7, 4),
    (5, 6, 3), (5, 7, 6),
    (6, 8, 2), (7, 8, 3),
    (1, 8, 9),
]
N_NODES = 9
START = 0


def build_adjacency(edges, n):
    """返回邻接表 {u: [(v, w), ...]}（无向）。"""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def degrees(edges, n):
    deg = [0] * n
    for u, v, _ in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def floyd_warshall(edges, n):
    """全源最短路：返回 dist 矩阵与 next 矩阵（用于还原路径）。"""
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:  # 平行边取最短
            dist[u][v] = w
            dist[v][u] = w
            nxt[u][v] = v
            nxt[v][u] = u
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]
    return dist, nxt


def reconstruct_path(nxt, i, j):
    if nxt[i][j] is None:
        return []
    path = [i]
    while i != j:
        i = nxt[i][j]
        path.append(i)
    return path


# ============================================================
# 2. 最小权完美匹配（本题要求「最小化」，故用取负变换的 KM，
#    或对小规模直接 DP 枚举。这里两者都实现：KM 作主算法，DP 作核验）
# ============================================================
def km_min_weight_perfect_matching(cost, forbid_self=True):
    """
    Kuhn-Munkres（KM）求【最小权完美匹配】（方阵、非负代价）。
    做法：用经典 e-maxx 最小化形式（行/列势 + 松弛）。
    ⚠️ 该标准实现假设「任意行都能配任意列」，若矩阵对角线为 0（自己配自己=0 代价），
       会退化成「人人配自己、总代价 0」的假解 → 必须把对角线抬成大数禁止自配对。
    返回 (总权重, 匹配数组 match[行] = 列)。
    """
    n = len(cost)
    BIG = max(max(r) for r in cost) * (n + 1) + 1 if n > 0 else 0
    cost = [[(BIG if (forbid_self and i == j) else cost[i][j]) for j in range(n)]
            for i in range(n)]
    # 标准最小化 KM（要求代价非负；cos 已满足）
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = 匹配到列 j 的行
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    match = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            match[p[j] - 1] = j - 1
    return match  # 由调用方按「原始代价矩阵」计算总权重（本函数改变了对角线）


def brute_min_weight_perfect_matching(cost):
    """暴力枚举完美匹配（n! 或双阶乘），作独立裁判，仅适合 n 不大。"""
    n = len(cost)
    best = float("inf")
    best_pairs = None
    idx = list(range(n))

    def rec(remaining, pairs, acc):
        nonlocal best, best_pairs
        if acc >= best:
            return
        if not remaining:
            if acc < best:
                best = acc
                best_pairs = list(pairs)
            return
        first = remaining[0]
        for k in range(1, len(remaining)):
            other = remaining[k]
            rest = remaining[1:k] + remaining[k + 1:]
            pairs.append((first, other))
            rec(rest, pairs, acc + cost[first][other])
            pairs.pop()

    rec(idx, [], 0)
    return best, best_pairs


# ============================================================
# 3. 中国邮递员问题主算法
# ============================================================
def chinese_postman(edges, n, start=0):
    """
    返回 dict：
      total_weight      —— 最优巡游总里程
      odd_nodes         —— 奇度顶点
      match_pairs       —— 奇度点配对（用最短路距离配对）
      duplicated_edges  —— 为配对而重复走的边（多重边列表）
      euler_tour        —— 欧拉回路（顶点序列，闭合）
      duplicated_weight —— 重复部分里程
    """
    dist, nxt = floyd_warshall(edges, n)
    deg = degrees(edges, n)
    odd = [i for i in range(n) if deg[i] % 2 == 1]

    base_weight = sum(w for _, _, w in edges)
    duplicated = []

    if not odd:
        # 已是欧拉图：所有边各走一次
        pass
    else:
        m = len(odd)
        # 代价矩阵 = 奇度点之间的最短路距离
        cost = [[dist[odd[i]][odd[j]] for j in range(m)] for i in range(m)]
        match = km_min_weight_perfect_matching(cost)
        km_val = sum(cost[i][match[i]] for i in range(m) if i < match[i])
        # 还原每对之间的最短路，把路径上的边加入 duplicated
        for i in range(m):
            j = match[i]
            if i < j:
                a, b = odd[i], odd[j]
                path = reconstruct_path(nxt, a, b)
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    w = dist[u][v]
                    duplicated.append((u, v, w))

    # 构造多重图（原边 + 重复边），求欧拉回路
    multi = list(edges) + duplicated
    tour = hierholzer(multi, n, start)

    dup_weight = sum(w for _, _, w in duplicated)
    return {
        "total_weight": base_weight + dup_weight,
        "base_weight": base_weight,
        "duplicated_weight": dup_weight,
        "odd_nodes": odd,
        "duplicated_edges": duplicated,
        "euler_tour": tour,
        "dist": dist,
    }


def hierholzer(edges, n, start):
    """Hierholzer 算法求欧拉回路（顶点序列，首尾相接）。多重图用边 id 去重。"""
    adj = defaultdict(list)
    for eid, (u, v, w) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used_edge = [False] * len(edges)
    # 选择有边的起点（若 start 无边则换）
    if not adj.get(start):
        start = next((u for u in adj if adj[u]), start)
    stack = [start]
    circuit = []
    while stack:
        u = stack[-1]
        # 从 u 找一条未使用的边
        edge = None
        while adj[u]:
            v, eid = adj[u][-1]
            if used_edge[eid]:
                adj[u].pop()
                continue
            edge = (v, eid)
            adj[u].pop()
            break
        if edge is None:
            # u 无可用边 → 回溯，记录到路径
            circuit.append(stack.pop())
        else:
            v, eid = edge
            used_edge[eid] = True
            if v != u:  # 非自环：从对方邻接表删掉这条边
                adj[v].remove((u, eid))
            stack.append(v)
    circuit.reverse()
    return circuit


# ============================================================
# 4. 对照方案
# ============================================================
def naive_double_all_edges(edges):
    """对照 A：把每条街都走两遍（原路往返），一定可行但通常远非最优。"""
    return 2 * sum(w for _, _, w in edges)


def greedy_nearest_walk(edges, n, start=0):
    """
    对照 B：贪心最近邻巡游——从起点走，每次在「未走过的边」里选距离最近的走，
    没有未走过的边时，沿最短路返回起点。总里程 = 走过的边 + 全程移动（含未覆盖时缩回）。
    这里统计「实际行驶里程」：每次移动按两点间直线边权累计（走没走过的边则覆盖它）。
    """
    dist, nxt = floyd_warshall(edges, n)
    edge_pairs = {frozenset((u, v)) for u, v, _ in edges}
    weight_of = {}
    for u, v, w in edges:
        key = frozenset((u, v))
        weight_of[key] = min(weight_of.get(key, float("inf")), w)
    covered = set()
    cur = start
    total = 0
    while len(covered) < len(edge_pairs):
        # 找当前点能直连、且未覆盖的边中最近的一条
        best = None
        for (u, v, w) in edges:
            key = frozenset((u, v))
            if key in covered:
                continue
            if u == cur or v == cur:
                nxt_node = v if u == cur else u
                if best is None or w < best[0]:
                    best = (w, nxt_node, key)
        if best is None:
            # 当前点无未覆盖直连边：走到最近的「有未覆盖边的点」（沿最短路）
            target = None
            bestd = float("inf")
            for (u, v, w) in edges:
                key = frozenset((u, v))
                if key in covered:
                    continue
                for node in (u, v):
                    if dist[cur][node] < bestd:
                        bestd = dist[cur][node]
                        target = node
            if target is None:
                break
            total += dist[cur][target]
            cur = target
        else:
            w, nxt_node, key = best
            total += w
            covered.add(key)
            cur = nxt_node
    # 返回起点
    total += dist[cur][start]
    return total


# ============================================================
# 5. 校验
# ============================================================
def check_tour_covers_all_edges(tour, edges):
    """核验巡游（含重复边）确实覆盖每条原边至少一次。"""
    from collections import Counter
    tour_edges = Counter()
    for k in range(len(tour) - 1):
        a, b = tour[k], tour[k + 1]
        tour_edges[frozenset((a, b))] += 1
    # 每条原边至少出现一次
    need = Counter()
    for u, v, _ in edges:
        need[frozenset((u, v))] += 1
    for e, c in need.items():
        if tour_edges[e] < c:
            return False, f"边 {tuple(e)} 覆盖不足 {tour_edges[e]}<{c}"
    return True, "OK"


def tour_closed_and_connected(tour):
    if tour[0] != tour[-1]:
        return False, "巡游非闭合"
    return True, "OK"


# ============================================================
# 6. 主流程
# ============================================================
def main():
    adj = build_adjacency(EDGES, N_NODES)
    deg = degrees(EDGES, N_NODES)
    base = sum(w for _, _, w in EDGES)

    print("=" * 72)
    print("社区网格道路巡检 · 中国邮递员问题（Chinese Postman Problem）")
    print("=" * 72)
    print(f"顶点数 N = {N_NODES}，街道边数 M = {len(EDGES)}，总里程(各走一次) = {base}")
    print(f"各顶点度数 = {deg}")
    odd = [i for i in range(N_NODES) if deg[i] % 2 == 1]
    print(f"奇度顶点 = {odd}（共 {len(odd)} 个）")
    print()

    # ---- 最优解 ----
    res = chinese_postman(EDGES, N_NODES, START)
    print("【中国邮递员最优解（最小权完美匹配 + 欧拉回路）】")
    print(f"  奇度点配对后需重复走的边: {res['duplicated_edges']}")
    print(f"  重复里程 = {res['duplicated_weight']}")
    print(f"  最优总里程 = {res['total_weight']}  ( = 各走一次 {base} + 重复 {res['duplicated_weight']} )")
    print(f"  欧拉回路顶点序列: {res['euler_tour']}")
    ok1, m1 = check_tour_covers_all_edges(res["euler_tour"], EDGES)
    ok2, m2 = tour_closed_and_connected(res["euler_tour"])
    print(f"  覆盖所有边? {ok1} ({m1})   闭合? {ok2} ({m2})")
    print()

    # ---- 独立裁判：暴力枚举奇度点配对 ----
    dist = res["dist"]
    m = len(odd)
    cost = [[dist[odd[i]][odd[j]] for j in range(m)] for i in range(m)]
    km_match = km_min_weight_perfect_matching(cost)
    km_val = sum(cost[i][km_match[i]] for i in range(m) if i < km_match[i])
    brute_val, brute_pairs = brute_min_weight_perfect_matching(cost)
    print("【独立裁判】")
    print(f"  KM 最小权完美匹配（奇度点配对距离）= {km_val}")
    print(f"  暴力枚举所有配对 = {brute_val}   配对方案 = {brute_pairs}")
    print(f"  KM 复算的巡游总里程 = {base + km_val}")
    print()

    # ---- 对照方案 ----
    naive = naive_double_all_edges(EDGES)
    greedy = greedy_nearest_walk(EDGES, N_NODES, START)
    print("【对照方案】")
    print(f"  A. 每条街往返两遍（笨办法上界） = {naive}  （比最优多 {naive - res['total_weight']}，"
          f"+{(naive - res['total_weight']) / res['total_weight'] * 100:.1f}%）")
    print(f"  B. 最近邻贪心巡游（每次都抢最近的未走街） = {greedy}  （比最优多 {greedy - res['total_weight']}）")
    print()

    # ---- 断言 ----
    assert abs(km_val - brute_val) < 1e-9, "KM 与暴力枚举不一致！"
    assert abs(res["total_weight"] - (base + brute_val)) < 1e-9, "巡游总里程与裁判不符"
    assert ok1 and ok2, "巡游不合法"
    assert res["total_weight"] <= greedy, "最优应不劣于贪心"
    assert res["total_weight"] <= naive, "最优应不劣于往返两遍"
    assert len(res["euler_tour"]) >= len(EDGES) + 1, "巡游边数异常"
    print("✅ 全部断言通过：KM==暴力裁判、巡游覆盖全部边且闭合、最优不劣于两个对照。")

    # ========================================================
    # 7. 绘图
    # ========================================================
    pos = {
        0: (0.0, 0.0), 1: (-1.4, 1.4), 2: (0.4, 1.9), 3: (1.7, 1.2),
        4: (-1.9, -0.6), 5: (-0.4, -1.1), 6: (1.3, -1.2), 7: (-1.7, -2.1), 8: (1.7, -2.4),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 左上：原图 + 度数 + 奇度点高亮
    ax = axes[0, 0]
    for u, v, w in EDGES:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#888", lw=1.5, zorder=1)
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ax.text(mx, my, str(w), fontsize=8, color="#555", zorder=3)
    for i in range(N_NODES):
        c = "#d62728" if deg[i] % 2 == 1 else "#1f77b4"
        ax.scatter(*pos[i], s=260, c=c, zorder=4, edgecolors="white")
        ax.text(pos[i][0], pos[i][1], str(i), color="white", ha="center", va="center",
                fontsize=9, zorder=5)
    ax.set_title("① 社区道路图（红=奇度路口 蓝=偶度，数字=里程）", fontsize=12)
    ax.axis("off")

    # 右上：奇度点配对示意（在最短路图上连 KM 匹配对）
    ax = axes[0, 1]
    for u, v, w in EDGES:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#ccc", lw=1.2, zorder=1)
    for i in range(N_NODES):
        c = "#d62728" if deg[i] % 2 == 1 else "#1f77b4"
        ax.scatter(*pos[i], s=200, c=c, zorder=4, edgecolors="white")
        ax.text(pos[i][0], pos[i][1], str(i), color="white", ha="center", va="center",
                fontsize=8, zorder=5)
    # KM 匹配对
    km_match = km_min_weight_perfect_matching(cost)
    for i in range(m):
        j = km_match[i]
        if i < j:
            a, b = odd[i], odd[j]
            path = reconstruct_path(res["dist"] and floyd_warshall(EDGES, N_NODES)[1], a, b)
            xs = [pos[p][0] for p in path]
            ys = [pos[p][1] for p in path]
            ax.plot(xs, ys, color="#9467bd", lw=3.0, alpha=0.8, zorder=2,
                    label=f"配对 {a}↔{b} (最短路 {cost[i][j]})" if i == 0 else None)
    ax.set_title("② 奇度路口两两配对（沿最短路重复走，紫色）", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.axis("off")

    # 左下：三种方案总里程对比
    ax = axes[1, 0]
    labels = ["中国邮递员\n(最优)", "最近邻\n贪心巡游", "每条街\n往返两遍"]
    vals = [res["total_weight"], greedy, naive]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("总行驶里程（公里）")
    ax.set_title("③ 三种巡游方案总里程对比", fontsize=12)

    # 右下：最优巡游路线（欧拉回路轨迹示意：按顶点访问顺序连线，颜色渐变）
    ax = axes[1, 1]
    for u, v, w in EDGES:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#eee", lw=6, zorder=1)
    tour = res["euler_tour"]
    cmap = plt.cm.viridis
    L = len(tour) - 1
    for k in range(L):
        a, b = tour[k], tour[k + 1]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=cmap(k / max(L - 1, 1)), lw=2.2, alpha=0.85, zorder=2)
    for i in range(N_NODES):
        ax.scatter(*pos[i], s=220, c="#333", zorder=4, edgecolors="white")
        ax.text(pos[i][0], pos[i][1], str(i), color="white", ha="center", va="center",
                fontsize=9, zorder=5)
    ax.set_title(f"④ 最优巡游路线（{L} 段，总里程 {res['total_weight']}）", fontsize=12)
    ax.axis("off")

    fig.suptitle("中国邮递员问题：怎么走遍每条街、总里程最短", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "chinese_postman_matching_euler.png")
    fig.savefig(out, dpi=130)
    print(f"图已保存: {out}")
    print(f"（中文字体: {FONT_NAME}）")


if __name__ == "__main__":
    main()
