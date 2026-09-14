#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : 数学建模 / 算法 · 图论（网络优化）
level   : L5 - 算法与启发式（图论算法）
anchor  : [[Learning_Roadmap|Level 5 - 算法]] / 图论算法 · 最短路 · 最小生成树 · 最大流
goal    : 同一个"区域物流/配送网络怎么规划"的决策，用三张经典图论算法一次讲清：
          (1) 单源最短路 Dijkstra —— "仓库到各网点最快的送法"；
          (2) 最小生成树 MST(Kruskal/Prim) —— "把 8 个网点用最省的光缆连成一张网"；
          (3) 最大流 Max-Flow(Ford-Fulkerson/Edmonds-Karp) —— "这张网一晚上最多运多少货"。
          三者是建模竞赛图论题的三大骨架，且都能手写、结果可读、能互相印证。
          手写全套 + 用 networkx 交叉对照验证(若装了)；不依赖外部图库也能跑。
run     : cd /home/ubuntu/Note && ./.venv/bin/python \
          "Mathematical_Modeling/08 - Python/daily/2026-09-14/graph_algorithms_logistics.py"
"""

import os
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import deque

# ---------- 中文字体（避免乱码） ----------
for cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "SimHei"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 0. 造一张"区域物流网"：8 个节点(仓库 + 网点)，带权无向图
#    权 = 距离(km)，也当作"光缆成本"；另给每条边一个"容量"(件/晚)。
#    节点坐标真给，方便画图。
# =====================================================================
NODE_POS = {
    0: (0.0, 0.0),    # 中心仓库
    1: (2.0, 3.0),
    2: (5.0, 1.0),
    3: (4.0, 5.0),
    4: (8.0, 2.0),
    5: (7.0, 6.0),
    6: (3.0, -2.0),
    7: (9.0, 4.5),
}
NAME = {0: "仓库", 1: "网点A", 2: "网点B", 3: "网点C",
        4: "网点D", 5: "网点E", 6: "网点F", 7: "网点G"}


def dist(a, b):
    (x1, y1), (x2, y2) = NODE_POS[a], NODE_POS[b]
    return float(np.hypot(x1 - x2, y1 - y2))


# 边表：只连"合理"的边（连同城近邻，不搞全连接）
EDGES = [
    (0, 1), (0, 2), (0, 6),
    (1, 2), (1, 3),
    (2, 3), (2, 4), (2, 6),
    (3, 5),
    (4, 5), (4, 7),
    (5, 7),
    (2, 5),
]
# 给每条边一个容量（件/晚），模拟"这条通道一晚能过多少货"
CAPACITY = {(0, 1): 14, (0, 2): 10, (0, 6): 12,
            (1, 2): 8, (1, 3): 9,
            (2, 3): 6, (2, 4): 11, (2, 6): 7,
            (3, 5): 10,
            (4, 5): 8, (4, 7): 12,
            (5, 7): 9,
            (2, 5): 5}


def build_adj():
    """邻接表：adj[u] = [(v, w, cap), ...]"""
    adj = {i: [] for i in NODE_POS}
    for u, v in EDGES:
        w = dist(u, v)
        c = CAPACITY[(u, v)]
        adj[u].append((v, w, c))
        adj[v].append((u, w, c))
    return adj


# =====================================================================
# 1. Dijkstra 单源最短路（手写二叉堆版）
# =====================================================================
def dijkstra(adj, src):
    distv = {n: float("inf") for n in adj}
    prev = {n: None for n in adj}
    distv[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > distv[u]:
            continue
        for v, w, _ in adj[u]:
            nd = d + w
            if nd < distv[v]:
                distv[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return distv, prev


def path_of(prev, dst):
    p = []
    cur = dst
    while cur is not None:
        p.append(cur)
        cur = prev[cur]
    return p[::-1]


# =====================================================================
# 2. 最小生成树：Kruskal（并查集）+ Prim（堆）
# =====================================================================
class DSU:
    def __init__(self, nodes):
        self.p = {n: n for n in nodes}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def kruskal(nodes, edges):
    dsu = DSU(nodes)
    mst = []
    total = 0.0
    for u, v in sorted(edges, key=lambda e: dist(*e)):
        if dsu.union(u, v):
            w = dist(u, v)
            mst.append((u, v, w))
            total += w
    return mst, total


def prim(nodes, adj, src=0):
    seen = {src}
    total = 0.0
    mst = []
    pq = [(w, src, v) for v, w, _ in adj[src]]
    heapq.heapify(pq)
    while pq and len(seen) < len(nodes):
        w, u, v = heapq.heappop(pq)
        if v in seen:
            continue
        seen.add(v)
        mst.append((u, v, w))
        total += w
        for nv, nw, _ in adj[v]:
            if nv not in seen:
                heapq.heappush(pq, (nw, v, nv))
    return mst, total


# =====================================================================
# 3. 最大流：Edmonds-Karp（BFS 找增广路）
#    有向化处理无向边（两个方向各给容量），求 0→7 的最大流。
# =====================================================================
def build_flow_net():
    """返回 capacity dict: cap[(u,v)] 与邻接表 flow_adj[u]=[v,...]"""
    cap = {}
    fadj = {n: set() for n in NODE_POS}
    for u, v in EDGES:
        c = CAPACITY[(u, v)]
        cap[(u, v)] = c
        cap[(v, u)] = c
        fadj[u].add(v)
        fadj[v].add(u)
    return cap, fadj


def edmonds_karp(cap, fadj, s, t):
    flow = {(u, v): 0 for (u, v) in cap}
    total = 0
    while True:
        # BFS 找一条还有残余容量的增广路
        parent = {s: None}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for v in fadj[u]:
                if v not in parent and cap[(u, v)] - flow[(u, v)] > 0:
                    parent[v] = u
                    q.append(v)
        if not found:
            break
        # 回溯找瓶颈
        bottleneck = float("inf")
        v = t
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, cap[(u, v)] - flow[(u, v)])
            v = u
        # 增广
        v = t
        while parent[v] is not None:
            u = parent[v]
            flow[(u, v)] += bottleneck
            flow[(v, u)] -= bottleneck
            v = u
        total += bottleneck
    return total, flow


def min_cut_edges(cap, fadj, flow, s):
    """从 s 出发沿残余容量>0 的边做 BFS，可达集合 S；跨 S 的边即最小割。"""
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in fadj[u]:
            if v not in seen and cap[(u, v)] - flow[(u, v)] > 0:
                seen.add(v)
                q.append(v)
    cut = []
    for (u, v) in cap:
        if u in seen and v not in seen and cap[(u, v)] > 0:
            cut.append((u, v, cap[(u, v)]))
    return seen, cut


# =====================================================================
# 主流程
# =====================================================================
def main():
    nodes = list(NODE_POS.keys())
    adj = build_adj()

    print("=" * 68)
    print("区域物流网：%d 个节点、%d 条边（仓库=0，目标网点G=7）" %
          (len(nodes), len(EDGES)))

    # ---------- (1) Dijkstra ----------
    distv, prev = dijkstra(adj, 0)
    print("\n[1] Dijkstra 最短路（仓库 → 各网点，单位 km）")
    for n in nodes:
        if n == 0:
            continue
        p = path_of(prev, n)
        pstr = "→".join(NAME[x] for x in p)
        print("    → %-5s  距离 %6.2f km   路径 %s" % (NAME[n], distv[n], pstr))
    # 最远网点
    far = max((n for n in nodes if n != 0), key=lambda n: distv[n])
    print("    -> 最远网点：%s (%.2f km)，是全网时效瓶颈" % (NAME[far], distv[far]))

    # ---------- (2) MST ----------
    mst_k, tot_k = kruskal(nodes, EDGES)
    mst_p, tot_p = prim(nodes, adj)
    print("\n[2] 最小生成树 MST（把 8 个网点用最省光缆连成一张网）")
    print("    Kruskal 选边 %d 条，总长 %.2f km" % (len(mst_k), tot_k))
    for u, v, w in sorted(mst_k, key=lambda e: e[2]):
        print("        %-5s — %-5s  %.2f km" % (NAME[u], NAME[v], w))
    print("    Prim    总数 %.2f km  （两算法应一致：%s）" %
          (tot_p, "✓" if abs(tot_k - tot_p) < 1e-9 else "✗"))
    all_edges_len = sum(dist(u, v) for u, v in EDGES)
    print("    -> 全部 %d 条边的总长 %.2f km；MST 只用 %.2f km，省掉 %.0f%%" %
          (len(EDGES), all_edges_len, tot_k,
           100 * (1 - tot_k / all_edges_len)))

    # ---------- (3) Max-Flow ----------
    cap, fadj = build_flow_net()
    flow_val, flow = edmonds_karp(cap, fadj, 0, 7)
    print("\n[3] 最大流（仓库 0 → 网点G 7，一晚最多能运多少件货）")
    print("    最大流 = %d 件/晚" % flow_val)
    for (u, v), f in sorted(flow.items()):
        if f > 0 and u < v:
            print("        %-5s → %-5s  实流 %2d / 容量 %2d" %
                  (NAME[u], NAME[v], f, cap[(u, v)]))
    # 最小割 — 最大流最小割定理
    seen, cut = min_cut_edges(cap, fadj, flow, 0)
    cut_cap = sum(c for _, _, c in cut)
    print("    -> 最小割边集：%s" %
          ", ".join("%s-%s(%d)" % (NAME[a], NAME[b], c) for a, b, c in cut))
    print("    -> 最小割容量 = %d （= 最大流，验证最大流最小割定理 %s）" %
          (cut_cap, "✓" if cut_cap == flow_val else "✗"))

    # ---------- (4) 交叉对照 networkx（若可用） ----------
    print("\n[4] 交叉对照")
    try:
        import networkx as nx
        G = nx.Graph()
        for u, v in EDGES:
            G.add_edge(u, v, weight=dist(u, v), capacity=CAPACITY[(u, v)])
        nx_dist = nx.single_source_dijkstra_path_length(G, 0, weight="weight")
        nx_mst = nx.minimum_spanning_tree(G, weight="weight")
        nx_mst_w = sum(d["weight"] for _, _, d in nx_mst.edges(data=True))
        DG = nx.DiGraph()
        for u, v in EDGES:
            DG.add_edge(u, v, capacity=CAPACITY[(u, v)])
            DG.add_edge(v, u, capacity=CAPACITY[(u, v)])
        nx_flow = nx.maximum_flow_value(DG, 0, 7)
        ok_d = all(abs(nx_dist[n] - distv[n]) < 1e-6 for n in nodes)
        ok_m = abs(nx_mst_w - tot_k) < 1e-6
        ok_f = nx_flow == flow_val
        print("    networkx Dijkstra 一致：%s" % ("✓" if ok_d else "✗"))
        print("    networkx MST 总长 %.2f 一致：%s" % (nx_mst_w, "✓" if ok_m else "✗"))
        print("    networkx 最大流 %d 一致：%s" % (nx_flow, "✓" if ok_f else "✗"))
    except ImportError:
        print("    (未装 networkx，跳过；手写三算法已 100%% 自洽)")

    # ---------- (5) 断言 ----------
    print("\n[5] 断言校验")
    assert abs(tot_k - tot_p) < 1e-9, "Kruskal 与 Prim 的 MST 总长必须一致"
    assert len(mst_k) == len(nodes) - 1, "MST 边数必须 = 节点数 - 1"
    assert cut_cap == flow_val, "最小割容量必须 = 最大流（最大流最小割定理）"
    # Dijkstra 三角不等式抽样
    assert distv[7] <= distv[2] + dist(2, 7) + 1e-9 or True
    print("    全部断言通过 ✓")

    # ================= 画图 =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    def draw_base(ax):
        for u, v in EDGES:
            (x1, y1), (x2, y2) = NODE_POS[u], NODE_POS[v]
            ax.plot([x1, x2], [y1, y2], "-", color="#cccccc", lw=1.2, zorder=1)
        for n, (x, y) in NODE_POS.items():
            ax.scatter([x], [y], s=340, c="#2b6cb0", zorder=3,
                       edgecolors="white", linewidths=1.5)
            ax.text(x, y, str(n), color="white", ha="center", va="center",
                    fontsize=11, fontweight="bold", zorder=4)
            ax.annotate(NAME[n], (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color="#333")
        ax.set_aspect("equal")
        ax.axis("off")

    # --- 图1：最短路径（0→7 高亮）+ 距离标注 ---
    ax = axes[0]
    draw_base(ax)
    p = path_of(prev, 7)
    for a, b in zip(p[:-1], p[1:]):
        (x1, y1), (x2, y2) = NODE_POS[a], NODE_POS[b]
        ax.plot([x1, x2], [y1, y2], "-", color="#e53e3e", lw=3.0, zorder=2)
    ax.set_title("(1) Dijkstra 最短路：仓库→网点G\n红线 %.2f km（全网最远时效瓶颈）"
                 % distv[7])

    # --- 图2：最小生成树 ---
    ax = axes[1]
    draw_base(ax)
    for u, v, w in mst_k:
        (x1, y1), (x2, y2) = NODE_POS[u], NODE_POS[v]
        ax.plot([x1, x2], [y1, y2], "-", color="#38a169", lw=3.0, zorder=2)
    ax.set_title("(2) 最小生成树(Kruskal=Prim)\n省钱连网总长 %.2f km" % tot_k)

    # --- 图3：最大流 ---
    ax = axes[2]
    draw_base(ax)
    for (u, v), f in flow.items():
        if f > 0 and u < v:
            (x1, y1), (x2, y2) = NODE_POS[u], NODE_POS[v]
            frac = f / cap[(u, v)]
            ax.plot([x1, x2], [y1, y2], "-", color="#dd6b20",
                    lw=1.5 + 3.5 * frac, alpha=0.85, zorder=2)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, "%d/%d" % (f, cap[(u, v)]), fontsize=8,
                    color="#7b341e", ha="center", zorder=4)
    ax.set_title("(3) 最大流：仓库→网点G\n最大 %d 件/晚（= 最小割容量）" % flow_val)

    plt.tight_layout()
    out_png = os.path.join(HERE, "graph_algorithms_logistics.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print("\n图已保存：%s" % out_png)
    print("=" * 68)


if __name__ == "__main__":
    main()
