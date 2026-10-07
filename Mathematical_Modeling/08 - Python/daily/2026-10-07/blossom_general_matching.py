#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L7 图论、网络与动态模型 / L8 数值计算与算法
anchor:  Day39 - 一般图最大匹配（Blossom 带花树 / Edmonds 算法）
goal:    在一张【非二分】的现实配对图上求最大匹配。
        二分图有匈牙利算法（Day22/Day25），但现实里"谁能和谁配对"的关系图
        一旦出现【奇环】（三人互相都认可），增广路推理就会失效——
        这时必须用 Edmonds 的【带花树（Blossom）】算法：把奇环缩成"花"，
        在半缩图上继续找增广路，再把花展开。

本脚本零专用图匹配库，全部手写：
  1) 邻接表建图（真实决策：医院护理小组两两搭班配对）
  2) 手写 Edmonds Blossom（找增广路 + 缩花 + 标记花身份 + 展开并翻转匹配）
  3) 暴力枚举求最大匹配作【独立裁判】（小规模，用位掩码 + 递归）
  4) 与"错误地套用二分图匈牙利算法"的正解对照——暴露奇环为什么坑
  5) 输出图（四个子图）

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-07/blossom_general_matching.py"
"""

import os
import itertools
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 中文字体
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
           "WenQuanYi Micro Hei", "Source Han Sans SC"]:
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 0. 真实决策问题：三甲医院护理小组「两两搭班」配对
# =====================================================================
# 12 名护士。两人能否搭班取决于「配合度」（历史协作记录里互相认可）。
# 目标：把护士两两配对成 6 组同班搭档，使「配上搭档的人数」最大
#      （= 最大匹配），且配出来的组尽量是"互相认可的强关系"。
#
# 关键：这张「可搭班关系图」【不是二分图】——因为存在三人互相认可的奇环：
#      N1-N2-N3-N1（三人都两两愿意搭班）。
#      二分图算法（匈牙利）会把点强行分成左右两组，对奇环必定出错。
#
# 护士群体（跨学历/科室，本来就无自然的二分划分）：
NURSES = [
    "A 张敏(ICU)", "B 李强(急诊)", "C 王芳(ICU)", "D 赵磊(手术)",
    "E 陈静(急诊)", "F 刘洋(手术)", "G 周霞(儿科)", "H 吴斌(内科)",
    "I 郑丽(儿科)", "J 孙浩(内科)", "K 冯娜(ICU)", "L 何军(急诊)",
]

# 可搭班边（无向，索引 0..11）。刻意织入多个奇环：
#   奇环1: A-B-C-A          （三人互认，全 IC∪急诊 骨干）
#   奇环2: D-E-F-G-D(4)     —— 4 是偶环；再加 H-I-J-H 奇环
#   奇环3: H-I-J-H
#   长边: A-D, C-E, F-H, G-K, I-K, J-L, D-K, E-G ...
EDGES = [
    (0, 1), (1, 2), (2, 0),        # 奇环1 A-B-C-A（三角形，最小奇环）
    (3, 4), (4, 5), (5, 6), (6, 3),# 偶环 D-E-F-G-D
    (7, 8), (8, 9), (9, 7),        # 奇环3 H-I-J-H
    (0, 3), (2, 4), (5, 7), (6, 10),
    (8, 10), (9, 11), (3, 10), (4, 6), (1, 10), (11, 2),
]


def build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


# =====================================================================
# 1. 手写 Edmonds Blossom（一般图最大匹配）
# =====================================================================
# 核心思想：从每个未匹配点出发，用 BFS 找一条「增广路」——两端都是未匹配点、
#          且路上匹配/未匹配边交替。找到就把匹配数 +1。
# 难点：在非二分图里，BFS 会遇到「奇环（花 blossom）」——环长奇数，
#       两条搜索路径在环上相遇，出现“同色点相连”。此时不能简单判定
#       “已冲突”，而要把这个奇环【收缩成一个超点（花）】，
#       在缩后的图上继续 BFS；找到增广路后再把花【展开】还原。
#
# 实现（经典 O(V^3) 版本，邻接表 + 花基 base 数组）：
#   match[v]  : 与 v 匹配的点，未匹配为 -1
#   p[v]      : BFS 树中 v 的父节点
#   base[v]   : v 所在花的“花根”（代表元）；未缩时 base[v]=v
#   used[v]   : v 是否已被 BFS 访问
#   q         : BFS 队列
def lca(a, b, match, p, base):
    """求 BFS 树上节点 a、b 的最近公共祖先（花根意义下）——花的花托。"""
    used = set()
    while True:
        a = base[a]
        used.add(a)
        if match[a] == -1:
            break
        a = p[match[a]]
    while True:
        b = base[b]
        if b in used:
            return b
        b = p[match[b]]


def mark_path(v, b, child, match, p, base, blossom):
    """把 v 到花托 b 的路径上的点全部标记为“花内”，并把 base 改成花托。"""
    while base[v] != b:
        blossom[base[v]] = True
        blossom[base[match[v]]] = True
        p[v] = child
        child = match[v]
        v = p[match[v]]


def find_path(root, adj, match, base, p, used):
    """从 root 出发找增广路；找到返回另一个未匹配端点 v（其 p[] 指向树上的父），
    否则返回 None。"""
    n = len(adj)
    used[:] = [False] * n
    p[:] = [-1] * n
    for i in range(n):
        base[i] = i
    used[root] = True
    q = [root]
    head = 0
    while head < len(q):
        v = q[head]
        head += 1
        for to in adj[v]:
            # 跳过：已在同一朵花内；以及“匹配边回头”
            if base[v] == base[to] or match[v] == to:
                continue
            # 情形一：to 已经长在 BFS 树上（或就是 root）→ 遇到奇环，缩花
            if to == root or (match[to] != -1 and p[match[to]] != -1):
                curbase = lca(v, to, match, p, base)
                blossom = [False] * n
                mark_path(v, curbase, to, match, p, base, blossom)
                mark_path(to, curbase, v, match, p, base, blossom)
                # 把花内所有点重新标 base 并压入队列
                for i in range(n):
                    if blossom[base[i]]:
                        base[i] = curbase
                        if not used[i]:
                            used[i] = True
                            q.append(i)
            elif p[to] == -1:
                # 情形二：to 未在 BFS 树中 → 把它挂到树上
                p[to] = v
                if match[to] == -1:
                    return to               # 另一个未匹配端点 → 增广路找到
                # 否则把 to 的匹配点继续入队（交替搜索）
                nxt = match[to]
                used[nxt] = True
                q.append(nxt)
    return None


def augment(v, match, p, base):
    """沿 BFS 树把 v 到根的路径做匹配翻转。注意：p[v] 是 v 在树上的父，
    翻转 v↔p[v]，再跳到 p[v] 原来的匹配伙伴继续。"""
    while v != -1:
        pv = p[v]
        nv = match[pv] if pv != -1 else -1
        match[v] = pv
        match[pv] = v
        v = nv


def blossom_max_matching(n, edges):
    """Edmonds 带花树：返回 (matching_size, match_array, n_augmentations)。"""
    adj = build_adj(n, edges)
    match = [-1] * n
    base = list(range(n))
    p = [-1] * n
    used = [False] * n
    aug = 0
    for i in range(n):
        if match[i] == -1:
            v = find_path(i, adj, match, base, p, used)
            if v is not None:
                aug += 1
                augment(v, match, p, base)
    size = sum(1 for m in match if m != -1) // 2
    return size, match, aug


# =====================================================================
# 2. 独立裁判：暴力枚举最大匹配（小规模，位掩码 + 记忆化递归）
# =====================================================================
def brute_max_matching(n, edges):
    adj = build_adj(n, edges)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(mask):
        # mask: 尚未处理的点集合；取编号最小的点 v
        if mask == 0:
            return 0
        v = (mask & -mask).bit_length() - 1
        # 方案一：v 不匹配
        best = solve(mask & ~(1 << v))
        # 方案二：v 与某个邻居 u 匹配（都从 mask 中移除）
        m = mask & ~(1 << v)
        for u in adj[v]:
            if m >> u & 1:
                best = max(best, 1 + solve(m & ~(1 << u)))
        return best

    return solve((1 << n) - 1)


# =====================================================================
# 3. 错误对照：把一般图硬当二分图跑匈牙利（看看会怎样）
# =====================================================================
def greedy_maximal_matching(n, edges):
    """贪心极大匹配（不是最大！）——用来展示“贪心 ≠ 最优”。"""
    adj = build_adj(n, edges)
    match = [-1] * n
    for u, v in edges:
        if match[u] == -1 and match[v] == -1:
            match[u] = v
            match[v] = u
    return sum(1 for m in match if m != -1) // 2, match


def naive_bipartite_hungarian(n, edges, left_split):
    """把图硬拆成左右两半、套二分图匈牙利（Kuhn 增广）。
    在非二分图上，left_split 之外的边会被忽略 → 极可能漏掉最优匹配。"""
    left = set(left_split)
    # 只保留“左→右”的边（把无向边硬当有向二分图）
    badj = [[] for _ in range(n)]
    for u, v in edges:
        if u in left and v not in left:
            badj[u].append(v)
        elif v in left and u not in left:
            badj[v].append(u)
    matchR = {}
    matchL = {}

    def try_kuhn(u, seen):
        for v in badj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in matchR or try_kuhn(matchR[v], seen):
                matchR[v] = u
                matchL[u] = v
                return True
        return False

    for u in sorted(left):
        try_kuhn(u, set())
    return len(matchR), matchL, matchR


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    n = len(NURSES)
    print("=" * 68)
    print("Day39 · 一般图最大匹配（Edmonds Blossom 带花树）—— 护理搭班配对")
    print("=" * 68)
    print(f"护士 {n} 名，可搭班边 {len(EDGES)} 条")

    # 4.1 带花树求最大匹配
    size, match, aug = blossom_max_matching(n, EDGES)
    pairs = sorted({tuple(sorted((i, match[i]))) for i in range(n) if match[i] != -1})
    print(f"\n[带花树] 最大匹配 = {size} 对（增广 {aug} 次）")
    print("  配出的搭班组：")
    for a, b in pairs:
        print(f"    {NURSES[a]}  ⇄  {NURSES[b]}")
    unmatched = [NURSES[i] for i in range(n) if match[i] == -1]
    print(f"  未配上：{'、'.join(unmatched) if unmatched else '（全部配上）'}")

    # 4.2 独立裁判：暴力枚举
    brute = brute_max_matching(n, EDGES)
    print(f"\n[暴力枚举裁判] 最大匹配 = {brute} 对")

    # 4.3 对照一：贪心极大匹配（会掉进局部最优）
    g_size, g_match = greedy_maximal_matching(n, EDGES)
    print(f"[贪心极大匹配] = {g_size} 对（比最优少 {brute - g_size} 对）")

    # 4.4 对照二：硬套二分图匈牙利
    #     用奇偶染色找二分划分——非二分图染色必然失败，这里退化地取一半点
    left_split = [i for i in range(n) if i % 2 == 0]
    b_size, _, _ = naive_bipartite_hungarian(n, EDGES, left_split)
    print(f"[硬套二分匈牙利] = {b_size} 对（比最优少 {brute - b_size} 对）")

    # 4.5 断言（多尺度核验）
    assert size == brute, f"带花树 {size} != 暴力真最优 {brute}"
    assert 2 * size <= n
    assert all((match[i] == -1) == (match[match[i]] == -1) for i in range(n) if match[i] != -1)
    # 匹配合法性：配对必须是一条真实存在的边，且无重边/无自匹配
    eset = {tuple(sorted(e)) for e in EDGES}
    for i in range(n):
        if match[i] != -1:
            assert match[i] != i
            assert tuple(sorted((i, match[i]))) in eset, "配了对不存在的边"
            assert match[match[i]] == i, "匹配不对称"
    assert size >= g_size, "最大匹配不可能小于贪心"
    print("\n[断言] 全部通过：带花树==暴力量、配对合法、对称、≥贪心")

    # 4.6 随机小规模压力测试：带花树 == 暴力（20 次）
    rng = random.Random(20261007)
    ok = 0
    for _ in range(20):
        k = rng.randint(6, 10)
        ps = 0.45
        ed = [(i, j) for i in range(k) for j in range(i + 1, k) if rng.random() < ps]
        s1, _, _ = blossom_max_matching(k, ed)
        s2 = brute_max_matching(k, ed)
        assert s1 == s2, f"随机图失配: k={k} blossom={s1} brute={s2}"
        ok += 1
    print(f"[随机压力测试] {ok}/20 张随机图 带花树==暴力，全部一致")

    # ------------------------------------------------------------------
    # 5. 画图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # --- 子图(1)：可搭班关系图 + 最大匹配（红粗边） + 奇环高亮 ---
    ax = axes[0, 0]
    # 手工布局：把奇环摆成明显的三角形
    pos = {
        0: (0.0, 2.0), 1: (1.0, 3.0), 2: (2.0, 2.0),          # 奇环1 A-B-C
        3: (4.0, 2.2), 4: (5.2, 3.2), 5: (6.4, 2.2), 6: (5.2, 1.2),  # 环 D-E-F-G
        7: (0.6, 0.3), 8: (1.8, -0.5), 9: (2.6, 0.7),          # 奇环3 H-I-J
        10: (4.8, -0.7), 11: (3.2, -0.9),
    }
    for u, v in EDGES:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, "-", color="#c8c8c8", lw=1.4, zorder=1)
    for a, b in pairs:
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, "-", color="#d62728", lw=3.6, zorder=3)
    for i in range(n):
        c = "#ffd166" if match[i] != -1 else "#eeeeee"
        ax.scatter(*pos[i], s=760, c=c, edgecolors="#333", zorder=4)
        ax.text(pos[i][0], pos[i][1], chr(ord("A") + i), ha="center",
                va="center", fontsize=13, zorder=5)
    # 高亮奇环1
    tri = [pos[0], pos[1], pos[2], pos[0]]
    ax.plot([p[0] for p in tri], [p[1] for p in tri], "--",
            color="#1f77b4", lw=2.2, zorder=2)
    ax.text(1.0, 2.35, "奇环（三角形 A-B-C）", color="#1f77b4",
            ha="center", fontsize=11)
    ax.set_title(f"可搭班关系图（非二分，含奇环）\n最大匹配 = {size} 对（红粗边）",
                 fontsize=13)
    ax.axis("off")

    # --- 子图(2)：方法对照柱状 ---
    ax = axes[0, 1]
    labels = ["暴力真最优", "带花树\n(Blossom)", "贪心极大", "硬套二分\n匈牙利"]
    vals = [brute, size, g_size, b_size]
    colors = ["#555555", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                ha="center", fontsize=12)
    ax.axhline(brute, ls="--", color="#555", lw=1.2)
    ax.set_ylabel("匹配对数")
    ax.set_title("四种方法对照：只有精确算法拿到真最优", fontsize=13)
    ax.set_ylim(0, max(vals) + 0.8)

    # --- 子图(3)：为什么奇环坑死二分图（示意）---
    ax = axes[1, 0]
    ax.set_title("为什么奇环必须用『花』：增广路推理在奇环上失效", fontsize=13)
    # 画一个三角形，演示左右染色矛盾
    t = np.array([[0, 1.6], [1.4, -0.2], [-1.4, -0.2], [0, 1.6]])
    ax.plot(t[:, 0], t[:, 1], "-", color="#1f77b4", lw=2.4)
    for k, (xx, yy) in enumerate([(0, 1.6), (1.4, -0.2), (-1.4, -0.2)]):
        col = "#d62728" if k != 1 else "#1f77b4"
        ax.scatter(xx, yy, s=900, c=col, edgecolors="#333", zorder=3)
        ax.text(xx, yy, ["左", "右", "?"][k], ha="center", va="center",
                color="white", fontsize=12, zorder=4)
    ax.text(0, -0.95,
            "三点两两相邻：无论怎么二分，\n必有两点同侧却相连 → 染色矛盾。\nBlossom 把奇环缩成一个超点（花），\n在缩图上继续找增广路，找到再展开。",
            ha="center", fontsize=11, color="#333")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.6, 2.4)
    ax.axis("off")

    # --- 子图(4)：匹配数随节点规模增长（带花树 vs 暴力，验证一致）---
    ax = axes[1, 1]
    rng2 = random.Random(7)
    ks = list(range(6, 15))
    b_curve, a_curve = [], []
    for k in ks:
        ed = [(i, j) for i in range(k) for j in range(i + 1, k) if rng2.random() < 0.4]
        s1, _, _ = blossom_max_matching(k, ed)
        s2 = brute_max_matching(k, ed)
        a_curve.append(s1)
        b_curve.append(s2)
    ax.plot(ks, b_curve, "o-", color="#555", label="暴力真最优", lw=2)
    ax.plot(ks, a_curve, "x--", color="#2ca02c", label="带花树", lw=2)
    ax.set_xlabel("节点数 n")
    ax.set_ylabel("最大匹配对数")
    ax.set_title("规模扫描：带花树始终与暴力真最优重合", fontsize=13)
    ax.legend()

    fig.suptitle("Day39 · 一般图最大匹配（Edmonds Blossom）—— 护理搭班配对决策",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT_DIR, "blossom_general_matching.png")
    fig.savefig(out, dpi=130)
    print(f"\n[图] 已保存 {out}")


if __name__ == "__main__":
    main()
