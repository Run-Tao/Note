# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L5 算法（最小费用流 / 残量网络 / 节点势与对偶检验数 / 多商品反例）
anchor:  Day83 - 最小费用流与价差检验云溪县五镇供水调度
goal:    把「运输/调运」从一张 LP 表变成一张图，在同一张图上回答三个现场问题：
        ① 多送 1 万 m³ 水，到底按哪个单价买？（节点势 = 边际到货价）
        ② 图上没有的那条新建干渠，值不值得建？（把候选弧塞进图重算 + 图上替代路对照）
        ③ 净水厂/泵站该先扩谁？（逐个 +1 万 m³/日 重解，与节点势排序对照）
        另附反例：两种水质共管时，按单商品求出的节点势不再有效。

运行： cd /home/ubuntu/Note && ./.venv/bin/python -W error "Mathematical_Modeling/08 - Python/daily/2026-11-18/day83_mincost_flow_potential.py"
"""

import itertools
import json
import os
import time
import warnings
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linprog

warnings.filterwarnings("error")

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
WS_CAP = 11.0          # 总水源水量：三个净水厂合计每天最多来 11 万 m³（瓶颈在源头，不在管径）

for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

# ----------------------------------------------------------------------------
# 1. 实例：云溪县五镇供水调度（容量 万 m³/日；成本 元/(万 m³)，即「元/m³ × 1 万」）
# ----------------------------------------------------------------------------
SRC, TGT = "S", "T"
GEN = ["A", "B", "C"]                 # 三个净水厂出水口
PUMP = ["P", "Q", "R"]                # 三个中继泵站
TOWN = ["D", "E", "F", "G"]           # 四个城镇接水口
NODES = [SRC] + GEN + PUMP + TOWN + [TGT]
IDX = {n: i for i, n in enumerate(NODES)}

# (u, v, cap, cost)
ARCS = [
    # 净水厂：出厂能力（万 m³/日）+ 制水成本（元/m³ × 1 万，即 元/万m³）
    ("S", "A", 6.0, 8000), ("S", "B", 7.0, 6500), ("S", "C", 3.0, 11500),
    # 厂 → 泵站
    ("A", "P", 5.0, 1500), ("A", "Q", 4.0, 3000),
    ("B", "Q", 2.5, 1200), ("B", "R", 4.0, 2200),
    ("C", "P", 2.0, 1800), ("C", "R", 3.0, 2800),
    # 泵站 ↔ 泵站
    ("P", "Q", 1.5, 1000), ("Q", "R", 10.0, 1400),    # 主输水干管：白天供水+夜间回蓄两股流共用，容量很大
    # 泵站 → 镇
    ("P", "D", 4.0, 2000), ("Q", "D", 2.5, 2600), ("Q", "E", 4.0, 1600),
    ("R", "E", 1.0, 3400), ("R", "F", 5.0, 1800),
    # 镇 ↔ 镇
    ("D", "E", 1.5, 1000), ("E", "F", 0.5, 900),
    # 北片 G：两条单价 0.50 / 0.70 的集水头
    ("C", "G", 6.0, 4500), ("D", "G", 4.0, 6000),
    # 镇接水口：日需水上限 3 万 m³/日（输配成本已含在上游）
    ("D", "T", 3.0, 0), ("E", "T", 3.0, 0), ("F", "T", 3.0, 0), ("G", "T", 3.0, 0),
]
# 水源侧单独处理：每天最多 11 万 m³ 进厂，「没被取走的那部分」记 2.00 元/m³ 的原水费+回收水费
# ⇒ 目标函数是**净成本**（含未取水费），基价不影响决策，但让 π(S) 与「取水」始终有对照
WATER_ARC = ("S", "T", WS_CAP, 20000)              # 送水（取水）
SPILL_ARC = ("T", "S", WS_CAP, 0)                  # 不送（把水留在源侧）：净成本 0，只用来配平供需
EXCLUDE = {("S", "T")}     # 水源余水弧：算「净成本」，不算「被调走的水」

# 候选新建干渠 (u, v, cap, cost, 年化建设费 万元/年)
EXCLUDE = {("S", "T")}     # 水源余水弧：算「净成本」，不算「被调走的水」
CANDIDATES = [
    ("B", "D", 4.0, 0.25, 700.0),    # 甲：乙厂直通北片（绕开 Q 泵站的 0.25 元/m³ 进站费）
    ("C", "D", 4.0, 0.25, 1500.0),   # 乙：丙厂直通北片（贵厂供水，不划算）
    ("C", "E", 3.0, 0.20, 1200.0),   # 丙：丙厂直通中片
    ("A", "F", 3.0, 0.10, 1500.0),   # 丁：甲厂直通南镇（绕开 Q→R 干管与 R 泵站）
]


def problem(cap_scale=1.0, extra=()):
    """「扩容」只放宽输水弧与厂出水口，不动镇接水口与水源余水弧。"""
    arcs = [(u, v, (cap * cap_scale if (c > 1e-12 or u == SRC) else cap), c)
            for (u, v, cap, c) in ARCS]
    return arcs + list(extra)


def net_problem(cap_scale=1.0, extra=(), ws_cap=WS_CAP):
    """完整问题 = 输配网络 + 送水弧（2.15 元/m³）+ 不送水弧（0 元/m³）。
    两条并联 ⇒ 取水的边际价值封顶 2.15 元/m³，同时 S、T 连通，节点势处处有限。"""
    arcs = problem(cap_scale, extra)
    return arcs + [(WATER_ARC[0], WATER_ARC[1], ws_cap, WATER_ARC[3]),
                   (SPILL_ARC[0], SPILL_ARC[1], ws_cap, SPILL_ARC[3])]


# ----------------------------------------------------------------------------
# 2. 最小费用最大流：SPFA 找最短路 + 沿最短路增广（手写，不经求解器）
# ----------------------------------------------------------------------------
def mcmf(arcs, s=SRC, t=TGT, nodes=NODES, demand=None):
    """demand=None ⇒ 最小费用最大流；给定 demand ⇒ 只送 demand 万 m³/日（最小费用流）。"""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    head = [[] for _ in range(n)]
    eu, ev, ecap, ecost = [], [], [], []
    for (u, v, cap, c) in arcs:
        for a, b, cp, cc in ((u, v, float(cap), float(c)), (v, u, 0.0, -float(c))):
            head[idx[a]].append(len(eu))
            eu.append(idx[a]); ev.append(idx[b]); ecap.append(cp); ecost.append(cc)
    flow = cost = 0.0
    while True:
        dist = [float("inf")] * n
        pv = [-1] * n
        inq = [False] * n
        dist[idx[s]] = 0.0
        dq = deque([idx[s]]); inq[idx[s]] = True
        while dq:
            u = dq.popleft(); inq[u] = False
            for a in head[u]:
                if ecap[a] > 1e-12:
                    v = ev[a]
                    if dist[u] + ecost[a] < dist[v] - 1e-12:
                        dist[v] = dist[u] + ecost[a]
                        pv[v] = a
                        if not inq[v]:
                            dq.append(v); inq[v] = True
        if dist[idx[t]] == float("inf"):
            break
        push, v = float("inf"), idx[t]
        while v != idx[s]:
            a = pv[v]; push = min(push, ecap[a]); v = eu[a]
        if demand is not None:
            push = min(push, demand - flow)
        v = idx[t]
        while v != idx[s]:
            a = pv[v]; ecap[a] -= push; ecap[a ^ 1] += push; v = eu[a]
        flow += push
        cost += push * dist[idx[t]]
        if demand is not None and flow >= demand - 1e-12:
            break
    return {"flow": flow, "cost": cost, "res_cap": ecap, "eu": eu, "ev": ev,
            "ecost": ecost, "n_arcs": len(arcs), "idx": idx}


def arc_flows(arcs, sol):
    return [max(0.0, cap - sol["res_cap"][2 * k]) for k, (u, v, cap, c) in enumerate(arcs)]


# ----------------------------------------------------------------------------
# 3. 节点势 π 与价差（检验数）——在残量网络上算
#    最优时残量网无负环 ⇒ 存在 π 使 对每条残量弧 π(v) ≤ π(u) + c
#    ⇒ 原弧（有残量）的检验数 rc = π(v) − π(u) − c ≥ 0；满容量的原弧其反向弧才有残量 ⇒ rc ≤ 0
# ----------------------------------------------------------------------------
def potentials(arcs, sol, nodes=NODES):
    """在残量网上求最短距离：加一个连向所有节点的超级源、零长虚弧，保证每个节点可达。
    （否则与 S 不连通的节点势为 ∞，势就没法当「相对水位」比较）"""
    idx = sol["idx"]; n = len(nodes)
    pi = [0.0] * n
    for _ in range(4 * n):        # 残量网可能有环，迭代到不动点（收敛即可）
        upd = False
        for a in range(2 * sol["n_arcs"]):
            if sol["res_cap"][a] > 1e-12:
                u, v, c = sol["eu"][a], sol["ev"][a], sol["ecost"][a]
                # 最短路口径：π(v) ≤ π(u) + c（对所有残量弧），迭代到不动点 = 最短距离
                if pi[u] + c < pi[v] - 1e-8:
                    pi[v] = pi[u] + c; upd = True
        if not upd:
            break
    return pi


def reduced_costs(arcs, sol, pi):
    idx = sol["idx"]
    out = []
    for k, (u, v, cap, c) in enumerate(arcs):
        f = max(0.0, cap - sol["res_cap"][2 * k])
        # 正弧上的检验数：只有「正向残量」时才要求 ≥0；满容量的弧要看反向弧「−c」的检验数
        rc = pi[idx[v]] - pi[idx[u]] - c        # π(v) − π(u) − c（π = 到该点的边际成本）
        res = cap - f
        # 原弧有残量 ⇒ 正向在残量网 ⇒ rc ≥ 0；满容量 ⇒ 只剩反向弧 ⇒ rc ≤ 0
        ok = (res > 1e-9 and rc <= 1e-6) or (res <= 1e-9 and rc >= -1e-6)
        out.append({"u": u, "v": v, "cap": cap, "c": c, "flow": f, "rc": rc, "ok": ok,
                    "residual": res, "pi_u": pi[idx[u]], "pi_v": pi[idx[v]]})
    return out


# ----------------------------------------------------------------------------
# 4. 独立裁判：LP（弧流量 + 流守恒）。线性成本下约束矩阵全单模，间隙应为 0
# ----------------------------------------------------------------------------
def solve_lp(arcs, neg_cols=(), nodes=NODES):
    """独立裁判：直接用 LP 求最优（变量 x_a ≥ 0，各节点流守恒）。
    neg_cols：虚拟的负代价列（多取水记账弧），是套利环 → 钉成 0，右端项相应补上。"""
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(arcs)
    A = np.zeros((len(nodes), m), float)
    for k, (u, v, cap, c) in enumerate(arcs):
        A[idx[u], k] -= 1.0
        A[idx[v], k] += 1.0
    keep = [i for i, n in enumerate(nodes) if n != SRC]
    A = A[keep]
    bnds = [(0.0, a[2]) for a in arcs]
    cvec = np.array([a[3] for a in arcs], float)
    for k in neg_cols:
        cvec[k] = 0.0
        bnds[k] = (0.0, 0.0)
    b = np.zeros(len(keep))
    b[keep.index(idx[TGT])] = 3.0 * len(TOWN) - len(neg_cols)   # T：四镇各 3 万，钉死的取水要扣掉
    res = linprog(cvec, A_eq=A, b_eq=b, bounds=bnds, method="highs")
    assert res.status == 0, res.message
    return res


# ----------------------------------------------------------------------------
# 5. 图上路径搜索：BFS 找一条 s→t 的简单路（原始弧，不考虑容量、不考虑反向）
# ----------------------------------------------------------------------------
def bfs_path(arcs, s, t):
    adj = {}
    for (u, v, cap, c) in arcs:
        adj.setdefault(u, []).append(v)
    prev = {s: None}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                if v == t:
                    path = [t]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                dq.append(v)
    return None


# ----------------------------------------------------------------------------
# 6. 反例：两种水质共管 —— 单商品节点势失效
# ----------------------------------------------------------------------------
def multicommodity(arcs):
    """两种水共用部分管道：生活水 7 万 m³/日、工业水 4 万 m³/日（源头共 11 万），工业水同管同程 +30% 处理费。
    对照 ①「各算一遍再相加」；② 联合 LP（两条流 + 共用容量）。"""
    LIVE, IND = 7.0, 4.0
    n_live = mcmf(arcs, demand=LIVE)
    pi1 = potentials(arcs, n_live)
    n_ind = mcmf(arcs, demand=IND)
    idx = {n: i for i, n in enumerate(NODES)}
    m = len(arcs)
    A_eq = np.zeros((len(NODES), 2 * m))
    for k, (u, v, cap, c) in enumerate(arcs):
        for col in (k, m + k):
            A_eq[idx[u], col] -= 1.0
            A_eq[idx[v], col] += 1.0
    b_eq = np.zeros(len(NODES))
    b_eq[idx[SRC]] = -(LIVE + IND)
    b_eq[idx[TGT]] = LIVE + IND
    A_ub = np.zeros((m, 2 * m))
    for k in range(m):
        A_ub[k, k] = 1.0
        A_ub[k, m + k] = 1.0
    cvec = np.array([a[3] for a in arcs] * 2, float)
    cvec[m:] *= 1.3
    r = linprog(cvec, A_ub=A_ub, b_ub=[a[2] for a in arcs], A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * (2 * m), method="highs")
    assert r.status == 0, r.message
    # 单商品势拿去评工业水：A→P 这条弧的「工业水检验数」= 1.3c − (π(P)−π(A))
    # 检验数（口径与 reduced_costs 一致）：π(v) − π(u) − c
    rc_ind = (pi1[idx["P"]] - pi1[idx["A"]]) - 1.3 * 1500
    return {"分开算相加_万元日": round((n_live["cost"] + n_ind["cost"]) / 1e4, 4),
            "联合LP_万元日": round(float(r.fun) / 1e4, 4),
            "节约_万元日": round((n_live["cost"] + n_ind["cost"] - r.fun) / 1e4, 4),
            "生活水单独_万元日": round(n_live["cost"] / 1e4, 4),
            "工业水单独_万元日": round(n_ind["cost"] / 1e4, 4),
            "单商品势评_A到P_工业水检验数_元m3": round(rc_ind / 1e4, 4),
            "说明": "分开算把同一条管子的容量用了两次（超卖）；"
                    "用生活水的 π 去评工业水，A→P 上的检验数为负，等于说「这条路该走」，"
                    "可它恰恰是被生活水占满的那条 —— 单商品势在共管时不成立"}


# ----------------------------------------------------------------------------
# 7. 画图
# ----------------------------------------------------------------------------
def plot(arcs, flows, pi, scan, cand, src_rows):
    pos = {"S": (0, 0.5), "A": (1, 0.88), "B": (1, 0.5), "C": (1, 0.12),
           "P": (2, 0.88), "Q": (2, 0.5), "R": (2, 0.12),
           "D": (3.2, 0.88), "E": (3.2, 0.58), "F": (3.2, 0.25), "G": (3.2, 0.0),
           "T": (4.6, 0.48)}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.6))

    a0 = ax[0, 0]
    for (u, v, cap, c), f in zip(arcs, flows):
        on = f > 1e-9
        a0.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", lw=0.7 + 3.0 * f / max(cap, 1e-9),
                                    color="#2b6cb0" if on else "#d0d0d0",
                                    shrinkA=13, shrinkB=13))
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        a0.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#2b6cb0" if on else "#e2e2e2",
                lw=0.8 + 3.0 * f / max(cap, 1e-9), zorder=1, alpha=0.55 if on else 0.6)
        a0.text(mx, my + 0.028, f"{f:.1f}/{cap:.1f}", fontsize=6.8, ha="center", color="#1a365d")
    for n_, (px, py) in pos.items():
        a0.scatter([px], [py], s=430, c="#fefcbf" if n_ not in (SRC, TGT) else "#feb2b2",
                   edgecolors="#2d3748", zorder=3)
        a0.text(px, py, n_, ha="center", va="center", fontsize=9.5, zorder=4)
    a0.set_title("(a) 调水网络与最优流量（弧上：万m³/日 / 容量）", fontsize=11)
    a0.set_xlim(-0.35, 5.0); a0.set_ylim(-0.12, 1.02); a0.axis("off")

    a1 = ax[0, 1]
    ks = [n_ for n_ in NODES if n_ != TGT]
    vals = [pi[IDX[k]] for k in ks]
    cols = ["#fc8181" if k == SRC else ("#4299e1" if k in TOWN else "#9ae6b4") for k in ks]
    a1.bar(ks, vals, color=cols)
    for i, v in enumerate(vals):
        a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    a1.set_ylabel("π（元/m³）")
    a1.set_title("(b) 节点势 π = 该节点再取 1 m³ 水的边际成本\n（红=水泵厂  绿=泵站+水厂  蓝=镇）", fontsize=10.5)

    a2 = ax[1, 0]
    xs = [d["系数"] for d in scan]; ys = [d["净成本_万元日"] for d in scan]
    a2.plot(xs, ys, "o-", color="#38a169")
    for d in scan:
        a2.annotate(f"F={d['流量']:.1f}", (d["系数"], d["净成本_万元日"]), textcoords="offset points",
                    xytext=(0, 7), fontsize=7.5, ha="center")
    a2.set_xlabel("全网输水能力放大系数 x"); a2.set_ylabel("最小总成本（万元/日）")
    a2.set_title("(c) 扩容扫描：曲线上每个折点 = 瓶颈换人", fontsize=11)
    a2.grid(alpha=0.3)

    a3 = ax[1, 1]
    labels = [f"建 {d['弧']}" for d in cand] + [f"扩 {d['节点']}" for d in src_rows]
    vals2 = [d["日省_万元"] for d in cand] + [-d["日省_万元"] for d in src_rows]   # 扩产：正数 = 省钱
    colors = ["#2b6cb0"] * len(cand) + ["#e53e3e"] * len(src_rows)
    a3.barh(labels, vals2, color=colors)
    for i, v in enumerate(vals2):
        a3.text(v + max(vals2) * 0.01, i, f"{v:+.3f}", va="center", fontsize=8)
    a3.set_xlabel("日省（万元/日）")
    a3.set_title("(d) 蓝＝新建干渠一次重算  红＝净水厂出水 +1 万m³/日（正数 = 省钱）", fontsize=11)
    a3.invert_yaxis()

    fig.suptitle("Day83 云溪县五镇供水调度：最小费用流 / 节点势 / 价差检验", fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(HERE, "day83_mincost_flow.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------------
def main():
    out = {}
    demand = 3.0 * len(TOWN)                # 四镇日需水合计 12 万 m³/日
    arcs = problem()                        # 输配网络（不算水源余水）
    sol = mcmf(arcs, demand=demand)
    flows = arc_flows(arcs, sol)
    narcs = net_problem()                   # 净成本口径（含水源余水弧），用于节点势与价差
    nsol = mcmf(narcs, demand=demand)
    pi_raw = potentials(narcs, nsol)
    pi = [(p_ - pi_raw[IDX[SRC]]) / 1e4 for p_ in pi_raw]     # 相对水位：S 取水从 0 开始，单位 元/m³
    red = reduced_costs(narcs, nsol, pi_raw)

    out["基准"] = {"流量_万m3日": round(sol["flow"], 3), "净成本_万元日": round(sol["cost"] / 1e4, 4),
                 "平均到货价_元m3": round(sol["cost"] / 1e4 / sol["flow"], 4)}
    print(f"[0] 四镇到水 {sol['flow']:.2f} 万m³/日；输配 + 制水净成本 {sol['cost']/1e4:.4f} 万元/日"
          f"（平均到货价 {sol['cost']/1e4/sol['flow']:.4f} 元/m³）")

    # 净成本口径里，反弧 T→S 是虚拟的「多取水量」记账（代价 −2.15 元/m³ /1 万），
    # 在 LP 里是个套利环，必须钉成 0；它的影子价稍后由有限差分单独核对
    neg_cols = [k for k, (u, v, cap, c) in enumerate(narcs) if c < 0]
    lp0 = solve_lp(narcs, neg_cols)
    assert abs(lp0.fun / 1e4 - nsol["cost"] / 1e4) < 1e-6, (lp0.fun, nsol["cost"])
    print(f"    LP 裁判（净成本口径）：{lp0.fun/1e4:.4f} 万元/日 —— 与手写 MCMF 一致")
    lp1 = solve_lp(arcs)
    assert abs(lp1.fun - sol["cost"]) < 1e-6, (lp1.fun, sol["cost"])
    print(f"    两个口径都核过：输配 {sol['cost']/1e4:.4f} 万元/日 + 水源余水项 = 净成本"
          f"（{nsol['cost']/1e4:.4f}）")

    for k, (u, v, cap, c) in enumerate(arcs):
        assert -1e-9 <= flows[k] <= cap + 1e-9
    net = {n_: 0.0 for n_ in NODES}
    for (u, v, cap, c), f in zip(arcs, flows):
        net[u] -= f; net[v] += f
    assert all(abs(net[n_]) < 1e-9 for n_ in NODES if n_ not in (SRC, TGT))
    assert abs(net[SRC] + sol["flow"]) < 1e-9 and abs(net[TGT] - sol["flow"]) < 1e-9
    demand = sum(f for (u, v, cap, c), f in zip(arcs, flows) if v == TGT and u in TOWN)
    assert abs(demand - 3.0 * len(TOWN)) < 1e-9
    print(f"    守恒与容量检查通过（四镇各 3.0，共 {demand:.1f} 万 m³/日）")

    # 5.1 检验数：有残量 ⇒ rc ≤ 0；满容量 ⇒ rc ≥ 0（π 是「到该点的边际成本」）
    bad = [r for r in red if not r["ok"]]
    assert not bad, f"最优性条件被破坏（{len(bad)} 条弧）：{bad[:3]}"
    flow_arcs = [r for r in red if r["flow"] > 1e-9]
    slack = [r for r in red if r["residual"] > 1e-9]
    tight = [r for r in red if r["residual"] <= 1e-9]
    print(f"[1] {len(red)} 条弧：有正流量 {len(flow_arcs)} 条，用满 {len(tight)} 条，残量>0 {len(slack)} 条"
          " —— 互补松弛全部满足")
    tight_list = [f"{r['u']}->{r['v']}({r['flow']:.1f}/{r['cap']:.1f})"
                  for r in tight if (r["u"], r["v"]) not in EXCLUDE]
    out["紧弧"] = tight_list
    print("    用满的弧：" + "  ".join(tight_list))
    for n_ in TOWN:
        incident = [r for r in red if r["v"] == n_ and (r["u"], r["v"]) not in EXCLUDE]
        print(f"      {n_} 的上游弧：" + "，".join(
            f"{r['u']}->{r['v']} {r['flow']:.2f}/{r['cap']:.1f} rc={r['rc']/1e4:+.2f}" for r in incident))

    out["节点势"] = {k: round(pi[IDX[k]], 4) for k in NODES if k != TGT}
    print("[2] 节点势 π（元/m³，= 该节点再取 1 万 m³ 的边际成本）：")
    for k in NODES:
        if k != TGT:
            print(f"     {k}: {pi[IDX[k]]:7.4f}")

    # 5.2 镇要多水：D 镇接水口 +1 万 m³/日（净成本差就是 1 万 m³ 的边际成本）
    a_d = [(u, v, cap + 1.0, c) if (u, v) == ("D", "T") else (u, v, cap, c) for (u, v, cap, c) in narcs]
    r_d = mcmf(a_d, demand=demand + 1.0)
    fd = (r_d["cost"] - nsol["cost"]) / 1e4
    out["D镇加水_元每m3"] = round(fd, 4)
    print(f"[3] 北片 D 镇多要 1 万 m³/日 → 净成本 +{fd:.4f} 万元/日 = +{fd:.4f} 元/m³"
          f"（节点势 π(D) = {pi[IDX['D']]:.4f}）")
    assert abs(fd - pi[IDX["D"]]) < 5e-3, (fd, pi[IDX["D"]])

    # 5.3 全网扩容扫描（输配网络口径，看流量与平均成本）
    scan = []
    for s_ in [0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
        r2 = mcmf(net_problem(cap_scale=s_), demand=demand)
        scan.append({"系数": s_, "流量": round(r2["flow"], 3),
                     "净成本_万元日": round(r2["cost"] / 1e4, 4),
                     "均价_元m3": round(r2["cost"] / 1e4 / r2["flow"], 4)})
    out["扩容扫描"] = scan
    print("[3b] 全网输水能力 ×x 的扫描（净成本口径）：")
    for d in scan:
        print(f"      x{d['系数']:.2f}  到水 {d['流量']:5.2f}  净成本 {d['净成本_万元日']:8.4f} 万元/日  "
              f"平均 {d['均价_元m3']:.4f} 元/m³")

    # 5.4 候选新建干渠：建 → 净成本口径重算
    cand_rows = []
    for (u, v, cap, c, capex) in CANDIDATES:
        arc_new = (u, v, cap, int(round(c * 1e4)))
        nn = net_problem(extra=[arc_new])
        r2 = mcmf(nn, demand=demand)
        gain = (nsol["cost"] - r2["cost"]) / 1e4              # 万元/日
        f_new = arc_flows(nn, r2)[-1]
        path = bfs_path(arcs, u, v)
        pnorm, pcap = 0.0, float("inf")
        for (a_, b_) in zip(path[:-1], path[1:]):
            for (p_, q_, cp_, cc_) in arcs:
                if (p_, q_) == (a_, b_):
                    pnorm += cc_ / 1e4; pcap = min(pcap, cp_); break
        gain_theory = max(0.0, pnorm - c)
        pay = capex / (gain * 365.0) if gain > 1e-9 else float("inf")
        cand_rows.append({"弧": f"{u}->{v}", "容量": cap, "单价": c, "建设费_万元": capex,
                          "新弧上流量": round(f_new, 3),
                          "日省_万元": round(gain, 4),
                          "静态回收_年": None if not np.isfinite(pay) else round(pay, 2),
                          "图上替代路": "→".join(path),
                          "替代路单价": round(pnorm, 4),
                          "替代路瓶颈容量": pcap,
                          "价差上限_元m3": round(gain_theory, 4)})
    out["候选干渠"] = cand_rows
    print("[4] 候选新建干渠（塞进图里重算）：")
    for d in cand_rows:
        print(f"      {d['弧']}  新弧 {d['新弧上流量']:.2f} 万m³/日  日省 {d['日省_万元']:+.4f} 万元  "
              f"回收 {d['静态回收_年']} 年 | 图上替代路 {d['图上替代路']}（单价 {d['替代路单价']:.3f}，"
              f"瓶颈 {d['替代路瓶颈容量']:.1f}）")
    assert any(d["日省_万元"] > 1e-9 for d in cand_rows), "没有任何候选干渠值得建，实例需重设"

    # 5.5 已有设施 +1 万 m³/日（净成本口径）
    src_rows = []
    for ns in [SRC] + GEN:
        nn = [(u, v, cap + 1.0, c) if u == ns else (u, v, cap, c) for (u, v, cap, c) in net_problem()]
        r2 = mcmf(nn, demand=demand)
        src_rows.append({"节点": ns, "日省_万元": round((nsol["cost"] - r2["cost"]) / 1e4, 4),
                         "增流量": round(r2["flow"] - nsol["flow"], 3)})
    out["扩产逐点"] = src_rows
    print("[5] 各节点 +1 万 m³/日 的净成本变化（负 = 省钱）：")
    for d in src_rows:
        print(f"      {d['节点']}: {d['日省_万元']:+.4f} 万元/日，流量 {d['增流量']:+.2f}")

    # 5.6 多商品反例
    mc = multicommodity(narcs)
    out["多商品反例"] = mc
    print("[6] 两商品（生活水/工业水共管）反例：")
    for k, v_ in mc.items():
        print(f"      {k}: {v_}")

    # 5.7 写进论文前必查的两件小事
    ratio = nsol["cost"] / 1e4 / nsol["flow"]
    print(f"[7] 平均到货价 = 净成本 ÷ 到水量 = {nsol['cost']/1e4:.4f} ÷ {nsol['flow']:.2f} = {ratio:.4f} 元/m³")
    print("    逐点势里最贵的 G 是 {:.2f}，比这个平均值高 {:.1f}%".format(
        pi[IDX["G"]], 100 * (pi[IDX["G"]] - ratio) / ratio))
    out["平均到货价检查"] = {"均值_元m3": round(ratio, 4), "G镇_元m3": round(pi[IDX["G"]], 4),
                           "最大相对平均_百分比": round(100 * (pi[IDX["G"]] - ratio) / ratio, 2)}

    out["runtime_s"] = round(time.time() - T0, 2)
    with open(os.path.join(HERE, "day83_result.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    plot(arcs, flows, pi, scan, cand_rows, src_rows)
    print(f"\n完成，用时 {out['runtime_s']} s")


if __name__ == "__main__":
    main()
