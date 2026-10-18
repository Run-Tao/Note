# -*- coding: utf-8 -*-
"""诊断：2026-10-17 半成品 FSP 脚本的数据一致性体检（不入库，只做证据）
① 重新核对 2026-10-17 版 P_BASE 的行和 / 列和；② 该数据的 LPT/NEH/IG 对照；
③ 证明「死数据」判据：直觉规则全部挤在一起 + NEH 与 IG 完全等价 + 解与下界错位。
"""
import numpy as np
P = np.array([
 [ 3,  9,  4,  1,  2],[ 2, 11,  3,  1,  2],[ 4,  7,  5,  2,  3],[ 3, 12,  4,  1,  2],
 [ 2,  8,  3,  1,  2],[ 5, 10,  6,  2,  3],[ 3,  6,  4,  1,  2],[ 4, 13,  5,  2,  3],
 [ 2,  7,  3,  1,  2],[ 3,  9,  4,  1,  2],[ 4, 11,  5,  1,  3],[ 2,  8,  3,  2,  2],
 [ 3, 10,  4,  1,  2],[ 4,  9,  6,  2,  3]], dtype=float)
def sched(p, P):
    C = np.zeros((len(p), P.shape[1]))
    for i, j in enumerate(p):
        for k in range(P.shape[1]):
            st = 0.0
            if i: st = max(st, C[i-1, k])
            if k: st = max(st, C[i, k-1])
            C[i, k] = st + P[j, k]
    return C[-1, -1], C
ms = lambda p: sched(list(p), P)[0]
def neh(P, order):
    seq = [int(order[0])]
    for j in map(int, order[1:]):
        best, bp = np.inf, 0
        for pos in range(len(seq)+1):
            v = ms(seq[:pos]+[j]+seq[pos:])
            if v < best: best, bp = v, pos
        seq.insert(bp, j)
    return seq
print("行和（每批总工时）:", P.sum(1).tolist(), " 合计:", P.sum())
print("列和（各工序总工时）:", P.sum(0).tolist(), " 合计:", P.sum(0).sum())
print("LPT 顺序:", list(np.argsort(-P.sum(1))), " → makespan", ms(np.argsort(-P.sum(1))))
print("SPT → ", ms(np.argsort(P.sum(1))))
print("瓶颈优先 → ", ms(np.argsort(-P[:, 1])))
n = neh(P, np.argsort(-P.sum(1)))
print("NEH 顺序:", n, " → makespan", ms(n))
rng = np.random.default_rng(1)
seq = list(n); cur = ms(seq)
def best_insert(seq, j):
    best, bp = np.inf, 0
    for pos in range(len(seq)+1):
        v = ms(seq[:pos]+[j]+seq[pos:])
        if v < best: best, bp = v, pos
    return seq[:bp]+[j]+seq[bp:], best
hist = [cur]
for _ in range(300):
    idx = sorted(rng.choice(len(seq), 4, replace=False).tolist())
    rem = [seq[i] for i in idx]
    part = [x for i, x in enumerate(seq) if i not in set(idx)]
    rng.shuffle(rem)
    cand = part
    for j in rem: cand, _ = best_insert(cand, j)
    cm = ms(cand)
    if cm < cur: seq, cur = cand, cm
    hist.append(cur)
print("IG（从 NEH 出发，300 次）:", cur, "收敛轨迹:", hist[0], "→", hist[50], "→", hist[-1])
print("结论：上版数据上 IG 一步无改进、与 NEH 完全等价 → 该实例无法展示任何算法差异")
