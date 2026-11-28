# -*- coding: utf-8 -*-
"""探路 6：SAA n 扫描的统计显著性（同一份 n=2000 测试集，配对差分 t 值）"""
import numpy as np

PRICE, H_RATE = 2800.0, 0.20
H = PRICE * H_RATE / 52
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)
Q_EV = np.array([90, 90, 150, 150, 150, 150, 110, 110], float)
Q_SAA = {5: np.array([84, 84, 140, 140, 140, 140, 103, 103], float),
         'SAA_n': np.array([83, 83, 138, 138, 138, 138, 102, 102], float)}


def sim_per_path(q, peaks, pen, r_u=1200.0):
    dem = peaks[:, None] * SEASON[None, :]
    inv = np.zeros(len(peaks))
    cost = np.zeros(len(peaks))
    for w in range(8):
        a = inv + q[w]
        sales = np.minimum(a, dem[:, w])
        lost = np.maximum(dem[:, w] - a, 0.0)
        inv = a - sales
        cost += PRICE * q[w] + H * sales + pen * lost
    cost += (PRICE - r_u) * inv
    return cost / 8


rng = np.random.default_rng(999)
peaks = rng.normal(120, 49, 100000).clip(0, 240)
for pen in [5400.0, 9000.0]:
    base = sim_per_path(Q_EV, peaks, pen)
    for name, q in Q_SAA.items():
        c = sim_per_path(q, peaks, pen)
        d = c - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"pen={pen:6.0f} {str(name):6s} EV={base.mean():8.1f} alt={c.mean():8.1f} diff={d.mean():7.1f} t={t:6.2f}")
