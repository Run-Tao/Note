# -*- coding: utf-8 -*-
"""临时探路 3：显式列平衡 inv_{w+1} = inv_w + U_w - d_w + lost_w"""
import numpy as np
from scipy.optimize import linprog

PRICE, H_RATE, WEEKS = 2800.0, 0.20, 52.0
H = PRICE * H_RATE / WEEKS
PEN = 1800.0
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)


def solve_SAA(peak_samp, pen=PEN, r_u=1200.0):
    S_, W = len(peak_samp), len(SEASON)
    dem = peak_samp[:, None] * SEASON[None, :]
    nx = W + (W + 1) * S_ + W * S_
    iU = lambda w: w
    iI = lambda s, w: W + s * (W + 1) + w
    iL = lambda s, w: W + (W + 1) * S_ + s * W + w
    c = np.zeros(nx)
    Aeq, beq = [], []
    for w in range(W):
        c[iU(w)] = PRICE
        assert iU(w) != iL(0, 0)
    for s in range(S_):
        for w in range(W):
            row = np.zeros(nx)
            row[iI(s, w + 1)] = 1
            row[iI(s, w)] = -1
            row[iL(s, w)] = -1         # 缺货
            row[iU(w)] = -1
            Aeq.append(row); beq.append(-dem[s, w])
        for w in range(W):
            c[iI(s, w)] += H / S_
            c[iL(s, w)] += pen / S_
        c[iI(s, W)] += (PRICE - r_u) / S_
    r = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * nx, method="highs")
    assert r.status == 0, r.message
    return r.fun / W, r.x[:W]


def sim(U, dem, pen=PEN, r_u=1200.0):
    R, W = dem.shape
    inv = np.zeros(R)
    cost = np.zeros(R) + sum(PRICE * U)
    for w in range(W):
        a = np.maximum(inv, U[w])
        sales = np.minimum(a, dem[:, w])
        lost = np.maximum(dem[:, w] - a, 0.0)
        inv = a - sales
        cost += H * sales + pen * lost
    cost += (PRICE - r_u) * inv
    return cost / W


for pen in [1800.0, 5400.0]:
    for r_u in [1200.0, 672.0, 0.0]:
        out = []
        for nsc in [1, 5, 20, 200]:
            ps = np.random.default_rng(5).normal(120, 49, nsc).clip(0, 240)
            ceq, U = solve_SAA(ps, pen, r_u)
            test = np.random.default_rng(999).normal(120, 49, 4000).clip(0, 240)[:, None] * SEASON[None, :]
            out.append((nsc, ceq, sim(U, test, pen, r_u).mean(), U.mean()))
        print(f"pen={pen:6.0f} r_u={r_u:6.0f} " + " | ".join(
            f"n={o[0]:3d} ceq={o[1]:8.1f} true={o[2]:8.1f} Ū={o[3]:6.1f}" for o in out))
