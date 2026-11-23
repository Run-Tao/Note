# -*- coding: utf-8 -*-
"""临时探路 4：干净重写 —— 场景 s、周 w：q_{s,w} 进货、l_{s,w} 缺货、i_{s,w} 期初库存"""
import numpy as np
from scipy.optimize import linprog

PRICE, H_RATE = 2800.0, 0.20
H = PRICE * H_RATE / 52
PEN = 1800.0
R_U = 1200.0
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)


def solve_SAA(peaks, pen=PEN, r_u=R_U):
    S_, W = len(peaks), len(SEASON)
    dem = peaks[:, None] * SEASON[None, :]
    nq, ni, nl = S_ * W, S_ * (W + 1), S_ * W
    nx = nq + ni + nl
    Q = lambda s, w: s * W + w
    I = lambda s, w: nq + s * (W + 1) + w
    L = lambda s, w: nq + ni + s * W + w
    c = np.zeros(nx)
    Aeq, beq = [], []
    for s in range(S_):
        for w in range(W):
            row = np.zeros(nx)
            row[I(s, w + 1)] = 1
            if w > 0:
                row[I(s, w)] = -1
            row[Q(s, w)] = -1
            row[L(s, w)] = -1
            Aeq.append(row)
            beq.append(-dem[s, w])
        for w in range(W):
            c[Q(s, w)] += PRICE / S_
            c[I(s, w)] += H / S_
            c[L(s, w)] += pen / S_
        c[I(s, W)] += (PRICE - r_u) / S_
    r = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * nx, method="highs")
    assert r.status == 0, r.message
    q = np.array([sum(r.x[Q(s, w)] for s in range(S_)) / S_ for w in range(W)])
    return r.fun / W, q


def sim(q, dem, pen=PEN, r_u=R_U):
    """q: 每周订货量（预案），到货即入库；缺货不欠"""
    R, W = dem.shape
    inv = np.zeros(R)
    cost = np.zeros(R)
    for w in range(W):
        a = inv + q[w]
        sales = np.minimum(a, dem[:, w])
        lost = np.maximum(dem[:, w] - a, 0.0)
        inv = a - sales
        cost += PRICE * q[w] + H * sales + pen * lost
    cost += (PRICE - r_u) * inv
    return cost / W


if __name__ == "__main__":
    for pen in [1800.0, 5400.0]:
        for r_u in [1200.0, 672.0]:
            line = []
            for nsc in [1, 5, 20, 200]:
                ps = np.random.default_rng(5).normal(120, 49, nsc).clip(0, 240)
                ceq, q = solve_SAA(ps, pen, r_u)
                test = np.random.default_rng(999).normal(120, 49, 4000).clip(0, 240)[:, None] * SEASON[None, :]
                line.append(f"n={nsc:3d} ceq={ceq:8.0f} true={sim(q, test, pen, r_u).mean():8.0f} q={np.round(q, 0)}")
            print(f"pen={pen:5.0f} r_u={r_u:5.0f} " + " | ".join(line))
