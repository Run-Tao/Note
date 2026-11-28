# -*- coding: utf-8 -*-
"""临时探路 5：SAA 场景树 vs 仿真真值（pen 抬高），外加 EV / EVPI / VSS 口径"""
import numpy as np
from scipy.optimize import linprog

PRICE, H_RATE = 2800.0, 0.20
H = PRICE * H_RATE / 52
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)


def solve_SAA(peaks, pen, r_u):
    S_, W = len(peaks), len(SEASON)
    dem = peaks[:, None] * SEASON[None, :]
    nq, ni, nl = S_ * W, S_ * (W + 1), S_ * W
    nx = nq + ni + nl   # 列：q(s,w) | i(s,0..W) | l(s,w)
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
            assert len(np.nonzero(row)[0]) == len(set(np.nonzero(row)[0].tolist())), (s, w)
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


def sim(q, dem, pen, r_u):
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


def oracle_gap(peaks_true, p_true, pen, r_u, n=400):
    """perfect information：给未来一周的 q 也算出来（每周独立订），拿期望"""
    print("  dbg peaks:", p_true[:5])
    tot = 0.0
    best = []
    for w in range(8):
        # 单周最优 q：min 2800q + H*sales + pen*lost（期末剩余按 r_u 回收）
        qs = np.linspace(0, 400, 401)
        dd = np.random.default_rng(11 + w).choice(d[:, w], n)
        costs = []
        for q in qs:
            sales = np.minimum(q, dd)
            lost = np.maximum(dd - q, 0)
            left = q - sales
            costs.append((PRICE * q + H * sales + pen * lost - r_u * left).mean())
        best.append(qs[int(np.argmin(costs))])
    return np.array(best)


if __name__ == "__main__":
    for pen in [5400.0, 7200.0, 9000.0]:
        p_true = np.random.default_rng(999).normal(120, 49, 200000).clip(0, 240)
        test = p_true[:6000, None] * SEASON[None, :]
        pbar = 120.0
        # EV：用均值场景的 q
        _, q_ev = solve_SAA(np.array([pbar]), pen, 1200.0)
        c_ev = sim(q_ev, test, pen, 1200.0).mean()
        # SAA
        out = []
        for nsc in [5, 20, 100, 500]:
            ceq, q = solve_SAA(np.random.default_rng(5).normal(120, 49, nsc).clip(0, 240), pen, 1200.0)
            out.append((nsc, ceq, sim(q, test, pen, 1200.0).mean(), q))
        print(f"pen={pen:6.0f} q_ev={np.round(q_ev,0)} EV={c_ev:8.0f} " + " | ".join(
            f"n={o[0]:3d} q={np.round(o[3],0)} ceq={o[1]/8:8.0f} true={o[2]:8.0f} VSS={o[2]-c_ev:7.0f}({100*(o[2]-c_ev)/o[2]:5.1f}%)" for o in out))
