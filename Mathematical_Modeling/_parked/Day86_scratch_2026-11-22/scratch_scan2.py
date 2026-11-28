# -*- coding: utf-8 -*-
"""临时探路 2：两阶段随机库存，变量就是各期「补到多少」(补货策略)，场景内确定性执行"""
import numpy as np
from scipy.optimize import linprog

PRICE, H_RATE, WEEKS = 2800.0, 0.20, 52.0
H = PRICE * H_RATE / WEEKS
R_U = 1200.0
PEN = 1800.0
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)


def solve_SAA(peak_samp, pen=PEN, r_u=R_U):
    S_, W = len(peak_samp), len(SEASON)
    dem = peak_samp[:, None] * SEASON[None, :]
    n_u = W
    inv = (W + 1) * S_
    lost = W * S_
    nx = n_u + inv + lost
    iU = lambda w: w
    iI = lambda s, w: n_u + s * (W + 1) + w
    iL = lambda s, w: n_u + inv + s * W + w
    c = np.zeros(nx)
    for w in range(W):
        c[iU(w)] = PRICE
    A, b = [], []
    for s in range(S_):
        for w in range(W):
            base = np.zeros(nx)
            base[iI(s, w + 1)] = -1        # 期末库存
            if w > 0:
                base[iI(s, w)] = 1         # 期初库存 = 上期期末
            base[iU(w)] = 1
            r1 = base.copy(); r1[iL(s, w)] = 1; A.append(r1); b.append(dem[s, w])
            r2 = base.copy(); r2[iL(s, w)] = -1; A.append(r2); b.append(0.0)
        for w in range(W):
            c[iI(s, w)] += H / S_
            c[iL(s, w)] += pen / S_
        c[iI(s, W)] += (PRICE - r_u) / S_
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)] * nx,
                method="highs")
    assert r.status == 0, r.message
    return r.fun / W, r.x[:W]


def sim(U, dem, pen=PEN, r_u=R_U):
    """按 U 的补货策略执行：每期『可用量 + 补货量 = U_w』（补货期提前期忽略，等价于 JIT 补货）"""
    R, W = dem.shape
    inv = np.zeros(R)
    cost = np.zeros(R) + sum(PRICE * U for U in U)
    for w in range(W):
        a = np.maximum(U[w], inv)     # 已经够多就不补（这里 U 是上限）
        a = np.minimum(U[w], inv) if False else np.maximum(inv, U[w])
        sales = np.minimum(a, dem[:, w])
        lost = np.maximum(dem[:, w] - a, 0.0)
        inv = a - sales
        cost += H * sales + pen * lost
    cost += (PRICE - r_u) * inv
    return cost / W


for pen in [1800.0, 5400.0]:
    for r_u in [1200.0, 672.0]:
        out = []
        for nsc in [1, 5, 20, 200]:
            ps = np.random.default_rng(5).normal(120, 49, nsc).clip(0, 240)
            ceq, U = solve_SAA(ps, pen, r_u)
            test = np.random.default_rng(999).normal(120, 49, 4000).clip(0, 240)[:, None] * SEASON[None, :]
            out.append((nsc, ceq, sim(U, test, pen, r_u).mean(), U.sum()))
        print(f"pen={pen:6.0f} r_u={r_u:6.0f} " + " | ".join(
            f"n={o[0]:3d} ceq={o[1]:8.1f} true={o[2]:8.1f} ΣU={o[3]:6.1f}" for o in out))
