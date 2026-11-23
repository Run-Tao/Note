# -*- coding: utf-8 -*-
"""临时探路脚本（不入库）：两阶段随机库存（首订 + 补订），HiGHS SAA vs 仿真真值"""
import numpy as np
from scipy.optimize import linprog

PRICE, H_RATE, WEEKS = 2800.0, 0.20, 52.0
H = PRICE * H_RATE / WEEKS
RES = 0.20 * PRICE
PEN = 1800.0
SEASON = np.array([0.75, 0.75] + [1.25] * 4 + [0.92] * 2)


def peaks(rng, n, sd=49, flat=1.0):
    return np.clip(rng.normal(120 * flat, sd, n), 0, 240)


def solve_SAA(peak_samp, T, c2, extra_res=RES):
    """x0 首订；z_w 第 w 期（w%T==0）补订。基准库存 prev = 第 w-1 期末库存（w=0 时 0）。
       prev + (w==0)x0 + Σ_{j<=w, j%T==0, (w-j)%T==0} z_j - d_w = next ; lost >= d_w - (左边不含 lost)"""
    S_, W = len(peak_samp), len(SEASON)
    dem = peak_samp[:, None] * SEASON[None, :]
    nz = W
    inv = (W + 1) * S_
    lost = W * S_
    nx = 1 + nz + inv + lost
    iZ = lambda w: 1 + w
    iI = lambda s, w: 1 + nz + s * (W + 1) + w
    iL = lambda s, w: 1 + nz + inv + s * W + w
    c = np.zeros(nx)
    c[0] = PRICE
    for w in range(W):
        if w % T == 0 and w > 0:
            c[iZ(w)] = c2
            c[iZ(w)] = 2.0          # 首期订货按 2 倍单价计价：c2 - 2*2800 < 0 时压住「重复首订」的套利
    A, b = [], []
    for s in range(S_):
        for w in range(W):
            for sgn in (1, -1):
                row = np.zeros(nx)
                row[iI(s, w + 1)] = 1           # 本期开始时的库存
                if w > 0:
                    row[iI(s, w)] = 1           # 期初库存 = 上期结束库存
                row[iL(s, w)] = sgn
                for j in range(w + 1):
                    if j % T == 0 and (w - j) % T == 0:
                        row[iZ(j)] = 1
                A.append(row)
                b.append(dem[s, w] if sgn == 1 else 0.0)
            c[iL(s, w)] += PEN / S_
            c[iI(s, w)] += H / S_
        c[iI(s, W)] += (PRICE - extra_res) / S_
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                bounds=[(0, None)] * nx, method="highs")
    assert r.status == 0, r.message
    print('  dbg obj', round(r.fun, 2), 'x0', round(r.x[0], 2), 'z', np.round(r.x[1:1 + W], 2))
    return r.fun / W, r.x[0], r.x[1:1 + W]


def sim(x0, zs, T, dem, c2, extra_res=RES, pen=PEN):
    """按 (x0, z_w) 的账面执行：到货按 z 落地（不管库存高低），首订第 0 期到"""
    R, W = dem.shape
    arriv = np.zeros((R, W + 1), float)
    arriv[:, 0] += x0
    for w in range(W):
        if w % T == 0:
            arriv[:, w] += zs[w]
    inv = np.zeros(R)
    cost = np.zeros(R) + PRICE * x0 + sum(c2 * zs[w] for w in range(W) if w % T == 0 and w > 0)
    for w in range(W):
        a = inv + arriv[:, w]
        sales = np.minimum(a, dem[:, w])
        lost = np.maximum(dem[:, w] - a, 0.0)
        inv = a - sales
        cost += H * sales + pen * lost
    cost += (PRICE - extra_res) * inv
    return cost / W


for c2 in [2800.0, 3360.0, 3920.0]:
    for T in [1, 2, 3, 4]:
        out = []
        for nsc in [1, 5, 20, 200]:
            r2 = np.random.default_rng(5)
            ps = peaks(r2, nsc)
            ceq, x0, zs = solve_SAA(ps, T, c2)
            test = peaks(np.random.default_rng(999), 4000)[:, None] * SEASON[None, :]
            out.append((nsc, x0, zs.sum(), ceq, sim(x0, zs, T, test, c2).mean()))
        print(f"c2={c2:.0f} T={T} " + " | ".join(
            f"n={o[0]:3d} x0={o[1]:5.1f} z={o[2]:6.1f} ceq={o[3]:8.1f} true={o[4]:8.1f}" for o in out))
