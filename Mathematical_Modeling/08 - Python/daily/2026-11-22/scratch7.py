# -*- coding: utf-8 -*-
"""最小可复现：8 周、1 场景、固定提前期问题的最小 LP"""
import numpy as np
from scipy.optimize import linprog

PRICE, H, PEN, RU, W = 2800.0, 2800 * 0.2 / 52, 1800.0, 1400.0, 8
SEA = np.array([.75, .75, 1.25, 1.25, 1.25, 1.25, .92, .92])
d = SEA * 120

# 变量：a_w(8), i_w(9: 期初), u_w(8)  ；目标 = c*a + p*(d-u) + (c-ru)*i_W − c*d
n_a, n_i, n_u = W, W + 1, W
nx = n_a + n_i + n_u
ia = lambda w: w
ii = lambda w: n_a + w
iu = lambda w: n_a + n_i + w

c = np.zeros(nx)
c[:W] = PRICE
for w in range(W):
    c[iu(w)] = -PEN
c[ii(W)] += PRICE - RU
const = -(PEN * d.sum()) - PRICE * d.sum()

Aeq, beq, Aub, bub = [], [], [], []
r = np.zeros(nx); r[ii(0)] = 1; Aeq.append(r); beq.append(0.0)
for w in range(W):
    # iv_w = iv_{w-1} + a_w - u_w
    r = np.zeros(nx); r[ii(w + 1)] = -1; r[ii(w)] = 1; r[ia(w)] = 1; r[iu(w)] = -1
    Aeq.append(r); beq.append(0.0)
    # u_w <= iv_{w-1} + a_w
    r = np.zeros(nx); r[iu(w)] = 1; r[ii(w)] = -1; r[ia(w)] = -1
    Aub.append(r); bub.append(0.0)

for ru_, pen_ in [(RU, PEN), (PRICE, PEN)]:
    c2 = c.copy(); c2[ii(W)] = PRICE - ru_; c2[iu(0):iu(W)] = -pen_
    const2 = -(pen_ * d.sum())
    res = linprog(c2, A_eq=np.array(Aeq), b_eq=np.array(beq),
                  A_ub=np.array(Aub), b_ub=np.array(bub),
                  bounds=[(0, None)] * n_a + [(0, 2000)] * (nx - n_a), method="highs")
    print("ru", ru_, "status", res.status, "obj", round(res.fun + const2, 0))
    print("  a", np.round(res.x[:W], 1))
    print("  i", np.round(res.x[n_a:n_a + W + 1], 1))
    print("  u", np.round(res.x[n_a + n_i:], 1))
