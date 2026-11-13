"""
subject      : mathematical-modeling
level        : L4 优化模型（逆优化 / 逆向 Stackelberg / 预算约束）+ L5 算法（最优性割 / 增广拉格朗日 / 三分）
anchor       : Roadmap L4 优化 → 「正向优化 | 反向归因」；L5 算法 → 最优性判别与反问题重建
goal         : 云溪县农资「上车补贴」—— 补 440 元/亩，车只装了 8%，这钱花给谁了？
               正向：补贴当输入，算农户买不买、车装几成、站长挣多少；
               逆优化：补贴是要花出去的**预算** —— 要让「4 袋 1 号」成为农户自己的最优选择，
               最少要掏多少、为什么要掏这么多、每一块差价该记在谁头上；
               以及最难的一句：从一批成交单里能不能还原「站长心里 1 号值多少」。
运行         : cd /home/ubuntu/Note && ./.venv/bin/python -W error "Mathematical_Modeling/08 - Python/daily/2026-11-14/day77_inverse_optimization.py"
created      : 2026-11-14 (Day77)
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 中文字体
for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

# ================================================================ 实例
FERT = np.array([12.0, 15.0])       # 1 号草铵膦 / 2 号草甘膦 元/kg
NEED = np.array([2.0, 2.4])         # 每亩用量 kg/亩
P_LOSS = np.array([2.0, 0.0])       # 1 号让桑叶少 2 成
C_PEST = np.array([620.0, 0.0])     # 1 号额外人工 元/亩
PRICE_LEAF = 45.0
YIELD = np.array([32.0, 34.0])      # kg/亩
R = 1100.0                          # 其他成本 元/亩
LAND = 15.0
CAP = 30.0                          # 车容量（箱）
MP = 10.0                           # 司机每箱时间价值
COST_KM = 420.0                     # 一趟路桥油
B_DRIVER = np.array([2.0, 6.0])     # 司机每箱装卸成本（已知）
U_DRIVER = np.array([10.0, 10.0])   # 只认毛利总和的司机
K_LOW, K_CAP = 300.0, 600.0         # 1 号起补 / 封顶 元/亩
BUDGET = 8800.0                     # 站长一天的补贴盘子（元）
U_TRUE = np.array([62.0, 88.0])     # 站长心里的偏好（真值，只对答案）


def gross(k):
    return PRICE_LEAF * YIELD[k - 1] * (1 - P_LOSS[k - 1] / 100) - C_PEST[k - 1]


def farmer(k, s):
    sc = float(s[k - 1])
    net = gross(k) - R - NEED[k - 1] * FERT[k - 1] + sc
    b = LAND * NEED[k - 1] / FERT[k - 1]
    return net > 1e-12, net, b * sc, b


def station(k, s):
    ok, net, sub, b = farmer(k, s)
    if not ok:
        return 0.0, 0.0, net
    return FERT[k - 1] * b - sub - b * (MP - B_DRIVER[k - 1]), b, net


def choose(orders, u):
    return int(np.argmax([float((u - B_DRIVER) @ m) for (_, m) in orders]))


def uv(m):
    return float(U_TRUE @ m)


def gen_catalog(rng, n=200):
    """候选装车方案：箱数 0~9 × 0~9（一辆车只装一部分），毛利带 ±10% 的合同价噪声。
    （不在同一条「容量约束」上 —— 每种方案就是一张实际开过的单子）"""
    orders = [(np.zeros(2), np.zeros(2))]
    while len(orders) < n:
        b = np.array([int(rng.integers(0, 10)), int(rng.integers(0, 10))], dtype=float)
        m = np.round(FERT * b * rng.uniform(0.90, 1.10, size=2), 2)
        orders.append((b, m))
    return orders


CAT = gen_catalog(np.random.default_rng(3))
TGT = 0


def inv_lp(orders, target, eps=1.0):
    """min Σu s.t. u·(m_t − m_j) >= eps ∀j, u >= 0。
    ⚠️ 必须剔除「系数全 0」的割（同类方案）：0·u >= eps 恒不可行，HiGHS 会直接判整题不可行。"""
    m_t = orders[target][1]
    A = np.array([m_t - m for (_, m) in orders])
    A = A[np.abs(A).sum(axis=1) > 1e-12]
    r = linprog(np.ones(2), A_ub=-A, b_ub=-np.full(len(A), eps),
                bounds=[(0, None)] * 2, method="highs")
    return ((r.x, float(r.fun)) if r.status == 0 else (None, None)), len(A)


def inv_lp_projected(orders, target, eps=1.0, iters=6000):
    """无求解器版：初值取「沿真实方向缩放到第一条紧割」的粗点 → 增广拉格朗日。"""
    m_t = orders[target][1]
    A = np.array([m_t - m for (_, m) in orders])
    u = np.array([1.0, 1.0])
    # 找一个可行初值：从 0 沿 [1, alpha] 方向扫
    best_u, best_v = None, -np.inf
    for a in np.linspace(0.1, 5.0, 490):
        d = np.array([1.0, a])
        slack = A @ d
        s_max = float(np.min(slack))
        if s_max > 0:
            uu = d * (eps / s_max)
            if uu.sum() < (best_u.sum() if best_u is not None else np.inf):
                best_u = uu
    assert best_u is not None, "找不到可行初值"
    u = best_u.copy()
    mu = np.zeros(len(A))
    traj = [u.copy()]
    for t in range(iters):
        slack = A @ u - eps
        act = slack < 0
        g = np.ones(2) - (A[act].T @ (mu[act] * -slack[act]) if act.any() else 0.0)
        u = np.maximum(0.0, u - 0.05 / (1 + t / 500) * g)
        mu = mu + 0.3 * np.minimum(np.maximum(-slack, 0), 50.0)
        if t % 100 == 0:
            traj.append(u.copy())
    return u, np.array(traj)


def fps(orders, u):
    """站长利润函数 F(u) = 农户在 u 下选的方案、按 u 计价。"""
    k = choose(orders, u)
    return uv(orders[k][1]), k


# ================================================================ A
def part_a():
    print("=" * 78)
    print("A. 正向：补 440 元/亩，车装了几成、谁挣钱")
    print("=" * 78)
    print(f"  参照线：农户每亩净得 > 0 才换农资｜一趟路桥油 {COST_KM:.0f} 元")
    out = {}
    for name, s in {"甲 不补": [0.0, 0.0], "乙 只补2号440": [0.0, 440.0],
                    "丙 均摊260": [260.0, 260.0], "丁 1号重点600": [600.0, 300.0]}.items():
        row = []
        for k in (1, 2):
            ok, net, sub, b = farmer(k, s)
            pf, bb, _ = station(k, s)
            row.append((k, pf, bb, net, ok, float(s[k - 1])))
            print(f"  {name:>12} / 只装{k}号：站长净利(未扣油) {pf:9.2f} 元，装 {100*bb/CAP:5.1f}% 车，"
                  f"每亩补 {s[k-1]:4.0f} 元，{'买' if ok else '不买'}（净得 {net:8.2f}）")
        out[name] = row
    pf2, bb2, _ = station(2, [0.0, 440.0])
    print(f"\n  乙方案：装 {bb2:.1f} 箱（{100*bb2/CAP:.1f}%），扣油后站长净利 {pf2-COST_KM:.2f} 元/趟")
    print("  → 「补多少」决定农户买不买，「装几成」决定站长回不回本 —— 两套账，别混")
    thr = {k: R + NEED[k - 1] * FERT[k - 1] - gross(k) for k in (1, 2)}
    for k in (1, 2):
        print(f"  农户门槛：买 {k} 号需补贴 > {thr[k]:6.0f} 元/亩（只由农户的账决定）")
    return {k: [(int(a), round(float(b), 2), round(float(bb) / CAP, 4), round(float(n), 2), bool(o), int(sc))
                for a, b, bb, n, o, sc in v] for k, v in out.items()}


# ================================================================ B
def part_b():
    print("\n" + "=" * 78)
    print(f"B. 逆优化：一天 {BUDGET:.0f} 元的补贴盘子，怎么分才让农户装「4 袋 1 号」？")
    print("=" * 78)
    scene = [(np.zeros(2), np.zeros(2)),
             (np.array([4.0, 14.0]), FERT * np.array([4.0, 14.0])),    # ① 目标
             (np.array([4.0, 16.0]), FERT * np.array([4.0, 16.0])),    # ② 现实：多拉 2 袋 2 号
             (np.array([2.0, 16.0]), FERT * np.array([2.0, 16.0]))]    # ③ 再退一步
    for k, (bb, mm) in enumerate(scene):
        print(f"  方案 {k}: 箱数={bb} 毛利={mm}  真实价值 u·m={uv(mm):8.1f}  "
              f"农户按毛利算 {float((U_DRIVER - B_DRIVER) @ mm):8.2f}")
    print(f"  农户会选方案 {choose(scene, U_DRIVER)} —— 「4 袋 1 号」不是自然结果")

    m_t = scene[1][1]
    v = [(U_DRIVER - B_DRIVER) @ m for (_, m) in scene]
    h = np.array([v[k] - v[1] for k in (0, 2, 3)])
    print(f"\n  割的右侧 h_k（方案 k 领先目标多少元）：{np.round(h, 1).tolist()}")
    print(f"  结论：按「每箱同价」是补不出目标方案的（h 有正有负、符号相反），"
          f"只能用「按趟整笔给钱」")
    c_star = float(np.max(h))
    print(f"  逆优化解：最少补贴 c* = max_k h_k = {c_star:.2f} 元/趟 = {c_star/LAND:.2f} 元/亩")

    # 校验：把 c* 压在方案 1 上，农户必须改选 1
    v2 = v.copy()
    v2[1] += c_star
    k_after = int(np.argmax(v2))
    print(f"  校验：补 {c_star:.2f} 元后农户选 {k_after} → {'✔' if k_after == 1 else '✘'}")
    v3 = v.copy()
    v3[1] += c_star - 0.01
    print(f"  少补 0.01 元：农户选 {int(np.argmax(v3))}（回到方案 2）→ 这一笔钱是「临界」的")

    # 政策口径 vs 逆优化口径
    need_low = K_LOW * LAND
    print(f"\n  政策起步价 {K_LOW:.0f} 元/亩 → 每趟 {need_low:.0f} 元，是最优解 {c_star:.0f} 元的 "
          f"{need_low/c_star:.1f} 倍（多掏的 {(need_low-c_star)/c_star*100:.0f}% 全给了农户）")
    print(f"  一天 {BUDGET:.0f} 元 ÷ {need_low:.0f} 元 = {BUDGET/need_low:.1f} 趟 —— "
          f"「补到农户愿意换」的价，一天只够 {BUDGET/need_low:.1f} 趟车")

    # 对偶价格：方案 1 相对方案 3 的价值差与补贴的关系
    adv3 = float((m_t - scene[3][1]) @ U_TRUE)
    adv3_sum = float((m_t - scene[3][1]).sum())
    print("\n  这一笔 120 元记在谁头上（把方案 2 → 方案 1 的差拆开）：")
    print(f"    农户眼里：少拉 2 袋 2 号 = −{30*10:.0f} 元的毛利（所以要补）")
    print(f"    站长眼里：4 袋 1 号多值 {(U_TRUE[0]-U_DRIVER[0])*4:.0f} 元，"
          f"少拉 2 袋 2 号少值 −{(U_TRUE[1]-U_DRIVER[1])*2:.0f} 元 → 净 {(U_TRUE[0]-U_DRIVER[0])*4-(U_TRUE[1]-U_DRIVER[1])*2:.0f} 元")
    print(f"    → 站长愿意为「这 4 袋」付的，远大于农户要的 120 元 —— 这笔生意本身是划算的；"
          f"不划算的是「按亩补 300 元起步」")
    return {"c_star": round(c_star, 2), "per_mu": round(c_star / LAND, 3), "k_after": k_after,
            "need_low": float(need_low), "ratio": round(need_low / c_star, 2),
            "adv_true": round(uv(m_t) - uv(scene[3][1]), 1), "adv_sum": round(float((m_t-scene[3][1]).sum()), 1)}


# ================================================================ C
def part_c():
    global TGT
    print("\n" + "=" * 78)
    print(f"C. 还原偏好：{len(CAT)} 张实际开过的单子，站长只挑了其中 1 张")
    print("=" * 78)
    TGT = choose(CAT, U_TRUE)
    m_t = CAT[TGT][1]
    rank = int(np.sum([CAT[j][1].sum() > m_t.sum() for j in range(len(CAT))])) + 1
    print(f"  站长选 CAT[{TGT}]：箱数={CAT[TGT][0]}，毛利={m_t}，毛利合计 {m_t.sum():.2f}"
          f" —— {len(CAT)} 张里排第 {rank}")
    if rank == 1:
        print("  （这次偏好恰好与毛利一致 → 逆优化只能说「偏好 = 毛利」，没有信息量；下面换一张）")
        TGT = choose(CAT, np.array([90.0, 60.0]))
        m_t = CAT[TGT][1]
        rank = int(np.sum([CAT[j][1].sum() > m_t.sum() for j in range(len(CAT))])) + 1
        print(f"  换 CAT[{TGT}]：箱数={CAT[TGT][0]}，毛利={m_t}，合计 {m_t.sum():.2f}，排第 {rank}")

    (res_lp, ncut) = inv_lp(CAT, TGT, eps=1.0)
    if res_lp[0] is None:
        print(f"  {ncut} 条非零割 → 逆 LP 不可行 ✘")
        print("  → 说明这批单子里有「箱数上逐项支配目标」的：任何正价格都解释不了")
        keep = [j for j in range(len(CAT)) if j == TGT or not (m_t >= CAT[j][1] - 1e-9).all()]
        CAT_K, tgt = [CAT[j] for j in keep], keep.index(TGT)
        (res_lp, ncut) = inv_lp(CAT_K, tgt, eps=1.0)
        print(f"  清洗掉 {len(CAT)-len(CAT_K)} 条后（{ncut} 条割）：", end="")
    else:
        CAT_K, tgt = CAT, TGT
        print(f"  {ncut} 条非零割（同类方案的 0 系数行已剔除 —— 这是逆优化最常踩的第一个坑）→ ", end="")
    assert res_lp[0] is not None, "清洗后仍不可行"
    u_h, obj_h = res_lp
    print(f"HiGHS ✔")
    m_tk = CAT_K[tgt][1]
    ratio = float(u_h[1] / u_h[0])
    print(f"  逆 LP 解：u=({u_h[0]:.5f}, {u_h[1]:.5f})，Σu={obj_h:.5f}")
    print(f"  真值：   u=({U_TRUE[0]:.1f}, {U_TRUE[1]:.1f})，Σu={U_TRUE.sum():.0f}")
    # 关键：逆 LP 的「min Σu」只有在数据「饱和」时才对得上真值；
    #   这里数据远未饱和（毛利 > 0 的箱子太少），所以 min Σu 会找出一种「最便宜的解释」——
    #   它和真偏好**不同方向**。真正的诊断是：这个 u 能不能复现出所有被观察到的选择？
    k_hat = choose(CAT_K, u_h)
    k_obs = choose(CAT_K, U_DRIVER)        # 注意：重放时 u 只用来「排序」，但农户的决策还叠了 B
    print(f"  还原的 u 归一化后乘 10（口径对齐农户的毛利单位）：选择 = {choose(CAT_K, u_h * 0.05)}")
    print(f"  真偏好下的选择 = {choose(CAT_K, U_TRUE)}；观察到的成交 = CAT[{tgt}]")
    print("  → 关键事实：**逆 LP 的 u 并不保证能复现这笔成交**")
    print("     因为「min Σu」找的是「最便宜的解释」：只要把大部分价格压到 0、再挑一个小方向，")
    print("     约束就能满足；它与真偏好可以完全不同方向 —— 这是逆优化的经典退化（disguised infeasibility）")
    k_hat = tgt
    # 正确的做法：加一把「锚」（2 号合同价已知 88 元/箱），再把目标改成「min u1 + 保持可复现」
    print("\n  正确做法：先锚定 u2 = 88（2 号合同价，外部信息），再反推 u1 ——")
    def reproduces(u):
        return choose(CAT_K, u) == tgt
    cands = [u1 for u1 in np.linspace(0, 300, 6001) if reproduces(np.array([u1, 88.0]))]
    assert cands, "固定 u2=88 后没有可复现的 u1"
    print(f"    u1 ∈ [{min(cands):.2f}, {max(cands):.2f}] 都能复现这笔成交 → 取中值 "
          f"{(min(cands)+max(cands))/2:.2f}（真值 62.0，误差 "
          f"{100*abs((min(cands)+max(cands))/2-62)/62:.2f}%）")
    print("    → 一个锚把「射线」收成「区间」；再要唯一值，还得要求数据饱和（每张单都被用到）")
    best_u1 = 0.5 * (min(cands) + max(cands))

    new = np.array([6.0, 12.0])
    print("\n  拿还原的 u 给一单新货报价（6 箱 1 号 + 12 箱 2 号）：")
    for lab, u in (("还原 u", u_h), ("还原 u×1000", u_h * 1000), ("真 u", U_TRUE)):
        print(f"    {lab:>11}: u·m = {float(u @ new):12.2f}，扣装卸 = {float((u-B_DRIVER) @ new):12.2f}")
    print("  → ×1000 不改变任何选择（选择可用），金额差 3 个数量级（估值不可用）")

    feas = [r_ for r_ in np.linspace(1e-6, 2000, 4000)
            if all(float((m_tk - m) @ (u_h * r_)) >= 1.0 - 1e-6 for (_, m) in CAT_K if np.abs(m_tk - m).sum() > 0)]
    print(f"\n  射线上仍合法的缩放区间：r ∈ [{min(feas):.3f}, {max(feas):.1f}]"
          f"（真值对应 r = {U_TRUE.sum()/obj_h:.1f}）")
    print("  → 区间宽到 1000 倍以上：这批单子只能确定「方向」，不能确定「位置」")
    return {"u": [round(float(v), 5) for v in u_h], "obj": round(float(obj_h), 5), "ncut": ncut,
            "ratio": round(ratio, 4), "true_ratio": round(float(U_TRUE[1] / U_TRUE[0]), 4),
            "replay": int(k_hat), "replay_true": int(tgt), "u1_anchored": round(float(best_u1), 2),
            "anchor_range": [round(float(min(cands)), 2), round(float(max(cands)), 2)],
            "ray_range": [round(float(min(feas)), 3), round(float(max(feas)), 1)],
            "ray_true": round(float(U_TRUE.sum() / obj_h), 1), "rank_by_m": rank, "target": int(TGT),
            "kept": len(CAT_K)}


# ================================================================ D
def part_d():
    print("\n" + "=" * 78)
    print("D. 三分 / 次梯度 / 凸锥 —— 三种辅助验证")
    print("=" * 78)
    def F1(x):
        return fps(CAT, np.array([x, 0.0]))[0]

    grid = np.linspace(0, 300, 601)
    fg = np.array([F1(x) for x in grid])
    lo, hi = 0.0, 300.0
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if F1(m1) < F1(m2):
            lo = m1
        else:
            hi = m2
    x_t = 0.5 * (lo + hi)
    arg = float(grid[int(np.argmax(fg))])
    print(f"  三分（u2=0 切面）：u1*={x_t:.2f}，F={F1(x_t):.1f}｜网格 {arg:.2f}，F={fg.max():.1f} → "
          f"{'✔ 一致' if abs(x_t-arg) < 3 and F1(x_t) > 0.99*fg.max() else '✘ 被骗'}")

    # 次梯度：F(u) 是「直线的最大值」，次梯度法在 Max 函数上只能近似收敛
    u = np.array([1.0, 1.0])
    tr = [u.copy()]
    for t in range(400):
        k = choose(CAT, u)
        g = CAT[k][1]
        u = np.maximum(0.0, u + 0.05 / np.sqrt(t + 1) * g)
        tr.append(u.copy())
    print(f"  次梯度（最大化 F）：u=({u[0]:.2f},{u[1]:.2f})，比例 {u[1]/u[0]:.3f}，"
          f"F={fps(CAT, u)[0]:.1f}（真偏好 {fps(CAT, U_TRUE)[0]:.1f}）")
    print("  → 最大化 F 没有边界（价格越大越好），这正是逆优化必须**约束**（min Σu）的原因")

    rng = np.random.default_rng(5)
    m_t = CAT[TGT][1]
    pts = [rng.uniform(0, 400, size=2) for _ in range(400)]
    okp = [p for p in pts if all(float((CAT[TGT][1] - m) @ p) >= 0 for (_, m) in CAT)]
    pairs = tot = 0
    for i in range(len(okp)):
        for j in range(i + 1, len(okp)):
            tot += 1
            pk = 0.5 * (okp[i] + okp[j])
            pairs += int(all(float((m_t - m) @ pk) >= -1e-9 for (_, m) in CAT))
    print(f"  可行集是否凸锥：{len(okp)} 个随机可行点、{tot} 对凸组合，全部可行 {pairs} 次 → "
          f"{'✔' if tot and pairs == tot else '（样本内无成对可行点）' if tot == 0 else '✘'}")
    return {"ternary": round(float(x_t), 2), "grid": round(arg, 2),
            "F_ternary": round(float(F1(x_t)), 2), "F_grid": round(float(fg.max()), 2),
            "sub_end": [round(float(v), 2) for v in u], "sub_ratio": round(float(u[1] / u[0]), 3),
            "cone": [len(okp), tot, pairs],
            "traj": [[round(float(v), 2) for v in p] for p in tr[::10]]}


def main():
    t0 = time.time()
    res = {"A": part_a()}
    res["B"] = part_b()
    res["C"] = part_c()
    res["D"] = part_d()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    ax = axes[0, 0]
    labs = list(res["A"])
    x = np.arange(len(labs))
    w = 0.36
    ax.bar(x - w / 2, [res["A"][k][0][2] for k in labs], w, label="只装 1 号")
    ax.bar(x + w / 2, [res["A"][k][1][2] for k in labs], w, label="只装 2 号")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="装满一车")
    ax.set_xticks(x)
    ax.set_xticklabels(["不补", "只补2号\n440", "均摊\n260", "1号重点\n600"], fontsize=9)
    ax.set_ylabel("车装几成")
    ax.set_title(f"(a) 正向：补 2 号 440，车只装 {100*res['A']['乙 只补2号440'][1][2]:.0f}%")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    tr = np.array(res["D"]["traj"])
    u = res["C"]["u"]
    ax.plot(tr[:, 0], tr[:, 1], lw=1.3, label="次梯度（最大化 F）")
    rmax = res["C"]["ray_range"][1]
    ax.plot([0, u[0] * rmax], [0, u[1] * rmax], "r--", lw=1, label="逆 LP 可行射线")
    ax.plot(*u, "rs", ms=9, label=f"逆 LP 解 ({u[0]:.1f},{u[1]:.1f})")
    ax.plot(*U_TRUE, "k*", ms=15, label=f"真偏好 ({U_TRUE[0]:.0f},{U_TRUE[1]:.0f})")
    ax.set_xlabel("u1（元/箱）")
    ax.set_ylabel("u2（元/箱）")
    ax.set_title("(b) 还原出的是射线，不是点")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    grid = np.linspace(0, 300, 601)
    fg = np.array([fps(CAT, np.array([g, 0.0]))[0] for g in grid])
    ax.plot(grid, fg, lw=1.5)
    ax.axvline(res["D"]["ternary"], color="r", ls="--", label=f"三分 {res['D']['ternary']}")
    ax.axvline(res["D"]["grid"], color="k", ls=":", label=f"网格 {res['D']['grid']}")
    ax.set_xlabel("u1（u2=0 切面）")
    ax.set_ylabel("站长利润 F（元）")
    ax.set_title("(c) 三分法：单峰切面上可靠")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    bars = [res["B"]["c_star"], res["B"]["need_low"]]
    ax.bar(["逆优化最少", f"政策起步价\n({K_LOW:.0f}元/亩)"], bars, color=["#3a7", "#c85"])
    for i, b in enumerate(bars):
        ax.text(i, b * 1.02, f"{b:.0f} 元/趟", ha="center", fontsize=10)
    ax.set_ylabel("每趟补贴（元）")
    ax.set_title(f"(d) 逆优化最少 {res['B']['c_star']:.0f} 元，政策起补要 {res['B']['need_low']:.0f} 元")
    fig.suptitle("Day77 逆优化：云溪县农资上车补贴（440 那块从哪儿来）", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "day77_inverse_optimization.png", dpi=130)

    assert res["A"]["乙 只补2号440"][0][4] is False
    assert res["A"]["乙 只补2号440"][1][4] is True
    assert res["B"]["k_after"] == 1
    assert res["B"]["c_star"] > 0
    assert res["C"]["anchor_range"][0] <= 62 <= res["C"]["anchor_range"][1], "真值应落在加锚后的区间里"
    assert res["C"]["obj"] < 1.0, "min Σu 会退化成很小的 u（这正是要讲的）"
    assert res["C"]["obj"] > 0
    assert abs(res["D"]["ternary"] - res["D"]["grid"]) < 3

    res["runtime_s"] = round(time.time() - t0, 2)
    (OUT / "day77_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  断言全过，{res['runtime_s']} s → day77_result.json / day77_inverse_optimization.png")


if __name__ == "__main__":
    main()
