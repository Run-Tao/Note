# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L4 优化模型（线性规划 · 目标系数灵敏度 · 100% 法则 / Wendell 容忍度法）+ L5 算法（基复用的单纯形 / 顶点枚举鲁棒检验）
anchor: L4 优化敏感度主线，接 Day26（影子价格）、Day52（目标系数单项允许区间）、Day58（热启动重解）、
        Day73（值函数曲线）、Day77（逆优化）、Day78（100% 法则 · 只看右端项版）
goal: 云溪县农机站 2027 春耕「油补怎么调，机手还得按老计划下地」——
      ① 单个作业包的补贴价能涨/降多少，原计划才不变（Day52 的单区间）；
      ② 五个包**同时调、方向还不限**，最多能一起挪多少（Wendell 容忍度 t*，全库零命中）；
      ③ 100% 法则（把各项变动量按各自单项允许幅度归一后求和 ≤1）在目标系数方面到底是充分、必要还是都不够；
      ④ 为什么「容忍度」必须绑定一个基，基一跳它就作废。
run: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-15/day79_lp_obj_tolerance.py"
"""

import itertools
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体（Noto Sans CJK 优先，避免中文图乱码） ----------
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
RNG = np.random.default_rng(20261115)

# ============================================================
# 0. 数据：云溪县农机站 2027 春耕作业计划
#    决策变量 x[j]：第 j 类作业包的本周下地面积（百亩）
#    作业包 = (镇, 作业类型)，两镇田块条件、订单结构都不同
#    ⚠️ 老版 Day78 用的是 linprog 读无效约束集；本讲把"基"这件事当真，
#       用自己写的单纯形（见 §1），因为「容忍度」本来就是基的性质。
# ============================================================
NAMES = ["东镇·机耕", "东镇·机播", "东镇·植保", "西镇·机耕", "西镇·植保"]
# 净利润 c（元/百亩，2027 油补新政后的口径）
C = np.array([4400.0, 5900.0, 4400.0, 5700.0, 3500.0])
# 每百亩消耗：柴油(升) / 人工(工时) / 作业窗口内可用机时(小时)
USE = np.array([
    [47.70, 10.46, 1.71],   # 东镇·机耕
    [34.57,  8.79, 1.34],   # 东镇·机播
    [23.13,  3.09, 0.80],   # 东镇·植保
    [44.19,  9.86, 1.25],   # 西镇·机耕
    [21.30,  4.25, 1.08],   # 西镇·植保
])
A_USE = USE.T                                   # (3 资源 × 5 包)
BUD = np.array([3058.8, 782.5, 132.6])          # 柴油 / 人工 / 机时
RES = ["柴油", "人工", "机时"]
CAP = np.array([40.0, 35.0, 26.0, 30.0, 22.0])  # 订单上限（百亩）
LOW = np.zeros(5)                               # 不设下限：可选可不做

N_R = 3        # 资源行数
N_V = 5        # 变量数


# ============================================================
# 1. 自写单纯形（基 = 列号元组），带"基复用"判据
#    标准形：max c^T x  s.t.  A x <= b,  0 <= x <= u,  x >= 0
#    化简：x = xl + y（这里 xl = 0），y 进基；上界用"上界松弛行" y_j + t_j = u_j 处理
#    列编号：0..n-1 真实变量 / n..n+m-1 资源松弛 / n+m..n+m+n-1 上界松弛
# ============================================================
def build(c, use=A_USE, bud=BUD, cap=CAP, low=LOW):
    m = use.shape[0]
    n = len(c)
    A = np.hstack([use, np.eye(m), np.zeros((m, n))])
    U = np.hstack([np.eye(n), np.zeros((n, m)), np.eye(n)])
    M = np.vstack([A, U])
    rhs = np.concatenate([bud - use @ low, cap - low])
    obj = np.concatenate([np.asarray(c, float), np.zeros(m + n)])
    base = [n + i for i in range(m)] + [n + m + j for j in range(n)]
    return M, rhs, obj, base, n, m


def simplex(c, use=A_USE, bud=BUD, cap=CAP, low=LOW, maxiter=5000):
    """返回 (x, 最优值, 基元组, 资源行对偶值, 上界行对偶值)。"""
    M, rhs, obj, base, n, m = build(c, use, bud, cap, low)
    B = M[:, base]
    Binv = np.linalg.inv(B)
    for _ in range(maxiter):
        cB = obj[base]
        y = cB @ Binv
        red = obj - y @ M
        inb = set(base)
        # Bland 规则防循环（退化实例下面会出现多次换基）
        j = -1
        for k in range(M.shape[1]):
            if k not in inb and red[k] > 1e-9:
                j = k
                break
        if j < 0:
            break
        d = Binv @ M[:, j]
        xb = Binv @ rhs
        best, i_best = np.inf, -1
        for i in range(len(d)):
            if d[i] > 1e-12:
                r = xb[i] / d[i]
                if r < best - 1e-12:
                    best, i_best = r, i
        if i_best < 0:
            raise RuntimeError("unbounded")
        base = list(base)
        base[i_best] = j
        B = M[:, base]
        Binv = np.linalg.inv(B)
    xb = Binv @ rhs
    yv = np.zeros(n)
    for k, j in enumerate(base):
        if j < n:
            yv[j] = xb[k]
    y = c[list(base)] if False else None  # 占位，下面统一算
    cB = obj[base]
    y = cB @ Binv
    dual_res = y[:m].copy()      # 资源约束的对偶（<=0 表示紧）
    dual_up = y[m:m + n].copy()  # 上界约束的对偶
    x = yv + np.asarray(low, float)
    x = np.clip(x, 0.0, np.asarray(cap, float))
    return x, float(np.dot(c, x)), tuple(base), dual_res, dual_up


def same_base(cvec, base):
    """只比较基（列号元组）；基不变 ⟺ 解不变、影子价格不变、结论照样能用。"""
    try:
        return simplex(cvec)[2] == base
    except RuntimeError:
        return False


def tight_rows(x):
    """给读者看的"卡死结构"：哪几行资源紧、哪几个包顶到上限。"""
    s = []
    for i in range(N_R):
        if abs(A_USE[i] @ x - BUD[i]) < 1e-6:
            s.append(f"{RES[i]}紧")
    for j in range(N_V):
        if abs(x[j] - CAP[j]) < 1e-6:
            s.append(f"{NAMES[j]}顶格")
        if abs(x[j]) < 1e-6:
            s.append(f"{NAMES[j]}零")
    return s


# ============================================================
# 主流程
# ============================================================
x0, z0, BASE, DUAL_R, DUAL_U = simplex(C)
print("=" * 78)
print("云溪县农机站 2027 春耕作业计划（5 个作业包 × 3 类资源共用）")
print("=" * 78)
print("最优作业计划（百亩）：")
for j, nm in enumerate(NAMES):
    tag = "  ← 不做" if x0[j] < 1e-6 else ("  ← 顶到订单上限" if abs(x0[j] - CAP[j]) < 1e-6 else "")
    print(f"  {nm:10s} {x0[j]:7.2f}   补贴后净利润 {C[j]:6.0f} 元/百亩{tag}")
print(f"  本周净利润 z* = {z0:,.2f} 元")
res_use = A_USE @ x0
for i, nm in enumerate(RES):
    print(f"  资源 {nm}: 用 {res_use[i]:8.2f} / 可用 {BUD[i]:8.2f}   余 {BUD[i]-res_use[i]:8.2f}")
print(f"  紧凑结构：{'、'.join(tight_rows(x0))}")

# ---------- [1] 影子价格 ----------
print("\n[1] 影子价格（右端项 +1 的有限差分 vs 单纯形对偶值）")
sh_fd = []
for i in range(N_R):
    b2 = BUD.copy()
    b2[i] += 1.0
    _, z2, _, _, _ = simplex(C, bud=b2)
    sh_fd.append(z2 - z0)
print(f"  {'资源':6s} {'有限差分':>10s} {'对偶值':>10s}")
for i, nm in enumerate(RES):
    print(f"  {nm:6s} {sh_fd[i]:10.4f} {-DUAL_R[i]:10.4f}")

# ---------- [2] 单项允许区间 ----------
print("\n[2] 单项允许区间：c_j 单独上下挪，最优基不变的最大幅度")
RANGE = {}
for j in range(5):
    hal = {}
    for sg in (1.0, -1.0):
        lo, hi = 0.0, 1.0
        trial = C.copy()
        while True:
            trial[j] = C[j] + sg * hi
            if trial[j] <= 0:
                break
            if not same_base(trial, BASE):
                break
            lo = hi
            hi *= 2.0
            if hi > 1e6:
                lo = np.inf
                break
        if not np.isfinite(lo):
            hal[sg] = np.inf
            continue
        for _ in range(70):
            m = 0.5 * (lo + hi)
            trial[j] = C[j] + sg * m
            if trial[j] > 0 and same_base(trial, BASE):
                lo = m
            else:
                hi = m
        hal[sg] = lo
    RANGE[j] = (C[j] - hal[-1.0], C[j] + hal[1.0])
    up = "∞" if not np.isfinite(hal[1.0]) else f"{hal[1.0]:9.1f}"
    dn = "∞" if not np.isfinite(hal[-1.0]) else f"{hal[-1.0]:9.1f}"
    print(f"  {NAMES[j]:10s} c={C[j]:6.0f}  可降 {dn}  可涨 {up}  区间 [{RANGE[j][0]:8.1f}, {RANGE[j][1]:8.1f}]")

# 单项允许幅度的"半宽"，100% 法则要用
HALF = np.array([min(abs(C[j] - RANGE[j][0]) if np.isfinite(RANGE[j][0]) else np.inf,
                     abs(RANGE[j][1] - C[j]) if np.isfinite(RANGE[j][1]) else np.inf) for j in range(5)])


# ---------- [3] 100% 法则（目标系数版）----------
def law_sum(delta, half=None):
    """把每个 Δc_j 折成"占它自己单项允许幅度的比例"，求和。≤1 就放行。"""
    half = HALF if half is None else half
    tot = 0.0
    for j in range(5):
        if half[j] <= 0:
            return np.inf
        tot += abs(delta[j]) / half[j]
    return tot


def law_pass(delta):
    return law_sum(delta) <= 1.0 + 1e-12


print("\n[3] 100% 法则在目标系数上的实际含义")
print("  如果五项都按同一个绝对量 t 改动（都涨或都降），法则门限 t_law = 1 / Σ(1/半宽_j)")
inv = 1.0 / HALF
t_law_eq = 1.0 / inv.sum()
print(f"  t_law（等量同向）= {t_law_eq:,.2f} 元/百亩")

# ---------- [4] Wendell 容忍度 t*：盒子顶点枚举 + 二分 ----------
print("\n[4] Wendell 容忍度 t*：五项同时改、**方向不限**都保持最优基的最大 t")


def box_ok(t):
    for signs in itertools.product([-1.0, 1.0], repeat=5):
        d = np.array(signs) * t
        if np.any(C + d <= 0):
            return False
        if not same_base(C + d, BASE):
            return False
    return True


lo, hi = 0.0, 1.0
while box_ok(hi):
    lo = hi
    hi *= 2.0
for _ in range(70):
    mid = 0.5 * (lo + hi)
    if box_ok(mid):
        lo = mid
    else:
        hi = mid
T = lo
print(f"  t* = {T:,.2f} 元/百亩（五项各自上下各挪这么多，最优基都不变）")
print("  相对量：" + " / ".join(f"{100*T/C[j]:.1f}%" for j in range(5)))
print(f"  最小单项半宽 {HALF.min():,.2f}；t* / 最小半宽 = {T/HALF.min():.3f}")

# ---------- [5] 同一方向上的"真拐点" vs 法则拐点 ----------
print("\n[5] 一条最不利方向：前四种降 t、最后一个涨 t —— 法则说什么时候停，真的什么时候停")
d_dir = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])
lo2, hi2 = 0.0, T
while same_base(C + hi2 * d_dir, BASE):
    lo2 = hi2
    hi2 *= 2.0
    if hi2 > 1e6:
        break
for _ in range(70):
    m = 0.5 * (lo2 + hi2)
    if same_base(C + m * d_dir, BASE):
        lo2 = m
    else:
        hi2 = m
true_t = lo2
law_t = 1.0 / law_sum(d_dir)
print(f"  真拐点 t_true = {true_t:,.2f}     法则门限 t_law = {law_t:,.2f}     倍数 = {true_t/law_t:.2f}×")
print("  → 法则「放行」的地方真解还稳；法则「拦下」的地方真解其实也稳 —— 目标系数版的法则是充分但极不必要")

# ---------- [6] 随机抽样：法则放行 ⟹ 基不变？ ----------
print("\n[6] 3000 组随机扰动：法则放行的组里，真解有无翻车")
n_tr, passed, wrong, right = 3000, 0, 0, 0
for _ in range(n_tr):
    d = RNG.uniform(-1, 1, 5) * HALF * 1.2
    if np.any(C + d <= 0):
        continue
    if law_pass(d):
        passed += 1
        if same_base(C + d, BASE):
            right += 1
        else:
            wrong += 1
print(f"  法则放行 {passed}/{n_tr} 组；其中基不变 {right} 组，**基变了 {wrong} 组**")

# ---------- [7] 超过 t* 会怎样 ----------
print("\n[7] 越过 t*：解换了人，但 z 变化多少")
bad_dir = None
for signs in itertools.product([-1.0, 1.0], repeat=5):
    dd = np.array(signs) * T * 1.05
    if not same_base(C + dd, BASE):
        bad_dir = np.array(signs)
        break
x1, z1, base1, _, _ = simplex(C + bad_dir * T * 1.05)
print(f"  越界 1.05t* 的方向 {bad_dir.astype(int).tolist()}：z = {z1:,.2f}（原 {z0:,.2f}，差 {z1-z0:+,.2f}）")
print("  新方案：" + "、".join(f"{NAMES[j]} {x1[j]:.2f}" for j in range(5)))
print(f"  新基的容忍度已与旧基无关（基元组 {'变了' if base1 != BASE else '没变'}）")

# ---------- [8] 30 个随机 LP：t* 与单项半宽的关系 ----------
print("\n[8] 400 个随机小 LP：t* / 最小单项半宽 的分布（这是法则保守程度的普适指纹）")
ratios = []
for trial in range(400):
    rng = np.random.default_rng(3000 + trial)
    n, m = 3, 2
    use = rng.uniform(0.5, 2.0, (m, n))
    bud = rng.uniform(10, 25, m)
    cap = rng.uniform(6, 20, n)
    low = np.zeros(n)
    c = rng.uniform(2.0, 8.0, n)
    try:
        xx, zz, bb, _, _ = simplex(c, use=use, bud=bud, cap=cap, low=low)
    except RuntimeError:
        continue
    if zz < 1e-9:
        continue

    def okb(cc):
        try:
            return simplex(cc, use=use, bud=bud, cap=cap, low=low)[2] == bb
        except RuntimeError:
            return False

    hs = []
    for j in range(n):
        for sg in (1.0, -1.0):
            l2, h2 = 0.0, 0.5
            while cc_ok := okb(c + sg * h2 * np.eye(n)[j]):
                l2 = h2
                h2 *= 1.6
                if h2 > 1e6:
                    break
            for _ in range(40):
                md = 0.5 * (l2 + h2)
                if okb(c + sg * md * np.eye(n)[j]):
                    l2 = md
                else:
                    h2 = md
            hs.append(l2)
    hs = np.array(hs)
    lo3, hi3 = 0.0, float(hs.min())
    good_all = True
    while good_all:
        for sg in itertools.product([-1.0, 1.0], repeat=n):
            if not okb(c + np.array(sg, float) * hi3):
                good_all = False
                break
        if good_all:
            lo3 = hi3
            hi3 *= 1.5
            if hi3 > 1e5:
                break
    for _ in range(40):
        md = 0.5 * (lo3 + hi3)
        if all(okb(c + np.array(s, float) * md) for s in itertools.product([-1.0, 1.0], repeat=n)):
            lo3 = md
        else:
            hi3 = md
    if hs.min() > 1e-9:
        ratios.append(lo3 / hs.min())
ratios = np.array(ratios)
print(f"  样本 {len(ratios)} 个：均值 {ratios.mean():.3f}，中位 {np.median(ratios):.3f}，"
      f"最小 {ratios.min():.3f}，最大 {ratios.max():.3f}")
print(f"  t* < 最小单项半宽 的比例：{(ratios < 1).mean()*100:.1f}%  "
      "→ 「同时改」通常比「单项最紧的那一项」更严，这是法则存在的理由")
print(f"  t* > 最小单项半宽 的比例：{(ratios > 1).mean()*100:.1f}%")

# ============================================================
# 出图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
ax = axes[0, 0]
jj = np.arange(5)
lo_d = np.array([C[j] - RANGE[j][0] if np.isfinite(RANGE[j][0]) else 9999 for j in range(5)])
up_d = np.array([RANGE[j][1] - C[j] if np.isfinite(RANGE[j][1]) else 9999 for j in range(5)])
ax.bar(jj - 0.18, np.minimum(lo_d, 3000), 0.36, label="可降（截断到 3000）", color="#4C78A8")
ax.bar(jj + 0.18, np.minimum(up_d, 3000), 0.36, label="可涨（截断到 3000）", color="#F58518")
ax.axhline(T, ls="--", color="crimson", lw=1.6, label=f"Wendell t* = {T:.0f}")
ax.axhline(t_law_eq, ls=":", color="black", lw=1.4, label=f"100%法则等量门限 {t_law_eq:.0f}")
ax.set_xticks(jj)
ax.set_xticklabels(NAMES, rotation=18, fontsize=9)
ax.set_ylabel("元/百亩")
ax.set_title("(a) 单项允许幅度 vs 同时改的两个门限")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.bar(RES, [-v for v in DUAL_R], color="#54A24B")
ax.set_ylabel("元/单位")
ax.set_title("(b) 影子价格（只有一条资源在说话）")

ax = axes[1, 0]
ts = np.linspace(0, true_t * 1.6, 160)
zs = []
for t in ts:
    cc = C + t * d_dir
    try:
        zs.append(simplex(cc)[1])
    except RuntimeError:
        zs.append(np.nan)
ax.plot(ts, zs, lw=2, color="#E45756")
ax.axvline(law_t, ls=":", color="black", lw=1.4, label=f"法则停手 {law_t:.0f}")
ax.axvline(true_t, ls="--", color="crimson", lw=1.4, label=f"真拐点 {true_t:.0f}")
ax.set_xlabel("沿最不利方向改动的幅度 t（元/百亩）")
ax.set_ylabel("本周净利润（元）")
ax.set_title("(c) 法则停手的地方，曲线还是一条直线")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.hist(ratios, bins=20, color="#B279A2", edgecolor="white")
ax.axvline(1.0, ls="--", color="crimson", label="t* = 最小单项半宽")
ax.set_xlabel("t* / 最小单项半宽")
ax.set_ylabel("LP 个数")
ax.set_title(f"(d) {len(ratios)} 个随机 LP：同时改比单项更严吗")
ax.legend(fontsize=8)

fig.suptitle("Day79 · 目标系数同时改的 Wendell 容忍度法 —— 云溪县农机站油补账本", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "day79_lp_obj_tolerance.png"), dpi=130)
print(f"\n图已保存：day79_lp_obj_tolerance.png  用时 {time.time()-T0:.1f}s")

payload = {
    "x_star": [round(float(v), 4) for v in x0],
    "z_star": round(z0, 2),
    "tight": tight_rows(x0),
    "shadow_fd": [round(float(v), 4) for v in sh_fd],
    "shadow_dual": [round(float(-v), 4) for v in DUAL_R],
    "single_range": {NAMES[j]: [round(float(RANGE[j][0]), 2), round(float(RANGE[j][1]), 2)]
                     for j in range(5)},
    "half": [round(float(v), 4) for v in HALF],
    "wendell_t": round(float(T), 4),
    "t_law_equal": round(float(t_law_eq), 4),
    "worst_dir_true_t": round(float(true_t), 4),
    "worst_dir_law_t": round(float(law_t), 4),
    "law_multiple": round(float(true_t / law_t), 4),
    "law_pass": passed, "law_right": right, "law_wrong": wrong,
    "ratio_stats": {"mean": round(float(ratios.mean()), 4),
                    "median": round(float(np.median(ratios)), 4),
                    "min": round(float(ratios.min()), 4),
                    "max": round(float(ratios.max()), 4),
                    "share_lt1": round(float((ratios < 1).mean()), 4),
                    "n": int(len(ratios))},
    "beyond": {"dir": bad_dir.astype(int).tolist(), "z": round(z1, 2),
               "x": [round(float(v), 3) for v in x1]},
}
with open(os.path.join(OUT, "day79_result.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("json 已保存")
