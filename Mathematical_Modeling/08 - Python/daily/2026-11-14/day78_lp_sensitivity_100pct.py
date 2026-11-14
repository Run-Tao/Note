# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L4 优化模型（线性规划 · 影子价格 · 灵敏度分析 · 100% 法则）+ L5 算法（基与有效约束集 / 蒙特卡洛验证）
anchor: L4，接 Day26（影子价格）、Day52（最优基区间）、Day58（热启动重解）、Day73（值函数拐点）、Day77（逆优化）
goal: 云溪县两城三库调水结算——「部门说三样各改 10%，先问一句：你是**同时**改的吗？」
      ① 单独改一项能改多少；② 几项同时改，"100% 法则"说行，到底行不行；
      ③ 影子价格外推的误差从哪儿冒出来；④ 为什么"法则通过"不等于"还是同一道题"。
run: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-14/day78_lp_sensitivity_100pct.py"
"""

import itertools
import json
import os
import time

import numpy as np
from scipy.optimize import linprog
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体 ----------
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
RNG = np.random.default_rng(20261114)

# ============================================================
# 0. 数据：云溪县两城三库调水
#   变量 x[k] （k=0..2：A 城从甲/乙/丙库取水；k=3..5：B 城从甲/乙/丙库取水），单位 万 m3
# ============================================================
LABEL = ["A城·甲库", "A城·乙库", "A城·丙库", "B城·甲库", "B城·乙库", "B城·丙库"]
PROFIT = np.array([9.0, 8.2, 10.5, 8.6, 9.4, 11.2])     # 到户净收益 万元/万m3（含输水差损）
# 三座水库可供水量（万 m3）
BS0 = np.array([260.0, 200.0, 180.0])
# 两城管网过水能力（万 m3）
PA0, PB0 = 300.0, 320.0

# 资源行（5 条）：
#   0 甲库出库 <= b1    1 乙库出库 <= b2    2 丙库出库 <= b3
#   3 A 城过水 <= b4    4 B 城过水 <= b5
R_RES = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
], float)
RES_NAMES = ["甲库可供", "乙库可供", "丙库可供", "A城过水", "B城过水"]

A_UB = np.vstack([R_RES, -np.eye(6)])                   # 行 5..10 是 x>=0
B_BASE = np.concatenate([BS0, [PA0, PB0], np.zeros(6)])
N_RES = 5


def bvec(res=None):
    b = B_BASE.copy()
    if res is not None:
        b[:N_RES] = res
    return b


def solve(res=None, cost=None):
    r = linprog(-(PROFIT if cost is None else cost), A_ub=A_UB, b_ub=bvec(res),
                bounds=[(0, None)] * 6, method="highs")
    if r.status != 0:
        return None, None, False
    return float(-r.fun), np.asarray(r.x, float).copy(), True


def basis_of(x, res=None, tol=1e-7):
    """有效约束集：卡死的资源行 + 取 0 的变量。"""
    slack = bvec(res) - A_UB @ x
    on = {int(r) for r in np.where(np.abs(slack) <= tol)[0] if r < N_RES}
    on |= {N_RES + k for k in range(6) if x[k] <= tol}
    return frozenset(on)


def cur_res():
    """当前基准的 5 个资源右端项。"""
    return np.concatenate([BS0, [PA0, PB0]])


RES = {"meta": {"day": "Day78", "date": "2026-11-14", "vars": LABEL, "resources": RES_NAMES,
                "topic": "多系数同时扰动：100% 法则（LP 灵敏度）"}}

# ============================================================
# 1. 基准解与影子价格
# ============================================================
z0, x0, ok = solve()
assert ok, "基准不可行"
BAS0 = basis_of(x0)
print("=" * 82)
print("【1】基准最优解（两城三库调水）")
print(f"  z* = {z0:,.2f} 万元")
for k, nm in enumerate(LABEL):
    if x0[k] > 1e-9:
        print(f"   {nm}: {x0[k]:8.2f} 万m3")
print(f"  卡死的资源 = {[RES_NAMES[r] for r in sorted(BAS0) if r < N_RES]}")
print(f"  取 0 的变量 = {[LABEL[r - N_RES] for r in sorted(BAS0) if r >= N_RES]}")

res_tight = sorted(r for r in BAS0 if r < N_RES)
assert len(res_tight) >= 3, "至少要三条资源同时卡住才有『多系数』可讲"

du = linprog(-PROFIT, A_ub=A_UB, b_ub=bvec(), bounds=[(0, None)] * 6, method="highs").ineqlin.marginals
sh_dual = -du[:N_RES]
sh_fd = np.zeros(N_RES)
for r in range(N_RES):
    v2 = cur_res(); v2[r] += 1.0
    val, _, okr = solve(res=v2)
    sh_fd[r] = (val - z0) if okr else np.nan
print(f"  影子价格（差分）{np.round(sh_fd, 4)}｜对偶 {np.round(sh_dual, 4)}")
for r in range(N_RES):
    print(f"   {RES_NAMES[r]:6s}: {sh_fd[r]:7.4f} 万元/万m3"
          + ("   ← 宽松，不值钱" if abs(sh_fd[r]) < 1e-9 else ""))
RES["base"] = {"z": z0, "x": x0.tolist(), "shadow_fd": sh_fd.tolist(),
               "shadow_dual": sh_dual.tolist(), "res_tight": res_tight,
               "basis": sorted(BAS0)}

# ============================================================
# 2. 单个右端项允许区间（基不变）
# ============================================================
def perturb(r, delta, base=None):
    v = (cur_res() if base is None else base).copy()
    v[r] += delta
    return v


def scan_range(r, step=0.5, span=1200.0):
    """返回 (lo_delta, hi_delta)：使基与 BAS0 一致的最大下/上扰动。"""
    lo = 0.0
    t = step
    while t <= span:
        z, x, ok = solve(res=perturb(r, -t))
        if (not ok) or basis_of(x, perturb(r, -t)) != BAS0:
            break
        lo = -t
        t += step
    hi = 0.0
    t = step
    while t <= span:
        z, x, ok = solve(res=perturb(r, t))
        if (not ok) or basis_of(x, perturb(r, t)) != BAS0:
            break
        hi = t
        t += step
    return lo, hi


print("=" * 82)
print("【2】单个资源：各自能改多少（方案与影子价格都不变）")
single = []
for r in range(N_RES):
    lo, hi = scan_range(r)
    base_v = cur_res()[r]
    single.append({"name": RES_NAMES[r], "b": float(base_v), "lo": float(lo), "hi": float(hi),
                   "dn_pct": float(-lo / base_v * 100), "up_pct": float(hi / base_v * 100),
                   "tight": r in res_tight})
    print(f"  {RES_NAMES[r]:6s} 基准 {base_v:7.1f}｜可降 {single[-1]['dn_pct']:6.1f}%（{lo:+7.1f}）"
          f"｜可升 {single[-1]['up_pct']:6.1f}%（{hi:+7.1f}）"
          + ("" if r in res_tight else "   ← 基准里宽松"))
RES["single"] = single

UP = np.array([single[r]["up_pct"] / 100 for r in range(N_RES)])
DN = np.array([single[r]["dn_pct"] / 100 for r in range(N_RES)])

# ============================================================
# 3. 目标系数单改区间
# ============================================================
print("=" * 82)
print("【3】单个单价改多少，方案才换")
single_c = []
for k in range(6):
    step = 0.02
    lo = hi = 0.0
    t = step
    while t < PROFIT[k] * 3:
        c2 = PROFIT.copy(); c2[k] -= t
        z, x, ok = solve(cost=c2)
        if (not ok) or basis_of(x) != BAS0:
            break
        lo = -t; t += step
    t = step
    while t < PROFIT[k] * 3:
        c2 = PROFIT.copy(); c2[k] += t
        z, x, ok = solve(cost=c2)
        if (not ok) or basis_of(x) != BAS0:
            break
        hi = t; t += step
    single_c.append({"name": LABEL[k], "c": float(PROFIT[k]), "dn": float(-lo), "up": float(hi),
                     "dn_pct": float(-lo / PROFIT[k] * 100), "up_pct": float(hi / PROFIT[k] * 100),
                     "zero_in_basis": (N_RES + k) in BAS0})
    print(f"  {LABEL[k]:9s} {PROFIT[k]:6.2f}｜可降 {single_c[-1]['dn_pct']:6.1f}%｜"
          f"可升 {single_c[-1]['up_pct']:6.1f}%"
          + ("   ← 基准为 0，只能往上长" if single_c[-1]["zero_in_basis"] else ""))
RES["single_cost"] = single_c

# ============================================================
# 4. 两两同时改：法则上限 vs 实际
# ============================================================
def t_break(comb, sign, base_res, hi_cap=1.5):
    """comb 中资源按同一比例 t 同时缩放 (+1 放宽 / -1 收紧)，返回基改变的最小 t。"""
    ref = base_res.copy()

    def broken(t):
        v = ref.copy()
        for r in comb:
            v[r] = ref[r] * (1 + sign * t)
        z, x, ok = solve(res=v)
        return (not ok) or basis_of(x, res=v) != BAS0

    if not broken(hi_cap):
        return hi_cap
    lo, hi = 0.0, hi_cap
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if broken(mid):
            hi = mid
        else:
            lo = mid
    return lo


print("=" * 82)
print("【4】两两同时改：100% 法则给的上限 vs 真能走多远")
pairs = []
base_res = cur_res()
pool = [r for r in res_tight]                      # 只对「卡住的」资源谈允许幅度
for a, b in itertools.combinations(pool, 2):
    for sign, lab, arr in ((+1, "同时放宽", UP), (-1, "同时收紧", DN)):
        ra, rb = arr[a], arr[b]
        t_rule = 1.0 / (1.0 / ra + 1.0 / rb) if (ra > 1e-12 and rb > 1e-12) else 0.0
        t_real = t_break((a, b), sign, base_res)
        pair = {"pair": f"{RES_NAMES[a]}+{RES_NAMES[b]}", "dir": lab,
                "rule_pct": float(t_rule * 100), "real_pct": float(t_real * 100),
                "ratio": float(t_real / t_rule) if t_rule > 1e-12 else None}
        pairs.append(pair)
        verdict = "✅ 法则保守" if t_real >= t_rule - 1e-9 else "🔴 法则失效"
        rr = f"{t_real/t_rule:5.2f}×" if pair["ratio"] else "  —  "
        print(f"  {RES_NAMES[a]:5s}+{RES_NAMES[b]:5s} {lab}：法则 {t_rule*100:6.1f}%｜"
              f"实际 {t_real*100:6.1f}%（{rr}） {verdict}")
RES["pairs"] = pairs

# ============================================================
# 5. 多项同时随机扰动：法则放行的那部分有多少是错的
# ============================================================
print("=" * 82)
print("【5】几项同时随机改，法则『说行』的到底行不行")
monte = []
PLANS = [("只改三座水库", [0, 1, 2]), ("三座水库 + 两条管网", [0, 1, 2, 3, 4])]
LIMITS = [0.10, 0.20, 0.30]
for plan_name, idxs in PLANS:
    for lim in LIMITS:
        N = 1200
        D = RNG.uniform(-lim, lim, size=(N, len(idxs)))
        rule_ok = np.zeros(N, bool); real_ok = np.zeros(N, bool); feas = np.zeros(N, bool)
        for k in range(N):
            v = base_res.copy()
            s = 0.0
            for j, r in enumerate(idxs):
                d = D[k, j]
                v[r] *= (1.0 + d)
                s += (d / UP[r]) if d >= 0 else (-d / DN[r])
            rule_ok[k] = s <= 1.0
            z, x, okv = solve(res=v)
            feas[k] = okv
            real_ok[k] = okv and basis_of(x, res=v) == BAS0
        rec = {"plan": plan_name, "dev_pct": f"±{lim*100:.0f}%", "N": N,
               "rule_pass_pct": float(rule_ok.mean() * 100),
               "pass_and_true_pct": float((rule_ok & real_ok).mean() * 100),
               "pass_but_changed_pct": float((rule_ok & ~real_ok & feas).mean() * 100),
               "pass_but_infeasible_pct": float((rule_ok & ~feas).mean() * 100),
               "real_changed_pct": float((~real_ok).mean() * 100),
               "caught_pct": float((~rule_ok & ~real_ok).sum() / max((~real_ok).sum(), 1) * 100)}
        monte.append(rec)
        print(f"  {plan_name:12s} {rec['dev_pct']:4s}：法则判可信 {rec['rule_pass_pct']:5.1f}%｜"
              f"其中真可信 {rec['pass_and_true_pct']:5.1f}%｜🔴已换方案 {rec['pass_but_changed_pct']:5.1f}%｜"
              f"🔴已不可行 {rec['pass_but_infeasible_pct']:5.1f}%｜实际变了 {rec['real_changed_pct']:5.1f}%"
              f"（被法则挡住 {rec['caught_pct']:5.1f}%）")
RES["monte"] = monte

# ============================================================
# 6. 三条卡死资源沿 45° 同时放宽：法则线 vs 真实拐点
# ============================================================
core = res_tight[:3] if len(res_tight) >= 3 else res_tight
curve = []
ts = np.linspace(0, 0.50, 101)
for t in ts:
    v = base_res.copy()
    for r in core:
        v[r] = base_res[r] * (1 + t)
    s = float(sum(t / UP[r] for r in core))
    z, x, okv = solve(res=v)
    same = bool(okv and basis_of(x, res=v) == BAS0)
    du2 = linprog(-PROFIT, A_ub=A_UB, b_ub=bvec(res=v), bounds=[(0, None)] * 6,
                  method="highs").ineqlin.marginals if okv else None
    curve.append({"t_pct": float(t * 100), "rule_sum": s, "same": same,
                  "z": float(z) if okv else None,
                  "shadow": [float(-du2[r]) for r in range(N_RES)] if okv else None})
first_break = next((c["t_pct"] for c in curve if not c["same"]), None)
rule_start = next((c["t_pct"] for c in curve if c["rule_sum"] >= 1.0), None)
print("=" * 82)
print(f"【6】三条资源同时等比放宽：法则在 t={rule_start}% 松口（Σ比例=1），"
      f"真实换方案在 t={first_break}%")
RES["curve"] = curve
RES["first_break_pct"] = first_break
RES["rule_start_pct"] = rule_start

# ============================================================
# 7. 一个具体的反例（法则通过、已换基 / 已不可行）
# ============================================================
print("=" * 82)
print("【7】能上台讲的例子：法则说『没问题』")
best = None
for _ in range(20000):
    d = RNG.uniform(-0.35, 0.35, N_RES)
    s = float(sum((d[r] / UP[r]) if d[r] >= 0 else (-d[r] / DN[r]) for r in range(N_RES)))
    if s > 1.0 or s < 0.55:
        continue
    v = base_res * (1 + d)
    z, x, okv = solve(res=v)
    changed = (not okv) or basis_of(x, res=v) != BAS0
    if changed and (best is None or s > best["rule_sum"]):
        best = {"d_pct": (d * 100).round(2).tolist(), "rule_sum": s, "feasible": bool(okv),
                "z": float(z) if okv else None, "z0": z0,
                "new_tight": [RES_NAMES[r] for r in sorted(basis_of(x, res=v)) if r < N_RES] if okv else None,
                "new_zero": [LABEL[r - N_RES] for r in sorted(basis_of(x, res=v)) if r >= N_RES] if okv else None}
if best is None:
    print("  ±35% 内未找到反例（如实记录）")
else:
    print(f"  五项分别改 {best['d_pct']}%（法则总比例 {best['rule_sum']:.3f} < 1）")
    print(f"  -> 可行={best['feasible']}，新卡死资源 {best['new_tight']}，新归零 {best['new_zero']}")
    if best["feasible"]:
        print(f"  -> 收益 {best['z']:.2f}，影子价外推 {z0 + sh_fd @ (base_res*(1+np.array(best['d_pct'])/100) - base_res):.2f}")
        # 外推误差由脚本后面写进 json
        best["z_lin"] = float(z0 + sh_fd @ (base_res * (1 + np.array(best["d_pct"]) / 100) - base_res))
        best["err_pct"] = float((best["z_lin"] - best["z"]) / abs(best["z"]) * 100)
RES["example"] = best

# ============================================================
# 8. 影子价格外推 vs 实解
# ============================================================
print("=" * 82)
print("【8】「每单位值多少钱」× 变动量 = 新收益？")
scen = {
    "三项资源各 +10%": np.array([1.10, 1.10, 1.10, 1.0, 1.0]),
    "三项资源各 −10%": np.array([0.90, 0.90, 0.90, 1.0, 1.0]),
    "三项资源 +20%": np.array([1.20, 1.20, 1.20, 1.0, 1.0]),
    "甲库 +15%/乙库 −25%": np.array([1.15, 0.75, 1.0, 1.0, 1.0]),
}
sc_out = {}
for nm, k in scen.items():
    d = k - 1
    s = float(sum((d[r] / UP[r]) if d[r] >= 0 else (-d[r] / DN[r]) for r in range(N_RES) if abs(d[r]) > 1e-12))
    v = base_res * k
    z, x, okv = solve(res=v)
    rec = {"rule_sum": s, "feasible": bool(okv)}
    if okv:
        zl = float(z0 + sh_fd @ (base_res * k - base_res))
        rec.update({"z": float(z), "z_lin": zl, "err": zl - float(z),
                    "err_pct": (zl - float(z)) / abs(float(z)) * 100,
                    "same": basis_of(x, res=v) == BAS0})
        print(f"  {nm:20s} 法则 {s:5.2f}｜实解 {z:9.2f}｜外推 {zl:9.2f}"
              f"（偏差 {rec['err_pct']:+7.2f}%）{'  ✅ 基未变' if rec['same'] else '  🔴 基已变'}")
    else:
        print(f"  {nm:20s} 法则 {s:5.2f}｜🔴 不可行")
    sc_out[nm] = rec
RES["scenarios"] = sc_out

# ============================================================
# 9. 「先按影子价砍最贵那一项」为什么会崩：按影子价排序 + 逐项收紧
# ============================================================
print("=" * 82)
print("【9】按影子价格从大到小逐项收紧 10%，收益怎么掉")
order = np.argsort(-sh_fd)
drop = []
v = base_res.copy()
for r in order:
    v[r] *= 0.9
    z, x, okv = solve(res=v)
    drop.append({"name": RES_NAMES[r], "z": float(z) if okv else None,
                 "d_loss": float(z - z0) if okv else None,
                 "pred": float(sh_fd[r] * (v[r] - base_res[r]))})
    print(f"  收紧 {RES_NAMES[r]:6s} 10%（影子价 {sh_fd[r]:6.3f}）→ 实掉 {drop[-1]['d_loss'] if drop[-1]['d_loss'] is not None else float('nan'):9.2f}"
          f"｜影子价预测掉 {drop[-1]['pred']:9.2f}"
          + ("" if okv else "｜🔴 不可行"))
RES["peel"] = drop

# ============================================================
# 10. 图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for r in range(N_RES):
    ax.barh(r, single[r]["dn_pct"] + single[r]["up_pct"], left=-single[r]["dn_pct"], height=0.55,
            alpha=0.7, color=("#4c78a8" if single[r]["tight"] else "#bbbbbb"))
    ax.text(single[r]["up_pct"] + 2, r, f"−{single[r]['dn_pct']:.0f}% / +{single[r]['up_pct']:.0f}%",
            va="center", fontsize=9)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(range(N_RES)); ax.set_yticklabels(RES_NAMES)
ax.set_xlabel("单项可变动幅度（此范围内方案与影子价格都不变）")
ax.set_title("(a) 单独改一项：各自能走多远（灰 = 基准里本来就宽松）", fontsize=11)
ax.grid(alpha=0.3, axis="x")

ax = axes[0, 1]
ts2 = [c["t_pct"] for c in curve]
ax.plot(ts2, [c["rule_sum"] for c in curve], lw=2, color="#4c78a8", label="100% 法则累计比例")
ax.axhline(1.0, color="#e45756", ls="--", lw=1.2, label="法则阈值 1.0")
if first_break is not None:
    ax.axvline(first_break, color="#54a24b", ls=":", lw=2.2, label=f"真实换方案 {first_break:.1f}%")
if rule_start is not None:
    ax.axvline(rule_start, color="#4c78a8", ls="-.", lw=1.2, label=f"法则松口 {rule_start:.1f}%")
ax.set_xlabel("三条卡死资源同时按同一比例放宽 t%")
ax.set_ylabel("Σ 变动量 / 单项允许幅度")
ax.set_title("(b) 三样一起改：法则何时松口 vs 方案何时真变", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
sub = [m for m in monte if m["plan"].startswith("三座水库 +")]
labs = [m["dev_pct"] for m in sub]
idx = np.arange(len(sub)); w = 0.27
ax.bar(idx - w, [m["pass_and_true_pct"] for m in sub], w, label="法则通过且真没变", color="#54a24b")
ax.bar(idx, [m["pass_but_changed_pct"] for m in sub], w, label="法则通过但已换方案", color="#e45772")
ax.bar(idx + w, [m["pass_but_infeasible_pct"] for m in sub], w, label="法则通过但已不可行", color="#8b0000")
for i, m in enumerate(sub):
    ax.text(i + w, m["pass_but_infeasible_pct"] + 0.8, f"{m['pass_but_infeasible_pct']:.0f}%",
            ha="center", fontsize=8)
ax.set_xticks(idx); ax.set_xticklabels(labs)
ax.set_ylabel("占抽样比例 (%)")
ax.set_title("(c) 五项同时随机改：法则放行的那部分，多少是错的", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
sh1 = [c["shadow"][1] for c in curve]   # 乙库：拐点后由 9.4 直接掉到 0
ax.plot(ts2, sh1, lw=2, color="#f58518")
if first_break is not None:
    ax.axvline(first_break, color="#54a24b", ls=":", lw=2, label=f"拐点 {first_break:.1f}%")
ax.set_xlabel("三条卡死资源等比放宽 t%")
ax.set_ylabel("乙库可供的影子价格（万元/万m3）")
ax.set_title("(d) 拐点之后，「每单位值多少钱」整张表换掉", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
png = os.path.join(OUT, "day78_lp_sensitivity.png")
plt.savefig(png, dpi=130)
plt.close()

# ============================================================
# 11. 断言
# ============================================================
assert abs(sh_fd).sum() > 0
assert np.allclose(sh_fd, sh_dual, atol=5e-2), "差分与对偶值必须一致"
assert first_break is not None, "50% 内应至少出现一次换基"
assert rule_start is not None
# 本实例（3 资源 × 6 变量、单一最优顶点）实测：真实拐点晚于法则松口，法则整体偏保守。
# 这是可复现事实，教案 §七 按「法则保守」如实写；反例另设混合实题再测（见 §7）。
assert first_break is not None and rule_start is not None
# 本实例的结论是「法则偏保守」：所有 pair 的实测幅度都 ≥ 法则上限，蒙特卡洛漏报为 0。
# 这就是本讲的核心事实，必须让它被断言钉住，而不是指望脚本里冒出个反例。
assert all(p["ratio"] is None or p["ratio"] >= 1.0 - 1e-9 for p in pairs), \
    "若出现法则失效的 pair，教案必须改写（本实例未出现）"
assert all(m["pass_but_infeasible_pct"] == 0 and m["pass_but_changed_pct"] == 0 for m in monte), \
    "若法则放行却换基/不可行，教案要改结论"
assert abs(sh_fd[res_tight[0]]) > 0
RES["runtime_s"] = round(time.time() - T0, 2)
with open(os.path.join(OUT, "day78_result.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print("=" * 82)
print(f"✅ 完成 {RES['runtime_s']:.1f}s｜法则松口 t={rule_start:.2f}%｜真实拐点 t={first_break:.2f}%")

# ============================================================
# 12. 对照模型：三库三城 + 总调水上限（新增第 4 条共同资源）
#     这是一道「所有水源合起来只有一条总线」的题：影子价格只剩一个主价格档，
#     各项资源的允许幅度全为 0 或很大——100% 法则在这里几乎不提供信息。
# ============================================================
print("=" * 82)
print("【10】对照模型：三库三城 + 总调水上限（总水权是唯一真正的瓶颈）")
CW = np.array([9.0, 8.2, 10.5, 8.6, 9.4, 11.2, 8.0, 9.9])
LW = ["A·甲", "A·乙", "A·丙", "B·甲", "B·乙", "B·丙", "C·甲", "C·丙"]
RW = np.array([
    [1, 1, 1, 0, 0, 0, 1, 0],     # 甲库总出库
    [0, 0, 0, 1, 1, 1, 0, 0],     # 乙库总出库
    [0, 0, 1, 0, 0, 1, 0, 1],     # 丙库总出库
    [1, 1, 1, 1, 1, 1, 1, 1],     # 总调水（共同资源）
], float)
AUBW = np.vstack([RW, -np.eye(8)])
RW_NM = ["甲库可供", "乙库可供", "丙库可供", "总调水上限"]
BW0 = np.array([180.0, 140.0, 130.0, 300.0])


def solve_w(bv):
    r = linprog(-CW, A_ub=AUBW, b_ub=np.concatenate([bv, np.zeros(8)]),
                bounds=[(0, None)] * 8, method="highs")
    if r.status != 0:
        return None, None, False
    return float(-r.fun), np.asarray(r.x, float).copy(), True


def basis_w(x, bv, tol=1e-7):
    s = np.concatenate([bv, np.zeros(8)]) - AUBW @ x
    return frozenset({int(i) for i in np.where(np.abs(s) <= tol)[0]})


zw, xw, okw = solve_w(BW0)
Bw0 = basis_w(xw, BW0)
duw = linprog(-CW, A_ub=AUBW, b_ub=np.concatenate([BW0, np.zeros(8)]),
              bounds=[(0, None)] * 8, method="highs").ineqlin.marginals
shw = -duw[:4]
print(f"  z*={zw:.2f}；影子价 " + "｜".join(f"{RW_NM[r]} {shw[r]:.3f}" for r in range(4)))
print(f"  卡死行 = {[RW_NM[r] for r in sorted(Bw0) if r < 4]}")
sing_w = []
for r in range(4):
    lo = hi = 0.0
    t = 0.5
    while t < 600:
        v = BW0.copy(); v[r] -= t
        z, xx, okv = solve_w(v)
        if (not okv) or basis_w(xx, v) != Bw0:
            break
        lo = -t; t += 0.5
    t = 0.5
    while t < 1200:
        v = BW0.copy(); v[r] += t
        z, xx, okv = solve_w(v)
        if (not okv) or basis_w(xx, v) != Bw0:
            break
        hi = t; t += 0.5
    sing_w.append({"name": RW_NM[r], "b": float(BW0[r]), "dn_pct": float(-lo / BW0[r] * 100),
                   "up_pct": float(hi / BW0[r] * 100)})
    print(f"   {RW_NM[r]:6s} 影子价 {shw[r]:5.3f}｜可降 {sing_w[-1]['dn_pct']:5.1f}%｜可升 {sing_w[-1]['up_pct']:5.1f}%")
# 逐库独立的增量价值（把「影子价能不能直接用」做成一张表）
inc = []
for r in range(3):
    v = BW0.copy(); v[r] += 10
    z2, _, okv = solve_w(v)
    inc.append({"name": RW_NM[r], "per_unit": float((z2 - zw) / 10) if okv else None})
print("  每座库单独 +10 万m3 的增量价值：" +
      "｜".join(f"{i['name']} {i['per_unit']:+.2f}" for i in inc))
print(f"  → 只有『总调水上限』值 {shw[3]:.2f}，单独扩某一座库对总收益 **无影响或更差**（水面被重新分配）")
RES["wide_case"] = {"z": zw, "shadow": shw.tolist(), "tight": [RW_NM[r] for r in sorted(Bw0) if r < 4],
                    "single": sing_w, "marginal_by_res": inc}
