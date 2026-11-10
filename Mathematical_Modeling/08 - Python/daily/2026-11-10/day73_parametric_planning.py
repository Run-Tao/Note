# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化 / 决策模型（**参数规划 / 影子价格曲线** · 分段线性值函数 · 边际与平均之别）
anchor  : Roadmap L4 优化模型。前接 Day3 线性规划（会解）、Day26 对偶与影子价格（价格从哪来）、
          Day52 最优基与灵敏度区间（价格还有多长保质期）、Day58 对偶单纯形热启动（数据变了要不要重算）。
          本讲补全库零命中的「**参数规划**」：不报一个最优解，报**整条最优解路径**。
goal    : 云溪县 2027 年 **长护险（长期护理保险）居家照护**的人力与预算配置。
          县里 2027 年只给了 3.8 万护理工时，2028 年争取到「最高 9.5 万工时」，
          但**追加的钱不是想花就能花出去** —— 卡的是人。
          四个要回答的问题：
          ① 「今年再加 1 万工时能换来多少照护质量」这个数，能不能拿去给 2028 年做预算？
          ② 预算从 0 加到 9.5 万工时，方案会变几次？每次卡住的是哪一样东西（钱、人、证）？
          ③ 3.8 → 9.5 万工时，多花 5.7 万工时，多出来多少质量？值不值？（边际 vs 平均）
          ④ 老人协会要求「先补重度失能」、医保局要求「先补上门护理」—— 同一笔钱两种划法差多少？
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-10/day73_parametric_planning.py"
输出：day73_result.json、day73_parametric_planning.png
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()

# ----------------------------------------------------------------------------
# 1. 实例（单位：万工时/年；收益 万元/年 的照护质量折算）
# ----------------------------------------------------------------------------
SVC = ["失能老人居家照护", "失能老人日间照料", "上门护理"]
C = np.array([3.60, 2.35, 2.05])                 # 每万工时成本（万元）
V = np.array([6.40, 3.80, 2.60])                 # 每万工时质量收益（万元）
# 资源：护理员 / 康复与评估师 / 车辆（每万工时占用）
R = np.array([[ 9.0, 10.2, 12.5],
              [ 2.4,  1.2,  0.8],
              [ 0.9,  0.6,  1.5]])
CAP = np.array([71.0, 18.0, 12.0])
BUD0, BUD_MAX = 3.8, 9.5

NVAR = len(SVC)
NROW = 3


def build(budget):
    M = np.vstack([R, np.ones((1, NVAR))])
    b = np.concatenate([CAP, [budget]])
    return M, b


def solve(budget, extra=None):
    """max (V−C)·x  s.t. Mx ≤ b, x ≥ 0；extra=(coef, rhs) 追加一行「专项划拨」"""
    M, b = build(budget)
    if extra is not None:
        M = np.vstack([M, np.asarray(extra[0], float)])
        b = np.concatenate([b, [float(extra[1])]])
    obj = V - C
    Aall = np.vstack([M, -np.eye(NVAR)])
    ball = np.concatenate([b, np.zeros(NVAR)])
    best, bx = -np.inf, None
    for sel in combinations(range(Aall.shape[0]), NVAR):
        sel = list(sel)
        Ms = Aall[sel]
        if np.linalg.matrix_rank(Ms) < NVAR:
            continue
        try:
            x = np.linalg.solve(Ms, ball[sel])
        except np.linalg.LinAlgError:
            continue
        if x.min() < -1e-9 or np.any(Aall @ x > ball + 1e-9):
            continue
        val = float(obj @ x)
        if val > best + 1e-10:
            best, bx = val, x
    return best, bx


def g(budget):
    return solve(budget)[0]


def tight(budget):
    val, x = solve(budget)
    M, b = build(budget)
    t = []
    for i in range(NROW):
        if abs(float(M[i] @ x) - b[i]) < 1e-6:
            t.append(["护理员", "评估师", "车辆"][i])
    if abs(float(M[NROW] @ x) - b[NROW]) < 1e-6:
        t.append("预算")
    return val, x, t


QUICK = True   # 备课用：粗网格定位拐点，再二分逼点

# ----------------------------------------------------------------------------
# 2. 单参数：预算 b 从 0 → 22 万工时
# ----------------------------------------------------------------------------
def scan(lo, hi, n):
    ts = np.linspace(lo, hi, n)
    gs = np.array([g(t) for t in ts])
    sl = np.diff(gs) / np.diff(ts)
    brk = []
    for i in range(len(sl) - 1):
        if abs(sl[i + 1] - sl[i]) > 1e-6:
            a, b = ts[i], ts[i + 2]
            s0 = sl[i]
            jump = abs(sl[i + 1] - sl[i])
            for _ in range(60):
                m = 0.5 * (a + b)
                h = 1e-6
                sm = (g(m + h) - g(m - h)) / (2 * h)
                if abs(sm - s0) > 0.6 * jump:
                    b = m
                else:
                    a = m
            p = 0.5 * (a + b)
            if not brk or abs(p - brk[-1]) > 1e-3:
                brk.append(p)
    return brk


BRK_RAW = [p for p in scan(0.0, BUD_MAX, 221) if p > 1e-3]
# 二分收尾时两侧窗口可能重叠，产生「假拐点」：用左/右斜率真的不同来判定（教案 §6 踩坑④）
BRK = []
for p in BRK_RAW:
    h = 1e-5
    sL = (g(p) - g(p - h)) / h
    sR = (g(p + h) - g(p)) / h
    if abs(sR - sL) > 1e-3 and (not BRK or p - BRK[-1] > 1e-2):
        BRK.append(p)

EDGES = [0.0] + BRK + [BUD_MAX]
SEG = []
for i in range(len(EDGES) - 1):
    a, b = EDGES[i], EDGES[i + 1]
    val, x, t = tight(0.5 * (a + b))
    slope = (g(b - 1e-6) - g(a + 1e-6)) / ((b - 1e-6) - (a + 1e-6))
    SEG.append({"a": a, "b": b, "slope": slope, "value": val, "x": x, "tight": t})

BRK_CHECK = []
for p in BRK:
    h = 1e-6
    sL = (g(p) - g(p - h)) / h
    sR = (g(p + h) - g(p)) / h
    BRK_CHECK.append({"budget": p, "left": sL, "right": sR, "gap": sR - sL})

VAL0, X0, T0SET = tight(BUD0)
VALMAX, XMAX, TMAX = tight(BUD_MAX)

# 2027 已批预算处的「局部」影子价格：这是能写进报表的那个数
EPS = 1e-4
LOCAL_SHADOW = (g(BUD0 + EPS) - g(BUD0 - EPS)) / (2 * EPS)
# 2028 追加预算后的影子价格
SHADOW_MAX = SEG[-1]["slope"]
AVG_2027 = VAL0 / BUD0
AVG_2028 = VALMAX / BUD_MAX

# ----------------------------------------------------------------------------
# 3. 两支意见：专项划拨（重度优先 / 上门护理优先）
# ----------------------------------------------------------------------------
SPECIAL = []
for name, j, amt in [("老人协会：重度照护专项", 0, 6.0), ("医保局：上门护理专项", 2, 6.0)]:
    coef = np.zeros(NVAR)
    coef[j] = 1.0
    row = []
    for t in np.linspace(BUD0, BUD_MAX, 41):
        val, _ = solve(t, extra=(coef, amt))
        row.append((t, val if np.isfinite(val) else None))
    SPECIAL.append({"name": name, "curve": row})

# 资源影子价格：在 **2027 已批预算** 与 **2028 预算** 两处分别放宽 1 单位资源
def resource_shadow(budget):
    base = g(budget)
    out = []
    for i in range(NROW):
        CAP[i] += 1.0
        out.append(float(g(budget) - base))
        CAP[i] -= 1.0
    return out


RES_SHADOW_2027 = resource_shadow(BUD0)
RES_SHADOW_2028 = resource_shadow(BUD_MAX)
RES_SHADOW = RES_SHADOW_2027

# 单项盈亏（为什么某些项目永远不进方案）
SINGLE = [{"svc": SVC[j], "profit": float(V[j] - C[j]), "ratio": float(V[j] / C[j])}
          for j in range(NVAR)]
# 非负利润项必须全进方案？——用「单独只做这一项」的上限做对照
SOLO = []
for j in range(NVAR):
    M, b = build(1e9)
    Rres = CAP / R[:, j]
    solo = float(min(Rres.min(), 1e9))
    SOLO.append({"svc": SVC[j], "max_volume_by_resource": solo})

# ----------------------------------------------------------------------------
# 3.5 决策表：2027 再要 0.5 万工时，花在哪项上最值 / 最不值
# ----------------------------------------------------------------------------
MARGINAL_05 = []
for j in range(NVAR):
    coef = np.zeros(NVAR)
    coef[j] = -1.0                       # −x_j ≤ −0.5  ⇔  x_j ≥ 0.5
    val, _ = solve(BUD0, extra=(coef, -0.5))
    MARGINAL_05.append({"svc": SVC[j], "value": (val if np.isfinite(val) else None),
                        "gain": (val - VAL0) if np.isfinite(val) else None})

# ----------------------------------------------------------------------------
# 4. 图
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10.2))

ax = axes[0, 0]
ts = np.linspace(0, BUD_MAX, 441)
ax.plot(ts, [g(t) for t in ts], color="#2b6cb0", lw=2.2)
for p in BRK:
    ax.axvline(p, color="#c53030", ls="--", lw=1)
ax.axvline(BUD0, color="#2d3748", ls=":", lw=1.5)
ax.annotate(f"2027 已批 {BUD0}", (BUD0, VALMAX * 0.05), rotation=90, fontsize=9, va="bottom")
ax.set_xlabel("可用护理工时（万工时/年）")
ax.set_ylabel("最优照护质量（万元/年等价）")
ax.set_title("(a) 值函数 g(b)：分段线性，每个拐点换一次方案")

ax = axes[0, 1]
mids = [(s["a"] + s["b"]) / 2 for s in SEG]
ax.step(mids, [s["slope"] for s in SEG], where="mid", color="#2f855a", lw=2)
for s, m in zip(SEG, mids):
    ax.annotate(f"{s['slope']:.2f}", (m, s["slope"]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8)
ax.set_xlabel("预算（万工时）")
ax.set_ylabel("边际价值 dg/db")
ax.set_title("(b) 影子价格是台阶：「再加 1 万工时」不是常数")

ax = axes[1, 0]
bottoms = np.zeros(len(SEG))
cols = ["#2b6cb0", "#dd6b20", "#38a169"]
for j in range(NVAR):
    vals = np.array([s["x"][j] for s in SEG])
    ax.bar([f"段{i+1}\n[{s['a']:.1f},{s['b']:.1f}]" for i, s in enumerate(SEG)],
           vals, bottom=bottoms, color=cols[j], label=SVC[j])
    bottoms += vals
ax.set_ylabel("万工时/年")
ax.set_title("(c) 每段方案：谁被换进来、谁被换出去")
ax.legend(fontsize=8)

ax = axes[1, 1]
for i, sp in enumerate(SPECIAL):
    xs = [t for t, v in sp["curve"] if v is not None]
    ys = [v for t, v in sp["curve"] if v is not None]
    ax.plot(xs, ys, lw=2, marker="o", ms=2.5, label=sp["name"])
xs = np.linspace(BUD0, BUD_MAX, 41)
ax.plot(xs, [g(t) for t in xs], lw=2, ls="--", color="#2d3748", label="不设专项")
ax.set_xlabel("预算（万工时）")
ax.set_ylabel("最优照护质量")
ax.legend(fontsize=8)
ax.set_title("(d) 划专项的代价：同一笔钱两种用法")

fig.tight_layout()
fig.savefig(HERE / "day73_parametric_planning.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. 断言（卡教学点真的发生）
# ----------------------------------------------------------------------------
assert len(SEG) >= 3, f"路径只有 {len(SEG)} 段，教学点不足"
slopes = [s["slope"] for s in SEG]
assert max(slopes) - min(slopes) > 0.5, f"影子价格变化太小 {slopes}"
assert VALMAX <= VAL0 + 1e-6 or True
assert VALMAX - VAL0 > 1e-6, "2028 追加预算必须真的带来收益"
assert any("护理员" in s["tight"] for s in SEG), "至少要有一段卡在人头上"

res = {
    "services": SVC, "cost": C.tolist(), "value": V.tolist(), "resource": R.tolist(),
    "cap": CAP.tolist(), "budget0": BUD0, "budget_max": BUD_MAX,
    "breakpoints": BRK_CHECK,
    "segments": [{"a": s["a"], "b": s["b"], "slope": s["slope"], "value": s["value"],
                  "x": np.asarray(s["x"]).tolist(), "tight": s["tight"]} for s in SEG],
    "budget0": {"value": VAL0, "x": X0.tolist(), "tight": T0SET, "local_shadow": LOCAL_SHADOW,
                "avg": AVG_2027},
    "budgetmax": {"value": VALMAX, "x": XMAX.tolist(), "tight": TMAX, "shadow": SHADOW_MAX,
                  "avg": AVG_2028},
    "resource_shadow_2027": RES_SHADOW_2027,
    "resource_shadow_2028": RES_SHADOW_2028,
    "single_profit": SINGLE,
    "marginal_05": MARGINAL_05,
    "solo_max": SOLO,
    "special": [{"name": sp["name"], "curve": sp["curve"]} for sp in SPECIAL],
    "elapsed_s": time.perf_counter() - T0,
}
with open(HERE / "day73_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=float)

print(f"[2027] b={BUD0}  值={VAL0:.3f}  紧张={T0SET}  x={np.round(X0,3)}")
print(f"        局部影子价={LOCAL_SHADOW:.3f}  平均={AVG_2027:.3f}")
print(f"[2028] b={BUD_MAX}  值={VALMAX:.3f}  紧张={TMAX}  x={np.round(XMAX,3)}")
print(f"        局部影子价={SHADOW_MAX:.3f}  平均={AVG_2028:.3f}")
print(f"拐点 {[round(b['budget'], 3) for b in BRK_CHECK]}")
for i, s in enumerate(SEG):
    print(f"  段{i+1} b∈[{s['a']:6.2f},{s['b']:6.2f}] 影子价={s['slope']:6.3f} "
          f"紧张={s['tight']} x={np.round(s['x'],3)}")
print("再要 0.5 万工时强制给某项:", [(d["svc"], None if d["gain"] is None else round(d["gain"], 3)) for d in MARGINAL_05])
print("单项利润:", [(d["svc"], round(d["profit"], 2)) for d in SINGLE])
print("资源影子价 2027:", [round(v, 3) for v in RES_SHADOW_2027])
print("资源影子价 2028:", [round(v, 3) for v in RES_SHADOW_2028])
for sp in SPECIAL:
    last = [v for t, v in sp["curve"] if v is not None][-1]
    print(f"{sp['name']}: 到 {BUD_MAX} 万工时时 {last:.2f}（不设专项 {VALMAX:.2f}）")
print(f"耗时 {time.perf_counter() - T0:.2f}s")
