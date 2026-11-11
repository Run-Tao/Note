# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化模型（集合函数 / 次模 submodular 最大化 · 预算约束下的覆盖价值）/ L5 算法（贪心 (1-1/e) · 局部搜索）
anchor  : Roadmap L4 优化模型 / L5 算法。前接 Day60（急救站集合覆盖 / 最大覆盖：目标只数「被覆盖的点数」，
          边际递减是天然的、贪心对不对没人追问）、Day71（后悔准则选址）、Day73（参数规划：预算当参数）。
          本讲补全库零命中的一族：**目标函数本身就是集合函数** ——
          「贪心为什么能带保证」「保证是给谁的」「预算版为什么一加钱就崩」「保证什么时候不管用」。
goal    : 云溪县 2027 年山洪预警雨量站补点。县财政只追加一笔钱；主沟 8 个待保护对象。
          五个决策问题：
          1) 雨量站只能预警**它下游**的人（提前量 = 距离 / 洪峰流速）：站在下游、或离得太近，都是废站；
          2) 「多一个站多一份保险」到底值多少？—— 用 g(k)=1-ρ^k 把「第 k 重覆盖」标上价；
          3) 固定站数 K 时贪心 vs 精确差多少？(1-1/e) 保证摆在一起看；
          4) 换成**预算**（每个站造价不同）以后，贪心会不会崩？崩在哪个预算点上？
          5) 站间雨量相关 ρ 一高，「覆盖两次」就同步失效 —— 多建的站还剩多少价值？

运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-11/day74_submodular_rain_gauge.py"
输出：day74_result.json、day74_submodular.png
"""
import itertools
import json
import math
import random
import time
from pathlib import Path

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
# 1. 实例
# ----------------------------------------------------------------------------
#            村名        沿主沟里程 km   权重(人口千×重要度)
TARGETS = [
    ("上坪村", 1.8, 24.0),   # 最上游；只有「上坪北」一类站护得住
    ("石桥村", 4.4, 10.0),
    ("柳湾村", 6.4, 12.0),
    ("谷口村", 8.2, 9.0),
    ("云岭村", 10.8, 26.0),  # 人口最多
    ("坳上村", 12.6, 8.0),
    ("河湾村", 15.6, 18.0),
    ("桥头村", 18.4, 9.0),
]
NT = len(TARGETS)
T_NAME = [t[0] for t in TARGETS]
T_KM = [float(t[1]) for t in TARGETS]
T_W = [float(t[2]) for t in TARGETS]

#     站名      沟口里程 km  上游半径 km   造价(万)
CAND = [
    ("甲站", 0.4, 6.6, 30),   # 守上坪/石桥/柳湾三村
    ("乙站", 4.6, 5.0, 26),   # 守柳湾/谷口（与甲站在柳湾叠了一层）
    ("丙站", 8.6, 5.6, 10),   # 共享站：县气象局已建一部分，县级只出 10 万配套
    ("丁站", 13.4, 5.2, 26),  # 守河湾/桥头
]
NC = len(CAND)
C_NAME = [c[0] for c in CAND]
C_KM = [float(c[1]) for c in CAND]
C_RAD = [float(c[2]) for c in CAND]
C_COST = [float(c[3]) for c in CAND]

FLOW_SPEED = 0.8      # 主沟洪峰传播 0.8 km/h
LEAD_MIN_H = 1.5      # 提前量门槛 1.5 h：低于它，「敲锣→撤到安全点」来不及
NEED_KM = FLOW_SPEED * LEAD_MIN_H      # = 1.2 km
RHO = 0.35            # 第二重覆盖的「残值」：第 k 重只值 ρ^(k-1)
BUDGET = 36           # 2027 年县财政实际批复（万）
B_SCAN = (10, 16, 22, 28, 30, 36, 40, 48, 56, 60, 66, 78)


def g(k):
    """第 k 重覆盖的效用：g(k) = 1 - ρ^k。
    g(0)=0；g(1)=0.65、g(2)=0.8775、g(3)=0.9571 —— 递增、增量递减。
    「多一个站多一份保险」这句话，价格就在这里：第二份保险只值第一份的 0.35。"""
    return 0.0 if k <= 0 else 1.0 - RHO ** k


def cover_list(sites):
    """每个站「护得住」哪些对象：站位于对象上游（d>0）、d ≤ 半径、d ≥ 流速 × 门槛。
    物理直觉：雨从上游来、洪峰往下游走。所以
      · 站在对象下游 = 对本对象没用（水已经过去了）；
      · 站离对象太近 = 报警太晚（d < 流速 × 门槛）；
      · 站离对象太远 = 不在同一片雨团里（d > 半径）。"""
    out = []
    for (_, km, rad, _) in sites:
        seg = []
        for j in range(NT):
            d = float(T_KM[j]) - float(km)
            if d > 0 and d <= float(rad) and d >= NEED_KM:
                seg.append(j)
        out.append(seg)
    return out


COVER = cover_list(CAND)


def layers(S):
    n = [0] * NT
    for i in S:
        for j in COVER[i]:
            n[j] += 1
    return n


def value(S, rho=None):
    rr = RHO if rho is None else rho
    n = layers(S)
    return float(sum(T_W[j] * (0.0 if n[j] <= 0 else 1.0 - rr ** n[j]) for j in range(NT)))


def cost(S):
    return float(sum(C_COST[i] for i in S))


# ----------------------------------------------------------------------------
# 2. 三种算法
# ----------------------------------------------------------------------------
def greedy_k(k):
    """固定站数 K 的贪心：每步取边际增益最大的站。"""
    S, steps = [], []
    for _ in range(k):
        base = value(S)
        best_i, best_g = None, -1e-12
        for i in range(NC):
            if i in S:
                continue
            gain = value(S + [i]) - base
            if gain > best_g:
                best_g, best_i = gain, i
        if best_i is None:
            break
        S.append(best_i)
        steps.append({"step": len(steps) + 1, "pick": C_NAME[best_i],
                      "marginal": round(best_g, 4), "value": round(value(S), 4)})
    return S, steps


def greedy_budget(B):
    """预算版贪心：每步取「增益最大」的站（不是「每万块增益最大」），钱不够就停。
    这就是最自然的写法，也是出事的地方。"""
    S, steps, spent = [], [], 0.0
    while True:
        base, best_i, best_g = value(S), None, 1e-12
        for i in range(NC):
            if i in S or spent + C_COST[i] > B + 1e-9:
                continue
            gain = value(S + [i]) - base
            if gain > best_g:
                best_g, best_i = gain, i
        if best_i is None:
            break
        S.append(best_i)
        spent += C_COST[best_i]
        steps.append({"pick": C_NAME[best_i], "marginal": round(best_g, 4),
                      "spent": spent, "value": round(value(S), 4)})
    return S, steps, spent


def greedy_budget_density(B):
    """预算版贪心（按性价比）：每步取「增益 / 造价」最大的站。"""
    S, spent = [], 0.0
    while True:
        base, best_i, best_r = value(S), None, 1e-12
        for i in range(NC):
            if i in S or spent + C_COST[i] > B + 1e-9:
                continue
            r = (value(S + [i]) - base) / C_COST[i]
            if r > best_r:
                best_r, best_i = r, i
        if best_i is None:
            break
        S.append(best_i)
        spent += C_COST[best_i]
    return S, spent


def exact_budget(B):
    """精确裁判：预算下全枚举（2^4 种）。小实例上枚举比通用求解器更可信。"""
    best_v, best_S = -1.0, None
    for r in range(NC + 1):
        for c in itertools.combinations(range(NC), r):
            if cost(c) <= B + 1e-9:
                v = value(c)
                if v > best_v + 1e-12:
                    best_v, best_S = v, list(c)
    return best_S, best_v


def local_search(S0, B):
    """1-swap + 1-add 局部搜索，预算可行才接受。"""
    S, cur = list(S0), value(S0)
    it = 0
    while it < 50:
        move = None
        for a in list(S) + [None]:
            rem = [x for x in S if x != a] if a is not None else list(S)
            for b in range(NC):
                if b in S:
                    continue
                cand = rem + [b]
                if cost(cand) > B + 1e-9:
                    continue
                v = value(cand)
                if v > cur + 1e-9:
                    cur, move = v, (a, b)
        if move is None:
            break
        a, b = move
        S = ([x for x in S if x != a] if a is not None else list(S)) + [b]
        it += 1
    return S, cur, it


# ----------------------------------------------------------------------------
# 3. 求解
# ----------------------------------------------------------------------------
V0 = value([])

S_GK, STEPS_K = greedy_k(2)
S_EB, V_EB = exact_budget(BUDGET)
S_GB, STEPS_B, SPENT_B = greedy_budget(BUDGET)
S_GD, SPENT_D = greedy_budget_density(BUDGET)
S_LS, V_LS, LS_IT = local_search(S_GB, BUDGET)

# 预算扫描
SCAN = []
for B in B_SCAN:
    Se, Ve = exact_budget(B)
    Sg, _, sp = greedy_budget(B)
    Sd, spd = greedy_budget_density(B)
    SCAN.append({"B": B,
                 "exact": round(Ve, 4), "exact_S": sorted(C_NAME[i] for i in Se), "exact_cost": cost(Se),
                 "value_greedy": round(value(Sg), 4), "greedy_S": [C_NAME[i] for i in Sg],
                 "greedy_spent": sp, "ratio": round(value(Sg) / Ve, 4) if Ve > 0 else None,
                 "value_density": round(value(Sd), 4), "density_S": [C_NAME[i] for i in Sd],
                 "density_ratio": round(value(Sd) / Ve, 4) if Ve > 0 else None})

# 固定 K 下的贪心 vs 精确（K = 1..NC）
KCURVE = []
for k in range(1, NC + 1):
    Sg, _ = greedy_k(k)
    best_v, best_S = -1.0, None
    for c in itertools.combinations(range(NC), k):
        v = value(c)
        if v > best_v + 1e-12:
            best_v, best_S = v, list(c)
    KCURVE.append({"K": k, "greedy": round(value(Sg), 4), "exact": round(best_v, 4),
                   "ratio": round(value(Sg) / best_v, 4)})

# 洪峰流速扫描（门槛折算成距离后，谁能护住谁）
SPEED = []
_flow0 = FLOW_SPEED
for v in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6):
    globals()["NEED_KM"] = v * LEAD_MIN_H
    segs = cover_list(CAND)
    SPEED.append({"流速": v, "门槛km": round(v * LEAD_MIN_H, 2),
                  "覆盖边": sum(len(x) for x in segs),
                  "可保护对象": len({j for seg in segs for j in seg})})
globals()["NEED_KM"] = _flow0 * LEAD_MIN_H

# 站间相关 ρ：「第二重覆盖」还剩多少
CORR = []
def exact_budget_rho(B, rho):
    best = (-1.0, None)
    for r in range(NC + 1):
        for c in itertools.combinations(range(NC), r):
            if cost(c) <= B + 1e-9:
                v = value(c, rho)
                if v > best[0] + 1e-12:
                    best = (v, list(c))
    return best[1], best[0]


for rc in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
    v1 = exact_budget_rho(30, rc)[1]
    v2 = exact_budget_rho(40, rc)[1]
    CORR.append({"rho": rc, "best1": round(v1, 4), "best2": round(v2, 4),
                 "gain_of_2nd": round(v2 - v1, 4)})

# 次模性抽样核验：v(S+i) - v(S) ≥ v(T+i) - v(T)，其中 S ⊆ T，i ∉ T
rnd = random.Random(2026)
CHECKED = VIOL = 0
for _ in range(4000):
    a = rnd.randint(0, NC - 1)
    S = rnd.sample(range(NC), a)
    rest = [x for x in range(NC) if x not in S]
    T = S + rnd.sample(rest, rnd.randint(0, len(rest)))
    cand = [x for x in range(NC) if x not in T]
    if not cand:
        continue
    i = rnd.choice(cand)
    CHECKED += 1
    if (value(S + [i]) - value(S)) < (value(T + [i]) - value(T)) - 1e-9:
        VIOL += 1

# ----------------------------------------------------------------------------
# 4. 断言（教学点必须成立，否则脚本必须炸）
# ----------------------------------------------------------------------------
BOUND = 1 - 1 / math.e
assert sum(len(x) for x in COVER) > 0, "覆盖关系为空：半径 / 流速 / 门槛三个数打架"
assert all(len(COVER[i]) > 0 for i in range(NC)), "有候选站一个都护不住，删掉它"
assert VIOL == 0, f"次模性被破坏 {VIOL} 次 / {CHECKED}"
assert value(S_GK) >= BOUND * KCURVE[1]["exact"] - 1e-9, "固定 K 下贪心必须 ≥ (1-1/e)×精确"
assert local_search(S_GK, 10 ** 9)[1] >= value(S_GK) - 1e-9, "固定 K 下局部搜索不得低于贪心"
# 预算版：便宜站先把预算吃掉，导致后面买不起大站
_ib = [i for i, r in enumerate(SCAN) if r["B"] == BUDGET][0]
assert SCAN[_ib]["ratio"] < 0.8, "批复预算处贪心必须崩（否则本讲核心不成立）"
assert SCAN[_ib]["density_ratio"] > 0.99, "按性价比的贪心在该点必须能救回来"
assert all(SCAN[i]["exact"] <= SCAN[i + 1]["exact"] + 1e-9 for i in range(len(SCAN) - 1)), "更多预算不该更差（精确解单调）"
assert V_LS >= value(S_GB) - 1e-9, "局部搜索应从贪心出发并至少不更差"
assert CORR[0]["gain_of_2nd"] > CORR[-1]["gain_of_2nd"] + 1e-6, "ρ 变大必须让第二重覆盖贬值"

# ----------------------------------------------------------------------------
# 5. 输出
# ----------------------------------------------------------------------------
res = {
    "base_value": round(V0, 4),
    "bound": round(BOUND, 4),
    "site_cover": {C_NAME[i]: [T_NAME[j] for j in sorted(COVER[i])] for i in range(NC)},
    "site_cost": {C_NAME[i]: C_COST[i] for i in range(NC)},
    "budget": BUDGET,
    "k_curve": KCURVE,
    "main": {
        "exact": {"S": sorted(C_NAME[i] for i in S_EB), "value": round(V_EB, 4), "cost": cost(S_EB)},
        "greedy": {"S": [C_NAME[i] for i in S_GB], "value": round(value(S_GB), 4),
                   "spent": SPENT_B, "ratio": round(value(S_GB) / V_EB, 4), "steps": STEPS_B},
        "density": {"S": [C_NAME[i] for i in S_GD], "value": round(value(S_GD), 4),
                    "spent": SPENT_D, "ratio": round(value(S_GD) / V_EB, 4)},
        "local_search": {"S": sorted(C_NAME[i] for i in S_LS), "value": round(V_LS, 4), "iter": LS_IT},
    },
    "scan": SCAN,
    "speed": SPEED,
    "corr": CORR,
    "submodularity": {"checked": CHECKED, "violations": VIOL},
    "layers_main": {T_NAME[j]: n for j, n in enumerate(layers(S_EB))},
    "layers_greedy": {T_NAME[j]: n for j, n in enumerate(layers(S_GB))},
}
with open(HERE / "day74_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6. 作图
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))

a = ax[0, 0]
for j in range(NT):
    a.plot([T_KM[j]], [0.0], "o", color="#c0392b", ms=9)
    a.text(T_KM[j], 0.55, f"{T_NAME[j]}\n{T_W[j]:.0f}", ha="center", fontsize=7.2)
colors = ["#1f618d", "#117a65", "#b9770e", "#8e44ad"]
for idx, (nm, km, rad, cst) in enumerate(CAND):
    yy = 8.0 - idx * 1.1
    col = colors[idx]
    sel = idx in S_EB
    a.plot([km, km + rad], [yy, yy], "-", lw=7 if sel else 4,
           color=col, alpha=0.9 if sel else 0.35)
    a.plot([km], [yy], "|", color="black", ms=9)
    a.text(km, yy + 0.30, f"{nm} {cst:.0f}万", fontsize=7.5, ha="left")
a.annotate("站的预警段只往下游走\n（上游 1.2 km 内报警太晚，半径外不同雨团）",
           xy=(11.5, 5.2), xytext=(2.0, 6.2),
           arrowprops=dict(arrowstyle="->", color="#7f8c8d"), fontsize=8.5, color="#7f8c8d")
a.set_ylim(-1.2, 9.4)
a.set_yticks([])
a.set_xlabel("沿主沟里程 / km")
a.set_title(f"(a) 4 个候选站的预警段（粗线 = 预算 {BUDGET} 万下的精确最优）", fontsize=10)

b = ax[0, 1]
Bs = [r["B"] for r in SCAN]
b.plot(Bs, [r["exact"] for r in SCAN], "o-", color="#c0392b", label="精确（全枚举）")
b.plot(Bs, [r["value_greedy"] for r in SCAN], "s--", color="#1f618d", label="贪心（按增益最大）")
b.plot(Bs, [r["value_density"] for r in SCAN], "^:", color="#117a65", label="贪心（按性价比）")
b.axvline(BUDGET, color="#7f8c8d", lw=1, ls="--")
b.text(BUDGET + 0.8, 8, f"批复 {BUDGET} 万", fontsize=8, color="#7f8c8d")
b.set_xlabel("预算 B / 万元")
b.set_ylabel("覆盖价值")
b.set_title("(b) 预算曲线：批复额处贪心掉到最优的 75%\n按性价比贪心把它救回来", fontsize=10)
b.legend(fontsize=8)
b.grid(alpha=0.3)

c = ax[1, 0]
Ks = [r["K"] for r in KCURVE]
c.plot(Ks, [r["exact"] for r in KCURVE], "o-", color="#c0392b", label="精确枚举")
c.plot(Ks, [r["greedy"] for r in KCURVE], "s--", color="#1f618d", label="贪心")
c.plot(Ks, [BOUND * r["exact"] for r in KCURVE], ":", color="gray", label="(1-1/e)×精确")
c.set_xlabel("站数 K")
c.set_ylabel("覆盖价值")
c.set_title("(c) 固定站数 K：贪心每次都命中最优，保证是下界不是预言", fontsize=10)
c.legend(fontsize=8)
c.grid(alpha=0.3)

d = ax[1, 1]
rcs = [r["rho"] for r in CORR]
d.plot(rcs, [r["best1"] for r in CORR], "o-", color="#1f618d", label="预算 30 万（1 站）")
d.plot(rcs, [r["best2"] for r in CORR], "s-", color="#c0392b", label="预算 40 万（2 站）")
d.plot(rcs, [r["gain_of_2nd"] for r in CORR], "^--", color="#117a65", label="第 2 站带来的增量")
d.set_xlabel("站间雨量相关系数 ρ")
d.set_ylabel("覆盖价值")
d.set_title("(d) ρ 越大，第二重覆盖越像重复建设\n柳湾村被甲乙两站重叠覆盖，第 2 站增量随 ρ 缩水", fontsize=10)
d.legend(fontsize=8)
d.grid(alpha=0.3)

fig.suptitle("Day74 · 次模最大化：山洪预警雨量站「贪心的保证是给谁的，预算一加怎么就崩」", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE / "day74_submodular.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. 控制台速览
# ----------------------------------------------------------------------------
print(f"基线（一个站都没有）V = {V0:.3f}")
print("站—村覆盖：")
for i in range(NC):
    print(f"  {C_NAME[i]}({C_COST[i]:.0f}万): {[T_NAME[j] for j in COVER[i]]}")
print(f"\n固定 K=2：贪心 {[C_NAME[i] for i in S_GK]} V={value(S_GK):.3f}")
print(f"预算 {BUDGET} 万：精确 {res['main']['exact']['S']} V={V_EB:.3f}（花 {cost(S_EB):.0f}）")
print(f"            贪心 {res['main']['greedy']['S']} V={value(S_GB):.3f}"
      f"（花 {SPENT_B:.0f}）ratio={value(S_GB)/V_EB:.4f}")
print(f"            性价比 {res['main']['density']['S']} V={value(S_GD):.3f}"
      f"（花 {SPENT_D:.0f}）ratio={value(S_GD)/V_EB:.4f}")
print(f"            局部搜索 {res['main']['local_search']['S']} V={V_LS:.3f}（{LS_IT} 次移动）")
print("\n预算扫描：")
for r in SCAN:
    print(f"  B={r['B']:>3} 精确 {r['exact']:>7.3f} {r['exact_S']} | "
          f"贪心 {r['value_greedy']:>7.3f} {r['greedy_S']} ratio={r['ratio']:.3f} | "
          f"性价比 {r['value_density']:>7.3f} ratio={r['density_ratio']:.3f}")
print(f"\n次模性抽样 {CHECKED} 对，破坏 {VIOL} 次")
print(f"耗时 {time.perf_counter() - T0:.2f}s")
