# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化模型（集合函数 / 次模 submodular 最大化 · 覆盖价值）/ L5 算法（贪心 (1-1/e) 保证 · 局部搜索）
anchor  : Roadmap L4 优化模型 / L5 算法。前接 Day60（急救站集合覆盖 / 最大覆盖：目标只数「被覆盖的点数」，
          边际递减是天然的、贪心对不对没人追问）、Day71（后悔准则选址）、Day73（参数规划：预算当参数）。
          本讲补全库零命中的一族：**目标函数本身就是集合函数** ——
          「贪心为什么能带保证」「保证是多少」「保证什么时候不管用」。
goal    : 云溪县 2027 年山洪预警雨量站补点。县财政只追加 K 个站；主沟 8 个待保护对象（7 村 + 1 园区），
          已有 1 个市建老站（不占县级预算但**必须算进覆盖**）。
          四个决策问题：
          1) 雨量站只能预警**它下游**的人（提前量 = 距离 / 洪峰流速）：站在下游、或离得太近，都是废站；
          2) 贪心 vs 最优差多少？把差距和 (1-1/e)=63.2% 的理论保证摆在一起看；
          3) 加一句「每个村都得有人管」的公平约束以后，保证还在、但差距会变大 —— 变大多少、贵多少？
          4) 站越密，「覆盖两次」越像重复建设：站间雨量相关 ρ 一高，第二重覆盖就同步失效。
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-11/day74_submodular_sensor.py"
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
# 1. 实例（全部用公里为单位的 Python 浮点，全程不依赖 numpy 的逐元素运算）
# ----------------------------------------------------------------------------
#          名称        沿沟里程 km  人口   重要性权重  类型
TARGETS = [
    ("上坪村",    2.3, 1600, 3.2, "村"),   # 最上游小村：只有「起点站」护得住
    ("石桥村",    4.2,  820, 1.20, "村"),
    ("柳湾村",    6.4, 2200, 1.0, "村"),
    ("云岭村",   11.0, 2600, 1.80, "村"),
    ("坳上村",   11.8,  300, 1.80, "村"),
    ("河湾村",   15.4, 2600, 1.20, "村"),
    ("城关社区", 18.0, 1700, 1.45, "村"),
    ("园区片区", 20.2, 1400, 1.20, "园"),
]
NT = len(TARGETS)
T_NAME = [t[0] for t in TARGETS]
T_KM = [float(t[1]) for t in TARGETS]
T_POP = [float(t[2]) for t in TARGETS]
T_W = [float(t[3]) for t in TARGETS]
T_KIND = [t[4] for t in TARGETS]

#     名称     里程 km  半径 km（谷宽） 建设费(万)  年运维(万)
CAND = [
    ("上坪北",  1.0, 3.2, 26, 1.2),   # 只护上坪村+石桥村
    ("石桥北",  2.4, 4.0, 27, 1.2),   # 只护石桥村+柳湾村
    ("柳湾北",  5.2, 2.6, 30, 1.4),
    ("南坳短站",  5.1, 3.2, 28, 1.3),   # 只护柳湾村（短站，便宜）
    ("云岭北",  9.6, 2.6, 33, 1.5),
    ("坳上北",  9.2, 3.6, 36, 1.6),
    ("云岭短站", 9.0, 2.6, 30, 1.4),   # 短站：只护住云岭村
    ("河湾北", 12.4, 3.2, 36, 1.6),
    ("河湾短站", 14.1, 1.4, 30, 1.4),   # 只护河湾村（短站）
]
NC = len(CAND)
C_NAME = [c[0] for c in CAND]
C_KM = [float(c[1]) for c in CAND]
C_RAD = [float(c[2]) for c in CAND]
C_COST = [float(c[3]) for c in CAND]
C_OPEX = [float(c[4]) for c in CAND]

FLOW_SPEED = 0.8      # 主沟洪峰传播 0.8 km/h（暴雨后沟道缓流，经验值）
LEAD_MIN_H = 1.5      # 提前量门槛 1.5 h：低于它，「敲锣→撤到安全点」来不及
NEED_KM = FLOW_SPEED * LEAD_MIN_H      # 折算成距离：1.2 km 以内 = 报警太晚
K_BUDGET = 3          # 县财政只追加 3 个站（K 曲线会说明第 4 个值不值）


def cover_list(sites):
    """每个站「护得住」哪些对象：站位于对象上游（d>0）、d ≤ 半径、d ≥ 流速 × 门槛。

    物理直觉：雨从上游来、洪峰往下游走。所以
      · 站在对象下游 = 对本对象没用（水已经过去了）；
      · 站离对象太近 = 报警太晚（d < 流速 × 门槛）；
      · 站离对象太远 = 不在同一片雨团里（d > 半径）。
    三条合起来就是本讲第一个反直觉：**站只保护它下游的人。**
    """
    out = []
    for (_, km, rad, _, _) in sites:
        seg = []
        for j in range(NT):
            d = float(T_KM[j]) - float(km)
            if d > 0 and d <= float(rad) and d >= NEED_KM:
                seg.append(j)
        out.append(seg)
    return out


COVER_CAND = cover_list(CAND)

# 市建老站：不占县级预算，但本来就在覆盖里（漏算它会把「已经有人管的村」重复买一遍）
CITY = [("城关老站", 16.2, 2.0)]
COVER_CITY = cover_list([(c[0], c[1], c[2], 0.0, 0.0) for c in CITY])
BASE_LAYER = [1 if any(j in seg for seg in COVER_CITY) else 0 for j in range(NT)]

# ----------------------------------------------------------------------------
# 2. 覆盖价值 v(S) = Σ_j 权重_j · 人口_j/1000 · g(重数_j)
# ----------------------------------------------------------------------------
# 为什么不写「覆盖 ≥1 次就满分」？—— 单个站在暴雨当天会因雷击 / 通讯中断 / 被冲而失效，
# 第二重覆盖不是重复建设，是备份。g(k) = 1 - ρ^k，ρ = 0.45：
#   第 1 重拿 55%，第 2 重再补 24.75%，第 3 重补 11%，之后基本饱和。
# g 递增且**增量递减** —— 这正是 v 单调次模（边际递减）的来源，
# 也是贪心 (1-1/e) 保证成立的前提。
RHO = 0.45


def g(k):
    return 0.0 if k <= 0 else 1.0 - RHO ** k


def layers(S):
    """每个对象被覆盖的重数 = 市建老站（算 1 层）+ 选中的补点"""
    n = list(BASE_LAYER)
    for i in S:
        for j in COVER_CAND[i]:
            n[j] += 1
    return n


def value(S):
    n = layers(S)
    return float(sum(T_W[j] * T_POP[j] / 1000.0 * g(n[j]) for j in range(NT)))


def value_once(S):
    """旧口径（Day60 全局最大覆盖）：覆盖 ≥1 次就满分"""
    n = layers(S)
    return float(sum(T_W[j] * T_POP[j] / 1000.0 * (1.0 if n[j] > 0 else 0.0) for j in range(NT)))


# ----------------------------------------------------------------------------
# 3. 算法：贪心 / 精确枚举 / 局部搜索（1-swap）/ 懒贪心（CELF）
# ----------------------------------------------------------------------------
def greedy(k=K_BUDGET, ok=None):
    S, steps = [], []
    for _ in range(k):
        best_i, best_g = None, -1e-12
        base = value(S)
        for i in range(NC):
            if i in S:
                continue
            if ok is not None and not ok(S + [i]):
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


def exact(k=K_BUDGET, ok=None):
    best, bestS = -1.0, None
    for comb in itertools.combinations(range(NC), k):
        S = list(comb)
        if ok is not None and not ok(S):
            continue
        v = value(S)
        if v > best + 1e-12:
            best, bestS = v, S
    return bestS, best


def local_swap(start, ok=None):
    """单点替换（1-swap）：拿掉一个、换进一个，只接受有改进的移动。
    —— 用它回答「贪心错了以后，能不能靠『换个站』修回来」。"""
    S, cur, it = list(start), value(start), 0
    while it < 60:
        best_gain, mv = 1e-12, None
        for a in list(S):
            for b in range(NC):
                if b in S:
                    continue
                S2 = [x for x in S if x != a] + [b]
                if ok is not None and not ok(S2):
                    continue
                gain = value(S2) - cur
                if gain > best_gain:
                    best_gain, mv = gain, (a, b)
        if mv is None:
            break
        S = [x for x in S if x != mv[0]] + [mv[1]]
        cur += best_gain
        it += 1
    return S, cur, it


def lazy_greedy(k=K_BUDGET):
    """CELF：上界优先队列代替「每步重算全部边际」；答案应与朴素贪心一致，只是更快"""
    base = value([])
    heap = sorted(([-(value([i]) - base), i] for i in range(NC)))
    S = []
    while len(S) < k and heap:
        _, i = heap.pop(0)
        if i in S:
            continue
        gain = value(S + [i]) - value(S)
        if not heap or gain >= -heap[0][0] - 1e-12:
            S.append(i)
        else:
            heap.append([-gain, i])
            heap.sort()
    return S


# ----------------------------------------------------------------------------
# 4. 公平约束：7 个村每一个都至少有一重覆盖（园区不算「村」）
# ----------------------------------------------------------------------------
VILLAGES = [j for j, kd in enumerate(T_KIND) if kd == "村"]


FAIR_TARGETS = [0, 1, 4]   # 上游三个村：人不多，但撤不出来是「人命」问题


def fair_ok(S):
    """公平约束：**上游四个村每一个都得有人管**（园区不算村）。"""
    n = layers(S)
    return all(n[j] >= 1 for j in FAIR_TARGETS)


def prefix_ok(S, k=K_BUDGET):
    """贪心可用的版本：**每一步都不许把「以后补不上」的组合走进去**。
    规则：已选 |S| 个之后，还没被管的上游村数量 ≤ 剩余名额 (k-|S|)。
    直接拿 fair_ok 当剪枝条件会把贪心卡死在第 0 步（任何单站都不可行），
    所以约束要写成「过程性」的：允许暂时不可行，但必须还得补得上。"""
    n = layers(S)
    need = sum(1 for j in FAIR_TARGETS if n[j] == 0)
    return need <= k - len(S)


def start_ok(S):
    """更弱的对照口径：只要求「最上游的上坪村有人管」—— 用来说明公平约束写多严会改变什么。"""
    n = layers(S)
    return n[0] >= 1


# ----------------------------------------------------------------------------
# 5. 相关雨量：多重覆盖 ≠ 多重备份
# ----------------------------------------------------------------------------
P_FAIL_SINGLE = 0.06     # 单站年失效概率（雷击 / 通讯 / 设备被冲）


def effective_value(S, rho_corr):
    """站间雨量相关 ρ_corr：多站**同时**失效的概率 = p·ρ^(k-1)。
    ρ=0 → 备份全额有效（本讲主口径）；ρ=1 → 多站等于一站（覆盖两次也不多一层保险）。"""
    n = layers(S)
    tot = 0.0
    for j in range(NT):
        if n[j] == 0:
            continue
        # 单站失效 p；k 重覆盖下「同时失效」概率 p·ρ^(k-1)（ρ=0 独立、ρ=1 完全同步）
        p_ok = 1.0 - P_FAIL_SINGLE * (rho_corr ** (n[j] - 1))
        # 有效层数 κ：k 重覆盖的「等效独立层数」，ρ 越大越打折扣
        kappa = 1.0 + (n[j] - 1) * (1.0 - rho_corr)
        tot += T_W[j] * T_POP[j] / 1000.0 * g(kappa) * p_ok
    return float(tot)


# ----------------------------------------------------------------------------
# 6. 求解
# ----------------------------------------------------------------------------
V_BASE = value([])

S_G, STEPS_G = greedy()
V_G = value(S_G)
S_E, V_E = exact()
V_E = value(S_E)
S_LS, V_LS, LS_IT = local_swap(S_G)

S_GF, _ = greedy(ok=prefix_ok)
V_GF = value(S_GF)
S_EF, V_EF = exact(ok=fair_ok)
S_LSF, V_LSF, LSF_IT = local_swap(S_GF, ok=prefix_ok)

CURVE = []
for k in range(1, NC + 1):
    Sg, _ = greedy(k)
    _, Ve = exact(k)
    CURVE.append({"K": k, "greedy": round(value(Sg), 4), "exact": round(Ve, 4),
                  "ratio": round(value(Sg) / Ve, 4) if Ve > 0 else None})

SPEED = []
_flow0 = FLOW_SPEED
for v in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6):
    globals()["FLOW_SPEED"] = v
    globals()["NEED_KM"] = v * LEAD_MIN_H
    segs = cover_list(CAND)
    SPEED.append({"流速": v,
                  "覆盖边": sum(len(x) for x in segs),
                  "可保护对象": len({j for seg in segs for j in seg})})
globals()["FLOW_SPEED"] = _flow0
globals()["NEED_KM"] = _flow0 * LEAD_MIN_H

CORR = [{"rho": rc, "greedy": round(effective_value(S_G, rc), 4),
         "exact": round(effective_value(S_E, rc), 4),
         "multi": round(effective_value([0, 2, 5], rc), 4)} for rc in (0.0, 0.3, 0.6, 0.8, 0.95, 1.0)]

# 次模性抽样核验：v(S+i) - v(S) ≥ v(T+i) - v(T)，其中 S ⊆ T，i ∉ T
rnd = random.Random(2026)
CHECKED = VIOL = 0
for _ in range(3000):
    a, b = rnd.randint(0, 2), rnd.randint(3, 6)
    S = rnd.sample(range(NC), a)
    rest = [x for x in range(NC) if x not in S]
    T = S + rnd.sample(rest, min(b, len(rest)))
    cand = [x for x in range(NC) if x not in T]
    if not cand:
        continue
    i = rnd.choice(cand)
    CHECKED += 1
    if (value(S + [i]) - value(S)) < (value(T + [i]) - value(T)) - 1e-9:
        VIOL += 1

# ----------------------------------------------------------------------------
# 7. 断言（教学点必须成立，否则脚本必须炸）
# ----------------------------------------------------------------------------
BOUND = 1 - 1 / math.e
assert sum(len(x) for x in COVER_CAND) > 0, "覆盖关系为空：半径 / 流速 / 门槛三个数打架"
assert all(len(COVER_CAND[i]) > 0 for i in range(NC)), "有候选站一个都护不住，删掉它"
assert VIOL == 0, f"次模性被破坏 {VIOL} 次 / {CHECKED}"
assert V_E >= V_G > V_BASE, "贪心必须不优于精确、优于基线"
assert V_G / V_E >= BOUND + 1e-6, "贪心比必须 ≥ 1-1/e"
assert V_LS >= V_G, "局部搜索不得低于贪心起点"
assert S_EF is not None and V_EF > V_GF, "公平约束下贪心同样劣于精确"
assert sorted(lazy_greedy()) == sorted(S_G), "CELF 必须给出与贪心相同的集合"
assert V_EF < V_E, "公平约束必须付出代价"

# ----------------------------------------------------------------------------
# 8. 输出
# ----------------------------------------------------------------------------
res = {
    "base_value": round(V_BASE, 4),
    "bound": round(BOUND, 4),
    "greedy": {"S": [C_NAME[i] for i in S_G], "value": round(V_G, 4),
               "ratio": round(V_G / V_E, 4), "steps": STEPS_G},
    "exact": {"S": sorted(C_NAME[i] for i in S_E), "value": round(V_E, 4)},
    "local_swap": {"S": sorted(C_NAME[i] for i in S_LS), "value": round(V_LS, 4), "iter": LS_IT},
    "fair": {"greedy": {"S": [C_NAME[i] for i in S_GF], "value": round(V_GF, 4)},
             "exact": {"S": sorted(C_NAME[i] for i in S_EF), "value": round(V_EF, 4)},
             "ratio": round(V_GF / V_EF, 4), "price_vs_main": round(V_E - V_EF, 4)},
    "once_cover": {"greedy": round(value_once(S_G), 4), "exact": round(value_once(S_E), 4)},
    "curve": CURVE,
    "speed": SPEED,
    "corr": CORR,
    "submodularity": {"checked": CHECKED, "violations": VIOL},
    "layers": {T_NAME[j]: {"base": BASE_LAYER[j],
                           "by": [C_NAME[i] for i in range(NC) if j in COVER_CAND[i]]}
               for j in range(NT)},
}
with open(HERE / "day74_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 9. 作图
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))

a = ax[0, 0]
for j in range(NT):
    col = "#c0392b" if T_KIND[j] == "村" else "#8e44ad"
    a.plot([T_KM[j]], [0.0], "o", color=col, ms=9)
    a.text(T_KM[j], 0.55, f"{T_NAME[j]}\n{int(T_POP[j])}人", ha="center", fontsize=7.2)
for nm, km, rad in CITY:
    a.plot([km, km + rad], [0.0, 0.0], "-", lw=4, color="#117a65", alpha=0.40)
    a.text(km, -0.85, nm, fontsize=7, color="#117a65")
for idx, (nm, km, rad) in enumerate(zip(C_NAME, C_KM, C_RAD)):
    yy = 8.2 - idx * 0.95
    if idx in S_G and idx in S_E:
        col = "#1f618d"
    elif idx in S_E:
        col = "#f0b27a"
    elif idx in S_G:
        col = "#7fb3d5"
    else:
        col = "#d5d8dc"
    a.plot([km, km + rad], [yy, yy], "-", lw=6, color=col, alpha=0.85)
    a.plot([km], [yy], "|", color="black", ms=9)
    a.text(km, yy + 0.28, nm, fontsize=7, ha="left")
a.annotate("站的预警段只往下游走", xy=(12, 5.9), xytext=(6.5, 6.6),
           arrowprops=dict(arrowstyle="->", color="#7f8c8d"), fontsize=9, color="#7f8c8d")
a.set_ylim(-1.3, 9.4)
a.set_yticks([])
a.set_xlabel("沿主沟里程 / km")
a.set_title("(a) 7 个候选站的预警段（只向下游）\n深蓝=两者都选  橙=只精确选  浅蓝=只贪心选  绿=市建老站", fontsize=10)

b = ax[0, 1]
gs = [s["marginal"] for s in STEPS_G]
b.bar(range(1, len(gs) + 1), gs, color="#1f618d")
for x, y in zip(range(1, len(gs) + 1), gs):
    b.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
b.set_xlabel("贪心第几步")
b.set_ylabel("边际增益")
b.set_title("(b) 贪心的边际增益逐级下降\n（这就是 (1-1/e) 保证的来源）", fontsize=10)
b.grid(alpha=0.3, axis="y")

c = ax[1, 0]
Ks = [r["K"] for r in CURVE]
c.plot(Ks, [r["exact"] for r in CURVE], "o-", color="#c0392b", label="精确枚举")
c.plot(Ks, [r["greedy"] for r in CURVE], "s--", color="#1f618d", label="贪心")
c.plot(Ks, [BOUND * r["exact"] for r in CURVE], ":", color="gray", label="(1-1/e)×精确")
c.axvline(K_BUDGET, color="#7f8c8d", lw=1, ls="--")
c.text(K_BUDGET + 0.12, min(r["exact"] for r in CURVE), f"K={K_BUDGET}", fontsize=8, color="#7f8c8d")
c.set_xlabel("预算点数 K")
c.set_ylabel("覆盖价值")
c.set_title("(c) 各 K 下贪心 vs 精确：保证是下界，实际常贴着它", fontsize=10)
c.legend(fontsize=8)
c.grid(alpha=0.3)

d = ax[1, 1]
rcs = [r["rho"] for r in CORR]
d.plot(rcs, [r["greedy"] for r in CORR], "o-", color="#1f618d", label="贪心方案")
d.plot(rcs, [r["exact"] for r in CORR], "s--", color="#c0392b", label="精确方案")
d.set_xlabel("站间雨量相关系数 ρ")
d.set_ylabel("有效覆盖价值（含失效概率）")
d.set_title("(d) ρ 越高，「覆盖两次」越像重复建设\nρ=1 时多建的站只剩账面好看", fontsize=10)
d.legend(fontsize=8)
d.grid(alpha=0.3)

fig.suptitle("Day74 · 次模最大化：山洪预警雨量站「贪心为什么能带保证，保证什么时候不管用」", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE / "day74_submodular.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# 10. 控制台速览
# ----------------------------------------------------------------------------
print(f"基线（只有市建老站）V = {V_BASE:.3f}")
print(f"贪心 {res['greedy']['S']}  V = {V_G:.3f}")
print(f"精确 {res['exact']['S']}  V = {V_E:.3f}")
print(f"贪心/精确 = {V_G / V_E:.4f}   (1-1/e = {BOUND:.4f})")
print(f"局部搜索 {res['local_swap']['S']}  V = {V_LS:.3f}（{LS_IT} 次替换）")
print(f"公平约束 贪心 {V_GF:.3f} / 精确 {V_EF:.3f} = {V_GF / V_EF:.4f}（代价 {V_E - V_EF:.3f}）")
print(f"次模性抽样 {CHECKED} 对，破坏 {VIOL} 次；耗时 {time.perf_counter() - T0:.2f}s")
