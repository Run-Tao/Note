# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L2 优化模型（组合优化 · 区间排序 / 区间图 DP）/ L8 算法（贪心 vs 加权区间 DP vs 精确枚举 vs 整数规划）
anchor  : Roadmap L4（优化模型）→ L5（图与网络）→ L8（算法复杂度与正确性）
goal    : 云溪水库「汛前水下检修」排班 —— 用同一份 36 条作业的数据回答
          ① 最多能塞几条（区间排序 / 最早完工优先）
          ② 每条价钱不同时怎么选（加权区间排序 / 欧拉 DP）
          ③ 有 4 支班组、同刻可并行时，为什么「两两相容」是最容易犯的错（区间图 max-weight 独立集）
          ④ 加一条班组值不值（敏感性：瓶颈在资源还是在排序）

运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-05/day68_interval_scheduling_memo.py"

输出：day68_result.json、day68_intervals.png
"""
import json
import math
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
TICK_MIN = 30  # 1 tick = 半小时；06:00 开工

# ---------------------------------------------------------------- 1. 实例
# 作业表: (名称, 起 tick, 止 tick, 需班组数, 报酬/百元)
# 窗口只有一处来源：**潮位/水位**（水下作业必须在高水位到来前完工）与**施工放行时间**（船闸过船、通航管制）
JOBS = [
    # 闸门组（可白天全天）
    ("闸1-底座打磨", 0, 3, 2, 6), ("闸1-止水带更换", 3, 6, 3, 9), ("闸1-喷锌防腐", 6, 10, 2, 8),
    ("闸1-启闭机调试", 10, 12, 1, 5), ("闸1-土建修补", 12, 16, 2, 7),
    ("闸2-底座打磨", 0, 3, 2, 6), ("闸2-止水带更换", 3, 6, 3, 9), ("闸2-喷锌防腐", 6, 10, 2, 8),
    ("闸2-启闭机调试", 10, 12, 1, 5), ("闸2-土建修补", 12, 16, 2, 7),
    # 栏栅组
    ("栅-水下清淤", 0, 5, 3, 14), ("栅-格栅校正", 5, 8, 2, 7), ("栅-阳极块安装", 8, 12, 2, 8),
    ("栅-水下补焊", 11, 16, 3, 11), ("栅-岸上除锈", 0, 4, 1, 7),
    # 船闸区
    ("船-闸墙注浆", 0, 8, 3, 15), ("船-航槽清障", 2, 6, 2, 10), ("船-护舷更换", 6, 10, 2, 9),
    ("船-门叶清理", 10, 14, 2, 8), ("船-测量复核", 13, 16, 1, 4),
    # 上游引航道
    ("引-拦污网检修", 0, 5, 3, 10), ("引-浮标复位", 4, 7, 1, 4), ("引-护坡勾缝", 5, 10, 2, 8),
    ("引-疏浚取样", 9, 12, 2, 7), ("引-航标灯检修", 12, 15, 1, 4),
    # 下游尾水（水位涨得快，窗口短）
    ("尾-护坦清石", 1, 3, 3, 9), ("尾-消力池检查", 3, 5, 3, 9), ("尾-导墙加固", 4, 8, 3, 11),
    ("尾-混凝土修补", 8, 12, 2, 8), ("尾-测量放样", 12, 15, 1, 4),
    # 启闭房（岸上，不受潮位限制，晚班可延后）
    ("房-配电柜改造", 0, 6, 2, 12), ("房-CCTV 调试", 5, 9, 1, 5), ("房-水位计标定", 9, 11, 1, 4),
    ("房-消防整改", 11, 15, 1, 9), ("房-网络布线", 13, 17, 1, 7), ("房-抄表建档", 16, 18, 1, 4),
]

N = len(JOBS)
S = np.array([j[1] for j in JOBS], dtype=int)
E = np.array([j[2] for j in JOBS], dtype=int)
R = np.array([j[3] for j in JOBS], dtype=int)   # 每条作业要几支班组（并行度）
W = np.array([j[4] for j in JOBS], dtype=int)
NAME = [j[0] for j in JOBS]
K = 4  # 本项目一共 4 支班组
HORIZON = int(E.max())
N = len(JOBS)
S = np.array([j[1] for j in JOBS], dtype=int)
E = np.array([j[2] for j in JOBS], dtype=int)
W = np.array([j[4] for j in JOBS], dtype=int)
NAME = [j[0] for j in JOBS]
CREW = 2  # 只有 2 支班组，同一时刻最多 2 条作业（潮位窗口已把「只涨不落」写进区间，见 §5 说明）

assert N == 36
assert (E > S).all()


def fmt(t):
    h = 6 + t * 0.5
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}"


items = [(NAME[i], int(S[i]), int(E[i]), int(W[i]), int(R[i])) for i in range(N)]


# ---------------------------------------------------------------- 2. 求解器
def greedy_earliest_finish(its, key=lambda x: x[2]):
    """区间排序（最大数量）：按「结束早」贪心；无权重时这是精确算法"""
    chosen, last = [], -math.inf
    for it in sorted(its, key=key):
        if it[1] >= last:          # 首尾相接不算重叠
            chosen.append(it)
            last = it[2]
    return chosen


def weighted_interval_dp(its):
    """加权区间排序（欧拉 DP）：p(j)=最后一个与 j 相容的区间，OPT(j)=max(OPT(j-1), w_j+OPT(p(j)))"""
    it = sorted(its, key=lambda x: x[2])
    n = len(it)
    p = []
    for j in range(n):
        k = -1
        for i in range(j):
            if it[i][2] <= it[j][1]:
                k = i
        p.append(k)
    opt = [0] * (n + 1)
    take = [False] * n
    for j in range(n):
        v_take = it[j][3] + (opt[p[j] + 1] if p[j] >= 0 else 0)
        if v_take > opt[j]:
            opt[j + 1] = v_take
            take[j] = True
        else:
            opt[j + 1] = opt[j]
    sel, j = [], n
    while j > 0:
        if take[j - 1]:
            sel.append(it[j - 1])
            j = p[j - 1] + 1
        else:
            j -= 1
    return sel, opt[n]


def exact_weighted_enum(its, cap_n=20):
    """精确：按结束排序的 2^n 子集枚举（只用于小规模核对 DP 的正确性）"""
    n = len(its)
    assert n <= cap_n
    it = sorted(its, key=lambda x: x[2])
    best, best_sel = -1, []
    for mask in range(1 << n):
        last, val, ok, sel = -math.inf, 0, True, []
        for i in range(n):
            if mask >> i & 1:
                if it[i][1] < last:
                    ok = False
                    break
                last, val = it[i][2], val + it[i][3]
                sel.append(it[i])
        if ok and val > best:
            best, best_sel = val, sel
    return best_sel, best


def profile(sel):
    """每 tick 的并行占用（人数）"""
    prof = np.zeros(HORIZON + 1, dtype=int)
    for it in sel:
        prof[it[1]:it[2]] += it[4]
    return prof


def feasible(sel, crews=K, strict=False):
    ev = []
    for it in sel:
        ev.append((it[1], it[4]))
        ev.append((it[2], -it[4]))
    ev.sort(key=lambda x: (x[0], x[1]))   # 同一时刻先释放再占用
    cur = mx = 0
    for _, d in ev:
        cur += d
        mx = max(mx, cur)
    assert len({it[0] for it in sel}) == len(sel)
    if strict:
        assert mx <= crews, f"并行班组 {mx} > {crews}"
    return True, mx


def greedy_by_start_with_crews(its, crews=K):
    """常见错解 A：按开始时间排，能塞就塞（不看价钱）"""
    prof = np.zeros(HORIZON + 1, dtype=int)
    sel = []
    for it in sorted(its, key=lambda x: (x[1], x[2])):
        if (prof[it[1]:it[2]] + it[4]).max() <= crews:
            prof[it[1]:it[2]] += it[4]
            sel.append(it)
    return sel


def greedy_by_weight_density(its, crews=K):
    """常见错解 B：按「报酬 / 时长」从高到低塞（贪心最能骗人的一个版本）"""
    prof = np.zeros(HORIZON + 1, dtype=int)
    sel = []
    for it in sorted(its, key=lambda x: -x[3] / (x[2] - x[1])):
        if (prof[it[1]:it[2]] + it[4]).max() <= crews:
            prof[it[1]:it[2]] += it[4]
            sel.append(it)
    return sel


def wrong_pairwise_pool_formulation(its):
    """常见错解 C（本讲核心）：把「两两相容」当成「可以同批」，再把同批当并行池。
    两两相容 = 区间图上的独立集；但独立集里的区间完全可以**在时间上交叉**，
    只要 (a) 能两两配对、或 (b) 把重边放大倍数改系数。两种做法都会给出可行域过大的假最优。"""
    from itertools import combinations
    pool_value = 0
    pool = []
    for i, j in combinations(range(len(its)), 2):
        a, b = its[i], its[j]
        if a[2] <= b[1] or b[2] <= a[1]:
            # 「两两相容 → 可同批」：直接把两人的报酬相加当成一批的收益（错在忽略了需 3 支班组的作业）
            if a[4] + b[4] <= K and a[3] + b[3] > pool_value:
                pool_value, pool = a[3] + b[3], [a, b]
    return pool, pool_value


def pooled_parallel_greedy(its, crews=K, weights_scaled=True):
    """错解 C 的「求解版」：只检查两两是否相容，而不检查同一时刻的**总人数**。
    做法：按时间窗把两两相容的作业合成一个「批次」，批次内当作并行池，只要 w_i+w_j ≤ 4 就同批开工。"""
    order = sorted(its, key=lambda x: (x[1], x[2]))
    prof = np.zeros(HORIZON + 1, dtype=int)   # 按「条数」计数（而不是按人数）—— 这就是错在哪
    sel = []
    for it in order:
        # 只看：已有几条在同一时刻开工（条数 ≤ 4 就行），不看每条要几支班组
        if prof[it[1]:it[2]].max() < crews:
            prof[it[1]:it[2]] += 1
            sel.append(it)
    return sel


# ---------------------------------------------------------------- 3. 四种口径
t0 = time.perf_counter()

# ① 单班组串行、只看条数 → 区间排序
g_count = greedy_earliest_finish(items)

# ② 单班组串行、带价钱 → 加权区间 DP
dp_sel, dp_val = weighted_interval_dp(items)

# ③ 4 支班组（每条作业需 1~3 支）、带价钱 → 按人数判可行 + DP 不成立，改用精确/启发式
g_start = greedy_by_start_with_crews(items)
g_density = greedy_by_weight_density(items)
g_counted = pooled_parallel_greedy(items)

# ④ 单班组口径下 DP 的精确核对（取前 18 条按结束时间）
small = sorted(items, key=lambda x: x[2])[:18]
ex_sel, ex_val = exact_weighted_enum(small)
dp_small_sel, dp_small_val = weighted_interval_dp(small)

t_solve = time.perf_counter() - t0

for nm, sel, strict in (("贪心条数", g_count, True), ("DP", dp_sel, True),
                        ("按开始", g_start, True), ("按密度", g_density, True),
                        ("按条数池(错解C)", g_counted, False)):
    ok, mx = feasible(sel, strict=strict)
    assert ok
    print(f"[可行] {nm}: 条数 {len(sel)}, 价值 {sum(it[3] for it in sel)}, 并发 {mx}")
assert ex_val == dp_small_val, (ex_val, dp_small_val)
# 教学点断言：错解 C 必须「看起来更贵但实际并行超限」
assert sum(it[3] for it in g_counted) > dp_val
assert profile(g_counted).max() > K
# 错解 C：只数条数、不数人数 → 账面看起来最好，真实并发必然超限
print("[核对] 错解C 报账收益", sum(it[3] for it in g_counted),
      "；按人数算峰值", int(profile(g_counted).max()), "> 4 支班组 → 方案不可行")
# 关键反例：同一份数据，「只看条数」给出 6 条 / 价值 34，「看人数」给出 6 条 / 价值 44
#   → 条数相同、名单不同：**最优化问题一旦把「条数」当 K，就不再是最初那个问题**
assert len(g_count) == len(dp_sel) and int(sum(it[3] for it in g_count)) < int(dp_val)

# ---------------------------------------------------------------- 3.5 4 班组口径的精确解（区间图 + 时间维 DP）
# 观察：任何时刻在作业数 ≤ K ⇒ 每条作业的开工时刻必落在 {某条作业的开始, 某条作业的结束} 里，
# 于是按时间推进、用「仍在进行的 4 条作业各自的结束时刻」当状态（至多 4 维），状态空间极小 → 精确到毫秒。
def exact_crew_dp(its, crews=K):
    chosen = []          # 记录入选作业（供报表用）
    it = sorted(its, key=lambda x: (x[1], x[2]))
    allpts = sorted({x[1] for x in it} | {x[2] for x in it})
# ---------------------------------------------------------------- 3.5 4 班组口径的精确解（区间图 + 时间维 DP）
# 观察：任何时刻在作业数 ≤ K ⇒ 每条作业的开工时刻必落在 {某条作业的开始, 某条作业的结束}，
# 于是按时间推进、用「仍在进行的 K 条作业各自的结束时刻」当状态（至多 K 维），状态空间极小 → 精确到毫秒。
# 这是区间图「最大权 K-可着色子集」的直接 DP，不需要 ILP，也不需要 2^36 枚举。
def exact_crew_dp(its, crews=K):
    it = sorted(its, key=lambda x: (x[1], x[2]))
    allpts = sorted({x[1] for x in it} | {x[2] for x in it})
    cur = {(allpts[0], ()): 0}
    n_states = 0
    for t in allpts:
        nxt = {}
        # ① 释放：把已结束的作业从「在进行」里拿掉
        for (tt, busy), val in cur.items():
            b = tuple(x for x in busy if x > t)
            if nxt.get((t, b), -1) < val:
                nxt[(t, b)] = val
        # ② 在 t 时刻开工的作业（同刻可能有多条，逐条决策）
        for job in [x for x in it if x[1] == t]:
            r, e, w = job[4], job[2], job[3]
            for (tt, busy), val in list(nxt.items()):
                if crews - len(busy) < r:
                    continue
                k = (t, busy + (e,) * r)
                if nxt.get(k, -1) < val + w:
                    nxt[k] = val + w
        n_states = max(n_states, len(nxt))
        cur = nxt
    key = max(cur, key=lambda k: cur[k])
    return cur[key], n_states, len(cur)


t_ex = time.perf_counter()
ex4_val, n_states, n_final = exact_crew_dp(items, K)
t_ex = time.perf_counter() - t_ex
for name, sel in (("按开始", g_start), ("按密度", g_density)):
    v = sum(x[3] for x in sel)
    assert v <= ex4_val, f"{name}贪心 {v} 超过精确解 {ex4_val} —— 精确解或可行性判定写错了"
print(f"[核对] 4 班组精确解 {ex4_val}（状态峰值 {n_states}，耗时 {t_ex * 1000:.1f} ms）；"
      f"贪心按开始 {sum(x[3] for x in g_start)}、按密度 {sum(x[3] for x in g_density)}")

# 敏感性：班组数 K = 1,2,3,4,5,6,8（每档都用精确 DP）
sens = {}
for k in (1, 2, 3, 4, 5, 6, 8):
    val, st, _ = exact_crew_dp(items, k)
    sens[k] = {"最优收益": int(val), "状态峰值": st,
               "贪心(按开始)收益": int(sum(it[3] for it in greedy_by_start_with_crews(items, k)))}

# 需求剖面
demand = np.zeros(HORIZON + 1, dtype=int)
for it in items:
    demand[it[1]:it[2]] += it[4]
peak = int(demand.max())

# ---------------------------------------------------------------- 5. 出图
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0])
sel_dp = {it[0] for it in dp_sel}
sel_gs = {it[0] for it in g_start}
for i in range(N):
    if NAME[i] in sel_dp:
        col = "#4c72b0"      # 单班组加权 DP 选中
    elif NAME[i] in sel_gs:
        col = "#dd8452"      # 4 班组贪心选中
    else:
        col = "#d9d9d9"
    ax.barh(i, E[i] - S[i], left=S[i], height=0.7, color=col,
            edgecolor="black", linewidth=0.4)
    ax.text(E[i] + 0.12, i, NAME[i], va="center", fontsize=7)
ax.set_yticks([])
ax.invert_yaxis()
ax.set_xlabel("开工后半小时数（06:00 起算）")
ax.set_title("36 条检修作业：蓝 = 单班组加权 DP 选中 / 橙 = 4 班组贪心选中 / 灰 = 未选", fontsize=10)
ax.set_xlim(0, HORIZON + 6)

ax2 = fig.add_subplot(gs[1])
xs = np.arange(HORIZON + 1)
ax2.step(xs, demand, where="post", lw=2.0, color="#333333",
         label="每刻需求班组数（峰值 %d）" % peak)
for k, c in ((4, "#c44e52"), (5, "#8172b2"), (6, "#55a868")):
    ax2.axhline(k, ls="--", lw=1.0, color=c, label=f"K={k} 时可用 {k} 支")
ax2.fill_between(xs, 0, demand, step="post", alpha=0.15, color="#333333")
t_bad = np.where(demand > K)[0]
ax2.scatter(t_bad, demand[t_bad], color="#c44e52", zorder=5, s=18)
ax2.set_xlabel("时刻（半小时数）")
ax2.set_ylabel("需求班组数")
ax2.set_title("需求剖面：总工时够 ≠ 峰值够（红点 = 4 支班组扛不住的时段）", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "day68_intervals.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- 6. 结果
out = {
    "实例": {"作业数": N, "总工时tick": int((E - S).sum()),
             "总工时小时": round(float((E - S).sum()) * TICK_MIN / 60, 2),
             "班组数K": K, "需求峰值": peak},
    "①单班组_只看条数": {"贪心条数": len(g_count), "贪心价值": int(sum(it[3] for it in g_count)),
                     "名单": [it[0] for it in g_count]},
    "②单班组_加权DP": {"价值": int(dp_val), "条数": len(dp_sel), "名单": [it[0] for it in dp_sel]},
    "②核对_枚举vsDP": {"n": len(small), "枚举": int(ex_val), "DP": int(dp_small_val),
                      "一致": bool(ex_val == dp_small_val)},
    "③4班组_按开始时间贪心": {"价值": int(sum(it[3] for it in g_start)), "条数": len(g_start),
                        "名单": [it[0] for it in g_start]},
    "③4班组_按密度贪心": {"价值": int(sum(it[3] for it in g_density)), "条数": len(g_density)},
    "③4班组_错解C(只数条数)": {"账面价值": int(sum(it[3] for it in g_counted)),
                           "条数": len(g_counted),
                           "真实并发": int(profile(g_counted).max()),
                           "可行": bool(profile(g_counted).max() <= K)},
    "③4班组_精确解(时间维DP)": {"价值": int(ex4_val), "状态峰值": n_states,
                             "耗时毫秒": round(t_ex * 1000, 2),
                             "贪心缺口": int(ex4_val - sum(it[3] for it in g_start))},
    "④班组敏感性": sens,
    "耗时秒": round(t_solve, 4),
}
with open(HERE / "day68_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
print("全部断言通过")
