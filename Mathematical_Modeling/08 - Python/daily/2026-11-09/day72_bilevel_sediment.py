# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化/决策模型（**双层规划 bi-level** · Stackelberg 主从决策）/ L8 算法（响应台阶 · 下层全枚举裁判）
anchor  : Roadmap L4 优化模型。前接 Day60 / Day69 / Day71（**一个人拍板**：覆盖、分层覆盖、后悔准则）、
          Day64 VCG（**规则**是决策变量，但参与者只改报价、不重新优化）、Day27 报童（一个决策者 + 随机需求）。
          本讲补全库零命中的一族：**上层定规则、下层据此重新求最优** —— 主从博弈 / 双层规划。
goal    : 云溪县 2027 年「水源地保护 vs 林场采伐」。县水务局（上层）每年给 3 个水源地逐条批「采伐面积额度」，
          林场（下层）在额度内自己挑林班采、追求木材利润最大。四个问题：
          1) 上层只批最下游那个水源的额度，能保住下游取水口吗？
          2) 「三个水源共用一条线」和「每个水源一个数」差多少？
          3) 下层对额度的响应是曲线还是台阶？「多批 40 公顷」到底值多少钱？
          4) 上游泥沙传递系数从 0.2 变到 1.0，同一个额度的含金量变多少？
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-09/day72_bilevel_sediment.py"
输出：day72_result.json、day72_bilevel.png
"""
import itertools
import json
import time
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
# 1. 实例：一条河上的 3 个取水口 + 10 个林班（全部硬编码，可复现）
# ----------------------------------------------------------------------------
SRC = ["城关(最上游)", "石桥(中游)", "云岭(最下游)"]
BASIN = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 2])                       # 林班所属流域
AREA = np.array([260, 310, 180, 240, 150, 290, 220, 330, 140, 280.])    # 公顷
SED = np.array([2.1, 2.6, 3.4, 4.1, 5.2, 3.0, 4.4, 2.8, 6.0, 3.7])      # 吨泥沙/公顷
PROF_AREA = np.array([0.64, 0.56, 0.82, 0.48, 0.38, 0.76, 0.52, 0.86, 0.30, 0.68])  # 万元/公顷
BASE_NTU = np.array([0.40, 0.60, 0.90])
DECAY = 0.25                       # 泥沙 → 浊度：0.25 NTU/吨
TRANS = 0.80                       # 上游泥沙被下游承接比例
LIMIT = np.array([1.05, 1.45, 99.0])
W_NTU = 1.0                        # 每 NTU 的水厂加药成本（万元/年）
N = len(AREA)

def scheme(bits, trans=TRANS):
    x = np.array(bits, float)
    reg_area = np.bincount(BASIN, weights=AREA * x, minlength=3)
    profit = float(PROF_AREA * AREA @ x)
    own = np.bincount(BASIN, weights=SED * AREA * x, minlength=3) / 1000.0
    ntu = BASE_NTU + DECAY * own
    ntu[1] += trans * ntu[0]
    ntu[2] += trans * ntu[1]
    return reg_area, profit, own, ntu

ALL_BITS = list(itertools.product([0, 1], repeat=N))
SCHEMES = [scheme(b) for b in ALL_BITS]
REG_AREA = np.array([s[0] for s in SCHEMES])
PROF = np.array([s[1] for s in SCHEMES])
OWN = np.array([s[2] for s in SCHEMES])
NTU = np.array([s[3] for s in SCHEMES])
NAMES = [tuple(i for i in range(N) if b[i]) for b in ALL_BITS]
assert len(SCHEMES) == 2 ** N == 1024

OK = (NTU[:, 0] <= LIMIT[0] + 1e-9) & (NTU[:, 1] <= LIMIT[1] + 1e-9)
K_FREE = int(PROF.argmax())
K_OK = int(np.where(OK)[0][PROF[OK].argmax()])

def fmt(k):
    return "+".join("R%d" % (i + 1) for i in NAMES[k])

# ----------------------------------------------------------------------------
# 2. 下层（林场）：给定额度，从 1024 个方案里挑利润最高的可行方案
# ----------------------------------------------------------------------------
def follower(cap=None, trans=None):
    """cap=(A0,A1,A2)：三个水源允许采伐的公顷数（None = 不限）。返回 (利润, 方案下标)。"""
    if trans is None:
        own, prof = OWN, PROF
        ok = OK.copy()
    else:
        sc = [scheme(b, trans) for b in ALL_BITS]
        prof = PROF
        ok = np.array([(s[3][0] <= LIMIT[0] + 1e-9) and (s[3][1] <= LIMIT[1] + 1e-9) for s in sc])
    if cap is not None:
        ok = ok & (REG_AREA <= np.asarray(cap, float) + 1e-9).all(axis=1)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return None, None
    k = int(idx[prof[idx].argmax()])
    return float(prof[k]), k

def upper_value(k, w=W_NTU):
    """上层目标：木材收益 − 水质代价（取水口浊度折算成水厂加药费）。"""
    cost = w * (NTU[k, 0] + NTU[k, 1] + 0.5 * NTU[k, 2])
    return PROF[k] - cost, cost

# ----------------------------------------------------------------------------
# 3. 上层：三种批法，全部穷举 + 下层精确响应
# ----------------------------------------------------------------------------
GRID_COARSE = np.arange(100, 861, 20)            # 三水源共用的粗网格
GRID_FINE = np.arange(700, 861, 10)              # 下游那一维加密（860 就是 R9 整块面积）
t0 = time.perf_counter()
CAP_TABLE = {}
for a0 in GRID_COARSE:
    for a1 in GRID_COARSE:
        for a2 in GRID_FINE:
            if (a0, a1, a2) in CAP_TABLE:
                continue
            net, k = follower((a0, a1, a2))
            if net is None:
                continue
            CAP_TABLE[(int(a0), int(a1), int(a2))] = (net, k, upper_value(k)[0])
cap_time = time.perf_counter() - t0

LOOSE = float(GRID_FINE[-1])
GRID = np.sort(np.unique(np.concatenate([GRID_COARSE, GRID_FINE])))
UNI = [(float(t), *follower((t, t, t))) for t in GRID_FINE]          # 一条线：本来就要最细的档
DOWN_ONLY = [(float(a2), *follower((LOOSE, LOOSE, a2))) for a2 in GRID_FINE]
UP_ONLY = [(float(a0), *follower((a0, LOOSE, LOOSE))) for a0 in GRID_COARSE]
MID_ONLY = [(float(a1), *follower((LOOSE, a1, LOOSE))) for a1 in GRID_COARSE]

_UMAX = max(v[2] for v in CAP_TABLE.values())
# 上层最优有并列（多个额度组合都能拿下同一批林子）：取「总额度最小」的那个，即不白白多批
STAR = min(((v[2], key, v[1], v[0]) for key, v in CAP_TABLE.items() if v[2] >= _UMAX - 0.5),
           key=lambda r: sum(r[1]))
BEST_UNI = max([(upper_value(k)[0], t, net, k) for t, net, k in UNI if k is not None])
BEST_DOWN = max([(upper_value(k)[0], a, net, k) for a, net, k in DOWN_ONLY if k is not None])
BEST_UP = max([(upper_value(k)[0], a, net, k) for a, net, k in UP_ONLY if k is not None])

# 响应台阶：主案额度按比例放大（0.05 步长，40 档），看下层换方案的点
STEPS, SWITCH = [], []
prev = -1
for lam in np.round(np.arange(0.20, 1.61, 0.05), 2):
    cap = tuple(float(v) * lam for v in STAR[1])
    net, k = follower(cap)
    STEPS.append((float(lam), cap, net, k))
    if k is not None and k != prev:
        SWITCH.append((float(lam), cap, float(net), int(k)))
        prev = k
assert len(STEPS) == 29, len(STEPS)

TR_ROWS = []
for tr in (0.2, 0.4, 0.6, 0.8, 1.0):
    net_s, k_s = follower(STAR[1], trans=tr)
    net_f, k_f = follower(None, trans=tr)
    TR_ROWS.append((tr, net_s, k_s, net_f, k_f))

# ----------------------------------------------------------------------------
# 4. 断言：把教学点钉死在代码里（换参数就会红）
# ----------------------------------------------------------------------------
assert PROF[K_FREE] > PROF[K_OK], "红线必须真的顶掉利润"
# 教学点 1：只批下游 = 拿最下游自己的林子抵账（曲线单调、从不站上守红线解）
assert all(n is not None for _, n, _ in DOWN_ONLY)
assert abs(max(n for _, n, _ in DOWN_ONLY) - PROF[K_OK]) < 1e-6, "只批下游也能达到守红线解（但要付出下游自己的林子）"
# 教学点 2：统一额度只能到 1092，守红线解本身 1134 → 一条线套不住三个水源
assert BEST_UNI[2] <= PROF[K_OK] + 1e-6, "统一额度至多等于守红线解"
AT420 = follower((420, LOOSE, LOOSE))[0]
UNI420 = follower((420, 420, 420))[0]
assert AT420 > UNI420 + 300, "同样的 420 公顷，只卡上游比一条线好得多"
# 教学点 3：台阶
assert 5 <= len(SWITCH) <= 20, len(SWITCH)
gaps = np.array([SWITCH[i + 1][0] - SWITCH[i][0] for i in range(len(SWITCH) - 1)])
platforms = np.array([SWITCH[i + 1][2] - SWITCH[i][2] for i in range(len(SWITCH) - 1)])
assert gaps.max() >= 0.15 - 1e-9, "至少一处：额度挪了 15% 方案不动"
assert platforms.max() >= 200, "至少一处：跨过一道台阶利润直接涨 200 万以上"
assert gaps.max() >= 0.15 - 1e-9, "至少一处：额度挪了 15% 方案不动"
# 教学点 4：传递系数
assert TR_ROWS[0][1] > TR_ROWS[-1][1], "传递系数越大，主案额度下利润越低"

# ----------------------------------------------------------------------------
# 5. 输出
# ----------------------------------------------------------------------------
def pack(rows):
    return [[a, None if n is None else round(n, 1), None if k is None else fmt(k)] for a, n, k in rows]
def area(k):
    return [int(round(v)) for v in REG_AREA[k]]

jsonr = {
    "林场无约束最优": {"profit": round(float(PROF[K_FREE]), 1), "plan": fmt(K_FREE),
                 "area": area(K_FREE), "ntu": [round(float(v), 2) for v in NTU[K_FREE]]},
    "守红线的最优": {"profit": round(float(PROF[K_OK]), 1), "plan": fmt(K_OK),
                "area": area(K_OK), "ntu": [round(float(v), 2) for v in NTU[K_OK]],
                "距无约束": round(float(PROF[K_FREE] - PROF[K_OK]), 1)},
    "主案_分水源额度": {"cap": list(STAR[1]), "profit": round(STAR[3], 1), "plan": fmt(STAR[2]),
                 "area": area(STAR[2]),
                 "ntu": [round(float(v), 2) for v in NTU[STAR[2]]],
                 "upper": round(STAR[0], 2), "water": round(upper_value(STAR[2])[1], 2)},
    "统一额度最优": {"cap": BEST_UNI[1], "profit": round(BEST_UNI[2], 1), "plan": fmt(BEST_UNI[3]),
               "area": area(BEST_UNI[3]),
               "ntu": [round(float(v), 2) for v in NTU[BEST_UNI[3]]], "upper": round(BEST_UNI[0], 2)},
    "只批下游最优": {"cap": BEST_DOWN[1], "profit": round(BEST_DOWN[2], 1), "plan": fmt(BEST_DOWN[3]),
               "upper": round(BEST_DOWN[0], 2)},
    "只批上游最优": {"cap": BEST_UP[1], "profit": round(BEST_UP[2], 1), "plan": fmt(BEST_UP[3]),
               "area": area(BEST_UP[3]),
               "ntu": [round(float(v), 2) for v in NTU[BEST_UP[3]]], "upper": round(BEST_UP[0], 2)},
    "统一额度扫描": pack(UNI),
    "只批下游扫描": pack(DOWN_ONLY),
    "只批上游扫描": pack(UP_ONLY),
    "只批中游扫描": pack(MID_ONLY),
    "台阶": [[lam, [int(round(c)) for c in cap], round(n, 1), fmt(k)] for lam, cap, n, k in SWITCH],
    "传递敏感性": [[tr, None if ns is None else round(ns, 1), None if ks is None else fmt(ks),
              None if nf is None else round(nf, 1), None if kf is None else fmt(kf)]
             for tr, ns, ks, nf, kf in TR_ROWS],
    "耗时": {"额度枚举": round(cap_time, 2), "总计": round(time.perf_counter() - T0, 2)},
}
(HERE / "day72_result.json").write_text(json.dumps(jsonr, ensure_ascii=False, indent=1), encoding="utf-8")

print("=== 一、林场自己会选什么 ===")
print("无约束 %s 利润 %.0f 面积 %s 浊度 %s" % (jsonr["林场无约束最优"]["plan"], PROF[K_FREE],
                                          jsonr["林场无约束最优"]["area"], jsonr["林场无约束最优"]["ntu"]))
print("守红线 %s 利润 %.0f 面积 %s 浊度 %s（少赚 %.0f 万）" % (
    jsonr["守红线的最优"]["plan"], PROF[K_OK], jsonr["守红线的最优"]["area"],
    jsonr["守红线的最优"]["ntu"], PROF[K_FREE] - PROF[K_OK]))
print("\n=== 二、三种批法 ===")
for name in ("主案_分水源额度", "只批上游最优", "统一额度最优", "只批下游最优"):
    d = jsonr[name]
    print("%-12s 额度 %-22s 利润 %7.1f 面积 %-16s 浊度 %s" % (
        name, str(d["cap"]), d["profit"], str(d.get("area", "-")), d.get("ntu", "-")))
print("\n=== 三、统一额度扫描 ===")
for a, n, p in UNI:
    print("  A=%4.0f  %s" % (a, "无解" if n is None else "利润 %6.0f  %s" % (n, p)))
print("\n=== 四、台阶（额度 × lam） ===")
for lam, cap, n, p in SWITCH:
    print("  ×%.2f  额度 %s  利润 %7.0f  %s" % (lam, [int(round(c)) for c in cap], n, p))
print("\n=== 五、传递系数 ===")
for tr, ns, ks, nf, kf in TR_ROWS:
    print("  tr=%.1f  主案额度下 %6s  无约束 %6s" % (tr, "—" if ns is None else "%.0f" % ns,
                                             "—" if nf is None else "%.0f" % nf))
print("\n耗时 %.2f s（额度枚举 %.2f s / %d 组）" % (jsonr["耗时"]["总计"], cap_time, len(CAP_TABLE)))

# ----------------------------------------------------------------------------
# 6. 图：四面板
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

ax = axes[0, 0]
xs = [r[0] for r in UNI]
ys = [np.nan if r[1] is None else r[1] for r in UNI]
ax.step(xs, ys, where="post", lw=2.2, color="#2b6cb0", label="下层最优利润（统一额度 A）")
ax.plot([BEST_UNI[1]], [BEST_UNI[2]], "o", ms=10, mfc="none", mec="#c53030", mew=2,
        label="一条线的尽头：A=%.0f，/%.0f" % (BEST_UNI[1], BEST_UNI[2]))
ax.axhline(PROF[K_OK], ls="--", color="#38a169", lw=1.5, label="守红线解 %.0f" % PROF[K_OK])
ax.set_xlabel("统一额度 A（公顷/水源）")
ax.set_ylabel("林场利润（万元/年）")
ax.set_title("① 统一额度：台阶上爬到 %.0f，前半段几乎白给" % BEST_UNI[2])
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

ax = axes[0, 1]
for rows, color, label in ((UP_ONLY, "#c53030", "只批上游 A0"),
                            (MID_ONLY, "#dd6b20", "只批中游 A1"),
                            (DOWN_ONLY, "#2b6cb0", "只批下游 A2")):
    ax.step([r[0] for r in rows], [np.nan if r[1] is None else r[1] for r in rows],
            where="post", lw=2.0, color=color, label=label)
ax.axhline(PROF[K_FREE], ls=":", color="gray", lw=1.2, label="无约束 %.0f" % PROF[K_FREE])
ax.set_xlabel("该水源的额度（公顷，其余两个放开）")
ax.set_ylabel("林场利润（万元/年）")
ax.set_title("② 三个水源各卡一条：下游卡到 860 才解锁 R9")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1, 0]
labels = ["林场无约束", "统一额度", "只批上游", "守红线解", "分水源额度"]
vals = [PROF[K_FREE], BEST_UNI[2], BEST_UP[2], PROF[K_OK], STAR[3]]
cols_ = ["#a0aec0", "#805ad5", "#c53030", "#38a169", "#2f855a"]
bars = ax.bar(labels, vals, color=cols_)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 15, "%.0f" % v, ha="center", fontsize=10)
ax.set_ylim(0, PROF[K_FREE] * 1.18)
ax.set_ylabel("林场利润（万元/年）")
ax.set_title("③ 三条路都能到 %.0f：但额度总额差 %.0f 公顷" % (PROF[K_OK], sum(STAR[1]) - 3 * BEST_UNI[1]))
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 1]
lam = [r[0] for r in STEPS]
pv = [np.nan if r[2] is None else r[2] for r in STEPS]
ax.step(lam, pv, where="post", lw=2.4, color="#2f855a")
ax.axhline(PROF[K_OK], ls="--", color="#38a169", lw=1.2, label="守红线解 %.0f" % PROF[K_OK])
ax.axhline(PROF[K_FREE], ls=":", color="gray", lw=1.2, label="无约束 %.0f" % PROF[K_FREE])
for i in range(len(SWITCH) - 1):
    ax.annotate("+%.0f" % (SWITCH[i + 1][2] - SWITCH[i][2]),
                (SWITCH[i + 1][0], SWITCH[i + 1][2]), fontsize=8, color="#2f855a",
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("主案额度 × lam（面积等比放缩）")
ax.set_ylabel("林场利润（万元/年）")
ax.set_title("④ 台阶：限额挪 15%% 方案不动，跨一步 +%.0f 万" % platforms.max())
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

fig.suptitle("Day72 · 双层规划：上层批额度，下层重新选片 —— 同一份数据，三种批法三个结局", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE / "day72_bilevel.png", dpi=140)
print("图已存：", HERE / "day72_bilevel.png")
