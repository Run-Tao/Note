#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day47 - 置换流水车间调度 FSP：NEH 构造 + 迭代贪心 IG（破坏-重建）
subject : mathematical-modeling
level   : L5 算法（元启发式 / 构造-局部搜索）/ L6 优化模型（组合优化 / 调度）
anchor  : 05 - Algorithms（算法主线：调度元启发式）
goal    : 用「NEH 构造 + 迭代贪心 IG」求解一道「县域果酱加工厂 5 道工序、14 批
          原料的流水车间排产」决策问题（置换流水车间 FSP，最小化 makespan）——
          ① 讲清「为什么瓶颈设备上把长活排前面」反而拖垮全局（候选规则全部吃亏）；
          ② NEH：按总加工时间降序，逐个插到「当前最好的位置」——构造即高质量的奥秘；
          ③ 迭代贪心 IG：破坏 d 个工序 → 用 NEH 贪心重建 → 接受准则，在 NEH 之上再进步；
          ④ 用「暴力全排列（小规模）+ 多起点随机排产」当独立裁判交叉验证；
          ⑤ 用 CP-SAT（scipy.milp 无调度原生支持，故用 OR-tools 可选）不做——仅用
             独立枚举与多起点做裁判，保持零专用调度库。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "<本文件路径>"
"""

import os
import time
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------
# 中文字体注册（Noto Sans CJK SC / 文泉驿），避免图中中文乱码
# ----------------------------------------------------------------------
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_registered = []
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            _registered.append(font_manager.FontProperties(fname=_p).get_name())
        except Exception:
            pass
if _registered:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _registered + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
print("CJK 字体已注册：", _registered if _registered else "（无，可能乱码）")

RNG_SEED = 20261017
RNG = np.random.default_rng(RNG_SEED)

# ======================================================================
# 一、真实决策问题：县域果酱加工厂的「流水车间排产」
# ======================================================================
# 背景：一家县域果酱厂有 5 台设备**串联成一条流水线**：
#   0 清洗去核 → 1 蒸煮浓缩 → 2 灌装封口 → 3 贴标打码 → 4 装箱入库
# 今天来了 14 批原料（不同水果/不同规格），每批都要**按同样顺序**走完 5 道工序。
# 设备不能并行处理两批；一批在某道工序做完才能进下一道（不等待、不插队）。
#
#   ★ 决策：14 批原料的加工顺序（一个排列）。
#   ★ 目标：最后一批装箱入库的完工时刻（makespan）最早 —— 整条线尽早收工。
#
# 这就是经典「置换流水车间调度（Permutation Flow Shop, FSP）」，n≥3 时 NP-hard。
# 现实含义：makespan 每缩 1 小时，就是 5 台设备、14 名工位人手少开 1 小时。

STAGES = 5
N_JOBS = 14
STAGE_NAMES = ["清洗去核", "蒸煮浓缩", "灌装封口", "贴标打码", "装箱入库"]

# 加工时间矩阵 p[j, m]（小时）：刻意让 蒸煮浓缩(m=1) 是瓶颈工序
# （各批在该工序耗时最长、且波动大 → 顺序影响最大），贴标打码(m=3) 很短。
P_BASE = np.array([
    #  m0  m1  m2  m3  m4
    [ 3,  9,  4,  1,  2],   # 草莓
    [ 2, 11,  3,  1,  2],   # 蓝莓
    [ 4,  7,  5,  2,  3],   # 苹果
    [ 3, 12,  4,  1,  2],   # 山楂
    [ 2,  8,  3,  1,  2],   # 梨
    [ 5, 10,  6,  2,  3],   # 黄桃
    [ 3,  6,  4,  1,  2],   # 葡萄
    [ 4, 13,  5,  2,  3],   # 桑葚
    [ 2,  7,  3,  1,  2],   # 枇杷
    [ 3,  9,  4,  1,  2],   # 杏
    [ 4, 11,  5,  1,  3],   # 无花果
    [ 2,  8,  3,  2,  2],   # 樱桃
    [ 3, 10,  4,  1,  2],   # 覆盆子
    [ 4,  9,  6,  2,  3],   # 柑橘
], dtype=float)
assert P_BASE.shape == (N_JOBS, STAGES)

# ======================================================================
# 二、核心评价函数：给定顺序 → makespan（以及每道工序/每批的开始时刻）
# ======================================================================
def schedule(perm, P):
    """置换流水车间：按 perm 的顺序依次加工，返回 (makespan, 开始时刻表 C)。"""
    perm = np.asarray(perm, dtype=int)
    n, m = len(perm), P.shape[1]
    # C[i, k] = 第 i 个（按顺序）作业在第 k 台设备上的完工时刻
    C = np.zeros((n, m))
    for i in range(n):
        job = perm[i]
        for k in range(m):
            start = 0.0
            if i > 0:               # 上一批在本工序结束
                start = max(start, C[i - 1, k])
            if k > 0:               # 本批在上一道工序结束
                start = max(start, C[i, k - 1])
            C[i, k] = start + P[job, k]
    return C[-1, -1], C


def makespan(perm, P):
    return schedule(perm, P)[0]

# ======================================================================
# 三、候选规则（工程师凭直觉会想到的排产法）
# ======================================================================
def rule_spt(perm, P):
    """SPT：总加工时间最短的先做（交期最短优先，最省平均等待）。"""
    return np.argsort(P.sum(axis=1))


def rule_lpt(perm, P):
    """LPT：总加工时间最长的先做（经典「瓶颈设备先喂长活」）。"""
    return np.argsort(-P.sum(axis=1))


def rule_bottleneck(perm, P):
    """瓶颈优先：按瓶颈工序（蒸煮浓缩）耗时最长的先做。"""
    return np.argsort(-P[:, 1])


def rule_eedd_like(perm, P):
    """按「前段工序总和 − 后段工序总和」排序（Johnson 规则推广到多工序的近似）。"""
    return np.argsort(P[:, 0] - P[:, -1])

# ======================================================================
# 四、NEH 构造启发式（Nawaz–Enscore–Ham, 1983）
# ======================================================================
def neh(P, order=None):
    """NEH：按总加工时间降序，逐个把作业插入到使当前 makespan 最小的位置。

    复杂度 O(n^2·m) 的朴素实现，n=14 秒级完成。
    """
    n, m = P.shape
    if order is None:
        order = np.argsort(-P.sum(axis=1))  # 总工时降序
    seq = [int(order[0])]
    for idx in range(1, n):
        job = int(order[idx])
        best_ms, best_pos = np.inf, 0
        for pos in range(len(seq) + 1):
            cand = np.array(seq[:pos] + [job] + seq[pos:], dtype=int)
            ms = makespan(cand, P)
            if ms < best_ms - 1e-9:
                best_ms, best_pos = ms, pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return np.array(seq, dtype=int)

# ======================================================================
# 五、迭代贪心 Iterated Greedy（Ruiz & Stützle, 2007）
# ======================================================================
def best_insert(seq, job, P):
    """把 job 插入 seq 的某个位置，使 makespan 最小；返回 (新序列, ms)。"""
    seq = list(np.asarray(seq, dtype=int))
    best_ms, best_seq = np.inf, None
    for pos in range(len(seq) + 1):
        cand = seq[:pos] + [job] + seq[pos:]
        ms = makespan(cand, P)
        if ms < best_ms - 1e-9:
            best_ms, best_seq = ms, cand
    return best_seq, best_ms


def iterated_greedy(P, init_seq, iterations=150, d=4, seed=RNG_SEED, T_factor=0.4):
    """迭代贪心（IG）：破坏（随机移除 d 个作业）→ 重建（逐个最优插入）→ 接受准则。

    接受准则：更好必收；变差则以 temperature = T_factor × makespan/n/... 的
    Metropolis 概率接受（与模拟退火同源，但邻域是「破坏-重建」而非「交换」）。
    """
    rng = np.random.default_rng(seed)
    cur = list(np.asarray(init_seq, dtype=int))
    cur_ms = makespan(cur, P)
    best, best_ms = list(cur), cur_ms
    history = [cur_ms]

    # 温度取 「一个作业×一台设备的平均工时」量级，保证初期能翻过小坎
    temp = T_factor * P.mean()

    for _ in range(iterations):
        # ---- 破坏：随机挑 d 个作业拿出来 ----
        removed_idx = rng.choice(len(cur), size=min(d, len(cur)), replace=False)
        removed = [cur[i] for i in removed_idx]
        partial = [x for i, x in enumerate(cur) if i not in set(removed_idx)]
        rng.shuffle(removed)  # 随机化重建顺序

        # ---- 重建：逐个插回最优位置 ----
        cand = list(partial)
        for job in removed:
            cand, _ = best_insert(cand, job, P)
        cand_ms = makespan(cand, P)

        # ---- 接受准则 ----
        if cand_ms < cur_ms - 1e-9 or rng.random() < np.exp(-(cand_ms - cur_ms) / max(temp, 1e-9)):
            cur, cur_ms = list(cand), cand_ms
            if cur_ms < best_ms - 1e-9:
                best, best_ms = list(cur), cur_ms
        history.append(best_ms)
    return np.array(best, dtype=int), best_ms, history

# ======================================================================
# 六、独立裁判：小规模暴力全排列 + 多起点随机排产
# ======================================================================
def brute_force(P):
    """暴力枚举全部 n! 个排列（仅用于 n<=8 的小规模子问题）。"""
    n = P.shape[0]
    best_ms, best_perm = np.inf, None
    for perm in itertools.permutations(range(n)):
        ms = makespan(perm, P)
        if ms < best_ms - 1e-9:
            best_ms, best_perm = ms, perm
    return best_perm, best_ms


def random_multistart(P, n_trials=2000, seed=RNG_SEED):
    """随机多重起点：随机排列取最好者（判断「顺序到底有多重要」）。"""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    best_ms, best_perm = np.inf, None
    vals = []
    for _ in range(n_trials):
        perm = rng.permutation(n)
        ms = makespan(perm, P)
        vals.append(ms)
        if ms < best_ms - 1e-9:
            best_ms, best_perm = ms, perm
    return best_perm, best_ms, np.array(vals)

# ======================================================================
# 七、主实验
# ======================================================================
print("=" * 72)
print("Day47 · 置换流水车间调度 FSP：NEH 构造 + 迭代贪心 IG")
print("=" * 72)
print(f"作业数 n={N_JOBS}，工序数 m={STAGES}，工序：{' → '.join(STAGE_NAMES)}")
print(f"理论下界 LB = max(各工序总工时) = {P_BASE.sum(axis=0).max():.1f} 小时")

lb = P_BASE.sum(axis=0).max()
print(f"\n各工序总工时：{dict(zip(STAGE_NAMES, np.round(P_BASE.sum(axis=0), 1)))}")

# ---------- 7.1 候选规则 ----------
print("\n【实验1】工程师直觉规则 vs NEH vs IG")
results = {}
for name, fn in [("SPT 短活先做", rule_spt),
                 ("LPT 长活先做", rule_lpt),
                 ("瓶颈优先(蒸煮)", rule_bottleneck),
                 ("前段-后段差", rule_eedd_like)]:
    perm = fn(None, P_BASE)
    ms = makespan(perm, P_BASE)
    results[name] = (perm, ms)
    print(f"  {name:<16s} makespan = {ms:7.2f}  h   （超下界 {100*(ms/lb-1):5.1f}%）")

init = rule_lpt(None, P_BASE)
t0 = time.time()
neh_perm = neh(P_BASE, order=init)
neh_ms = makespan(neh_perm, P_BASE)
t_neh = time.time() - t0
results["NEH 构造"] = (neh_perm, neh_ms)
print(f"  {'NEH 构造':<16s} makespan = {neh_ms:7.2f}  h   （超下界 {100*(neh_ms/lb-1):5.1f}%）  用时 {t_neh:.3f}s")

t0 = time.time()
ig_perm, ig_ms, ig_hist = iterated_greedy(P_BASE, neh_perm, iterations=200, d=4)
t_ig = time.time() - t0
results["IG 迭代贪心"] = (ig_perm, ig_ms)
print(f"  {'IG 迭代贪心':<16s} makespan = {ig_ms:7.2f}  h   （超下界 {100*(ig_ms/lb-1):5.1f}%）  用时 {t_ig:.2f}s")

# ---------- 7.2 独立裁判：小规模暴力 ----------
print("\n【实验2】独立裁判：小规模（n=7 子集，最后 7 批）暴力全排列")
sub = P_BASE[7:, :]  # 7 批 → 7! = 5040
t0 = time.time()
bf_perm, bf_ms = brute_force(sub)
t_bf = time.time() - t0
# 同样在小规模上跑 NEH 与 IG，验证其在有真值时的表现
sub_neh = neh(sub)
sub_neh_ms = makespan(sub_neh, sub)
sub_ig, sub_ig_ms, _ = iterated_greedy(sub, sub_neh, iterations=100, d=3)
print(f"  暴力最优        makespan = {bf_ms:7.2f}  h   排列 {list(bf_perm)}  （{t_bf:.2f}s）")
print(f"  NEH 构造        makespan = {sub_neh_ms:7.2f}  h   排列 {list(sub_neh)}   差 {100*(sub_neh_ms/bf_ms-1):+.2f}%")
print(f"  IG 迭代贪心     makespan = {sub_ig_ms:7.2f}  h   排列 {list(sub_ig)}   差 {100*(sub_ig_ms/bf_ms-1):+.2f}%")

# ---------- 7.3 随机多重起点 ----------
print("\n【实验3】随机多重起点 2000 次（看「顺序」到底值多少钱）")
t0 = time.time()
rs_perm, rs_ms, rs_vals = random_multistart(P_BASE, n_trials=2000)
t_rs = time.time() - t0
print(f"  随机最好  makespan = {rs_ms:7.2f}  h（{t_rs:.1f}s）")
print(f"  随机最差  makespan = {rs_vals.max():7.2f}  h")
print(f"  随机均值  makespan = {rs_vals.mean():7.2f}  h ± {rs_vals.std():.2f}")
print(f"  ⭐ NEH 比随机均值省 {100*(1-neh_ms/rs_vals.mean()):.1f}%，比随机最好还省 {100*(1-neh_ms/rs_ms):.1f}%")
print(f"  ⭐ IG 比随机均值省 {100*(1-ig_ms/rs_vals.mean()):.1f}%，比随机最好还省 {100*(1-ig_ms/rs_ms):.1f}%")

# ---------- 7.4 鲁棒性：IG 多次独立运行 ----------
print("\n【实验4】IG 鲁棒性：10 次不同随机种子")
ig_runs = []
for s in range(10):
    p_, ms_, _ = iterated_greedy(P_BASE, neh_perm, iterations=200, d=4, seed=RNG_SEED + 101 * s)
    ig_runs.append(ms_)
ig_runs = np.array(ig_runs)
print(f"  最好 {ig_runs.min():.2f} / 最差 {ig_runs.max():.2f} / 均值 {ig_runs.mean():.2f} / 浮动 {100*(ig_runs.max()-ig_runs.min())/ig_runs.mean():.2f}%")

# ---------- 7.5 甘特图数据 ----------
ms_best, C_best = schedule(ig_perm, P_BASE)

# ======================================================================
# 八、断言（硬验收）
# ======================================================================
print("\n【验收断言】")
assert np.array_equal(np.sort(ig_perm), np.arange(N_JOBS)), "IG 解必须是 0..n-1 的合法排列"
assert np.array_equal(np.sort(neh_perm), np.arange(N_JOBS)), "NEH 解必须是合法排列"
assert ig_ms <= neh_ms + 1e-9, "IG 不应劣于其初始解 NEH"
assert neh_ms < results["LPT 长活先做"][1], "NEH 应优于 LPT 直觉规则"
assert ig_ms <= rs_ms + 1e-9, "IG 不应劣于随机 2000 次的最好解"
assert sub_ig_ms <= sub_neh_ms + 1e-9, "小规模上 IG 不应劣于 NEH"
assert sub_neh_ms >= bf_ms - 1e-9, "NEH 不可能优于暴力真最优"
assert abs(C_best[-1, -1] - ig_ms) < 1e-9, "甘特表最后完工时刻应等于 makespan"
assert np.all(C_best > 0), "所有完工时刻应为正"
print("  ✅ 全部断言通过")

# ======================================================================
# 九、画图
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 左上：各方法对比
ax = axes[0, 0]
names = list(results.keys())
vals = [results[k][1] for k in names]
colors = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#b0b0b0", "#4C78A8", "#D62728"]
bars = ax.barh(range(len(names)), vals, color=colors)
ax.axvline(lb, color="green", ls="--", lw=1.5, label=f"理论下界 {lb:.1f}h")
ax.axvline(rs_vals.mean(), color="orange", ls=":", lw=1.5, label=f"随机均值 {rs_vals.mean():.1f}h")
for i, v in enumerate(vals):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("完工时刻 makespan（小时）")
ax.set_title("① 直觉规则 vs NEH 构造 vs IG：谁真的更早收工")
ax.legend(fontsize=8, loc="lower right")
ax.invert_yaxis()

# 右上：IG 收敛
ax = axes[0, 1]
ax.plot(ig_hist, color="#D62728", lw=1.8, label="IG 历史最优")
ax.axhline(neh_ms, color="#4C78A8", ls="--", lw=1.5, label=f"NEH 起点 {neh_ms:.1f}h")
ax.axhline(lb, color="green", ls=":", lw=1.2, label=f"理论下界 {lb:.1f}h")
ax.set_xlabel("迭代次数")
ax.set_ylabel("makespan（小时）")
ax.set_title("② 迭代贪心：破坏 d=4 → 最优插入重建 → 接受准则")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 左下：最优排产甘特图（按设备一行、按批次着色）
ax = axes[1, 0]
cmap = plt.get_cmap("tab20")
for i in range(N_JOBS):
    job = ig_perm[i]
    for k in range(STAGES):
        start = C_best[i, k] - P_BASE[job, k]
        ax.barh(k, P_BASE[job, k], left=start, height=0.7,
                color=cmap(job % 20), edgecolor="white", lw=0.8)
        if P_BASE[job, k] >= 3:
            ax.text(start + P_BASE[job, k] / 2, k, str(job), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
ax.set_yticks(range(STAGES))
ax.set_yticklabels(STAGE_NAMES)
ax.invert_yaxis()
ax.set_xlabel("时间（小时）")
ax.set_title(f"③ IG 最优排产甘特图（makespan = {ig_ms:.1f}h，数字=批次号）")
ax.grid(axis="x", alpha=0.3)
ax.axvline(ig_ms, color="#D62728", ls="--", lw=1.5)
ax.axvline(lb, color="green", ls=":", lw=1.2)

# 右下：随机分布
ax = axes[1, 1]
ax.hist(rs_vals, bins=40, color="#b0b0b0", edgecolor="white", alpha=0.85)
ax.axvline(ig_ms, color="#D62728", lw=2, label=f"IG {ig_ms:.1f}h")
ax.axvline(neh_ms, color="#4C78A8", lw=2, ls="--", label=f"NEH {neh_ms:.1f}h")
ax.axvline(rs_vals.mean(), color="orange", lw=1.5, ls=":", label=f"随机均值 {rs_vals.mean():.1f}h")
ax.set_xlabel("makespan（小时）")
ax.set_ylabel("随机排列次数")
ax.set_title("④ 2000 个随机排产的完工时刻分布")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowshop_neh_ig.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n图已保存：{out_png}")

print("\n【结论摘要】")
print(f"  NEH 构造 makespan = {neh_ms:.1f}h（超下界 {100*(neh_ms/lb-1):.1f}%）")
print(f"  IG 迭代贪心 makespan = {ig_ms:.1f}h（超下界 {100*(ig_ms/lb-1):.1f}%）")
print(f"  最优顺序：{list(ig_perm)}")
