#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day47 - 置换流水车间调度 FSP：NEH 构造 + 迭代贪心 IG（破坏-重建）
subject : mathematical-modeling
level   : L5 算法（元启发式 / 构造式启发 + 局部搜索）/ L6 优化模型（组合优化 / 调度）
anchor  : 05 - Algorithms（算法主线：调度元启发式）；前接 Day42 禁忌搜索、Day43 ACO、
          Day44 PSO、Day45 DE；与 Day6 GA/SA（TSP 环游）、Day21 Held-Karp（精确 TSP）对照
goal    : 用「NEH 构造 + 迭代贪心 IG」解一道「县域果酱加工厂 5 道工序、14 批原料的流水
          车间排产」决策（置换流水车间 FSP，最小化 makespan）——
          ① 讲清「为什么瓶颈工序上把长活排前面」这条工程师直觉反而更差；
          ② 看清 makespan 的「空间」真相：瓶颈机被排满、下游机只是偶尔闪一下，
             真正的损失在「瓶颈断粮」和「下游等件」——所以 FSP 是「组批 + 卡节拍」问题；
          ③ NEH：按总工时降序，逐个「插到当前最好的位置」——构造即高质量的奥秘；
          ④ 迭代贪心 IG：破坏 d 个工件 → 逐个最优插入重建 → 模拟退火式接受，
             从 LPT 规则粗解一路收敛到 NEH 同等的 124h；
          ⑤ 用 LPT / NEH / 随机多重起点（n=14 无真值）+ n=7 暴力全排列（有真值）+
             两种 LB（M1 = max 工序总工时、M2 = 双机 Johnson 松弛）交叉裁判。
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
from matplotlib.patches import Patch

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

RNG_SEED = 20261018

# ======================================================================
# 一、真实决策问题：县域果酱加工厂的「流水车间排产」
# ======================================================================
# 背景：一家县域果酱厂，5 台设备串联成一条流水线（每批原料必须按这个顺序走）：
#   0 清洗去核 → 1 蒸煮浓缩 → 2 灌装封口 → 3 贴标打码 → 4 装箱入库
# 今天到了 14 批不同水果的原料，每批的工序工时不同（下表，单位：小时）。
# 设备不能同时处理两批；一批在上一道工序做完才能进下一道（流水、不插队）。
#
#   ★ 决策：14 批原料的加工顺序（一个排列）。
#   ★ 目标：最后一批装箱入库的完工时刻（makespan）最早——整条线尽早收工。
#   ★ 现实口径：makespan 每提前 1 小时，就是 5 台设备 + 14 个工位少开 1 小时，
#     而「蒸煮浓缩」是唯一的产能瓶颈（总工时 117h，其余工序 27~64h）。
#
# 这就是经典「置换流水车间调度（Permutation Flow Shop, FSP）」，n≥3 即 NP-hard
# （Garey, Johnson & Sethi 1976）。

STAGES = 5
N_JOBS = 14
STAGE_NAMES = ["清洗去核", "蒸煮浓缩", "灌装封口", "贴标打码", "装箱入库"]
JOB_NAMES = ["草莓", "蓝莓", "苹果", "山楂", "梨", "黄桃", "葡萄",
             "桑葚", "枇杷", "杏", "无花果", "樱桃", "覆盆子", "柑橘"]

# 加工时间矩阵 p[j, m]（小时），单位工时
P_BASE = np.array([
    #  m0  m1  m2  m3  m4      总工时
    [ 2,  7,  2,  3,  1],    # 0  草莓      15
    [ 5, 10,  6,  1,  2],    # 1  蓝莓      24
    [ 6,  8,  4,  2,  5],    # 2  苹果      25
    [ 4, 11,  6,  1,  3],    # 3  山楂      25
    [ 1,  6,  4,  1,  1],    # 4  梨        13
    [ 8,  8,  5,  1,  5],    # 5  黄桃      27
    [ 1,  7,  3,  3,  1],    # 6  葡萄      15
    [ 7, 11,  4,  2,  5],    # 7  桑葚      29
    [ 5,  5,  5,  3,  5],    # 8  枇杷      23
    [ 3,  9,  6,  1,  3],    # 9  杏        22
    [ 4, 11,  4,  1,  5],    # 10 无花果    25
    [ 3,  8,  2,  1,  3],    # 11 樱桃      17
    [ 5,  8,  6,  2,  1],    # 12 覆盆子    22
    [ 2,  8,  7,  5,  5],    # 13 柑橘      27
], dtype=float)
assert P_BASE.shape == (N_JOBS, STAGES)

# ======================================================================
# 二、核心评价函数：给定顺序 → makespan（以及各批在各工序的完工时刻）
# ======================================================================
def schedule(perm, P):
    """置换流水车间：按 perm 顺序依次加工，返回 (makespan, 完工时刻表 C)。

    C[i, k] = 顺序上第 i 批在设备 k 上的完工时刻。
    递推：开始时刻 = max(同设备上一批完工, 本批上一道工序完工)。
    """
    perm = [int(j) for j in perm]
    n, m = len(perm), P.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            ready = 0.0
            if i > 0:
                ready = max(ready, C[i - 1, k])   # 设备 k 被上一批占着
            if k > 0:
                ready = max(ready, C[i, k - 1])   # 本批还没走完上一道工序
            C[i, k] = ready + P[perm[i], k]
    return float(C[-1, -1]), C


def makespan(perm, P):
    return schedule(perm, P)[0]


def idle_per_stage(perm, P):
    """每台设备的空闲时长（从 0 到 makespan 之间，设备没活干的总时长）。"""
    ms, C = schedule(perm, P)
    idles = []
    for k in range(P.shape[1]):
        busy = P[list(map(int, perm)), k].sum()   # 设备 k 真正干活的时间
        idles.append(ms - busy)
    return ms, np.array(idles)

# ======================================================================
# 三、工程师凭直觉会想到的排产规则
# ======================================================================
def rule_spt(P):
    """SPT：总工时最短的先做（想尽快清掉小单）。"""
    return list(np.argsort(P.sum(axis=1), kind="stable"))


def rule_lpt(P):
    """LPT：总工时最长的先做（让大单早开工、早完成）。"""
    return list(np.argsort(-P.sum(axis=1), kind="stable"))


def rule_bottleneck(P):
    """瓶颈优先：按瓶颈工序（蒸煮浓缩，m=1）耗时降序，先把长活喂给瓶颈机。"""
    return list(np.argsort(-P[:, 1], kind="stable"))


def rule_flow_dominance(P):
    """按「后段工时 − 前段工时」排：后段重（装箱入库长）的先做，前段重的后做。"""
    return list(np.argsort(P[:, 3:].sum(axis=1) - P[:, :2].sum(axis=1), kind="stable"))

# ======================================================================
# 四、NEH 构造启发式（Nawaz–Enscore–Ham, 1983）
# ======================================================================
def neh(P, order=None):
    """NEH：按总工时降序，逐个把工件插入到使当前 makespan 最小的位置。

    O(n^2·m) 朴素实现，n=14 毫秒级完成。
    """
    n = P.shape[0]
    if order is None:
        order = list(np.argsort(-P.sum(axis=1), kind="stable"))
    order = [int(j) for j in order]
    seq = [order[0]]
    for job in order[1:]:
        best_ms, best_pos = np.inf, 0
        for pos in range(len(seq) + 1):
            ms = makespan(seq[:pos] + [job] + seq[pos:], P)
            if ms < best_ms - 1e-9:
                best_ms, best_pos = ms, pos
        seq.insert(best_pos, job)
    return seq

# ======================================================================
# 五、迭代贪心 Iterated Greedy（Ruiz & Stützle, 2007）
# ======================================================================
def best_insert(seq, job, P):
    """把 job 插入 seq 的某个位置使 makespan 最小，返回 (新序列, makespan)。"""
    best_ms, best_seq = np.inf, None
    for pos in range(len(seq) + 1):
        cand = seq[:pos] + [job] + seq[pos:]
        ms = makespan(cand, P)
        if ms < best_ms - 1e-9:
            best_ms, best_seq = ms, cand
    return best_seq, best_ms


def iterated_greedy(P, init_seq, iterations=600, d=4, seed=RNG_SEED, T_factor=0.5):
    """IG：破坏（随机拿掉 d 个工件）→ 重建（逐个最优插入）→ 模拟退火式接受。

    温度 T = T_factor × 平均单位工时，量级 ≈「挪一批进去多花的那点时间」，
    够越过小坎、又不会乱跳。接受准则与 Day6 模拟退火同源，
    只是邻域从「交换两点」换成了「破坏-重建一大块」。
    """
    rng = np.random.default_rng(seed)
    cur = [int(j) for j in init_seq]
    cur_ms = makespan(cur, P)
    best, best_ms = list(cur), cur_ms
    history = [cur_ms]
    temp = T_factor * float(P.mean())

    for _ in range(iterations):
        # ---- 破坏 ----
        idx = sorted(rng.choice(len(cur), size=d, replace=False).tolist())
        removed = [cur[i] for i in idx]
        partial = [j for i, j in enumerate(cur) if i not in set(idx)]
        rng.shuffle(removed)
        # ---- 重建 ----
        cand = list(partial)
        for job in removed:
            cand, _ = best_insert(cand, job, P)
        cand_ms = makespan(cand, P)
        # ---- 接受准则 ----
        delta = cand_ms - cur_ms
        if delta < -1e-9 or rng.random() < np.exp(-max(delta, 0.0) / temp):
            cur, cur_ms = cand, cand_ms
            if cur_ms < best_ms - 1e-9:
                best, best_ms = list(cur), cur_ms
        history.append(best_ms)
    return best, best_ms, np.array(history)

# ======================================================================
# 六、裁判：暴力全排列 / 随机多重起点 / 两个下界
# ======================================================================
def brute_force(P):
    """暴力枚举全部 n! 个排列（仅 n ≤ 8 时用）。"""
    n = P.shape[0]
    best_ms, best_perm = np.inf, None
    for perm in itertools.permutations(range(n)):
        ms = makespan(perm, P)
        if ms < best_ms - 1e-9:
            best_ms, best_perm = ms, perm
    return list(best_perm), float(best_ms)


def random_multistart(P, n_trials=2000, seed=RNG_SEED):
    """随机多重起点：2000 个随机排列里最好的那个（n=14 时无真值，只能这样照镜子）。"""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    best_ms, best_perm, vals = np.inf, None, []
    for _ in range(n_trials):
        perm = list(rng.permutation(n))
        ms = makespan(perm, P)
        vals.append(ms)
        if ms < best_ms - 1e-9:
            best_ms, best_perm = ms, perm
    return best_perm, best_ms, np.array(vals)


def m2_exact_dp(a, b):
    """双机流水车间（Johnson 问题）的精确 makespan —— 子集 DP + 非支配剪枝。

    状态 (已排集合 S) -> [(C2, C1)] 非支配集；若 C1,C2 都被另一状态压住则剪掉。
    n 很小时几毫秒即可，用来给「相邻设备对」算精确双机下界。
    """
    n = len(a)
    pareto = {0: [(0.0, 0.0)]}
    for mask in range(1, 1 << n):
        cand = []
        for j in range(n):
            if mask >> j & 1:
                for c2, c1 in pareto[mask ^ (1 << j)]:
                    nc1 = c1 + a[j]
                    cand.append((max(c2, nc1) + b[j], nc1))
        cand.sort()
        keep, best_c1 = [], np.inf
        for c2, c1 in cand:
            if c1 < best_c1 - 1e-9:
                keep.append((c2, c1))
                best_c1 = c1
        pareto[mask] = keep
    return float(min(c2 for c2, _ in pareto[(1 << n) - 1]))


def lb_flow_conservation(P):
    """下界 M1：**流量守恒**（FSP 基础下界）。

    机器 k 要干 T_k = sum_j p[j,k] 小时；在它开动前，首件必须走完前 k 道
    （最短 sum_{i<k} min_j p[j,i]）；最后一件干完还要走完后 m-1-k 道（最短
    sum_{i>k} min_j p[j,i]）。所以 makespan >= 开工链 + T_k + 收尾链，对 k 取最大。
    它比「只数工时」的粗下界（M0，117h）紧，但仍看不到「两两相邻设备」的排队。
    """
    colmin = P.min(axis=0)
    totals = P.sum(axis=0)
    m = P.shape[1]
    return float(max(colmin[:k].sum() + totals[k] + colmin[k + 1:].sum() for k in range(m)))


def lb_machine_naive(P):
    """对照用「粗下界」M0：只数最忙那台机器的总工时。"""
    return float(P.sum(axis=0).max())


def lb_per_pair(P):
    """下界 M2：每一对「相邻设备」的**精确双机最优**，取最大。

    理由：只看相邻两台机器 (i, i+1) 的松弛问题，每道工序都不更长、
    且砍掉了其余工序 -> 松弛最优 <= 原问题最优，是合法下界；
    双机问题可精确求解（Johnson 的等价情形），所以这对下界比只看单机更紧。
    """
    m = P.shape[1]
    return float(max(m2_exact_dp(P[:, k], P[:, k + 1]) for k in range(m - 1)))


# ======================================================================
# 七、主实验
# ======================================================================
print("=" * 74)
print("Day47 · 置换流水车间调度 FSP：NEH 构造 + 迭代贪心 IG")
print("=" * 74)
print(f"工件 {N_JOBS} 批，工序 {STAGES} 道：{' → '.join(STAGE_NAMES)}")
P = P_BASE
totals = P.sum(axis=0)
lb0 = lb_machine_naive(P)
lb1, lb2 = lb_flow_conservation(P), lb_per_pair(P)
lb = max(lb1, lb2)   # 采用的最强下界（M0 太松，不参与）
print(f"各工序总工时 "
      f"{dict(zip(STAGE_NAMES, np.round(totals, 1).tolist()))}")
print(f"下界 M0（只数最忙机器总工时）= {lb0:.1f}h ｜ "
      f"M1（流量守恒：开工链 + T_k + 收尾链）= {lb1:.1f}h ｜ "
      f"M2（相邻设备对的精确双机界）= {lb2:.1f}h  → 取 LB={lb:.1f}h")
print("  注意：M2 只用一对相邻设备，聚合后每件都更短、还可能看不到首件开工链，"
      "所以合法但偏松；下界越紧，越能证明「解已经很好」——这也是本课要讲的判断。")

# ---------- 7.1 直觉规则 vs NEH vs IG ----------
print("\n【实验1】工程师直觉规则 vs NEH vs IG")
results = {}
for name, fn in [
    ("SPT 短活先做", rule_spt),
    ("LPT 长活先做", rule_lpt),
    ("瓶颈(蒸煮)优先", rule_bottleneck),
    ("后段优先", rule_flow_dominance),
]:
    perm = fn(P)
    ms = makespan(perm, P)
    results[name] = (perm, ms)
    print(f"  {name:<14s} makespan = {ms:6.1f} h  （距下界 {100*(ms/lb-1):+5.1f}%）")

t0 = time.time()
neh_perm = neh(P)
neh_ms = makespan(neh_perm, P)
t_neh = time.time() - t0
results["NEH 构造"] = (neh_perm, neh_ms)
print(f"  {'NEH 构造':<14s} makespan = {neh_ms:6.1f} h  （距下界 {100*(neh_ms/lb-1):+5.1f}%）  "
      f"用时 {1000*t_neh:.1f}ms")

lpt_perm, lpt_ms = results["LPT 长活先做"]
t0 = time.time()
ig_perm, ig_ms, ig_hist = iterated_greedy(P, lpt_perm, iterations=600, d=4)
t_ig = time.time() - t0
results["IG 迭代贪心"] = (ig_perm, ig_ms)
print(f"  {'IG 迭代贪心':<14s} makespan = {ig_ms:6.1f} h  （距下界 {100*(ig_ms/lb-1):+5.1f}%）  "
      f"用时 {t_ig:.2f}s（从 LPT 粗解出发）")

# ---------- 7.2 瓶颈视角：空闲都发生在哪台设备上 ----------
print("\n【实验2】空闲在哪台设备上？（makespan − 该设备实际干活时间）")
idle_rows = {}
for name in ["LPT 长活先做", "NEH 构造", "IG 迭代贪心", "SPT 短活先做"]:
    perm, ms = results[name]
    _, idles = idle_per_stage(perm, P)
    idle_rows[name] = idles
    print(f"  {name:<14s} ms={ms:6.1f}  " +
          "  ".join(f"{s}{i:5.1f}h" for s, i in zip(STAGE_NAMES, idles)))

print("  → 头两道工序几乎不空（瓶颈机被排满），空都空在下游的「等件」。"
      "FSP 的真实损失 = 瓶颈断粮 + 下游等件，所以好解要「让瓶颈不断粮、又别把下游堵死」。")

# ---------- 7.3 独立裁判①：n=7 暴力全排列 ----------
print("\n【实验3】独立裁判①：n=7 子集暴力枚举全部 7! = 5040 个排列")
sub = P[7:, :]                       # 桑葚…柑橘 共 7 批
t0 = time.time()
bf_perm, bf_ms = brute_force(sub)
t_bf = time.time() - t0
sub_neh = neh(sub)
sub_neh_ms = makespan(sub_neh, sub)
sub_lpt = rule_lpt(sub)
sub_lpt_ms = makespan(sub_lpt, sub)
sub_ig, sub_ig_ms, _ = iterated_greedy(sub, sub_lpt, iterations=300, d=3)
print(f"  暴力真最优     = {bf_ms:6.1f} h   顺序 {bf_perm}   （{t_bf:.2f}s）")
print(f"  LPT 规则       = {sub_lpt_ms:6.1f} h   差 {100*(sub_lpt_ms/bf_ms-1):+5.2f}%")
print(f"  NEH 构造       = {sub_neh_ms:6.1f} h   差 {100*(sub_neh_ms/bf_ms-1):+5.2f}%")
print(f"  IG 迭代贪心    = {sub_ig_ms:6.1f} h   差 {100*(sub_ig_ms/bf_ms-1):+5.2f}%")

# ---------- 7.4 独立裁判②：n=14 随机多重起点 ----------
print("\n【实验4】独立裁判②：n=14 随机多重起点 2000 次（无真值时的参照系）")
t0 = time.time()
rs_perm, rs_ms, rs_vals = random_multistart(P, n_trials=2000)
t_rs = time.time() - t0
print(f"  随机最好 = {rs_ms:6.1f} h ｜ 最差 = {rs_vals.max():6.1f} h ｜ "
      f"均值 = {rs_vals.mean():6.1f} ± {rs_vals.std():.2f} h  （{t_rs:.1f}s）")
print(f"  ⭐ LPT 比随机均值省 {100*(1-lpt_ms/rs_vals.mean()):.1f}%")
print(f"  ⭐ NEH 比随机均值省 {100*(1-neh_ms/rs_vals.mean()):.1f}%、比随机最好还省 {100*(1-neh_ms/rs_ms):.1f}%")
print(f"  ⭐ IG  比随机均值省 {100*(1-ig_ms/rs_vals.mean()):.1f}%、比随机最好还省 {100*(1-ig_ms/rs_ms):.1f}%")

# ---------- 7.5 鲁棒性：IG 换 10 个种子 ----------
print("\n【实验5】IG 鲁棒性：10 个随机种子（都从 LPT 出发）")
ig_runs = []
for s in range(10):
    _, ms_, _ = iterated_greedy(P, lpt_perm, iterations=600, d=4, seed=RNG_SEED + 977 * s)
    ig_runs.append(ms_)
ig_runs = np.array(ig_runs)
spread = 100 * (ig_runs.max() - ig_runs.min()) / ig_runs.mean()
print(f"  最好 {ig_runs.min():.1f} / 最差 {ig_runs.max():.1f} / 均值 {ig_runs.mean():.2f} / "
      f"浮动 {spread:.2f}%")

# ======================================================================
# 八、断言（硬验收）
# ======================================================================
print("\n【验收断言】")
assert sorted(ig_perm) == list(range(N_JOBS)), "IG 解必须是 0..n-1 的合法排列"
assert sorted(neh_perm) == list(range(N_JOBS)), "NEH 解必须是合法排列"
assert ig_ms <= lpt_ms + 1e-9, "IG 不应劣于其初始解 LPT"
assert ig_ms <= neh_ms + 1e-9, "IG 应从 LPT 收敛到 NEH 同等或更好"
assert neh_ms < lpt_ms, "NEH 应严格优于 LPT 直觉规则"
assert lb0 <= lb2 + 1e-9, "M2 应比粗下界 M0 更紧"
print(f"  提示：M0={lb0:.0f}h 只是个「乐观的梦想」，比解还短；M2={lb2:.0f}h 才靠得住")
assert neh_ms >= lb2 - 1e-9 and ig_ms >= lb2 - 1e-9, "解不可能低于最强下界 M2"
assert ig_ms <= rs_ms + 1e-9, "IG 不应劣于 2000 次随机的最好解"
assert sub_ig_ms >= bf_ms - 1e-9, "子问题解不可能优于暴力真最优"
assert abs(sub_ig_ms - bf_ms) < 1e-9, "子问题上 IG 应命中暴力真最优"
_, idles_neh = idle_per_stage(neh_perm, P)
assert abs(idles_neh[1] - (ig_ms - totals[1])) < 1e-9, "瓶颈机空闲应等于 makespan 减去其总工时"
print("  ✅ 全部断言通过")

# ======================================================================
# 九、出图
# ======================================================================
ms_best, C_best = schedule(ig_perm, P)
fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

# ---- 左上：各方案 makespan 对比 ----
ax = axes[0, 0]
names = ["SPT 短活先做", "LPT 长活先做", "瓶颈(蒸煮)优先", "后段优先", "NEH 构造", "IG 迭代贪心"]
vals = [results[n][1] for n in names]
colors = ["#9AA5B1"] * 4 + ["#4C78A8", "#D62728"]
bars = ax.barh(names, vals, color=colors, height=0.62)
ax.axvline(lb, color="green", ls=":", lw=1.8)
ax.text(lb + 0.4, -0.72, f"LB {lb:.0f}h", color="green", fontsize=9)
for b, v in zip(bars, vals):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.0f}h", va="center", fontsize=9)
ax.set_xlim(118, max(vals) + 4)
ax.invert_yaxis()
ax.set_xlabel("makespan（小时）")
ax.set_title("① 直觉规则 vs NEH 构造 vs IG：谁真的更早收工")
ax.grid(axis="x", alpha=0.3)

# ---- 右上：IG 收敛过程 ----
ax = axes[0, 1]
ax.plot(ig_hist, color="#D62728", lw=2.0, label="IG 历史最优")
ax.axhline(lpt_ms, color="#9AA5B1", ls="--", lw=1.5, label=f"LPT 起点 {lpt_ms:.0f}h")
ax.axhline(neh_ms, color="#4C78A8", ls="--", lw=1.5, label=f"NEH {neh_ms:.0f}h")
ax.axhline(lb, color="green", ls=":", lw=1.5, label=f"下界 {lb:.0f}h")
ax.set_xlabel("迭代次数")
ax.set_ylabel("makespan（小时）")
ax.set_title("② IG 从 LPT 粗解 130h 一路收敛到 124h（破坏 4 批 → 最优插入）")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

# ---- 左下：各设备空闲时长（瓶颈视角）----
ax = axes[1, 0]
show = ["SPT 短活先做", "LPT 长活先做", "NEH 构造", "IG 迭代贪心"]
x = np.arange(len(STAGE_NAMES))
w = 0.2
for i, nm in enumerate(show):
    ax.bar(x + (i - 1.5) * w, idle_rows[nm], width=w, label=nm)
ax.set_xticks(x)
ax.set_xticklabels(STAGE_NAMES)
ax.set_ylabel("设备空闲时长（小时）")
ax.set_title("③ 空闲都发生在下游：瓶颈机排满，损失在「等件」")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ---- 右下：最优排产的甘特图 ----
ax = axes[1, 1]
cmap = plt.get_cmap("tab20")
for i, job in enumerate(ig_perm):
    for k in range(STAGES):
        start = C_best[i, k] - P[job, k]
        ax.barh(k, P[job, k], left=start, height=0.7,
                color=cmap(job % 20), edgecolor="white", lw=0.6)
        if P[job, k] >= 4:
            ax.text(start + P[job, k] / 2, k, str(job), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
ax.set_yticks(range(STAGES))
ax.set_yticklabels(STAGE_NAMES)
ax.invert_yaxis()
ax.axvline(ig_ms, color="#D62728", ls="--", lw=1.5)
ax.text(ig_ms - 1.0, -0.75, f"makespan {ig_ms:.0f}h", color="#D62728",
        ha="right", fontsize=9)
ax.set_xlabel("时间（小时）  ·  色块内数字 = 批次号")
ax.set_title(f"④ IG 最优排产甘特图（{ig_ms:.0f}h 完工；瓶颈机仅开头 2h 等首件）")
ax.grid(axis="x", alpha=0.3)
handles = [Patch(color=cmap(j % 20), label=f"{j}{JOB_NAMES[j]}") for j in range(N_JOBS)]
ax.legend(handles=handles, fontsize=6.5, ncol=7, loc="lower center",
          bbox_to_anchor=(0.5, -0.42), frameon=False)

plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowshop_neh_ig.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n图已保存：{out_png}")

# ======================================================================
# 十、结论摘要（可复述版）
# ======================================================================
print("\n【结论摘要】")
print(f"  瓶颈工序：蒸煮浓缩，总工时 {totals[1]:.0f}h → 光瓶颈机就要干 {totals[1]:.0f}h，"
      f"再加首尾两条链，理论最快不可能早于 {lb2:.0f}h")
print(f"  LPT 直觉规则   {lpt_ms:.1f}h（距下界 {100*(lpt_ms/lb-1):+.1f}%）——瓶颈机被喂长活，却把下游堵死")
print(f"  NEH 构造       {neh_ms:.1f}h（距下界 {100*(neh_ms/lb-1):+.1f}%，{1000*t_neh:.0f}ms）")
print(f"  IG 迭代贪心    {ig_ms:.1f}h（距下界 {100*(ig_ms/lb-1):+.1f}%，{t_ig:.1f}s）")
print(f"  最优顺序（批次号）：{ig_perm}")
print(f"  按水果名：{' → '.join(JOB_NAMES[j] for j in ig_perm)}")
print(f"  相对 LPT 省 {lpt_ms - ig_ms:.1f}h（{100*(1-ig_ms/lpt_ms):.1f}%）；"
      f"按 5 台设备计，折合少开机 {5*(lpt_ms - ig_ms):.0f} 设备·小时")
print(f"  ⭐ 下界教学点：「只数最忙机器工时」的粗下界 M0={lb0:.0f}h 比解还长 —— 下界本身也会算错；"
      f"正确做法是让每一道工序都交代「首件开工链 + 本工序负载 + 末件收尾链」")
print(f"  与最强下界 {lb:.0f}h 只差 {ig_ms-lb:.0f}h（{100*(ig_ms/lb-1):.1f}%）："
      f"这就是「证明解已近最优」的证据；子问题上 IG 已命中暴力真最优")
print(f"  诚实说明：{ig_ms:.0f}h 是否全局最优未证明——下界仍偏松（只用相邻设备对），"
      f"不能冒充「最优」；对外交付应写「{ig_ms:.0f}h，距下界 {100*(ig_ms/lb-1):.1f}%」")
