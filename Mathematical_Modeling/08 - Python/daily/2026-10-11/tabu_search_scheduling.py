# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L6 优化模型 / L5 算法（元启发式 / 禁忌搜索）
anchor:  Roadmap Level 5-6 · 组合优化的邻域搜索：禁忌搜索 Tabu Search
goal:    用「禁忌搜索（Tabu Search）+ 交换邻域」求解 **单机总加权延误排产问题 1||Σ w_j T_j**，
        并与「暴力枚举（小规模真最优）」「EDD 规则（交期优先）」「WSPT 规则（权重优先）」
        「爬山法（只走改进方向）」「随机重启局部搜索」对照，量化禁忌表+特赦准则带来的增量。

本脚本零专用调度库：手写目标函数、交换邻域、禁忌表、期望准则(aspiration)、
禁忌长度自适应与多样化的重启机制；scipy 仅用于可选参考（本教学实例不需要 MILP）。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-11/tabu_search_scheduling.py"
"""

import itertools
import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------------
# 中文字体（避免乱码 / glyph 缺失）
# ----------------------------------------------------------------------------
for _cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=_cand), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261011)


# ============================================================================
# 1. 真实决策背景：一台瓶颈设备的排产（1 || Σ w_j T_j）
# ----------------------------------------------------------------------------
# 某机加车间有 1 台热处理炉（瓶颈设备），一批 n 个订单只能【一台一台】做，不能抢单。
#   p_j   加工时间（小时）
#   d_j   承诺交期（自开工起算，小时）
#   w_j   延误惩罚权重（元/小时）—— 大客户 / 急件权重高
# 完工时刻 C_j 由排产顺序决定；延误 T_j = max(0, C_j - d_j)。
# 目标：找一个加工顺序 σ，使【总加权延误 Σ w_j T_j 最小】。
# 这是经典 NP-hard 问题（Lawler 1977 证明 1||Σw_jT_j 为强 NP-hard），
# 现实里只能靠「邻域搜索 / 元启发式」在可接受时间内找好解——正是禁忌搜索的用武之地。
# ============================================================================

def make_instance(n, seed=0):
    """随机生成一件排产实例（保证有意义的延误张力）。"""
    rng = np.random.default_rng(seed)
    p = rng.integers(5, 30, size=n)          # 加工时间 5~29 h
    w = rng.integers(1, 10, size=n)          # 权重 1~9
    # 交期围绕总加工时间的一部分生成，制造「赶不上交期」的张力
    total_p = int(p.sum())
    d = rng.integers(int(0.25 * total_p), int(0.75 * total_p), size=n)
    return p, d, w


def objective(order, p, d, w):
    """给定加工顺序(order 为订单编号的排列)，返回 (总加权延误, 完工时刻数组)。"""
    t = 0
    C = np.zeros(len(order))
    for k, job in enumerate(order):
        t += p[job]
        C[k] = t
    T = np.maximum(0, C - d[np.array(order)])
    return float(np.sum(w[np.array(order)] * T)), C


# ----------------------------------------------------------------------------
# 2. 构造型规则解（作为初始解 / 对照）
# ----------------------------------------------------------------------------
def rule_edd(p, d, w):
    """EDD：交期最早优先（Earliest Due Date）——经典交期型启发式。"""
    return list(np.argsort(d, kind="stable"))


def rule_wspt(p, d, w):
    """WSPT：加权最短加工时间优先（按 w_j/p_j 降序）。"""
    return list(np.argsort(-(w / p), kind="stable"))


def rule_atc(p, d, w, k=2.0):
    """ATC：Apparent Tardiness Cost 规则（Rachamadugu & Morton 1982），
    在交期型排产里通常优于 EDD/WSPT。"""
    n = len(p)
    remaining = set(range(n))
    order = []
    t = 0
    pbar = p.mean()
    while remaining:
        best, best_val = None, -np.inf
        for j in remaining:
            slack = d[j] - t - p[j]
            val = (w[j] / p[j]) * np.exp(-max(0.0, slack) / (k * pbar))
            if val > best_val:
                best, best_val = j, val
        order.append(best)
        t += p[best]
        remaining.remove(best)
    return order


# ----------------------------------------------------------------------------
# 3. 邻域：交换（swap）——任取两个位置交换
# ----------------------------------------------------------------------------
def all_swap_neighbors(order):
    """生成全部 2-opt 交换邻居：(新顺序, (i,j) 移动标识)。"""
    n = len(order)
    for i in range(n - 1):
        for j in range(i + 1, n):
            nb = order.copy()
            nb[i], nb[j] = nb[j], nb[i]
            yield nb, (order[i], order[j])   # 标识用「两个被换的订单号」做禁忌键


# ----------------------------------------------------------------------------
# 4. 禁忌搜索（Tabu Search, Glover 1986/1989）
# ----------------------------------------------------------------------------
def tabu_search(p, d, w, init_order, max_iter=400, tabu_tenure=8,
                restart_after=60, seed=0):
    """
    标准禁忌搜索：
      - 每轮在「交换邻域」里选**最优允许移动**（不一定是改进方向 → 能爬出局部最优）；
      - 禁忌表记录最近做过的『订单对交换』，防止立刻走回头路；
      - 特赦准则(aspiration)：若某禁忌移动能刷新冠军解，则破禁；
      - 停滞超过 restart_after 轮 → 从**冠军解**扰动重启（多样化）。
    返回：best_order, best_val, 目标轨迹, 历史最优轨迹, 重启次数。
    """
    rng = np.random.default_rng(seed)
    cur = init_order.copy()
    cur_val, _ = objective(cur, p, d, w)
    best = cur.copy()
    best_val = cur_val

    tabu = deque(maxlen=max(tabu_tenure, max_iter))   # 存 (i,j) 无向禁忌键
    tabu_until = {}                                   # 键 -> 解禁的迭代编号
    track, track_best = [cur_val], [best_val]
    no_improve = 0
    restarts = 0

    for it in range(1, max_iter + 1):
        best_nb, best_nb_val, best_move = None, np.inf, None
        for nb, key in all_swap_neighbors(cur):
            val, _ = objective(nb, p, d, w)
            k = tuple(sorted(key))
            is_tabu = tabu_until.get(k, -1) > it
            # 特赦准则：禁忌但更优于历史最优 → 允许
            if is_tabu and val >= best_val - 1e-9:
                continue
            if val < best_nb_val - 1e-12:
                best_nb, best_nb_val, best_move = nb, val, k

        if best_nb is None:      # 全被禁忌（罕见）：随机扰动一跳
            i, j = rng.choice(len(cur), size=2, replace=False)
            best_nb = cur.copy()
            best_nb[i], best_nb[j] = best_nb[j], best_nb[i]
            best_nb_val, _ = objective(best_nb, p, d, w)
            best_move = tuple(sorted((cur[i], cur[j])))

        cur, cur_val = best_nb, best_nb_val
        tabu_until[best_move] = it + int(rng.integers(1, tabu_tenure + 1))

        if cur_val < best_val - 1e-9:
            best, best_val = cur.copy(), cur_val
            no_improve = 0
        else:
            no_improve += 1

        track.append(cur_val)
        track_best.append(best_val)

        # 停滞 → 从冠军解扰动重启（多样化，保持精英解）
        if no_improve >= restart_after:
            cur = best.copy()
            for _ in range(max(2, len(cur) // 4)):
                a, b = rng.choice(len(cur), size=2, replace=False)
                cur[a], cur[b] = cur[b], cur[a]
            cur_val, _ = objective(cur, p, d, w)
            tabu_until.clear()
            no_improve = 0
            restarts += 1

    return best, best_val, track, track_best, restarts


def hill_climb(p, d, w, init_order, max_iter=400):
    """对照 A：爬山法——只接受改进移动，第一步就卡死在局部最优。"""
    cur = init_order.copy()
    cur_val, _ = objective(cur, p, d, w)
    track = [cur_val]
    for _ in range(max_iter):
        improved = False
        for nb, _k in all_swap_neighbors(cur):
            val, _ = objective(nb, p, d, w)
            if val < cur_val - 1e-9:
                cur, cur_val = nb, val
                improved = True
                break
        track.append(cur_val)
        if not improved:
            break
    return cur, cur_val, track


def random_restart_ls(p, d, w, n, tries=200, seed=1, max_iter=400):
    """对照 B：随机重启局部搜索——多次随机起点爬山，取最好。"""
    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    track = []
    for _ in range(tries):
        order = list(rng.permutation(n))
        _o, val, _tr = hill_climb(p, d, w, order, max_iter)
        if val < best_val:
            best, best_val = _o, val
        track.append(best_val)
    return best, best_val, track


def brute_force(p, d, w):
    """独立裁判：枚举全部排列（仅小 n 可用），返回 (真最优值, 最优顺序)。"""
    n = len(p)
    P, D, W = np.array(p), np.array(d), np.array(w)
    best_val, best_order = np.inf, None
    for perm in itertools.permutations(range(n)):
        a = np.array(perm)
        C = np.cumsum(P[a])
        val = float((W[a] * np.maximum(0, C - D[a])).sum())
        if val < best_val:
            best_val, best_order = val, list(perm)
    return best_val, best_order


# ============================================================================
# 主流程
# ============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("Day42 禁忌搜索 (Tabu Search) 求解单机总加权延误排产  1 || Σ w_j T_j")
    print("=" * 78)

    p, d, w = make_instance(n=9, seed=12)
    n = len(p)
    print(f"\n实例：n={n} 个订单")
    print(f"  p(加工时间) = {list(p)}")
    print(f"  d(交期)     = {list(d)}")
    print(f"  w(权重)     = {list(w)}")
    print(f"  总加工时间  = {int(p.sum())} h")

    # 真最优（独立裁判）
    import math
    t0 = time.time()
    bf_val, bf_order = brute_force(p, d, w)
    print(f"\n[独立裁判] 暴力枚举 {n}! = {math.factorial(n)} 种排列 ...")
    print(f"  真最优 ΣwT = {bf_val:.0f}  (顺序 {bf_order})   耗时 {time.time()-t0:.2f}s")

    # 规则解
    edd_o = rule_edd(p, d, w); edd_v, _ = objective(edd_o, p, d, w)
    wspt_o = rule_wspt(p, d, w); wspt_v, _ = objective(wspt_o, p, d, w)
    atc_o = rule_atc(p, d, w); atc_v, _ = objective(atc_o, p, d, w)
    print(f"\n[构造规则] EDD  = {edd_v:.0f}   WSPT = {wspt_v:.0f}   ATC = {atc_v:.0f}")

    # 禁忌搜索（以 ATC 起始）
    t0 = time.time()
    ts_best, ts_val, ts_track, ts_best_track, ts_restarts = tabu_search(
        p, d, w, atc_o, max_iter=400, tabu_tenure=8, restart_after=50, seed=42)
    ts_time = time.time() - t0
    print(f"\n[禁忌搜索] ΣwT = {ts_val:.0f}  顺序 {ts_best}  重启 {ts_restarts} 次  耗时 {ts_time:.3f}s")

    # 对照：爬山法、随机重启局部搜索
    hc_o, hc_v, hc_track = hill_climb(p, d, w, atc_o)
    rr_o, rr_v, rr_track = random_restart_ls(p, d, w, n, tries=200, seed=3)
    print(f"[对照] 爬山法(ATC 起点) = {hc_v:.0f}   随机重启局部搜索(200 次) = {rr_v:.0f}")
    gap_hc = (hc_v - bf_val) / max(1.0, bf_val)
    gap_atc = (atc_v - bf_val) / max(1.0, bf_val)
    print(f"       爬山法距真最优 +{gap_hc*100:.1f}% ; ATC 距真最优 +{gap_atc*100:.1f}%")

    # ---------------------------- 断言（跑通即验证） ----------------------------
    assert abs(ts_val - bf_val) < 1e-6, f"禁忌搜索={ts_val} 未达到真最优={bf_val}"
    assert ts_val <= min(edd_v, wspt_v, atc_v) + 1e-9, "禁忌搜索不应劣于规则解"
    assert hc_v >= bf_val - 1e-9, "爬山解不应优于真最优"
    assert rr_v >= bf_val - 1e-9, "随机重启解不应优于真最优"
    assert sorted(ts_best) == list(range(n)), "最优顺序应是 0..n-1 的一个排列"
    print("\n✅ 断言全过：禁忌搜索达到暴力真最优；解为合法排列；对照不优于真最优。")

    # ---------------------------- 可视化 ----------------------------
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # (0,0) 收敛轨迹
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts_track, color="#9aa0a6", lw=1.0, alpha=0.9, label="当前解 ΣwT")
    ax.plot(ts_best_track, color="#d62728", lw=2.2, label="历史最优 ΣwT")
    ax.axhline(bf_val, color="#2ca02c", ls="--", lw=1.6, label=f"暴力真最优 = {bf_val:.0f}")
    ax.set_title("禁忌搜索收敛轨迹（含停滞重启）", fontsize=13)
    ax.set_xlabel("迭代轮次"); ax.set_ylabel("总加权延误 Σ w_j·T_j")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (0,1) 六法对照柱状
    ax = fig.add_subplot(gs[0, 1])
    labels = ["EDD\n规则", "WSPT\n规则", "ATC\n规则", "爬山法", "随机重启\n局部搜索", "禁忌搜索", "暴力\n真最优"]
    vals = [edd_v, wspt_v, atc_v, hc_v, rr_v, ts_val, bf_val]
    colors = ["#c7c7c7", "#c7c7c7", "#8ab4f8", "#f4b183", "#e69138", "#d62728", "#2ca02c"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("七种方法总加权延误对比（越低越好）", fontsize=13)
    ax.set_ylabel("Σ w_j·T_j"); ax.grid(axis="y", alpha=0.3)

    # (1,0) 最优排产甘特图
    ax = fig.add_subplot(gs[1, 0])
    t = 0
    ymax = 0
    for k, job in enumerate(ts_best):
        start, end = t, t + p[job]
        tardy = max(0, end - d[job])
        col = "#d62728" if tardy > 0 else "#2ca02c"
        ax.barh(0, p[job], left=start, height=0.6, color=col, edgecolor="black", lw=0.6)
        ax.text((start + end) / 2, 0, f"#{job}", ha="center", va="center", fontsize=9, color="white")
        # 交期刻度线
        ax.plot([d[job], d[job]], [-0.42, 0.42], color="#1a73e8", lw=1.2, alpha=0.8)
        t = end
        ymax = max(ymax, end)
    ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_title(f"最优排产甘特图（红=延误，绿=准时，蓝刻度=各单交期）\nΣwT={ts_val:.0f}", fontsize=13)
    ax.set_xlabel("时间 (h)"); ax.grid(axis="x", alpha=0.3)

    # (1,1) 各订单延误明细
    ax = fig.add_subplot(gs[1, 1])
    _val, C = objective(ts_best, p, d, w)
    order_arr = np.array(ts_best)
    tardy = np.maximum(0, C - d[order_arr]) * w[order_arr]
    bars = ax.bar([f"#{j}" for j in ts_best], tardy, color="#d62728", edgecolor="black", lw=0.6)
    for b, v in zip(bars, tardy):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_title(f"最优顺序下各订单的加权延误贡献（总 {ts_val:.0f}）", fontsize=13)
    ax.set_ylabel("w_j·T_j"); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Day42 · 禁忌搜索求解单机总加权延误排产（1 || Σ w_j·T_j）", fontsize=15, y=0.98)
    out = os.path.join(here, "tabu_search_scheduling.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[图] 已保存 {out}")

    # ---------------------------- 多实例鲁棒性压力测试 ----------------------------
    print("\n" + "=" * 78)
    print("鲁棒性：20 个 n=8 随机实例，禁忌搜索 vs 暴力真最优")
    print("=" * 78)
    ok, worst_gap = 0, 0.0
    for s in range(20):
        pp, dd, ww = make_instance(n=8, seed=100 + s)
        start = rule_atc(pp, dd, ww)
        bval, _border = brute_force(pp, dd, ww)
        _o, tval, _tr, _bt, _rs = tabu_search(pp, dd, ww, start, max_iter=300,
                                              tabu_tenure=7, restart_after=40, seed=s)
        gap = (tval - bval) / max(1.0, bval)
        worst_gap = max(worst_gap, gap)
        if abs(tval - bval) < 1e-6:
            ok += 1
    print(f"  禁忌搜索与暴力真最优一致的实例数：{ok}/20")
    print(f"  最大相对间隙：{worst_gap*100:.2f}%")
    assert ok >= 18, f"禁忌搜索鲁棒性不足：仅 {ok}/20 达到真最优"
    print("✅ 鲁棒性断言通过（≥18/20 达真最优）。")
    print("\n完成。")


if __name__ == "__main__":
    main()
