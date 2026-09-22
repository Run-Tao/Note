# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L7 图论、网络与动态模型 / L6 优化模型
anchor: Day22 — 指派问题(Assignment)与二分图匹配：匈牙利算法 + 非方阵/最大化/禁忌指派扩展
goal: 把"谁干哪件活"的运筹决策翻译成二分图上的最小权完美匹配，
      用手写匈牙利算法(Kuhn-Munkres, O(n^3))求全局最优指派，
      并与暴力穷举、scipy.optimize.linear_sum_assignment 交叉验证；
      再延伸到"项目调度(一台设备连续加工多任务)"这一同源问题的应用。

真实决策背景：
  某咨询公司在一个周末收到 5 个客户项目(任务 T1..T5)，
  所内 5 位顾问(C1..C5)各有所长，把项目交给不同顾问的"预期工时(小时)"如下。
  每位顾问同一时间只能做一个项目、每个项目只需一位顾问。
  问：怎样分配才能让"团队总工时"最小？

  追加决策：① 若预算只够请 4 位顾问，5 个项目怎么排(非方阵/虚设行)？
            ② 若公司目标是"总客户满意度(打分)最大"，怎么排(最大化 → 取负/大数法)？
            ③ 顾问 C3 因故不能接 T5(硬性禁忌)，最优怎么变？

运行：
  cd /home/ubuntu/Note && ./.venv/bin/python \
    "Mathematical_Modeling/08 - Python/daily/2026-09-22/assignment_hungarian_scheduling.py"
"""

import os
import itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 中文字体注册，避免图中中文乱码 ----
from matplotlib import font_manager

_CJK_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Source Han Sans SC",
    "SimHei",
]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in _CJK_CANDIDATES:
    if _name in _available:
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False

INF = 10**9


# =====================================================================
# 1. 手写匈牙利算法 (Kuhn-Munkres, O(n^3) Jonker-Volgenant 简介版)
# =====================================================================
def hungarian(cost):
    """求解方阵最小权完美指派。

    参数
    ----
    cost : (n, n) ndarray，cost[i, j] = 工人 i 做任务 j 的代价。

    返回
    ----
    row_assignment : (n,) ndarray，row_assignment[i] = 分给工人 i 的任务列号。
    total_cost     : float，最优总代价。

    算法：经典 O(n^3) 匈牙利法（对偶势 + 增广路）。
    维护行势 u、列势 v，使 reduced cost  slack = cost - u - v >= 0；
    对每个工人找一条"零费用增广路"，找不到就用最小 slack 更新势，
    直到每位工人都匹配上一个任务 → 得到最小权完美匹配。
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    assert n == m, "本实现只接方阵，非方阵请先补虚设行/列(见 solve_general)。"

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)   # p[j] = 匹配到列 j 的行
    way = np.zeros(m + 1, dtype=int)  # 增广路前驱

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # 回溯增广路，翻转匹配
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    row_assignment = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            row_assignment[p[j] - 1] = j - 1
    total_cost = float(cost[np.arange(n), row_assignment].sum())
    return row_assignment, total_cost


def brute_force(cost):
    """暴力穷举所有排列，仅用于小规模(<=9)真值核对。"""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    best_perm, best = None, np.inf
    for perm in itertools.permutations(range(n)):
        c = sum(cost[i, perm[i]] for i in range(n))
        if c < best:
            best, best_perm = c, perm
    return np.array(best_perm), float(best)


# =====================================================================
# 2. 通用求解：非方阵 / 最大化 / 禁忌指派
# =====================================================================
def solve_general(cost, maximize=False, forbidden=None):
    """把一般指派问题归约成方阵最小化，再调 hungarian。

    - maximize=True : 目标改为最大化 → 用 (max - cost) 转最小化(等价取负+平移)。
    - forbidden     : set of (i, j) 硬性禁止该指派 → 置为大 M。
    - 非方阵        : 行少补"虚设行(代价 0)"，列少补"虚设列"，
                      虚设匹配表示"该工人闲置 / 该任务没人做"。
    """
    cost = np.asarray(cost, dtype=float).copy()
    rows, cols = cost.shape

    if forbidden:
        big = cost.max() * 100 + 1000 if cost.size else 1e6
        for (i, j) in forbidden:
            cost[i, j] = big

    if maximize:
        maxv = cost.max()
        cost = maxv - cost  # 最大化 → 最小化

    n = max(rows, cols)
    padded = np.zeros((n, n), dtype=float)
    # 补虚设行列的代价设为 0(不产生成本)，但需保证不会挤掉真实指派：
    # 由于我们用大 M 处理禁忌、最大化已平移，补 0 即让"闲置/无人做"代价为 0。
    padded[:rows, :cols] = cost

    assign, total = hungarian(padded)
    real_pairs = [(i, assign[i]) for i in range(rows) if assign[i] < cols]
    return assign, total, real_pairs


# =====================================================================
# 3. 案例数据
# =====================================================================
consultants = ["顾问C1", "顾问C2", "顾问C3", "顾问C4", "顾问C5"]
tasks = ["项目T1", "项目T2", "项目T3", "项目T4", "项目T5"]

# 预期工时(小时)，行=顾问，列=项目
#    数值刻意做成：全局最小格与次小格落在同一行/列，
#    使"每轮锁定全局最省的单格"贪心会抢先占掉关键格，从而陷入局部最优。
hours = np.array([
    [10, 15, 17, 14, 20],   # C1  最擅长 T1
    [18, 11, 12, 16, 19],   # C2  最擅长 T2
    [13, 19, 10, 18, 16],   # C3  与 C5 争 T5、与 C1 争 T1
    [17,  9, 18, 11, 15],   # C4  最擅长 T2(与 C2 撞车)
    [19, 16, 20, 12, 11],   # C5  最擅长 T5，次擅长 T4
], dtype=float)

# 客户满意度打分(0-100)，行=顾问，列=项目。
# 刻意让"满意度最高"的配对与"工时最短"的配对部分错位(如 C3 做 T1 满意度最高、
# C2 做 T4 满意度也很高)，制造真实的双目标冲突。
satisfaction = np.array([
    [70, 88, 66, 84, 60],   # C1
    [72, 80, 90, 86, 74],   # C2  满意度上最擅长 T3，但也高分 T4
    [95, 62, 80, 58, 70],   # C3  满意度上 T1 最高，与工时最优争抢
    [74, 78, 68, 92, 76],   # C4
    [66, 76, 88, 74, 90],   # C5
], dtype=float)


def fmt_pairs(pairs, row_names, col_names):
    return "、".join(f"{row_names[i]}→{col_names[j]}" for i, j in pairs)


# =====================================================================
# 4. 求解与验证
# =====================================================================
def main():
    print("=" * 72)
    print("【案例】5 位顾问 × 5 个客户项目：谁干哪件活，团队总工时最小？")
    print("=" * 72)
    print("预期工时矩阵(小时)：")
    for i, name in enumerate(consultants):
        print(f"  {name}: " + "  ".join(f"{tasks[j]}={int(hours[i, j])}" for j in range(5)))

    # ---- 4.1 方阵最小化：手写匈牙利 ----
    assign, total = hungarian(hours)
    pairs = [(i, assign[i]) for i in range(5)]
    print("\n[主问题] 手写匈牙利算法最优指派：")
    print("  " + fmt_pairs(pairs, consultants, tasks))
    print(f"  最优团队总工时 = {total:.0f} 小时")

    # 交叉验证 1：暴力穷举
    bf_perm, bf_cost = brute_force(hours)
    bf_pairs = [(i, bf_perm[i]) for i in range(5)]
    print(f"\n[交叉验证] 暴力穷举(5!=120 种)：{fmt_pairs(bf_pairs, consultants, tasks)}")
    print(f"            总工时 = {bf_cost:.0f} 小时  → 与匈牙利一致: {abs(bf_cost - total) < 1e-9}")

    # 交叉验证 2：scipy 参考实现
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(hours)
        scipy_cost = float(hours[r, c].sum())
        print(f"[交叉验证] scipy.linear_sum_assignment：总工时 = {scipy_cost:.0f} 小时  "
              f"→ 与手写一致: {abs(scipy_cost - total) < 1e-9}")
    except Exception as e:  # pragma: no cover
        print(f"[交叉验证] scipy 不可用：{e}")

    # 贪心对照：按"单格工时从小到大"依次锁定最划算的格(经典贪心陷阱)
    greedy_cost, greedy_pairs = greedy_lowest_cell(hours)
    print(f"[对照] 贪心(每次锁定全局最省的那一格)：{fmt_pairs(greedy_pairs, consultants, tasks)}")
    print(f"        总工时 = {greedy_cost:.0f} 小时 → 比最优多 {greedy_cost - total:.0f} 小时")

    # ---- 4.2 最大化：满意度 ----
    _, neg_total, sat_pairs = solve_general(satisfaction, maximize=True)
    sat_total = float(sum(satisfaction[i, j] for i, j in sat_pairs))
    print("\n[变体A·最大化] 若目标是客户满意度总分最大：")
    print("  " + fmt_pairs(sat_pairs, consultants, tasks))
    print(f"  满意度总分 = {sat_total:.0f} (满分 500)")
    # 与工时最优解对照
    hours_of_sat = float(sum(hours[i, j] for i, j in sat_pairs))
    sat_of_hours = float(sum(satisfaction[i, j] for i, j in pairs))
    print(f"  ⚠️ 满意度最优解的工时 = {hours_of_sat:.0f} 小时（比工时最优多 {hours_of_sat - total:.0f}）；"
          f"工时最优解的满意度 = {sat_of_hours:.0f} 分（比满意度最优少 {sat_total - sat_of_hours:.0f}）")
    print("     → 两个目标打架：省工时与讨客户欢心不可兼得，需显式权衡。")

    # ---- 4.3 非方阵：只请 4 位顾问 ----
    sub = hours[:4, :]  # 只有 C1..C4 可用，5 个任务
    _, _, part_pairs = solve_general(sub)
    part_names = consultants[:4]
    done = {j for _, j in part_pairs}
    idle_task = [tasks[j] for j in range(5) if j not in done]
    print("\n[变体B·非方阵] 预算只够 4 位顾问(C1..C4)、5 个项目：")
    print("  " + fmt_pairs(part_pairs, part_names, tasks))
    print(f"  被放弃的项目 = {idle_task}（虚设列吸收）")
    part_cost = float(sum(sub[i, j] for i, j in part_pairs))
    print(f"  完成 4 项的总工时 = {part_cost:.0f} 小时")

    # ---- 4.4 禁忌指派：C3 不能接 T5 ----
    _, t_total, t_pairs = solve_general(hours, forbidden={(2, 4)})
    print("\n[变体C·禁忌] 顾问 C3 因故不能接项目 T5（硬约束）：")
    print("  " + fmt_pairs(t_pairs, consultants, tasks))
    print(f"  最优总工时 = {t_total:.0f} 小时 → 比无约束最优多 {t_total - total:.0f} 小时（禁忌的代价）")

    # ---- 4.5 敏感性：工时扰动后最优指派是否稳定 ----
    rng = np.random.default_rng(42)
    same = 0
    n_trials = 2000
    base_set = set(pairs)
    cost_bucket = {}
    for _ in range(n_trials):
        noise = rng.normal(1.0, 0.15, size=hours.shape)
        perturbed = np.clip(hours * noise, 1, None)
        a, _c = hungarian(perturbed)
        key = tuple((i, a[i]) for i in range(5))
        cost_bucket[key] = cost_bucket.get(key, 0) + 1
        if set(key) == base_set:
            same += 1
    print(f"\n[敏感性] 工时±15% 高斯扰动 × {n_trials} 次：")
    print(f"  原最优指派出现频率 = {same}/{n_trials} = {same / n_trials:.1%}")
    top = sorted(cost_bucket.items(), key=lambda kv: -kv[1])[:3]
    for key, cnt in top:
        print(f"    {fmt_pairs(list(key), consultants, tasks)} : {cnt / n_trials:.1%}")

    # ---- 4.6 项目调度应用：一台设备连续加工 5 个零件，最小化总完成时间 ----
    # 同源思想：一台机器依次加工多个任务，找"次序"使平均完成时间最小
    # 吉田法则/SPT：加工时间短的先做能最小化总完成时间(等价问题)
    job_names = ["零件P1", "零件P2", "零件P3", "零件P4", "零件P5"]
    proc_time = np.array([7, 3, 9, 2, 5], dtype=float)
    spt_order = np.argsort(proc_time)          # 短作业优先 SPT
    random_order = np.array([0, 1, 2, 3, 4])   # 原始(未排序)

    def total_completion(order, pt):
        t = 0.0
        total = 0.0
        for k in order:
            t += pt[k]
            total += t
        return total

    spt_total = total_completion(spt_order, proc_time)
    base_total = total_completion(random_order, proc_time)
    print("\n[应用·项目调度] 一台设备依次加工 5 个零件，最小化完工时间之和：")
    print(f"  原始次序总完工时间 = {base_total:.0f}")
    print(f"  SPT(短作业优先)次序 = {[job_names[k] for k in spt_order]}，"
          f"总完工时间 = {spt_total:.0f} → 省 {base_total - spt_total:.0f}")
    # 穷举核对 SPT 最优
    best_ord, best_tc = None, np.inf
    for perm in itertools.permutations(range(5)):
        tc = total_completion(perm, proc_time)
        if tc < best_tc:
            best_tc, best_ord = tc, perm
    print(f"  穷举 120 种核对：最优总完工时间 = {best_tc:.0f}，SPT 是否最优 = "
          f"{abs(best_tc - spt_total) < 1e-9}")

    # =================================================================
    # 5. 断言：全部通过才认为实现正确
    # =================================================================
    assert abs(bf_cost - total) < 1e-9, "匈牙利与暴力穷举不一致"
    # 最优工时应小于等于任何贪心/可行解
    assert total <= greedy_cost + 1e-9, "最优解不应劣于贪心"
    assert greedy_cost > total, "本例应演示贪心明显劣于最优，请检查数据"
    # 指派是双射(每行每列恰一个)
    assert len(set(assign)) == 5, "指派不是双射"
    # 非方阵：4 位顾问接 4 个不同任务
    assert len({j for _, j in part_pairs}) == 4, "非方阵指派列重复"
    # 禁忌：C3 不得接 T5
    assert (2, 4) not in t_pairs, "禁忌指派被违反"
    # 禁忌后的最优不比无约束更优(约束放宽后只能更好或持平)
    assert t_total >= total - 1e-9, "加约束后反而更优，异常"
    # SPT 为调度最优
    assert abs(best_tc - spt_total) < 1e-9, "SPT 非最优"
    print("\n[断言] 匈牙利=穷举、双射、非方阵完备、禁忌生效、加约束不更优、SPT 最优 —— 全部通过 ✅")

    # =================================================================
    # 6. 出图
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 子图1：工时矩阵热力图 + 最优指派
    ax = axes[0, 0]
    im = ax.imshow(hours, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(tasks, rotation=20)
    ax.set_yticks(range(5)); ax.set_yticklabels(consultants)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(hours[i, j]), ha="center", va="center", color="black", fontsize=10)
    for i, j in pairs:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="blue", lw=3))
    ax.set_title(f"工时矩阵与最优指派(蓝框)\n团队总工时 = {total:.0f} h")
    fig.colorbar(im, ax=ax, shrink=0.8, label="工时(小时)")

    # 子图2：方法对照柱状
    ax = axes[0, 1]
    labels = ["匈牙利\n(最优)", "暴力穷举\n(真值)", "贪心\n(按项目挑)", "SPT调度\n(同源)"]
    vals = [total, bf_cost, greedy_cost, spt_total]
    colors = ["#2e7d32", "#1565c0", "#c62828", "#6a1b9a"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("总工时 / 总完工时间")
    ax.set_title("不同解法对照：最优 vs 贪心")

    # 子图3：敏感性——前 3 指派出现频率
    ax = axes[1, 0]
    top_labels = []
    top_freqs = []
    for key, cnt in top:
        short = "、".join(f"C{i+1}→T{j+1}" for i, j in key)
        top_labels.append(short)
        top_freqs.append(cnt / n_trials * 100)
    ypos = np.arange(len(top_labels))
    ax.barh(ypos, top_freqs, color=["#2e7d32", "#90a4ae", "#90a4ae"])
    ax.set_yticks(ypos); ax.set_yticklabels(top_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("出现频率 (%)")
    ax.set_title(f"工时±15%扰动×{n_trials}次：指派稳定性")

    # 子图4：满意度 vs 工时 双目标权衡
    ax = axes[1, 1]
    h_pts = [total, hours_of_sat]
    s_pts = [sat_of_hours, sat_total]
    names = ["工时最优解", "满意度最优解"]
    ax.scatter(h_pts, s_pts, s=180, c=["#2e7d32", "#c62828"], zorder=3)
    ax.plot(h_pts, s_pts, "--", color="gray", zorder=1, label="权衡前沿(两解连线)")
    for x, y, n in zip(h_pts, s_pts, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("团队总工时 (小时) ↓ 越省越好")
    ax.set_ylabel("客户满意度总分 ↑ 越高越好")
    ax.set_title("两目标打架：省工时 vs 讨欢心")
    ax.legend()

    fig.suptitle("指派问题(匈牙利算法)与项目调度 —— Day22", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "assignment_hungarian_scheduling.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


def greedy_lowest_cell(cost):
    """贪心：每轮锁定当前"未被占用行列中工时最小"的那一格，锁定后再处理下一个。

    这是指派问题最典型的"局部最优陷阱"：单格看起来最省，
    但锁死后可能逼迫后续搭配出更贵的组合 → 全局并非最优。
    """
    n, m = cost.shape
    rows_used, cols_used = set(), set()
    chosen = []
    total = 0.0
    while len(chosen) < min(n, m):
        best_i, best_j, best_c = -1, -1, np.inf
        for i in range(n):
            if i in rows_used:
                continue
            for j in range(m):
                if j in cols_used:
                    continue
                if cost[i, j] < best_c:
                    best_c, best_i, best_j = cost[i, j], i, j
        if best_i < 0:
            break
        rows_used.add(best_i)
        cols_used.add(best_j)
        chosen.append((best_i, best_j))
        total += best_c
    return total, chosen


if __name__ == "__main__":
    main()
