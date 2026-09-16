# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 (04 - Optimization)   # 优化主线：LP -> ILP -> NLP -> 启发式 -> 动态规划(DP)
anchor: Day16 动态规划 DP / Bellman 最优性原理 / 多阶段决策(设备更新 & 0-1 背包)
goal  : 演示当决策"分阶段、后效、且阶段间由状态耦合"时，贪心为何失效。
        用同一套(状态, 决策, 转移, 收益)语言建模两个真实问题：
        ① 设备更新(capital replacement)：求 5 年总净成本最小，并把最优路径
           读成"资产账本"(何时卖、收回残值多少)；
        ② 0-1 背包(resource allocation)：DP 求最大价值。
        两者用 (a) 手写记忆化递归 top-down、(b) 自底向上迭代 DP
        互相交叉验证，背包再用暴力穷举做真值核对（绝不把没验证的代码入库）。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 注册中文字体，避免图乱码 ----------
for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

# =====================================================================
# 决策背景一（真实问题）：设备更新 / 资产处置
# 你是一家物流公司的资产经理，2026 年初手上有一台分拣机（车龄 t=1）。
# 机器会老化：车龄越老，"本年运行维护费 C(t)"越高；但老机器仍能卖点残值。
# 每个年初你可以做两个决策之一：
#   - 继续用一年（付运行费，车龄 +1）；
#   - 卖掉旧机器，买一台新的投入运营（付新车价 P、旧机按残值 s 收回）。
# 计划期 T=5 年，第 5 年末机器按残值清算（回收残值，不再投入运营）。
# 问：5 年总净成本最小是多少？每年该继续用还是换新？
#
# 为什么贪心必错：每一步"本年看着最省"(继续用)会把车龄拖老，
# 让后面几年的运行费爆炸；而"早换"当步看似贵，却摊掉了未来的运行费。
# 阶段之间的耦合(车龄)正是 DP 的用武之地，而贪心对它视而不见。
# =====================================================================

PRICE_NEW = 100.0                     # 购置一台新机器价格 P
# 运行维护成本 C(t)：车龄 t=1..6（单位：万元/年）
RUN_COST = {1: 8.0, 2: 12.0, 3: 18.0, 4: 27.0, 5: 40.0, 6: 57.0}
# 残值 s(t)：车龄 t 的机器当期卖掉能回收多少
RESALE = {1: 70.0, 2: 50.0, 3: 35.0, 4: 20.0, 5: 10.0, 6: 4.0}
T = 5                                  # 计划期年数
MAX_AGE = 6                            # 车龄上限（保持 DP 状态空间有限）


def year_cashflow(age, action, year):
    """给定'开年这台机器车龄 age'与动作，返回(该年现金流, 下一年开年车龄)。

    现金流约定（统一口径，保证账本能闭合）：
      - keep    : 继续用这台机器一年，付运行费 RUN_COST[age]；车龄 +1。
      - replace : 卖掉旧机回收残值 s(age)；买一台新机(付 P)，
                  新机本年即投入运营，按车龄 1 付运行费 RUN_COST[1]；
                  故下一年开年这台车龄为 2。
    计划期末(T 年末)：把当时的手上机器按残值卖掉落袋（在 V 的边界处理）。
    注：第 T 年初若 replace，等于"买了台新车当年就清算"，DP 会因
    s(1)=70 < P=100 自然排除它——这正是模型不会给出荒诞决策的原因。
    """
    if action == "keep":
        return -RUN_COST[min(age, MAX_AGE)], min(age + 1, MAX_AGE)
    elif action == "replace":
        return -PRICE_NEW + RESALE[age] - RUN_COST[1], 2
    raise ValueError(action)


def solve_equipment_topdown():
    """top-down 记忆化递归：V(year, age) = 从'第 year 年开年、车龄 age'到期末的最小未来净成本(负=净支出)。"""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def V(year, age):
        if year > T:
            return -RESALE[min(age, MAX_AGE)]     # 期末清算：卖机器回收残值
        best = float("inf")
        for action in ("keep", "replace"):
            cash, next_age = year_cashflow(age, action, year)
            val = cash + V(year + 1, next_age)
            if val < best:
                best = val
        return best

    # 回溯最优路径
    path, year, age = [], 1, 1
    while year <= T:
        best, best_action, best_next_age = float("inf"), None, None
        for action in ("keep", "replace"):
            cash, next_age = year_cashflow(age, action, year)
            val = cash + V(year + 1, next_age)
            if val < best:
                best, best_action, best_next_age = val, action, next_age
        path.append((year, age, best_action))
        year, age = year + 1, best_next_age
    return V(1, 1), path


def solve_equipment_bottomup():
    """bottom-up 迭代 DP：从期末往前推，dp[year][age] = 最小未来净成本(同口径)。"""
    dp = {T + 1: {age: -RESALE[min(age, MAX_AGE)] for age in range(1, MAX_AGE + 1)}}
    choice = {}
    for year in range(T, 0, -1):
        dp[year], choice[year] = {}, {}
        for age in range(1, MAX_AGE + 1):
            best, best_action = float("inf"), None
            for action in ("keep", "replace"):
                cash, next_age = year_cashflow(age, action, year)
                val = cash + dp[year + 1][next_age]
                if val < best:
                    best, best_action = val, action
            dp[year][age], choice[year][age] = best, best_action
    # 回溯
    path, year, age = [], 1, 1
    while year <= T:
        action = choice[year][age]
        _, next_age = year_cashflow(age, action, year)
        path.append((year, age, action))
        year, age = year + 1, next_age
    return dp[1][1], path, dp, choice


def simulate(strategy_fn):
    """按给定'逐阶段策略'模拟 T 年现金流，返回(总净成本, 逐年记录)。用于对照实验。"""
    total, age, recs = 0.0, 1, []
    for y in range(1, T + 1):
        action = strategy_fn(age, y)
        cash, next_age = year_cashflow(age, action, y)
        total += cash
        recs.append((y, age, action, cash, total))
        age = next_age
    end_resale = RESALE[min(age, MAX_AGE)]
    total -= end_resale
    recs.append((T + 1, min(age, MAX_AGE), "liquidate", -end_resale, total))
    return total, recs


# =====================================================================
# 决策背景二（真实问题）：0-1 背包 / 预算分配
# 有限货运预算 50 万，在若干候选"装备/项目"里选一批，每件有重量(成本)
# 与价值(收益)，每件最多选一次，装不下就放弃。
# 贪心按"性价比"排序会错：一件高性价比物品可能占满预算，
# 导致剩下额度浪费，反而不如两件稍低性价比但恰好装满的组合。
# DP 对"容量"做阶段(逐件物品)，状态=已用容量，求最大价值。
# =====================================================================

KNAP_W = [12, 20, 26, 8, 15, 33]        # 重量/成本(万)
KNAP_V = [24, 40, 45, 14, 25, 52]       # 价值/收益(万)
KNAP_CAP = 50                            # 总容量预算


def knapsack_bottomup(cap):
    """自底向上 0-1 背包 DP，返回(最大价值, 选择列表)。"""
    n = len(KNAP_W)
    dp = [0.0] * (cap + 1)
    keep = [[False] * (cap + 1) for _ in range(n)]
    for i in range(n):
        for c in range(cap, KNAP_W[i] - 1, -1):   # 逆序：保证每件只用一次
            cand = dp[c - KNAP_W[i]] + KNAP_V[i]
            if cand > dp[c]:
                dp[c] = cand
                keep[i][c] = True
    chosen, c = [], cap
    for i in range(n - 1, -1, -1):
        if keep[i][c]:
            chosen.append(i)
            c -= KNAP_W[i]
    chosen.reverse()
    return dp[cap], chosen


def knapsack_bruteforce(cap):
    """穷举 2^n 组合求真值，用于核对 DP 结果（n 很小）。"""
    n = len(KNAP_W)
    best_val, best_mask = -1.0, 0
    for mask in range(1 << n):
        w = sum(KNAP_W[i] for i in range(n) if mask >> i & 1)
        if w <= cap:
            v = sum(KNAP_V[i] for i in range(n) if mask >> i & 1)
            if v > best_val:
                best_val, best_mask = v, mask
    return best_val, [i for i in range(n) if best_mask >> i & 1]


def greedy_knapsack(cap):
    """按'单位重量价值'贪心，作为'贪心失效'的反面教材。"""
    order = sorted(range(len(KNAP_W)), key=lambda i: KNAP_V[i] / KNAP_W[i], reverse=True)
    rem, val, chosen = cap, 0.0, []
    for i in order:
        if KNAP_W[i] <= rem:
            rem -= KNAP_W[i]
            val += KNAP_V[i]
            chosen.append(i)
    return val, sorted(chosen)


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 70)
    print("一、设备更新（多阶段决策 · DP 的主战场）")
    print("=" * 70)
    v_td, path_td = solve_equipment_topdown()
    v_bu, path_bu, dp, choice = solve_equipment_bottomup()
    print(f"top-down 记忆化递归  最小 5 年总净成本 = {v_td:.2f} 万元")
    print(f"bottom-up 迭代 DP     最小 5 年总净成本 = {v_bu:.2f} 万元")
    assert abs(v_td - v_bu) < 1e-9, "两种实现必须一致"
    print(f"[一致] 两种 DP 实现结果相同 = {v_bu:.2f} 万元（含期末残值清算，负号=净支出）\n")

    print("最优决策序列（逐年）：")
    print(f"{'年份':<8}{'开年车龄':<12}{'决策':<16}{'本年现金流':<12}")
    for (yr, age, action), (yr2, age2, action2) in zip(path_td, path_bu):
        assert action == action2 and age == age2, "两条回溯路径必须一致"
        cash, _ = year_cashflow(age, action, yr)
        label = "继续用一年" if action == "keep" else "卖旧+买新"
        print(f"{yr:<8}{age:<12}{label:<16}{cash:+.0f}")
    print()

    # 对照策略：永不换机 / 每年都换
    greedy_keep, _ = simulate(lambda age, y: "keep")
    greedy_rep, _ = simulate(lambda age, y: "replace")
    print(f"对照① 贪心'永不换机'   总净成本 = {greedy_keep:.2f} 万元")
    print(f"对照② 贪心'每年都换新' 总净成本 = {greedy_rep:.2f} 万元")
    print(f"→ DP 最优 {v_bu:.2f} 万元，比'永不换机'省 {greedy_keep - v_bu:.2f} 万元，"
          f"比'每年都换'省 {greedy_rep - v_bu:.2f} 万元\n")

    # 资产账本：把最优路径读成现金流
    print("最优路径资产账本（现金流，负=支出）：")
    ledger, total, year, age = [], 0.0, 1, 1
    for y in range(1, T + 1):
        action = choice[y][age]
        cash, next_age = year_cashflow(age, action, y)
        if action == "keep":
            note = f"继续用(龄{age})  运行费 -{RUN_COST[age]:.0f}"
        else:
            note = f"卖旧机(龄{age}) +{RESALE[age]:.0f}，买新车 -{PRICE_NEW:.0f}，新机运行费 -{RUN_COST[1]:.0f}"
        total += cash
        ledger.append((y, age, action, cash, note, total))
        print(f"  第{y}年 龄{age} {'用' if action=='keep' else '换'} | {note:<48} 累计 {total:+.1f}")
        age = next_age
    end_resale = RESALE[min(age, MAX_AGE)]
    total -= end_resale
    print(f"  期末清算 | 车龄{min(age, MAX_AGE)} 卖旧机 +{end_resale:.0f}"
          f"{'':<40} 累计 {total:+.1f}")
    assert abs(total - v_bu) < 1e-9, "账本累计必须等于 DP 最优值"
    print(f"[一致] 账本累计 = DP 最优 = {total:.2f} 万元\n")

    print("=" * 70)
    print("二、0-1 背包（资源分配 · 逐件物品逐容量）")
    print("=" * 70)
    dp_val, dp_sel = knapsack_bottomup(KNAP_CAP)
    bf_val, bf_sel = knapsack_bruteforce(KNAP_CAP)
    gr_val, gr_sel = greedy_knapsack(KNAP_CAP)
    print(f"DP     最大价值 = {dp_val:.0f} 万，选物品 {[i+1 for i in dp_sel]} "
          f"(重量 {sum(KNAP_W[i] for i in dp_sel)}/{KNAP_CAP})")
    print(f"穷举   最大价值 = {bf_val:.0f} 万，选物品 {[i+1 for i in bf_sel]}  ← 真值")
    print(f"贪心   价值 = {gr_val:.0f} 万，选物品 {[i+1 for i in gr_sel]} "
          f"(重量 {sum(KNAP_W[i] for i in gr_sel)}/{KNAP_CAP})")
    assert abs(dp_val - bf_val) < 1e-9, "DP 必须与穷举一致"
    print(f"[一致] DP 与穷举结果相同 = {dp_val:.0f} 万")
    if gr_val < dp_val:
        print(f"[贪心失效] 按性价比贪心只拿到 {gr_val:.0f} 万，比最优少 "
              f"{dp_val - gr_val:.0f} 万（浪费容量 {KNAP_CAP - sum(KNAP_W[i] for i in gr_sel)} 万）")
    else:
        print("[注意] 本例贪心恰好也最优，但贪心对 0-1 背包无最优性保证（见论文讨论）")
    print()

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    # 左：运行费 vs 残值 随车龄，标出最优换机点
    ax = axes[0]
    ages = sorted(RUN_COST.keys())
    ax.plot(ages, [RUN_COST[a] for a in ages], "o-", color="#c0392b", lw=2, label="运行维护费 C(t)")
    ax.plot(ages, [RESALE[a] for a in ages], "s-", color="#2980b9", lw=2, label="残值 s(t)")
    ymax = max(max(RUN_COST.values()), max(RESALE.values()))
    for (yr, age0, action, *_) in ledger:
        if action == "replace":
            ax.axvline(age0, color="#27ae60", ls="--", alpha=0.6)
            ax.text(age0, ymax * 0.92, f"第{yr}年换\n(龄{age0})", color="#27ae60",
                    ha="center", fontsize=9)
    ax.set_xlabel("车龄 t（年）")
    ax.set_ylabel("金额（万元）")
    ax.set_title("设备老化：运行费随龄升、残值随龄降\n（绿虚线=最优路径的换机点）", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()

    # 右：三种策略总净成本对比
    ax = axes[1]
    names = ["DP 最优", "贪心 永不换", "贪心 每年都换"]
    costs = [abs(v_bu), abs(greedy_keep), abs(greedy_rep)]
    colors = ["#27ae60", "#95a5a6", "#e67e22"]
    bars = ax.bar(names, costs, color=colors, width=0.6)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.5, f"{c:.1f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("5 年总净成本（万元，越低越好）")
    ax.set_title("多阶段决策：贪心两种极端 vs DP 最优", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(costs) * 1.18)

    fig.tight_layout()
    png = OUT / "dynamic_programming_multistage.png"
    fig.savefig(png, dpi=130)
    print(f"[输出] 图已保存：{png}")
    print("全部断言通过 ✓")


if __name__ == "__main__":
    main()
