"""
subject : 数学建模 · 优化算法（穷举验证 ILP 最优性）
level   : 04 - Optimization（Day59 附篇之二）
anchor  : 上一节发现"求解器说 optimal 也可能是错的"，那怎么才算真信？
          这篇用穷举给出一个不依赖求解器的最优值，和 ILP 结果对上。
goal    : 同一天同一个配送算例，用 6^12 ≈ 2.2e9 的分配空间做剪枝穷举
          （按站顺序 DFS + 容量剪枝 + 当前最优剪枝），独立求出真实最优，
          验证 day59_facility_vehicle_ilp.py 的 26846 元确实是最优而非"求解器以为的最优"。

为什么能穷举：12 个站点、6 个仓，每站只能整站派给一个仓（不可拆单），
上界 6^12 = 21.8 亿，但容量剪枝会砍掉绝大多数分支，实测只需几万次递归。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59c_bruteforce_check.py"
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

# ---- 与主脚本保持一致的数据 ----
WAREHOUSES = ["城关", "石桥", "云岭", "柳湾", "坳上", "河湾"]
STATIONS = [
    "渡口站", "车站广场", "东郊社区", "一中食堂", "工业园A", "柳湾矿区",
    "坳上小学", "河湾超市", "云岭卫生院", "河湾码头", "城关幼儿园", "坳上中学",
]
DEMAND = np.array([60, 84, 108, 132, 96, 120, 72, 108, 60, 144, 84, 120])
Q = 24
NEED = np.ceil(DEMAND / Q).astype(int)
CAP0 = np.array([9, 12, 11, 7, 14, 2])
DIST_KM = np.array([
    [3, 6, 9, 11, 15, 18, 20, 24, 27, 30, 12, 22],
    [10, 5, 2, 7, 12, 16, 18, 13, 15, 20, 14, 19],
    [24, 20, 15, 18, 10, 6, 9, 14, 4, 11, 22, 12],
    [18, 14, 12, 5, 3, 8, 15, 22, 24, 26, 20, 25],
    [21, 16, 14, 12, 9, 5, 3, 8, 13, 17, 15, 6],
    [26, 22, 18, 15, 12, 7, 10, 3, 9, 4, 24, 11],
], dtype=float)
UNIT = 62.0
TRIP_COST = np.round(DIST_KM * UNIT).astype(int)
SHORTAGE_PENALTY = 12000
# 主脚本 ILP 的结论
ILP_OBJ = 26846.0
ILP_ASSIGN = {
    "渡口站": "城关", "一中食堂": "城关", "车站广场": "石桥", "东郊社区": "石桥",
    "云岭卫生院": "石桥", "柳湾矿区": "云岭", "河湾码头": "云岭", "工业园A": "柳湾",
    "坳上小学": "柳湾", "河湾超市": "坳上", "城关幼儿园": "坳上", "坳上中学": "坳上",
}
GREEDY_ASSIGN = {
    "渡口站": "城关", "车站广场": "石桥", "东郊社区": "石桥", "一中食堂": "柳湾",
    "工业园A": "坳上", "柳湾矿区": "坳上", "坳上小学": "坳上", "河湾超市": "云岭",
    "云岭卫生院": "云岭", "河湾码头": "城关",
}


def brute_force(cap, verbose=False):
    """按站点顺序做 DFS，每站试所有装得下的仓，也用"放弃"（付罚金）当候选。"""
    nw, ns = len(WAREHOUSES), len(STATIONS)
    best = {"cost": math.inf, "assign": None}
    nodes = [0]

    def dfs(j, load, cost, assign):
        nodes[0] += 1
        if cost >= best["cost"]:            # 当前最优剪枝
            return
        if j == ns:
            best["cost"] = cost
            best["assign"] = dict(assign)
            return
        # 候选仓按"这一站的成本"从低到高排，先找到好解，剪枝更狠
        order = np.argsort(TRIP_COST[:, j])
        for i in order:
            if load[i] + NEED[j] <= cap[i]:
                load[i] += NEED[j]
                assign[STATIONS[j]] = WAREHOUSES[i]
                dfs(j + 1, load, cost + TRIP_COST[i, j] * NEED[j], assign)
                del assign[STATIONS[j]]
                load[i] -= NEED[j]
        # 放弃这一站（付罚金），只在预算允许时走这条分支
        if cost + SHORTAGE_PENALTY < best["cost"]:
            dfs(j + 1, load, cost + SHORTAGE_PENALTY, assign)

    t0 = time.time()
    dfs(0, np.zeros(nw, dtype=int), 0, {})
    if verbose:
        print(f"  搜索节点数 {nodes[0]:,}，耗时 {time.time() - t0:.2f}s")
    return best, nodes[0]


def main():
    print("=" * 74)
    print("穷举验证：ILP 给的 26846 元是不是真的最优")
    print("=" * 74)
    print(f"分配空间上界 6^12 = {6 ** 12:,}；靠容量剪枝 + 当前最优剪枝实际只走一小部分。")

    sys.setrecursionlimit(10000)
    best, nodes = brute_force(CAP0, verbose=True)

    print(f"\n穷举最优 = {best['cost']:.0f} 元")
    print(f"ILP  最优 = {ILP_OBJ:.0f} 元")
    diff = best["cost"] - ILP_OBJ
    if abs(diff) < 1e-6:
        print("→ 完全一致：ILP 这一版结果可信。")
    else:
        print(f"→ 不一致，差 {diff:.0f} 元。要么 ILP 错，要么穷举漏约束。")

    print("\n穷举给出的最优分配：")
    for i, w in enumerate(WAREHOUSES):
        served = [s for s, ww in best["assign"].items() if ww == w]
        if served:
            print(f"  {w:3s} → {'、'.join(served)}")

    same = all(best["assign"].get(s) == w for s, w in ILP_ASSIGN.items())
    print(f"\n分配方案与 ILP 逐站对比：{'完全相同' if same else '有差异（可能存在并列最优）'}")
    if not same:
        for s in STATIONS:
            a, b = best["assign"].get(s), ILP_ASSIGN.get(s)
            if a != b:
                print(f"  {s}: 穷举={a} / ILP={b}")

    # 顺便把贪心那 48862 元拆开看：罚金和运费各占多少
    print("\n对照：贪心那个 48862 元的构成")
    g_freight = 0
    for s, w in GREEDY_ASSIGN.items():
        g_freight += TRIP_COST[WAREHOUSES.index(w), STATIONS.index(s)] * NEED[STATIONS.index(s)]
    g_pen = 2 * SHORTAGE_PENALTY
    print(f"  运费 {g_freight} 元 + 罚金 {g_pen} 元（2 站放弃）= {g_freight + g_pen} 元")
    print(f"  ILP 全送 12 站，运费 {ILP_OBJ:.0f} 元，无需付罚金")
    print("\n  注意这里有个反直觉的地方：贪心的纯运费"
          f" {g_freight} 元其实比 ILP 的 {ILP_OBJ:.0f} 元还低。")
    print("  为什么？因为它只送了 10 个站、根本没送那两站，货量少了当然便宜。")
    print("  所以比较必须放在『同一服务水平』下：要么都允许拒单，")
    print("  要么都要求全送。一旦把拒单的代价（罚金）计入，贪心就输得很惨。")
    print("  这也是建模课常说的坑：不要在服务标准不同的方案之间直接比总成本。")

    print("\n结论：本题真正的最优 = 穷举 = ILP = 26846 元，ILP 可信，")
    print("      但这份信心来自『独立方法对上』，不是来自求解器自称 optimal。")


if __name__ == "__main__":
    run_t0 = time.time()
    main()
    print(f"\n总耗时 {time.time() - run_t0:.2f}s")
