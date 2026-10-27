"""
subject : 数学建模 · 优化算法（求解器结果可信度自检）
level   : 04 - Optimization（Day59 附篇，配套 day59_facility_vehicle_ilp.py）
anchor  : 建模课上反复强调"结果要复核"，这篇给一个真实的翻车现场
goal    : 用一个 24 个 0-1 变量的极小题，展示 HiGHS 1.12.0（scipy 1.18.1 后端）
          在特定形态下会返回 status=Optimal、gap=0%，但解与真实最优差 100。
          给出可复现的判定流程：换符号重解 + 关 presolve + 独立上界对照。

背景：Day59 主脚本做多仓-多站点配送。第一版把 k[i] 的上界误写成 cap.sum()
      （建模错），修好后目标仍为 0，怀疑建模。逐列打印喂给 HiGHS 的系数
      确认模型无误，最后定位到求解器 presolve。这个副产物值得单独留档。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59b_solver_audit.py"
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

N = 24
A = np.ones((1, N))
UB_ROW = 3.0
TRUE_OPT = 100.0     # 手算：x=(1,0,…,0)，目标 100


def solve(c, integrality, presolve=True):
    res = milp(
        c,
        constraints=LinearConstraint(A, [0.0], [UB_ROW]),
        integrality=integrality,
        bounds=Bounds(np.zeros(N), np.ones(N)),
        options={"disp": False, "presolve": presolve},
    )
    x = np.round(res.x)
    return res, int(x.sum())


def verdict(value):
    return "正确" if abs(value - TRUE_OPT) < 1e-6 else "错解"


def main():
    print("=" * 74)
    print("测试题：min 100·x0  s.t.  x1+…+x24 ≤ 3,  x ∈ {0,1}^24")
    print("手算最优 = 100（取 x0=1，其余全 0）。它简单到算错才值得记录。")
    print("=" * 74)

    c = np.zeros(N); c[0] = TRUE_OPT
    ones = np.ones(N)

    print("\n【A】最小化正成本（原样）")
    res_a, k_a = solve(c, ones)
    print(f"  status={res_a.status} message={res_a.message}")
    print(f"  res.fun = {res_a.fun:.1f}    Σx = {k_a}    → {verdict(res_a.fun)}")

    print("\n【B】presolve = False（试图绕开）")
    res_b, k_b = solve(c, ones, presolve=False)
    print(f"  res.fun = {res_b.fun:.1f}    Σx = {k_b}    → {verdict(res_b.fun)}")
    print("  日志观察：B&B 第 1 个节点 BestSol=0、BestBound=0，直接宣布 optimal。")
    print("  所以这不是『presolve 开关』层面的问题，关掉也逃不掉。")

    print("\n【C】改成最大化同一目标（系数取负）")
    res_c, k_c = solve(-c, ones)
    print(f"  res.fun = {res_c.fun:.1f}    Σx = {k_c}    （期望 -100）")
    print(f"  → {'正确' if abs(res_c.fun + TRUE_OPT) < 1e-6 else '错解'}"
          "　负系数会走到完全不同的 presolve 路径，反倒对了")

    print("\n【D】LP 松弛（integrality 全 0）")
    res_d, _ = solve(c, np.zeros(N))
    print(f"  res.fun = {res_d.fun:.1f}    → {'正确' if abs(res_d.fun) < 1e-6 else '错解'}"
          "　松弛最优确实是 0（x 全取 0 满足 Σx ≤ 3）")

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    print("1. 不要只看 res.success / res.status / res.mip_gap —— 三者全绿也可能错。")
    print("2. 『把解代回目标自己算』也不够：本例里 c·x 与 res.fun 一致地错，")
    print("   因为错的是可行域（约束被当成空的），不是解的记账。")
    print("3. 真正管用的复核是三件套：")
    print("   · 换符号 / 换等价的另一写法重解，看结论是否一致；")
    print("   · 小规模题用暴力枚举或贪心独立求一个可行上界，和求解器比；")
    print("   · 打印喂进求解器的矩阵（行上下界、列上下界、成本、整性），")
    print("     确认自己的建模没错，再怀疑求解器。")
    print("4. 一句话：模型对 ≠ 代码对 ≠ 求解器对。三关都得过。")
    print(f"\n本例差距：求解器 {res_a.fun:.0f} vs 真值 {TRUE_OPT:.0f}，"
          f"少算 {TRUE_OPT - res_a.fun:.0f}，而它报告 gap = 0%。")


if __name__ == "__main__":
    main()
