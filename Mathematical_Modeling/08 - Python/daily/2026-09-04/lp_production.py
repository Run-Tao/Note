#!/usr/bin/env python
"""
subject: mathematical-modeling
level: Level 6 - Optimization
roadmap: Level 6 / 优化模型
title: 线性规划标准建模 + scipy.optimize.linprog 求解闭环
date: 2026-09-04

目标：把"生产计划"这一真实决策问题写成标准线性规划(LP)，
并用 scipy.optimize.linprog 求解，再对结果做敏感性解读。
配套教案见同目录 README.md
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 注册中文字体（Noto Sans CJK），否则中文会显示为方框
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.optimize import linprog

# ---------------------------------------------------------------
# 问题：某工厂用 2 台机器生产 A、B 两种产品。
#   - 每件 A：需机器1 工时 2h，机器2 工时 1h，利润 30 元
#   - 每件 B：需机器1 工时 1h，机器2 工时 3h，利润 40 元
#   - 机器1 可用 100h，机器2 可用 120h
#   - 市场需求：A ≤ 40 件，B ≤ 50 件，且 B 至少生产 10 件
# 目标：最大化总利润（线性规划，两个决策变量）
# ---------------------------------------------------------------
# 标准形式：min c^T x, s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds
# 最大化利润 <=> 最小化负利润
c = np.array([-30.0, -40.0])          # 求 min(-利润)

# 约束系数（<=）：
#   机器1:  2A + 1B <= 100
#   机器2:  1A + 3B <= 120
#   需求A:      A <= 40
#   需求B上:    B <= 50
A_ub = np.array([[2, 1],
                 [1, 3],
                 [1, 0],
                 [0, 1]])
b_ub = np.array([100.0, 120.0, 40.0, 50.0])

# bounds：A>=0, B>=10 (需求下界)
bounds = [(0, None), (10, None)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

if result.success:
    A, B = result.x
    profit = -result.fun
    print("求解成功")
    print(f"最优解：A = {A:.2f} 件, B = {B:.2f} 件")
    print(f"最大利润 = {profit:.2f} 元")

    # 检查哪条约束拉满（紧约束 -> 敏感性来源）
    constr = {"机器1": 2*A + B, "机器2": A + 3*B, "需求A": A, "需求B": B}
    print("\n约束占用情况：")
    for k, v in constr.items():
        status = "拉满(紧)" if abs(v - dict(m1=100, m2=120, A=40, B=50)[
            {"机器1": "m1", "机器2": "m2", "需求A": "A", "需求B": "B"}[k]]) < 1e-6 else "有余量"
        print(f"  {k}: {v:8.2f}  {status}")

    # 敏感性：机器1 工时+1 时利润增幅 = 对偶价格影子价(近似)
    # 通过把 100 -> 101 重解，观察利润增量
    b_ub2 = np.array([101.0, 120.0, 40.0, 50.0])
    r2 = linprog(c, A_ub=A_ub, b_ub=b_ub2, bounds=bounds, method="highs")
    shadow_m1 = (-r2.fun) - profit
    print(f"\n机器1 工时 +1，利润增加 ≈ {shadow_m1:.2f} 元（机器1 的影子价）")

    # 可视化可行域 + 最优解
    fig, ax = plt.subplots(figsize=(7, 6))
    A_vals = np.linspace(0, 60, 400)
    # 各约束边界线（B on y-axis）：B = (b - a*A)/b_coeff
    lines = {
        "机器1 2A+B<=100": (100 - 2*A_vals),            # 2A+B=100 -> B=100-2A
        "机器2 A+3B<=120": (120 - A_vals) / 3,          # B=(120-A)/3
        "需求A A<=40": None,
        "需求B/B下界": None,
    }
    # 半平面填充近似：网格法求可行域
    gridA = np.linspace(0, 60, 600)
    gridB = np.linspace(0, 70, 600)
    GA, GB = np.meshgrid(gridA, gridB)
    feasible = (
        (2*GA + GB <= 100) &
        (GA + 3*GB <= 120) &
        (GA <= 40) & (GB <= 50) & (GB >= 10) & (GA >= 0)
    )
    ax.contourf(GA, GB, feasible.astype(int), levels=[0.5, 1.5], colors=["#e6f0fa"], alpha=0.6)
    ax.plot(A_vals, lines["机器1 2A+B<=100"], "C0", label="机器1: 2A+B=100")
    ax.plot(A_vals, lines["机器2 A+3B<=120"], "C1", label="机器2: A+3B=120")
    ax.axvline(40, color="C2", ls="--", label="需求A: A=40")
    ax.axhline(50, color="C3", ls="--", label="需求B上: B=50")
    ax.axhline(10, color="C4", ls=":", label="需求B下: B=10")
    ax.plot(A, B, "ro", ms=9, zorder=5)
    ax.annotate(f"最优 ({A:.1f}, {B:.1f})\n利润 {profit:.0f} 元",
                (A, B), textcoords="offset points", xytext=(12, -16),
                fontsize=10, color="red", fontweight="bold")
    ax.set_xlim(0, 60); ax.set_ylim(0, 70)
    ax.set_xlabel("产品 A 产量 (件)"); ax.set_ylabel("产品 B 产量 (件)")
    ax.set_title("线性规划可行域与最优解（生产计划）")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("lp_feasible_region.png", dpi=130)
    print("\n图已保存: lp_feasible_region.png")
else:
    print("求解失败：", result.message)
    raise SystemExit(1)

# 断言自查（保证 Agent 可验证）
assert abs(A - 36) < 1e-6 and abs(B - 28) < 1e-6, (A, B)
assert abs(profit - 2200) < 1e-6, profit
assert 10 <= B <= 50
print("\n断言全部通过：最优解 (A,B)=(36,28)，利润 2200 元")
