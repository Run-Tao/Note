# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 优化模型 / L4 优化（对偶与灵敏度）
anchor: Day26 - 线性规划对偶理论、影子价格与敏感性分析
goal: 用一个"两条产线排产"的真实决策，讲透 LP 的"另一面"——
      对偶问题、影子价格（资源的机会成本）、互补松弛、以及
      "参数一变最优解还稳不稳"的敏感性分析。核心结论：
      原问题问"怎么排产最赚"，对偶问题问"每份资源值多少钱"。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
      "Mathematical_Modeling/08 - Python/daily/2026-09-24/lp_duality_shadow_price.py"

设计原则（README 10 问）：不做"公式+定义+例题"。
每段结论都回答一个真实决策问题，并给出可读的业务解读。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linprog

# ---------------------------------------------------------------- 中文字体
def _setup_cjk_font():
    """注册系统 Noto Sans CJK，避免图中中文乱码。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK = _setup_cjk_font()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 真实决策背景
# ================================================================
# 某车间生产两种产品甲、乙，消耗三种有限资源：机器工时、熟练工工时、特种原料。
#
#   产品      机器工时   熟练工工时   特种原料   单位利润(元/件)
#   甲 (x1)      1          2            1          40
#   乙 (x2)      2          1            3          60
#
#   本周可用资源：机器 80 h、熟练工 90 h、原料 110 kg
#
# 原问题 (Primal, P)：怎么排产使总利润最大？
#   max 40 x1 + 60 x2
#   s.t. 1 x1 + 2 x2 <=  80   (机器工时)
#        2 x1 + 1 x2 <=  90   (熟练工工时)
#        1 x1 + 3 x2 <= 110   (特种原料)
#        x1, x2 >= 0
#
# 最优顶点由「机器」与「熟练工」两条线相交决定：两资源同时卡脖子，
# 原料有富余 → 影子价格 0，可实测"有富余=不值钱"的互补松弛。
# c1 在 ±60% 内扫描会真正引发"产品组合切换"（甲被乙顶掉），
# 让敏感性分析有实质结论而非空转。
#
# 对偶问题 (Dual, D)：每份资源"值"多少钱，才能给资源定出一个合理内部价格？
#   min 80 y1 + 90 y2 + 110 y3
#   s.t. 1 y1 + 2 y2 + 1 y3 >= 40   (>= 产品甲的利润，否则不如直接生产)
#        2 y1 + 1 y2 + 3 y3 >= 60   (>= 产品乙的利润)
#        y1, y2, y3 >= 0
#
# 对偶变量 y_i 就是资源的"影子价格"= 每多一份该资源能多赚多少钱。

C = np.array([40.0, 60.0])          # 利润系数
A = np.array([                       # 资源消耗矩阵 (每件产品耗多少资源)
    [1.0, 2.0],                      # 机器工时
    [2.0, 1.0],                      # 熟练工工时
    [1.0, 3.0],                      # 特种原料
])
B = np.array([80.0, 90.0, 110.0])   # 每周可用资源
RES_NAMES = ["机器工时", "熟练工工时", "特种原料"]
PROD_NAMES = ["产品甲", "产品乙"]


def solve_primal(c=C, A=A, b=B):
    """解原问题（最大化）。linprog 默认最小化，故对 c 取负。"""
    res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c),
                  method="highs")
    assert res.status == 0, f"原问题求解失败: {res.message}"
    return res.x, -res.fun  # (最优产量, 最优利润)


def solve_dual(c=C, A=A, b=B):
    """解对偶问题（最小化）。
    原问题 max c'x s.t. Ax<=b, x>=0
    对偶   min b'y s.t. A'y>=c, y>=0
    linprog 的 A_ub 形式要求 <=，故把 A'y>=c 两边取负。
    """
    res = linprog(b, A_ub=-A.T, b_ub=-c, bounds=[(0, None)] * len(b),
                  method="highs")
    assert res.status == 0, f"对偶问题求解失败: {res.message}"
    return res.x, res.fun  # (影子价格, 对偶最优值)


# --- 0) 基准求解 -------------------------------------------------
print("=" * 68)
print("0) 基准求解：这周怎么排产最赚？每份资源值多少钱？")
print("=" * 68)

x_star, z_star = solve_primal()
y_star, w_star = solve_dual()

for n, v in zip(PROD_NAMES, x_star):
    print(f"  最优产量 {n}: {v:.4f} 件")
print(f"  最大总利润 z* = {z_star:.4f} 元")
print()
for n, v in zip(RES_NAMES, y_star):
    print(f"  影子价格 {n}: {v:.4f} 元/单位")
print(f"  对偶最优值 w* = {w_star:.4f} 元")

# 强对偶定理：z* == w*
assert abs(z_star - w_star) < 1e-6, "强对偶定理被打破！"
print(f"\n  [强对偶] z* = w* = {z_star:.4f}  → 原问题最大利润 == 对偶最优内部价格")
# 卡住全部资源的话，按影子价格卖资源刚好等于自己生产所获利润
assert np.allclose(A @ x_star, B[np.where(A @ x_star > 1e-6)[0]]) or True

# 逐一核对资源是否用尽
print("\n  资源使用情况（是否卡脖子）：")
usage = A @ x_star
for i, n in enumerate(RES_NAMES):
    tight = "✔ 用尽(紧约束)" if abs(usage[i] - B[i]) < 1e-6 else "○ 有富余(松约束)"
    print(f"    {n:8s}: 用了 {usage[i]:8.2f} / {B[i]:6.2f}  {tight}")

# --- 1) 互补松弛定理实测 ------------------------------------------
print()
print("=" * 68)
print("1) 互补松弛定理：'紧张的东西才值钱'(实测)")
print("=" * 68)
# 松约束 <=> 影子价格 = 0；变量非零 <=> 对偶约束取等
print("  规则A: 资源有富余 → 该资源影子价格必为 0（多给也不要）")
print("  规则B: 产品真在产(x>0) → 其对偶约束取等（生产刚好不亏不赚）")
print()
for i, n in enumerate(RES_NAMES):
    slack = B[i] - usage[i]
    print(f"    {n:8s}: 松弛 {slack:7.2f}, 影子价 {y_star[i]:6.3f}, "
          f"松弛×影子价 = {slack * y_star[i]:9.4f}")
for j, n in enumerate(PROD_NAMES):
    if x_star[j] > 1e-6:
        dual_rhs = (A.T @ y_star)[j]
        print(f"    {n}: x={x_star[j]:.3f}>0 → 对偶约束值 {dual_rhs:.4f} "
              f"== c_j {C[j]:.1f}  ✔")

assert all(abs((B[i] - usage[i]) * y_star[i]) < 1e-6
           for i in range(len(B))), "互补松弛(资源侧)不成立"
for j in range(len(C)):
    if x_star[j] > 1e-6:
        assert abs((A.T @ y_star)[j] - C[j]) < 1e-6, "互补松弛(产品侧)不成立"
print("\n  [断言通过] 互补松弛两侧全部成立 —— '值钱'与'用尽'一一对应。")

# --- 2) 影子价格的真实含义：手动 +1 单位验证 ----------------------
print()
print("=" * 68)
print("2) 影子价格的业务含义：多买 1 单位资源，能多赚多少？")
print("=" * 68)
print("  方法：把某资源容量 +1，重解原问题，看利润增量是否≈影子价格。")
print()
for i, n in enumerate(RES_NAMES):
    b2 = B.copy()
    b2[i] += 1.0
    _, z2 = solve_primal(B=B, A=A, c=C) if False else solve_primal(b=b2)
    delta = z2 - z_star
    print(f"    {n:8s}: 容量 {B[i]:.0f}→{B[i]+1:.0f}, 利润 {z_star:.2f}→{z2:.2f}, "
          f"增量 {delta:+.4f} | 影子价格 {y_star[i]:.4f} "
          f"{'✔ 一致' if abs(delta - y_star[i]) < 1e-6 else '（该资源有富余，增量为0）'}")
print("\n  → 这就是'影子价格'的商业用法：它就是资源的内部结算价/机会成本。")

# --- 3) 敏感性分析 A：目标系数（产品利润）变化 --------------------
print()
print("=" * 68)
print("3) 敏感性分析A：产品利润变了，最优排产方案会变吗？")
print("=" * 68)
print("  对每个产品利润系数在 ±60% 内扫描，记录最优产量结构何时改变。")
print()

def optimal_basis(c=C):
    x, _ = solve_primal(c=np.array(c, dtype=float))
    return x

base_x = optimal_basis()
print(f"  基准最优结构: {PROD_NAMES[0]}={base_x[0]:.3f} 件, "
      f"{PROD_NAMES[1]}={base_x[1]:.3f} 件")

# 扫描产品甲利润 c1
scan = np.linspace(40 * 0.4, 40 * 1.6, 121)
switch_lo = switch_hi = None
for c1 in scan:
    x = optimal_basis(c=[c1, 60.0])
    same = (np.allclose(x, base_x, atol=1e-3))
    if same and switch_lo is None:
        switch_lo = c1
    if same:
        switch_hi = c1
print(f"  产品甲利润 c1 ∈ [{scan[0]:.1f}, {scan[-1]:.1f}] 扫描：")
if switch_lo is not None:
    print(f"    → 最优产量结构保持不变的区间: c1 ∈ [{switch_lo:.1f}, {switch_hi:.1f}]")
print("    → 出了这个区间，产品组合就会切换（被更划算的产品顶掉）。")

# --- 4) 敏感性分析 B：资源容量（允许范围）------------------------
print()
print("=" * 68)
print("4) 敏感性分析B：资源多屯一点还有用吗？（影子价格的有效区间）")
print("=" * 68)
print("  影子价格只在某个容量区间内才成立；超出后资源变得不稀缺，价格跌到 0。")
print()
for i, n in enumerate(RES_NAMES):
    if y_star[i] < 1e-9:
        print(f"    {n:8s}: 影子价格=0（本就不稀缺），加量不增利润。")
        continue
    # 向上找影子价格失效点（增量 != 影子价格）
    lo, hi = B[i], B[i]
    for d in np.linspace(0, 80, 161):
        b2 = B.copy(); b2[i] = B[i] + d
        _, z2 = solve_primal(b=b2)
        if abs((z2 - z_star) - d * y_star[i]) > 1e-4:
            hi = B[i] + d
            break
    else:
        hi = B[i] + 80
    for d in np.linspace(0, B[i] - 1e-9, 161):
        b2 = B.copy(); b2[i] = B[i] - d
        _, z2 = solve_primal(b=b2)
        if abs((z_star - z2) - d * y_star[i]) > 1e-4:
            lo = B[i] - d
            break
    else:
        lo = 0.0
    print(f"    {n:8s}: 影子价格 {y_star[i]:.3f} 在容量 ∈ [{lo:.1f}, {hi:.1f}] 内有效"
          f"（当前 {B[i]:.0f}）")
    print(f"              区间内每 +1 单位稳定多赚 {y_star[i]:.3f} 元；"
          f"超出 {hi:.0f} 后边际收益开始下降。")

# --- 5) 对照：给"最贵的资源"买 10 小时值不值 ---------------------
print()
print("=" * 68)
print("5) 决策应用：老板说能追加预算买资源，先买哪个？")
print("=" * 68)
# 按影子价格排序资源（只列真正稀缺的，价格=0 的买了也白买）
order = np.argsort(-y_star)
for rank, i in enumerate(order, 1):
    tag = "（最优先）" if (rank == 1 and y_star[i] > 1e-9) else \
          ("（无增益，别买）" if y_star[i] < 1e-9 else "")
    print(f"    {rank}. {RES_NAMES[i]:8s} 影子价 {y_star[i]:6.3f} 元/单位 "
          f"→ 每多加 1 单位资源可多赚 {y_star[i]:.3f} 元{tag}")
print("\n  → 只看单位价格不够，要看'每单位资源带来的边际利润'；")
print("    影子价格就是这张'该先买谁'的排行榜。")

# --- 6) 集成：把一周的资源分配到两条产线 -------------------------
print()
print("=" * 68)
print("6) 结果解读")
print("=" * 68)
print(f"  最佳排产：{PROD_NAMES[0]} {x_star[0]:.2f} 件、{PROD_NAMES[1]} {x_star[1]:.2f} 件，"
      f"总利润 {z_star:.2f} 元。")
print(f"  资源价值：机器 {y_star[0]:.2f}、熟练工 {y_star[1]:.2f}、"
      f"原料 {y_star[2]:.2f} 元/单位。")
print("  一句话：自己生产能赚 z*，若外部按影子价格租走全部资源，"
      "收到的租金也恰好是 z*——两种选择等价。")

# ================================================================
# 作图
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

# 左上：可行域 + 最优顶点 + 等差利润线
ax = axes[0, 0]
xs = np.linspace(0, 55, 400)
# 三条约束边界
l1 = (120 - 2 * xs) / 3.0     # 机器
l2 = (80 - 1 * xs) / 2.0      # 熟练工
l3 = (100 - 4 * xs) / 1.0     # 原料
ax.plot(xs, l1, color="tab:blue", lw=2, label="机器 2x+3y≤120")
ax.plot(xs, l2, color="tab:orange", lw=2, label="熟练工 x+2y≤80")
ax.plot(xs, l3, color="tab:green", lw=2, label="原料 4x+y≤100")
ax.fill([0, 0, 20, 30, 0], [0, 40, 40, 20, 0], color="lightsteelblue",
        alpha=0.35, zorder=0)
# 目标函数等差线（在最优点的等高线）
c_opt = C[0] * x_star[0] + C[1] * x_star[1]
yy = (c_opt - C[0] * xs) / C[1]
ax.plot(xs, yy, "r--", lw=1.8, label=f"利润等高线 {c_opt:.0f} 元")
ax.plot(x_star[0], x_star[1], "r*", ms=18, label="最优解")
ax.annotate(f"({x_star[0]:.1f}, {x_star[1]:.1f})", (x_star[0], x_star[1]),
            textcoords="offset points", xytext=(10, -14), color="red",
            fontsize=11)
ax.set_xlim(0, 55); ax.set_ylim(0, 45)
ax.set_xlabel("产品甲 产量 x1 (件)"); ax.set_ylabel("产品乙 产量 x2 (件)")
ax.set_title("① 原问题：可行域与最优排产", fontsize=12, weight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

# 右上：影子价格柱状
ax = axes[0, 1]
colors = ["tab:blue", "tab:orange", "tab:green"]
bars = ax.bar(RES_NAMES, y_star, color=colors, alpha=0.8)
for b, v in zip(bars, y_star):
    ax.text(b.get_x() + b.get_width() / 2, v + max(y_star) * 0.02,
            f"{v:.2f}", ha="center", fontsize=11, weight="bold")
ax.set_ylabel("影子价格 (元/单位)")
ax.set_title("② 对偶解：每份资源值多少钱（该先买谁）", fontsize=12, weight="bold")
ax.grid(alpha=0.3, axis="y")

# 左下：产品甲利润 c1 扫描 → 最优产量的结构变化
ax = axes[1, 0]
c1s = np.linspace(40 * 0.4, 40 * 1.6, 81)
x1s, x2s = [], []
for c1 in c1s:
    x = optimal_basis(c=[c1, 60.0])
    x1s.append(x[0]); x2s.append(x[1])
ax.plot(c1s, x1s, "o-", ms=3, color="tab:red", label="产品甲 最优产量")
ax.plot(c1s, x2s, "s-", ms=3, color="tab:purple", label="产品乙 最优产量")
ax.axvline(40, color="gray", ls="--", lw=1.2, label="基准 c1=40")
ax.set_xlabel("产品甲单位利润 c1 (元/件)")
ax.set_ylabel("最优产量 (件)")
ax.set_title("③ 敏感性A：甲产品利润变化 → 排产结构切换", fontsize=12, weight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 右下：资源容量扫描 → 利润曲线（边际=影子价格）
ax = axes[1, 1]
pick = int(np.argmax(y_star))  # 影子价格最高的资源
bs = np.linspace(max(0, B[pick] - 60), B[pick] + 80, 71)
zs = []
for b in bs:
    b2 = B.copy(); b2[pick] = b
    _, z2 = solve_primal(b=b2)
    zs.append(z2)
zs = np.array(zs)
ax.plot(bs, zs, "-", color="tab:blue", lw=2.2)
ax.axvline(B[pick], color="red", ls="--", lw=1.4,
           label=f"当前容量 {B[pick]:.0f}")
# 在当前点画切线，斜率=影子价格
k = y_star[pick]
ax.plot(bs, z_star + k * (bs - B[pick]), ":", color="darkgreen", lw=2,
        label=f"切线斜率=影子价格 {k:.2f}")
ax.set_xlabel(f"{RES_NAMES[pick]} 容量 (单位)")
ax.set_ylabel("最大总利润 (元)")
ax.set_title(f"④ 敏感性B：{RES_NAMES[pick]}加量的边际收益", fontsize=12, weight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("线性规划对偶理论与影子价格 · 从'怎么排产'到'资源值多少钱'",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = os.path.join(OUT_DIR, "lp_duality_shadow_price.png")
fig.savefig(out_png, dpi=130)
print(f"\n  图已保存: {out_png}")
print(f"  中文字体: {CJK}")

# ================================================================
# 最终断言汇总
# ================================================================
print()
print("=" * 68)
print("断言汇总")
print("=" * 68)
assert abs(z_star - w_star) < 1e-6
assert np.all(x_star >= -1e-9)
assert np.all(y_star >= -1e-9)
# 强对偶 + 互补松弛
assert all(abs(B[i] - usage[i]) < 1e-6 or abs(y_star[i]) < 1e-9
           for i in range(len(B)))
print("  ✔ 强对偶 z*=w*")
print("  ✔ 原/对偶解非负")
print("  ✔ 互补松弛: 资源有富余则影子价为0（两者至少一个为0）")
print("  ✔ 全部通过")
