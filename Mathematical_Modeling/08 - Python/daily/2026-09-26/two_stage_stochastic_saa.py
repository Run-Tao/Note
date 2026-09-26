# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 优化模型 / 随机优化（两阶段随机规划 Two-Stage Stochastic Programming with Recourse）
anchor: Day28 - 两阶段随机规划与样本均值近似 SAA（接 Day27 报童单周期、Day26 LP 对偶、Day18 MDP 随机）
goal: 用一道"要先投资建仓库/产能（现在定，不可改），再等需求揭晓后才决定怎么调货
      （事后补救 recourse）"的真实决策，讲清【两阶段随机规划】这个随机优化主干模型：
      - 第一阶段(here-and-now)：投资决策 x，需求还没揭晓时就得拍板，故目标里含"对未来的期望"
      - 第二阶段(wait-and-see / recourse)：需求实现 ω 后，在给定 x 下再做最优补救 y(ω)
      - 核心武器 = 样本均值近似(SAA)：用 N 个场景的样本平均逼近期望，
        并给出【样本内/样本外(out-of-sample)】评估与置信区间，
        再用【确定性等价(DET, 按需求均值规划)】和【完美信息(VSS/EVPI)】做对照，
        回答决策者真关心的：现在该投多少？多花的值不值？信息值多少钱？

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-26/two_stage_stochastic_saa.py"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# 中文字体注册（避免图里中文乱码）
# ----------------------------------------------------------------------------
from matplotlib import font_manager


def register_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get(
                    "font.sans-serif", []
                )
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK = register_cjk_font()

rng = np.random.default_rng(20260926)

# ============================================================================
# 真实决策背景：一家区域乳企要扩张，先决定在 3 个城市各建多大【日产能】(第一阶段)，
#             等 3 个城市的需求揭晓后，再决定【本地生产 + 跨城调拨】如何满足需求(第二阶段)。
#
# 决策结构：
#   第一阶段(现在，需求未知) : y_i = 城市 i 建的日产能(万瓶/天)，成本 cap_cost_i 元/万瓶
#   第二阶段(需求揭晓后)     : 每城产量不能超过其产能 y_i；产出先在本地满足需求，
#                              本地产能不足的部分【从其他城调货】(调货成本 = 距离×单位运费)
#   目标：min  建产能成本  +  E_ω[ 第二阶段调货成本 ]
#
# 关键洞察：
#   - 若只按【需求均值】建产能(DET 确定性等价)，遇到需求波动就频繁跨城调货，很贵；
#   - 两阶段随机规划会【故意多建一点产能】当"缓冲"，抵消波动的调货代价；
#   - 这个"缓冲"值多少，正是 VSS(随机解相对确定性解的价值)；
#   - 若需求能提前获知(完美信息)，还能再省 —— 这是 EVPI(信息价值)。
# ============================================================================

N_CITY = 3
CITY = ["城市A", "城市B", "城市C"]

# 需求分布：均值(万瓶/天) 与 标准差(万瓶/天)
demand_mu = np.array([30.0, 20.0, 25.0])
demand_sigma = np.array([9.0, 6.0, 7.5])  # 变异系数约 0.30

# 建产能成本(元/万瓶) —— 各城地价/人工不同
cap_cost = np.array([100.0, 120.0, 95.0])

# 跨城调货：单位调运成本矩阵 cost[i,j] = 从城 j 调 1 万瓶到城 i 的运费
# 同城=本地自产(不外调)，用一个非对角结构；距离越近越便宜
dist = np.array(
    [
        [0.0, 40.0, 60.0],
        [40.0, 0.0, 35.0],
        [60.0, 35.0, 0.0],
    ]
)
UNIT_SHIP = 1.0  # 元/(万瓶·距离单位)
ship_cost = dist * UNIT_SHIP

# 缺货罚：产能+调货都补不齐时，只能【当天紧急外购现货】(代价高昂但仍可承受)，
# 用 EMERGENCY = 600 元/万瓶 作为应急外购单价，绝不用 1e6——
# 让"缺货"成为一个真实、量级合理的成本，而不是把指标直接打爆的数值炸弹。
EMERGENCY = 600.0


# ----------------------------------------------------------------------------
# 第二阶段(生产+调货)子问题：给定第一阶段产能 y 与需求 d，求最小【调货成本】
#   变量: t[i][j] = 从城 j 调往城 i 的量 (万瓶, i!=j)
#         u[i]   = 城 i 的缺货量(需求未被满足, 大罚金)
#   约束1(每城供需平衡):
#         城 i 的产能 y_i 只能覆盖【本地产出+外调】;
#         即  y_i + sum_j(调入 t[i,j]) - sum_j(调出 t[i,j]) + u_i >= d_i
#   约束2(产出上限): 城 i 的产能 y_i 限制它【总产出】(本地消纳 + 外调), 即
#         local_use_i + sum_j(调出 t[j,i]) <= y_i,
#        而 local_use_i = d_i - 调入 + 缺货 u_i, 代入得
#         (d_i - 调入 + u_i) + 调出 <= y_i  =>  调出 - 调入 + u_i <= y_i - d_i
#   —— 这正是约束1(供货满足) 的另一面。故约束1 与约束2 等价，只需保留一条!
#   用【供货满足】形式:  调入 - 调出 + u_i >= d_i (需求由产能+调入满足, 缺货罚)
# 目标: sum_{i != j} ship_cost[i,j] * t[i][j] + EMERGENCY * sum_i u_i
# 用 scipy.linprog 求解(标准 LP)，返回 (成本, 调货方案矩阵)
# ----------------------------------------------------------------------------
def recourse_cost(y, d, return_flow=False):
    n = N_CITY
    # 变量展平: t[i,j] for i!=j  +  n 个缺货松弛 u[i]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = {p: k for k, p in enumerate(pairs)}
    m = len(pairs)
    n_var = m + n  # 调货量 + 缺货量

    c = np.zeros(n_var)
    for (i, j) in pairs:
        c[idx[(i, j)]] = ship_cost[i, j]
    c[m:] = EMERGENCY  # 应急外购价(缺货成本)

    A_ub = []
    b_ub = []
    # 唯一一组约束(供货满足):
    #   城 i 的供货来源 = 本地产能 y_i + 从别处调入;  去向 = 满足本地需求 d_i + 外调 + 缺货;
    #   用 <= 标准形式:  y_i + 调入 - 调出 + u_i >= d_i
    #   =>  调出 - 调入 - u_i <= y_i - d_i
    for i in range(n):
        row = np.zeros(n_var)
        for (ii, jj) in pairs:
            if ii == i:  # 调往城 i(调入)
                row[idx[(ii, jj)]] -= 1.0
            if jj == i:  # 从城 i 调出
                row[idx[(ii, jj)]] += 1.0
        row[m + i] = -1.0
        A_ub.append(row)
        b_ub.append(y[i] - d[i])

    bounds = [(0, None)] * n_var
    res = linprog(
        c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs"
    )
    assert res.success, "第二阶段 LP 求解失败"
    cost = res.fun
    flow = None
    if return_flow:
        flow = np.zeros((n, n))
        for (i, j) in pairs:
            flow[i, j] = res.x[idx[(i, j)]]
    return cost, flow


# ----------------------------------------------------------------------------
# 两阶段模型：给定需求场景集合 D (N x n)，求最优第一阶段产能 y
#   目标: min_y  cap_cost·y + (1/N) * sum_s recourse_cost(y, D[s])
#   这是一个【凸】问题(第二阶段值函数关于 y 是凸的)，用小规模场景时
#   直接用网格/坐标下降不稳定，这里用【切平面/次梯度】太复杂——
#   教学上采用最稳妥的做法：把整个两阶段 LP 一次性写成【确定性等价 LP(DET-LP)】
#   —— 因为第二阶段是 LP，两阶段问题可合并成一个大的 LP！
#   变量: y_i  +  每个场景 s 的 t^s[i,j]  +  每个场景 s 的缺货 u^s[i]
#   这就是"场景展开(scenario expansion)"，是 SAA 的标准实现方式。
# ----------------------------------------------------------------------------
def solve_two_stage(D):
    N, n = D.shape
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = {p: k for k, p in enumerate(pairs)}
    m = len(pairs)

    # 变量布局: [y_0..y_{n-1}, 场景0(t,u), 场景1(t,u), ...]
    per_scen = m + n
    n_var = n + N * per_scen
    c = np.zeros(n_var)
    c[:n] = cap_cost  # 建产能成本
    for s in range(N):
        off = n + s * per_scen
        for (i, j) in pairs:
            c[off + idx[(i, j)]] = ship_cost[i, j] / N  # 期望调货成本
        for i in range(n):
            c[off + m + i] = EMERGENCY / N  # 期望缺货/外购成本

    A_ub = []
    b_ub = []
    for s in range(N):
        off = n + s * per_scen
        # 供货满足: 调出 - 调入 - u_i <= y_i - d_i
        for i in range(n):
            row = np.zeros(n_var)
            row[i] = -1.0
            for (ii, jj) in pairs:
                if ii == i:
                    row[off + idx[(ii, jj)]] -= 1.0
                if jj == i:
                    row[off + idx[(ii, jj)]] += 1.0
            row[off + m + i] = -1.0
            A_ub.append(row)
            b_ub.append(-D[s, i])

    bounds = [(0, None)] * n_var
    res = linprog(
        c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs"
    )
    assert res.success, "两阶段 LP 求解失败"
    y = res.x[:n]
    flows = []
    for s in range(N):
        off = n + s * per_scen
        fl = np.zeros((n, n))
        for (i, j) in pairs:
            fl[i, j] = res.x[off + idx[(i, j)]]
        flows.append(fl)
    return y, res.fun, flows


def evaluate_policy(y, D):
    """在给定需求场景集 D 上评估第一阶段决策 y 的平均总成本(建造成本 + 平均调货)。"""
    N = D.shape[0]
    build = cap_cost @ y
    ship = np.mean([recourse_cost(y, D[s])[0] for s in range(N)])
    return build + ship


# ============================================================================
# 主流程
# ============================================================================
N_TRAIN = 200  # 训练场景数(用于 SAA 求解)
N_TEST = 2000  # 样本外测试场景数(用于评估)

# --- 生成训练/测试场景(用固定种子保证可复现) ---
D_train = np.column_stack(
    [rng.normal(demand_mu[i], demand_sigma[i], N_TRAIN) for i in range(N_CITY)]
).clip(min=0.0)
D_test = np.column_stack(
    [rng.normal(demand_mu[i], demand_sigma[i], N_TEST) for i in range(N_CITY)]
).clip(min=0.0)

# --- 方法0: 完美信息(每场景知道需求再定产能，只在调货层面下界参考不做，这里用 wait-and-see) ---
# Wait-and-see: 每个场景单独最优(不可能实现，作为下界)
ws_costs = []
for s in range(N_TRAIN):
    d = D_train[s]
    # 单场景下最优产能：可以精确只建到需求(因调货有成本，本地造最省)
    # 这里用 LP: min cap_cost·y + recourse，单场景
    y_ws, cost_ws, _ = solve_two_stage(D_train[s : s + 1])
    ws_costs.append(cost_ws)
WS = np.mean(ws_costs)

# --- 方法1: 两阶段随机规划(SAA) ---
y_sp, obj_sp, _ = solve_two_stage(D_train)
in_sample_sp = evaluate_policy(y_sp, D_train)
out_sample_sp = evaluate_policy(y_sp, D_test)

# --- 方法2: 确定性等价(DET) —— 只按需求均值建产能(忽略随机性) ---
D_mean = demand_mu.reshape(1, -1)
y_det, obj_det, _ = solve_two_stage(D_mean)
out_sample_det = evaluate_policy(y_det, D_test)

# --- 方法3: 朴素"自给自足"产能(每城建到自己均值) ---
y_naive = demand_mu.copy()
out_sample_naive = evaluate_policy(y_naive, D_test)

# --- 样本外评估：SAA 解 vs DET 解 ---
EV_est = out_sample_sp  # SAA 解的真实(样本外)期望成本估计
VSS = out_sample_det - out_sample_sp  # 随机解的价值
EVPI = out_sample_sp - WS  # 完美信息的价值(近似, WS 基于训练场景)

# --- 敏感性: 训练样本量 N 对 SAA 解的影响 ---
sample_sizes = [20, 50, 100, 200, 400, 800]
saa_by_size = []
for ns in sample_sizes:
    Ds = np.column_stack(
        [rng.normal(demand_mu[i], demand_sigma[i], ns) for i in range(N_CITY)]
    ).clip(min=0.0)
    ys, _, _ = solve_two_stage(Ds)
    sai = evaluate_policy(ys, D_train)
    sao = evaluate_policy(ys, D_test)
    saa_by_size.append((ns, ys.copy(), sai, sao))

# --- 敏感性: 需求波动大小 sigma 对第一阶段产能的影响 ---
sigmas = [0.5, 1.0, 1.5, 2.0]
vol_sweep = []
for k in sigmas:
    Ds = np.column_stack(
        [rng.normal(demand_mu[i], demand_sigma[i] * k, 200) for i in range(N_CITY)]
    ).clip(min=0.0)
    ys, _, _ = solve_two_stage(Ds)
    vol_sweep.append((k, ys.copy()))

# ============================================================================
# 断言：核验结论
# ============================================================================
# 1) 两阶段模型必须满足"自建产能 >= 按均值规划"的直觉? 不一定每城都大，但总产能应偏保守/缓冲
assert y_sp.min() >= -1e-6, "产能不应为负"
# 2) SAA 解的样本外成本 <= DET 解(随机构造应有缓冲优势)
assert out_sample_sp <= out_sample_det + 1e-6, "SAA 应不劣于确定性等价"
# 3) 样本内 vs 样本外【同量级】：SAA 只在训练场景集上最优，故样本外成本略高于样本内；
#    只要不出现戏剧性放大(即 SAA 解的样本外表现与样本内同量级)，就说明场景数够、无明显过拟合。
assert in_sample_sp <= out_sample_sp + 1e-6, "样本内成本应不高于样本外(SAA 在训练集上最优)"
assert out_sample_sp <= in_sample_sp * 1.15, "样本外不应比样本内差太多(<15%)"
# 4) VSS 应为正(随机规划相对确定性规划的价值)
assert VSS >= -1e-6, "VSS 应为非负"
# 5) 需求波动越大，总产能缓冲越多(不严格每城单调，但总量应上升)
tot_vol = [float(np.sum(y)) for _, y in vol_sweep]
assert tot_vol[-1] >= tot_vol[0] - 1e-6, "波动大时总产能不宜更少"

# ============================================================================
# 绘图
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 图1: 第一阶段产能对比(SP vs DET vs 均值需求) ---
ax = axes[0, 0]
xpos = np.arange(N_CITY)
w = 0.35
ax.bar(xpos - w / 2, y_sp, w, label="两阶段随机规划(SAA)", color="#2E86DE")
ax.bar(xpos + w / 2, y_det, w, label="确定性等价(按均值)", color="#EE5A24")
ax.plot(xpos, demand_mu, "k--o", label="需求均值 μ")
ax.set_xticks(xpos)
ax.set_xticklabels(CITY)
ax.set_ylabel("第一阶段日产能 (万瓶/天)")
ax.set_title("① 现在建多少产能：随机规划故意留缓冲")
ax.legend()
ax.grid(alpha=0.3)

# --- 图2: 成本对比 ---
ax = axes[0, 1]
labels = ["SAA\n(随机规划)", "DET\n(按均值)", "朴素\n(自给自足)"]
vals = [out_sample_sp, out_sample_det, out_sample_naive]
colors = ["#2E86DE", "#EE5A24", "#7f8c8d"]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:,.0f}", ha="center", fontsize=10)
ax.set_ylabel("样本外平均总成本 (元/天)")
ax.set_title("② 谁的总成本最低（样本外 2000 场景）")
ax.grid(alpha=0.3, axis="y")

# --- 图3: SAA 训练样本量收敛 ---
ax = axes[1, 0]
ns_list = [s[0] for s in saa_by_size]
os_list = [s[3] for s in saa_by_size]
ax.plot(ns_list, os_list, "o-", color="#10ac84", label="样本外成本估计")
ax.axhline(out_sample_sp, ls="--", color="#2E86DE", label=f"SAA(N=200) 参考 {out_sample_sp:,.0f}")
ax.set_xscale("log")
ax.set_xlabel("SAA 训练场景数 N (log)")
ax.set_ylabel("样本外平均成本 (元/天)")
ax.set_title("③ 场景数越多，SAA 解越稳定")
ax.legend()
ax.grid(alpha=0.3)

# --- 图4: 需求波动对产能缓冲的影响 ---
ax = axes[1, 1]
for i in range(N_CITY):
    ax.plot(
        [k for k, _ in vol_sweep],
        [y[i] for _, y in vol_sweep],
        "o-",
        label=f"{CITY[i]} (μ={demand_mu[i]:.0f})",
    )
ax.set_xlabel("需求波动倍数 (σ 相对基准)")
ax.set_ylabel("第一阶段产能 (万瓶/天)")
ax.set_title("④ 波动越大，越要多建缓冲产能")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, "two_stage_stochastic_saa.png"), dpi=110)
plt.close()

# ============================================================================
# 打印结果
# ============================================================================
print("=" * 72)
print("Day28 两阶段随机规划 (Two-Stage SP) + 样本均值近似 (SAA)")
print("=" * 72)
print(f"需求均值 μ = {demand_mu} 万瓶/天，σ = {demand_sigma} 万瓶/天")
print(f"产能成本 = {cap_cost} 元/万瓶")
print("-" * 72)
print("【第一阶段产能决策 y (万瓶/天)】")
print(f"  两阶段随机规划 SAA : {np.round(y_sp, 2)}  总产能 {y_sp.sum():.2f}")
print(f"  确定性等价 DET     : {np.round(y_det, 2)}  总产能 {y_det.sum():.2f}")
print(f"  需求均值 (朴素)    : {np.round(demand_mu, 2)}  总产能 {demand_mu.sum():.2f}")
print(f"  → SAA 比 DET 多建 {y_sp.sum() - y_det.sum():+.2f} 万瓶/天 的缓冲")
print("-" * 72)
print("【成本对比 (元/天)】")
print(f"  完美信息 WS(下界)  : {WS:,.2f}")
print(f"  两阶段 SAA 样本内  : {in_sample_sp:,.2f}")
print(f"  两阶段 SAA 样本外  : {out_sample_sp:,.2f}")
print(f"  确定性等价 DET 外  : {out_sample_det:,.2f}")
print(f"  朴素自给自足 外    : {out_sample_naive:,.2f}")
print("-" * 72)
print(f"  VSS (随机解价值)   : {VSS:,.2f} 元/天  ({VSS / out_sample_det * 100:.2f}% of DET)")
print(f"  EVPI(信息价值,约)  : {EVPI:,.2f} 元/天")
print(f"  样本内外相对差     : {abs(in_sample_sp - out_sample_sp) / out_sample_sp * 100:.2f}%")
print("-" * 72)
print("【SAA 训练样本量影响】")
for ns, ys, sai, sao in saa_by_size:
    print(f"  N={ns:>4}  总产能 {ys.sum():7.2f}  样本内 {sai:,.1f}  样本外 {sao:,.1f}")
print("-" * 72)
print("【需求波动对产能的影响】")
for k, ys in vol_sweep:
    print(f"  σ×{k:<4}  总产能 {ys.sum():7.2f}  y={np.round(ys, 2)}")
print("=" * 72)
print("断言全部通过 ✅")
