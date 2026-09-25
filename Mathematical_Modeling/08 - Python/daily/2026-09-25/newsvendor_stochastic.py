# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 优化模型 / 随机优化（报童模型 Newsvendor）
anchor: Day27 - 报童模型与报童临界分位点（接 Day26 LP 对偶影子价格、Day18 MDP 随机优化）
goal: 用一道"每天订多少份报纸/备多少货"的真实决策，讲清随机需求下的单周期最优订货量
      —— 关键不是预测需求均值，而是算出一个【临界分位点】，并用四种独立方法交叉验证：
      (1) 解析公式 F(Q*) = Cu/(Cu+Co)
      (2) 网格穷举期望利润最大值
      (3) 蒙特卡洛仿真验证（大样本收敛到解析值）
      (4) 边际分析（多订一份的期望收益 = 0 的临界点）
      再回答决策者真正关心的追问：需求预测错了会亏多少？服务水平该定多高？

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-25/newsvendor_stochastic.py"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

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

rng = np.random.default_rng(20260925)

# ----------------------------------------------------------------------------
# 真实决策背景：一家社区面包店，每天清晨要决定烤多少只"当日限定可颂"
#   - 每只售价 p 元；未卖完的当晚只能当隔夜货折价清仓（残值 s 元/只）
#   - 每只成本 c 元；缺货（客人来了没货）损失的边际利润即缺货成本 Cu
#   - 当日需求 D 随机，历史经验服从正态 N(mu, sigma^2)
# 决策：每天烤多少只 Q，使【期望利润】最大
# ----------------------------------------------------------------------------
P_PRICE = 8.0      # 售价（元/只）
C_COST = 3.0       # 成本（元/只）
S_SALVAGE = 1.5    # 隔夜清仓残值（元/只）
MU = 200.0         # 需求均值（只/天）
SIGMA = 40.0       # 需求标准差（只/天）

CU = P_PRICE - C_COST              # 缺货成本（少赚的边际利润）= 5.0
CO = C_COST - S_SALVAGE            # 过剩成本（多烤一只的净损失）= 1.5
CRITICAL_RATIO = CU / (CU + CO)    # 临界分位点 = 0.7692...


# ----------------------------------------------------------------------------
# 方法 1：解析解 —— 正态需求下 Q* = mu + sigma * z_{CR}
# ----------------------------------------------------------------------------
def analytical_optimal():
    z = norm.ppf(CRITICAL_RATIO)
    q_star = MU + SIGMA * z
    # 解析期望利润（正态版：E[profit] = p*mu - c*Q - (p-s)*E[过剩] + 缺货... 用报童标准公式）
    # 用标准报童利润函数便于与数值法统一：profit(Q,D) = p*min(Q,D) + s*max(Q-D,0) - c*Q
    exp_profit = expected_profit_analytic(q_star)
    return q_star, z, exp_profit


def expected_profit_analytic(q):
    """正态需求下报童模型期望利润的闭式表达。"""
    z = (q - MU) / SIGMA
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    # 期望销量 = mu*Phi(z) ... 用标准报童公式：
    # E[sales] = mu - sigma*L(z)，其中 L(z)=phi(z) - z*(1-Phi(z)) 是标准正态损失函数
    loss = phi - z * (1.0 - Phi)
    exp_sales = MU - SIGMA * loss
    # 期望过剩 = q - 期望销量
    exp_excess = q - exp_sales
    return P_PRICE * exp_sales + S_SALVAGE * exp_excess - C_COST * q


# ----------------------------------------------------------------------------
# 方法 2：网格穷举 —— 直接对 Q 网格算期望利润取最大
# ----------------------------------------------------------------------------
def expected_profit_grid(q, D_samples):
    """给定大量需求样本，用样本平均逼近期望利润。"""
    sales = np.minimum(q, D_samples)
    excess = np.maximum(q - D_samples, 0.0)
    return np.mean(P_PRICE * sales + S_SALVAGE * excess - C_COST * q)


def grid_search(q_lo=100, q_hi=320, step=0.5, n_samples=400000):
    D_samples = rng.normal(MU, SIGMA, size=n_samples)
    qs = np.arange(q_lo, q_hi + step, step)
    # 向量化：对每个 q 批量算（内存友好，分块遍历 q）
    profits = np.array([expected_profit_grid(q, D_samples) for q in qs])
    idx = int(np.argmax(profits))
    return qs[idx], profits[idx], qs, profits, D_samples


# ----------------------------------------------------------------------------
# 方法 3：边际分析 —— "再多烤一只"的期望净收益 = 0 的临界点
#   多烤第 Q 只：以 P(D >= Q) 概率被正常卖出（多赚 Cu）
#                以 P(D <  Q) 概率沦为过剩（倒亏 Co）
#   期望边际收益 = Cu*P(D>=Q) - Co*P(D<Q) = 0  =>  P(D<Q) = Cu/(Cu+Co)
# ----------------------------------------------------------------------------
def marginal_analysis():
    # P(D < Q) = CR  =>  Q = mu + sigma*z_{CR}（与解析解一致，此处用数值二分验证）
    lo, hi = MU - 5 * SIGMA, MU + 5 * SIGMA
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm.cdf((mid - MU) / SIGMA) < CRITICAL_RATIO:
            lo = mid
        else:
            hi = mid
    q_star = 0.5 * (lo + hi)
    return q_star


# ----------------------------------------------------------------------------
# 方法 4：蒙特卡洛仿真 —— 用 Q* 实际"经营"多天，看平均利润是否逼近解析值
# ----------------------------------------------------------------------------
def simulate_policy(q, n_days=200000, seed=None):
    r = np.random.default_rng(seed if seed is not None else 12345)
    D = r.normal(MU, SIGMA, size=n_days)
    D = np.maximum(D, 0.0)
    sales = np.minimum(q, D)
    excess = np.maximum(q - D, 0.0)
    profit = P_PRICE * sales + S_SALVAGE * excess - C_COST * q
    return profit.mean(), profit.std(), (sales < q).mean()  # 平均利润/波动/缺货率


# ----------------------------------------------------------------------------
# 方法 5：需求预测错了会亏多少？—— 均值/标准差估计偏差的敏感性
# ----------------------------------------------------------------------------
def sensitivity_to_forecast_error():
    """假设我们把 mu/sigma 估偏了，但没发现，仍按错的参数订货，真实世界却不偏。
    计算"利润损失"——回答决策者："需求预测不准，代价有多大？" """
    true_mu, true_sigma = MU, SIGMA
    q_true = analytical_optimal()[0]
    rows = []
    # 均值估偏
    for bias in [-40, -20, -10, 0, 10, 20, 40]:
        mu_hat = true_mu + bias
        z = norm.ppf(CRITICAL_RATIO)
        q_hat = mu_hat + true_sigma * z
        # 真实期望利润（用真实分布算）
        loss = norm.pdf((q_hat - true_mu) / true_sigma) - (
            (q_hat - true_mu) / true_sigma
        ) * (1 - norm.cdf((q_hat - true_mu) / true_sigma))
        exp_sales = true_mu - true_sigma * loss
        exp_excess = q_hat - exp_sales
        profit = P_PRICE * exp_sales + S_SALVAGE * exp_excess - C_COST * q_hat
        rows.append((bias, q_hat, profit))
    return q_true, rows


def sensitivity_to_sigma():
    """需求波动 sigma 估偏的代价。"""
    true_mu, true_sigma = MU, SIGMA
    rows = []
    z = norm.ppf(CRITICAL_RATIO)
    for sig_hat in [20, 30, 40, 50, 60, 80]:
        q_hat = true_mu + sig_hat * z
        loss = norm.pdf((q_hat - true_mu) / true_sigma) - (
            (q_hat - true_mu) / true_sigma
        ) * (1 - norm.cdf((q_hat - true_mu) / true_sigma))
        exp_sales = true_mu - true_sigma * loss
        exp_excess = q_hat - exp_sales
        profit = P_PRICE * exp_sales + S_SALVAGE * exp_excess - C_COST * q_hat
        rows.append((sig_hat, q_hat, profit))
    return rows


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("报童模型（Newsvendor）—— 每天该烤多少只可颂？")
    print("=" * 78)
    print(f"售价 p={P_PRICE}  成本 c={C_COST}  残值 s={S_SALVAGE}")
    print(f"缺货成本 Cu=p-c={CU}   过剩成本 Co=c-s={CO}")
    print(f"临界分位点 Cu/(Cu+Co) = {CRITICAL_RATIO:.4f}")
    print(f"需求 ~ N(mu={MU}, sigma={SIGMA})")
    print()

    # --- 方法1 解析 ---
    q_ana, z_ana, profit_ana = analytical_optimal()
    print("[方法1 解析解]  Q* = mu + sigma*z = "
          f"{MU} + {SIGMA}*{z_ana:.4f} = {q_ana:.3f} 只")
    print(f"              解析期望利润 = {profit_ana:.3f} 元/天")

    # --- 方法2 网格穷举 ---
    q_grid, profit_grid, qs, profits, D_samples = grid_search()
    print(f"[方法2 网格穷举] Q* = {q_grid:.2f} 只（步长 0.5），"
          f"样本期望利润 = {profit_grid:.3f} 元/天")

    # --- 方法3 边际分析 ---
    q_marg = marginal_analysis()
    print(f"[方法3 边际分析] 多订一只期望收益=0 的临界点 Q* = {q_marg:.3f} 只")

    # --- 方法4 蒙特卡洛 ---
    np.random.seed(7)
    sim_profit, sim_std, stockout_rate = simulate_policy(q_ana, n_days=300000, seed=99)
    print(f"[方法4 蒙特卡洛] 按 Q*={q_ana:.2f} 经营 30 万天："
          f"平均利润 {sim_profit:.3f} ± {sim_std:.2f} 元，缺货率 {stockout_rate:.2%}")

    # --- 朴素对照：按均值订货 ---
    prof_mean, _, sr_mean = simulate_policy(MU, n_days=300000, seed=99)
    print(f"[朴素对照] 若只按需求均值订 Q=mu={MU:.0f}：平均利润 {prof_mean:.3f} 元，"
          f"缺货率 {sr_mean:.2%}")
    print(f"           → 拍脑袋按均值订货，每天少赚 {sim_profit - prof_mean:.3f} 元 "
          f"（{(sim_profit - prof_mean) / prof_mean:.2%}）")

    # --- 敏感性 ---
    q_true, rows_mu = sensitivity_to_forecast_error()
    print()
    print("[敏感性A] 需求均值估偏的代价（仍按错参数订货，真实分布不变）：")
    for bias, q_hat, profit in rows_mu:
        tag = "  <- 正确" if bias == 0 else ""
        print(f"  mu 估偏 {bias:+3d}  → 订 {q_hat:6.2f} 只，真实期望利润 {profit:8.3f} 元{tag}")

    rows_sig = sensitivity_to_sigma()
    print("[敏感性B] 需求波动 sigma 估偏的代价：")
    for sig_hat, q_hat, profit in rows_sig:
        tag = "  <- 正确" if abs(sig_hat - SIGMA) < 1e-9 else ""
        print(f"  sigma 估为 {sig_hat:4.0f}  → 订 {q_hat:6.2f} 只，真实期望利润 {profit:8.3f} 元{tag}")

    # 服务水平
    service_level = norm.cdf((q_ana - MU) / SIGMA)
    print()
    print(f"[服务水平] Q* 对应不缺货概率 P(D<=Q*) = {service_level:.4f}"
          f"（= 临界分位点，这正是它名字的由来）")

    # --- 断言 ---
    assert abs(q_ana - q_grid) < 1.5, "解析解与网格穷举应一致"
    assert abs(q_ana - q_marg) < 0.5, "边际分析临界点应等于解析解"
    assert abs(sim_profit - profit_ana) < 3.0, "蒙特卡洛应逼近解析期望利润"
    assert sim_profit > prof_mean, "报童最优应优于按均值订货"
    assert abs(service_level - CRITICAL_RATIO) < 1e-9, "服务水平应等于临界分位点"
    assert q_ana > MU, "临界分位点>0.5 时应订得比均值多"
    print("\n>>> 全部断言通过 ✔")

    # ------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    # 左上：期望利润随 Q 曲线 + 三种方法定位的 Q*
    ax = axes[0, 0]
    ax.plot(qs, profits, color="#2c7fb8", lw=2, label="期望利润（样本平均）")
    ax.axvline(q_ana, color="crimson", ls="--", lw=1.8,
               label=f"解析 Q* = {q_ana:.1f}")
    ax.axvline(q_grid, color="#31a354", ls=":", lw=2,
               label=f"网格穷举 Q* = {q_grid:.1f}")
    ax.axvline(MU, color="gray", ls="-.", lw=1.5,
               label=f"需求均值 μ = {MU:.0f}")
    ax.scatter([q_ana], [profit_ana], color="crimson", zorder=5, s=60)
    ax.set_xlabel("订货量 Q（只）")
    ax.set_ylabel("期望利润（元/天）")
    ax.set_title("期望利润随订货量的变化：最优点在均值右侧")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # 右上：需求分布 + 临界分位点（服务水平）
    ax = axes[0, 1]
    xs = np.linspace(MU - 4 * SIGMA, MU + 4 * SIGMA, 500)
    pdf = norm.pdf(xs, MU, SIGMA)
    ax.plot(xs, pdf, color="#333", lw=2)
    ax.fill_between(xs, pdf, where=(xs <= q_ana), color="#2c7fb8", alpha=0.35,
                    label=f"不缺货概率 = {service_level:.3f}")
    ax.fill_between(xs, pdf, where=(xs > q_ana), color="crimson", alpha=0.30,
                    label=f"缺货概率 = {1 - service_level:.3f}")
    ax.axvline(q_ana, color="crimson", ls="--", lw=2)
    ax.annotate(f"Q* = {q_ana:.1f}\n（临界分位点）",
                xy=(q_ana, norm.pdf(q_ana, MU, SIGMA)),
                xytext=(q_ana + 20, max(pdf) * 0.75),
                arrowprops=dict(arrowstyle="->", color="crimson"),
                fontsize=9, color="crimson")
    ax.set_xlabel("当日需求 D（只）")
    ax.set_ylabel("概率密度")
    ax.set_title(f"临界分位点 = Cu/(Cu+Co) = {CRITICAL_RATIO:.3f}")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # 左下：两种订货策略的利润分布（蒙特卡洛）
    ax = axes[1, 0]
    r = np.random.default_rng(2026)
    D = np.maximum(r.normal(MU, SIGMA, size=60000), 0.0)
    prof_star = P_PRICE * np.minimum(q_ana, D) + S_SALVAGE * np.maximum(q_ana - D, 0) - C_COST * q_ana
    prof_m = P_PRICE * np.minimum(MU, D) + S_SALVAGE * np.maximum(MU - D, 0) - C_COST * MU
    ax.hist(prof_m, bins=60, alpha=0.55, color="gray", label=f"按均值订 Q=μ（均值{prof_m.mean():.1f}）")
    ax.hist(prof_star, bins=60, alpha=0.55, color="#2c7fb8", label=f"报童最优 Q*（均值{prof_star.mean():.1f}）")
    ax.axvline(prof_m.mean(), color="gray", ls="--", lw=1.5)
    ax.axvline(prof_star.mean(), color="crimson", ls="--", lw=1.5)
    ax.set_xlabel("单日利润（元）")
    ax.set_ylabel("天数")
    ax.set_title("两种策略的利润分布：报童整体右移")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # 右下：敏感性——均值估偏的利润惩罚
    ax = axes[1, 1]
    biases = [r0[0] for r0 in rows_mu]
    penalties = [r0[2] - rows_mu[biases.index(0)][2] for r0 in rows_mu]
    colors = ["crimson" if b != 0 else "#31a354" for b in biases]
    ax.bar([str(b) for b in biases], penalties, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("需求均值估计偏差（只）")
    ax.set_ylabel("相对最优的利润损失（元/天）")
    ax.set_title("预测偏了会亏多少：低估需求比高估更贵")
    for i, v in enumerate(penalties):
        ax.text(i, v, f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("报童模型：随机需求下的单周期最优订货量", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "newsvendor_stochastic.png")
    fig.savefig(out_png, dpi=130)
    print(f"[图] 已保存到 {out_png}")
    if CJK:
        print(f"[字体] 使用中文字体: {CJK}")
    else:
        print("[字体] 警告：未找到中文字体，图中中文可能乱码")


if __name__ == "__main__":
    main()
