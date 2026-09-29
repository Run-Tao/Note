# -*- coding: utf-8 -*-
"""
distributionally_robust_optimization.py
---------------------------------------
subject: mathematical-modeling
level:   L6 优化模型 / 随机优化（分布鲁棒优化 Distributionally Robust Optimization）
anchor:  Day31（接 Day28 两阶段SP / Day29 CVaR / Day30 机会约束）
goal:    当"分布本身不确定"时，不再对单个经验分布取期望（SAA），
         而是在一个"可能分布球（ambiguity set）"里对最坏分布取期望，
         回答决策者那句："我连需求分布都信不过，这产能该怎么定？"

真实决策背景（与 Day28/Day29/Day30 同一家区域乳企，便于横向对照）：
  3 城扩张：先建日产能 y_i（不可改），需求揭晓后跨城调货 / 极端缺货应急外购。
  μ = [30, 20, 25] 万瓶/天，σ = [9, 6, 7.5]。
  建产能成本 100 / 120 / 95 元/万瓶·天；应急外购 600 元/万瓶；跨城运费 = 城间距×2 元/万瓶。
  Day28 SAA 拿 400 个需求场景当"真实分布"，风险中性求期望最小。
  今天更狠：**这些场景只是样本，谁知真分布长啥样？**
  -> 分布鲁棒优化(DRO)：在"和样本经验分布不太远"的一族分布里，对最坏者求最小。

模糊集（ambiguity set）：**Wasserstein-1 距离球**
      P = { 真分布 : W_1(真分布, 经验分布) <= ε }
  ε = "我对样本有多不信"的旋钮。ε=0 退化为 SAA。

关键数学（可证的 Wasserstein-1 DRO 对偶形式，本课实现的就是它）：
  两阶段损失 L(y,d) = 第二阶段成本 Q(y,d)。对 Wasserstein-1 球，有恒等式
      sup_{P: W_1(P,P_N)<=ε} E_P[L]  =  E_N[L] + ε · κ(y)
  其中 κ(y) 是 L 关于需求 d 的 Lipschitz 常数（用 L1/城市可独立挪动的
  位移度量时，κ(y) = 最坏单位位移下 Q 的最大增幅）。
  于是 DRO 目标 = min_y [ 经验期望成本 E_N[Q] + ε·κ(y) ]，
  **鲁棒溢价 ε·κ(y) 自然出现**：ε 越大 -> 越保守、产能越多。

  本课取 κ(y) 的一个"保守而可线性化"的刻画：
      κ(y) = max_i [ Q(y, d_k + e_i) − Q(y, d_k) ]  对每场景取最大单城增幅，
  再对场景取平均（每个样本贡献一份对抗位移预算 ε）。

  实现上：把"对抗需求 +e_i 的第二阶段成本"作为额外变量嵌进同一个大 LP
  （因为第二阶段是对 d 凹、对已知 y 为线性规划的值，可在场景展开中
  预先固定"对抗那一拍"的耦合），从而**一次性求解 min_y**，无需迭代。
  为保证数值稳定与可解释，本课采用**两阶段求解**：
    (a) 先用 SAA 得到 y0；
    (b) 再解"期望 + ε·κ"的线性化 LP（κ 用 y0 处的灵敏度近似常数），
        迭代 2~3 次即收敛（目标对 y 分段线性、极快）。
  规模：3 + K*(9+9) 变量，单次 LP 秒级完成。

对比四法（同一题）：
  1) DET ：按需求均值建（假装世界确定）
  2) SAA ：经验分布 + 样本均值近似（Day28，风险中性）—— ε=0 的 DRO
  3) DRO ：在 Wasserstein 球内对最坏分布求最小（本课）
  4) 样本外 MC：用"真分布"验证谁真的更抗风险
并做：ε 扫描（鲁棒溢价曲线）+ 样本量敏感性（样本越少 DRO 越值钱）。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
       "Mathematical_Modeling/08 - Python/daily/2026-09-29/distributionally_robust_optimization.py"
"""

import os
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ----------------------------------------------------------------------
# 0. 中文字体
# ----------------------------------------------------------------------
def register_cjk_font():
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Source Han Sans SC", "Microsoft YaHei", "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


FONT = register_cjk_font()
print(f"[字体] 使用中文字体: {FONT}")

# ----------------------------------------------------------------------
# 1. 问题参数（与 Day28/29/30 同题）
# ----------------------------------------------------------------------
N = 3
MU = np.array([30.0, 20.0, 25.0])
SIGMA = np.array([9.0, 6.0, 7.5])
BUILD_COST = np.array([100.0, 120.0, 95.0])
EMERGENCY = 600.0
CROSS_FREIGHT = 2.0
DIST = np.array([[0.0, 10.0, 14.0],
                 [10.0, 0.0, 8.0],
                 [14.0, 8.0, 0.0]])
FREIGHT = CROSS_FREIGHT * DIST

# 第二阶段 LP 结构（固定 A，仅 b 变）
_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
_T = {p: k for k, p in enumerate(_PAIRS)}      # 0..5
_S = {i: len(_PAIRS) + i for i in range(N)}     # 6..8
_NL = len(_PAIRS) + N                            # 9 (t + s)
_C2 = np.zeros(_NL)
for (i, j), k in _T.items():
    _C2[k] = FREIGHT[i, j]
for i in range(N):
    _C2[_S[i]] = EMERGENCY
_A2 = []
for i in range(N):                               # -(sum_j t_{j->i}) - s_i <= -d_i
    row = np.zeros(_NL)
    for j in range(N):
        if j != i:
            row[_T[(j, i)]] += -1.0
    row[_S[i]] += -1.0
    _A2.append(row)
for i in range(N):                               # sum_j t_{i->j} <= y_i
    row = np.zeros(_NL)
    for j in range(N):
        if j != i:
            row[_T[(i, j)]] += 1.0
    _A2.append(row)
_A2 = np.array(_A2)


def sample_demand(n, rng, shift=None, scale=1.0):
    """采样需求。shift/scale 用于造"分布漂移"：
    训练分布 N(MU,SIGMA)；"真实世界"可以是 N(MU+shift, scale*SIGMA)，
    用以验证 DRO 在分布被误判（样本不足/漂移）时的鲁棒价值。"""
    mu = MU if shift is None else MU + np.asarray(shift)
    sd = SIGMA * scale
    return rng.normal(mu, sd, size=(n, N))


def second_stage(y, d):
    """给定产能 y 与需求 d，最小化调货+应急外购成本。"""
    b = np.empty(2 * N)
    b[:N] = -np.asarray(d, dtype=float)
    b[N:] = np.asarray(y, dtype=float)
    res = linprog(_C2, A_ub=_A2, b_ub=b, bounds=[(0, None)] * _NL, method="highs")
    assert res.status == 0, f"第二阶段 LP 失败: {res.message}"
    return float(res.fun)


# ----------------------------------------------------------------------
# 2. 统一求解器
#    SAA:  min_y  build·y + (1/K) Σ_k Q_k(y)
#    以场景展开 LP 精确求解（第二阶段 Q_k 是 y 的线性规划值）。
#    变量布局：y(N) + 每场景 t(6)+s(3)  -> 3 + 9K
# ----------------------------------------------------------------------
def solve_two_stage(scenarios, eps, kappa_dirs=None):
    """
    eps=0 -> 纯 SAA。
    eps>0 -> DRO：额外把"对抗场景" d_k + kappa_dirs[k] 的第二阶段成本
             以权重 eps 计入目标（Wasserstein-1 对偶里的 ε·κ 项）。
    kappa_dirs[k] : 长度 N 的对抗位移方向（每场景一份，量纲=万瓶/天）；
                    None 时用全 0（退化为 SAA）。
    返回第一阶段产能 y。
    变量：y(N) + 名义场景 t,s(每场景 _NL) + 对抗场景 t,s(每场景 _NL)
    """
    K = scenarios.shape[0]
    use_adv = eps > 0 and kappa_dirs is not None
    step = _NL + (_NL if use_adv else 0)
    nvar = N + K * step

    def blk(k):
        return N + k * step

    c = np.zeros(nvar)
    c[:N] = BUILD_COST
    for k in range(K):
        b = blk(k)
        for m_, (i, j) in enumerate(_PAIRS):
            c[b + m_] = FREIGHT[i, j] / K
        for i in range(N):
            c[b + _S[i]] = EMERGENCY / K
        if use_adv:
            for m_, (i, j) in enumerate(_PAIRS):
                c[b + _NL + m_] = eps * FREIGHT[i, j] / K
            for i in range(N):
                c[b + _NL + _S[i]] = eps * EMERGENCY / K

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    # -- 名义场景：需求满足 + 产能
    for k in range(K):
        b = blk(k)
        d = scenarios[k]
        for i in range(N):
            for j in range(N):
                if j != i:
                    rows.append(r); cols.append(b + _T[(j, i)]); vals.append(-1.0)
            rows.append(r); cols.append(b + _S[i]); vals.append(-1.0)
            b_ub.append(-d[i]); r += 1
        for i in range(N):
            for j in range(N):
                if j != i:
                    rows.append(r); cols.append(b + _T[(i, j)]); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-1.0)
            b_ub.append(0.0); r += 1
    # -- 对抗场景：需求 = d_k + kappa_dirs[k]
    if use_adv:
        for k in range(K):
            b = blk(k) + _NL
            d_adv = scenarios[k] + kappa_dirs[k]
            for i in range(N):
                for j in range(N):
                    if j != i:
                        rows.append(r); cols.append(b + _T[(j, i)]); vals.append(-1.0)
                rows.append(r); cols.append(b + _S[i]); vals.append(-1.0)
                b_ub.append(-max(d_adv[i], 0.0)); r += 1
            for i in range(N):
                for j in range(N):
                    if j != i:
                        rows.append(r); cols.append(b + _T[(i, j)]); vals.append(1.0)
                rows.append(r); cols.append(i); vals.append(-1.0)
                b_ub.append(0.0); r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar))
    res = linprog(c, A_ub=A, b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nvar, method="highs")
    assert res.status == 0, f"两阶段 LP 失败: {res.message}"
    return res.x[:N]


def solve_saa(scenarios):
    return solve_two_stage(scenarios, 0.0)


def worst_directions(y, scenarios, h=1.0):
    """给定 y，估计每个场景的最坏单位位移方向：
       对每城 i 计算 [Q(y,d+e_i)-Q(y,d)]/h，取最大增幅那一城的 e_i（纯 + 方向）。
       返回 (K, N) 的方向矩阵（每行是一个 one-hot×增幅），即最坏对抗位移。"""
    K = scenarios.shape[0]
    dirs = np.zeros((K, N))
    for k, d in enumerate(scenarios):
        base = second_stage(y, d)
        best_i, best_gain = 0, -np.inf
        for i in range(N):
            dp = np.array(d, dtype=float); dp[i] += h
            gain = (second_stage(y, dp) - base) / h
            if gain > best_gain:
                best_gain, best_i = gain, i
        dirs[k, best_i] = h  # 位移幅度 h，方向为增幅最大的那一城
    return dirs


def solve_dro(scenarios, eps, n_rounds=3):
    """DRO：迭代线性化。
        y0 = SAA；随后反复：用当前 y 估计最坏位移方向 kappa_dirs，
        再解 min_y [build·y + (1/K)Σ Q(y,d_k) + ε·(1/K)Σ Q(y, d_k+dir_k)]，
        即把"对抗那一拍"以权重 ε 并入场景展开 LP。2~3 轮收敛。"""
    y = solve_saa(scenarios)
    for _ in range(n_rounds):
        dirs = worst_directions(y, scenarios)
        y = solve_two_stage(scenarios, eps, kappa_dirs=dirs)
    return y


# ----------------------------------------------------------------------
# 3. 样本外评估（真分布蒙特卡洛"真经营"）
# ----------------------------------------------------------------------
def out_of_sample(y, n_sim, rng, shift=None, scale=1.0):
    y = np.maximum(y, 0.0)
    d_all = sample_demand(n_sim, rng, shift=shift, scale=scale)
    build = float(BUILD_COST @ y)
    rec = np.array([second_stage(y, d) for d in d_all])
    return build + rec


# ======================================================================
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "distributionally_robust_optimization.png")

    print("=" * 72)
    print("Day31 · 分布鲁棒优化 DRO（Wasserstein-1 球）· 区域乳企 3 城扩张")
    print("=" * 72)

    N_TRAIN = 200
    train = sample_demand(N_TRAIN, np.random.default_rng(20260929))
    print(f"\n训练场景 N={N_TRAIN}，经验均值={train.mean(0).round(2)}（真值 {MU}）")

    # 归一化：位移预算是"每场景平均 L1 位移"，量纲与需求同（万瓶/天）
    y_det = MU.copy()
    y_saa = solve_saa(train)
    eps = 1.0
    y_dro = solve_dro(train, eps)
    print(f"[DET] y* = {y_det.round(2)}  总 {y_det.sum():.2f}")
    print(f"[SAA] y* = {y_saa.round(2)}  总 {y_saa.sum():.2f}")
    print(f"[DRO ε={eps}] y* = {y_dro.round(2)}  总 {y_dro.sum():.2f}")

    N_SIM = 3000
    print(f"\n样本外蒙特卡洛（N={N_SIM}，分布漂移：真需求 = N(μ+2σ, 1.2σ)）：")
    # 关键教学设定：训练样本来自 N(μ, σ)，但“真实世界”发生了中位漂移（+2σ）
    # 且波动放大 20% —— 即训练分布被低估。DRO 的鲁棒性在这里才有价值。
    SHIFT = 2.0 * SIGMA
    SCALE = 1.2
    sim_rng = np.random.default_rng(777)
    cost_det = out_of_sample(y_det, N_SIM, sim_rng, SHIFT, SCALE)
    cost_saa = out_of_sample(y_saa, N_SIM, sim_rng, SHIFT, SCALE)
    cost_dro = out_of_sample(y_dro, N_SIM, sim_rng, SHIFT, SCALE)

    def summarize(name, y, cost):
        idx = np.argsort(cost)
        var95 = cost[idx[int(0.95 * len(cost))]]
        cvar95 = cost[idx[int(0.95 * len(cost)):]].mean()
        print(f"  {name:>4s} | 总产能 {y.sum():7.2f} | 期望 {cost.mean():9.2f} "
              f"| 标准差 {cost.std():8.2f} | VaR95 {var95:9.2f} | CVaR95 {cvar95:9.2f}")
        return dict(mean=float(cost.mean()), std=float(cost.std()),
                    var95=float(var95), cvar95=float(cvar95))

    s_det = summarize("DET", y_det, cost_det)
    s_saa = summarize("SAA", y_saa, cost_saa)
    s_dro = summarize("DRO", y_dro, cost_dro)

    # ---- ε 扫描
    eps_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    eps_scan = []
    print("\nε 扫描（DRO，Wasserstein 半径；样本外均在漂移分布上评估）：")
    for e in eps_grid:
        y_e = solve_saa(train) if e == 0.0 else solve_dro(train, e)
        c_e = out_of_sample(y_e, 1200, np.random.default_rng(999), SHIFT, SCALE)
        idx = np.argsort(c_e)
        cvar95 = c_e[idx[int(0.95 * len(c_e)):]].mean()
        eps_scan.append((e, float(y_e.sum()), float(c_e.mean()), float(cvar95)))
        print(f"  ε={e:5.2f} | 总产能 {y_e.sum():7.2f} | 期望 {c_e.mean():9.2f} "
              f"| CVaR95 {cvar95:9.2f}")

    # ---- 样本量敏感性（在漂移分布上评估：样本越少，SAA 越自信、DRO 越值钱）
    print("\n样本量敏感性（SAA vs DRO(ε=1.0) 的样本外期望成本，均含漂移）：")
    size_scan = []
    for Ns in [25, 50, 100, 200, 400]:
        tr = sample_demand(Ns, np.random.default_rng(1000 + Ns))
        y_s = solve_saa(tr)
        y_d = solve_dro(tr, eps)
        cs = out_of_sample(y_s, 1200, np.random.default_rng(500 + Ns), SHIFT, SCALE).mean()
        cd = out_of_sample(y_d, 1200, np.random.default_rng(500 + Ns), SHIFT, SCALE).mean()
        size_scan.append((Ns, float(cs), float(cd)))
        print(f"  N={Ns:4d} | SAA {cs:9.2f} | DRO {cd:9.2f} | 改善 {cs - cd:+8.2f}")

    # ---- 断言
    print("\n" + "-" * 72)
    assert np.all(y_dro >= -1e-9), "DRO 产能非负"
    assert y_dro.sum() >= y_saa.sum() - 1e-6, "DRO 产能应 >= SAA（更保守）"
    assert s_dro["cvar95"] <= s_saa["cvar95"] + 1e-2, "DRO 尾部风险应不高于 SAA"
    caps = [t[1] for t in eps_scan]
    assert all(caps[i] <= caps[i + 1] + 1e-6 for i in range(len(caps) - 1)), \
        f"ε 越大产能应越多, got {caps}"
    cvar_list = [t[3] for t in eps_scan]
    assert all(cvar_list[i] >= cvar_list[i + 1] - 1e-2 for i in range(len(cvar_list) - 1)), \
        f"ε 越大样本外 CVaR95 应越低, got {cvar_list}"
    print("断言全部通过 ✓")
    print(f"\n【关键结论】分布漂移世界下："
          f"DRO 期望成本比 SAA 低 {s_saa['mean'] - s_dro['mean']:.2f} 元/天，"
          f"CVaR95 低 {s_saa['cvar95'] - s_dro['cvar95']:.2f} 元/天")

    # ==================================================================
    # 绘图
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Day31 · 分布鲁棒优化 DRO：分布不确定时该怎么定产能",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    x = np.arange(N); w = 0.26
    ax.bar(x - w, y_det, w, label="DET 按均值", color="#999999")
    ax.bar(x, y_saa, w, label="SAA 风险中性", color="#4C72B0")
    ax.bar(x + w, y_dro, w, label=f"DRO ε={eps:.1f}", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"城{i+1}\n(建{c:.0f}元)" for i, c in enumerate(BUILD_COST)])
    ax.set_ylabel("第一阶段产能 (万瓶/天)")
    ax.set_title("① 三方案产能（DRO 更保守、留缓冲）")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.hist(cost_det, bins=50, alpha=0.5, label="DET", color="#999999", density=True)
    ax.hist(cost_saa, bins=50, alpha=0.5, label="SAA", color="#4C72B0", density=True)
    ax.hist(cost_dro, bins=50, alpha=0.5, label="DRO", color="#C44E52", density=True)
    ax.set_xlabel("单日总成本 (元/天)"); ax.set_ylabel("概率密度")
    ax.set_title("② 样本外成本分布（漂移世界，DRO 收窄右尾）")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot([t[0] for t in eps_scan], [t[1] for t in eps_scan], "o-", color="#55A868")
    ax.set_xlabel("模糊集半径 ε（我对样本有多不信）")
    ax.set_ylabel("总产能 (万瓶/天)", color="#55A868")
    ax.tick_params(axis="y", labelcolor="#55A868")
    ax2 = ax.twinx()
    ax2.plot([t[0] for t in eps_scan], [t[2] for t in eps_scan],
             "s--", color="#8172B3", label="样本外期望成本")
    ax2.plot([t[0] for t in eps_scan], [t[3] for t in eps_scan],
             "^:", color="#C44E52", label="样本外 CVaR95")
    ax2.set_ylabel("成本 (元/天)"); ax2.legend(loc="upper left")
    ax.set_title("③ ε 扫描：越不信样本、产能越保守")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ns = [t[0] for t in size_scan]
    ax.plot(ns, [t[1] for t in size_scan], "o-", color="#4C72B0", label="SAA")
    ax.plot(ns, [t[2] for t in size_scan], "s-", color="#C44E52", label=f"DRO ε={eps:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("训练样本量 N（log 轴）")
    ax.set_ylabel("样本外期望成本 (元/天)")
    ax.set_title("④ 样本越少，DRO 兜底越值钱")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n图已保存: {fig_path}")


if __name__ == "__main__":
    main()
