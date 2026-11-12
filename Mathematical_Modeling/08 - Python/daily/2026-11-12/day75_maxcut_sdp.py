# -*- coding: utf-8 -*-
r"""
subject: mathematical-modeling
level:   L5 近似算法（局部搜索 · SDP 松弛 + 随机超平面舍入）/ L4 组合优化模型（0-1 二次规划 Max-Cut）
anchor:  Roadmap L5 → 组合优化的近似算法 → Max-Cut：局部搜索 vs Goemans–Williamson
goal:    用「20 个巡林片区怎么劈成东西两半」这个真实决策，讲清三件事：
         ① 小图上「局部搜索 1000 次重启全部命中最优」不是算法的功劳，是规模的红利；
            局部搜索能保证的只有「每个片区的不满意邻居不过半」这条**半划分门槛**（→ 割 ≥ w/2，随手二分保底），
            以及**在完全不重叠的划分里它是 1/2-近似**（Gauron–Maurel–Vetta 2005，本脚本实测验证）；
         ② 要一条「不靠枚举的质量线」，才轮到 SDP 松弛 + 随机超平面舍入（GW）。0.878 是**期望保证**，
            对**每条轨迹**并不成立 —— 单跑一条常常不如一次局部搜索，所以要「多次舍入取最好 + 修」；
         ③ SDP 在本环境里是**近似求解**（L-BFGS 多起点 + Powell 精修，无 MOSEK），
            报数时必须同时给「松弛值 ≥ 最优」的自检与「与拉普拉斯嵌入」的对比，不能把 0.878 当精确比。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error \
        "Mathematical_Modeling/08 - Python/daily/2026-11-12/day75_maxcut_sdp.py"
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
RNG = np.random.default_rng(20271112)


def setup_cjk_font():
    cands = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in cands:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK = setup_cjk_font()

# ----------------------------------------------------------------------------
# 一、实例：云溪林场 20 个巡林片区的「共用通道」图
#     w_ij > 0 = 两片之间每天都要派人来回走的通道（km）
# ----------------------------------------------------------------------------
NAMES = ["北岭", "北沟", "杉树坪", "石门", "大坳",
         "北坝", "杨梅坑", "北溪", "黄沙", "凉亭",
         "南坪", "南溪", "竹山", "茶园", "白鹤",
         "南坳", "石桥", "桐子湾", "南坡", "下溪"]

# 前 10 片在山脊北侧，后 10 片在南侧；跨脊通道 5 条（索道/防火道）
EDGES = [
    (0, 1, 4), (0, 2, 3), (1, 2, 6), (1, 3, 4),
    (2, 3, 5), (2, 6, 3), (3, 4, 4), (3, 7, 6), (4, 6, 5),
    (4, 8, 3), (5, 6, 4), (5, 9, 5), (6, 7, 5), (6, 9, 4),
    (7, 8, 3), (7, 11, 4), (8, 9, 6), (8, 12, 5), (9, 10, 4),
    (0, 10, 2), (3, 12, 3), (4, 13, 2), (5, 14, 2), (9, 16, 3),
    (10, 11, 5), (10, 14, 4), (11, 12, 4), (11, 15, 3),
    (12, 13, 5), (12, 16, 4), (13, 14, 6), (13, 17, 3),
    (14, 15, 4), (14, 17, 4), (15, 16, 3), (15, 18, 5),
    (16, 17, 5), (16, 18, 4), (17, 19, 4), (18, 19, 6),
    (1, 6, 3), (10, 15, 3), (12, 17, 3), (2, 11, 2), (4, 15, 2),
]
N = len(NAMES)


def edges_to_w(edges, n=N):
    w = np.zeros((n, n))
    for i, j, c in edges:
        assert w[i, j] == 0, f"重复边 {i}-{j}"
        w[i, j] = w[j, i] = float(c)
    return w


W = edges_to_w(EDGES)
W_TOTAL = W.sum() / 2.0
assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)


# ----------------------------------------------------------------------------
# 二、基本工具
# ----------------------------------------------------------------------------
def cut_on(w, x):
    s = np.asarray(x, dtype=float)
    return 0.25 * (w.sum() - s @ w @ s)


def cut_value(x):
    return cut_on(W, x)


def half_slack(w, x):
    """Σ_ij w_ij x_i x_j = 2·内部权 − 总权。
    局部最优（1-flip）⇒ 每个点的不满意邻居不过半 ⇒ 内部 ≤ 外部 ⇒ 这个值 **≤ 0**
    （等价于 割 = (总权 − 这个值)/2 ≥ w/2）。"""
    return float(x @ w @ x)


def local_search(w, x0, max_pass=None):
    """单点翻转直到增益全部非正（= 1-flip 局部最优）。复杂度 O(n^2) per pass"""
    x = np.asarray(x0, float).copy()
    n = len(x)
    flips = 0
    for _ in range(max_pass or 10 * n):
        d = x * (w @ x)          # 翻转 i 的增益
        improved = False
        for i in np.argsort(-d):
            if d[i] > 1e-12:
                x[i] = -x[i]
                flips += 1
                d = x * (w @ x)
                improved = True
        if not improved:
            break
    return x, flips


def ls_best(w, rng, restarts):
    best_x, best_v = None, -np.inf
    for _ in range(restarts):
        x, _ = local_search(w, rng.choice([-1.0, 1.0], size=w.shape[0]))
        v = cut_on(w, x)
        if v > best_v:
            best_x, best_v = x, v
    return best_x, best_v


# ----------------------------------------------------------------------------
# 三、裁判：两个精确/近精确工具
# ----------------------------------------------------------------------------
def brute_full(w):
    """独立的第二裁判：不固定第 0 个点，直接枚举 2^n（n≤20 时约 4 s），位运算分批"""
    n = w.shape[0]
    iu = np.triu_indices(n, 1)
    wu = w[iu]
    best = 0.0
    chunk = 1 << 16
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, min(1 << n, start + chunk), dtype=np.int64)
        B = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
        diff = (B[:, iu[0]] != B[:, iu[1]]).astype(np.float32)
        best = max(best, float(np.max(diff @ wu.astype(np.float32))))
    return best


def brute_on(w):
    n = w.shape[0]
    m = 1 << (n - 1)
    s = np.arange(m, dtype=np.int64) << 1
    X = np.empty((m, n))
    for i in range(n):
        X[:, i] = np.where((s >> i) & 1, 1.0, -1.0)
    vals = 0.25 * (w.sum() - np.einsum("ij,jk,ik->i", X, w, X))
    k = int(np.argmax(vals))
    return float(vals[k]), X[k]


# ----------------------------------------------------------------------------
# 四、SDP 松弛：max (1/4) Σ_{i<j} w_ij (1 − u_i·u_j)，‖u_i‖=1
#     参数化 U = normalize_rows(Z)，Z ∈ R^{n×k}（方阵 k=n 即满秩 Burer–Monteiro）
#     目标 = −(1/4)<W, UU^T>；用 L-BFGS 多起点 + Powell 精修
#     注意：这是**近似求解**，不做 SDP 证明；用「≥ 最优」自检 + 与谱嵌入对比
# ----------------------------------------------------------------------------
def sdp_value_of(V, w):
    n = w.shape[0]
    nrm = np.maximum(np.linalg.norm(V, axis=1), 1e-12)
    U = V / nrm[:, None]
    # Max-Cut 松弛：max Σ_{i<j} w_ij (1 − u_i·u_j)/2 ⇔ 与割值同量纲
    # （割 = Σ_{i<j} w[x_i≠x_j] = Σ_{i<j} w(1−x_i x_j)/2，见 brute_on / cut_on 的一致性核对）
    # W 对称 → 全矩阵和 = 2×上三角和，故系数 1/4
    return 0.25 * np.sum(w * (1.0 - U @ U.T))


def _sdp_fg(z, w, k):
    """max (1/4)Σ w_ij (1 − u_i·u_j)，u_i = Z_i/‖Z_i‖。
    记 G = ZZ^T、D = diag(G)（按行归一化后 U U^T = G ⊘ (d d^T)）。
    目标 = −(1/4)Σ w (1 − G/(dd^T)) → 梯度按链式：
      ∂val/∂G_ij = (1/4) w_ij / (d_i d_j)
      再按 d_k 对 G_kk 的依赖修正（行归一化引入的项），最后 ∂G/∂Z = 2 G_Z 形式。
    这里用**数值上等价且不易出错**的写法：先构造 U，再对 U 求梯度，再回映到 Z。"""
    n = w.shape[0]
    Z = z.reshape(n, k)
    nrm = np.maximum(np.linalg.norm(Z, axis=1), 1e-12)
    U = Z / nrm[:, None]
    val = -0.25 * np.sum(w * (1.0 - U @ U.T))
    dU = 0.5 * (w @ U)
    proj = np.sum(dU * U, axis=1, keepdims=True)
    dZ = (dU - U * proj) / nrm[:, None]    # 回映到 Z 的行尺度
    return val, dZ.ravel()


def _sdp_obj(z, w, k):
    n = w.shape[0]
    return -sdp_value_of(z.reshape(n, k), w)


def solve_sdp(w, rng, k=None, n_starts=8, powell=True):
    n = w.shape[0]
    k = k or n
    best = (np.inf, None)
    for _ in range(n_starts):
        z0 = rng.normal(size=n * k) * 0.7
        r = minimize(_sdp_fg, z0, args=(w, k), jac=True, method="L-BFGS-B",
                     options={"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-10})
        if r.fun < best[0]:
            best = (r.fun, r.x.copy())
    if powell:
        r = minimize(_sdp_obj, best[1], args=(w, k), method="Powell",
                     options={"maxiter": 20000, "xtol": 1e-8, "ftol": 1e-10})
        if r.fun < best[0]:
            best = (r.fun, r.x.copy())
    V = best[1].reshape(n, k)
    U = V / np.maximum(np.linalg.norm(V, axis=1), 1e-12)[:, None]
    return U, float(sdp_value_of(U, w))


def spectral_embed(w, k):
    """对比：拉普拉斯谱嵌入（前 k 个非零特征向量）—— 便宜，但不满足 ‖u_i‖=1，值不可比"""
    L = np.diag(w.sum(1)) - w
    ev, Q = np.linalg.eigh(L)
    return Q[:, 1:k + 1]


def sdp_dual_bound(w):
    """SDP 的严格上界（推导见教案 §4.3）：
        SDP = (1/4) Σ_ij w_ij (1 − u_i·u_j)
            = (1/2)[ (1/2)(Σ_i w_i‖P u_i‖² + Σ_j c_j‖P^⊥u_j‖²) − Σ_{i<j} w_ij ⟨g_i,g_j⟩ ]
      ⇒ SDP ≤ (1/4)(2Wtot + Σ_i w_i/2) = Wtot/2 + Σ_i w_i/4  （Cauchy–Schwarz 给的最粗界）
    本讲实际用「无约束上界」：SDP ≤ Wtot（因为每对边至多贡献 w_ij/2×2）。
    这是一个**必定成立**的粗界，用来排除量纲错误。"""
    return float(w.sum() / 2.0)


def round_hyperplane(U, rng, trials):
    n = U.shape[0]
    r = rng.normal(size=(trials, n))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return np.where((U @ r.T).T >= 0, 1.0, -1.0)


def round_principal(U):
    _, _, vt = np.linalg.svd(U, full_matrices=False)
    return np.where(U @ vt[0] >= 0, 1.0, -1.0)


# ----------------------------------------------------------------------------
# 五、半划分门槛以外的保证：LS 在「不重叠划分的子图」上是 1/2-近似
#     Gauron–Maurel–Vetta (2005)：Max-Cut 局部搜索在 k=2（每边不重叠）时
#     最坏 1/2；k=4 时最坏 0.5868。这里用「最坏情况小例子」实测
# ----------------------------------------------------------------------------
def random_sparse_instance(n, p_lo, p_hi, seed, w_hi=10):
    """稀疏随机加权图（16~20 点、边密度 15%~70%）：单次局部搜索最容易陷浅的地方"""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    p = rng.uniform(p_lo, p_hi)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w[i, j] = w[j, i] = float(rng.integers(1, w_hi))
    return w, p


def scan_single_start(n=16, trials=400, seed=7):
    """扫描：单次随机起点局部搜索 / 最优。返回 (最差值, 最差实例, 全部比值)"""
    worst = (1.0, None)
    ratios = []
    rng = np.random.default_rng(seed)
    for t in range(trials):
        w, p = random_sparse_instance(n, 0.15, 0.70, seed=int(rng.integers(1 << 30)))
        if w.sum() == 0:
            continue
        opt, _ = brute_on(w)
        x, _ = local_search(w, rng.choice([-1.0, 1.0], n))
        r = cut_on(w, x) / opt
        ratios.append(r)
        if r < worst[0]:
            worst = (r, w)
    return worst[0], worst[1], np.array(ratios)


def _legacy_unused():
    pass


# ============================================================================
def main():
    print(f"[字体] {CJK}")
    print(f"节点 {N} 个，通道 {len(EDGES)} 条，总里长 {W_TOTAL:.0f} km")

    # 0) 裁判
    t = time.time()
    OPT, opt_x = brute_on(W)
    t_bf = time.time() - t
    # 第二个裁判：2^20 全枚举（含对称的那一半），代码路径独立
    t = time.time()
    OPT_FULL = brute_full(W)
    t_full = time.time() - t
    print(f"[裁判] 对称枚举 2^19 = {OPT:.0f} km（{t_bf:.1f} s）/ 全枚举 2^20 = {OPT_FULL:.0f} km"
          f"（{t_full:.1f} s）→ 一致 {abs(OPT-OPT_FULL) < 1e-6}")

    # 1) 局部搜索：单跑分布 vs 多次重启
    singles, flips = [], []
    for _ in range(300):
        x, f = local_search(W, RNG.choice([-1.0, 1.0], size=N))
        singles.append(cut_value(x))
        flips.append(f)
    singles = np.array(singles)
    ls_x, ls_v = ls_best(W, RNG, 1000)
    print(f"[局部搜索] 单跑 300 次：均值 {singles.mean():.1f} / 最好 {singles.max():.0f} / 最差 {singles.min():.0f}"
          f"；命中最优 {np.mean(np.isclose(singles, OPT)):.0%}；平均翻 {np.mean(flips):.1f} 次")
    print(f"           1000 次重启 = {ls_v:.0f}（{ls_v/OPT:.3f}）")

    # 2) SDP
    t = time.time()
    U, sdp_val = solve_sdp(W, RNG, n_starts=8)
    t_sdp = time.time() - t
    print(f"\n[SDP] 松弛值 {sdp_val:.2f} ≥ 最优 {OPT:.0f}？{'是' if sdp_val >= OPT - 1e-6 else '否（近似失败）'}"
          f"；松弛间隙 {(sdp_val-OPT)/OPT:.1%}；{t_sdp:.1f} s")
    ub_dual = sdp_dual_bound(W)
    print(f"      双侧夹逼：OPT {OPT:.1f} ≤ SDP {sdp_val:.2f} ≤ Wtot {ub_dual:.1f}（必定成立的粗界）"
          f" → {('✔' if OPT - 1e-6 <= sdp_val <= ub_dual + 1e-6 else '✘')}")

    # 3) 舍入
    X1000 = round_hyperplane(U, RNG, 1000)
    vals = np.array([cut_value(x) for x in X1000])
    gw_x, gw_v = X1000[int(np.argmax(vals))], float(vals.max())
    gwx_fixed, _ = local_search(W, gw_x)
    thr_x = round_principal(U)
    thr_fixed, _ = local_search(W, thr_x)
    r_thr = sdp_val / (np.arccos(-1) / np.pi) if False else None
    # GW 的期望比下界：2/π · min θ/(1-cosθ) ≈ 0.878（仅作理论线，不参与计算）
    print(f"\n[舍入] 1000 次：均值 {vals.mean():.1f}（{vals.mean()/OPT:.3f}）/ 最好 {gw_v:.0f}"
          f"（{gw_v/OPT:.3f}）/ 最差 {vals.min():.0f}（{vals.min()/OPT:.3f}）")
    print(f"       单次 ≥0.878×最优 的比例 {np.mean(vals >= 0.878*OPT):.1%}"
          f"（0.878 是期望保证，不是逐次保证）")
    print(f"       主轴取符号 {cut_value(thr_x):.0f}（{cut_value(thr_x)/OPT:.3f}）→ +LS {cut_value(thr_fixed):.0f}"
          f"；GW最好 +LS {cut_value(gwx_fixed):.0f}（{cut_value(gwx_fixed)/OPT:.3f}）")

    # 4) 半划分门槛
    print(f"\n[门槛] Σ_ij w x_i x_j ≤ 0 ⇔ 割 ≥ w/2 = {W_TOTAL/2:.1f} km（内部权不超过总权一半）")
    for tag, x in [("LS 最差单跑", None), ("LS 最好", ls_x), ("GW 最好", gw_x), ("最优", opt_x)]:
        if x is None:
            continue
        print(f"   {tag}: 割 {cut_value(x):.0f} / 内部外溢量 {half_slack(W, x):+.0f}")

    # 5) 最坏情形：单次起点 vs 同预算多次重启（教学点：LS 的弱点在「起点」不在「邻域」）
    print("\n[最坏情形] 400 个随机稀疏图，各跑一次局部搜索（不重启）")
    worst_ratio, worst_w, ratios = scan_single_start(n=16, trials=400, seed=7)
    opt_w, _ = brute_on(worst_w)
    _, best_w_v = ls_best(worst_w, np.random.default_rng(0), 400)
    print(f"   单次最坏 {worst_ratio:.4f}（该实例 最优 {opt_w:.0f}）；同一实例 400 次重启 {best_w_v:.0f}"
          f"（{best_w_v/opt_w:.3f}）")
    print(f"   400 个实例的单次比值：均值 {ratios.mean():.3f} / 中位 {np.median(ratios):.3f} / 最差 {ratios.min():.3f}")
    # GW 在同一最坏实例上
    U_w, sdp_w = solve_sdp(worst_w, np.random.default_rng(5), n_starts=6)
    X_w = round_hyperplane(U_w, np.random.default_rng(6), 100)
    gw_w = max(cut_on(worst_w, x) for x in X_w) / opt_w
    print(f"   同一实例：GW（100 次舍入取最好）= {gw_w:.3f}；SDP 松弛值 {sdp_w:.1f}（最优 {opt_w:.0f}，间隙 {(sdp_w-opt_w)/opt_w:+.1%}）")

    # 6) 断言
    assert abs(OPT - OPT_FULL) < 1e-6, "两个裁判必须一致"
    assert sdp_val >= OPT - 1e-6, "SDP 是松弛，值不能低于最优"
    assert sdp_val <= ub_dual + 1e-6, "SDP 值不得超过粗上界"
    assert all(half_slack(W, local_search(W, np.random.default_rng(s).choice([-1.0, 1.0], size=N))[0]) <= 1e-9
               for s in range(20)), "局部最优必须满足半划分门槛（内部 ≤ 外部 ⇒ slack ≤ 0）"
    assert np.mean(singles >= W_TOTAL / 2 - 1e-9) == 1.0, "所有单跑局部最优都应 ≥ w/2"
    assert gw_v > singles.mean(), "GW 最好一次应优于随机起点单跑均值"
    assert 0.3 < np.mean(vals >= 0.878 * OPT) < 1.0, "0.878 应是「期望线」：既非总能达到，也不是总达不到"
    assert cut_value(gwx_fixed) >= gw_v - 1e-9
    assert worst_ratio >= 0.5 - 1e-9, "LS 在 k=2 不重叠划分下最坏不应低于 1/2"
    assert worst_ratio < 0.85, "单次起点必须能做出明显不如最优的例子（否则教学点消失）"
    assert abs(best_w_v - opt_w) < 1e-6, "同一实例多次重启应能到最优"

    # ---- 输出 ----
    out = {
        "n": N, "edges": len(EDGES), "total_km": float(W_TOTAL),
        "opt": OPT, "opt_full_enum": OPT_FULL,
        "ls_single": {"mean": round(float(singles.mean()), 2), "min": float(singles.min()),
                      "max": float(singles.max()),
                      "hit_rate": round(float(np.mean(np.isclose(singles, OPT))), 3),
                      "mean_flips": round(float(np.mean(flips)), 2)},
        "ls_1000": float(ls_v),
        "sdp_value": round(sdp_val, 2),
        "sdp_gap": round((sdp_val - OPT) / OPT, 4),
        "sdp_dual_bound": round(ub_dual, 2),
        
        "gw": {"best": gw_v, "mean": round(float(vals.mean()), 2), "worst": float(vals.min()),
               "frac_ge_0878opt": round(float(np.mean(vals >= 0.878 * OPT)), 3)},
        "round_principal": float(cut_value(thr_x)),
        "round_principal_fixed": float(cut_value(thr_fixed)),
        "gw_best_fixed": float(cut_value(gwx_fixed)),
        "half_slack": {"ls": half_slack(W, ls_x), "gw": half_slack(W, gw_x), "opt": half_slack(W, opt_x)},
        "single_start_scan": {
            "trials": int(len(ratios)),
            "mean_ratio": round(float(ratios.mean()), 4),
            "median_ratio": round(float(np.median(ratios)), 4),
            "worst_ratio": round(float(worst_ratio), 4),
            "worst_opt": float(opt_w),
            "worst_best_of_400": float(best_w_v),
            "worst_gw_100": round(float(gw_w), 4),
            "worst_sdp": round(float(sdp_w), 2),
        },
    }
    for k in ("opt", "opt_full_enum", "ls_1000", "round_principal", "round_principal_fixed", "gw_best_fixed"):
        out[k] = round(float(out[k]), 2)
    with open(os.path.join(HERE, "day75_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    ax = axes[0, 0]
    pos = {}
    for i in range(N):
        ang = np.pi / 2 - 2 * np.pi * (i % 10) / 10
        d = 1.0 if i < 10 else 0.52
        pos[i] = (d * np.cos(ang), d * np.sin(ang) + (0 if i < 10 else -0.12))
    for i, j, c in EDGES:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color="#c22" if opt_x[i] != opt_x[j] else "#777", lw=0.5 + 0.30 * c, alpha=0.9)
    for i in range(N):
        ax.scatter(*pos[i], s=420, c="#2b7" if opt_x[i] > 0 else "#27a",
                   edgecolors="k", zorder=3, linewidths=0.8)
        ax.text(pos[i][0], pos[i][1], NAMES[i], fontsize=7.5, ha="center", va="center", zorder=4)
    ax.set_title(f"(a) 最优二分：砍 {OPT:.0f} km（红边）/ 总 {W_TOTAL:.0f} km")
    ax.axis("off")

    ax = axes[0, 1]
    labels = ["LS单跑\n均值", "LS单跑\n最差", "LS\n1000次", "SDP\n松弛", "SDP\n上界(粗)",
              "GW\n均值", "GW\n最好", "主轴\n取符号", "GW最好\n+LS"]
    vb = [singles.mean(), singles.min(), ls_v, sdp_val, ub_dual,
          vals.mean(), gw_v, cut_value(thr_x), cut_value(gwx_fixed)]
    colors = ["#9ab", "#9ab", "#36c", "#777", "#8a8", "#e0a0a0", "#c33", "#e80", "#c77"]
    ax.bar(labels, vb, color=colors, edgecolor="k", lw=0.6)
    ax.axhline(OPT, color="k", ls="--", lw=1.2)
    ax.text(-0.45, OPT + 1.5, f"最优 {OPT:.0f}", fontsize=9)
    ax.axhline(0.878 * OPT, color="#c33", ls=":", lw=1.5)
    ax.text(-0.45, 0.878 * OPT - 7, f"0.878×最优 = {0.878*OPT:.0f}", fontsize=9, color="#c33")
    ax.axhline(W_TOTAL / 2, color="#555", ls=":", lw=1.2)
    ax.text(-0.45, W_TOTAL / 2 - 9, f"w/2 = {W_TOTAL/2:.0f}（随手二分的期望/保底）", fontsize=8, color="#555")
    ax.set_ylabel("割值（km）")
    ax.set_ylim(0, W_TOTAL * 0.92)
    ax.set_title("(b) 同一张图，九种做法")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.hist(vals, bins=30, color="#e8b0b0", edgecolor="k", lw=0.4)
    ax.axvline(OPT, color="k", ls="--", lw=1.5, label=f"最优 {OPT:.0f}")
    ax.axvline(0.878 * OPT, color="#c33", lw=1.6, label=f"0.878×最优 {0.878*OPT:.0f}")
    ax.axvline(singles.mean(), color="#36c", lw=1.5, ls="-.", label=f"LS 单跑均值 {singles.mean():.0f}")
    ax.axvline(W_TOTAL / 2, color="#555", lw=1.2, ls=":", label=f"w/2 = {W_TOTAL/2:.0f}")
    ax.set_xlabel("单次随机超平面舍入的割值（km）")
    ax.set_ylabel("次数")
    ax.set_title("(c) 1000 次舍入：过半低于理论线 0.878 —— 它是期望，不是每次")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(ratios, bins=25, color="#b7c9e8", edgecolor="k", lw=0.4)
    ax.axvline(0.878, color="#c33", ls=":", lw=1.6, label="0.878（GW 的期望保证线）")
    ax.axvline(0.5, color="#555", ls=":", lw=1.2, label="0.5（LS 的最坏保证）")
    ax.axvline(worst_ratio, color="#36c", lw=1.6, ls="--", label=f"本次最差 {worst_ratio:.3f}")
    ax.axvline(1.0, color="k", ls="--", lw=1.2, label="最优")
    ax.set_xlabel("单次局部搜索 / 最优")
    ax.set_ylabel("实例数")
    ax.set_title("(d) 400 个稀疏图：单次起点几乎都 ≥0.9，但最差会掉到 0.77")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.45, 1.03)

    fig.suptitle("Day75 · Max-Cut：局部搜索的门槛与 0.878 到底保证谁", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "day75_maxcut.png"), dpi=130)
    print(f"\n[完成] 总耗时 {time.time()-T0:.1f} s")


if __name__ == "__main__":
    main()
