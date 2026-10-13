# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L5 算法（元启发式 / 群体智能）
anchor:  Day44 - 粒子群优化 PSO：县域医共体冷链中转站连续选址（多设施 Weber 问题）
goal:    用连续型群体智能算法 PSO，求解"k 个中转站选址使加权运输距离最小"的非凸多峰
         连续优化问题；并与梯度类方法、多起点局部搜索、以及 k=1 时的 Weiszfeld 解析迭代
         交叉验证，讲清"梯度走不出去、PSO 能翻山"的机制差异。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-13/pso_facility_location.py"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 中文字体
def setup_cjk_font():
    """注册系统 CJK 字体，避免图中中文乱码（Noto Sans CJK / 文泉驿）。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


# ---------------------------------------------------------------- 问题数据
def make_demand_points(seed=7):
    """20 个村卫生室的坐标（km）与疫苗需求权重。

    刻意设计成 3 个"聚落 + 一个远处孤点"的簇状分布：
    局部最优极多（每个聚落中心都像"最优"），逼梯度类方法卡住。
    """
    rng = np.random.default_rng(seed)
    # 一个"环 + 圆心 + 孤点"的刻意地貌：
    # 难点在于：单一设施放圆心得".中间解"，放环上某个点会卡在局部最优;
    # 多设施时各站会"抢"同一个聚落，天真方法极易漏掉环上的点。
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    ring = np.column_stack([50 + 40 * np.cos(angles), 50 + 40 * np.sin(angles)])
    ring += rng.normal(0, 2.0, ring.shape)
    core = np.array([[50.0, 50.0]])
    outlier = np.array([[50.0, 6.0]])   # 环下方的偏远重镇（需求大）
    pts = np.vstack([ring, core, outlier])
    wts = np.concatenate([
        rng.uniform(0.6, 1.6, size=len(ring)),   # 环上点权重小
        np.array([1.8]),                          # 圆心中权重
        np.array([4.5]),                          # 孤点大权重——拉偏重心
    ])
    return pts, wts


def total_cost(facilities, pts, wts):
    """加权运输成本 = Σ_i w_i · min_j ||x_i - c_j||₂（欧氏距离，km·需求单位）。

    facilities: shape (k, 2)
    """
    facilities = np.asarray(facilities, dtype=float).reshape(-1, 2)
    # (n, k) 距离矩阵
    d = np.linalg.norm(pts[:, None, :] - facilities[None, :, :], axis=2)
    return float(np.sum(wts * d.min(axis=1)))


# ---------------------------------------------------------------- 主角：PSO
def pso_facility_location(pts, wts, k, n_particles=60, iters=400,
                          w=0.72, c1=1.49445, c2=1.49445, seed=0,
                          bounds=(0.0, 100.0)):
    """粒子群优化：k 个中转站连续选址。

    每个粒子 = 一个 k×2 的坐标摊平成长度 2k 的向量（位置）。
    速度按标准 PSO 更新：v ← w·v + c1·r1·(pbest-x) + c2·r2·(gbest-x)。
    pbest = 粒子历史最优（个体经验），gbest = 群体历史最优（集体经验）。
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    dim = 2 * k

    # 初始化：位置在界内均匀撒点，速度取界宽的 10%
    x = rng.uniform(lo, hi, size=(n_particles, dim))
    v = rng.uniform(-0.1, 0.1, size=(n_particles, dim)) * (hi - lo)

    # 评估初始适应度
    fit = np.array([total_cost(x[i].reshape(k, 2), pts, wts) for i in range(n_particles)])
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history = [gbest_fit]           # 每轮记录群体最优（收敛曲线）
    for _ in range(iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        # 限速，防止粒子"飞出太阳系"
        v_max = 0.2 * (hi - lo)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, lo, hi)      # 触界则吸附在边界

        fit = np.array([total_cost(x[i].reshape(k, 2), pts, wts) for i in range(n_particles)])
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()
        history.append(gbest_fit)

    return gbest.reshape(k, 2), gbest_fit, np.array(history)


# ---------------------------------------------------------------- 独立裁判
def weiszfeld_k1(pts, wts, iters=2000, tol=1e-12):
    """k=1 的经典解析迭代法（Weiszfeld 算法）——给单设施问题一个"真值"。"""
    x = np.average(pts, axis=0, weights=wts)
    for _ in range(iters):
        d = np.linalg.norm(pts - x, axis=1)
        d = np.maximum(d, 1e-12)
        ratio = wts / d
        x_new = np.sum(pts * ratio[:, None], axis=0) / np.sum(ratio)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def brute_force_grid_k(pts, wts, k, grid=60):
    """网格暴力枚举（独立裁判）：
    对 k 个站点做全组合网格搜索，得到该分辨率下的"准全局最优"。
    规模 = C(grid^2, k) 太大，故只对 k=1 全网格、k>=2 用粗网格 + 组合抽样。
    """
    g = np.linspace(0, 100, grid)
    GX, GY = np.meshgrid(g, g)
    cells = np.column_stack([GX.ravel(), GY.ravel()])
    if k == 1:
        best, bx = np.inf, None
        for c in cells:
            val = total_cost(c.reshape(1, 2), pts, wts)
            if val < best:
                best, bx = val, c
        return bx.reshape(1, 2), best
    # k>=2：网格点上随机组合采样（Monte-Carlo 式粗裁判，只作下界参照）
    rng = np.random.default_rng(123)
    best, bx = np.inf, None
    for _ in range(200000):
        idx = rng.choice(len(cells), size=k, replace=False)
        cand = cells[idx]
        val = total_cost(cand, pts, wts)
        if val < best:
            best, bx = val, cand
    return bx, best


def multistart_local(pts, wts, k, n_starts=60, seed=1, maxiter=2000, method="Nelder-Mead"):
    """对照：多次随机起点 + 局部搜索，模拟"梯度类/局部"方法。

    n_starts / maxiter 是"计算预算"：预算小 → 局部方法容易被非凸地形困住。
    method="L-BFGS-B" 时用数值梯度，代表 Day33 讲过的拟牛顿法（真·梯度类）。
    """
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    best, bx = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(0, 100, size=2 * k)
        opts = {"maxiter": maxiter}
        if method == "Nelder-Mead":
            opts.update({"xatol": 1e-4, "fatol": 1e-4})
        res = minimize(lambda z: total_cost(z.reshape(k, 2), pts, wts), x0,
                       method=method, options=opts)
        # 局部方法也要保证解不离边界
        xb = np.clip(res.x, 0, 100)
        fv = total_cost(xb.reshape(k, 2), pts, wts)
        if fv < best:
            best, bx = float(fv), xb.reshape(k, 2)
    return bx, best


def kmeans_centroid_heuristic(pts, wts, k, seed=0, n_restart=3):
    """天真方案：把需求点按"加权 KMeans"聚成 k 簇、各站建在簇质心。
    这是很多人第一反应的做法，但它只在"簇结构干净"时才对——
    本课实例里它会漏掉孤点/远簇，成本明显偏高。
    """
    from sklearn.cluster import KMeans
    best, bx = np.inf, None
    for s in range(n_restart):
        km = KMeans(n_clusters=k, n_init=1, random_state=seed + s)
        km.fit(pts, sample_weight=wts)
        val = total_cost(km.cluster_centers_, pts, wts)
        if val < best:
            best, bx = float(val), km.cluster_centers_
    return bx, best


# ---------------------------------------------------------------- 主流程
def main():
    font = setup_cjk_font()
    print(f"[字体] 使用中文字体：{font}")

    pts, wts = make_demand_points(seed=7)
    n = len(pts)
    print(f"需求点数 n = {n}；总需求权重 = {wts.sum():.2f}")

    results = {}

    # ---------- k = 1：单设施（有 Weiszfeld 真值可比） ----------
    k = 1
    fac_pso, cost_pso, hist_pso = pso_facility_location(pts, wts, k, seed=0)
    fac_w, cost_w = weiszfeld_k1(pts, wts), None
    cost_w = total_cost(fac_w.reshape(1, 2), pts, wts)
    fac_grid, cost_grid = brute_force_grid_k(pts, wts, k, grid=200)
    print("\n===== k = 1（单中转站）=====")
    print(f"[PSO]       成本 = {cost_pso:.4f}  位置 = {np.round(fac_pso.ravel(), 3)}")
    print(f"[Weiszfeld] 成本 = {cost_w:.4f}  位置 = {np.round(fac_w, 3)}")
    print(f"[网格暴力]  成本 = {cost_grid:.4f}  位置 = {np.round(fac_grid.ravel(), 3)}")
    results["k1"] = dict(pso=(fac_pso, cost_pso, hist_pso),
                         weiszfeld=(fac_w.reshape(1, 2), cost_w),
                         grid=(fac_grid, cost_grid))

    # ---------- k = 2：两设施（多峰，局部/天真方法易卡） ----------
    k = 2
    # 只给少量起点的局部搜索（"预算受限"的局部方法）
    fac_loc, cost_loc = multistart_local(pts, wts, k, n_starts=3, seed=1, maxiter=400)
    fac_one, cost_one = multistart_local(pts, wts, k, n_starts=1, seed=9, maxiter=400)
    fac_grad, cost_grad = multistart_local(pts, wts, k, n_starts=1, seed=9, maxiter=400,
                                           method="L-BFGS-B")
    fac_km, cost_km = kmeans_centroid_heuristic(pts, wts, k, seed=0)
    # 现实中最常见的做法：用天真的 KMeans 质心当起点，再开梯度法“微调”。
    # 结果：梯度法被锁在KMeans解附近，只能小改进——这就是局部方法的"真坑"。
    from scipy.optimize import minimize as _min
    _r = _min(lambda z: total_cost(z.reshape(k, 2), pts, wts), fac_km.ravel(),
              method="L-BFGS-B", options={"maxiter": 800})
    _x = np.clip(_r.x, 0, 100)
    fac_grad2, cost_grad2 = _x.reshape(k, 2), total_cost(_x.reshape(k, 2), pts, wts)
    fac_pso2, cost_pso2, hist2 = pso_facility_location(pts, wts, k, seed=3)
    fac_ref, cost_ref = brute_force_grid_k(pts, wts, k)
    print("\n===== k = 2（两座中转站）=====")
    print(f"[KMeans 质心]  成本 = {cost_km:.4f}  位置 = {np.round(fac_km, 3).tolist()}")
    print(f"[KMeans起点+梯度微调] 成本 = {cost_grad2:.4f}  位置 = {np.round(fac_grad2, 3).tolist()}")
    print(f"[单起点梯度(L-BFGS)] 成本 = {cost_grad:.4f}  位置 = {np.round(fac_grad, 3).tolist()}")
    print(f"[单起点局部]   成本 = {cost_one:.4f}  位置 = {np.round(fac_one, 3).tolist()}")
    print(f"[局部搜索 x3]  成本 = {cost_loc:.4f}  位置 = {np.round(fac_loc, 3).tolist()}")
    print(f"[PSO]          成本 = {cost_pso2:.4f}  位置 = {np.round(fac_pso2, 3).tolist()}")
    print(f"[网格粗裁判]   成本 = {cost_ref:.4f}")
    results["k2"] = dict(km=(fac_km, cost_km), grad2=(fac_grad2, cost_grad2),
                         grad=(fac_grad, cost_grad),
                         one=(fac_one, cost_one), local=(fac_loc, cost_loc),
                         pso=(fac_pso2, cost_pso2, hist2), ref=(fac_ref, cost_ref))

    # ---------- k = 3：四簇结构，PSO 优势更明显 ----------
    k = 3
    fac_km3, cost_km3 = kmeans_centroid_heuristic(pts, wts, k, seed=0)
    fac_loc3, cost_loc3 = multistart_local(pts, wts, k, n_starts=3, seed=2, maxiter=400)
    fac_pso3, cost_pso3, hist3 = pso_facility_location(pts, wts, k, seed=5)
    print("\n===== k = 3（三座中转站）=====")
    print(f"[KMeans 质心]  成本 = {cost_km3:.4f}")
    print(f"[局部搜索 x3]  成本 = {cost_loc3:.4f}")
    print(f"[PSO]          成本 = {cost_pso3:.4f}  位置 = {np.round(fac_pso3, 3).tolist()}")
    results["k3"] = dict(km=(fac_km3, cost_km3), local=(fac_loc3, cost_loc3),
                         pso=(fac_pso3, cost_pso3, hist3))

    # ---------- 鲁棒性：PSO 多次独立运行 ----------
    print("\n===== 鲁棒性（k=2，PSO 独立运行 20 次）=====")
    runs = [pso_facility_location(pts, wts, 2, seed=s)[1] for s in range(20)]
    runs = np.array(runs)
    print(f"最好 {runs.min():.4f} / 最差 {runs.max():.4f} / 均值 {runs.mean():.4f} / 标准差 {runs.std():.4f}")
    print(f"自身浮动 {(runs.max()-runs.min())/runs.mean()*100:.2f}%")
    results["robust"] = runs

    # ---------- 断言 ----------
    assert cost_pso <= cost_grid * 1.001, "PSO(单设施) 不应明显劣于网格暴力"
    assert abs(cost_pso - cost_w) / cost_w < 0.01, "PSO 单设施应与 Weiszfeld 一致(1%内)"
    assert cost_pso2 <= cost_loc + 1e-6, "PSO(两设施) 不应劣于局部搜索"
    assert cost_pso2 < cost_grad2, "PSO 应优于‘KMeans起点+梯度微调’（否则’翻山’教学点落空）"
    assert cost_pso2 < cost_km, "PSO(两设施) 应优于天真的 KMeans 质心方案"
    assert cost_pso2 <= cost_one + 1e-6, "PSO(两设施) 不应劣于单起点局部搜索"
    assert cost_pso2 <= cost_ref * 1.01, "PSO(两设施) 不应明显劣于粗裁判"
    assert cost_pso3 <= cost_loc3 + 1e-6, "PSO(三设施) 不应劣于局部搜索"
    assert cost_pso3 < cost_km3, "PSO(三设施) 应优于 KMeans 质心方案"
    assert results["robust"].max() <= results["robust"].mean() * 1.05, "PSO 多次运行应稳定"
    print("\n[断言] 全部通过 ✔")

    # ---------------------------------------------------------------- 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 左上：单设施 —— PSO 位置 + Weiszfeld + 网格最优
    ax = axes[0, 0]
    ax.scatter(pts[:, 0], pts[:, 1], s=wts * 60, c="#2c7fb8", alpha=0.7, edgecolor="k", zorder=3)
    f1 = results["k1"]["pso"][0][0]
    ax.scatter(*f1, marker="*", s=420, c="red", edgecolor="k", zorder=5, label="PSO 最优位置")
    ax.set_title("k=1：PSO / Weiszfeld / 网格 均落在同一点", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 右上：收敛曲线（k=1,2,3）
    ax = axes[0, 1]
    for k, key, color in [(1, "k1", "#d62728"), (2, "k2", "#2ca02c"), (3, "k3", "#9467bd")]:
        hist = results[key]["pso"][2]
        ax.plot(hist, color=color, lw=1.8, label=f"k={k} PSO 群体最优")
    ax.set_xlabel("迭代轮次"); ax.set_ylabel("加权运输成本")
    ax.set_title("PSO 收敛曲线：阶梯式下探", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 左下：k=2 —— KMeans质心/局部搜索 vs PSO
    ax = axes[1, 0]
    ax.scatter(pts[:, 0], pts[:, 1], s=wts * 60, c="#2c7fb8", alpha=0.6, edgecolor="k", zorder=3)
    fk = results["k2"]["km"][0]
    fg = results["k2"]["grad2"][0]
    fp = results["k2"]["pso"][0]
    ax.scatter(fk[:, 0], fk[:, 1], marker="s", s=150, c="orange", edgecolor="k",
               zorder=4, label=f"KMeans质心 成本={results['k2']['km'][1]:.2f}")
    ax.scatter(fg[:, 0], fg[:, 1], marker="^", s=190, c="#9467bd", edgecolor="k",
               zorder=4, label=f"质心起点+梯度 成本={results['k2']['grad2'][1]:.2f}")
    ax.scatter(fp[:, 0], fp[:, 1], marker="*", s=420, c="red", edgecolor="k",
               zorder=5, label=f"PSO 成本={results['k2']['pso'][1]:.2f}")
    ax.set_title("k=2：天真/梯度法卡在次优 vs PSO 翻山", fontsize=12)
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

    # 右下：k=3 方案
    ax = axes[1, 1]
    ax.scatter(pts[:, 0], pts[:, 1], s=wts * 60, c="#2c7fb8", alpha=0.6, edgecolor="k", zorder=3)
    fp3 = results["k3"]["pso"][0]
    ax.scatter(fp3[:, 0], fp3[:, 1], marker="*", s=460, c="red", edgecolor="k", zorder=5)
    for i, (x, y) in enumerate(fp3):
        ax.annotate(f"站{i}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_title(f"k=3 最优方案（成本={results['k3']['pso'][1]:.2f}）", fontsize=12)
    ax.grid(alpha=0.3)

    fig.suptitle("粒子群优化 PSO · 县域医共体冷链中转站选址（多设施 Weber 问题）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pso_facility_location.png")
    fig.savefig(out, dpi=130)
    print(f"\n[图] 已保存 -> {out}")

    return results


if __name__ == "__main__":
    main()
