# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 算法（元启发式 / 群体智能）/ L6 优化模型（组合优化）
anchor: Day43 蚁群算法 ACO（Ant Colony Optimization）求解 TSP —— 信息素 + 启发式 + 挥发
goal: 用「蚂蚁怎么找最短送货车路」回答四种递进问题：
      ① 为什么最近邻贪心会把自己越走越死（本例差 +X%）；
      ② 蚂蚁凭什么「集体」找到好路（信息素正反馈 + 距离启发式 + 挥发防早熟）；
      ③ 信息素浓度、挥发率、启发权重这三个旋钮各管什么；
      ④ 蚂蚁解怎么才算可信（Held-Karp 精确 DP 当独立裁判 + 多次运行统计）。

零专用元启发式/TSP 库，全手写：构造（概率选择下一城）→ 信息素更新（挥发 + 沉积）
→ 精英/迭代最优强化；对照：最近邻贪心；裁判：Held-Karp 精确 DP。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-12/ant_colony_tsp.py"
"""

import os
import time
import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 中文字体（避免乱码） ----------
from matplotlib import font_manager as fm

for _name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
              "Noto Sans CJK JP", "SimHei"):
    try:
        fm.findfont(fm.FontProperties(family=_name), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 实例：城市坐标（可复现，刻意设计使最近邻贪心吃亏）
# ============================================================
def make_coords(seed=7, n=16):
    """生成 n 个城市坐标。簇状分布 + 外圈，制造「近邻诱人但拖累全局」的结构，
    使最近邻贪心做短视选择、锁死后续（教学点：贪心会吃亏）。"""
    rng = np.random.default_rng(seed)
    pts = []
    # 三个簇：主干城市聚成团，团间距离远 —— 贪心容易「困在一个簇里」
    centers = [(15, 20), (75, 25), (50, 80)]
    per = [6, 5, 5]
    for (cx, cy), k in zip(centers, per):
        for _ in range(k):
            pts.append([cx + rng.normal(0, 7), cy + rng.normal(0, 7)])
    pts = np.array(pts[:n], dtype=float)
    pts = np.clip(pts, 0, 100)
    return pts


def make_coords_random(seed=2026, n=30):
    """更大的随机实例：精确 DP 已算不动，只能用「多次运行的最好解」互相参照，
    用来诚实展示「元启发式无最优性保证」——同一实例每跑一次结果会略有浮动。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2))


def dist_matrix(pts):
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


# ============================================================
# 1. 精确裁判之一：暴力全排列（仅小 n 可用，此处作交叉校验）
# ============================================================
def brute_force_tsp(D):
    n = D.shape[0]
    best, best_tour = np.inf, None
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        length = D[tour[-1], tour[0]] + sum(D[tour[i], tour[i + 1]]
                                            for i in range(n - 1))
        if length < best:
            best, best_tour = length, list(tour)
    return best, best_tour


# ============================================================
# 2. 精确裁判之二：Held-Karp 动态规划（O(2^n n^2)，n<=16 可跑）
# ============================================================
def held_karp_tsp(D):
    n = D.shape[0]
    INF = np.inf
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)
    dp[1 << 0][0] = 0.0
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        row = dp[mask]
        for j in range(n):
            if row[j] == INF:
                continue
            dj = D[j]
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                cand = row[j] + dj[k]
                if cand < dp[nm][k]:
                    dp[nm][k] = cand
                    parent[nm][k] = j
    full = (1 << n) - 1
    best_end = int(np.argmin([dp[full][j] + D[j][0] for j in range(n)]))
    best = dp[full][best_end] + D[best_end][0]
    tour, mask, j = [], full, best_end
    while j != -1:
        tour.append(int(j))
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = int(pj)
    tour = tour[::-1]
    return float(best), tour


# ============================================================
# 3. 构造启发式对照：最近邻贪心
# ============================================================
def nearest_neighbor_tsp(D, start=0):
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.discard(start)
    tour, cur = [start], start
    while unvisited:
        nxt = min(unvisited, key=lambda c: D[cur][c])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour_length(D, tour), tour


def tour_length(D, tour):
    n = len(tour)
    return float(sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n)))


# ============================================================
# 4. 主角：蚁群算法 ACO（Ant System + 精英强化）
# ============================================================
def ant_colony_optimization(
    D,
    n_ants=None,
    n_iter=150,
    alpha=1.0,       # 信息素重要性
    beta=4.0,        # 启发式（1/距离）重要性
    rho=0.1,         # 挥发率
    Q=100.0,         # 信息素沉积总量系数
    elite=3,         # 每轮额外强化的精英解个数
    seed=42,
):
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    if n_ants is None:
        n_ants = n

    eta = 1.0 / (D + 1e-12)
    np.fill_diagonal(eta, 0.0)
    tau = np.ones((n, n))

    best_tour, best_len = None, np.inf
    hist_best, hist_mean = [], []

    for it in range(n_iter):
        tours, lens = [], []
        for _ in range(n_ants):
            tour = _build_tour(D, tau, eta, alpha, beta, rng)
            L = tour_length(D, tour)
            tours.append(tour)
            lens.append(L)
            if L < best_len:
                best_len, best_tour = L, tour[:]
        lens = np.array(lens)
        hist_best.append(best_len)
        hist_mean.append(lens.mean())

        tau *= (1.0 - rho)
        for tour, L in zip(tours, lens):
            _deposit(tau, tour, Q / L)
        for idx in np.argsort(lens)[:elite]:
            _deposit(tau, tours[idx], Q / (lens[idx] * 2.0))
        _deposit(tau, best_tour, Q / (best_len * 2.0))

    return best_len, best_tour, hist_best, hist_mean


def _build_tour(D, tau, eta, alpha, beta, rng):
    n = D.shape[0]
    start = rng.integers(n)
    visited = [start]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    cur = start
    for _ in range(n - 1):
        w = (tau[cur] ** alpha) * (eta[cur] ** beta)
        w = w * unvisited
        s = w.sum()
        if s <= 0:
            nxt = int(rng.choice(np.where(unvisited)[0]))
        else:
            nxt = int(rng.choice(n, p=w / s))
        visited.append(nxt)
        unvisited[nxt] = False
        cur = nxt
    return visited


def _deposit(tau, tour, amount):
    n = len(tour)
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        tau[a][b] += amount
        tau[b][a] += amount


# ============================================================
# 5. 主流程
# ============================================================
def main(do_brute=False):
    print("=" * 68)
    print("Day43 蚁群算法 ACO 求解 TSP —— 「蚂蚁怎么找最短送货车路」")
    print("=" * 68)

    n = 16
    pts = make_coords(seed=7, n=n)
    D = dist_matrix(pts)
    print(f"\n城市数 n = {n}，坐标见正文图。")

    # --- 裁判：Held-Karp 精确最优 ---
    t0 = time.time()
    hk_len, hk_tour = held_karp_tsp(D)
    t_hk = time.time() - t0
    print(f"\n[Held-Karp 精确 DP] 最优路长 = {hk_len:.3f}，耗时 {t_hk:.3f}s")
    print(f"  最优路径 = {hk_tour}")

    # --- 对照：最近邻贪心 ---
    nn_len, nn_tour = nearest_neighbor_tsp(D)
    print(f"\n[最近邻贪心] 路长 = {nn_len:.3f}（比最优多 {nn_len - hk_len:.3f}，"
          f"{(nn_len / hk_len - 1) * 100:+.1f}%）")

    # --- 主角：ACO ---
    t0 = time.time()
    aco_len, aco_tour, hist_best, hist_mean = ant_colony_optimization(D, seed=42)
    t_aco = time.time() - t0
    print(f"\n[蚁群算法 ACO] 路长 = {aco_len:.3f}，耗时 {t_aco:.3f}s")
    print(f"  路径 = {[int(c) for c in aco_tour]}")
    print(f"  与精确最优差距 = {aco_len - hk_len:.3f}"
          f"（{(aco_len / hk_len - 1) * 100:+.2f}%）")

    # --- 多次运行统计（鲁棒性） ---
    runs = []
    for s in range(20):
        L, _, _, _ = ant_colony_optimization(D, n_iter=120, seed=1000 + s)
        runs.append(L)
    runs = np.array(runs)
    opt_hits = int(np.sum(np.isclose(runs, hk_len, atol=1e-6)))
    print(f"\n[鲁棒性] ACO 独立运行 20 次：命中精确最优 {opt_hits}/20，"
          f"均值 {runs.mean():.3f}，最差 {runs.max():.3f}"
          f"（相对最差 {(runs.max() / hk_len - 1) * 100:+.2f}%）")

    # --- 更大实例：诚实展示「元启发式无最优保证 / 每次结果会浮动」 ---
    print("\n" + "-" * 68)
    print("[更大实例 n=30] 精确 DP 已算不动 → 用「多次运行最好解」互相参照")
    big_pts = make_coords_random(seed=2026, n=30)
    Dbig = dist_matrix(big_pts)
    nn_big, _ = nearest_neighbor_tsp(Dbig)
    big_runs = []
    for s in range(6):
        L, _, _, _ = ant_colony_optimization(Dbig, n_iter=200, seed=5000 + s)
        big_runs.append(L)
    big_runs = np.array(big_runs)
    best_known = big_runs.min()
    print(f"  最近邻贪心 = {nn_big:.3f}")
    print(f"  ACO 6 次：最好 {best_known:.3f} / 最差 {big_runs.max():.3f} / "
          f"均值 {big_runs.mean():.3f}")
    print(f"  → ACO 相对最近邻贪心最好改善 "
          f"{(nn_big / best_known - 1) * 100:.1f}%；"
          f"自身 6 次浮动 "
          f"{(big_runs.max() / best_known - 1) * 100:.2f}%（无最优保证，但很稳）")

    # ---------- 断言核验 ----------
    def is_permutation(tour, n):
        return sorted(int(c) for c in tour) == list(range(n))

    assert is_permutation(aco_tour, n), "ACO 解不是合法排列"
    assert is_permutation(hk_tour, n), "Held-Karp 解不是合法排列"
    assert is_permutation(nn_tour, n), "最近邻解不是合法排列"
    assert aco_len >= hk_len - 1e-6, "ACO 解优于精确最优（不可能，裁判有误）"
    assert nn_len >= hk_len - 1e-6, "最近邻解优于精确最优（不可能，裁判有误）"
    assert abs(tour_length(D, aco_tour) - aco_len) < 1e-6, "ACO 记录长度与重算不符"
    assert opt_hits >= 15, f"ACO 鲁棒性不足（仅 {opt_hits}/20 命中最优）"
    print("\n✅ 断言全过：解为合法排列、ACO/最近邻均不优于精确最优、"
          "长度自洽、鲁棒性达标。")

    if do_brute:
        bf_len, _ = brute_force_tsp(D)
        print(f"[交叉校验] 暴力全排列 = {bf_len:.3f}（应等于 Held-Karp {hk_len:.3f}）")
        assert abs(bf_len - hk_len) < 1e-6

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.plot(hist_best, color="#d62728", lw=2, label="迭代最优")
    ax.plot(hist_mean, color="#1f77b4", lw=1.2, alpha=0.7, label="蚁群平均")
    ax.axhline(hk_len, color="k", ls="--", lw=1.5, label=f"精确最优 {hk_len:.1f}")
    ax.set_title("ACO 收敛轨迹：信息素正反馈逐步逼近最优", fontsize=13)
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("路径长度")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    names = ["最近邻贪心", "蚁群算法\nACO", "精确最优\n(Held-Karp)"]
    vals = [nn_len, aco_len, hk_len]
    colors = ["#ff7f0e", "#2ca02c", "#7f7f7f"]
    bars = ax.bar(names, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=11)
    ax.set_title("三法总路程对比（越低越好）", fontsize=13)
    ax.set_ylabel("路径长度")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0, nn_len * 0.55, f"贪心\n{(nn_len / hk_len - 1) * 100:+.1f}%",
            ha="center", color="#b35900", fontsize=11)

    ax = axes[1, 0]
    _plot_tour(ax, pts, aco_tour, f"ACO 最优送货车路（长度 {aco_len:.1f}）")
    ax = axes[1, 1]
    _plot_tour(ax, pts, hk_tour, f"Held-Karp 精确最优（长度 {hk_len:.1f}）")

    fig.suptitle("Day43 蚁群算法 ACO 求解 TSP：信息素 + 启发式 + 挥发",
                 fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ant_colony_tsp.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


def _plot_tour(ax, pts, tour, title):
    tour = [int(c) for c in tour]
    tt = tour + [tour[0]]
    ax.plot(pts[tt, 0], pts[tt, 1], "-", color="#1f77b4", lw=1.6, alpha=0.85,
            zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=90, c="#d62728", zorder=2,
               edgecolors="k")
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i), (x, y), fontsize=8, ha="center", va="center",
                    color="white", zorder=3)
    ax.scatter([pts[tour[0], 0]], [pts[tour[0], 1]], s=200, marker="s",
               facecolors="none", edgecolors="#2ca02c", lw=2, zorder=4)
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    main()
