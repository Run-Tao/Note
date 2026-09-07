#!/usr/bin/env python
"""
subject: mathematical-modeling
level: Level 6 - Optimization / 启发式与元启发式
roadmap: Level 6 / 优化模型
title: 启发式与元启发式：遗传算法(GA)+模拟退火(SA)求解 TSP 城市巡回路线
date: 2026-09-07
day: 7
goal:
    1) 用真实"多城市巡回送货/巡检"决策切入 NP-hard 的组合优化(TSP)；
    2) 说明为何精确求解器(LP/ILP/穷举)高维会炸——n=25 穷举 25!≈1.55e25；
    3) 手写遗传算法(GA)：编码+适应度+锦标赛选择+顺序交叉(OX)+变异+精英保留；
    4) 手写模拟退火(SA)：2-opt 邻域 O(1) 增量 + Metropolis 接受 + 指数降温；
    5) GA/SA 近似最优与"最近邻贪心"对比，讲清 速度↔质量(最优间隙)权衡。
配套教案见 05 - Algorithms/Day6 笔记。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import factorial as _fact

# ---------- 中文字体注册(避免乱码) ----------
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                   "Noto Serif CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)


# ----------------------------------------------------------------------
# 决策问题背景：快递/巡检公司每天要跑一批固定点位(城市),
#   车辆从仓库(0)出发, 把所有城市各拜访一次后回仓库。
#   问何种拜访顺序总路程最短? 这是旅行商问题 TSP——
#   决策是"排列/顺序", 组合爆炸, 是元启发式(遗传/退火)的经典战场。
# ----------------------------------------------------------------------
N = 25
x = rng.uniform(0, 100, N)          # 城市横坐标
y = rng.uniform(0, 100, N)          # 城市纵坐标
D = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)


def tour_length(perm):
    """给定拜访顺序排列(含回起点0), 返回总路程。"""
    return float(np.sum(D[perm[:-1], perm[1:]]) + D[perm[-1], perm[0]])


# ----------------------------------------------------------------------
# 方法A：最近邻贪心(Nearest Neighbour) —— 简单构造启发式即"基准"
#   思想: 从仓库出发每一步去最近且未拜访的城市(局部贪心)。快、直观,
#   但短视, 结果通常只是"还行", 远非最优——给 GA/SA 做下界对照。
# ----------------------------------------------------------------------
def nearest_neighbour(dist):
    n = len(dist)
    start = 0
    unvisited = set(range(1, n))
    route = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur, j])
        route.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return route


# ----------------------------------------------------------------------
# 遗传算法 GA (Genetic Algorithm)
#   编码: 个体=一条拜访顺序排列(0 固定开头). 遗传算子:
#     - 选择: 锦标赛 tour=3, 随机挑3个取最优 —— 保留多样性又择优;
#     - 交叉: 顺序交叉 OX —— 继承父本一段, 其余按另一父本相对顺序补全;
#     - 变异: 随机交换两点 / 随机翻转一段(2-opt 微扰);
#     - 精英保留: 每代原文保留当前最优个体免被破坏.
# ----------------------------------------------------------------------
def ga_tsp(dist, pop_size=200, n_gen=300, tourn=3, mut_rate=0.2, seed=0):
    r = np.random.default_rng(seed)
    n = len(dist)
    init = np.arange(n)

    def rand_route():
        p = init.copy()
        r.shuffle(p[1:])          # 起点固定 0, 其余洗牌
        return p

    pop = np.array([rand_route() for _ in range(pop_size)])
    fits = np.array([tour_length(p) for p in pop])
    best = pop[int(np.argmin(fits))].copy()
    best_len = fits.min()
    history = [best_len]

    def order_crossover(p1, p2):
        # 去起点0后做标准 OX: 在 [a,b) 继承 t1, 其余空缺按 t2 相对顺序补 → 合法排列
        t1 = p1[1:].tolist(); t2 = p2[1:].tolist()
        m = n - 1
        a, b = sorted(r.choice(m, size=2, replace=False))
        seg = set(t1[a:b])
        tail = [None] * m
        tail[a:b] = t1[a:b]                     # 继承父本 t1 的连续段
        fill = [g for g in t2 if g not in seg]  # 其余城市按 t2 顺序
        it = iter(fill)
        for i in range(m):
            if tail[i] is None:
                tail[i] = next(it)
        return np.concatenate([[0], np.array(tail, dtype=int)])

    for _ in range(n_gen):
        new_pop = []
        while len(new_pop) < pop_size:
            c1 = r.integers(0, pop_size, size=tourn)
            i = c1[np.argmin(fits[c1])]
            c2 = r.integers(0, pop_size, size=tourn)
            j = c2[np.argmin(fits[c2])]
            if r.random() < 0.9:
                c = order_crossover(pop[i], pop[j])
            else:
                c = pop[i].copy()
            if r.random() < mut_rate:
                if r.random() < 0.5:                      # swap 两点
                    u, v = r.choice(np.arange(1, n), size=2, replace=False)
                    c[u], c[v] = c[v], c[u]
                else:                                     # 翻转一段
                    a, b = sorted(r.choice(np.arange(1, n), size=2, replace=False))
                    c[a:b] = c[a:b][::-1]
            new_pop.append(c)
        pop = np.array(new_pop)
        fits = np.array([tour_length(p) for p in pop])
        # 精英保留: 最差个体用上一代的全集最优替换(避免退化)
        if best_len < fits.min():
            worst = int(np.argmax(fits))
            pop[worst] = best; fits[worst] = best_len
        else:
            best = pop[int(np.argmin(fits))].copy()
            best_len = fits.min()
        history.append(best_len)
    return best, best_len, history


# ----------------------------------------------------------------------
# 模拟退火 SA (Simulated Annealing) —— 高效版
#   思想: 借金属退火, 高温时允许接受更差的解(跳局部坑), T 下降逐渐咬定最优。
#   邻域: 2-opt 翻转一段; 其长度增量可 O(1) 算(只动被翻段首尾两条边)。
#   Metropolis: 更差解以概率 exp(-Δ/T) 接受。
# ----------------------------------------------------------------------
def _delta_2opt(perm, dist, i, k):
    """翻转 perm[i:k] 后总路长变化(回路看作循环, 含收尾回起点), O(1)。
    翻转段只改边界两条“环上相邻”边: 原边 (i-1,i) 与 (k-1,k), 换成 (i-1,k-1) 与 (i,k)
    (下标做环向模运算, 处理 k 触及结尾/回起点的情况).
    """
    n = len(perm)
    def E(a, b):          # 环上相邻两城市的边长(含回起点边)
        return dist[perm[a % n], perm[b % n]]
    removed = E(i - 1, i) + E(k - 1, k)
    added = E(i - 1, k - 1) + E(i, k)
    return float((added - removed))


def sa_tsp(dist, T0=100.0, n_steps=80000, alpha=0.9992, seed=0):
    r = np.random.default_rng(seed)
    n = len(dist)
    cur = np.concatenate([[0], r.permutation(np.arange(1, n))])
    best = cur.copy()
    best_len = cur_len = tour_length(cur)
    T = T0
    history = [best_len]
    for _ in range(n_steps):
        i, k = sorted(r.choice(np.arange(1, n), size=2, replace=False))
        j = k if k > i else i          # 语义统一为翻转 [i,k]
        if k - i == 1:                 # 相邻没必要翻(等宽)
            continue
        delta = _delta_2opt(cur, dist, i, k)
        if delta < 0 or r.random() < np.exp(-delta / T):
            cur[i:k] = cur[i:k][::-1]
            cur_len += delta
            if cur_len < best_len:
                best_len = cur_len
                best = cur.copy()
                history.append(best_len)
        T *= alpha
    return best, best_len, history


# ----------------------------------------------------------------------
# 三方运行对比
# ----------------------------------------------------------------------
nn_route = nearest_neighbour(D)
nn_len = tour_length(nn_route)
print("最近邻贪心 求解中 ...")
ga_route, ga_len, ga_hist = ga_tsp(D)
print("遗传算法GA 求解中 ...")
sa_route, sa_len, sa_hist = sa_tsp(D)

print("=" * 68)
print(f"N={N} 城市巡回(含回仓库0)。穷举 N! = {_fact(N):.2e} 种 —— 精确枚举/分支定界都难扛")
print(f"最近邻贪心 NN : {nn_len:7.1f}   (构造启发式, 秒出, 作为基准)")
print(f"遗传算法  GA : {ga_len:7.1f}   优于NN {(nn_len-ga_len)/nn_len*100:.1f}%")
print(f"模拟退火  SA : {sa_len:7.1f}   优于NN {(nn_len-sa_len)/nn_len*100:.1f}%")
print(f"GA 路线: {ga_route.tolist()}")
print(f"SA 路线: {sa_route.tolist()}")
print("=" * 68)

# 可行性断言: 都是 0..N-1 合法排列、仓库0在首位
for name, rte in [("GA", ga_route), ("SA", sa_route), ("NN", nn_route)]:
    assert sorted(rte) == list(range(N)), f"{name} 非合法排列"
    assert rte[0] == 0, f"{name} 未从仓库0出发"
    print(f"[验证] {name} 合法巡回: 起点0、每城恰一次, 自算路程 {tour_length(rte):.1f}")

# 元启发式须明显好于贪心(说明跳出局部)
assert ga_len < nn_len * 0.95, "GA 应显著优于最近邻贪心"
assert sa_len < nn_len * 0.95, "SA 应显著优于最近邻贪心"
print("[验证] GA/SA 均显著优于最近邻贪心(跳出局部贪心陷阱) ✅")
print("[验证] SA 不比 GA 差太多(都属'优秀近似')  max差 = "
      f"{max(ga_len,sa_len)-min(ga_len,sa_len):.1f}")
print("ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# 可视化: 上排三条路线, 加上收敛曲线
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
styles = [(nn_route, "最近邻贪心 NN", "#8a8a8a", f"{nn_len:.0f}"),
          (ga_route, "遗传算法 GA", "#c0392b", f"{ga_len:.0f}"),
          (sa_route, "模拟退火 SA", "#2980b9", f"{sa_len:.0f}")]

def draw_route(ax, route, title, color, label):
    rx = x[route]; ry = y[route]
    ax.plot(rx, ry, "-o", color=color, lw=1.5, ms=4, alpha=0.9)
    ax.plot([rx[-1], rx[0]], [ry[-1], ry[0]], "-o", color=color, lw=1.5, ms=4)
    ax.scatter(x[0], y[0], c="k", s=100, marker="s", zorder=5, label="仓库0")
    for i in range(1, N):
        ax.annotate(str(i), (x[i], y[i]), textcoords="offset points",
                    xytext=(3, 3), fontsize=7, color="gray")
    ax.set_title(f"{title}  路程={label}", fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)

for (rte, t, c, lab), ax in zip(styles, axes):
    draw_route(ax, rte, t, c, lab)
fig.suptitle(f"{N}-城市 TSP：构造启发式 vs 元启发式路线对比(仓库□=0)", fontsize=14)

fig2, ax2 = plt.subplots(figsize=(8, 4.3))
ax2.plot(ga_hist, label=f"GA 最优 {ga_len:.0f}", color="#c0392b")
ax2.plot(sa_hist, label=f"SA 最优 {sa_len:.0f}", color="#2980b9")
ax2.axhline(nn_len, color="gray", ls="--", label=f"NN 基准 {nn_len:.0f}")
ax2.set_xlabel("迭代步/代"); ax2.set_ylabel("当前最优路程")
ax2.set_title("元启发式收敛：遗传 GA vs 模拟退火 SA")
ax2.legend(); ax2.grid(alpha=0.3)

fig.tight_layout(); fig2.tight_layout()
import os
_save_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(_save_dir, "routes_ga_sa.png"), dpi=130)
fig2.savefig(os.path.join(_save_dir, "convergence_ga_sa.png"), dpi=130)
print(f"[图] 已保存至 {_save_dir}: routes_ga_sa.png 与 convergence_ga_sa.png")
