#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L6 优化模型 / L7 图论、网络与动态模型
anchor:  Day40 - 最大权匹配 / 指派问题（KM 算法 Kuhn-Munkres，匈牙利法的加权版）
goal:    在一张【加权】的二分图上求【总权重最大】的完美匹配（人员—任务最优配对）。
        Day22 的匈牙利法只解"最多配几对"（最大基数匹配），权值一进来就不够用了；
        今天的 KM 算法把"收益表"翻译成一组【对偶势】u_i(行)/v_j(列)，
        维持 u_i + v_j >= w_ij（对偶可行），在"紧边"构成的相等子图里找增广路，
        找不到就整体调价（收紧势 δ），最终匹配一完美即全局最优。

本脚本零专用指派求解器，全部手写：
  1) 6x6 收益矩阵（真实决策：咨询公司派 6 名顾问各领 1 个项目的创收最大化）
  2) 手写 KM 算法（在取负的最小化问题上跑标准 KM，等价于最大化）
  3) 暴力枚举 n! 排列求真最优，作【独立裁判】
  4) 贪心对照——暴露"只顾眼前必亏"
  5) 对偶可行性核验、匹配合法性核验 + 随机图压力测试
  6) 输出四联图

说明：标准 KM（e-maxx 版）要求代价非负；本课收益全为正，转成最小化 -w 后符合条件。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-09/kuhn_munkres_weighted_matching.py"
"""

import os
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 中文字体
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
           "WenQuanYi Micro Hei", "Source Han Sans SC"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=_f),
                              fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

INF = float("inf")


# ================================================================ 手写 KM 算法
def _km_min(a):
    """KM 最小权完美匹配（标准 e-maxx 实现，要求 a 非负）。

    a: n×n 非负代价矩阵。返回 (min_cost, match, u, v)。
      match[i] = 第 i 行匹配的列
      u, v     = 行势 / 列势（对偶变量）
    可行性：u[i] + v[j] <= a[i][j]（最小化版的对偶可行）
    紧边：  u[i] + v[j] == a[i][j] → 相等子图里求完美匹配即最优。
    """
    n = len(a)
    w = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            w[i][j] = a[i - 1][j - 1]
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)      # p[j] = 匹配到列 j 的行（0 = 未匹配）
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = w[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    total = 0
    match = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            total += w[p[j]][j]
            match[p[j] - 1] = j - 1
    return total, match, u, v


def kuhn_munkres_max(weight):
    """最大权完美匹配 = 在 -weight 上跑标准 KM 最小化。返回 (total, match, u, v)。"""
    n = len(weight)
    neg = [[-weight[i][j] for j in range(n)] for i in range(n)]
    t, match, u, v = _km_min(neg)
    return -t, match, u, v


# ================================================================ 贪心对照
def greedy_matching(weight):
    """贪心：每次挑当前最大收益，划掉对应行/列。"""
    n = len(weight)
    w = np.array(weight, dtype=float)
    used_r = np.zeros(n, bool)
    used_c = np.zeros(n, bool)
    match = [-1] * n
    total = 0.0
    for _ in range(n):
        best = -1.0
        bi = bj = -1
        for i in range(n):
            if used_r[i]:
                continue
            for j in range(n):
                if used_c[j]:
                    continue
                if w[i, j] > best:
                    best, bi, bj = w[i, j], i, j
        used_r[bi] = used_c[bj] = True
        match[bi] = bj
        total += best
    return total, match


# ================================================================ 暴力裁判
def brute_force(weight):
    """暴力枚举 n! 排列求真最优（n<=8 可用）。"""
    n = len(weight)
    w = np.array(weight)
    best = -np.inf
    best_perm = None
    for perm in itertools.permutations(range(n)):
        s = sum(w[i, perm[i]] for i in range(n))
        if s > best:
            best = s
            best_perm = perm
    return best, list(best_perm)


# ================================================================ 主程序
def main():
    # ---- 真实决策：咨询公司派 6 名顾问各领 1 个项目的创收矩阵 (万元) ----
    # 行 = 6 名顾问 (A0..A5)，列 = 6 个项目 (P0..P5)
    # 刻意设计：顾问3 在 P0 上收益最高(87)、顾问0 在 P0 上也很强(85)——
    # 贪心会先抢 87，却锁死两个强顾问，导致全局总收益明显偏低。
    W = np.array([
        [85, 48, 79, 32, 63, 13],   # 顾问0
        [35, 18, 58, 70, 41, 53],   # 顾问1
        [84, 34, 40, 23, 17, 33],   # 顾问2
        [87, 29, 75, 17, 57, 83],   # 顾问3 最高收益在 P0
        [39, 17, 19, 58, 63, 10],   # 顾问4
        [62, 77, 80, 10, 53, 43],   # 顾问5
    ], dtype=float)

    n = W.shape[0]
    names_r = [f"顾问{i}" for i in range(n)]
    names_c = [f"项目{j}" for j in range(n)]

    print(f"顾问 {n} 名 / 项目 {n} 个，收益矩阵 {n}x{n}")
    for i in range(n):
        print("  " + names_r[i] + ": " + " ".join(f"{int(W[i,j]):4d}" for j in range(n)))

    # ---- 1) KM 算法 ----
    total_km, match_km, u, v = kuhn_munkres_max(W.tolist())
    print(f"\n[KM 算法] 最优总收益 = {total_km:.0f}")
    for i in range(n):
        j = match_km[i]
        print(f"    {names_r[i]} → {names_c[j]}（收益 {int(W[i,j])}）")

    # ---- 2) 暴力枚举裁判 ----
    best_bf, perm_bf = brute_force(W)
    print(f"\n[暴力枚举裁判] 最优总收益 = {best_bf:.0f}")

    # ---- 3) 贪心对照 ----
    total_gr, match_gr = greedy_matching(W)
    gap = best_bf - total_gr
    print(f"[贪心对照] 总收益 = {total_gr:.0f}（比最优少 {gap:.0f}，{gap/best_bf*100:.1f}%）")

    # ---- 断言 ----
    assert abs(total_km - best_bf) < 1e-6, "KM 应等于暴力最优"
    assert sorted(match_km) == list(range(n)), "KM 应为完美匹配（一人一项）"
    assert total_gr <= best_bf + 1e-9, "贪心不应超过最优"
    assert sorted(match_gr) == list(range(n)), "贪心也应是完美匹配"
    # 势的弱对偶核验：对最大化，-u[i]-v[j] >= -w[i][j]（因在 -w 上最小化）
    for i in range(n):
        for j in range(n):
            assert (-u[i + 1]) + (-v[j + 1]) >= (-W[i, j]) - 1e-9, "对偶可行性被破坏"
    print("\n断言全通过：KM==暴力最优 / 完美匹配 / 对偶可行 / 贪心<=最优")

    # ---- 随机小图压力测试 ----
    rng = np.random.default_rng(40)
    ok = 0
    for _ in range(50):
        m = rng.integers(1, 7)
        M = rng.integers(1, 100, size=(m, m)).astype(float)
        t_km, _, _, _ = kuhn_munkres_max(M.tolist())
        t_bf, _ = brute_force(M)
        if abs(t_km - t_bf) < 1e-6:
            ok += 1
    print(f"[随机压力测试] {ok}/50 张随机图 KM == 暴力真最优")
    assert ok == 50

    # ================================================================ 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (左上) 收益矩阵热力图 + KM 选中格
    ax = axes[0, 0]
    im = ax.imshow(W, cmap="YlGnBu")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(W[i, j])), ha="center", va="center",
                    color="black", fontsize=9)
    for i in range(n):
        j = match_km[i]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=3))
    ax.set_xticks(range(n)); ax.set_xticklabels(names_c, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names_r, fontsize=9)
    ax.set_title("收益矩阵热力图（红框 = KM 最优指派）", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (右上) 三方对照柱状
    ax = axes[0, 1]
    bars = ax.bar(["KM 最优", "暴力枚举\n(裁判)", "贪心"],
                  [total_km, best_bf, total_gr],
                  color=["#2b8cbe", "#8c96c6", "#e34a33"])
    for b, val in zip(bars, [total_km, best_bf, total_gr]):
        ax.text(b.get_x() + b.get_width() / 2, val + 4, f"{val:.0f}",
                ha="center", fontsize=11)
    ax.axhline(best_bf, color="gray", ls="--", lw=1)
    ax.set_ylabel("总收益 (万元)")
    ax.set_title(f"算法对照：贪心比最优少赚 {gap:.0f}（-{gap/best_bf*100:.1f}%）", fontsize=12)

    # (左下) 二部图：左顾问、右项目、红边=KM 匹配
    ax = axes[1, 0]
    ys_r = np.linspace(0.9, 0.1, n)
    ys_c = np.linspace(0.9, 0.1, n)
    for i in range(n):
        for j in range(n):
            ax.plot([0.3, 0.7], [ys_r[i], ys_c[j]], color="lightgray", lw=0.8, zorder=1)
    for i in range(n):
        j = match_km[i]
        ax.plot([0.3, 0.7], [ys_r[i], ys_c[j]], color="red", lw=2.5, zorder=2)
        ax.text(0.29, ys_r[i], names_r[i], ha="right", va="center", fontsize=9)
        ax.text(0.71, ys_c[j], names_c[j], ha="left", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("最优配对（红边 = KM 匹配）", fontsize=12)

    # (右下) 对偶松弛 u_i+v_j-w_ij：紧边处为 0
    ax = axes[1, 1]
    # 在最大化视角下定义 slack = w_ij - (u_i + v_j) 的等价形式；
    # -u,-v 为最小化 -w 的势，故 slack_signed = (-u_i)+(-v_j)-(-w_ij) = w_ij - u_i - v_j
    slacks = np.array([[W[i, j] - (-u[i + 1]) - (-v[j + 1]) for j in range(n)]
                       for i in range(n)])
    im = ax.imshow(slacks, cmap="OrRd")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{slacks[i,j]:.0f}", ha="center", va="center",
                    color="black", fontsize=9)
    for i in range(n):
        j = match_km[i]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="blue", lw=2.5))
    ax.set_xticks(range(n)); ax.set_xticklabels(names_c, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names_r, fontsize=8)
    ax.set_title("紧边判据（蓝框处 = 0 → 紧边，即最优匹配）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Day40 · 最大权匹配（KM 算法）：人员—任务最优配对", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "kuhn_munkres_weighted_matching.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
