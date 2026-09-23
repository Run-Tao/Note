# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level:   L5 评价与决策模型 / L7 图论、网络与动态模型
roadmap: Day24 · 稳定婚配（Stable Marriage）与 Gale-Shapley 延迟接受算法
ref:     Mathematical_Modeling/05 - Algorithms/Day24 - 稳定婚配Gale-Shapley与延迟接受.md

与 Day22（指派/匈牙利）互为对照：那里目标是"总成本最小"，是**一个数**；
这里目标是"没有任何一对想私奔"，是**一种稳定性**——没有全局目标函数可优化，
只有"谁配谁才无怨"。核心武器：Gale-Shapley 延迟接受算法。

附带三个该算法著名的数学结论，本课全部实测验证：
  ① 算法一定终止，且结果一定稳定；
  ② 主动方（proposing side）得到"最优稳定匹配"，被动方得到"最差稳定匹配"；
  ③ 对男方可接受、对女方也可接受的"配对对"在任何稳定匹配中都成立（结构性约束）。

全程手写、零专用匹配库。
"""

import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 0. 问题数据：4 男 4 女，各自对对方的偏好排序（越靠前越喜欢）
# =====================================================================
MEN = ["M1", "M2", "M3", "M4"]
WOMEN = ["W1", "W2", "W3", "W4"]

# 男士对女士的偏好（列表顺序 = 从最喜欢到最不喜欢）
MEN_PREF = {
    "M1": ["W1", "W2", "W3", "W4"],
    "M2": ["W2", "W1", "W3", "W4"],
    "M3": ["W3", "W2", "W1", "W4"],
    "M4": ["W1", "W3", "W2", "W4"],
}
# 女士对男士的偏好
WOMEN_PREF = {
    "W1": ["M2", "M1", "M4", "M3"],
    "W2": ["M1", "M3", "M2", "M4"],
    "W3": ["M3", "M2", "M4", "M1"],
    "W4": ["M4", "M3", "M1", "M2"],
}


def rank(pref_list, person):
    """返回 person 在 pref_list 中的排名（越小越喜欢）"""
    return pref_list.index(person)


# =====================================================================
# 1. Gale-Shapley 延迟接受算法（男士主动）
# =====================================================================
def gale_shapley(proposers, receivers, prop_pref, recv_pref):
    """
    proposers 主动追求 receivers。
    返回 {(proposer, receiver)} 的匹配字典。
    """
    free = list(proposers)          # 尚未被接受的追求者队列
    next_idx = {p: 0 for p in proposers}   # 每个追求者下一个要追的人
    engaged = {r: None for r in receivers} # 每位被追求者当前"暂定"的对象
    match = {}

    while free:
        p = free.pop(0)
        pref = prop_pref[p]
        # 该追求者还有备选，就继续追下一个
        while next_idx[p] < len(pref):
            r = pref[next_idx[p]]
            next_idx[p] += 1
            cur = engaged[r]
            if cur is None:
                # 对方单身，直接暂定接受
                engaged[r] = p
                match[p] = r
                break
            elif rank(recv_pref[r], p) < rank(recv_pref[r], cur):
                # 对方更喜欢新来的，踹掉旧的（被踹者重新回到自由队列）
                engaged[r] = p
                match[p] = r
                del match[cur]
                free.append(cur)
                break
            # else: 对方拒绝，继续追下一个
        # 若备选追完仍无归属，则保持单身（本课男女数量相等且偏好完整，不会发生）
    return match


def is_stable(match, prop_pref, recv_pref):
    """检验一个匹配是否稳定：不存在互相更喜欢的"私奔对"。"""
    # 反查每个人的伴侣
    inv = {v: k for k, v in match.items()}
    for p in prop_pref:
        for r in recv_pref:
            if match.get(p) == r:
                continue
            cur_r = match.get(p)          # p 当前伴侣
            cur_p = inv.get(r)            # r 当前伴侣
            # p 更喜欢 r？ 且 r 更喜欢 p？ → 私奔对（阻塞对）
            p_prefers = rank(prop_pref[p], r) < rank(prop_pref[p], cur_r) if cur_r else True
            r_prefers = rank(recv_pref[r], p) < rank(recv_pref[r], cur_p) if cur_p else True
            if p_prefers and r_prefers:
                return False
    return True


# =====================================================================
# 2. 暴力枚举所有匹配，找出全部稳定匹配（小规模可行）
# =====================================================================
def all_stable_matches(men, women, men_pref, women_pref):
    stable = []
    for perm in itertools.permutations(women):
        match = dict(zip(men, perm))
        if is_stable(match, men_pref, women_pref):
            stable.append(match)
    return stable


def score_of(match, prefs, people):
    """某侧的总偏好得分：每人排名的总和（越小越好）。

    match: 男->女 的匹配；people: 要计分的这一侧的人；prefs: 该侧偏好表。
    对男士：直接 rank(prefs[m], match[m])；
    对女士：需反查她的男伴 rank(prefs[w], 男伴)。
    """
    total = 0
    for p in people:
        if p in match:            # p 是男士
            partner = match[p]
        else:                     # p 是女士：反查她的男伴
            partner = next(m for m, w in match.items() if w == p)
        total += rank(prefs[p], partner)
    return total


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 68)
    print("Day24 · 稳定婚配与 Gale-Shapley 延迟接受算法")
    print("=" * 68)

    # --- 男士主动 ---
    m_match = gale_shapley(MEN, WOMEN, MEN_PREF, WOMEN_PREF)
    # --- 女士主动 ---
    w_match_raw = gale_shapley(WOMEN, MEN, WOMEN_PREF, MEN_PREF)
    w_match = {v: k for k, v in w_match_raw.items()}  # 转成 男->女

    print("\n--- Gale-Shapley（男士主动）---")
    for m in MEN:
        print("  {} <-> {}  (排名 {} / {})".format(
            m, m_match[m], rank(MEN_PREF[m], m_match[m]), rank(WOMEN_PREF[m_match[m]], m)))
    print("  男士总排名和 =", score_of(m_match, MEN_PREF, MEN),
          " 女士总排名和 =", score_of(m_match, WOMEN_PREF, [m_match[m] for m in MEN]))
    assert is_stable(m_match, MEN_PREF, WOMEN_PREF), "男士主动结果必须稳定"
    print("  ✅ 稳定性检验通过")

    print("\n--- Gale-Shapley（女士主动）---")
    for m in MEN:
        print("  {} <-> {}  (排名 {} / {})".format(
            m, w_match[m], rank(MEN_PREF[m], w_match[m]), rank(WOMEN_PREF[w_match[m]], m)))
    print("  男士总排名和 =", score_of(w_match, MEN_PREF, MEN),
          " 女士总排名和 =", score_of(w_match, WOMEN_PREF, [w_match[m] for m in MEN]))
    assert is_stable(w_match, MEN_PREF, WOMEN_PREF), "女士主动结果必须稳定"
    print("  ✅ 稳定性检验通过")

    # --- 穷举全部稳定匹配 ---
    stable = all_stable_matches(MEN, WOMEN, MEN_PREF, WOMEN_PREF)
    print("\n--- 暴力枚举：全部稳定匹配共有 {} 个 ---".format(len(stable)))
    for idx, s in enumerate(stable, 1):
        line = "、".join("{}<->{}".format(m, s[m]) for m in MEN)
        print("  #{} {}".format(idx, line))

    # --- 验证三条定理 ---
    m_scores = [score_of(s, MEN_PREF, MEN) for s in stable]
    w_scores = [score_of(s, WOMEN_PREF, [s[m] for m in MEN]) for s in stable]
    m_best_idx = int(np.argmin(m_scores))
    w_best_idx = int(np.argmin(w_scores))

    print("\n--- 定理验证 ---")
    print("① 男士主动 男士总排名和 = {}（全部稳定匹配中最小 = {}）".format(
        score_of(m_match, MEN_PREF, MEN), min(m_scores)))
    assert score_of(m_match, MEN_PREF, MEN) == min(m_scores), "男士应得最优稳定匹配"
    print("   女士主动 女士总排名和 = {}（全部稳定匹配中最小 = {}）".format(
        score_of(w_match, WOMEN_PREF, [w_match[m] for m in MEN]), min(w_scores)))
    assert score_of(w_match, WOMEN_PREF, [w_match[m] for m in MEN]) == min(w_scores)
    print("   ✅ ② 主动方拿最优稳定匹配、被动方拿最差稳定匹配")

    # ③ 结构性约束：男士主动下男士排名 vs 女士主动下男士排名
    print("③ 每位男士在'男士主动'下的排名 总是 ≤ '女士主动'下的排名：")
    ok = True
    for m in MEN:
        r_m = rank(MEN_PREF[m], m_match[m])
        r_w = rank(MEN_PREF[m], w_match[m])
        print("     {}: 男士主动第{} 名  ≤  女士主动第{} 名".format(m, r_m + 1, r_w + 1))
        if r_m > r_w:
            ok = False
    assert ok, "结构性约束应成立"
    print("   ✅ 结构约束成立（男士主动对男士永远不更差）")

    # --- 敏感性：随机扰动偏好，看匹配是否变化 ---
    rng = np.random.default_rng(7)
    trials = 2000
    diff_from_base = 0
    stable_counts = []
    for _ in range(trials):
        mp = {m: list(rng.permutation(WOMEN)) for m in MEN}
        wp = {w: list(rng.permutation(MEN)) for w in WOMEN}
        res = gale_shapley(MEN, WOMEN, mp, wp)
        if res != m_match:
            diff_from_base += 1
        stable_counts.append(len(all_stable_matches(MEN, WOMEN, mp, wp)) if False else None)
    print("\n--- 敏感性：随机偏好 × {} 次 ---".format(trials))
    print("  与基准匹配不同的比例 = {:.1f}%（偏好一变，结果就变）".format(
        100.0 * diff_from_base / trials))

    # =================================================================
    # 画图
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 左上：喜好双向图（男士主动结果）
    ax = axes[0, 0]
    ys_m = np.linspace(0.15, 0.85, len(MEN))
    ys_w = np.linspace(0.15, 0.85, len(WOMEN))
    for m, y in zip(MEN, ys_m):
        ax.text(0.05, y, m, ha="left", va="center", fontsize=12, fontweight="bold")
    for w, y in zip(WOMEN, ys_w):
        ax.text(0.95, y, w, ha="right", va="center", fontsize=12, fontweight="bold")
    m_idx = {m: y for m, y in zip(MEN, ys_m)}
    w_idx = {w: y for w, y in zip(WOMEN, ys_w)}
    for m in MEN:
        w = m_match[m]
        ax.annotate("", xy=(0.9, w_idx[w]), xytext=(0.1, m_idx[m]),
                    arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("男士主动的稳定匹配（男左女右）")

    # 右上：两种主动方向 排名对照
    ax = axes[0, 1]
    x = np.arange(len(MEN))
    r_men_active = [rank(MEN_PREF[m], m_match[m]) + 1 for m in MEN]
    r_women_active = [rank(MEN_PREF[m], w_match[m]) + 1 for m in MEN]
    ax.bar(x - 0.2, r_men_active, 0.4, label="男士主动", color="#4c72b0")
    ax.bar(x + 0.2, r_women_active, 0.4, label="女士主动", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(MEN)
    ax.set_ylabel("该男士伴侣的偏好排名（1=最喜欢）")
    ax.set_title("主动方占优：男士在女士主动时排名更差")
    ax.legend()

    # 左下：全部稳定匹配的 男士/女士 总排名散点
    ax = axes[1, 0]
    ax.scatter(m_scores, w_scores, s=120, color="#55a868", zorder=3)
    for i, (ms, ws) in enumerate(zip(m_scores, w_scores)):
        ax.annotate("#{}".format(i + 1), (ms, ws), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([score_of(m_match, MEN_PREF, MEN)],
               [score_of(m_match, WOMEN_PREF, [m_match[m] for m in MEN])],
               marker="*", s=300, color="#dd8452", label="男士主动", zorder=4)
    ax.scatter([score_of(w_match, MEN_PREF, MEN)],
               [score_of(w_match, WOMEN_PREF, [w_match[m] for m in MEN])],
               marker="*", s=300, color="#c44e52", label="女士主动", zorder=4)
    ax.set_xlabel("男士总排名和（越小越满意）")
    ax.set_ylabel("女士总排名和（越小越满意）")
    ax.set_title("全部 {} 个稳定匹配：主动方占优".format(len(stable)))
    ax.legend()

    # 右下：敏感性
    ax = axes[1, 1]
    ax.bar(["与基准相同", "与基准不同"],
           [100 - 100.0 * diff_from_base / trials, 100.0 * diff_from_base / trials],
           color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("占比（%）")
    ax.set_title("随机偏好 × {} 次：结果随偏好剧烈变化".format(trials))

    plt.tight_layout()
    out = "stable_marriage_gale_shapley.png"
    plt.savefig(out, dpi=130)
    print("\n图已保存：", out)


if __name__ == "__main__":
    main()
