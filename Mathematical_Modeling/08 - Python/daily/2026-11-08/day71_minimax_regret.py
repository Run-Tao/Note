# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化/决策模型（离散选址 · 不确定条件下的决策准则）/ L8 算法（极小极大后悔 · 全枚举裁判）
anchor  : Roadmap L4（优化模型 → 不确定下的建站选址）→ 接 Day60 确定性 p-中值/最大覆盖、
           Day28 两阶段随机规划（SAA）、Day31 DRO、Day64 VCG（「没有自然概率时到底该按什么准则选」）
goal    : 云溪县 2027 年**渗滤液应急暂存点**（邻避型设施）选址 ——
          建几个、建在哪是县里能定的；但「这套设施该判多贵」取决于 4 套互不相同的态度
          （人口账 / 水源地优先 / 重点建设区 / 只认保护区）。这些态度**没有自然概率**，
          算不了期望，只能按「最坏」或「最不后悔」来选。
          1) 建 K 个站：按「平均态度」最省、按「最坏态度」最省、按「最不后悔」（极小极大后悔），
             三套准则选出来的点位一样吗？不一样时差在哪几个村？
          2) 「最不后悔」的最佳值若不是 0，说明 4 套态度里没有任何一个能同时兜住 —— 那下一步该干什么？
          3) 多开 1 个站能把后悔压到多少？少开 1 个站又让后悔涨多少？边际价是连续的吗？
          4) 敏感性：把最坏那套态度的权重再抬、把固定成本整体抬 30%，点位会不会换？
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-08/day71_minimax_regret.py"
输出：day71_result.json、day71_minimax_regret.png
"""
import json
import time
import itertools
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()

# ----------------------------------------------------------------------------
# 1. 实例（全部硬编码；距离 = 平面距离 × 路网系数，不手填）
# ----------------------------------------------------------------------------
SITES = ["城关东", "新坝", "河东", "城北", "石桥", "柳湾", "坳上", "云岭", "河湾"]
N = len(SITES)
XY = np.array([
    [2.0, 9.0], [0.6, 8.6], [3.6, 7.4], [1.2, 10.4], [4.2, 4.0],
    [6.4, 6.6], [1.4, 3.0], [8.2, 2.2], [4.6, 11.2],
])
# 4 套态度下「每片区受影响强度」（千人当量）——本讲的核心数据：
# 它不是距离的函数，而是各套态度**自己那份账**。同一个村，在四份账里的分量可以差 5 倍。
IMPACT = np.array([
    # 城关东 新坝 河东 城北 石桥 柳湾 坳上 云岭 河湾
    [10.0, 6.0, 9.0, 8.0, 6.0, 6.0, 5.0, 3.0, 5.0],   # 人口账：谁人多谁嗓门大
    [4.0, 4.0, 4.0, 4.0, 6.0, 3.0, 8.0, 2.0, 8.0],    # 水源地优先：石桥/坳上/河湾在下游
    [12.0, 10.0, 11.0, 9.0, 4.0, 8.0, 2.0, 4.0, 2.0],  # 重点建设区：城关/新坝/河东要发展
    [2.0, 9.0, 2.0, 2.0, 4.0, 2.0, 9.0, 1.0, 9.0],    # 只认保护区：其他一律不算
])
SCEN_NAMES = ["基线（人口账）", "严管（水源地优先）", "就业（重点建设区）", "督办（只认保护区）"]
N_SC = IMPACT.shape[0]
NET_FACTOR = 1.7            # 山区路网系数（绕行严重）
PPK = 1500.0                # 每（千人·公里）年负担折算，元
FIXED = 320000.0            # 一个站年度固定成本（造价摊 8 年 + 运维 + 封场提留），元/年
K_MAIN = 4                  # 主案：建 4 个站（K≤3 时三套准则重合，K=4 起可能分开；第 3 节用 K=5 复核）

DIST = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1)) * NET_FACTOR


def build(K, impact=None, fix_mult=1.0, impact_mult=None):
    """成本表：行 = 候选站点集合，列 = 态度。

    impact_mult: 长度 N_SC 的数组，只乘「受影响强度」（即抬高某套态度的嗓门），不动固定成本。
    若直接乘整列成本，固定成本那半截也会被放大，站少的解会被人为压低 —— 那是口径错误。
    """
    impact = IMPACT if impact is None else impact
    im = np.ones(N_SC) if impact_mult is None else np.asarray(impact_mult)
    sets_ = list(itertools.combinations(range(N), K))
    C = np.zeros((len(sets_), N_SC))
    for i, s in enumerate(sets_):
        b = DIST[:, list(s)].min(axis=1)
        for w in range(N_SC):
            C[i, w] = (b * impact[w]).sum() * PPK * im[w] + K * FIXED * fix_mult
    return sets_, C


def criteria(C):
    """三套准则 + 后悔表。后悔按列各自减本列最小值 —— 同列同单位才可比。"""
    regret = C - C.min(axis=0)[None, :]
    mean_c = C.mean(axis=1)
    worst_c = C.max(axis=1)
    mx = regret.max(axis=1)
    return {
        "mean": mean_c, "mean_i": int(mean_c.argmin()),
        "worst": worst_c, "worst_i": int(worst_c.argmin()),
        "regret": regret, "mx": mx, "regret_i": int(mx.argmin()),
    }


def r(x):
    return round(float(x), 1)


def names(s):
    return [SITES[j] for j in s]


def solve():
    out = {"实例": {
        "站点": SITES, "坐标km": XY.tolist(), "路网系数": NET_FACTOR,
        "受影响强度千人当量": IMPACT.tolist(), "态度": SCEN_NAMES,
        "单价元每千人公里年": PPK, "单站年固定元": FIXED, "主案K": K_MAIN}}

    # 1. 一刀切：K=1..8 三套准则
    scan = {}
    for K in range(1, min(9, N)):
        sets_, C = build(K)
        c = criteria(C)
        scan[K] = {
            "候选数": len(sets_),
            "平均态度": {"站点": names(sets_[c["mean_i"]]), "目标": r(c["mean"][c["mean_i"]]),
                     "最大后悔": r(c["mx"][c["mean_i"]])},
            "最坏态度": {"站点": names(sets_[c["worst_i"]]), "目标": r(c["worst"][c["worst_i"]]),
                     "最大后悔": r(c["mx"][c["worst_i"]])},
            "极小极大后悔": {"站点": names(sets_[c["regret_i"]]), "最大后悔": r(c["mx"][c["regret_i"]]),
                        "基线成本": r(C[c["regret_i"], 0]),
                        "四态度成本": [r(v) for v in C[c["regret_i"]]],
                        "四态度后悔": [r(v) for v in c["regret"][c["regret_i"]]],
                        "最坏态度": SCEN_NAMES[int(c["regret"][c["regret_i"]].argmax())]},
            "平均≠最不后悔": bool(c["mean_i"] != c["regret_i"]),
            "最坏≠最不后悔": bool(c["worst_i"] != c["regret_i"]),
            "三种并列个数": int(len({c["mean_i"], c["worst_i"], c["regret_i"]})),
        }
    out["一刀切"] = scan

    # 2. 主案 K=4
    sets4, C4 = build(K_MAIN)
    c4 = criteria(C4)
    order = np.argsort(c4["mx"])[:8]
    out["主案后悔表"] = [
        {"站点": names(sets4[i]), "四态度成本": [r(v) for v in C4[i]],
         "四态度后悔": [r(v) for v in c4["regret"][i]], "最大后悔": r(c4["mx"][i])}
        for i in order
    ]
    km = out["主案后悔表"][0]
    S_mean, S_worst, S_reg = names(sets4[c4["mean_i"]]), names(sets4[c4["worst_i"]]), names(sets4[c4["regret_i"]])
    out["主案三套准则"] = {
        "平均态度最优": {"站点": S_mean, "基线成本": r(C4[c4["mean_i"], 0]), "最大后悔": r(c4["mx"][c4["mean_i"]])},
        "最坏态度最优": {"站点": S_worst, "基线成本": r(C4[c4["worst_i"], 0]), "最大后悔": r(c4["mx"][c4["worst_i"]])},
        "极小极大后悔": {"站点": S_reg, "基线成本": r(C4[c4["regret_i"], 0]), "最大后悔": r(c4["mx"][c4["regret_i"]]),
                     "最坏态度": SCEN_NAMES[int(c4["regret"][c4["regret_i"]].argmax())]},
        "平均最优换掉的村": sorted(set(S_mean) - set(S_reg)),
        "最不后悔补上的村": sorted(set(S_reg) - set(S_mean)),
        "基线多付": r(C4[c4["regret_i"], 0] - C4[c4["mean_i"], 0]),
        "最坏后悔少付": r(c4["mx"][c4["mean_i"]] - c4["mx"][c4["regret_i"]]),
        "平均最优的后悔明细": [r(v) for v in c4["regret"][c4["mean_i"]]],
        "最不后悔的后悔明细": [r(v) for v in c4["regret"][c4["regret_i"]]],
    }

    # 2b. 复核：K=5 是否也是三套准则分开（判断「分开」是不是只在某一站数上偶然发生）
    sets5, C5 = build(5)
    c5 = criteria(C5)
    out["K5复核"] = {
        "平均态度": names(sets5[c5["mean_i"]]), "最大后悔": r(c5["mx"][c5["mean_i"]]),
        "最坏态度": names(sets5[c5["worst_i"]]), "最坏后悔": r(c5["mx"][c5["worst_i"]]),
        "极小极大后悔": names(sets5[c5["regret_i"]]), "最小遗憾": r(c5["mx"][c5["regret_i"]]),
        "三者不同": bool(len({c5["mean_i"], c5["worst_i"], c5["regret_i"]}) == 3),
    }

    # 3. 边际
    out["边际"] = {
        "各K最小遗憾": {K: scan[K]["极小极大后悔"]["最大后悔"] for K in scan},
        "多开1个(K→K+1)": {str(K): r(scan[K + 1]["极小极大后悔"]["最大后悔"] - scan[K]["极小极大后悔"]["最大后悔"])
                         for K in scan if K + 1 in scan},
        "少开1个(K→K-1)": {str(K): r(scan[K - 1]["极小极大后悔"]["最大后悔"] - scan[K]["极小极大后悔"]["最大后悔"])
                         for K in scan if K - 1 in scan},
        "全局最小遗憾的K": min(scan, key=lambda K: scan[K]["极小极大后悔"]["最大后悔"]),
    }

    # 4. 敏感性（每次都重算整张表，不沿用上一档的排序）
    sens = {}
    cases = [
        ("督办嗓门×1.5（只放大受影响强度）", IMPACT, 1.0, [1.0, 1.0, 1.0, 1.5]),
        ("就业嗓门×1.5（只放大受影响强度）", IMPACT, 1.0, [1.0, 1.0, 1.5, 1.0]),
        ("水源地嗓门×1.5（只放大受影响强度）", IMPACT, 1.0, [1.0, 1.5, 1.0, 1.0]),
        ("固定成本×1.3", IMPACT, 1.3, None),
        ("合并成一套「逐片区取最狠」", np.repeat(IMPACT.max(0, keepdims=True), N_SC, axis=0), 1.0, None),
    ]
    for tag, imp, fm, ow in cases:
        sets_, C = build(K_MAIN, imp, fm, ow)
        c = criteria(C)
        pick = names(sets_[c["regret_i"]])
        sens[tag] = {"站点": pick, "最大后悔": r(c["mx"][c["regret_i"]]),
                     "与主案一致": bool(set(pick) == set(S_reg)),
                     "换点": sorted(set(pick) ^ set(S_reg))}
    out["敏感性"] = sens

    # 5. 断言：教学点必须真的发生
    assert scan[K_MAIN]["平均≠最不后悔"], "主案必须出现「平均态度最优 ≠ 最不后悔」"
    assert len({c4["mean_i"], c4["worst_i"], c4["regret_i"]}) >= 2, "主案三套准则至少要有两套不同"
    assert c4["mx"][c4["regret_i"]] > 0, "最不后悔的解必须仍有正后悔（不存在全能解）"
    assert out["K5复核"]["三者不同"], "K=5 必须也是三套准则各自分开"
    step = out["边际"]["多开1个(K→K+1)"]
    # 本实例的边界：K=1 的「最不后悔」就是一个站，但它只有一种便宜解法（12.8 千元）；
    # 加一个站反而把「最坏态度」这一列的基准抬高 → 最小遗憾升到 23.4 千元。这不是 bug，
    # 是「多建一个站」与「少后悔」本身不等价：建站费在每列都翻倍，而省的只是各态度下的距离账。
    assert step["2"] > 0 and step["4"] < 0, "边际必须出现一次反向（K=1→2）和一次正常下降（K=4→5）"
    assert out["边际"]["少开1个(K→K-1)"][str(K_MAIN)] > 0, "主案少开一个站必须让后悔上升"
    out["耗时秒"] = round(time.perf_counter() - T0, 3)
    return out, sets4, C4, c4


def figure(out, sets4, C4, c4):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.6))
    k4 = out["主案三套准则"]

    # (a) 地图
    ax = axes[0, 0]
    base = IMPACT[0]
    ax.scatter(XY[:, 0], XY[:, 1], s=base * 70, c="#ededed", edgecolor="#8a8a8a", lw=1.1, zorder=1, label="片区（圆面积∝基线受影响强度）")
    for j, s in enumerate(SITES):
        ax.annotate(s, (XY[j, 0], XY[j, 1]), fontsize=9, xytext=(7, 5), textcoords="offset points")
    styles = [("平均态度最优", k4["平均态度最优"]["站点"], "#1f77b4", "o"),
              ("最坏态度最优", k4["最坏态度最优"]["站点"], "#ff7f0e", "s"),
              ("极小极大后悔", k4["极小极大后悔"]["站点"], "#d62728", "*")]
    for lab, pick, col, mk in styles:
        for s in pick:
            j = SITES.index(s)
            ax.scatter(XY[j, 0], XY[j, 1], s=320, marker=mk, facecolor="none",
                       edgecolor=col, lw=2.2, label=lab if s == pick[0] else None)
    ax.set_title("(a) 同样建 4 个站：平均值、最坏值、最不后悔各选一套", fontsize=11)
    ax.set_xlabel("东西向 / km"); ax.set_ylabel("南北向 / km")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.25); ax.set_aspect("equal")

    # (b) 后悔表热图
    ax = axes[0, 1]
    order = np.argsort(c4["mx"])[:10]
    mat = c4["regret"][order] / 1e4
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["+".join(names(sets4[i])) for i in order], fontsize=7)
    ax.set_xticks(range(N_SC))
    ax.set_xticklabels([s[:4] for s in SCEN_NAMES], fontsize=8.5)
    ax.set_title("(b) 后悔表（万元/年）：每格 = 比「早知道是这套态度」多花多少", fontsize=11)
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            ax.text(b, a, f"{mat[a, b]:.1f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=.045)

    # (c) 两个方案的账单对照
    ax = axes[1, 0]
    i_mean, i_reg = c4["mean_i"], c4["regret_i"]
    x = np.arange(N_SC); w = 0.38
    ax.bar(x - w / 2, C4[i_mean] / 1e4, w, label="平均态度最优", color="#1f77b4")
    ax.bar(x + w / 2, C4[i_reg] / 1e4, w, label="极小极大后悔", color="#d62728")
    for i in range(N_SC):
        ax.text(i - w / 2, C4[i_mean, i] / 1e4 + 1.2, f"{C4[i_mean, i]/1e4:.0f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, C4[i_reg, i] / 1e4 + 1.2, f"{C4[i_reg, i]/1e4:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([s[:4] for s in SCEN_NAMES], fontsize=9)
    ax.set_ylabel("年度总成本 / 万元"); ax.legend(fontsize=8)
    ax.set_title("(c) 账单对照：平均值赢一格、最坏值输一格 —— 差额就是后悔", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    # (d) K 与最小遗憾
    ax = axes[1, 1]
    ks = sorted(out["一刀切"])
    vals = [out["一刀切"][K]["极小极大后悔"]["最大后悔"] / 1e4 for K in ks]
    ax.plot(ks, vals, marker="o", color="#2ca02c", lw=2)
    for K, v in zip(ks, vals):
        ax.annotate(f"{v:.1f}", (K, v), textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")
    ax.axvline(K_MAIN, ls="--", color="#888", lw=1)
    ax.set_xlabel("站点数 K"); ax.set_ylabel("最优方案的最大后悔 / 万元·年")
    ax.set_title("(d) 多开一个站买到的「不后悔」，边际不单调", fontsize=11)
    ax.grid(alpha=.25)

    fig.suptitle("Day71 · 极小极大后悔选址：四种态度、没有概率时，先算后悔再谈最优", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = HERE / "day71_minimax_regret.png"
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return path


def main():
    out, sets4, C4, c4 = solve()
    print("=" * 88)
    print("一刀切（K=1..8）：三套准则各选谁")
    for K in sorted(out["一刀切"]):
        s = out["一刀切"][K]
        rg = s["极小极大后悔"]
        print(f"  K={K} 候选 {s['候选数']:>4} | 平均 {s['平均态度']['站点']} {s['平均态度']['目标']:>11,.0f}")
        print(f"        | 最坏 {s['最坏态度']['站点']} {s['最坏态度']['目标']:>11,.0f}")
        print(f"        | 最不后悔 {rg['站点']} 最大后悔 {rg['最大后悔']:>9,.0f} 最坏态度={rg['最坏态度']}")
    print("\n主案 K=4 后悔表（前 8）")
    for row in out["主案后悔表"]:
        print(f"  {row['最大后悔']:>11,.0f}  {'+'.join(row['站点']):<30} 成本 {row['四态度成本']}")
    print("\n主案三套准则")
    for k, v in out["主案三套准则"].items():
        print(f"  {k}: {v}")
    print("\nK=5 复核:", out["K5复核"])
    print("\n边际:", out["边际"])
    print("敏感性:", json.dumps(out["敏感性"], ensure_ascii=False))
    print(f"耗时 {out['耗时秒']} s")

    path = figure(out, sets4, C4, c4)
    out["图"] = path.name
    with open(HERE / "day71_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("图已保存:", path.name)


if __name__ == "__main__":
    main()
