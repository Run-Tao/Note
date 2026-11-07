#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化模型（动态规划 DP）/ 整数规划
anchor  : Day55 - 多阶段预算分配：动态规划 · 9^6 个组合为什么不用真的数完
goal    : 用「县域 6 片区 / 12 个村级道路子项目 / 1200 万元投资」这一个决策，
          回答四件事：
          ① 9^6 = 53 万个可行组合，凭什么用「填一张表」在 1 毫秒内解完；
          ② 每条路都「先花钱后见效」（慢启动），为什么「按每元回报排序」的贪心会把 1200 万
             花成「六个片区各挖一路沟」，只拿到 8% 的效益；
          ③ 预算与效益的整条曲线说明什么：最后一档只值 43 万/年，为什么管理结论是
             「要么砍预算，要么放宽片区上限」，而不是继续加钱；
          ④ 两个片区之间出现「联营」后，为什么一维状态装不下，扩维后答案变不变、怎么验。

运行：cd /home/ubuntu/Note && ./.venv/bin/python -W all \
        "Mathematical_Modeling/08 - Python/daily/2026-10-24/dp_budget_allocation.py"
输出：dp_budget_allocation.png / dp_budget_allocation.json
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
BUDGET = 1200            # 万元（本年度指标 1200 万；县财政按 40 万一档下拨，不拆零）
STEP = 40                # 一档 = 40 万元（县财政按 40 万整数下拨，不拆零）
N_STAGE = 6              # 6 个片区 = DP 的 6 个阶段
N_SUB = 2                # 每片区 2 个子项目（各 4 档 → 每片区 8 档 = 320 万）
LEVELS = [STEP * k for k in range(9)]   # 0,40,...,320
CAP_UNITS = BUDGET // STEP              # 30 档
L = len(LEVELS) - 1

# ---------------------------------------------------------------- 中文字体
for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# F[i][k] = 片区 i 拿到 40k 万元时的年效益（万元/年）
# 前段刻意做「慢启动」：首期 40 万主要花在勘测、放线、施工便道上，几乎不产生通行效益；
# 第二、三期集中铺装才出效果 —— 这就是「边际收益先升后降」，它让「按每元回报排序」的贪心彻底失效。
F = np.array([
    [0.0,  33.0, 110.0, 236.0, 366.0, 486.0, 582.0, 644.0, 682.0],   # 0 城关（车流最密）
    [0.0,  27.0, 101.0, 228.0, 356.0, 468.0, 562.0, 594.0, 622.0],   # 1 石桥
    [0.0,  37.0, 121.0, 237.0, 316.0, 379.0, 429.0, 454.0, 448.0],   # 2 柳湾
    [0.0,  31.0, 100.0, 197.0, 263.0, 318.0, 361.0, 393.0, 399.0],   # 3 云岭
    [0.0,  25.0,  82.0, 150.0, 200.0, 250.0, 285.0, 305.0, 322.0],   # 4 河湾
    [0.0,  30.0,  67.0, 120.0, 160.0, 200.0, 230.0, 250.0, 262.0],   # 5 坳上（最偏远）
])
DISTRICT = ["城关", "石桥", "柳湾", "云岭", "河湾", "坳上"]
SYNERGY_PAIR = (0, 1)     # 片区 1 的河砂改从片区 0 的弃运点走
SYNERGY = 55.0            # 每年省 55 万
SYNERGY_TRIGGER = 5       # 前提：片区 0 的「二期完工」（5 档 = 200 万）才接通弃运点


def total_value(x: np.ndarray, synergy: bool = True) -> float:
    v = float(sum(F[i][int(k)] for i, k in enumerate(x)))
    if synergy and x[SYNERGY_PAIR[0]] >= SYNERGY_TRIGGER and x[SYNERGY_PAIR[1]] > 0:
        v += SYNERGY
    return v


# ---------------------------------------------------------------- 1. 暴力枚举
def brute_force(synergy: bool = False) -> dict:
    best, best_x, n_feasible = -1.0, None, 0
    for combo in itertools.product(range(L + 1), repeat=N_STAGE):
        if STEP * sum(combo) > BUDGET:
            continue
        n_feasible += 1
        v = float(sum(F[i][k] for i, k in enumerate(combo)))
        if synergy and combo[SYNERGY_PAIR[0]] >= SYNERGY_TRIGGER and combo[SYNERGY_PAIR[1]] > 0:
            v += SYNERGY
        if v > best:
            best, best_x = v, combo
    return {"best": best, "x": np.array(best_x), "n_feasible": n_feasible}


# ---------------------------------------------------------------- 2. 一维 DP（状态 = 已花的钱）
def dp_naive(level_cap: int | None = None) -> dict:
    cap_i = L if level_cap is None else level_cap
    g = np.full(CAP_UNITS + 1, -1e18)
    g[0] = 0.0
    choice = np.zeros((N_STAGE, CAP_UNITS + 1), dtype=int)
    for i in range(N_STAGE):
        gn = np.full(CAP_UNITS + 1, -1e18)
        for s in range(CAP_UNITS + 1):
            if g[s] <= -1e17:
                continue
            for k in range(0, min(cap_i, CAP_UNITS - s) + 1):
                if k > L:
                    break
                cand = g[s] + F[i][k]
                if cand > gn[s + k]:
                    gn[s + k] = cand
                    choice[i, s + k] = k
        g = gn
    s_star = int(np.argmax(g))
    alloc, s = np.zeros(N_STAGE, dtype=int), s_star
    for i in range(N_STAGE - 1, -1, -1):
        alloc[i] = choice[i, s]
        s -= alloc[i]
    return {"best": float(g[s_star]), "x": alloc, "used": STEP * int(alloc.sum()), "by_budget": g.copy()}


# ---------------------------------------------------------------- 3. 扩维 DP（联营让状态带上片区 0 的档位）
def dp_synergy() -> dict:
    g = np.full((CAP_UNITS + 1, L + 1), -1e18)
    for k0 in range(0, min(L, CAP_UNITS) + 1):
        g[k0, k0] = F[0][k0]
    choice = {}
    for i in range(1, N_STAGE):
        gn = np.full((CAP_UNITS + 1, L + 1), -1e18)
        ch = np.zeros((CAP_UNITS + 1, L + 1), dtype=int)
        for s in range(CAP_UNITS + 1):
            for k0 in range(L + 1):
                if g[s, k0] <= -1e17:
                    continue
                for k in range(0, min(L, CAP_UNITS - s) + 1):
                    gain = F[i][k]
                    if i == SYNERGY_PAIR[1] and k > 0 and k0 >= SYNERGY_TRIGGER:
                        gain += SYNERGY
                    cand = g[s, k0] + gain
                    if cand > gn[s + k, k0]:
                        gn[s + k, k0] = cand
                        ch[s + k, k0] = k
        choice[i] = ch
        g = gn
    s_star, k0_star = np.unravel_index(int(np.argmax(g)), g.shape)
    s_star, k0_star = int(s_star), int(k0_star)
    alloc, s = np.zeros(N_STAGE, dtype=int), s_star
    for i in range(N_STAGE - 1, 0, -1):
        alloc[i] = choice[i][s, k0_star]
        s -= alloc[i]
    alloc[0] = s
    return {"best": float(g[s_star, k0_star]), "x": alloc, "used": STEP * int(alloc.sum()),
            "states": g.size}


# ---------------------------------------------------------------- 4. 贪心：按「每元回报」排队买档
def greedy_by_ratio() -> dict:
    units = []
    for i in range(N_STAGE):
        for k in range(1, L + 1):
            units.append(((F[i][k] - F[i][k - 1]) / STEP, i, k))
    units.sort(reverse=True)
    alloc = np.zeros(N_STAGE, dtype=int)
    left = CAP_UNITS
    for _, i, k in units:
        if left == 0:
            break
        if alloc[i] == k - 1:
            alloc[i] = k
            left -= 1
    return {"x": alloc, "best": total_value(alloc)}


def main() -> None:
    res: dict = {}
    res["规模"] = {"阶段(片区)": N_STAGE, "每片区子项目": N_SUB, "档位(万元)": LEVELS,
                 "预算(万元)": BUDGET, "组合数": (L + 1) ** N_STAGE,
                 "最大可投(万元)": STEP * L * N_STAGE}

    t0 = time.perf_counter()
    bf = brute_force()
    t_bf = time.perf_counter() - t0
    t0 = time.perf_counter()
    dp = dp_naive()
    t_dp = time.perf_counter() - t0

    res["暴力枚举"] = {"最优值": round(bf["best"], 2), "最优解(档)": bf["x"].tolist(),
                    "可行组合": bf["n_feasible"], "耗时s": round(t_bf, 3)}
    res["一维DP"] = {"最优值": round(dp["best"], 2), "最优解(档)": dp["x"].tolist(),
                  "已投(万元)": dp["used"], "闲置(万元)": BUDGET - dp["used"],
                  "填表次数": CAP_UNITS * N_STAGE * (L + 1), "耗时s": round(t_dp, 5)}
    res["加速比"] = round(t_bf / max(t_dp, 1e-9), 1)

    gr = greedy_by_ratio()
    res["贪心(按每元回报)"] = {"最优值": round(gr["best"], 2), "解(档)": gr["x"].tolist()}

    dp_s = dp_synergy()
    res["联营返利"] = {
        "一维DP(看不见联营)": round(dp["best"], 2),
        "一维DP解补算联营": round(total_value(dp["x"]), 2),
        "扩维DP": round(dp_s["best"], 2),
        "扩维DP解(档)": dp_s["x"].tolist(),
        "扩维后状态数": dp_s["states"],
        "带联营枚举核对": round(brute_force(synergy=True)["best"], 2),
        "联营增量(万元/年)": round(total_value(dp["x"], True) - dp["best"], 2),
    }

    curve = [round(float(v), 1) for v in dp["by_budget"]]
    res["最优值随预算"] = curve
    res["平均每档增益"] = round(curve[-1] / CAP_UNITS, 1)
    res["末档增益"] = round(curve[-1] - curve[-2], 1)
    best_k = {}
    for i, k in enumerate(dp["x"]):
        best_k[DISTRICT[i]] = {"档位": int(k), "投入(万元)": int(40 * k),
                            "效益(万元/年)": float(F[i][k]),
                            "下一档边际(万元/年)": round(float(F[i][min(int(k) + 1, L)] - F[i][k]) / STEP, 3)}
    res["最优解明细"] = best_k
    res["整档下拨的检查"] = {"预算（万元）": BUDGET, "预算能被 40 整除": BUDGET % STEP == 0,
                        "最优解用款（万元）": dp["used"], "闲置（万元）": BUDGET - dp["used"]}

    # ---- 断言
    assert abs(bf["best"] - dp["best"]) < 1e-9, "一维 DP 与枚举不一致"
    assert t_dp * 100 < t_bf, "DP 没有明显快于枚举"


    assert dp["used"] == BUDGET, "预算没花满（与「封顶」教学点冲突）"
    assert gr["best"] < dp["best"] - 50, "贪心没输够（教学点没发生）"
    assert total_value(dp["x"], True) == dp_s["best"], "「一维DP解 + 事后补联营」居然达到了扩维最优（教学点不成立）"
    assert abs(res["联营返利"]["带联营枚举核对"] - dp_s["best"]) < 1e-9, "扩维 DP 与带联营枚举不一致"

    (HERE / "dp_budget_allocation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"{k}: {v}")

    # ---------------------------------------------------------------- 四联图
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for i, nm in enumerate(DISTRICT):
        ax[0, 0].plot(LEVELS, F[i], "o-", lw=1.4, ms=4, label=nm)
    ax[0, 0].set_title("各片区效益曲线：慢启动 → 陡升 → 饱和（边际先升后降）")
    ax[0, 0].set_xlabel("片区投入（万元）")
    ax[0, 0].set_ylabel("年效益（万元）")
    ax[0, 0].legend(fontsize=8, ncol=2)
    ax[0, 0].grid(alpha=0.3)

    bars = ax[0, 1].bar(DISTRICT, STEP * dp["x"], color="#4a7ebb")
    ax[0, 1].bar_label(bars, fmt="%d万", fontsize=9)
    ax[0, 1].axhline(STEP * L, ls=":", color="k", lw=1, label="片区封顶 320 万")
    ax[0, 1].set_title(f"一维 DP 最优解：效益 {dp['best']:.0f} 万/年，只花 {dp['used']} 万（闲置 {BUDGET-dp['used']} 万）")
    ax[0, 1].set_ylabel("投入（万元）")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3, axis="y")

    # 左下：斜率（边际收益）—— 慢启动的直接证据
    for i, nm in enumerate(DISTRICT):
        sl = np.diff(F[i]) / STEP
        ax[1, 0].plot(LEVELS[1:], sl, "o--", lw=1.2, ms=3, label=nm)
    ax[1, 0].axhline(1.0, ls="-", color="k", lw=1)
    ax[1, 0].set_title("每 40 万的边际收益：前 1~2 档 < 1（红线下=不划算），中段才划算")
    ax[1, 0].set_xlabel("片区已投（万元）")
    ax[1, 0].set_ylabel("万元/年 每 40 万元")
    ax[1, 0].legend(fontsize=7, ncol=2)
    ax[1, 0].grid(alpha=0.3)

    xs = np.arange(0, BUDGET + 1, STEP)
    ax[1, 1].plot(xs, curve, "o-", color="#4a7ebb", label="一维 DP：最优效益 g(B)")
    ax[1, 1].axhline(dp_s["best"], ls="--", color="#27ae60", lw=1.2, label=f"扩维 DP（含联营）{dp_s['best']:.0f}")
    ax[1, 1].axhline(gr["best"], ls=":", color="#c0392b", lw=1.2, label=f"贪心（每元回报）{gr['best']:.0f}")
    ax[1, 1].scatter([dp["used"]], [dp["best"]], s=70, color="#e67e22", zorder=5, label=f"实际只用到 {dp['used']} 万")
    ax[1, 1].set_title("预算越大收益越高但越平；贪心被「慢启动」钉死在第一档")
    ax[1, 1].set_xlabel("预算（万元）")
    ax[1, 1].set_ylabel("可达年效益（万元）")
    ax[1, 1].legend(fontsize=8, loc="lower right")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "dp_budget_allocation.png", dpi=150)
    print("图已保存：", HERE / "dp_budget_allocation.png")


if __name__ == "__main__":
    main()
