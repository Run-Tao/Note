# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L2 优化模型（整数规划）/ L8 算法（列生成 · 定价子问题）
anchor      : Roadmap L7-L8 —— 接 Day36 列生成与 Dantzig-Wolfe（一般框架）、
              Day37 割平面（「削顶点」的另一种思路）、Day4 0-1 ILP 分支定界、
              Day52 灵敏度分析（最优基的有效期）、Day59「求解器说 Optimal ≠ 最优」的审计习惯
goal        : 云溪县门窗型材下料（一维下料问题 1D-CSP；钢管切割、卷材分切、钢筋定尺同型）。
              14 m 母料 + 5 种规格 + 10% 报废率。回答五件事：
              ① 「一根切满再切下一根」的按根贪心 —— 多花多少料、它的浪费到底浪费在哪
              ② 只用两种切法（全切一种 / 满排一份）能省多少根 —— 简单切法并不差多少
              ③ 整数规划 + 全部可达切法 —— 最优根数是多少、这怎么变成「列生成」的定价子问题
              ④ ★ LP 松弛给的是「13.1 根」这种小数 —— 向上取整补种，哪个方向补最省钱
              ⑤ 板材加长一定省料吗、短料各放宽 10 公分为什么反而多买 1 根、哪一条料构成瓶颈
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-02/day65_cutting_stock.py"
输出：day65_result.json + day65_cutting_stock.png（四面板）
"""

import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

warnings.simplefilter("always")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day65_result.json")
OUT_PNG = os.path.join(HERE, "day65_cutting_stock.png")

# ---------------------------------------------------------------- 中文字体
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
]
_installed = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in CJK_CANDIDATES if n in _installed), None)
if CJK is not None:
    plt.rcParams["font.sans-serif"] = [CJK] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 场景数据
STOCK = 14000.0        # 母料 14 m（mm）
COST_PER_BAR = 116.0   # 14 m 铝合金型材单价（元/根，含切断）
WASTE_RATE = 0.10      # 下料报废率（料头、气孔、切歪）
KERF = 3.0             # 锯口 3 mm

PARTS = [              # (名称, 单支长 mm, 需求量 支)
    ("A 落地窗竖框", 2700, 25),
    ("B 飘窗横档", 1150, 45),
    ("C 推拉门下轨", 900, 50),
    ("D 纱窗框", 700, 20),
    ("E 封边压条", 450, 10),
]
N_SPEC = len(PARTS)
NAMES = [p[0] for p in PARTS]
LEN_MM = np.array([p[1] for p in PARTS], dtype=float)
DEMAND = np.array([p[2] for p in PARTS], dtype=int)


def gross_need(mult=1.0):
    """含报废率的毛需求（向上取整）。"""
    return np.ceil(DEMAND * mult * (1.0 + WASTE_RATE)).astype(int)


def max_count(lmm, stock_mm):
    """一根母料能切几支该规格（含每支一刀 3 mm 锯缝）。"""
    if lmm > stock_mm:
        return 0
    return int((stock_mm + KERF) // (lmm + KERF))


def pattern_len(cnt, lens=None):
    """一套切法的实际占用长度（mm），含锯缝。"""
    lens = LEN_MM if lens is None else np.asarray(lens, dtype=float)
    cnt = np.asarray(cnt, dtype=float)
    n = int(cnt.sum())
    if n == 0:
        return 0.0
    return float(np.dot(cnt, lens + KERF) - KERF)


# ---------------------------------------------------------------- 切法（列）枚举
def gen_patterns(stock_mm, need, lens=None):
    """穷举一根母料的所有可达切法（每规格用 ≤ 毛需求；DFS + 可行性剪枝）。"""
    lens = LEN_MM if lens is None else np.asarray(lens, dtype=float)
    caps = [max_count(l, stock_mm) for l in lens]
    out = []

    def dfs(i, cnt, used):
        if i == N_SPEC:
            if sum(cnt) > 0:
                out.append(tuple(cnt))
            return
        for k in range(min(caps[i], int(need[i])), -1, -1):
            add = k * (lens[i] + KERF) - (KERF if k else 0.0)
            if used + add <= stock_mm + 1e-9:
                dfs(i + 1, cnt + [k], used + add)
            elif k > 0:
                continue          # 太长，试更小的 k
            else:
                dfs(i + 1, cnt + [0], used)   # 这一种不切，继续下一种

    dfs(0, [], 0.0)
    return [np.array(p, dtype=int) for p in out]


def simple_patterns(stock_mm=STOCK):
    """两种「傻瓜」切法：① 一根只切一种规格（不混搭）② 按图纸单套一份（每种规格各一支）。"""
    pats = [tuple(1 if j == i else 0 for j in range(N_SPEC)) for i in range(N_SPEC)]
    pats.append(tuple(1 for _ in range(N_SPEC)))
    return [np.array(p, dtype=int) for p in pats]


# ---------------------------------------------------------------- 求解（显式对偶，不用求解器取影子价格）
def lp_primal(pool, need):
    """受限主问题（LP）：min Σx s.t. A^T x ≥ need, x ≥ 0。返回 (值, x, y)。"""
    A = np.array(pool, dtype=float).T                 # 规格 × 列
    c = np.ones(A.shape[1])
    r = linprog(c, A_ub=-A, b_ub=-need.astype(float), bounds=(0, None), method="highs")
    assert r.success, r.message
    y = np.maximum(-np.asarray(r.ineqlin.marginals, dtype=float), 0.0)   # 对偶价格 ≥ 0
    return float(r.fun), np.asarray(r.x, dtype=float), y


def ip_solve(pool, need, fixed_bars=None):
    """整数主问题：min Σx s.t. A^T x ≥ need（+ Σx = fixed_bars），x ∈ Z+。"""
    A = np.array(pool, dtype=float).T
    cons = [LinearConstraint(A, need.astype(float), np.full(N_SPEC, np.inf))]
    if fixed_bars is not None:
        cons.append(LinearConstraint(np.ones((1, A.shape[1])), [fixed_bars], [fixed_bars]))
    return milp(c=np.ones(A.shape[1]), constraints=cons, bounds=Bounds(0, np.inf),
                integrality=np.ones(A.shape[1]))


def solve_duals(patterns, need):
    """对偶：max need·y  s.t. A^T y ≤ 1, y ≥ 0（列生成的「每支规格值多少根母料」）。"""
    A = np.array(patterns, dtype=float).T
    r = milp(c=-need.astype(float),
             constraints=[LinearConstraint(A.T, -np.inf, np.ones(A.shape[1]))],
             bounds=Bounds(0, np.inf))
    if not r.success:
        return None
    return np.asarray(r.x, dtype=float)


def stats(patterns, x, need, stock_mm=STOCK, lens=None):
    """根数、实得料（只算需求内）、超产料、利用率、型材费。"""
    lens = LEN_MM if lens is None else np.asarray(lens, dtype=float)
    x = np.asarray(x, dtype=float)
    bars = int(round(float(x.sum())))
    made = np.zeros(N_SPEC)
    for p, xi in zip(patterns, x):
        made += p * np.round(xi)
    excess_mm = float(np.dot(np.maximum(made - need, 0), lens + KERF))
    used_mm = float(np.dot(np.minimum(made, need), lens + KERF)) - KERF * bars
    unit_price = COST_PER_BAR * stock_mm / 14000.0   # 按长度线性计价
    return {
        "bars": bars,
        "used_m": round(used_mm / 1000.0, 3),
        "excess_m": round(excess_mm / 1000.0, 3),
        "waste_m": round((bars * stock_mm - used_mm - excess_mm) / 1000.0, 3),
        "yield": round(used_mm / (bars * stock_mm), 4),
        "cost": round(bars * unit_price, 1),
        "made": made.astype(int).tolist(),
    }


def greedy(stock_mm, need, lens=None):
    """按根贪心：每根都从最长的规格开始切，谁装得下就装，装满换下一根（「先切好的、剩料再说」）。

    注意：这是「按根优先」——每一根都想要最大化自身利用，不考虑「把这根留给后面的长料」。
    """
    lens = LEN_MM if lens is None else np.asarray(lens, dtype=float)
    rem = np.zeros(N_SPEC, dtype=float)
    rem[:] = need
    bars = []
    while rem.sum() > 1e-9:
        cnt = np.zeros(N_SPEC, dtype=float)
        used = 0.0
        for i in np.argsort(-lens):
            while rem[i] - cnt[i] > 1e-9 and used + lens[i] + (KERF if used > 0 else 0.0) <= stock_mm + 1e-9:
                cnt[i] += 1
                used += lens[i] + (KERF if used > 0 else 0.0)
        assert cnt.sum() > 0, "有规格切不进母料"
        bars.append(cnt)
        rem -= cnt
    return bars


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    need = gross_need()
    out = {"stock_mm": STOCK, "cost_per_bar": COST_PER_BAR, "waste_rate": WASTE_RATE,
           "specs": [{"name": n, "len_mm": int(l), "demand": int(d), "gross": int(g)}
                     for (n, l, d), g in zip(PARTS, need)]}
    print("毛需求（含 10% 报废）：", dict(zip([n[0] for n in NAMES], need.tolist())))
    print("总净长 %.2f m" % (np.dot(need, LEN_MM) / 1000))

    # ---- 0. 按需求直接买整支（不切、多截少用的极端对照）
    naive = {"bars": int(need.sum())}
    out["naive_bars"] = naive["bars"]

    # ---- 1. 按根贪心
    bars_g = greedy(STOCK, need)
    sg = stats(bars_g, np.ones(len(bars_g)), need)
    sg["patterns_used"] = len(bars_g)
    print(f"[1] 按根贪心：{sg['bars']} 根，利用率 {sg['yield']*100:.1f}%，"
          f"料头 {sg['waste_m']:.2f} m，超产 {sg['excess_m']:.2f} m，{sg['cost']} 元")
    out["greedy"] = sg

    # ---- 2. 两种切成法
    ps = simple_patterns()
    rs = ip_solve(ps, need)
    xs = rs.x
    assert rs.success, "两种切法应可行"
    ss = stats(ps, xs, need)
    ss["patterns_used"] = {",".join(str(int(v)) for v in p): round(float(x), 2)
                           for p, x in zip(ps, xs) if x > 1e-9}
    print(f"[2] 两种切法：{ss['bars']} 根，利用率 {ss['yield']*100:.1f}%，料头 {ss['waste_m']:.2f} m，{ss['cost']} 元")
    out["simple"] = ss

    # ---- 3. 整数规划 + 全部可达切法
    pf = gen_patterns(STOCK, need)
    rf = ip_solve(pf, need)
    assert rf.success, rf.message
    xf = rf.x
    sf = stats(pf, xf, need)
    sf["n_patterns"] = len(pf)
    sf["patterns_used"] = {",".join(str(int(v)) for v in p): int(round(x))
                           for p, x in zip(pf, xf) if x > 0.5}
    print(f"[3] 整数规划（{len(pf)} 种切法）：{sf['bars']} 根，利用率 {sf['yield']*100:.1f}%，"
          f"料头 {sf['waste_m']:.2f} m，超产 {sf['excess_m']:.2f} m，{sf['cost']} 元")
    out["ip"] = sf

    # ---- 3b. LP 松弛 + 列生成（定价子问题 = 背包）
    lp_bars, x_lp, y_lp = lp_primal(pf, need)   # 全部列时的 LP 值 = 根数下界
    print(f"[3b] LP 松弛 {lp_bars:.3f} 根（直接由全部 {len(pf)} 列求得）")

    def price(d):
        """定价：min 1 − Σ d_i n_i s.t. 一根母料装得下（精确 DFS，相当于 5 维背包）。"""
        best = [1.0, ()]

        def dfs(i, cnt, used, gain):
            if i == N_SPEC:
                if 1.0 - gain < best[0] - 1e-12:
                    best[0], best[1] = 1.0 - gain, tuple(cnt)
                return
            for k in range(max_count(LEN_MM[i], STOCK), -1, -1):
                nl = used + k * (LEN_MM[i] + KERF) - (KERF if k else 0.0)
                if nl <= STOCK + 1e-9:
                    dfs(i + 1, cnt + [k], nl, gain + k * d[i])

        dfs(0, [], 0.0, 0.0)
        return best[0], best[1]

    # 列生成从 6 列起步：5 种「只切一种规格」+ 1 种「一份单套料（每规格一支）」
    pool = [np.asarray(p, float) for p in simple_patterns()]
    pool[-1] = np.ones(N_SPEC, float)     # 起点最后一列：一份单套料（每规格各一支）
    cg_log = []
    for it in range(1, 31):
        lp_val, x_cg, du = lp_primal(pool, need)
        rc_, pat = price(du)
        lb_print = float(np.dot(need, du))
        print(f"      第 {it} 轮：{len(pool)} 列，LP={lp_val:.4f}，"
              f"need·y={lb_print:.4f}，"
              f"定价列 {pat}，约化成本 {rc_:.4f}")
        cg_log.append({"iter": it, "cols": len(pool), "lp": round(lp_val, 4),
                       "lb": round(float(lp_val), 9),
                       "pattern": list(pat), "reduced_cost": round(float(rc_), 4)})
        if rc_ >= -1e-9 or any(np.allclose(np.asarray(pat, float), q) for q in pool):
            break
        pool.append(np.asarray(pat, float))
    out["column_generation"] = cg_log
    cg_lp = cg_log[-1]["lp"]   # 收敛时（定价约化成本 ≥ 0）受限主问题的 LP 值 = 完整 LP 值
    out["cg_lp_bars"] = cg_lp
    print(f"[3b] 列生成：{len(cg_log)} 轮、{cg_log[0]['cols']} → {cg_log[-1]['cols']} 列，"
          f"LP {cg_lp:.4f} 根，全枚举 LP {lp_bars:.4f} 根")
    print(f"     一路最优下界 max(need·y) 只爬到 {max(c['lb'] for c in cg_log):.3f} 根"
          f"（对偶退化：同样最优的对偶解里，最差的那个只值这么多）")
    print("     对偶价格（每支规格值多少根母料）：",
          [round(float(v), 2) for v in du])
    out["duals"] = [round(float(v), 4) for v in du]

    # ---- 4. ★ 取整补种：向上取整 vs 只补缺
    ceil_bars = int(np.ceil(lp_bars - 1e-9))
    rc = ip_solve(pf, need, fixed_bars=ceil_bars)
    xc = rc.x if rc.success else None
    if xc is None:
        sc = None
        print(f"[4] 强制 {ceil_bars} 根：不可行")
    else:
        sc = stats(pf, xc, need)
        print(f"[4] 强制 {ceil_bars} 根（LP 向上取整）：可行，利用率 {sc['yield']*100:.1f}%，{sc['cost']} 元")
    out["ceil"] = sc
    out["ceil_bars"] = ceil_bars
    gap = (sf["bars"] - lp_bars) / lp_bars * 100
    out["integrality_gap_pct"] = round(gap, 2)
    print(f"    整性间隙：LP {lp_bars:.3f} → IP {sf['bars']} 根（+{gap:.2f}%）")

    # ---- 5. 母料长度扫描
    scan = []
    for s_m in [12.0, 13.0, 14.0, 15.0, 16.0]:
        s = s_m * 1000.0
        pp = gen_patterns(s, need)
        rr = ip_solve(pp, need)
        assert rr.success, rr.message
        st = stats(pp, rr.x, need, stock_mm=s)
        st["stock_m"] = s_m
        scan.append(st)
    print("[5] 母料长度扫描：", [(d["stock_m"], d["bars"], d["cost"], round(d["yield"]*100, 1)) for d in scan])
    out["stock_scan"] = scan

    # ---- 6. 短料各放宽 10 cm
    lens6 = LEN_MM + 100.0
    p6 = gen_patterns(STOCK, need, lens=lens6)
    r6 = ip_solve(p6, need)
    assert r6.success, r6.message
    x6 = r6.x
    s6 = stats(p6, x6, need, lens=lens6)
    print(f"[6] 短料各 +10 cm：{s6['bars']} 根（基准 {sf['bars']}），利用率 {s6['yield']*100:.1f}%，"
          f"料头 {s6['waste_m']:.2f} m")
    out["relax_100mm"] = s6

    # ---- 7. 瓶颈：最小单支规格决定「每根几件」 → 规格 E 加需求
    n_max_C = max_count(LEN_MM[2], STOCK)
    n_max_E = max_count(LEN_MM[4], STOCK)
    neck = {"C_per_bar_max": n_max_C, "E_per_bar_max": n_max_E,
            "C_bound_bars": int(np.ceil(need[2] / n_max_C)),
            "E_bound_bars": int(np.ceil(need[4] / n_max_E))}
    print(f"[7] 瓶颈：C 每根最多 {n_max_C} 支 → 至少 {neck['C_bound_bars']} 根；"
          f"E 每根最多 {n_max_E} 支 → 至少 {neck['E_bound_bars']} 根；IP 实得 {sf['bars']} 根")
    out["bottleneck"] = neck

    # ---------------------------------------------------------------- 出图
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))

    ax = axes[0, 0]
    lab = ["按根\n贪心", "两种\n切法", "整数规划", "LP 松弛"]
    bars_n = [sg["bars"], ss["bars"], sf["bars"], lp_bars]
    axes_col = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]
    ax.bar(lab, bars_n, color=axes_col)
    for i, v in enumerate(bars_n):
        ax.text(i, v + 0.3, f"{v:.2f}" if isinstance(v, float) else f"{v}", ha="center", fontsize=11)
    ax.set_ylim(0, max(bars_n) * 1.18)
    ax.set_ylabel("母料根数")
    ax.set_title("同一批料、三种做法：根数一样，利用率不一样")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    sm = [d["stock_m"] for d in scan]
    sc_ = [d["cost"] for d in scan]
    ax.plot(sm, sc_, "o-", color="#1f77b4", lw=2)
    for x, y, d in zip(sm, sc_, scan):
        ax.annotate(f"{y:.0f} 元 / {d['bars']} 根", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("母料长度（m）")
    ax.set_ylabel("型材费（元，按长度线性计价）")
    ax.set_title("母料加长 ≠ 省钱：12 m 与 16 m 同价，14 m 与 15 m 最贵")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    names = ["需求净长", "实得料", "真废料", "母料总长"]
    vals = [np.dot(need, LEN_MM) / 1000, sg["used_m"], sg["waste_m"], sg["bars"] * STOCK / 1000]
    ax.bar(names, vals, color=["#2ca02c", "#1f77b4", "#d62728", "#7f7f7f"])
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.2f} m", ha="center")
    ax.set_ylabel("长度（m）")
    ax.set_title(f"按根贪心 {sg['bars']} 根切开后是什么：真浪费只有 {sg['waste_m']:.1f} m")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    text = []
    for k, v in sorted(sf["patterns_used"].items(), key=lambda kv: -kv[1]):
        cnts = [int(c) for c in k.split(",")]
        txt = " + ".join(f"{n[0]}×{c}" for n, c in zip(NAMES, cnts) if c)
        text.append(f"({k}) × {v} 根   =  {txt}")
    ax.text(0.0, 1.0, "整数规划最优方案用到的切法：\n\n" + "\n".join(text),
            va="top", fontsize=10, transform=ax.transAxes)
    ax.axis("off")
    ax.set_title("每根母料怎么下（A/B/C/D/E 的支数）")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print("图已输出：", OUT_PNG)

    # ---- 断言卡教学点
    assert ss["bars"] > sg["bars"], "两种切法必须比按根贪心浪费得多（浪费藏在已切的短料里）"
    assert sf["bars"] == sg["bars"], "本实例根数两法相同（差别只在利用率，必须在正文写清）"
    assert sf["excess_m"] > 5.0, "最优方案的超产料约 5 m，必须报出来"
    assert sg["excess_m"] < 0.5, "按根贪心几乎不超产（它切到装不下为止）"
    assert max(c["lb"] for c in cg_log) > 40, "退化对偶：早期下界应远低于 LP 值（教学点）"
    assert 0 < gap < 5, "整性间隙应在 0~5%"
    assert abs(sum(sf["patterns_used"].values()) - sf["bars"]) <= 1e-9, "切法根数之和应等于总根数"
    assert sc_ is not None
    assert max(d["cost"] for d in scan) - min(d["cost"] for d in scan) > 0
    assert abs(sum(c["lb"] for c in cg_log[-1:]) - lp_bars) < 1e-6, "列生成必须与全枚举 LP 收敛到同一值"
    assert out["cg_lp_bars"] <= sf["bars"], "LP 下界不能高于 IP"
    out["runtime_s"] = round(time.time() - t0, 3)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"总耗时 {out['runtime_s']} s；json 已输出")
    print("✅ 全部断言通过")


if __name__ == "__main__":
    main()
