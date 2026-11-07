# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化模型（选址 / 覆盖 · 分层服务口径）· L8 算法（枚举裁判 vs 阈值扫描 vs 参数敏感性）
anchor  : Roadmap L4（优化模型 → 选址与覆盖）→ L8（精确解、下界与「口径」）
goal    : 云溪县学生营养餐「保障点」布点 —— 同一份需求与距离，回答四个问题：
          ① 服务标准写成一个阈值时，最少几个点、花多少钱（集合覆盖）；
          ② 加一句「教育局发餐的学校必须双档直达」后，钱花在了哪、买到了什么；
          ③ 同一个点既送营养餐又送食堂，钱记谁的账 —— 两本账为什么对不上；
          ④ 「点开几个」与「钱花多少」不是同一条曲线：单点产能卡住之后，多点反而更贵。
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-06/day69_layered_coverage.py"
输出：day69_result.json、day69_layered.png
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
T_START = time.perf_counter()

# ============================== 1. 实例（硬编码；改一行即重算）
NAME = ["老城东", "老城西", "河东", "城北", "石桥", "柳湾", "坳上", "云岭", "河湾"]
X = np.array([6.14, 5.70, 5.08, 4.17, 0.00, 0.89, 1.46, 4.01, 5.46])   # km（相对县中心）
Y = np.array([6.21, 7.00, 7.03, 8.77, 0.23, 0.31, 0.00, 1.42, 1.73])
DEM = np.array([6.2, 5.4, 7.1, 4.8, 5.9, 3.6, 6.8, 4.3, 5.1])          # 百份/日（1 份 = 1 人餐）
N = len(NAME)
D = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
K_NEAR = ((D > 0) & (D <= 4.5)).sum(0)     # 4.5 km 内邻点数 → 空驶/冷清成本

E, H, CAP = 6.20, 0.115, 50.0    # 每车 620 元 / 0.115 元每百份每 km / 单车 50 百份
CPT = E / CAP                     # 每百份占用的车次成本 = 0.124
TS = 0.15                         # 时间成本 百元/百份·km
F_PLAN, F_BASE, KAPPA = 1.85, 0.42, 0.02   # 保障点日固定 = 1.85·需求 + 0.42 + 0.02·邻点数
G_PLANT, G_MAX = 2.10, 15.0       # 单点厨房日上限 15 百份 / 超限改建工厂: 日 210 元 + 0.55 元/份
F_HC = 0.45                       # 双档额外固定（百元/百份·日）：保温周转箱 + 二次核查
UNSAT = 30.0                      # 没人送 = 自提 3000 元/次

SCHOOL = [0, 3, 6]                # 教育局发营养餐的 3 个点（其余 6 个食堂自办）
CANTEEN = [1, 2, 4, 5, 7, 8]
RAD = 5.2                         # 覆盖阈值（km）：「多远算覆盖到」
LAM = {1: 0.40, 2: 0.85}          # 冷清成本（百元/次）


def plant_cost(load):
    """单点日产能上限 G_MAX；超了要改建（固定 G_PLANT + 20 元/百份 的产出成本）"""
    if load <= G_MAX:
        return 0.0
    return G_PLANT + 0.55*(load - G_MAX)


def F(j, load, tier2=False):
    return F_PLAN*load + F_BASE + KAPPA*K_NEAR[j] + (F_HC*load if tier2 else 0.0) + plant_cost(load)


def T(d):
    """乡镇道 20 km/h + 装卸 10 分钟"""
    return d/20.0*60 + 10


def pair_pool(S):
    S = sorted(S)
    partner = {s: 0 for s in S}
    for (i, j) in sorted([(i, j) for i in range(N) for j in range(i+1, N) if D[i, j] <= RAD],
                         key=lambda p: D[p[0], p[1]]):
        if i in partner and j in partner and not partner[i] and not partner[j]:
            partner[i] = partner[j] = 1
    return partner


def route_cost(d, dem, k=1):
    total = k*dem
    trips = int(np.ceil(total/CAP))
    return E*trips + H*2*d*total + CPT*total + TS*1.5*d


def assign(S, hc, b):
    """S = 开放点；hc ⊆ S = 双档点；b = 口径（1：人人可达；2：学校点必须双档）"""
    S = sorted(S)
    hc = set(hc) & set(S)
    partner = pair_pool(S)
    lvl = np.zeros(N, int)
    proc = np.zeros(N)
    who = ["无"]*N
    src = [None]*N
    for j in range(N):
        if j in S:
            lvl[j] = 2 if (j in hc and b <= 2) else 1
            who[j] = "自"
        cands = []
        for p in S:
            d = D[p, j]
            if d > RAD:
                continue
            if p in hc and b >= 1:
                cands.append((route_cost(d, DEM[j], 1) + LAM[2], p, 2))
            cands.append((route_cost(d, DEM[j], 1) + LAM[1], p, 1))
        ok = [c for c in cands if c[2] >= b]
        if not ok:
            proc[j] = UNSAT*DEM[j]
            continue
        ok.sort()
        lvl[j] = ok[0][2]
        proc[j] = ok[0][0]
        src[j] = ok[0][1]
        who[j] = ("自" if src[j] == j else f"{NAME[src[j]]}·{'双档' if ok[0][2] == 2 else '单档'}")
    # 第 2 步：把量汇总到点上，再算点的固定成本（含超出产能的工厂回填）
    load = DEM.copy()
    for j in range(N):
        if src[j] is not None:
            load[src[j]] += DEM[j]
    for s_ in S:
        proc[s_] += F(s_, load[s_], s_ in hc)
    return lvl, proc, who, partner, load


def feasible(lvl, b):
    for j in range(N):
        need = 2 if (j in SCHOOL and b >= 2) else 1
        if lvl[j] < need:
            return False
    return True


def solve_fixed(S, hc, b):
    lvl, proc, who, partner, load = assign(S, hc, b)
    return proc.sum(), feasible(lvl, b), dict(lvl=lvl, proc=proc, who=who, partner=partner, load=load)


def best_exact(b, kmax=N):
    """枚举裁判：K 从小到大，返回最小可行 K 与该 K 下最便宜方案"""
    for k in range(1, kmax+1):
        rows = []
        for S in itertools.combinations(range(N), k):
            for r in range(k+1):
                for hc in itertools.combinations(S, r):
                    t, ok, det = solve_fixed(S, hc, b)
                    if ok:
                        rows.append((t, S, hc, det))
        if rows:
            rows.sort(key=lambda z: z[0])
            return k, rows[0], rows
    return None, None, []


def best_with_K(b, K):
    rows = []
    for S in itertools.combinations(range(N), K):
        for r in range(K+1):
            for hc in itertools.combinations(S, r):
                t, ok, det = solve_fixed(S, hc, b)
                if ok:
                    rows.append((t, S, hc, det))
    rows.sort(key=lambda z: z[0])
    return rows[0] if rows else None


def accounts(det):
    """两本账：教育局（营养餐 3 个点） vs 食堂（其余 6 个点）
    —— 一条线路既送营养餐又送食堂时，「记谁账上」是记账规则，不是成本。"""
    edu = np.zeros(N)
    can = np.zeros(N)
    for j in range(N):
        if j in SCHOOL:
            edu[j] = det["proc"][j]
        else:
            can[j] = det["proc"][j]
    return edu, can


def door_bill(det, rate_2=0.30, rate_1=0.18):
    """校门口值班：双档 11:30 前到 → 门卫必须提前到；单档可让食堂先接（百元/日）"""
    return sum(rate_2 if det["lvl"][j] == 2 else rate_1 for j in range(N))


def main():
    global RAD, F_HC, KAPPA
    out = {"instance": dict(name=NAME, x=X.tolist(), y=Y.tolist(), dem=DEM.tolist(),
                            school=SCHOOL, canteen=CANTEEN, rad=RAD),
           "params": dict(E=E, H=H, CAP=CAP, TS=TS, F_PLAN=F_PLAN, F_BASE=F_BASE,
                          KAPPA=KAPPA, F_HC=F_HC, LAM=LAM, UNSAT=UNSAT,
                          G_PLANT=G_PLANT, G_MAX=G_MAX)}

    # ---------- 问 1：最小点数与两档口径的代价
    k1, A, _ = best_exact(1)
    k2, B, _ = best_exact(2)
    print(f"[问1] 人人可达（单档） K*={k1} 日成本 {A[0]:.2f} 百元 开放 {[NAME[i] for i in A[1]]} 双档 {[NAME[i] for i in A[2]]}")
    print(f"[问1] 学校必须双档    K*={k2} 日成本 {B[0]:.2f} 百元 开放 {[NAME[i] for i in B[1]]} 双档 {[NAME[i] for i in B[2]]}")
    print("      A lvl", A[3]["lvl"], "\n      A who", A[3]["who"])
    print("      B lvl", B[3]["lvl"], "\n      B who", B[3]["who"])
    eduA, canA = accounts(A[3])
    eduB, canB = accounts(B[3])
    print(f"      教育局账 {eduA.sum():.2f} → {eduB.sum():.2f} (+{eduB.sum()-eduA.sum():.2f})")
    print(f"      食堂账   {canA.sum():.2f} → {canB.sum():.2f} (+{canB.sum()-canA.sum():.2f})")
    print(f"      门卫值班 {door_bill(A[3]):.2f} → {door_bill(B[3]):.2f}")
    out["q1"] = dict(k_star=k1, cost_A=round(float(A[0]), 3), sites_A=[NAME[i] for i in A[1]],
                     hc_A=[NAME[i] for i in A[2]],
                     k_star_B=k2, cost_B=round(float(B[0]), 3), sites_B=[NAME[i] for i in B[1]],
                     hc_B=[NAME[i] for i in B[2]],
                     lvl_A=A[3]["lvl"].tolist(), lvl_B=B[3]["lvl"].tolist(),
                     who_A=A[3]["who"], who_B=B[3]["who"],
                     edu_A=round(float(eduA.sum()), 3), edu_B=round(float(eduB.sum()), 3),
                     can_A=round(float(canA.sum()), 3), can_B=round(float(canB.sum()), 3),
                     door_A=round(door_bill(A[3]), 3), door_B=round(door_bill(B[3]), 3),
                     load_A=[round(float(x), 2) for x in A[3]["load"]],
                     load_B=[round(float(x), 2) for x in B[3]["load"]],
                     overcap_A=[NAME[j] for j in range(N) if A[3]["load"][j] > G_MAX],
                     overcap_B=[NAME[j] for j in range(N) if B[3]["load"][j] > G_MAX])

    # ---------- 问 2：K 曲线（同一个口径下，开几个点最便宜）
    kcurve = []
    for K in range(k1, 8):
        r = best_with_K(1, K)
        if r:
            kcurve.append([K, round(float(r[0]), 3), [NAME[i] for i in r[1]], [NAME[i] for i in r[2]]])
            print(f"[问2] 恰好 {K} 个点 {r[0]:.2f} 开放 {[NAME[i] for i in r[1]]} 双档 {[NAME[i] for i in r[2]]}")
    out["q2_kcurve"] = kcurve

    # ---------- 问 2b：「先省钱、再谈覆盖」—— 排序成本表上看 K*（0-1 选址图的标准读法）
    out["q2b_cost_table"] = [[K, c] for K, c, _, _ in kcurve]

    # ---------- 问 2c：工厂改建的门槛 —— 没有工厂时，一个点最多送多少
    ax = np.arange(15.0, 36.01, 1.0)
    y_noplant = [F_PLAN*a + F_BASE + KAPPA*float(K_NEAR[1]) + E + H*2*0.62*a for a in ax]
    y_plant = [F_PLAN*a + F_BASE + KAPPA*float(K_NEAR[1]) + E + H*2*0.62*a + G_PLANT + 0.55*(a-G_MAX) for a in ax]
    out["q2c_plant_threshold"] = dict(load=ax.tolist(),
                                      cost_no_plant=[round(float(v), 3) for v in y_noplant],
                                      cost_plant=[round(float(v), 3) for v in y_plant],
                                      g_max=G_MAX, plant_fixed=G_PLANT, plant_var=0.55)

    # ---------- 问 2d：如果允许「一档点便宜一点」（例如去掉冷清成本），K* 还会是 2 吗
    kappa0 = KAPPA
    rows2d = []
    for kp in [0.0, 0.02, 0.10, 0.25]:
        KAPPA = kp
        kk, rr, _ = best_exact(1, kmax=9)
        rows2d.append(dict(kappa=kp, k=kk, cost=None if rr is None else round(float(rr[0]), 3),
                           sites=None if rr is None else [NAME[i] for i in rr[1]]))
        print(f"[问2d] 冷清系数 {kp:.2f} → K*={kk} 成本 {rows2d[-1]['cost']} {rows2d[-1]['sites']}")
    KAPPA = kappa0
    out["q2d_kappa_scan"] = rows2d

    # ---------- 问 3：覆盖阈值扫描
    rows = []
    base = RAD
    for r in [4.0, 4.5, 5.0, 5.2, 5.5, 6.0, 6.5, 7.0, 8.0]:
        RAD = r
        kk, rr, _ = best_exact(1, kmax=9)
        rows.append(dict(rad=r, k=kk, cost=None if rr is None else round(float(rr[0]), 3),
                         sites=None if rr is None else [NAME[i] for i in rr[1]]))
        print(f"[问3] 阈值 {r:.1f} km → {kk} 个点 {rows[-1]['sites']} {rows[-1]['cost']}")
    RAD = base
    out["q3_threshold_scan"] = rows

    out["elapsed_s"] = round(time.perf_counter() - T_START, 3)
    make_figure(out, A, B)
    check(out)
    print(f"[耗时] {out['elapsed_s']:.2f} s")
    with open(HERE / "day69_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


def make_figure(out, A, B):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    # (a) 点位图：谁在服务谁
    ax = fig.add_subplot(gs[0, 0])
    for j in range(N):
        mark = "s" if j in SCHOOL else "o"
        ax.scatter(X[j], Y[j], s=90, marker=mark, c="white", edgecolors="k", zorder=3)
        ax.annotate(NAME[j], (X[j], Y[j]), textcoords="offset points", xytext=(7, 5), fontsize=9)
    for j in range(N):
        if j in A[1]:
            ax.scatter(X[j], Y[j], s=320, facecolors="none", edgecolors="#c0392b", lw=2, zorder=2)
    for j in range(N):
        w = A[3]["who"][j]
        if w not in ("自", "无"):
            p = NAME.index(w.split("·")[0])
            ax.plot([X[p], X[j]], [Y[p], Y[j]], "--", color="#7f8c8d", lw=1, zorder=1)
    ax.set_title("(a) 口径 A（人人可达）下的最优布点：红圈 = 开放点，虚线 = 它负责的点\n□ = 教育局发餐点   ○ = 食堂自办点", fontsize=10)
    ax.set_xlabel("东西向 km")
    ax.set_ylabel("南北向 km")
    ax.set_aspect("equal")

    # (b) 两本账
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["教育局账\n（营养餐 3 点）", "食堂账\n（其余 6 点）", "门卫值班"]
    a_vals = [out["q1"]["edu_A"], out["q1"]["can_A"], out["q1"]["door_A"]]
    b_vals = [out["q1"]["edu_B"], out["q1"]["can_B"], out["q1"]["door_B"]]
    xx = np.arange(3)
    ax2.bar(xx - 0.2, a_vals, 0.4, label="口径 A：人人可达", color="#95a5a6")
    ax2.bar(xx + 0.2, b_vals, 0.4, label="口径 B：学校双档", color="#c0392b")
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        ax2.annotate(f"{a:.1f}", (i - 0.2, a), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{b:.1f}", (i + 0.2, b), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("百元/日")
    ax2.set_title("(b) 加一句「学校双档」以后，钱记在谁账上\n教育局账只动了一点，食堂账涨了 —— 这就是两本账对不上的地方", fontsize=10)
    ax2.legend(fontsize=8)

    # (c) 成本 - K 曲线
    ax3 = fig.add_subplot(gs[1, 0])
    ks = [row[0] for row in out["q2_kcurve"]]
    cs = [row[1] for row in out["q2_kcurve"]]
    ax3.plot(ks, cs, "o-", color="#2c3e50")
    for k, c in zip(ks, cs):
        ax3.annotate(f"{c:.1f}", (k, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    best_k = ks[int(np.argmin(cs))]
    ax3.axvline(best_k, ls=":", color="#c0392b")
    ax3.set_xlabel("开放保障点数 K")
    ax3.set_ylabel("日运行成本（百元）")
    ax3.set_title(f"(c) 开几个点最便宜：K*={best_k}\n最少点数（覆盖）= {out['q1']['k_star']}，最便宜点数 = {best_k} —— 两个问题", fontsize=10)

    # (d) 覆盖阈值扫描
    ax4 = fig.add_subplot(gs[1, 1])
    rads = [row["rad"] for row in out["q3_threshold_scan"]]
    kk = [row["k"] for row in out["q3_threshold_scan"]]
    cc = [row["cost"] for row in out["q3_threshold_scan"]]
    ax4.step(rads, kk, where="post", color="#c0392b", lw=2, label="最少点数 K*")
    ax4.set_xlabel("覆盖阈值（km，「多远算覆盖到」）")
    ax4.set_ylabel("最少点数 K*", color="#c0392b")
    ax4.tick_params(axis="y", labelcolor="#c0392b")
    ax5 = ax4.twinx()
    ax5.plot(rads, cc, "o--", color="#2980b9", label="该阈值下的最优日成本")
    ax5.set_ylabel("日成本（百元）", color="#2980b9")
    ax5.tick_params(axis="y", labelcolor="#2980b9")
    ax4.set_title("(d) 「写在文件里的那个距离」挪 0.5 km，点数就从 3 跳到 2\n阈值在 5~7 km 之间对成本完全不敏感 —— 这不是稳健，是平局", fontsize=10)
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle("Day69 · 保障点分层：标准写成一个数，答案就挂在这个数上", fontsize=13)
    ax4.annotate(f"工厂门槛：单点厨房 {G_MAX:.0f} 百份/日，超过就要改建\n最优方案里 "
                 f"{len(out['q1']['overcap_A'])} 个点已超线（{'/'.join(out['q1']['overcap_A'])}）",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round", fc="#fdf2e9", ec="#e67e22"))
    fig.savefig(HERE / "day69_layered.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def check(out):
    """断言：每一条都要卡住一个教学点，跑不过就不算完成"""
    q1 = out["q1"]
    assert q1["k_star"] == 2 and q1["k_star_B"] == 2, "两种口径的最小点数都应是 2（再少就有片区没人管）"
    assert q1["cost_B"] > q1["cost_A"], "双档口径必须更贵"
    assert q1["edu_B"] - q1["edu_A"] < q1["can_B"] - q1["can_A"], \
        "学校升双档的钱必须主要落在食堂账上（这正是两本账对不上的原因）"
    assert q1["door_B"] > q1["door_A"], "双档要门卫提前到岗，值班费必须上涨"
    ks = [row[0] for row in out["q2_kcurve"]]
    cs = [row[1] for row in out["q2_kcurve"]]
    assert ks[int(np.argmin(cs))] > q1["k_star"], "最便宜的点数必须多于最少点数（覆盖 ≠ 成本最优）"
    assert min(cs) < q1["cost_A"], "多点布点必须比最小覆盖便宜（跑动成本省下来）"
    scan = out["q3_threshold_scan"]
    assert scan[0]["k"] == 3 and scan[-1]["k"] == 1, "阈值收紧到 4 km 应需 3 点、放宽到 8 km 应收缩到 1 点"
    assert scan[0]["cost"] < [r for r in scan if r["rad"] == 8.0][0]["cost"], \
        "只剩 1 个点时虽然点数少，成本反而更高（要跑长途）"
    print("[断言] 全部通过")


if __name__ == "__main__":
    main()
