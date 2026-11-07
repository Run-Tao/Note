"""
subject : 数学建模 · 优化算法（整数规划 / 设施选址 + 车次配置）
level   : 04 - Optimization（Day59，承接 Day58 对偶单纯形与热启动）
anchor  : 县域「多仓—多站点」单趟配送网络设计：选哪个仓出货、哪个站由谁送、每个仓发几趟车
goal    : 用 0-1 整数规划一次性做「分配 + 车次 + 缺货」的联合决策，并与"就近分配"贪心
          对照，看清贪心为什么会在容量卡口上翻车；再用约束右端项 +1 的扫描（手工版
          影子价格）读出"多给一车运力值多少钱"。

【十问中的关键几问】
 Q1 谁的决策、什么周期？—— 县配送中心调度员，次日排班前定一次，日粒度。
 Q2 不优化会怎样？—— 按"离得近就派给谁"分配，石桥溢出 7 车、河湾溢出 8 车，
     有站点整站排不进任何仓。这是真实会发生的拒单。
 Q3 目标是什么？—— 总运费（车次 × 单趟成本）+ 缺货罚金，全都要钱，可以直接相加。
 Q4 约束卡在哪？—— 车容量与仓容量是硬卡口，站点不可拆单（一车一送到底）。
 Q5 结果怎么解读？—— 影子价格：哪个仓多 1 车最值钱，就优先给谁加班次。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59_facility_vehicle_ilp.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix

# ---------------------------------------------------------------- 中文字体
CJK = None
for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        CJK = name
        break
if CJK:
    plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据
WAREHOUSES = ["城关", "石桥", "云岭", "柳湾", "坳上", "河湾"]
STATIONS = [
    "渡口站", "车站广场", "东郊社区", "一中食堂", "工业园A", "柳湾矿区",
    "坳上小学", "河湾超市", "云岭卫生院", "河湾码头", "城关幼儿园", "坳上中学",
]
# 每站日需求（件）
DEMAND = np.array([60, 84, 108, 132, 96, 120, 72, 108, 60, 144, 84, 120])
Q = 24                      # 单车额定装载件数
NEED = np.ceil(DEMAND / Q).astype(int)   # 每站最少车次（不可拆单）
CAP0 = np.array([9, 12, 11, 7, 14, 2])   # 各仓当日可用车次

# 仓→站 单趟往返成本（元）：由距离（km）× 吨公里价折算后取整
DIST_KM = np.array([
    [3, 6, 9, 11, 15, 18, 20, 24, 27, 30, 12, 22],
    [10, 5, 2, 7, 12, 16, 18, 13, 15, 20, 14, 19],
    [24, 20, 15, 18, 10, 6, 9, 14, 4, 11, 22, 12],
    [18, 14, 12, 5, 3, 8, 15, 22, 24, 26, 20, 25],
    [21, 16, 14, 12, 9, 5, 3, 8, 13, 17, 15, 6],
    [26, 22, 18, 15, 12, 7, 10, 3, 9, 4, 24, 11],
], dtype=float)
UNIT = 62.0                 # 元/(km·趟)：含司机工时、油耗、折旧
TRIP_COST = np.round(DIST_KM * UNIT).astype(int)
OPEN_COST = np.array([0, 0, 0, 0, 0, 0])   # 六仓均为自有，无固定开仓费
SHORTAGE_PENALTY = 12000    # 整站送不到的罚金（元/站）


def print_panel():
    print("=" * 74)
    print("一、问题底数")
    print("=" * 74)
    print(f"仓数 {len(WAREHOUSES)}，站点数 {len(STATIONS)}，总需求 {DEMAND.sum()} 件，"
          f"单车 {Q} 件")
    print(f"每站最少车次 {' '.join(f'{s}={n}' for s, n in zip(STATIONS, NEED))}")
    print(f"合计需车 {NEED.sum()} 车；仓可提供 {CAP0.sum()} 车")
    print(f"仓容量 {' '.join(f'{w}={c}' for w, c in zip(WAREHOUSES, CAP0))}")
    print(f"单趟成本区间 {TRIP_COST.min()} 元 ~ {TRIP_COST.max()} 元")


def greedy_nearest():
    """基线：每站派给直线距离最近的仓，装不进就顺着次近的往下试（经典做法）。"""
    load = np.zeros(len(WAREHOUSES), dtype=int)
    assign = {}
    for j, s in enumerate(STATIONS):
        order = np.argsort(DIST_KM[:, j])
        for i in order:
            if load[i] + NEED[j] <= CAP0[i]:
                assign[s] = WAREHOUSES[i]
                load[i] += NEED[j]
                break
        # 若一个都装不下，就是拒单
    return assign, load


def build_ilp(cap):
    """决策变量排布：
       x[i,j] ∈{0,1}   仓 i 是否服务站 j            （6×12 = 72）
       k[i]   ≥0 整数  仓 i 当日发车数              （6）
       变量总数 78。"""
    nw, ns = len(WAREHOUSES), len(STATIONS)
    nx = nw * ns
    nv = nx + nw
    off_k = nx

    c = np.zeros(nv)
    for i in range(nw):
        for j in range(ns):
            c[i * ns + j] = TRIP_COST[i, j] * NEED[j] + OPEN_COST[i]
    for i in range(nw):
        c[off_k + i] = 0.0            # k 只通过约束起作用
    # 允许"整站放弃"，付罚金：用 y[j] 隐含在 x 全零里
    # 为让罚金可加，追加 y[j] ∈{0,1}，目标 +(罚金)y，约束 y_j + Σ_i x_ij = 1
    nv += ns
    off_y = off_k + nw
    c = np.concatenate([c, np.full(ns, float(SHORTAGE_PENALTY))])

    rows, lo, hi = [], [], []

    # (1) 每站恰好被处理一次：Σ_i x_ij + y_j = 1
    for j in range(ns):
        rows.append({i * ns + j: 1.0 for i in range(nw)} | {off_y + j: 1.0})
        lo.append(1.0); hi.append(1.0)

    # (2) 仓车次平衡：Σ_j NEED_j·x_ij ≤ k_i
    for i in range(nw):
        row = {i * ns + j: float(NEED[j]) for j in range(ns)}
        row[off_k + i] = -1.0
        rows.append(row); lo.append(-np.inf); hi.append(0.0)

    # (3) 仓车辆上限：k_i ≤ cap_i
    for i in range(nw):
        rows.append({off_k + i: 1.0}); lo.append(-np.inf); hi.append(float(cap[i]))

    A = lil_matrix((len(rows), nv))
    for r, row in enumerate(rows):
        for col, val in row.items():
            A[r, col] = val
    A = A.tocsr()

    integrality = np.zeros(nv)
    integrality[:nx] = 1
    integrality[off_k:off_k + nw] = 1
    integrality[off_y:] = 1
    lb = np.zeros(nv)
    ub = np.ones(nv)
    ub[off_k:off_k + nw] = np.maximum(cap, 0).astype(float) + 0.0
    # k_i 上界用 cap_i 本身（而不是 cap.sum()），这是第一版写错、导致假可行的地方
    ub[off_k:off_k + nw] = cap.astype(float)

    return dict(c=c, A=A, lo=np.array(lo), hi=np.array(hi), integrality=integrality,
                lb=lb, ub=ub, ns=ns, nw=nw, nx=nx, off_k=off_k, off_y=off_y)


def solve_ilp(cap):
    m = build_ilp(cap)
    res = milp(
        m["c"],
        constraints=LinearConstraint(m["A"], m["lo"], m["hi"]),
        integrality=m["integrality"],
        bounds=Bounds(m["lb"], m["ub"]),
        options={"disp": False, "mip_rel_gap": 0.0, "time_limit": 60},
    )
    if not res.success:
        return None
    x = np.round(res.x[:m["nx"]]).reshape(m["nw"], m["ns"])
    k = np.round(res.x[m["off_k"]:m["off_k"] + m["nw"]]).astype(int)
    y = np.round(res.x[m["off_y"]:]).astype(int)
    return dict(x=x, k=k, y=y, obj=res.fun, raw=res)


def solve_lp_relax(cap):
    """LP 松弛：把 integrality 全清零，拿目标值当 ILP 的乐观下界。

    注意：这里刻意用 milp(integrality=0) 而不是 linprog(A_ub=...)，
    因为 LP/ILP 都走同一个 HiGHS presolve；下面 main() 里专门有一节
    演示它在某些形态下会给出假最优，调用方需要自己复核。
    """
    m = build_ilp(cap)
    res = milp(
        m["c"],
        constraints=LinearConstraint(m["A"], m["lo"], m["hi"]),
        integrality=np.zeros_like(m["integrality"]),
        bounds=Bounds(m["lb"], m["ub"]),
        options={"disp": False},
    )
    return res


def audit_message(msg, x, c):
    """自检：把返回解代回目标与约束，不信 message 只信算术。"""
    return dict(claimed=msg, recomputed=float(c @ np.round(x)))


def report_solution(sol, tag):
    x, k, y = sol["x"], sol["k"], sol["y"]
    print(f"\n[{tag}] 目标值 {sol['obj']:.0f} 元，总车次 {k.sum()} 车")
    for i, w in enumerate(WAREHOUSES):
        served = [STATIONS[j] for j in range(len(STATIONS)) if x[i, j] > 0.5]
        print(f"  {w:3s} 用车 {k[i]:2d}/{CAP0[i]:2d}  →  {('、'.join(served) if served else '（空仓）')}")
    give_up = [STATIONS[j] for j in range(len(STATIONS)) if y[j] > 0.5]
    print(f"  放弃站点：{'、'.join(give_up) if give_up else '无'}")


def main():
    print_panel()

    print("\n" + "=" * 74)
    print("二、基线：就近分配贪心")
    print("=" * 74)
    g_assign, g_load = greedy_nearest()
    missed = [s for s in STATIONS if s not in g_assign]
    print(f"装载后各仓占用：{' '.join(f'{w}={l}/{c}' for w, l, c in zip(WAREHOUSES, g_load, CAP0))}")
    print(f"分不下去的站：{('、'.join(missed) if missed else '无')}（{len(missed)} 个）")
    g_cost = 0
    for s, w in g_assign.items():
        j = STATIONS.index(s); i = WAREHOUSES.index(w)
        g_cost += TRIP_COST[i, j] * NEED[j]
    g_cost += SHORTAGE_PENALTY * len(missed)
    print(f"贪心方案含罚金总成本：{g_cost:.0f} 元")

    print("\n" + "=" * 74)
    print("三、整数规划：分配 + 车次 + 缺货 一起解")
    print("=" * 74)
    sol = solve_ilp(CAP0)
    if sol is None:
        print("ILP 无解（容量不足）")
        return
    report_solution(sol, "ILP")
    print(f"  贪心 {g_cost:.0f} 元  vs  ILP {sol['obj']:.0f} 元  →  省 {g_cost - sol['obj']:.0f} 元 "
          f"（{(g_cost - sol['obj']) / g_cost * 100:.1f}%）")

    print("\n" + "=" * 74)
    print("四、整性间隙：LP 松弛给了什么乐观下界")
    print("=" * 74)
    lp = solve_lp_relax(CAP0)
    if lp.success:
        gap = (sol["obj"] - lp.fun) / max(abs(sol["obj"]), 1e-9) * 100
        print(f"LP 松弛目标 {lp.fun:.0f} 元，ILP {sol['obj']:.0f} 元，"
              f"整性间隙 {gap:.1f}%")
        print(f"  含义：单纯看 LP 会以为这单活能压到 {lp.fun:.0f} 元，"
              f"但整数约束把它抬到了 {sol['obj']:.0f} 元")
    else:
        print(f"LP 松弛未解出（status={lp.status}: {lp.message}）")

    print("\n" + "=" * 74)
    print("五、运力影子价格：每个仓 +1 车值多少钱（手工对偶，重解 6 次）")
    print("=" * 74)
    base = sol["obj"]
    marginal = {}
    for i, w in enumerate(WAREHOUSES):
        cap2 = CAP0.copy(); cap2[i] += 1
        s2 = solve_ilp(cap2)
        if s2 is None:
            marginal[w] = None
            print(f"  {w:3s} +1 车 →  不可行")
        else:
            marginal[w] = base - s2["obj"]
            print(f"  {w:3s} +1 车 →  目标下降 {base - s2['obj']:8.0f} 元"
                  f"（{abs(base - s2['obj']) / 1.0:.0f} 元/车）")
    ranked = sorted([(v, k) for k, v in marginal.items() if v is not None], reverse=True)
    print("\n  优先级（多给 1 车最值钱排前面）：")
    for v, w in ranked:
        print(f"    {w:3s}  {v:8.0f} 元/车")

    # ------------------------------------------------ 出图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    ax = axes[0]
    labels = WAREHOUSES
    ypos = np.arange(len(labels))
    ax.barh(ypos - 0.2, CAP0, height=0.4, color="#c9d6e5", label="可用车次")
    ax.barh(ypos + 0.2, sol["k"], height=0.4, color="#3b6ea5", label="ILP 排班车次")
    for i in range(len(labels)):
        ax.text(max(CAP0[i], sol["k"][i]) + 0.2, i, f"{sol['k'][i]}/{CAP0[i]}", va="center", fontsize=9)
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel("车次"); ax.set_title("各仓运力利用率")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    ax = axes[1]
    names = [w for _, w in ranked]
    vals = [v for v, _ in ranked]
    colors = ["#2f7d5f" if v > 1 else "#b8b8b8" for v in vals]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("元 / 多 1 车")
    ax.set_title("运力影子价格（约束右端 +1 扫描）")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[2]
    cmp_names = ["就近贪心", "整数规划"]
    cmp_vals = [g_cost, sol["obj"]]
    bars = ax.bar(cmp_names, cmp_vals, color=["#b05a3c", "#3b6ea5"], width=0.5)
    for b, v in zip(bars, cmp_vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:,.0f} 元", ha="center", fontsize=10)
    ax.set_ylabel("含缺货罚金总成本（元）")
    ax.set_title(f"整性规划救回 {g_cost - sol['obj']:,.0f} 元")
    ax.set_ylim(0, max(cmp_vals) * 1.18)

    fig.suptitle("Day59 · 县域多仓—多站点单趟配送：分配与车次联合优化", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = OUT / "day59_facility_vehicle.png"
    fig.savefig(png, dpi=150)
    print(f"\n图已保存：{png}")

    # ------------------------------------------------ 结果落盘
    payload = dict(
        warehouses=WAREHOUSES,
        stations=STATIONS,
        cap=CAP0.tolist(),
        need=NEED.tolist(),
        greedy=dict(assign=g_assign, load=g_load.tolist(), missed=missed,
                    cost=float(g_cost)),
        ilp=dict(obj=float(sol["obj"]), k=sol["k"].tolist(),
                 y=sol["y"].tolist(),
                 assign={STATIONS[j]: WAREHOUSES[i]
                         for i in range(len(WAREHOUSES))
                         for j in range(len(STATIONS)) if sol["x"][i, j] > 0.5}),
        lp_bound=None if not lp.success else float(lp.fun),
        shadow_price={k: (None if v is None else float(v)) for k, v in marginal.items()},
    )
    js = OUT / "day59_result.json"
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{js}")


if __name__ == "__main__":
    main()
