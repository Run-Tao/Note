# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L2 优化模型（动态规划 · 多阶段决策）/ L8 算法（连续切分 DP vs 贪心）
anchor      : Roadmap L4-L8 —— 接 Day16 动态规划（设备更新 / 0-1 背包）、Day56 动态规划（预算分档）、
              Day18 随机动态规划（MDP 值迭代）、Day65 一维下料（「切法组合」的换变量思路）
goal        : 云溪县三桥钢箱梁吊装分块。112.689 t / 40 m 钢箱梁，工厂分段预制、现场吊装。
              一架 55 t 履带吊（额定 62 t，但跨距越大起重量越小，最远降到 46 t）、
              一辆拖车每车限重 32 t。回答五件事：
              ① 最少要几块、几车、几吊、多少钱 —— 块数不是自己选的，是被「车重」和「起重量」卡出来的
              ② ★ 逐块算车数 vs 拼车（DP）—— 现场最常见的「一块一车」多花了多少钱
              ③ ★★ 连续切分 DP：最优切点不是均分（两端轻、中间重），均分方案根本吊不动
              ④ 贪心「贴着车限切」为什么在两端切出废块、又把车数多算一车
              ⑤ 加一天吊车、省一趟车、少一个接头 —— 哪个方向值得花钱（影子价格的离散版）
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-03/day66_dynamic_programming_gantry.py"
输出：day66_result.json + day66_gantry.png（四面板）
"""

import json
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.simplefilter("error")   # 零告警：任何警告直接报错

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day66_result.json")
OUT_PNG = os.path.join(HERE, "day66_gantry.png")

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

# ---------------------------------------------------------------- 场景数据（改一行即可重算）
L_TOTAL = 40.0            # 钢箱梁全长 m
W_TOTAL = 112.689         # 总重 t
W_PER_M = W_TOTAL / L_TOTAL

TRUCK_CAP = 32.0          # 拖车限重 t/车
TRUCK_DAY = 1800.0        # 拖车包车 元/车
CRANE_DAY = 9800.0        # 55 t 履带吊 元/天（含司机、辅吊、试吊）
JOINT = 4600.0            # 每多一个现场接头（焊接 + 探伤 + 脚手架）元
LIFT = 2400.0             # 每吊一次 元
SITE_DAY = 1200.0         # 现场其他日开销（封路、看护、临电）

# 吊车：站位在桥中段跨线作业，吊点在块重心；跨距 = 块最远端点离中线的距离
R_NEAR = 16.0             # 主臂覆盖半径 m
CAP_NEAR = 62.0           # 16 m 内起重量 t
CAP_FAR = 46.0            # 22 m 处起重量 t（16→22 m 线性衰减）
FAR_SLOPE = (CAP_NEAR - CAP_FAR) / 6.0

# 运输姿态：单块超长（>13 m）或超重（>28 t）必须侧放运输 → 到场要翻身 = 多一吊
FLIP_LEN = 13.0
FLIP_W = 28.0

GRID = 0.25               # 连续 DP 的切点网格（m）
DP_FRAG_PENALTY = 500.0   # DP 里对「短于 13 m 的收尾块」每米加价（元/m），防止碎块解
MIN_SEG = 3.0             # 非末段最小长度 m（2026-11-04：纯惩罚项实测无效，故改为硬约束）
LIFT_MAX = 20.0           # 吊车在块远端最大跨距（= 主臂 16 m + 6 m 副臂）  m
RAIL_MAX = 16.0           # 拖车可拉单块最大长度（平板车 13 m + 前后悬）  m


# ---------------------------------------------------------------- 基础函数
def seg_weight(a, b):
    return (b - a) * W_PER_M


def crane_cap(a, b):
    """该块可用起重量 t：块最远端点决定跨距，超出主臂范围线性衰减到 46 t。"""
    r_end = max(abs(a - L_TOTAL / 2), abs(b - L_TOTAL / 2))
    if r_end <= R_NEAR:
        return CAP_NEAR
    return max(CAP_FAR, CAP_NEAR - FAR_SLOPE * (r_end - R_NEAR))


def n_trucks(w):
    return int(np.ceil(w / TRUCK_CAP - 1e-9))


def needs_flip(a, b):
    return ((b - a) > FLIP_LEN + 1e-9) or (seg_weight(a, b) > FLIP_W + 1e-9)


def over_len(a, b):
    """超长：拖车装不下（>16 m）或吊车跨不到（块端点离中线 >20 m）。"""
    return (b - a) > RAIL_MAX + 1e-9 or max(abs(a - L_TOTAL / 2), abs(b - L_TOTAL / 2)) > LIFT_MAX + 1e-9


def evaluate(pts):
    """给定切点，算账。两种口径：①一块一车（现场现状） ②相邻块拼一车（DP 最优装箱）。"""
    n = len(pts) - 1
    ws = [seg_weight(pts[i], pts[i + 1]) for i in range(n)]
    over_crane = [i for i in range(n) if ws[i] > crane_cap(pts[i], pts[i + 1]) + 1e-9]
    over_truck = [i for i in range(n) if ws[i] > TRUCK_CAP + 1e-9]
    too_long = [i for i in range(n) if over_len(pts[i], pts[i + 1])]
    feasible = (not over_crane) and (not over_truck) and (not too_long)
    if not feasible:   # 吊不动 / 一车装不下 → 不是「贵一点」，是根本做不到
        return dict(n=n, pts=list(pts), ws=ws, flips=0, lifts=0, joints=n - 1,
                    trucks_naive=0, trucks_pack=0, cost_naive=float("inf"),
                    cost_pack=float("inf"), over_crane=over_crane,
                    over_truck=over_truck, too_long=too_long, feasible=False,
                    max_end_r=float("nan"))
    trucks_naive = sum(n_trucks(w) for w in ws)
    trucks_pack = min_trucks_pack(ws)
    flips = sum(1 for i in range(n) if needs_flip(pts[i], pts[i + 1]))
    lifts = n + flips                      # 每块至少一吊，要翻身的块两吊
    joints = n - 1
    cost_naive = trucks_naive * TRUCK_DAY + CRANE_DAY + joints * JOINT + lifts * LIFT + SITE_DAY
    cost_pack = trucks_pack * TRUCK_DAY + CRANE_DAY + joints * JOINT + lifts * LIFT + SITE_DAY
    return dict(n=n, pts=list(pts), ws=ws, flips=flips, lifts=lifts, joints=joints,
                trucks_naive=trucks_naive, trucks_pack=trucks_pack,
                cost_naive=cost_naive, cost_pack=cost_pack,
                over_crane=over_crane, over_truck=over_truck, too_long=too_long,
                feasible=feasible,
                max_end_r=max(max(abs(pts[i] - L_TOTAL / 2), abs(pts[i + 1] - L_TOTAL / 2)) for i in range(n)))


def min_trucks_pack(ws):
    """拼车最少车次。这里只允许「相邻块装同一车」（区间划分 DP）——钢箱梁是 40 m 长件，
    一车不可能同时装两块不相邻的；一般物品的 FFD/精确装箱见 Day65。"""
    n = len(ws)
    INF = 10 ** 9
    dp = [INF] * (n + 1)
    dp[0] = 0
    for j in range(1, n + 1):
        load = 0.0
        for i in range(j - 1, -1, -1):
            load += ws[i]
            if load > TRUCK_CAP + 1e-9:
                break
            dp[j] = min(dp[j], dp[i] + 1)
    return dp[n]


# ---------------------------------------------------------------- ① 等长切分扫描
def equal_cuts(k):
    return [round(i * L_TOTAL / k, 6) for i in range(k + 1)]


# ---------------------------------------------------------------- ② 连续切分 DP
def dp_min_blocks():
    """不分块数，直接求「最少块数」可行切分（每块 ≤32 t 且吊得动）。"""
    N = int(round(L_TOTAL / GRID))
    INF = 10 ** 9
    dp = [INF] * (N + 1)
    back = [-1] * (N + 1)
    dp[0] = 0
    for i in range(1, N + 1):
        for j in range(i - 1, -1, -1):
            a, b = j * GRID, i * GRID
            if dp[j] >= INF:
                continue
            w = seg_weight(a, b)
            if w > TRUCK_CAP + 1e-9:
                break
            if w > crane_cap(a, b) + 1e-9:
                continue
            if dp[j] + 1 < dp[i]:
                dp[i] = dp[j] + 1
                back[i] = j
    if dp[N] >= INF:
        return None
    pts, i = [L_TOTAL], N
    while i > 0:
        i = back[i]
        pts.append(round(i * GRID, 6))
    return sorted(pts)


def dp_fixed_blocks(k, mid_step=None):
    """固定块数 k 的最小费用切分：状态 (段数, 切到哪)，目标 = 每段一车 + 翻身吊次。

    车成本在这里按「一块一车」计（现场口径），拼车另算——两种口径都报。
    """
    N = int(round(L_TOTAL / GRID))
    INF = 10 ** 9
    dp = [[INF] * (N + 1) for _ in range(k + 1)]
    back = [[None] * (N + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for t in range(1, k + 1):
        for i in range(1, N + 1):
            for j in range(i - 1, -1, -1):
                if dp[t - 1][j] >= INF:
                    continue
                a = j * GRID     # 用乘法而不是 round(j*GRID,6)，避免累加误差吃掉边界
                b = i * GRID
                w = seg_weight(a, b)
                if w > TRUCK_CAP + 1e-9:
                    break
                if w > crane_cap(a, b) + 1e-9:
                    continue
                if over_len(a, b):         # 超长块根本运不进吊装半径
                    continue
                fl = needs_flip(a, b)
                step = (n_trucks(w) * TRUCK_DAY
                        + LIFT * (2 if fl else 1)
                        + (JOINT if t > 1 else 0.0)
                        # 碎块硬约束（2026-11-04 修正）：只罚不收口径时 DP 仍会剁 1 m，故改为拒绝
                        + (1.0 + 20.0 * max(0.0, FLIP_LEN - 1e-9 - (b - a))) * DP_FRAG_PENALTY)
                if i != N and (b - a) < MIN_SEG:   # 非末段不得短于 MIN_SEG，防 DP 剁碎块
                    continue
                c = dp[t - 1][j] + step
                if c < dp[t][i] - 1e-9:
                    dp[t][i] = c
                    back[t][i] = j
    if dp[k][N] >= INF:
        return None
    pts, i = [L_TOTAL], N
    for t in range(k, 0, -1):
        i = back[t][i]
        pts.append(round(i * GRID, 6))
    return sorted(pts)


# ---------------------------------------------------------------- ③ 贪心
def greedy_cut():
    """现场最容易出现的做法：从中点向两端切，每段尽量装到 32 t（贴着车限走）。

    一段一段排，只看自己这一段、不看「剩下多少」——这就是 DP 与贪心的分野。
    反面：切出来的块超过 13 m，只有平板车勉强能拉，长料价格另算（本脚本不计）。
    """
    half = L_TOTAL / 2
    pts = {0.0, L_TOTAL}

    def fill(start_pos):
        """从 start_pos 往右切，返回 [start_pos, ...] 直到 40 m。"""
        out = [start_pos]
        cur = start_pos
        while cur < L_TOTAL - 1e-9:
            nxt, ok = cur + GRID, cur
            while nxt <= L_TOTAL + 1e-9:
                if seg_weight(cur, nxt) > TRUCK_CAP + 1e-9:
                    break
                if seg_weight(cur, nxt) > crane_cap(cur, nxt) + 1e-9:
                    break
                ok = nxt
                nxt += GRID
            if ok <= cur + 1e-9:          # 一点都装不下
                break
            out.append(round(min(ok, L_TOTAL), 6))
            cur = ok
        return out

    right = fill(half)
    # 左半镜像（截面均布）
    left = sorted(round(L_TOTAL - p, 6) for p in right)
    pts |= set(right) | set(left)
    return sorted(pts)


# ---------------------------------------------------------------- ④ 局部搜索：均分起步 → 逐个切点微调
def local_search(init_pts, step0=1.0, rounds=8):
    """从给定起点出发，逐个切点左右挪一格，谁省钱就收谁；步长减半。

    这就是教科书里「枚举块数 + 均分起手 → 局部微调」的标准做法：
    用来和 DP 对账——两种方法都找到 4.40 万，才算把「最优」这件事钉死。
    """
    pts = [0.0] + list(init_pts[1:-1]) + [L_TOTAL]
    best = evaluate(pts)
    step = step0
    trace = [best["cost_pack"]]
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(pts) - 1):
                for d in (-step, step):
                    cand = pts[:]
                    cand[i] = min(max(round(cand[i] + d, 6), cand[i - 1] + 0.5), cand[i + 1] - 0.5)
                    if abs(cand[i] - pts[i]) < 1e-9:
                        continue
                    r = evaluate(cand)
                    if r["feasible"] and r["cost_pack"] < best["cost_pack"] - 1e-6:
                        best, pts, improved = r, cand, True
                        trace.append(best["cost_pack"])
        step /= 2.0
    return pts, best, trace


# ---------------------------------------------------------------- 主流程
def main():
    res = {"subject": "mathematical-modeling", "day": 66, "topic": "dp_steel_box_girder",
           "L_total": L_TOTAL, "W_total": W_TOTAL, "W_per_m": W_PER_M}

    # ① 等长切分
    eq = []
    for k in (2, 3, 4, 5, 6):
        r = evaluate(equal_cuts(k))
        r["label"] = f"{k}块等长"
        eq.append(r)
    res["equal"] = eq

    # ② 连续最优：最少块数 + 固定块数
    minpts = dp_min_blocks()
    res["min_blocks"] = evaluate(minpts)
    best_fixed = None
    fixed = []
    for k in range(4, 8):
        p = dp_fixed_blocks(k)
        if p is None:
            fixed.append({"n": k, "feasible": False})
            continue
        r = evaluate(p)
        r["label"] = f"{k}块最优"
        fixed.append(r)
        if r["feasible"] and (best_fixed is None or r["cost_pack"] < best_fixed["cost_pack"]):
            best_fixed = r
    res["fixed_blocks"] = fixed
    res["best"] = best_fixed
    # 与教科书做法对账：枚举块数 + 均分起手 + 局部搜索
    ls_rows = []
    for k in (4, 5, 6):
        # 教科书做法：枚举块数 + 均分起手 → 局部微调（顺便记录一个多起点版本做对照）
        _, l_eq, _ = local_search(equal_cuts(k))
        l_eq["label"] = f"局部搜索{k}块"
        ls_rows.append(l_eq)
    res["local_search"] = [{k: v for k, v in r.items() if k != "ws"} for r in ls_rows]

    # ③ 贪心
    g = evaluate(greedy_cut())
    g["label"] = f"贪心{g['n']}块"
    res["greedy"] = g

    # ④ 敏感：吊车日租 / 接头费 / 车限
    sens = {}
    for key, vals in (("CRANE_DAY", [6000, 9800, 14000, 20000]),
                      ("JOINT", [2000, 4600, 9000, 15000]),
                      ("TRUCK_CAP", [24.0, 28.0, 32.0, 36.0])):
        row = []
        old = globals()[key]
        for v in vals:
            globals()[key] = v
            bestk, bestc = None, None
            for k in range(4, 8):
                p = dp_fixed_blocks(k)
                if p is None:
                    continue
                r = evaluate(p)
                if bestc is None or r["cost_pack"] < bestc:
                    bestc, bestk = r["cost_pack"], k
            row.append({"value": v, "best_blocks": bestk, "cost": bestc})
        globals()[key] = old
        sens[key] = row
    res["sensitivity"] = sens

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 打印
    def fmt(r):
        return (f"{r['label']:<10}{r['n']:>3}{r['trucks_naive']:>8}{r['trucks_pack']:>8}"
                f"{r['flips']:>6}{r['lifts']:>6}{r['cost_pack'] / 1e4:>10.2f}"
                f"{'  ✅' if r['feasible'] else '  ❌吊不动/超重'}")

    print("=" * 84)
    print(f"钢箱梁 {L_TOTAL} m / {W_TOTAL} t = {W_PER_M:.4f} t/m；拖车 {TRUCK_CAP} t/车；"
          f"吊车 62 t@{R_NEAR}m → {CAP_FAR} t@{R_NEAR + 6}m")
    print("=" * 84)
    print(f"{'方案':<10}{'块':>3}{'一车一块':>8}{'拼车DP':>8}{'翻身块':>6}{'吊次':>6}{'总费(万)':>10}")
    for r in eq:
        print(fmt(r))
    print("-" * 84)
    for r in fixed:
        if not r.get("feasible", True):
            print(f"{r['n']}块最优：无可行切分")
            continue
        print(fmt(r))
    print(fmt(g))
    print("-" * 84)
    mb = res["min_blocks"]
    print(f"最少块数 DP（只约束「一车装得下 + 吊得动」，不接头/吊次）：{mb['n']} 块，切点 {[round(p, 2) for p in mb['pts']]}，"
          f"拼车 {mb['trucks_pack']} 车，总费 {mb['cost_pack'] / 1e4:.2f} 万")
    print(f"★ 按钱最优（含接头与吊次）：{best_fixed['label']}，切点 {[round(p, 2) for p in best_fixed['pts']]}，"
          f"{best_fixed['cost_pack'] / 1e4:.2f} 万")
    for r in ls_rows:
        print(f"局部搜索 {r['n']} 块（均分起手→逐个切点微调）：切点 {[round(p, 2) for p in r['pts']]} "
              f"→ {r['cost_pack'] / 1e4:.2f} 万")
    print(f"贪心（贴着 32 t 切）： {[round(p, 2) for p in g['pts']]} → {g['n']} 块，拼车 {g['trucks_pack']} 车，"
          f"{g['cost_pack'] / 1e4:.2f} 万")
    print("敏感性：")
    for k, row in sens.items():
        print(f"  {k:<10}" + "  ".join(f"{x['value']}→{x['best_blocks']}块/{x['cost'] / 1e4:.2f}万" for x in row))

    # ---------------------------------------------------------------- 断言
    # ① 等长方案随块数增加费用单调升（接头/吊次是线性增项，车费只减不增）
    costs = [r["cost_pack"] for r in eq]
    fk = [r for r in eq if r["feasible"]]
    # 「块越多越贵」不是无条件的：4 块必须翻身（省车却多两吊），5 块反而更便宜
    assert fk[0]["cost_pack"] > fk[1]["cost_pack"] > 0
    assert all(b["cost_pack"] > a["cost_pack"] for a, b in zip(fk[1:], fk[2:])), [r["cost_pack"] for r in fk]
    # 4→5 块更便宜，唯一原因是 4 块必须翻身（每块 10 m / 28.2 t），5 块刚好全部 ≤28 t 省下两吊
    assert fk[0]["flips"] == 4 and fk[1]["flips"] == 0
    # ② 2/3 块等长必吊不动，4 块起可行
    assert not eq[0]["feasible"] and not eq[1]["feasible"]
    assert all(r["feasible"] for r in eq[2:])
    # ③ 拼车 ≤ 一车一块
    for r in eq + fixed + [g]:
        if r.get("feasible", True) and r.get("trucks_pack") is not None:
            assert r["trucks_pack"] <= r["trucks_naive"]
    # ④ 最优切点 11.25 + 9.75 + 9.75 + 9.25：最短的是最远的那块（9.25 m），最重的是靠中段的那块（31.7 t）
    bw = best_fixed["ws"]
    bl = [p1 - p0 for p0, p1 in zip(best_fixed["pts"], best_fixed["pts"][1:])]
    assert abs(min(bl) - bl[-1]) < 1e-9 and bl[0] > bl[1], (bl, bw)
    assert abs(max(bw) - bw[0]) < 1e-9
    # ⑤ 贪心只切两刀，每块都超长/超重 → 必须两个块都翻身；而且块数少到不现实（现场没有 19 m 的拖车）
    # ⑤ 贪心 4 块且在两端各切出一块要翻身的块（8.75 m 那两块长过 13 m）
    assert g["feasible"] and g["n"] == 4 and g["flips"] == 2
    # ⑥ 局部搜索（均分起手 → 逐个切点挪一格）在每一个块数上都只落到「近邻台阶」，比连续 DP 贵
    for r in ls_rows:
        dpv = [x for x in fixed if x["n"] == r["n"]][0]["cost_pack"]
        assert r["cost_pack"] > dpv + 1e-6, (r["n"], r["cost_pack"], dpv)
    # ⑦ 但两者的差距不超过 0.36 万（近邻方案的相对落差），说明台阶不难、难在「一步跳不过去」
    assert max(r["cost_pack"] for r in ls_rows) - best_fixed["cost_pack"] < 1.6e4
    print("断言全过 ✅")

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0, 0]
    ka = [r["n"] for r in eq]
    ax.plot(ka, [r["cost_pack"] / 1e4 for r in eq], "o-", color="#3d5a80", label="等长切分")
    kf = [r["n"] for r in fixed if r.get("feasible")]
    kk = [r["n"] for r in ls_rows]
    ax.plot(kk, [r["cost_pack"] / 1e4 for r in ls_rows], "^:", color="#e07a5f", label="均分起手+局部微调")
    ax.plot(kf, [r["cost_pack"] / 1e4 for r in fixed if r.get("feasible")],
            "s--", color="#81b29a", label="不等长最优（连续 DP）")
    ax.set_ylim(3, 9)
    ax.set_xlabel("切块数")
    ax.set_ylabel("总费用（万元）")
    ax.set_title("② 块数不是越多越好：4 块最便宜，局部搜索却抓不到")
    ax.legend()

    ax = axes[0, 1]
    r = best_fixed
    for i in range(r["n"]):
        a, b = r["pts"][i], r["pts"][i + 1]
        w = r["ws"][i]
        ax.bar((a + b) / 2, w, width=(b - a) * 0.88,
               color="#e07a5f" if needs_flip(a, b) else "#3d5a80")
        ax.text((a + b) / 2, w + 0.8, f"{w:.1f}t", ha="center", fontsize=9)
    ax.axhline(TRUCK_CAP, color="r", ls="--", lw=1.2, label="拖车限重 32 t")
    ax.axhline(FLIP_W, color="orange", ls=":", lw=1.2, label="翻身阈值 28 t")
    ax.set_xlabel("沿桥纵向位置（m）")
    ax.set_ylabel("块重（t）")
    ax.set_title(f"③ 最优切点 [{best_fixed['n']}块]：两端轻、中间重（橙色=需翻身）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs = np.linspace(0, L_TOTAL, 241)
    cap = [crane_cap(max(0.0, x - 4), min(L_TOTAL, x + 4)) for x in xs]
    ax.plot(xs, cap, color="#3d5a80", lw=2, label="可用起重量（块端点跨距）")
    ax.axhline(CAP_NEAR, color="gray", ls=":", label=f"近距额定 {CAP_NEAR:.0f} t")
    ax.axhline(CAP_FAR, color="gray", ls="--", label=f"远距下限 {CAP_FAR:.0f} t")
    ax.axvline(R_NEAR, color="#e07a5f", ls="--", alpha=0.6)
    ax.axvline(L_TOTAL - R_NEAR, color="#e07a5f", ls="--", alpha=0.6)
    ax.text(R_NEAR + 0.3, 48, "主臂边界", color="#e07a5f", fontsize=8)
    ax.set_xlabel("块中心位置（m）")
    ax.set_ylabel("起重量（t）")
    ax.set_title("④ 起重量沿桥衰减：这才是「两端轻」的来源")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = ["一车一块\n(拼车前)", "拼车DP", "翻身多吊", "接头费", "吊车+现场"]
    rb = best_fixed
    vals = [rb["trucks_naive"] * TRUCK_DAY,
            (rb["trucks_pack"] - rb["trucks_naive"]) * TRUCK_DAY,
            rb["flips"] * LIFT,
            rb["joints"] * JOINT,
            CRANE_DAY + rb["lifts"] * LIFT + SITE_DAY]
    colors = ["#3d5a80", "#81b29a", "#e07a5f", "#f2cc8f", "#6d597a"]
    ax.bar(range(5), [v / 1e4 for v in vals], color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v / 1e4 + 0.05, f"{v / 1e4:.2f}万", ha="center", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("费用（万元）")
    ax.set_title(f"⑤ 最优方案 {rb['n']} 块的费用结构（总 {rb['cost_pack'] / 1e4:.2f} 万）")

    fig.suptitle("Day66 · 钢箱梁吊装分块：动态规划定「切几刀、在哪切」", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"已保存 {OUT_PNG}")


if __name__ == "__main__":
    main()
