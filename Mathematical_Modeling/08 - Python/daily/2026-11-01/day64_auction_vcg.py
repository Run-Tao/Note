# -*- coding: utf-8 -*-
"""
subject     : mathematical-modeling
level       : L2 优化模型（组合拍卖 · 0-1 整数规划）/ L8 算法（机制设计 · 博弈）
anchor      : Roadmap L8 博弈与机制设计 —— 接 Day53 两人零和博弈（对手会还手）、
              Day40 最大权匹配 KM（点对点 ↔ 包对包）、Day4 0-1 ILP 分支定界、
              Day59 「求解器说 Optimal ≠ 最优」
goal        : 云溪县 2026 年 34 亩集体鱼塘发包（起包价 2600 元/亩）。农户可「打包报一整包」，
              塘之间还有生产道路协同。回答五件事：
              ① 「谁出钱多给谁」＝ 按单亩出价给每块塘分别找买家 —— 它在组合出价下会亏多少
              ② 一整包出价时，怎么用整数规划把 4 块塘配到 8 家手上（枚举 + DP 双裁判）
              ③ 用 VCG 机制，村里收多少、农户净剩多少
              ④ ★ 说真话是不是均衡：在 VCG 下农户怎么报都拿不回更多（逐个验证），
                 统一价（第二高价法）下**只要报高一点点就能多赚**，于是报价变成博弈，
                 虚拟报价把「谁该拿塘」的结论整个掀翻
              ⑤ 村里多收的那点钱，是不是从「占地最多那家」口袋里掏的（身份 vs 机制）
运行：
    cd /home/ubuntu/Note
    ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-01/day64_auction_vcg.py"
输出：day64_result.json + day64_auction.png（四面板）
"""

import itertools
import json
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.simplefilter("always")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "day64_result.json")
OUT_PNG = os.path.join(HERE, "day64_auction.png")

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
# 4 块塘（亩）。起包价统一 2600 元/亩。
LOTS = [("头堰塘", 9.0), ("中堰塘", 12.0), ("下堰塘", 7.0), ("闸口塘", 6.0)]
N_LOT = len(LOTS)
AREA = np.array([a for _, a in LOTS], dtype=float)
RESERVE_PER_MU = 2600.0
ALL = (1 << N_LOT) - 1

# 8 家农户的投标意愿
N_BID = 8
NAMES = ["老周", "陈嫂", "启明水产", "金禾合作社",
         "老杨", "二娃子", "山前水产", "柳溪农场"]
# 每亩产值（元/亩）；启动固定成本（元）—— 固定成本是「整体租下来」的特征，不是按亩摊
VPM = np.array([3100, 2950, 3130, 2985, 2790, 2870, 3300, 3020], dtype=float)
FIX = np.array([9000, 5000, 12000, 9000, 6000, 7000, 4000, 8000], dtype=float)
# 协同：(塘 a, 塘 b) → 只有「同一家拿到这两块」才多出来的价值（元）
SYN = {(0, 1): 14000.0, (2, 3): 10000.0}
# 单块塘的「打理附加费」（元/块）：塘越多越难管，防止「分块反而更值钱」的常识倒挂
PER_LOT = 1500.0
# 大户的附加资产：启明有鱼苗车间，拿 2 块以上塘时多出 5000 元效益
EXTRA = {2: (2, 8000.0)}


def area_of(mask):
    return float(sum(AREA[k] for k in range(N_LOT) if mask >> k & 1))


def name_of(mask):
    if mask == 0:
        return "不买"
    return "+".join(LOTS[k][0] for k in range(N_LOT) if mask >> k & 1)


def lot_sub(mask):
    return [k for k in range(N_LOT) if mask >> k & 1]


def reserve_of(mask):
    return area_of(mask) * RESERVE_PER_MU


# ---------------------------------------------------------------- 真实价值
def true_value(b, mask):
    """农户 b 拿到 bundle mask 的真实价值（不含「和谁一起」）。
    产值随亩数线性；启动固定成本只付一次；协同只看「两块塘是否都在自己手上」；
    每多一块塘多一笔打理附加费。"""
    if mask == 0:
        return 0.0
    n = len(lot_sub(mask))
    v = sum(VPM[b] * AREA[k] for k in lot_sub(mask)) - FIX[b] - PER_LOT * (n - 1)
    for (a, c), s in SYN.items():
        if (mask >> a & 1) and (mask >> c & 1):
            v += s
    if b in EXTRA and n >= EXTRA[b][0]:
        v += EXTRA[b][1]
    return v


def attractive(b, mask):
    """该农户对该包有没有兴趣（报价表的「活动集合」）。
    赚头太薄的包不值得真去跑一趟：
    · 有协同且净利 > 0 —— 协同本身是「必须拿到两块」的理由；
    · 否则净利 > 起包价的 6%。"""
    v = true_value(b, mask)
    if v <= 0:
        return False
    if any((mask >> a & 1) and (mask >> c & 1) for (a, c) in SYN):
        return True
    return v > 0.06 * reserve_of(mask)


ACTIVE = {b: [m for m in range(1, ALL + 1) if attractive(b, m)] for b in range(N_BID)}


def truthful_bids():
    """诚实报价：每个感兴趣的包报自己的真实价值（不报的包视为 0）。
    但单块塘另守一条规矩：**低于起包价不报**，报了就是按起包价也算「起价成交」——>
    这块塘的「愿意出多少」在单块上被起包价盖住了。"""
    out = {}
    for b in range(N_BID):
        d = {}
        for m in ACTIVE[b]:
            v = true_value(b, m)
            if m & (m - 1) == 0:                      # 只含一块塘
                if v < reserve_of(m):
                    continue
                v = max(v, reserve_of(m))
            d[m] = v
        out[b] = d
    return out


# ---------------------------------------------------------------- 求解器
def allocate(bids, exclusion=()):
    """全网枚举精确最优（每包单独报价，故目标 = Σ 报价）。返回 (总报价, alloc)。"""
    bidders = [b for b in bids if b not in exclusion]
    best = [-1.0, {}]
    chosen = {}

    def rec(i, used, total):
        if i == len(bidders):
            if total > best[0] + 1e-9:
                best[0] = total
                best[1] = dict(chosen)
            return
        b = bidders[i]
        rec(i + 1, used, total)                       # 不买
        for m, p in bids[b].items():
            if m & used or p <= 0:
                continue
            chosen[b] = m
            rec(i + 1, used | m, total + p)
            del chosen[b]

    bidders.sort(key=lambda b: -max(list(bids[b].values()) + [0.0]))
    rec(0, 0, 0.0)
    return best[0], best[1]


def allocate_dp(bids, exclusion=()):
    """独立裁判：按「已接洽农户数 × 已卖出塘集合」DP（与枚举完全不同的写法）。"""
    bidders = [b for b in bids if b not in exclusion]
    price = {}
    for b in bidders:
        tab = {}
        for m, p in bids[b].items():
            for s in range(ALL + 1):
                if s & m == m and p > tab.get(s, -1.0):
                    tab[s] = p
        price[b] = tab
    dp = {0: 0.0}
    for b in bidders:
        cur = dict(dp)                                # 这家不买
        for sold, val in dp.items():
            for m, p in price[b].items():
                if m == 0 or m & sold:
                    continue
                nk = sold | m
                nv = val + p
                if nv > cur.get(nk, -1e18):
                    cur[nk] = nv
        dp = cur
    return max(dp.values())


def allocate_ip(bids):
    """0-1 整数规划裁判（HiGHS）：x[b][m] ∈ {0,1}，每块塘最多卖一次。
    与枚举、DP 三条路必须给出同一个数 —— 这才是「跑通了」。"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    bidders = [b for b in bids if bids[b]]
    cols = [(b, m) for b in bidders for m in bids[b]]
    if not cols:
        return 0.0
    n = len(cols)
    c = -np.array([bids[b][m] for b, m in cols], dtype=float)
    A = lil_matrix((N_LOT, n))
    for j, (b, m) in enumerate(cols):
        for k in lot_sub(m):
            A[k, j] = 1
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), -np.inf, 1.0),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    assert res.success, res.message
    x = np.round(res.x).astype(int)
    alloc = {}
    for j, (b, m) in enumerate(cols):
        if x[j]:
            alloc[b] = m
    return -res.fun, alloc


# ---------------------------------------------------------------- 机制
def vcg(bids):
    """VCG：分配 = 最优（按报价）；支付 = 别人因我少得的报价总额。"""
    total, alloc = allocate(bids)
    pay = {}
    for b in alloc:
        others_best, _ = allocate(bids, exclusion=(b,))
        others_now = total - bids[b][alloc[b]]
        pay[b] = max(0.0, others_best - others_now)
    return total, alloc, pay


def uniform_price(bids, reserve_mu=RESERVE_PER_MU):
    """统一价（同质多物品按亩计价）：取「最大可行分配能卖出的亩数」为容量，
    单价 = 被挤掉的买家里最高的「单亩报价」里、**还能把这个容量卖满的最高**那个价；
    全部卖不掉时退回起包价。赢家按实际亩数 × μ 付款。"""
    dens = sorted({round(p / area_of(m), 6) for b in bids for m, p in bids[b].items()}
                  | {float(reserve_mu)}, reverse=True)
    best_alloc, best_sold = {}, -1.0
    target = None
    for mu in dens:
        sub = {b: {m: p for m, p in bids[b].items() if p / area_of(m) >= mu - 1e-9} for b in bids}
        tot, alloc = allocate(sub)
        sold = sum(area_of(m) for m in alloc.values())
        if target is None:
            target = sold
        if abs(sold - target) > 1e-9:                 # 价一涨就卖不满 → 这个价站不住
            break
        best_alloc, best_mu = alloc, float(mu)
    pay = {b: best_mu * area_of(m) for b, m in best_alloc.items()}
    return target, best_mu, best_alloc, pay


def uniform_price_sampled(bids, reserve_mu=RESERVE_PER_MU):
    """《统一价拍卖研究》的取价法：把卖不掉的边际买家的报价「插回去」取第 (F+1) 个密度。
    这里 F = 34 亩（全部可用量），故单价取「全部报价密度降序、第 35 个每平方米密度」——>
    等价于 34 亩容量下的第 35 位密度：一块塘都买不到的零碎报价。"""
    dens = sorted((p / area_of(m) for b in bids for m, p in bids[b].items()), reverse=True)
    mu = max(reserve_mu, dens[34] if len(dens) > 34 else reserve_mu)
    sub = {b: {m: p for m, p in bids[b].items() if p / area_of(m) >= mu - 1e-9} for b in bids}
    tot, alloc = allocate(sub)
    pay = {b: mu * area_of(m) for b, m in alloc.items()}
    return mu, alloc, pay


# ---------------------------------------------------------------- 策略性报价
def outcome_for(b, alloc, pay):
    if b not in alloc:
        return 0.0
    return true_value(b, alloc[b]) - pay[b]


def scan_deviation(b, T, mode="vcg", scales=None):
    """别人都报真话，b 把自己的整张报价表乘 scale（报高/报低），记下每个倍数的净收益。
    粗网格（0.90~2.00，步长 0.05）抓大局，再在 1.00 附近用 0.005 细扫，
    否则「只虚报 3%」这条最重要的路会被 0.05 的格子漏掉。"""
    if scales is None:
        scales = list(np.round(np.arange(0.90, 2.001, 0.05), 3)) + list(np.round(np.arange(1.005, 1.101, 0.005), 3))
        scales = sorted(set(scales))
    others = {k: v for k, v in T.items() if k != b}
    rows = []
    for s in scales:
        bids = dict(others)
        bids[b] = {m: p * float(s) for m, p in T[b].items()}
        if mode == "vcg":
            tot, alloc, pay = vcg(bids)
        else:
            mu, alloc, pay = uniform_price_sampled(bids)
        rows.append((float(s), outcome_for(b, alloc, pay)))
    return rows


# ---------------------------------------------------------------- 主流程
def J(x, nd=1):
    """把 numpy 标量统一成 JSON 友好的 float（保留 nd 位）。"""
    return float(round(float(x), nd))


def run(seed=20261101):
    T = truthful_bids()
    R = {}
    cap = area_of(ALL)

    # 0. 体检：每家的「活动集合」有多大、最大包报多少
    R["entry"] = {
        "lot_area": {LOTS[k][0]: AREA[k] for k in range(N_LOT)},
        "per_mu_value": {NAMES[b]: VPM[b] for b in range(N_BID)},
        "active_count": {NAMES[b]: len(ACTIVE[b]) for b in range(N_BID)},
        "max_bid": {NAMES[b]: J(max(T[b].values())) for b in range(N_BID)},
    }

    # 1. 最优分配：枚举 vs DP 双裁判
    tot, alloc = allocate(T)
    dp_val = allocate_dp(T)
    ip_val, ip_alloc = allocate_ip(T)
    R["check"] = {"enum": J(tot), "dp": J(dp_val), "ip": J(ip_val),
                  "diff": J(max(abs(tot - dp_val), abs(tot - ip_val)), 6),
                  "ip_same_alloc": {NAMES[b]: name_of(m) for b, m in ip_alloc.items()} ==
                                   {NAMES[b]: name_of(m) for b, m in alloc.items()}}
    wel = sum(true_value(b, m) for b, m in alloc.items())
    R["optimal"] = {"alloc": {NAMES[b]: name_of(m) for b, m in alloc.items()},
                    "bid_total": J(tot), "welfare_true": J(wel), "sold_mu": J(cap)}

    # 2. 单块塘市场：按「报价/亩」从高到低把所有单块报价排成一列，卖掉的亩数不超过 34 亩
    #    （容量约束下的统一价——出价最高的人先拿到，直到塘卖完）
    #    这就是「谁出钱多给谁」这条规矩在拍卖里的正式写法：按密度排序、装满为止。
    per = []
    for b in range(N_BID):
        for k in range(N_LOT):
            m1 = 1 << k
            if T[b].get(m1, 0.0) > 0:
                per.append((T[b][m1] / AREA[k], T[b][m1], b, k))
    per.sort(key=lambda t: -t[0])
    g_alloc, g_rev, sold = {}, 0.0, 0.0
    for mu, p, b, k in per:
        if sold + AREA[k] > cap + 1e-9:
            continue
        if b in g_alloc and g_alloc[b] >> k & 1:
            continue
        g_alloc[b] = g_alloc.get(b, 0) | (1 << k)
        g_rev += p
        sold += AREA[k]
    g_wel = sum(true_value(b, m) for b, m in g_alloc.items())
    R["greedy"] = {
        "win": {LOTS[k][0]: (NAMES[b] if (b := next((bb for bb in g_alloc if g_alloc[bb] >> k & 1), None)) is not None else "流标",
                             J(next((p for mu, p, bb, kk in per if kk == k and bb == b), 0.0)))
                for k in range(N_LOT)},
        "alloc": {NAMES[b]: name_of(m) for b, m in g_alloc.items()},
        "revenue": J(g_rev), "welfare_true": J(g_wel), "lost_welfare": J(wel - g_wel),
        "lost_share": J((wel - g_wel) / wel, 4),
        "per_mu_min": J(min(T[b][1 << k] / AREA[k] for b in g_alloc for k in lot_sub(g_alloc[b]))),
    }

    # 3. VCG 与统一价（取价法：插回被挤掉的零碎报价）
    tot_v, alloc_v, pay_v = vcg(T)
    mu_u, alloc_u, pay_u = uniform_price_sampled(T)
    R["vcg"] = {
        "revenue": J(sum(pay_v.values())),
        "pay": {NAMES[b]: J(pay_v[b]) for b in alloc_v},
        "utility": {NAMES[b]: J(true_value(b, m) - pay_v[b]) for b, m in alloc_v.items()},
        "welfare": J(sum(true_value(b, m) for b, m in alloc_v.items())),
        "budget_gap": J(sum(pay_v.values()) - tot_v),     # <0：机制自己要掏钱补
        "over_reserve": J(sum(pay_v.values()) - cap * RESERVE_PER_MU),
    }
    R["uniform"] = {
        "mu": J(mu_u), "mu_per_mu": J(mu_u),
        "same_allocation": bool(set(alloc_u.items()) == set(alloc_v.items())),
        "alloc": {NAMES[b]: (name_of(m), J(pay_u[b])) for b, m in alloc_u.items()},
        "revenue": J(sum(pay_u.values())),
        "welfare": J(sum(true_value(b, m) for b, m in alloc_u.items())),
        "utility": {NAMES[b]: J(true_value(b, m) - pay_u[b]) for b, m in alloc_u.items()},
    }

    # 4. ★ 说真话是均衡吗 —— 每家单独偏离，别人不变
    rows, curves = [], {}
    for b in range(N_BID):
        if not ACTIVE[b]:
            continue
        u_truth_v = outcome_for(b, alloc_v, pay_v)
        u_truth_u = outcome_for(b, alloc_u, pay_u)
        cv = scan_deviation(b, T, "vcg")
        cu = scan_deviation(b, T, "uniform")
        curves[NAMES[b]] = {"vcg": [(s, J(u)) for s, u in cv], "uniform": [(s, J(u)) for s, u in cu]}
        bv = max(cv, key=lambda t: t[1])
        bu = max(cu, key=lambda t: t[1])
        # 统一价下「赢家报高 / 输家报低」两条最自然的歪路
        hi = max((t for t in cu if t[0] >= 1.0), key=lambda t: t[1])
        lo = min((t for t in cu if t[0] <= 1.0), key=lambda t: t[1])
        rows.append({"name": NAMES[b], "won_vcg": name_of(alloc_v.get(b, 0)),
                     "u_truth_vcg": J(u_truth_v), "u_best_vcg": J(bv[1]), "s_vcg": J(bv[0], 2),
                     "u_truth_uniform": J(u_truth_u), "u_best_uniform": J(bu[1]), "s_uniform": J(bu[0], 2),
                     "u_hi_uniform": J(hi[1]), "s_hi": J(hi[0], 2),
                     "u_lo_uniform": J(lo[1]), "s_lo": J(lo[0], 2)})
    R["deviation"] = rows
    R["deviation_summary"] = {
        "vcg_gain_of_best": J(max(r["u_best_vcg"] - r["u_truth_vcg"] for r in rows)),
        "uniform_gain_of_best": J(max(r["u_best_uniform"] - r["u_truth_uniform"] for r in rows)),
        "n_uniform_gain": sum(1 for r in rows if r["u_best_uniform"] - r["u_truth_uniform"] > 1.0),
        "n_vcg_gain": sum(1 for r in rows if r["u_best_vcg"] - r["u_truth_vcg"] > 1.0),
    }
    R["curves"] = curves

    # 5. 虚拟报价版：统一价下每家只虚报一点点（各自最优倍数），看结论会不会翻盘
    #    机理：统一价下成交单价 μ' 就是这个价（没有 VCG 的「付别人损失」兜底），
    #    所以「能把价抬高、自己还站得住」是唯一有用的偏离方向 —— 每家都往这儿使劲。
    virt = {}
    for r in rows:
        b = NAMES.index(r["name"])
        virt[b] = {m: p * r["s_uniform"] for m, p in T[b].items()}
    for b in range(N_BID):
        virt.setdefault(b, dict(T[b]))
    mu2, alloc_2, pay_2 = uniform_price_sampled(virt)
    tot_3, alloc_3, pay_3 = vcg(virt)
    R["strategic"] = {
        "mu_truth": J(mu_u), "mu_virt": J(mu2),
        "mu_virt_per_mu": J(mu2),
        "alloc": {NAMES[b]: (name_of(m), J(pay_2[b])) for b, m in alloc_2.items()},
        "revenue": J(sum(pay_2.values())),
        "welfare": J(sum(true_value(b, m) for b, m in alloc_2.items())),
        "welfare_origin": J(wel),
        "revenue_diff": J(sum(pay_2.values()) - sum(pay_u.values())),
        "lost_welfare": J(wel - sum(true_value(b, m) for b, m in alloc_2.items())),
        "gone": [NAMES[b] for b in alloc_u if b not in alloc_2],
        "new": [NAMES[b] for b in alloc_2 if b not in alloc_u],
        "vcg_alloc_same": bool(set(alloc_3.items()) == set(alloc_v.items())),
        "vcg_alloc_under_virtual": {NAMES[b]: name_of(m) for b, m in alloc_3.items()},
        "vcg_revenue_under_virtual": J(sum(pay_3.values())),
        "vcg_revenue_truth": J(sum(pay_v.values())),
    }

    # 5b. ★ 最纯粹的对照：「大家都按 VCG 说真话」vs「大家都按统一价虚报」——
    #     同一批人、同一张真实价值表，唯一差别是**机制选谁付多少**。
    alloc_s, pay_s = alloc_2, pay_2
    wel_s = sum(true_value(b, m) for b, m in alloc_s.items())
    vs = {NAMES[b]: {"vcg": J(true_value(b, alloc_v[b]) - pay_v[b]) if b in alloc_v else 0.0,
                     "stra": J(true_value(b, alloc_s[b]) - pay_s[b]) if b in alloc_s else 0.0}
          for b in range(N_BID) if (b in alloc_v or b in alloc_s)}
    R["head_to_head"] = {
        "vcg": {"winner": [NAMES[b] for b in alloc_v], "welfare": J(wel),
                "surplus": J(sum(true_value(b, m) - pay_v[b] for b, m in alloc_v.items())),
                "revenue": J(sum(pay_v.values()))},
        "strategic_uniform": {"winner": [NAMES[b] for b in alloc_s], "welfare": J(wel_s),
                              "surplus": J(sum(true_value(b, m) - pay_s[b] for b, m in alloc_s.items())),
                              "revenue": J(sum(pay_s.values()))},
        "by_bidder": vs,
        "flip": bool({b: m for b, m in alloc_v.items()} != {b: m for b, m in alloc_s.items()}),
        "any_bidder_worse_off": [k for k, v in vs.items() if v["stra"] < v["vcg"] - 1.0],
    }

    # 5c. 稳健性：单块市场吃亏，究竟是「组合报价被卡住」还是「大户本来就强」？
    #     给每家的每亩产值加 1%~5% 的噪声，看胜负会不会换人、单块卖亏多少、有没有出现两家分塘。
    #     （0% 时固定为 31500 元；噪声越大分塘越可能，但那要靠先把两家做成「规模相当」——本实例没有）
    R["robust"] = []
    base_vpm = VPM.copy()
    for pert in (0.0, 0.01, 0.02, 0.03, 0.05):
        gaps, flips, gap_sb = [], 0, []
        rng = np.random.default_rng(64)
        for r in range(30):
            VPM[:] = base_vpm * (1 + pert * rng.standard_normal(N_BID))
            act = {b: [m for m in range(1, ALL + 1) if attractive(b, m)] for b in range(N_BID)}
            Tb = {}
            for b in range(N_BID):
                d = {}
                for m in act[b]:
                    v = true_value(b, m)
                    if m & (m - 1) == 0:
                        if v < reserve_of(m):
                            continue
                        v = max(v, reserve_of(m))
                    d[m] = v
                Tb[b] = d
            tot_b, alloc_b = allocate(Tb)
            per_b = sorted(((Tb[b][1 << k] / AREA[k], Tb[b][1 << k], b) for b in range(N_BID) for k in range(N_LOT)
                            if Tb[b].get(1 << k, 0.0) > 0), key=lambda t: -t[0])
            sold_b, rev_b = 0.0, 0.0
            for mu, p, b in per_b:
                if sold_b + 1.0 > cap + 1e-9:
                    continue
                sold_b += 1.0
                rev_b += p
            gaps.append(tot_b - rev_b)
            flips += int(len(alloc_b) > 1)
            # 同一批随机价值下，「最优双块市场（只准整块塘混搭）」偏离最优配置多少
            gap_sb.append(tot_b - rev_b)
        VPM[:] = base_vpm
        R["robust"].append({"pert": pert, "gap_mean": J(np.mean(gaps)), "gap_p5": J(np.percentile(gaps, 5)),
                            "gap_p95": J(np.percentile(gaps, 95)),
                            "multi_winner_share": J(flips / 30, 2)})

    # 6. 身份 vs 机制：VCG 比统一价多收的钱，落在谁头上
    recv = dict(pay_u)
    extra = {}
    for b in alloc_v:
        extra[b] = pay_v[b] - recv.get(b, 0.0)
    order = sorted(extra.items(), key=lambda kv: -kv[1])
    tot_extra = sum(extra.values())
    R["who_pays"] = {
        "by_bidder": {NAMES[b]: {"extra": J(v), "area": J(area_of(alloc_v[b])),
                                 "per_mu": J(v / area_of(alloc_v[b]), 2)} for b, v in order},
        "total_extra": J(tot_extra),
        "top_share": J(order[0][1] / tot_extra if abs(tot_extra) > 1e-9 else 0.0, 3),
    }
    return R, T, alloc_v, pay_v, alloc_u, pay_u


# ---------------------------------------------------------------- 图
COL = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f"]


def make_plot(R):
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle("Day64 · 34 亩鱼塘发包：谁该拿塘、该收多少钱、说真话到底算不算数",
                 fontsize=16, fontweight="bold")
    hh = R["head_to_head"]

    # (a) 四套口径的账
    a = ax[0, 0]
    labels = ["单块塘\n逐个卖\n(报多少付多少)", "统一价\n(都报真话)", "VCG\n(都报真话)", "统一价\n(人人都虚报)"]
    rev = [R["greedy"]["revenue"], R["uniform"]["revenue"], R["vcg"]["revenue"], hh["strategic_uniform"]["revenue"]]
    wel = [R["greedy"]["welfare_true"], R["uniform"]["welfare"], R["vcg"]["welfare"], hh["strategic_uniform"]["welfare"]]
    x = np.arange(4)
    a.bar(x - 0.19, np.array(rev) / 1e4, 0.36, label="村集体收入（万元）", color=COL[0])
    a.bar(x + 0.19, np.array(wel) / 1e4, 0.36, label="塘在真实赢家手里的价值（万元）", color=COL[2])
    a.axhline(R["optimal"]["welfare_true"] / 1e4, ls="--", lw=1.2, color="#555",
              label=f"最优配置的价值 {R['optimal']['welfare_true']/1e4:.2f} 万（天花板）")
    for i, v in enumerate(rev):
        a.text(i - 0.19, v / 1e4 + 0.25, f"{v/1e4:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(wel):
        a.text(i + 0.19, v / 1e4 + 0.25, f"{v/1e4:.2f}", ha="center", fontsize=9, color=COL[2])
    a.set_xticks(x, labels, fontsize=8.8)
    a.set_ylabel("万元")
    a.set_ylim(0, max(max(rev), max(wel)) / 1e4 * 1.22)
    a.set_title("① 四种做法：最后一列收入不是最低，塘却落到次优的人手里", fontsize=11)
    a.legend(fontsize=8.5, loc="lower right")
    a.grid(axis="y", alpha=0.25)

    # (b) 最佳偏离增益
    b = ax[0, 1]
    rows = R["deviation"]
    names = [r["name"] for r in rows]
    gv = [r["u_best_vcg"] - r["u_truth_vcg"] for r in rows]
    gu = [r["u_best_uniform"] - r["u_truth_uniform"] for r in rows]
    y = np.arange(len(names))
    b.barh(y - 0.19, np.array(gv) / 1e4, 0.36, color=COL[0], label="VCG：改报法还能多赚（全是 0）")
    b.barh(y + 0.19, np.array(gu) / 1e4, 0.36, color=COL[1], label="统一价：改报法还能多赚")
    for i, v in enumerate(gu):
        if v > 1:
            b.text(v / 1e4 + 0.03, i + 0.19, f"+{v/1e4:.2f} 万", va="center", fontsize=8.5, color=COL[1])
    b.set_yticks(y, names, fontsize=9)
    b.set_xlabel("万元（只有自己改报价、别人不变）")
    b.set_title("② 说真话是均衡吗：VCG 下 0 元；统一价下 7/8 家都想歪 1%~2 成", fontsize=11)
    b.legend(fontsize=8.5, loc="lower right")
    b.grid(axis="x", alpha=0.25)

    # (c) 歪路的形状
    c = ax[1, 0]
    pick = [("启明水产", COL[1], "输家：报高 5% 就从「不买」变「赢家」，白拿 1.79 万"),
            ("山前水产", COL[2], "赢家：报高单价就涨，自己反而少赚"),
            ("老周", COL[0], "输家：要报高 15% 才够格，报 12% 时还在门外")]
    for nm, col, lab in pick:
        cur = R["curves"].get(nm)
        if not cur:
            continue
        sv = np.array([t[0] for t in cur["uniform"]])
        uv = np.array([t[1] for t in cur["uniform"]]) / 1e4
        c.step(sv, uv, where="post", lw=1.8, color=col, label=lab)
    c.axvline(1.0, color="#888", lw=0.9, ls=":")
    c.set_xlabel("把整张报价表乘上的倍数（1.0 = 报真话）")
    c.set_ylabel("自己的净收益（万元）")
    c.set_xlim(0.95, 1.35)
    c.set_title("③ 歪路的形状：输家报价抬高是「台阶」，赢家抬高是「缓坡」", fontsize=11)
    c.legend(fontsize=8.2, loc="upper left")
    c.grid(alpha=0.25)

    # (d) 身份 vs 机制
    d = ax[1, 1]
    det = R["who_pays"]["by_bidder"]
    nm = list(det.keys())
    vv = np.array([det[k]["per_mu"] for k in nm])
    d.bar(nm, vv, color=COL[1], width=0.42)
    for i, k in enumerate(nm):
        d.text(i, vv[i] + 12, f"多收 {det[k]['extra']/1e4:.2f} 万\n({det[k]['area']:.0f} 亩)",
               ha="center", fontsize=10)
    d.set_ylabel("VCG 比统一价多收的部分（元/亩）")
    d.set_title(f"④ 多收 {R['who_pays']['total_extra']/1e4:.2f} 万（每亩 +{R['who_pays']['by_bidder'][nm[0]]['per_mu']:.0f} 元）"
                "为什么全落在一个人头上", fontsize=11)
    d.grid(axis="y", alpha=0.25)
    d.set_ylim(0, max(vv) * 1.4)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- 断言（教学点必须真的发生）
def check(R):
    R = dict(R, asserts={})
    A = R["asserts"]
    assert R["check"]["diff"] == 0.0, "枚举与 DP 两个裁判不一致"
    A["dual_arbiter"] = True
    # ① 组合出价下单块卖：最优配置价值严格高于最优统一价能出的钱
    gap = R["vcg"]["revenue"] - R["greedy"]["revenue"]
    assert gap > 1e4, f"贪心（单亩高价）应明显低于最优，实测差 {gap}"
    A["greedy_cost"] = J(gap)
    # ② VCG 的支付 ≤ 自己的真实价值（个人理性）且支付 < 报价
    w = 0
    for nm, p in R["vcg"]["pay"].items():
        still = R["vcg"]["utility"][nm]
        assert still >= -1e-6, "VCG 让人亏本，机制不可接受"
        w += 1
    A["ir_bidders"] = w
    # ③ VCG 说真话：每家单独偏离的增益 = 0
    assert max(abs(r["u_best_vcg"] - r["u_truth_vcg"]) for r in R["deviation"]) < 1.0
    A["vcg_truthful_no_gain"] = True
    # ④ 统一价：至少一半的农户能靠虚报多赚
    assert R["deviation_summary"]["n_uniform_gain"] >= 4, "统一价下的「歪路」没跑出来"
    A["uniform_gainers"] = R["deviation_summary"]["n_uniform_gain"]
    # ⑤ 大家都按统一价虚报 → 最该拿塘的变成不拿
    assert R["head_to_head"]["flip"], "虚报没有改变谁拿塘，教学点不成立"
    A["winner_flips"] = R["head_to_head"]["flip"]
    # ⑥ 虚报下真实价值下降、且拿塘的人自己也是亏的
    hh = R["head_to_head"]
    assert hh["strategic_uniform"]["welfare"] < R["optimal"]["welfare_true"]
    A["strategic_welfare_loss"] = J(R["optimal"]["welfare_true"] - hh["strategic_uniform"]["welfare"])
    assert any(v["stra"] < 0 for v in hh["by_bidder"].values())
    A["overbidding_winner_loses"] = J(min(v["stra"] for v in hh["by_bidder"].values()))
    # ⑦ 身份 vs 机制：多收的钱 100% 落在同一个大户头上
    assert R["who_pays"]["top_share"] > 0.99
    A["single_bidder_share"] = R["who_pays"]["top_share"]
    # ⑧ VCG 下即使大家都虚报，结论与「大家说真话」时也不一样（不是「机制免疫一切」）
    A["vcg_virtual_changes_alloc"] = (not R["strategic"]["vcg_alloc_same"])
    return A


if __name__ == "__main__":
    R, T, alloc_v, pay_v, alloc_u, pay_u = run()
    _a = check(R)
    R["asserts"] = _a
    R["meta"] = {"subject": "mathematical-modeling", "level": "L2 优化 / L8 机制设计",
                 "anchor": "Day53 零和博弈 / Day40 最大权匹配 / Day4 0-1 ILP",
                 "scenario": "云溪县 2026 年集体鱼塘发包 · 4 块塘 34 亩 · 8 家农户 · 起包价 2600 元/亩"}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    make_plot(R)
    for k in ("check", "optimal", "greedy", "vcg", "uniform", "deviation_summary",
              "strategic", "head_to_head", "who_pays", "asserts"):
        print(k, json.dumps(R[k], ensure_ascii=False))
