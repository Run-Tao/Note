#!/usr/bin/env python
"""
subject: mathematical-modeling
level: Level 6 - Optimization / ILP
roadmap: Level 6 / 优化模型
title: 整数规划(ILP) 0-1 项目选择 + scipy.optimize.milp + 手工分支定界
date: 2026-09-05
goal:
    1) 用 0-1 整数规划建模"有限预算 + 项目选投"这一真实决策；
    2) scipy.optimize.milp(HiGHS/分支定界) 求真正可落地(整)的整数最优；
    3) 演示"LP 松弛直接四舍五入"是陷阱——会超预算(不可行)；
    4) 手工实现 0-1 背包的分支定界(Branch & Bound)看清"搜索树+剪枝"；
    5) 用 2^n 穷举作 ground truth 交叉验证。
配套教案见 04 - Optimization/Day4 笔记。
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ----------------------------------------------------------------------
# 问题背景：某公司有 100 万元年度研发预算，6 个候选研发项目。
#   每个项目要么整体立项(x_j=1)、要么不立(x_j=0)，不能"投一半"。
#   cost: 立项所需资金(万元);  ret: 预计净收益(万元)
# ----------------------------------------------------------------------
cost = np.array([55., 45., 65., 50., 70., 35.])      # 万元
ret  = np.array([110., 100., 180., 170., 150., 95.])  # 万元
BUDGET = 100.0
n = len(cost)

def solve_lp(integer):
    """解连续(LP)或整数(ILP)版本，返回 (最优目标值[收益], 决策x)。"""
    A = cost.reshape(1, -1)
    con = LinearConstraint(A, -np.inf, BUDGET)
    bds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    r = milp(c=-ret, constraints=con,          # min(-收益) <=> max 收益
             integrality=np.full(n, integer), bounds=bds)
    assert r.success, f"求解失败: {r.message}"
    return float(-r.fun), r.x                   # 还原为正的收益

# ======================================================================
# 解法 A：0-1 整数规划 (scipy.optimize.milp / HiGHS)
# ======================================================================
obj_ilp, x_lp_cont = solve_lp(integer=1)      # integrality=1 -> 整数/0-1
x_ilp = np.round(x_lp_cont).astype(int)
ilp_val = obj_ilp
sel = x_ilp.astype(bool)
chosen_ids = (np.where(sel)[0] + 1).tolist()
print("=" * 64)
print("解法A：0-1 整数规划 (scipy.optimize.milp / HiGHS 分支定界)")
print(f"选投项目: {chosen_ids}   (项目编号 1~6)")
print(f"  总成本 = {cost[sel].sum():.0f} 万元 ≤ 预算 {BUDGET:.0f}")
print(f"  整数最优总收益 = {ilp_val:.0f} 万元   ← 真正可落地的方案")

# ======================================================================
# 解法 B：先 LP 松弛(允许"投 0.77 个项目")，再四舍五入——反面教材
# ======================================================================
obj_lp, x_lp = solve_lp(integer=0)            # 连续松弛
lp_val = obj_lp
x_round = np.clip(np.round(x_lp), 0, 1).astype(int)
r_cost = (cost * x_round).sum()
print("-" * 64)
print("解法B：LP 松弛后再四舍五入（⚠️ 反面教材：直接照抄会翻车）")
print(f"  LP连续最优收益 = {lp_val:.1f} 万元  (模型里'投 0.9 个项目',现实中不行)")
print(f"  LP松弛解 x     = {np.round(x_lp,2)}")
print(f"  四舍五入 后     = 选 {np.where(x_round)[0]+1},",
      f"成本 {r_cost:.0f} 万,",
      "超预算 ❌ 不可行!" if r_cost > BUDGET else "可行但常非最优")
print(f"  → 结论: 对 LP 解直接 round 会得到**不可落地的方案**，必须先做整数规划。")

# ======================================================================
# 解法 C：穷举 2^n 组合 —— ground truth 交叉验证
# ======================================================================
best_val, best_mask = -np.inf, None
for m in range(1 << n):
    mask = np.array([(m >> j) & 1 for j in range(n)], dtype=bool)
    if cost[mask].sum() <= BUDGET and ret[mask].sum() > best_val:
        best_val, best_mask = ret[mask].sum(), mask
print("-" * 64)
print(f"解法C：穷举 {2**n} 种组合 → 最优收益 {best_val:.0f} 万，" +
      ("与 milp 一致 ✅" if abs(best_val - ilp_val) < 1e-6 else "不一致 ❌"))

# ======================================================================
# 解法 D：手写分支定界 —— 直观看到"搜索树 + 剪枝"
# ======================================================================
# 0-1 背包的分支定界：按"收益/成本"比降序，便于贪心算 LP 上界去剪枝
order = sorted(range(n), key=lambda j: -ret[j] / cost[j])
ids = [order_i for order_i in order]

best_bb, best_set_bb, node_cnt, pruned = 0.0, [], 0, 0

def fr_bound(cap, start):
    """分数背包(松弛)上界：剩下的容量按单价优先放满，用于判断能否剪枝。"""
    v = 0.0
    for j in ids[start:]:
        if cap <= 0:
            break
        take = min(1.0, cap / cost[j])
        v += take * ret[j]
        cap -= take * cost[j]
    return v

def branch(idx, cap, val, chosen):
    """在 0/1 决策树上深搜: idx 之前已决定，还剩容量 cap。"""
    global best_bb, best_set_bb, node_cnt, pruned
    node_cnt += 1
    # 剪枝：即便把剩余容量全按最高性价比放满(上界)也打不过当前最优 -> 整支砍掉
    if val + fr_bound(cap, idx) <= best_bb + 1e-9:
        pruned += 1
        return
    if idx == n:                                 # 到叶子，更新最优
        if val > best_bb:
            best_bb, best_set_bb = val, list(chosen)
        return
    j = ids[idx]
    if cost[j] <= cap:                           # 分支1：选 j（可行才分）
        branch(idx + 1, cap - cost[j], val + ret[j], chosen + [j])
    branch(idx + 1, cap, val, chosen)            # 分支2：不选 j

branch(0, BUDGET, 0.0, [])
ok_bb = abs(best_bb - best_val) < 1e-6
print("-" * 64)
print(f"解法D：手写分支定界 → 最优收益 {best_bb:.0f} 万，" +
      ("与穷举一致 ✅" if ok_bb else "不一致 ❌"))
print(f"       选 {np.array(sorted(best_set_bb))+1}；搜索树节点数 = {node_cnt}"
      f"，其中被剪枝 {pruned}（穷举要全看 {2**n} 个叶子）")

# ======================================================================
# 结论 & 断言
# ======================================================================
print("=" * 64)
# 与穷举选出集合一致
assert sel.tolist() == best_mask.tolist(), "ILP 与穷举选出的方案应一致"
# 期望数值(代码实际跑通值)
assert abs(ilp_val - 275.0) < 1e-6, f"整数最优收益应为 275 万, got {ilp_val}"
assert cost[sel].sum() <= BUDGET + 1e-9, "ILP 方案必须不超预算(可落地)"
# LP 松弛解必须是"超预算(需 round)"或是小数——演示陷阱存在
assert np.any(np.abs(x_lp - np.round(x_lp)) > 1e-6), "应存在分数解说明松弛与现实不符"
assert r_cost > BUDGET + 1e-9, "直接 round LP 应超预算(不可行), 证明必须做 ILP"
# LP 收益 >= ILP 收益(整数用去"不可分的余地"，故 LP 是上界)
assert lp_val >= ilp_val - 1e-9
assert ok_bb
print("全部断言通过 ✅")
print(f"整数间隙 = LP松弛({lp_val:.1f}) − ILP整数({ilp_val:.0f}) = {lp_val-ilp_val:.1f} 万")
print("→ 那区间的 33.5 万是'项目不可分割'的代价：松弛说能赚更多，但现实中投不了一半项目。")
