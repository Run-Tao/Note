# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L4 优化/决策模型（序贯决策 · 探索与利用）· L8 算法（UCB / Thompson / UCT / rollout）
anchor  : Roadmap L4（优化模型 → 在线序贯决策）→ 接 Day18 折扣 MDP 值迭代、Day16/56 DP
goal    : 云溪烟叶站两件事 ——
          ① 6 种烘烤工艺的达标率事先没人知道，一季 400 炉，只能一炉一炉试：先试还是先用最好的？
             （值迭代要「转移概率 P」，bandit 只要「拉了才知道」——本讲第一件事：**给「试」定价**）
          ② 4 户烟农共用 1 间烤房、每家分两批先后上炉：UCT 能给出什么，比静态规则好多少，
             以及它的**系统偏差**从哪里来（「再往下模拟一层」把 UCT 变成了不同的算法）。
运行：
  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-11-07/day70_bandits_mcts.py"
输出：day70_result.json、day70_bandit.png
"""
import json
import math
import time
import itertools
from functools import lru_cache
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
RNG = np.random.default_rng(20261107)

# ============================================================================
# 第一部分：多臂老虎机 —— 6 种烘烤工艺，达标率未知
# ============================================================================
ARM_NAME = ["传统三段式", "低温慢变黄", "先蒸后烤", "分段升温", "带杆密集烤", "电热泵烤"]
K = 6
T_BANDIT = 400
REPS = 300
P_ONE = np.array([0.82, 0.55, 0.61, 0.70, 0.38, 0.74])    # 「一个冠军」结构：亚军差 8 个点
P_TIGHT = np.array([0.82, 0.55, 0.61, 0.70, 0.74, 0.72])  # 「三个差不多」结构：差 2 个点


def run_bandit(P, policy, rng, **kw):
    n = np.zeros(K); s = np.zeros(K); pulls = []
    for t in range(T_BANDIT):
        a = policy(t, n, s, rng, **kw)
        r = 1.0 if rng.random() < P[a] else 0.0
        n[a] += 1; s[a] += r; pulls.append(a)
    pulls = np.asarray(pulls)
    return np.cumsum(P.max() - P[pulls]), n, s / np.maximum(n, 1)


def pol_greedy(t, n, s, rng, **kw):
    un = np.flatnonzero(n == 0)
    return int(un[0]) if len(un) else int(np.argmax(s / n))


def pol_eps(t, n, s, rng, eps=0.1, **kw):
    if rng.random() < eps:
        return int(rng.integers(K))
    return int(np.argmax(s / np.maximum(n, 1)))


def pol_ucb(t, n, s, rng, c=0.25, **kw):
    nn = np.maximum(n, 1)
    u = s / nn + c * np.sqrt(2.0 * math.log(t + 1.0) / nn)
    u[n == 0] = np.inf
    return int(np.argmax(u))


def pol_ts(t, n, s, rng, **kw):
    return int(np.argmax(rng.beta(s + 1.0, n - s + 1.0)))


# (名称, 函数, 参数, 用哪个真值表)
POLS = [("纯贪心(只热身一次)", pol_greedy, {}, P_ONE),
        ("ε-贪心 ε=0.1", pol_eps, {}, P_ONE),
        ("UCB c=1（教材默认）", pol_ucb, {"c": 1.0}, P_ONE),
        ("UCB c=0.25", pol_ucb, {"c": 0.25}, P_ONE),
        ("Thompson", pol_ts, {}, P_ONE),
        ("UCB c=0.25 · 三强贴身", pol_ucb, {"c": 0.25}, P_TIGHT),
        ("纯贪心 · 三强贴身", pol_greedy, {}, P_TIGHT)]

bandit, single = {}, {}
for name, fn, kw, P in POLS:
    curve = np.zeros(T_BANDIT); acc = np.zeros(K); fin = []
    for _ in range(REPS):
        r, n, _ = run_bandit(P, fn, RNG, **kw)
        curve += r; acc += n; fin.append(r[-1])
    fin = np.asarray(fin)
    bandit[name] = {"curve": curve / REPS, "pulls": acc / REPS, "P": P,
                    "mean": float(fin.mean()), "p90": float(np.percentile(fin, 90)),
                    "sd": float(fin.std())}
    r, n, mu = run_bandit(P, fn, np.random.default_rng(7), **kw)
    single[name] = {"curve": r, "pulls": n, "mu": mu}

# ============================================================================
# 第二部分：UCT 排产 —— 4 户烟农共用 1 间烤房
# ============================================================================
FARMER = ["老周", "阿兰", "德生", "小陶"]
JI = [2, 2, 2, 2]
PIECES = [[30, 22], [26, 28], [18, 34], [24, 20]]
SHED = [1.0, 1.4, 0.9, 1.2]
N_JOB = len(JI)
FIN = tuple(JI)


def evaluate(seq, t0=0.0):
    """真正评估序贯决策的唯一办法：按时间轴往前走，边看边付钱。"""
    t, done, cost = t0, [0] * N_JOB, 0.0
    for j in seq:
        cost += t * PIECES[j][done[j]]
        t += SHED[j]; done[j] += 1
    return cost


def used(st):
    return float(sum(SHED[j] * st[j] for j in range(N_JOB)))


@lru_cache(maxsize=None)
def F(st):
    """精确 DP：F(st) = 从 st 出发（当前时刻 = used(st)）的最小剩余等待成本。状态 3^4 = 81。"""
    if st == FIN:
        return 0.0
    t = used(st)
    return min(t * PIECES[j][st[j]] + F(st[:j] + (st[j] + 1,) + st[j + 1:])
               for j in range(N_JOB) if st[j] < JI[j])


def dp_plan():
    st, seq = (0,) * N_JOB, []
    while st != FIN:
        t = used(st); opt = [j for j in range(N_JOB) if st[j] < JI[j]]
        j = min(opt, key=lambda x: t * PIECES[x][st[x]] + F(st[:x] + (st[x] + 1,) + st[x + 1:]))
        seq.append(j); st = st[:j] + (st[j] + 1,) + st[j + 1:]
    return seq


def greedy_plan(rule):
    done, seq = [0] * N_JOB, []
    while any(done[j] < JI[j] for j in range(N_JOB)):
        o = [j for j in range(N_JOB) if done[j] < JI[j]]
        if rule == "short_first":
            j = min(o, key=lambda x: SHED[x])
        elif rule == "big_first":
            j = max(o, key=lambda x: (PIECES[x][done[x]], -SHED[x]))
        else:
            j = max(o, key=lambda x: (PIECES[x][done[x]] / SHED[x], -done[x]))
        seq.append(j); done[j] += 1
    return seq


class Node:
    __slots__ = ("state", "children", "N", "W", "untried")

    def __init__(self, st):
        self.state = st
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.untried = [j for j in range(N_JOB) if st[j] < JI[j]]


def rollout(st, rng, rule="big"):
    """从 st 的「下一步」开始把剩下排完（不是从 st 本身开始），返回整条路径的累计等待。

    ⚠️ 这是本讲最贵的坑：写成「从 st 开始」时，根节点自己也会被当成一条模拟，
    根 N 与子节点 N 差一位，「均值」变成「含根节点自身的滚动」。本脚本里两者已分开。
    """
    done, seq = list(st), []
    while any(done[j] < JI[j] for j in range(N_JOB)):
        o = [j for j in range(N_JOB) if done[j] < JI[j]]
        if rule == "random":
            j = int(rng.choice(o))
        else:  # 担数大、批时短先上 —— Day47 NEH 的老经验
            j = max(o, key=lambda x: (PIECES[x][done[x]], -SHED[x]))
        seq.append(j); done[j] += 1
    return evaluate(seq, used(st))


def uct_plan(n_sims, c=1.0, seed=0, rule="big", pick="visit"):
    """一次建树、逐层下棋。pick = 'visit'（访问最多，UCT 的经典读法）/ 'value'（均值最好）。"""
    rng = np.random.default_rng(seed)
    root = Node((0,) * N_JOB)
    for _ in range(n_sims):
        node, path = root, [root]
        while not node.untried and node.children:
            node = max(node.children.values(),
                       key=lambda ch: -ch.W / ch.N + c * math.sqrt(math.log(max(root.N, 1)) / ch.N))
            path.append(node)
        if node.untried:
            j = node.untried.pop(int(rng.integers(len(node.untried))))
            ch = Node(node.state[:j] + (node.state[j] + 1,) + node.state[j + 1:])
            node.children[j] = ch
            node = ch; path.append(ch)
        cost = rollout(node.state, rng, rule)
        for p in path:
            p.N += 1; p.W += -cost
    seq, node = [], root
    while node.children:
        if pick == "visit":
            j = max(node.children, key=lambda x: node.children[x].N)
        else:
            j = max(node.children, key=lambda x: -node.children[x].W / node.children[x].N)
        seq.append(j); node = node.children[j]
    return seq


def name_seq(seq):
    return " → ".join(FARMER[j] for j in seq)


def main():
    res = {}
    print("=" * 84)
    print("第一部分：6 种烘烤工艺 · bandit（K=%d，一季 T=%d 炉，%d 次重复）" % (K, T_BANDIT, REPS))
    print("「一个冠军」真值：" + "，".join("%s %.2f" % (ARM_NAME[i], P_ONE[i]) for i in range(K)))
    print("=" * 84)
    print("%-22s %11s %11s %11s" % ("策略", "平均regret", "最差10%", "冠军臂拉动"))
    for name, _, _, _ in POLS:
        b = bandit[name]
        print("%-22s %11.1f %11.1f %11.1f" % (name, b["mean"], b["p90"],
                                             b["pulls"][int(np.argmax(b["P"]))]))
        res[name] = {"regret_mean": round(b["mean"], 2), "regret_sd": round(b["sd"], 2),
                     "p90": round(b["p90"], 2),
                     "pulls_mean": [round(float(x), 1) for x in b["pulls"]],
                     "single_pulls": [int(x) for x in single[name]["pulls"]]}

    print("\n单条轨迹（种子 7，共 %d 炉）的拉动分配：" % T_BANDIT)
    print("%-22s" % "策略" + "".join("%9s" % a[:3] for a in ARM_NAME))
    for name, _, _, _ in POLS:
        print("%-22s" % name + "".join("%9d" % x for x in single[name]["pulls"]))

    print("\n" + "=" * 84)
    print("第二部分：4 户共用 1 间烤房 · 分批排产")
    print("=" * 84)
    base = [j for j in range(N_JOB) for _ in range(JI[j])]
    n_orders = len(set(itertools.permutations(base)))
    best_c = F((0,) * N_JOB)
    brute_c = min(evaluate(s) for s in set(itertools.permutations(base)))
    assert abs(brute_c - best_c) < 1e-6, "DP 与全枚举不一致"
    dseq = dp_plan()
    assert abs(evaluate(dseq) - best_c) < 1e-6, "DP 策略还原失败"
    print("全枚举 %d 种，精确最优 = %.1f 担·天（DP 81 个状态逐格核对）" % (n_orders, best_c))
    print("最优方案：" + name_seq(dseq))
    res["exact"] = {"cost": round(best_c, 3), "seq": [FARMER[j] for j in dseq], "n_orders": n_orders}

    basec = {}
    print("\n静态规则：")
    for rule in ("short_first", "big_first", "ratio"):
        s = greedy_plan(rule); v = evaluate(s); basec[rule] = v
        print("  %-11s %7.1f  (gap %+5.1f%%)  %s" % (rule, v, (v - best_c) / best_c * 100, name_seq(s)))
    res["baselines"] = {k: round(v, 3) for k, v in basec.items()}

    print("\nUCT（c=1、规则 rollout、10 个种子）：")
    uct = {}
    for pick in ("visit", "value"):
        for n_sims in (10, 30, 100, 300, 1000):
            vals = [evaluate(uct_plan(n_sims, 1.0, s, "big", pick)) for s in range(10)]
            uct[(pick, n_sims)] = vals
            print("  %-5s %5d 次 → 均值 %7.1f  最好 %7.1f  最差 %7.1f  gap %+6.1f%%"
                  % (pick, n_sims, np.mean(vals), np.min(vals), np.max(vals),
                     (np.mean(vals) - best_c) / best_c * 100))
    res["uct"] = {("%s_%d" % k): {"mean": round(float(np.mean(v)), 2), "best": round(float(np.min(v)), 2),
                                  "worst": round(float(np.max(v)), 2), "vals": [round(x, 1) for x in v]}
                  for k, v in uct.items()}

    print("\nrollout 设计对照（300 次模拟 / 10 种子 / 访问最多选枝）：")
    ro = {}
    for rule in ("big", "random"):
        vals = [evaluate(uct_plan(300, 1.0, s, rule, "visit")) for s in range(10)]
        ro[rule] = vals
        print("  %-7s 均值 %7.1f  最好 %7.1f  最差 %7.1f" % (rule, np.mean(vals), np.min(vals), np.max(vals)))
    res["rollout"] = {k: {"mean": round(float(np.mean(v)), 2)} for k, v in ro.items()}

    print("\n探索常数 c（300 次模拟 / 10 种子 / 访问最多选枝）：")
    cr = {}
    for c in (0.1, 0.3, 1.0, 2.0, 5.0):
        vals = [evaluate(uct_plan(300, c, s, "big", "visit")) for s in range(10)]
        cr[c] = {"mean": round(float(np.mean(vals)), 2), "best": round(float(np.min(vals)), 2)}
        print("  c=%.1f → 均值 %7.1f  最好 %7.1f" % (c, cr[c]["mean"], cr[c]["best"]))
    res["c_scan"] = {str(k): v for k, v in cr.items()}
    res["runtime_s"] = round(time.perf_counter() - T_START, 2)

    # ---------------------------------- 图 ----------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13.6, 9))
    show = ["纯贪心(只热身一次)", "ε-贪心 ε=0.1", "UCB c=1（教材默认）", "UCB c=0.25", "Thompson"]
    for name in show:
        ax[0, 0].plot(single[name]["curve"], label=name, lw=1.6)
    ax[0, 0].plot([1, T_BANDIT], [0, math.log(T_BANDIT)], "k--", lw=1, label="ln T（量级参考）")
    ax[0, 0].set_title("一条真实轨迹：累积 regret（好臂没被选中而亏掉的炉数）")
    ax[0, 0].set_xlabel("炉次"); ax[0, 0].set_ylabel("regret（次）")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=.3)

    w = 0.2; xs = np.arange(K)
    for i, name in enumerate(["纯贪心(只热身一次)", "ε-贪心 ε=0.1", "UCB c=1（教材默认）", "Thompson"]):
        ax[0, 1].bar(xs + (i - 1.5) * w, bandit[name]["pulls"], w, label=name)
    ax[0, 1].set_xticks(xs)
    ax[0, 1].set_xticklabels(["%s\n%.2f" % (ARM_NAME[k][:3], P_ONE[k]) for k in range(K)], fontsize=8)
    ax[0, 1].set_ylabel("平均拉动次数（%d 次重复）" % REPS)
    ax[0, 1].set_title("探索花在哪：冠军 0.82 / 亚军 0.74 / 最差 0.38")
    ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=.3, axis="y")

    for name, ls, colr in [("UCB c=0.25 · 三强贴身", "--", "#4a8c5a"),
                           ("纯贪心 · 三强贴身", "-.", "#8a6d3b")]:
        ax[1, 0].plot(np.arange(1, T_BANDIT + 1), bandit[name]["curve"], ls, color=colr, lw=1.8, label=name)
        ax[1, 0].fill_between(np.arange(1, T_BANDIT + 1),
                              bandit[name]["curve"] - bandit[name]["sd"],
                              bandit[name]["curve"] + bandit[name]["sd"], color=colr, alpha=.15)
    ax[1, 0].plot(np.arange(1, T_BANDIT + 1), bandit["UCB c=0.25"]["curve"], color="#4a6fa1", lw=2,
                  label="UCB c=0.25 · 一个冠军")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("换个结构：冠军与亚军只差 2 个点时，纯贪心（只热身一次）开始崩")
    ax[1, 0].set_xlabel("炉次（对数轴）"); ax[1, 0].set_ylabel("平均累积 regret")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    ns = [10, 30, 100, 300, 1000]
    xs2 = np.arange(len(ns))
    ax[1, 1].bar(xs2 - 0.2, [np.mean(uct[("visit", n)]) for n in ns], 0.4, color="#7aa6c2", label="访问最多选枝")
    ax[1, 1].bar(xs2 + 0.2, [np.mean(uct[("value", n)]) for n in ns], 0.4, color="#c98b4b", label="均值最好选枝")
    ax[1, 1].axhline(best_c, color="k", ls="--", lw=1.2, label="精确最优 %.1f" % best_c)
    for rule, v in basec.items():
        ax[1, 1].axhline(v, color="gray", ls=":", lw=.9)
        ax[1, 1].text(-0.45, v, rule, fontsize=7, color="gray", va="bottom")
    ax[1, 1].set_xticks(xs2); ax[1, 1].set_xticklabels([str(n) for n in ns])
    ax[1, 1].set_xlabel("每次建树的模拟次数"); ax[1, 1].set_ylabel("总加权等待（担·天）")
    ax[1, 1].set_title("UCT：模拟越多越准，但两条曲线都收敛在「规整解」上")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3, axis="y")

    fig.suptitle("Day70 · 探索与利用：bandit 的 regret，与 UCT 在带先后约束的排产上", fontsize=14)
    fig.tight_layout()
    fig.savefig(HERE / "day70_bandit.png", dpi=130)
    plt.close(fig)

    print("\n总耗时 %.2f s" % res["runtime_s"])
    with open(HERE / "day70_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    main()
