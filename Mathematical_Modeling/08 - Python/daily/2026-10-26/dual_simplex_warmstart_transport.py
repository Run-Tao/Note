# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L6 优化模型 / L8 算法（线性规划 · 单纯形 · 对偶单纯形 · 热启动）
anchor  : Day26 对偶与影子价格 + Day52 灵敏度分析（最优基 = 解的有效期）
goal    : 县域果蔬集散中心「三基地 → 五片区」周度调运。线路没变，数据每周在变。
          回答四件事：① 上周那张最优调运表，这周还能不能用？
                     ② 不能用的话，要改几笔？
                     ③ 影子价格（哪个基地最金贵）什么时候过期？
                     ④ 周五临时加一条约束，几次迭代就能给它定价？

模型口径：
  * x[i,j]：基地 i 调往片区 j 的运量（百吨/周），成本 c[i,j] 元/吨
      基地  城关主产区 6.0 / 石桥次产区 6.6 / 柳湾新建仓 3.0（百吨/周可供）
      片区  甲城区 2.8 / 乙东片 3.5 / 丙西片 4.2 / 丁南岭 3.2 / 戊高原 1.9（百吨/周需求下限）
    第一周产销平衡（15.6 = 15.6）
  * 基地约束 ≤（运不完留仓，加松弛 s）；片区约束 ≥（缺口得外调，减剩余 t）
    8 个约束 = 8 个基变量
  * 方案 A 冷启动：每个新数据都从零重跑（Phase-I 人工变量 + 原始单纯形）
  * 方案 B 热启动：留住上周的最优基，只换右端 xB ← B⁻¹b，用对偶单纯形恢复可行性。
    表底那一行检验数一动不动 —— 因为在位基的影子价格没变。这是全课的核心。

依赖：numpy matplotlib（仓库 .venv 已装）
运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-26/dual_simplex_warmstart_transport.py"
输出：dual_simplex_warmstart_transport.png / .json（与本脚本同目录）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "dual_simplex_warmstart_transport.png"
OUT_JSON = HERE / "dual_simplex_warmstart_transport.json"

BASE_NAMES = ["城关主产区", "石桥次产区", "柳湾新建仓"]
BASE_SHORT = ["城关", "石桥", "柳湾"]
AREA_NAMES = ["甲 城区", "乙 东片", "丙 西片", "丁 南岭", "戊 高原"]
AREA_SHORT = ["甲", "乙", "丙", "丁", "戊"]

COST0 = np.array([
    [3.2, 4.6, 5.1, 6.8, 7.4],
    [5.0, 3.4, 3.9, 5.2, 6.1],
    [6.3, 5.4, 4.1, 3.6, 4.9],
])
S0 = np.array([6.0, 6.6, 3.0])
D0 = np.array([2.8, 3.5, 4.2, 3.2, 1.9])

NB, NA = COST0.shape
NV = NB * NA                  # 决策变量 15（x[i,j] = i*NA+j）
NR = NB + NA                  # 约束 8
IDX_S = NV                    # 15..17 基地松弛 s / 18..22 片区剩余 t
IDX_ART = NV + NR             # 23..30 人工 a
NCOL = NV + 2 * NR
TOL = 1e-9
BIG = 1e9                     # 人工列检验数的"封锁值"，保证第二阶段永不再入基


# ============================================================
# 一、模型矩阵与标签
# ============================================================
def build_A():
    """不含人工列的 A：基地行 ≤、片区行 ≥。"""
    A = np.zeros((NR, NCOL))
    for i in range(NB):
        for j in range(NA):
            A[i, i * NA + j] = 1.0
            A[NB + j, i * NA + j] = 1.0
    for i in range(NB):
        A[i, IDX_S + i] = 1.0                 # 松弛：留仓量
    for j in range(NA):
        A[NB + j, IDX_S + NB + j] = -1.0      # 剩余：缺口
    return A


def build_Abar(A):
    """含人工列的矩阵：只在片区行是 +1。"""
    Abar = A.copy()
    for j in range(NA):
        Abar[NB + j, IDX_ART + j] = 1.0
    return Abar


A_MAT = build_A()
ABAR = build_Abar(A_MAT)
GEO_CLEAN = A_MAT.copy()                      # 第二阶段用（人工列恒 0）
GEO_CLEAN[:, IDX_ART:] = 0.0


def cv(cost):
    """目标系数：决策用 c，松弛/剩余/人工都是 0。"""
    return np.concatenate([np.asarray(cost).ravel(), np.zeros(2 * NR)])


def label(j):
    j = int(j)
    if j < NV:
        return f"x({BASE_SHORT[j // NA]},{AREA_SHORT[j % NA]})"
    if j < IDX_ART:
        k = j - IDX_S
        return f"s({BASE_SHORT[k]})" if k < NB else f"t({AREA_SHORT[k - NB]})"
    return f"a({AREA_SHORT[j - IDX_ART]})"


def basis_matrix(basis, mat=None):
    return (A_MAT if mat is None else mat)[:, [int(k) for k in basis]]


def price(basis, cvec):
    """在位基的影子价格 y = c_B B⁻¹。B 用含人工列的矩阵取，才和表里的基对应。"""
    return np.asarray(cvec)[[int(k) for k in basis]] @ np.linalg.inv(basis_matrix(basis, ABAR))


# ============================================================
# 二、两个主元规则
# ============================================================
def pivot(T, basis, i, j):
    T[i] = T[i] / T[i, j]
    col = T[:, j].copy()
    col[i] = 0.0
    T -= np.outer(col, T[i])
    T[np.abs(T) < 1e-10] = 0.0
    basis[i] = int(j)


def primal(T, basis, m, rhs=-1):
    """原始单纯形：右端已 ≥0，消掉负检验数。表底行存检验数。"""
    it = 0
    while True:
        neg = np.where(T[m, :IDX_ART] < -TOL)[0]
        if not len(neg):
            return it
        j = int(neg[np.argmin(T[m, neg])])
        col = T[:m, j]
        ok = col > TOL
        if not ok.any():
            raise RuntimeError(f"原始问题无界（{label(j)} 可以无限增加）")
        ratio = np.where(ok, T[:m, rhs] / np.where(ok, col, 1e300), np.inf)
        i = int(np.argmin(ratio))
        pivot(T, basis, i, j)
        it += 1


def dual_simplex(T, basis, m, rhs=-1, nvar=IDX_ART):
    """对偶单纯形：检验数已 ≥0，只修负右端。
    最负右端行出基，入基列按 min{检验数/|该行系数|} 选（只比系数 < 0 的列）。"""
    it = 0
    while True:
        neg = np.where(T[:m, rhs] < -TOL)[0]
        if not len(neg):
            return it
        i = int(neg[np.argmin(T[neg, rhs])])
        row = T[i, :nvar].copy()
        ok = row < -TOL
        if not ok.any():
            raise RuntimeError("约束互相矛盾：原始不可行、对偶无界")
        ratio = np.where(ok, T[m, :nvar] / np.abs(np.where(ok, row, -1.0)), np.inf)
        j = int(np.argmin(ratio))
        pivot(T, basis, i, j)
        it += 1


# ============================================================
# 三、方案 A：冷启动（Phase-I 人工变量 + Phase-II 原始单纯形）
# ============================================================
def cold(b, cost):
    """从零算一遍。返回 (表, 基, 主元次数, 目标值)。b = [可供量..., 需求下限...]。"""
    cvec = cv(cost)
    T = np.zeros((NR + 1, NCOL + 1))
    T[:NR, :NCOL] = ABAR
    T[:NR, -1] = b

    # ---- Phase-I：min Σ 人工。初始基 = 基地松弛(3) + 片区人工(5)，正好是单位阵
    basis = [IDX_S + i for i in range(NB)] + [IDX_ART + j for j in range(NA)]
    w = np.zeros(NCOL)
    w[IDX_ART:] = 1.0
    # 目标行 = w − y·Ā，初始 y = w_B = (0,0,0,1,1,1,1,1)：基内人工列恰好消成 0
    T[NR, :NCOL] = w - np.asarray(w)[basis] @ ABAR
    it1 = primal(T, basis, NR)

    xart = sum(max(T[i, -1], 0.0) for i, k in enumerate(basis) if k >= IDX_ART)
    if xart > 1e-7:
        raise RuntimeError(f"模型不可行：人工变量还留着 {xart:.3f}（需求总计超过可供总计）")

    # ---- 基里若还留着 0 值人工变量，纯换基把它请出去（基变、解不变）
    for i in range(NR):
        if basis[i] >= IDX_ART:
            cand = np.where(np.abs(T[i, :IDX_ART]) > 1e-9)[0]
            if not len(cand):
                raise RuntimeError(f"第 {i} 行约束冗余（人工行全零）")
            pivot(T, basis, i, int(cand[0]))
            it1 += 1

    # ---- Phase-II：换回真实成本。人工已全部出基，用干净矩阵重算检验数
    y = price(basis, cvec)
    T[NR, :NCOL] = cvec - y @ GEO_CLEAN
    T[NR, IDX_ART:] = BIG                     # 人工列封锁，永不再入基
    it2 = primal(T, basis, NR)

    x = read_x(T, basis)
    z = obj_of(x, cost)
    T[NR, -1] = -z
    return T, basis, it1 + it2, z


# ============================================================
# 四、方案 B：热启动
# ============================================================
def warm_b(T_prev, basis_prev, b_prev, b_new, cost):
    """右端变了：xB ← xB + B⁻¹Δb，检验数一行不动，交给对偶单纯形收尾。"""
    T = T_prev.copy()
    Binv = np.linalg.inv(basis_matrix(basis_prev, ABAR))
    T[:NR, -1] = T_prev[:NR, -1] + Binv @ (b_new - b_prev)
    basis = [int(k) for k in basis_prev]
    it = dual_simplex(T, basis, NR)
    T[NR, -1] = -obj_of(read_x(T, basis), cost)
    return T, basis, it


def warm_c(T_prev, basis_prev, cost):
    """成本变了：基不动，只重算检验数（y 跟着 c_B 变），再用原始单纯形收尾。"""
    cvec = cv(cost)
    T = T_prev.copy()
    basis = [int(k) for k in basis_prev]
    y = price(basis, cvec)
    T[NR, :NCOL] = cvec - y @ GEO_CLEAN
    T[NR, IDX_ART:] = BIG
    it = primal(T, basis, NR)
    T[NR, -1] = -obj_of(read_x(T, basis), cost)
    return T, basis, it


def read_x(T, basis, rhs=-1):
    x = np.zeros(NCOL)
    for i, k in enumerate(basis):
        if k < NCOL:
            x[k] = max(T[i, rhs], 0.0)
    return x


def obj_of(x, cost):
    return float(np.asarray(cost).ravel() @ x[:NV])


def n_diff(xa, xb, tol=1e-6):
    A = xa[:NV].reshape(NB, NA)
    B = xb[:NV].reshape(NB, NA)
    return int((np.abs(A - B) > tol).sum())


def changed_cells(xa, xb, tol=1e-6):
    A, B = xa[:NV].reshape(NB, NA), xb[:NV].reshape(NB, NA)
    return [(BASE_SHORT[i], AREA_SHORT[j], round(float(B[i, j] - A[i, j]), 2))
            for i in range(NB) for j in range(NA) if abs(A[i, j] - B[i, j]) > tol]


# ============================================================
# 五、五周剧本：数据在变，方案不变
# ============================================================
def weeks_plan():
    plan = []
    plan.append(("第1周 基准", S0.copy(), D0.copy(), COST0.copy(), "产销平衡 15.6 = 15.6"))
    plan.append(("第2周 柳湾仓扩租", np.array([6.0, 6.6, 3.5]), D0.copy(), COST0.copy(),
                 "新建仓临时多租 0.5 百吨"))
    plan.append(("第3周 城关丰产", np.array([8.0, 6.6, 3.0]), np.r_[D0[:4], 3.3], COST0.copy(),
                 "城关 +2.0，戊高原冬储 +1.4"))
    plan.append(("第4周 乙东片加码", np.array([6.0, 7.3, 3.0]), np.r_[D0[0], 4.2, D0[2:]], COST0.copy(),
                 "乙东片 +0.7，石桥 +0.7 补上"))
    c5 = COST0.copy()
    c5[0, 2] -= 1.5
    plan.append(("第5周 绕道费下调", S0.copy(), D0.copy(), c5, "城关→丙西片 运费降 1.5 元/吨"))
    return plan


def solve_all():
    rows = []
    T_w = B_w = b_prev = c_prev = None
    for tag, S, D, C, note in weeks_plan():
        b = np.r_[S, D]
        t0 = time.perf_counter()
        T_c, B_c, it_c, z_c = cold(b, C)
        t_cold = (time.perf_counter() - t0) * 1e3
        B_c = [int(k) for k in B_c]
        x_c = read_x(T_c, B_c)

        if T_w is None:
            T_w, B_w, it_w, z_w = T_c, B_c, it_c, z_c
            t_warm = t_cold
            detail = "冷启动（第一周，无历史可借）"
        else:
            n_c = n_b = 0
            detail = []
            if not np.allclose(C, c_prev):
                T_w, B_w, n_c = warm_c(T_w, B_w, C)
                detail.append(f"成本变：重算检验数 {n_c} 次主元")
            t0 = time.perf_counter()
            T_w, B_w, n_b = warm_b(T_w, B_w, b_prev, b, C)
            t_warm = (time.perf_counter() - t0) * 1e3
            it_w = n_c + n_b
            z_w = obj_of(read_x(T_w, B_w), C)
            detail.append(f"右端变：对偶单纯形 {n_b} 次主元")
            detail = "；".join(detail)

        x_w = read_x(T_w, B_w)
        rows.append(dict(
            week=tag, note=note,
            supply=[round(float(v), 2) for v in S],
            demand=[round(float(v), 2) for v in D],
            z_cold=round(z_c, 4), z_warm=round(z_w, 4),
            it_cold=it_c, it_warm=it_w, t_cold_ms=round(t_cold, 2), t_warm_ms=round(t_warm, 2),
            plan=[[round(float(v), 3) for v in row] for row in x_w[:NV].reshape(NB, NA)],
            plan_cold=[[round(float(v), 3) for v in row] for row in x_c[:NV].reshape(NB, NA)],
            cells_changed=n_diff(x_c, x_w),
            changes=changed_cells(x_c, x_w),
            y=[round(float(v), 3) for v in price(B_w, cv(C))],
            detail=detail,
        ))
        b_prev, c_prev = b, C
    return rows


def shadow_decay():
    """影子价格的有效期：把某个基地的可用量一点点放宽，看 z 什么时候拐弯。
    城关不是瓶颈（y=0），把量压下去会先撞上"总供给不够"的墙；
    石桥、柳湾才是真稀缺，每多 1 百吨就省下 |y| 万元。"""
    T0, B0, _, _ = cold(np.r_[S0, D0], COST0)
    y0 = price([int(k) for k in B0], cv(COST0))
    out = []
    for i in range(NB):
        grid, zs = [], []
        for delta in np.arange(-2.0, 2.51, 0.25):
            S = S0.copy()
            S[i] = S0[i] + delta
            grid.append(round(float(S[i]), 2))
            if S.sum() < D0.sum() - 1e-9:
                zs.append(None)               # 总供给不够，模型直接不可行
                continue
            try:
                _, B, _, z = cold(np.r_[S, D0], COST0)
                zs.append(round(float(z), 3))
            except RuntimeError:
                zs.append(None)
        out.append(dict(base=BASE_NAMES[i], y=round(float(y0[i]), 3),
                        grid=grid, z=zs, unit="万元/周"))
    return out


def add_constraint_demo():
    """周五临时加一条：城关→甲城区 每日最多 2.0 百吨（基准解 2.8，是硬需求）。
    对比"热启动加行"和"从零加行"，看增量分析省了多少迭代。"""
    b = np.r_[S0, D0]
    T, B, _, z0 = cold(b, COST0)
    B = [int(k) for k in B]
    x0 = read_x(T, B)
    src = 0 * NA + 0                        # x(城关, 甲城区)：基准解 2.8，是硬需求，卡它就是卡脖子
    coef = np.zeros(NCOL)
    coef[src] = 1.0
    cvec = cv(COST0)

    # ---------- 热启动：旧最优基原样搬到"多一行"的新模型上 ----------
    # 新约束：x_src + u = 2.0（u ≥ 0 是新行的松弛）。旧基 B 在新模型里依然可行，
    # 新表 = [旧表 B⁻¹A | B⁻¹·a_new] + 新行 a_new、右端加一项；u 先当基变量挂着，
    # 它的值是 2.0 − x_src；若为负（超了），跑一次对偶单纯形就自动把它替掉。
    n_new = NCOL + 1                       # 多一个变量 u
    a_new = np.r_[coef, 1.0]               # 新约束在 (x, u) 上的系数
    T2 = np.zeros((NR + 2, n_new + 1))
    T2[:NR, :NCOL] = T[:NR, :NCOL]
    T2[:NR, n_new] = T[:NR, -1]            # 右端列先落在最后一列
    B2 = [int(k) for k in B]
    # 新行 = 单位向量方向（因为它还没有被旧基表达过）；用旧基把它表示出来：
    # 新行原始形式 [a_new, 2.0]，消掉基内变量后剩下 u 的系数
    T2[NR, :n_new] = a_new
    T2[NR, n_new] = 2.0
    for i, k in enumerate(B):
        c = T2[NR, int(k)]
        if abs(c) > 1e-12:
            T2[NR] -= c * T2[i]            # 用旧表第 i 行消掉基变量 k 的贡献
    # 此时 T2[NR] 里 x 部分应为 0，只剩 u 的系数 1 和右端；u 就做新行的基变量
    B2 = B2 + [NCOL]
    # 目标行：沿用旧检验数（y 不变），再把 x_src 的新影子价补上
    T2[NR + 1, :n_new] = np.r_[cvec, 0.0] - price(B, cvec) @ np.hstack([GEO_CLEAN, np.zeros((NR, 1))])
    T2[NR + 1, IDX_ART:] = BIG
    T2[NR + 1, n_new] = 0.0
    it_warm = dual_simplex(T2, B2, NR + 1, rhs=n_new, nvar=n_new)
    x2 = read_x(T2, B2, rhs=n_new)
    z1 = obj_of(x2, COST0)

    # ---------- 冷启动：从零带着这条约束做两阶段 ----------
    wcol = np.zeros(NCOL + 2)
    wcol[IDX_ART:] = 1.0
    A3 = np.zeros((NR + 1, NCOL + 2))
    A3[:NR, :NCOL] = ABAR
    A3[NR, :NCOL] = coef
    A3[NR, NCOL] = 1.0
    T3 = np.zeros((NR + 2, NCOL + 2))
    T3[:NR + 1, :NCOL + 2] = A3
    T3[:NR, NCOL + 1] = b
    T3[NR, NCOL + 1] = 2.0
    basis3 = [IDX_S + i for i in range(NB)] + [IDX_ART + j for j in range(NA)] + [NCOL]
    T3[NR + 1, :NCOL + 2] = wcol - np.asarray(wcol)[basis3] @ A3
    it_cold = primal(T3, basis3, NR + 1, rhs=NCOL + 1)
    for i in range(NR + 1):
        if basis3[i] >= IDX_ART:
            cand = np.where(np.abs(T3[i, :IDX_ART]) > 1e-9)[0]
            if len(cand):
                pivot(T3, basis3, i, int(cand[0]))
                it_cold += 1
    A3c = A3.copy()
    A3c[:, IDX_ART:] = 0.0
    y3 = np.asarray(cvec)[[int(k) for k in basis3 if k < NCOL]] @ np.linalg.inv(
        A3[:, [int(k) for k in basis3]])
    T3[NR + 1, :NCOL + 2] = np.r_[cvec, 0.0, 0.0] - y3 @ A3c
    T3[NR + 1, IDX_ART:] = BIG
    T3[NR + 1, NCOL] = BIG
    it_cold += primal(T3, basis3, NR + 1, rhs=NCOL + 1)
    z_cold = obj_of(read_x(T3, basis3, rhs=NCOL + 1), COST0)

    moves = [(BASE_SHORT[i], AREA_SHORT[j], round(float(x2[i * NA + j] - x0[i * NA + j]), 2))
             for i in range(NB) for j in range(NA)
             if abs(x2[i * NA + j] - x0[i * NA + j]) > 1e-6]
    return dict(z0=round(z0, 3), z_warm=round(z1, 3), z_cold=round(z_cold, 3),
                cost_of_rule=round(z1 - z0, 3),
                it_warm=it_warm, it_cold=it_cold,
                plan=[[round(float(v), 3) for v in r] for r in x2[:NV].reshape(NB, NA)],
                moves=moves,
                rule="城关→甲城区 每日 ≤ 2.0 百吨（原解 2.8，缺口 0.8 必须外调）")


# ============================================================
# 六、作图
# ============================================================
def draw_plan(ax, plan, cost, title, demand, supply):
    M = np.array(plan)
    ax.imshow(M, cmap="YlOrBr", aspect="auto", vmin=0, vmax=max(M.max(), 1) * 1.15)
    ax.set_xticks(range(NA))
    ax.set_xticklabels([f"{AREA_SHORT[j]}\n需{demand[j]}" for j in range(NA)], fontsize=8)
    ax.set_yticks(range(NB))
    ax.set_yticklabels([f"{BASE_SHORT[i]}\n供{supply[i]}" for i in range(NB)], fontsize=8)
    for i in range(NB):
        for j in range(NA):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="#333" if M[i, j] < M.max() * 0.6 else "white")
    ax.set_title(title, fontsize=10)


def plot(rows, shadow, addrow, path):
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # (a) 五周方案一览
    ax = fig.add_subplot(gs[0, :2])
    wk = [r["week"].split(" ")[0] for r in rows]
    itc = [r["it_cold"] for r in rows]
    itw = [r["it_warm"] for r in rows]
    xp = np.arange(len(wk))
    ax.bar(xp - 0.2, itc, 0.4, label="冷启动（每次从零）", color="#c44e52")
    ax.bar(xp + 0.2, itw, 0.4, label="热启动（借上周最优基）", color="#4c72b0")
    for k, v in enumerate(itc):
        ax.text(k - 0.2, v + 0.3, str(v), ha="center", fontsize=8)
    for k, v in enumerate(itw):
        ax.text(k + 0.2, v + 0.3, str(v), ha="center", fontsize=8)
    ax.set_xticks(xp)
    ax.set_xticklabels([f"{r['week']}\n{r['note']}" for r in rows], fontsize=7.5)
    ax.set_ylabel("主元次数")
    ax.set_title("(a) 五周数据变动：同样的答案，代价差一个数量级", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) 每周最优成本
    ax = fig.add_subplot(gs[0, 2])
    zc = [r["z_cold"] for r in rows]
    ax.plot(xp, zc, "o-", color="#333", lw=2)
    for k, v in enumerate(zc):
        ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(xp)
    ax.set_xticklabels(wk, fontsize=8)
    ax.set_ylabel("周总运费（万元）")
    ax.set_title("(b) 每周最优成本", fontsize=10)
    ax.grid(alpha=0.3)

    # (c) 第一周方案
    ax = fig.add_subplot(gs[1, 0])
    r0 = rows[0]
    draw_plan(ax, r0["plan"], None, f"第1周最优调运（{r0['z_warm']:.2f} 万元）", D0, S0)

    # (d) 影子价格的有效期
    ax = fig.add_subplot(gs[1, 1])
    for s in shadow:
        g = np.array(s["grid"])
        z = np.array([np.nan if v is None else v for v in s["z"]], dtype=float)
        ok = ~np.isnan(z)
        ax.plot(g[ok], z[ok], "o-", ms=5,
                label=f"{s['base']}（偏导 {s['y']:+.2f} 万元/百吨）")
        if ok.any():
            k0 = int(np.argmax(ok))
            ax.plot(g[k0], z[k0], "k*", ms=12)
    ax.set_xlabel("该基地可供量（百吨/周）")
    ax.set_ylabel("周总运费（万元）")
    ax.set_title("(d) 影子价格的有效期：★ 当前计划，虚线段=基变了", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (e) 临时加约束
    ax = fig.add_subplot(gs[1, 2])
    draw_plan(ax, addrow["plan"], None,
              f"加约束后（{addrow['z_warm']:.2f} 万元，+{addrow['cost_of_rule']:.2f}）", D0, S0)
    ax.set_xlabel(f"热启动 {addrow['it_warm']} 次 vs 从零 {addrow['it_cold']} 次主元", fontsize=8)

    fig.suptitle("对偶单纯形热启动：上周的最优基，这周还能接着用", fontsize=13, y=0.98)
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 七、主流程
# ============================================================
def main():
    t0 = time.perf_counter()
    rows = solve_all()
    shadow = shadow_decay()
    addrow = add_constraint_demo()

    print("=" * 78)
    print("五周数据变动：冷启动 vs 热启动")
    print("=" * 78)
    for r in rows:
        print(f"{r['week']:<16s} 冷 {r['it_cold']:>2d} 次  z={r['z_cold']:>7.2f} | "
              f"热 {r['it_warm']:>2d} 次  z={r['z_warm']:>7.2f} | 改动 {r['cells_changed']} 格")
        print(f"{'':<16s} {r['detail']}")
        for i in range(NB):
            print(f"{'':<18s}{BASE_SHORT[i]}  " +
                  "  ".join(f"{v:5.2f}" for v in r["plan"][i]))
        print(f"{'':<18s}影子价格 y = {r['y']}")
        print()
    print("=" * 78)
    print(f"临时加约束：{addrow['rule']}")
    print(f"  从 {addrow['z0']:.2f} 涨到 {addrow['z_warm']:.2f}（+{addrow['cost_of_rule']:.2f} 万元/周）")
    print(f"  热启动 {addrow['it_warm']} 次主元；从零重算要 {addrow['it_cold']} 次")
    print(f"  需要改动的运量：{addrow['moves']}")
    print("=" * 78)

    out = dict(weeks=rows, shadow=shadow, add_row=addrow,
               cost0=COST0.tolist(), supply0=S0.tolist(), demand0=D0.tolist(),
               runtime_ms=round((time.perf_counter() - t0) * 1e3, 1))
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(rows, shadow, addrow, OUT_PNG)
    print(f"图：{OUT_PNG.name}  JSON：{OUT_JSON.name}  用时 {out['runtime_ms']:.0f} ms")


if __name__ == "__main__":
    main()
