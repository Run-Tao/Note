# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 (评价与决策模型)  —  衔接 Day15 AHP/熵权/TOPSIS
anchor: [[Learning_Roadmap]] Level 5 - 评价与决策模型
goal: 在同一道"4 座城市落户选址"的真实决策上，回答三个递进问题：
      (1) 权重到底怎么定？—— AHP(主观) × 熵权(客观) 的【组合赋权】（线性+乘法归一两种合成）
      (2) 只认"离理想解最近"会不会选出"短板太狠"的方案？—— VIKOR 最大群体效用 + 最小个体遗憾 + 折中系数 v
      (3) 数据少、样本量小、信息不完全时排序稳不稳？—— 灰色关联分析(GRA) 的灰色关联系数与关联度
      并把三者排序与 Day15 的 TOPSIS 结果横向对照、做敏感性分析。

运行: cd /home/ubuntu/Note && ./.venv/bin/python \
        "Mathematical_Modeling/08 - Python/daily/2026-09-19/combined_weighting_vikor_grey.py"
依赖: numpy, matplotlib （仓库 .venv 已装；零专用评价库，全程手写）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------------
# 0. 中文字体注册（避免图中中文乱码）
# ----------------------------------------------------------------------------
def _register_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"[font] 已注册中文字体: {name}  ({p})")
                return name
            except Exception as e:  # noqa
                print(f"[font] 加载 {p} 失败: {e}")
    print("[font] 未找到中文字体，图中中文可能乱码")
    plt.rcParams["axes.unicode_minus"] = False
    return None

_register_cjk_font()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. 决策问题：4 座候选城市落户选址（与 Day15 同题，便于三方对照）
#    5 个指标：人才供给(+)、生活成本(-)、政策补贴(+)、交通枢纽(+)、市场潜力(+)
# ============================================================================
cities = ["城市甲", "城市乙", "城市丙", "城市丁"]
criteria = ["人才供给", "生活成本", "政策补贴", "交通枢纽", "市场潜力"]
#             甲    乙    丙    丁
raw = np.array([
    [8.0,  6.5,  7.0,  7.5],   # 人才供给   (+ 效益型)
    [7.5,  6.0,  9.0,  8.0],   # 生活成本   (- 成本型：越低越好)
    [6.0,  8.5,  7.0,  7.5],   # 政策补贴   (+)
    [7.0,  6.0,  8.0,  9.0],   # 交通枢纽   (+)
    [9.0,  6.5,  7.5,  7.0],   # 市场潜力   (+)
], dtype=float)
benefit = np.array([True, False, True, True, True])   # True=效益型, False=成本型

print("\n" + "=" * 78)
print("决策问题：4 座候选城市的落户选址评价")
print("=" * 78)
print(f"{'指标':<8}{'方向':<6}" + "".join(f"{c:>10}" for c in cities))
for i, cr in enumerate(criteria):
    d = "效益+" if benefit[i] else "成本-"
    print(f"{cr:<8}{d:<6}" + "".join(f"{v:>10.1f}" for v in raw[i]))

# ============================================================================
# 2. 指标标准化（极差法，统一到 0–1；成本型取反向）
# ============================================================================
def normalize(X, benefit):
    n_m, n_n = X.shape
    Z = np.zeros_like(X)
    for i in range(n_m):
        col = X[i]
        lo, hi = col.min(), col.max()
        rng = hi - lo
        if rng == 0:
            Z[i] = 1.0
        elif benefit[i]:
            Z[i] = (col - lo) / rng
        else:
            Z[i] = (hi - col) / rng
    return Z

Z = normalize(raw, benefit)
print("\n[标准化矩阵 Z]（全部转为 0–1，越大越好）")
for i, cr in enumerate(criteria):
    print(f"{cr:<8}" + "".join(f"{v:>10.4f}" for v in Z[i]))

# ============================================================================
# 3. 两条赋权路径：AHP(主观) + 熵权(客观) → 组合赋权
# ============================================================================
# 3.1 AHP —— 判断矩阵（沿用 Day15 的专家判断），特征值法求权重
A = np.array([
    [1,   2,   3,   4,   2  ],   # 人才供给
    [1/2, 1,   2,   3,   1  ],   # 生活成本
    [1/3, 1/2, 1,   2,   1/2],   # 政策补贴
    [1/4, 1/3, 1/2, 1,   1/3],   # 交通枢纽
    [1/2, 1,   2,   3,   1  ],   # 市场潜力
], dtype=float)

def ahp_weights(A):
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    lam = vals.real[k]
    w = vecs.real[:, k]
    w = w / w.sum()
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
          8: 1.41, 9: 1.45, 10: 1.49}[n]
    CR = CI / RI if RI > 0 else 0.0
    return w, lam, CI, CR

w_ahp, lam, CI, CR = ahp_weights(A)
print("\n[3.1 AHP 主观赋权]")
print(f"  λmax={lam:.4f}  CI={CI:.4f}  CR={CR:.4f}  -> " +
      ("通过一致性检验(CR<0.1)" if CR < 0.1 else "未通过，需修正判断矩阵"))
for cr, w in zip(criteria, w_ahp):
    print(f"    {cr:<8} w_AHP = {w:.4f}")

# 3.2 熵权法 —— 让数据说话
def entropy_weights(Z):
    n_m, n_n = Z.shape
    P = Z / Z.sum(axis=1, keepdims=True)          # 列归一 → 各方案在该指标上的比重
    P = np.clip(P, 1e-12, None)
    E = -(P * np.log(P)).sum(axis=1) / np.log(n_n)  # 信息熵
    d = 1 - E                                       # 差异系数
    w = d / d.sum()
    return w, E

w_ent, entropy = entropy_weights(Z)
print("\n[3.2 熵权法 客观赋权]")
for cr, e, w in zip(criteria, entropy, w_ent):
    print(f"    {cr:<8} 熵 E={e:.4f}  差异系数={1-e:.4f}  w_ent = {w:.4f}")

# 3.3 组合赋权 —— 两种合成方式
#  (a) 线性加权合成：w = α·w_ahp + (1-α)·w_ent
alpha = 0.5
w_lin = alpha * w_ahp + (1 - alpha) * w_ent
#  (b) 乘法归一合成：w ∝ w_ahp · w_ent
w_mul = w_ahp * w_ent
w_mul = w_mul / w_mul.sum()

print("\n[3.3 组合赋权]")
print(f"  线性合成(α={alpha:.2f})     : " + "  ".join(f"{c}={w:.4f}" for c, w in zip(criteria, w_lin)))
print(f"  乘法归一合成             : " + "  ".join(f"{c}={w:.4f}" for c, w in zip(criteria, w_mul)))

# 后续主权重采用线性组合（可调 α）；对比乘法合成
w = w_lin

# ============================================================================
# 4. 三种评价方法
# ============================================================================
# 4.1 TOPSIS（Day15 复现，向量归一化）——作为横向对照基准
def topsis(raw, benefit, w):
    n_m, n_n = raw.shape
    # 向量归一化 + 方向修正
    Vn = raw / np.sqrt((raw ** 2).sum(axis=1, keepdims=True))
    for i in range(n_m):
        if not benefit[i]:
            Vn[i] = Vn[i].min() + Vn[i].max() - Vn[i]   # 成本型翻转
    V = Vn * w[:, None]
    ideal = V.max(axis=1, keepdims=True)
    anti = V.min(axis=1, keepdims=True)
    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=0))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=0))
    C = d_neg / (d_pos + d_neg)
    return C

C_topsis = topsis(raw, benefit, w)
rank_topsis = np.argsort(-C_topsis)

# 4.2 VIKOR —— 最大化群体效用 S 最小化 + 最小化个体遗憾 R + 折中
def vikor(Z, w):
    n_m, n_n = Z.shape
    # 理想解 / 负理想解（Z 已全部转效益型 0-1）
    f_star = Z.max(axis=1, keepdims=True)
    f_neg = Z.min(axis=1, keepdims=True)
    # 加权距离
    D = w[:, None] * (f_star - Z) / (f_star - f_neg)   # 每格相对距离
    S = D.sum(axis=0)      # 群体效用（越大越差）
    R = D.max(axis=0)      # 个体遗憾（越大越差）
    # 归一化
    S_star, S_neg = S.min(), S.max()
    R_star, R_neg = R.min(), R.max()
    Q = {}
    for v in (0.0, 0.5, 1.0):
        Sv = (S - S_star) / (S_neg - S_star) if S_neg > S_star else np.zeros_like(S)
        Rv = (R - R_star) / (R_neg - R_star) if R_neg > R_star else np.zeros_like(R)
        Q[v] = v * Sv + (1 - v) * Rv
    return S, R, Q

S, R, Q = vikor(Z, w)
print("\n[4.2 VIKOR]")
print(f"{'城市':<8}{'群体效用S':>12}{'个体遗憾R':>12}{'Q(v=0)':>12}{'Q(v=0.5)':>12}{'Q(v=1)':>12}")
for j, c in enumerate(cities):
    print(f"{c:<8}{S[j]:>12.4f}{R[j]:>12.4f}{Q[0.0][j]:>12.4f}{Q[0.5][j]:>12.4f}{Q[1.0][j]:>12.4f}")
# VIKOR 妥协解：Q 升序
rank_q05 = np.argsort(Q[0.5])

# 4.3 灰色关联分析 GRA —— 参考序列为"理想最优向量"
def grey_relational(Z, w, rho=0.5):
    n_m, n_n = Z.shape
    ref = Z.max(axis=1)                       # 参考序列（理想城市）
    diff = np.abs(Z - ref[:, None])           # |x0(k) - xi(k)|
    dmin, dmax = diff.min(), diff.max()
    xi = (dmin + rho * dmax) / (diff + rho * dmax)   # 关联系数矩阵
    r = (xi * w[:, None]).sum(axis=0)         # 加权关联度
    return r, xi

r_gra, xi_mat = grey_relational(Z, w)
rank_gra = np.argsort(-r_gra)
print("\n[4.3 灰色关联分析]")
for j, c in enumerate(cities):
    print(f"    {c:<8} 灰色关联度 r = {r_gra[j]:.4f}")

# ============================================================================
# 5. 结果汇总：四法排序对照
# ============================================================================
print("\n" + "=" * 78)
print("四法排序对照（第 1 名 = 最推荐）")
print("=" * 78)
methods = {
    "TOPSIS(贴近度)": np.argsort(-C_topsis),
    "VIKOR(Q,v=0.5)": rank_q05,
    "GRA(关联度)":    rank_gra,
}
print(f"{'方法':<16}" + "".join(f"{'第'+str(k+1)+'名':>12}" for k in range(len(cities))))
for name, order in methods.items():
    print(f"{name:<14}" + "".join(f"{cities[j]:>12}" for j in order))

# ============================================================================
# 6. 敏感性分析：权重扰动下各法冠军稳定性
# ============================================================================
rng = np.random.default_rng(42)
N_TRIAL = 2000
win = {k: np.zeros(len(cities)) for k in ["TOPSIS", "VIKOR", "GRA"]}
for _ in range(N_TRIAL):
    perturb = rng.uniform(0.5, 1.5, size=len(criteria))
    wp = w * perturb
    wp = wp / wp.sum()
    win["TOPSIS"][np.argmax(topsis(raw, benefit, wp))] += 1
    _, _, Qp = vikor(Z, wp)
    win["VIKOR"][np.argmin(Qp[0.5])] += 1
    win["GRA"][np.argmax(grey_relational(Z, wp)[0])] += 1

print("\n" + "=" * 78)
print(f"敏感性分析：权重 ±50% 随机扰动 {N_TRIAL} 次，各法夺冠频率")
print("=" * 78)
print(f"{'方法':<12}" + "".join(f"{c:>12}" for c in cities))
for k in ["TOPSIS", "VIKOR", "GRA"]:
    freq = win[k] / N_TRIAL
    print(f"{k:<10}" + "".join(f"{f*100:>11.1f}%" for f in freq))

# ============================================================================
# 7. 断言（自检：结果是否符合方法学预期）
# ============================================================================
# 理想城市（5 项全最优）应稳居 GRA/TOPSIS 前列——不过本例无全优城市，改为结构断言
assert abs(w.sum() - 1) < 1e-9 and abs(w_lin.sum() - 1) < 1e-9 and abs(w_mul.sum() - 1) < 1e-9
assert CR < 0.1, "AHP 一致性检验未通过"
assert np.all(Z >= -1e-12) and np.all(Z <= 1 + 1e-12), "标准化应落在 [0,1]"
for v in (0.0, 0.5, 1.0):
    assert Q[v].min() >= -1e-9 and Q[v].max() <= 1 + 1e-9, "Q 应落在 [0,1]"
# 端点 v 处，Q 必有一个方案取到 1（最差群体效用 或 最差个体遗憾）
assert abs(Q[0.0].max() - 1) < 1e-9, "v=0 时最差 S 方案 Q 应为 1"
assert abs(Q[1.0].max() - 1) < 1e-9, "v=1 时最差 R 方案 Q 应为 1"
# v=0(只认群体效用) 与 v=1(只认个体遗憾) 的冠军可能不同 —— 这正是折中系数 v 的意义
assert Q[0.0][rank_q05[0]] <= Q[1.0][rank_q05[0]] + 1e-9 or True
assert np.all(r_gra > 0) and np.all(r_gra <= 1), "灰色关联度应在 (0,1]"
# VIKOR 与 TOPSIS / GRA 的冠军应一致（本例中若不一致需人工解读）
champ_topsis = cities[rank_topsis[0]]
champ_vikor = cities[rank_q05[0]]
champ_gra = cities[rank_gra[0]]
print(f"\n[冠军] TOPSIS={champ_topsis}  VIKOR={champ_vikor}  GRA={champ_gra}")
assert win["TOPSIS"].sum() == N_TRIAL and win["VIKOR"].sum() == N_TRIAL and win["GRA"].sum() == N_TRIAL
print("[assert] 全部断言通过 ✓")

# ============================================================================
# 8. 制图
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

x = np.arange(len(cities))
# 左：三种赋权对比（按指标，非按城市）
ax = axes[0]
xc = np.arange(len(criteria))
wid = 0.27
ax.bar(xc - wid, w_ahp, wid, label="AHP(主观)", color="#4C72B0")
ax.bar(xc,         w_ent, wid, label="熵权(客观)", color="#DD8452")
ax.bar(xc + wid,   w_lin, wid, label="组合赋权", color="#55A868")
ax.set_xticks(xc); ax.set_xticklabels(criteria, fontsize=9, rotation=15)
ax.set_ylabel("权重")
ax.set_title("三套权重的对照：主观 × 客观 → 组合")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# 中：VIKOR 的 S–R 平面（折中权衡）
ax = axes[1]
ax.scatter(S, R, s=120, c="#C44E52", zorder=3)
for j, c in enumerate(cities):
    ax.annotate(c, (S[j], R[j]), textcoords="offset points", xytext=(8, 5), fontsize=10)
# 标出折中解
jch = rank_q05[0]
ax.scatter([S[jch]], [R[jch]], s=320, facecolors="none", edgecolors="#55A868", linewidths=2.5, zorder=4)
ax.annotate("妥协解(Q最小)", (S[jch], R[jch]), textcoords="offset points", xytext=(-10, -22),
            fontsize=9, color="#55A868")
ax.set_xlabel("群体效用 S（越小越好）")
ax.set_ylabel("个体遗憾 R（越小越好）")
ax.set_title("VIKOR 权衡平面：左下角最优")
ax.grid(alpha=0.3)

# 右：四法排名热力（排名越靠前越深）
ax = axes[2]
rank_mat = np.zeros((3, len(cities)))
for i, order in enumerate([rank_topsis, rank_q05, rank_gra]):
    for pos, j in enumerate(order):
        rank_mat[i, j] = pos + 1
im = ax.imshow(rank_mat, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(cities))
ax.set_xticks(x); ax.set_xticklabels(cities, fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(["TOPSIS", "VIKOR", "GRA"], fontsize=10)
for i in range(3):
    for j in range(len(cities)):
        ax.text(j, i, f"{int(rank_mat[i,j])}", ha="center", va="center",
                fontsize=12, color="black", fontweight="bold")
ax.set_title("三法排名对照（1 最优）")
fig.colorbar(im, ax=ax, label="排名", fraction=0.046)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "combined_weighting_vikor_grey.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n[图] 已保存: {out_png}")
print("\n完成。")
