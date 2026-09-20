# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L5 (评价与决策模型)  —  衔接 Day15 AHP/熵权/TOPSIS 与 Day19 组合赋权/VIKOR/GRA
anchor: [[Learning_Roadmap]] Level 5 - 评价与决策模型
goal: 在同一道"4 座城市落户选址"的真实决策上，补上评价类最后三块短板：
      (1) 评语是"好/较好/一般/较差"这种模糊话，怎么变成一个可比的分数？—— 模糊综合评价(FCE)
          （隶属度矩阵 × 权重 → 综合评价向量；最大隶属度 vs 加权平均两种"读结论"方式）
      (2) 主观(AHP)会偏心、客观(熵权)只看"数据分不分得开"，有没有更硬的客观权重？—— CRITIC
          （对比强度=标准差 × 冲突性=1-相关系数 → 信息量 C → 权重）
      (3) 指标之间高度相关、信息冗余时，能不能让数据自己"造出新维度"再排序？—— PCA 综合评价
          （标准化 → 协方差/相关阵特征分解 → 主成分得分 → 按方差贡献率加权合成）
      四法（FCE / CRITIC-TOPSIS / PCA / Day15 熵权TOPSIS）排序横向对照 + CRITIC 权重敏感性。

运行: cd /home/ubuntu/Note && ./.venv/bin/python \
        "Mathematical_Modeling/08 - Python/daily/2026-09-20/fuzzy_critic_pca_evaluation.py"
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
rng = np.random.default_rng(20260920)

# ============================================================================
# 1. 决策问题：4 座候选城市落户选址（与 Day15/Day19 同题，便于四法对照）
#    5 个指标：人才供给(+)、生活成本(-)、政策补贴(+)、交通枢纽(+)、市场潜力(+)
#    效益型(+): 越大越好；成本型(-): 越小越好
# ============================================================================
cities = ["城市甲", "城市乙", "城市丙", "城市丁"]
criteria = ["人才供给", "生活成本", "政策补贴", "交通枢纽", "市场潜力"]
benefit = np.array([True, False, True, True, True])  # 指标方向
# 原始得分矩阵 X：行=城市 列=指标（原始量纲，生活成本为"成本"越大越差）
X = np.array([
    [8.0, 7.5, 6.0, 7.0, 9.0],   # 城市甲
    [6.5, 6.0, 8.5, 6.0, 6.5],   # 城市乙
    [7.0, 9.0, 7.0, 8.0, 7.5],   # 城市丙
    [7.5, 8.0, 7.5, 9.0, 7.0],   # 城市丁
], dtype=float)

m, n = X.shape  # m 个方案(城市)，n 个指标
print("=" * 78)
print(f"决策：{m} 座候选城市落户选址，{n} 个指标 | 指标方向: "
      + ", ".join(f"{c}{'+' if b else '-'}" for c, b in zip(criteria, benefit)))
print("=" * 78)


# ----------------------------------------------------------------------------
# 2. 极差标准化（向量归一/离差化），把不同量纲、不同方向的指标拉到同一把尺子
#    效益型:  (x - min) / (max - min)
#    成本型:  (max - x) / (max - min)
#    → 标准化后 Z ∈ [0,1]，越大越好（方向已统一）
# ----------------------------------------------------------------------------
def range_normalize(X, benefit):
    Z = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if benefit[j]:
            Z[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            Z[:, j] = (col.max() - col) / (col.max() - col.min())
    return Z

Z = range_normalize(X, benefit)
print("\n[标准化] 极差标准化后（越大越好，方向已统一）：")
for i, c in enumerate(cities):
    print(f"  {c}: " + "  ".join(f"{criteria[j]}={Z[i,j]:.3f}" for j in range(n)))


# ============================================================================
# 3. 第一问：模糊综合评价 FCE（评语是"好/较好/一般/较差"，怎么变成分数？）
# ============================================================================
print("\n" + "=" * 78)
print("【第一问】模糊综合评价 FCE —— 把模糊评语变成可比分数")
print("=" * 78)

# 3.1 评语等级 + 分值向量（5 级）
levels = ["优", "良", "中", "及格", "差"]
level_scores = np.array([95, 85, 75, 65, 55], dtype=float)  # 各评语的中心分

# 3.2 构造隶属度矩阵 R：每个指标对各评语等级的"隶属程度"（行=指标，列=评语）
#     真实场景来自多位专家打分统计；此处用"以标准化得分为中心的高斯隶属函数"生成，
#     相当于让数据既保留专家评语的语言习惯、又与客观分数挂钩（可复现）。
def gaussian_membership(score01, levels_centers=None):
    """把 [0,1] 的标准化得分映射成 5 级评语的隶属度（高斯型），归一化后返回。"""
    centers = np.array([0.90, 0.70, 0.50, 0.30, 0.10])  # 各评语对应的标准化中心
    sigma = 0.18
    w = np.exp(-((score01 - centers) ** 2) / (2 * sigma ** 2))
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

# 对每个城市，各指标先在"该指标上跨城市的相对位置"上取标准化得分，
# 这样"城市甲的人才供给在4城里算不错"能在评语里体现出来（横向比较的口径）。
R_by_city = {}
for i in range(m):
    R = np.zeros((n, len(levels)))
    for j in range(n):
        # 该指标上，本城市相对其他城市的归一化位置 ∈[0,1]
        col = Z[:, j]
        rel = (Z[i, j] - col.min()) / (col.max() - col.min() + 1e-12)
        R[j, :] = gaussian_membership(rel)
    R_by_city[cities[i]] = R

# 3.3 权重：这里用 CRITIC（客观）权重（下面第 4 节算），FCE 与 CRITIC 共享同一套权重，
#     也便于四法口径一致。先占位，第 4 节算完后回填。
def fuzzy_synthesize(R, w):
    """B = w ∘ R —— 模糊合成算子（加权平均型：B_k = Σ_j w_j R_jk），再归一化。"""
    B = w @ R          # 形状 (len(levels),)
    return B / B.sum()

print("\n[FCE] 评语等级:", " / ".join(levels), " 中心分:", level_scores.tolist())
print("[FCE] 隶属度矩阵示例（城市甲的 5 个指标 × 5 级评语，行=指标）：")
for j in range(n):
    print(f"    {criteria[j]:<8}" + "  ".join(f"{v:.3f}" for v in R_by_city[cities[0]][j]))


# ============================================================================
# 4. 第二问：CRITIC 客观赋权（比熵权更硬：对比强度 × 指标间冲突性）
# ============================================================================
print("\n" + "=" * 78)
print("【第二问】CRITIC 客观赋权 —— 对比强度 × 冲突性 = 信息量")
print("=" * 78)

# 4.1 在标准化矩阵 Z 上计算（Z 已统一方向，∈[0,1]）
#     - 对比强度：标准差 σ_j（某指标在方案间越"分得开"，越有信息）
#     - 冲突性：   Σ_k (1 - r_jk)，与其他指标相关性越低、越提供独立信息
std = Z.std(axis=0, ddof=1)             # 样本标准差
corr = np.corrcoef(Z, rowvar=False)      # 指标相关矩阵
conflict = (1.0 - corr).sum(axis=1)      # 每列与其他指标的"独立性"之和
info = std * conflict                    # 信息量 C_j = σ_j × Σ(1-r_jk)
w_critic = info / info.sum()             # 归一化权重

print("\n[CRITIC] 逐指标：")
print(f"    {'指标':<8}{'标准差σ':>10}{'冲突性Σ(1-r)':>14}{'信息量C':>10}{'权重':>10}")
for j in range(n):
    print(f"    {criteria[j]:<8}{std[j]:>10.4f}{conflict[j]:>14.4f}{info[j]:>10.4f}{w_critic[j]:>10.4f}")
print(f"    权重和 = {w_critic.sum():.6f}（应为1）")
print("\n[CRITIC] 指标相关矩阵 r_jk：")
print("        " + "".join(f"{c:>9}" for c in criteria))
for j in range(n):
    print(f"    {criteria[j]:<6}" + "".join(f"{corr[j,k]:>9.3f}" for k in range(n)))

# 4.2 回填：用 CRITIC 权重做 FCE 合成
print("\n[FCE] 用 CRITIC 权重做模糊合成（B = w ∘ R）：")
fce_score = {}
fce_vector = {}
for city in cities:
    B = fuzzy_synthesize(R_by_city[city], w_critic)
    fce_vector[city] = B
    # 两种"读结论"方式
    best_level = int(np.argmax(B))                 # 最大隶属度原则
    wavg_score = float(B @ level_scores)           # 加权平均（把评语折算成分）
    fce_score[city] = wavg_score
    print(f"    {city}: 评语向量 " + "  ".join(f"{v:.3f}" for v in B)
          + f" | 最大隶属度={levels[best_level]} | 加权平均分={wavg_score:.2f}")

fce_rank = sorted(cities, key=lambda c: fce_score[c], reverse=True)
print("    FCE 排序:", " > ".join(fce_rank))


# ============================================================================
# 5. 第三问：PCA 综合评价（数据自己造维度 → 按方差贡献率加权合成综合分）
# ============================================================================
print("\n" + "=" * 78)
print("【第三问】PCA 综合评价 —— 让数据自己造出新维度再排序")
print("=" * 78)

# 5.1 用 z-score 标准化（PCA 要求各指标均值为0、方差为1，避免量纲主导方差）
Zz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
# 成本型指标取负号，使所有指标"越大越好"，PCA 主成分方向才有统一意义
Zz[:, ~benefit] *= -1.0

# 5.2 相关矩阵特征分解（等价于对 Zz 做 PCA）
corr_z = np.corrcoef(Zz, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(corr_z)
order = np.argsort(eigvals)[::-1]          # 从大到小
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
eigvals_clip = np.clip(eigvals, 0, None)
explained = eigvals_clip / eigvals_clip.sum()

print("\n[PCA] 主成分方差解释率：")
cum = 0.0
for k in range(n):
    cum += explained[k]
    print(f"    PC{k+1}: 特征值={eigvals[k]:.4f}  解释率={explained[k]*100:5.2f}%  累计={cum*100:6.2f}%")

# 5.3 主成分得分：PC_k = Zz · v_k；综合分 = Σ 解释率_k × PC_k（越大越好）
scores_pc = Zz @ eigvecs                    # 形状 (m, n)
pca_score_raw = scores_pc @ explained       # 按方差贡献率加权
# 平移到 [0,1] 方便读（保持大小次序不变）
pca_score = (pca_score_raw - pca_score_raw.min()) / (pca_score_raw.max() - pca_score_raw.min() + 1e-12)

# 取前 2 个主成分做"评价平面"可视化
pc1_loading = eigvecs[:, 0]
print("\n[PCA] 第一主成分载荷（各指标在 PC1 上的权重方向）：")
for j in range(n):
    print(f"    {criteria[j]:<8}{pc1_loading[j]:+.4f}")
print("\n[PCA] 综合得分（0~1 归一）与排序：")
pca_rank = sorted(cities, key=lambda c: pca_score[cities.index(c)], reverse=True)
for city in cities:
    i = cities.index(city)
    print(f"    {city}: PC1={scores_pc[i,0]:+.4f}  PC2={scores_pc[i,1]:+.4f}  综合分={pca_score[i]:.4f}")
print("    PCA 排序:", " > ".join(pca_rank))


# ============================================================================
# 6. 对照：Day15 熵权 + TOPSIS（同一套标准化，客观权重另一把尺子）
# ============================================================================
print("\n" + "=" * 78)
print("【对照】熵权法 + TOPSIS（Day15 口径，沿用同题同标准化）")
print("=" * 78)

# 熵权：在 Z 上算（Z ∈[0,1] 且方向已统一）
P = Z / (Z.sum(axis=0, keepdims=True) + 1e-12)
P = np.clip(P, 1e-12, 1.0)
E = -(P * np.log(P)).sum(axis=0) / np.log(m)   # 信息熵
d = 1.0 - E                                     # 差异系数
w_entropy = d / d.sum()
print("\n[熵权] 权重:", {criteria[j]: round(float(w_entropy[j]), 4) for j in range(n)})

# TOPSIS：向量归一化 → 加权 → 正负理想解 → 贴近度
V = Z / (np.sqrt((Z ** 2).sum(axis=0, keepdims=True)) + 1e-12) * w_entropy
v_plus = V.max(axis=0)
v_minus = V.min(axis=0)
d_plus = np.sqrt(((V - v_plus) ** 2).sum(axis=1))
d_minus = np.sqrt(((V - v_minus) ** 2).sum(axis=1))
closeness = d_minus / (d_plus + d_minus + 1e-12)
topsis_rank = sorted(cities, key=lambda c: closeness[cities.index(c)], reverse=True)
print("[TOPSIS] 贴近度 C*：")
for city in cities:
    i = cities.index(city)
    print(f"    {city}: C*={closeness[i]:.4f}")
print("    TOPSIS 排序:", " > ".join(topsis_rank))


# ============================================================================
# 7. CRITIC 权重敏感性分析（±30% 扰动 × N 次，看冠军稳不稳）
# ============================================================================
print("\n" + "=" * 78)
print("【敏感性】CRITIC 权重 ±30% 随机扰动，看各城市夺冠频率")
print("=" * 78)

N_TRIALS = 2000
perturb = 1.0 + 0.30 * (2 * rng.random((N_TRIALS, n)) - 1.0)  # 每指标一个 ±30% 乘子
champion_counts = {c: 0 for c in cities}
for t in range(N_TRIALS):
    w = w_critic * perturb[t]
    w = w / w.sum()
    # 用扰动权重重算 FCE 加权平均分（与主结果同口径）
    sc = {c: float(fuzzy_synthesize(R_by_city[c], w) @ level_scores) for c in cities}
    champ = max(sc, key=sc.get)
    champion_counts[champ] += 1
print(f"    扰动 {N_TRIALS} 次，冠军频率：")
for c in cities:
    print(f"    {c}: {champion_counts[c]/N_TRIALS*100:5.1f}%  ({champion_counts[c]}次)")


# ============================================================================
# 8. 四法排序横向对照
# ============================================================================
print("\n" + "=" * 78)
print("【对照】四法排序总表（名次=1 最好）")
print("=" * 78)
methods = {
    "模糊综合(FCE)": fce_rank,
    "CRITIC-TOPSIS": sorted(cities, key=lambda c: closeness[cities.index(c)], reverse=True),
    "PCA 综合": pca_rank,
}
# 熵权TOPSIS 与 CRITIC-TOPSIS 用同一 TOPSIS，此处另列出便于区分权重口径
method_ranks = {
    "模糊综合评价(FCE)": fce_rank,
    "CRITIC-TOPSIS": topsis_rank,   # 注意：closeness 基于熵权，这里单列 CRITIC-TOPSIS 见下
    "PCA 综合评价": pca_rank,
}
# 真正用 CRITIC 权重的 TOPSIS（与熵权 TOPSIS 对照）
V_c = Z / (np.sqrt((Z ** 2).sum(axis=0, keepdims=True)) + 1e-12) * w_critic
dp = np.sqrt(((V_c - V_c.max(axis=0)) ** 2).sum(axis=1))
dm = np.sqrt(((V_c - V_c.min(axis=0)) ** 2).sum(axis=1))
closeness_critic = dm / (dp + dm + 1e-12)
critic_topsis_rank = sorted(cities, key=lambda c: closeness_critic[cities.index(c)], reverse=True)

method_ranks = {
    "模糊综合评价(FCE)": fce_rank,
    "CRITIC-TOPSIS": critic_topsis_rank,
    "熵权-TOPSIS": topsis_rank,
    "PCA 综合评价": pca_rank,
}
rank_dict = {}
for name, order in method_ranks.items():
    for pos, city in enumerate(order, start=1):
        rank_dict[(name, city)] = pos

print(f"\n    {'城市':<8}" + "".join(f"{name:>18}" for name in method_ranks))
for city in cities:
    print(f"    {city:<8}" + "".join(f"{rank_dict[(name,city)]:>18}" for name in method_ranks))

# 冠军一致性
champions = {name: order[0] for name, order in method_ranks.items()}
uniq_champ = set(champions.values())
print("\n    各法冠军:", champions)
print(f"    冠军是否唯一一致: {len(uniq_champ) == 1}  →  {list(uniq_champ)}")


# ============================================================================
# 9. 图：FCE 评语向量热力 + CRITIC 权重与信息量 + PCA 评价平面 + 四法名次对照
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) FCE 评语向量热力图
ax = axes[0, 0]
Bmat = np.vstack([fce_vector[c] for c in cities])
im = ax.imshow(Bmat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=Bmat.max())
ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_yticks(range(m))
ax.set_yticklabels(cities)
for i in range(m):
    for k in range(len(levels)):
        ax.text(k, i, f"{Bmat[i,k]:.2f}", ha="center", va="center", fontsize=9,
                color="black" if Bmat[i, k] < Bmat.max() * 0.7 else "white")
ax.set_title("(a) 模糊综合评价：各城市评语向量 $B=w\\circ R$", fontsize=12)
ax.set_xlabel("评语等级")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (b) CRITIC：权重 vs 信息量
ax = axes[0, 1]
xpos = np.arange(n)
ax.bar(xpos - 0.2, w_critic, width=0.4, label="CRITIC 权重", color="#d9534f")
ax2 = ax.twinx()
ax2.bar(xpos + 0.2, info, width=0.4, label="信息量 C=σ×冲突", color="#5bc0de", alpha=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(criteria, fontsize=9)
ax.set_ylabel("CRITIC 权重", color="#d9534f")
ax2.set_ylabel("信息量 C", color="#5bc0de")
ax.set_title("(b) CRITIC 客观赋权：对比强度 × 冲突性", fontsize=12)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

# (c) PCA 评价平面（PC1–PC2）
ax = axes[1, 0]
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.axvline(0, color="grey", lw=0.8, ls="--")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, city in enumerate(cities):
    ax.scatter(scores_pc[i, 0], scores_pc[i, 1], s=260, color=colors[i],
               edgecolor="k", zorder=5)
    ax.annotate(city, (scores_pc[i, 0], scores_pc[i, 1]),
                xytext=(8, 8), textcoords="offset points", fontsize=11)
# 画载荷箭头
scale = 2.2
for j in range(n):
    ax.arrow(0, 0, eigvecs[j, 0] * scale, eigvecs[j, 1] * scale,
             head_width=0.08, color="grey", alpha=0.7, length_includes_head=True)
    ax.annotate(criteria[j], (eigvecs[j, 0] * scale, eigvecs[j, 1] * scale),
                fontsize=8, color="dimgrey")
ax.set_xlabel(f"PC1（解释 {explained[0]*100:.1f}%）")
ax.set_ylabel(f"PC2（解释 {explained[1]*100:.1f}%）")
ax.set_title("(c) PCA 评价平面：城市位置 + 指标载荷", fontsize=12)

# (d) 四法名次对照热力（名次越小越好）
ax = axes[1, 1]
Rk = np.array([[rank_dict[(name, c)] for name in method_ranks] for c in cities])
im2 = ax.imshow(Rk, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=m)
ax.set_xticks(range(len(method_ranks)))
ax.set_xticklabels(list(method_ranks.keys()), fontsize=8, rotation=15)
ax.set_yticks(range(m))
ax.set_yticklabels(cities)
for i in range(m):
    for k in range(len(method_ranks)):
        ax.text(k, i, f"#{Rk[i,k]}", ha="center", va="center",
                fontsize=11, color="black")
ax.set_title("(d) 四法排名对照（#1 最优）", fontsize=12)
fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, ticks=range(1, m + 1))

fig.suptitle("Day20 评价类收尾：模糊综合评价 / CRITIC 客观赋权 / PCA 综合评价", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = os.path.join(OUT_DIR, "fuzzy_critic_pca_evaluation.png")
fig.savefig(out_png, dpi=130)
print(f"\n[图] 已保存: {out_png}")


# ============================================================================
# 10. 断言：验证实现正确性与结果自洽
# ============================================================================
print("\n" + "=" * 78)
print("【断言】")
print("=" * 78)

# 权重和为 1
assert abs(w_critic.sum() - 1.0) < 1e-9, "CRITIC 权重和应为1"
assert abs(w_entropy.sum() - 1.0) < 1e-9, "熵权权重和应为1"
print("  [ok] CRITIC/熵权权重和均=1")

# 标准化后 Z ∈[0,1]，且每个指标上最小值恰为0、最大值为1
assert Z.min() >= -1e-12 and Z.max() <= 1 + 1e-12, "Z 应落在[0,1]"
for j in range(n):
    assert abs(Z[:, j].min()) < 1e-12 and abs(Z[:, j].max() - 1) < 1e-12
print("  [ok] 极差标准化后每列 min=0,max=1")

# 隶属度矩阵每行和为1
for city in cities:
    R = R_by_city[city]
    assert np.allclose(R.sum(axis=1), 1.0), "隶属度矩阵每行应为1"
    B = fce_vector[city]
    assert abs(B.sum() - 1.0) < 1e-9, "FCE 综合向量应为1"
    assert B.min() >= -1e-12
print("  [ok] 隶属度矩阵每行和=1；FCE 综合向量归一")

# 主成分解释率非负且和为 1，累计满足解析
assert abs(explained.sum() - 1.0) < 1e-9, "PCA 解释率之和应为1"
assert np.all(np.diff(explained) <= 1e-9), "解释率应递减"
print(f"  [ok] PCA 解释率和=1 且递减（PC1={explained[0]*100:.1f}%）")

# 相关矩阵对角线为 1
assert np.allclose(np.diag(corr), 1.0, atol=1e-9)
print("  [ok] 指标相关矩阵对角线=1")

# 排名是完整排列（每法恰好覆盖4个城市）
for name, order in method_ranks.items():
    assert sorted(order) == sorted(cities), f"{name} 排名应覆盖全部城市"
print("  [ok] 每法排名均为完备排列（无重复/遗漏）")

# 敏感性计数守恒
assert sum(champion_counts.values()) == N_TRIALS, "夺冠计数应守恒"
print(f"  [ok] 敏感性计数守恒（合计={N_TRIALS}）")

print("\n全部断言通过 ✅")
print("=" * 78)
print("完成。四法冠军:", champions)
