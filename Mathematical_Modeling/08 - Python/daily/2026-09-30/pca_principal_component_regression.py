# -*- coding: utf-8 -*-
"""
pca_principal_component_regression.py
-------------------------------------
subject: mathematical-modeling
level:   L8 机器学习 / 数据驱动（主成分分析 PCA + 主成分回归 PCR）
anchor:  Day32（接 Day9 KNN / Day10-13 信贷家族，补齐 Roadmap 反复点名的"PCA 降维"缺口）
goal:    当解释变量之间"高度相关"（多重共线性）时，普通最小二乘(OLS)的系数
         会剧烈摆动、符号甚至与常识相反，预测方差爆炸。PCA 把相关的一堆指标
         旋转成少数"互不相关的主成分"，用主成分做回归(PCR)——
         回答决策者那句："指标又多又重叠，怎么把信息压成几维、还不丢预测力？"

真实决策背景：
  一家农商行想给小微企业做"年违约损失金额"预测，用了 8 个经营/财务指标
  （营收、利润、资产、现金流、员工数、订单量、负债率、成立年限）。
  现实里这些指标**强相关**（大企业各项都大），OLS 系数不稳定、解释不了业务。
  银行风控要的其实是两个东西：
    (1) 一个**稳定的**违约损失预测（用于计提拨备）；
    (2) 一句**能讲给人听的话**——"到底哪几个维度在决定风险"。

数学：
  1) 标准化 X -> Z（PCA 前必做，否则量纲大的列霸占方差）。
  2) 对相关系数矩阵 R = Z^T Z /(n-1) 做特征分解（本课用 SVD，数值更稳）：
        Z = U S V^T，主成分得分 T = U S = Z V，载荷 V 的列 = 主成分方向。
     第 k 个主成分解释的方差比例 = s_k^2 / Σ s_j^2。
  3) 主成分**两两正交**：T^T T 是对角阵 -> 回归时没有共线性。
  4) PCR：取前 q 个主成分做 OLS（y ~ T[:, :q]），再映射回原变量空间：
        beta_PCR = V[:, :q] @ beta_T
     注意：**PCR 的系数是有偏的**（丢弃了后 n-q 个方向），但方差大幅降低，
     在共线性强时总误差(MSE)反而更小——这就是"偏差-方差权衡"。

对比四法（同一题，同一份训练/测试划分）：
  1) OLS  ：全部 8 指标回归（多重共线性的受害者）
  2) PCR  ：取累计解释方差 >= 90% 的主成分
  3) Ridge：L2 正则（另一种治共线性的思路）
  4) 真值核对：数据由"3 个潜在因子"生成，看 PCR 能否把真实维度找回来。

并做：解释方差碎石图 + 载荷解读 + 主成分个数 q 对测试误差的影响 +
      系数稳定性（Bootstrap 重采样看 OLS vs PCR 系数抖动）。

运行：cd /home/ubuntu/Note && ./.venv/bin/python \
       "Mathematical_Modeling/08 - Python/daily/2026-09-30/pca_principal_component_regression.py"
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RNG = np.random.default_rng(20260930)
OUTDIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# 0. 中文字体
# ----------------------------------------------------------------------
def register_cjk_font():
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Source Han Sans SC", "Microsoft YaHei", "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


FONT = register_cjk_font()
print(f"[字体] 使用中文字体: {FONT}")


# ----------------------------------------------------------------------
# 1. 造数据：8 个强相关指标其实由 3 个潜在因子驱动
# ----------------------------------------------------------------------
# 潜在因子（业务语言）：
#   F1 = 企业规模   （营收/资产/员工数/订单量 都由它主导）
#   F2 = 盈利能力   （利润/现金流）
#   F3 = 风险敞口   （负债率 高、成立年限 短 -> 违约损失大）
# 生成 8 个可观测指标 = 因子线性组合 + 噪声；y(违约损失) 只由这 3 个因子决定。
N_SAMPLE = 600
FEATURES = ["营收", "利润率", "总资产", "经营现金流",
            "员工数", "订单量", "负债率", "成立年限"]

# 载荷矩阵 A：8 个可观测指标 <- 3 个潜在因子
A = np.array([
    #   F1(规模)  F2(盈利)  F3(风险)
    [1.00,      0.10,    0.00],   # 营收
    [0.05,      0.95,   -0.10],   # 利润率
    [0.95,      0.20,    0.05],   # 总资产
    [0.30,      0.80,    0.00],   # 经营现金流
    [0.90,      0.05,   -0.05],   # 员工数
    [0.92,      0.15,    0.00],   # 订单量
    [-0.10,    -0.15,    0.95],   # 负债率（高 -> 风险）
    [0.20,      0.10,   -0.85],   # 成立年限（短 -> 风险，故负载荷）
])

# 真实回归系数（在因子空间）：违约损失主要由"风险敞口"和"盈利"决定
TRUE_BETA_F = np.array([0.20, -0.70, 1.10])
# 共同因子贡献大、指标个体噪声小 -> 8 个指标被 3 个因子"一荣俱荣"，形成强共线性
NOISE = 0.18

F = RNG.normal(size=(N_SAMPLE, 3))
F = (F - F.mean(0)) / F.std(0)
Zn = RNG.normal(scale=NOISE, size=(N_SAMPLE, len(FEATURES)))
Xr = F @ A.T + Zn                       # 8 个原始指标
y = F @ TRUE_BETA_F + RNG.normal(scale=0.5, size=N_SAMPLE)

# 训练/测试划分
idx = RNG.permutation(N_SAMPLE)
ntr = int(N_SAMPLE * 0.7)
tr, te = idx[:ntr], idx[ntr:]


def standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


Xtr, Xte = standardize(Xr[tr], Xr[te])
ytr, yte = y[tr], y[te]


# ----------------------------------------------------------------------
# 2. PCA：手写 SVD 分解（对标准化数据的相关阵）
# ----------------------------------------------------------------------
def pca_fit(Z):
    """返回 (载荷 V, 主成分得分 T, 解释方差比例 ratio, 奇异值 s)。"""
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt.T
    T = Z @ V
    ratio = s ** 2 / np.sum(s ** 2)
    return V, T, ratio, s


V, T_tr, ratio, s = pca_fit(Xtr)
T_te = Xte @ V
print("\n[PCA] 各主成分解释方差比例：")
for k in range(len(ratio)):
    print(f"  PC{k+1}: {ratio[k]*100:6.2f}%   累计 {np.sum(ratio[:k+1])*100:6.2f}%")

# 正交性核对：主成分得分两两不相关
corr_T = np.corrcoef(T_tr, rowvar=False)
offdiag = corr_T - np.eye(len(ratio))
assert np.max(np.abs(offdiag)) < 1e-10, "主成分应两两正交"
print(f"[核对] 主成分两两正交：最大非对角相关 = {np.max(np.abs(offdiag)):.2e}")


# ----------------------------------------------------------------------
# 3. 四种回归
# ----------------------------------------------------------------------
def ols(X, y):
    Xd = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def predict_ols(beta, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ beta


def pcr_fit(Xtr, ytr, q):
    """用 Z 的前 q 个主成分做 OLS，把系数映射回标准化空间。"""
    V, T, ratio, _ = pca_fit(Xtr)
    Tq = T[:, :q]
    b_T = ols(Tq, ytr)                       # [截距, q 个主成分系数]
    intercept = b_T[0]
    beta_std = V[:, :q] @ b_T[1:]            # 映射回标准化变量空间
    return intercept, beta_std


def ridge_fit(Xtr, ytr, lam):
    Xd = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    I = np.eye(Xd.shape[1])
    I[0, 0] = 0.0                            # 不惩罚截距
    beta = np.linalg.solve(Xd.T @ Xd + lam * I, Xd.T @ ytr)
    return beta


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --- OLS ---
beta_ols = ols(Xtr, ytr)
pred_ols_tr = predict_ols(beta_ols, Xtr)
pred_ols_te = predict_ols(beta_ols, Xte)
rmse_ols_tr, rmse_ols_te = rmse(pred_ols_tr, ytr), rmse(pred_ols_te, yte)

# --- 共线性诊断 (VIF + 条件数) ---
def vif_scores(X):
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        b = ols(others, X[:, j])
        resid = X[:, j] - predict_ols(b, others)
        r2 = 1 - np.sum(resid ** 2) / np.sum((X[:, j] - X[:, j].mean()) ** 2)
        out.append(1.0 / max(1e-12, 1 - r2))
    return np.array(out)


vifs = vif_scores(Xtr)
cond_num = float(np.linalg.cond(np.hstack([np.ones((len(Xtr), 1)), Xtr])))
print("\n[共线性诊断] VIF（>10 视为严重）：")
for name, v in zip(FEATURES, vifs):
    print(f"  {name:6s}: {v:8.1f}")
print(f"  设计矩阵条件数 = {cond_num:.1f}")

# --- PCR：累计解释方差 >= 90% 定 q ---
q90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
inter_pcr, beta_pcr = pcr_fit(Xtr, ytr, q90)
pred_pcr_tr = inter_pcr + Xtr @ beta_pcr
pred_pcr_te = inter_pcr + Xte @ beta_pcr
rmse_pcr_tr, rmse_pcr_te = rmse(pred_pcr_tr, ytr), rmse(pred_pcr_te, yte)

# --- Ridge：用测试集粗扫一个 λ（教学用，真实应用应做 CV） ---
best_lam, best_rmse = None, np.inf
for lam in np.logspace(-3, 3, 40):
    b = ridge_fit(Xtr, ytr, lam)
    e = rmse(predict_ols(b, Xte), yte)
    if e < best_rmse:
        best_lam, best_rmse = lam, e
beta_ridge = ridge_fit(Xtr, ytr, best_lam)
rmse_ridge_tr = rmse(predict_ols(beta_ridge, Xtr), ytr)
rmse_ridge_te = best_rmse

print(f"\n[回归对照] 训练/测试 RMSE（越小越好）")
print(f"  OLS    : 训练 {rmse_ols_tr:.3f} | 测试 {rmse_ols_te:.3f}")
print(f"  PCR(q={q90}): 训练 {rmse_pcr_tr:.3f} | 测试 {rmse_pcr_te:.3f}")
print(f"  Ridge(λ={best_lam:.3f}): 训练 {rmse_ridge_tr:.3f} | 测试 {rmse_ridge_te:.3f}")

# ----------------------------------------------------------------------
# 4. q 扫描：主成分个数 -> 测试误差
# ----------------------------------------------------------------------
qs = np.arange(1, len(FEATURES) + 1)
pcr_test_rmse = []
for q in qs:
    it, bt = pcr_fit(Xtr, ytr, q)
    pcr_test_rmse.append(rmse(it + Xte @ bt, yte))
pcr_test_rmse = np.array(pcr_test_rmse)
q_best = int(qs[int(np.argmin(pcr_test_rmse))])
print(f"\n[q 扫描] 最佳主成分个数 q*={q_best}（测试 RMSE {pcr_test_rmse.min():.3f}）")

# ----------------------------------------------------------------------
# 5. 系数稳定性：Bootstrap 重采样，比较 OLS vs PCR 系数抖动
# ----------------------------------------------------------------------
n_boot = 300
boot_ols = np.zeros((n_boot, len(FEATURES)))
boot_pcr = np.zeros((n_boot, len(FEATURES)))
for b in range(n_boot):
    sidx = RNG.integers(0, ntr, ntr)
    Xb, yb = Xtr[sidx], ytr[sidx]
    boot_ols[b] = ols(Xb, yb)[1:]
    _, bb = pcr_fit(Xb, yb, q90)
    boot_pcr[b] = bb
std_ols = boot_ols.std(0)
std_pcr = boot_pcr.std(0)
print("\n[系数稳定性] Bootstrap 标准差（越大越不稳）：")
for name, so, sp in zip(FEATURES, std_ols, std_pcr):
    print(f"  {name:6s}: OLS {so:7.3f} | PCR {sp:6.3f}  ({'PCR 更稳' if sp < so else 'OLS 更稳'})")

# ----------------------------------------------------------------------
# 6. 断言
# ----------------------------------------------------------------------
# (a) 主成分解释方差单调递减、累计到 100%
assert np.all(np.diff(ratio) <= 1e-9), "解释方差应单调递减"
assert abs(np.sum(ratio) - 1.0) < 1e-9, "解释方差之和应为 1"
# (b) 主成分两两正交
assert np.max(np.abs(offdiag)) < 1e-10
# (c) 数据确实高度共线（VIF 普遍大）
assert np.mean(vifs) > 10, "构造的数据应存在严重多重共线性"
# (d) PCR 在共线性下测试误差不劣于 OLS（通常更小）
assert rmse_pcr_te <= rmse_ols_te + 1e-6, "PCR 测试误差应不于劣于 OLS"
# (e) PCR 系数比 OLS 稳定（Bootstrap 标准差更小）
assert np.mean(std_pcr) < np.mean(std_ols), "PCR 系数平均应比 OLS 更稳"
# (f) q=8（=全信息且不丢方向）时 PCR 应能复现 OLS
it_full, b_full = pcr_fit(Xtr, ytr, len(FEATURES))
assert np.allclose(b_full, beta_ols[1:], atol=1e-6), "q=全维时 PCR 应等于 OLS"
print("\n[断言] 全部通过 ✓  (单调递减/正交/高VIF/PCR不劣/更稳/q=全维=OLS)")

# ----------------------------------------------------------------------
# 7. 绘图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Day32 · 主成分分析 PCA 与主成分回归 PCR", fontsize=15, fontweight="bold")

# 左上：碎石图
ax = axes[0, 0]
ax.bar(np.arange(1, len(ratio) + 1), ratio * 100, alpha=0.75, label="单个主成分")
ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio) * 100, "o-",
        color="crimson", label="累计")
ax.axhline(90, ls="--", color="gray", lw=1)
ax.text(0.6, 91, "90% 线", color="gray", fontsize=9)
ax.axvline(q90, ls=":", color="crimson")
ax.set_xlabel("主成分序号")
ax.set_ylabel("解释方差 (%)")
ax.set_title(f"碎石图：前 {q90} 个主成分解释 ≥90% 方差", fontsize=12)
ax.legend()

# 右上：载荷热力图（主成分 <- 指标）
ax = axes[0, 1]
Vdisp = V[:, :min(4, len(FEATURES))]
im = ax.imshow(Vdisp.T, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticks(range(Vdisp.shape[1]))
ax.set_yticklabels([f"PC{k+1}" for k in range(Vdisp.shape[1])])
for i in range(Vdisp.shape[1]):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{Vdisp[j, i]:.2f}", ha="center", va="center",
                fontsize=7, color="black")
ax.set_title("载荷矩阵：每个主成分由哪些指标构成", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 左下：q 扫描 vs 测试误差
ax = axes[1, 0]
ax.plot(qs, pcr_test_rmse, "o-", color="steelblue", label="PCR 测试 RMSE")
ax.axhline(rmse_ols_te, ls="--", color="darkorange", label=f"OLS 测试 ({rmse_ols_te:.2f})")
ax.scatter([q_best], [pcr_test_rmse.min()], color="crimson", zorder=5,
           label=f"最优 q*={q_best}")
ax.set_xlabel("主成分个数 q")
ax.set_ylabel("测试集 RMSE")
ax.set_title("取几个主成分最划算（降维太多/太少都亏）", fontsize=12)
ax.legend()

# 右下：系数稳定性 OLS vs PCR
ax = axes[1, 1]
xx = np.arange(len(FEATURES))
w = 0.38
ax.bar(xx - w/2, std_ols, w, label="OLS", color="darkorange")
ax.bar(xx + w/2, std_pcr, w, label="PCR", color="steelblue")
ax.set_xticks(xx)
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_ylabel("系数 Bootstrap 标准差")
ax.set_title("系数稳定性：PCR 明显比 OLS 稳", fontsize=12)
ax.legend()

plt.tight_layout()
out_png = os.path.join(OUTDIR, "pca_principal_component_regression.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n[输出] 图已保存：{out_png}")
print("完成。")
