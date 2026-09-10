# -*- coding: utf-8 -*-
"""
subject      : mathematical-modeling
level        : L8 (Learning Roadmap Level 8 — 数据分析与机器学习 / 监督分类)
anchor       : Day10 — 监督分类第二课：Logistic 回归(参数化概率模型)
              接 Day9 KNN(记忆型/惰性) → 本课"参数化"对照：一次性学出权重 w,b
goal         : 用历史客户(收入×负债率, 已知好坏)学一个"违约概率"评分卡，
               对新申请输出 p(违约)，并按业务阈值决定贷/拒。
               核心增量：①线性打分 z=w·x+b 经 sigmoid 压到 [0,1] 得概率；
               ②用极大似然(交叉熵)/梯度下降自己把 w 训出来(手写)并与 sklearn 对齐；
               ③概率天然带"置信度"，阈值可调以平衡两类错误代价；
               ④决策边界是直线(线性可分性) vs KNN 的不规则边界。

运行: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-10/logistic_credit_scoring.py"
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 中文字体（避免图里中文乱码）----
from matplotlib import font_manager as fm

_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        _name = fm.FontProperties(fname=_p).get_name()
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

# ============================================================
# 1. 造数据：600 名历史客户（月收入 万元 × 负债率），已知好坏
#    真实规律：收入越高越安全、负债率越高越危险；边界带噪声(借款不是数学)
# ============================================================
N = 600
income = rng.normal(1.6, 0.7, N).clip(0.3, 4.5)          # 月收入(万元)
dti = rng.uniform(0.05, 0.95, N)                          # 负债率 debt-to-income
risk_z = 2.4 * (dti - 0.5) - 1.5 * (income - 1.6)        # 潜在违约倾向
p_true = 1.0 / (1.0 + np.exp(-(risk_z - 0.05) / 0.55))    # 真实概率
y = (rng.uniform(0, 1, N) < p_true).astype(int)           # 1=违约(坏), 0=守约(好)
X = np.column_stack([income, dti])

print("=" * 64)
print("数据概览：600 名历史客户")
print(f"  违约(坏) {y.sum()} 人  |  守约(好) {N - y.sum()} 人")
print(f"  好客户 平均收入 {income[y==0].mean():.2f} 万  平均负债率 {dti[y==0].mean():.2f}")
print(f"  坏客户 平均收入 {income[y==1].mean():.2f} 万  平均负债率 {dti[y==1].mean():.2f}")

# ============================================================
# 2. 训练/测试切分 + 标准化（用训练集统计量，防泄漏）
# ============================================================
idx = rng.permutation(N)
n_tr = int(0.7 * N)
tr, te = idx[:n_tr], idx[n_tr:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr_s = (Xtr - mu) / sd
Xte_s = (Xte - mu) / sd
print(f"\n切分：训练 {len(tr)} / 测试 {len(te)}；标准化用训练集统计量(防泄漏)")

# ============================================================
# 3. 手写 Logistic 回归：sigmoid + 交叉熵 + 梯度下降
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def fit_logreg(X, y, lr=0.5, epochs=4000, l2=0.0):
    """手写批量梯度下降。返回 (w, b) 与损失历史。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    hist = []
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y                       # dL/dz  (交叉熵对 z 的梯度)
        grad_w = X.T @ err / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        # 交叉熵(对数似然取负)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        hist.append(loss)
    return w, b, np.array(hist)

w, b, hist = fit_logreg(Xtr_s, ytr, lr=0.5, epochs=4000)

def predict_proba(Xs, w, b):
    return sigmoid(Xs @ w + b)

# 可解释性：z = w·x + b，标准化后 w 的符号/大小 = 特征影响力
print("\n" + "=" * 64)
print("手写 Logistic 梯度下降拟合结果（标准化尺度）")
print(f"  w_收入 = {w[0]:+.3f}  (负 => 收入越高越不易违约)")
print(f"  w_负债率 = {w[1]:+.3f}  (正 => 负债率越高越易违约)")
print(f"  b = {b:+.3f}")
# 还原到原始单位，便于业务讲：log-odds 对 1 万元 / 100% 负债率的边际
w_inc_raw, w_dti_raw = w[0] / sd[0], w[1] / sd[1]
print(f"  原始单位 log-odds 斜率：收入 {w_inc_raw:+.3f}/万元, 负债率 {w_dti_raw:+.3f}/单位")
print(f"  交叉熵损失 首/末 = {hist[0]:.4f} → {hist[-1]:.4f}")

# ============================================================
# 4. 评估：准确率/混淆矩阵 + 与 sklearn 对齐
# ============================================================
def metrics(y_true, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    acc = (yhat == y_true).mean()
    # 混淆矩阵元素
    tp = int(((yhat == 1) & (y_true == 1)).sum())  # 判坏且真坏
    tn = int(((yhat == 0) & (y_true == 0)).sum())  # 判好且真好
    fp = int(((yhat == 1) & (y_true == 0)).sum())  # 判坏但真好(误拒)
    fn = int(((yhat == 0) & (y_true == 1)).sum())  # 判好但真坏(漏放)
    return acc, (tn, fp, fn, tp)

p_tr = predict_proba(Xtr_s, w, b)
p_te = predict_proba(Xte_s, w, b)
acc_tr, cm_tr = metrics(ytr, p_tr)
acc_te, cm_te = metrics(yte, p_te)
print("\n" + "=" * 64)
print("手写模型评估(阈值 0.5)")
print(f"  训练准确率 = {acc_tr:.4f}   测试准确率 = {acc_te:.4f}")
tn, fp, fn, tp = cm_te
print(f"  测试混淆矩阵: 好客户判对(tn)={tn}  误拒坏客户(fp)={fp}  漏放坏客户(fn)={fn}  坏客户判对(tp)={tp}")

# sklearn 交叉验证实现正确性
from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(C=1e6, max_iter=5000)  # 近似无正则，对齐手写
sk.fit(Xtr_s, ytr)
w_sk, b_sk = sk.coef_[0], sk.intercept_[0]
p_te_sk = sk.predict_proba(Xte_s)[:, 1]
acc_te_sk = ((p_te_sk >= 0.5).astype(int) == yte).mean()
print("\n与 sklearn 对照")
print(f"  w 手写 {np.round(w, 3)}  vs  sklearn {np.round(w_sk, 3)}")
print(f"  b 手写 {b:+.3f}  vs  sklearn {b_sk:+.3f}")
print(f"  测试准确率 手写 {acc_te:.4f}  vs  sklearn {acc_te_sk:.4f}")
assert abs(acc_te - acc_te_sk) < 1e-9, "手写与 sklearn 准确率不一致"
assert np.allclose(w, w_sk, atol=2e-2), "系数偏差过大"

# ============================================================
# 5. 阈值可调：两类错误的代价不同，判定线该动
# ============================================================
print("\n" + "=" * 64)
print("阈值权衡（业务代价不同，判定线可动）")
print(f"{'阈值':>6}{'准确率':>10}{'误拒好客户':>12}{'漏放坏客户':>12}")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    a, (tn_, fp_, fn_, tp_) = metrics(yte, p_te, thr)
    print(f"{thr:>6.1f}{a:>10.4f}{fp_:>12d}{fn_:>12d}")

# 概率分：坏客户概率显著高于好客户
print("\n违约概率分布(测试集)")
print(f"  真实好客户 平均 p(违约) = {p_te[yte==0].mean():.3f}")
print(f"  真实坏客户 平均 p(违约) = {p_te[yte==1].mean():.3f}")

# ============================================================
# 6. 画图：决策边界(概率热力) + sigmoid + 损失曲线
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# --- (a) 概率热力图 + 线性决策边界 ---
ax = axes[0]
gx = np.linspace(0.3, 4.5, 200)
gy = np.linspace(0.05, 0.95, 200)
GX, GY = np.meshgrid(gx, gy)
grid = np.column_stack([GX.ravel(), GY.ravel()])
grid_s = (grid - mu) / sd
P = predict_proba(grid_s, w, b).reshape(GX.shape)
cs = ax.contourf(GX, GY, P, levels=20, cmap="RdYlBu_r", alpha=0.85)
plt.colorbar(cs, ax=ax, label="预测违约概率 p")
ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=18, c="white",
           edgecolor="blue", label="守约(好)")
ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=18, c="black",
           marker="x", label="违约(坏)")
# 决策边界 p=0.5 => z=0 的直线
ax.contour(GX, GY, P, levels=[0.5], colors="k", linewidths=2)
ax.set_xlabel("月收入 (万元)")
ax.set_ylabel("负债率")
ax.set_title("(a) 违约概率热力图 + 线性决策边界(p=0.5)")
ax.legend(loc="upper right", fontsize=9)

# --- (b) sigmoid 曲线 ---
ax = axes[1]
zz = np.linspace(-6, 6, 300)
ax.plot(zz, sigmoid(zz), color="crimson", lw=2.5)
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.scatter(Xtr_s @ w + b, ytr, alpha=0.15, s=10, c="steelblue")
ax.set_xlabel("线性打分 z = w·x + b")
ax.set_ylabel("违约概率 σ(z)")
ax.set_title("(b) sigmoid：把线性打分压成概率")
ax.text(0.05, 0.92, "σ(z)=1/(1+e^{-z})", transform=ax.transAxes, fontsize=11)

# --- (c) 损失收敛 ---
ax = axes[2]
ax.plot(hist, color="darkgreen", lw=2)
ax.set_xlabel("梯度下降迭代")
ax.set_ylabel("交叉熵损失")
ax.set_title(f"(c) 手写 GD 收敛({hist[0]:.3f}→{hist[-1]:.3f})")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "logistic_credit_scoring.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n图已保存: {out_png}")

# ============================================================
# 7. 断言：验证结果合理性
# ============================================================
assert w[0] < 0, "收入系数应为负"
assert w[1] > 0, "负债率系数应为正"
assert acc_te > 0.8, "测试准确率过低"
assert p_te[yte == 1].mean() > p_te[yte == 0].mean(), "坏客户平均概率应更高"
assert hist[-1] < hist[0], "损失应下降"
print("\n[OK] 全部断言通过。")
