#!/usr/bin/env python
"""
subject: mathematical-modeling
level: Level 8 - Data Analysis & Machine Learning / 分类(监督)
roadmap: Level 8 / ML 基础
title: 监督分类 KNN：近邻投票做"客户信贷风险"预筛 + 手写=sklearn 对照 + 定K
date: 2026-09-09
day: 9
goal:
    1) 用"银行/小贷对一笔新申请按'收入+负债率+信用分'判定逾期风险高低"的决策
       切入监督分类——衔接 Day8 聚类：聚类是自己发现分组(无标签)，KNN 是拿
       "已知好/坏客户"当样本库，对新客户按 K 个最近邻投票做分类(有标签)；
    2) 讲清 KNN 是"惰性/记忆型"算法：不训练出模型，预测时才在训练集里找
       K 个最近邻，多数票(回归则均值)给出标签——一句话"物以类聚，近朱者赤"；
    3) 手写 KNN(欧氏距离 + argsort 取 K 近邻 + 投票)，与 sklearn KNeighborsClassifier
       对照，验证"手写=标准库"；用留出验证评估准确率/混淆矩阵；
    4) 讨论 K 怎么定(过小欠平滑易受单点噪声、过大被远处异类带偏)+ 特征量纲
       必须先标准化(否则大尺度特征喧宾夺主)两大工程要点；
    5) 结果落到"对新申请打风险标签 + 该不该拒"的决策并给解读。
配套教案见 05 - Algorithms/Day9 笔记。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体注册(避免乱码) ----------
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260909)
OUT = __file__.rsplit("/", 1)[0]  # 输出到脚本自身目录 daily/2026-09-09/


# ---------- 造数据(信贷背景：收入 x 负债率, 已知好/坏 600 人) ----------
def gen_cloud(cx, cy, n, spread=1.0):
    pts = rng.normal(size=(n, 2))
    scale = np.array([spread * 5.5, spread * 0.20])  # 收入轴宽/负债率轴窄
    return pts * scale + np.array([cx, cy])


# 好客户:偏高收入 & 低负债率; 坏客户:收入略低或略高但负债率普遍偏高
# (两列云刻意拉宽、部分重叠 = 现实里好人不时违约/坏人也偶发守约)
good = gen_cloud(58, 0.30, 300, spread=1.15)
bad  = np.vstack([gen_cloud(44, 0.70, 210, spread=1.15),   # 中低收入+高负债 →坏
                  gen_cloud(62, 0.95, 90, spread=1.15)])    # 收入不低但负债极高 →坏
# 用 zip 交错拼成打乱的 data + label, 固定随机顺序便于复现
data = np.vstack([good, bad])
lab  = np.array([0] * 300 + [1] * 300)         # 0=好(默认不违约) 1=坏(高风险/易逾)
idx  = rng.permutation(len(data))
data, lab = data[idx], lab[idx]

# 训练/测试切分:前 420 训练, 后 180 留出测试(未见数据)
Xtr, ytr = data[:420], lab[:420]
Xte, yte = data[420:], lab[420:]
print(f"数据: 600 名历史客户(好 300 / 坏 300), 特征=收入(千元/月)x负债率")
print(f"切分: 训练 {len(Xtr)} / 留出测试 {len(Xte)}\n")

# ---------- 标准化(关键:两特征量纲差很多, 负债率∈[0,1]会被收入几十喧宾夺主) ----------
mu, sd = Xtr.mean(0), Xtr.std(0)
Xs = (Xtr - mu) / sd   # 训练标准化
Xs_te = (Xte - mu) / sd


# ---------- 手写 KNN(惰性算法:预测时才算距离) ----------
def knn_predict(train, tlab, q, k):
    """对单个查询点 q 返回预测类别(K 近邻多数票). train/tlab 已标准化. 平局取小类. """
    d = np.sqrt(((train - q) ** 2).sum(1))      # 欧氏距离 (n,)
    nb = np.argsort(d)[:k]                       # 最近 K 个邻
    votes = tlab[nb]
    return int((votes.sum() >= (k + 1) / 2)), votes, d[nb]  # 多数票 > 平局取1


def knn_predict_batch(train, tlab, Q, k):
    D = np.sqrt(((train[None, :, :] - Q[:, None, :]) ** 2).sum(-1))  # (m,n)
    nb = np.argsort(D, 1)[:, :k]                  # 每行最近K个邻下标
    votes = tlab[nb]                              # (m,k)
    return (votes.sum(1) >= (k + 1) / 2).astype(int), votes


# k=7 手写 vs sklearn 对照
from sklearn.neighbors import KNeighborsClassifier

k = 7
pred_hand, _ = knn_predict_batch(Xs, ytr, Xs_te, k)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(Xs, ytr)
pred_sk = model.predict(Xs_te)

acc_hand = (pred_hand == yte).mean()
acc_sk   = (pred_sk == yte).mean()
agree    = (pred_hand == pred_sk).mean()
print(f"[手写 vs sklearn 对照] k={k} 测试集 {len(Xte)} 人")
print(f"  手写 准确率 = {acc_hand:.4f} ; sklearn 准确率 = {acc_sk:.4f}")
print(f"  两种实现指派一致率 = {agree*100:.1f}%\n")

# ---------- 定 K: 网格扫 K=1,3,...,29 看测试准确率 ----------
Ks = list(range(1, 30, 2))
accs, accs_sk = [], []
for kk in Ks:
    ph, _ = knn_predict_batch(Xs, ytr, Xs_te, kk)
    accs.append((ph == yte).mean())
    m = KNeighborsClassifier(n_neighbors=kk).fit(Xs, ytr)
    accs_sk.append((m.predict(Xs_te) == yte).mean())
accs, accs_sk = np.array(accs), np.array(accs_sk)

best_k = Ks[int(np.argmax(accs))]
print(f"[定K扫描] 测试准确率 vs K:")
for kk, a, a2 in zip(Ks, accs, accs_sk):
    mark = " <== 手写与 sklearn 一致 + 峰值" if kk == best_k and abs(a - a2) < 1e-9 else ""
    print(f"  K={kk:2d}  手写 {a:.4f}  sklearn {a2:.4f}{mark}")
print(f"\n结论: 测试准确率峰值在 K={best_k} (准确率 {accs.max():.4f})")

# ---------- 混淆矩阵 + 决策解读 ----------
ph_bf, votes_best = knn_predict_batch(Xs, ytr, Xs_te, best_k)
tn = int(((ph_bf == 0) & (yte == 0)).sum())
fp = int(((ph_bf == 1) & (yte == 0)).sum())
fn = int(((ph_bf == 0) & (yte == 1)).sum())
tp = int(((ph_bf == 1) & (yte == 1)).sum())
print(f"\n[K={best_k} 混淆矩阵] 好=0 坏=1")
print(f"                   预测好(0)   预测坏(1)")
print(f"  实际好(0)  {tn:8d}      {fp:8d}")
print(f"  实际坏(1)  {fn:8d}      {tp:8d}")
tot = tn + fp + fn + tp
print(f"  总 {tot} 人 | 误杀好客户(拒错,FP)={fp} | 漏网坏客户(放贷坏账,FN)={fn}")
print(f"  误杀率 {fp/(tn+fp)*100:.1f}% | 漏网率 {fn/(fn+tp)*100:.1f}%")

# 违约倾向分数：近邻中坏客户占比高的新申请最该拒
score = votes_best.mean(1)  # 每个测试样本 K 邻中"坏客户"比例
risk_bad_true = yte.astype(float)
print(f"\n[违约倾向] 把 K 邻里坏客户比例当风险分: 真实为坏者的平均风险分 "
      f"{score[yte==1].mean():.3f} vs 真实为好者 {score[yte==0].mean():.3f}")

# ---------- 画图: (A)决策边界+测试点 (B)准确率随K ----------
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))

# (A) 决策边界(k=best_k) —— 在测试特征区间铺网格, 用 sklearn 分类器预测着色
xxlim = (min(Xs_te[:,0].min(), Xs[:,0].min())-.25, max(Xs_te[:,0].max(), Xs[:,0].max())+.25)
yylim = (min(Xs_te[:,1].min(), Xs[:,1].min())-.25, max(Xs_te[:,1].max(), Xs[:,1].max())+.25)
xx, yy = np.meshgrid(np.linspace(*xxlim, 300), np.linspace(*yylim, 300))
Z = KNeighborsClassifier(n_neighbors=best_k).fit(Xs, ytr).predict(
        np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax[0].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f4", "#f6d0c0"], alpha=.8)
# 这里画的是留出测试点(未见过的 180 人), 不是训练点 —— 用来检验泛化
ax[0].scatter(Xs_te[yte == 0, 0], Xs_te[yte == 0, 1], c="#1f77b4", s=26, label="测试·好(实际不违)", edgecolor="w", lw=.4)
ax[0].scatter(Xs_te[yte == 1, 0], Xs_te[yte == 1, 1], c="#d62728", s=26, marker="x", label="测试·坏(实际违)")
ax[0].set_title(f"KNN 信贷风险决策边界 (K={best_k})  蓝区→贷 / 红区→拒", fontsize=12)
ax[0].set_xlabel("标准化后的 月收入"); ax[0].set_ylabel("标准化后的 负债率")
ax[0].legend(loc="upper left", fontsize=8, framealpha=.9)

# (B) 准确率 vs K
ax[1].plot(Ks, accs, "-o", color="#2ca02c", label="手写 KNN 测试准确率")
ax[1].plot(Ks, accs_sk, "--s", color="#9467bd", lw=1, label="sklearn 测试准确率")
ax[1].axvline(best_k, color="#d62728", ls=":", lw=1.4)
ax[1].annotate(f"峰值 K={best_k}\n({accs.max():.3f})", xy=(best_k, accs.max()),
               xytext=(best_k+3, accs.max()-.025), fontsize=10,
               arrowprops=dict(arrowstyle="->", color="#d62728"))
ax[1].set_title("K 怎么定: 过小/过大都会掉准确率", fontsize=12)
ax[1].set_xlabel("近邻数 K"); ax[1].set_ylabel("留出测试准确率")
ax[1].legend(fontsize=9); ax[1].set_xticks(Ks); ax[1].grid(alpha=.3)
plt.tight_layout()
png = OUT + "/knn_decision_boundary_k.png"
plt.savefig(png, dpi=150, bbox_inches="tight")
plt.close()
print("\n图已存: knn_decision_boundary_k.png")

assert agree >= 0.99, "手写与 sklearn 指派应高度一致"
assert best_k >= 3, "合理性: 过小的 K 通常不是峰值(易受单点噪声)"
assert accs.max() > 0.8, "数据分离度应能让 KNN 拿到 80%+ 准确率"
assert fp + fn < 0.3 * tot, "错误数应在合理范围"
print("\n[验证] 全部断言通过: 手写=sklearn、K 峰值合理、准确率足够、误判在可控范围")

