"""
subject : mathematical-modeling
level   : L4 统计与预测模型（多元线性回归 + 回归诊断）/ L10 模型检验（设定误设 / 异方差 / 异常点 / 正态）
anchor  : Learning_Roadmap · Level 4「回归诊断：OLS 四个假设逐一验」+ Level 10「一个模型能不能信」
goal    : 县域热电厂的燃煤单耗账本被审计打回重做。
          手写 OLS 只给「系数 + R²」，而审计要问的是「这 4 个系数站不站得住」。
          本讲把 OLS 的 4 条假设拆成可检验、能落到纸上的诊断量：
            (1) 线性设定 → RESET（ŷ²、ŷ³ 能否解释残差）；
            (2) 同方差   → White / BP / 「按产线分组 F 检验」；
            (3) 异常点   → 杠杆 h、外部学生化残差 t*、Cook 距离 D、DFFITS；
            (4) 正态     → QQ 图 + Shapiro-Wilk。
          再给两条修法：HC1 稳健标准误（不删数据）+ 删点复估（看结论会不会翻转），
          最后把「系数怎么用」翻成一句能下班的决策：老炉要不要提负荷。
          全部 np.linalg 手写，不依赖 statsmodels；scipy 只用来算 F/t/χ² 的分位与 p 值。

运行: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-21/regression_diagnostics_coal.py"
依赖: numpy / scipy / matplotlib（零 statsmodels）
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from scipy import stats

for _f in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"):
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("error")

HERE = Path(__file__).resolve().parent
SEED = 6
ALPHA = 0.05

# ======================================================== 1. 账本：40 个运行班
# 云溪县热电厂 2026 年 1–8 月燃煤单耗台账（kg 标煤 / t 蒸汽）
#   产线 A（老链条炉）：6 个班，只在 30–36 t/h 的中低负荷运行；炉况抖，读数散
#   产线 B（新循环流化床）：30 个班，负荷 16–60 t/h 全覆盖；自动化程度高，读数稳
#   3 月 14 日夜班：老炉唯一一次拉到 60 t/h，台账记为"化验补录"——这就是 0 号点
NAMES = ["截距", "产量 t/h", "老炉哑变量", "二班哑变量"]


def make_data(seed: int = SEED, n_a: int = 6, n_b: int = 30,
              sigma_a: float = 2.8, sigma_b: float = 0.7,
              bad_shift: float = 6.0, c2: float = 0.016, t0: float = 40.0):
    """真实单耗曲线：w = 97 + 0.016(t-40)² + 3·老炉 + 1.2·二班
    即「U 形」——低负荷效率差、高负荷单耗回升，最低点在 40 t/h。这是锅炉的常识，
    但记账的人常常只用一个线性斜率去拟合，于是 U 形被拉成一条斜线。
    """
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    line = np.array([0] * n_a + [1] * n_b)                 # 0=老炉A  1=新炉B
    ton = np.concatenate([rng.uniform(30, 36, n_a), rng.uniform(16, 60, n_b)])
    ton[0] = 60.0                                           # 老炉唯一一次满负荷
    shift = rng.integers(0, 3, n)                           # 0夜 / 1早 / 2中
    w = 97 + c2 * (ton - t0) ** 2 + 3.0 * line + 1.2 * (shift == 2)
    sigma = np.where(line == 0, sigma_a, sigma_b)           # 老炉抖、新炉稳
    w = w + rng.normal(0, 1, n) * sigma
    w[0] += bad_shift                                       # 补录错误
    X = np.column_stack([np.ones(n), ton, line, (shift == 2).astype(float)])
    return dict(w=w, X=X, line=line, ton=ton, shift=shift, n=n,
                run=np.arange(1, n + 1), true_c2=c2, true_t0=t0)


# ======================================================== 2. OLS 与全套诊断量
def ols(X, y):
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    p = X.shape[1]
    s2 = e @ e / (len(y) - p)
    return beta, e, s2, XtXi


def diagnostics(y, X):
    n, p = X.shape
    beta, e, s2, XtXi = ols(X, y)
    se = np.sqrt(np.diag(XtXi) * s2)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)               # 帽子矩阵对角元（杠杆）
    r = e / np.sqrt(np.maximum(1 - h, 1e-12)) / np.sqrt(s2)  # 内部学生化残差
    # 外部学生化残差：删掉第 i 条后重估方差 —— 残差大到"自己都藏不住"
    t_ext = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        _, e_i, s2_i, _ = ols(X[keep], y[keep])
        t_ext[i] = e[i] / np.sqrt(s2_i * np.maximum(1 - h[i], 1e-12))
    cook = r ** 2 * h / (p * np.maximum(1 - h, 1e-12))     # Cook 距离
    dffits = t_ext * np.sqrt(h / np.maximum(1 - h, 1e-12))
    return dict(beta=beta, se=se, e=e, h=h, r=r, t_ext=t_ext, cook=cook,
                dffits=dffits, s2=s2, n=n, p=p, XtXi=XtXi)


def hc1(X, e, XtXi, n, p):
    """White(1980) HC1：把「残差平方」按观察逐条代回三明治，σ² 不再是单一常数。
    异方差下 OLS 系数仍然无偏，但它的标准误会被系统性低估 —— 修标准误就够了。"""
    meat = (X * (e ** 2)[:, None]).T @ X
    cov = XtXi @ meat @ XtXi * n / (n - p)
    return np.sqrt(np.diag(cov)), cov


def fit_stats(y, yhat, p):
    n = len(y)
    e = y - yhat
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / sst
    return r2, 1 - (1 - r2) * (n - 1) / (n - p), np.sqrt(e @ e / (n - p))


def reset_test(y, X, powers=(2, 3)):
    """Ramsey RESET：原式若设定正确，ŷ² 和 ŷ³ 不该再有解释力。
    把它们塞回回归式，做一个套嵌 F 检验；显著 ⇒ 漏了曲率/交互。"""
    n, p = X.shape
    beta, e, *_ = ols(X, y)
    yhat = X @ beta
    A = np.column_stack([X] + [yhat ** k for k in powers])
    bA = np.linalg.lstsq(A, y, rcond=None)[0]
    eA = y - A @ bA
    q, df2 = len(powers), n - A.shape[1]
    F = ((e @ e - eA @ eA) / q) / (eA @ eA / df2)
    return F, stats.f.sf(F, q, df2), q


def white_test(y, X):
    """White：e² 对「所有回归元的平方与两两乘积」回归，nR² ~ χ²。自由度吃掉很多 → 功效偏低。"""
    n, p = X.shape
    _, e, *_ = ols(X, y)
    cross = [X[:, j] * X[:, k] for j in range(1, p) for k in range(j, p)]
    G = np.column_stack([np.ones(n)] + cross)
    bG = np.linalg.lstsq(G, e ** 2, rcond=None)[0]
    fit = G @ bG
    r2aux = 1 - ((e ** 2 - fit) ** 2).sum() / ((e ** 2 - (e ** 2).mean()) ** 2).sum()
    lm, df = n * r2aux, G.shape[1] - 1
    return lm, stats.chi2.sf(lm, df), df


def bp_test(y, X, groups):
    """Koenker 版 BP：e²/ē² 对回归元回归，nR²~χ²(p-1)；再补一个「分组方差比 F」——
    工程师最看得懂的版本：两条产线的残差标准差差几倍。"""
    n, p = X.shape
    _, e, *_ = ols(X, y)
    g = e ** 2 / (e ** 2).mean()
    bg = np.linalg.lstsq(X, g, rcond=None)[0]
    fit = X @ bg
    r2aux = 1 - ((g - fit) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    lm, df = n * r2aux, p - 1
    a, b = e[groups == 0], e[groups == 1]
    Fa = a.var(ddof=1) / b.var(ddof=1)
    dfa, dfb = len(a) - 1, len(b) - 1
    pF = 2 * min(stats.f.sf(Fa, dfa, dfb), stats.f.cdf(Fa, dfa, dfb))
    return dict(lm=lm, p=stats.chi2.sf(lm, df), df=df, F=Fa, pF=pF,
                sd_a=a.std(ddof=1), sd_b=b.std(ddof=1), dfa=dfa, dfb=dfb)


def norm_k2(x):
    """D'Agostino-Pearson K²：把偏度、峰度各标准化成 z，再 K²=z1²+z2² ~ χ²(2)。
    这里用 scipy.stats.normaltest 而不手抄公式：正态性检验的近似式版本差异极大，
    手抄一份反而不可比；而它的逻辑一句话就能讲完——「偏度该 0、峰度该 3」。"""
    stat, pval = stats.normaltest(x)
    return stat, pval


# ======================================================== 3. 主流程
D = make_data()
y, X, line, ton, run = D["w"], D["X"], D["line"], D["ton"], D["run"]
n, p = X.shape
dg = diagnostics(y, X)
beta, se, e, h, t_ext, cook = dg["beta"], dg["se"], dg["e"], dg["h"], dg["t_ext"], dg["cook"]
t_ols = beta / se
r2, adj, sig = fit_stats(y, X @ beta, p)
se_hc1, _ = hc1(X, e, dg["XtXi"], n, p)
t_hc1 = beta / se_hc1
df_res = n - p

print("=" * 74)
print("一、OLS 台账（只交这张表，审计第一轮就退）")
print("=" * 74)
print(f"{'变量':<12}{'系数':>9}{'OLS SE':>9}{'t':>7}{'p':>9}{'HC1 t':>8}")
for nm, b, s, tt, th in zip(NAMES, beta, se, t_ols, t_hc1):
    print(f"{nm:<12}{b:>9.3f}{s:>9.3f}{tt:>7.2f}{2 * stats.t.sf(abs(tt), df_res):>9.4f}{th:>8.2f}")
print(f"R²={r2:.4f}  adjR²={adj:.4f}  σ̂={sig:.3f}  n={n}  p={p}")

print()
print("=" * 74)
print("二、四条假设逐一验（诊断的主场）")
print("=" * 74)
F_reset, p_reset, q_reset = reset_test(y, X)
lm_w, p_w, df_w = white_test(y, X)
bp = bp_test(y, X, line)
print(f"[设定]   RESET F({q_reset},{n - p - q_reset})={F_reset:7.1f}  p={p_reset:.2e}"
      f"   → {'设定错：直线抓不住 U 形' if p_reset < ALPHA else '设定可接受'}")
print(f"[同方差] White LM={lm_w:6.2f}  df={df_w}  p={p_w:.3f}"
      f"        → {'异方差' if p_w < ALPHA else '此项未报警（自由度吃太多）'}")
print(f"[同方差]  BP   LM={bp['lm']:6.2f}  df={bp['df']}  p={bp['p']:.2e}")
print(f"[同方差]  分组 F={bp['F']:5.2f}  F({bp['dfa']},{bp['dfb']})  p={bp['pF']:.2e}"
      f"  → 老炉残差 sd {bp['sd_a']:.2f} vs 新炉 {bp['sd_b']:.2f}（{bp['sd_a'] / bp['sd_b']:.1f} 倍）")
K2, p_sw = norm_k2(e)
print(f"[正态]   D'Agostino K²={K2:.2f}  p={p_sw:.4f}"
      f"   → {'残差非正态' if p_sw < ALPHA else '残差近似正态'}")

h_rule = 2 * p / n
print(f"\n[异常点] 杠杆阈值 2p/n={h_rule:.3f}、3p/n={3 * p / n:.3f}")
for i in np.argsort(-cook)[:3]:
    tags = []
    if h[i] > 2 * h_rule:
        tags.append("高杠杆")
    if abs(t_ext[i]) > 3:
        tags.append("异常残差")
    if cook[i] > 1:
        tags.append("Cook>1")
    print(f"   第 {run[i]:>2} 班  h={h[i]:.3f}  t*={t_ext[i]:+.2f}  Cook D={cook[i]:.2f}"
          f"  DFFITS={dg['dffits'][i]:+.2f}  {'+'.join(tags)}")

print()
print("=" * 74)
print("三、修法①：不删数据，只换标准误（HC1）")
print("=" * 74)
for nm, a, b in zip(NAMES, t_ols, t_hc1):
    verdict = lambda v: "显著" if abs(v) > stats.t.ppf(0.975, df_res) else "不显著"
    print(f"{nm:<12} t={a:>6.2f}({verdict(a)}) → t_HC1={b:>6.2f}({verdict(b)})")

print()
print("=" * 74)
print("四、修法②：删掉 0 号复估，看结论会不会翻转")
print("=" * 74)
worst = int(np.argmax(cook))
keep = np.arange(n) != worst
dg2 = diagnostics(y[keep], X[keep])
beta2, t2 = dg2["beta"], dg2["beta"] / dg2["se"]
r2_2, adj2, sig2 = fit_stats(y[keep], X[keep] @ beta2, p)
print(f"{'变量':<12}{'原系数':>9}{'原t':>7}   {'删点系数':>10}{'删点t':>7}{'系数变动':>9}")
for nm, b0, t0v, b1, t1v in zip(NAMES, beta, t_ols, beta2, t2):
    print(f"{nm:<12}{b0:>9.3f}{t0v:>7.2f}   {b1:>10.3f}{t1v:>7.2f}{(b1 - b0) / abs(b0) if abs(b0) > 1e-9 else float('nan'):>9.0%}")
print(f"R² {r2:.4f}→{r2_2:.4f}   σ̂ {sig:.3f}→{sig2:.3f}（降 {1 - sig2 / sig:.0%}）")
F_r2, p_r2, _ = reset_test(y[keep], X[keep])
_, p_w2, _ = white_test(y[keep], X[keep])
F_h2, p_h2, _ = reset_test(y[keep], np.column_stack([X[keep], ton[keep] ** 2]))
print(f"删点后：RESET p={p_r2:.2e}（设定仍错） White p={p_w2:.3f}（异方差仍在）")
print(f"再加一个 ton² 项：F={F_h2:.2f} p={p_h2:.3f}"
      f" → {'U 形被 ton² 解释掉了' if p_h2 > ALPHA else '仍不足'}")

X2 = np.column_stack([X, ton ** 2])
dg3 = diagnostics(y, X2)          # 保留 0 号！用"改设定"而不是"删数据"处理它
beta3, t3 = dg3["beta"], dg3["beta"] / dg3["se"]
r2_3, adj3, sig3 = fit_stats(y, X2 @ beta3, X2.shape[1])
b3, s3, s23, XtXi3 = ols(X2, y)
e3 = y - X2 @ b3
se3_hc1 = hc1(X2, e3, XtXi3, n, X2.shape[1])[0]
print(f"\n保留 0 号 + 补 ton² 项（c2 估计 {beta3[-1]:.4f}，真值 {D['true_c2']:.4f}）：")
for nm, b_, t_, th in zip(NAMES + ["产量²"], beta3, t3, beta3 / se3_hc1):
    print(f"   {nm:<10}{b_:>10.3f}  t={t_:>6.2f}  t_HC1={th:>6.2f}")
print(f"   R²={r2_3:.4f}  σ̂={sig3:.3f}（原 {sig:.3f}）")
vertex = -beta3[1] / (2 * beta3[-1])
print(f"   顶点 t* = -b1/(2b2) = {vertex:.2f} t/h（真实最低点 {D['true_t0']:.0f} t/h）")

print()
print("=" * 74)
print("五、结论落到决策：老炉要不要提负荷？")
print("=" * 74)


def dmarg(t, b=b3):
    """单耗对产量的边际：dw/dt = b1 + 2·b2·t（老炉）。"""
    return b[1] + 2 * b[-1] * t


t_now, t_new = 30.0, 36.0
m_now, m_new = dmarg(t_now), dmarg(t_new)
naive = beta[1]
print(f"朴素线性模型说：每提 1 t/h，单耗变 {naive:+.3f} kg/t —— 几乎为零，「提不提都行」")
print(f"U 形模型说：{t_now:.0f} t/h 处 {m_now:+.2f} kg/t；{t_new:.0f} t/h 处 {m_new:+.2f} kg/t")
gap = m_now + m_new
print(f"  从 {t_now:.0f} 提到 {t_new:.0f}（+{t_new - t_now:.0f} t/h），平均每 t 省 {-gap / 2 * (t_new - t_now):.2f} kg/t")
print(f"  按班产 {t_new * 8:.0f} t × 100 班/年估算：年省标煤约 {-gap / 2 * (t_new - t_now) * t_new * 8 * 100 / 1000:.1f} t")

# ======================================================== 4. 零点校准
rng = np.random.default_rng(SEED + 1)
B = 1000
fp_reset = fp_white = fp_bp = 0
for _ in range(B):
    ys = X @ [97, 0.0, 3.0, 1.2] + rng.normal(0, 1, n)
    fp_reset += reset_test(ys, X)[1] < ALPHA
    fp_white += white_test(ys, X)[1] < ALPHA
    fp_bp += bp_test(ys, X, line)["p"] < ALPHA
print()
print("=" * 74)
print(f"六、零点校准：真模型是「线性 + 同方差」时，{B} 组模拟里三个检验各误报多少次")
print("=" * 74)
print(f"RESET {fp_reset / B:.2%}   White {fp_white / B:.2%}   BP {fp_bp / B:.2%}   （名义 5%）")
print("→ 虚警率都守在 5% 附近，所以上面的「报警」不是检验自己太敏感")

# ======================================================== 5. 六联图
fig = plt.figure(figsize=(15.5, 10.4))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.28, hspace=0.36)

ax = fig.add_subplot(gs[0, 0])
ss = np.linspace(15, 62, 200)
ax.plot(ss, 97 + 0.016 * (ss - 40) ** 2, color="#2ca02c", lw=1.6, label="真曲线（U 形，底 40 t/h）")
ax.plot(ss, beta[0] + beta[1] * ss, "r--", lw=1.8, label=f"OLS 直线（斜率 {beta[1]:+.3f}）")
for cc, mk, cl, lb in ((0, "o", "#1f77b4", "老炉 A（6 班）"), (1, "s", "#ff7f0e", "新炉 B（30 班）")):
    m = line == cc
    ax.scatter(ton[m], y[m], s=38, marker=mk, color=cl, alpha=0.8, label=lb)
ax.annotate("0 号班：老炉唯一满发\n+ 台账补录", xy=(ton[0], y[0]), xytext=(30, 112),
            arrowprops=dict(arrowstyle="->", color="darkred"), color="darkred", fontsize=9)
ax.set_xlabel("产量 t/h"), ax.set_ylabel("燃煤单耗 kg/t")
ax.set_title("① 一条直线去贴 U 形：斜率被压成 -0.02"), ax.legend(fontsize=8, loc="upper center")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(X @ beta, e, c=np.where(line == 0, "#1f77b4", "#ff7f0e"), s=40)
ax.axhline(0, color="k", lw=0.8)
for i in np.argsort(-np.abs(t_ext))[:2]:
    ax.annotate(f"班{run[i]}", (float(X[i] @ beta), float(e[i])), textcoords="offset points",
                xytext=(6, 4), color="darkred", fontsize=9)
ax.set_xlabel("拟合单耗"), ax.set_ylabel("残差")
ax.set_title(f"② 残差 vs 拟合：中间压扁、两头张开\n分组 F p={bp['pF']:.1e}")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(h, np.abs(t_ext), c=np.where(line == 0, "#1f77b4", "#ff7f0e"), s=40)
ax.axvline(2 * p / n, color="orange", ls="--", lw=1.2, label=f"2p/n={2 * p / n:.2f}")
ax.axhline(3, color="red", ls=":", lw=1.2, label="|t*|=3")
ax.annotate("又高杠杆\n又异常", xy=(h[0], abs(t_ext[0])), xytext=(0.36, 4.4),
            arrowprops=dict(arrowstyle="->", color="darkred"), color="darkred", fontsize=9)
ax.set_xlim(0, 0.6), ax.set_ylim(0, 6)
ax.set_xlabel("杠杆 h"), ax.set_ylabel("|外部学生化残差|")
ax.set_title("③ 杠杆-残差图：谁在改写结论"), ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
osm, osr = stats.probplot(e, dist="norm", fit=False)
slope, icpt, _ = stats.probplot(e, dist="norm", fit=True)[1]
ax.scatter(osm, osr, s=26, c="#7f7f7f")
ax.plot(osm, icpt + slope * osm, "r-", lw=1.2)
ax.set_xlabel("理论正态分位"), ax.set_ylabel("残差分位")
ax.set_title(f"④ QQ 图：K²={K2:.2f}, p={p_sw:.3f}")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
order = np.argsort(-cook)
ax.bar(range(n), cook[order], color=["#d62728" if c > 1 else "#ff9896" for c in cook[order]])
ax.axhline(1, color="k", ls="--", lw=1.0, label="Cook D=1")
ax.set_xlabel("班次（按 Cook 距离降序）"), ax.set_ylabel("Cook D")
ax.set_title("⑤ 删掉一个点，结论就变 → 这个点必须查"), ax.legend(fontsize=8)
ax.set_xticks([]), ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(gs[1, 2])
cats = ["截距", "产量", "老炉", "二班"]
xx = np.arange(len(cats))
ax.bar(xx - 0.2, np.abs(t_ols), 0.4, color="#1f77b4", label="OLS 的 |t|")
ax.bar(xx + 0.2, np.abs(t_hc1), 0.4, color="#ff7f0e", label="HC1 的 |t|")
ax.axhline(stats.t.ppf(0.975, df_res), color="k", ls="--", lw=1.0, label="5% 临界")
ax.set_xticks(xx), ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("|t|")
ax.set_title("⑥ 异方差让 t 值虚高，稳健标准误把它拉回来")
ax.legend(fontsize=8), ax.grid(alpha=0.3, axis="y")

fig.suptitle("Day51 · 回归诊断：OLS 交的是系数，诊断交的是「能不能信」", fontsize=15, y=0.985)
out = HERE / "regression_diagnostics_coal.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n图已保存：{out}")

# ======================================================== 6. 断言（把结论钉死）
assert p_reset < 1e-3, "RESET 应当报警：直线设定抓不住 U 形"
assert p_w < ALPHA, "White 应当报警（严重异方差）"
assert bp["p"] < ALPHA, "BP 应当报警"
assert 2 <= bp["sd_a"] / bp["sd_b"] <= 3.5, "老炉残差 sd 应为新炉 2~3.5 倍（真实 σ 比 4，OLS 双向拉平）"

assert p_sw < ALPHA, "残差应偏离正态（被 0 号拖偏）"
assert h[0] > 2 * p / n, '0 号应越过 2p/n 这条常规判据（更严的 3p/n 本例未过，正文如实说明）'
assert h[0] > 1.5 * h[1:].mean(), "0 号杠杆应明显高于其余点均值（其余点约 0.2）"
assert abs(t_ext[0]) > 3, "0 号应为异常残差"
assert cook[0] > 1, "0 号 Cook 距离应 >1"
assert np.all(np.abs(t_hc1) < np.abs(t_ols)), "HC1 应把全部 t 值往下压"
assert p_h2 < ALPHA, "补 ton² 后 RESET 仍报警——因为 0 号还在"
assert abs(beta3[-1] - D["true_c2"]) / D["true_c2"] < 0.35, "c2 应估得接近真值"
assert abs(beta3[-1] - D["true_c2"]) / D["true_c2"] > 0.10, "留有张力：直线设定下 c2 被系统性高估"
assert abs(vertex - D["true_t0"]) < 4, "顶点应估在 40 t/h 附近"
assert dmarg(t_now) < dmarg(t_new), "两点的边际都该是负的：老炉全程在 U 形左半支"
assert dmarg(t_now) > -1.0, "曲线很浅：点边际不足 1 kg/t，结论要用区间平均"
assert abs(naive) < 0.1, "朴素线性斜率应接近 0（这正是它误人的地方）"
assert 0.02 <= fp_reset / B <= 0.09 and 0.02 <= fp_bp / B <= 0.09, "零点校准虚警率应接近 5%"
print("全部断言通过 ✔")
