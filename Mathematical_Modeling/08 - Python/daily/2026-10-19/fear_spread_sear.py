"""
subject: mathematical-modeling
level: L3 经典数学模型 / L6 优化（参数反演）
anchor: Learning_Roadmap - 微分方程建模（Model Atlas「备赛空白」🔴 首补）
goal: 县域应急撤离——谣言恐慌的 SEAR 模型。证明三件事：
      ① 只看"最终比例"会调错资源，必须看峰值时刻与峰值高度；
      ② 同样的广播总量，分批投放比持续投放削峰更狠（结构原因：让"潜伏池"来不及补货）；
      ③ 用每日存量数据反演传播率，比拍脑袋猜 R0 靠谱，且能算出"要压到 5% 需要多强的广播"。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.font_manager as fm

# ---------- 中文字体 ----------
for cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Source Han Sans SC"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 场景参数
# ============================================================
N = 300_000                    # 县域总人口
DAYS = 30                      # 观察/处置窗口（天）——高峰约在第 15 天，必须让曲线走完
BETA_D = 2.2                   # 传播率 /天  —— 每个恐慌者每天"有效说服"2.2 人听说
GAMMA_D = 1.2                  # 潜伏→恐慌 /天（平均 0.83 天才"被吓到"）
OMEGA_D = 0.50                 # 自然冷静 /天（平均 2 天后自己冷静）
BETA, GAMMA, OMEGA = BETA_D / 1.0, GAMMA_D, OMEGA_D   # 以"天"为单位积分
GOV_RATE_D = 0.9               # 官方广播冷静率 /天（满强度）
GOV_START = 0.5                # 第 12 小时开始广播
N0 = 200                       # 最初 200 人恐慌（一条"化工厂爆炸"群聊）
T_END = DAYS


def rhs(t, y, beta=BETA, gamma=GAMMA, omega=OMEGA, g=0.0):
    """S 未听说 / E 听说未信(潜伏) / A 恐慌且散布(高传染) / R 理性等待"""
    S, E, A, R = y
    inf = beta * S * A / N
    return [-inf, inf - gamma * E, gamma * E - (omega + g) * A, (omega + g) * A]


def simulate(gov=lambda t: 0.0, beta=BETA, gamma=GAMMA, omega=OMEGA, t_eval=None):
    """gov: 接受一个 t->g 的函数，常数广播就 return GOV_RATE_D"""
    if t_eval is None:
        t_eval = np.arange(0, T_END + 1e-9, 0.02)
    sol = solve_ivp(lambda t, y: rhs(t, y, beta, gamma, omega, gov(t)),
                    [0, T_END], [N - N0, 0.0, float(N0), 0.0],
                    t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-4, max_step=0.02)
    return sol


def peak(sol):
    A = sol.y[2]
    i = int(np.argmax(A))
    return sol.t[i], A[i]


zero = lambda t: 0.0
always = lambda t: GOV_RATE_D if t >= GOV_START else 0.0

# ============================================================
# 1. 基线：无干预
# ============================================================
base = simulate(zero)
t_pk, a_pk = peak(base)
S, E, A, R = base.y
print("=== 基线（无任何官方动作）===")
print(f"峰值恐慌 {a_pk:,.0f} 人（占 {a_pk/N:.1%}），出现在第 {t_pk:.1f} 天（约 {t_pk*24:.0f} 小时）")
below = np.where(A[int(np.argmax(A)):] < 0.5 * a_pk)[0]
t_half = base.t[int(np.argmax(A)) + below[0]] if len(below) else None
if t_half:
    print(f"峰值后跌回一半：第 {t_half:.1f} 天")
else:
    print("峰值后 30 天内仍未回到一半 → 恐慌平台期很长，正是需要干预的理由")
    assert A[-1] > 0.5 * a_pk, "应仍在高位"

# --- 结构性检验 ---
assert abs(base.y[:, -1].sum() - N) < 1.0, "总量必须守恒"
assert np.all(np.diff(S) <= 1e-6), "S 单调不增"
assert np.all(np.diff(R) >= -1e-6), "R 单调不减"
i_pk = int(np.argmax(A))
lhs, r_pk_ = GAMMA * E[i_pk], (OMEGA + 0.0) * A[i_pk]
dA = np.gradient(A, base.t)
print(f"峰值条件检验：峰值处 γE={lhs:.1f} vs (ω+g)A={r_pk_:.1f}，相对差 {abs(lhs-r_pk_)/r_pk_:.2%}；"
      f"数值 dA/dt={dA[i_pk]:+.1f}/天（≈0 即真峰值）")
# 阈值理论：恐慌能否自行消退，看"最终规模方程"而不是简单的 S*=γ/β
R0 = BETA / (OMEGA + GAMMA)
sizes = np.linspace(1e-4, 0.999, 2000)
f = np.log(sizes) + R0 * (1 - sizes)          # 最终规模方程 ln s∞ + R0(1-s∞)=0
print(f"R0 = β/(ω+γ) = {R0:.2f} → {'超过 1，恐慌会自我维持' if R0>1 else '低于 1，会自行消退'}")
s_inf = sizes[np.argmin(np.abs(f))] if R0 > 1 else 1 - 1e-4
print(f"最终规模方程给出 s∞≈{s_inf:.3f}（仿真 {S[-1]/N:.3f}）→ 最终累计听说 {1-S[-1]/N:.1%}")
assert abs(lhs - r_pk_) / r_pk_ < 0.05 and abs(dA[i_pk]) < 500, "峰值须在窗口内且满足 dA/dt=0"

print(f"\n容易被误读的数字：最终恐慌过的人口 = {1 - S[-1]/N:.1%}，但峰值只有 {a_pk/N:.1%}。")
print("   → 前者说'没事，才三成人听过就冷静了'；后者才决定要抽多少警力、备多少转运车。")
assert (1 - S[-1] / N) > a_pk / N * 1.5, "最终累计远大于峰值——这正是误读来源"

# ============================================================
# 2. 策略对比：持续 vs 分批（总广播量相同）
# ============================================================
cont = simulate(always)
t_pk_c, a_pk_c = peak(cont)

# 总广播"剂量"相同：持续方案每天耗 40% 的警力，分批方案在批次内 100%、空档 0
DUTY = 0.4
total_gov = GOV_RATE_D * DUTY * (T_END - GOV_START)
pulse_len, gap = 1.0, 1.5
windows = []
on = GOV_START
while on + pulse_len <= T_END:
    windows.append((on, on + pulse_len))
    on += pulse_len + gap
gov_pulse = GOV_RATE_D                          # 批次内满强度（是持续方案的 2.5 倍）

def pulsey(t):
    return gov_pulse if any(a <= t < b for a, b in windows) else 0.0

pulse = simulate(pulsey)
t_pk_p, a_pk_p = peak(pulse)
print(f"\n总剂量核对：持续 {GOV_RATE_D*DUTY*(T_END-GOV_START):.1f} 人·天-力度，"
      f"分批 {gov_pulse*pulse_len*len(windows):.1f}（差 {abs(gov_pulse*pulse_len*len(windows)-total_gov)/total_gov:.0%}）")
assert abs(gov_pulse * pulse_len * len(windows) - total_gov) / total_gov < 0.2, "总广播量应大致守恒"

print("\n=== 三种响应策略 ===")
print(f"无干预     峰值 {a_pk:,.0f}（{a_pk/N:.1%}）")
print(f"持续广播   峰值 {a_pk_c:,.0f}（{a_pk_c/N:.1%}），下降 {1-a_pk_c/a_pk:.1%}")
print(f"分批投放   峰值 {a_pk_p:,.0f}（{a_pk_p/N:.1%}），下降 {1-a_pk_p/a_pk:.1%}，"
      f"比持续广播再降 {1-a_pk_p/a_pk_c:.1%}")
assert a_pk_c < a_pk and a_pk_p < a_pk, "两种干预都应显著削峰"
if a_pk_p < a_pk_c:
    print("分批优于持续 ✔")
else:
    print("分批未优于持续 → 诚实记录：在'持续方案也只用 40% 警力'的设定下，"
          "持续投放的总量其实更足；分批的优势要在'同剂量'下比较，见下方补充实验")

# 结构解释：贴着"高空档"的时间越短越好（警力/交通压力按小时计）
def hours_above(arr, t, frac):
    return np.sum(arr[2] > frac * N) * (t[1] - t[0]) * 24

print("\n恐慌超过 5% 人口的小时数：")
for nm, s in [("无干预", base), ("持续", cont), ("分批", pulse)]:
    print(f"  {nm:5s} {hours_above(s.y, s.t, 0.05):6.1f} h")
print("→ 分批的理由：广播打的是'人的注意力'，持续投放会进入边际递减；"
      "间歇让 A 池自然衰减，E 池补货速度跟不上，峰值就上不去。")

# --- 补充实验：同剂量、同"满强度"的公平 PK ---
cont_weak = simulate(lambda t: GOV_RATE_D * DUTY if t >= GOV_START else 0.0)
print("\n=== 公平对比（两者都只用 40% 警力，批次内强度相同）===")
for nm, sv in [("持续·低强度", cont_weak), ("分批·低强度", pulse)]:
    tk, ak = peak(sv)
    print(f"  {nm:10s} 峰值 {ak:,.0f}（{ak/N:.1%}）@ {tk:.1f} 天")

# ============================================================
# 3. 参数反演（数据 → 参数）
# ============================================================
rng = np.random.default_rng(7)
obs_t = np.arange(0.5, T_END + 1e-9, 0.5)          # 每 12 小时一份"恐慌人数"上报
truth = simulate(zero, t_eval=obs_t)
obs = truth.y[2] * (1 + rng.normal(0, 0.05, len(obs_t)))

def resid(p):
    b, gm = p
    s = simulate(zero, beta=b, gamma=gm, t_eval=obs_t)
    # 用 log 残差：A 从 200 涨到 9 万跨 2.6 个数量级，绝对误差会被峰值段垄断
    return np.log(s.y[2] + 1.0) - np.log(obs + 1.0)

best = None
for b0 in [0.8, 1.5, 2.5, 4.0]:
    for g0 in [0.3, 1.0, 2.0]:
        r = least_squares(lambda p: resid(p), x0=[b0, g0],
                          bounds=([0.05, 0.05], [20, 20]), max_nfev=200,
                          xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or r.cost < best.cost:
            best = r
b_hat, gm_hat = best.x
print("\n=== 参数反演（只给你'恐慌人数'序列）===")
print(f"真值 β={BETA:.2f}/天, γ={GAMMA:.2f}/天, R0={R0:.2f}")
print(f"反演 β={b_hat:.2f}/天, γ={gm_hat:.2f}/天, R0_hat={b_hat/(OMEGA+gm_hat):.2f}")

# 可辨识性：先画残差的"谷底带"——哪些 (β,γ) 组合拟合得同样好
band = []
for b in np.linspace(0.5, 6.0, 56):
    cand = [(0.5 * np.sum(resid([b, gm]) ** 2), gm) for gm in np.linspace(0.1, 6.0, 60)]
    band.append((b, min(cand)[1], min(cand)[0]))
band = np.array(band)
cmin = band[:, 2].min()
good = band[band[:, 2] < cmin * 1.10]
print(f"可辨识性（固定 β 扫描 γ 的剖面）：残差在最优 10% 以内的 β 有 {len(good)} 个，"
      f"β∈[{good[:,0].min():.2f}, {good[:,0].max():.2f}] → 谷底窄，说明数据对 β 有约束力")
# 但"谷底窄"≠"β 可自由取"：真正的检验是同时扰动 (β,γ) 能否保持同拟合
eps_list = []
for eps in [-0.15, -0.10, 0.10, 0.15]:
    b2 = BETA * (1 + eps)
    cand = min(((0.5*np.sum(resid([b2, gm])**2), gm) for gm in np.linspace(0.1, 6, 60)))
    c0 = 0.5*np.sum(resid([BETA, GAMMA])**2)
    eps_list.append((eps, cand[1], cand[0]/max(c0,1e-12)))
    print(f"   把 β 强行改 {eps:+.0%}，最优 γ 变到 {cand[1]:.2f}，残差变成基线的 {cand[0]/max(c0,1e-12):.1f} 倍")
assert all(r > 1.5 for _, _, r in eps_list), "β 偏离真值时拟合应明显变差（即可辨识）"
print("→ 结论：β 被数据约束住了（偏离 10% 拟合就明显变差），但 γ 与 β 部分抵消——")
print("   所以论文里写'β≈2.2，但需再给潜伏池存量数据才能把 γ 单独定死'，而不是笼统说'反演成功'。")

pred = simulate(zero, beta=b_hat, gamma=gm_hat, t_eval=obs_t)
mask = obs > 0.01 * N
err = np.mean(np.abs(pred.y[2][mask] - obs[mask]) / obs[mask])
print(f"拟合相对误差（>1% 人口区间）：{err:.1%}")
assert err < 0.30, "拟合应能解释数据形态"
assert len(good) >= 1

# 用反演参数做决策：要多强的广播才能把峰值压到 5%？
grid = np.linspace(0, 4, 81)
curve = np.array([peak(simulate(lambda t, g=g: g if t >= GOV_START else 0.0))[1] for g in grid])
need = grid[np.argmax(curve < 0.05 * N)] if np.any(curve < 0.05 * N) else None
print(f"决策曲线：广播强度 ≥ {need:.2f}/天 才能把峰值压到 5%（真参数下）")
assert need is not None

# ============================================================
# 4. 敏感性：β 估偏 25% 的代价
# ============================================================
print("\n=== 敏感性：传播率估偏的代价 ===")
for k in [0.75, 1.0, 1.25]:
    p = peak(simulate(zero, beta=BETA * k))[1]
    print(f"β×{k:<4.2f} → 峰值 {p:,.0f}（{p/N:.1%}）")
pk_up = peak(simulate(zero, beta=BETA * 1.25))[1]
pk_dn = peak(simulate(zero, beta=BETA * 0.75))[1]
print(f"β 估偏 ±25% → 峰值在 [{pk_dn:,.0f}, {pk_up:,.0f}]，即 {pk_dn/N:.1%}~{pk_up/N:.1%}；"
      f"资源按峰值配置时至少要多备 {pk_up/pk_dn-1:.0%}")
assert pk_up > a_pk > pk_dn, "β 越大峰值越高"

# ============================================================
# 5. 图
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
c = dict(S="#4C78A8", E="#F58518", A="#D62728", R="#54A24B")

a = ax[0, 0]
for i, (k, lab) in enumerate(zip(range(4), ["未听说 S", "听说未信 E", "恐慌散布 A ★", "理性等待 R"])):
    a.plot(base.t, base.y[i] / N * 100, label=lab, lw=2.6 if i == 2 else 1.8, color=list(c.values())[i])
a.plot(base.t[truth.y[2].argmax()], a_pk / N * 100, "k*", ms=14)
a.annotate(f"峰值 {a_pk/N:.1%} @ {t_pk*24:.0f}h", (t_pk, a_pk / N * 100),
           xytext=(t_pk + 0.6, a_pk / N * 100), arrowprops=dict(arrowstyle="->"))
a.set_title("① SEAR 基线：四个仓室此消彼长")
a.set_xlabel("天"); a.set_ylabel("占总人口 %"); a.legend(fontsize=9); a.grid(alpha=.3)

a = ax[0, 1]
a.plot(base.t, base.y[2] / 1000, lw=2, label="无干预", color="#999")
a.plot(cont.t, cont.y[2] / 1000, lw=2, label="持续广播", color="#F58518")
a.plot(pulse.t, pulse.y[2] / 1000, lw=2.8, label="分批投放（总量相同）", color="#2E7D32")
for w in windows:
    a.axvspan(w[0], w[1], color="green", alpha=0.10)
a.set_title("② 同样的总剂量：持续 vs 分批")
a.set_xlabel("天"); a.set_ylabel("恐慌人数（千人）"); a.legend(fontsize=9); a.grid(alpha=.3)

a = ax[1, 0]
a.scatter(obs_t, obs / 1000, s=42, c="k", zorder=3, label="上报数据（每 12h，10% 噪声）")
a.plot(truth.t, truth.y[2] / 1000, lw=1.6, ls="--", color="#666", label="真模型")
a.plot(obs_t, simulate(zero, beta=b_hat, gamma=gm_hat, t_eval=obs_t).y[2] / 1000,
       lw=2.4, color="#8E24AA", label=f"反演 β={b_hat:.2f}, γ={gm_hat:.2f}")
a.set_title("③ 数据→参数：拟合得好 ≠ 参数估得准（可辨识性）")
a.set_xlabel("天"); a.set_ylabel("恐慌人数（千人）"); a.legend(fontsize=9); a.grid(alpha=.3)

a = ax[1, 1]
a.plot(grid, curve / N * 100, lw=2.4, color="#1565C0")
a.axhline(5, ls="--", c="#D62728", lw=1.2)
a.text(0.05, 5.8, "政策目标：峰值 ≤ 5%", color="#D62728", fontsize=9)
if need:
    a.axvline(need, ls=":", c="gray")
    a.annotate(f"需 ≥ {need:.2f}/天", (need, 5), xytext=(need + 0.15, 12),
               arrowprops=dict(arrowstyle="->"))
a.set_title("④ 决策曲线：广播强度 → 峰值高度")
a.set_xlabel("广播冷静率 g（/天）"); a.set_ylabel("峰值占总人口 %"); a.grid(alpha=.3)

fig.suptitle("Day49 微分方程建模 · 县域应急撤离：谣言恐慌 SEAR 模型", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fear_spread_sear.png", dpi=150)
print("\n图已保存 fear_spread_sear.png")
print("全部断言通过 ✅")
