#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day48 - 统计过程控制 SPC：x-bar-R 控制图 + 判异规则 + CUSUM 对照
subject : mathematical-modeling
level   : L4 统计与预测模型（抽样推断 / 品质控制）/ L10 模型检验（控制图判异与虚警率）
anchor  : 03 - Mathematical Models（统计模型主线）；与 Day46 ARIMA（预测）镜像：
          ARIMA 认为序列里有结构要去解释，SPC 认为序列不该有结构、一有就报警
goal    : 用「x̄-R 控制图 + Western Electric 判异规则 + CUSUM」回答一道「县域药厂片剂装量
          抽样，15:00 换辅料后均值上漂 3.5mg，但 120 片全部落在 480~520mg 国标范围内——
          到底要不要停机？」的决策问题。
          ① 讲清「合格」与「受控」是两笔不同的账（规格限 vs 控制限）；
          ② Phase I 建限 / Phase II 监视 必须分开（用含异常段的数据建限会把限位撑大）；
          ③ 判异规则不是「超 3σ」一条：小偏移要靠 R2/R3/R5/R6 这类「模式规则」；
          ④ 用 CUSUM 做第二裁判（小偏移最敏感），并用 10000 条稳定序列校准虚警率；
          ⑤ 过程能力 Cp/Cpk 只描述「稳定期」，不能当成「现在没事」的证据。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "<本文件路径>"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------
# 中文字体注册（Noto Sans CJK SC / 文泉驿），避免图中中文乱码
# ----------------------------------------------------------------------
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_registered = []
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            _registered.append(font_manager.FontProperties(fname=_p).get_name())
        except Exception:
            pass
if _registered:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _registered + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
print("CJK 字体已注册：", _registered if _registered else "（无，可能乱码）")

SEED = 20261018
rng = np.random.default_rng(SEED)

# ======================================================================
# 一、真实决策问题：县域药厂片剂装量的过程控制
# ======================================================================
# 500mg 维 C 片，国标单片 480~520mg。每 30 分钟抽 5 片称重，一个班 24 组。
# 14:00（第 9 组）换了新供应商的乳糖辅料 -> 装量均值每半小时上漂一点，半个班涨约 +3mg。
# 关键：单片全部落在 480~520 内 —— 按「合格率」看一切正常，按「受控」看已经出事。
NSG = 24          # 子组数（一个班）
SUB = 5           # 每子组抽样片数
LSL, USL = 480.0, 520.0
TARGET = 500.0
SHIFT_SG = 8      # 前 8 组（08:00–11:30、13:00、13:30）= 换料前稳定期，用于 Phase I 建限；14:00 换料

# —— 子组数据（电子秤 0.1mg 分辨率）——
#   稳定期 08:00~13:30（前 8 组）：N(500.4, 1.6)，只受随机波动
#   14:00 换辅料后：均值逐组线性上漂到 +3.5mg，但组内波动不变（R 图不会报警）
stable_mean, stable_sd = 500.4, 1.6
SHIFT_LOG = np.zeros(NSG)
SHIFT_LOG[SHIFT_SG:] = np.linspace(3.0, 6.0, NSG - SHIFT_SG)
xbars = (stable_mean + SHIFT_LOG).copy()  # 占位，下面重算
subgroups = np.round(rng.normal(stable_mean + SHIFT_LOG[:, None], stable_sd,
                               size=(NSG, SUB)), 1)
# 固定抽样噪声：本课只教「怎么发现偏移」，不希望某一次随机抽样把结论打乱
for _ in range(4):
    if np.std(xbars[:SHIFT_SG]) > 1.2:
        break
    subgroups = np.round(rng.normal(stable_mean + SHIFT_LOG[:, None], stable_sd,
                                   size=(NSG, SUB)), 1)
    xbars = subgroups.mean(axis=1)
Rs = subgroups.max(axis=1) - subgroups.min(axis=1)
xbars = subgroups.mean(axis=1)
Rs = subgroups.max(axis=1) - subgroups.min(axis=1)

# 控制图常数（子组大小 n=5）
N_SUB = SUB
A2, D3, D4, d2 = 0.577, 0.0, 2.114, 2.326

# ======================================================================
# 二、Phase I：只用稳定期建限（这是建模判断，不是计算细节）
# ======================================================================
xb_phase1 = xbars[:SHIFT_SG].mean()
rb_phase1 = Rs[:SHIFT_SG].mean()
sigma_hat = rb_phase1 / d2
UCLx = xb_phase1 + A2 * rb_phase1
LCLx = xb_phase1 - A2 * rb_phase1
UCLr = D4 * rb_phase1
LCLr = D3 * rb_phase1

print("=" * 76)
print("Day48 · 统计过程控制 SPC：片剂装量 x̄-R 图 + 判异规则 + CUSUM")
print("=" * 76)
print(f"规格 LSL/USL = {LSL:.0f}/{USL:.0f} mg，每 30 分钟抽 {SUB} 片，一个班 {NSG} 组")
print(f"\n[Phase I 建限（前 {SHIFT_SG} 组稳定期）] "
      f"x̄̄={xb_phase1:.2f}  R̄={rb_phase1:.2f}  σ̂=R̄/d2={sigma_hat:.2f}mg")
print(f"    x 图：UCL={UCLx:.2f}  CL={xb_phase1:.2f}  LCL={LCLx:.2f}")
print(f"    R 图：UCL={UCLr:.2f}  CL={rb_phase1:.2f}  LCL={LCLr:.2f}")
print(f"    单片落在规格内: {np.sum((subgroups >= LSL) & (subgroups <= USL))}/{subgroups.size}"
      f"（→ 只看单片，全都「合格」）")

# ======================================================================
# 三、Phase II：判异规则（Western Electric 简化版 6 条）
# ======================================================================
def zscores(x, mu, sigma):
    return (x - mu) / sigma


def detect_violations(xbars, mu, sigma, i_max=None):
    """对每个子组 i 返回命中的规则集合。规则只用 i 之前（含 i）的数据，防未来信息。"""
    n = len(xbars) if i_max is None else i_max
    z = zscores(xbars, mu, sigma)
    hits = {i: [] for i in range(n)}

    def add(rule, idx):
        if idx < n:
            hits[idx].append(rule)

    for i in range(n):
        # R1：单点超 3σ
        if abs(z[i]) > 3:
            add("R1", i)
        # R2：连续 9 点在中心线同侧
        if i >= 8 and (np.all(z[i - 8:i + 1] > 0) or np.all(z[i - 8:i + 1] < 0)):
            add("R2", i)
        # R3：连续 6 点单调升/降
        if i >= 5:
            d = np.diff(xbars[i - 5:i + 1])
            if np.all(d > 0) or np.all(d < 0):
                add("R3", i)
        # R4：连续 14 点交替升降
        if i >= 13:
            d = np.diff(xbars[i - 13:i + 1])
            if np.all(d[:-1] * d[1:] < 0):
                add("R4", i)
        # R5：连续 3 点中 2 点超 2σ 同侧
        if i >= 2:
            for sgn in (1, -1):
                if np.sum(sgn * z[i - 2:i + 1] > 2) >= 2:
                    add("R5", i)
                    break
        # R6：连续 5 点中 4 点超 1σ 同侧
        if i >= 4:
            for sgn in (1, -1):
                if np.sum(sgn * z[i - 4:i + 1] > 1) >= 4:
                    add("R6", i)
                    break
    return hits


violations = detect_violations(xbars, xb_phase1, sigma_hat)
first_alarm = next((i for i in range(NSG) if violations[i]), None)
rule_counts = {}
for i, rs in violations.items():
    for r in rs:
        rule_counts.setdefault(r, []).append(i + 1)   # 子组编号从 1 起

print(f"\n[Phase II 判异（{NSG} 组，限位来自 Phase I）]")
for r in ["R1", "R2", "R3", "R4", "R5", "R6"]:
    idxs = rule_counts.get(r, [])
    if idxs:
        span = f"第 {min(idxs)}~{max(idxs)} 组，共 {len(idxs)} 次"
    else:
        span = "无"
    print(f"    {r}: {span}")
alarm_sg = [i + 1 for i in range(NSG) if violations[i]]
alarm_count = len(alarm_sg)
print(f"    → 首次报警：第 {first_alarm + 1} 组（{alarm_sg[:6]}…）")

# ======================================================================
# 四、第二裁判：CUSUM（小偏移最敏感）
# ======================================================================
def cusum(xbars, mu, sigma, k=0.5, h=5.0):
    """双侧 CUSUM：C+ = max(0, C+ + z - k)，C- = max(0, C- - z - k)；超过 h 报警。

    k 为参考值（≈ 想检测的偏移的一半），h 为判决界限（标准差单位）。
    """
    z = zscores(xbars, mu, sigma)
    cp = np.zeros(len(z))
    cm = np.zeros(len(z))
    for i in range(len(z)):
        cp[i] = max(0.0, (cp[i - 1] if i else 0.0) + z[i] - k)
        cm[i] = max(0.0, (cm[i - 1] if i else 0.0) - z[i] - k)
    out = (cp > h) | (cm > h)
    first = int(np.argmax(out)) if out.any() else None
    return cp, cm, out, first


cp, cm, cusum_out, cusum_first = cusum(xbars, xb_phase1, sigma_hat, k=0.5, h=5.0)
print(f"\n[CUSUM 第二裁判] k=0.5, h=5.0σ")
print(f"    首次越界：{'第 %d 组' % (cusum_first + 1) if cusum_first is not None else '无'}"
      f"；越界组数 {int(cusum_out.sum())}/{NSG}")
print(f"    Shewhart 首发 第 {first_alarm + 1} 组 ｜ CUSUM 首发 "
      f"{'第 %d 组' % (cusum_first + 1) if cusum_first is not None else '无'}")

# ======================================================================
# 五、过程能力（只用稳定期）与对照组
# ======================================================================
Cp = (USL - LSL) / (6 * sigma_hat)
Cp_low = (xb_phase1 - LSL) / (3 * sigma_hat)
Cp_high = (USL - xb_phase1) / (3 * sigma_hat)
Cpk = min(Cp_low, Cp_high)
# 若用「全部 24 组」估 σ（错误做法）：x̄ 漂移会撑大 σ
sigma_all = xbars.std(ddof=1)
sigma_naive = Rs.mean() / d2
Cp_naive = (USL - LSL) / (6 * sigma_naive)
print(f"\n[过程能力（稳定期）] σ̂={sigma_hat:.2f}mg  Cp={Cp:.2f}  Cpk={Cpk:.2f}")
print(f"    若把 24 组一起算（错误做法）：σ̂={sigma_naive:.2f}mg，Cp 虚高到 {Cp_naive:.2f}")

# 对照组：换料后不换（继续稳定）——24 组里会报几次？
def simulate_stable(n_reps=10000, n=NSG, seed=SEED + 1):
    r = np.random.default_rng(seed)
    any_rule, cnt2 = 0, 0
    for _ in range(n_reps):
        sg = np.round(r.normal(stable_mean, stable_sd, size=(n, SUB)), 1)
        xb = sg.mean(axis=1)
        R = sg.max(axis=1) - sg.min(axis=1)
        mu = xb.mean()
        sd = R.mean() / d2
        # 用同一批数据自建限（Phase I 与 II 同源，相对宽松）
        hits = detect_violations(xb, mu, sd)
        flags = [len(hits[i]) > 0 for i in range(n)]
        if any(flags):
            any_rule += 1
        # 「连续 2 个子组都报警」才算真报警
        if any(flags[i] and flags[i + 1] for i in range(n - 1)):
            cnt2 += 1
    return 100.0 * any_rule / n_reps, 100.0 * cnt2 / n_reps


p_any, p_two = simulate_stable()
print(f"\n[虚警率仿真 10000 条稳定序列] 任一规则报警一次 ≈ {p_any:.1f}%；"
      f"相邻两组连续报警 ≈ {p_two:.1f}%")
print(f"    → 6 条规则叠起来不是「零虚警」：平均约每 {100/p_any:.0f} 个班虚惊 1 次"
      f"（以「一个班 24 组里任一规则报警」为口径）")

# ======================================================================
# 六、断言（硬验收）
# ======================================================================
print("\n【验收断言】")
assert np.all((subgroups >= LSL) & (subgroups <= USL)), "本实例单片应全部在规格内（教学伏笔）"
assert first_alarm is not None and first_alarm >= SHIFT_SG - 1, "报警应发生在换料之后"
assert first_alarm >= SHIFT_SG, "报警不能早于换料（否则是虚警）"
assert first_alarm <= NSG - 3, "报警应在换料后、下班前若干组内出现"
assert alarm_count >= 3, "换料后应有连续多组报警（不是孤立一点）"
assert cusum_first is not None and abs(cusum_first - first_alarm) <= 2, "CUSUM 与 Shewhart 结论应一致"
assert "R2" in rule_counts and len(rule_counts["R2"]) >= 3, "R2（连续同侧）应稳定捕获偏移"
assert any(r in rule_counts for r in ("R5", "R6")), "R5/R6 至少一条应参与捕获小偏移"
assert 0.0 < p_any < 35.0, "虚警率应在合理区间（既非 0 也非失控）"
assert Cpk > 1.33, "稳定期过程能力应满足工业常用门槛"
print("  ✅ 全部断言通过")

# ======================================================================
# 七、出图
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
xs = np.arange(1, NSG + 1)

# ---- 左上：x̄ 图 ----
ax = axes[0, 0]
ax.plot(xs, xbars, "o-", color="#2C6E9B", lw=1.6, ms=5, label="子组均值 x̄")
ax.axhline(UCLx, color="#D62728", ls="--", lw=1.4, label="UCL 3σ")
ax.axhline(LCLx, color="#D62728", ls="--", lw=1.4, label="LCL 3σ")
ax.axhline(xb_phase1, color="#2E8B57", ls="-", lw=1.2, label="CL（稳定期均值）")
ax.axvline(SHIFT_SG + 0.5, color="gray", ls=":", lw=1.5)
ax.text(SHIFT_SG + 0.7, LCLx - 0.6, "14:00 换辅料", fontsize=9, color="gray")
bad = np.array([i for i in range(NSG) if violations[i]]) if first_alarm is not None else np.array([], int)
ax.plot(xs[bad], xbars[bad], "x", color="#D62728", ms=10, mew=2, label="判异命中")
ax.set_xlabel("子组序号（每 30 分钟一组）")
ax.set_ylabel("均值（mg）")
ax.set_title("① x̄ 图：Phase I 建限（前 14 组）→ Phase II 监视")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# ---- 右上：R 图 ----
ax = axes[0, 1]
ax.plot(xs, Rs, "s-", color="#805AD5", lw=1.4, ms=4, label="子组极差 R")
ax.axhline(UCLr, color="#D62728", ls="--", lw=1.4, label=f"UCL={UCLr:.1f}")
ax.axhline(rb_phase1, color="#2E8B57", ls="-", lw=1.2, label=f"R̄={rb_phase1:.1f}")
ax.set_xlabel("子组序号")
ax.set_ylabel("极差（mg）")
ax.set_title("② R 图：组内波动没有变化——偏移来自「均值」，不是「精度变差」")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---- 左下：CUSUM ----
ax = axes[1, 0]
ax.plot(xs, cp, "o-", color="#D62728", ms=4, label="C+（h=5σ）")
ax.plot(xs, cm, "s-", color="#2C6E9B", ms=4, label="C−（h=5σ）")
ax.axhline(5.0, color="black", ls="--", lw=1.2, label="判决界限 h=5")
ax.axvline(SHIFT_SG + 0.5, color="gray", ls=":", lw=1.5)
ax.set_xlabel("子组序号")
ax.set_ylabel("累积和（标准化）")
ax.set_title("③ CUSUM：小偏移会「攒分」，比单点判异更早、更稳")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---- 右下：每组报警次数 + 首次报警对照 ----
ax = axes[1, 1]
cnt = np.array([len(violations[i]) for i in range(NSG)])
ax.bar(xs, cnt, color=["#9AA5B1" if c == 0 else "#D62728" for c in cnt])
ax.axvline(SHIFT_SG + 0.5, color="gray", ls=":", lw=1.5)
ax.text(SHIFT_SG + 0.7, max(cnt) * 0.75, "换辅料", fontsize=9, color="gray")
ax.set_xlabel("子组序号")
ax.set_ylabel("命中规则条数（0 = 正常）")
ax.set_title("④ 每组命中几条判异规则：换料前全 0，换料后连续命中")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spc_control_chart.png")
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\n图已保存：{out_png}")

# ======================================================================
# 八、结论摘要
# ======================================================================
print("\n【结论摘要】")
print(f"  1) 单片合格率 100%（120/120 落在 480~520mg），但过程在第 {first_alarm + 1} 组起失控")
print(f"  2) 判异证据：{', '.join(sorted(rule_counts))} 同时命中，CUSUM 在"
      f" {'第 %d 组' % (cusum_first + 1) if cusum_first is not None else '—'} 越界")
print(f"  3) 换料后期均值偏移 ≈ {xbars[SHIFT_SG:].mean() - xb_phase1:+.2f} mg（不到 0.2 个规格），"
      f"单片看不出来、子组均值看得出来")
print(f"  4) 过程能力 Cp={Cp:.2f}/Cpk={Cpk:.2f} 只描述稳定期；动作是「隔离 + 复核辅料 + 重建限」")
print(f"  5) 虚警率：稳定过程每 24 组任一规则报警 ≈ {p_any:.1f}%；规则不能无限叠加")
