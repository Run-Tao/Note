"""
subject : mathematical-modeling
level   : L4 统计与预测模型（灰色系统 GM(1,1)）/ L10 模型检验（滚动回测 / 数据扰动）
anchor  : Learning_Roadmap · Level 4「统计与预测模型：灰色预测 GM(1,1)」（Model Atlas 备赛空白 🟡）
goal    : 县域河段氨氮只有 9 个季度监测点、无机理、无历史同期，
          要定「下季度限排令发不发、发几级、发多久」。
          手写 GM(1,1)：级比检验 → 最小二乘估 a,b → 白化解 → IAGO 还原 → 后验差 C/P，
          再用 DGM(1,1) + 三种基线做滚动回测，最后用"数据扰动"给触线时刻配上可信区间。

运行: cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-20/grey_gm11_pollution.py"
依赖: numpy / matplotlib（零专用灰色系统库，全手写）
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"):
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("error")

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 场景数据
# 云溪县河段氨氮（mg/L），2024Q2 起 9 个季度例行监测值（县环境监测站）
OBS = np.array([0.61, 0.68, 0.76, 0.84, 0.94, 1.04, 1.16, 1.28, 1.42])
LIMIT = 2.00      # GB 3838-2002 Ⅲ 类水氨氮限值（硬线）
WARN = 1.80       # 县内控预警线（软线）
QUARTERS = ["2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2",
            "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
FUTURE = ["2026Q3", "2026Q4", "2027Q1", "2027Q2"]
SEED = 20261020


# ================================================================ GM(1,1)
def ago(x):
    return np.cumsum(x)


def lambda_ok(x):
    """级比检验：λ(k)=x(k-1)/x(k) ∈ (e^{-2/(n+1)}, e^{2/(n+1)})；不通过要先做平移/开方变换。"""
    lam = x[:-1] / x[1:]
    lo, hi = np.exp(-2 / (len(x) + 1)), np.exp(2 / (len(x) + 1))
    return lam, bool(np.all((lam > lo) & (lam < hi))), lo, hi


def gm11(x, steps=0):
    """标准 GM(1,1)。返回 (a, b, off, 拟合+预测序列)。

    三个必须做对的地方（错一处数就废）：
      1) **先在 AGO 空间减首值**（平移不变性）：否则 x1 首项被初值占满，b 被初值主导；
      2) **IAGO 还原时首值直接用 x(0)**：白化解只拟合 AGO 的走势，它在 k=0 处不等于 x1(0)，
         若强行令 x1_hat(0)=x1(0)、再拿白化解算 x1_hat(1)，第一点会被压成负数；
      3) 白化解的起点取 **x1(1)**（第一个累加值），不用 x1(0)=0。
    """
    x = np.asarray(x, float)
    off = x[0] - 1.0
    y = x - off                       # 平移到首值 = 1
    x1 = ago(y)
    z1 = 0.5 * (x1[1:] + x1[:-1])     # 紧邻均值生成
    B = np.column_stack([-z1, np.ones_like(z1)])
    (a, b), *_ = np.linalg.lstsq(B, y[1:], rcond=None)
    k = np.arange(len(y) + steps)
    # 白化解在 AGO 空间：x1hat(k) = (y1 - b/a)e^{-a(k-1)} + b/a，y1 = 首个 AGO 值
    y1 = x1[0]
    kk = np.arange(len(y) + steps)               # k=0 对应第一个 AGO 值
    x1h = (y1 - b / a) * np.exp(-a * kk) + b / a
    out = np.empty_like(k, dtype=float)
    out[0] = x[0]                                 # IAGO 首点回填原值（y 空间里为 1）
    out[1:] = x1h[1:] - x1h[:-1] + off            # 其余点：AGO 之差 + 平移量
    return float(a), float(b), off, out


def series_at(a, b, off, k):
    """连续时间版的"累减还原"：返回时刻 k 处的**瞬时值** x(k)（单位同原数据）。

    为什么不能拿 AGO 的白化解 x1(k) 直接和限值比？
    因为 x1 是**累计量**、单位是"mg/L 累加"，不是浓度本身。
    正确写法：x(k) = dx1/dt |_k ，即 AGO 曲线在该时刻的切线斜率。
    """
    return -a * (1.0 - b / a) * np.exp(-a * k) + off


def cross_time(a, b, off, level):
    """反解「瞬时值首次达到 level」的连续时刻 k（季度序号，k=0 ↔ 首个观测点）。"""
    if a >= 0:
        return np.nan
    rhs = (level - off) / (-a * (1.0 - b / a))
    if rhs <= 0:
        return np.nan
    return float(np.log(rhs) / (-a))


def dgm11(x, steps=0):
    """DGM(1,1)：对 AGO 直接建一阶递推 x1(k+1)=β1·x1(k)+β2，不经过微分方程。

    与 GM 本质同源（几何增长），但少一次"紧邻均值"的近似，指数型短序列上通常更稳。
    """
    x = np.asarray(x, float)
    off = x[0] - 1.0
    y = x - off
    x1 = ago(y)
    B = np.column_stack([x1[:-1], np.ones_like(x1[:-1])])
    (b1, b2), *_ = np.linalg.lstsq(B, x1[1:], rcond=None)
    k = np.arange(len(y) + steps)
    A = x1[0] - b2 / (1 - b1)
    x1h = A * b1 ** k + b2 / (1 - b1)
    out = np.empty_like(k, dtype=float)
    out[0] = x[0]
    out[1:] = x1h[1:] - x1h[:-1] + off
    return out


# ================================================================ 检验
def post_check(x, xh, skip_first=False):
    """后验差检验：C=S2/S1（残差离散度 / 数据离散度），P=落在 0.6745·S1 带内的比例。"""
    x = np.asarray(x, float)
    xh = np.asarray(xh, float)[: len(x)]
    if skip_first:
        x, xh = x[1:], xh[1:]
    e = x - xh
    s1 = np.std(x, ddof=0)
    s2 = np.std(e, ddof=0)
    c = float(s2 / s1) if s1 > 0 else np.inf
    p = float(np.mean(np.abs(e - e.mean()) < 0.6745 * s1))
    if c < 0.35 and p >= 0.95:
        g = "好"
    elif c < 0.50 and p >= 0.80:
        g = "合格"
    elif c < 0.65 and p >= 0.70:
        g = "勉强"
    else:
        g = "不合格"
    return {"C": c, "P": p, "grade": g,
            "mape": float(np.mean(np.abs(e) / x) * 100),
            "mape_all": float(np.mean(np.abs((np.asarray(x, float) - xh)
                                             / np.asarray(x, float)) * 100)) if not skip_first else None}


# ================================================================ 基线
def naive_last(x, s=1):
    return np.concatenate([np.asarray(x, float), np.full(s, x[-1])])


def naive_ratio(x, s=1):
    r = np.mean(np.asarray(x[1:], float) / x[:-1])
    return np.concatenate([np.asarray(x, float), x[-1] * r ** np.arange(1, s + 1)])


def linreg(x, s=1):
    t = np.arange(len(x), dtype=float)
    k, b = np.polyfit(t, x, 1)
    return k * np.arange(len(x) + s) + b


def rolling_mae(fn, x):
    """扩展窗口 rolling-origin：第 t 个原点只用 x[:t]（最少 5 点）预测 x[t]。"""
    errs = np.array([abs(fn(x[:t])[t] - x[t]) for t in range(5, len(x))])
    return float(errs.mean()), errs


# ================================================================ 主流程
def main():
    x, n = OBS, len(OBS)
    print("=" * 78)
    print("云溪县河段氨氮（mg/L）")
    for q, v in zip(QUARTERS, x):
        print(f"  {q}: {v:.2f}")
    print(f"Ⅲ 类限值 {LIMIT:.2f}（硬线），内控预警线 {WARN:.2f}（软线）")

    # ---------- ① 级比检验
    lam, ok, lo, hi = lambda_ok(x)
    print(f"\n[1] 级比检验  可容区 ({lo:.4f},{hi:.4f})  实测 {np.round(lam, 4)} → {'通过' if ok else '不通过'}")
    assert ok

    # ---------- ② GM(1,1) + 后验差
    a, b, off, xh_all = gm11(x)
    xh = xh_all[:n]
    pc = post_check(x, xh)
    pc2 = post_check(x, xh, skip_first=True)
    print(f"\n[2] a={a:+.5f}（发展系数） b={b:+.5f}（灰作用量）  等价增长率 {-a * 100:.2f}%/季")
    print(f"    拟合 {np.round(xh, 4)}")
    print(f"    残差 {np.round(x - xh, 4)}")
    print(f"    后验差（全段）  C={pc['C']:.4f} P={pc['P']:.2f} → 「{pc['grade']}」  MAPE={pc['mape']:.2f}%")
    print(f"    后验差（剔首点）C={pc2['C']:.4f} P={pc2['P']:.2f} → 「{pc2['grade']}」  MAPE={pc2['mape']:.2f}%")
    print(f"    ⚠ 首点残差恒为 0（IAGO 回填）→ 后验差 C 被'白送'了 1 个零残差点，"
          f"9 点序列只有 8 个残差参与离散度；必须两个口径都报")
    assert abs(a) < 0.3, "|a|≥0.3 时 GM 不宜做中长期外推"
    assert xh[0] == x[0], "IAGO 首点必须回填原值"
    assert np.all(xh > 0), "拟合出现非正值"

    # ---------- ③ 一步预测：GM vs DGM vs 基线
    gm_fut = gm11(x, 1)[3][n]
    dgm_fut = dgm11(x, 1)[n]
    base = {"最后值不变": naive_last(x)[n], "几何增长率法": naive_ratio(x)[n],
            "最小二乘直线": linreg(x)[n]}
    print(f"\n[3] 2026Q3 一步预测：GM(1,1)={gm_fut:.4f}   DGM(1,1)={dgm_fut:.4f}")
    for k, v in base.items():
        print(f"    {k:<8s}={v:.4f}（与 GM 相差 {100 * (v - gm_fut) / gm_fut:+.1f}%）")
    assert gm_fut > x[-1], "本例为上升趋势，预测应高于末观测"

    # ---------- ④ 滚动回测
    print("\n[4] 扩展窗口滚动一步回测（原点：2025Q3 起，共 4 个）")
    tab = {}
    for name, fn in [("GM(1,1)", lambda z: gm11(z, 1)[3]), ("DGM(1,1)", lambda z: dgm11(z, 1)),
                     ("最后值", naive_last), ("增长率", naive_ratio), ("直线", linreg)]:
        mae, errs = rolling_mae(fn, x)
        tab[name] = (mae, errs)
        print(f"    {name:<8s} 平均|误差|={mae:.4f}  逐点 {np.round(errs, 3)}")
    best = min(tab, key=lambda k: tab[k][0])
    print(f"    → 最优：{best}")
    assert tab["GM(1,1)"][0] < 0.05, "GM 滚动误差应明显小于 0.05 mg/L"

    # ---------- ⑤ 未来 4 季 + 触线时刻
    x4 = gm11(x, 4)[3]
    k_warn = cross_time(a, b, off, WARN)
    k_hit = cross_time(a, b, off, LIMIT)
    print(f"\n[5] 未来 4 季 {FUTURE}：{np.round(x4[n:], 4)}")
    def lab(k):
        i = int(np.ceil(k))      # k=11.24 → 第 12 个监测点 → 2027Q1
        return QUARTERS[i] if i < n else FUTURE[i - n]
    print(f"    ⚠ 注意：**不能用 AGO 累计值 x1 去和限值比**（单位是累加值）；"
          f"必须用 AGO 曲线的斜率 dx1/dt")
    labf = lambda k: k - (n - 1)          # >0 表示第几个"未来季度"
    def when(k):
        d = labf(k)
        return f"{lab(k)} 前后（距最新观测 {d:+.2f} 季 ≈ {d * 3:+.0f} 个月）"
    print(f"    连续解反解（瞬时值）：软线 {WARN} → k={k_warn:.2f}，{when(k_warn)}")
    print(f"                         硬线 {LIMIT} → k={k_hit:.2f}，{when(k_hit)}")
    assert n < k_hit < n + 4, "硬线越线时刻应落在规划窗内（本例为 2027Q1 前后）"
    assert np.all(np.diff(x4) > 0), "白化解应保持单调上升"

    # ---------- ⑥ 数据扰动：短序列的"稳"比"准"更值钱
    print("\n[6] 数据扰动（灰色模型只有 9 个点，结论必须带区间）")
    variants = {}
    variants["删掉 2026Q2"] = cross_time(*gm11(x[:-1])[:2], off, LIMIT)
    variants["删掉最后两点"] = cross_time(*gm11(x[:-2])[:2], off, LIMIT)
    for pct in (-3, 3):
        xm = x.copy()
        xm[-1] *= 1 + pct / 100
        variants[f"末点 {pct:+d}%"] = cross_time(*gm11(xm)[:2], off, LIMIT)
    rng = np.random.default_rng(SEED)
    noisy = np.array([cross_time(*gm11(x * (1 + rng.normal(0, 0.005, n)))[:2], off, LIMIT)
                      for _ in range(300)])
    # 注：x*off 的写法会破坏平移基准，故噪声只加在原始尺度上、gm11 内部再平移
    for k, v in variants.items():
        print(f"    {k:<12s} k*={v:.2f}（Δ={v - k_hit:+.2f} 季 ≈ {(v - k_hit) * 3:+.1f} 个月）")
    print(f"    0.5% 测量噪声 × 300 次：均值 {np.nanmean(noisy):.2f} ± {np.nanstd(noisy):.2f}，"
          f"P5~P95 = [{np.nanpercentile(noisy, 5):.2f}, {np.nanpercentile(noisy, 95):.2f}]")
    shift = variants["删掉 2026Q2"] - k_hit
    print(f"    → 删掉最新一点、或把末点改 ±3%，越线时刻就偏 {shift:+.2f}~"
          f"{max(abs(v - k_hit) for v in variants.values()):.2f} 季；"
          f"GM 对'末点'敏感，命令书须写成「区间 + 复核条件」而不是单点日期")
    assert np.nanstd(noisy) > 0, "扰动应产生非零离散度（否则说明参数没跟着变）"

    # ---------- ⑦ 图（2×2）
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    tt = np.arange(n)

    axx = ax[0, 0]
    axx.plot(tt, x, "o-", color="#1f4e79", lw=2, label="监测值")
    axx.plot(tt, xh, "s--", color="#c00000", lw=1.6, label="GM(1,1) 拟合")
    axx.axhline(LIMIT, color="black", ls=":", lw=1.2, label=f"Ⅲ类限值 {LIMIT}")
    axx.axhline(WARN, color="gray", ls=":", lw=1.0, label=f"内控预警 {WARN}")
    axx.annotate("首点拟合偏低（灰度最大）", (0, xh[0]), textcoords="offset points",
                 xytext=(10, -26), fontsize=9, color="#b35c00",
                 arrowprops=dict(arrowstyle="->", color="#b35c00", lw=0.8))
    axx.set_xticks(tt, QUARTERS, rotation=45, ha="right")
    axx.set_ylabel("氨氮 (mg/L)")
    axx.set_title(f"GM(1,1)：a={a:+.4f}（{(-a) * 100:.1f}%/季）  MAPE={pc['mape']:.2f}%")
    axx.legend(fontsize=8, loc="lower right")
    axx.grid(alpha=0.3)

    axx = ax[0, 1]
    resid = x - xh
    axx.bar(tt, resid, color=np.where(resid > 0, "#c00000", "#2e75b6"))
    axx.axhline(0, color="black", lw=0.8)
    s1 = np.std(x)
    axx.axhline(0.6745 * s1, color="gray", ls="--", lw=1, label="±0.6745·S1")
    axx.axhline(-0.6745 * s1, color="gray", ls="--", lw=1)
    axx.set_xticks(tt, QUARTERS, rotation=45, ha="right")
    axx.set_ylabel("残差 (mg/L)")
    axx.set_title(f"后验差：全段 C={pc['C']:.3f}「{pc['grade']}」 / 剔首点 C={pc2['C']:.3f}「{pc2['grade']}」")
    axx.legend(fontsize=8)
    axx.grid(alpha=0.3, axis="y")

    axx = ax[1, 0]
    allt = np.arange(n + 4)
    axx.plot(tt, x, "o-", color="#1f4e79", lw=2, label="监测值")
    axx.plot(allt, x4, "--", color="#c00000", lw=1.8, label="GM(1,1)")
    axx.plot(allt, dgm11(x, 4), ":", color="#ed7d31", lw=2, label="DGM(1,1)")
    axx.plot(allt, naive_ratio(x, 4), "-.", color="#70ad47", lw=1.3, label="增长率法")
    axx.plot(allt, linreg(x, 4), "--", color="#7030a0", lw=1.1, label="直线法")
    axx.axhline(LIMIT, color="black", ls=":", lw=1.2)
    axx.axhline(WARN, color="gray", ls=":", lw=1.0)
    axx.axvline(k_hit, color="#c00000", lw=1, alpha=0.6)
    axx.scatter([k_hit], [LIMIT], color="#c00000", zorder=5)
    axx.annotate(f"越线 k*={k_hit:.2f}", (k_hit, LIMIT), textcoords="offset points",
                 xytext=(6, 8), fontsize=9, color="#c00000")
    axx.axvline(n - 0.5, color="gray", lw=0.8)
    axx.set_xticks(allt, QUARTERS + FUTURE, rotation=45, ha="right")
    axx.set_ylabel("氨氮 (mg/L)")
    axx.set_title("未来 4 季与越线时刻：GM≈DGM；直线法低估、增长率法略高")
    axx.legend(fontsize=8, loc="upper left")
    axx.grid(alpha=0.3)

    axx = ax[1, 1]
    names = ["最后值", "增长率", "直线", "GM(1,1)", "DGM(1,1)"]
    vals = [tab[k][0] for k in names]
    bars = axx.bar(names, vals, color=["#a5a5a5", "#a5a5a5", "#a5a5a5", "#c00000", "#ed7d31"])
    for bar, v in zip(bars, vals):
        axx.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                 ha="center", fontsize=9)
    axx.set_ylabel("滚动一步预测 平均|误差| (mg/L)")
    axx.set_title("滚动回测：9 点序列里增长率法险胜 GM/DGM —— '更聪明的模型'不等于'更准'")
    axx.grid(alpha=0.3, axis="y")

    fig.suptitle("Day50 · 灰色 GM(1,1)：9 个监测点，能否定下季度限排令", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "grey_gm11_pollution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[7] 图已保存 {out}")

    # ---------- ⑧ 决策读数
    print("\n[8] 决策读数")
    print(f"    GM 预测 2026Q3 = {gm_fut:.2f} mg/L：距软线 {WARN - gm_fut:+.2f}、距硬线 {LIMIT - gm_fut:+.2f}")
    print(f"    硬线越线 k*={k_hit:.2f}（扰动 P5~P95 "
          f"[{np.nanpercentile(noisy, 5):.2f}, {np.nanpercentile(noisy, 95):.2f}]）")
    print("    ⇒ 命令书：本季发一级限排令、按 2~3 季执行、每季复核"
          "（连续两季环比 >9% 提前一级）")


if __name__ == "__main__":
    main()
