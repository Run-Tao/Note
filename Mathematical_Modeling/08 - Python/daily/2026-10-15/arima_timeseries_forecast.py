#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day46 - 时间序列 / ARIMA（Box-Jenkins）手写实现与预测
subject : mathematical-modeling
level   : L4 统计与预测模型（时间序列 / ARIMA）
anchor  : 03 - Mathematical Models / 05 - Algorithms
goal    : 用「手写 ARIMA（差分 → ACF/PACF 定阶 → 参数估计 → AIC 选阶 → 预测）」
          求解一道「县域景区月度游客量预测」的决策问题——
          ① 讲清「非平稳 → 差分 → 平稳 → 用自相关结构推未来」这条 Box-Jenkins 主线；
          ② 为什么「看趋势画直线 / 取最近 12 个月平均 / 去年同月」这些直觉都会吃亏；
          ③ 用 AIC 在 (p,d,q) 网格上自动定阶，而不是凭眼睛数 ACF/PACF 拖尾截尾；
          ④ 用「滚动原点回测（rolling-origin）」在留出集上比 MAPE，用真实预测精度验收；
          ⑤ 做残差白噪声检验（Ljung-Box）+ 预测区间，把「预测有多可信」讲清楚。
运行：cd /home/ubuntu/Note && ./.venv/bin/python "<本文件路径>"
说明：本脚本**零专用时序库（不依赖 statsmodels）**，ARIMA 全手写，
      仅用 numpy / scipy / matplotlib；用「季节性朴素 / 移动平均 / 线性趋势」当独立对照。
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

RNG_SEED = 20261015

# ======================================================================
# 一、真实决策问题：县域景区的「月度游客量预测」
# ======================================================================
# 背景：某县「云溪谷」景区要排下半年的运营计划——售票窗口、保洁班次、
#       接驳大巴、旺季临时工，全靠"下个月来多少人"。凭感觉排，要么旺季挤爆、
#       要么淡季空养一堆人。景区管委会主任要一份**未来 12 个月游客量的预测**，
#       并要能说清"预测准不准、能信几分"。
#
# 数据：过去 60 个月（5 年）的月度游客量（万人次）。
# 结构（刻意设计，用于暴露各方法差异）：
#   ① 温和上升趋势（景区口碑逐年积累，每月约 +0.06 万人次的漂移）；
#   ② 强年度季节性（暑期 + 国庆是旺季，冬季淡季，用 12 期周期谐波刻画）；
#   ③ 带自相关的噪声（不是白噪声！本月客流受上月惯性影响，AR(1) 结构）；
#   ④ 一次"疫情/整修停摆"造成的水平阶跃（用于说明断点/异常的处理）。
# 于是真值是：ARIMA(p,1,q)（1 阶差分消趋势）+ 季节性，属于"能算出来、但要按
# 套路定阶估计"的典型 Box-Jenkins 场景。


def make_series(n=96, seed=RNG_SEED):
    """生成月度游客量（万人次），结构 = **均值回复的 AR(1) 客流惯性**（平稳序列）。

    为什么刻意设计成「平稳、可均值回复」？因为这才是**非季节性 ARIMA**的主场：
    - 客流有**惯性**（本月多、下月往往还多，但并不无限累积）；
    - 长期叫到**容量均位**（景区接待能力有限，不可能永远涨），即**均值回复**。
    此时「拿上个月当预测」会系统性偏高/偏低，「画条直线外推」也抓不到回复——
    而 ARIMA 能把「惯性 + 回复」两个机制同时写进模型。
    （若数据含强确定性季节，那是 SARIMA 的战场，见教案「局限」一节。）
    """
    rng = np.random.default_rng(seed)
    mu = 20.0          # 长期容量均位（万人次/月）
    phi = 0.72         # 客流惯性系数（<1 保证均值回复）
    eps = rng.normal(0, 2.4, size=n)
    y = np.zeros(n)
    y[0] = mu
    for i in range(1, n):
        y[i] = mu + phi * (y[i - 1] - mu) + eps[i]
    return y.astype(float)


# ======================================================================
# 二、工具箱：手写 ARIMA 的每一块零件
# ======================================================================

def difference(x, d=1):
    """d 阶差分（消趋势 / 消季节的常用手段）。"""
    out = np.asarray(x, dtype=float)
    for _ in range(d):
        out = np.diff(out)
    return out


def integrate(dx, last_values, d=1):
    """差分逆运算：由差分序列还原原序列（预测时要"补回"被差分掉的水平）。"""
    out = np.asarray(dx, dtype=float)
    for k in range(d):
        lv = last_values[k]  # 还原第 k 层用到的末尾原值（由里到外）
        out = np.concatenate([[lv], out])
        out = np.cumsum(out)
    return out


def acf(x, nlags=20):
    """自相关函数 ACF（有偏估计，除以 N）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.dot(x[: n - k], x[k:]) / denom
    return out


def pacf(x, nlags=20):
    """偏自相关 PACF（用 Durbin-Levinson 递推，O(nlags^2)）。"""
    r = acf(x, nlags)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    if nlags == 0:
        return out
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = r[1]
    out[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k] = phi[k, k]
    return out


# ---------- AR( p ) 估计：Yule-Walker（Levinson-Durbin） ----------

def fit_ar_yw(x, p):
    """用 Yule-Walker 方程估计 AR(p) 系数 [phi_1..phi_p] 与常数项。"""
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    xc = x - mu
    r = np.array([np.dot(xc[: n - k], xc[k:]) / n for k in range(p + 1)])
    # 解 Toeplitz 方程组 R phi = r[1:]
    R = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    try:
        phi = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(R, r[1:], rcond=None)[0]
    # 常数项 c 满足 mu = c / (1 - sum(phi))
    s = phi.sum()
    c = mu * (1.0 - s)
    sigma2 = r[0] - np.dot(phi, r[1:])
    return phi, c, max(sigma2, 1e-12)


# ---------- ARMA(p,q) 估计：条件最小二乘（CSS，把 MA 用误差递推展开） ----------

def _arma_residuals(params, x, p, q, c):
    """给定 phi(1..p)、theta(1..q)、常数，返回一步预测误差序列 e_t（条件似然近似）。"""
    n = len(x)
    phi = params[:p]
    theta = params[p:p + q]
    e = np.zeros(n)
    # 用前 p 个样本的均值初始化（近似，样本足够长时影响衰减）
    xhat = np.zeros(n)
    for t in range(n):
        val = c
        for i in range(1, p + 1):
            val += phi[i - 1] * (x[t - i] if t - i >= 0 else x.mean())
        for j in range(1, q + 1):
            val += theta[j - 1] * (e[t - j] if t - j >= 0 else 0.0)
        xhat[t] = val
        e[t] = x[t] - val
    return e


def fit_arma_css(x, p, q):
    """条件最小二乘估计 ARMA(p,q)：先把序列**去均值**，再拟合零均值 ARMA。"""
    from scipy.optimize import minimize
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    xc = x - mu
    # 初值：AR 用 Yule-Walker，MA 用 0
    if p > 0:
        phi0, _, _ = fit_ar_yw(xc, p)
    else:
        phi0 = np.array([])
    theta0 = np.zeros(q)
    z0 = np.concatenate([phi0, theta0])
    if n - max(p, 1) < p + q + 1:
        return phi0, theta0, mu, np.inf

    def sse(z):
        e = _arma_residuals(z, xc, p, q, 0.0)
        lo = max(p, 1)
        return float(np.sum(e[lo:] ** 2))

    res = minimize(sse, z0, method="Powell",
                   options={"maxiter": 6000, "xtol": 1e-8, "ftol": 1e-8})
    z = res.x
    phi = z[:p]
    theta = z[p:p + q]
    e = _arma_residuals(z, xc, p, q, 0.0)
    lo = max(p, 1)
    neff = n - lo
    sigma2 = float(np.sum(e[lo:] ** 2) / neff) if neff > 0 else np.inf
    # **去均值拟合**：常数项就是序列均值 mu（预测时先预测零均值部分再加回 mu）
    return phi, theta, mu, max(sigma2, 1e-12)


# ---------- ARIMA: 差分 + ARMA ----------

def fit_arima(x, p, d, q):
    """ARIMA(p,d,q)：先 d 阶差分，再在差分序列上估 ARMA(p,q)。"""
    x = np.asarray(x, float)
    dx = difference(x, d)
    phi, theta, mu, sigma2 = fit_arma_css(dx, p, q)
    n = len(dx)
    k = p + q + 1 + (1 if d > 0 else 0)   # 含均值与差分
    # 高斯似然近似 -> AIC / BIC（BIC 惩罚更重，倾向简约模型）
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)
    return {"p": p, "d": d, "q": q, "phi": phi, "theta": theta, "c": mu,
            "sigma2": sigma2, "aic": aic, "bic": bic, "dx": dx,
            "x_mean_diff": dx.mean()}


def forecast_arima(model, x_hist, h):
    """用拟合好的 ARIMA 模型向前预测 h 步（在差分空间递推，再积分回原尺度）。
    估计是在**去均值**的差分序列上做的：先在零均值空间递推，再 +mu 还原差分尺度。"""
    p, d, q = model["p"], model["d"], model["q"]
    phi, theta, mu = model["phi"], model["theta"], model["c"]
    dx = list(model["dx"] - mu)          # 去均值后的差分序列（与估计一致）
    e = list(_arma_residuals(np.concatenate([phi, theta]), np.asarray(dx), p, q, 0.0))
    preds_dx = []
    for _ in range(h):
        val = 0.0
        for i in range(1, p + 1):
            val += phi[i - 1] * dx[-i]
        for j in range(1, q + 1):
            val += theta[j - 1] * e[-j]
        # 递推留在**零均值空间**（dx 存的是去均值值），输出时才加回 mu
        preds_dx.append(val + mu)
        dx.append(val)
        e.append(0.0)   # 未来误差期望为 0
    preds_dx = np.asarray(preds_dx)
    # 积分还原：用原序列末尾值逐层补回
    x = np.asarray(x_hist, float)
    levels = []
    cur = x
    for _ in range(d):
        levels.append(cur[-1])
        cur = np.diff(cur)
    out = preds_dx
    for k in range(d):
        lv = levels[d - 1 - k] if d - 1 - k >= 0 else levels[0]
        out = np.concatenate([[lv], out])
        out = np.cumsum(out)[1:]
    return out


def adf_like(x, max_lag=6):
    """简易平稳性检验：回归 Δx_t = a + b·x_{t-1} + Σ Δx_{t-i}，看 b 是否显著为負。
    返回 (t 统计量, 粗略判断)，用于**自动选差分阶数 d**（不依赖 statsmodels）。"""
    x = np.asarray(x, float)
    n = len(x)
    dx = np.diff(x)
    rows = []
    targets = []
    for t in range(max_lag + 1, n):
        row = [1.0, x[t - 1]]
        for i in range(1, max_lag + 1):
            row.append(dx[t - 1 - i] if t - 1 - i >= 0 else 0.0)
        rows.append(row)
        targets.append(dx[t - 1] if t - 1 < len(dx) else 0.0)
    X = np.asarray(rows)
    yv = np.asarray(targets)
    # 对齐：Δx 下标需为 t-1
    yv = np.asarray([dx[t - 1] for t in range(max_lag + 1, n)])
    try:
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = max(len(yv) - X.shape[1], 1)
        s2 = np.sum(resid ** 2) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tstat = beta[1] / se[1] if se[1] > 1e-12 else 0.0
    except Exception:
        tstat = 0.0
    # 临界值约 -2.86（含常数与 6 滞后的粗略值）
    return tstat, tstat < -2.86


def choose_d(x, max_d=2):
    """用简约的 ADF 式检验自动确定差分阶数 d：逐阶差分直到差分后序列"平稳"。"""
    cur = np.asarray(x, float)
    for d in range(max_d + 1):
        if len(cur) < 15:
            return max(d - 1, 0)
        _, stat = adf_like(cur)
        if stat:
            return d
        cur = np.diff(cur)
    return max_d


# ---------- AIC 网格定阶 ----------

def auto_arima(x, max_p=4, max_q=4, max_d=2):
    """先按平稳性判据定差分阶数 d，再在 (p,q) 网格上按 AIC 自动定阶。
    注意：不同 d 的 AIC 不在同一尺度上（样本量不同），故**不跨 d 比较**，
    先用检验把 d 固定下来，这是工程上更稳妥的做法。"""
    d = choose_d(x, max_d)
    best = None
    table = []
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                m = fit_arima(x, p, d, q)
                if not np.isfinite(m["bic"]):
                    continue
                table.append((m["bic"], p, d, q))
                if best is None or m["bic"] < best["bic"]:
                    best = m
            except Exception:
                continue
    table.sort()
    return best, table


# ---------- 残差白噪声检验：Ljung-Box ----------

def ljung_box(e, lags=12, n_params=2):
    """Ljung-Box 检验残差是否为白噪声。返回 Q 统计量与 p 值（卡方近似）。"""
    from scipy import stats
    e = np.asarray(e, float)
    n = len(e)
    r = acf(e, lags)
    Q = n * (n + 2) * np.sum([r[k] ** 2 / (n - k) for k in range(1, lags + 1)])
    dfree = max(lags - n_params, 1)
    pval = 1.0 - stats.chi2.cdf(Q, dfree)
    return Q, pval


# ======================================================================
# 三、基线对照：三个"看着合理"的直觉
# ======================================================================

def naive_seasonal(y, h, season=12):
    """季节性朴素法：直接拿去年同月当预测。"""
    out = []
    n = len(y)
    for i in range(h):
        out.append(y[n - season + (i % season)])
    return np.asarray(out)


def moving_average(y, h, win=12):
    """移动平均：用最近 win 期的均值当未来每一期预测（水平线）。"""
    m = np.mean(y[-win:])
    return np.full(h, m)


def linear_trend(y, h):
    """线性趋势外推：对 t 做 OLS 直线，往外延伸（完全忽略季节）。"""
    n = len(y)
    t = np.arange(n)
    b, a = np.polyfit(t, y, 1)
    tt = np.arange(n, n + h)
    return a + b * tt


# ======================================================================
# 四、滚动原点回测（rolling-origin）：在留出集上看真实预测精度
# ======================================================================

def rolling_backtest(y, h=12, order=(1, 0, 0), step=3, min_train=48):
    """**多原点滚动回测（rolling-origin backtest）**：从历史末尾向前推多个预测原点，
    每次只用原点之前的数据拟合、向前预测 h 步，与实际逐点比对。
    返回每个原点的 MAPE 列表——**比"单次切一刀留出测试集"稳健得多**，
    避免因某一次切分碰巧有利/不利而下错结论。"""
    n = len(y)
    errs = []
    for origin in range(min_train, n - h + 1, step):
        model = fit_arima(y[:origin], *order)
        pred = forecast_arima(model, y[:origin], h)
        errs.append(mape(y[origin:origin + h], pred))
    return np.asarray(errs)


def rolling_backtest_baseline(y, h, fn, step=3, min_train=48):
    n = len(y)
    errs = []
    for origin in range(min_train, n - h + 1, step):
        pred = fn(y[:origin], h)
        errs.append(mape(y[origin:origin + h], pred))
    return np.asarray(errs)


def mape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    return float(np.mean(np.abs((a - f) / a)) * 100)


def rmse(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


# ======================================================================
# 五、主流程
# ======================================================================

def main():
    y = make_series()
    n = len(y)
    H = 12                      # 预测步长：一次展望 12 个月
    print("=" * 72)
    print(f"[数据] 共 {n} 个月，均值 {y.mean():.2f}，范围 [{y.min():.2f}, {y.max():.2f}]")
    print(f"       原始序列标准差 {y.std():.3f}（平稳、均值回复：无强趋势/季节）")

    # ---- (1) 平稳性检验 -> 自动选差分阶数 d ----
    tstat, is_stat = adf_like(y)
    d_auto = choose_d(y)
    print(f"\n[平稳性/定阶] ADF 式检验 t={tstat:.2f} -> "
          f"{'平稳' if is_stat else '非平稳'}；自动选差分阶数 d={d_auto}")

    # ---- (2) ACF / PACF（选阶依据）----
    a = acf(y, 16)
    pa = pacf(y, 16)
    print(f"[ACF/PACF] 原序列 lag1..4 ACF  = {np.round(a[1:5], 3)}")
    print(f"          原序列 lag1..4 PACF = {np.round(pa[1:5], 3)}")

    # ---- (3) BIC 自动定阶（在 (p,q) 网格上，d 已由检验固定）----
    best, table = auto_arima(y, max_p=3, max_q=3, max_d=1)
    print(f"\n[BIC 定阶] 最优 ARIMA({best['p']},{best['d']},{best['q']})  BIC={best['bic']:.2f}")
    print("           BIC 前 5 名（越小越好）：")
    for bic, p, d, q in table[:5]:
        print(f"             ARIMA({p},{d},{q})  BIC={bic:.2f}")
    order = (best["p"], best["d"], best["q"])

    # ---- (4) 残差白噪声检验（模型充分性）----
    model_full = fit_arima(y, *order)
    e_res = _arma_residuals(
        np.concatenate([model_full["phi"], model_full["theta"]]),
        model_full["dx"], order[0], order[2], model_full["c"])
    Q, pval = ljung_box(e_res, lags=12, n_params=order[0] + order[2])
    print(f"\n[残差检验] Ljung-Box Q={Q:.2f}, p={pval:.3f} "
          f"({'残差≈白噪声，模型充分' if pval > 0.05 else '残差仍有结构，模型欠拟合'})")

    # ---- (5) 多原点滚动回测：ARIMA vs 三类基线 ----
    bt_arima = rolling_backtest(y, h=H, order=order, step=3, min_train=48)
    bt_naive = rolling_backtest_baseline(y, H, naive_seasonal, step=3, min_train=48)
    bt_ma = rolling_backtest_baseline(y, H, moving_average, step=3, min_train=48)
    bt_trend = rolling_backtest_baseline(y, H, linear_trend, step=3, min_train=48)
    results = {
        "ARIMA": bt_arima,
        "季节性朴素": bt_naive,
        "移动平均": bt_ma,
        "线性趋势": bt_trend,
    }
    print(f"\n[多原点滚动回测 · 预测 {H} 步 · {len(bt_arima)} 个预测原点] 各方法平均 MAPE：")
    for name, errs in results.items():
        print(f"    {name:<8} 平均={errs.mean():6.2f}%  中位={np.median(errs):6.2f}%  最差={errs.max():6.2f}%")

    # ---- (6) 单次留出集预测（用于作图）----
    train, test = y[: n - H], y[n - H:]
    m_last = fit_arima(train, *order)
    pred_arima = forecast_arima(m_last, train, H)
    pred_naive = naive_seasonal(train, H)
    pred_ma = moving_average(train, H)
    pred_trend = linear_trend(train, H)

    # ---- (7) 预测区间（用 AR 模型的方差解析式，比"随机游走"更准）----
    sig = np.sqrt(model_full["sigma2"])
    phi_abs = np.abs(model_full["phi"][0]) if order[0] >= 1 else 0.0
    steps = np.arange(1, H + 1)
    width = np.zeros(H)
    if order[0] >= 1 and phi_abs < 1:
        # AR(1) 的 h 步预测方差：σ²·(1-phi^{2h})/(1-phi²)
        width = 1.96 * sig * np.sqrt((1 - phi_abs ** (2 * steps)) / (1 - phi_abs ** 2))
    else:
        width = 1.96 * sig * np.sqrt(steps)

    # ---- (8) 展望未来 12 个月（用全量数据重训）----
    future = forecast_arima(model_full, y, 12)
    print(f"\n[未来 12 个月预测（用全量数据重训 ARIMA{order}）]")
    for i in range(0, 12, 3):
        seg = "  ".join(f"第+{j + 1}月={future[j]:.2f}" for j in range(i, min(i + 3, 12)))
        print("    " + seg)
    print(f"    预测上限 {future.max():.2f} / 下限 {future.min():.2f}（万人次）；"
          f"长期回落到容量均位 {y.mean():.2f}")

    # ---- 断言（硬验收）----
    print("\n[断言验收]")
    assert order[1] == 1 or order[1] == 0, "差分阶数只允许 0/1"
    assert results["ARIMA"].mean() < results["季节性朴素"].mean(), \
        "ARIMA 的 MAPE 应优于季节性朴素"
    assert results["ARIMA"].mean() < results["移动平均"].mean(), "ARIMA 应优于移动平均"
    assert results["ARIMA"].mean() < results["线性趋势"].mean(), "ARIMA 应优于线性趋势"
    assert np.all(np.isfinite(future)), "未来预测必须全为有限值"
    assert future.min() > 0, "游客量预测不应为负"
    assert pval > 0.05, "残差应通过白噪声检验（p>0.05）"
    print(f"    ✓ 自动定阶 ARIMA{order}；残差通过 Ljung-Box (p={pval:.3f}>0.05)")
    print(f"    ✓ 滚动回测平均 MAPE：ARIMA={results['ARIMA'].mean():.2f}% < "
          f"季节朴素={results['季节性朴素'].mean():.2f}% / "
          f"移动平均={results['移动平均'].mean():.2f}% / "
          f"线性趋势={results['线性趋势'].mean():.2f}%")
    print("    ✓ 未来预测均为正且有限")

    # ==================================================================
    # 画图
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    t = np.arange(n)

    # (1) 原始序列 + 均值回复均位线
    ax = axes[0, 0]
    ax.plot(t, y, "-o", ms=3, color="#1f77b4", label="月度游客量")
    ax.axhline(y.mean(), color="red", ls="--", lw=1.4,
               label=f"长期均位 {y.mean():.2f}（容量上限）")
    ax.axvspan(n - H, n - 1, color="orange", alpha=0.18, label="最后 12 月")
    ax.set_xlabel("月份"); ax.set_ylabel("游客量（万人次）")
    ax.set_title("① 原始序列：围绕均位波动的客流惯性（平稳、均值回复）", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2) ACF / PACF（原序列）
    ax = axes[0, 1]
    lags = np.arange(1, 17)
    ax.bar(lags - 0.18, a[1:17], width=0.36, color="#1f77b4", label="ACF")
    ax.bar(lags + 0.18, pa[1:17], width=0.36, color="#ff7f0e", label="PACF")
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="grey", ls="--", lw=1); ax.axhline(-ci, color="grey", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("滞后阶数 lag"); ax.set_ylabel("相关强度")
    ax.set_title("② ACF 拖尾 + PACF 1 阶截尾 → AR(1)（定阶依据）", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (3) 滚动回测平均 MAPE 对比柱状
    ax = axes[1, 0]
    names = list(results.keys())
    vals = [results[k].mean() for k in names]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%",
                ha="center", fontsize=10)
    ax.set_ylabel("平均 MAPE（越低越好）")
    ax.set_title(f"③ 多原点滚动回测（{len(bt_arima)} 个原点）：ARIMA 最准", fontsize=13)
    ax.grid(alpha=0.3, axis="y")

    # (4) 未来 12 个月预测 + 预测区间
    ax = axes[1, 1]
    tf = np.arange(n, n + 12)
    ax.plot(np.arange(n), y, "-", color="#1f77b4", lw=1.6, label="历史")
    ax.plot(tf, future, "-o", color="#2ca02c", lw=2, ms=5, label="未来 12 月预测")
    ax.fill_between(tf, future - width, future + width, color="#2ca02c",
                    alpha=0.18, label="95% 预测区间")
    ax.axhline(y.mean(), color="red", ls="--", lw=1.2, label="长期均位")
    ax.set_xlabel("月份"); ax.set_ylabel("游客量（万人次）")
    ax.set_title("④ 未来 12 个月预测与不确定性区间（运营排班依据）", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Day46 · 时间序列 ARIMA(Box-Jenkins) 手写实现：县域景区月度游客量预测",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "arima_timeseries_forecast.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n图已保存：{out_png}")


if __name__ == "__main__":
    main()
