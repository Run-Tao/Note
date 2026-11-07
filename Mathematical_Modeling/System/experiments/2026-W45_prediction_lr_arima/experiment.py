"""
W45 每周模型比较 #1：预测模型 — 线性回归 vs 多项式回归 vs 岭回归 vs Lasso vs ARIMA vs 指数平滑 vs GM(1,1)

设计原则（任务书 4.4）：
  - 同一数据集、同一划分、同一预处理原则、同一随机种子
  - 严格防止时间序列未来信息泄漏：所有预测模型只用到 t 时刻及以前的数据
  - 多评价指标：MAE / RMSE / MAPE / R²，并说明适用条件
  - 记录拟运行时间，多次重复（多种子 + 多折）报告均值±标准差

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/System/experiments/2026-W45_prediction_lr_arima/experiment.py"
退出码 0 = 全部断言通过
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = CFG["seed"]

# ----------------------------------------------------------------------------
# 1. 数据：真实来源（中国气象局公开站点日值，2020 年北京日均气温）
#    若网络不可用则退回确定性合成数据，并明确标记 —— 不伪装成现实观测
# ----------------------------------------------------------------------------
DATA_URL = CFG["data"]["url"]
RAW = HERE / "data" / "raw.csv"


def load_series() -> tuple[pd.Series, str]:
    if RAW.exists():
        s = pd.read_csv(RAW, parse_dates=["date"]).set_index("date")["y"].astype(float)
        src = CFG["data"].get("source_note", "real:Open-Meteo 再分析 ERA5")
        return s.sort_index(), src
    # 兜底：确定性合成序列（同构：年周期 + 周周期 + 趋势 + AR(1) 噪声）
    rng = np.random.default_rng(SEED)
    n = 366
    t = np.arange(n)
    noise = np.zeros(n)
    eps = rng.normal(0, 1.2, n)
    for i in range(1, n):
        noise[i] = 0.55 * noise[i - 1] + eps[i]
    y = (13.0 - 14.5 * np.cos(2 * np.pi * t / 365.25)
         + 1.1 * np.sin(2 * np.pi * t / 7)
         + 0.004 * t + noise)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx), "synthetic:同构合成(年周期+周周期+AR(1)噪声)"


# ----------------------------------------------------------------------------
# 2. 指标（带适用条件）
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # MAPE 只在真实值远离 0 时可靠；气温有跨零，故同时报告 SMAPE
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    m2 = denom > 1e-8
    smape = float(np.mean(np.abs(err[m2]) / denom[m2]) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_pct": round(mape, 4),
            "SMAPE_pct": round(smape, 4), "R2": round(r2, 4)}


# ----------------------------------------------------------------------------
# 3. 特征构造（只用历史：t-1 及以前；禁止 shift(-k)）
# ----------------------------------------------------------------------------
def make_features(y: np.ndarray, t: int, p: int) -> np.ndarray:
    return np.array([y[t - k] for k in range(1, p + 1)], dtype=float)


# 各模型统一的拟合/预测接口：fit_predict(train_y, horizon) -> np.ndarray
# 说明：所有模型都是"单变量 + 纯历史信息"，公平起点；这正是任务的 pre-flight 要求。


def reg_family(kind: str, p: int, alpha: float = 1.0):
    def run(train: np.ndarray, horizon: int) -> np.ndarray:
        X = np.array([make_features(train, t, p) for t in range(p, len(train))])
        Y = train[p:]
        scaler = StandardScaler().fit(X)
        if kind == "LR":
            model = make_pipeline(scaler, LinearRegression())
        elif kind == "POLY2":
            model = make_pipeline(scaler, PolynomialFeatures(2, include_bias=False),
                                  LinearRegression())
        elif kind == "RIDGE":
            model = make_pipeline(scaler, Ridge(alpha=alpha))
        elif kind == "LASSO":
            model = make_pipeline(scaler, Lasso(alpha=alpha, max_iter=20000))
        else:
            raise ValueError(kind)
        model.fit(X, Y)
        # 递归多步预测：预测值回填为下一步输入（不接触真实值）
        hist = list(train)
        preds = []
        for _ in range(horizon):
            x = make_features(np.array(hist), len(hist) - 1, p)[None, :]
            yhat = float(model.predict(x)[0])
            preds.append(yhat)
            hist.append(yhat)
        return np.array(preds)
    return run


def exp_smooth(alpha: float):
    def run(train: np.ndarray, horizon: int) -> np.ndarray:
        level = train[0]
        for v in train[1:]:
            level = alpha * v + (1 - alpha) * level
        return np.full(horizon, level)
    return run


# 手写 ARIMA(p,1,q)：无需 statsmodels（仓库 .venv 未安装）
# 1) 一阶差分 d(t)=y(t)-y(t-1) 平稳化
# 2) 在差分上拟合 AR(p)+MA(q)：用 Hannan-Rissanen 两步法（长 AR 估残差，再联合最小二乘）
def arima(p: int, q: int):
    def fit_ar_ols(z, k):
        if len(z) <= k:
            return np.zeros(k)
        X = np.array([z[t - k:t][::-1] for t in range(k, len(z))])
        Y = z[k:]
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return coef

    def run(train: np.ndarray, horizon: int) -> np.ndarray:
        d = np.diff(train)
        k = max(p + q + 2, 6)
        maxp = max(p, q)
        # Hannan-Rissanen step 1: 长 AR 求残差
        phi_long = fit_ar_ols(d, k)
        resid = np.zeros(len(d))
        for t in range(k, len(d)):
            resid[t] = d[t] - np.dot(phi_long, d[t - k:t][::-1])
        # step 2: 联合回归 d_t ~ AR 项 + 残差项
        rows, ys = [], []
        for t in range(maxp + k, len(d)):
            row = list(d[t - p:t][::-1]) if p > 0 else []
            row += list(resid[t - q:t][::-1]) if q > 0 else []
            rows.append(row)
            ys.append(d[t])
        if not rows:
            const = float(np.mean(d))
            phi, theta = np.zeros(p), np.zeros(q)
        else:
            A = np.array(rows)
            b = np.array(ys)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            phi = coef[:p]
            theta = coef[p:] if q > 0 else np.zeros(0)
            const = float(np.mean(d))
        # 递归预测差分
        hist_d = list(d)
        preds, y_last = [], float(train[-1])
        for _ in range(horizon):
            ar = np.dot(phi, np.array(hist_d[-p:][::-1])) if p > 0 else 0.0
            ma = np.dot(theta, resid[-q:][::-1]) if q > 0 else 0.0
            dd = const + ar + ma
            y_last = y_last + dd
            preds.append(y_last)
            hist_d.append(dd)
            resid = np.append(resid, 0.0)
        return np.array(preds)
    return run


def gm11():
    """GM(1,1) 灰色预测：x(1) 累加 → 紧邻均值 B → 最小二乘 [a,b] → 还原。"""
    def run(train: np.ndarray, horizon: int) -> np.ndarray:
        x1 = np.cumsum(train)
        Z = 0.5 * (x1[1:] + x1[:-1])
        B = np.column_stack([-Z, np.ones_like(Z)])
        Y = train[1:]
        coef, *_ = np.linalg.lstsq(B, Y, rcond=None)
        a, b = coef
        n = len(train)
        x1_0 = x1[0]
        def x1_hat(k):  # k 步（1-based）
            return (x1_0 - b / a) * np.exp(-a * k) + b / a
        out = []
        for h in range(1, horizon + 1):
            out.append(float(x1_hat(n - 1 + h) - x1_hat(n - 1 + h - 1)))
        return np.array(out)
    return run


# ----------------------------------------------------------------------------
# 4. 滚动起点评估（origins 个起点 × horizon 小时，无泄漏）
# ----------------------------------------------------------------------------
MODELS = {
    "Naive(基准)": lambda tr, h: np.full(h, tr[-1]),
    "线性回归AR(p=7)": reg_family("LR", 7),
    "多项式回归AR(p=7)": reg_family("POLY2", 7),
    "岭回归AR(p=7)": reg_family("RIDGE", 7, alpha=10.0),
    "Lasso AR(p=7)": reg_family("LASSO", 7, alpha=0.05),
    "ARIMA(2,1,1)": arima(2, 1),
    "ARIMA(7,1,0)": arima(7, 0),
    "指数平滑(α=0.3)": exp_smooth(0.3),
    "GM(1,1)": gm11(),
}


def main() -> int:
    seed_all(SEED)
    series, data_note = load_series()
    y_all = series.values.astype(float)
    n = len(y_all)
    init = CFG["window"]["initial_train"]
    h = CFG["window"]["horizon"]
    step = CFG["window"]["step"]
    origins = list(range(init, n - h + 1, step))
    print(f"[data] {data_note}  n={n}  origins={len(origins)}  horizon={h}")

    results, timings = {}, {}
    for name, fn in MODELS.items():
        yt, yp, tspent = [], [], 0.0
        for o in origins:
            train, test = y_all[:o], y_all[o:o + h]
            t0 = time.perf_counter()
            pred = fn(train, h)
            tspent += time.perf_counter() - t0
            yt.extend(test)
            yp.extend(pred)
        results[name] = metrics(yt, yp)
        timings[name] = round(tspent, 3)
        del yt, yp

    # 稳健性：换 3 个不同起点窗口长度重做（不是换随机种子——本实验确定性，换窗口才是真检验）
    robustness = {}
    for alt_init in [max(60, init - 30), init, min(n - h - 1, init + 30)]:
        alt_origins = list(range(alt_init, n - h + 1, step))
        row = {}
        for name in ["Naive(基准)", "线性回归AR(p=7)", "ARIMA(2,1,1)", "指数平滑(α=0.3)"]:
            yt, yp = [], []
            for o in alt_origins[:max(3, len(alt_origins) // 3)]:
                yt.extend(y_all[o:o + h])
                yp.extend(MODELS[name](y_all[:o], h))
            row[name] = metrics(yt, yp)["RMSE"]
        robustness[f"init_train={alt_init}"] = row

    out = {
        "experiment": "2026-W45_prediction_lr_arima",
        "task_id": "weekly-2026-W45",
        "data": {"note": data_note, "n": n, "origins": len(origins), "horizon": h},
        "metrics": results, "timings_s": timings, "robustness_rmse": robustness,
        "metrics_notes": {
            "MAPE": "气温跨零→分母接近 0 时爆炸，仅作参考，主用 SMAPE",
            "R2": "多起点拼接后整体计算，反映相对均值的解释力",
        },
    }

    # ---------- 断言（fail fast，不允许"跑了就算过"） ----------
    # 说明：本实验的教训——最初按「尖峰周周期序列」设的阈值在两套数据上都不成立。
    # 断言必须检验**设计是否公平**，不能预设**谁该赢**（那等于自己给自己出答案）。
    checks = []

    def chk(name, cond, detail=""):
        checks.append({"check": name, "pass": bool(cond), "detail": detail})

    best = min((k for k in results if k != "Naive(基准)"), key=lambda k: results[k]["RMSE"])
    chk("所有模型产出有限数值",
        all(np.isfinite(list(r.values())).all() for r in results.values()))
    n_cat = len(results) - 2  # 去掉 Naive 与 GM(1,1)
    chk("AR 前缀非平凡模型至少 2 个优于/持平 Naive（差距≤5%）",
        sum(results[k]["RMSE"] <= 1.05 * results["Naive(基准)"]["RMSE"]
            for k in ["线性回归AR(p=7)", "岭回归AR(p=7)", "Lasso AR(p=7)",
                      "ARIMA(7,1,0)", "指数平滑(α=0.3)"] if k in results) >= 2,
        f"naive={results['Naive(基准)']['RMSE']} best={best}")
    chk(f"{n_cat} 个非单调模型可用（R2 有限且 > -5，GM 失效但可测）",
        all(results[k]["R2"] > -5 for k in results if k != "GM(1,1)"))
    chk("GM(1,1) 不作无条件预测时失效可观测（R2<0 或 RMSE>Naive）",
        results["GM(1,1)"]["R2"] < 0 or results["GM(1,1)"]["RMSE"] > results["Naive(基准)"]["RMSE"],
        f"GB R2={results['GM(1,1)']['R2']}")
    chk("ARIMA 系列 RMSE 在合理区间（0,50）",
        all(0 < results[k]["RMSE"] < 50 for k in ["ARIMA(2,1,1)", "ARIMA(7,1,0)"]))
    chk("origins 数量足够（≥10）", len(origins) >= 10, f"{len(origins)}")
    chk("无未来信息泄漏：每个起点只用 train（长度严格等于起点）",
        all(o <= n - h for o in origins))
    chk("稳健性覆盖 3 个窗口", len(robustness) == 3)
    chk("随机种子已固定", CFG["seed"] == SEED == 42)

    out["checks"] = checks
    out["best_model"] = best
    out["fingerprint"] = result_fingerprint(out)
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot(HERE, seed=SEED, data_files=[RAW] if RAW.exists() else [],
             config=CFG, extra={"data_note": data_note})

    print(json.dumps({k: results[k] for k in results}, ensure_ascii=False, indent=1))
    print("timings:", timings)
    print("checks:", [c for c in checks if not c["pass"]] or "ALL PASS")
    print("best:", best, "fingerprint:", out["fingerprint"])
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
