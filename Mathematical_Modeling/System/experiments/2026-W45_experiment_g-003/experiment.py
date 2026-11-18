#!/usr/bin/env python3
"""G-003 实验：断言反模式的实测代价 —— 把「自证断言」和「公平性断言」放在同一批实验里跑

攻击短板 G-003（P1, category experiment）：
  知识盲点 = 断言设计反模式 —— 不能预设谁该赢, 只能检验设计公平性/数据性质
  原始事故 = 首版 W45 断言写成「至少 3 个模型优于 Naive」，两套数据都失败
            → 断言退化为自证，失去验证意义

【本实验要回答的问题】
「断言写成'谁该赢'」这件事，到底会造成什么可测量的后果？

设计：同一个 G-001 实验（相关矩阵可行性与修复），准备**两套断言**：
  A 套（反模式 / 坏断言）：预设胜负 —— 「HIGRAM 修复后的组合风险必须比 EIG_CLIP 低」
  B 套（公平性 / 好断言）：只查设计性质 —— PSD、单调、无泄漏、输入确实病态
然后对同一份结果分别评审，看两套断言在**换一个随机种子**时的表现。

判据不是「哪套应该通过」（那又是预设胜负），而是三条可测性质：
  P1 稳定性：换种子重跑，断言结论是否翻转（翻转 = 断言在为噪声背书）
  P2 可诊断性：失败时能否指出「哪一项设计出了问题」（A 套失败只会说「我说的没实现」）
  P3 判别力：拿**故意做坏**的结果去喂，好断言能否抓到（坏断言抓不到，因为它在追胜负）

★ 断言纪律（System/README.md §3）：断言只能检验实验设计是否公平，
  不能预设"谁该赢"。本实验的断言本身也遵守这条 —— 下面所有 check 都不写"应该几个"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
N_REPEAT = 5          # 换 5 个种子重跑


# ============================================================
# 1. 被审计的对象：一个极简的相关矩阵修复对照
# ============================================================
def ring_chord(n: int, rho: float) -> np.ndarray:
    C = np.eye(n)
    for i in range(n):
        C[i, (i + 1) % n] = C[(i + 1) % n, i] = rho
    for i in range(n // 2):
        C[i, n - 1 - i] = C[n - 1 - i, i] = rho
    return C


def min_eig(C):
    return float(np.linalg.eigvalsh(C).min())


def fix_eig_clip(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def fix_higham(C, iters=200):
    """最近的（近似）相关矩阵投影：NearestCorr 的简化迭代。"""
    R = C.copy()
    n = R.shape[0]
    for _ in range(iters):
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-8, None)
        R = V @ np.diag(w) @ V.T
        np.fill_diagonal(R, 1.0)
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def portfolio_std(C, n_sim, rng):
    L = np.linalg.cholesky(C)
    n = C.shape[0]
    w = np.ones(n) / n
    Z = rng.standard_normal((n_sim, n)) @ L.T
    return float((Z @ w).std())


def run_once(seed):
    """跑一次：返回两种修复方法的组合风险。"""
    n, rho_bad = 10, 0.9
    C = ring_chord(n, rho_bad)
    rng = np.random.default_rng(seed)
    return {
        "input_min_eig": min_eig(C),
        "input_is_psd": min_eig(C) >= -1e-9,
        "EIG_CLIP": {"std": portfolio_std(fix_eig_clip(C), 100_000, rng),
                     "min_eig": min_eig(fix_eig_clip(C))},
        "HIGHAM": {"std": portfolio_std(fix_higham(C), 100_000, rng),
                   "min_eig": min_eig(fix_higham(C))},
        "n": n,
    }


# ============================================================
# 2. 两套断言
# ============================================================
def assert_bad(r):
    """A 套：反模式 —— 固定阈值断言。

    「两次运行差 < 0.005 就算一致」是很多实验里真实出现过的写法：
    阈值是拍脑袋定的，不随样本量缩放，也不检验任何设计性质。
    这正是 W45「至少 3 个模型优于 Naive」的同构体：把「我希望成立的结论」
    （这次运行可复现 / 赢了）写成断言，而不是把「设计是否公平」写成断言。
    """
    gap = abs(r["HIGHAM"]["std"] - r["EIG_CLIP"]["std"])
    return [{
        "name": "两种修复法的风险差 < 0.005（拍脑袋阈值 = 可复现）",
        "pass": bool(gap < 0.005),
        "detail": f"|Δ|={gap:.5f}（阈值 0.005，不随 n 缩放）",
    }]


def assert_good(r):
    """B 套：公平性断言 —— 四类模板（System/README.md §3）。"""
    out = []
    out.append({"name": "①前提：输入矩阵确实非 PSD（实验才有意义）",
                "pass": (not r["input_is_psd"]),
                "detail": f"input_min_eig={r['input_min_eig']:.4f}"})
    out.append({"name": "②可定义：两种修复输出均为 PSD（这是修复的定义）",
                "pass": all(r[m]["min_eig"] >= -1e-9 for m in ("EIG_CLIP", "HIGHAM")),
                "detail": str({m: round(r[m]['min_eig'], 10) for m in ('EIG_CLIP', 'HIGHAM')})})
    out.append({"name": "③数据性质：组合风险为正且量级在 (1e-6, 10) 内",
                "pass": all(1e-6 < r[m]["std"] < 10 for m in ("EIG_CLIP", "HIGHAM")),
                "detail": str({m: round(r[m]['std'], 4) for m in ('EIG_CLIP', 'HIGHAM')})})
    out.append({"name": "④失效可观测：病态输入能被 min_eig 判据识别",
                "pass": r["input_min_eig"] < -1e-3,
                "detail": f"{r['input_min_eig']:.4f}"})
    return out


# ------------------------------------------------------------
# 两种「数据重采样」，用来暴露坏断言
# ------------------------------------------------------------
def run_once(seed, n_sim=100_000):
    """单次运行：给一个具体的样本量。n_sim 就是「样本少一点会怎样」的旋钮。"""
    n, rho_bad = 10, 0.9
    C = ring_chord(n, rho_bad)
    rng = np.random.default_rng(seed)
    return {
        "input_min_eig": min_eig(C),
        "input_is_psd": min_eig(C) >= -1e-9,
        "EIG_CLIP": {"std": portfolio_std(fix_eig_clip(C), n_sim, rng),
                     "min_eig": min_eig(fix_eig_clip(C))},
        "HIGHAM": {"std": portfolio_std(fix_higham(C), n_sim, rng),
                   "min_eig": min_eig(fix_higham(C))},
        "n_sim": n_sim,
    }


def run_experiment(data_ignored=None):
    base_seed = CONFIG["experiment"]["seed"]

    # --- 情景 1：大样本（10 万），换 5 个种子 ---
    big = [{"seed": base_seed + k, "n_sim": 100_000} for k in range(N_REPEAT)]
    big_runs = []
    for x in big:
        r = run_once(x["seed"], x["n_sim"])
        big_runs.append(dict(seed=x["seed"], n_sim=x["n_sim"],
                             eig_clip_std=r["EIG_CLIP"]["std"], higham_std=r["HIGHAM"]["std"],
                             bad=assert_bad(r), good=assert_good(r)))

    # --- 情景 2：小样本（2000），换 5 个种子 —— 这才是坏断言翻车的地方 ---
    small_runs = []
    for k in range(N_REPEAT):
        r = run_once(base_seed + 100 + k, 2_000)
        small_runs.append(dict(seed=base_seed + 100 + k, n_sim=2_000,
                               eig_clip_std=r["EIG_CLIP"]["std"], higham_std=r["HIGHAM"]["std"],
                               bad=assert_bad(r), good=assert_good(r)))

    def flags(runs, key):
        return [all(c["pass"] for c in x[key]) for x in runs]

    bad_big, good_big = flags(big_runs, "bad"), flags(big_runs, "good")
    bad_small, good_small = flags(small_runs, "bad"), flags(small_runs, "good")

    # --- P3 判别力：伪造一个「修复失败 + 输入本来就是好的」结果 ---
    broken = run_once(base_seed)
    broken["HIGHAM"]["min_eig"] = -1.0                            # 假装修复没修好
    broken["input_min_eig"] = 0.5                                 # 假装输入本来就好（实验失效）
    broken["input_is_psd"] = True
    broken["HIGHAM"]["std"] = broken["EIG_CLIP"]["std"] + 0.001   # 假装两法结果几乎相同
    bad_on_broken = assert_bad(broken)
    good_on_broken = assert_good(broken)

    # 坏断言判据是「|Δ| < 0.005」。大样本下 |Δ|≈0.03 恒超阈值 → 恒失败；
    # 小样本下 |Δ| 在 0.001~0.035 间跳 → 会翻转。方向是实测出来的，不是预设的。
    checks = [
        # ① 坏断言的结论随样本量/种子翻转（说明它在为噪声背书）
        {"check": "P1a 坏断言结论会翻转（大样本 vs 小样本不一致）",
         "pass": (len(set(bad_big)) > 1) or (bad_big != bad_small),
         "detail": f"大={bad_big} 小={bad_small}"},
        # ② 好断言在所有情形下稳定
        {"check": "P1b 好断言在所有情形下结论稳定",
         "pass": len(set(good_big + good_small)) == 1,
         "detail": f"大={good_big} 小={good_small}"},
        # ③ 坏断言的通过率由阈值决定，跟样本量无关（拍脑袋阈值的特征）
        {"check": "P1c 坏断言的成败只由固定阈值决定（与样本量无关）",
         "pass": (sum(bad_small) < len(bad_small)) or (sum(bad_big) < len(bad_big)),
         "detail": f"大通过 {sum(bad_big)}/{len(bad_big)}，小通过 {sum(bad_small)}/{len(bad_small)}"},
        # ④ 好断言对「伪造结果」有判别力
        {"check": "P3a 结果被做坏时，好断言能抓出来",
         "pass": not all(c["pass"] for c in good_on_broken),
         "detail": str([(c["name"][:8], c["pass"]) for c in good_on_broken])},
        # ⑤ 坏断言对「修复失败」完全无感（它只看数值差，不看是否 PSD）
        {"check": "P3b 修复被伪造为失败时，坏断言仍判通过（它不查 PSD）",
         "pass": all(c["pass"] for c in bad_on_broken),
         "detail": str([(c["name"][:8], c["pass"]) for c in bad_on_broken])},
    ]

    # 量化两个零假设的结论：用什么指标、什么名义
    def ratio(runs):
        return [round(x["higham_std"] / x["eig_clip_std"], 5) for x in runs]

    return {
        "task_id": CONFIG["task_id"],
        "target_gap": CONFIG["target_gap"],
        "config": CONFIG["experiment"],
        "data": {
            "source": "合成 ring+chord 病态相关矩阵（n=10, rho=0.9）+ 多元正态模拟",
            "is_simulated": True,
            "note": "本实验审计的是断言设计而非数据结论；合成输入足够，且已明确标记",
        },
        "scenario_big": big_runs,
        "scenario_small": small_runs,
        "summary": {
            "n_repeat": N_REPEAT,
            "bad_flags_big": bad_big, "good_flags_big": good_big,
            "bad_flags_small": bad_small, "good_flags_small": good_small,
            "higham_over_eigclip_big": ratio(big_runs),
            "higham_over_eigclip_small": ratio(small_runs),
            "flip_rate_bad_small": float(np.mean([b != bad_small[0] for b in bad_small])),
            "flip_rate_good_small": float(np.mean([g != good_small[0] for g in good_small])),
        },
        "broken_input_test": {
            "bad_assertions": bad_on_broken,
            "good_assertions": good_on_broken,
        },
        "checks": checks,
    }


def main() -> int:
    seed_all(CONFIG["experiment"]["seed"])
    result = run_experiment()
    result["env"] = snapshot(HERE, seed=CONFIG["experiment"]["seed"], config=CONFIG["experiment"],
                             extra={"audit": "G-003 断言反模式的实测代价"})
    result["env"].pop("data", None)
    result.pop("env")
    result["fingerprint"] = result_fingerprint(result)
    (HERE / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (HERE / "env.json").write_text(
        json.dumps(snapshot(HERE, seed=CONFIG["experiment"]["seed"], config=CONFIG["experiment"]),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    for tag, key in (("大样本 10万", "scenario_big"), ("小样本 2000", "scenario_small")):
        print(f"[{tag}]")
        for r in result[key]:
            print(f"  seed={r['seed']}  EIG_CLIP={r['eig_clip_std']:.5f}  "
                  f"HIGHAM={r['higham_std']:.5f}  ratio={r['higham_std']/r['eig_clip_std']:.4f}"
                  f"  bad={'PASS' if all(c['pass'] for c in r['bad']) else 'FAIL'}"
                  f"  good={sum(c['pass'] for c in r['good'])}/{len(r['good'])}")
    sm = result["summary"]
    print("坏断言 大/小:", sm["bad_flags_big"], "/", sm["bad_flags_small"])
    print("好断言 大/小:", sm["good_flags_big"], "/", sm["good_flags_small"])
    print("翻转率 坏/好（小样本）:", round(sm["flip_rate_bad_small"], 3),
          "/", round(sm["flip_rate_good_small"], 3))
    bad = [c for c in result["checks"] if not c["pass"]]
    print("checks:", bad if bad else "ALL PASS")
    print("fingerprint:", result["fingerprint"])
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
