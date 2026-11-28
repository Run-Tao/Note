#!/usr/bin/env python3
"""2026-W48 每周实验：G-004 退化情形验证 —— 并列最优 / 多解集合比对

★ 背景：benchmark-2026-11-B1 的 S1+S3 与 S1+S2 **同价 15682.20**，而评审脚本按「唯一解」
  判分，把正确的等价解判成错误。本实验不复现那道赛题打分器，而是**用一个构造好的退化实例
  把「并列最优」的三种真实形态造出来**，然后量化三件事：
    Q1 同价解有几族？一个「只比一个解」的判分器对正确解族的**漏放率**是多少？
    Q2 集合比对（JS 距离 / 对称差占比）比单解比对**早多少步**能把「另一个正确解」区分出来？
    Q3 退化实例上「选一个解去比较」用不同 tie-break 会产生多大的假差异？
  以及实验自身要证明的：**不做二分搜索、不动 eps，能否判出「这题退化」**。

数据：**模拟/合成**（构造自明、参数全写在 config 与本文档；见 data/README.md）
纪律（System/README.md §3）：
  - 断言只检验设计公平性（判分器对 correct/incorrect 的区分能力、覆盖率逻辑），不预设"哪个模型赢"
  - 本实验不横向比较模型，只比较**判分口径**；`models_to_compare` 里的 LP/MLR/RIDGE 骨架占位不用
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lib.env_record import seed_all, snapshot, result_fingerprint  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = CONFIG["experiment"]["seed"]

TOL = 1e-9          # 「同价」的判定容差（现实打分器用 1e-6；这里收紧，避免把数值噪声当并列）
TIE_MARGIN = 1e-3   # 「并列最优」的业务容差：性价比差 0.1% 内都算同价（报告口径）


# ---------- 数据 / 实例 ----------
def build_instance(n_items: int = 6, seed: int = SEED):
    """构造 0-1 背包式退化实例：前 n_items 个项目两两成对，配对价值比完全相等，
    容量恰好卡在「每对只能选一个」的边界上 → 产生 2^(n_pairs) 个同价最优解。"""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        w = int(rng.integers(5, 20))
        v = int(rng.integers(5, 20))
        items.append({"id": f"P{i}", "w": w, "v": v})
    # 强制配对：把 (0,1),(2,3),(4,5) 改成完全同价同重（副本对）—— 这就是退化来源
    pairs = [(0, 1), (2, 3), (4, 5)]
    for a, b in pairs:
        items[b]["w"] = items[a]["w"]
        items[b]["v"] = items[a]["v"]
    cap = sum(min(items[a]["w"], items[b]["w"]) for a, b in pairs)
    return {"items": items, "capacity": cap, "pairs": [list(p) for p in pairs],
            "n": n_items}


def enumerate_all(inst):
    """穷举 2^n 全部子集，返回 (mask, weight, value)。n<=12，可穷举。"""
    rows = []
    n = inst["n"]
    w = np.array([it["w"] for it in inst["items"]])
    v = np.array([it["v"] for it in inst["items"]])
    for mask in range(1 << n):
        sel = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
        if w[sel].sum() > inst["capacity"]:
            continue
        rows.append({"mask": mask, "w": int(w[sel].sum()), "v": int(v[sel].sum())})
    return rows


# ---------- 判分口径 ----------
def score_single(inst, cand_mask: int, opt: dict) -> dict:
    """口径 A：只比一个解（现状打分器的做法）。"""
    v = sum(inst["items"][i]["v"] for i in range(inst["n"]) if (cand_mask >> i) & 1)
    w = sum(inst["items"][i]["w"] for i in range(inst["n"]) if (cand_mask >> i) & 1)
    feasible = w <= inst["capacity"]
    correct = feasible and abs(v - opt["v"]) <= TOL
    return {"feasible": feasible, "value": int(v), "weight": int(w),
            "correct_by_single": bool(correct)}


def score_set(cand_masks, opt_family, opt_v: int, n: int) -> dict:
    """口径 B：解集合比对（允许提交多个等价解；一个解命中即算对，同时报告精确匹配度）。"""
    fam = set(opt_family)
    hits = [m for m in cand_masks if m in fam]
    exact = bool(cand_masks) and set(cand_masks) == fam
    return {"n_candidates": len(cand_masks), "n_in_family": len(hits),
            "covered": bool(hits), "exact_match": exact}


def jaccard(masks_a, masks_b) -> float:
    a, b = set(masks_a), set(masks_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------- 主实验 ----------
def run_experiment(data):
    inst = data["inst"]
    n = inst["n"]
    rows = enumerate_all(inst)
    best_v = max(r["v"] for r in rows)
    opt_family = sorted(r["mask"] for r in rows if abs(r["v"] - best_v) <= TOL)
    opt = {"v": best_v, "family": opt_family, "n_opt": len(opt_family)}

    # ---- Q1 并列最优的族结构与单解判分的漏放率 ----
    q1 = []
    for r in rows:
        s = score_single(inst, r["mask"], opt)
        q1.append({**s, "mask": r["mask"],
                   "in_opt_family": r["mask"] in set(opt_family)})

    n_equiv_correct = sum(1 for x in q1 if x["correct_by_single"])

    # 单解判分器若 tie-break 选错（选到族外同价？不存在）→ 它拒绝多少族内解：
    # 现实打分器拿到「一个正确解」是能判对的；错的是**它只按第一个解建立基准**，
    # 于是别的族内解一律被判「与基准不符」。这里量化：以第一个族内解为基准，
    # 其余族内解在「单解比对」下被判错的比例。
    base_mask = opt_family[0]
    base_vec = np.array([(base_mask >> i) & 1 for i in range(n)])
    # 评审脚本的「解集合比对」典型实现：拿**基准解**做集合（可能只有 1 个，也可能是一族），
    # 然后要求候选解在集合内。下面两种基准都要量：
    #   (a) 基准 = 单解 → 族内其他解全部被判错（无提示的判定）；
    #   (b) 基准 = 完整族 → 候选真命中集合，集合口径值钱
    base_set = {base_mask}
    cand_hit_global_family = [m for m in opt_family if m in set(opt_family)]
    off_family_vecs = {m: np.array([(m >> i) & 1 for i in range(n)]) for m in opt_family}
    hamming_to_base = {m: int((v != base_vec).sum()) for m, v in off_family_vecs.items()}
    wrongly = sum(1 for m, d in hamming_to_base.items() if m != base_mask and d > 0)

    # ---- Q2 集合比对 vs 单解比对：谁更早承认「等价」 ----
    # 参照集 = 完整最优族 vs 参照集 = 基准单解（现实中两种都存在）
    q2 = []
    for m in opt_family:
        single = score_single(inst, m, opt)["correct_by_single"]
        st_full = score_set([m], opt_family, best_v, n)
        st_base = score_set([m], [base_mask], best_v, n)
        cand_vec = np.array([(m >> i) & 1 for i in range(n)])
        q2.append({"mask": int(m), "hamming_to_base": int((cand_vec != base_vec).sum()),
                   "single_vs_base_only": bool(np.array_equal(cand_vec, base_vec)),
                   "set_vs_base_only": st_base["covered"],
                   "set_vs_full_family": st_full["covered"]})
    q2_full_pass = sum(1 for x in q2 if x["set_vs_full_family"])
    q2_base_pass = sum(1 for x in q2 if x["set_vs_base_only"])

    # ---- Q2b 「差一点」的候选：族外但同价值？→ 本实例不存在；改量「全子集中同价值者」
    same_value_any = [r["mask"] for r in rows if r["v"] == best_v]
    same_value_info = {"n_same_value_in_feasible": len(same_value_any),
                       "equals_opt_family": sorted(same_value_any) == opt_family}

    # ---- Q3 tie-break 假差异：同一目标值下，不同「选哪个解”会造出假差异 ----
    weights = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    # 只取族里两个解，比较它们在「辅助指标」上的差 —— 辅助指标与主目标无关时的假差异
    aux = {}
    for m in opt_family[: min(len(opt_family), 8)]:
        vec = np.array([(m >> i) & 1 for i in range(n)])
        aux[m] = {
            "n_selected": int(vec.sum()),
            "weighted_sum": float((vec * weights).sum()),
            "first_item_included": int(vec[0]),
        }
    aux_ranges = {k: max(v[k] for v in aux.values()) - min(v[k] for v in aux.values())
                  for k in ("n_selected", "weighted_sum", "first_item_included")}

    # ---- Q3b 判分器「早停早判」：族内前 k 个解一一提交时的表现 ----
    early = []
    for k in range(1, min(len(opt_family), 8) + 1):
        sub = opt_family[:k]
        single_pass = sum(score_single(inst, m, opt)["correct_by_single"] for m in sub)
        set_pass = score_set(sub, opt_family, best_v, n)["covered"]
        early.append({"k_submitted": k,
                      "single_correct": int(single_pass),
                      "set_covered": bool(set_pass)})

    # ---- Q4 可行性 / 构造自检（把边界情形钉住） ----
    capl = inst["capacity"]
    tight = [r for r in rows if abs(r["v"] - best_v) <= TOL]
    loose_family = sorted(r["mask"] for r in rows if abs(r["v"] - best_v) <= TIE_MARGIN * max(1, best_v))
    nondegenerate = len(opt_family) == 1

    # 反例：把容量 +1 是否会让族变小 / 变大（退化消失或加剧）
    inst2 = json.loads(json.dumps(inst))
    inst2["capacity"] = capl + min(it["w"] for it in inst["items"])
    rows2 = enumerate_all(inst2)
    best2 = max(r["v"] for r in rows2)
    family2 = sorted(r["mask"] for r in rows2 if abs(r["v"] - best2) <= TOL)

    # 反例 2：把**每一对**里的拷贝价值都扰动掉（只扰一对时其余两对仍在撑族）
    #   —— 首版只改 items[1]，族仍为 4，A6 当场 FAIL。这是真实的实验设计错误，保留记录。
    inst3 = json.loads(json.dumps(inst))
    for k, (a, b) in enumerate(inst["pairs"]):
        inst3["items"][b]["v"] = inst3["items"][a]["v"] + (k + 1)
    rows3 = enumerate_all(inst3)
    best3 = max(r["v"] for r in rows3)
    family3 = sorted(r["mask"] for r in rows3 if abs(r["v"] - best3) <= TOL)

    # ---- 断言（只检验设计公平性） ----
    asserts = []
    def check(name, cond, detail=""):
        asserts.append({"name": name, "pass": bool(cond), "detail": str(detail)})

    check("A1 构造成立：存在 >1 个同价最优解（否则本实验无意义）",
          len(opt_family) > 1, f"n_opt={len(opt_family)}")
    check("A2 族内每个解都真可行（不是靠放宽容量造出来的）",
          all(r["mask"] in set(opt_family) and r["w"] <= capl for r in tight),
          f"cap={capl}")
    check("A3 单解基准判分对「族内但非基准」的解漏放 > 0（这正是 B1 事故的机理）",
          wrongly > 0, f"wrongly={wrongly}/{len(opt_family)-1}")
    check("A4 集合比对（基准=完整族）对族内任一解全部放行；**基准只给单解时集合比对退化成单解比对**",
          q2_full_pass == len(opt_family) and q2_base_pass == 1,
          f"full={q2_full_pass}/{len(opt_family)}  base_only={q2_base_pass}/{len(opt_family)}")
    check("A5 反例 1：容量放宽一个单位后族结构发生变化（说明退化对参数敏感、可判别）",
          family2 != opt_family, f"{len(opt_family)}->{len(family2)}")
    check("A6 反例 2：破坏全部配对价值后族缩为 1（成功造出非退化对照）",
          len(family3) == 1, f"n_opt={len(family3)}")
    check("A7 判分容差 1e-9 下「同价」为精确相等（不靠 eps 蒙）",
          max(abs(r["v"] - best_v) for r in tight) <= TOL, "tol=1e-9")
    check("A8 tie-break 假差异确实存在（辅助指标在族内并非唯一）",
          aux_ranges["weighted_sum"] > 0, f"range={aux_ranges['weighted_sum']}")

    return {
        "inst": inst,
        "opt_value": int(best_v),
        "opt_family": opt_family,
        "n_opt": len(opt_family),
        "q1_single_scoring": {"n_candidates": len(q1),
                              "n_equivalent_correct": n_equiv_correct,
                              "n_family_wrongly_rejected": wrongly,
                              "hamming_to_base": hamming_to_base},
        "q2_set_vs_single": q2,
        "q2b_same_value": same_value_info,
        "q3_aux_under_ties": aux_ranges,
        "q3_aux_raw": {str(k): v for k, v in aux.items()},
        "q3b_early_stop": early,
        "q4_counterexamples": {"cap_relaxed_family_size": len(family2),
                               "pair_broken_family_size": len(family3),
                               "typo_cap_in": capl, "typo_cap_out": inst2["capacity"]},
        "assertions": asserts,
        "data_note": "模拟/合成：0-1 背包退化实例，参数见 config.json seed 与 build_instance，可完全重造",
    }


def main() -> int:
    seed_all(SEED)
    data = {"inst": build_instance()}
    result = run_experiment(data)
    result["task_id"] = CONFIG["task_id"]
    result["config"] = CONFIG["experiment"]
    # 本实验无磁盘数据文件（实例由 seed 生成），如实记录 data_files=[] 而不伪造
    result["env"] = snapshot(HERE, seed=SEED, data_files=[],
                             config=CONFIG["experiment"],
                             extra={"data_note": "实例由 seed 生成，"
                                    "重跑 build_instance(seed) 可完全复现；无外部数据文件"})
    result["fingerprint"] = result_fingerprint(result)
    (HERE / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (HERE / "env.json").write_text(
        json.dumps(result["env"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("task_id:", result["task_id"])
    print("opt_value:", result["opt_value"], " n_opt:", result["n_opt"])
    print("opt_family:", result["opt_family"])
    print("族内被单解基准误判数:", result["q1_single_scoring"]["n_family_wrongly_rejected"])
    print("tie-break 辅助指标假差异:", result["q3_aux_under_ties"])
    print("反例:", result["q4_counterexamples"])
    allpass = all(a["pass"] for a in result["assertions"])
    for a in result["assertions"]:
        print(("PASS " if a["pass"] else "FAIL "), a["name"], "|", a["detail"])
    print("断言:", "全过" if allpass else "有失败")
    print("fingerprint:", result["fingerprint"])
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
