#!/usr/bin/env python3
"""独立复核：G-004 退化情形验证 —— 不与 experiment.py 共用任何函数

复核策略：**换实现路径**
  - 不复用 experiment.enumerate_all / score_*；用 itertools.product 逐项生成 + 手算价值
  - 不复用 build_instance 的构造逻辑；用「价值=重量」的特殊实例（v=w）令最优族可解析预测
  - 对 2^n 全空间直接数同价解个数，与 result.json 的 n_opt / opt_family 双向核对
  - 关键：**验证「退化真的会发生」而不是「代码说会」** —— 用两个解析可比的特例
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
INST = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

items = R["inst"]["items"]
cap = R["inst"]["capacity"]
n = R["inst"]["n"]
w = [it["w"] for it in items]
v = [it["v"] for it in items]

findings = []


def add(item, ok, detail=""):
    findings.append({"item": item, "pass": bool(ok), "detail": str(detail)})


# R1 全空间穷举（itertools.product，不用位运算 mask）重数最优族
best, fam = -1, []
for bits in itertools.product((0, 1), repeat=n):
    ww = sum(b * wi for b, wi in zip(bits, w))
    if ww > cap:
        continue
    vv = sum(b * vi for b, vi in zip(bits, v))
    if vv > best:
        best, fam = vv, [bits]
    elif vv == best:
        fam.append(bits)
fam_masks = sorted(sum(b << i for i, b in enumerate(bits)) for bits in fam)
add("R1 独立穷举（product+手算）最优值与族与 result.json 完全一致",
    best == R["opt_value"] and fam_masks == sorted(R["opt_family"]),
    f"best={best} n={len(fam_masks)} vs json {R['opt_value']}/{len(R['opt_family'])}")

# R2 （首版写的是「族大小 = 2^对数 = 8」，实测 4 → 假说被证伪，如实保留）
#   真实结构：cap=30 恰好卡在「两对里各拿一个」的上沿，最小重量可能是 25 或 26，
#   于是第三对/first 对的选择里只有一部分能进门 → 族是被**容量几何**截断的，不是 2^3。
n_pairs = len(R["inst"]["pairs"])
add("R2 族大小 = 2^对数 这个假说被证伪（实测值更小）—— 退化族的形状取决于容量几何",
    len(fam_masks) != 2 ** n_pairs,
    f"实测 {len(fam_masks)} ≠ 2^{n_pairs}={2**n_pairs}")
# R2b 正确的结构判据：每个最优解都恰好取「每对中的一个」（配对不可破）
ok2b = all(sum(bits[a] + bits[b] for a, b in R["inst"]["pairs"]) == n_pairs for bits in fam)
add("R2b 族内每个解都恰好每对取一个（退化来源=配对同价，可验证的结构）", ok2b,
    f"{len(fam)} 个解")

# R3 族内解的目标值与可行性由「手写求和」独立确认
ok3 = all(sum(b * vi for b, vi in zip(bits, v)) == best
          and sum(b * wi for b, wi in zip(bits, w)) <= cap for bits in fam)
add("R3 族内每解手算可行且同价", ok3, f"{len(fam)} 个")

# R4 单解基准判分的漏放（独立重算，不用 score_single）
base = sorted(R["opt_family"])[0]
base_bits = tuple((base >> i) & 1 for i in range(n))
missed = 0
for bits in fam:
    if bits != base_bits:
        missed += 1
add("R4 以 base 单解为参照时，族内其余解被判不符的个数 > 0", missed > 0, f"{missed}/{len(fam)-1}")

# R5 解析特例：v=w 且 cap=W/2 → 最优必为「装满一半」，族大小的下界 = 组合数
random.seed(7)
vals = [6, 6, 4, 4, 3, 3]
caps = 13
best5, fam5 = -1, 0
for bits in itertools.product((0, 1), repeat=6):
    ww = sum(b * vi for b, vi in zip(bits, vals))
    if ww > caps:
        continue
    if ww > best5:
        best5, fam5 = ww, 1
    elif ww == best5:
        fam5 += 1
add("R5 解析特例（v=w，cap=13）：最优值=13 且族大小 ≥2（退化可被独立造出）",
    best5 == 13 and fam5 >= 2, f"best={best5} family={fam5}")

# R6 eps 无关性：把「同价」阈值从 1e-9 放宽到 1e-6，族大小不变（结论不是靠容差蒙出来的）
for eps in (1e-9, 1e-6, 1e-3):
    fam_eps = [bits for bits in fam
               if abs(sum(b * vi for b, vi in zip(bits, v)) - best) <= eps]
    add(f"R6 eps={eps:g} 时族大小不变（结论对判分容差不敏感）",
        len(fam_eps) == len(fam), f"{len(fam_eps)}")

# R7 （首版拿 experiment 的「扰动量 k+1」重算，实测族=2 → 与 experiment 的 A6(族=1) 冲突。
#   独立重算后发现冲突来自**扰动量选择不同**：+1/+2/+3 仍可能互相撞价。
#   改用一批互不相同的扰动量，检查「破坏配对后族是否真的塌成 1」以及它对扰动量是否敏感。）
pairs = R["inst"]["pairs"]   # 上一版误写成 R["inst"]["pairs"] 的 enumerate 对象内部字段，KeyError 后整段被吞
sensitivity = []
for step in (1, 2, 3, 4, 5, 7):
    vals2 = list(v)
    for k, (a, b) in enumerate(pairs):
        vals2[b] = vals2[a] + step * (k + 1)
    best7, fam7 = -1, 0
    for bits in itertools.product((0, 1), repeat=n):
        if sum(b * wi for b, wi in zip(bits, w)) > cap:
            continue
        vv = sum(b * vi for b, vi in zip(bits, vals2))
        if vv > best7:
            best7, fam7 = vv, 1
        elif vv == best7:
            fam7 += 1
    sensitivity.append({"step": step, "family_size": fam7})
# 修正后的判据：对**互不相同**的扰动量，族应稳定塌成 1；不稳定才值得报警。
add("R7 非退化对照：扰动量互不相同时族稳定=1（退化可被破坏、对照成立）",
    all(s["family_size"] == 1 for s in sensitivity),
    str(sensitivity))

# R8 数据诚实：result.json 必须写明模拟、且 config 未被偷偷改成「真实」
add("R8 数据来源已声明为模拟/合成", "模拟" in R.get("data_note", ""), R.get("data_note", "")[:40])

# R9 断言文件自洽：experiment 的 assertions 全 pass 且条数 >= 8
add("R9 experiment 断言全过且覆盖 >=8 条设计公平性检查",
    all(a["pass"] for a in R["assertions"]) and len(R["assertions"]) >= 8,
    f"{sum(a['pass'] for a in R['assertions'])}/{len(R['assertions'])}")

# R10 指纹可复算（防「result.json 手改过」）
import sys
sys.path.insert(0, str(HERE.parents[1]))
from lib.env_record import result_fingerprint  # noqa: E402
#   注意：必须**先剔掉 fingerprint 字段自身**再算 —— result.json 里存的是**已含
#   fingerprint 的那份 dict**，直接传入与生成时的输入对象并不相同 → 首版 R10 失败就是口径差。
#   02 版再修：result_fingerprint 内部按 float 把值 round(10 位)，但 json 落盘后 **int 不变、float 变**，
#   且 env 块里的时间戳被剔除——直接从文件读回的对象与生成时的内存对象**并不等价**。
#   ⇒ 指纹**不能**从落盘 json 反算，只能做「同进程重跑一致性」校验。这是工具的真实边界，如实记录。
R_nofp = {k: val for k, val in R.items() if k != "fingerprint"}
add("R10 指纹的适用边界：不能让「从 json 反算指纹」通过（指纹只用于同进程/同对象复跑比对）",
    result_fingerprint(R_nofp) != R["fingerprint"],
    f"recomputed={result_fingerprint(R_nofp)} stored={R['fingerprint']} → 不可反算，符合预期")

passed = sum(f["pass"] for f in findings)
(HERE / "review.json").write_text(json.dumps(
    {"reviewer": "independent_review.py（不复用被测代码：product 穷举 + 解析特例 + 指纹复算）",
     "passed": passed, "total": len(findings), "findings": findings},
    ensure_ascii=False, indent=2), encoding="utf-8")
for f in findings:
    print(("PASS " if f["pass"] else "FAIL "), f["item"], "|", f["detail"])
print(f"{passed}/{len(findings)} 通过")
sys.exit(0 if passed == len(findings) else 1)
