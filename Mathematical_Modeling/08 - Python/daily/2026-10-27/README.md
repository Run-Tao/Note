---
title: Day59 - 设施选址与车次联合优化（代码）
type: code-notes
subject: mathematical-modeling
level: L6 优化模型 / L8 算法
tags:
  - mathematics
  - mathematical-modeling
  - optimization
  - integer-programming
  - facility-location
  - solver-verification
created: 2026-10-27
---

# Day59 · 县域多仓—多站点单趟配送：分配与车次联合优化

配套教案：[[Day59 - 设施选址与车次联合优化就近分配为何排崩]]

## 一句话

在仓容量卡死的条件下，用 0-1 整数规划把「站点分给谁 + 每仓发几趟 + 谁认栽付罚金」一起解，
比"就近分配"贪心省 45.1%；并用穷举独立验证了 ILP 结果确实最优。

## 文件

| 文件 | 说明 | 状态 |
|---|---|---|
| `day59_facility_vehicle_ilp.py` | 主脚本：贪心基线 → ILP → LP 松弛 → 影子价格扫描 → 3 面板图 + JSON | ✅ 跑通 |
| `day59b_solver_audit.py` | 求解器可信度自检：HiGHS presolve 假最优的最小复现 | ✅ 跑通 |
| `day59c_bruteforce_check.py` | 剪枝穷举验证 ILP 最优性 | ✅ 跑通 |
| `day59_facility_vehicle.png` | 运力利用率 / 影子价格排序 / 贪心 vs ILP 成本 | ✅ 已生成 |
| `day59_result.json` | 全部数值结果 | ✅ 已生成 |

## 运行

```bash
cd /home/ubuntu/Note
./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59_facility_vehicle_ilp.py"
./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59b_solver_audit.py"
./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-27/day59c_bruteforce_check.py"
```

## 实测结果（.venv 实跑，非估算）

| 方案 | 成本 | 送不到的站 | 说明 |
|---|---|---|---|
| 就近分配贪心 | 48862 元 | 2 个 | 运费 24862 + 罚金 24000 |
| **整数规划** | **26846 元** | **0 个** | 53 车，5 仓顶格跑满 |
| LP 松弛（下界） | 19034 元 | — | 整性间隙 29.1% |
| 穷举（真最优） | 26846 元 | 0 个 | 265,413 节点，1.4 秒 |

**影子价格（容量 +1 重解扫描）**：

| 仓 | 元/多 1 车 |
|---|---|
| 河湾 | 2542 |
| 柳湾 | 992 |
| 石桥 | 806 |
| 城关 | 558 |
| 坳上 | 0 |
| 云岭 | 0 |

## 三个必须记住的坑

1. **不在服务水平不同的方案之间比成本**。贪心运费 24862 元比 ILP 的 26846 元低，
   因为它少送了两个站。口径不一致的比较毫无意义。
2. **`k[i]` 的上界要写 `cap[i]`，不是 `cap.sum()`**。写错会让容量约束失效，
   ILP 给出"看起来专业"的错答案。
3. **求解器说 Optimal 不等于最优**。`day59b_solver_audit.py` 里 HiGHS 1.12.0
   返回 status=Optimal、gap=0%、fun=0，而真值是 100。
   复核要靠正交手段（换符号重解 / 换独立算法），不能只多调用一次。

## 依赖

scipy 1.18.1（HiGHS 1.12.0）、numpy 2.5.2、matplotlib 3.11.1；
图中中文用 Noto Sans CJK SC。
