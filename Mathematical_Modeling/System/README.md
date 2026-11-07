# Mathematical_Modeling/System — 数学建模能力进化系统

> 定位：在**不破坏**现有每日备课/日报管线的前提下，为数学建模研究增加
> **可验证的实验层、比较层、能力评估层、短板反馈层、成熟度台账层**。
> 创建：2026-11-07　审计：见 `AUDIT.md`

## 0. 它解决什么问题

原系统 = 5 条 cron 提示词 + 一个 Git 仓库（`~/Note`）。它能"讲模型"，但**无法证明自己会用模型**：
无模型比较实验、无能力测试、无独立评分、无成熟度分级、短板台账停在 2026-10-11。

本系统用**可执行代码 + 证据链**补这五条，而不是再写更多教案。

## 1. 目录职责

```
System/
├── AUDIT.md              # Phase 1 系统审计（先审后改）
├── README.md             # 本文件（规范 + 操作手册）
├── cli.py                # 任务台账 CLI（唯一ID/锁/状态机/历史查询）
├── lib/env_record.py     # 实验可复现记录（种子/版本/数据哈希/结果指纹）
├── registry/
│   ├── models.yaml       # 模型成熟度真值源（L1–L5，必须有证据）
│   └── gaps.yaml         # 知识短板台账（P0–P3，需复测才能关闭）
├── experiments/<YYYY-Www>_<slug>/   # 每周模型比较（config + 代码 + 数据 + result + report + review）
├── benchmarks/<YYYY-MM>_<slug>/     # 每月综合建模测试（题目 + solver + 评审 + 评分）
├── runs/ledger.jsonl     # 任务运行流水（append-only）
└── reports/              # 跨周期汇总（能力趋势、系统健康）
```

## 2. 任务状态机（强制）

```
planned → running → completed → verified
                ↘ failed
                ↘ skipped
```

- `completed` = 代码跑完，退出码 0
- **`verified` = 有独立复核**（另一个脚本/另一个上下文，走不同实现路径重算关键数字）
- **绝不允许**把"生成了报告"写成 `completed`，把"自己说自己对"写成 `verified`

## 3. 实验强制规范（任务书第十部分）

每个实验目录必须包含：

| 文件 | 必需 | 内容 |
| --- | --- | --- |
| `config.json` | ✅ | 种子、窗口、模型清单、数据来源 |
| `experiment.py` | ✅ | 可独立运行，**末尾内置断言**，退出码 0/1 |
| `result.json` | ✅ | 全部指标 + `checks` + `fingerprint` |
| `env.json` | ✅ | Python/依赖版本、种子、git commit、数据 sha256（缺失字段记 `null`，**不估算**） |
| `report.md` | ✅ | 结构化比较报告（含选择建议、局限、待研究问题） |
| `independent_review.py` | ✅ | **不 import 实验代码**，独立重算关键数字 |
| `review.json` | ✅ | 复核结论 `verified` / `rejected` |
| `data/raw.csv` | 视情况 | 数据落盘，保证可复现 |

**断言纪律（本系统最重要的一条，来自 W45 的真实教训）**：
断言只能检验**实验设计是否公平**（无数据泄漏、基准存在、失效可观测、权重和为 1），
**不能预设"谁该赢"**。首版 W45 断言写成"至少 3 个模型优于 Naive"，结果两套数据都失败 —— 那不是发现，是自己给自己出答案。

## 4. 模型成熟度（任务书第八部分）

| 等级 | 通过条件（缺一不可） |
| --- | --- |
| L1 已了解 | 有资料/教案；能解释基本思想 |
| L2 理论掌握 | 数学原理 + 公式推导 + 变量/参数/假设 + 适用条件 |
| L3 实现掌握 | 有实现代码 + 断言通过 + 复跑指纹一致 |
| L4 应用掌握 | 真实数据上解决过问题 + 与 ≥1 个替代模型同条件比较 + 局限性分析 |
| L5 综合应用 | 在综合测试中被**组合使用**，且通过独立评审（总分 ≥70 且无 P0 错误） |

**禁止自评升级**：每次升级必须在 `evidence` 字段写明对应的实验/测试目录与指纹。

## 5. 每月综合测试（benchmark）流程

1. 题目与完整参考解答存放在 `benchmarks/<id>/private/`，**不进入日常备课知识库**
2. Solver 在**独立上下文**中解题，只能读知识库，不能读 `private/`
3. 评审分 5 个角色（数学/代码/实验/论文/汇总），至少 3 个角色由独立复核脚本或独立上下文承担
4. 评分写入 `score.json`，短板写入 `registry/gaps.yaml`
5. 题目来源与是否可能泄漏，必须在 `benchmark.md` 头部标注

## 6. 操作手册

```bash
cd ~/Note/Mathematical_Modeling/System

# 系统总览（成熟度分布、短板数、最近任务）
../../.venv/bin/python cli.py status

# 查看成熟度 / 短板 / 历史
../../.venv/bin/python cli.py models
../../.venv/bin/python cli.py gaps
../../.venv/bin/python cli.py run list --last 20
../../.venv/bin/python cli.py run show <task_id>
../../.venv/bin/python cli.py history experiments
../../.venv/bin/python cli.py history benchmarks

# 手动跑一次每周比较
../../.venv/bin/python "experiments/2026-W45_prediction_lr_arima/experiment.py"
../../.venv/bin/python "experiments/2026-W45_prediction_lr_arima/independent_review.py"

# 失败任务恢复：换 id 重跑（幂等保护会拒绝同 id 重复）
../../.venv/bin/python cli.py run start weekly-2026-W60-retry weekly-2026-W60
```

## 7. 回滚

全部新增内容在 `Mathematical_Modeling/System/` 一个目录内，
`git revert <commit>` 或删除该目录即可完全回滚，**不触碰现有教案、日报、cron**。
新增 cron 3 条可用 `openclaw cron disable <id>` 单独关闭。
