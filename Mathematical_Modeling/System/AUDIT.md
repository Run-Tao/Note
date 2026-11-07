# 系统审计报告（Phase 1）

> 审计时间：2026-11-07 21:30 (Asia/Shanghai)
> 审计对象：已部署的「数学建模每日研究系统」
> 审计方式：静态读代码/配置 + `openclaw cron list/get` 实读 + 仓库 `git log` + 实跑环境探测。**未修改任何现有文件。**

## 1. 系统真实位置与形态

| 项 | 事实 |
| --- | --- |
| 主体仓库 | `/home/ubuntu/Note`（Obsidian Vault），remote `git@ssh.github.com:Run-Tao/Note.git`，branch `main` |
| 建模知识库 | `Note/Mathematical_Modeling/`（298 文件，27 MB） |
| Agent 产出区 | `Note/Learning_Output/`（Daily_Report 72 篇 / Weekly_Report 8 篇 / Knowledge_Gaps 9 篇） |
| 运行形态 | **无独立 Python 服务、无 crontab、无 systemd unit**。全部由 OpenClaw cron → 隔离 Agent 会话执行「提示词任务」 |
| Python 环境 | 仓库 `.venv`（Python 3.12.3）：numpy 2.5.2 / scipy 1.18.1 / pandas 3.0.5 / scikit-learn 1.9.0 / matplotlib 3.11.1 / networkx 3.7 / ortools 9.15 / highspy 1.15 / numba 0.68 / statsmodels **缺失** |
| 机器资源 | 2 vCPU / 1.9 GB RAM（可用 0.8 GB）/ 磁盘 50G（剩 27G） |
| 模型 | 全部任务共用 `deepseek/deepseek-v4-flash`（OpenClaw agent `main`） |

**关键判断：现有系统没有任何结构化代码，所谓"系统"= 5 条 cron 提示词 + 一个 Git 仓库。** 因此升级必须自建可执行层，而不是改现有代码。

## 2. 现有任务清单（实测 `openclaw cron list/get`）

| 任务 | 周期 | 状态 | 实际行为 |
| --- | --- | --- | --- |
| `数学建模每日备课` (baf03772…) | 每天 23:00 | 🔴 **error ×10**（`TranscriptNotContinuableError`） | 读 `Teaching_Plan.md` → 沿模型/算法主线写下一讲教案（十件套）→ 代码落 `08 - Python/daily/<日期>/` → `.venv` 实跑 → 追加进度日志 → git push |
| `Note-Daily-Report` (5a6e200e…) | 每天 23:00 | source 源 OK | 复盘当日备课：独立复跑脚本、核对数字、写 `Daily_Report/YYYY-MM-DD.md` |
| `Note-Weekly-Report` (55c87230…) | 周日 23:00 | 🔴 error ×3 | 周进度/知识体系/漏洞分析 → `Weekly_Report/YYYY-Www.md` |
| `建模每周复习` (5ad560a8…) | 周一 10:00 | 🔴 error ×2 | 复习上周讲次、更新 `06 - Model Atlas/模型备赛图谱.md` |
| `Note-Morning-Digest` (58621030…) | 每天 07:00 | 🔴 error ×6 | 早晨摘要 |
| `练字提醒` | 每天 19:00 | OK | 无关 |

**竞态缺陷（历史遗留，第 2x 期）**：每日备课与日报同为 23:00，日报窗口会吞掉同一晚的备课 commit。

## 3. 审计八问

1. **每天具体做什么？** 每晚 23:00 生成一篇「Day N」教案（模型/算法主线，已到 Day69 / 61 讲），配可运行脚本 + 图 + `result.json` + README，`.venv` 实跑断言后 git push；同一时刻再写一篇对你学习情况的复盘日报。
2. **输入/输出？** 输入：`Teaching_Plan.md`（规程+进度日志）、`Learning_Roadmap.md`、MOC、上期日报。输出：`04/05/03` 目录教案 md、`08 - Python/daily/<date>/`（py/png/json/README）、`Learning_Output/Daily_Report/*.md`、git commit。
3. **模型如何研究与保存？** 单篇教案：决策场景 → 数学形式 → 假设 → 求解 → 实跑数字 → 教学点。Markdown + Obsidian LaTeX。
4. **如何进知识库？** 直接落目录 + 手工维护 `00 - MOC` 与 `06 - Model Atlas`（两者已出现口径不一致：MOC 61 讲 / Atlas 60 讲）。
5. **是否有代码运行与结果验证？** **有，且这是现有系统最强的一环**：`.venv` 实跑、断言、PNG md5 复跑一致性、日报独立复跑核对数字（近 72 篇日报里已多次证伪教案 headline）。
6. **是否重复研究？** 选题前会全库 grep 查零命中，重复度低；但**没有机器可读的模型索引**，靠 Agent 临场判断。
7. **最值得保留的？** ① 每日备课的"实跑+断言+数字口径"纪律；② 日报的独立复跑复核文化；③ `06 - Model Atlas` 选型图谱；④ `.venv` 工具链。
8. **新增功能需改哪些模块？** 无需改现有代码（无代码）。需新增：结构化注册表（YAML）、实验目录、cron 新任务、轻量 CLI。**现有 cron 提示词不改（除错峰/修复工作区归属）。**

## 4. 关键缺口（对照任务书）

| 任务书要求 | 现状 | 判定 |
| --- | --- | --- |
| 模型横向比较 + 真实实验 | ❌ 无（教案只在单一场景内打分） | 缺失 |
| 综合建模能力测试 | ❌ 无（无测试集、无隔离测试题） | 缺失 |
| 独立评分机制 | ❌ 无（日报是复盘，非评分） | 缺失 |
| 知识短板定位闭环 | 🟡 `Knowledge_Gaps` 停在 2026-10-11，无闭环追踪 | 半缺失 |
| 模型成熟度分级 | ❌ 无（只有"讲过/没讲过"） | 缺失 |
| 算法库（可测试实现） | 🟡 散落在 `daily/<date>/*.py`，无统一接口/无 pytest | 半缺失 |
| 案例库（训练/测试隔离） | ❌ `09 Case Studies` 为空 | 缺失 |
| 任务可靠性（ID/锁/重试/断点） | 🟡 cron 自带 ID，但无锁、无断点、4 条任务长期 error | 半缺失 |

## 5. 风险清单

| 风险 | 证据 | 等级 |
| --- | --- | --- |
| 未提交改动被覆盖 | `daily/2026-11-06/README.md` 在工作区未提交 | 中 |
| 任务长期静默失败 | 每日备课 error×10、早间摘要 error×6 | 高 |
| 内存紧张 | 1.9 GB 总量，2 vCPU；跑多实验并发会 OOM | 中 |
| statsmodels 缺失 | 做 ARIMA 对照需加依赖（纯 Python 可用 numpy 手写 ARIMA 或装 statsmodels） | 低 |
| 每日备课 23:00 与日报竞态 | 每日两次同分钟启动 | 中 |
| 知识库口径不一致 | MOC 61 / Atlas 60 | 低 |

## 6. 最小修改方案

- **不动**：现有 6 条 cron、`Mathematical_Modeling/` 现有文件、`.venv` 已装包。
- **新增一个独立目录** `/home/ubuntu/Note/Mathematical_Modeling/System/`（与现有知识库同仓，便于 Obsidian 查看与 git 回滚），内含：
  - `registry/models.yaml`（模型成熟度真值源）
  - `registry/gaps.yaml`（短板台账）
  - `lib/`（可复用实测代码：IO/环境记录/评分器）
  - `experiments/YYYY-Www_<slug>/`（每周比较：config + 代码 + result.json + report.md）
  - `benchmarks/YYYY-MM_<slug>/`（每月测试：题目、solver 产出、独立评审、评分 json）
  - `runs/`（任务运行台账：run id / 状态机 / 锁 / 日志）
  - `cli.py`（手动触发 + 历史查询 + 失败恢复）
- **新增 cron 3 条**（周日比较、每月测试、系统健康检查），全部 `delivery.mode=announce` 且与现有任务错峰。
- 所有改动走 git commit，可 `git revert` 回滚。

## 7. 状态机约定

`planned → running → completed → verified`；异常分支 `failed` / `skipped`。
**`completed` 只代表代码跑完，`verified` 必须有一次独立复跑（不同进程、同结果哈希或断言通过）才可写入。**
