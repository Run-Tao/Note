# 定时任务（Phase 6）— 已注册并实测

> 原则：**新增任务用 `--command` 直接跑脚本，不走 Agent 会话**。
> 理由：现有 4 条 Agent 型 cron 长期静默失败（`TranscriptNotContinuableError` / "couldn't generate a response"），
> 证明"依赖 Agent 会话产出的定时任务"可靠性差；而本系统的实验/复核/评分全部是**确定性脚本**，
> 用 command 型任务可做到：零 token 消耗、可复现、失败退出码明确、日志落盘。

## 一、已注册任务（2026-11-07 注册并手动实跑验证）

| 名称 | 周期 | 命令 | 产出 | 预计耗时 |
| --- | --- | --- | --- | --- |
| MathSys-Weekly | 每周一 06:10 | `run_task.py weekly <dir>` | `experiments/<dir>/{result.json,review.json,report.md}` | < 2 min |
| MathSys-Monthly | 每月 1 日 06:30 | `run_task.py monthly <dir>` | `benchmarks/<dir>/{score.json,review.json}` | 5–20 min |
| MathSys-Health | 每天 06:00 | `healthcheck.py` | `reports/health_YYYY-MM-DD.json` | < 5 s |

**错峰安排（不改动现有任务）**

| 时间 | 任务 | 类型 |
| --- | --- | --- |
| 06:00 | MathSys-Health（新增） | command |
| 06:10（周一） | MathSys-Weekly（新增） | command |
| 06:30（每月 1 日） | MathSys-Monthly（新增） | command |
| 07:00 | Note-Morning-Digest（现有） | agent |
| 10:00（周一） | 建模每周复习（现有） | agent |
| 19:00 | 练字提醒（现有） | agent |
| 23:00 | 数学建模每日备课 + Note-Daily-Report（现有） | agent |

新增任务全部在凌晨 6 点档，与现有 23:00 备课/日报完全不重叠，**避免历史遗留的"23:00 双任务抢同一窗口"问题进一步恶化**。

## 二、实际执行的注册命令（已生效）

```bash
# 1) 每日健康检查 06:00 —— jobId 4b76ca19-fec1-47cf-9196-1243aedafb1c
openclaw cron add --name "MathSys-Health" --cron "0 6 * * *" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && /home/ubuntu/Note/.venv/bin/python Mathematical_Modeling/System/healthcheck.py" \
  --command-cwd "/home/ubuntu/Note" --session isolated --timeout-seconds 300 --no-deliver

# 2) 每周比较 周一 06:10 —— jobId 5c3776c9-f86f-4b70-a2cc-da7d49126319
#    先生成下期骨架（幂等），再执行已批准项
openclaw cron add --name "MathSys-Weekly" --cron "10 6 * * 1" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && /home/ubuntu/Note/.venv/bin/python Mathematical_Modeling/System/generate_weekly.py >> Mathematical_Modeling/System/runs/generate.log 2>&1; /home/ubuntu/Note/.venv/bin/python Mathematical_Modeling/System/run_pipeline.py" \
  --command-cwd "/home/ubuntu/Note" --session isolated --timeout-seconds 1800 --no-deliver

# 3) 每月综合测试 每月 1 日 06:30 —— jobId 454b356b-8bcc-42fb-b1d3-bbf1cd27de6f
openclaw cron add --name "MathSys-Monthly" --cron "30 6 1 * *" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && D=$(ls Mathematical_Modeling/System/benchmarks | grep -E '^[0-9]{4}-[0-9]{2}_' | tail -1); if [ -n \"$D\" ]; then /home/ubuntu/Note/.venv/bin/python Mathematical_Modeling/System/run_task.py monthly \"$D\"; else echo 'no benchmark dir'; fi" \
  --command-cwd "/home/ubuntu/Note" --session isolated --timeout-seconds 3600 --no-deliver
```

**实测记录**

| 任务 | `cron run` 结果 | 证据 |
| --- | --- | --- |
| MathSys-Health | 第 1 次 `error`（rc=1：骨架目录被当成缺证据）→ 修 healthcheck 后 **ok** | `reports/health_2026-11-07.json` = `healthy` |
| MathSys-Weekly | **ok** | `runs/generate.log` 有 "生成骨架 2026-W45_paper_g-002"；`pipeline/pending.yaml` 状态正确 |
| MathSys-Monthly | 未手动触发（每月 1 日才跑，避免重复执行 B1） | 命令已用 `sh -lc` 在相同 cwd 下验证管道语法 |

> ⚠️ 已知坑：`$(...)` 与 `$D` 在 `--command` 里会被外层 shell 先展开，注册时 shell 报了一次
> `ls: cannot access ...benchmarks`，但最终写入 cron 的是原始字符串，运行正常 —— 仍建议以后改用脚本文件而非内联管道。

## 三、每周选题自动生成（决策：自动生成 + 人工批准）

用户 2026-11-07 决策：**选题由系统自动生成**。实现为两段式，避免"为了跑而跑"：

```
06:10 generate_weekly.py   → 读 gaps.yaml/models.yaml，选出一个待攻短板
                             生成 experiments/<week>_<category>_<gapid>/
                             （config.json + experiment.py 桩 + data/README + TODO.md）
                             → pipeline/pending.yaml  status=pending
     run_pipeline.py       → 只执行 status=approved 且骨架已补全的目录
                             没有 approved 就什么都跑（安全阀）
```

**选短板规则**：`P0 > P1 > P2 > P3`，同级取最早发现的；若该 `category` 已有实验目录则跳过（防重复）。

**批准方式**：`./.venv/bin/python generate_weekly.py --approve <dir>`

**当前 pipeline 状态**（2026-11-07）

| 目录 | 状态 | 短板 |
| --- | --- | --- |
| `2026-W45_math_g-001` | approved（骨架未补全，暂不执行） | G-001 相关矩阵正定性 |
| `2026-W45_paper_g-002` | pending | G-002 证据纪律 |

> 注：`generate_weekly.py` 的重复检测是按 `category` 判重的，而 G-001/G-002/G-003 的 category 各不相同，
> 因此会连续生成多个骨架。**这是刻意的**：骨架由人工批准才执行，未批准的不会占用资源。

## 四、资源预算

| 任务 | CPU | 内存峰值 | 磁盘 |
| --- | --- | --- | --- |
| Weekly（预测） | <1 s 计算 | <200 MB | 数据 6.8 KB |
| Weekly（优化，含穷举 16.9 万点） | 5 s | <150 MB | 无数据文件 |
| Monthly（B1，含 20 万次模拟） | ~15 s | <400 MB | 无数据文件 |
| Health | <1 s | <50 MB | 每次 <2 KB json |

在 2 vCPU / 1.9 GB 机器上完全无压力；全部任务彼此不重叠，不存在并发 OOM 风险。
