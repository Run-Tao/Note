# 定时任务设计（Phase 6）

> 原则：**新增任务用 `--command` 直接跑脚本，不走 Agent 会话**。
> 理由：现有 4 条 Agent 型 cron 长期静默失败（`TranscriptNotContinuableError` / "couldn't generate a response"），
> 证明"依赖 Agent 会话产出的定时任务"可靠性差；而本系统的实验/复核/评分全部是**确定性脚本**，
> 用 command 型任务可做到：零 token 消耗、可复现、失败退出码明确、日志落盘。

## 一、待注册任务（尚未注册，需人工确认）

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

## 二、注册命令（需人工确认后执行）

```bash
cd /home/ubuntu

# 1) 每日健康检查
openclaw cron add --name "MathSys-Health" \
  --cron "0 6 * * *" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && ./.venv/bin/python Mathematical_Modeling/System/healthcheck.py" \
  --command-cwd "/home/ubuntu/Note" \
  --session isolated --timeout-seconds 300 --no-deliver

# 2) 每周模型比较（每周一 06:10）
openclaw cron add --name "MathSys-Weekly" \
  --cron "10 6 * * 1" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && ./.venv/bin/python Mathematical_Modeling/System/run_task.py weekly \$(date +\%Y-W\%V)_prediction_lr_arima" \
  --command-cwd "/home/ubuntu/Note" \
  --session isolated --timeout-seconds 1800 --no-deliver

# 3) 每月综合测试（每月 1 日 06:30）——目录需先人工选题创建
openclaw cron add --name "MathSys-Monthly" \
  --cron "30 6 1 * *" --tz "Asia/Shanghai" \
  --command "cd /home/ubuntu/Note && ls Mathematical_Modeling/System/benchmarks | grep \$(date +\%Y-\%m) | tail -1 | xargs -I{} ./.venv/bin/python Mathematical_Modeling/System/run_task.py monthly {}" \
  --command-cwd "/home/ubuntu/Note" \
  --session isolated --timeout-seconds 3600 --no-deliver
```

> ⚠️ 注册前必须验证 `--command` 型任务在 cron 环境中的 PATH/Python 是否可用（已用绝对路径 `.venv/bin/python` 规避）。
> 建议先用 `openclaw cron add ... --disabled` 创建，再 `openclaw cron run <id>` 做一次手动验证，确认输出后 `enable`。

## 三、为什么周任务当前写死"重复跑同一个实验目录"

`run_task.py` 有幂等保护：同 `task_id` 已 `verified` 会**直接跳过**（退出 0，不重跑、不重复写）。
因此"每周跑同一目录"实际只在**第一次**真正执行，之后每周返回"已跳过"。
真正的每周更新需要**每周新建实验目录**（选题由人工或备课 Agent 决定），这是刻意的设计：

- 避免 Agent 自动生成"为了跑而跑"的实验；
- 每周比较的选题应当来自知识库的短板与上期"待研究问题"，而不是机械轮换。

**下一阶段改进方向**：让周任务读取 `reports/` 中最新一期的"待研究问题"，自动创建下一个实验目录骨架（config + 数据 + 待填代码），由人工确认后再执行。

## 四、资源预算

| 任务 | CPU | 内存峰值 | 磁盘 |
| --- | --- | --- | --- |
| Weekly（预测） | <1 s 计算 | <200 MB | 数据 6.8 KB |
| Weekly（优化，含穷举 16.9 万点） | 5 s | <150 MB | 无数据文件 |
| Monthly（B1，含 20 万次模拟） | ~15 s | <400 MB | 无数据文件 |
| Health | <1 s | <50 MB | 每次 <2 KB json |

在 2 vCPU / 1.9 GB 机器上完全无压力；全部任务彼此不重叠，不存在并发 OOM 风险。
