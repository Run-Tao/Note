# Agent_Output — 长期学习规划 Agent 输出目录

本目录用于存放**学习规划 Agent** 的分析输出，供本地 Agent / 用户进一步处理。

## 结构

```
Agent_Output/
├── Daily_Report/                  # 每日报告（读取24h变化）
├── Weekly_Report/                 # 每周报告（7天深度分析）
├── Learning_Plans/                # 学习计划
├── Research_Recommendations/      # 资料推荐（含推荐理由）
└── Knowledge_Gaps/                # 知识缺口分析
```

## 输入（Agent 读取）

- `Inbox/`、`Notes/`、`Projects/`、`Research/`、`Learning_Status.md`
- `git log`（commit 历史）
- 各知识库 MOC / 路线 / 周记

## 输出原则

1. 不修改正式知识库，只写入本目录
2. 不删除内容
3. 不编造资料——推荐必须能说明"为什么推荐"
4. 输出可被本地 Agent 进一步处理（Markdown 规范格式）

## 状态

- 首个基线分析：2026-08-23（见 `Knowledge_Gaps/2026-08-23_baseline.md`）
- 学习状态快照与缺口清单见 `Learning_Status.md`（Agent 维护）
