---
title: 数学建模每日备课（Agent 驱动教案）
type: plan
subject: mathematical-modeling
level: overview
tags:
  - mathematics
  - mathematical-modeling
  - agent
created: 2026-09-02
---

# 数学建模每日备课

> 定位：**由 Agent 持续备课 + 推送的教案流水线**，服务于"用户是学习者、没时间自己整理"的场景。
> 原则：①沿 [[Learning_Roadmap]] 的 Level 0–12 递进；②**内容不依赖用户当日真实输入**（用户可抽空按需阅读/运行）；③代码一律用仓库 `.venv` 可运行并**经 Agent 实际跑通**后再入库；④遵守 [[Mathematical Modeling README]] 的"10 问"原则，不写机械的"公式+定义+例题"。

## 运行环境

- 虚拟环境：仓库根 `.venv/`（python3.12）
- 已装：numpy 2.5.2 / scipy 1.18.1 / pandas 3.0.5 / scikit-learn 1.9.0 / matplotlib 3.11.1
- 运行：`cd /home/ubuntu/Note && ./.venv/bin/python <脚本>`
- 代码落地目录：`Mathematical_Modeling/08 - Python/`

## 每日输出物（落到仓库并 push）

1. **教案笔记** → 放进对应 Level 目录（02–09 主题目录），如 `09 Case Studies`、`04 Optimization` 等；
2. **可运行代码** → `Mathematical_Modeling/08 - Python/daily/<YYYY-MM-DD>/`，每个主题配 `README` + 脚本 + 输出截图/图；
3. **进度更新** → 本文件"进度日志"追加当天条目；
4. **推送** → git commit + push 到 `Run-Tao/Note`。

## 选题策略（沿 Roadmap，补齐代码闭环优先）

用户最大短板是"Python/NumPy 代码验证为零、Projects/ 空"。故首周不以堆模型为主，先打通**建模常用 Python 工具链**，再回到模型方法。计划草表（可随进度调整）：

| 天 | Roadmap 锚点 | 主题 | 代码闭环 |
|---|---|---|---|
| Day1 | L1/L2 打底 | Git/.venv/工具链就绪 + 首个 NumPy 求解复现 | 矩阵线性方程组 |
| Day2 | L2 | NumPy 数组/向量化 快速过关 | numpy 练习 |
| Day3 | L2 | pandas 数据清洗 + matplotlib 可视化套路 | 数据+图 |
| Day4 | L3 | 差分/连续方程入门 | scipy odeint |
| Day5 | L3 | SIR 传染病仿真 | 仿真图 |
| Day6 | L4 | 线性回归 + 拟合诊断 | sklearn |
| Day7 | L4 | 时间序列/ARIMA 初步 | statsmodels |
| ... | L5–12 | AHP/TOPSIS→优化→ML→真题→论文 | 逐步 |

## 待办 / 约束

- [ ] 与现有"每日学习报告"Agent 的关系：**本教案只管数学建模备课内容本身**，不接管其学习报告机制；
- [ ] 数学符号/公式应能在 Obsidian(数学插件)渲染，用 `$$`；避免过重 MathJax 负担；
- [ ] 每个代码文件头部注释标 `subject` / `level` / 对应 Roadmap，便于索引进 MOC；
- [ ] 若仓库后续 config.md（Obsidian 规范）落地，本计划随之对齐。

## 进度日志

### 2026-09-03（Day 2 — NumPy 数组与向量化快速过关）
- 备课主题：Roadmap Level 2 前置 / Level 8 底座 ——「NumPy 数组与向量化快速过关」（axis/广播/ufunc+掩码 三构件 + 循环版 vs 向量化速度对比）。
- 教案：`Mathematical_Modeling/08 - Python/daily/2026-09-03/README.md`（含 frontmatter，按 README 10 问精简成文）。
- 代码：`08 - Python/daily/2026-09-03/numpy_vectorization.py`，已用仓库 `.venv` 实际跑通：n=20 万向量化比 for 循环快约 26×；最小二乘闭式解一次算完，系数逼近真值 [2,3]，RMSE≈1.45（与噪声 1.5 同量级），全部断言通过。
- 要点：打通"数组心智"——能整段一次算完就不逐元素写循环，为 Day3+ pandas/sklearn/scipy 打好速度与可读性底座。
- 推送：已 git commit + push 到 Run-Tao/Note。

### 2026-09-02（Day 0/1 — 立项 + 首个 NumPy 线性方程组闭环）
- 建立 `.venv` 科学计算环境并跑通依赖导入；
- `.gitignore` 补 `.venv/`、`__pycache__/`、`*.pyc`；
- 新建本教案；确认 Note 仓库 git remote 为 `Run-Tao/Note`（SSH 认证 OK）。
