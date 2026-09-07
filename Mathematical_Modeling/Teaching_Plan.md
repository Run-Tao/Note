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

### 2026-09-07（Day 7 — 模型课第 4 篇：元启发式 GA/SA 求解 TSP 城市巡回）
- 沿模型主线续 Day5 NLP：收编先前滞留的 Day6 GA/SA 工作（此前代码已闭环但**配套教案待补**），正式成稿为模型课第 4 篇——**离散启发性元启发式**。用“25 城巡回送货/巡检走哪条顺序最短”真实决策切入 **TSP(旅行商)**，讲清为何精确求解(LP/ILP/穷举/分支定界)几十城后组合爆炸：$25!\approx1.55\times10^{25}$。
- 教案：`05 - Algorithms/Day6 - 启发式元启发式GA与SA求解TSP.md`（新建 05 目录；沿用课程内 Day6 连续编号）。
- 代码：`08 - Python/daily/2026-09-07/heuristic_ga_sa_tsp.py`（手写 GA：编码→适应度→锦标赛选择→顺序交叉OX→变异→精英保留；手写 SA：2-opt 邻域 O(1) 增量 + Metropolis 接受 + 指数降温），仓库 `.venv` 实际跑通：NN=473 → **GA≈404(−14.7%)、SA≈414(−12.5%)**，合法性+有效性断言全过；图已按脚本自身目录输出到 daily/2026-09-07/。
- 图：`routes_ga_sa.png`(NN/GA/SA 三路线并排)、`convergence_ga_sa.png`(收敛曲线+NN 基准虚线)，已注册系统 Noto Sans CJK 字体。
- 要点：**“允许有纪律地变坏”才能跳出局部贪心陷阱**——GA 靠群体+顺序交叉重组路线块、SA 靠 Metropolis 概率接受更差再随温度指数降温收网；元启发式交付“高品质近似+可复现”，不保证全局最优（规模小仍优先精确求解）。
- 整理：将原先误置于仓库根的 2 张 PNG 与代码一并归位到 daily/2026-09-07/（原 2026-09-06/ 只留 Day5 NLP 全套）。
- 推送：已 git commit + push 到 Run-Tao/Note。
- ⚠️ 下一步：同属启发式的禁忌搜索/蚁群/PSO，或评价类 AHP/TOPSIS(Day2 工具链已就绪)、聚类/KNN(ML) 等模型+算法选题；元启发式可与多目标/调度结合继续加深。

### 2026-09-06（Day 5 — 模型课第 3 篇：非线性规划 NLP / 二次规划 / 均值-方差投资组合）
- 沿模型主线续 Day4 ILP，收尾**数学规划三件套**：升级到**非线性规划(NLP/二次规划QP)**——当目标里出现**风险(方差 $w^\top\Sigma w$ 二次型)**这类非线性时，LP/ILP 都表达不了。用"三条产品线资金怎么分"的真实决策切入（Markowitz 均值-方差）。
- 教案：`04 - Optimization/Day5 - 非线性规划NLP_均值方差投资组合.md`，及 `08 - Python/daily/2026-09-06/README.md`（frontmatter + 10 问成文）。
- 代码：`08 - Python/daily/2026-09-06/nlp_portfolio_quadprog.py`，仓库 `.venv` 实际跑通：LP式全押最高回报线1→波动≈20%;NLP 最优 = **线1 66.7% + 线3(避险)33.3%**(线2=0,被支配),波动 20%→14.4% 而回报仍保 10%;曲线2=0的决策洞察来自协方差遍历;纯最小方差解析解 $w_{gmv}=[ -9,10,0]$ 可卖空,NLP 是其 $w\ge0$ 现实版。SLSQP 与 trust-constr 交叉一致(<5e-3)+约束断言全过。
- 图：`nlp_efficient_frontier.png`(单资产 vs 优化组合★ / 有效前沿)，已注册 Noto Sans CJK SC 字体。
- 要点：**风险=二次型不可线性相加**是 NLP 入场券；分散化(40%→14.4%)优于孤注一掷；有效前沿标定"收益是用风险换的"价签。
- 推送：已 git commit + push 到 Run-Tao/Note。
- ⚠️ 下一步：连续 NLP 之后接 **离散启发性元启发式**(遗传/模拟退火/禁忌，Day4 ILP 已预告) 或聚类/KNN、评价类 AHP/TOPSIS。

### 2026-09-05（Day 4 — 模型课第 2 篇：整数规划 ILP / 分支定界 / 0-1 项目选择）
- 沿模型主线续 Day3 LP：升级到**整数规划(ILP)**——当决策"不可切半"(整投/整数件)时的建模与求解；揭开 **分支定界(Branch & Bound)** 算法。
- 教案：`04 - Optimization/Day4 - 整数规划ILP分支定界与0-1项目选择.md`（新建 04 目录），及 `08 - Python/daily/2026-09-05/README.md`（frontmatter + 10 问成文）。
- 代码：`08 - Python/daily/2026-09-05/ilp_project_selection.py`，仓库 `.venv` 实际跑通：预算100万/6项目，LP松弛=308.5 → **四舍五入选{3,4}超预算不可行(反面教材)**；ILP整数最优=275，选{3,6}=100万花满预算；穷举64种 & 手写分支定界三方一致(21节点/剪3支)，全部断言通过。
- 要点：LP 解直接 round 是陷阱；**整数间隙 = LP−ILP = 33.5 万** 即"项目不可分割"的代价；剪枝比穷举快。
- 推送：已 git commit + push 到 Run-Tao/Note。
- ⚠️ 下一步：非线性规划 NLP / 启发式(遗传、模拟退火) / 聚类 等模型+算法选题陆续推进。

### 2026-09-04（Day 3 — 切换模型课第 1 篇：线性规划 LP 标准建模闭环）
- 选题调整：应学习者要求，从"工具链地基"(pandas/可视化) **直接切换到模型/算法类课程**，并改为**每日自动定时**推送。今天开优化模型口子(Level 6 第一课)：线性规划。
- 教案：`Mathematical_Modeling/08 - Python/daily/2026-09-04/README.md`（LP 三要素 + scipy 标准形式 + 10 问成文）。
- 代码：`08 - Python/daily/2026-09-04/lp_production.py`，仓库 `.venv` 实际跑通：最优解 (A,B)=(36,28)、利润 2200 元、机器1 影子价 ≈10 元/h，断言全过；
- 图：`lp_feasible_region.png`（可行域 + 最优顶点），已注册系统 Noto Sans CJK 字体解决中文乱码；
- 环境注记：PuLP 本机未装，本轮用 scipy.optimize.linprog(HiGHS) 求解，无需额外依赖；
- 推送：已 commit + push 到 Run-Tao/Note。
- ⚠️ 下一步：配置 cron 每日自动执行备课动作（待接线自动化）。

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
