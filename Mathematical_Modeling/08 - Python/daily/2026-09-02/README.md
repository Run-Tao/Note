# Day 2026-09-02 · 首个 NumPy 闭环：线性方程组求解

> 位置：`08 - Python/daily/2026-09-02/`
> 锚点：Linear Algebra Level 1–2（高斯消元 / 秩 / 线性方程组）

## 今天解决了什么

仓库"最大短板=代码验证为零"，从今天起开始补齐。首个闭环：把已学过（纸上）的
"3×3 线性方程组求解"，用 NumPy 真正**跑通并验证**。

## 代码：`linear_system_solve.py`

运行方式（务必用仓库虚拟环境，勿污染系统 python）：

```bash
cd /home/ubuntu/Note && ./.venv/bin/python \
  "Mathematical_Modeling/08 - Python/daily/2026-09-02/linear_system_solve.py"
```

产出截图见 `output.png`（若生成）。

## 关键知识点（对应 README 的"10 问"）— 精简版

**它解决什么问题**：给定系数矩阵 $A$ 与右端 $b$，求 $Ax=b$。

**何时用**：线性回归、差分/连续方程离散化、图网络流量、最短路等大量建模问题最后都
落到"解一个线性方程组"。

**数学模型**：$Ax=b$，其中 $x$ 是未知向量。

**如何求解（Python）**：`np.linalg.solve(A, b)`；内部等价于 LU 分解（消元+回代），
对应高中学的高斯消元。

**验证**：算残差 $\|Ax-b\|$ 应接近 0。本脚本输出 `max|残差| = 0.0`。

## 今天踩的坑（值得记入 Exercises）

1. **先查秩，再决定用什么解**。对不满秩（奇异/欠定）系统：
   - 新版 numpy 的 `solve` 会**直接抛 `LinAlgError: Singular matrix`**（不是静默给错值）；
   - 若想知道最接近的解，用 `np.linalg.lstsq`（最小二乘意义），但要说清"并非唯一解"。
2. 自编方程 `A=[[1,2,-1],[3,1,1],[2,-1,2]]` 第 3 行其实是第 1、2 行的线性组合
   → rank=2 欠定，`np.linalg.solve` 结果无意义。**以后造题先 `matrix_rank` 验算。**

## 下一步（待办）

- [ ] Day2：NumPy 数组/向量化快速过关
- [ ] Day3：pandas + matplotlib 数据可视化套路（配合 Level2 数据分析）
