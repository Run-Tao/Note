"""
subject: mathematical-modeling
level: Level 2 (Data Analysis) / Level 8 (Numerical) prep
anchor: Linear Algebra Level 1-2, `Matrix → linear systems`
goal: 首个 NumPy 代码闭环 —— 解线性方程组 + 用"秩"预判解 + 识别数值陷阱
run:  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-02/linear_system_solve.py"
"""
import numpy as np

# ---------------------------------------------------------------------------
# 背景：仓库每日报告连续多天把"代码验证为零"标为最大短板。
# 线性代数 Level1-2 已学过 高斯消元/秩/线性方程组，但从未用 NumPy 落地。
# 本文件 = 让"纸上知识"变成"跑得通、看得见"的第一步闭环。
# ---------------------------------------------------------------------------

# 1. 构造一个 3x3 方程组   A x = b   （练习自编数据，已验证满秩）
A = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 1.0, 1.0],
    [1.0, 1.0, 3.0],
])
b = np.array([6.0, 5.0, 5.0])

print("A =\n", A)
print("b =", b, "\n")

# 2. 用"秩"预判解的结构：这是解方程组前必做的一步
rank_A = np.linalg.matrix_rank(A)
rank_aug = np.linalg.matrix_rank(np.column_stack([A, b]))   # 增广矩阵 [A | b]
n = A.shape[1]
print(f"rank(A)     = {rank_A}")
print(f"rank([A|b]) = {rank_aug}   (n={n})")

if rank_A != rank_aug:
    print("判定：rank(A) != rank([A|b]) -> 无解")
elif rank_A == n:
    print("判定：满秩（唯一解）-> 可放心用 np.linalg.solve")
else:
    print("判定：rank(A)==rank([A|b]) < n -> 无穷多解，solve 给出的只是特解")

# 3. 满秩时数值求解
x = np.linalg.solve(A, b)
print("\nx =", x)

# 4. 回代验证残差（好习惯：永远验证一下）
res = A @ x - b
print("残差 A@x-b   =", res)
print("max|残差|    =", np.max(np.abs(res)), "\n")

# 5. 展示"高斯消元"到底在做什么：LU 分解 = 消元 + 回代
from scipy.linalg import lu
P, L, U = lu(A)
print("[scipy.lu] 消元结构视角：")
print("L(下三角, 消元乘数) =\n", np.round(L, 4))
print("U(上三角, 消元结果) =\n", np.round(U, 4), "\n")

# 6. ⚠️ 数值陷阱示范：对"欠定(rank<n)"系统，先查秩，别裸调 solve
#    备课过程 Agent 实测：numpy 新版对奇异矩阵会直接抛 LinAlgError，而不是静默返回错值；
#    旧版本/某些库则可能返回误导值。无论哪种，正确姿势都是“先算秩再决定”。
A2 = np.array([[1.,2.,-1.],[2.,4.,-2.],[3.,1.,1.]])   # 第2行 = 2×第1行 -> 秩不足
b2 = np.array([1.,2.,4.])
r2 = np.linalg.matrix_rank(A2)
print(f"⚠️ 欠定系统 A2 的 rank = {r2} (<3, 不满秩)")
try:
    x2 = np.linalg.solve(A2, b2)
    print("   solve 竟没报错，x2 =", x2, "← 满秩前提不成立，此值无意义")
except np.linalg.LinAlgError as e:
    print("   直接 solve 会抛错:", e)
# 正确姿势：承认“无唯一解”，用 lstsq 求最小二乘意义下的解
x2b, res_, rank_, s_ = np.linalg.lstsq(A2, b2, rcond=None)
print("   np.linalg.lstsq → x2b =", np.round(x2b, 4), "(最小二乘/最小范数意义，并非唯一解)")

print("\n✅ 闭环完成：满秩方程组 -> 秩预判 -> 数值解 -> 残差验证 -> 消元透视 -> 欠定陷阱识别")
