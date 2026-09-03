"""
subject: mathematical-modeling
level: Level 2 (Data Analysis) prep / Level 8 (Numerical) spine
anchor: Roadmap Level 2, `NumPy 数组与向量化` —— 建模 Python 工具链第 2 环
goal: 建立"数组心智"：标量 for 循环 → 向量化广播，彻底摆脱慢速 Python 循环
run:  cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-09-03/numpy_vectorization.py"
"""
import numpy as np
import time

# ===========================================================================
# 背景：上一环(Day1)已经用 np.linalg.solve 解过一次方程组。但那时我们只是
#       "调用一个函数"，还没有"用数组思考"。建模 90% 的算式用不上线代库，
#       而是这种重复运算：对每个样本/每个时刻/每个网格点 算同一个式子。
#       若用 Python for 循环去算，数据一上万就慢到不能忍 —— 竞赛里就是超时。
#       今天这节课把它一次性练成"条件反射"。
# ===========================================================================

print("=" * 72)
print("Day2 · NumPy 数组与向量化：从“逐个算”到“整体算”")
print("=" * 72)

# ---------------------------------------------------------------------------
# 情景 1：为什么要向量化？用速度说话（先建立动机）
# ---------------------------------------------------------------------------
# 常见建模场景：给 10 万只股票/人群/样本算某个标量公式，
# 例如 净需求 q_i = 供给 s_i - 需求项
n = 200_000
s = np.random.rand(n)          # 供给
d = np.random.rand(n)          # 需求基础项
lb = np.linspace(0.9, 1.1, n)  # 扰动系数

# (a) 慢：显式 Python for 循环
t0 = time.perf_counter()
out_slow = np.empty(n)
for i in range(n):
    out_slow[i] = s[i] * lb[i] - d[i]
t_slow = time.perf_counter() - t0

# (b) 快：整段数组一起算（向量化）
t0 = time.perf_counter()
out_fast = s * lb - d
t_fast = time.perf_counter() - t0

print("\n[速度对比] n =", n, "条记录：")
print(f"  for 循环版本 : {t_slow*1000:8.2f} ms")
print(f"  向量化版本   : {t_fast*1000:8.2f} ms")
print(f"  加速约       : {t_slow/t_fast:6.0f} 倍")
assert np.allclose(out_slow, out_fast, atol=1e-12), "两版结果不一致!"
print("  结果一致性   : ✅ 两版输出逐元素相同")

# ---------------------------------------------------------------------------
# 向量化的三个基础构件：轴/切片、广播、通用函数 ufunc
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("① 轴 (axis) 与 切片：告别手写双重循环求行列统计")
print("-" * 72)
# 模拟：12 个月 × 5 个地区 的某指标面板数据
X = np.arange(12 * 5, dtype=float).reshape(12, 5) + 1
print("面板 X 形状 (月×地区):", X.shape)
# 不用 for：直接沿 axis 归约
row_mean = X.mean(axis=1)      # 每个月的地区平均
col_mean = X.mean(axis=0)      # 每个地区跨月平均
print("每月均值 X.mean(axis=1) =", np.round(row_mean, 1))
print("每地区均值 X.mean(axis=0) =", np.round(col_mean, 1))
# axis=0 是"跨样本的行方向"，axis=1 是"每样本内的列方向" —— 建模里反复用到

print("\n" + "-" * 72)
print("② 广播 (broadcasting)：不同形状自动对齐，消除手工复制")
print("-" * 72)
# 中心化：每个地区减去自己的均值（列方向缩放到长度1)
centered = X - X.mean(axis=0)          # (12,5) 减 (5,)  -> 广播
# 标准化到 Z 分数：再除以每地标准差
z = (X - X.mean(axis=0)) / X.std(axis=0)
print("中心化后示例（前3行）：\n", np.round(centered[:3], 2))
print("核对：中心化后每列均值应≈0 ->", np.round(centered.mean(axis=0), 12))
print("核对：标准化后每列标准差应≈1 ->", np.round(z.std(axis=0), 6))

print("\n" + "-" * 72)
print("③ 通用函数 ufunc + 条件掩码：连“判断/裁剪/开关”都整段做")
print("-" * 72)
prices = np.array([12.5, -3.0, 88.0, 0.0, 45.6])   # -3 可能是录入错误
# 逻辑条件 -> 布尔掩码，整段筛选，不用 if
mask = prices >= 0
print("非负掩码 prices>=0 :", mask)
print("提取有效价格       :", prices[mask])
# 裁剪异常值（把取值夹到 [0, 120]）
clipped = np.clip(prices, 0, 120)
print("clip 到 [0,120]    :", clipped)
# 用 where 做“若 a 否则 b”：把负数标记为“疑误”
flag = np.where(prices < 0, "疑误(负价)", "正常")
print("where 标记          :", flag)

# ---------------------------------------------------------------------------
# 把三件套用到一个"更像个模型"的小案例上：
# 线性拟合的向量一步到位的实现（这就是 sklearn 背后在干的事）
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("连接：向量化很自然地把 Day1 的线代 & 后续回归串起来")
print("=" * 72)
# 造数据 y ≈ 2 + 3x + 噪声
m = 400
rng = np.random.default_rng(7)
x = rng.uniform(0, 10, m)
y_true = 2.0 + 3.0 * x
y = y_true + rng.normal(0, 1.5, m)

# 最小二乘闭式解：一次性向量化写出  (XᵀX)⁻¹ Xᵀy
# 设计矩阵：先用 column_stack + ones_like 拼出解释变量矩阵（含截距列）
X_des = np.column_stack([np.ones_like(x), x])      # (m,2)
# 闭式解公式（对比 Day1 的 solve 视角：这里 solve 解的是正规方程）
XtX = X_des.T @ X_des
Xty = X_des.T @ y
beta = np.linalg.solve(XtX, Xty)                   # 等价于 np.linalg.lstsq 最小二乘解
print("最小二乘闭式解 β =", np.round(beta, 4), "（真值 ≈ [2, 3]）")

# 预测 + 残差（全向量化，无 for）
yhat = X_des @ beta
resid = y - yhat
sse = float(resid @ resid)                          # 残差平方和 = 向量点积
rmse = float(np.sqrt(sse / m))
print(f"RMSE = {rmse:.4f}  （噪声标准差设 1.5，RMSE 应量级相近 -> 拟合合理）")
print(f"系数β0≈{beta[0]:.2f} 即为截距、β1≈{beta[1]:.2f} 即为斜率，一次矩阵乘法全出")

# ---------------------------------------------------------------------------
# 小结 / 复述（对齐 —— README 的"如何验证/写论文"）
# ---------------------------------------------------------------------------
print("\n[本课一句话] 能“整段数组一次算完”就不要“逐元素写循环”；")
print("   axis 定方向、broadcasting 省复制、ufunc/掩码 代替 if ——")
print("   这三点是后续 pandas/sklearn/scipy 全部代码的速度与可读性底座。")
print("\n[验证] 速度对比 + 归约均值≈0 + 最小二乘系数逼近真值 (2,3)，闭环成立 ✅")
