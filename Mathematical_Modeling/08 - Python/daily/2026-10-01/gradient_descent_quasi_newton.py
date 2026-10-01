# -*- coding: utf-8 -*-
"""
subject : mathematical-modeling
level   : L6 优化模型 / L8 算法（无约束非线性优化：梯度下降家族 + 拟牛顿法）
anchor  : Learning_Roadmap Level 6 (优化模型) · 算法主线
goal    : 用同一个「工厂配方成本最小化」决策，横向对比
          GD / Momentum / RMSProp / Adam / BFGS 五类优化算法的
          收敛速度、对学习率的敏感性、能否逃出病态/非凸地形，
          并回答"该选哪个算法、学习率怎么设"。

运行： cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-01/gradient_descent_quasi_newton.py"
"""
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体：注册系统 Noto Sans CJK，避免乱码 ----
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# ============================================================
# 一、决策背景：工厂「配方 × 温度」调优（非凸成本函数）
# ------------------------------------------------------------
# 一家化工厂要选 2 个连续工艺参数 x=(配方浓度 c, 反应温度 t)。
# 成本 = 原料成本 + 能耗 + 不良率惩罚。因为存在"两个工艺窗口"（低温低浓 或
# 高温高浓 都能达标），真实成本地形是**双井（bimodal）非凸**：
#   f(x) = 谷底浅井(局部最优) + 谷底深井(全局最优) + 一个陡峭的"病态"窄谷
# 再叠加一个**病态二次项**（Hessian 条件数很大），用来暴露
# "梯度下降在病态地形上抖成锯齿"这一经典现象。
# ============================================================
A_COND = 200.0  # 病态方向 / 良态方向 的曲率比 → 条件数 ~200（故意病态）

# 两个工艺窗口中心（用来生成"双井"）：浅井(局部) 与 深井(全局)
W_LOCAL = np.array([0.8, 2.6])
W_GLOBAL = np.array([4.8, 2.2])

# 用「成对的高斯势阱」叠出一个双井地形：
#   两项等深+一项额外的"全局加深"，保证两个井都是真正的盆地，
#   而深井略深一点点（避免两井完全对称导致数值抖动）。
_SIG = 1.15          # 井的宽度
_DEPTH = 6.0         # 井深（够深 → 起点落进井内就跑不出去）

# 病态二次目标（单独用于"学习率敏感性"与"BFGS 学曲率"两个实验）：
#   quad(x) = 0.5*x0^2 + 0.5*K*x1^2 ，K=A_COND=200 → Hessian=diag(1,200)
# 两个方向的曲率差 200 倍，梯度下降会沿陡方向来回"锯齿"，
# 稳定步长上限 lr<2/200=0.01 → 稍大就发散，这就是"病态"的代价。
K_COND = A_COND
quad      = lambda z: 0.5 * z[0] ** 2 + 0.5 * K_COND * z[1] ** 2
quad_grad = lambda z: np.array([z[0], K_COND * z[1]])

# ---- 非凸"双井"成本函数（用于轨迹/多起点实验，不带病态碗）----
_BOWL_C = np.array([4.8, 2.2])   # 只保留很弱的碗，充当"背景"

def _well(r2):
    return -_DEPTH * np.exp(-r2 / (2.0 * _SIG ** 2))

def cost(x):
    """非凸双井工艺成本函数（我们要求最小化它）。"""
    x = np.asarray(x, dtype=float)
    c, t = x[..., 0], x[..., 1]
    # 很弱的背景碗（防止远处跑到无穷），不主导最优
    bowl = 0.02 * ((c - _BOWL_C[0]) ** 2) + 0.02 * ((t - _BOWL_C[1]) ** 2)
    well_local = _well((c - W_LOCAL[0]) ** 2 + (t - W_LOCAL[1]) ** 2)
    # 全局井额外加深 6.0 → 它比局部井更深，是真正的全局最优
    well_global = _well((c - W_GLOBAL[0]) ** 2 + (t - W_GLOBAL[1]) ** 2) - 6.0 \
        * np.exp(-((c - W_GLOBAL[0]) ** 2 + (t - W_GLOBAL[1]) ** 2) / (2.0 * _SIG ** 2))
    return bowl + well_local + well_global + 8.0

def grad(x):
    """解析梯度（算法只用到梯度，不喂 Hessian）。"""
    x = np.asarray(x, dtype=float)
    c, t = x[..., 0], x[..., 1]
    g_c = 0.02 * 2 * (c - _BOWL_C[0])
    g_t = 0.02 * 2 * (t - _BOWL_C[1])
    # 局部井梯度
    r2a = (c - W_LOCAL[0]) ** 2 + (t - W_LOCAL[1]) ** 2
    ea = -_DEPTH * np.exp(-r2a / (2.0 * _SIG ** 2))
    g_c += ea * (-(c - W_LOCAL[0]) / _SIG ** 2)
    g_t += ea * (-(t - W_LOCAL[1]) / _SIG ** 2)
    # 全局井梯度（含额外 0.6 加深项）
    r2b = (c - W_GLOBAL[0]) ** 2 + (t - W_GLOBAL[1]) ** 2
    eb = -_DEPTH * np.exp(-r2b / (2.0 * _SIG ** 2))
    g_c += eb * (-(c - W_GLOBAL[0]) / _SIG ** 2)
    g_t += eb * (-(t - W_GLOBAL[1]) / _SIG ** 2)
    ec = 6.0 * np.exp(-r2b / (2.0 * _SIG ** 2))
    g_c += ec * ((c - W_GLOBAL[0]) / _SIG ** 2)
    g_t += ec * ((t - W_GLOBAL[1]) / _SIG ** 2)
    return np.stack([g_c, g_t], axis=-1)

# 数值梯度自检（确保解析梯度没写错）—— 优化算法课最容易被"梯度写错"坑死
def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

_x0 = np.array([3.0, 0.0])
_g_t = grad(_x0)
_g_n = num_grad(cost, _x0)
assert np.allclose(_g_t, _g_n, atol=1e-4), f"解析梯度与数值梯度不符 {_g_t} vs {_g_n}"
print(f"[自检] 解析梯度与数值梯度一致（最大偏差 {np.max(np.abs(_g_t - _g_n)):.2e}）")

# ============================================================
# 二、五类优化算法（全部手写，零 scipy.optimize）
# ============================================================
def gd(gf, x0, lr, steps):
    """朴素梯度下降 GD。"""
    x = x0.astype(float).copy(); traj = [x.copy()]
    for _ in range(steps):
        x = x - lr * gf(x); traj.append(x.copy())
    return np.array(traj)

def momentum(gf, x0, lr, steps, beta=0.9):
    """动量法 Momentum：给梯度积累速度，冲过窄谷。"""
    x = x0.astype(float).copy(); v = np.zeros_like(x); traj = [x.copy()]
    for _ in range(steps):
        v = beta * v + (1 - beta) * gf(x)
        x = x - lr * v; traj.append(x.copy())
    return np.array(traj)

def rmsprop(gf, x0, lr, steps, beta=0.9, eps=1e-8):
    """RMSProp：按各维梯度平方的滑动平均自适应缩放步长。"""
    x = x0.astype(float).copy(); s = np.zeros_like(x); traj = [x.copy()]
    for _ in range(steps):
        g = gf(x)
        s = beta * s + (1 - beta) * g * g
        x = x - lr * g / (np.sqrt(s) + eps); traj.append(x.copy())
    return np.array(traj)

def adam(gf, x0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    """Adam：一阶矩 + 二阶矩 + 偏差校正。"""
    x = x0.astype(float).copy(); m = np.zeros_like(x); v = np.zeros_like(x); traj = [x.copy()]
    for k in range(1, steps + 1):
        g = gf(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** k); vhat = v / (1 - b2 ** k)
        x = x - lr * mhat / (np.sqrt(vhat) + eps); traj.append(x.copy())
    return np.array(traj)

def bfgs(gf, x0, steps, tol=1e-9):
    """
    BFGS 拟牛顿法：用梯度差分近似 Hessian 的逆，自己"学"出地形曲率。
    步长用最速下降方向上的回溯线搜索（Armijo）。
    """
    x = x0.astype(float).copy()
    n = len(x)
    H = np.eye(n)
    traj = [x.copy()]
    g = gf(x)
    for _ in range(steps):
        p = -H @ g                      # 拟牛顿方向
        if np.linalg.norm(p) < tol:
            break
        # Armijo 回溯：保证充分下降
        t = 1.0; c1 = 1e-4; f0 = cost(x)
        while t > 1e-12 and cost(x + t * p) > f0 + c1 * t * (g @ p):
            t *= 0.5
        x_new = x + t * p
        g_new = gf(x_new)
        s = x_new - x; y = g_new - g
        sy = s @ y
        if sy > 1e-12:                  # 保正曲率才更新
            rho = 1.0 / sy
            M = np.eye(n) - rho * np.outer(s, y)
            H = M @ H @ M.T + rho * np.outer(s, s)
        x, g = x_new, g_new
        traj.append(x.copy())
    return np.array(traj)

# ============================================================
# 三、跑实验：收敛速度 vs 学习率敏感性
# ============================================================
x0 = np.array([3.0, 0.0])          # 起点在两井之间的"分水岭"上
STEPS = 600
f_star = cost(W_GLOBAL)            # 全局最优附近的成本（作为收敛下界参考）

print("\n=== 实验1：五类算法在最优学习率下的收敛 ===")
runs = {}
# 为一阶方法各自挑一个"表现最好"的学习率（体现"学习率要靠调"）
best_lrs = {"GD": 0.006, "Momentum": 0.006, "RMSProp": 0.15, "Adam": 0.35}
for name, fn, lr in [("GD", gd, best_lrs["GD"]),
                     ("Momentum", momentum, best_lrs["Momentum"]),
                     ("RMSProp", rmsprop, best_lrs["RMSProp"]),
                     ("Adam", adam, best_lrs["Adam"])]:
    traj = fn(grad, x0, lr, STEPS)
    runs[name] = traj
    f = cost(traj)
    print(f"  {name:9s} lr={lr:<6} 末点 x*=[{traj[-1,0]:6.3f},{traj[-1,1]:6.3f}] "
          f"成本={f[-1]:7.4f}  首次进入 f<f*+0.05 的步数="
          f"{np.argmax(f < f_star + 0.05) if np.any(f < f_star + 0.05) else '未达到'}")

bfgs_traj = bfgs(grad, x0, 80)
runs["BFGS"] = bfgs_traj
f_bfgs = cost(bfgs_traj)
print(f"  {'BFGS':9s} {'(线搜索)':<8} 末点 x*=[{bfgs_traj[-1,0]:6.3f},{bfgs_traj[-1,1]:6.3f}] "
      f"成本={f_bfgs[-1]:7.4f}  迭代步数={len(bfgs_traj)-1}")

# 收敛到全局最优解才算真的成功
opt_pt = W_GLOBAL
def hit_global(traj, tol=0.25):
    return np.linalg.norm(traj - opt_pt, axis=1) < tol

# ============================================================
# 四、学习率敏感性：GD 步长太大/太小会怎样
# ============================================================
print("\n=== 实验2：GD 在病态二次目标上的学习率敏感性（条件数≈200） ===")
quad_x0 = np.array([3.0, 2.0])   # 病态二次目标起点(与双井地形无关)
lr_scan = [0.0005, 0.002, 0.006, 0.011, 0.02]
lr_records = []
for lr in lr_scan:
    traj = gd(quad_grad, quad_x0, lr, 300)
    f = quad(traj.T).astype(float)
    diverged = (not np.all(np.isfinite(f))) or f[-1] > 1e6
    lr_records.append((lr, f[-1] if not diverged else np.inf, diverged))
    tag = "发散!" if diverged else f"成本={f[-1]:.5f}"
    print(f"  lr={lr:<7} {tag}  (稳定上限 lr*<2/K={2.0/K_COND:.3f} → 超了就沿陡方向振荡/发散)")
assert lr_records[0][1] > lr_records[1][1], "过小学习率应收敛更慢（成本更高）"
assert any(d for _, _, d in lr_records), "病态地形下应存在某个学习率导致发散"

# ============================================================
# 五、非凸性：不同起点会不会掉进不同的井？
# ============================================================
print("\n=== 实验3：非凸地形下不同起点的归宿 ===")
starts = np.array([[3.0, 0.0], [0.2, 2.0], [6.5, 1.0], [0.9, 3.4]])
start_results = []
for s in starts:
    traj = adam(grad, s, 0.35, 600)
    end = traj[-1]
    to_glob = np.linalg.norm(end - opt_pt)
    to_local = np.linalg.norm(end - W_LOCAL)
    where = "全局最优" if to_glob < to_local else "局部最优(浅井)"
    start_results.append((s, end, cost(end), where))
    print(f"  起点 [{s[0]:4.1f},{s[1]:4.1f}] → 终点 [{end[0]:5.2f},{end[1]:5.2f}] "
          f"成本={cost(end):7.4f}  落在：{where}")
assert any(r[3] == "局部最优(浅井)" for r in start_results), "应至少有一个起点陷入局部最优"

# ============================================================
# 六、拟牛顿的"秘密"：它隐式学到了 Hessian 逆
# ------------------------------------------------------------
# 为了把"曲率"讲透，这里单独用一个**纯病态二次函数**（Hessian 已知）：
#   q(x) = 0.5*(x0)^2 + 0.5*K*(x1)^2 ，K=200 → Hessian=diag(1,200)
# BFGS 只拿到梯度，却能把 H≈Hessian^{-1}=diag(1, 1/200) 学出来。
# ============================================================
print("\n=== 实验4：BFGS 隐式逼近的曲率 vs 真实 Hessian 对角 ===")
H_true_inv_diag = np.array([1.0 / 1.0, 1.0 / K_COND])

z = np.array([3.0, 2.0]); n = 2; H = np.eye(n); g = quad_grad(z)
diag_hist = []
for _ in range(50):
    p = -H @ g
    if np.linalg.norm(p) < 1e-12: break
    tt = 1.0; c1 = 1e-4; f0 = quad(z)
    while tt > 1e-12 and quad(z + tt * p) > f0 + c1 * tt * (g @ p): tt *= 0.5
    zn = z + tt * p; gn = quad_grad(zn); s = zn - z; y = gn - g; sy = s @ y
    if sy > 1e-12:
        rho = 1.0 / sy
        M = np.eye(n) - rho * np.outer(s, y)
        H = M @ H @ M.T + rho * np.outer(s, s)
    z, g = zn, gn
    diag_hist.append(np.diag(H).copy())
diag_hist = np.array(diag_hist)
print(f"  真实 Hessian 逆对角 = [{H_true_inv_diag[0]:.4f}, {H_true_inv_diag[1]:.5f}]")
print(f"  BFGS 终态 H 对角    = [{diag_hist[-1,0]:.4f}, {diag_hist[-1,1]:.5f}]  "
      f"(用了 {len(diag_hist)} 步)")
print("  → BFGS 只靠梯度，却把'陡方向（逆曲率小）/缓方向（逆曲率大）'的"
      "尺度差异自动学了出来，这正是它比 GD 快 1~2 个量级的根本原因。")
assert abs(diag_hist[-1, 1] - H_true_inv_diag[1]) < 0.01, "BFGS 应逼近真实逆 Hessian"
assert diag_hist[-1, 1] < 0.2 * diag_hist[-1, 0], "陡方向的逆曲率应远小于缓方向"

# ============================================================
# 七、汇总断言
# ============================================================
print("\n=== 断言汇总 ===")
assert np.allclose(_g_t, _g_n, atol=1e-4)
assert cost(W_GLOBAL) < cost(W_LOCAL), "全局井应比局部井更深"
# BFGS 应最快到全局最优
bfgs_steps_to_glob = np.argmax(hit_global(bfgs_traj)) if np.any(hit_global(bfgs_traj)) else 999
print(f"  BFGS 进入全局最优邻域用了 {bfgs_steps_to_glob} 步（GD lr=0.006 需 "
      f"{np.argmax(hit_global(runs['GD'])) if np.any(hit_global(runs['GD'])) else '未达'} 步）")
assert bfgs_steps_to_glob < 40, "BFGS 应在 ~40 步内收敛"
assert np.isfinite(cost(bfgs_traj[-1]))
print("  全部断言通过 ✓")

# ============================================================
# 八、绘图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (1) 成本地形 + 各算法轨迹
ax = axes[0, 0]
gx = np.linspace(-1.5, 7.0, 400); gy = np.linspace(-1.0, 5.5, 400)
GX, GY = np.meshgrid(gx, gy)
GZ = cost(np.stack([GX, GY], axis=-1))
cs = ax.contourf(GX, GY, np.log1p(GZ - GZ.min()), levels=30, cmap="viridis")
ax.contour(GX, GY, GZ, levels=12, colors="white", linewidths=0.4, alpha=0.5)
colors = {"GD": "#ff4d4d", "Momentum": "#ffb000", "RMSProp": "#00b0ff", "Adam": "#00e676"}
for name in ["GD", "Momentum", "RMSProp", "Adam"]:
    tr = runs[name]
    ax.plot(tr[:, 0], tr[:, 1], "-", color=colors[name], lw=1.6, label=name, alpha=0.9)
ax.plot(bfgs_traj[:, 0], bfgs_traj[:, 1], "-o", color="white", lw=1.6, ms=2.5, label="BFGS")
ax.scatter([W_GLOBAL[0]], [W_GLOBAL[1]], marker="*", s=260, c="red", edgecolors="white",
           zorder=5, label="全局最优(深井)")
ax.scatter([W_LOCAL[0]], [W_LOCAL[1]], marker="X", s=140, c="orange", edgecolors="white",
           zorder=5, label="局部最优(浅井)")
ax.scatter([x0[0]], [x0[1]], marker="o", s=90, c="black", edgecolors="white", zorder=6, label="起点")
ax.set_title("成本地形与五类算法收敛轨迹（非凸+病态）", fontsize=12)
ax.set_xlabel("工艺参数 c（配方浓度）"); ax.set_ylabel("工艺参数 t（反应温度）")
ax.legend(fontsize=8, loc="upper right")

# (2) 收敛曲线（成本 vs 步数）
ax = axes[0, 1]
for name in ["GD", "Momentum", "RMSProp", "Adam"]:
    f = cost(runs[name])
    ax.plot(np.arange(len(f)), f - f_star, "-", color=colors[name], lw=1.8, label=name)
f = cost(bfgs_traj)
ax.plot(np.arange(len(f)), f - f_star, "-o", color="black", lw=1.8, ms=3, label="BFGS")
ax.set_yscale("log")
ax.set_title("收敛速度对比（纵轴=与全局最优的成本差，对数轴）", fontsize=12)
ax.set_xlabel("迭代步数"); ax.set_ylabel("成本 - 全局最优成本（log）")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (3) 学习率敏感性
ax = axes[1, 0]
for lr, fend, div in lr_records:
    traj = gd(grad, x0, lr, 400)
    f = cost(traj)
    f = np.where(f > 1e5, np.nan, f)   # 发散截断
    lbl = f"lr={lr}" + ("（发散）" if div else "")
    ax.plot(f - f_star, lw=1.6, label=lbl)
ax.set_yscale("log")
ax.set_title("GD 学习率敏感性：太小→慢，太大→病态方向振荡/发散", fontsize=12)
ax.set_xlabel("迭代步数"); ax.set_ylabel("成本 - 全局最优（log）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (4) BFGS 隐式学曲率
ax = axes[1, 1]
ax.plot(diag_hist[:, 0], "-", color="#d32f2f", lw=2, label="H[0]（缓方向：真实逆曲率 1.0）")
ax.plot(diag_hist[:, 1], "-", color="#1976d2", lw=2, label="H[1]（陡方向：真实逆曲率 0.005）")
ax.axhline(1.0 / 1.0, ls="--", color="#d32f2f", alpha=0.6, label="真实 1/Hessian[0]=1.0")
ax.axhline(1.0 / A_COND, ls="--", color="#1976d2", alpha=0.6,
           label=f"真实 1/Hessian[1]={1.0/A_COND:.4f}")
ax.set_yscale("log")
ax.set_title("BFGS 隐式逼近 Hessian 逆：自动学出病态尺度", fontsize=12)
ax.set_xlabel("BFGS 迭代步数"); ax.set_ylabel("H 对角元（log）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("Day33 · 无约束非线性优化：梯度下降家族 vs 拟牛顿法 BFGS", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gradient_descent_quasi_newton.png"), dpi=130)
print(f"\n[图已保存] {os.path.join(OUT, 'gradient_descent_quasi_newton.png')}")
print("完成。")
