# -*- coding: utf-8 -*-
"""
subject: mathematical-modeling
level: L6 优化模型 / 线性规划灵敏度分析（LP Sensitivity Analysis：最优基视角）
anchor: Learning_Roadmap L4–L6「优化模型：从求解到解释」；前接 Day26 对偶与影子价格、Day4 ILP 分支定界
goal: 回答「系数动了，方案还作数吗、有效期到哪天」——用「最优基 + 区间算术 + 全基枚举」给出
      LP 目标系数 / 右端项的可变区间与对偶价格的失效点，并以暴力采样做裁判。

场景：云溪县味源酱菜厂三条产线（爽口萝卜 / 香辣豆腐乳 / 老坛酸菜），三道工序（腌渍 / 灌装 / 杀菌），
      求周产量使周利润最大。

运行：cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-21/sensitivity_analysis_basis.py"
"""

import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

for _name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"]:
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_analysis_basis.png")

NAMES = ["爽口萝卜", "香辣豆腐乳", "老坛酸菜"]
PROC = ["腌渍", "灌装", "杀菌"]
A = np.array([[0.6, 1.0, 1.4],
              [1.5, 0.8, 0.7],
              [0.8, 1.1, 0.9]], dtype=float)
B0 = np.array([1210.0, 1560.0, 1330.0])
C0 = np.array([170.0, 210.0, 260.0])
NEW_C = C0 + np.array([45.0, -60.0, 75.0])


def to_std(A, b, c):
    m = A.shape[0]
    return np.hstack([A, np.eye(m)]), np.asarray(b, float), np.concatenate([c, np.zeros(m)]), A.shape[1], m


def analyze(A, b, c):
    """枚举全部基：可行基里检验数全 <= 0 的即最优。"""
    Ah, bh, ch, n, m = to_std(A, b, c)
    optimal = []
    for basis in itertools.combinations(range(n + m), m):
        basis = list(basis)
        B = Ah[:, basis]
        if abs(np.linalg.det(B)) < 1e-9:
            continue
        xB = np.linalg.solve(B, bh)
        if np.any(xB < -1e-9):
            continue
        y = np.linalg.solve(B.T, np.asarray(ch[basis], dtype=float))
        r = ch - np.asarray(Ah, dtype=float).T @ np.asarray(y, dtype=float)
        if not np.all(r <= 1e-9):
            continue
        x = np.zeros(n + m)
        x[basis] = xB
        optimal.append(dict(basis=basis, x=x, z=float(ch @ x), y=y, r=r))
    if not optimal:
        for dbg in (locals().get("c"),):
            pass
        raise RuntimeError(f"没有找到最优基：c={c}, b={b}")
    optimal.sort(key=lambda d: -d["z"])
    best = optimal[0]
    return dict(best=best, n=n, m=m, alt=len(optimal) > 1,
                nondeg=int(np.sum(best["x"] > 1e-9)) == m, Ah=Ah, ch=ch)


def c_interval(res, dshape):
    """最优基不变时 c(λ)=c0+λ·dshape 的 λ 区间。

    记 R(λ) = 成本按 dshape 方向挪 λ 之后的检验数向量，R(λ) 对 λ 是仿射的。
    最优基不变的充要条件：R_k(λ) <= 0 对所有 k 成立。
    这里不推解析式，直接对「是否有检验数变正」做**二分求临界 λ**，正负两侧各找一次。
    """
    basis = res["best"]["basis"]
    Ah, ch = res["Ah"], res["ch"]
    d = np.concatenate([np.asarray(dshape, float), np.zeros(Ah.shape[0])])
    B = np.asarray(Ah[:, basis], dtype=float)

    def reduc(lam):
        y = np.linalg.solve(B.T, ch[basis] + lam * d[basis])
        return (ch + lam * d) - y @ Ah

    assert np.all(reduc(0.0) <= 1e-8), "起点不是最优基"

    def bisect(direction):
        """沿 direction（+1/-1）找第一个使某检验数变正的 λ。"""
        hi_step = 1.0
        while hi_step < 1e7:
            if np.any(reduc(direction * hi_step) > 1e-9):
                lo_, hi_ = 0.0, hi_step
                for _ in range(80):
                    mid = (lo_ + hi_) / 2
                    if np.any(reduc(direction * mid) > 1e-9):
                        hi_ = mid
                    else:
                        lo_ = mid
                return direction * hi_
            hi_step *= 2.0
        return direction * float("inf")

    up = bisect(+1.0)
    down = bisect(-1.0)
    return down, up


def b_interval(res, b, i):
    """最优基不变时第 i 个右端项区间：B^{-1}(b+Δe_i) >= 0。"""
    basis = res["best"]["basis"]
    Binv = np.linalg.inv(res["Ah"][:, basis])
    xB = Binv @ b
    col = Binv[:, i]
    lo, hi = -np.inf, np.inf
    for t in range(len(col)):
        if abs(col[t]) < 1e-12:
            continue
        v = -xB[t] / col[t]
        if col[t] > 0:
            lo = max(lo, v)
        else:
            hi = min(hi, v)
    return b[i] + lo, b[i] + hi


def brute_z(A, b, c, grid=60, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.array([b[i] / A[:, i].min() for i in range(3)])
    best = -np.inf
    for _ in range(20000):
        x = rng.random(3) * ub
        if np.all(A @ x <= b + 1e-12):
            best = max(best, float(c @ x))
    for x in itertools.product(*[np.linspace(0, u, grid) for u in ub]):
        x = np.array(x)
        if np.all(A @ x <= b + 1e-12):
            best = max(best, float(c @ x))
    return best


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("A. 基准 LP：解出来了吗，落在哪组基上")
    print("=" * 78)
    res = analyze(A, B0, C0)
    best = res["best"]
    x = best["x"][:3]
    print(f"最优解 x = {x}   z* = {best['z']:.3f} 元 / 周")
    print(f"最优基（0-2 真实变量，3-5 松弛变量）= {best['basis']}")
    print(f"非退化? {res['nondeg']}   最优解唯一? {not res['alt']}")
    print(f"影子价格 y = {best['y']}")
    print(f"检验数 r = {best['r']}")
    assert abs(best["z"] - C0 @ x) < 1e-7
    assert np.all(A @ x <= B0 + 1e-9) and res["nondeg"] and not res["alt"]
    zb = brute_z(A, B0, C0)
    print(f"网格+随机采样最好值 {zb:.3f}（应 ≤ z*）")
    assert zb <= best["z"] + 1e-6

    print()
    print("=" * 78)
    print("B. 影子价格 + 有效期（右端项区间）")
    print("=" * 78)
    b_rng = {}
    for i, nm in enumerate(PROC):
        lo, hi = b_interval(res, B0, i)
        b_rng[nm] = (lo, hi)
        print(f"  {nm}: y={best['y'][i]:7.3f} 元/工分 | {B0[i]:.0f} → [{lo:.1f}, {hi:.1f}]"
              f"（{lo - B0[i]:+.1f} / {hi - B0[i]:+.1f}）")
        assert lo < B0[i] < hi

    print()
    print("=" * 78)
    print("C. 目标系数 c 的可变区间")
    print("=" * 78)
    c_rng = {}
    for j, nm in enumerate(NAMES):
        e = np.zeros(3)
        e[j] = 1.0
        lam_lo, lam_hi = c_interval(res, e)
        lo, hi = C0[j] + lam_lo, C0[j] + lam_hi
        c_rng[nm] = (lo, hi)
        print(f"  {nm}: {C0[j]:.0f} → [{lo:.2f}, {hi:.2f}]（降 {C0[j] - lo:.2f} / 涨 {hi - C0[j]:.2f}）")
        assert lo < C0[j] < hi

    print()
    print("=" * 78)
    print("C2. 零点核对：区间两端点处的真实最优基是否与解析区间吻合")
    n_ok, n_tot = 0, 0
    for j, nm in enumerate(NAMES):
        lo_, hi_ = c_rng[nm]
        for sign, edge in ((-1, lo_), (+1, hi_)):
            c_test = C0.copy()
            c_test[j] = edge + sign * (-0.5)       # 稍往区间内收 0.5 元
            n_ok += int(analyze(A, B0, c_test)["best"]["basis"] == best["basis"])
            n_tot += 1
    print(f"  6 个端点内收后仍保持最优基：{n_ok}/{n_tot}")
    assert n_ok == n_tot, "解析区间端点与真实最优基不一致"

    print()
    print("=" * 78)
    print("D. 边界核对：区间内一阶精确，越界后失真")
    print("=" * 78)
    lo_, hi_ = b_rng["腌渍"]
    for trial in [B0[0], hi_ - 1e-3, hi_ + 5, hi_ + 150]:
        b2 = B0.copy()
        b2[0] = trial
        r2 = analyze(A, b2, C0)
        dz = r2["best"]["z"] - best["z"]
        pred = best["y"][0] * (trial - B0[0])
        print(f"  b1={trial:8.2f} [{'区间内' if trial <= hi_ else '已越界'}] z={r2['best']['z']:9.2f} "
              f"Δz={dz:9.2f} 首阶={pred:9.2f} 误差={dz - pred:8.2f} 基={r2['best']['basis']}")
    r_in = analyze(A, np.array([hi_ - 1e-3, B0[1], B0[2]]), C0)
    assert abs((r_in["best"]["z"] - best["z"]) - best["y"][0] * (hi_ - 1e-3 - B0[0])) < 1e-3

    print()
    print("=" * 78)
    print("E. 情景：三种酱菜利润同时变（+45 / −60 / +75）")
    print("=" * 78)
    delta = NEW_C - C0
    dn = [float(c_rng[n][0] - C0[j]) for j, n in enumerate(NAMES)]
    up = [float(c_rng[n][1] - C0[j]) for j, n in enumerate(NAMES)]
    frac = sum(abs(delta[j]) / (up[j] if delta[j] > 0 else -dn[j]) for j in range(3))
    print(f"  Δc = {delta}")
    print("  余量：" + " / ".join(f"{NAMES[j]} 降{-dn[j]:.1f}涨{up[j]:.1f}" for j in range(3)))
    print(f"  100% 规则满格度 = {frac:.3f}")
    r3 = analyze(A, B0, NEW_C)
    x3 = r3["best"]["x"][:3]
    old_new = float(NEW_C @ x)
    print(f"  换价后最优：x={x3}  z*={r3['best']['z']:.2f}  基={r3['best']['basis']}")
    print(f"  旧方案按新价：{old_new:.2f} → 少赚 {r3['best']['z'] - old_new:.2f} 元")
    assert r3["best"]["z"] > old_new + 1e-6

    print()
    print("=" * 78)
    print("F. 决策读数")
    print("=" * 78)
    print(f"  1) 本周排产 {x[0]:.0f}/{x[1]:.0f}/{x[2]:.0f} 箱，周利润 {best['z']:.0f} 元")
    print(f"  2) 腌渍加班封顶价 {best['y'][0]:.2f} 元/工分，只对 {hi_ - B0[0]:.0f} 工分以内成立")
    print(f"  3) 酸菜利润只涨 {c_rng[NAMES[2]][1] - 260:.0f} 元就该改排产")
    print(f"  4) 三品种同调后旧方案少赚 {r3['best']['z'] - old_new:.0f} 元")

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    c1 = np.linspace(60, 340, 120)
    ax.plot(c1, [analyze(A, B0, np.array([v, C0[1], C0[2]]))["best"]["z"] for v in c1], lw=2, label="重新求解 z*")
    l1, h1 = c_rng[NAMES[0]]
    ax.axvspan(l1, h1, color="tab:green", alpha=0.15, label=f"最优基不变 [{l1:.0f}, {h1:.0f}]")
    ax.plot(c1, best["z"] + x[0] * (c1 - C0[0]), "--", color="tab:red", label=f"斜率 x1*={x[0]:.0f}")
    ax.axvline(C0[0], color="k", ls=":", lw=1)
    ax.set_xlabel("爽口萝卜利润 c1（元/箱）")
    ax.set_ylabel("最优目标值 z*（元）")
    ax.set_title("目标系数：区间内是直线")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    b1 = np.linspace(B0[0] - 400, B0[0] + 700, 120)
    ax.plot(b1, [analyze(A, np.array([v, B0[1], B0[2]]), C0)["best"]["z"] for v in b1], lw=2, label="重新求解 z*")
    lo1, hi1 = b_rng["腌渍"]
    ax.axvspan(lo1, hi1, color="tab:green", alpha=0.15, label=f"影子价格有效期 [{lo1:.0f}, {hi1:.0f}]")
    ax.plot(b1, best["z"] + best["y"][0] * (b1 - B0[0]), "--", color="tab:red", label=f"斜率 y1={best['y'][0]:.1f}")
    ax.axvline(B0[0], color="k", ls=":", lw=1)
    ax.set_xlabel("腌渍工分 b1")
    ax.set_ylabel("最优目标值 z*（元）")
    ax.set_title("右端项：影子价格只在有效期里值钱")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.barh(np.arange(3) - 0.18, [(c_rng[NAMES[j]][0] - C0[j]) / C0[j] * 100 for j in range(3)],
            height=0.36, color="tab:blue", label="可降 %")
    ax.barh(np.arange(3) + 0.18, [(c_rng[NAMES[j]][1] - C0[j]) / C0[j] * 100 for j in range(3)],
            height=0.36, color="tab:orange", label="可涨 %")
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{NAMES[j]} {C0[j]:.0f}" for j in range(3)], fontsize=9)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("相对基准的允许变动（%）")
    ax.set_title("三种酱菜的利润系数能扛多大波动")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    for j, nm in enumerate(NAMES):
        ax.plot(c1, [analyze(A, B0, np.array([v, C0[1], C0[2]]))["best"]["x"][j] for v in c1], lw=2, label=nm)
    ax.axvline(C0[0], color="k", ls=":", lw=1, label="基准 c1")
    ax.set_xlabel("爽口萝卜利润 c1（元/箱）")
    ax.set_ylabel("最优产量（箱/周）")
    ax.set_title("越过区间边界，最优基换人")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("线性规划灵敏度分析：解的「有效期」由最优基决定", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"\n图已保存：{OUT}")
    print("\n全部断言通过。")


if __name__ == "__main__":
    main()
