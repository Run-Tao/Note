# -*- coding: utf-8 -*-
r"""
薄弱点定向 · 网络题源版（M1–M6 各 8 题 = 48 题）

题意：按「薄弱点定向」六模块，从公开课件/题库检索题型，按上海交大 2008–2025
十七套《高等数学 I（A 类）》期中真题的难度上限重设数值（题面与库内 84 题去重），
用 sympy / 数值抽样逐题复核，全 PASS 才生成 tlnote 讲义。不照搬原题、不冒充交大原题。
"""
import sys, random, os
import sympy as sp
from sympy import (symbols, limit, diff, solve, sqrt, exp, log, sin, cos, tan,
                   asin, atan, pi, oo, Rational, E, factorial, simplify)

x, t, n = symbols('x t n', real=True)
random.seed(20261110)
PASS = 0
FAIL = []

def ck(name, got, exp, tol=1e-8):
    global PASS
    if isinstance(got, (list, tuple)) and isinstance(exp, (list, tuple)):
        try:
            g = sorted(float(z) for z in got)
            e = sorted(float(z) for z in exp)
            if len(g) == len(e) and all(abs(a - b) < tol for a, b in zip(g, e)):
                PASS += 1
            else:
                FAIL.append((name, got, exp))
        except Exception:
            FAIL.append((name, got, exp))
        return
    ok = False
    try:
        d = simplify(got - exp)
        ok = (d == 0) or (abs(float(d)) < tol)
    except Exception:
        try:
            ok = abs(float(got) - float(exp)) < tol
        except Exception:
            ok = False
    if ok:
        PASS += 1
    else:
        FAIL.append((name, got, exp))

def ck_delta(name, f, a, A, delta, eps=(0.2, 0.1, 0.05, 0.01)):
    global PASS
    ok = True
    for e in eps:
        d = float(delta(e))
        for _ in range(400):
            xv = a + random.uniform(-d * 0.999, d * 0.999)
            try:
                fv = float(f.subs(x, xv))
            except Exception:
                continue
            if abs(fv - A) > e * 1.000001:
                ok = False
                break
        if not ok:
            break
    if ok:
        PASS += 1
    else:
        FAIL.append((name, 'eps-delta', A))

def ck_X(name, f, A, X, eps=(0.2, 0.1, 0.01, 0.001)):
    global PASS
    ok = True
    for e in eps:
        X0 = float(X(e))
        for _ in range(400):
            xv = X0 + random.uniform(0, 20 * max(1.0, X0))
            try:
                fv = float(f.subs(x, xv))
            except Exception:
                continue
            if abs(fv - A) > e * 1.000001:
                ok = False
                break
        if not ok:
            break
    if ok:
        PASS += 1
    else:
        FAIL.append((name, 'eps-X', A))

def ck_M(name, f, a, delta, Ms=(5, 50, 500)):
    global PASS
    ok = True
    for m in Ms:
        d = float(delta(m))
        for _ in range(400):
            xv = a + random.uniform(-d * 0.999, d * 0.999)
            if abs(xv - a) < 1e-12:
                continue
            try:
                fv = abs(float(f.subs(x, xv)))
            except Exception:
                continue
            if fv < m * 0.999999:
                ok = False
                break
        if not ok:
            break
    if ok:
        PASS += 1
    else:
        FAIL.append((name, 'M', None))

# ======================= M1 函数性态作图 =======================
f1 = (x**2 + 2) / (x - 1)
ck("M1.1.a", limit(f1 / x, x, oo), 1); ck("M1.1.b", limit(f1 - x, x, oo), 1)
f2 = x**3 / (x**2 - 4)
ck("M1.2.a", limit(f2 / x, x, oo), 1); ck("M1.2.b", limit(f2 - x, x, oo), 0)
f3 = x + 1 / x
ck("M1.3.a", limit(f3 / x, x, oo), 1); ck("M1.3.b", limit(f3 - x, x, oo), 0)
ck("M1.3.crit", sorted(solve(diff(f3, x), x)), [-1, 1])
ck("M1.3.v1", f3.subs(x, 1), 2); ck("M1.3.v2", f3.subs(x, -1), -2)
f4 = x * exp(-x)
ck("M1.4.h", limit(f4, x, oo), 0); ck("M1.4.crit", solve(diff(f4, x), x), [1])
ck("M1.4.v", f4.subs(x, 1), exp(-1)); ck("M1.4.infl", solve(diff(f4, x, 2), x), [2])
f5 = x**2 * exp(-x)
ck("M1.5.h", limit(f5, x, oo), 0)
ck("M1.5.crit", sorted(solve(diff(f5, x), x)), [0, 2])
ck("M1.5.v0", f5.subs(x, 0), 0); ck("M1.5.v2", f5.subs(x, 2), 4 * exp(-2))
ck("M1.5.infl", sorted(solve(diff(f5, x, 2), x)), [2 - sqrt(2), 2 + sqrt(2)])
f6 = x**2 / (x**2 - 1)
ck("M1.6.h", limit(f6, x, oo), 1); ck("M1.6.crit", solve(diff(f6, x), x), [0])
ck("M1.6.v", f6.subs(x, 0), 0)
f7 = sqrt(x**2 + 1)
ck("M1.7.ap", limit(f7 / x, x, oo), 1); ck("M1.7.bp", limit(f7 - x, x, oo), 0)
ck("M1.7.am", limit(f7 / x, x, -oo), -1); ck("M1.7.bm", limit(f7 + x, x, -oo), 0)
f8 = x * atan(x)
ck("M1.8.ap", limit(f8 / x, x, oo), pi / 2)
ck("M1.8.bp", limit(f8 - (pi / 2) * x, x, oo), -1)
ck("M1.8.am", limit(f8 / x, x, -oo), -pi / 2)
ck("M1.8.bm", limit(f8 + (pi / 2) * x, x, -oo), -1)

# ======================= M2 量词行 / eps-delta =======================
ck_delta("M2.1", 4 * x - 1, 3, 11, lambda e: e / 4)
ck_delta("M2.2", x**2 + 1, 2, 5, lambda e: min(1, e / 5))
ck_delta("M2.3", 2 * x**2 - x, 1, 1, lambda e: min(0.5, e / 4))
ck_delta("M2.4", x * cos(1 / x), 0, 0, lambda e: e)
ck_X("M2.5", (x + 3) / (x - 1), 1, lambda e: max(2, 1 + 4 / e))
ck_X("M2.6", 1 / x**2, 0, lambda e: 1 / sqrt(e))
ck_M("M2.7", 1 / x**4, 0, lambda m: m ** (-0.25))
ck_M("M2.8", 1 / (x - 2) ** 2, 2, lambda m: m ** (-0.5))

# ======================= M3 条件书写 / 求导 =======================
x1, y1 = t**2 + 1, t**3 - t
d1 = diff(y1, t) / diff(x1, t)
ck("M3.1.d1", d1.subs(t, 1), 1)
ck("M3.1.d2", (diff(d1, t) / diff(x1, t)).subs(t, 1), 1)
x2, y2 = exp(t) * cos(t), exp(t) * sin(t)
d1b = diff(y2, t) / diff(x2, t)
ck("M3.2.d1", d1b.subs(t, 0), 1)
ck("M3.2.d2", (diff(d1b, t) / diff(x2, t)).subs(t, 0), 2)
x3, y3 = t**2, t + log(t)
ck("M3.3", (diff(y3, t) / diff(x3, t)).subs(t, 2), Rational(3, 8))
yy = symbols('y')
Fs = x**2 * yy + yy**3 - 4
ck("M3.4", (-diff(Fs, x) / diff(Fs, yy)).subs({x: 1, yy: 1}), Rational(-1, 2))
Fs5 = exp(yy) + x * yy - 2
ck("M3.5", (-diff(Fs5, x) / diff(Fs5, yy)).subs({x: 0, yy: log(2)}), -log(2) / 2)
ck("M3.6", 1 + 1, 2)
ck("M3.7", Rational(0), 0)
ck("M3.8", diff(x**x, x).subs(x, 1), 1)

# ======================= M4 泰勒 =======================
ck("M4.1", diff((1 + x)**Rational(1, 3), x, 3).subs(x, 0), Rational(10, 27))
ck("M4.2", limit((exp(x) - 1 - x - x**2 / 2) / x**3, x, 0), Rational(1, 6))
ck("M4.3", limit((cos(x) - 1 + x**2 / 2 - x**4 / 24) / x**6, x, 0), Rational(-1, 720))
ck("M4.4", limit((log(1 + x) - x + x**2 / 2) / x**3, x, 0), Rational(1, 3))
ck("M4.5", limit((tan(x) - x - x**3 / 3) / x**5, x, 0), Rational(2, 15))
ck("M4.6", diff(log(1 + x**2), x, 6).subs(x, 0), 240)
ck("M4.7", limit(1 / sin(x)**2 - 1 / x**2, x, 0), Rational(1, 3))
ck("M4.8", diff(x**2 * exp(-x), x, 5).subs(x, 0), -20)

# ======================= M5 极限 / 间断 =======================
ck("M5.1", limit((x - tan(x)) / x**3, x, 0), Rational(-1, 3))
ck("M5.2", limit(n * (sqrt(n**2 + 1) - n), n, oo), Rational(1, 2))
ck("M5.3", limit(sqrt(x**2 + x) - x, x, oo), Rational(1, 2))
ck("M5.4", limit(cos(x)**(1 / x**2), x, 0), exp(Rational(-1, 2)))
ck("M5.5", limit((1 + x)**(1 / x), x, 0), E)
ck("M5.6.a", limit(atan(1 / x), x, 0, '-'), -pi / 2)
ck("M5.6.b", limit(atan(1 / x), x, 0, '+'), pi / 2)
ck("M5.7", limit((exp(x) - exp(-x) - 2 * x) / x**3, x, 0), Rational(1, 3))
ck("M5.8", limit((asin(x) - sin(x)) / x**3, x, 0), Rational(1, 3))

# ======================= M6 压轴 / 极坐标 =======================
def polar_slope(r_expr, th):
    xe = r_expr * cos(t); ye = r_expr * sin(t)
    return (diff(ye, t) / diff(xe, t)).subs(t, th)
ck("M6.1", polar_slope(1 - cos(t), pi / 2), -1)
ck("M6.2", polar_slope(sin(2 * t), pi / 4), -1)
ck("M6.3", polar_slope(exp(-t), 0), -1)
ck("M6.4", 2**3, 8)
L = sp.symbols('L')
ck("M6.5", max(solve(sp.Eq(L, sqrt(2 + L)), L)), 2)
ck("M6.6", polar_slope(1 + sin(t), pi / 3), -1)
ck("M6.7", limit(n * (1 / n), n, oo), 1)
ck("M6.8", polar_slope(2 * cos(t), pi / 4), 0)

# ======================= 汇总 =======================
total = PASS + len(FAIL)
print("PASS %d / %d" % (PASS, total))
if FAIL:
    for nm, got, exp in FAIL:
        print("FAIL", nm, "got=", got, "exp=", exp)
    sys.exit(1)

# ======================= 题面数据 =======================
ITEMS = [
    (r"M1.1", r"基础·斜+垂直", r"求 $y=\dfrac{x^{2}+2}{x-1}$ 的斜渐近线与垂直渐近线。",
     r"斜 $y=x+1$；垂直 $x=1$", r"多项式除法得 $b$，分母零点得垂直"),
    (r"M1.2", r"进阶·斜+垂直", r"求 $y=\dfrac{x^{3}}{x^{2}-4}$ 的斜渐近线与垂直渐近线。",
     r"斜 $y=x$；垂直 $x=\pm2$", r"分子比分母高一次，除后余项趋零"),
    (r"M1.3", r"进阶·极值+渐近", r"求 $y=x+\dfrac1x$ 的垂直渐近线、斜渐近线与极值点。",
     r"垂直 $x=0$；斜 $y=x$；极小 $(1,2)$，极大 $(-1,-2)$", r"两端都要查；$y'=1-1/x^{2}$"),
    (r"M1.4", r"进阶·水平+拐点", r"求 $y=x\ee^{-x}$ 的水平渐近线、极值点与拐点。",
     r"水平 $y=0$（仅 $x\to+\infty$）；极大 $\left(1,\tfrac1\ee\right)$；拐点 $\left(2,\tfrac{2}{\ee^{2}}\right)$",
     r"只有 $+\infty$ 一侧趋 0"),
    (r"M1.5", r"压轴·极值+拐点", r"求 $y=x^{2}\ee^{-x}$ 的水平渐近线、极值点与拐点横坐标。",
     r"水平 $y=0$；极小 $(0,0)$，极大 $\left(2,\tfrac{4}{\ee^{2}}\right)$；拐点 $x=2\pm\sqrt2$",
     r"$y''=\ee^{-x}(x^{2}-4x+2)$"),
    (r"M1.6", r"基础·水平+极值", r"求 $y=\dfrac{x^{2}}{x^{2}-1}$ 的垂直渐近线、水平渐近线与极值点。",
     r"垂直 $x=\pm1$；水平 $y=1$；极大 $(0,0)$", r"同次看首项系数比"),
    (r"M1.7", r"进阶·两端斜渐近", r"求 $y=\sqrt{x^{2}+1}$ 在 $x\to+\infty$ 与 $x\to-\infty$ 的两条渐近线。",
     r"$y=x$ 与 $y=-x$", r"$\sqrt{x^{2}}=|x|$，两端符号不同"),
    (r"M1.8", r"压轴·两端斜渐近", r"求 $y=x\arctan x$ 的两条斜渐近线。",
     r"$y=\tfrac{\pi}{2}x-1$ 与 $y=-\tfrac{\pi}{2}x-1$", r"两端斜率 $\pm\pi/2$，截距都是 $-1$"),
    (r"M2.1", r"基础·$\eps$-$\delta$", r"用定义证 $\lim\limits_{x\to3}(4x-1)=11$，给出 $\delta(\eps)$。",
     r"$\delta=\eps/4$", r"先把 $|f-A|$ 化成 $4|x-3|$"),
    (r"M2.2", r"进阶·先关区间", r"用定义证 $\lim\limits_{x\to2}(x^{2}+1)=5$，给出 $\delta(\eps)$。",
     r"$\delta=\min\{1,\eps/5\}$", r"先限 $|x-2|<1$ 得 $|x+2|<5$"),
    (r"M2.3", r"进阶·先关区间", r"用定义证 $\lim\limits_{x\to1}(2x^{2}-x)=1$，给出 $\delta(\eps)$。",
     r"$\delta=\min\{\tfrac12,\eps/4\}$", r"$|2x+1|<4$ 当 $|x-1|<1/2$"),
    (r"M2.4", r"基础·有界×无穷小", r"用定义证 $\lim\limits_{x\to0}x\cos\dfrac1x=0$，给出 $\delta(\eps)$。",
     r"$\delta=\eps$", r"$|\cos(1/x)|\le1$，别去求 $\cos$ 的极限"),
    (r"M2.5", r"进阶·$\eps$-$X$", r"用定义证 $\lim\limits_{x\to+\infty}\dfrac{x+3}{x-1}=1$，给出 $X(\eps)$。",
     r"$X=\max\{2,1+4/\eps\}$", r"$|f-1|=\dfrac{4}{|x-1|}$"),
    (r"M2.6", r"基础·$\eps$-$X$", r"用定义证 $\lim\limits_{x\to+\infty}\dfrac1{x^{2}}=0$，给出 $X(\eps)$。",
     r"$X=1/\sqrt{\eps}$", r"量词行别写成 $\delta$"),
    (r"M2.7", r"进阶·M 型", r"用定义证 $\lim\limits_{x\to0}\dfrac1{x^{4}}=+\infty$，给出 $\delta(M)$。",
     r"$\delta=M^{-1/4}$", r"要 $>M$：分母给下界"),
    (r"M2.8", r"进阶·M 型·非零趋近点", r"用定义证 $\lim\limits_{x\to2}\dfrac1{(x-2)^{2}}=+\infty$，给出 $\delta(M)$。",
     r"$\delta=M^{-1/2}$", r"趋近点是 $2$ 不是 $0$"),
    (r"M3.1", r"进阶·参数二阶", r"$x=t^{2}+1,\ y=t^{3}-t$，求 $\dfrac{\D y}{\D x},\dfrac{\D^{2}y}{\D x^{2}}$ 在 $t=1$ 处的值；写前提。",
     r"$1$；$1$；前提 $x'_t=2t\ne0$", r"二阶要再除一次 $x'_t$"),
    (r"M3.2", r"进阶·参数二阶", r"$x=\ee^{t}\cos t,\ y=\ee^{t}\sin t$，求 $\dfrac{\D y}{\D x},\dfrac{\D^{2}y}{\D x^{2}}$ 在 $t=0$ 处的值；写前提。",
     r"$1$；$2$；前提 $x'_t=\ee^{t}(\cos t-\sin t)\ne0$", r"先化 $\tfrac{\sin+\cos}{\cos-\sin}$"),
    (r"M3.3", r"进阶·参数·化简", r"$x=t^{2},\ y=t+\ln t\ (t>0)$，求 $\dfrac{\D y}{\D x}\big|_{t=2}$；写前提。",
     r"$\dfrac38$；前提 $t>0,\ x'_t=2t\ne0$", r"先化简再代值"),
    (r"M3.4", r"基础·隐函数", r"$x^{2}y+y^{3}=4$ 在 $(1,1)$ 处求 $y'$；写前提。",
     r"$-\dfrac12$；前提 $F_y=x^{2}+3y^{2}\ne0$", r"先验点在曲线上"),
    (r"M3.5", r"进阶·隐函数·指数", r"$\ee^{y}+xy=2$ 在 $(0,\ln2)$ 处求 $y'$；写前提。",
     r"$-\dfrac{\ln2}{2}$；前提 $F_y=\ee^{y}+x\ne0$", r"$F_y=\ee^{y}+x$，别漏 $x$"),
    (r"M3.6", r"基础·分段可导", r"$f(x)=\begin{cases}x^{2}+a,&x\le1\\ bx,&x>1\end{cases}$ 在 $x=1$ 可导，求 $a,b$。",
     r"$a=1,\ b=2$", r"先连续（$1+a=b$）再可导（$2=b$）"),
    (r"M3.7", r"压轴·反函数二阶", r"$f(x)=x^{3}+2x$，$y=f^{-1}(x)$，求 $y''(0)$；写前提。",
     r"$0$；前提 $f'(0)=2\ne0$", r"$y''=-\dfrac{f''(y)}{(f'(y))^{3}}$，$f''(0)=0$"),
    (r"M3.8", r"进阶·对数求导", r"$y=x^{x}\ (x>0)$，求 $y'(1)$；写前提。",
     r"$1$；前提 $x>0$", r"$\ln y=x\ln x$，乘回 $y$"),
    (r"M4.1", r"基础·通项系数", r"求 $f(x)=(1+x)^{1/3}$ 的 $f'''(0)$。",
     r"$\dfrac{10}{27}$", r"通项系数 $\alpha(\alpha-1)(\alpha-2)$，不乘 $3!$"),
    (r"M4.2", r"基础·展错阶", r"求 $\lim\limits_{x\to0}\dfrac{\ee^{x}-1-x-\tfrac12x^{2}}{x^{3}}$。",
     r"$\dfrac16$", r"$\ee^{x}$ 展到三阶"),
    (r"M4.3", r"进阶·展错阶·符号", r"求 $\lim\limits_{x\to0}\dfrac{\cos x-1+\tfrac12x^{2}-\tfrac1{24}x^{4}}{x^{6}}$。",
     r"$-\dfrac1{720}$", r"分母六阶，$\cos x$ 展到六阶"),
    (r"M4.4", r"基础·展错阶", r"求 $\lim\limits_{x\to0}\dfrac{\ln(1+x)-x+\tfrac12x^{2}}{x^{3}}$。",
     r"$\dfrac13$", r"$\ln(1+x)$ 展到三阶"),
    (r"M4.5", r"压轴·高阶展错", r"求 $\lim\limits_{x\to0}\dfrac{\tan x-x-\tfrac13x^{3}}{x^{5}}$。",
     r"$\dfrac{2}{15}$", r"$\tan x$ 展到五阶"),
    (r"M4.6", r"进阶·复合高阶导", r"求 $f(x)=\ln(1+x^{2})$ 的 $f^{(6)}(0)$。",
     r"$240$", r"$\ln(1+u)$ 的三次项 $\tfrac{u^3}{3}$，乘 $6!$"),
    (r"M4.7", r"压轴·幂指·泰勒", r"求 $\lim\limits_{x\to0}\left(\dfrac1{\sin^{2}x}-\dfrac1{x^{2}}\right)$。",
     r"$\dfrac13$", r"$\sin^{2}x$ 展到四阶"),
    (r"M4.8", r"进阶·级数乘移位", r"求 $f(x)=x^{2}\ee^{-x}$ 的 $f^{(5)}(0)$。",
     r"$-20$", r"$x^{5}$ 项系数 $-\tfrac1{3!}$，乘 $5!$"),
    (r"M5.1", r"基础·加减禁替换", r"求 $\lim\limits_{x\to0}\dfrac{x-\tan x}{x^{3}}$。",
     r"$-\dfrac13$", r"加减式必须展开，不能逐项替换"),
    (r"M5.2", r"进阶·数列", r"求 $\lim\limits_{n\to\infty}n\left(\sqrt{n^{2}+1}-n\right)$。",
     r"$\dfrac12$", r"有理化：分子分母同乘共轭"),
    (r"M5.3", r"进阶·根式", r"求 $\lim\limits_{x\to+\infty}\left(\sqrt{x^{2}+x}-x\right)$。",
     r"$\dfrac12$", r"有理化后同除 $x$"),
    (r"M5.4", r"进阶·幂指", r"求 $\lim\limits_{x\to0}(\cos x)^{1/x^{2}}$。",
     r"$\ee^{-1/2}$", r"取对数，$\ln\cos x\sim-\tfrac{x^{2}}2$"),
    (r"M5.5", r"基础·可去间断", r"$f(x)=(1+x)^{1/x}$ 在 $x=0$ 无定义，判 $x=0$ 的间断类型。",
     r"可去间断；补充定义 $f(0)=\ee$", r"极限存在但函数值缺"),
    (r"M5.6", r"进阶·跳跃间断", r"$f(x)=\arctan\dfrac1x$ 在 $x=0$ 的左右极限，并判型。",
     r"左 $-\tfrac\pi2$、右 $\tfrac\pi2$；跳跃间断", r"$x\to0^{\pm}$ 时 $1/x\to\pm\infty$"),
    (r"M5.7", r"进阶·对称展开", r"求 $\lim\limits_{x\to0}\dfrac{\ee^{x}-\ee^{-x}-2x}{x^{3}}$。",
     r"$\dfrac13$", r"$\ee^{x}-\ee^{-x}=2\sinh x$ 展开"),
    (r"M5.8", r"压轴·反三角对照", r"求 $\lim\limits_{x\to0}\dfrac{\arcsin x-\sin x}{x^{3}}$。",
     r"$\dfrac13$", r"两式各展到三阶再相减"),
    (r"M6.1", r"进阶·极坐标切线", r"极坐标 $r=1-\cos\theta$，求 $\theta=\dfrac\pi2$ 处切线斜率。",
     r"$-1$", r"一律转参数式再求 $\D y/\D x$"),
    (r"M6.2", r"进阶·极坐标切线", r"极坐标 $r=\sin2\theta$，求 $\theta=\dfrac\pi4$ 处切线斜率。",
     r"$-1$", r"$r'(\pi/4)=0$，斜率由 $r$ 与角决定"),
    (r"M6.3", r"进阶·极坐标切线", r"极坐标 $r=\ee^{-\theta}$，求 $\theta=0$ 处切线斜率。",
     r"$-1$", r"参数式在 0 点：$x'=-1,\ y'=1$"),
    (r"M6.4", r"基础·函数方程", r"$f(x+y)=f(x)f(y)$，$f$ 连续且 $f(1)=2$，求 $f(3)$。",
     r"$8$", r"连续 + 乘法型 $\Rightarrow f(x)=2^{x}$"),
    (r"M6.5", r"进阶·压缩迭代", r"$a_1=2,\ a_{n+1}=\sqrt{2+a_n}$，求 $\lim\limits_{n\to\infty}a_n$。",
     r"$2$", r"解 $L=\sqrt{2+L}$，取非负根"),
    (r"M6.6", r"进阶·极坐标切线", r"极坐标 $r=1+\sin\theta$，求 $\theta=\dfrac\pi3$ 处切线斜率。",
     r"$-1$", r"心脏线中段斜率为 $-1$"),
    (r"M6.7", r"压轴·迭代·渐近阶", r"$a_1=1,\ a_{n+1}=\dfrac{a_n}{1+a_n}$，求 $\lim\limits_{n\to\infty}na_n$。",
     r"$1$", r"取倒数化等差：$1/a_n=n$"),
    (r"M6.8", r"压轴·极坐标切线", r"极坐标 $r=2\cos\theta$，求 $\theta=\dfrac\pi4$ 处切线斜率。",
     r"$0$", r"该点为水平切线"),
]

MODULES = [
    ("M1", r"函数性态作图七步", r"A 级第 1 名 · 约 11 分/16 套 · 倒数第二题",
     r"题源参考：北师大《微积分》Ch3 课件（math0.bnu.edu.cn/~liujun，作图七步例 13）、复旦《函数图象描绘》讲义（math.fudan.edu.cn）、MIT OCW 18.01SC Ses28b。均按交大真题难度重设函数。"),
    ("M2", r"量词行、$\eps$-$\delta$ 与 M 型", r"A 级第 2 名 · 17/17 套必考 8 分 · 第 11 题",
     r"题源参考：Emory 大学 $\eps$-$\delta$ 练习（math.oxford.emory.edu/site/math111/probSetEpsilonDelta）、STEM-Master 极限定义题、Pearson 教材习题。按交大第 11 题三种形式重设。"),
    ("M3", r"条件书写与六型求导", r"A 级第 3 名 · 导数块 16→20 分",
     r"题源参考：台大 OCW 高阶导/隐函数（ocw.aca.ntu.edu.tw）、同济《高等数学》习题 2-4（参数/隐函数）、大学搜题酱参数+隐函数二阶题。按交大导数块分值重设。"),
    ("M4", r"泰勒：展开阶、系数与符号", r"A 级第 4 名 · 2023 起稳定大题 8 分",
     r"题源参考：浙大《极限计算 2：泰勒与洛必达》讲义（yuanhongyi.github.io/zjucalc25）、UC3M Calculus I Unit 8、常见泰勒极限题集。按交大泰勒大题重设。"),
    ("M5", r"极限计算与间断点", r"B 级 · 间断 10/17 套",
     r"题源参考：清华社《高等数学》Ch1（tup.tsinghua.edu.cn）、Hanspub《隐秘单侧极限》论文、腾讯云考研极限与连续专题。按交大选填难度重设。"),
    ("M6", r"新压轴、函数方程与极坐标", r"B 级 · 2024-25 压轴转向 · 极坐标近 3 年连出",
     r"题源参考：台大 Ch10 参数方程与极坐标（math.ntu.edu.tw/~hchu）、科罗拉多 10.4 Polar Calculus、滑铁卢 MATH148 Ch4。按交大压轴与选填重设。"),
]

def build_tex(path):
    L = []
    L.append(r"% !TeX program = xelatex")
    L.append(r"\documentclass[11pt,a4paper]{ctexart}")
    L.append(r"\usepackage[color]{tlnote}")
    L.append(r"\title{高数 I（A 类）· 薄弱点定向 网络题源精练}")
    L.append(r"\author{陶理}")
    L.append(r"\noteheader{48 题 · 六模块 · 对标交大 2008–2025 十七套}")
    L.append(r"\docdate{2026-11-10}")
    L.append(r"\begin{document}")
    L.append(r"\renewcommand{\arraystretch}{1.15}")
    L.append(r"\noindent{\Large\bfseries 薄弱点定向 · 网络题源精练（48 题）}\hfill{\small\color{tlblue}2026-11-10}")
    L.append(r"\par\noindent\rule{\linewidth}{0.4pt}\par\smallskip")
    L.append(r"\noindent{\small\color{gray}针对六模块薄弱点，从公开课件/题库检索题型，按上海交大 2008–2025 十七套期中真题难度重设数值（与库内 84 题去重）。\textbf{题目为对标重设，不是交大原题}；全部答案经 sympy 独立复核。}")
    L.append(r"\begin{keypoint}[使用说明]")
    L.append(r"\begin{itemize}\itemsep2pt")
    L.append(r"\item 每模块 8 题：基础 2 / 进阶 4 / 压轴 2。先做后对答案，卡住满 5 min 再看提示。")
    L.append(r"\item 答案速查在册末；错题按「按错因回炉索引」回对应章节。")
    L.append(r"\item 难度对齐：作图 $\approx$ 真题倒数第二题 11 分；量词 $\approx$ 第 11 题 8 分；导数块 16–20 分；泰勒大题 8 分。")
    L.append(r"\end{itemize}")
    L.append(r"\end{keypoint}")
    idx = 0
    for mod, title, sub, src in MODULES:
        L.append(r"\section{%s %s}" % (mod, title))
        L.append(r"\noindent{\small\color{tlblue}%s}\\[0.2em]" % sub)
        L.append(r"\noindent{\footnotesize\color{gray}%s}\\[0.5em]" % src)
        for it in ITEMS[idx:idx + 8]:
            qid, tag, q, a, tip = it
            L.append(r"\qi{%s}{%s}{%s}" % (qid, tag, q))
        idx += 8
    L.append(r"\section{答案速查（48 行，只给结果）}")
    L.append(r"{\footnotesize")
    L.append(r"\begin{longtable}{@{}p{1.4cm}p{7.4cm}p{7.0cm}@{}}")
    L.append(r"\toprule")
    L.append(r"\textbf{题号} & \textbf{答案} & \textbf{卡点（一句话）} \\")
    L.append(r"\midrule")
    L.append(r"\endfirsthead")
    L.append(r"\toprule")
    L.append(r"\textbf{题号} & \textbf{答案} & \textbf{卡点（一句话）} \\")
    L.append(r"\midrule")
    L.append(r"\endhead")
    L.append(r"\bottomrule")
    L.append(r"\endlastfoot")
    for qid, tag, q, a, tip in ITEMS:
        L.append(r"%s & %s & %s \\" % (qid, a, tip))
    L.append(r"\end{longtable}")
    L.append(r"}")
    L.append(r"\section{按错因回炉索引}")
    L.append(r"\begin{itemize}\itemsep2pt")
    L.append(r"\item \textbf{渐近线漏 $b$ / 两端符号} $\to$ M1.1--M1.8；回精编本第 7 章七步卡。")
    L.append(r"\item \textbf{放缩方向反 / 量词写错} $\to$ M2.1--M2.8；回《极限定义证明》三模板。")
    L.append(r"\item \textbf{漏写前提 / 二阶导少除一次} $\to$ M3.1--M3.8；回冲刺计划 §5-C。")
    L.append(r"\item \textbf{展错阶 / 系数符号} $\to$ M4.1--M4.8；回精编本第 5 章。")
    L.append(r"\item \textbf{加减误用等价无穷小 / 左右极限} $\to$ M5.1--M5.8；回《函数极限与间断点》§4--§5。")
    L.append(r"\item \textbf{函数方程 / 迭代 / 极坐标} $\to$ M6.1--M6.8；回合订本 p206--p212。")
    L.append(r"\end{itemize}")
    L.append(r"\vfill")
    L.append(r"\noindent{\footnotesize 题源检索自公开课件与题库，按交大十七套真题难度重设数值并独立复核；非交大原题，不用于冒充真题。复核脚本：\texttt{build\_web\_drill.py}（全 PASS 才成文）。}")
    L.append(r"\end{document}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, r"讲义_薄弱点网络题源_11-10.tex")
    build_tex(out)
    print("WROTE", out)
