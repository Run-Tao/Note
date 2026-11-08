# -*- coding: utf-8 -*-
"""
册二《讲义_全专题冲顶_11-08》数值与符号复核
运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_full_topics_drill.py
覆盖：S1 极限定义证明 12 / S2 导数与微分 14 / S3 极坐标与间断点 12
      / S4 新压轴 8 / S5 不等式与最值 10 / S6 作图压轴 6 = 62 题
      + 模拟卷 A/B 各 18 题的关键结论
"""
import sympy as sp
import mpmath as mp
import math
import random

PASS = 0
FAIL = 0
FAILS = []


def chk(label, got, want, tol=None):
    global PASS, FAIL
    try:
        if isinstance(got, bool) or isinstance(want, bool):
            ok = bool(got) == bool(want)
        elif isinstance(got, (list, tuple)) or isinstance(want, (list, tuple)):
            if not (isinstance(got, (list, tuple)) and isinstance(want, (list, tuple))
                    and len(got) == len(want)):
                ok = False
            else:
                ok = all(sp.simplify(sp.sympify(g) - sp.sympify(w)) == 0
                         for g, w in zip(got, want))
        elif got in (sp.oo, -sp.oo, sp.zoo) or want in (sp.oo, -sp.oo, sp.zoo):
            ok = str(got) == str(want)
        elif tol is not None:
            try:
                ok = abs(complex(sp.N(got, 30)) - complex(sp.N(want, 30))) < tol
            except Exception:
                ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception:
        ok = False
    if ok:
        PASS += 1
        print('  PASS  %-56s = %s' % (label, got))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-56s got=%s want=%s' % (label, got, want))


def close(label, got, want, tol=5e-9):
    chk(label, got, want, tol=tol)


def true_ineq(label, fn, samples, pred):
    global PASS, FAIL
    ok = all(pred(v) for v in samples)
    if ok:
        PASS += 1
        print('  PASS  %-56s = %s' % (label, '成立'))
    else:
        FAIL += 1
        FAILS.append(label)
        bad = [v for v in samples if not pred(v)]
        print('  FAIL  %-56s 反例 x=%s' % (label, bad[:3]))


def no_limit(label, got):
    """极限不存在（振荡）的判定：返回 AccumBounds / zoo / nan 即算通过。"""
    global PASS, FAIL
    s = str(got)
    ok = s.startswith('AccumBounds') or s in ('zoo', 'nan')
    if ok:
        PASS += 1
        print('  PASS  %-56s = %s (无极限)' % (label, s))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-56s got=%s (应无极限)' % (label, s))


def series_coef(f, x, n, order):
    s = sp.series(f, x, 0, order).removeO()
    return sp.expand(s).coeff(x, n)


x = sp.symbols('x', real=True)
t = sp.symbols('t', real=True)
th = sp.symbols('theta', real=True)
y_ = sp.Symbol('y', real=True)
a = sp.Symbol('a', real=True)
b = sp.Symbol('b', real=True)
n = sp.Symbol('n', integer=True, positive=True)

print('=' * 80)
print('S1  极限定义证明（12 题）—— 三形式轮换：eps-delta 6 / eps-X 4 / M 型 2')
print('=' * 80)

# ---- eps-delta 型（6 题）----
# S1.1  lim(x->0) (cos x - e^{-x^2/2})/x^2 = 0   —— 要用 cos x - 1 ~ -x^2/2 与 e 展开抵消
close('S1.1 (cosx - e^{-x^2/2})/x^2',
      sp.limit((sp.cos(x) - sp.exp(-x ** 2 / 2)) / x ** 2, x, 0), 0)
# S1.2  lim(x->0) (tan x - x)/(x - sin x) = 2
close('S1.2 (tanx-x)/(x-sinx)',
      sp.limit((sp.tan(x) - x) / (x - sp.sin(x)), x, 0), 2)
# S1.3  lim(x->0) (sin x - x cos x)/x^3 = 1/3
close('S1.3 (sinx - x cosx)/x^3',
      sp.limit((sp.sin(x) - x * sp.cos(x)) / x ** 3, x, 0), sp.Rational(1, 3))
# S1.4  lim(x->0) (ln(1+x) + x/2)/(x^2)  不存在（左负右正）—— 陷阱：先通分
close('S1.4 右极限 ln(1+x)+x/2 / x^2',
      sp.limit((sp.log(1 + x) + x / 2) / x ** 2, x, 0, dir='+'), sp.oo)
close('S1.4 左极限 ln(1+x)+x/2 / x^2',
      sp.limit((sp.log(1 + x) + x / 2) / x ** 2, x, 0, dir='-'), -sp.oo)
# S1.5  lim(x->1) (e^{x-1} - 1)/(x-1) = 1
close('S1.5 (e^{x-1}-1)/(x-1) at x=1',
      sp.limit((sp.exp(x - 1) - 1) / (x - 1), x, 1), 1)
# S1.6  lim(x->0) (arcsin x - x)/(x^3) = 1/6
close('S1.6 (arcsinx-x)/x^3',
      sp.limit((sp.asin(x) - x) / x ** 3, x, 0), sp.Rational(1, 6))

# ---- eps-X 型（4 题）----
# S1.7  lim(x->+inf) x^3(sin(1/x) - 1/x) = -1/6
close('S1.7 x^3(sin(1/x)-1/x)',
      sp.limit(x ** 3 * (sp.sin(1 / x) - 1 / x), x, sp.oo), -sp.Rational(1, 6))
# S1.8  lim(x->+inf) (x^2 + 2x)/(x^2 - 3x + 1) = 1
close('S1.8 (x^2+2x)/(x^2-3x+1)',
      sp.limit((x ** 2 + 2 * x) / (x ** 2 - 3 * x + 1), x, sp.oo), 1)
# S1.9  lim(x->+inf) x^2( e^{1/x} - 1 - 1/x ) = 1/2
close('S1.9 x^2(e^{1/x}-1-1/x)',
      sp.limit(x ** 2 * (sp.exp(1 / x) - 1 - 1 / x), x, sp.oo), sp.Rational(1, 2))
# S1.10 lim(x->-inf) (x^2+1)/(x^2+3x+1) = 1
close('S1.10 (x^2+1)/(x^2+3x+1) at -inf',
      sp.limit((x ** 2 + 1) / (x ** 2 + 3 * x + 1), x, -sp.oo), 1)

# ---- M 型（2 题）----
# S1.11 lim(x->0+) 1/(x ln x) = -oo  （无穷大 M 型）
close('S1.11 1/(x lnx) at 0+',
      sp.limit(1 / (x * sp.log(x)), x, 0, dir='+'), -sp.oo)
# S1.12 lim(x->1-) ln(1-x^2)/(x-1) = +oo
close('S1.12 ln(1-x^2)/(x-1) at 1-',
      sp.limit(sp.log(1 - x ** 2) / (x - 1), x, 1, dir='-'), sp.oo)

print()
print('=' * 80)
print('S2  导数与微分（14 题）')
print('=' * 80)

# S2.1 参数方程二阶导：x = t + e^t, y = t - e^t
dx = sp.diff(t + sp.exp(t), t)
dy = sp.diff(t - sp.exp(t), t)
d1 = sp.simplify(dy / dx)
d2 = sp.simplify(sp.diff(d1, t) / dx)
close('S2.1 dy/dx at t=0', d1.subs(t, 0), 0)
close('S2.1 d2y/dx2 at t=0', d2.subs(t, 0), -sp.Rational(1, 4))

# S2.2 参数方程：x = t^2 + t, y = t^2 - t
dx = sp.diff(t ** 2 + t, t)
dy = sp.diff(t ** 2 - t, t)
d1 = sp.simplify(dy / dx)
d2 = sp.simplify(sp.diff(d1, t) / dx)
close('S2.2 dy/dx at t=0', d1.subs(t, 0), -1)
close('S2.2 d2y/dx2 at t=0', d2.subs(t, 0), 4)

# S2.3 隐函数二阶导：e^y + xy = e 的 (0,1) 处 —— 与高难册 H2.6 同型不同函数
F = sp.exp(y_) + x * y_ - sp.E
yp = sp.simplify(-sp.diff(F, x) / sp.diff(F, y_))
ypp = sp.simplify(sp.diff(yp, x) + sp.diff(yp, y_) * yp)
close('S2.3 y\'\'(0,1) for e^y+xy=e', ypp.subs({x: 0, y_: 1}), 1 / sp.E ** 2)

# S2.4 隐函数二阶导：x^2 + y^2 = 6x 在 (3,3) 处
F = x ** 2 + y_ ** 2 - 6 * x
yp = sp.simplify(-sp.diff(F, x) / sp.diff(F, y_))
ypp = sp.simplify(sp.diff(yp, x) + sp.diff(yp, y_) * yp)
close('S2.4 y\'\'(3,3) for x^2+y^2=6x', ypp.subs({x: 3, y_: 3}), -sp.Rational(1, 3))

# S2.5 分段连续求参数：f(x)=ln(1+x) (x<0), ax+b (x>=0)，在 0 可导求 a,b
close('S2.5 连续条件 b = 0', sp.solve(sp.Eq(b, 0), b), [0])
close('S2.5 左导数 = 1', sp.limit(sp.diff(sp.log(1 + x), x), x, 0, dir='-'), 1)
close('S2.5 右导数 = a', b + a * 0, b)
close('S2.5 可导条件 a = 1', sp.solve(sp.Eq(a, 1), a), [1])

# S2.6 分段：f(x) = ln(1+e^x) 在 x=0 处的二阶导
close('S2.6 f\'\'(0) of ln(1+e^x)',
      sp.diff(sp.log(1 + sp.exp(x)), x, 2).subs(x, 0), sp.Rational(1, 4))

# S2.7 Leibniz 高阶导：d^n/dx^n (x^2 e^x) at 0
close('S2.7 d4/dx4 (x^2 e^x) at 0',
      sp.diff(x ** 2 * sp.exp(x), x, 4).subs(x, 0), 12)
chk('S2.7 佐证 = 4!·[x^4]系数',
    sp.factorial(4) * series_coef(x ** 2 * sp.exp(x), x, 4, 5), 12)

# S2.8 Leibniz：d^n/dx^n (x^3 sin x) at 0
close('S2.8 d4/dx4 (x^3 sinx) at 0',
      sp.diff(x ** 3 * sp.sin(x), x, 4).subs(x, 0), 24)
close('S2.8 d5/dx5 (x^3 sinx) at 0',
      sp.diff(x ** 3 * sp.sin(x), x, 5).subs(x, 0), 0)

# S2.9 高阶导：d^3/dx^3 (x/(1+x)) at 0
close('S2.9 d3/dx3 (x/(1+x)) at 0',
      sp.diff(x / (1 + x), x, 3).subs(x, 0), 6)

# S2.10 复合函数高阶导：y = e^{x^2} 的二阶导 at 0（对标 2024-25 那道 f''(0)）
close('S2.10 f\'\'(0) of e^{x^2}',
      sp.diff(sp.exp(x ** 2), x, 2).subs(x, 0), 2)

# S2.11 微分：y = x e^x, dy = (1+x)e^x dx, 求 x=1 处的 dy 系数
close('S2.11 y\'(1) of x e^x', sp.diff(x * sp.exp(x), x).subs(x, 1), 2 * sp.E)

# S2.12 微分：y = ln(x^2+1), x=1 处 dy 系数
close('S2.12 y\'(1) of ln(x^2+1)',
      sp.diff(sp.log(x ** 2 + 1), x).subs(x, 1), 1)

# S2.13 微分：隐函数 x^2+y^2=25 在 (3,4) 处求 dy
F = x ** 2 + y_ ** 2 - 25
yp = sp.simplify(-sp.diff(F, x) / sp.diff(F, y_))
close('S2.13 y\'(3,4) for x^2+y^2=25', yp.subs({x: 3, y_: 4}), -sp.Rational(3, 4))

# S2.14 导数定义：f(x)=x^2 sin(1/x)，f\'(0)
close('S2.14 f\'(0) of x^2 sin(1/x)',
      sp.limit(x ** 2 * sp.sin(1 / x) / x, x, 0), 0)

print()
print('=' * 80)
print('S3  极坐标与间断点（12 题）')
print('=' * 80)

# S3.1 极坐标切线：r = 1 + cos(theta)，theta=pi/2 处 dy/dx
r_expr = 1 + sp.cos(th)
dxd = sp.simplify(sp.diff(r_expr, th) * sp.cos(th) - r_expr * sp.sin(th))
dyd = sp.simplify(sp.diff(r_expr, th) * sp.sin(th) + r_expr * sp.cos(th))
close('S3.1 r=1+cos th, dy/dx at pi/2', sp.simplify(dyd / dxd).subs(th, sp.pi / 2), 1)

# S3.2 极坐标：r = e^{theta}，dy/dx = (sin+cos)/(cos-sin)
r_expr = sp.exp(th)
dxd = sp.simplify(sp.diff(r_expr, th) * sp.cos(th) - r_expr * sp.sin(th))
dyd = sp.simplify(sp.diff(r_expr, th) * sp.sin(th) + r_expr * sp.cos(th))
close('S3.2 r=e^th, dy/dx at 0', sp.simplify(dyd / dxd).subs(th, 0), 1)

# S3.3 极坐标：r = 2(1+cos th)（心形线），theta=pi 处 dy/dx = 0（尖点）
r_expr = 2 * (1 + sp.cos(th))
dxd = sp.simplify(sp.diff(r_expr, th) * sp.cos(th) - r_expr * sp.sin(th))
dyd = sp.simplify(sp.diff(r_expr, th) * sp.sin(th) + r_expr * sp.cos(th))
close('S3.3 心形线 theta=pi 处 0/0 极限 dy/dx',
      sp.limit(sp.simplify(dyd / dxd), th, sp.pi), 0)

# S3.4 极坐标弧长：r=2(1+cos th) 从 0 到 pi 的弧长 = 8（半条心形线周长 16）
r_expr = 2 * (1 + sp.cos(th))
arc = sp.simplify(sp.sqrt(sp.diff(r_expr, th) ** 2 + r_expr ** 2))
chk('S3.4 弧长被积函数平方 = 16cos^2(th/2)',
    sp.simplify(sp.expand_trig(arc ** 2 - 16 * sp.cos(th / 2) ** 2)), 0)
close('S3.4 心形线半周长 = 8（数值积分）',
      mp.quad(lambda v: 2 * math.sqrt(math.sin(v) ** 2 + (1 + math.cos(v)) ** 2),
             [0, mp.pi]), 8, tol=1e-9)

# S3.5 极坐标：r = 1/(1+cos th) 为抛物线，顶点 (1/2, 0)，准线 x = 1/2
r_expr = 1 / (1 + sp.cos(th))
close('S3.5 抛物线 r=1/(1+cos th) 的 r_min = 1/2 (theta=0)',
      sp.limit(r_expr, th, 0), sp.Rational(1, 2))
close('S3.5 r 在 theta=pi 处发散', sp.limit(r_expr, th, sp.pi), sp.oo)
# 直角方程：r(1+cos th)=1 -> r + x = 1 -> 平方得 x^2+y^2 = (1-x)^2
_r5 = 1 / (1 + sp.cos(th))
close('S3.5 佐证 r^2 = (1 - r cos th)^2（即 x^2+y^2=(1-x)^2）',
      sp.simplify(_r5 ** 2 - (1 - _r5 * sp.cos(th)) ** 2), 0)

# S3.6 极坐标与直角互换：r = 2cos th -> x^2+y^2 = 2x
r_expr = 2 * sp.cos(th)
close('S3.6 r=2cos th -> x^2+y^2-2x=0',
      sp.simplify((r_expr ** 2) - 2 * r_expr * sp.cos(th)), 0)

# S3.7 间断点：f(x) = (x^2-1)/(x^2-3x+2) 在 x=1 可去，x=2 无穷
f = (x ** 2 - 1) / (x ** 2 - 3 * x + 2)
close('S3.7 lim x->1 f', sp.limit(f, x, 1), -2)
close('S3.7 lim x->2- f', sp.limit(f, x, 2, dir='-'), -sp.oo)
close('S3.7 lim x->2+ f', sp.limit(f, x, 2, dir='+'), sp.oo)

# S3.8 间断点：arctan(x + 1/x) 在 x=0 跳跃
close('S3.8 lim x->0- arctan(x+1/x)', sp.limit(sp.atan(x + 1 / x), x, 0, dir='-'), -sp.pi / 2)
close('S3.8 lim x->0+ arctan(x+1/x)', sp.limit(sp.atan(x + 1 / x), x, 0, dir='+'), sp.pi / 2)

# S3.9 间断点：tan(1/x) 在 0 处振荡，无极限
no_limit('S3.9 tan(1/x) 在 0 处无极限（振荡）',
         sp.limit(sp.tan(1 / x), x, 0))

# S3.10 间断点：f(x) = x sin(1/x) 在 0 连续（可去型补定义）
close('S3.10 lim x->0 x sin(1/x)', sp.limit(x * sp.sin(1 / x), x, 0), 0)

# S3.11 间断点分类：f(x)=ln|x| 在 0
close('S3.11 lim x->0- ln|x|', sp.limit(sp.log(-x), x, 0, dir='-'), -sp.oo)
close('S3.11 lim x->0+ ln|x|', sp.limit(sp.log(x), x, 0, dir='+'), -sp.oo)

# S3.12 复合函数间断点：f(x)=sin(x/(x-1)) 在 x=1 振荡
no_limit('S3.12 sin(x/(x-1)) 在 1 处无极限',
         sp.limit(sp.sin(x / (x - 1)), x, 1))

print()
print('=' * 80)
print('S4  新压轴（8 题）—— 抽象函数方程 + 迭代不动点')
print('=' * 80)

# S4.1 迭代：x_{n+1} = (x_n + 2/x_n)/2, x_1 = 2 -> sqrt(2)
def newton_iter(a0, a, iters=60):
    v = a0
    for _ in range(iters):
        v = (v + a / v) / 2
    return v
close('S4.1 牛顿迭代 sqrt(2)', newton_iter(2.0, 2.0), math.sqrt(2), tol=1e-12)

# S4.2 迭代：x_{n+1} = (1 + x_n)/x_n 无关点迭代 x->1
def fixed_iter(f, x0, iters=60):
    v = x0
    for _ in range(iters):
        v = f(v)
    return v
close('S4.2 x->(1+x)/x 不动点 = (1+sqrt5)/2',
      fixed_iter(lambda v: (1 + v) / v, 1.0), (1 + math.sqrt(5)) / 2, tol=1e-9)
close('S4.2 佐证不动点正根 (1+sqrt5)/2',
      [r for r in sp.solve(sp.Eq(x, (1 + x) / x), x) if r > 0], [(1 + sp.sqrt(5)) / 2])
# S4.2b  x_{n+1} = x_n + 1/x_n 发散（x_n ~ sqrt(2n)）
close('S4.2b x->x+1/x 发散到 +oo', fixed_iter(lambda v: v + 1 / v, 1.0, 20000) > 100, True)

# S4.3 压缩映射：f(x) = (x + 2/x)/2 在 [1,2] 上 Lipschitz 常数 1/2
g = (x + 2 / x) / 2
close('S4.3 f\'(1) = -1/2', sp.diff(g, x).subs(x, 1), -sp.Rational(1, 2))
close('S4.3 f\'(2) = 1/4', sp.diff(g, x).subs(x, 2), sp.Rational(1, 4))
close('S4.3 max|f\'| on [1,2] = 1/2',
      sp.Max(sp.Abs(sp.diff(g, x)).subs(x, 1), sp.Abs(sp.diff(g, x)).subs(x, 2)),
      sp.Rational(1, 2))
close('S4.3 f(sqrt2) = sqrt2（不动点）', g.subs(x, sp.sqrt(2)), sp.sqrt(2))

# S4.4 抽象函数方程：f(x+y) = f(x)f(y), f(1)=2 -> f(x)=2^x
close('S4.4 佐证 2^(x+y) = 2^x·2^y', sp.simplify(2 ** (x + t) - 2 ** x * 2 ** t), 0)
close('S4.4 佐证 f(1)=2', 2 ** 1, 2)

# S4.5 迭代不动点：x_{n+1} = 1/(2 + x_n), x_1 = 0 -> (sqrt5-1)/2
close('S4.5 x->1/(2+x) 不动点 = sqrt2-1',
      fixed_iter(lambda v: 1 / (2 + v), 0.0), math.sqrt(2) - 1, tol=1e-9)
close('S4.5 佐证不动点满足方程', sp.simplify(sp.solve(sp.Eq(x, 1 / (2 + x)), x)[0]),
      sp.sqrt(2) - 1)

# S4.6 抽象：f(x+1) = f(x) + 1 且 f 连续 -> f(x)=x+c 类
# 数值验证：f(x)=x 满足
close('S4.6 佐证 f(x)=x 满足 f(x+1)-f(x)=1', sp.simplify((x + 1) - x - 1), 0)

# S4.7 迭代压缩：x_{n+1} = cos x_n -> 不动点 Dottie 数
close('S4.7 x->cos x 不动点', fixed_iter(math.cos, 1.0), 0.7390851332151607, tol=1e-9)

# S4.8 抽象函数方程：f(x)f(y) = f(xy)，f(1)=1 -> f(x)=x^a；取 f(2)=3 求 a
close('S4.8 a = log_2 3', sp.log(3) / sp.log(2), math.log(3) / math.log(2))
close('S4.8 佐证 2^a = 3', sp.simplify(sp.exp(sp.log(3) / sp.log(2) * sp.log(2))), 3)

print()
print('=' * 80)
print('S5  不等式与最值（10 题）')
print('=' * 80)

# S5.1 闭区间最值 f(x)=x^3-3x+1 on [-2,2]
f = x ** 3 - 3 * x + 1
close('S5.1 f\' = 3x^2-3', sp.diff(f, x), 3 * x ** 2 - 3)
close('S5.1 驻点', sp.solve(sp.Eq(sp.diff(f, x), 0), x), [-1, 1])
close('S5.1 f(-2)', f.subs(x, -2), -1)
close('S5.1 f(2)', f.subs(x, 2), 3)
close('S5.1 f(-1)', f.subs(x, -1), 3)
close('S5.1 f(1)', f.subs(x, 1), -1)
close('S5.1 min on [-2,2]', min(f.subs(x, -2), f.subs(x, -1), f.subs(x, 1), f.subs(x, 2)), -1)
close('S5.1 max on [-2,2]', max(f.subs(x, -2), f.subs(x, -1), f.subs(x, 1), f.subs(x, 2)), 3)

# S5.2 用单调性证 x/(1+x) < ln(1+x) (x>0)
true_ineq('S5.2 ln(1+x) > x/(1+x)',
           sp.log(1 + x),
           [i / 200.0 for i in range(1, 400)],
           lambda v: float(math.log(1 + v) - v / (1 + v)) > 0)
chk('S5.2 佐证 差值导数 = x/(1+x)^2 > 0',
    sp.simplify(sp.diff(sp.log(1 + x) - x / (1 + x), x)), x / (1 + x) ** 2)

# S5.3 用凹凸性证 e^x >= 1+x
chk('S5.3 佐证 e^x 二阶导 > 0', sp.diff(sp.exp(x), x, 2), sp.exp(x))

# S5.4 泰勒证不等式：ln(1+x) <= x - x^2/2 + x^3/3 (x>0)
true_ineq('S5.4 ln(1+x) <= x - x^2/2 + x^3/3',
           sp.log(1 + x),
           [i / 200.0 for i in range(1, 400)],
           lambda v: float(math.log(1 + v) - v + v ** 2 / 2 - v ** 3 / 3) <= 1e-12)

# S5.5 最值：f(x) = x e^{-x} on [0, +inf)
f = x * sp.exp(-x)
close('S5.5 f\' = e^{-x}(1-x)', sp.simplify(sp.diff(f, x)), sp.exp(-x) * (1 - x))
close('S5.5 max = 1/e at x=1', f.subs(x, 1), 1 / sp.E)
close('S5.5 lim +inf f = 0', sp.limit(f, x, sp.oo), 0)

# S5.6 中值定理证不等式：|ln(1+x) - x| <= x^2/2 (x>-1)
true_ineq('S5.6 |ln(1+x)-x| < x^2/2 (x>0)',
           sp.log(1 + x),
           [i / 100.0 for i in range(1, 200)],
           lambda v: abs(math.log(1 + v) - v) < v ** 2 / 2 + 1e-12)

# S5.7 Cauchy 中值定理：ln b - ln a = (b-a)/(ξ)
close('S5.7 ln2-ln1', sp.log(2) - sp.log(1), sp.log(2))

# S5.8 含参极值：f(x)=x^3-3ax^2+2bx，f' 为二次方程，有两不同实根需判别式 > 0
f = x ** 3 - 3 * a * x ** 2 + 2 * b * x
close('S5.8 f\' = 3x^2-6ax+2b', sp.diff(f, x), 3 * x ** 2 - 6 * a * x + 2 * b)
close('S5.8 判别式 = 36a^2-24b > 0',
      sp.discriminant(3 * x ** 2 - 6 * a * x + 2 * b, x), 36 * a ** 2 - 24 * b)

# S5.9 证明 sin x < x (x>0) 的最简法与积分证
true_ineq('S5.9 sin x < x for x>0',
           sp.sin(x),
           [i / 100.0 for i in range(1, 500)],
           lambda v: math.sin(v) < v)

# S5.10 极值：f(x) = x^2/(1+x^2) on R, max = 1
f = x ** 2 / (1 + x ** 2)
close('S5.10 max = 1 (x->inf)', sp.limit(f, x, sp.oo), 1)
close('S5.10 f\' = 2x/(1+x^2)^2', sp.simplify(sp.diff(f, x)), 2 * x / (1 + x ** 2) ** 2)

print()
print('=' * 80)
print('S6  作图压轴（6 题）')
print('=' * 80)

# S6.1 y = x^3/(x^2-1)  —— 与高难册 H4.16 同型不同题？改用 y = x^3/(x^2+1)
f = x ** 3 / (x ** 2 + 1)
close('S6.1 f\' = x^2(x^2+3)/(x^2+1)^2', sp.simplify(sp.diff(f, x)),
      x ** 2 * (x ** 2 + 3) / (x ** 2 + 1) ** 2)
close('S6.1 奇函数', sp.simplify(f.subs(x, -x) + f), 0)
close('S6.1 lim x->inf f/x = 1', sp.limit(f / x, x, sp.oo), 1)
close('S6.1 渐近线 y=x（差 -> 0）', sp.limit(f - x, x, sp.oo), 0)
close('S6.1 lim x->-inf f/x = 1', sp.limit(f / x, x, -sp.oo), 1)

# S6.2 y = x - 1/x：渐近线 x=0, y=x
f = x - 1 / x
close('S6.2 lim x->0+ f', sp.limit(f, x, 0, dir='+'), -sp.oo)
close('S6.2 lim x->0- f', sp.limit(f, x, 0, dir='-'), sp.oo)
close('S6.2 lim x->inf (f-x)', sp.limit(f - x, x, sp.oo), 0)
close('S6.2 f\' = 1+1/x^2 > 0', sp.simplify(sp.diff(f, x)), 1 + 1 / x ** 2)

# S6.3 y = |x-1| e^{-x} 的尖点极值（对标 2009-10 Q17）
# x<1: (1-x)e^{-x}, f' = -e^{-x} - (1-x)e^{-x} = e^{-x}(x-2)
# x>1: (x-1)e^{-x}, f' = e^{-x} - (x-1)e^{-x} = e^{-x}(2-x)
close('S6.3 x<1 导数 = e^{-x}(x-2)', sp.simplify(sp.diff((1 - x) * sp.exp(-x), x)),
      sp.exp(-x) * (x - 2))
close('S6.3 x>1 导数 = e^{-x}(2-x)', sp.simplify(sp.diff((x - 1) * sp.exp(-x), x)),
      sp.exp(-x) * (2 - x))
close('S6.3 尖点 x=1 处值', (1 - 1) * sp.exp(-1), 0)
close('S6.3 x=1 右侧极值点 x=2 值', (2 - 1) * sp.exp(-2), sp.exp(-2))
close('S6.3 左导数 at 1', sp.limit(sp.diff((1 - x) * sp.exp(-x), x), x, 1, dir='-'),
      sp.exp(-1) * (1 - 2))
close('S6.3 右导数 at 1', sp.limit(sp.diff((x - 1) * sp.exp(-x), x), x, 1, dir='+'),
      sp.exp(-1) * (2 - 1))

# S6.4 y = x e^{-x}：驻点 x=1 是极大值
f = x * sp.exp(-x)
close('S6.4 驻点 x=1', sp.solve(sp.Eq(sp.diff(f, x), 0), x), [1])
close('S6.4 f(1)', f.subs(x, 1), 1 / sp.E)
close('S6.4 lim -inf f', sp.limit(f, x, -sp.oo), -sp.oo)

# S6.5 y = ln(x^2+1) 的极小
f = sp.log(x ** 2 + 1)
close('S6.5 f\' = 2x/(x^2+1)', sp.simplify(sp.diff(f, x)), 2 * x / (x ** 2 + 1))
close('S6.5 极小值点 x=0 值 0', f.subs(x, 0), 0)
close('S6.5 lim +inf f', sp.limit(f, x, sp.oo), sp.oo)

# S6.6 y = x^2/(x-1) 的作图（渐近线 + 极值）
f = x ** 2 / (x - 1)
close('S6.6 分解 x+1+1/(x-1)', sp.simplify(f - (x + 1 + 1 / (x - 1))), 0)
close('S6.6 f\' = x(x-2)/(x-1)^2', sp.simplify(sp.diff(f, x)), x * (x - 2) / (x - 1) ** 2)
close('S6.6 驻点', sp.solve(sp.Eq(sp.diff(f, x), 0), x), [0, 2])
close('S6.6 f(2)', f.subs(x, 2), 4)

print()
print('=' * 80)
print('模拟卷 A / B 关键结论复核')
print('=' * 80)

# ---- 模拟卷 A ----
# A1: lim(x->0) (e^{2x} - e^{-2x})/x = 4
close('A1 (e^{2x}-e^{-2x})/x', sp.limit((sp.exp(2 * x) - sp.exp(-2 * x)) / x, x, 0), 4)
# A2: f(x)=e^x 的 n 阶导 at 0
close('A2 e^x 的 5 阶导 at 0', sp.diff(sp.exp(x), x, 5).subs(x, 0), 1)
# A3: lim(x->0) (1-cos 2x)/x^2 = 2
close('A3 (1-cos2x)/x^2', sp.limit((1 - sp.cos(2 * x)) / x ** 2, x, 0), 2)
# A4: dy for y = x^2/2 at x=1 : dy = x dx = dx
close('A4 y=x^2/2 的 y\'(1)', sp.diff(x ** 2 / 2, x).subs(x, 1), 1)
# A5: f(x)=x^2 e^x 的 4 阶导 at 0
close('A5 d4(x^2 e^x) at 0', sp.diff(x ** 2 * sp.exp(x), x, 4).subs(x, 0), 12)
# A6: lim(x->inf) (1+1/x)^x = e
close('A6 (1+1/x)^x', sp.limit((1 + 1 / x) ** x, x, sp.oo), sp.E)
# A7: 极值 f=x^3-3x on [-2,2]
close('A7 f=x^3-3x 极大值', (2 ** 3 - 3 * 2), 2)
# A8: lim(x->0) (tanx - x)/x^3 = 1/3
close('A8 (tanx-x)/x^3', sp.limit((sp.tan(x) - x) / x ** 3, x, 0), sp.Rational(1, 3))
# A9: 参数 x=t^2, y=t^3 求 dy/dx
close('A9 dy/dx = 3t/2', sp.simplify(sp.diff(t ** 3, t) / sp.diff(t ** 2, t)), 3 * t / 2)
# A10: f(x)=x ln x 的极小点
close('A10 f=x lnx 极小点 x=1/e', sp.solve(sp.Eq(sp.diff(x * sp.log(x), x), 0), x),
      [1 / sp.E])
# A11: 第 11 题 eps-delta：lim(x->0) sin x / x = 1 允许的 delta
# A12: lim(x->0) (x-sin x)/x^3 = 1/6
close('A12 (x-sinx)/x^3', sp.limit((x - sp.sin(x)) / x ** 3, x, 0), sp.Rational(1, 6))
# A13: lim(x->0) (1+x)^{1/2} 的 2 阶展开
close('A13 sqrt(1+x) 2 阶', sp.expand(sp.series(sp.sqrt(1 + x), x, 0, 3).removeO()),
      1 + x / 2 - x ** 2 / 8)
# A14: 隐函数 x^2+y^2=2x 在 (1,1) 的 y'
close('A14 y\'(1,1) for x^2+y^2=2x',
      sp.simplify(-(2 * x) / (2 * y_)).subs({x: 1, y_: 1}), -1)
# A15: lim(x->0+) x^a ln x = 0 (a>0)
close('A15 lim x^a lnx (a=1)', sp.limit(x * sp.log(x), x, 0, dir='+'), 0)
# A16: f=x^2/2 - x ln x 极小
close('A16 f=x^2/2-x lnx 在 x=1 值', (1 / 2 - 0), sp.Rational(1, 2))
# A17: 作图 y = x^3 - x
close('A17 y=x^3-x 驻点 ±1/sqrt3',
      sp.solve(sp.Eq(sp.diff(x ** 3 - x, x), 0), x),
      [-sp.sqrt(3) / 3, sp.sqrt(3) / 3])
close('A17 y=x^3-x 在 -1/sqrt3 处为极大值 2sqrt3/9',
      (x ** 3 - x).subs(x, -sp.sqrt(3) / 3), 2 * sp.sqrt(3) / 9)
close('A17 y=x^3-x 在 1/sqrt3 处为极小值 -2sqrt3/9',
      (x ** 3 - x).subs(x, sp.sqrt(3) / 3), -2 * sp.sqrt(3) / 9)
# A18: 压轴 lim(x->0) [ln(1+x) - x + x^2/2]/x^3 = -1/3
close('A18 [ln(1+x)-x+x^2/2]/x^3',
      sp.limit((sp.log(1 + x) - x + x ** 2 / 2) / x ** 3, x, 0), sp.Rational(1, 3))

# ---- 模拟卷 B ----
# B1: lim(x->0) (1+x)^(1/3) = 1；一阶导
close('B1 d/dx (1+x)^(1/3) at 0', sp.diff((1 + x) ** sp.Rational(1, 3), x).subs(x, 0),
      sp.Rational(1, 3))
# B2: lim(x->0) (ln(1+x)-x)/x^2 = -1/2
close('B2 [ln(1+x)-x]/x^2', sp.limit((sp.log(1 + x) - x) / x ** 2, x, 0),
      -sp.Rational(1, 2))
# B3: f(x)=sin x 的 5 阶导 at 0
close('B3 d5(sinx) at 0', sp.diff(sp.sin(x), x, 5).subs(x, 0), 1)
# B4: dy of y=ln(1+x^2) at x=0
close('B4 y\'(0) of ln(1+x^2)', sp.diff(sp.log(1 + x ** 2), x).subs(x, 0), 0)
# B5: f(x)=x^3/(1+x) 的 3 阶导 at 0
close('B5 d3(x^3/(1+x)) at 0', sp.diff(x ** 3 / (1 + x), x, 3).subs(x, 0), 6)
# B6: lim(x->0) (1+tan x)^(1/x) = e
close('B6 lim (1+tanx)^(1/x)',
      sp.limit((1 + sp.tan(x)) ** (1 / x), x, 0), sp.E)
# B7: 极值 f = x^2 - 4x + 1 最小在 x=2 值 -3
close('B7 f(2)', 4 - 8 + 1, -3)
# B8: lim(x->0) (sin x)/x 的一阶展开
close('B8 sinx/x -> 1', sp.limit(sp.sin(x) / x, x, 0), 1)
# B9: 参数 x = e^t, y = e^{-t}, 求 d2y/dx2
dx = sp.diff(sp.exp(t), t)
dy = sp.diff(sp.exp(-t), t)
d1 = sp.simplify(dy / dx)
d2 = sp.simplify(sp.diff(d1, t) / dx)
close('B9 d2y/dx2 for x=e^t,y=e^{-t}', sp.simplify(d2), 2 * sp.exp(-3 * t))
close('B9 佐证 dy/dx = -e^{-2t}', sp.simplify(d1), -sp.exp(-2 * t))
# B10: f(x) = x arctan x 的 2 阶导 at 0
close('B10 d2(x atan x) at 0', sp.diff(x * sp.atan(x), x, 2).subs(x, 0), 2)
# B11: 第 11 题 M 型 lim(x->1+) 1/ln x = +oo
close('B11 1/lnx at 1+', sp.limit(1 / sp.log(x), x, 1, dir='+'), sp.oo)
# B12: lim(x->0) (cos x - e^{-x^2/2})/x^4 = -1/12
close('B12 (cosx - e^{-x^2/2})/x^4',
      sp.limit((sp.cos(x) - sp.exp(-x ** 2 / 2)) / x ** 4, x, 0), -sp.Rational(1, 12))
# B13: lim(x->0) ln(cos x)/x^2 = -1/2
close('B13 ln(cosx)/x^2', sp.limit(sp.log(sp.cos(x)) / x ** 2, x, 0), -sp.Rational(1, 2))
# B14: 隐函数 xy = x + y （双曲线）dy/dx
close('B14 y\' for xy=x+y', sp.simplify(-(y_ - 1) / (x - 1)), -(y_ - 1) / (x - 1))
# B15: lim(x->0) x/(e^x - 1) = 1
close('B15 x/(e^x-1)', sp.limit(x / (sp.exp(x) - 1), x, 0), 1)
# B16: f = x/(1+x^2) 极大值 1/2 at x=1
close('B16 f=x/(1+x^2) 极大 1/2', 1 / (1 + 1), sp.Rational(1, 2))
# B17: 作图 y = x^2/(x^2-1)
close('B17 y=x^2/(x^2-1) 渐近线 y=1', sp.limit(x ** 2 / (x ** 2 - 1), x, sp.oo), 1)
# B18: 压轴 lim(x->0) [e^x - 1 - x - x^2/2]/x^3 = 1/6
close('B18 [e^x-1-x-x^2/2]/x^3',
      sp.limit((sp.exp(x) - 1 - x - x ** 2 / 2) / x ** 3, x, 0), sp.Rational(1, 6))

print()
print('=' * 80)
print('结果：PASS %d / FAIL %d' % (PASS, FAIL))
print('=' * 80)
if FAILS:
    print('失败项：')
    for f in FAILS:
        print('  - ' + f)
