# -*- coding: utf-8 -*-
"""复核「考前知识点清单」中所有公式表：等价无穷小、麦克劳林系数、参数方程二阶导、极坐标切线斜率。
全部 PASS 才允许写进讲义。"""
import sympy as sp

x, t, th = sp.symbols('x t theta', real=True)
PASS = 0
FAIL = 0


def ck(name, got, exp, tol=1e-9):
    global PASS, FAIL
    try:
        ok = abs(float(got) - float(exp)) <= tol
    except Exception:
        ok = sp.simplify(got - exp) == 0
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print('FAIL', name, got, exp)


def lim1(expr):
    return sp.limit(expr, x, 0)


print('== 1. 等价无穷小（比值极限应为 1）==')
eqs = [
    (sp.sin(x), x),
    (sp.tan(x), x),
    (sp.asin(x), x),
    (sp.atan(x), x),
    (1 - sp.cos(x), x**2 / 2),
    (sp.log(1 + x), x),
    (sp.exp(x) - 1, x),
    ((1 + x)**sp.Rational(1, 2) - 1, x / 2),
    ((1 + x)**sp.Rational(1, 3) - 1, x / 3),
    (2**x - 1, x * sp.log(2)),
    (x - sp.sin(x), x**3 / 6),
    (sp.tan(x) - x, x**3 / 3),
    (x - sp.log(1 + x), x**2 / 2),
    (sp.asin(x) - x, x**3 / 6),
    (sp.atan(x) - x, -x**3 / 3),
    (sp.tan(x) - sp.sin(x), x**3 / 2),
    (sp.exp(x) - 1 - x, x**2 / 2),
]
for i, (f, g) in enumerate(eqs):
    ck('eq%d %s' % (i, f), lim1(f / g), 1)

print('== 2. 麦克劳林系数 ==')
series = {
    sp.exp(x): [1, 1, sp.Rational(1, 2), sp.Rational(1, 6), sp.Rational(1, 24)],
    sp.sin(x): [0, 1, 0, sp.Rational(-1, 6), 0, sp.Rational(1, 120)],
    sp.cos(x): [1, 0, sp.Rational(-1, 2), 0, sp.Rational(1, 24)],
    sp.log(1 + x): [0, 1, sp.Rational(-1, 2), sp.Rational(1, 3), sp.Rational(-1, 4)],
    sp.atan(x): [0, 1, 0, sp.Rational(-1, 3), 0, sp.Rational(1, 5)],
    sp.tan(x): [0, 1, 0, sp.Rational(1, 3), 0, sp.Rational(2, 15)],
    sp.sqrt(1 + x): [1, sp.Rational(1, 2), sp.Rational(-1, 8), sp.Rational(1, 16), sp.Rational(-5, 128)],
    (1 + x)**sp.Rational(-1, 2): [1, sp.Rational(-1, 2), sp.Rational(3, 8), sp.Rational(-5, 16)],
}
for f, coefs in series.items():
    s = sp.series(f, x, 0, len(coefs)).removeO()
    for n, c in enumerate(coefs):
        ck('%s x^%d' % (f, n), sp.expand(s).coeff(x, n), c)

# (1+x)^alpha 一般系数
a = sp.symbols('alpha')
s = sp.series((1 + x)**a, x, 0, 4).removeO()
ck('binom a1', sp.expand(s).coeff(x, 1), a)
ck('binom a2', sp.expand(s).coeff(x, 2), a * (a - 1) / 2)
ck('binom a3', sp.expand(s).coeff(x, 3), a * (a - 1) * (a - 2) / 6)

print('== 3. 参数方程二阶导公式 ==')
xt = t - sp.sin(t)
yt = 1 - sp.cos(t)
dydx = sp.diff(yt, t) / sp.diff(xt, t)
d2_formula = sp.diff(yt, t, 2) * sp.diff(xt, t) - sp.diff(yt, t) * sp.diff(xt, t, 2)
d2_formula = d2_formula / sp.diff(xt, t)**3
d2_direct = sp.diff(dydx.subs(t, sp.Function('u')(x)) if False else dydx, x)  # 占位
# 用参数法直接验：d/dx(dy/dx) = (d/dt dy/dx)/(dx/dt)
d2_check = sp.simplify(sp.diff(dydx, t) / sp.diff(xt, t) - d2_formula)
ck('param d2', d2_check, 0)

print('== 4. 极坐标切线斜率 ==')
r = sp.Function('r')(th)
yP = r * sp.sin(th)
xP = r * sp.cos(th)
slope_param = sp.simplify(sp.diff(yP, th) / sp.diff(xP, th))
slope_table = (sp.diff(r, th) * sp.sin(th) + r * sp.cos(th)) / (sp.diff(r, th) * sp.cos(th) - r * sp.sin(th))
ck('polar slope', sp.simplify(slope_param - slope_table), 0)
# 具体例：r = 1 + cos(theta)，theta = pi/2 处斜率为 -1
r0 = 1 + sp.cos(th)
sl = slope_table.subs(r, r0).doit().subs(th, sp.pi / 2)
ck('polar ex r=1+cos@pi/2', sl, 1)
# 心形线 theta=pi/3（分子恰好为 0，切线水平）
sl2 = slope_table.subs(r, r0).doit().subs(th, sp.pi / 3)
ck('polar ex r=1+cos@pi/3', sl2, 0)

print('== 5. 反函数 / 对数求导 ==')
ck('arctan derivative', sp.diff(sp.atan(x), x).subs(x, 1), sp.Rational(1, 2))
ck('arcsin derivative', sp.diff(sp.asin(x), x).subs(x, sp.Rational(1, 2)), 2 / sp.sqrt(3))
ck('tan derivative', sp.diff(sp.tan(x), x).subs(x, sp.pi / 4), 2)

print('== 6. Leibniz 例：(x^2 e^x)^(n) ==')
n = sp.symbols('n', positive=True, integer=True)
for k in (1, 2, 3, 5):
    lhs = sp.diff(x**2 * sp.exp(x), x, k)
    rhs = sp.exp(x) * (x**2 + 2 * k * x + k * (k - 1))
    ck('leibniz n=%d' % k, sp.simplify(lhs - rhs).subs(x, 0), 0)

print()
print('PASS', PASS, 'FAIL', FAIL)
if FAIL:
    raise SystemExit(1)
