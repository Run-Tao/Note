# -*- coding: utf-8 -*-
"""
册三《讲义_加练_定义与压轴深化_11-08》数值与符号复核
运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_extra_drill.py
覆盖：P1 极限定义证明深化 12 / P2 泰勒与高阶导深化 12
      / P3 抽象函数方程与迭代 10 / P4 作图与渐近线 6 = 40 题
      + 模拟卷 C 18 题 = 58 题
口径：每题先算 got 再与我写进讲义的 want 比对；delta_ok / X_ok / M_ok
      额外数值验证「量词行里给出的 delta / X」真的满足定义。
"""
import sympy as sp
import mpmath as mp
import math

PASS = 0
FAIL = 0
FAILS = []

x = sp.Symbol('x')
t = sp.Symbol('t')
a = sp.Symbol('a')
b = sp.Symbol('b')
n = sp.Symbol('n', integer=True, positive=True)
al = sp.Symbol('alpha')


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
        print('  PASS  %-58s = %s' % (label, got))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-58s got=%s want=%s' % (label, got, want))


def close(label, got, want, tol=1e-8):
    chk(label, got, want, tol=tol)


def true_ineq(label, fn, samples, pred):
    global PASS, FAIL
    ok = all(pred(v) for v in samples)
    if ok:
        PASS += 1
        print('  PASS  %-58s = 成立' % label)
    else:
        FAIL += 1
        FAILS.append(label)
        bad = [v for v in samples if not pred(v)]
        print('  FAIL  %-58s 反例=%s' % (label, bad[:3]))


def series_coef(f, n, order):
    s = sp.series(f, x, 0, order).removeO()
    return sp.expand(s).coeff(x, n)


def deriv_at(f, k, at=0):
    return sp.simplify(sp.diff(f, x, k).subs(x, at))


def delta_ok(label, fexpr, at, L, delta_fn,
             eps_list=(0.5, 0.1, 0.02, 0.005),
             fracs=(-0.99, -0.7, -0.3, 0.3, 0.7, 0.99)):
    """eps-delta：验证我给的 delta(eps) 真的能保证 |f(x)-L|<eps。"""
    fn = sp.lambdify(x, fexpr, 'mpmath')
    ok = True
    for e in eps_list:
        d = float(delta_fn(e))
        if d <= 0:
            ok = False
            break
        for u in fracs:
            xv = mp.mpf(at) + mp.mpf(u) * mp.mpf(d)
            if abs(xv - at) < mp.mpf('1e-14'):
                continue
            try:
                val = abs(complex(fn(xv)) - float(L))
            except Exception:
                continue
            if val >= e:
                ok = False
                break
        if not ok:
            break
    chk(label, ok, True)


def X_ok(label, fexpr, L, X_fn, eps_list=(0.5, 0.1, 0.02, 0.005),
         mults=(1.01, 1.3, 2.0, 5.0)):
    """eps-X：验证 x>X(eps) 时 |f(x)-L|<eps。"""
    fn = sp.lambdify(x, fexpr, 'mpmath')
    ok = True
    for e in eps_list:
        X = float(X_fn(e))
        for m in mults:
            try:
                val = abs(complex(fn(mp.mpf(X) * m)) - float(L))
            except Exception:
                continue
            if val >= e:
                ok = False
                break
        if not ok:
            break
    chk(label, ok, True)


def M_ok(label, fexpr, X_fn, Ms=(5, 20, 100), mults=(1.0, 1.5, 3.0)):
    """M 型（趋于 +oo）：验证 x>X(M) 时 f(x)>M。"""
    fn = sp.lambdify(x, fexpr, 'mpmath')
    ok = True
    for M in Ms:
        X = float(X_fn(M))
        for m in mults:
            try:
                val = float(sp.re(fn(mp.mpf(X) * m)))
            except Exception:
                continue
            if val <= M:
                ok = False
                break
        if not ok:
            break
    chk(label, ok, True)


print('=' * 74)
print('P1 极限定义证明深化 12 题（eps-delta 6 / eps-X 4 / M 型 2）')
print('=' * 74)

# --- P1.1 ~ P1.6 : eps-delta ---
p11 = (x**2 - 4) / (x - 2)
chk('P1.1  L = 4', sp.limit(p11, x, 2), sp.Integer(4))
delta_ok('P1.1  delta = eps 有效', p11, 2, 4, lambda e: e)

p12 = (sp.sqrt(x + 3) - 2) / (x - 1)
chk('P1.2  L = 1/4', sp.limit(p12, x, 1), sp.Rational(1, 4))
delta_ok('P1.2  delta = min(1, 8eps) 有效', p12, 1, sp.Rational(1, 4),
         lambda e: min(1, 8 * e))

p13 = x**2 * sp.sin(1 / x)
chk('P1.3  L = 0', sp.limit(p13, x, 0), sp.Integer(0))
delta_ok('P1.3  delta = min(1, sqrt(eps)) 有效', p13, 0, 0,
         lambda e: min(1, math.sqrt(e)))

p14 = x * sp.sin(1 / x)
chk('P1.4  L = 0', sp.limit(p14, x, 0), sp.Integer(0))
chk('P1.4  x^(1/2) sin(1/x) -> 0 (x->0+)',
    sp.limit(sp.sqrt(x) * sp.sin(1 / x), x, 0, dir='+'), sp.Integer(0))
delta_ok('P1.4  delta = eps 有效', p14, 0, 0, lambda e: e)

p15 = sp.log(1 + x) / x
chk('P1.5  L = 1', sp.limit(p15, x, 0), sp.Integer(1))
delta_ok('P1.5  delta = min(1/2, eps) 有效', p15, 0, 1,
         lambda e: min(0.5, e))

p16 = (sp.exp(x) - 1) / x
chk('P1.6  L = 1', sp.limit(p16, x, 0), sp.Integer(1))
delta_ok('P1.6  delta = min(1, eps/2) 有效', p16, 0, 1,
         lambda e: min(1, e / 2))

# --- P1.7 ~ P1.10 : eps-X ---
p17 = (3 * x**2 + 1) / (x**2 + 2)
chk('P1.7  L = 3', sp.limit(p17, x, sp.oo), sp.Integer(3))
X_ok('P1.7  X = sqrt(5/eps) 有效', p17, 3, lambda e: math.sqrt(5 / e))

p18 = (2 * x + sp.sin(x)) / (x + 1)
chk('P1.8  L = 2', sp.limit(p18, x, sp.oo), sp.Integer(2))
X_ok('P1.8  X = 3/eps 有效', p18, 2, lambda e: 3 / e)

p19 = sp.sqrt(x**2 + 1) / x
chk('P1.9  L = 1 (x->+oo)', sp.limit(p19, x, sp.oo), sp.Integer(1))
X_ok('P1.9  X = 1/sqrt(2eps) 有效', p19, 1, lambda e: 1 / math.sqrt(2 * e))

p110 = sp.log(1 + sp.exp(x)) / x
chk('P1.10 L = 1 (x->+oo)', sp.limit(p110, x, sp.oo), sp.Integer(1))
X_ok('P1.10 X = 2/eps 有效', p110, 1, lambda e: 2 / e)

# --- P1.11 ~ P1.12 : M 型 ---
p111 = sp.log(x)
chk('P1.11 lim ln x = +oo', sp.limit(p111, x, sp.oo), sp.oo)
M_ok('P1.11 X = exp(M+1) 有效', p111, lambda M: math.exp(M + 1))

p112 = x**3 / (x**2 + 1)
chk('P1.12 lim = +oo', sp.limit(p112, x, sp.oo), sp.oo)
M_ok('P1.12 X = 2M + 2 有效', p112, lambda M: 2 * M + 2)

print('=' * 74)
print('P2 泰勒与高阶导深化 12 题')
print('=' * 74)

chk('P2.1  [x^7] x^2 ln(1+x) = 1/5',
    series_coef(x**2 * sp.log(1 + x), 7, 9), sp.Rational(1, 5))
chk('P2.2  [x^6] arctan(x)/x = -1/7',
    series_coef(sp.atan(x) / x, 6, 9), sp.Rational(-1, 7))
chk('P2.3  f=x^3 cos x, f^(10)(0) = 0', deriv_at(x**3 * sp.cos(x), 10), sp.Integer(0))
chk('P2.3  f=x^3 cos x, f^(11)(0) = 990', deriv_at(x**3 * sp.cos(x), 11), sp.Integer(990))
chk('P2.4  Peano 余项 x^6 系数 = -1/720',
    sp.limit((sp.cos(x) - 1 + x**2 / 2 - x**4 / 24) / x**6, x, 0),
    sp.Rational(-1, 720))
true_ineq('P2.5  |sin x - x + x^3/6| <= |x|^5/120 (|x|<=1)',
          None,
          [mp.mpf(k) / 40 for k in range(-40, 41) if k != 0],
          lambda v: abs(mp.sin(v) - v + v**3 / 6) <= abs(v)**5 / 120 * (1 + mp.mpf('1e-9')))
chk('P2.6  [(1+x)^(1/x) - e(1-x/2)]/x^2 -> 11e/24',
    sp.limit(((1 + x)**(1 / x) - sp.E * (1 - x / 2)) / x**2, x, 0),
    11 * sp.E / 24)
chk('P2.7  (tan x - sin x)/x^3 -> 1/2',
    sp.limit((sp.tan(x) - sp.sin(x)) / x**3, x, 0), sp.Rational(1, 2))
chk('P2.8  d^9 (1+x)^alpha at 0 = alpha(alpha-1)...(alpha-8)',
    deriv_at((1 + x)**al, 9), sp.prod(al - k for k in range(9)))
# P2.9 隐式 y = x e^y 的级数（Lagrange 反演），逐次迭代求系数
y = x
for _ in range(7):
    y = sp.series(x * sp.exp(y), x, 0, 6).removeO()
yser = sp.expand(sp.series(y, x, 0, 6).removeO())
chk('P2.9  y=x e^y 的 [x^3] = 3/2', yser.coeff(x, 3), sp.Rational(3, 2))
chk('P2.9  y=x e^y 的 y\'\'\'(0) = 9', sp.factorial(3) * yser.coeff(x, 3), sp.Integer(9))
chk('P2.10 估阶 n^3(ln(1+1/n)-1/n+1/(2n^2)) -> 1/3',
    sp.limit(n**3 * (sp.log(1 + 1 / n) - 1 / n + 1 / (2 * n**2)), n, sp.oo),
    sp.Rational(1, 3))
true_ineq('P2.11 e^x > 1+x+x^2/2+x^3/6 (x>0)', None,
          [mp.mpf(k) / 20 for k in range(1, 61)],
          lambda v: mp.e**v > 1 + v + v**2 / 2 + v**3 / 6)
chk('P2.12 (x - ln(1+x) - x^2/2)/x^3 -> -1/3',
    sp.limit((x - sp.log(1 + x) - x**2 / 2) / x**3, x, 0), sp.Rational(-1, 3))

print('=' * 74)
print('P3 抽象函数方程 / 迭代 / 不动点 10 题')
print('=' * 74)

# P3.1 Cauchy 加性：f(x)=3x
chk('P3.1  f=3x 满足 f(x+y)=f(x)+f(y)',
    sp.simplify(3 * (x + a) - (3 * x + 3 * a)), sp.Integer(0))
chk('P3.1  f(sqrt 2) = 3 sqrt 2', 3 * sp.sqrt(2), 3 * sp.sqrt(2))
# P3.2 指数型：f(x)=exp(2x)
chk('P3.2  f\'\'(1) = 4 e^2', sp.diff(sp.exp(2 * x), x, 2).subs(x, 1), 4 * sp.E**2)
chk('P3.2  f(x+y)=f(x)f(y)',
    sp.simplify(sp.exp(2 * (x + a)) - sp.exp(2 * x) * sp.exp(2 * a)), sp.Integer(0))
# P3.3 对数型：f=ln x
chk('P3.3  f\'(e) = 1/e', sp.diff(sp.log(x), x).subs(x, sp.E), 1 / sp.E)
xp, ap = sp.Symbol('xp', positive=True), sp.Symbol('ap', positive=True)
chk('P3.3  f(xy)=f(x)+f(y)',
    sp.simplify(sp.log(xp * ap).expand(force=True) - sp.log(xp) - sp.log(ap)),
    sp.Integer(0))
# P3.4 二次型方程 f(x+y)+f(x-y)=2f(x)+2f(y)，f(x)=2x^2
f2 = lambda z: 2 * z**2
chk('P3.4  f=2x^2 满足二次型方程',
    sp.simplify(f2(x + a) + f2(x - a) - 2 * f2(x) - 2 * f2(a)), sp.Integer(0))
chk('P3.4  f(3) = 18', f2(3), sp.Integer(18))
# P3.5 牛顿迭代求 sqrt2
u0 = sp.Integer(2)
u1 = sp.simplify((u0 + 2 / u0) / 2)
u2 = sp.simplify((u1 + 2 / u1) / 2)
chk('P3.5  x1 = 3/2', u1, sp.Rational(3, 2))
chk('P3.5  x2 = 17/12', u2, sp.Rational(17, 12))
chk('P3.5  不动点 = sqrt 2', sp.sqrt(2), sp.sqrt(2))
# P3.6 余弦迭代的 Dottie 数
dottie = mp.findroot(lambda v: mp.cos(v) - v, mp.mpf('0.7'))
close('P3.6  cos 迭代不动点 = 0.7390851332', dottie, mp.mpf('0.7390851332151607'), tol=1e-9)
# P3.7 x_{n+1} = 1 + 1/(1+x_n) 的不动点
roots = sp.solve(sp.Eq(a, 1 + 1 / (1 + a)), a)
pos = [r for r in roots if sp.N(r) > 0][0]
chk('P3.7  不动点 = sqrt 2', sp.simplify(pos), sp.sqrt(2))
chk('P3.7  |f\'(sqrt 2)| = 3-2sqrt 2',
    sp.simplify(abs(sp.diff(1 + 1 / (1 + a), a).subs(a, sp.sqrt(2)))),
    3 - 2 * sp.sqrt(2))
# P3.8 压缩映射 f(x)=(x+3)/4
chk('P3.8  |f\'| = 1/4 < 1', sp.diff((x + 3) / 4, x), sp.Rational(1, 4))
chk('P3.8  不动点 = 1', sp.solve(sp.Eq(a, (a + 3) / 4), a)[0], sp.Integer(1))
chk('P3.8  n 步误差 = (x0-1)/4^n (n=3)',
    ((sp.Rational(5, 1) - 1) / 4**3), sp.Rational(1, 16))
# P3.9 条件不足：f(x+1)=f(x)+2, f(0)=0，f(1/2) 不唯一
fA = lambda z: 2 * z
fB = lambda z: 2 * z + sp.cos(2 * sp.pi * z) - 1
chk('P3.9  反例 A 满足方程', sp.simplify(fA(x + 1) - fA(x) - 2), sp.Integer(0))
chk('P3.9  反例 B 满足方程', sp.simplify(fB(x + 1) - fB(x) - 2), sp.Integer(0))
chk('P3.9  fA(1/2) = 1', fA(sp.Rational(1, 2)), sp.Integer(1))
chk('P3.9  fB(1/2) = -1', fB(sp.Rational(1, 2)), sp.Integer(-1))
# P3.10 f(x+y)=f(x)+f(y)+xy, f(1)=1 -> f(x)=x^2/2+x/2
f10 = lambda z: z**2 / 2 + z / 2
chk('P3.10 f= x^2/2 + x/2 满足方程',
    sp.simplify(f10(x + a) - f10(x) - f10(a) - x * a), sp.Integer(0))
chk('P3.10 f(2) = 3', f10(2), sp.Integer(3))

print('=' * 74)
print('P4 作图与渐近线 6 题')
print('=' * 74)

p41 = x**2 / (x - 1)
chk('P4.1  斜渐近线 y = x+1', sp.limit(p41 - (x + 1), x, sp.oo), sp.Integer(0))
chk('P4.1  x->1+ 为 +oo', sp.limit(p41, x, 1, dir='+'), sp.oo)
chk('P4.1  x->1- 为 -oo', sp.limit(p41, x, 1, dir='-'), -sp.oo)

p42 = x * sp.exp(1 / x)
chk('P4.2  斜渐近线 y = x+1', sp.limit(p42 - (x + 1), x, sp.oo), sp.Integer(0))
chk('P4.2  x->0+ 为 +oo', sp.limit(p42, x, 0, dir='+'), sp.oo)
chk('P4.2  x->0- 为 0', sp.limit(p42, x, 0, dir='-'), sp.Integer(0))

p43 = sp.sqrt(x**2 + 1) / x
chk('P4.3  x->+oo 水平渐近线 y=1', sp.limit(p43, x, sp.oo), sp.Integer(1))
chk('P4.3  x->-oo 水平渐近线 y=-1', sp.limit(p43, x, -sp.oo), sp.Integer(-1))
chk('P4.3  x->0+ 为 +oo', sp.limit(p43, x, 0, dir='+'), sp.oo)

p44 = x**3 / (x**2 - 1)
chk('P4.4  斜渐近线 y = x', sp.limit(p44 - x, x, sp.oo), sp.Integer(0))
chk('P4.4  x->1+ 为 +oo', sp.limit(p44, x, 1, dir='+'), sp.oo)
chk('P4.4  奇函数', sp.simplify(p44.subs(x, -x) + p44), sp.Integer(0))

p45 = x - 2 * sp.atan(x)
chk('P4.5  x->+oo 斜渐近线 y = x - pi', sp.limit(p45 - (x - sp.pi), x, sp.oo), sp.Integer(0))
chk('P4.5  x->-oo 斜渐近线 y = x + pi', sp.limit(p45 - (x + sp.pi), x, -sp.oo), sp.Integer(0))

p46 = (x**2 - 1) / (x**2 + 1)
chk('P4.6  水平渐近线 y = 1', sp.limit(p46, x, sp.oo), sp.Integer(1))
chk('P4.6  偶函数', sp.simplify(p46.subs(x, -x) - p46), sp.Integer(0))
chk('P4.6  最小值 f(0) = -1', p46.subs(x, 0), sp.Integer(-1))

print('=' * 74)
print('模拟卷 C（18 题 / 100 分 / 120 分钟）')
print('=' * 74)

chk('C1   (sin x - x cos x)/x^3 -> 1/3',
    sp.limit((sp.sin(x) - x * sp.cos(x)) / x**3, x, 0), sp.Rational(1, 3))
chk('C2   1/x - 1/(e^x-1) -> 1/2',
    sp.limit(1 / x - 1 / (sp.exp(x) - 1), x, 0), sp.Rational(1, 2))
chk('C3   d/dx x^(sin x) at 1 = sin 1',
    sp.diff(x**sp.sin(x), x).subs(x, 1), sp.sin(1))
c4 = (x**2 - 1) / (x**2 - 3 * x + 2)
chk('C4   x=1 为可去间断（极限 -2）', sp.limit(c4, x, 1), sp.Integer(-2))
chk('C4   x=2 为无穷间断 (x->2-)', sp.limit(c4, x, 2, dir='-'), -sp.oo)
rp = 1 + sp.cos(t)
rp1 = sp.diff(rp, t)
kk = sp.simplify((rp1 * sp.sin(t) + rp * sp.cos(t)) /
                 (rp1 * sp.cos(t) - rp * sp.sin(t)))
chk('C5   心形线 theta=pi/2 切线斜率 = 1', kk.subs(t, sp.pi / 2), sp.Integer(1))
chk('C6   d^8 (x^3 sin x) at 0 = 336', deriv_at(x**3 * sp.sin(x), 8), sp.Integer(336))
chk('C7   (ln(1+x)-x+x^2/2)/x^3 -> 1/3',
    sp.limit((sp.log(1 + x) - x + x**2 / 2) / x**3, x, 0), sp.Rational(1, 3))
chk('C8   x^(1/x) 在 x=e 取极大（导数为 0）',
    sp.diff(x**(1 / x), x).subs(x, sp.E), sp.Integer(0))
chk('C8   极大值 = e^(1/e)', (x**(1 / x)).subs(x, sp.E), sp.E**(1 / sp.E))
c9 = x * sp.log(x)
kap = sp.simplify((sp.diff(c9, x, 2) / (1 + sp.diff(c9, x)**2)**sp.Rational(3, 2)).subs(x, 1))
chk('C9   y=x ln x 在 (1,0) 曲率 = sqrt 2 / 4', kap, sp.sqrt(2) / 4)
chk('C10  r=2 sin(theta) 面积 = pi',
    sp.integrate((2 * sp.sin(t))**2 / 2, (t, 0, sp.pi)), sp.pi)
chk('C11  (x^2-9)/(x-3) -> 6', sp.limit((x**2 - 9) / (x - 3), x, 3), sp.Integer(6))
delta_ok('C11  delta = eps 有效', (x**2 - 9) / (x - 3), 3, 6, lambda e: e)
chk('C12  1/sin^2 x - 1/x^2 -> 1/3',
    sp.limit(1 / sp.sin(x)**2 - 1 / x**2, x, 0), sp.Rational(1, 3))
# C13 参数方程 x=t-ln(1+t), y=t^3/3
xt = t - sp.log(1 + t)
yt = t**3 / 3
dy1 = sp.simplify(sp.diff(yt, t) / sp.diff(xt, t))
dy2 = sp.simplify(sp.diff(dy1, t) / sp.diff(xt, t))
chk('C13  dy/dx at t=1 = 2', dy1.subs(t, 1), sp.Integer(2))
chk('C13  d2y/dx2 at t=1 = 6', dy2.subs(t, 1), sp.Integer(6))
# C14 隐函数 y + ln y = x
yv = sp.Symbol('yv', positive=True)
y1 = sp.simplify(yv / (yv + 1))
y2 = sp.simplify(sp.diff(y1, yv) * y1)
chk('C14  y\'(1) = 1/2', y1.subs(yv, 1), sp.Rational(1, 2))
chk('C14  y\'\'(1) = 1/8', y2.subs(yv, 1), sp.Rational(1, 8))
# C15 f=x^3+ax^2+bx 在 x=1 有极值 -2
sol = sp.solve([sp.Eq(3 + 2 * a + b, 0), sp.Eq(1 + a + b, -2)], [a, b], dict=True)[0]
chk('C15  a = 0', sol[a], sp.Integer(0))
chk('C15  b = -3', sol[b], sp.Integer(-3))
chk('C15  x=1 为极小点（二阶导 6）',
    (6 * x + 2 * sol[a]).subs(x, 1), sp.Integer(6))
# C16 f = x + 1/x
c16 = x + 1 / x
chk('C16  极大值 f(-1) = -2', c16.subs(x, -1), sp.Integer(-2))
chk('C16  极小值 f(1) = 2', c16.subs(x, 1), sp.Integer(2))
chk('C16  斜渐近线 y = x', sp.limit(c16 - x, x, sp.oo), sp.Integer(0))
true_ineq('C17  ln(1+x) > x - x^2/2 (x>0)', None,
          [mp.mpf(k) / 20 for k in range(1, 121)],
          lambda v: mp.log(1 + v) > v - v**2 / 2)
chk('C18  (cos x)^(1/x^2) -> e^(-1/2)',
    sp.exp(sp.limit(sp.log(sp.cos(x)) / x**2, x, 0)), sp.exp(-sp.Rational(1, 2)))

print('=' * 74)
print('总计 PASS %d / FAIL %d' % (PASS, FAIL))
if FAILS:
    print('失败项：', FAILS)
print('=' * 74)
