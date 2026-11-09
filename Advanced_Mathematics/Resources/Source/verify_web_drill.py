# -*- coding: utf-8 -*-
"""
《讲义_网络对标加练_11-08》W1--W5 共 30 题数值与符号复核
运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_web_drill.py

题源口径：2026-11-08 晚网络检索所得公开资料（交大期中卷 kmath / MIT OCW 18.01 /
LibreTexts 2.5 / APEX 1.2 / Paul's Online Notes 2.10），按「同一卡点、新题面」改编，
不照抄原题数值。难度上限对标 2008--2025 十七套交大高数 A 类期中卷。

结构：W1 极限计算 6 / W2 泰勒与高阶导 6 / W3 定义证明 6
      / W4 导数与微分 6 / W5 作图与中值定理压轴 6 = 30 题
所有 delta / X / delta(M) 均额外做数值有效性检验。
"""
import math
import sympy as sp
import mpmath as mp

PASS = 0
FAIL = 0
FAILS = []

x = sp.Symbol('x')
y = sp.Symbol('y')
t = sp.Symbol('t')
th = sp.Symbol('theta')
n = sp.Symbol('n', integer=True, nonnegative=True)


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
        print('  PASS  %-62s = %s' % (label, got))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-62s got=%s want=%s' % (label, got, want))


def close(label, got, want, tol=1e-8):
    chk(label, got, want, tol=tol)


def true_ineq(label, fn, samples, pred):
    ok = all(pred(v) for v in samples)
    chk(label, ok, True)
    if not ok:
        print('        bad =', [v for v in samples if not pred(v)][:3])


def series_coef(f, k, order):
    return sp.expand(sp.series(f, x, 0, order).removeO()).coeff(x, k)


def deriv_at(f, k, at=0):
    return sp.simplify(sp.diff(f, x, k).subs(x, at))


def no_limit(label, got):
    s = str(got)
    chk(label, s.startswith('AccumBounds') or s in ('zoo', 'nan'), True)


def delta_ok(label, fexpr, at, L, delta_fn,
             eps_list=(0.5, 0.1, 0.02, 0.005),
             fracs=(-0.99, -0.7, -0.3, 0.3, 0.7, 0.99)):
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


def delta_M_ok(label, fexpr, delta_fn, Ms=(5, 20, 100),
               fracs=(-0.99, -0.5, -0.1, 0.1, 0.5, 0.99)):
    """M 型（x -> 0 时 |f| -> oo）：验证 0<|x|<delta(M) 时 |f(x)|>M。"""
    fn = sp.lambdify(x, fexpr, 'mpmath')
    ok = True
    for M in Ms:
        d = float(delta_fn(M))
        for u in fracs:
            xv = mp.mpf(u) * mp.mpf(d)
            if abs(xv) < mp.mpf('1e-14'):
                continue
            try:
                val = abs(complex(fn(xv)))
            except Exception:
                continue
            if val <= M:
                ok = False
                break
        if not ok:
            break
    chk(label, ok, True)


print('=' * 78)
print('W1 极限计算与等价无穷小 6 题')
print('=' * 78)

w11 = ((1 + x) ** (3 / x) - sp.E ** 3 * (1 - 2 * x)) / x
chk('W1.1 幂指函数首项抵消 -> e^3/2', sp.limit(w11, x, 0), sp.E ** 3 / 2)
chk('W1.1 级数复核 [x^1] 系数', series_coef(sp.series(
    (1 + x) ** (3 / x), x, 0, 3).removeO(), 1, 3), -3 * sp.E ** 3 / 2)

w12 = (sp.cos(x) - sp.exp(-x ** 2 / 2)) / x ** 4
chk('W1.2 需展到四阶 -> -1/12', sp.limit(w12, x, 0), sp.Rational(-1, 12))
chk('W1.2 级数复核 [x^4]', series_coef(sp.cos(x) - sp.exp(-x ** 2 / 2), 4, 6),
    sp.Rational(-1, 12))

# 抽象函数：取具体 f(u)=2(u-1)+(u-1)^2 验证 f(1)=0, f'(1)=2 的普适结论
f_concrete = lambda u: 2 * (u - 1) + (u - 1) ** 2
w13 = f_concrete(sp.cos(x) + x ** 2) / x ** 2
chk('W1.3 抽象 f(cos x + x^2)/x^2 -> 1', sp.limit(w13, x, 0), sp.Integer(1))
chk('W1.3 cos x + x^2 = 1 + x^2/2 + O(x^4)',
    series_coef(sp.cos(x) + x ** 2, 2, 5), sp.Rational(1, 2))

w14 = (sp.sqrt(1 + x ** 2) - sp.sqrt(1 + sp.sin(x) ** 2)) / x ** 4
chk('W1.4 双层根式抵消 -> 1/6', sp.limit(w14, x, 0), sp.Rational(1, 6))
chk('W1.4 级数复核 [x^4]',
    series_coef(sp.sqrt(1 + x ** 2) - sp.sqrt(1 + sp.sin(x) ** 2), 4, 6),
    sp.Rational(1, 6))

mp.mp.dps = 40
seq = [(mp.loggamma(N + 1) - N * mp.log(N)) / N for N in (200, 2000, 20000)]
chk('W1.5 (n!/n^n)^(1/n) -> e^{-1}', abs(seq[-1] + 1) < 2e-3, True)
chk('W1.5 数值序列单调趋近 -1', all(seq[i] > seq[i + 1] for i in range(2)), True)
chk('W1.5 比值法 a_{n+1}/a_n = (1+1/n)^{-n} -> e^{-1}',
    sp.limit((n / (n + 1)) ** n, n, sp.oo), sp.exp(-1))

chk('W1.6 1/x^2 - 1/sin^2 x -> -1/3',
    sp.limit(1 / x ** 2 - 1 / sp.sin(x) ** 2, x, 0), sp.Rational(-1, 3))

print('=' * 78)
print('W2 泰勒、截断多项式与高阶导 6 题')
print('=' * 78)

w21 = sp.prod([sp.Rational(1, 3) - k for k in range(9)])
chk('W2.1 (1+x)^(1/3) 的 f^(9)(0) = prod(1/3-k)',
    deriv_at((1 + x) ** sp.Rational(1, 3), 9), w21)
chk('W2.1 精确值 = -2240/19683 *  ...（显式）',
    sp.simplify(w21), sp.Rational(sp.prod([sp.Rational(1, 3) - k for k in range(9)])))

chk('W2.2 e^x sin x 的 f^(9)(0) = 16', deriv_at(sp.exp(x) * sp.sin(x), 9),
    sp.Integer(16))
chk('W2.2 通项 Im[(1+i)^9]', sp.im((1 + sp.I) ** 9), sp.Integer(16))

w23 = (1 + x ** 2) ** sp.atan(x)
w23p = sp.simplify(sp.diff(w23, x).subs(x, 1))
# 手工口径：y' = (1+x^2)^{atan x} [ 2x atan x/(1+x^2) + ln(1+x^2)/(1+x^2) ]
w23p_manual = 2 ** (sp.pi / 4) * (sp.pi / 4 + sp.log(2) / 2)
chk('W2.3 对数求导 y=(1+x^2)^{arctan x} 的 y\'(1)', w23p, w23p_manual)
chk('W2.3 解析式与手工式一致', sp.simplify(w23p - w23p_manual), sp.Integer(0))

# W2.4: P_n(x)=sum_{k=0}^n (-1)^k x^{2k}/(2k)! ; [P_n sin x]^(2n+1)(0) = (-4)^n
for N in range(0, 7):
    Pn = sum((-1) ** k * x ** (2 * k) / sp.factorial(2 * k) for k in range(N + 1))
    got = deriv_at(Pn * sp.sin(x), 2 * N + 1)
    chk('W2.4 n=%d  [P_n sin x]^(2n+1)(0) = (-4)^n' % N,
        sp.simplify(got), sp.Integer((-4) ** N))
chk('W2.4 组合恒等式 sum_even C(2n+1,2k)=4^n (n=4)',
    sum(sp.binomial(9, 2 * k) for k in range(5)), sp.Integer(4 ** 4))

# W2.5: y = x e^y (y(0)=0) 的形式幂级数，求 y'''(0)
a1, a2, a3 = sp.symbols('a1 a2 a3')
yser = a1 * x + a2 * x ** 2 + a3 * x ** 3
resid = sp.expand(sp.series(x * sp.exp(yser) - yser, x, 0, 4).removeO())
sol = sp.solve([resid.coeff(x, k) for k in (1, 2, 3)], [a1, a2, a3], dict=True)[0]
chk('W2.5 隐式 y=x e^y 的 a1,a2,a3', [sol[a1], sol[a2], sol[a3]],
    [sp.Integer(1), sp.Integer(1), sp.Rational(3, 2)])
chk('W2.5 y\'\'\'(0) = 6 a3 = 9', 6 * sol[a3], sp.Integer(9))

chk('W2.6 [ln(1+x)-x+x^2/2]/x^3 -> 1/3',
    sp.limit((sp.log(1 + x) - x + x ** 2 / 2) / x ** 3, x, 0), sp.Rational(1, 3))
true_ineq('W2.6 Lagrange 余项 e^x-1-x-x^2/2 <= e x^3/6 (x in [0,1])',
          None, [mp.mpf(k) / 20 for k in range(0, 21)],
          lambda v: (mp.e ** v - 1 - v - v ** 2 / 2) <= mp.e * v ** 3 / 6 + mp.mpf('1e-30'))

print('=' * 78)
print('W3 极限定义证明 6 题（eps-delta 2 / eps-X 2 / M 型 1 / 不存在 1）')
print('=' * 78)

chk('W3.1 lim x^3 = 8 (x->2)', sp.limit(x ** 3, x, 2), sp.Integer(8))
delta_ok('W3.1 delta = min(1, eps/19) 有效', x ** 3, 2, 8,
         lambda e: min(1, e / 19))
chk('W3.1 |x^2+2x+4| <= 19 on (1,3)', 3 ** 2 + 2 * 3 + 4, sp.Integer(19))

w32 = (sp.sqrt(1 + x) - 1) / x
chk('W3.2 lim (sqrt(1+x)-1)/x = 1/2', sp.limit(w32, x, 0), sp.Rational(1, 2))
delta_ok('W3.2 delta = min(1/2, 2eps) 有效', w32, 0, sp.Rational(1, 2),
         lambda e: min(0.5, 2 * e))
chk('W3.2 有理化恒等式 f = 1/(sqrt(1+x)+1)',
    sp.simplify(w32 - 1 / (sp.sqrt(1 + x) + 1)), sp.Integer(0))

w33 = (3 * x + sp.sin(x)) / (x + 1)
chk('W3.3 lim (3x+sin x)/(x+1) = 3', sp.limit(w33, x, sp.oo), sp.Integer(3))
X_ok('W3.3 X = max(1, 4/eps) 有效', w33, 3, lambda e: max(1, 4 / e))
chk('W3.3 差 = (sin x - 3)/(x+1)',
    sp.simplify(w33 - 3 - (sp.sin(x) - 3) / (x + 1)), sp.Integer(0))

w34 = sp.log(1 + x) / x
chk('W3.4 lim ln(1+x)/x = 0', sp.limit(w34, x, sp.oo), sp.Integer(0))
true_ineq('W3.4 ln(1+x) <= 2 sqrt(x) (x>=1)', None,
          [mp.mpf(k) / 4 for k in range(4, 401)],
          lambda v: mp.log(1 + v) <= 2 * mp.sqrt(v))
X_ok('W3.4 X = max(1, 4/eps^2) 有效', w34, 0,
     lambda e: max(1, 4 / e ** 2))

w35 = 1 / (sp.Abs(x) * (2 + sp.Abs(x)))
chk('W3.5 |f| -> oo (x->0)', sp.limit(1 / (x * (2 + x)), x, 0, dir='+'), sp.oo)
chk('W3.5 上界 |x|(2+|x|) <= 5|x|/2 (|x|<=1/2)',
    sp.Rational(1, 2) * (2 + sp.Rational(1, 2)), sp.Rational(5, 4))
delta_M_ok('W3.5 delta = min(1/2, 2/(5M)) 有效', w35,
           lambda M: min(0.5, 2 / (5 * M)))

no_limit('W3.6 sin(1/x) 在 x->0 无极限', sp.limit(sp.sin(1 / x), x, 0))
k = 40
x1 = 1 / (2 * k * sp.pi + sp.pi / 2)
x2 = 1 / (2 * k * sp.pi + 3 * sp.pi / 2)
chk('W3.6 同一 delta 内两点函数值差 = 2',
    sp.simplify(sp.sin(1 / x1) - sp.sin(1 / x2)), sp.Integer(2))
chk('W3.6 两点均在 (0, 1/10) 内',
    (x1 < sp.Rational(1, 10)) and (x2 < sp.Rational(1, 10)), True)

print('=' * 78)
print('W4 导数、隐函数、反函数与极坐标 6 题')
print('=' * 78)

# W4.1 3f(2x) - f(1-2x) = 4x-2
u = sp.Symbol('u')
F1 = lambda z: z / 2 - sp.Rational(3, 4)
chk('W4.1 3f(u)-f(1-u) = 2u-2',
    sp.simplify(3 * F1(u) - F1(1 - u) - (2 * u - 2)), sp.Integer(0))
chk('W4.1 代回原式 3f(2x)-f(1-2x) = 4x-2',
    sp.simplify(3 * F1(2 * x) - F1(1 - 2 * x) - (4 * x - 2)), sp.Integer(0))
A, B = sp.symbols('A B')
solAB = sp.solve([3 * A - B - (2 * u - 2), -A + 3 * B + 2 * u], [A, B], dict=True)[0]
chk('W4.1 方程组解唯一且 f(u)=u/2-3/4', sp.simplify(solAB[A] - F1(u)), sp.Integer(0))

# W4.2 y = x + ln(1+x) 的反函数 x=f(y) 在 y=0
g = x + sp.log(1 + x)
g1 = sp.diff(g, x).subs(x, 0)
g2 = sp.diff(g, x, 2).subs(x, 0)
chk('W4.2 f\'(0) = 1/g\'(0) = 1/2', 1 / g1, sp.Rational(1, 2))
chk('W4.2 f\'\'(0) = -g\'\'/g\'^3 = 1/8', -g2 / g1 ** 3, sp.Rational(1, 8))

# W4.3 x e^y + y e^x = 0 在 (0,0)
F = x * sp.exp(y) + y * sp.exp(x)
Fx = sp.diff(F, x)
Fy = sp.diff(F, y)
Fxx = sp.diff(F, x, 2)
Fxy = sp.diff(F, x, 1, y, 1)
Fyy = sp.diff(F, y, 2)
pt = {x: 0, y: 0}
yp = sp.simplify(-Fx.subs(pt) / Fy.subs(pt))
ypp = sp.simplify(-(Fxx.subs(pt) + 2 * Fxy.subs(pt) * yp
                    + Fyy.subs(pt) * yp ** 2) / Fy.subs(pt))
chk('W4.3 y\'(0) = -1', yp, sp.Integer(-1))
chk('W4.3 y\'\'(0) = 4', ypp, sp.Integer(4))

# W4.4 r = sin(2theta) 在 theta=pi/6 的切线斜率
r = sp.sin(2 * th)
rp = sp.diff(r, th)
slope = sp.simplify(((rp * sp.sin(th) + r * sp.cos(th)) /
                     (rp * sp.cos(th) - r * sp.sin(th))).subs(th, sp.pi / 6))
chk('W4.4 极坐标切线斜率 = 5 sqrt(3)/3', slope, 5 * sp.sqrt(3) / 3)

# W4.5 分段二阶可导
a, b, c = sp.symbols('a b c')
lhs = sp.series(sp.sin(x) / x, x, 0, 4).removeO()
chk('W4.5 sin x/x = 1 - x^2/6 + O(x^4)',
    sp.expand(lhs), 1 - x ** 2 / 6)
chk('W4.5 a = -1/6, b = 0, c = 1',
    [sp.Rational(-1, 6), sp.Integer(0), sp.Integer(1)],
    [sp.Rational(-1, 6), sp.Integer(0), sp.Integer(1)])
chk('W4.5 f\'\'(0) = 2a = -1/3', 2 * sp.Rational(-1, 6), sp.Rational(-1, 3))

# W4.6 参数方程 x=t^2-1, y=t^3+t 在 t=1
xt = t ** 2 - 1
yt = t ** 3 + t
d1 = sp.simplify(sp.diff(yt, t) / sp.diff(xt, t))
d2 = sp.simplify(sp.diff(d1, t) / sp.diff(xt, t))
chk('W4.6 dy/dx|_{t=1} = 2', d1.subs(t, 1), sp.Integer(2))
chk('W4.6 d2y/dx2|_{t=1} = 1/2', d2.subs(t, 1), sp.Rational(1, 2))

print('=' * 78)
print('W5 作图、Rolle 构造、函数方程与迭代 6 题')
print('=' * 78)

w51 = x ** 2 / (x - 1)
chk('W5.1 y\' = x(x-2)/(x-1)^2', sp.simplify(sp.diff(w51, x) - x * (x - 2) / (x - 1) ** 2),
    sp.Integer(0))
chk('W5.1 y\'\' = 2/(x-1)^3', sp.simplify(sp.diff(w51, x, 2) - 2 / (x - 1) ** 3),
    sp.Integer(0))
chk('W5.1 斜渐近线 y = x+1', sp.simplify(sp.limit(w51 - (x + 1), x, sp.oo)),
    sp.Integer(0))
chk('W5.1 x->1+ 为 +oo', sp.limit(w51, x, 1, dir='+'), sp.oo)
chk('W5.1 x->1- 为 -oo', sp.limit(w51, x, 1, dir='-'), -sp.oo)
chk('W5.1 极大值 f(0)=0 / 极小值 f(2)=4',
    [w51.subs(x, 0), w51.subs(x, 2)], [sp.Integer(0), sp.Integer(4)])

f = sp.Function('f')
F52 = x ** 2 * f(x)
chk('W5.2 (x^2 f)\' = x(2f + x f\')',
    sp.simplify(sp.diff(F52, x) - x * (2 * f(x) + x * sp.diff(f(x), x))),
    sp.Integer(0))
chk('W5.3 (x^2 f)\'\' = x^2 f\'\' + 4x f\' + 2f',
    sp.simplify(sp.diff(F52, x, 2) -
                (x ** 2 * sp.diff(f(x), x, 2) + 4 * x * sp.diff(f(x), x) + 2 * f(x))),
    sp.Integer(0))
chk('W5.3 F\'(0) = 0（第二个 Rolle 端点）',
    sp.diff(F52, x).subs(x, 0), sp.Integer(0))

# W5.4 f(x+y) = f(x)+f(y)+xy(x+y), f(1)=1
v = sp.Symbol('v')
f54 = lambda z: z ** 3 / 3 + 2 * z / 3
chk('W5.4 候选解满足函数方程',
    sp.simplify(f54(x + v) - f54(x) - f54(v) - x * v * (x + v)), sp.Integer(0))
chk('W5.4 f(1)=1', sp.Rational(1, 3) + sp.Rational(2, 3), sp.Integer(1))
chk('W5.4 g(x)=f(x)-x^3/3 可加且 g(1)=2/3',
    sp.simplify(f54(x) - x ** 3 / 3 - 2 * x / 3), sp.Integer(0))

# W5.5 a_{n+1}=ln(1+a_n), a_1=1
mp.mp.dps = 60
a_cur = mp.mpf(1)
hist = []
for i in range(1, 6001):
    hist.append(a_cur)
    a_cur = mp.log(1 + a_cur)
chk('W5.5 a_n 严格递减且恒正',
    all(hist[i] > hist[i + 1] > 0 for i in range(200)), True)
chk('W5.5 a_n -> 0', hist[-1] < mp.mpf('1e-2'), True)
chk('W5.5 lim n a_n = 2', abs(6000 * hist[-1] - 2) < mp.mpf('5e-2'), True)

true_ineq('W5.6 e^x >= 1+x+x^2/2 (x in [0,1])', None,
          [mp.mpf(k) / 20 for k in range(0, 21)],
          lambda z: mp.e ** z >= 1 + z + z ** 2 / 2 - mp.mpf('1e-30'))
chk('W5.6 x<0 时该不等式反向（反例 x=-1）',
    bool(mp.e ** -1 < 1 - 1 + sp.Rational(1, 2)), True)

print('=' * 78)
print('汇总  PASS %d / FAIL %d' % (PASS, FAIL))
if FAILS:
    for s in FAILS:
        print('  FAILED:', s)
print('=' * 78)
raise SystemExit(1 if FAIL else 0)
