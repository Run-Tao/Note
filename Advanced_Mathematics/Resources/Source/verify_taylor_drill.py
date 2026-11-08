# -*- coding: utf-8 -*-
"""
册一《讲义_泰勒与压轴强化_11-08》数值与符号复核
运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_taylor_drill.py
覆盖：泰勒专项 18 题（T1 展开与系数 5 / T2 余项与逼近阶 4 / T3 泰勒求导数 4
      / T4 泰勒应用 5）+ 泰勒限时卷 12 题 = 30 题
定位：先写脚本再写讲义；本脚本锁定的数值即为讲义答案的来源。
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
        elif got in (sp.oo, -sp.oo, sp.zoo) or want in (sp.oo, -sp.oo, sp.zoo):
            ok = str(got) == str(want)
        elif tol is not None:
            try:
                ok = abs(complex(sp.N(got, 30)) - complex(sp.N(want, 30))) < tol
            except Exception:
                ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
        elif isinstance(got, (list, tuple)) or isinstance(want, (list, tuple)):
            if not (isinstance(got, (list, tuple)) and isinstance(want, (list, tuple))
                    and len(got) == len(want)):
                ok = False
            else:
                ok = all(sp.simplify(sp.sympify(g) - sp.sympify(w)) == 0
                         for g, w in zip(got, want))
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception:
        ok = False
    if ok:
        PASS += 1
        print('  PASS  %-54s = %s' % (label, got))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-54s got=%s want=%s' % (label, got, want))


def close(label, got, want, tol=5e-9):
    chk(label, got, want, tol=tol)


def series_coef(f, x, n, order):
    """f 的 x^n 项系数（order 取 n+1 保证含 x^n）。"""
    s = sp.series(f, x, 0, order).removeO()
    return sp.expand(s).coeff(x, n)


def deriv_at(f, x, n, at=0):
    """f 在 at 处的 n 阶导数。"""
    return sp.simplify(sp.diff(f, x, n).subs(x, at))


def true_ineq(label, fn, domain_samples, pred):
    """不等式数值抽检：domain_samples 给出区间样本点，pred 返回布尔。"""
    global PASS, FAIL
    ok = all(pred(v) for v in domain_samples)
    if ok:
        PASS += 1
        print('  PASS  %-54s = %s' % (label, '成立'))
    else:
        FAIL += 1
        FAILS.append(label)
        bad = [v for v in domain_samples if not pred(v)]
        print('  FAIL  %-54s 反例 x=%s' % (label, bad[:3]))


x = sp.symbols('x', real=True)
t = sp.symbols('t', real=True)
a = sp.symbols('a', real=True)
al = sp.symbols('alpha', real=True)
y_ = sp.Symbol('y', real=True)

print('=' * 78)
print('T1  展开与系数（5 题）—— 母函数展开表的实战')
print('=' * 78)

# T1.1  e^{sin x} 的 3 阶麦克劳林展开（x^3 系数恰为 0，是陷阱）
s11 = sp.expand(sp.series(sp.exp(sp.sin(x)), x, 0, 4).removeO())
chk('T1.1 e^{sin x} 3 阶展开', s11, 1 + x + x ** 2 / 2)
chk('T1.1 佐证 x^3 项系数为 0', series_coef(sp.exp(sp.sin(x)), x, 3, 5), 0)

# T1.2  (1+x)^{1/3} 的 4 阶麦克劳林展开
s12 = sp.expand(sp.series((1 + x) ** sp.Rational(1, 3), x, 0, 5).removeO())
chk('T1.2 (1+x)^{1/3} 4 阶展开', s12,
    1 + x / 3 - x ** 2 / 9 + 5 * x ** 3 / 81 - 10 * x ** 4 / 243)

# T1.3  (1+2x)^{1/2} 的 4 阶麦克劳林展开
s13 = sp.expand(sp.series((1 + 2 * x) ** sp.Rational(1, 2), x, 0, 5).removeO())
chk('T1.3 (1+2x)^{1/2} 4 阶展开', s13, 1 + x - x ** 2 / 2 + x ** 3 / 2 - 5 * x ** 4 / 8)

# T1.4  ln(1+e^x) 的 3 阶麦克劳林展开（x^3 系数为 0）
s14 = sp.expand(sp.series(sp.log(1 + sp.exp(x)), x, 0, 4).removeO())
chk('T1.4 ln(1+e^x) 3 阶展开', s14, sp.log(2) + x / 2 + x ** 2 / 8)
chk('T1.4 佐证 x^3 项系数为 0', series_coef(sp.log(1 + sp.exp(x)), x, 3, 5), 0)

# T1.5  e^{x^2} cos x 的 4 阶麦克劳林展开
s15 = sp.expand(sp.series(sp.exp(x ** 2) * sp.cos(x), x, 0, 5).removeO())
chk('T1.5 e^{x^2}cosx 4 阶展开', s15, 1 + x ** 2 / 2 + x ** 4 / 24)

print()
print('=' * 78)
print('T2  余项与逼近阶（4 题）—— 该展到第几阶')
print('=' * 78)

close('T2.1 (sqrt(1+x)-1-x/2+x^2/8)/x^3',
      sp.limit((sp.sqrt(1 + x) - 1 - x / 2 + x ** 2 / 8) / x ** 3, x, 0), sp.Rational(1, 16))
close('T2.2 (sinx-x+x^3/6-x^5/120)/x^7',
      sp.limit((sp.sin(x) - x + x ** 3 / 6 - x ** 5 / 120) / x ** 7, x, 0),
      -sp.Rational(1, 5040))
close('T2.3 (ln cos x + x^2/2)/x^4',
      sp.limit((sp.log(sp.cos(x)) + x ** 2 / 2) / x ** 4, x, 0), -sp.Rational(1, 12))
close('T2.4 (arctanx-x+x^3/3-x^5/5)/x^7',
      sp.limit((sp.atan(x) - x + x ** 3 / 3 - x ** 5 / 5) / x ** 7, x, 0),
      -sp.Rational(1, 7))

# T2 佐证：逼近阶（余项是 x^k 的等价无穷小）
close('T2.1 佐证 余项 / x^3 -> 1/16',
      sp.limit((sp.sqrt(1 + x) - 1 - x / 2 + x ** 2 / 8) / x ** 3, x, 0), sp.Rational(1, 16))
close('T2.4 佐证 arctan 7 阶展开',
      sp.expand(sp.series(sp.atan(x), x, 0, 8).removeO()),
      x - x ** 3 / 3 + x ** 5 / 5 - x ** 7 / 7)

print()
print('=' * 78)
print('T3  泰勒求导数（5 题）—— 用展开式反求高阶导')
print('=' * 78)

# T3.1  f(x)=(1+x)^alpha 的 13 阶导数在 0（对标 2023-24 Q15）
#      注意：f^(13)(0) = alpha(alpha-1)...(alpha-12)（=13!× 展开式中 x^13 的系数）
d13 = deriv_at((1 + x) ** al, x, 13)
want13 = sp.prod(al - k for k in range(13))
chk('T3.1 f^(13)(0) = prod_{k=0..12}(alpha-k)', d13, want13)
close('T3.1 lim [(1+x)^alpha - 1 - alpha x]/x^2',
      sp.limit(((1 + x) ** al - 1 - al * x) / x ** 2, x, 0), al * (al - 1) / 2)
close('T3.1 佐证 alpha=1/3 时 f^(13)(0) 数值',
      deriv_at((1 + x) ** sp.Rational(1, 3), x, 13),
      sp.prod(sp.Rational(1, 3) - k for k in range(13)))
# T3.1 佐证：13 阶导 = 13! × x^13 系数
chk('T3.1 佐证 13!×[x^13]系数 == f^(13)(0)',
    sp.factorial(13) * series_coef((1 + x) ** al, x, 13, 14), want13)

# T3.2  f(x) = x e^{x^2} 的 5 阶导数在 0
chk('T3.2 f^(5)(0), f=x e^{x^2}', deriv_at(x * sp.exp(x ** 2), x, 5), 60)

# T3.3  f(x) = e^{sin x} 的 3 阶导数在 0（= 0）
chk('T3.3 f^(3)(0), f=e^{sin x}', deriv_at(sp.exp(sp.sin(x)), x, 3), 0)

# T3.4  f(x) = e^x cos x 的 4 阶导数在 0
chk('T3.4 f^(4)(0), f=e^x cos x', deriv_at(sp.exp(x) * sp.cos(x), x, 4), -4)

# T3.5  参数方程 x=t+t^2, y=t-t^3 的二阶导数在 t=0
dx = sp.diff(t + t ** 2, t)
dy = sp.diff(t - t ** 3, t)
d1 = sp.simplify(dy / dx)
d2 = sp.simplify(sp.diff(d1, t) / dx)
close('T3.5 dy/dx at t=0', d1.subs(t, 0), 1)
close('T3.5 d2y/dx2 at t=0', d2.subs(t, 0), -2)
# 泰勒法佐证：取 t(0)=0 的分支 t = (-1+sqrt(1+4x))/2
t_of_x = (-1 + sp.sqrt(1 + 4 * x)) / 2
close('T3.5 泰勒法佐证 y(x) 展开',
      sp.expand(sp.series((t_of_x - t_of_x ** 3), x, 0, 4).removeO()),
      x - x ** 2 + x ** 3)

print()
print('=' * 78)
print('T4  泰勒应用（5 题）—— 不等式 / 极值 / 渐近 / 隐函数')
print('=' * 78)

# T4.1  ln(1+x) > x - x^2/2  (x>0)
true_ineq('T4.1 ln(1+x) > x - x^2/2 (x>0)',
           sp.log(1 + x),
           [i / 200.0 for i in range(1, 400)],
           lambda v: float(sp.log(1 + v) - v + v ** 2 / 2) > 0)
# 佐证：差值的导数 = x^2/(1+x) > 0
chk('T4.1 佐证 差值导数 = x^2/(1+x) > 0',
    sp.simplify(sp.diff(sp.log(1 + x) - x + x ** 2 / 2, x)), x ** 2 / (x + 1))

# T4.2  sin x > x - x^3/6  (x>0)  —— 佩亚诺余项的典型结论
true_ineq('T4.2 sinx > x - x^3/6 (x>0)',
           sp.sin(x) - x + x ** 3 / 6,
           [i / 200.0 for i in range(1, 400)],
           lambda v: float(math.sin(v) - v + v ** 3 / 6) > 0)
chk('T4.2 佐证 差值导数 = cosx - 1 + x^2/2 >= 0',
    sp.simplify(sp.diff(sp.sin(x) - x + x ** 3 / 6, x)), sp.cos(x) - 1 + x ** 2 / 2)
true_ineq('T4.2 佐证 cosx >= 1 - x^2/2',
           sp.cos(x) - 1 + x ** 2 / 2,
           [i / 100.0 - 30 for i in range(1, 600)],
           lambda v: float(math.cos(v) - 1 + v ** 2 / 2) >= -1e-12)

# T4.3  f(x)=x^3 e^{-x} 在 (0,+inf) 上的极大值
f43 = x ** 3 * sp.exp(-x)
chk('T4.3 f\'(x) = x^2 e^{-x}(3-x)',
    sp.simplify(sp.diff(f43, x)), x ** 2 * sp.exp(-x) * (3 - x))
close('T4.3 极大值点 x=3 处的值', f43.subs(x, 3), 27 * sp.exp(-3))
close('T4.3 佐证 lim x->+inf f(x) = 0', sp.limit(f43, x, sp.oo), 0)

# T4.4  x->+inf 时 (1+1/x)^x 的渐近展开到 1/x^2
close('T4.4 lim [(1+1/x)^x - e + e/(2x)] x^2',
      sp.limit(((1 + 1 / x) ** x - sp.E + sp.E / (2 * x)) * x ** 2, x, sp.oo),
      11 * sp.E / 24)

# T4.5  隐函数 e^y + y = x 的 y' 与 y'' 在 x=1 处（以 y 为参数反解 x(y)）
dxdy = sp.diff(sp.exp(y_) + y_, y_)
d1p = sp.simplify(1 / dxdy)
d2p = sp.simplify(sp.diff(d1p, y_) / dxdy)
close('T4.5 y\'(1) = 1/2', d1p.subs(y_, 0), sp.Rational(1, 2))
close('T4.5 y\'\'(1) = -1/8', d2p.subs(y_, 0), -sp.Rational(1, 8))
# 数值佐证：x=1.001 时 y 的差商
_y_num = mp.mpf(0)
_x_num = mp.mpf('1.001')
def _yof(_x):
    _v = mp.mpf(0)
    for _ in range(60):
        _v = _v - (mp.e ** _v + _v - _x) / (mp.e ** _v + 1)
    return _v
_y1 = _yof(_x_num)
close('T4.5 数值佐证 y(1.001)', _y1, 0.0005, tol=2e-3)

print()
print('=' * 78)
print('W  泰勒限时卷（12 题 / 40 分钟 / 36 分）')
print('=' * 78)

close('W1 (e^x-1-x-x^2/2)/x^3',
      sp.limit((sp.exp(x) - 1 - x - x ** 2 / 2) / x ** 3, x, 0), sp.Rational(1, 6))
chk('W2 sqrt(1+x) 的 x^3 项系数', series_coef(sp.sqrt(1 + x), x, 3, 5), sp.Rational(1, 16))
close('W3 (tanx-x)/x^3', sp.limit((sp.tan(x) - x) / x ** 3, x, 0), sp.Rational(1, 3))
chk('W4 f^(4)(0), f=ln(1+x^2)', deriv_at(sp.log(1 + x ** 2), x, 4), -12)
close('W5 (arcsinx-x)/x^3',
      sp.limit((sp.asin(x) - x) / x ** 3, x, 0), sp.Rational(1, 6))
close('W6 (e^x-1-x)/x^2', sp.limit((sp.exp(x) - 1 - x) / x ** 2, x, 0), sp.Rational(1, 2))
s7 = sp.expand(sp.series(sp.exp(sp.cos(x)), x, 0, 3).removeO())
chk('W7 e^{cosx} 2 阶展开', s7, sp.E - sp.E * x ** 2 / 2)
close('W8 (sqrt(1+x)-1-x/2)/x^2',
      sp.limit((sp.sqrt(1 + x) - 1 - x / 2) / x ** 2, x, 0), -sp.Rational(1, 8))
close('W9 lim x[(1+1/x)^x - e]',
      sp.limit(x * ((1 + 1 / x) ** x - sp.E), x, sp.oo), -sp.E / 2)
chk('W10 由 lim[f(x)-x-2x^2]/x^3=3 求 f\'\'\'(0)',
    sp.Rational(3) * sp.factorial(3), 18)
close('W11 [(1+x)^(1/3) - 1 - x/3 + x^2/9]/x^3',
      sp.limit(((1 + x) ** sp.Rational(1, 3) - 1 - x / 3 + x ** 2 / 9) / x ** 3, x, 0),
      sp.Rational(5, 81))
close('W12 (ln(1+sinx)-x+x^2/2)/x^3',
      sp.limit((sp.log(1 + sp.sin(x)) - x + x ** 2 / 2) / x ** 3, x, 0),
      sp.Rational(1, 6))

# W10 佐证：构造一个满足条件的确切函数，验证其 f'''(0)
_g = x + 2 * x ** 2 + 3 * x ** 3
close('W10 佐证 满足条件的模型函数 lim = 3',
      sp.limit((_g - x - 2 * x ** 2) / x ** 3, x, 0), 3)
chk('W10 佐证 模型函数 f\'\'\'(0) = 18', deriv_at(_g, x, 3), 18)

print()
print('=' * 78)
print('母函数展开表抽样复核（讲义附录表逐行）')
print('=' * 78)
table = [
    ('e^x', sp.exp(x), 6, [1, 1, sp.Rational(1, 2), sp.Rational(1, 6),
                           sp.Rational(1, 24), sp.Rational(1, 120)]),
    ('sin x', sp.sin(x), 6, [0, 1, 0, -sp.Rational(1, 6), 0, sp.Rational(1, 120)]),
    ('cos x', sp.cos(x), 6, [1, 0, -sp.Rational(1, 2), 0, sp.Rational(1, 24), 0]),
    ('ln(1+x)', sp.log(1 + x), 5, [0, 1, -sp.Rational(1, 2), sp.Rational(1, 3), -sp.Rational(1, 4)]),
    ('arctan x', sp.atan(x), 6, [0, 1, 0, -sp.Rational(1, 3), 0, sp.Rational(1, 5)]),
    ('1-cos x', 1 - sp.cos(x), 5, [0, 0, sp.Rational(1, 2), 0, -sp.Rational(1, 24)]),
    ('(1+x)^{1/2}', sp.sqrt(1 + x), 5, [1, sp.Rational(1, 2), -sp.Rational(1, 8),
                                       sp.Rational(1, 16), -sp.Rational(5, 128)]),
    ('(1+x)^a', (1 + x) ** al, 5, [1, al, al * (al - 1) / 2,
                                   al * (al - 1) * (al - 2) / 6,
                                   al * (al - 1) * (al - 2) * (al - 3) / 24]),
    ('tan x', sp.tan(x), 5, [0, 1, 0, sp.Rational(1, 3), 0]),
]
for nm, f, order, coef in table:
    got = [sp.simplify(series_coef(f, x, i, order + 1)) for i in range(order)]
    chk('母表 %s 各项系数' % nm, got, coef)

print()
print('=' * 78)
print('结果：PASS %d / FAIL %d' % (PASS, FAIL))
print('=' * 78)
if FAILS:
    print('失败项：')
    for f in FAILS:
        print('  - ' + f)
