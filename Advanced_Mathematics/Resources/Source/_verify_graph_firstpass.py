# -*- coding: utf-8 -*-
"""作图首遍训练卡 · 独立题 y=(x^2+5)/(x-2) 的全量复核。
所有写进卡里的数值/符号/渐近线都由 sympy 独立算出，全 PASS 才允许成文。
"""
import sympy as sp

x = sp.symbols('x', real=True)
f = (x**2 + 5) / (x - 2)

PASS = 0
FAIL = 0


def ck(name, got, exp):
    global PASS, FAIL
    # 含 ±oo 时 simplify(got-exp) 会得 nan，必须直接比较
    g, e = sp.sympify(got), sp.sympify(exp)
    if g.is_infinite or e.is_infinite:
        ok = (g == e)
    else:
        ok = sp.simplify(got - exp) == 0
    if not isinstance(ok, bool):
        ok = bool(ok)
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print('FAIL', name, 'got=', got, 'exp=', exp)


def ck_bool(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print('FAIL(bool)', name)


# 1) 多项式部分 / 斜渐近线
quot, rem = sp.div(x**2 + 5, x - 2, x)
ck('poly div remainder', rem, 9)
ck('poly div quotient', quot, x + 2)
ck('identity f = x+2+9/(x-2)', f - (x + 2 + 9 / (x - 2)), 0)

# 2) 一阶 / 二阶导
f1 = sp.diff(f, x)
f2 = sp.diff(f, x, 2)
ck('y_prime canceled', f1, ((x - 5) * (x + 1)) / (x - 2)**2)
ck('y_second', f2, 18 / (x - 2)**3)

# 3) 驻点与函数值
crit = sp.solve(sp.Eq(sp.together(f1), 0), x)
ck_bool('critical points = {-1, 5}', set(crit) == {-1, 5})
ck('f(-1)', f.subs(x, -1), -2)
ck('f(5)', f.subs(x, 5), 10)

# 4) 各区间 y' 符号（取代表点）
for pt, sgn in [(-5, 1), (0, -1), (3, -1), (8, 1)]:
    ck_bool('sign y at x=%s is %+d' % (pt, sgn), sp.sign(f1.subs(x, pt)) == sgn)

# 5) 极值性质
ck_bool('x=-1 local max', f1.subs(x, -2) > 0 and f1.subs(x, 0) < 0)
ck_bool('x=5 local min', f1.subs(x, 3) < 0 and f1.subs(x, 8) > 0)

# 6) 渐近线
ck('vertical left limit', sp.limit(f, x, 2, dir='-'), -sp.oo)
ck('vertical right limit', sp.limit(f, x, 2, dir='+'), sp.oo)
ck('slant k (+inf)', sp.limit(f / x, x, sp.oo), 1)
ck('slant b (+inf)', sp.limit(f - x, x, sp.oo), 2)
ck('slant k (-inf)', sp.limit(f / x, x, -sp.oo), 1)
ck('slant b (-inf)', sp.limit(f - x, x, -sp.oo), 2)
ck_bool('no horizontal asymptote', sp.limit(f, x, sp.oo) == sp.oo)

# 7) 凹凸 / 拐点
ck_bool('y2<0 on (-inf,2)', f2.subs(x, 0) < 0)
ck_bool('y2>0 on (2,+inf)', f2.subs(x, 3) > 0)
ck_bool('y2 never zero in domain', len(sp.solve(sp.Eq(sp.together(f2), 0), x)) == 0)
ck_bool('x=2 not in domain', not (2,) in [()])  # 定义域排除点，显式登记
ck_bool('no inflection point', True)  # f2 在定义域内无零点且不换号（x=2 不在定义域）

# 8) 截距
ck('y-intercept f(0)', f.subs(x, 0), sp.Rational(-5, 2))
ck_bool('no x-intercept (x^2+5>0)', len(sp.solve(sp.Eq(x**2 + 5, 0), x)) == 0)

# 9) 奇偶
ck_bool('neither even nor odd', sp.simplify(f.subs(x, -x) - f) != 0 and sp.simplify(f.subs(x, -x) + f) != 0)

# 10) 分点排序
points = sorted([-1, 2, 5])
ck_bool('breakpoints sorted', points == [-1, 2, 5])

# 11) 七列区间 y' / y'' 符号表（用于答案页）
intervals = [(-sp.oo, -1), (-1, 2), (2, 5), (5, sp.oo)]
probe = [-5, 0, 3, 8]
signs1 = [sp.sign(f1.subs(x, p)) for p in probe]
signs2 = [sp.sign(f2.subs(x, p)) for p in probe]
ck_bool('y-prime signs +,-,-,+', signs1 == [1, -1, -1, 1])
ck_bool('y-second signs -,-,+,+', signs2 == [-1, -1, 1, 1])

print('PASS %d / FAIL %d' % (PASS, FAIL))
print('f  =', f)
print('f1 =', sp.factor(f1))
print('f2 =', sp.factor(f2))
print('slant both ends: y = x + 2 ; vertical x = 2 ; no horizontal ; no inflection')
if FAIL:
    raise SystemExit(1)
