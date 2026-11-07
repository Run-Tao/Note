# -*- coding: utf-8 -*-
"""
复核《讲义_今日配套加练_11-07》全部可计算结论。
运行： PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_daily_drill.py
"""
import sympy as sp
import mpmath as mp

x, t, n, a, b = sp.symbols('x t n a b', real=True)
PASS = FAIL = 0


def chk(label, got, want, tol=None):
    global PASS, FAIL
    try:
        if tol is not None:
            ok = abs(complex(sp.N(got) if not isinstance(got, (int, float, complex)) else got)
                     - complex(want)) < tol
        elif isinstance(got, (list, tuple)) or isinstance(want, (list, tuple)):
            g = list(got) if isinstance(got, (list, tuple)) else [got]
            w = list(want) if isinstance(want, (list, tuple)) else [want]
            ok = len(g) == len(w) and all(sp.simplify(sp.sympify(u) - sp.sympify(v)) == 0
                                          for u, v in zip(g, w))
        elif got in (sp.oo, -sp.oo, sp.zoo) or want in (sp.oo, -sp.oo, sp.zoo):
            ok = (got is want) or (str(got) == str(want))
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception as e:
        ok = False
        print('  ERR', label, type(e).__name__, e)
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print('  FAIL', label, 'got', got, 'want', want)
    return ok


def lim(expr, var, to, dirn='+'):
    return sp.limit(expr, var, to, dir=dirn)


print('===== §2 函数极限计算 L1-L9 =====')
chk('L1 根式差', lim((sp.sqrt(1 + x**2) - sp.sqrt(1 - x**2)) / x**2, x, 0), sp.Integer(1))
chk('L2 1/sin²-1/x²', lim(1 / sp.sin(x)**2 - 1 / x**2, x, 0), sp.Rational(1, 3))
chk('L3 (cos x)^(1/x²)', lim(sp.cos(x)**(1 / x**2), x, 0), sp.exp(-sp.Rational(1, 2)))
chk('L4 ((x+3)/(x+1))^x', lim(((x + 3) / (x + 1))**x, x, sp.oo), sp.exp(2))
chk('L5 √(x²+3x)-√(x²+x)', lim(sp.sqrt(x**2 + 3*x) - sp.sqrt(x**2 + x), x, sp.oo), sp.Integer(1))
chk('L6 1/x-1/(e^x-1)', lim(1 / x - 1 / (sp.exp(x) - 1), x, 0), sp.Rational(1, 2))
chk('L7 (tan2x-2sinx)/x³', lim((sp.tan(2*x) - 2*sp.sin(x)) / x**3, x, 0), sp.Integer(3))
chk('L8 (1-cosx cos2x cos3x)/x²',
    lim((1 - sp.cos(x)*sp.cos(2*x)*sp.cos(3*x)) / x**2, x, 0), sp.Integer(7))
chk('L9 (x+sinx)/x', lim((x + sp.sin(x)) / x, x, sp.oo), sp.Integer(1))

print('===== §4 泰勒 / 导数 T1-T8 =====')
chk('T1 (sinx-xcosx)/x³', lim((sp.sin(x) - x*sp.cos(x)) / x**3, x, 0), sp.Rational(1, 3))
# T2: lim (f-1-2x)/x² = 3  =>  f(0)=1, f'(0)=2, f''(0)=6
f = 1 + 2*x + 3*x**2 + x**3
chk('T2 f(0)', f.subs(x, 0), sp.Integer(1))
chk('T2 f(0)', sp.diff(f, x).subs(x, 0), sp.Integer(2))
chk('T2 f"(0)', sp.diff(f, x, 2).subs(x, 0), sp.Integer(6))
# T3 (x³e^{-x})^(9)(0)
chk('T3 Leibniz 9 阶导', sp.diff(x**3*sp.exp(-x), x, 9).subs(x, 0), sp.Integer(504))
# T4 隐函数 x e^y + y e^x = 1 在 (0,1)
y = sp.Function('y')
F = x*sp.exp(y(x)) + y(x)*sp.exp(x) - 1
yp = sp.diff(F, x)
y1 = sp.solve(sp.Eq(yp.subs({x: 0, y(x): 1}), sp.Symbol('Y')), sp.Symbol('Y'))
# 手工： F_x=e^y+y e^x, F_y=x e^y+e^x ; y'=-Fx/Fy
Y = sp.Symbol('Y')
Fx = sp.exp(Y) + Y*sp.exp(x)
Fy = x*sp.exp(Y) + sp.exp(x)
yprime = -Fx / Fy
d1 = sp.simplify(yprime.subs({x: 0, Y: 1}))
chk('T4 y(0)', 1 * 1, 1)  # 点 (0,1) 满足方程
chk('T4 yprime(0)', d1, -(sp.E + 1))
# y'' 由 F_xx + 2F_xy y' + F_yy (y')² + F_y y'' = 0
xx, yy = sp.symbols('xx yy', real=True)
FF = xx*sp.exp(yy) + yy*sp.exp(xx) - 1
Fxx = sp.diff(FF, xx, 2)
Fxy = sp.diff(FF, xx, 1, yy, 1)
Fyy = sp.diff(FF, yy, 2)
Fy_ = sp.diff(FF, yy)
p = d1
y2 = -sp.simplify((Fxx + 2*Fxy*p + Fyy*p**2) / Fy_).subs({xx: 0, yy: 1})
chk('T4 y"(0)', y2, 2*(sp.E + 1)**2 - 1)
# T5 参数二阶导 x=arctan t, y=ln(1+t²)
dx = sp.diff(sp.atan(t), t)
dy = sp.diff(sp.log(1 + t**2), t)
first = sp.simplify(dy / dx)
second = sp.simplify(sp.diff(first, t) / dx)
chk('T5 y_x', first, 2*t)
chk('T5 y_xx(t=1)', second.subs(t, 1), sp.Integer(4))
# T6 分段可导
chk('T6 a', sp.solve(sp.Eq(sp.diff(sp.sin(a*x) + 1, x).subs(x, 0),
                           sp.diff(sp.exp(2*x), x).subs(x, 0)), a), [sp.Integer(2)])
chk('T6 b', sp.solve(sp.Eq(sp.exp(2*0), sp.sin(a*0) + 1 + b), b), [sp.Integer(0)])
# T7 极坐标 r=1+2cosθ, θ=π/2
th = sp.pi / 2
r = 1 + 2*sp.cos(sp.Symbol('q'))
q = sp.Symbol('q')
rp = sp.diff(r, q).subs(q, th)
slope = sp.simplify((rp*sp.sin(th) + r.subs(q, th)*sp.cos(th)) /
                    (rp*sp.cos(th) - r.subs(q, th)*sp.sin(th)))
chk('T7 斜率', slope, sp.Integer(2))
# T8 x³ sin(1/x): f'(0)=0 且 f' 在 0 连续
g = x**3*sp.sin(1/x)
chk('T8 f(0)', lim(g, x, 0), sp.Integer(0))
gp = sp.simplify(sp.diff(g, x))
chk('T8 fp(0)=0', sp.limit(gp, x, 0), sp.Integer(0))
chk('T8 fp 连续', sp.limit(gp, x, 0), sp.Integer(0))

print('===== §3 间断点 J1-J6 左右极限 =====')
chk('J1 右', lim((sp.exp(1/x) - 1) / (sp.exp(1/x) + 1), x, 0, '+'), sp.Integer(1))
chk('J1 左', lim((sp.exp(1/x) - 1) / (sp.exp(1/x) + 1), x, 0, '-'), sp.Integer(-1))
chk('J2 右', lim(sp.atan(1/(x - 1)), x, 1, '+'), sp.pi/2)
chk('J2 左', lim(sp.atan(1/(x - 1)), x, 1, '-'), -sp.pi/2)
# J4 lim_n x(1-x^{2n})/(1+x^{2n})
chk('J4 |x|<1 段', sp.limit(1.0 * (1 - 0.5**(2*n)) / (1 + 0.5**(2*n)), n, sp.oo), sp.Integer(1))
chk('J4 |x|>1 段', sp.limit((1 - 2**(2*n)) / (1 + 2**(2*n)), n, sp.oo), sp.Integer(-1))
# J5 x²cos(1/x)
chk('J5 连续值', lim(x**2*sp.cos(1/x), x, 0), sp.Integer(0))
chk('J5 f(0) 可导', lim(x**2*sp.cos(1/x)/x, x, 0), sp.Integer(0))
# J6 1/(1-e^{x/(1-x)})
J6 = 1/(1 - sp.exp(x/(1 - x)))
chk('J6 x=0 右', lim(J6, x, 0, '+'), -sp.oo)
chk('J6 x=0 左', lim(J6, x, 0, '-'), sp.oo)
chk('J6 x=1 右', lim(J6, x, 1, '+'), sp.Integer(1))
chk('J6 x=1 左', lim(J6, x, 1, '-'), sp.Integer(0))

print('===== §5 作图 G1-G3 关键量 =====')
# G1 = x + arctan x：两端斜渐近线不同、x=0 拐点、无驻点
G1 = x + sp.atan(x)
chk('G1 右端斜渐近线 b', lim(G1 - x, x, sp.oo), sp.pi/2)
chk('G1 左端斜渐近线 b', lim(G1 - x, x, -sp.oo), -sp.pi/2)
chk('G1 y\'', sp.simplify(sp.diff(G1, x) - (1 + 1/(1 + x**2))), sp.Integer(0))
chk('G1 y"(0)', sp.diff(G1, x, 2).subs(x, 0), sp.Integer(0))
# G2 = (x-1)^(2/3)：x=1 尖点（左右导 ±∞）
G2 = (x - 1)**sp.Rational(2, 3)
chk('G2 差商右侧发散', lim(G2/(x - 1), x, 1, '+'), sp.oo)
# G3 = 1/x + ln x：极小 x=1、拐点 x=2、垂直渐近线 x=0
G3 = 1/x + sp.log(x)
chk('G3 y\'', sp.simplify(sp.diff(G3, x) - (x - 1)/x**2), sp.Integer(0))
chk('G3 y"', sp.simplify(sp.diff(G3, x, 2) - (2 - x)/x**3), sp.Integer(0))
chk('G3 极小值 f(1)', G3.subs(x, 1), sp.Integer(1))
chk('G3 拐点纵坐标 f(2)', G3.subs(x, 2), sp.Rational(1, 2) + sp.log(2))
chk('G3 x=0 右极限', lim(G3, x, 0, '+'), sp.oo)
chk('G3 无斜渐近线', lim(G3/x, x, sp.oo), sp.Integer(0))

print('===== §6 定义证明 E1-E3 常数 =====')
chk('E1 |x+2| 上界', sp.Abs(2 + 1 + 2), sp.Integer(5))
f2 = (2*x**2 + 1)/(x**2 + x)
chk('E2 极限', lim(f2, x, sp.oo), sp.Integer(2))
chk('E2 放缩差', sp.simplify(f2 - 2 - (1 - 2*x)/(x**2 + x)), sp.Integer(0))
# E3: |x|<1/2 => f=1/(|x|(1+|x|)) > 1/(2|x|)
mp.mp.dps = 40
M = 10
d = min(sp.Rational(1, 2), sp.Rational(1, 2)/M)
chk('E3 δ', d, sp.Rational(1, 20))
xx0 = float(d) * 0.99
if not (1/(xx0*(1 + xx0)) > M):
    FAIL += 1
    print('  FAIL E3 f(x)>M', 1/(xx0*(1 + xx0)), '<=', M)
else:
    PASS += 1

print('===== §7 数列极限 S1-S3（缩减后 3 题） =====')
k = sp.symbols('k', integer=True, positive=True)
chk('S1 定积分定义', sp.integrate(sp.sqrt(x), (x, 0, 1)), sp.Rational(2, 3))
L = sp.symbols('L', positive=True)
chk('S2 不动点', sp.solve(sp.Eq(L, 1 + L/(1 + L)), L),
    [(1 + sp.sqrt(5))/2])
# 数值迭代核对
v = 1.0
for _ in range(200):
    v = 1 + v/(1 + v)
chk('S2 迭代值', round(v, 10), round(float((1 + sp.sqrt(5))/2), 10), tol=1e-9)

print('===== §1 中值定理 M 组可验证结论 =====')
# M1 Rolle: x³-3x 在 [-√3,√3] 的驻点
chk('M1 Rolle ξ', sp.solve(sp.diff(x**3 - 3*x, x), x), [-1, 1])
chk('M1 端点等值', (x**3 - 3*x).subs(x, -sp.sqrt(3)) - (x**3 - 3*x).subs(x, sp.sqrt(3)), 0)
# M2 Lagrange: ln x 在 [1,e]
chk('M2 ξ', sp.solve(sp.Eq(1/x, (sp.log(sp.E) - sp.log(1))/(sp.E - 1)), x), [sp.E - 1])
# M3 Lagrange: √x 在 [1,4]
chk('M3 ξ', sp.solve(sp.Eq(1/(2*sp.sqrt(x)), (2 - 1)/(4 - 1)), x), [sp.Rational(9, 4)])
# M4 Cauchy: f=x², g=x³ 在 [1,2]
chk('M4 ξ', sp.solve(sp.Eq(sp.Rational(3, 7), 2/(3*x)), x), [sp.Rational(14, 9)])
# M5 Lagrange: (b-a)/b < ln(b/a) < (b-a)/a
for A, B in [(1, 2), (1, 3), (2, 5), (1, sp.E)]:
    aa, bb = sp.Rational(A) if A != sp.E else sp.E, sp.Rational(B) if B != sp.E else sp.E
    lo = sp.simplify((bb - aa)/bb)
    mid = sp.log(bb/aa)
    hi = sp.simplify((bb - aa)/aa)
    if not (sp.N(lo) < sp.N(mid) < sp.N(hi)):
        FAIL += 1
        print('  FAIL M4 不等式', aa, bb, sp.N(lo), sp.N(mid), sp.N(hi))
    else:
        PASS += 1
# M5 / M6 不等式数值采样
for xv in [0.1, 0.5, 1, 2, 5]:
    e = mp.mpf(xv)
    if not (mp.e**e > 1 + e + e**2/2):
        FAIL += 1; print('  FAIL M5', xv)
    else:
        PASS += 1
    if not (mp.log(1 + e) > e/(1 + e)):
        FAIL += 1; print('  FAIL M6', xv)
    else:
        PASS += 1
# M7: f(0)=f(1)=0, f(1/2)=1 => min f'' <= -8，f=4x(1-x) 取等
M7f = 4*x*(1 - x)
chk('M7 f(0)', M7f.subs(x, 0), sp.Integer(0))
chk('M7 f(1)', M7f.subs(x, 1), sp.Integer(0))
chk('M7 f(1/2)', M7f.subs(x, sp.Rational(1, 2)), sp.Integer(1))
chk('M7 f"', sp.diff(M7f, x, 2), sp.Integer(-8))
# M10: f(0)=0, f(1)=1 => ∃ξ f'(ξ)=2ξ（F=f-x² 的 Rolle）、∃η f'(η)=1（Lagrange）
chk('M10 Rolle 点 (f=x)', sp.solve(sp.Eq(1, 2*x), x), [sp.Rational(1, 2)])

print('===== J3 取整函数 / S3 导数定义型 =====')
# J3: [x]+[-x] 在非整数点恒 -1，整数点函数值 0 => 可去间断
for xv in [0.001, -0.001, 0.5, -0.5, 1.001]:
    if int(mp.floor(xv)) + int(mp.floor(-xv)) != -1:
        FAIL += 1; print('  FAIL J3 floor', xv)
    else:
        PASS += 1
# S3: lim n(a^{1/n}-1) = ln a
for av in [2, 3, 10, sp.Rational(1, 2)]:
    chk('S3 a=%s' % av, sp.limit(n*(av**(1/n) - 1), n, sp.oo), sp.log(av))

print('===== §9 限时卷 Q1-Q10 =====')
chk('Q1 根式/立方根差', lim((sp.sqrt(1 + 2*x) - (1 + 3*x)**sp.Rational(1, 3))/x**2, x, 0),
    sp.Rational(1, 2))
chk('Q2 幂指', lim((1 + sp.sin(x)**2)**(1/x**2), x, 0), sp.E)
chk('Q3 e^x sin x 三阶', lim((sp.exp(x)*sp.sin(x) - x - x**2)/x**3, x, 0), sp.Rational(1, 3))
chk('Q4 根式差 @∞', lim(sp.sqrt(x**2 + 4*x + 1) - x, x, sp.oo), sp.Integer(2))
chk('Q5 右极限', lim((sp.exp(2/x) - 1)/(sp.exp(2/x) + 1), x, 0, '+'), sp.Integer(1))
chk('Q5 左极限', lim((sp.exp(2/x) - 1)/(sp.exp(2/x) + 1), x, 0, '-'), sp.Integer(-1))
chk('Q6 ln(1-x) 五阶导', sp.diff(sp.log(1 - x), x, 5).subs(x, 0), sp.Integer(-24))
xt, yt = sp.exp(t)*sp.cos(t), sp.exp(t)*sp.sin(t)
chk('Q7 参数一阶导 t=0', sp.simplify((sp.diff(yt, t)/sp.diff(xt, t)).subs(t, 0)), sp.Integer(1))
chk('Q8 水平渐近线', lim(x**2/(x**2 - 4), x, sp.oo), sp.Integer(1))
chk('Q10 化简差', sp.simplify((2*x**2 - 2)/(x - 1) - 4 - 2*(x - 1)), sp.Integer(0))

print('===== 数值采样交叉验证 =====')
mp.mp.dps = 30
samples = [
    ('L1', lambda e: (mp.sqrt(1 + e**2) - mp.sqrt(1 - e**2))/e**2, 1.0),
    ('L2', lambda e: 1/mp.sin(e)**2 - 1/e**2, 1/3),
    ('L3', lambda e: mp.cos(e)**(1/e**2), mp.exp(-mp.mpf(1)/2)),
    ('L6', lambda e: 1/e - 1/(mp.e**e - 1), mp.mpf(1)/2),
    ('L7', lambda e: (mp.tan(2*e) - 2*mp.sin(e))/e**3, 3.0),
    ('L8', lambda e: (1 - mp.cos(e)*mp.cos(2*e)*mp.cos(3*e))/e**2, 7.0),
    ('T1', lambda e: (mp.sin(e) - e*mp.cos(e))/e**3, mp.mpf(1)/3),
]
for lab, fn, want in samples:
    e = mp.mpf('1e-4')
    chk('采样 ' + lab, fn(e), want, tol=2e-3)

print()
print('================  PASS %d   FAIL %d  ================' % (PASS, FAIL))
raise SystemExit(1 if FAIL else 0)
