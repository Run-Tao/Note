# -*- coding: utf-8 -*-
"""
2026-11-04 新增材料（Level2-6 笔记 / 4 篇题型专篇 / 3 本讲义 / 模拟卷 C）
中所有可计算结论的集中复核。

运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_new_materials.py
口径：全部用 sympy 符号计算；涉及 1^inf、数列极限、中值定理结论的，
      另给闭式推导或高精度数值采样，避免"恒真断言"。
"""
import sys
import sympy as sp

x, t, h = sp.symbols('x t h', real=True)
n = sp.symbols('n', integer=True, positive=True)
PASS = FAIL = 0


def chk(label, got, want, tol=None):
    global PASS, FAIL
    try:
        if tol is not None:
            ok = abs(complex(got) - complex(want)) < tol
        elif isinstance(got, (list, tuple)) or isinstance(want, (list, tuple)):
            # 列表逐项比较（solve 的返回值）
            ok = [str(sp.simplify(g)) for g in got] == [str(sp.simplify(w)) for w in want]
        elif got in (sp.oo, -sp.oo, sp.zoo) or want in (sp.oo, -sp.oo, sp.zoo):
            # 无穷大做差会得到 nan，改用同一对象判断
            ok = (got is want) or (str(got) == str(want))
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception:
        ok = False
    if ok:
        PASS += 1
        print('  PASS  %-52s = %s' % (label, got))
    else:
        FAIL += 1
        print('  FAIL  %-52s got %s, want %s' % (label, got, want))


def lim(expr, var, pt, dir='+'):
    return sp.simplify(sp.limit(expr, var, pt, dir=dir))


print('=== A. Level 2 导数与微分 ===')
# 分段可导求参数：f = a e^{2x} (x<=0), x^2 + b x + 1 (x>0)
a, b = sp.symbols('a b')
left0 = a                      # f(0-)
right0 = sp.Integer(1)         # f(0+)
dleft = 2 * a                  # 左导
dright = b                     # 右导
chk('分段可导 连续性 a', sp.solve(sp.Eq(left0, right0), a)[0], 1)
chk('分段可导 导数相等 b', sp.solve(sp.Eq(dleft.subs(a, 1), dright), b)[0], 2)

# 参数方程 x=t-sin t, y=1-cos t 的二阶导
xp = sp.diff(t - sp.sin(t), t)
yp = sp.diff(1 - sp.cos(t), t)
dydx = sp.simplify(yp / xp)
d2ydx2 = sp.simplify(sp.diff(dydx, t) / xp)
chk('参数方程 d2y/dx2 一般式', d2ydx2, -1 / (1 - sp.cos(t)) ** 2)
chk('参数方程 d2y/dx2 @ t=pi/2', d2ydx2.subs(t, sp.pi / 2), -1)

# 隐函数 x^2+xy+y^2=3 在 (1,1)
F = x ** 2 + x * sp.Symbol('y') + sp.Symbol('y') ** 2 - 3
y = sp.Symbol('y')
Fx, Fy = sp.diff(F, x), sp.diff(F, y)
Fxx, Fxy, Fyy = sp.diff(Fx, x), sp.diff(Fx, y), sp.diff(Fy, y)
y1 = sp.simplify(-Fx / Fy).subs({x: 1, y: 1})
y2 = sp.simplify(-(Fxx * Fy ** 2 - 2 * Fxy * Fx * Fy + Fyy * Fx ** 2) / Fy ** 3).subs({x: 1, y: 1})
chk('隐函数 (1,1) y_prime', y1, -1)
chk('隐函数 (1,1) y_second', y2, sp.Rational(-2, 3))

# Leibniz 代点消项
c10 = sp.factorial(10) * sp.series(x ** 2 * sp.exp(2 * x), x, 0, 11).removeO().coeff(x, 10)
chk('(x^2 e^{2x})^{(10)}(0)', c10, 23040)
c5 = sp.factorial(5) * sp.series(sp.log(1 + 2 * x), x, 0, 6).removeO().coeff(x, 5)
chk('(ln(1+2x))^{(5)}(0)', c5, 768)
al = sp.Rational(1, 3)
chk('(1+x)^{1/3} 的 f^{(4)}(0)', al * (al - 1) * (al - 2) * (al - 3), sp.Rational(-80, 81))

# 极坐标 r=1+cos(theta) 在 theta=pi/2 的切线斜率
th = sp.symbols('theta', real=True)
r = 1 + sp.cos(th)
rp = sp.diff(r, th)
slope = sp.simplify((rp * sp.sin(th) + r * sp.cos(th)) / (rp * sp.cos(th) - r * sp.sin(th)))
chk('极坐标 r=1+cos(theta) 斜率 @ pi/2', slope.subs(th, sp.pi / 2), 1)
chk('极坐标 该点坐标 x', (r * sp.cos(th)).subs(th, sp.pi / 2), 0)
chk('极坐标 该点坐标 y', (r * sp.sin(th)).subs(th, sp.pi / 2), 1)

print()
print('=== B. Level 3 / 极限与泰勒 ===')
chk('tan x - sin x 主部 x^3 系数', lim((sp.tan(x) - sp.sin(x)) / x ** 3, x, 0), sp.Rational(1, 2))
chk('sqrt(1+tan x)-sqrt(1+sin x) / x^3', lim((sp.sqrt(1 + sp.tan(x)) - sp.sqrt(1 + sp.sin(x))) / x ** 3, x, 0), sp.Rational(1, 4))
chk('(tan x - sin x)/(x^2 ln(1+x))', lim((sp.tan(x) - sp.sin(x)) / (x ** 2 * sp.log(1 + x)), x, 0), sp.Rational(1, 2))
chk('(e^{x^2}-cos x)/x^2', lim((sp.exp(x ** 2) - sp.cos(x)) / x ** 2, x, 0), sp.Rational(3, 2))
chk('sqrt(1+x^2)-cos x 的阶（除以 x^2）', lim((sp.sqrt(1 + x ** 2) - sp.cos(x)) / x ** 2, x, 0), 1)
chk('((x+3)/(x+1))^{2x}, x->inf', sp.exp(lim(2 * x * sp.log((x + 3) / (x + 1)), x, sp.oo)), sp.exp(4))
chk('((1+2x)/(1-x))^{1/x}', lim(((1 + 2 * x) / (1 - x)) ** (1 / x), x, 0), sp.exp(3))
# 1^inf 捷径的余项条件示例：(sin x / x)^{1/x^2}
chk('(sin x/x)^{1/x^2}', lim((sp.sin(x) / x) ** (1 / x ** 2), x, 0), sp.exp(-sp.Rational(1, 6)))

print()
print('=== C. 函数性态与作图（模拟卷 C Q17 与讲义例题） ===')
y17 = x ** 2 / (x - 1)
chk('Q17 y = x^2/(x-1) 多项式部分', sp.simplify(sp.div(sp.numer(sp.together(y17)), sp.denom(sp.together(y17)), x)[0]), x + 1)
chk('Q17 斜渐近线截距 b', lim(y17 - x, x, sp.oo), 1)
chk('Q17 y_prime', sp.factor(sp.diff(y17, x)), x * (x - 2) / (x - 1) ** 2)
chk('Q17 y_second', sp.factor(sp.diff(y17, x, 2)), 2 / (x - 1) ** 3)
chk('Q17 极大值 y(0)', y17.subs(x, 0), 0)
chk('Q17 极小值 y(2)', y17.subs(x, 2), 4)
chk('Q17 x->1-', lim(y17, x, 1, '-'), -sp.oo)
chk('Q17 x->1+', lim(y17, x, 1, '+'), sp.oo)

yA = x ** 2 * sp.exp(-x)
chk('y=x^2 e^{-x} 驻点', sp.solve(sp.Eq(sp.diff(yA, x), 0), x), [0, 2])
chk('y=x^2 e^{-x} 拐点数', len([s for s in sp.solve(sp.Eq(sp.diff(yA, x, 2), 0), x) if s.is_real]), 2)
chk('y=x^2 e^{-x} x->+inf', lim(yA, x, sp.oo), 0)
chk('y=x^2 e^{-x} x->-inf', lim(yA, x, -sp.oo), sp.oo)

yB = sp.log(x) / x
chk('y=ln x/x 驻点', sp.solve(sp.Eq(sp.diff(yB, x), 0), x), [sp.E])
chk('y=ln x/x 极大值', sp.simplify(yB.subs(x, sp.E)), 1 / sp.E)
chk('y=ln x/x 水平渐近线', lim(yB, x, sp.oo), 0)

yC = x ** 3 / (x ** 2 + 1)
chk('y=x^3/(x^2+1) 斜渐近线 k', lim(yC / x, x, sp.oo), 1)
chk('y=x^3/(x^2+1) 斜渐近线 b', lim(yC - x, x, sp.oo), 0)

print()
print('=== D. 数列极限 / 新压轴 ===')
k = sp.symbols('k', integer=True, positive=True)
# 夹逼：sum_{k=1}^n 1/sqrt(n^2+k) -> 1
lo = n / sp.sqrt(n ** 2 + n)
hi = n / sp.sqrt(n ** 2 + 1)
chk('夹逼下界极限', lim(lo, n, sp.oo), 1)
chk('夹逼上界极限', lim(hi, n, sp.oo), 1)
# 定积分定义：1/n sum sin(k pi/(2n)) -> 2/pi
chk('定积分定义求极限 解析值', sp.integrate(sp.sin(sp.pi * x / 2), (x, 0, 1)), 2 / sp.pi)
for N in [200, 2000, 20000]:
    v = sum(sp.N(sp.sin(sp.pi * kk / (2 * N)), 20) for kk in range(1, N + 1)) / N
    chk('  数值采样 n=%d' % N, sp.Float(v, 12), sp.N(2 / sp.pi, 12), tol=5e-3)
# 单调有界 a_{n+1}=sqrt(2+a_n)
chk('递推 a_{n+1}=sqrt(2+a_n) 不动点', [s for s in sp.solve(sp.Eq(sp.Symbol('L'), sp.sqrt(2 + sp.Symbol('L'))), sp.Symbol('L')) if s.is_real], [2])
# Newton 压缩 a_{n+1}=(a_n+3/a_n)/2
L = sp.Symbol('L', positive=True)
chk('Newton 迭代极限', sp.solve(sp.Eq(L, (L + 3 / L) / 2), L), [sp.sqrt(3)])
# Stolz
chk('Stolz (1+..+n)/n^2', lim(sp.summation(k, (k, 1, n)) / n ** 2, n, sp.oo), sp.Rational(1, 2))

print()
print('=== E. 模拟卷 C 逐题答案 ===')
chk('选1 sqrt(1+x^2)-cos x 主部系数（阶=2）', lim((sp.sqrt(1 + x ** 2) - sp.cos(x)) / x ** 2, x, 0), 1)
f2 = sp.sin(sp.pi * x) / (x ** 2 - x)
chk('选2 x=0 可去极限', lim(f2, x, 0), -sp.pi)
chk('选2 x=1 可去极限', lim(f2, x, 1), -sp.pi)
chk('填6 (ln(1+2x))^{(5)}(0)', c5, 768)
chk('填7 定积分定义极限', sp.integrate(sp.sin(sp.pi * x / 2), (x, 0, 1)), 2 / sp.pi)
chk('填8 斜渐近线 y=x 的 b', lim(yC - x, x, sp.oo), 0)
chk('填9 (1+x)^{1/3} 四阶导', al * (al - 1) * (al - 2) * (al - 3), sp.Rational(-80, 81))
chk('填10 ((x+3)/(x+1))^{2x}', sp.exp(lim(2 * x * sp.log((x + 3) / (x + 1)), x, sp.oo)), sp.exp(4))
chk('Q11 lim_{x->1}(x+1)/(x+2)', lim((x + 1) / (x + 2), x, 1), sp.Rational(2, 3))
chk('Q12 (tan x-sin x)/(x^2 ln(1+x))', lim((sp.tan(x) - sp.sin(x)) / (x ** 2 * sp.log(1 + x)), x, 0), sp.Rational(1, 2))
chk('Q14 隐函数 y_second', y2, sp.Rational(-2, 3))
# Q16 不等式：g'(x)=x^2/(1+x)>0
g16 = sp.log(1 + x) - x + x ** 2 / 2
chk('Q16 g_prime', sp.factor(sp.diff(g16, x)), x ** 2 / (x + 1))
chk('Q16 g(0)', g16.subs(x, 0), 0)

print()
print('=== F. 讲义_导数计算专题：限时卷 8 题 ===')
a2, b2 = sp.symbols('a2 b2')
chk('T1 分段连续 a', sp.solve(sp.Eq(a2, 2), a2)[0], 2)
chk('T1 分段可导 b', sp.solve(sp.Eq(-a2 * sp.sin(0), b2), b2)[0], 0)  # 左导 0 = 右导 b2
xt = sp.log(1 + t ** 2)
yt = t - sp.atan(t)
d2_t = sp.simplify(sp.diff(sp.diff(yt, t) / sp.diff(xt, t), t) / sp.diff(xt, t))
chk('T2 参数 d2y/dx2', d2_t, (1 + t ** 2) / (4 * t))
F3 = x ** 3 + sp.Symbol('y') ** 3 - 2 * x * sp.Symbol('y')
f3x, f3y = sp.diff(F3, x), sp.diff(F3, sp.Symbol('y'))
f3xx, f3xy, f3yy = sp.diff(f3x, x), sp.diff(f3x, sp.Symbol('y')), sp.diff(f3y, sp.Symbol('y'))
y2_3 = sp.simplify(-(f3xx * f3y ** 2 - 2 * f3xy * f3x * f3y + f3yy * f3x ** 2) / f3y ** 3).subs({x: 1, sp.Symbol('y'): 1})
chk('T3 隐函数 x^3+y^3=2xy @ (1,1) y_second', y2_3, -16)
c8 = sp.factorial(8) * sp.series(x ** 3 * sp.exp(x), x, 0, 9).removeO().coeff(x, 8)
chk('T4 (x^3 e^x)^{(8)}(0)', c8, 336)
r5 = 2 * sp.cos(th)
rp5 = sp.diff(r5, th)
num5 = sp.simplify((rp5 * sp.sin(th) + r5 * sp.cos(th)).subs(th, 0))
den5 = sp.simplify((rp5 * sp.cos(th) - r5 * sp.sin(th)).subs(th, 0))
chk('T5 极坐标 r=2cos(theta) @0 分子', num5, 2)
chk('T5 极坐标 r=2cos(theta) @0 分母（0=竖直切线）', den5, 0)
t6 = sp.diff(sp.atan(sp.exp(2 * x)), x)
chk('T6 (arctan e^{2x})_prime = 1/cosh(2x)', sp.simplify(t6.rewrite(sp.exp) - 2 * sp.exp(2 * x) / (1 + sp.exp(4 * x))), 0)

print()
print('=== G. 讲义_函数性态与作图：例题与限时卷 ===')
yg1 = x / (x ** 2 - 1)
chk('G4 y=x/(x^2-1) 驻点数', len(sp.solve(sp.Eq(sp.diff(yg1, x), 0), x)), 0)
chk('G4 y=x/(x^2-1) 水平渐近线', lim(yg1, x, sp.oo), 0)
chk('G4 y=x/(x^2-1) 拐点 x=0', sp.solve(sp.Eq(sp.diff(yg1, x, 2), 0), x), [0])
yg2 = x + 1 / x
chk('G5 y=x+1/x 驻点', sp.solve(sp.Eq(sp.diff(yg2, x), 0), x), [-1, 1])
chk('G5 y=x+1/x 斜渐近线 b', lim(yg2 - x, x, sp.oo), 0)
ylim1 = (x - 1) ** 2 / (x + 1)
chk('L1 y=(x-1)^2/(x+1) 斜渐近线 k', lim(ylim1 / x, x, sp.oo), 1)
chk('L1 y=(x-1)^2/(x+1) 斜渐近线 b', lim(ylim1 - x, x, sp.oo), -3)
chk('L1 y=(x-1)^2/(x+1) 驻点', sp.solve(sp.Eq(sp.diff(ylim1, x), 0), x), [-3, 1])
chk('L1 y=(x-1)^2/(x+1) 极小值', ylim1.subs(x, 1), 0)
chk('L1 y=(x-1)^2/(x+1) 极大值', ylim1.subs(x, -3), -8)
ylim2 = x * sp.exp(-x ** 2 / 2)
chk('L2 y=x e^{-x^2/2} 驻点', sp.solve(sp.Eq(sp.diff(ylim2, x), 0), x), [-1, 1])
chk('L2 y=x e^{-x^2/2} 拐点数', len(sp.solve(sp.Eq(sp.diff(ylim2, x, 2), 0), x)), 3)
chk('L2 y=x e^{-x^2/2} 水平渐近线 +inf', lim(ylim2, x, sp.oo), 0)
chk('L2 y=x e^{-x^2/2} 水平渐近线 -inf', lim(ylim2, x, -sp.oo), 0)

print()
print('=== H. 讲义_数列极限与压轴新题型 ===')
chk('H3 sum n/(n^2+k^2) 解析值', sp.integrate(1 / (1 + x ** 2), (x, 0, 1)), sp.pi / 4)
chk('H7 sum 1/(n+k) 解析值', sp.integrate(1 / (1 + x), (x, 0, 1)), sp.log(2))
for N in [500, 5000]:
    v7 = sum(sp.N(1 / (N + kk), 20) for kk in range(1, N + 1))
    v3 = sum(sp.N(N / (N ** 2 + kk ** 2), 20) for kk in range(1, N + 1))
    chk('  H3 数值采样 n=%d' % N, sp.Float(v3, 12), sp.N(sp.pi / 4, 12), tol=2e-3)
    chk('  H7 数值采样 n=%d' % N, sp.Float(v7, 12), sp.N(sp.log(2), 12), tol=2e-3)

print()
print('=== I. 讲义_数列极限与压轴新题型 / 模拟卷 C：限时卷 V1--V4 ===')
chk('V1 sum 1/sqrt(n^2+kn) 解析值', sp.integrate(1 / sp.sqrt(1 + x), (x, 0, 1)), 2 * (sp.sqrt(2) - 1))
chk('V2 (1/n)sum (k/n)^2 解析值', sp.integrate(x ** 2, (x, 0, 1)), sp.Rational(1, 3))
Lv = sp.Symbol('Lv', positive=True)
chk('V3 Newton a_{n+1}=(a_n+4/a_n)/2 极限', sp.solve(sp.Eq(Lv, (Lv + 4 / Lv) / 2), Lv), [2])
chk('V4 Stolz (1^2+..+n^2)/n^3', lim(sp.summation(k ** 2, (k, 1, n)) / n ** 3, n, sp.oo), sp.Rational(1, 3))
# Q18 P1 用具体函数验证构造：f=x(1-x) 满足 f(0)=f(1)，等距平移解应为 x=1/4
fz = lambda z: z * (1 - z)
chk('Q18 P1 具体函数 f=x(1-x) 的等距解', sp.solve(sp.Eq(fz(x), fz(x + sp.Rational(1, 2))), x), [sp.Rational(1, 4)])
chk('Q18 P1 端点条件 f(0)=f(1)', sp.simplify(fz(0) - fz(1)), 0)

print()
print('================  PASS %d   FAIL %d  ================' % (PASS, FAIL))
sys.exit(0 if FAIL == 0 else 1)
