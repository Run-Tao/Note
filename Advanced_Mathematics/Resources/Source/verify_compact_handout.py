# -*- coding: utf-8 -*-
"""精编本 11-09 的复核：先锁例题答案，再检查四册题库复核脚本仍全 PASS。

工作流约定：复核先于讲义。本脚本在 _build_compact_handout.py 之前运行，
例题讲解里的每个数值结论都在这里用 sympy / mpmath 独立验证。
"""
import os
import subprocess
import sys

import mpmath as mp
import sympy as sp

x = sp.Symbol('x')
n = sp.Symbol('n', integer=True, positive=True)
N_PASS = 0
N_FAIL = 0


def chk(label, got, want, tol=None):
    global N_PASS, N_FAIL
    if tol is None:
        ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    else:
        ok = abs(float(got) - float(want)) < tol
    if ok:
        N_PASS += 1
        print('  PASS  ' + label)
    else:
        N_FAIL += 1
        print('  FAIL  %s: got %s, want %s' % (label, got, want))


def true(label, cond):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print('  PASS  ' + label)
    else:
        N_FAIL += 1
        print('  FAIL  ' + label)


print('=== A. 八道例题的符号 / 数值复核 ===')

# CH1: lim (e^x sin x - x(1+x)) / x^3 = 1/3
f1 = (sp.exp(x) * sp.sin(x) - x * (1 + x)) / x ** 3
chk('CH1 例 极限 = 1/3', sp.limit(f1, x, 0), sp.Rational(1, 3))
chk('CH1 例 展开系数 e^x sin x 的 x^3 项 = 1/3',
    sp.series(sp.exp(x) * sp.sin(x), x, 0, 5).removeO().coeff(x, 3),
    sp.Rational(1, 3))
true('CH1 例 分子前两阶被抵消（x, x^2 系数为 0）',
     sp.series(sp.exp(x) * sp.sin(x) - x * (1 + x), x, 0, 4).removeO().coeff(x, 1) == 0
     and sp.series(sp.exp(x) * sp.sin(x) - x * (1 + x), x, 0, 4).removeO().coeff(x, 2) == 0)

# CH2: 1/x -> 1 (x -> 1), delta = min(1/2, eps/2)
for eps in [sp.Rational(1, 10), sp.Rational(1, 100), sp.Rational(7, 3)]:
    delta = min(sp.Rational(1, 2), eps / 2)
    # 取边界内最坏点 x = 1 - delta（分母更小，误差更大）
    worst = abs(sp.Rational(1, 1) / (1 - delta) - 1)
    true('CH2 例 delta=min(1/2,eps/2) 在 eps=%s 有效' % eps,
         sp.simplify(worst) <= eps)
true('CH2 例 只取 delta=eps 不足（反例 eps=1/10）',
     abs(sp.Rational(1, 1) / (1 - sp.Rational(1, 10)) - 1) > sp.Rational(1, 10))

# CH3: 1/(1+e^{1/x}) 左右极限
g = lambda t: 1 / (1 + mp.e ** (1 / t))
true('CH3 例 右极限 -> 0', abs(g(mp.mpf('1e-6'))) < 1e-5)
true('CH3 例 左极限 -> 1', abs(g(mp.mpf('-1e-6')) - 1) < 1e-5)
true('CH3 例 第一类跳跃（左右极限都存在且不等）',
     abs(g(mp.mpf('1e-7')) - g(mp.mpf('-1e-7'))) > 0.99)

# CH4: x^2 + y^2 = 25 在 (3,4) 的 y''
y = sp.Function('y')
yp1 = -x / y(x)
ypp = sp.simplify(sp.diff(yp1, x).subs(sp.Derivative(y(x), x), yp1))
ypp_val = sp.simplify(ypp.subs({x: 3, y(x): 4}))
chk('CH4 例 y\'\'(3,4) = -25/64', ypp_val, sp.Rational(-25, 64))
chk('CH4 例 y\'(3,4) = -3/4', yp1.subs({x: 3, y(x): 4}), sp.Rational(-3, 4))
chk('CH4 例 通式 y\'\'=-(1+y\'^2)/y 在 (3,4)',
    -(1 + sp.Rational(-3, 4) ** 2) / 4, sp.Rational(-25, 64))

# CH5: (1+x)^{1/3} 的三阶导
h = (1 + x) ** sp.Rational(1, 3)
chk('CH5 例 f\'\'\'(0) = 10/27', sp.diff(h, x, 3).subs(x, 0), sp.Rational(10, 27))
chk('CH5 例 3! [x^3] = 10/27',
    sp.factorial(3) * sp.series(h, x, 0, 4).removeO().coeff(x, 3),
    sp.Rational(10, 27))
chk('CH5 例 alpha(alpha-1)(alpha-2)',
    sp.Rational(1, 3) * sp.Rational(-2, 3) * sp.Rational(-5, 3),
    sp.Rational(10, 27))

# CH6: (b-a)/b < ln(b/a) < (b-a)/a
ok6 = True
for a, b in [(1, 2), (sp.Rational(1, 2), 3), (2, 5), (sp.Rational(7, 3), 11)]:
    lo = (b - a) / b
    mid = sp.log(b / a)
    hi = (b - a) / a
    ok6 = ok6 and bool(sp.N(lo - mid) < 0) and bool(sp.N(mid - hi) < 0)
true('CH6 例 Lagrange 双边不等式（四组 a<b）', ok6)

# CH7: y = x^2/(x-1)
y7 = x ** 2 / (x - 1)
chk('CH7 例 斜渐近线 y=x+1', sp.simplify(y7 - (x + 1)), 1 / (x - 1))
chk('CH7 例 y\' = x(x-2)/(x-1)^2',
    sp.simplify(sp.diff(y7, x) - x * (x - 2) / (x - 1) ** 2), 0)
chk('CH7 例 y\'\' = 2/(x-1)^3',
    sp.simplify(sp.diff(y7, x, 2) - 2 / (x - 1) ** 3), 0)
chk('CH7 例 极大值 y(0)=0', y7.subs(x, 0), 0)
chk('CH7 例 极小值 y(2)=4', y7.subs(x, 2), 4)

# CH8: 迭代 / 数列（W5.5 的 n a_n -> 2 与压缩常数）
mp.mp.dps = 40
# W5.5 原题：a_1=1, a_{n+1}=ln(1+a_n) => a_n ~ 2/n
a = mp.mpf('1.0')
for _ in range(6000 - 1):
    a = mp.log(1 + a)
chk('CH8 例 W5.5 迭代 6000 次后 n*a_n → 2', 6000 * a, 2.0, 1e-2)
true('CH8 例 W5.5 严格递减且 a_n>0', 0 < a < mp.mpf('1.0'))
# 对照：a_{k+1}=a_k/(1+a_k) 的闭式解 a_n=1/n
b = mp.mpf('1.0')
for _ in range(6000 - 1):
    b = b / (1 + b)
chk('CH8 对照迭代 a_n=1/n 的 n*a_n = 1', 6000 * b, 1.0, 1e-25)
true('CH8 例 压缩常数 3-2sqrt2 在 (0,1)',
     0 < 3 - 2 * sp.sqrt(2) < 1)
chk('CH8 例 3-2sqrt2 数值', float(3 - 2 * sp.sqrt(2)), 0.1715728752538099, 1e-12)

print('\n=== B. 四册题库复核脚本回归 ===')
TOOLS = r'D:/Run_Tao/Homework/_review/tools'
for name in ['verify_taylor_drill.py', 'verify_full_topics_drill.py',
             'verify_extra_drill.py', 'verify_web_drill.py']:
    p = os.path.join(TOOLS, name)
    r = subprocess.run([sys.executable, p], cwd=TOOLS,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    ok = (r.returncode == 0 and 'FAIL 0' in out)
    true('%s: rc=0 且 FAIL 0（%s）' %
         (name, [l.strip() for l in out.splitlines() if 'PASS' in l][-1][:48]
          if any('PASS' in l for l in out.splitlines()) else 'no summary'),
         ok)

print('\n=== 汇总 ===')
print('PASS %d / FAIL %d' % (N_PASS, N_FAIL))
sys.exit(1 if N_FAIL else 0)
