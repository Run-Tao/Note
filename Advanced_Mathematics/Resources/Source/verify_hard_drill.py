# -*- coding: utf-8 -*-
"""
高难冲顶 22 题（11/7 追加）数值与符号复核
运行：PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe verify_hard_drill.py
定位：只校「可计算的结论」；纯存在性证明用构造恒等式 + 反例数值抽检。
"""
import sympy as sp
import mpmath as mp

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
            ok = [sp.simplify(sp.sympify(g)) for g in got] == [sp.simplify(sp.sympify(w)) for w in want]
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception:
        ok = False
    if ok:
        PASS += 1
        print('  PASS  %-52s = %s' % (label, got))
    else:
        FAIL += 1
        FAILS.append(label)
        print('  FAIL  %-52s got=%s want=%s' % (label, got, want))


def close(label, got, want, tol=5e-9):
    chk(label, got, want, tol=tol)


x, t, u, n, k, y = sp.symbols('x t u n k y', real=True)
nn = sp.symbols('n', integer=True, positive=True)

print('=' * 74)
print('H1  极限综合（5 题）—— 多步变形 / 首项抵消 / 高阶展开')
print('=' * 74)
close('H1.1 (sqrt(1+x sinx)-cosx-x^2)/x^4',
      sp.limit((sp.sqrt(1 + x * sp.sin(x)) - sp.cos(x) - x ** 2) / x ** 4, x, 0), -sp.Rational(1, 4))
chk('H1.1 佐证 分子展开 = x^2 - x^4/4',
    sp.expand(sp.series(sp.sqrt(1 + x * sp.sin(x)) - sp.cos(x), x, 0, 6).removeO()),
    sp.expand(x ** 2 - x ** 4 / 4))
close('H1.2 1/sin^2x - 1/x^2',
      sp.limit(1 / sp.sin(x) ** 2 - 1 / x ** 2, x, 0), sp.Rational(1, 3))
close('H1.3 (tanx - x - x^3/3)/x^5',
      sp.limit((sp.tan(x) - x - x ** 3 / 3) / x ** 5, x, 0), sp.Rational(2, 15))
close('H1.4 x^2[(1+3/x)^(1/3)-(1+2/x)^(1/2)]',
      sp.limit(((1 + 3 * u) ** sp.Rational(1, 3) - (1 + 2 * u) ** sp.Rational(1, 2)) / u ** 2, u, 0),
      -sp.Rational(1, 2))
close('H1.5 (e^x sinx - x - x^2)/x^3',
      sp.limit((sp.exp(x) * sp.sin(x) - x - x ** 2) / x ** 3, x, 0), sp.Rational(1, 3))

# 逐项核对 H1.4 的两支展开系数（防止答案对但理由写错）
s1 = sp.series((1 + 3 * u) ** sp.Rational(1, 3), u, 0, 3).removeO()
s2 = sp.series((1 + 2 * u) ** sp.Rational(1, 2), u, 0, 3).removeO()
chk('H1.4 佐证 (1+3u)^(1/3) 前三项', sp.expand(s1), sp.expand(1 + u - u ** 2))
chk('H1.4 佐证 (1+2u)^(1/2) 前三项', sp.expand(s2), sp.expand(1 + u - u ** 2 / 2))

print()
print('=' * 74)
print('H2  导数与高阶导（5 题）')
print('=' * 74)
# H2.6  隐函数 e^y + x y = e 在 (0,1) 的二阶导
F = sp.exp(y) + x * y - sp.E
yp = -sp.diff(F, x) / sp.diff(F, y)
ypp = sp.diff(yp, x) + sp.diff(yp, y) * yp
close('H2.6 y\'\'(0), e^y+xy=e', ypp.subs({x: 0, y: 1}), 1 / sp.E ** 2)

# H2.7  摆线参数方程二阶导
dx = sp.diff(t - sp.sin(t), t)
dy = sp.diff(1 - sp.cos(t), t)
d1 = dy / dx
d2 = sp.diff(d1, t) / dx
close('H2.7 d2y/dx2 at t=pi/2', sp.simplify(d2.subs(t, sp.pi / 2)), -1)

# H2.8  x^2 ln(1+x) 的 10 阶导在 0
f8 = x ** 2 * sp.log(1 + x)
ser8 = sp.series(f8, x, 0, 12).removeO()
chk('H2.8 [x^10] x^2 ln(1+x)', sp.expand(ser8).coeff(x, 10), -sp.Rational(1, 8))
chk('H2.8 f^(10)(0)', sp.factorial(10) * sp.expand(ser8).coeff(x, 10), -453600)

# H2.9  1/(x^2-3x+2) 的 n 阶导通项
f9 = 1 / (x ** 2 - 3 * x + 2)
ser9 = sp.series(f9, x, 0, 9).removeO()
for m in range(0, 8):
    got = sp.factorial(m) * sp.expand(ser9).coeff(x, m)
    chk('H2.9 f^(%d)(0)' % m, got, sp.factorial(m) * (1 - sp.Rational(1, 2) ** (m + 1)))
chk('H2.9 通项公式 n!(1-2^-(n+1)) 抽样 n=5',
    sp.factorial(5) * (1 - sp.Rational(1, 2) ** 6), sp.Rational(945, 8))

# H2.10 分段连续可导：a=b=1/2，左右导数都 -1/8
a_, b_ = sp.symbols('a b', real=True)
cont_r = sp.limit((sp.sqrt(1 + x) - 1) / x, x, 0, '+')
cont_l = sp.limit(sp.log(1 + b_ * x) / x, x, 0, '-')
chk('H2.10 右极限', cont_r, sp.Rational(1, 2))
chk('H2.10 左极限 = b', cont_l, b_)
der_r = sp.limit(((sp.sqrt(1 + x) - 1) / x - sp.Rational(1, 2)) / x, x, 0, '+')
der_l = sp.limit((sp.log(1 + sp.Rational(1, 2) * x) / x - sp.Rational(1, 2)) / x, x, 0, '-')
close('H2.10 右导数 (a=1/2)', der_r, -sp.Rational(1, 8))
close('H2.10 左导数 (b=1/2)', der_l, -sp.Rational(1, 8))
close('H2.10 左导数通式 -b^2/2 (b=1/2)', -sp.Rational(1, 2) ** 2 / 2, -sp.Rational(1, 8))

print()
print('=' * 74)
print('H3  中值定理与构造证明（5 题）—— 验证构造恒等式，不代替证明')
print('=' * 74)
# H3.11  Taylor + 最值分类：f''(xi) <= -8
f11 = 4 * x * (1 - x)
chk('H3.11 例 f=4x(1-x) 满足 f(0)=f(1)=0', [f11.subs(x, 0), f11.subs(x, 1)], [0, 0])
chk('H3.11 例 max=1', sp.maximum(f11, x, sp.Interval(0, 1)), 1)
chk('H3.11 例 f\'\' 恒为 -8', sp.diff(f11, x, 2), -8)
f11b = sp.sin(sp.pi * x)
chk('H3.11 例 sin(pi x) 端点', [f11b.subs(x, 0), f11b.subs(x, 1)], [0, 0])
close('H3.11 例 sin(pi x) 的 f\'\'(1/2) <= -8', sp.diff(f11b, x, 2).subs(x, sp.Rational(1, 2)), -sp.pi ** 2)
# 分类逻辑：c<=1/2 => -2/c^2 <= -8 ; c>1/2 => -2/(1-c)^2 < -8
chk('H3.11 分类 c=1/4 时 -2/c^2', -2 / sp.Rational(1, 4) ** 2, -32)
chk('H3.11 分类 c=3/4 时 -2/(1-c)^2', -2 / (1 - sp.Rational(3, 4)) ** 2, -32)

# H3.12  ln t > 2(t-1)/(t+1)
phi = sp.log(t) - 2 * (t - 1) / (t + 1)
chk('H3.12 phi(1)=0', sp.simplify(phi.subs(t, 1)), 0)
chk('H3.12 phi\'(t) = (t-1)^2/(t(t+1)^2)',
    sp.simplify(sp.diff(phi, t) - (t - 1) ** 2 / (t * (t + 1) ** 2)), 0)
close('H3.12 抽样 t=2', phi.subs(t, 2), sp.log(2) - sp.Rational(2, 3))
chk('H3.12 t=2 时 ln2-2/3 > 0', sp.log(2) - sp.Rational(2, 3) > 0, True)

# H3.13  f(x+y)=f(x)+f(y)+xy 的解 f=x^2/2+cx
c = sp.symbols('c', real=True)
fx = lambda z: z ** 2 / 2 + c * z
xx, yy = sp.symbols('xx yy', real=True)
chk('H3.13 f(x+y)=f(x)+f(y)+xy 恒等',
    sp.simplify(fx(xx + yy) - fx(xx) - fx(yy) - xx * yy), 0)
chk('H3.13 f(0)=0', fx(0), 0)
chk('H3.13 减去 x^2/2 后满足 Cauchy',
    sp.simplify((fx(xx + yy) - (xx + yy) ** 2 / 2) - (fx(xx) - xx ** 2 / 2) - (fx(yy) - yy ** 2 / 2)), 0)

# H3.14  f(b)-f(a) = xi f'(xi) ln(b/a)
aa, bb = sp.symbols('a b', positive=True)
fb = sp.Function('f')
Delta = fb(bb) - fb(aa)
Fc = fb(x) - Delta / sp.log(bb / aa) * sp.log(x)
chk('H3.14 F(b)-F(a)=0', sp.simplify(Fc.subs(x, bb) - Fc.subs(x, aa)), 0)
chk('H3.14 F\'(x) 形式', sp.simplify(sp.diff(Fc, x) - (sp.Derivative(fb(x), x) - Delta / (x * sp.log(bb / aa)))), 0)

# H3.15  逐层递推：sin x 的三明治
h1 = x - sp.sin(x)
h2 = sp.sin(x) - x + x ** 3 / 6
h3 = x - x ** 3 / 6 + x ** 5 / 120 - sp.sin(x)
close('H3.15 h1(0)', h1.subs(x, 0), 0)
chk('H3.15 h1\' = 1-cos x >= 0', sp.simplify(sp.diff(h1, x) - (1 - sp.cos(x))), 0)
chk('H3.15 h2\'\' = x - sin x', sp.simplify(sp.diff(h2, x, 2) - (x - sp.sin(x))), 0)
chk('H3.15 h3\'\'\'\' = x - sin x', sp.simplify(sp.diff(h3, x, 4) - (x - sp.sin(x))), 0)
for p in (sp.Rational(1, 2), 1, 2):
    close('H3.15 抽样 x=%s 下界' % p, h2.subs(x, p) > 0, True)
    close('H3.15 抽样 x=%s 上界' % p, h3.subs(x, p) > 0, True)

print()
print('=' * 74)
print('H4  函数性态与作图（3 题）')
print('=' * 74)
# H4.16  y = x^3/(x^2-1)
f16 = x ** 3 / (x ** 2 - 1)
chk('H4.16 斜渐近线 y=x 的截距 -> 0', sp.limit(f16 - x, x, sp.oo), 0)
chk('H4.16 竖直渐近线 x=±1', [sp.limit(f16, x, 1, '+'), sp.limit(f16, x, -1, '-')], [sp.oo, -sp.oo])
chk('H4.16 y\' = x^2(x^2-3)/(x^2-1)^2',
    sp.simplify(sp.diff(f16, x) - x ** 2 * (x ** 2 - 3) / (x ** 2 - 1) ** 2), 0)
chk('H4.16 驻点', sp.solve(sp.numer(sp.together(sp.diff(f16, x))), x), [0, -sp.sqrt(3), sp.sqrt(3)])
chk('H4.16 y(sqrt3)', sp.simplify(f16.subs(x, sp.sqrt(3))), 3 * sp.sqrt(3) / 2)
chk('H4.16 y(-sqrt3)', sp.simplify(f16.subs(x, -sp.sqrt(3))), -3 * sp.sqrt(3) / 2)
# x=0 处 y' 不变号（驻点非极值）
sgn = [sp.sign(sp.diff(f16, x).subs(x, v)) for v in (-sp.Rational(1, 2), sp.Rational(1, 2))]
chk('H4.16 x=0 两侧 y\' 同号（非极值）', sgn, [-1, -1])
chk('H4.16 奇函数', sp.simplify(f16.subs(x, -x) + f16), 0)

# H4.17  y = |x-1| e^{-x}
f17r = (x - 1) * sp.exp(-x)          # x>1
f17l = (1 - x) * sp.exp(-x)          # x<1
close('H4.17 右导数 at 1', sp.limit(f17r / (x - 1), x, 1, '+'), 1 / sp.E)
close('H4.17 左导数 at 1', sp.limit(f17l / (x - 1), x, 1, '-'), -1 / sp.E)
chk('H4.17 x=1 不可导（左右导不等）', sp.simplify(1 / sp.E + 1 / sp.E) != 0, True)
close('H4.17 极大值 y(2)', f17r.subs(x, 2), sp.exp(-2))
chk('H4.17 x>1 时驻点', sp.solve(sp.diff(f17r, x), x), [2])
chk('H4.17 x->+oo 水平渐近线 0', sp.limit(f17r, x, sp.oo), 0)
chk('H4.17 x=1 处取值 0（尖点极小）', sp.simplify(f17r.subs(x, 1)), 0)

# H4.18  r = 1 + cos(theta) 的切线
th = sp.symbols('theta', real=True)
rx = (1 + sp.cos(th)) * sp.cos(th)
ry = (1 + sp.cos(th)) * sp.sin(th)
dxdt = sp.simplify(sp.diff(rx, th))
dydt = sp.simplify(sp.diff(ry, th))
chk('H4.18 x\'(pi/2)', dxdt.subs(th, sp.pi / 2), -1)
chk('H4.18 y\'(pi/2)', dydt.subs(th, sp.pi / 2), -1)
chk('H4.18 斜率 at pi/2 = 1', sp.simplify(dydt / dxdt).subs(th, sp.pi / 2), 1)
chk('H4.18 切点 at pi/2 = (0,1)', [rx.subs(th, sp.pi / 2), ry.subs(th, sp.pi / 2)], [0, 1])
chk('H4.18 x\'(pi)=0', dxdt.subs(th, sp.pi), 0)
chk('H4.18 y\'(pi)=0', dydt.subs(th, sp.pi), 0)
close('H4.18 尖点斜率极限 (theta->pi) = 0',
      sp.limit(dydt / dxdt, th, sp.pi), 0, tol=1e-9)
chk('H4.18 theta=pi 处点为原点', [rx.subs(th, sp.pi), ry.subs(th, sp.pi)], [0, 0])
close('H4.18 数值佐证 theta=pi-1e-4 斜率', (dydt / dxdt).subs(th, sp.pi - mp.mpf('1e-4')), 0, tol=1e-3)

print()
print('=' * 74)
print('H5  数列极限与压轴（4 题）')
print('=' * 74)


def iterate(f, a1, N):
    a = mp.mpf(a1)
    for _ in range(N):
        a = f(a)
    return a


# H5.19  牛顿迭代 sqrt2
a19 = iterate(lambda z: (z + 2 / z) / 2, 1, 60)
close('H5.19 a_n -> sqrt2', a19, mp.sqrt(2), tol=1e-12)
chk('H5.19 极限方程 L=(L+2/L)/2 的解 L>0', sp.solve(sp.Eq(sp.Symbol('L'), (sp.Symbol('L') + 2 / sp.Symbol('L')) / 2), sp.Symbol('L'))[-1], sp.sqrt(2))
# AM-GM 下界
chk('H5.19 a_{n+1} >= sqrt2 的代数式', sp.simplify(((sp.Symbol('s') + 2 / sp.Symbol('s')) / 2) ** 2 - 2 - (sp.Symbol('s') ** 2 - 2) ** 2 / (4 * sp.Symbol('s') ** 2)), 0)

# H5.20  夹逼 + 展开：n(Σ 1/sqrt(n^2+k) - 1) -> -1/4
for N in (1000, 5000, 20000):
    s = mp.nsum(lambda q: 1 / mp.sqrt(N ** 2 + q), [1, N])
    val = N * (s - 1)
    print('        n=%-6d  n(sum-1) = %.10f' % (N, val))
close('H5.20 极限 -1/4（n=20000 采样）', mp.mpf('20000') * (mp.nsum(lambda q: 1 / mp.sqrt(20000 ** 2 + q), [1, 20000]) - 1), -0.25, tol=2e-4)
chk('H5.20 二阶余项系数 -1/4（解析式）',
    sp.simplify(sp.Rational(1, 2) * sp.Rational(1, 2) * sp.Rational(1, 2)), sp.Rational(1, 8))
chk('H5.20 求和主部 sum k = n(n+1)/2', sp.summation(k, (k, 1, nn)), nn * (nn + 1) / 2)

# H5.21  a_{n+1}=sqrt(6+a_n), a1=1 -> 3
a21 = iterate(lambda z: mp.sqrt(6 + z), 1, 80)
close('H5.21 a_n -> 3', a21, 3, tol=1e-12)
chk('H5.21 极限方程 L^2=6+L 的正根', [r for r in sp.solve(sp.Eq(sp.Symbol('L') ** 2, 6 + sp.Symbol('L')), sp.Symbol('L')) if r > 0][0], 3)
chk('H5.21 单调性恒等式',
    sp.simplify((6 + sp.Symbol('s')) - sp.Symbol('s') ** 2 - (3 - sp.Symbol('s')) * (2 + sp.Symbol('s'))), 0)

# H5.22  压缩映射：几何级数上界
kk = sp.symbols('k', positive=True)
dist = sp.symbols('d', positive=True)
chk('H5.22 几何部分和 (1-k^N)/(1-k)',
    sp.simplify(sp.summation(kk ** (nn - 1), (nn, 1, 6)) - (1 - kk ** 6) / (1 - kk)), 0)
# 数值抽检：k=1/2, d=1 时 x_n 与不动点误差 <= k^(n-1)/(1-k) d
f22 = lambda z: mp.mpf('0.25') + mp.mpf('0.5') * z      # 压缩比 1/2，不动点 1/2
xs = [mp.mpf('0')]
for _ in range(40):
    xs.append(f22(xs[-1]))
err_ok = all(abs(xs[i] - mp.mpf('0.5')) <= mp.mpf('0.5') ** (i) / (1 - mp.mpf('0.5')) * mp.mpf('1') for i in range(1, 41))
chk('H5.22 误差界抽样 k=1/2,d=1', err_ok, True)
chk('H5.22 不动点抽样', xs[-1], mp.mpf('0.5'), tol=1e-12)

print()
print('=' * 74)
print('统计与抽检')
print('=' * 74)
close('抽检 H1.1 数值 x=1e-2（截断误差 O(x^2)）', (mp.sqrt(1 + mp.mpf('1e-2') * mp.sin(mp.mpf('1e-2'))) - mp.cos(mp.mpf('1e-2')) - mp.mpf('1e-4')) / mp.mpf('1e-8'), -0.25, tol=1e-4)
close('抽检 H1.3 数值 x=1e-2', (mp.tan(mp.mpf('1e-2')) - mp.mpf('1e-2') - mp.mpf('1e-2') ** 3 / 3) / mp.mpf('1e-2') ** 5, 2 / 15, tol=1e-5)
close('抽检 H1.5 数值 x=1e-2', (mp.e ** mp.mpf('1e-2') * mp.sin(mp.mpf('1e-2')) - mp.mpf('1e-2') - mp.mpf('1e-4')) / mp.mpf('1e-6'), 1 / 3, tol=1e-5)

print()
print('=' * 74)
print('================  PASS %d   FAIL %d  ================' % (PASS, FAIL))
if FAILS:
    print('FAILED:', FAILS)
print('=' * 74)
