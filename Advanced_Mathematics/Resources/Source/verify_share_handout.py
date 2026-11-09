# -*- coding: utf-8 -*-
"""《高等数学 I（A 类）期中复习讲义 · 分享版》例题答案独立复核。

先于讲义写成：所有例题精讲的可计算结论由 sympy 符号推导 / mpmath 高精度
数值独立给出，与人工推导的期望值比对。全绿（FAIL=0）才允许 build_share_handout.py
生成 LaTeX。运行：PYTHONIOENCODING=utf-8 python verify_share_handout.py
"""
import sys
import sympy as sp
import mpmath as mp

x, t, Y = sp.symbols('x t Y', real=True)
N = sp.symbols('N', integer=True, positive=True)
mp.mp.dps = 40

PASS = 0
FAIL = 0


def chk(label, got, want, tol=None):
    """符号/数值/列表三种比对；任一异常都记 FAIL，不放过。"""
    global PASS, FAIL
    ok = False
    try:
        if getattr(got, 'is_infinite', False) or getattr(want, 'is_infinite', False):
            # ±oo / zoo 不能走「差为零」，直接比较符号对象
            ok = got == want
        elif isinstance(got, bool) or isinstance(want, bool):
            ok = bool(got) == bool(want)
        elif isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
            ok = len(got) == len(want) and all(
                sp.simplify(sp.sympify(g) - sp.sympify(w)) == 0
                for g, w in zip(got, want))
        elif tol is not None:
            ok = abs(mp.mpf(str(got)) - mp.mpf(str(want))) <= tol
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception as exc:                                  # noqa: BLE001
        print('  [EXC] %s: %r' % (label, exc))
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print('FAIL  %-46s got=%s  want=%s' % (label, got, want))
    return ok


def true(label, cond):
    global PASS, FAIL
    if bool(cond):
        PASS += 1
    else:
        FAIL += 1
        print('FAIL  %-46s (assertion false)' % label)
    return bool(cond)


def coef(f, k):
    """f 在 x=0 的 x^k 系数（独立用 series 求）。"""
    return sp.simplify(sp.series(f, x, 0, k + 2).removeO().coeff(x, k))


# ============================================================
# 第 1 章 · 极限计算与无穷小阶
# ============================================================
chk('E1.1 (sin x - x)/x^3', sp.limit((sp.sin(x) - x) / x**3, x, 0), sp.Rational(-1, 6))
chk('E1.2 (e^x sin x - x(1+x))/x^3',
    sp.limit((sp.exp(x) * sp.sin(x) - x * (1 + x)) / x**3, x, 0), sp.Rational(1, 3))
chk('E1.3 sqrt(x^2+3x)-x', sp.limit(sp.sqrt(x**2 + 3 * x) - x, x, sp.oo), sp.Rational(3, 2))
chk('E1.4 ln(1+x^2)/(1-cos x)',
    sp.limit(sp.log(1 + x**2) / (1 - sp.cos(x)), x, 0), sp.Integer(2))
chk('E1.4 原极限 = e^2', sp.exp(sp.limit(sp.log(1 + x**2) / (1 - sp.cos(x)), x, 0)), sp.exp(2))
chk('E1.5 sqrt(1+x^2)-cos x 是 2 阶，主部 x^2',
    sp.limit((sp.sqrt(1 + x**2) - sp.cos(x)) / x**2, x, 0), sp.Integer(1))
chk('E1.5 与 x^4 之比发散（确认不是 4 阶）',
    sp.simplify(sp.series(sp.sqrt(1 + x**2) - sp.cos(x), x, 0, 6).removeO().coeff(x, 4)),
    -sp.Rational(1, 6))

# ============================================================
# 第 2 章 · 极限定义证明（δ(ε) / X(ε) / δ(M) 有效性数值检验）
# ============================================================
chk('E2.1 lim(3x-1)=5', sp.limit(3 * x - 1, x, 2), sp.Integer(5))
chk('E2.2 lim (x^2-4)/(x-2)=4', sp.limit((x**2 - 4) / (x - 2), x, 2), sp.Integer(4))
chk('E2.3 lim (2x+1)/(x-1)=2', sp.limit((2 * x + 1) / (x - 1), x, sp.oo), sp.Integer(2))
chk('E2.3 |f-2| = 3/(x-1)', sp.simplify((2 * x + 1) / (x - 1) - 2), 3 / (x - 1))
chk('E2.4 lim 1/x^2 (x->0+)', sp.limit(1 / x**2, x, 0, '+'), sp.oo)

def quant_delta(label, fnum, x0, L, delta, eps_list, side='two'):
    """对若干 ε 与若干 |u|<1 采样，验证 0<|x-x0|<δ ⇒ |f(x)-L|<ε。"""
    good = True
    for e in eps_list:
        e = mp.mpf(e)
        d = delta(e)
        if d <= 0:
            good = False
        for u in (mp.mpf('-0.9'), mp.mpf('-0.5'), mp.mpf('0.5'), mp.mpf('0.9')):
            xv = mp.mpf(x0) + d * u
            if abs(fnum(xv) - mp.mpf(L)) >= e:
                good = False
    return true(label, good)


EPS = ['0.1', '0.01', '0.001', '1e-5']
quant_delta('E2.1 δ=ε/3 满足 ε-δ 定义', lambda v: 3 * v - 1, 2, 5,
            lambda e: e / 3, EPS)
quant_delta('E2.2 δ=ε 满足 ε-δ 定义',
            lambda v: (v * v - 4) / (v - 2), 2, 4, lambda e: e, EPS)

def quant_X(label, fnum, X_of_eps, L, eps_list):
    good = True
    for e in eps_list:
        e = mp.mpf(e)
        Xv = X_of_eps(e)
        for fac in (mp.mpf('1.01'), mp.mpf('2'), mp.mpf('10')):
            xv = Xv * fac
            if abs(fnum(xv) - mp.mpf(L)) >= e:
                good = False
    return true(label, good)


quant_X('E2.3 X=1+3/ε 满足 ε-X 定义',
        lambda v: (2 * v + 1) / (v - 1), lambda e: 1 + 3 / e, 2, EPS)

def quant_M(label, fnum, delta_of_M, M_list):
    good = True
    for Mv in M_list:
        Mv = mp.mpf(Mv)
        d = delta_of_M(Mv)
        for u in (mp.mpf('0.1'), mp.mpf('0.5'), mp.mpf('0.9')):
            xv = d * u
            if fnum(xv) <= Mv:
                good = False
    return true(label, good)


quant_M('E2.4 δ=1/sqrt(M) 满足 M 型定义',
        lambda v: 1 / v**2, lambda Mv: 1 / mp.sqrt(Mv),
        ['10', '100', '1000', '1e5'])

# ============================================================
# 第 3 章 · 连续、间断点与分段 / 极坐标
# ============================================================
f3 = sp.sin(sp.pi * x) / (x**2 - x)
chk('E3.1 x=0 处极限', sp.limit(f3, x, 0), -sp.pi)
chk('E3.1 x=1 处极限', sp.limit(f3, x, 1), -sp.pi)
chk('E3.1 x=0 左极限', sp.limit(f3, x, 0, '-'), -sp.pi)
chk('E3.2 arctan(1/x) 左极限', sp.limit(sp.atan(1 / x), x, 0, '-'), -sp.pi / 2)
chk('E3.2 arctan(1/x) 右极限', sp.limit(sp.atan(1 / x), x, 0, '+'), sp.pi / 2)
chk('E3.2 跳跃量', sp.limit(sp.atan(1 / x), x, 0, '+')
    - sp.limit(sp.atan(1 / x), x, 0, '-'), sp.pi)
chk('E3.3 a=0,b=1 时右导数', sp.limit(((x**2 + 1) - 1) / x, x, 0, '+'), sp.Integer(0))
chk('E3.3 a=0,b=1 时左导数', sp.diff(sp.Integer(1), x), sp.Integer(0))
true('E3.3 a=0,b=1 时连续', sp.limit(x**2 + 1, x, 0, '+') == 1)

th = sp.symbols('th', real=True)
Xc = (1 + sp.cos(th)) * sp.cos(th)
Yc = (1 + sp.cos(th)) * sp.sin(th)
slope = sp.simplify(sp.diff(Yc, th) / sp.diff(Xc, th))
chk('E3.4 心形线 θ=π/2 切线斜率', sp.simplify(slope.subs(th, sp.pi / 2)), sp.Integer(1))
chk('E3.4 该点坐标 x', Xc.subs(th, sp.pi / 2), sp.Integer(0))
chk('E3.4 该点坐标 y', Yc.subs(th, sp.pi / 2), sp.Integer(1))

# ============================================================
# 第 4 章 · 导数与微分
# ============================================================
F = x**2 + x * Y + Y**2 - 3
Fx, FY = sp.diff(F, x), sp.diff(F, Y)
yp = sp.simplify(-Fx / FY)
chk('E4.1 隐函数 (1,1) 处 y\'', sp.simplify(yp.subs({x: 1, Y: 1})), -sp.Integer(1))
ypp = sp.simplify(-(sp.diff(F, x, 2) + 2 * sp.diff(F, x, Y) * yp
                    + sp.diff(F, Y, 2) * yp**2) / FY)
chk('E4.1 隐函数 (1,1) 处 y\'\'',
    sp.simplify(ypp.subs({x: 1, Y: 1})), -sp.Rational(2, 3))

Xp, Yp = t - sp.sin(t), 1 - sp.cos(t)
d1 = sp.diff(Yp, t) / sp.diff(Xp, t)
d2 = sp.simplify(sp.diff(d1, t) / sp.diff(Xp, t))
chk('E4.2 摆线 t=π/2 一阶导', sp.simplify(d1.subs(t, sp.pi / 2)), sp.Integer(1))
chk('E4.2 摆线 t=π/2 二阶导', sp.simplify(d2.subs(t, sp.pi / 2)), -sp.Integer(1))

leib = x**2 * sp.exp(x)
chk('E4.3 f^(5)(0)=5·4', sp.diff(leib, x, 5).subs(x, 0), sp.Integer(20))
for k in range(2, 9):
    chk('E4.3 通项 n(n-1) @ n=%d' % k,
        sp.diff(leib, x, k).subs(x, 0), sp.Integer(k * (k - 1)))

chk('E4.4 a=2,b=-1 时右极限', sp.limit(2 * x - 1, x, 1, '+'), sp.Integer(1))
chk('E4.4 a=2,b=-1 时右导数', sp.limit((2 * x - 1 - 1) / (x - 1), x, 1, '+'), sp.Integer(2))
chk('E4.4 a=2,b=-1 时左导数', sp.limit((x**2 - 1) / (x - 1), x, 1, '-'), sp.Integer(2))

# ============================================================
# 第 5 章 · 泰勒公式与高阶导数
# ============================================================
chk('E5.1 sqrt(1+x) 前 5 项系数',
    [coef(sp.sqrt(1 + x), k) for k in range(5)],
    [1, sp.Rational(1, 2), -sp.Rational(1, 8), sp.Rational(1, 16), -sp.Rational(5, 128)])
chk('E5.2 (cos x - e^{-x^2/2})/x^4',
    sp.limit((sp.cos(x) - sp.exp(-x**2 / 2)) / x**4, x, 0), -sp.Rational(1, 12))
chk('E5.3 sqrt(1+x+x^2) 的 x^3 系数',
    coef(sp.sqrt(1 + x + x**2), 3), -sp.Rational(3, 16))
chk('E5.4 (1+x)^{1/4} 的 f\'\'\'(0)',
    sp.diff((1 + x)**sp.Rational(1, 4), x, 3).subs(x, 0), sp.Rational(21, 64))
chk('E5.4 用通项公式复核 α(α-1)(α-2)',
    sp.Rational(1, 4) * (sp.Rational(1, 4) - 1) * (sp.Rational(1, 4) - 2),
    sp.Rational(21, 64))

# ============================================================
# 第 6 章 · 中值定理、不等式与最值
# ============================================================
g6 = x**3 - 3 * x + 1
chk('E6.1 f(0)', g6.subs(x, 0), sp.Integer(1))
chk('E6.1 f(1)', g6.subs(x, 1), -sp.Integer(1))
true('E6.1 f\'=3(x^2-1)<0 on (0,1)',
     all(sp.diff(g6, x).subs(x, v) < 0 for v in ('0.1', '0.5', '0.9')))

ineq = sp.log(1 + x) - x / (1 + x)
chk('E6.2 g\' = x/(1+x)^2', sp.simplify(sp.diff(ineq, x)), x / (1 + x)**2)
true('E6.2 g(x)>0 对 x>0 数值抽检',
     all(mp.log(1 + mp.mpf(v)) - mp.mpf(v) / (1 + mp.mpf(v)) > 0
         for v in ('0.1', '0.5', '1', '2', '10', '100')))

h6 = x**3 - 3 * x
vals6 = [h6.subs(x, v) for v in (-2, -1, 1, 2)]
chk('E6.3 [-2,2] 最大值', max(vals6), sp.Integer(2))
chk('E6.3 [-2,2] 最小值', min(vals6), -sp.Integer(2))
chk('E6.3 临界点', sp.solve(sp.diff(h6, x), x), [-1, 1])

true('E6.4 2x/π < sin x < x on (0,π/2)',
     all(2 * mp.mpf(v) / mp.pi < mp.sin(mp.mpf(v)) < mp.mpf(v)
         for v in ('0.2', '0.5', '1.0', '1.4')))

# ============================================================
# 第 7 章 · 函数性态与作图七步
# ============================================================
f7 = x**2 / (x - 1)
chk('E7.1 y\'', sp.simplify(sp.diff(f7, x)), x * (x - 2) / (x - 1)**2)
chk('E7.1 y\'\'', sp.simplify(sp.diff(f7, x, 2)), 2 / (x - 1)**3)
chk('E7.1 斜渐近线 k', sp.limit(f7 / x, x, sp.oo), sp.Integer(1))
chk('E7.1 斜渐近线 b', sp.limit(f7 - x, x, sp.oo), sp.Integer(1))
chk('E7.1 极大值 f(0)', f7.subs(x, 0), sp.Integer(0))
chk('E7.1 极小值 f(2)', f7.subs(x, 2), sp.Integer(4))
chk('E7.1 驻点', sp.solve(sp.diff(f7, x), x), [0, 2])

e7 = x * sp.exp(-x)
chk('E7.2 极大点', sp.solve(sp.diff(e7, x), x), [1])
chk('E7.2 极大值', sp.simplify(e7.subs(x, 1)), sp.exp(-1))
chk('E7.2 拐点', sp.solve(sp.diff(e7, x, 2), x), [2])
chk('E7.2 拐点处函数值', sp.simplify(e7.subs(x, 2)), 2 * sp.exp(-2))
chk('E7.2 水平渐近线', sp.limit(e7, x, sp.oo), sp.Integer(0))

s7 = (x**2 + 2) / (x - 1)
chk('E7.3 斜渐近线 k', sp.limit(s7 / x, x, sp.oo), sp.Integer(1))
chk('E7.3 斜渐近线 b', sp.limit(s7 - x, x, sp.oo), sp.Integer(1))
c7 = x**3 / (x**2 - 1)
chk('E7.4 斜渐近线 k', sp.limit(c7 / x, x, sp.oo), sp.Integer(1))
chk('E7.4 斜渐近线 b', sp.limit(c7 - x, x, sp.oo), sp.Integer(0))
chk('E7.4 竖直渐近线 x=1 右极限', sp.limit(c7, x, 1, '+'), sp.oo)
chk('E7.4 竖直渐近线 x=-1 左极限', sp.limit(c7, x, -1, '-'), -sp.oo)

# ============================================================
# 第 8 章 · 函数方程、迭代与数列极限
# ============================================================
def iterate_sqrt(a0, steps):
    v = mp.mpf(a0)
    for _ in range(steps):
        v = mp.sqrt(2 + v)
    return v


chk('E8.1 a_{n+1}=sqrt(2+a_n) 从 a1=0 收敛到 2',
    iterate_sqrt(0, 300), mp.mpf(2), tol=mp.mpf('1e-30'))
true('E8.1 a_n 单调递增且 <2',
     all(iterate_sqrt(0, k) < iterate_sqrt(0, k + 1) < 2 for k in range(1, 12)))

k = sp.symbols('k', integer=True, positive=True)
sum_expr = sp.summation(k / N**2, (k, 1, N))
chk('E8.2 sum_{k=1}^n k/n^2 = (n+1)/(2n)', sp.simplify(sum_expr), (N + 1) / (2 * N))
chk('E8.2 极限', sp.limit(sum_expr, N, sp.oo), sp.Rational(1, 2))

def harm_diff(nv):
    return mp.fsum([mp.mpf(1) / j for j in range(nv + 1, 2 * nv + 1)])


chk('E8.4 1/(n+1)+...+1/(2n) -> ln 2 (n=20000)',
    harm_diff(20000), mp.log(2), tol=mp.mpf('3e-5'))

def stolz_ratio(nv):
    Hn = mp.fsum([mp.mpf(1) / j for j in range(1, nv + 1)])
    return Hn / mp.log(nv)


# Stolz 的实质是差分比：(H_{n+1}-H_n)/(ln(n+1)-ln n) = 1/((n+1)ln(1+1/n)) -> 1
chk('E8.3 Stolz 差分比极限',
    sp.limit((1 / (N + 1)) / sp.log(1 + 1 / N), N, sp.oo), sp.Integer(1))
# H_n/ln n 本身收敛很慢（H_n = ln n + γ + o(1)，n=2e5 时 γ/ln n ≈ 0.047），
# 只作趋势佐证，容差按量级给，不拿它当 Stolz 的判据
true('E8.3 H_n/ln n 数值趋势（容差按 γ/ln n 量级给）',
     abs(stolz_ratio(200000) - 1) < mp.mpf('0.06'))

# ============================================================
print('\n' + '=' * 62)
print('分享版讲义例题复核：PASS %d / FAIL %d' % (PASS, FAIL))
print('=' * 62)
if FAIL:
    print('!! 复核未全绿，禁止生成讲义')
    sys.exit(1)
print('全绿：可以生成讲义')
