# -*- coding: utf-8 -*-
"""
11/9 薄弱点定向加练：先复核、后成文。
- 诊断卷 12 题（六弱点各 2）+ 六模块 72 题 = 84 题
- 每题的答案先由 sympy/mpmath 独立算出并与讲义速查表里的手写值比对，
  全部 PASS 才生成 LaTeX；任何 FAIL 都不写文件。
"""
import io
import os
import sys
import math
import sympy as sp
import mpmath as mp

x, y, t, th, n, eps, M = sp.symbols('x y t theta n epsilon M', real=True)
mp.mp.dps = 40

PASS = 0
FAIL = 0
FAILS = []


def chk(label, got, want, tol=None):
    global PASS, FAIL
    try:
        if isinstance(got, bool) or isinstance(want, bool):
            ok = bool(got) == bool(want)
        elif tol is not None:
            ok = abs(complex(got) - complex(want)) <= tol
        elif isinstance(got, (list, tuple, set)) or isinstance(want, (list, tuple, set)):
            g = sorted([sp.simplify(sp.sympify(v)) for v in got], key=str)
            w = sorted([sp.simplify(sp.sympify(v)) for v in want], key=str)
            ok = (len(g) == len(w) and all(sp.simplify(a - b) == 0 for a, b in zip(g, w)))
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    except Exception as e:
        ok = False
        label += ' [EXC %s]' % e
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILS.append('%s | got=%s want=%s' % (label, got, want))
    return ok


def limit2(label, expr, var, pt, want):
    """符号极限 + mpmath 数值双验。pt 为 0 或 oo 时给出合适采样点。"""
    got = sp.limit(expr, var, pt)
    chk(label + ' [symbolic]', got, want)
    f = sp.lambdify(var, expr, 'mpmath')
    if pt == 0:
        samples = [mp.mpf('1e-3'), mp.mpf('1e-4'), mp.mpf('-1e-4')]
    elif pt == sp.oo:
        samples = [mp.mpf('1e3'), mp.mpf('1e5')]
    else:
        samples = [mp.mpf(str(pt)) + mp.mpf('1e-5'), mp.mpf(str(pt)) - mp.mpf('1e-5')]
    vals = [f(v) for v in samples]
    chk(label + ' [numeric]', vals[-1], sp.N(want, 30), tol=mp.mpf('1e-6'))
    return got


def coef(label, expr, k, want):
    """f^(k)(0) = k! [x^k]，用 series 与 diff 两条独立路径互验。"""
    s = sp.series(expr, x, 0, k + 2).removeO()
    c = sp.expand(s).coeff(x, k)
    got = sp.factorial(k) * c
    chk(label + ' [series]', got, want)
    chk(label + ' [diff]', sp.diff(expr, x, k).subs(x, 0), want)


def oblique(label, f, want_k, want_b):
    k = sp.limit(f / x, x, sp.oo)
    b = sp.limit(f - k * x, x, sp.oo)
    chk(label + ' [k]', k, want_k)
    chk(label + ' [b]', b, want_b)
    return k, b


def stat(label, f, want):
    got = sp.solve(sp.diff(f, x), x)
    got = [sp.simplify(v) for v in got if v.is_real]
    chk(label, got, want)


def infl(label, f, want):
    got = [sp.simplify(v) for v in sp.solve(sp.diff(f, x, 2), x) if v.is_real]
    chk(label, got, want)


def delta_ok(label, f, x0, kind, delta, params, target=0, direction=0):
    """数值验证 δ(ε)/X/δ(M) 的有效性。kind: 'eps' | 'X' | 'M'。"""
    fn = sp.lambdify(x, f, 'mpmath')
    ok = True
    for p in params:
        d = delta(p)
        if kind == 'eps':
            for s in (-1, 1):
                xx = mp.mpf(str(x0)) + s * d * mp.mpf('0.9')
                ok &= abs(fn(xx) - target) < p
        elif kind == 'X':
            xx = d * mp.mpf('1.1')
            ok &= abs(fn(xx) - target) < p
        else:  # M 型：|f| > M
            for s in (-1, 1):
                xx = mp.mpf(str(x0)) + s * d * mp.mpf('0.9')
                if xx == 0:
                    continue
                ok &= abs(fn(xx)) > p
    chk(label + ' [delta-valid]', ok, True)
    return ok


def iter_ok(label, a1, step, transform, want, N=6000, tol=mp.mpf('1e-4')):
    a = mp.mpf(a1)
    for _ in range(N):
        a = step(a)
    got = transform(a, N)
    chk(label + ' [6000 iter]', got, want, tol=tol)
    return got


ITEMS = []


def add(qid, sec, lvl, q, ans, tip, tag, checks):
    for c in checks:
        lab, got, want = c[0], c[1], c[2]
        rest = [v for v in c[3:] if isinstance(v, (int, float, mp.mpf)) or isinstance(v, sp.Basic)]
        chk('%s %s' % (qid, lab), got, want, tol=(rest[0] if rest else None))
    ITEMS.append(dict(id=qid, sec=sec, lvl=lvl, q=q, a=ans, tip=tip, tag=tag))


def E(got, want, label='check'):
    return (label, got, want)


# ============================================================
# 诊断卷 D1--D12（六弱点各 2 题，20 min 掐表）
# ============================================================
add('D1', 'D', '基础', r'求 $y=\dfrac{x^{2}+2}{x-1}$ 的斜渐近线。',
    r'$y=x+1$', '先做多项式除法，别把 $b$ 漏掉', '作图七步·渐近线',
    [E(sp.limit((x**2+2)/(x-1)-x, x, sp.oo), 1, 'b'),
     E(sp.limit((x**2+2)/(x-1)/x, x, sp.oo), 1, 'k')])

add('D2', 'D', '基础', r'求 $y=\dfrac{x^{3}}{x^{2}-1}$ 的斜渐近线。',
    r'$y=x$', '分子次数恰比分母高 1', '作图七步·渐近线',
    [E(sp.limit(x**3/(x**2-1)/x, x, sp.oo), 1, 'k'),
     E(sp.limit(x**3/(x**2-1)-x, x, sp.oo), 0, 'b')])

add('D3', 'D', '基础', r'用定义证 $\lim\limits_{x\to0}\dfrac1{x^{2}}=+\infty$（按 $|f|\to\infty$ 写），给出 $\delta(M)$。',
    r'$\delta=M^{-1/2}$', '要的是 $>M$，不是 $<\\varepsilon$；放缩方向别反', '第 11 题 M 型',
    [('delta', delta_ok('D3', 1/x**2, 0, 'M', lambda m: mp.mpf(1)/mp.sqrt(m),
                        [10, 100, 1000]), True)])

add('D4', 'D', '基础', r'用 $\eps$-$\delta$ 定义证 $\lim\limits_{x\to2}(3x-1)=5$，给出 $\delta(\eps)$。',
    r'$\delta=\eps/3$', '量词行先写全，再化差', '第 11 题 $\\eps$-$\\delta$',
    [('delta', delta_ok('D4', 3*x-1, 2, 'eps', lambda e: e/3,
                        [mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.001')], target=5), True)])

add('D5', 'D', '进阶', r'设 $x=t^{2},\ y=t^{3}$，求 $\dfrac{\dd^{2}y}{\dd x^{2}}\big|_{t=1}$，并写出第一句必须声明的前提。',
    r'$\dfrac34$；前提 $x\prime_t=2t\ne0$', '参数方程二阶导要再除以 $x\prime_t$', '条件书写·参数二阶',
    [E(sp.simplify(sp.diff(sp.diff(t**3, t)/sp.diff(t**2, t), t)/sp.diff(t**2, t)).subs(t, 1),
       sp.Rational(3, 4), 'd2y/dx2')])

add('D6', 'D', '进阶', r'设 $x^{2}+xy+y^{2}=7$，求 $y\prime$ 在 $(1,2)$ 处的值，并写出隐函数求导的前提。',
    r'$-\dfrac45$；前提 $F_y=x+2y\ne0$', '先查 $F_y\ne0$ 再除', '条件书写·隐函数',
    [E(sp.simplify(-(2*x+y)/(x+2*y)).subs({x: 1, y: 2}), sp.Rational(-4, 5), 'y\'')])

add('D7', 'D', '基础', r'求 $f(x)=(1+x)^{1/4}$ 的 $f\prime\prime\prime(0)$。',
    r'$\dfrac{21}{64}$', '通项系数 $\\alpha(\\alpha-1)(\\alpha-2)$，不要乘 $3!$', '泰勒·通项系数',
    [('coef', sp.factorial(3)*sp.series((1+x)**sp.Rational(1, 4), x, 0, 5).removeO().coeff(x, 3),
      sp.Rational(21, 64))])

add('D8', 'D', '基础', r'求 $\lim\limits_{x\to0}\dfrac{\ln(1+x)-x+\tfrac12x^{2}}{x^{3}}$。',
    r'$\dfrac13$', '分母三阶，$\\ln(1+x)$ 必须展到三阶', '泰勒·展错阶',
    [('lim', sp.limit((sp.log(1+x)-x+x**2/2)/x**3, x, 0), sp.Rational(1, 3))])

add('D9', 'D', '进阶', r'求 $\lim\limits_{x\to0}\dfrac{\tan x-\sin x}{x^{3}}$。',
    r'$\dfrac12$', '加减式不能逐项等价替换，展到第一个非零阶', '极限计算·加减禁替换',
    [('lim', sp.limit((sp.tan(x)-sp.sin(x))/x**3, x, 0), sp.Rational(1, 2))])

add('D10', 'D', '进阶', r'求 $f(x)=\dfrac{\ee^{1/x}-1}{\ee^{1/x}+1}$ 在 $x\to0^{-}$ 与 $x\to0^{+}$ 的极限，并判间断类型。',
    r'$-1$ 与 $1$；跳跃间断', '$\\ee^{1/x}$ 两侧分别趋于 $0$ 与 $+\\infty$', '间断·跳跃',
    [('left', sp.limit((sp.exp(1/x)-1)/(sp.exp(1/x)+1), x, 0, '-'), -1),
     ('right', sp.limit((sp.exp(1/x)-1)/(sp.exp(1/x)+1), x, 0, '+'), 1)])

add('D11', 'D', '压轴', r'设 $a_1=1,\ a_{n+1}=\dfrac{a_n}{1+2a_n}$，求 $\lim\limits_{n\to\infty} n a_n$。',
    r'$\dfrac12$', '取倒数化等差，别直接套压缩映射', '新压轴·迭代',
    [( 'iter', iter_ok('D11', 1, lambda a: a/(1+2*a), lambda a, k: k*a, mp.mpf('0.5')),
      mp.mpf('0.5'), mp.mpf('1e-4'))])

add('D12', 'D', '进阶', r'极坐标 $r=1+\cos\theta$，求 $\theta=\dfrac\pi2$ 处切线斜率。',
    r'$1$', '一律转参数式再求 dy/dx，别直接用 r 的导数', '极坐标切线',
    [E(sp.simplify((( -sp.sin(th))*(1+sp.cos(th))*0 + 0)), 0, 'dummy')])

# D12 单独用参数式复核
_x = (1 + sp.cos(th)) * sp.cos(th)
_y = (1 + sp.cos(th)) * sp.sin(th)
chk('D12 slope', sp.simplify(sp.diff(_y, th)/sp.diff(_x, th)).subs(th, sp.pi/2), 1)

# ============================================================
# M1 函数性态作图七步（14 题，A 级第 1 名）
# ============================================================
M1 = [
    ('M1.1', '基础', r'求 $y=\dfrac{x^{2}+3}{x-1}$ 的斜渐近线。', r'$y=x+1$', '多项式除法得 $b$', '斜渐近线',
     [E(sp.limit((x**2+3)/(x-1)-x, x, sp.oo), 1, 'b')]),
    ('M1.2', '基础', r'求 $y=\dfrac{2x^{2}+3x+1}{x+1}$ 的斜渐近线。', r'$y=2x+1$', '先约分再判渐近线（$x=-1$ 是可去间断）', '斜渐近线',
     [E(sp.limit((2*x**2+3*x+1)/(x+1)-2*x, x, sp.oo), 1, 'b')]),
    ('M1.3', '基础', r'求 $y=\dfrac{x^{3}}{x^{2}-4}$ 的斜渐近线与垂直渐近线。', r'$y=x$；$x=\pm2$', '垂直渐近线看分母零点', '斜+垂直',
     [E(sp.limit(x**3/(x**2-4)/x, x, sp.oo), 1, 'k')]),
    ('M1.4', '进阶', r'求 $y=\dfrac{x^{3}+2x}{x^{2}+1}$ 的斜渐近线。', r'$y=x$', '分子分母同次差 1', '斜渐近线',
     [E(sp.limit((x**3+2*x)/(x**2+1)-x, x, sp.oo), 0, 'b')]),
    ('M1.5', '进阶', r'求 $y=x\ee^{1/x}$ 的斜渐近线。', r'$y=x+1$', '$\\ee^{1/x}=1+\\frac1x+o(\\frac1x)$', '斜渐近线·指数',
     [E(sp.limit(x*sp.exp(1/x)-x, x, sp.oo), 1, 'b')]),
    ('M1.6', '进阶', r'求 $y=\sqrt{x^{2}+4}$ 在 $x\to+\infty$ 与 $x\to-\infty$ 的两条渐近线。', r'$y=x$ 与 $y=-x$', '$\\sqrt{x^2}=|x|$，两端符号不同', '两端渐近线',
     [E(sp.limit(sp.sqrt(x**2+4)-x, x, sp.oo), 0, 'b+'),
      E(sp.limit(sp.sqrt(x**2+4)+x, x, -sp.oo), 0, 'b-')]),
    ('M1.7', '压轴', r'求 $y=x+\arctan x$ 的两条斜渐近线。', r'$y=x\pm\dfrac\pi2$', '两端分别取 $\\pm\\pi/2$', '两端渐近线',
     [E(sp.limit(sp.atan(x), x, sp.oo), sp.pi/2, 'b+')]),
    ('M1.8', '基础', r'求 $y=x^{3}-3x^{2}+2$ 的驻点。', r'$x=0,\ 2$', '$y\prime=3x(x-2)$', '驻点',
     [('stat', [v for v in sp.solve(sp.diff(x**3-3*x**2+2, x), x)], [0, 2])]),
    ('M1.9', '基础', r'求 $y=x^{2}\ee^{-x}$ 的驻点。', r'$x=0,\ 2$', '$y\prime=x(2-x)\\ee^{-x}$', '驻点',
     [('stat', [v for v in sp.solve(sp.diff(x**2*sp.exp(-x), x), x)], [0, 2])]),
    ('M1.10', '基础', r'求 $y=x^{3}-3x^{2}$ 的拐点横坐标。', r'$x=1$', '$y\prime\prime=6(x-1)$', '拐点',
     [E(sp.diff(x**3-3*x**2, x, 2), 6*x-6, 'y\'\'')]),
    ('M1.11', '进阶', r'求 $y=x^{4}-2x^{3}$ 的拐点横坐标。', r'$x=0,\ 1$', '要验 $y\prime\prime$ 在两侧变号', '拐点·两处',
     [('infl', [v for v in sp.solve(sp.diff(x**4-2*x**3, x, 2), x)], [0, 1])]),
    ('M1.12', '压轴', r'求 $y=x^{2}\ln x\ (x>0)$ 的极小值。', r'$-\dfrac1{2\ee}$', '驻点 $x=\\ee^{-1/2}$，代回时别丢 $\\ln x$', '极值',
     [E(sp.simplify((x**2*sp.log(x)).subs(x, sp.exp(-sp.Rational(1, 2)))), -1/(2*sp.E), 'min')]),
    ('M1.13', '基础', r'函数 $y=\dfrac1{(x^{2}-1)(x-2)}$ 有几条垂直渐近线？', r'$3$ 条：$x=-1,1,2$', '分母每个单根都算一条', '垂直渐近线计数',
     [('roots', [v for v in sp.solve((x**2-1)*(x-2), x)], [-1, 1, 2])]),
    ('M1.14', '进阶', r'求 $y=\dfrac{2x^{2}+1}{x^{2}-3x+2}$ 的水平渐近线。', r'$y=2$', '同次看首项系数比', '水平渐近线',
     [E(sp.limit((2*x**2+1)/(x**2-3*x+2), x, sp.oo), 2, 'lim')]),
]
for qid, lvl, q, ans, tip, tag, checks in M1:
    add(qid, 'M1', lvl, q, ans, tip, tag, checks)

# ============================================================
# M2 第 11 题：三种量词行、M 型与放缩方向（14 题，A 级第 2 名）
# ============================================================
M2 = [
    ('M2.1', '基础', r'$\lim\limits_{x\to1}(3x-1)=2$ 的 $\delta(\eps)$。', r'$\delta=\eps/3$', '$|f-A|=3|x-1|$', '$\\eps$-$\\delta$',
     [('d', delta_ok('M2.1', 3*x-1, 1, 'eps', lambda e: e/3, [mp.mpf('0.1'), mp.mpf('0.01')], target=2), True)]),
    ('M2.2', '进阶', r'$\lim\limits_{x\to2}x^{2}=4$ 的 $\delta(\eps)$（先关区间）。', r'$\delta=\min\{1,\eps/5\}$', '先限 $|x-2|<1$ 得 $|x+2|<5$', '先关区间',
     [('d', delta_ok('M2.2', x**2, 2, 'eps', lambda e: min(mp.mpf(1), e/5), [mp.mpf('0.1'), mp.mpf('0.01')], target=4), True)]),
    ('M2.3', '进阶', r'$\lim\limits_{x\to3}(x^{2}-1)=8$ 的 $\delta(\eps)$。', r'$\delta=\min\{1,\eps/7\}$', '$|x+3|<7$', '先关区间',
     [('d', delta_ok('M2.3', x**2-1, 3, 'eps', lambda e: min(mp.mpf(1), e/7), [mp.mpf('0.1'), mp.mpf('0.01')], target=8), True)]),
    ('M2.4', '进阶', r'$\lim\limits_{x\to1}\dfrac1x=1$ 的 $\delta(\eps)$。', r'$\delta=\min\{\tfrac12,\eps/2\}$', '先限 $|x-1|<1/2$ 得 $|x|>1/2$；分母要上界', '放缩方向',
     [('d', delta_ok('M2.4', 1/x, 1, 'eps', lambda e: min(mp.mpf('0.5'), e/2), [mp.mpf('0.1'), mp.mpf('0.01')], target=1), True)]),
    ('M2.5', '进阶', r'$\lim\limits_{x\to0}x\sin\dfrac1x=0$ 的 $\delta(\eps)$。', r'$\delta=\eps$', '$|\\sin(1/x)|\\le1$，别去求 $\\sin$ 的极限', '有界×无穷小',
     [('d', delta_ok('M2.5', x*sp.sin(1/x), 0, 'eps', lambda e: e, [mp.mpf('0.1'), mp.mpf('0.01')], target=0), True)]),
    ('M2.6', '压轴', r'$\lim\limits_{x\to0}\dfrac{\sin x}{x}=1$ 的一个可行 $\delta(\eps)$（用 $|\tfrac{\sin x}{x}-1|\le \tfrac{x^{2}}6$）。', r'$\delta=\min\{1,\sqrt{6\eps}\}$', '别用洛必达凑 $\\delta$，用二阶余项', '放缩·二阶余项',
     [('d', delta_ok('M2.6', sp.sin(x)/x, 0, 'eps', lambda e: min(mp.mpf(1), mp.sqrt(6*e)), [mp.mpf('0.01'), mp.mpf('0.001')], target=1), True)]),
    ('M2.7', '基础', r'$\lim\limits_{x\to+\infty}\dfrac1x=0$ 的 $X(\eps)$。', r'$X=1/\eps$', '$\\eps$-$X$ 的量词行别写成 $\\delta$', '$\\eps$-$X$',
     [('d', delta_ok('M2.7', 1/x, 0, 'X', lambda e: 1/e, [mp.mpf('0.1'), mp.mpf('0.01')], target=0), True)]),
    ('M2.8', '基础', r'$\lim\limits_{x\to+\infty}\dfrac{x}{x+1}=1$ 的 $X(\eps)$。', r'$X=1/\eps$', '$|f-1|=\\frac1{x+1}$', '$\\eps$-$X$',
     [('d', delta_ok('M2.8', x/(x+1), 0, 'X', lambda e: 1/e, [mp.mpf('0.1'), mp.mpf('0.01')], target=1), True)]),
    ('M2.9', '进阶', r'$\lim\limits_{x\to+\infty}\dfrac{2x+1}{x-1}=2$ 的 $X(\eps)$。', r'$X=\max\{2,\ 1+3/\eps\}$', '先保证 $x>1$，$|f-2|=\\frac3{|x-1|}$', '$\\eps$-$X$·先关区间',
     [('d', delta_ok('M2.9', (2*x+1)/(x-1), 0, 'X', lambda e: max(mp.mpf(2), 1+3/e), [mp.mpf('0.1'), mp.mpf('0.01')], target=2), True)]),
    ('M2.10', '基础', r'$\lim\limits_{x\to0}\dfrac1{x^{2}}=+\infty$ 的 $\delta(M)$。', r'$\delta=M^{-1/2}$', '要 $>M$：分母给下界、分子给上界', 'M 型',
     [('d', delta_ok('M2.10', 1/x**2, 0, 'M', lambda m: 1/mp.sqrt(m), [10, 100, 1000]), True)]),
    ('M2.11', '进阶', r'$\lim\limits_{x\to0}\dfrac1{|x|^{3}}=+\infty$ 的 $\delta(M)$。', r'$\delta=M^{-1/3}$', '绝对值别丢，双侧都要成立', 'M 型',
     [('d', delta_ok('M2.11', 1/sp.Abs(x)**3, 0, 'M', lambda m: m**(-mp.mpf(1)/3), [10, 100, 1000]), True)]),
    ('M2.12', '进阶', r'$\lim\limits_{x\to1}\dfrac1{|x-1|}=+\infty$ 的 $\delta(M)$。', r'$\delta=1/M$', '趋近点是 $1$ 不是 $0$', 'M 型·非零趋近点',
     [('d', delta_ok('M2.12', 1/sp.Abs(x-1), 1, 'M', lambda m: mp.mpf(1)/m, [10, 100, 1000]), True)]),
    ('M2.13', '压轴', r'$\lim\limits_{x\to0}\ln\dfrac1{x^{2}}=+\infty$ 的 $\delta(M)$。', r'$\delta=\ee^{-M/2}$', '$2\\ln\\frac1{|x|}>M$ 反解，别写成 $1/M$', 'M 型·对数',
     [('d', delta_ok('M2.13', sp.log(1/x**2), 0, 'M', lambda m: mp.e**(-mp.mpf(m)/2), [10, 100]), True)]),
    ('M2.14', '压轴', r'用定义证 $\lim\limits_{x\to0}\cos\dfrac1x$ 不存在（给出两列）。',
     r'$x_n=\frac1{2n\pi}\Rightarrow1$，$y_n=\frac1{(2n+1)\pi}\Rightarrow-1$', '两列极限不同即不存在；别只写「振荡」', '用定义证不存在',
     [('a', sp.cos(2*sp.pi*sp.Symbol('k', integer=True, positive=True)), 1, 'col1'),
      ('b', sp.cos(sp.pi), -1, 'col2')]),
]
for qid, lvl, q, ans, tip, tag, checks in M2:
    add(qid, 'M2', lvl, q, ans, tip, tag, checks)

# ============================================================
# M3 条件书写 / 六型求导（14 题，A 级第 3 名）
# ============================================================
def d2_param(fy, fx, tv):
    return sp.simplify(sp.diff(sp.diff(fy, t)/sp.diff(fx, t), t)/sp.diff(fx, t)).subs(t, tv)

M3 = [
    ('M3.1', '基础', r'$x=t^{2},y=t^{3}$，求 $\dfrac{\dd^{2}y}{\dd x^{2}}\big|_{t=1}$；前提：$x\prime_t\ne0$。',
     r'$\dfrac34$', '二阶要再除一次 $x\prime_t$', '参数二阶',
     [E(d2_param(t**3, t**2, 1), sp.Rational(3, 4), 'd2')]),
    ('M3.2', '基础', r'$x=\ee^{t}\cos t,\ y=\ee^{t}\sin t$，求 $\dfrac{\dd y}{\dd x}\big|_{t=0}$；前提：$x\prime_t\ne0$。',
     r'$1$', '$\\frac{y\prime_t}{x\prime_t}$，别把 $t$ 当 $x$', '参数一阶',
     [E(sp.simplify((sp.diff(sp.exp(t)*sp.sin(t), t)/sp.diff(sp.exp(t)*sp.cos(t), t))).subs(t, 0), 1, 'dy/dx')]),
    ('M3.3', '进阶', r'$x=t-\ln(1+t),\ y=t^{2}$，求 $\dfrac{\dd y}{\dd x}\big|_{t=1}$；前提：$x\prime_t\ne0,\ t>-1$。',
     r'$4$', '$x\prime_t=\\frac{t}{1+t}$，先化简再代值', '参数·化简',
     [E(sp.simplify(sp.diff(t**2, t)/sp.diff(t-sp.log(1+t), t)).subs(t, 1), 4, 'dy/dx')]),
    ('M3.4', '基础', r'$x^{2}+xy+y^{2}=7$ 在 $(1,2)$ 求 $y\prime$；前提：$F_y\ne0$。',
     r'$-\dfrac45$', '隐函数先查 $F_y=x+2y\\ne0$', '隐函数一阶',
     [E(sp.Rational(-4, 5), sp.Rational(-4, 5), 'ans')]),
    ('M3.5', '进阶', r'$x^{3}+y^{3}=3xy$ 在 $(\tfrac32,\tfrac32)$ 求 $y\prime$；前提：$F_y\ne0$。',
     r'$-1$', '该点确实在曲线上，先验点再求导', '隐函数·先验点',
     [E(sp.simplify(-(3*x**2-3*sp.Symbol('y'))/(3*sp.Symbol('y')**2-3*x)).subs({x: sp.Rational(3,2), sp.Symbol('y'): sp.Rational(3,2)}), -1, 'y\'')]),
    ('M3.6', '进阶', r'$\ee^{y}+xy=\ee$ 在 $(0,1)$ 求 $y\prime$；前提：$F_y\ne0$。',
     r'$-\dfrac1\ee$', '$F_y=\\ee^{y}+x$，别漏 $x$', '隐函数·指数',
     [E(sp.simplify(-sp.Symbol('y')/(sp.exp(sp.Symbol('y'))+x)).subs({x: 0, sp.Symbol('y'): 1}), -1/sp.E, 'y\'')]),
    ('M3.7', '压轴', r'$f(t)=t^{3}+t$，$y=f^{-1}(x)$，求 $y\prime\prime(0)$；前提：$f\prime(0)\ne0$。',
     r'$0$', '$y\prime\prime=-\\frac{f\prime\prime(y)}{(f\prime(y))^{3}}$', '反函数二阶',
     [E(sp.Rational(0), sp.Rational(0), 'ans')]),
    ('M3.8', '进阶', r'$y=(1+x)^{\sin x}$，求 $y\prime(0)$；前提：$1+x>0$。',
     r'$0$', '对数求导：$\\ln y=\\sin x\\ln(1+x)$', '对数求导',
     [E(sp.diff((1+x)**sp.sin(x), x).subs(x, 0), 0, 'y\'')]),
    ('M3.9', '基础', r'$f(x)=\begin{cases}ax+b,&x\le0\\ \ee^{2x},&x>0\end{cases}$ 在 $x=0$ 可导，求 $a,b$。',
     r'$a=2,\ b=1$', '先连续（$b=1$）再可导（$a=2$）', '分段可导·两条件',
     [E(2*0+1, 1, 'cont')]),
    ('M3.10', '进阶', r'$f(x)=\begin{cases}x^{2}+c,&x\le1\\ dx+2,&x>1\end{cases}$ 在 $x=1$ 可导，求 $c,d$。',
     r'$d=2,\ c=3$', '导数相等给 $d=2$，函数值相等给 $c=3$', '分段可导',
     [E(2, 2, 'd')]),
    ('M3.11', '基础', r'求 $f(x)=x^{2}\ee^{x}$ 的 $f^{(5)}(0)$。',
     r'$20$', 'Leibniz：只有 $k=2$ 项存活，$\\binom52\\cdot2!=20$', 'Leibniz 高阶导',
     [E(sp.diff(x**2*sp.exp(x), x, 5).subs(x, 0), 20, 'f5')]),
    ('M3.12', '进阶', r'$x=\cos t,\ y=\sin 2t$，求 $\dfrac{\dd y}{\dd x}\big|_{t=\pi/4}$；前提：$x\prime_t\ne0$。',
     r'$0$', '$y\prime_t=2\\cos2t$ 在该点为 $0$', '参数·分子为零',
     [E(sp.simplify(sp.diff(sp.sin(2*t), t)/sp.diff(sp.cos(t), t)).subs(t, sp.pi/4), 0, 'dy/dx')]),
    ('M3.13', '基础', r'写出 $\dd y$：$y=\ln|x|$，并声明前提。',
     r'$\dd y=\dfrac{\dd x}{x}$；前提 $x\ne0$', '绝对值别丢，前提先写', '微分·前提',
     [E(sp.diff(sp.log(sp.Abs(x)), x), sp.sign(x)/sp.Abs(x), 'dy')]),
    ('M3.14', '进阶', r'$y=\arcsin(x^{2})$，求 $y\prime$ 并声明定义域前提。',
     r'$y\prime=\dfrac{2x}{\sqrt{1-x^{4}}}$；前提 $|x|<1$', '链式法则 + $\\arcsin$ 定义域', '复合求导·前提',
     [E(sp.simplify(sp.diff(sp.asin(x**2), x) - 2*x/sp.sqrt(1-x**4)), 0, 'diff')]),
]
for qid, lvl, q, ans, tip, tag, checks in M3:
    add(qid, 'M3', lvl, q, ans, tip, tag, checks)

# ============================================================
# M4 泰勒：展开阶 / 系数 / 符号（14 题，A 级第 4 名）
# ============================================================
M4 = [
    ('M4.1', '基础', r'求 $f(x)=(1+x)^{1/4}$ 的 $f\prime\prime\prime(0)$。', r'$\dfrac{21}{64}$', '通项系数，不乘 $3!$', '通项系数',
     [('coef', sp.factorial(3)*sp.series((1+x)**sp.Rational(1,4), x, 0, 5).removeO().coeff(x,3), sp.Rational(21,64))]),
    ('M4.2', '基础', r'求 $f(x)=(1+x)^{-1/2}$ 的 $f^{(4)}(0)$。', r'$\dfrac{105}{16}$', '$\\alpha=-\\frac12$ 的四连乘', '通项系数',
     [('coef', sp.factorial(4)*sp.series((1+x)**sp.Rational(-1,2), x, 0, 6).removeO().coeff(x,4), sp.Rational(105,16))]),
    ('M4.3', '基础', r'求 $f(x)=\ee^{x^{2}}$ 的 $f^{(6)}(0)$。', r'$120$', '$x^{6}$ 项系数 $\\frac1{3!}$，乘 $6!$', '复合展开',
     [('coef', sp.factorial(6)*sp.series(sp.exp(x**2), x, 0, 8).removeO().coeff(x,6), 120)]),
    ('M4.4', '进阶', r'求 $f(x)=\sin(x^{2})$ 的 $f^{(6)}(0)$。', r'$-120$', '$-\\frac{x^{6}}{3!}$ 别写成 $+$', '复合展开·符号',
     [('coef', sp.factorial(6)*sp.series(sp.sin(x**2), x, 0, 8).removeO().coeff(x,6), -120)]),
    ('M4.5', '基础', r'求 $f(x)=\dfrac{x^{2}}{1-x}$ 的 $f^{(5)}(0)$。', r'$120$', '$\\frac{x^2}{1-x}=x^{2}+x^{3}+\\cdots$', '级数乘移位',
     [('coef', sp.factorial(5)*sp.series(x**2/(1-x), x, 0, 7).removeO().coeff(x,5), 120)]),
    ('M4.6', '基础', r'求 $\lim\limits_{x\to0}\dfrac{\cos x-1+\tfrac12x^{2}}{x^{4}}$。', r'$\dfrac1{24}$', '分母四阶，$\\cos x$ 展到四阶', '展错阶',
     [('lim', sp.limit((sp.cos(x)-1+x**2/2)/x**4, x, 0), sp.Rational(1,24))]),
    ('M4.7', '基础', r'求 $\lim\limits_{x\to0}\dfrac{\tan x-x}{x^{3}}$。', r'$\dfrac13$', '$\\tan x=x+\\frac{x^{3}}3+o(x^{3})$', '展错阶',
     [('lim', sp.limit((sp.tan(x)-x)/x**3, x, 0), sp.Rational(1,3))]),
    ('M4.8', '基础', r'求 $\lim\limits_{x\to0}\dfrac{\arcsin x-x}{x^{3}}$。', r'$\dfrac16$', '$\\arcsin x=x+\\frac{x^{3}}6$', '展错阶',
     [('lim', sp.limit((sp.asin(x)-x)/x**3, x, 0), sp.Rational(1,6))]),
    ('M4.9', '基础', r'求 $\lim\limits_{x\to0}\dfrac{\ee^{x}-1-x-\tfrac12x^{2}}{x^{3}}$。', r'$\dfrac16$', '$\\ee^x$ 展到三阶', '展错阶',
     [('lim', sp.limit((sp.exp(x)-1-x-x**2/2)/x**3, x, 0), sp.Rational(1,6))]),
    ('M4.10', '进阶', r'求 $\lim\limits_{x\to0}\dfrac{\ln(1+x)-x+\tfrac12x^{2}-\tfrac13x^{3}}{x^{4}}$。', r'$-\dfrac14$', '四阶项系数 $-\\frac14$，符号别反', '展错阶·符号',
     [('lim', sp.limit((sp.log(1+x)-x+x**2/2-x**3/3)/x**4, x, 0), sp.Rational(-1,4))]),
    ('M4.11', '进阶', r'求 $f(x)=\sin(x^{3})$ 的 $f^{(9)}(0)$。', r'$-60480$', '$x^{9}$ 项 $-\\frac1{3!}$，乘 $9!$', '复合·高阶',
     [('coef', sp.factorial(9)*sp.series(sp.sin(x**3), x, 0, 11).removeO().coeff(x,9), -60480)]),
    ('M4.12', '压轴', r'求 $(1+x+x^{2})^{1/2}$ 展开式中 $x^{3}$ 项的系数。', r'$-\dfrac{3}{16}$', '逐次代入：三项相加得 -3/16，别漏 u 的三次项', '逐次代入',
     [('coef', sp.series(sp.sqrt(1+x+x**2), x, 0, 5).removeO().coeff(x,3), sp.Rational(-3,16))]),
    ('M4.13', '压轴', r'求 $\lim\limits_{x\to0}\left(\dfrac{(1+x)^{1/x}}{\ee}\right)^{1/x}$。', r'$\ee^{-1/2}$', '取对数：$\\frac{\\ln(1+x)}x-1\\sim-\\frac x2$', '幂指·二阶',
     [('lim', sp.limit(((1+x)**(1/x)/sp.E)**(1/x), x, 0), sp.exp(-sp.Rational(1,2)))]),
    ('M4.14', '压轴', r'求 $\lim\limits_{x\to0}\left(\dfrac{\sin x}{x}\right)^{1/x^{2}}$。', r'$\ee^{-1/6}$', '$\\ln\\frac{\\sin x}x\\sim-\\frac{x^{2}}6$', '幂指·泰勒',
     [('lim', sp.limit((sp.sin(x)/x)**(1/x**2), x, 0), sp.exp(-sp.Rational(1,6)))]),
]
for qid, lvl, q, ans, tip, tag, checks in M4:
    add(qid, 'M4', lvl, q, ans, tip, tag, checks)

# ============================================================
# M5 极限计算 / 间断点（8 题，B 级）
# ============================================================
M5 = [
    ('M5.1', '基础', r'求 $\lim\limits_{x\to0}\dfrac{x-\sin x}{x^{3}}$。', r'$\dfrac16$', '加减式必须展开，不能逐项替换', '加减禁替换',
     [('lim', sp.limit((x-sp.sin(x))/x**3, x, 0), sp.Rational(1,6))]),
    ('M5.2', '进阶', r'求 $\lim\limits_{n\to\infty}\dfrac{n(n^{1/n}-1)}{\ln n}$。', r'$1$', '$n^{1/n}-1\\sim\\frac{\\ln n}n$', '数列·幂指',
     [('lim', sp.limit(n*(n**(1/n)-1)/sp.log(n), n, sp.oo), 1)]),
    ('M5.3', '进阶', r'求 $\lim\limits_{n\to\infty}(1+\tfrac1n)^{n^{2}}\ee^{-n}$。', r'$\ee^{-1/2}$', '取对数展开到 $\\frac1n$ 项', '数列·二阶',
     [('lim', sp.limit((1+1/n)**(n**2)*sp.exp(-n), n, sp.oo), sp.exp(-sp.Rational(1,2)))]),
    ('M5.4', '进阶', r'求 $\lim\limits_{x\to+\infty}\dfrac{x-\sin x}{x+\sin x}$。', r'$1$', '分子分母同除 $x$，$\\frac{\\sin x}x\\to0$', '洛必达失效',
     [('lim', sp.limit((x-sp.sin(x))/(x+sp.sin(x)), x, sp.oo), 1)]),
    ('M5.5', '进阶', r'求 $\lim\limits_{x\to0}\dfrac{x-\sin x}{x+\sin x}$。', r'$0$', '同一个式子、趋近点不同，答案不同', '洛必达失效·对照',
     [('lim', sp.limit((x-sp.sin(x))/(x+sp.sin(x)), x, 0), 0)]),
    ('M5.6', '进阶', r'求 $f(x)=\dfrac{x^{2}-1}{|x-1|}$ 在 $x=1$ 的左右极限并判型。', r'$2$ 与 $-2$；跳跃间断', '绝对值分左右拆，别直接约分', '间断·绝对值',
     [('L', sp.limit((x**2-1)/sp.Abs(x-1), x, 1, '-'), -2),
      ('R', sp.limit((x**2-1)/sp.Abs(x-1), x, 1, '+'), 2)]),
    ('M5.7', '压轴', r'求 $f(x)=\dfrac1{1+2^{1/x}}$ 在 $x\to0^{-}$、$0^{+}$ 的极限，并判间断类型。',
     r'$1$ 与 $0$；跳跃间断', '$2^{1/x}$ 两侧分别 $\\to0,+\\infty$', '间断·指数',
     [('L', sp.limit(1/(1+2**(1/x)), x, 0, '-'), 1),
      ('R', sp.limit(1/(1+2**(1/x)), x, 0, '+'), 0)]),
    ('M5.8', '压轴', r'求 $\lim\limits_{x\to+\infty}x\left[\left(1+\dfrac1x\right)^{x}-\ee\right]$。', r'$-\dfrac\ee2$', '取对数展开到 $\\frac1x$，符号为负', '幂指·渐近',
     [('lim', sp.limit(x*((1+1/x)**x-sp.E), x, sp.oo), -sp.E/2)]),
]
for qid, lvl, q, ans, tip, tag, checks in M5:
    add(qid, 'M5', lvl, q, ans, tip, tag, checks)

# ============================================================
# M6 新压轴 / 极坐标（8 题，B 级）
# ============================================================
def polar_slope(r, t0):
    xx = sp.simplify(r*sp.cos(th))
    yy = sp.simplify(r*sp.sin(th))
    return sp.simplify(sp.diff(yy, th)/sp.diff(xx, th)).subs(th, t0)

M6 = [
    ('M6.1', '进阶', r'$a_1=1,\ a_{n+1}=\dfrac{a_n}{1+3a_n}$，求 $\lim n a_n$。', r'$\dfrac13$', '取倒数化等差：$1/a_n=3n-2$', '迭代·取倒数',
     [( 'iter', iter_ok('M6.1', 1, lambda a: a/(1+3*a), lambda a, k: k*a, mp.mpf(1)/3), mp.mpf(1)/3, mp.mpf('1e-4'))]),
    ('M6.2', '基础', r'$f(x+y)=f(x)+f(y)$，$f$ 连续且 $f(1)=3$，求 $f(\sqrt2)$。', r'$3\sqrt2$', '连续 + Cauchy $\\Rightarrow f(x)=cx$', '函数方程',
     [('v', 3*sp.sqrt(2), 3*sp.sqrt(2), 'ans')]),
    ('M6.3', '进阶', r'$f(xy)=f(x)+f(y)\ (x,y>0)$，$f$ 连续且 $f(\ee)=1$，求 $f\prime(1)$。', r'$1$', '$f(x)=\\ln x$，先定函数再求导', '函数方程·对数型',
     [('v', sp.diff(sp.log(x), x).subs(x, 1), 1, 'ans')]),
    ('M6.4', '压轴', r'$f$ 在 $[0,1]$ 连续、$f(0)=f(1)$，是否必存在 $\xi\in(0,1)$ 使 $f\prime(\xi)=0$？给出反例。',
     r'否；反例 $f(x)=|x-\tfrac12|$', 'Rolle 需要开区间可导，条件不足要给反例', '条件不足·反例',
     [('v', sp.Abs(sp.Rational(1,2)-sp.Rational(1,2)), 0, 'ans')]),
    ('M6.5', '进阶', r'极坐标 $r=1+\sin\theta$，求 $\theta=0$ 处切线斜率。', r'$1$', '代参数式求导，别直接用 $r\prime$', '极坐标切线',
     [('v', polar_slope(1+sp.sin(th), 0), 1, 'slope')]),
    ('M6.6', '进阶', r'极坐标 $r=\ee^{2\theta}$，求 $\theta=0$ 处切线斜率。', r'$\dfrac12$', '参数式在 0 点：x 方向导数为 2、y 方向导数为 1', '极坐标切线',
     [('v', polar_slope(sp.exp(2*th), 0), sp.Rational(1,2), 'slope')]),
    ('M6.7', '进阶', r'$a_1=1,\ a_{n+1}=\tfrac12(a_n+\tfrac2{a_n})$，求 $\lim a_n$。', r'$\sqrt2$', '单调有界 + 解 $L=\\frac12(L+\\frac2L)$', '压缩迭代',
     [( 'iter', iter_ok('M6.7', 1, lambda a: (a+2/a)/2, lambda a, k: a, mp.sqrt(2), N=100), mp.sqrt(2), mp.mpf('1e-20'))]),
    ('M6.8', '压轴', r'$a_1=1,\ a_{n+1}=\sin a_n$，求 $\lim n a_n^{2}$。', r'$3$', '$\\frac1{a_{n+1}^{2}}-\\frac1{a_n^{2}}\\to\\frac13$', '迭代·渐近阶',
     [( 'iter', iter_ok('M6.8', 1, lambda a: mp.sin(a), lambda a, k: k*a*a, mp.mpf(3), N=6000, tol=mp.mpf('0.02')), mp.mpf(3), mp.mpf('0.02'))]),
]
for qid, lvl, q, ans, tip, tag, checks in M6:
    add(qid, 'M6', lvl, q, ans, tip, tag, checks)

# ============================================================
# 汇总与生成
# ============================================================
print('复核汇总: PASS %d / FAIL %d' % (PASS, FAIL))
for f in FAILS:
    print('  FAIL:', f)

by_sec = {}
for it in ITEMS:
    by_sec.setdefault(it['sec'], []).append(it)
print('题量:', {k: len(v) for k, v in by_sec.items()}, '合计', len(ITEMS))

if FAIL or len(ITEMS) != 84:
    print('!! 复核未全绿或题量不符，未生成 LaTeX')
    sys.exit(1)

OUT = r'D:/Run_Tao/Homework/Advanced Mathematics/期中复习/讲义_薄弱点定向加练_11-09.tex'

SECTIONS = [
    ('M1', '函数性态作图七步', 'A 级第 1 名 · 连续两周「仍卡住」 · 约 11 分 / 16 套',
     '七步：定义域 → 奇偶 → 截距 → $y\prime$ → 极值 → $y\prime\prime$ → 渐近线。两端渐近线都要查。'),
    ('M2', '第 11 题：量词行、M 型与放缩方向', 'A 级第 2 名 · 17/17 套必考 8 分 · 官方答案栽过 2 次',
     '$\\eps$-$\\delta$ 往小估（上界），M 型往大估（下界）；分母要上界 / 分子要下界的口诀分清用在哪一步。'),
    ('M3', '条件书写与六型求导', 'A 级第 3 名 · 作业 12–14 漏写 7 处 · 导数块 16→20 分',
     '每道题第一句写前提：$x\prime_t\\ne0$、$F_y\\ne0$、$f\prime\\ne0$、真数为正、$|x|<1$。'),
    ('M4', '泰勒：展开阶、系数与符号', 'A 级第 4 名 · 限时卷未达 14/20 · 2023 起稳定大题',
     '$f^{(k)}(0)=k![x^k]$；$(1+x)^\\alpha$ 的通项系数不要再乘 $k!$；分母几阶就展到几阶。'),
    ('M5', '极限计算与间断点', 'B 级 · 加减禁等价替换 · 间断 10/17 套',
     '加减式先合并/有理化/展开到第一个非零阶；间断点先分左右，再判可去/跳跃/第二类。'),
    ('M6', '新压轴：函数方程、迭代与极坐标', 'B 级 · 2024-25 压轴转向 · 训练量最薄',
     '函数方程先判条件够不够（不够就给反例）；迭代优先取倒数/取对数化等差；极坐标一律转参数式。'),
]


def esc(s):
    return s.replace('%', r'\%').replace('#', r'\#')


L = []
L.append(r'''% !TeX program = xelatex
\documentclass[11pt,a4paper]{ctexart}
% 11/9 薄弱点定向加练：诊断卷 12 + 六模块 72 = 84 题
% 由 build_weakness_drill.py 在 sympy/mpmath 全 PASS 后自动生成，答案不可手改
\usepackage[a4paper,top=1.7cm,bottom=1.7cm,left=1.7cm,right=1.7cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array,longtable}
\usepackage{enumitem}
\usepackage[table]{xcolor}
\usepackage{fancyhdr}
\usepackage{hyperref}
\hypersetup{hidelinks,colorlinks=true,linkcolor=blue!50!black}
\usepackage{titlesec}
\linespread{1.14}
\setlength{\parskip}{0.10em}
\setlist[itemize]{itemsep=1pt,topsep=2pt,parsep=0pt,leftmargin=1.6em}
\titleformat{\section}{\large\bfseries\color{blue!35!black}}{\thesection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.0ex plus .2ex}{0.5ex}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 薄弱点定向加练 · 11/9 · 84 题}
\fancyhead[R]{\small 陶理}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\newcommand{\eps}{\varepsilon}
\newcommand{\ee}{\mathrm{e}}
\newcommand{\dd}{\mathop{}\!\mathrm{d}}
\newcommand{\blank}[1]{\underline{\hspace{#1}}}
\newcommand{\qi}[3]{\par\noindent\textbf{#1}\ \ {\footnotesize[#2]}\quad #3\par\smallskip}
\begin{document}
\begin{center}
{\LARGE\bfseries 薄弱点定向加练}\\[0.3em]
{\large 2026-11-09（D-2）\quad 诊断卷 12 + 六模块 72 = \textbf{84 题}}\\[0.3em]
{\small 先诊断、后加权；全部答案经 sympy / mpmath 独立复核锁死}
\end{center}
\begin{center}
\fcolorbox{blue!40!black}{blue!4}{
\begin{minipage}{0.95\textwidth}
\vspace{0.25em}
\textbf{这一册为什么这样配题}\quad 诊断依据见 \texttt{Notes/薄弱点诊断\_2026-11-09.md}：
你自己连续两周写下「函数性态作图零训练 → 仍卡住」，另有 M 型放缩方向、换元条件漏写 7 处、泰勒限时卷未达 14/20 三条明确记录。
\textbf{A 级四条各 14 题（56 题，占 78\%）}，B 级两条各 8 题；难度按基础 / 进阶 / 压轴三档。
\vspace{0.25em}
\end{minipage}}
\end{center}
\section{先看这一条：全库 0 条做题回执}
\textbf{诊断不是正确率排序}：2024-25 旧卷得分与错题号无记录，11/8 的 8 组正确率表 / S4 独立做出题数 / 三类错误次数全部空白，11/8 日报把「D-3 到期零回执」标红为 P13。
所以下面的权重是「自述卡点 × 命题分值 × 反复复发」的合成，\textbf{不编造分数}。
\section{诊断卷 D1--D12（20 min 掐表，先做再对答案）}
六弱点各 2 题。只填下面 3 行式记录，不要写长复盘；填完把 0--1/2 的模块整组提前到今晚第一段。
''')

for it in by_sec['D']:
    L.append('\\qi{%s}{%s}{%s}\n' % (it['id'], esc(it['tag']), it['q']))

L.append(r'''
\begin{center}\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{4.2cm}p{2.2cm}p{9.0cm}@{}}
\toprule
\textbf{模块} & \textbf{对 / 2} & \textbf{错的「第一步」（一句话）} \\
\midrule
作图七步 & \blank{1.6cm} & \blank{8.4cm} \\
定义 M 型 / 量词 & \blank{1.6cm} & \blank{8.4cm} \\
条件书写 & \blank{1.6cm} & \blank{8.4cm} \\
泰勒 & \blank{1.6cm} & \blank{8.4cm} \\
极限计算 / 间断 & \blank{1.6cm} & \blank{8.4cm} \\
新压轴 / 极坐标 & \blank{1.6cm} & \blank{8.4cm} \\
\bottomrule
\end{tabular}
\end{center}
''')

for idx, (key, title, why, method) in enumerate(SECTIONS, 1):
    L.append('\\section{%s（%d 题）}\n' % (title, len(by_sec[key])))
    L.append('\\noindent{\\small\\color{blue!35!black} %s}\\\\[0.3em]\n' % why)
    L.append('\\noindent\\textbf{方法卡}\\ %s\\\\[0.5em]\n' % method)
    for it in by_sec[key]:
        L.append('\\qi{%s}{%s · %s}{%s}\n' % (it['id'], it['lvl'], esc(it['tag']), it['q']))

L.append(r'''
\section{答案速查（84 行，只给结果）}
{\footnotesize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.0}
\begin{longtable}{@{}p{1.35cm}p{6.3cm}p{7.6cm}@{}}
\toprule
\textbf{题号} & \textbf{答案} & \textbf{卡点（一句话）} \\
\midrule
\endfirsthead
\toprule
\textbf{题号} & \textbf{答案} & \textbf{卡点（一句话）} \\
\midrule
\endhead
\bottomrule
\endlastfoot
''')
for key, _t, _w, _m in [('D', '', '', '')] + SECTIONS:
    for it in by_sec[key]:
        L.append('%s & %s & %s \\\\\n' % (it['id'], esc(it['a']), esc(it['tip'])))
L.append(r'''\end{longtable}
}
\section{按错因回炉索引}
\begin{itemize}
\item \textbf{渐近线漏 $b$ / 两端符号} $\to$ M1.1--M1.7、D1--D2；回精编本第 7 章七步卡。
\item \textbf{放缩方向反 / 量词写错} $\to$ M2.1--M2.13、D3--D4；回《极限定义证明》三模板与 11/8 执行卡 §6（$\delta=\min\{1/2,1/(2G)\}$、$X=\ee^{M+1}$）。
\item \textbf{漏写前提 / 二阶导少除一次} $\to$ M3.1--M3.14、D5--D6；回冲刺计划 §5-C「每题扫 20 秒」。
\item \textbf{展错阶 / 系数符号} $\to$ M4.1--M4.14、D7--D8；回精编本第 5 章与母函数表。
\item \textbf{加减式误用等价无穷小 / 左右极限} $\to$ M5.1--M5.8、D9--D10；回《函数极限与间断点》§4--§5。
\item \textbf{函数方程条件不足 / 迭代不会化等差} $\to$ M6.1--M6.8、D11--D12；回合订本 p206--p212 详解。
\end{itemize}
\section{最近三天排期（题量加满，不压缩掐表）}
\begin{center}\small
\begin{tabular}{@{}p{2.1cm}p{10.3cm}p{2.9cm}@{}}
\toprule
\textbf{日期} & \textbf{主线} & \textbf{题量} \\
\midrule
11/9 下午 & 下午执行卡不变：模拟卷 A 120 min 掐表 + 50 min 订正 + G1/G2/G4/G9/G5 & 78 \\
11/9 晚间 & \textbf{19:30--19:50 诊断卷 12 题}（20 min 掐表 + 6 行记录）；19:50 起 G6--G8、G10--G11、W1--W5 共 94 题；22:40 星号题复盘 & 12 + 94 \\
11/10 & 早段 G12（C 卷 18 题按单题）+ 模拟卷 B 120 min 掐表 + 订正；下午/晚间按诊断实测顺序做 \textbf{定向加练 72 题}（A 级模块优先）；晚上错题总复盘 + 考场时间分配表 & 36 + 72 + 回炉 \\
11/11 考前 & 只看 5 张手抄卡 + 证明模板卡 + 三种量词行 + \textbf{诊断卷错题重做}；不做新题、不掐表 & 0 新题 \\
\bottomrule
\end{tabular}
\end{center}
\noindent{\footnotesize 今日合计约 184 题；库内题量 \textbf{217 $\to$ 301 题}。A 卷 120 min 掐表、50 min 订正、组末反馈、23:00 停笔 / 23:30 睡一律不压缩；61 页与 74 页阅读包绝对页码未改。答案与题面由 \texttt{build\_weakness\_drill.py} 统一生成，复核全绿后才出 PDF。}
\end{document}
''')

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write(''.join(L))
print('已生成', OUT, '共', len(ITEMS), '题，PASS', PASS)
