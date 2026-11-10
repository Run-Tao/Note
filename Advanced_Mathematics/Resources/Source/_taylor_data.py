# -*- coding: utf-8 -*-
"""泰勒/麦克劳林展开全表 —— 数据与 sympy 逐项复核。

每个条目的系数是我手推的结果，脚本从 sympy 独立 series 出来对比；
不一致即 FAIL。全 PASS 才允许排版。
"""
import sympy as sp
from sympy import Rational as R

x = sp.symbols('x')
t = sp.symbols('t')
N = sp.symbols('n', integer=True, nonnegative=True)

PASS, FAIL = [], []


def coef_map(expr, n, x0=0):
    """在 x0 处展开到 n 阶，返回 {k: 系数}。"""
    e = expr.subs(x, t + x0) if x0 != 0 else expr.subs(x, t)
    s = sp.series(e, t, 0, n + 1).removeO()
    return {k: sp.simplify(s.coeff(t, k)) for k in range(n + 1)}


def chk(name, expr, claimed, x0=0):
    """claimed: {k: 系数}，未列出的 k 视为 0（会被一并检查）。"""
    n = max(list(claimed) + [0])
    try:
        got = coef_map(expr, n, x0)
    except Exception as exc:  # noqa
        FAIL.append((name, 'series error: %s' % exc))
        return
    bad = []
    for k in range(n + 1):
        want = claimed.get(k, R(0))
        g = got[k]
        if sp.simplify(sp.nsimplify(g) - want) != 0:
            bad.append((k, str(want), str(g)))
    if bad:
        FAIL.append((name, bad))
    else:
        PASS.append(name)


# ---------------------------------------------------------------- §2 核心表
CORE = [
    ('exp', sp.exp(x), {0: R(1), 1: R(1), 2: R(1, 2), 3: R(1, 6), 4: R(1, 24), 5: R(1, 120), 6: R(1, 720)}),
    ('sin', sp.sin(x), {1: R(1), 3: R(-1, 6), 5: R(1, 120), 7: R(-1, 5040)}),
    ('cos', sp.cos(x), {0: R(1), 2: R(-1, 2), 4: R(1, 24), 6: R(-1, 720)}),
    ('tan', sp.tan(x), {1: R(1), 3: R(1, 3), 5: R(2, 15), 7: R(17, 315), 9: R(62, 2835)}),
    ('asin', sp.asin(x), {1: R(1), 3: R(1, 6), 5: R(3, 40), 7: R(5, 112), 9: R(35, 1152)}),
    ('atan', sp.atan(x), {1: R(1), 3: R(-1, 3), 5: R(1, 5), 7: R(-1, 7), 9: R(1, 9)}),
    ('ln(1+x)', sp.log(1 + x), {1: R(1), 2: R(-1, 2), 3: R(1, 3), 4: R(-1, 4), 5: R(1, 5), 6: R(-1, 6)}),
    ('ln(1-x)', sp.log(1 - x), {1: R(-1), 2: R(-1, 2), 3: R(-1, 3), 4: R(-1, 4), 5: R(-1, 5)}),
    ('1/(1-x)', 1 / (1 - x), {0: R(1), 1: R(1), 2: R(1), 3: R(1), 4: R(1)}),
]

# ---------------------------------------------------------------- §3 幂函数族
POWER = [
    ('1/(1+x)', 1 / (1 + x), {0: R(1), 1: R(-1), 2: R(1), 3: R(-1), 4: R(1)}),
    ('1/(1+x)^2', 1 / (1 + x) ** 2, {0: R(1), 1: R(-2), 2: R(3), 3: R(-4), 4: R(5)}),
    ('1/(1-x)^2', 1 / (1 - x) ** 2, {0: R(1), 1: R(2), 2: R(3), 3: R(4), 4: R(5)}),
    ('sqrt(1+x)', sp.sqrt(1 + x), {0: R(1), 1: R(1, 2), 2: R(-1, 8), 3: R(1, 16), 4: R(-5, 128)}),
    ('1/sqrt(1+x)', 1 / sp.sqrt(1 + x), {0: R(1), 1: R(-1, 2), 2: R(3, 8), 3: R(-5, 16), 4: R(35, 128)}),
    ('(1+x)^(1/3)', (1 + x) ** sp.Rational(1, 3), {0: R(1), 1: R(1, 3), 2: R(-1, 9), 3: R(5, 81), 4: R(-10, 243)}),
    ('1/(1+x^2)', 1 / (1 + x ** 2), {0: R(1), 2: R(-1), 4: R(1), 6: R(-1)}),
    ('(1-x)^(-1/2)', (1 - x) ** sp.Rational(-1, 2), {0: R(1), 1: R(1, 2), 2: R(3, 8), 3: R(5, 16)}),
]

# ---------------------------------------------------------------- §4 指对与双曲
MISC = [
    ('sinh', sp.sinh(x), {1: R(1), 3: R(1, 6), 5: R(1, 120), 7: R(1, 5040)}),
    ('cosh', sp.cosh(x), {0: R(1), 2: R(1, 2), 4: R(1, 24), 6: R(1, 720)}),
    ('tanh', sp.tanh(x), {1: R(1), 3: R(-1, 3), 5: R(2, 15), 7: R(-17, 315)}),
    ('arsinh', sp.asinh(x), {1: R(1), 3: R(-1, 6), 5: R(3, 40), 7: R(-5, 112)}),
    ('2^x', sp.Integer(2) ** x, {0: R(1), 1: sp.log(2), 2: sp.log(2) ** 2 / 2, 3: sp.log(2) ** 3 / 6}),
    ('lg(1+x)', sp.log(1 + x, 10), {1: 1 / sp.log(10), 2: -1 / (2 * sp.log(10)), 3: 1 / (3 * sp.log(10))}),
]

# ---------------------------------------------------------------- §5 复合代入
COMPOSITE = [
    ('exp(-x^2)', sp.exp(-x ** 2), {0: R(1), 2: R(-1), 4: R(1, 2), 6: R(-1, 6)}),
    ('exp(sinx)', sp.exp(sp.sin(x)),
     {0: R(1), 1: R(1), 2: R(1, 2), 3: R(0), 4: R(-1, 8), 5: R(-1, 15), 6: R(-1, 240)}),
    ('sin(x^2)', sp.sin(x ** 2), {2: R(1), 6: R(-1, 6)}),
    ('ln(1+x^2)', sp.log(1 + x ** 2), {2: R(1), 4: R(-1, 2), 6: R(1, 3)}),
    ('sqrt(1+x^2)', sp.sqrt(1 + x ** 2), {0: R(1), 2: R(1, 2), 4: R(-1, 8), 6: R(1, 16)}),
    ('sinx/x', sp.sin(x) / x, {0: R(1), 2: R(-1, 6), 4: R(1, 120), 6: R(-1, 5040)}),
    ('tanx-sinx', sp.tan(x) - sp.sin(x), {3: R(1, 2), 5: R(1, 8), 7: R(13, 240)}),
    ('asinx-atanx', sp.asin(x) - sp.atan(x), {3: R(1, 2), 5: R(-1, 8), 7: R(3, 16)}),
    ('x/(1-x^2)', x / (1 - x ** 2), {1: R(1), 3: R(1), 5: R(1), 7: R(1)}),
]

# ---------------------------------------------------------------- §6 x0 != 0
SHIFT = [
    ('exp@1', sp.exp(x), 1, {0: sp.E, 1: sp.E, 2: sp.E / 2, 3: sp.E / 6}),
    ('lnx@1', sp.log(x), 1, {0: R(0), 1: R(1), 2: R(-1, 2), 3: R(1, 3), 4: R(-1, 4)}),
    ('sqrtx@4', sp.sqrt(x), 4, {0: R(2), 1: R(1, 4), 2: R(-1, 64), 3: R(1, 512)}),
    ('1/x@1', 1 / x, 1, {0: R(1), 1: R(-1), 2: R(1), 3: R(-1)}),
    ('sinx@pi/2', sp.sin(x), sp.pi / 2, {0: R(1), 2: R(-1, 2), 4: R(1, 24)}),
]

# ---------------------------------------------------------------- §7 反求 f^(n)(0)
DERIV = [
    ('x^2e^x', x ** 2 * sp.exp(x), N * (N - 1), [0, 1, 2, 3, 4, 5, 6]),
    ('1/(1-x)', 1 / (1 - x), sp.factorial(N), [0, 1, 2, 3, 4, 5]),
    ('exp(-x^2)', sp.exp(-x ** 2), None, [0, 1, 2, 3, 4, 5, 6]),
    ('atanx', sp.atan(x), None, [0, 1, 2, 3, 4, 5, 6, 7]),
    ('sinx/x', sp.sin(x) / x, None, [0, 1, 2, 3, 4, 5, 6]),
    ('ln(1+x)/x', sp.log(1 + x) / x, None, [0, 1, 2, 3, 4, 5]),
    ('cos(sqrt x)', sp.cos(sp.sqrt(x)), None, [0, 1, 2, 3, 4, 5, 6]),
    ('x/(1-x^2)', x / (1 - x ** 2), None, [0, 1, 2, 3, 4, 5, 6, 7]),
]

# 反求题的「我给出的公式」用 python 函数写，方便逐 n 比
DERIV_FORMULA = {
    'exp(-x^2)': lambda n: (sp.Integer(-1) ** (n // 2) * sp.factorial(n) / sp.factorial(n // 2)) if n % 2 == 0 else sp.Integer(0),
    'atanx': lambda n: (sp.Integer(-1) ** ((n - 1) // 2) * sp.factorial(n - 1)) if n % 2 == 1 else sp.Integer(0),
    'sinx/x': lambda n: (sp.Integer(-1) ** (n // 2) / (n + 1)) if n % 2 == 0 else sp.Integer(0),
    'ln(1+x)/x': lambda n: sp.Integer(-1) ** n * sp.factorial(n) / (n + 1),
    # cos(sqrt x) = sum_k (-1)^k x^k / (2k)!  —— x 的幂次就是 k，故 [x^n]=(-1)^n/(2n)!
    'cos(sqrt x)': lambda n: sp.factorial(n) * sp.Integer(-1) ** n / sp.factorial(2 * n),
    'x/(1-x^2)': lambda n: sp.factorial(n) if n % 2 == 1 else sp.Integer(0),
}


def run():
    for name, expr, coefs in CORE:
        chk(name, expr, coefs)
    for name, expr, coefs in POWER:
        chk(name, expr, coefs)
    for name, expr, coefs in MISC:
        chk(name, expr, coefs)
    for name, expr, coefs in COMPOSITE:
        chk(name, expr, coefs)
    for name, expr, x0, coefs in SHIFT:
        chk(name, expr, coefs, x0=x0)

    # 反求 f^(n)(0)
    for name, expr, formula, ns in DERIV:
        ok, bad = True, []
        for n in ns:
            if formula is None:
                want = DERIV_FORMULA[name](n)
            else:
                want = formula.subs(N, n)
                if want == 0:
                    want = sp.Integer(0)
            # sinx/x、ln(1+x)/x、cos(sqrt x) 在 x=0 处本身无定义，只能取右极限
            got = sp.simplify(sp.limit(sp.diff(expr, x, n), x, 0, '+'))
            if sp.simplify(got - want) != 0:
                ok, _ = False, bad.append((n, str(want), str(got)))
        if ok:
            PASS.append('f^(n)(0):' + name)
        else:
            FAIL.append(('f^(n)(0):' + name, bad))

    print('=' * 56)
    print('PASS %d   FAIL %d' % (len(PASS), len(FAIL)))
    for name, bad in FAIL:
        print('FAIL', name)
        if isinstance(bad, list):
            for k, w, g in bad:
                print('    k=%s want=%s got=%s' % (k, w, g))
        else:
            print('   ', bad)
    return len(FAIL) == 0


if __name__ == '__main__':
    run()
