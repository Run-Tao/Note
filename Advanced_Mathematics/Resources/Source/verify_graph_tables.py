# -*- coding: utf-8 -*-
"""函数性态作图 · 列表式讲义：先复核、再生成 8 张五行性态列表。

用法：python verify_graph_tables.py
输出：期中复习/_graph_tables_generated.tex（表格片段，供讲义 \input）
      期中复习/讲义_函数性态作图_列表式_11-09.tex（模板注入后）
只有全部复核 PASS 才写出讲义。
"""
import io
import os
import sys

import sympy as sp

x = sp.symbols('x', real=True)
BASE = r'D:/Run_Tao/Homework/Advanced Mathematics/期中复习'
TEMPLATE = os.path.join(BASE, '_graph_list_notes_template.tex')
OUT_TEX = os.path.join(BASE, '讲义_函数性态作图_列表式_11-09.tex')
OUT_TABLES = os.path.join(BASE, '_graph_tables_generated.tex')

NP = NF = 0


def chk(label, got, want, tol=None):
    global NP, NF
    if tol is not None:
        ok = abs(float(sp.N(got)) - float(sp.N(want))) <= tol
    elif got in (sp.oo, -sp.oo, sp.zoo, sp.nan) or want in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        # oo - oo 会变成 nan，无穷大只能直接比较
        ok = got == want
    else:
        try:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
        except Exception:
            ok = str(got) == str(want)
    if ok:
        NP += 1
    else:
        NF += 1
        print('FAIL:', label, '| got =', got, '| want =', want)
    return ok


def true(label, cond):
    global NP, NF
    if cond:
        NP += 1
    else:
        NF += 1
        print('FAIL:', label)
    return cond


def sgn(v):
    try:
        f = float(sp.N(v, 30))
    except Exception:
        return None
    if abs(f) < 1e-9:
        return '0'
    return '+' if f > 0 else '-'


def at(expr, t):
    try:
        return sgn(sp.N(expr.subs(x, t), 30))
    except Exception:
        return None


def val_at(expr, p):
    """分点处导数值：'0' / '不存在' / '+' / '-'"""
    try:
        v = sp.simplify(expr.subs(x, p))
    except Exception:
        return '不存在'
    if v is sp.nan or v.is_finite is False:
        return '不存在'
    if v == 0:
        return '0'
    return sgn(v) or '?'


def tex_sym(s):
    if s == '不存在' or s == '—' or s == '不存在':
        return s
    return r'$%s$' % s


# ---------------------------------------------------------------------------
# 8 个函数：解析式、讲义给出的 y'/y''、分点、渐近线断言
# kind: max / min / break / infl / flat / boundary
# ---------------------------------------------------------------------------
E = sp.E
sqrt = sp.sqrt

CASES = [
    dict(key='F1', f=x ** 2 / (x - 1),
         fp_sym=x * (x - 2) / (x - 1) ** 2, fpp_sym=2 / (x - 1) ** 3,
         pts=[dict(v=sp.Integer(0), lab=r'$0$', kind='max', ytxt=r'极大 $0$'),
              dict(v=sp.Integer(1), lab=r'$1$', kind='break', ytxt=r'无定义'),
              dict(v=sp.Integer(2), lab=r'$2$', kind='min', ytxt=r'极小 $4$')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('F1 斜渐近线 k', sp.limit(f / x, x, sp.oo), 1),
             ('F1 斜渐近线 b', sp.limit(f - x, x, sp.oo), 1),
             ('F1 垂直左极限', sp.limit(f, x, 1, dir='-'), -sp.oo),
             ('F1 垂直右极限', sp.limit(f, x, 1, dir='+'), sp.oo),
             ('F1 极大值', f.subs(x, 0), 0),
             ('F1 极小值', f.subs(x, 2), 4),
         ]),
    dict(key='F2', f=x ** 2 * sp.exp(-x),
         fp_sym=x * (2 - x) * sp.exp(-x), fpp_sym=(x ** 2 - 4 * x + 2) * sp.exp(-x),
         pts=[dict(v=sp.Integer(0), lab=r'$0$', kind='min', ytxt=r'极小 $0$'),
              dict(v=2 - sqrt(2), lab=r'$2-\sqrt2$', kind='infl', ytxt=r'拐点'),
              dict(v=sp.Integer(2), lab=r'$2$', kind='max', ytxt=r'极大 $4/\ee^2$'),
              dict(v=2 + sqrt(2), lab=r'$2+\sqrt2$', kind='infl', ytxt=r'拐点')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('F2 右端水平 y=0', sp.limit(f, x, sp.oo), 0),
             ('F2 左端发散', sp.limit(f, x, -sp.oo), sp.oo),
             ('F2 极大值', sp.simplify(f.subs(x, 2) - 4 * sp.exp(-2)), 0),
             ('F2 拐点个数', len(sp.solve(sp.diff(f, x, 2), x)), 2),
         ]),
    dict(key='F3', f=sp.log(x) / x,
         fp_sym=(1 - sp.log(x)) / x ** 2, fpp_sym=(2 * sp.log(x) - 3) / x ** 3,
         pts=[dict(v=sp.Integer(0), lab=r'$0^+$', kind='boundary', ytxt=r'$-\infty$'),
              dict(v=E, lab=r'$\ee$', kind='max', ytxt=r'极大 $1/\ee$'),
              dict(v=E ** sp.Rational(3, 2), lab=r'$\ee^{3/2}$', kind='infl', ytxt=r'拐点')],
         left=None, right=r'+\infty',
         checks=lambda f: [
             ('F3 右端水平 y=0', sp.limit(f, x, sp.oo), 0),
             ('F3 垂直右极限', sp.limit(f, x, 0, dir='+'), -sp.oo),
             ('F3 极大值', sp.simplify(f.subs(x, E) - 1 / E), 0),
             ('F3 拐点位置', sp.simplify(sp.diff(f, x, 2).subs(x, E ** sp.Rational(3, 2))), 0),
         ]),
    dict(key='F4', f=x / (x ** 2 - 1),
         fp_sym=-(x ** 2 + 1) / (x ** 2 - 1) ** 2, fpp_sym=2 * x * (x ** 2 + 3) / (x ** 2 - 1) ** 3,
         pts=[dict(v=sp.Integer(-1), lab=r'$-1$', kind='break', ytxt=r'无定义'),
              dict(v=sp.Integer(0), lab=r'$0$', kind='infl', ytxt=r'拐点 $(0,0)$'),
              dict(v=sp.Integer(1), lab=r'$1$', kind='break', ytxt=r'无定义')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('F4 奇函数', sp.simplify(f.subs(x, -x) + f), 0),
             ('F4 水平渐近线', sp.limit(f, x, sp.oo), 0),
             ('F4 y\' 恒负', sp.simplify(sp.diff(f, x) + (x ** 2 + 1) / (x ** 2 - 1) ** 2), 0),
             ('F4 拐点', sp.simplify(sp.diff(f, x, 2).subs(x, 0)), 0),
         ]),
    dict(key='F5', f=x + 1 / x,
         fp_sym=(x ** 2 - 1) / x ** 2, fpp_sym=2 / x ** 3,
         pts=[dict(v=sp.Integer(-1), lab=r'$-1$', kind='max', ytxt=r'极大 $-2$'),
              dict(v=sp.Integer(0), lab=r'$0$', kind='break', ytxt=r'无定义'),
              dict(v=sp.Integer(1), lab=r'$1$', kind='min', ytxt=r'极小 $2$')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('F5 斜渐近线 k', sp.limit(f / x, x, sp.oo), 1),
             ('F5 斜渐近线 b', sp.limit(f - x, x, sp.oo), 0),
             ('F5 垂直右极限', sp.limit(f, x, 0, dir='+'), sp.oo),
             ('F5 极大值', f.subs(x, -1), -2),
             ('F5 极小值', f.subs(x, 1), 2),
         ]),
    dict(key='F6', f=x ** 3 / (x ** 2 + 1),
         fp_sym=x ** 2 * (x ** 2 + 3) / (x ** 2 + 1) ** 2,
         fpp_sym=2 * x * (3 - x ** 2) / (x ** 2 + 1) ** 3,
         pts=[dict(v=-sqrt(3), lab=r'$-\sqrt3$', kind='infl', ytxt=r'拐点'),
              dict(v=sp.Integer(0), lab=r'$0$', kind='flat', ytxt=r'非极值（拐点）'),
              dict(v=sqrt(3), lab=r'$\sqrt3$', kind='infl', ytxt=r'拐点')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('F6 斜渐近线 k', sp.limit(f / x, x, sp.oo), 1),
             ('F6 斜渐近线 b', sp.limit(f - x, x, sp.oo), 0),
             ('F6 奇函数', sp.simplify(f.subs(x, -x) + f), 0),
             ('F6 y\' 在 0 处为 0 但不变号', sp.simplify(sp.diff(f, x).subs(x, 0)), 0),
             ('F6 y\' 非负', sp.simplify(sp.diff(f, x).subs(x, 1) - sp.Rational(1, 1)), 0),
         ]),
    dict(key='L1', f=(x - 1) ** 2 / (x + 1),
         fp_sym=(x - 1) * (x + 3) / (x + 1) ** 2, fpp_sym=8 / (x + 1) ** 3,
         pts=[dict(v=sp.Integer(-3), lab=r'$-3$', kind='max', ytxt=r'极大 $-8$'),
              dict(v=sp.Integer(-1), lab=r'$-1$', kind='break', ytxt=r'无定义'),
              dict(v=sp.Integer(1), lab=r'$1$', kind='min', ytxt=r'极小 $0$')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('L1 多项式部分', sp.simplify(f - (x - 3 + 4 / (x + 1))), 0),
             ('L1 斜渐近线 k', sp.limit(f / x, x, sp.oo), 1),
             ('L1 斜渐近线 b', sp.limit(f - x, x, sp.oo), -3),
             ('L1 极大值', f.subs(x, -3), -8),
             ('L1 极小值', f.subs(x, 1), 0),
         ]),
    dict(key='L2', f=x * sp.exp(-x ** 2 / 2),
         fp_sym=(1 - x ** 2) * sp.exp(-x ** 2 / 2),
         fpp_sym=x * (x ** 2 - 3) * sp.exp(-x ** 2 / 2),
         pts=[dict(v=-sqrt(3), lab=r'$-\sqrt3$', kind='infl', ytxt=r'拐点'),
              dict(v=sp.Integer(-1), lab=r'$-1$', kind='min', ytxt=r'极小'),
              dict(v=sp.Integer(0), lab=r'$0$', kind='infl', ytxt=r'拐点'),
              dict(v=sp.Integer(1), lab=r'$1$', kind='max', ytxt=r'极大'),
              dict(v=sqrt(3), lab=r'$\sqrt3$', kind='infl', ytxt=r'拐点')],
         left=r'-\infty', right=r'+\infty',
         checks=lambda f: [
             ('L2 奇函数', sp.simplify(f.subs(x, -x) + f), 0),
             ('L2 右端水平 y=0', sp.limit(f, x, sp.oo), 0),
             ('L2 左端水平 y=0', sp.limit(f, x, -sp.oo), 0),
             ('L2 极大值', sp.simplify(f.subs(x, 1) ** 2 - sp.exp(-1)), 0),
             ('L2 拐点个数', len(sp.solve(sp.diff(f, x, 2), x)), 3),
         ]),
]


def interval_cell(c, a, b, la, lb, fp, fpp):
    if a is None:
        t = float(sp.N(b)) - 1.0
    elif b is None:
        t = float(sp.N(a)) + 1.0
    else:
        t = (float(sp.N(a)) + float(sp.N(b))) / 2.0
    s1, s2 = at(fp, t), at(fpp, t)
    true('%s 区间 (%s,%s) 的 y\' 符号可判' % (c['key'], la, lb), s1 is not None)
    true('%s 区间 (%s,%s) 的 y\'\' 符号可判' % (c['key'], la, lb), s2 is not None)
    return dict(head=r'$(%s,%s)$' % (la, lb),
                yp=tex_sym(s1 or '?'),
                y=(r'$\nearrow$' if s1 == '+' else (r'$\searrow$' if s1 == '-' else r'—')),
                ypp=tex_sym(s2 or '?'),
                cv=('凹' if s2 == '+' else ('凸' if s2 == '-' else '—')))


def point_cell(p, fp, fpp):
    lab, kind, v = p['lab'], p['kind'], p['v']
    if kind == 'break':
        return dict(head=lab, yp='不存在', y=p['ytxt'], ypp='不存在', cv='——')
    if kind == 'boundary':
        return dict(head=lab, yp='—', y=p['ytxt'], ypp='—', cv='—')
    s1, s2 = val_at(fp, v), val_at(fpp, v)
    curv = ('拐点' if s2 == '0' else
            ('——' if s2 == '不存在' else ('凹' if s2 == '+' else '凸')))
    return dict(head=lab, yp=tex_sym(s1), y=p['ytxt'], ypp=tex_sym(s2), cv=curv)


def make_table(c):
    f = c['f']
    fp = sp.factor(sp.diff(f, x))
    fpp = sp.factor(sp.diff(f, x, 2))
    # 复核讲义给出的导数解析式
    chk("%s 讲义 y' 与 sympy 一致" % c['key'], sp.simplify(fp - c['fp_sym']), 0)
    chk("%s 讲义 y'' 与 sympy 一致" % c['key'], sp.simplify(fpp - c['fpp_sym']), 0)
    for lab, got, want in c['checks'](f):
        chk(lab, got, want)

    pts = c['pts']
    n = len(pts)
    cells = []
    if pts[0]['kind'] != 'boundary':
        cells.append(interval_cell(c, None, pts[0]['v'], c['left'],
                                   pts[0]['lab'].strip('$'), fp, fpp))
    cells.append(point_cell(pts[0], fp, fpp))
    for i in range(n - 1):
        a, b = pts[i]['v'], pts[i + 1]['v']
        cells.append(interval_cell(c, a, b, pts[i]['lab'].strip('$'),
                                   pts[i + 1]['lab'].strip('$'), fp, fpp))
        cells.append(point_cell(pts[i + 1], fp, fpp))
    if pts[-1]['kind'] != 'boundary':
        cells.append(interval_cell(c, pts[-1]['v'], None,
                                   pts[-1]['lab'].strip('$'), c['right'], fp, fpp))

    widths = 'c' * len(cells)
    lines = [
        r'\begin{center}',
        r'\footnotesize\setlength{\tabcolsep}{1.5pt}',
        r'\begin{tabular}{@{}c|%s@{}}' % widths,
        '$x$ & ' + ' & '.join(z['head'] for z in cells) + r' \\',
        r'\hline',
        "$y'$ & " + ' & '.join(z['yp'] for z in cells) + r' \\',
        '$y$ & ' + ' & '.join(z['y'] for z in cells) + r' \\',
        "$y''$ & " + ' & '.join(z['ypp'] for z in cells) + r' \\',
        '凹凸 & ' + ' & '.join(z['cv'] for z in cells) + r' \\',
        r'\end{tabular}',
        r'\end{center}',
    ]
    return '\n'.join(lines)


def main():
    tables = {}
    for c in CASES:
        tables['@@TAB_%s@@' % c['key']] = make_table(c)
    print('复核汇总: PASS %d / FAIL %d' % (NP, NF))
    if NF:
        print('存在复核失败，拒绝生成讲义。')
        return 1

    with io.open(OUT_TABLES, 'w', encoding='utf-8') as fh:
        fh.write('% 由 verify_graph_tables.py 自动生成，请勿手改\n')
        for k, v in tables.items():
            fh.write('\n%% %s\n%s\n' % (k, v))

    tpl = io.open(TEMPLATE, encoding='utf-8').read()
    for k, v in tables.items():
        if k not in tpl:
            print('模板缺少占位符', k)
            return 1
        tpl = tpl.replace(k, v)
    io.open(OUT_TEX, 'w', encoding='utf-8').write(tpl)
    print('已生成', OUT_TEX)
    print('表格数', len(tables))
    return 0


if __name__ == '__main__':
    sys.exit(main())
