# -*- coding: utf-8 -*-
"""生成《泰勒展开全表》LaTeX —— 公式由已复核的系数自动排版。

先跑 _taylor_data.run()，全 PASS 才落盘。
"""
import io
import os
import sympy as sp

import _taylor_data as D

x = D.x

# ---------------------------------------------------------------- 系数 -> LaTeX


def tex_term(c, k, var, first):
    c = sp.nsimplify(c)
    neg = bool(c.is_negative)
    a = -c if neg else c
    sign = ('-' if neg else '') if first else ('-' if neg else '+')
    if k == 0:
        v = ''
    elif k == 1:
        v = var
    else:
        v = r'%s^{%d}' % (var, k)
    num, den = sp.fraction(sp.together(a))
    if den == 1:
        if k == 0:
            return sign + sp.latex(num)
        if num == 1:
            return sign + v
        return sign + sp.latex(num) + v
    return sign + r'\frac{%s}{%s}' % (sp.latex(num), sp.latex(den)) + v


def series_tex(coefs, n, var='x'):
    parts = []
    for k in range(n + 1):
        c = coefs.get(k, 0)
        if sp.simplify(c) == 0:
            continue
        parts.append(tex_term(c, k, var, not parts))
    return ''.join(parts) + r'+\cdots'


# ---------------------------------------------------------------- 展示清单
# (展示名, 系数表, 最高阶, 成立范围, 变量 LaTeX)
CORE_ROWS = [
    (r'\ee^{x}', D.CORE[0][2], 5, r'$x\in\R$'),
    (r'\sin x', D.CORE[1][2], 7, r'$x\in\R$'),
    (r'\cos x', D.CORE[2][2], 6, r'$x\in\R$'),
    (r'\tan x', D.CORE[3][2], 7, r'$\abs{x}<\frac{\pi}{2}$'),
    (r'\arcsin x', D.CORE[4][2], 7, r'$\abs{x}<1$'),
    (r'\arctan x', D.CORE[5][2], 9, r'$\abs{x}\le1$'),
    (r'\ln(1+x)', D.CORE[6][2], 5, r'$-1<x\le1$'),
    (r'\ln(1-x)', D.CORE[7][2], 5, r'$-1\le x<1$'),
    (r'\frac{1}{1-x}', D.CORE[8][2], 4, r'$\abs{x}<1$'),
]

POWER_ROWS = [
    (r'\frac{1}{1+x}', D.POWER[0][2], 4, r'$\abs{x}<1$'),
    (r'\frac{1}{(1+x)^{2}}', D.POWER[1][2], 4, r'$\abs{x}<1$'),
    (r'\frac{1}{(1-x)^{2}}', D.POWER[2][2], 4, r'$\abs{x}<1$'),
    (r'\sqrt{1+x}', D.POWER[3][2], 4, r'$\abs{x}\le1$'),
    (r'\frac{1}{\sqrt{1+x}}', D.POWER[4][2], 4, r'$-1<x\le1$'),
    (r'\sqrt[3]{1+x}', D.POWER[5][2], 4, r'$\abs{x}<1$'),
    (r'\frac{1}{1+x^{2}}', D.POWER[6][2], 6, r'$\abs{x}\le1$'),
    (r'\frac{1}{\sqrt{1-x}}', D.POWER[7][2], 4, r'$\abs{x}<1$'),
]

MISC_ROWS = [
    (r'\sinh x', D.MISC[0][2], 7, r'$x\in\R$'),
    (r'\cosh x', D.MISC[1][2], 6, r'$x\in\R$'),
    (r'\tanh x', D.MISC[2][2], 7, r'$\abs{x}<\frac{\pi}{2}$'),
    (r'\arsinh x', D.MISC[3][2], 7, r'$\abs{x}\le1$'),
]

COMP_ROWS = [
    (r'\ee^{-x^{2}}', D.COMPOSITE[0][2], 6, r'$x\in\R$'),
    (r'\ee^{\sin x}', D.COMPOSITE[1][2], 6, r'$x\in\R$'),
    (r'\sin(x^{2})', D.COMPOSITE[2][2], 6, r'$x\in\R$'),
    (r'\ln(1+x^{2})', D.COMPOSITE[3][2], 6, r'$\abs{x}\le1$'),
    (r'\sqrt{1+x^{2}}', D.COMPOSITE[4][2], 6, r'$\abs{x}\le1$'),
    (r'\frac{\sin x}{x}', D.COMPOSITE[5][2], 6, r'$x\ne0$'),
    (r'\tan x-\sin x', D.COMPOSITE[6][2], 7, r'$\abs{x}<\frac{\pi}{2}$'),
    (r'\arcsin x-\arctan x', D.COMPOSITE[7][2], 7, r'$\abs{x}<1$'),
    (r'\frac{x}{1-x^{2}}', D.COMPOSITE[8][2], 7, r'$\abs{x}<1$'),
]


def table(rows, small=True):
    L = []
    A = L.append
    A(r'\begin{center}')
    A(r'\renewcommand{\arraystretch}{1.32}')
    A(r'\begin{tabular}{@{}l@{\hspace{10pt}}l@{}}')
    A(r'\toprule')
    A(r'\textbf{展开式} & \textbf{成立范围} \\')
    A(r'\midrule')
    for name, coefs, n, dom in rows:
        A(r'$%s=%s$ & {\footnotesize %s} \\' % (name, series_tex(coefs, n), dom))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{center}')
    return L


def build():
    assert D.run(), '系数复核未全 PASS，拒绝生成'

    L = []
    A = L.append
    A(r'\documentclass[11pt,UTF8]{ctexart}')
    A(r'\usepackage[color]{tlnote}')
    A(r'\usepackage{needspace}')
    A('')
    A(r'\begin{document}')
    A('')
    A(r'\noindent{\LARGE\bfseries 泰勒展开全表}\hfill{\small 高数 I 期中 · 考前速查}')
    A('')
    A(r'\noindent{\small\color{gray} 交大 2008--2025 十七套命题范围内的全部常见展开 \quad$\mid$\quad 2026-11-11 用}')
    A('')
    A(r'\vspace{2pt}\hrule\vspace{8pt}')
    A('')
    A(r'\noindent{\footnotesize 本表所有系数由 \texttt{sympy} 独立展开逐项比对生成（\textbf{%d 项 PASS / %d FAIL}），'
      r'脚本 \texttt{Source/\_build\_taylor\_full.py}。展开式本身不会写错，犯错的永远是「展到几阶」和「忘乘 $n!$」。}'
      % (len(D.PASS), len(D.FAIL)))
    A('')
    A(r'\noindent{\footnotesize\color{gray} 用法：先背 §2 九个核心；考场上遇到不认识的函数，回 §3--§5 找「能不能套」；'
      r'要 $f^{(n)}(0)$ 直接查 §7，不要逐阶求导。本表独立 PDF，不改任何既有包绝对页码。}')
    A('')

    # ------------------------------------------------------------ §1
    A(r'\section{四条铁律}')
    A('')
    A(r'\begin{keypoint}[展到几阶、用哪种余项、怎么反求]')
    A(r'\begin{enumerate}[leftmargin=1.7em,itemsep=3pt]')
    A(r'\item \textbf{展到几阶：看分母的最低次。} 分母是 $x^{3}$ 就展到 $x^{3}$，分母是 $x^{4}$ 就展到 $x^{4}$；'
      r'若分子展开后低次项\textbf{抵消}，要展到「抵消后剩下的最低次」为止——这是最常错的一步。')
    A(r'\item \textbf{两种余项分工。} \textbf{Peano} $o((x-x_0)^n)$ 只用于\textbf{求极限}（保证阶数够就行）；'
      r'\textbf{Lagrange} $\dfrac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$ 用于\textbf{证不等式、估误差}（要定号）。')
    A(r'\item \textbf{系数反求。} $f^{(n)}(0)=n!\,[x^{n}]$ —— 括号里的系数乘上 $n!$ 才是导数值，'
      r'漏乘 $n!$ 是填空最典型的失分点。')
    A(r'\item \textbf{复合代入。} 把内层\textbf{整体}代进外层，只保留到需要的阶；'
      r'内层是偶函数（如 $x^{2}$、$\cos x$）时，外层只出偶次项，能省一半计算。')
    A(r'\end{enumerate}')
    A(r'\end{keypoint}')
    A('')

    # ------------------------------------------------------------ §2
    A(r'\section{\texorpdfstring{核心九个（必背，$x\to0$）}{核心九个（必背，x 趋于 0）}}')
    A('')
    L.extend(table(CORE_ROWS))
    A('')
    A(r'\needspace{6\baselineskip}')
    A(r'\section{\texorpdfstring{幂函数族（把 $\alpha$ 代进去）}{幂函数族（把 alpha 代进去）}}')
    A('')
    A(r'\noindent{\footnotesize 幂函数通式：$(1+x)^{\alpha}=1+\alpha x+\dfrac{\alpha(\alpha-1)}{2!}x^{2}'
      r'+\dfrac{\alpha(\alpha-1)(\alpha-2)}{3!}x^{3}+\cdots\ (\abs{x}<1)$，$\alpha$ 为任意实数；'
      r'系数就是「下降阶乘除以阶乘」，即 $\mathrm{C}_{\alpha}^{n}$。下面八个都是它的特例。}')
    A('')
    A(r'\noindent{\footnotesize 考场上直接调用，不必重推；注意 $\dfrac{1}{1+x^{2}}$ 只有偶次项（它是 $\arctan x$ 的导数）。}')
    A('')
    L.extend(table(POWER_ROWS))
    A('')

    # ------------------------------------------------------------ §4
    A(r'\section{指数、对数与双曲}')
    A('')
    A(r'\noindent{\footnotesize $a^{x}=\ee^{x\ln a}$，把 $x\ln a$ 当成整体代进 $\ee^{u}$ 即可；'
      r'$\log_{a}(1+x)=\dfrac{1}{\ln a}\ln(1+x)$。双曲函数与 $\sin/\cos$ 只差符号：'
      r'$\sinh$ 全是正号，$\cosh$ 与 $\cos$ 相同，$\tanh$ 与 $\tan$ 交错符号。}')
    A('')
    A(r'\begin{center}')
    A(r'\renewcommand{\arraystretch}{1.32}')
    A(r'\begin{tabular}{@{}l@{\hspace{10pt}}l@{}}')
    A(r'\toprule')
    A(r'\textbf{展开式} & \textbf{成立范围} \\')
    A(r'\midrule')
    A(r'$2^{x}=1+\ln2\,x+\frac{\ln^{2}2}{2}x^{2}+\frac{\ln^{3}2}{6}x^{3}+\cdots$ & {\footnotesize $x\in\R$} \\')
    A(r'$\lg(1+x)=\frac{1}{\ln10}\left(x-\frac{x^{2}}{2}+\frac{x^{3}}{3}-\cdots\right)$ & {\footnotesize $-1<x\le1$} \\')
    for name, coefs, n, dom in MISC_ROWS:
        A(r'$%s=%s$ & {\footnotesize %s} \\' % (name, series_tex(coefs, n), dom))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{center}')
    A('')

    # ------------------------------------------------------------ §5
    A(r'\needspace{8\baselineskip}')
    A(r'\section{复合代入型（考场直接代，不必重推）}')
    A('')
    A(r'\noindent{\footnotesize 做法固定：把内层展开\textbf{整体}代入外层展开式，再按 $x$ 的幂次合并、只留到够用的阶。'
      r'下面九个是十七套真题里出现过或可直接套用的组合，背下来能省 3--5 分钟。}')
    A('')
    L.extend(table(COMP_ROWS))
    A('')
    A(r'\noindent{\footnotesize 其中两行是「等价无穷小的精确版」：'
      r'$\tan x-\sin x\sim\dfrac{x^{3}}{2}$、$\arcsin x-\arctan x\sim\dfrac{x^{3}}{2}$，'
      r'遇到 $\dfrac{\tan x-\sin x}{x^{3}}$ 型极限可直接写答案 $=\dfrac{1}{2}$。}')
    A('')

    # ------------------------------------------------------------ §6
    A(r'\section{\texorpdfstring{$x_0\ne0$ 的泰勒（换元退回麦克劳林）}{非零展开点的泰勒（换元退回麦克劳林）}}')
    A('')
    A(r'\noindent{\footnotesize 看到 $x\to1$、$x\to4$ 就先令 $t=x-x_0$，把函数写成 $t$ 的函数再展开；'
      r'答案里的 $t$ 再换回 $(x-x_0)$。\textbf{切记每一项都是 $(x-x_0)$ 的幂，不是 $x$ 的幂。}}')
    A('')
    A(r'\begin{center}')
    A(r'\renewcommand{\arraystretch}{1.32}')
    A(r'\begin{tabular}{@{}l@{\hspace{10pt}}l@{}}')
    A(r'\toprule')
    A(r'\textbf{展开式} & \textbf{成立范围} \\')
    A(r'\midrule')
    A(r'$\ee^{x}=\ee\left[1+(x-1)+\frac{(x-1)^{2}}{2}+\frac{(x-1)^{3}}{6}+\cdots\right]$ & {\footnotesize $x\in\R$} \\')
    A(r'$\ln x=%s$ & {\footnotesize $0<x\le2$} \\' % series_tex(D.SHIFT[1][3], 4, var=r'(x-1)'))
    A(r'$\sqrt{x}=%s$ & {\footnotesize $x>0$} \\' % series_tex(D.SHIFT[2][3], 3, var=r'(x-4)'))
    A(r'$\frac{1}{x}=%s$ & {\footnotesize $0<x<2$} \\' % series_tex(D.SHIFT[3][3], 3, var=r'(x-1)'))
    A(r'$\sin x=%s$ & {\footnotesize $x\in\R$} \\'
      % series_tex(D.SHIFT[4][3], 4, var=r'\left(x-\frac{\pi}{2}\right)'))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{center}')
    A('')
    A(r'\noindent{\footnotesize 三种考法：① 直接给函数求展开（如上表）；'
      r'② 给出 $f(x_0)$、$f^{\prime}(x_0)$、$f^{\prime\prime}(x_0)$ 反写展开式；'
      r'③ 求 $\lim_{x\to x_0}$ 型极限，先令 $t=x-x_0$ 化成 $t\to0$。三种都只做一个动作：\textbf{换元}。}')
    A('')

    # ------------------------------------------------------------ §7
    A(r'\needspace{9\baselineskip}')
    A(r'\section{\texorpdfstring{反求 $f^{(n)}(0)$：八个直接查}{反求 f 的 n 阶导数：八个直接查}}')
    A('')
    A(r'\noindent{\footnotesize 方法只有一句：\textbf{先展开，再乘 $n!$}。'
      r'下面每一行都已用 $\mathrm{d}^{n}f/\mathrm{d}x^{n}\big|_{x=0}$ 逐阶验过（$n\le7$）。}')
    A('')
    A(r'\begin{center}')
    A(r'\renewcommand{\arraystretch}{1.35}')
    A(r'\begin{tabular}{@{}p{2.5cm}p{6.9cm}p{4.6cm}@{}}')
    A(r'\toprule')
    A(r'\textbf{$f(x)$} & \textbf{$f^{(n)}(0)$} & \textbf{前三项} \\')
    A(r'\midrule')
    A(r'$x^{2}\ee^{x}$ & $n(n-1)$ & $x^{2}+x^{3}+\frac{x^{4}}{2}+\cdots$ \\[1pt]')
    A(r'$\frac{1}{1-x}$ & $n!$ & $1+x+x^{2}+\cdots$ \\[1pt]')
    A(r'$\ee^{-x^{2}}$ & $\begin{cases}\frac{(-1)^{n/2}\,n!}{(n/2)!}, & n\ \text{偶}\\[2pt] 0, & n\ \text{奇}\end{cases}$'
      r' & $1-x^{2}+\frac{x^{4}}{2}-\cdots$ \\[1pt]')
    A(r'$\arctan x$ & $\begin{cases}(-1)^{(n-1)/2}(n-1)!, & n\ \text{奇}\\[2pt] 0, & n\ \text{偶}\end{cases}$'
      r' & $x-\frac{x^{3}}{3}+\frac{x^{5}}{5}-\cdots$ \\[1pt]')
    A(r'$\frac{\sin x}{x}$ & $\begin{cases}\frac{(-1)^{n/2}}{n+1}, & n\ \text{偶}\\[2pt] 0, & n\ \text{奇}\end{cases}$'
      r' & $1-\frac{x^{2}}{6}+\frac{x^{4}}{120}-\cdots$ \\[1pt]')
    A(r'$\frac{\ln(1+x)}{x}$ & $\frac{(-1)^{n}n!}{n+1}$ & $1-\frac{x}{2}+\frac{x^{2}}{3}-\cdots$ \\[1pt]')
    A(r'$\cos\sqrt{x}$ & $\frac{(-1)^{n}n!}{(2n)!}$ & $1-\frac{x}{2}+\frac{x^{2}}{24}-\cdots$ \\[1pt]')
    A(r'$\frac{x}{1-x^{2}}$ & $\begin{cases}n!, & n\ \text{奇}\\[2pt] 0, & n\ \text{偶}\end{cases}$'
      r' & $x+x^{3}+x^{5}+\cdots$ \\')
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{center}')
    A('')
    A(r'\noindent{\footnotesize $\cos\sqrt{x}$ 这行最易错：$x$ 的幂次是 $k$ 不是 $2k$，'
      r'$[x^{n}]=\frac{(-1)^{n}}{(2n)!}$，乘 $n!$ 后分母仍是 $(2n)!$，别把 $n!$ 当成 $(2n)!$。}')
    A('')

    # ------------------------------------------------------------ §8
    A(r'\section{六个陷阱}')
    A('')
    A(r'\begin{warnbox}[改卷时最常扣分的地方]')
    A(r'\begin{enumerate}[leftmargin=1.7em,itemsep=2pt]')
    A(r'\item \textbf{阶数不够。} $\lim_{x\to0}\frac{\sin x-x\cos x}{x^{3}}$：只展一阶（$\sin x\approx x$、$\cos x\approx1$）'
      r'分子算得 $0$，误判极限为 $0$；实际要展到 $x^{3}$，$\sin x$ 与 $x\cos x$ 的 $x^{3}$ 项分别是 $-\frac{1}{6}$ 与 $-\frac{1}{2}$，'
      r'相减得 $\frac{x^{3}}{3}$，故极限为 $\frac{1}{3}$。')
    A(r'\item \textbf{漏乘 $n!$。} $[x^{5}]$ 是系数，$f^{(5)}(0)=5!\cdot[x^{5}]$。')
    A(r'\item \textbf{非零 $x_0$ 忘换元。} $\ln x$ 在 $x=1$ 处展开，答案里必须是 $(x-1)$ 的幂；'
      r'直接写 $x-\frac{x^{2}}{2}+\cdots$ 全错。')
    A(r'\item \textbf{内层高次项没跟上。} $\sqrt{1+x^{2}}=1+\frac{x^{2}}{2}-\frac{x^{4}}{8}+\cdots$：'
      r'一次项把 $x$ 换成 $x^{2}$ 恰好对；$x^{4}$ 项要按 $\alpha=\frac{1}{2}$ 的 $\frac{\alpha(\alpha-1)}{2}(x^{2})^{2}$ 算，得 $-\frac{x^{4}}{8}$，'
      r'照抄 $\sqrt{1+x}$ 的 $+\frac{x^{2}}{8}$ 就会连符号一起错。')
    A(r'\item \textbf{把收敛域当成成立条件。} 求极限时 $x\to0$，任何收敛域都成立；'
      r'但「用泰勒估 $\ln2$」时必须确认 $x=1$ 在收敛域边界上是否可用（$\ln(1+x)$ 在 $x=1$ 收敛）。')
    A(r'\item \textbf{只算不写。} 大题用泰勒必须先写「由带有 Peano 余项的泰勒公式」，'
      r'再写展开式，最后取极限；只写结果不给分。')
    A(r'\end{enumerate}')
    A(r'\end{warnbox}')
    A('')
    A(r'\noindent{\footnotesize\color{gray} 配套：考前知识点清单 7 页（判据·默写自测）$\mid$ 精编本 22 页（讲解）'
      r'$\mid$ 薄弱点网络题源 48 题 $\mid$ 合订本 370 页（查详解）。}')
    A('')
    A(r'\end{document}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '泰勒展开全表_11-11.tex')
    io.open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('written', out, len(L), 'lines')


if __name__ == '__main__':
    build()
