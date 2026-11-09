# -*- coding: utf-8 -*-
r"""装配《高等数学 I（A 类）期中复习讲义 · 分享版》的 LaTeX。

面向「分享给学弟学妹」重新编写：8 章全新讲解 + 33 道例题精讲（完整解答）
+ 151 道章末习题与答案速查 + 5 个附录；剥离全部个人排期 / 掐表 / 诊断 / 回执内容。

两道硬闸门，任一不过就拒绝生成：
  1) verify_share_handout.py 必须 PASS 全绿（exit 0）；
  2) 151 题题面与答案按题号配对、8 章全覆盖、例题数 33、环境配对。

排版严格走 A 方案：ctexart + \usepackage[color]{tlnote}，不手写 preamble、
不重复定义 tlnote 已提供的任何命令。

运行：PYTHONIOENCODING=utf-8 python build_share_handout.py
编译：xelatex -interaction=nonstopmode -halt-on-error 讲义_高数I期中_分享版.tex （两遍）
"""
import collections
import importlib.util
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, 'verify_share_handout.py')
PARSER = os.path.join(HERE, '_build_compact_handout.py')
NOTES = os.path.join(HERE, '_share_notes.tex')
OUT_TEX = os.path.join(HERE, '讲义_高数I期中_分享版.tex')

CH_TITLES = [
    '极限计算与无穷小阶',
    '极限定义证明：三套量词行',
    '连续、间断点与分段函数',
    '导数与微分：六型求导',
    '泰勒公式与高阶导数',
    '中值定理、不等式与最值',
    '函数性态与作图七步',
    '数列极限、迭代与新压轴',
]


def gate_verify():
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, VERIFY], cwd=HERE, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = [ln for ln in (r.stdout or '').splitlines() if ln.strip()][-3:]
    print('[闸门 1] 复核脚本 exit=%d  %s' % (r.returncode, ' | '.join(tail)))
    if r.returncode != 0:
        print((r.stdout or '')[-1500:])
        print((r.stderr or '')[-1500:])
        sys.exit('复核未全绿，拒绝生成讲义')


def load_questions():
    spec = importlib.util.spec_from_file_location('_bch_parser', PARSER)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    # 分享版必须自包含：题库源只取 Note 仓库内的同步快照，不依赖本机 Homework 目录
    b.SRC = HERE
    q, a = b.load()
    ids = sorted(set(q) & set(a), key=lambda s: (s.split('.')[0], int(s.split('.')[1])))
    orphan_q, orphan_a = sorted(set(q) - set(a)), sorted(set(a) - set(q))
    unmapped = [i for i in ids if b.chapter_of(i) is None]
    print('[闸门 2] 配对题数=%d  仅题面=%d  仅答案=%d  未映射=%d'
          % (len(ids), len(orphan_q), len(orphan_a), len(unmapped)))
    if len(ids) != 151 or orphan_q or orphan_a or unmapped:
        sys.exit('题库配对 / 章节映射不完整，拒绝生成讲义')
    return b, q, a, ids


def toc_star(s):
    r"""给 \section* 手动挂目录与 PDF 书签（hyperref 已由 tlnote 加载）。"""
    def rep(m):
        title = m.group(1).replace(r'\quad', ' ').strip()
        return ('\\phantomsection\\addcontentsline{toc}{section}{%s}\n' % title
                + m.group(0))
    return re.sub(r'\\section\*\{([^{}]*)\}', rep, s)


def split_notes():
    s = io.open(NOTES, encoding='utf-8').read()
    s = toc_star(s)
    ms = [(m.start(), m.end(), m.group(1)) for m in
          re.finditer(r'^%%(FRONT|CH[1-8]|BACK)%%\s*$', s, re.M)]
    keys = [k for _, _, k in ms]
    expected = ['FRONT'] + ['CH%d' % i for i in range(1, 9)] + ['BACK']
    if keys != expected:
        sys.exit('讲解分段标记异常：%s' % keys)
    chunks = {}
    for i, (st, en, key) in enumerate(ms):
        end = ms[i + 1][0] if i + 1 < len(ms) else len(s)
        chunks[key] = s[en:end].strip()
    ex = sum(chunks['CH%d' % i].count(r'\begin{exa}') for i in range(1, 9))
    print('[闸门 2] 讲解分段 10/10，例题精讲 %d 题' % ex)
    if ex != 33:
        sys.exit('例题数量与附录 E 声明不符，拒绝生成讲义')
    return chunks


def check_envs(tex):
    names = set(re.findall(r'\\begin\{([^}]+)\}', tex))
    bad = []
    for nm in sorted(names):
        nb = len(re.findall(r'\\begin\{' + re.escape(nm) + r'\}', tex))
        ne = len(re.findall(r'\\end\{' + re.escape(nm) + r'\}', tex))
        if nb != ne:
            bad.append('%s(%d/%d)' % (nm, nb, ne))
    if bad:
        sys.exit('LaTeX 环境未配对：' + ', '.join(bad))
    print('[闸门 2] LaTeX 环境配对 %d 类全部通过' % len(names))


PREAMBLE = r'''% !TeX program = xelatex
% ============================================================
%  高等数学 I（A 类）期中考试复习讲义 · 分享版
%  由 build_share_handout.py 在 verify_share_handout.py 全绿后生成，请勿手改题面与答案。
%  排版：ctexart + \usepackage[color]{tlnote}（用户模板 A 方案，零手写 preamble）
%  编译：xelatex 两遍（目录 / 书签 / 交叉引用才完整）
% ============================================================
\documentclass[11pt,a4paper]{ctexart}
\usepackage[color]{tlnote}

\title{高等数学 I（A 类）期中复习讲义}
\author{陶理 编}
\noteheader{高等数学 I（A 类）· 期中复习 · 分享版}
\docdate{2026-11-09}

% 用户文档里 @ 不是字母，取 \@title / \@author / \@date 需先做一层别名
\makeatletter
\newcommand{\showtitle}{\@title}
\newcommand{\showauthor}{\@author}
\newcommand{\showdate}{\@date}
\makeatother

\begin{document}

\begin{center}
\vspace*{1.0cm}
{\Huge\bfseries\color{tlblue} 高等数学 I（A 类）}\\[6pt]
{\Huge\bfseries\color{tlblue} 期中考试复习讲义}\\[14pt]
{\Large 系统讲解 $\cdot$ 8 章 $\cdot$ 33 道例题精讲 $\cdot$ 151 道章末习题}\\[8pt]
{\large 分享版 $\cdot$ \showdate}\\[16pt]
\noindent{\color{tlrule}\rule{0.82\linewidth}{0.6pt}}\\[14pt]
\begin{tabular}{@{}ll@{}}
适用范围 & 函数与极限、连续、导数与微分、中值定理、泰勒、作图、数列极限与新压轴 \\
编排依据 & 2008--2025 十七套高数 A 类期中真题的考点频次与分值 \\
章节结构 & 概念与定理 $\to$ 方法卡 $\to$ 例题精讲 $\to$ 易错点 $\to$ 章末习题 $\to$ 答案速查 \\
答案复核 & 例题由 sympy / mpmath 独立复核；151 题答案由四册既有复核脚本锁定 \\
使用建议 & 先抄方法卡，例题先自己做再看解答，章末习题闭卷做，卡住回看方法卡 \\
\end{tabular}
\end{center}
'''


def main():
    gate_verify()
    b, questions, answers, ids = load_questions()
    chunks = split_notes()

    out = [PREAMBLE, '\n', chunks['FRONT'], '\n']
    counts = [0] * 8
    special = []
    for ch, title in enumerate(CH_TITLES):
        group = [i for i in ids if b.chapter_of(i) == ch]
        counts[ch] = len(group)
        # 章间只用垂直胶。个别 section 锚点落在前页导致的书签 off-by-one，
        # 交给 align_share_bookmarks.py 以 .toc 为真值统一修正（比强制预留空间更省纸）。
        out.append('\n\\bigskip\n')
        out.append('\\section{%s}\n' % title)
        out.append(chunks['CH%d' % (ch + 1)] + '\n')
        out.append('\\subsection{章末习题（%d 题 \\texorpdfstring{$\\cdot$}{·} 按题号排列）}\n'
                   % len(group))
        for qid in group:
            out.append('\\q{%s}{%s}\n' % (qid, b.esc(questions[qid])))
        out.append('\\subsection{答案速查（只给结论，卡住回看方法卡与例题）}\n')
        # 标准三段式表头；且最后一行不带 \\，否则 longtable 会在表尾
        # 再吐一页只有表头的空白页（实测 p28 出现过）
        out.append('{\\footnotesize\n\\begin{longtable}{@{}p{1.6cm}p{13.0cm}@{}}\n'
                   '\\toprule 题号 & 答案 \\\\ \\midrule\n'
                   '\\endfirsthead\n'
                   '\\toprule 题号 & 答案（续） \\\\ \\midrule\n'
                   '\\endhead\n'
                   '\\midrule\n'
                   '\\endfoot\n'
                   '\\bottomrule\n'
                   '\\endlastfoot\n')
        for idx, qid in enumerate(group):
            ans = b.esc(answers[qid])
            if '&' in ans or r'\\' in ans:
                special.append(qid)
                ans = ans.replace('&', r'\&').replace(r'\\', '; ')
            tail = ' \\\\\n' if idx < len(group) - 1 else '\n'
            out.append('%s & %s%s' % (qid, ans, tail))
        out.append('\\end{longtable}}\n')

    out.append('\n' + chunks['BACK'] + '\n')
    out.append('\n\\end{document}\n')
    tex = ''.join(out)

    if sum(counts) != len(ids) or min(counts) == 0:
        sys.exit('章末习题分章失败：%s' % counts)
    check_envs(tex)

    io.open(OUT_TEX, 'w', encoding='utf-8').write(tex)
    print('\n章末习题分章：%s  合计 %d 题' % (dict(zip(range(1, 9), counts)), sum(counts)))
    if special:
        print('答案单元格特殊字符已转义：', special)
    print('写出：%s  (%.1f KB)' % (OUT_TEX, len(tex.encode('utf-8')) / 1024))
    print('全部门禁通过，可以 xelatex 两遍编译。')


if __name__ == '__main__':
    main()
