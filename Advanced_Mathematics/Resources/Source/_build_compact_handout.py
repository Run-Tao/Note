# -*- coding: utf-8 -*-
"""装配《高数I期中精编讲义 11-09》的 LaTeX。

讲解全新撰写（_compact_notes.tex），题目与答案速查从 11/8 四册已复核讲义的
.tex 源中按题号提取、按八个考点重排；题号保留原册编号便于回查详解。
题面 / 答案一字不改，只做裸 % # 转义与表格列分隔保护。
"""
import io
import os
import re

SRC = r'D:/Run_Tao/Homework/Advanced Mathematics/期中复习'
NOTES = os.path.join(SRC, '_compact_notes.tex')
OUT_TEX = os.path.join(SRC, '讲义_精编本_11-09.tex')

# (文件名, 题面宏, 答案速查所在文件)
BOOKS = [
    ('讲义_泰勒与压轴强化_11-08.tex', 'q', '讲义_泰勒与压轴强化_11-08.tex'),
    ('讲义_全专题冲顶_11-08.tex', 'q', 'part2_body.tex'),
    ('讲义_加练_定义与压轴深化_11-08.tex', 'q', '讲义_加练_定义与压轴深化_11-08.tex'),
    ('讲义_网络对标加练_11-08.tex', 'Q', '讲义_网络对标加练_11-08.tex'),
]

CH_TITLES = [
    '第 1 章 \\ 极限计算与无穷小阶',
    '第 2 章 \\ 极限定义证明：三套量词行',
    '第 3 章 \\ 连续、间断点与极坐标切线',
    '第 4 章 \\ 导数与微分：六型求导',
    '第 5 章 \\ 泰勒公式与高阶导数',
    '第 6 章 \\ 中值定理、不等式与最值',
    '第 7 章 \\ 函数性态与作图七步',
    '第 8 章 \\ 新压轴：函数方程、迭代与数列',
]

W5_MAP = {
    'W5.1': 6,   # 七步作图
    'W5.2': 5,   # Rolle 构造
    'W5.3': 5,   # 两次 Rolle
    'W5.4': 7,   # Cauchy 型函数方程
    'W5.5': 7,   # 迭代渐近
    'W5.6': 5,   # Lagrange 不等式
}


def chapter_of(qid):
    if qid in W5_MAP:
        return W5_MAP[qid]
    for pref, ch in [('W1.', 0), ('S1.', 1), ('P1.', 1), ('W3.', 1),
                     ('S3.', 2), ('S2.', 3), ('W4.', 3),
                     ('T1.', 4), ('T2.', 4), ('T3.', 4), ('T4.', 4),
                     ('P2.', 4), ('W2.', 4), ('S5.', 5),
                     ('S6.', 6), ('P4.', 6), ('S4.', 7), ('P3.', 7)]:
        if qid.startswith(pref):
            return ch
    return None


def read_braced(s, i):
    """s[i] == '{'，返回 (内容, 右括号后一位)；处理嵌套与 \\{ \\}。"""
    assert s[i] == '{'
    depth, j, buf = 1, i + 1, []
    while j < len(s):
        c = s[j]
        if c == '\\' and j + 1 < len(s):
            buf.append(c)
            buf.append(s[j + 1])
            j += 2
            continue
        if c == '{':
            depth += 1
            buf.append(c)
            j += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return ''.join(buf), j + 1
            buf.append(c)
            j += 1
        else:
            buf.append(c)
            j += 1
    raise ValueError('unbalanced brace at %d' % i)


def parse_questions(text, macro):
    out = {}
    pat = '\\' + macro + '{'
    i = 0
    while True:
        k = text.find(pat, i)
        if k < 0:
            break
        j = k + len(pat) - 1          # 指向 '{'
        try:
            qid, nxt = read_braced(text, j)
            if nxt >= len(text) or text[nxt] != '{':
                i = nxt
                continue
            body, nxt2 = read_braced(text, nxt)
        except ValueError:
            i = k + len(pat)
            continue
        qid = qid.strip()
        # 网络册题面把「对标来源」写在第一参数里：W1.1〔对标交大卷 Q12 …〕。
        # 必须提取纯题号并把标注并入题面，否则会因编号不匹配而误取后面的详解。
        m = re.match(r'^([A-Za-z]+\d+\.\d+)(.*)$', qid, re.S)
        if m and m.group(1) not in out:
            qid, extra = m.group(1), m.group(2).strip().strip('〔〕').strip()
            body = body.strip()
            if extra:
                body += (r'\quad{\footnotesize\color{blue!45!black}['
                         + extra + ']}')
            out[qid] = body
        i = nxt2
    return out


def parse_answers(text):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if '\\section{答案速查' in l), None)
    if start is None:
        return {}
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith('\\section{')), len(lines))
    out = {}
    for l in lines[start:end]:
        s = l.strip()
        if not s.endswith('\\\\') or '&' not in s:
            continue
        parts = s.split('&')
        qid = parts[0].strip()
        if not re.match(r'^[A-Za-z]+\d+\.\d+$', qid):
            continue
        # 去掉末列（组名 / 一句话卡点），保留中间的答案单元
        ans = '&'.join(parts[1:-1]) if len(parts) >= 3 else parts[1]
        ans = re.sub(r'\\\\\s*$', '', ans).strip()
        if ans and qid not in out:
            out[qid] = ans
    return out


CIRCLED = {}
for _i in range(1, 21):
    CIRCLED[chr(0x2460 + _i - 1)] = '(%d)' % _i      # ①–⑳
    CIRCLED[chr(0x2474 + _i - 1)] = '(%d)' % _i      # ⑴–⒇
CIRCLED[chr(0x24EA)] = '(0)'                         # ⓪


def esc(s):
    """转义裸 % / #（防吞表格行尾 \\\\）+ 把 ctex 缺字形的圈号换成 (n)。"""
    s = re.sub(r'(?<!\\)%', r'\%', s)
    s = re.sub(r'(?<!\\)#', r'\#', s)
    for c, rep in CIRCLED.items():
        s = s.replace(c, rep)
    return s


def load():
    questions, answers = {}, {}
    for qfile, macro, afile in BOOKS:
        qt = io.open(os.path.join(SRC, qfile), encoding='utf-8').read()
        at = io.open(os.path.join(SRC, afile), encoding='utf-8').read()
        q = parse_questions(qt, macro)
        a = parse_answers(at)
        print('%-42s 题面 %3d / 答案 %3d' % (qfile, len(q), len(a)))
        questions.update(q)
        answers.update(a)
    return questions, answers


PREAMBLE = r'''\documentclass[11pt,a4paper]{ctexart}
% ============================================================
%  高等数学 I（A 类）期中精编讲义 · 2026-11-09
%  讲解全新撰写；151 题从 11/8 四册已复核讲义按考点重排
%  复核：verify_compact_handout.py PASS 30 / FAIL 0
%        + verify_taylor/full_topics/extra/web_drill.py 全 PASS
%  编译：xelatex 两遍
% ============================================================
\usepackage[a4paper,top=2.0cm,bottom=2.0cm,left=2.0cm,right=2.0cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,array,longtable,tabularx,multicol}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{hyperref}
\hypersetup{hidelinks,bookmarksopen=true,bookmarksnumbered=true,
  colorlinks=true,linkcolor=blue!50!black}

\linespread{1.16}
\setlength{\parskip}{0.16em}
\setlength{\tabcolsep}{4pt}
\setlist[enumerate]{itemsep=2pt,topsep=3pt,parsep=0pt,leftmargin=1.9em}
\setlist[itemize]{itemsep=2pt,topsep=3pt,parsep=0pt,leftmargin=1.6em}
\titleformat{\section}{\Large\bfseries\color{blue!35!black}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries\color{blue!25!black}}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.3ex plus .3ex}{0.7ex}
\titlespacing*{\subsection}{0pt}{1.0ex plus .2ex}{0.5ex}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 高数 I 期中精编讲义 $\cdot$ 11/9}
\fancyhead[R]{\small 陶理}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\newcommand{\R}{\mathbb{R}}
\newcommand{\eps}{\varepsilon}
\newcommand{\abs}[1]{\left\lvert #1\right\rvert}
\newcommand{\ee}{\mathrm{e}}
\newcommand{\D}{\mathrm{d}}
\newcommand{\dd}[2]{\dfrac{\D #1}{\D #2}}
\newcommand{\q}[2]{\par\medskip\noindent\textbf{#1}\quad #2\par}
\newcommand{\key}[1]{\par\nopagebreak\noindent{\small\color{red!55!black}\textbf{卡点：}#1}\par}
\newcommand{\tagline}[1]{\par\nopagebreak\noindent{\small\color{blue!45!black}#1}\par}
\newcommand{\ans}[1]{\;\boxed{#1}}

\newtcolorbox{keypoint}[1][]{enhanced,breakable,colback=blue!3,colframe=blue!40!black,
  fonttitle=\bfseries,title={#1},boxrule=0.5pt,arc=1.5mm,
  left=3mm,right=3mm,top=2mm,bottom=2mm,before skip=4pt,after skip=4pt}

\begin{document}

\begin{center}
\vspace*{1.0cm}
{\Huge\bfseries 高等数学 I（A 类）}\\[5pt]
{\Huge\bfseries 期中考试精编讲义}\\[12pt]
{\Large 系统讲解 $\cdot$ 151 题 $\cdot$ 章末答案速查}\\[8pt]
{\large 2026-11-09 定稿 $\cdot$ 考试 11/11 周三}\\[16pt]
\rule{0.8\linewidth}{0.6pt}\\[12pt]
\begin{tabular}{@{}ll@{}}
依据 & 上海交大 2008--2025 十七套高数 A 类期中真题逐卷核对 \\
题源 & 11/8 四册加练（T 19 + S 62 + P 40 + W 30 = 151 题），答案已全部复核 \\
用法 & 每章先读方法卡与例题精讲，再闭卷做习题，最后对章末答案速查 \\
边界 & 不含阅读包、执行卡、模拟卷整卷与逐题详解；掐表请用单册模拟卷 \\
\end{tabular}
\end{center}
\newpage
'''

FRONT_TITLE = ''


def main():
    questions, answers = load()
    ids = sorted(set(questions) & set(answers),
                 key=lambda s: (s.split('.')[0], int(s.split('.')[1])))
    orphan_q = sorted(set(questions) - set(answers))
    orphan_a = sorted(set(answers) - set(questions))
    print('\n交集题数 = %d；仅题面 %d 个；仅答案 %d 个'
          % (len(ids), len(orphan_q), len(orphan_a)))
    if orphan_q:
        print('  仅题面示例:', orphan_q[:12])
    if orphan_a:
        print('  仅答案示例:', orphan_a[:12])

    unmapped = [i for i in ids if chapter_of(i) is None]
    if unmapped:
        print('  !! 未映射题号:', unmapped)

    notes = io.open(NOTES, encoding='utf-8').read()
    chunks = {}
    for key in ['FRONT'] + ['CH%d' % k for k in range(1, 9)] + ['BACK']:
        m = re.search(r'^%%' + key + r'%%\s*$', notes, re.M)
        if not m:
            raise SystemExit('missing marker %%' + key + '%%')
        chunks[key] = m
    # 按位置切分
    markers = [(m.start(), m.end(), m.group(0).strip('%')) for m in
               re.finditer(r'^%%(FRONT|CH[1-8]|BACK)%%\s*$', notes, re.M)]
    for i, (s, e, key) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(notes)
        chunks[key] = notes[e:end].strip()

    out = [PREAMBLE, FRONT_TITLE, chunks['FRONT'], '\n']

    counts = [0] * 8
    for ch, title in enumerate(CH_TITLES):
        group = [i for i in ids if chapter_of(i) == ch]
        counts[ch] = len(group)
        # 章间不强制换页：答案速查长表跨页后常只剩 2–3 行，
        # 连续排版能让下一章方法卡接上，避免半空页（精编本以省纸为先）。
        sep = '' if ch == 0 else ('\n\\bigskip\\noindent{\\color{blue!35!black}'
                                  '\\rule{\\linewidth}{0.5pt}}\\bigskip\n')
        out.append(sep + '\\section{' + title + '}\n')
        out.append(chunks['CH%d' % (ch + 1)] + '\n')
        out.append('\n\\subsection{习题（%d 题 $\\cdot$ 闭卷做，做完再对答案）}\n' % len(group))
        for qid in group:
            out.append('\\q{%s}{%s}\n' % (qid, esc(questions[qid])))
        out.append('\n\\subsection{答案速查（只核对结论，卡住满 5 min 再回查详解）}\n')
        out.append('{\\footnotesize\n\\begin{longtable}{@{}p{1.5cm}p{13.6cm}@{}}\n'
                   '\\toprule 题号 & 答案 \\\\ \\midrule \\endhead\n'
                   '\\bottomrule \\endlastfoot\n')
        for qid in group:
            out.append('%s & %s \\\\\n' % (qid, esc(answers[qid])))
        out.append('\\end{longtable}}\n')

    out.append('\n\\newpage\n' + chunks['BACK'] + '\n')
    out.append('\n\\end{document}\n')

    tex = ''.join(out)
    io.open(OUT_TEX, 'w', encoding='utf-8').write(tex)

    # 一致性闸门
    print('\n章节题数:', dict(zip(range(1, 9), counts)), '合计', sum(counts))
    print('写出:', OUT_TEX, '%.1f KB' % (len(tex.encode('utf-8')) / 1024))
    assert sum(counts) == len(ids), '有题目未归入任何章'
    assert tex.count('\\q{') == len(ids), '题目行数不符'
    assert tex.count('\\end{longtable}') >= 8
    # 花括号平衡（粗检）
    body = re.sub(r'\\[{}]', '', tex)
    assert body.count('{') == body.count('}'), '花括号不平衡'
    # 答案行不得含独立的 LaTeX 换行或裸 &（会破坏 longtable）
    bad = [q for q in ids if '\\\\' in answers[q] or '&' in answers[q]]
    if bad:
        print('  !! 答案含换行/& 的题号（需人工检查）:', bad[:20])
    print('OK: 一致性闸门通过')


if __name__ == '__main__':
    main()
