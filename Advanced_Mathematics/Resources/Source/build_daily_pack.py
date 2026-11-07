# -*- coding: utf-8 -*-
"""
每日阅读包生成器 —— 把「今天要看的全部材料」合成一份 PDF。

用法：
    PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe build_daily_pack.py 11-07
    PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe build_daily_pack.py --list

产出的 PDF 分两部分：
    A 部分（新编 LaTeX）：封面 / 阅读路线图 / 可书写默写页 / 记录表 / 核对答案速查
    B 部分（原件合订）：按 DAY_CONFIG 指定的文件与页码，把讲义、练习册、真题的
                        对应页原样 insert_pdf 进来，加书签与页脚来源标注

新增一天：在 DAY_CONFIG 中加一个键（复制 11-07 的结构，改标题、时间表、SECTIONS）。
A 部分正文由 ``make_body()`` 按日期分支生成；B 部分由 SECTIONS 配置驱动，无需改代码。
"""
import io
import os
import re
import shutil
import subprocess
import sys

import pymupdf

ROOT = r'D:/Run_Tao'
DOWNLOADS = os.path.join(ROOT, 'Note/Advanced_Mathematics/Resources/Downloads')
SOURCE = os.path.join(ROOT, 'Note/Advanced_Mathematics/Resources/Source')
EXAMS = os.path.join(ROOT, 'Homework/_pdf_text/exam_2008_2024/高数A2008-2024第1学期期中')
WORK = os.path.join(ROOT, 'Homework/Advanced Mathematics/期中复习/_daily_pack')


def D(name):
    return os.path.join(DOWNLOADS, name)


def E(name):
    return os.path.join(EXAMS, name)


# ---------------------------------------------------------------------------
# B 部分：今天要看的材料原件（页码为 PDF 阅读器页码，1-based）
#   key     -> 路线图与书签编号
#   title   -> 书签 / 路线图显示名
#   items   -> [(文件绝对路径, [页码...]), ...]，允许一段跨多个文件
#   action  -> 路线图里「翻到这一段做什么」
#   minutes -> 建议分钟数
# ---------------------------------------------------------------------------
DAY_CONFIG = {
    '11-07': {
        'date': '2026-11-07',
        'weekday': '周六',
        'title': '补齐 11/6（D-4）＋ 总体复习',
        'subtitle': '中值定理保底 · 加练 50 题 · 高难 22 题冲顶 · 九大专题快扫',
        'plan_file': 'Note/Advanced_Mathematics/Notes/11-7_补齐D-4与总体复习.md',
        'sections': [
            dict(key='B1', title='资料分工总规则与九大专题表',
                 items=[(D('讲义_该翻哪一本.pdf'), [2, 3])],
                 action='开局读总规则；P1 九宫格按此表逐格走', minutes=5),
            dict(key='B2', title='中值定理 §3 万能构造表与倒推',
                 items=[(D('微分中值定理专题讲义.pdf'), [9, 10, 11, 12])],
                 action='P0③ 遮住解答，重推 6 个辅助函数 + 三道倒推示范', minutes=35),
            dict(key='B3', title='中值定理 §4 母题（15 类）',
                 items=[(D('微分中值定理专题讲义.pdf'), [13, 14, 15])],
                 action=r"P0④ 挑 3 类：$f'+\lambda f=0$ / 含 $f''$ 两次 Rolle / 结论含 $\ln$ 的 Cauchy", minutes=25),
            dict(key='B4', title='中值定理 §5 真题压轴',
                 items=[(D('微分中值定理专题讲义.pdf'), [16, 17, 18])],
                 action='P0⑤ 做 2010 级或 2017 级 七(18)，按答题模板五步写全', minutes=20),
            dict(key='B5', title='2009-10 真题卷 Q18',
                 items=[(E('高数A类_第1学期_期中_2009-2010学年.pdf'), [1, 2])],
                 action=r"P0⑤ 做 Q18（$f(c)f'(1-c)$ 型构造），条件与端点验证写全", minutes=15),
            # 2026-11-07 15:23 用户指示「最后数列极限那部分可以减少」：
            # 原 B6--B8 共 7 页（讲义 p1--4 + 模拟卷 C 题面/答案）压缩为 2 页最小保温，
            # 腾出的段位放新编《讲义_今日配套加练_11-07》9 页（题面 5 + 详解 4）。
            dict(key='B6', title='数列极限 / 新压轴（已压缩为最小保温）',
                 items=[(D('讲义_数列极限与压轴新题型.pdf'), [1, 4])],
                 action='只闭卷复述 P1--P3 三问思路；动笔只做加练讲义 S 组 3 题（B7）。原 §3 十例 / V1--V5 / 模拟卷 C Q18 今天不做', minutes=8),
            dict(key='B7', title='今日讲义配套加练 · 题面（50 题：M/L/J/T/G/E/S + Q 卷）',
                 items=[(D('讲义_今日配套加练_11-07.pdf'), [1, 2, 3, 4, 5])],
                 action='今天的动笔主战场：P0 后插 M 组 40 min；P1 各格「1 道代表题」直接改用本册对应组；Q 卷 30 min 掐表', minutes=70),
            dict(key='B8', title='配套加练 · 答案速查 / 逐题详解 / 错因回炉 / 验收',
                 items=[(D('讲义_今日配套加练_11-07.pdf'), [6, 7, 8, 9])],
                 action='做完再看；错题按第 12 节回炉索引定位，错的「第一步」写进记录表 C', minutes=30),
            dict(key='B9', title='第 11 题 ε-δ / ε-X / M 型三模板',
                 items=[(D('讲义_极限定义证明.pdf'), [4, 5, 6, 14])],
                 action=r'P1① 三形式框架各写一遍，重点 M 型 $\delta=\min\{1/2,1/(2G)\}$', minutes=15),
            dict(key='B10', title='函数极限决策树 / 间断点例题 / §7 限时卷',
                 items=[(D('讲义_函数极限与间断点.pdf'), [3, 4, 7, 8, 11, 12, 13, 14])],
                 action='P1②③ 挑 1 道有理化 + 1 道分界点；§7 得分补记；§9 全表打勾', minutes=27),
            dict(key='B11', title='泰勒练习册：决策卡 / B 组 / L 组 / 答案',
                 items=[(D('讲义_练习_函数极限计算与泰勒.pdf'), [3, 5, 6, 7, 8])],
                 action=r'P1④ B 组挑 1 题；P2③ L 组 30 min 掐表，目标 $\geq 14/20$', minutes=42),
            dict(key='B12', title='导数计算：六型例题 / T1–T8 / 速查卡',
                 items=[(D('讲义_导数计算专题.pdf'), [2, 3, 4, 5])],
                 action='P1⑤ 挑 1 道分段可导或参数二阶导；§7 速查卡', minutes=15),
            dict(key='B13', title='函数性态作图：例题 / L1–L2 / 四套 Q17 速查',
                 items=[(D('讲义_函数性态与作图.pdf'), [2, 3, 4])],
                 action='P1⑥ 挑 1 道例题只列七步表；P2④ 两道 Q17 闭卷七步', minutes=35),
            # 2026-11-07 16:30 用户指示「增加有难度的内容，匹配前几年真题卷的难度，
            # 不要保守型思考」：新编《讲义_高难冲顶_11-07》15 页（题面 p1--4 含难度对标表，
            # 答案 p5，详解 p6--14，验收 p15），拆成 B14 题面 / B15 答案与详解两段。
            dict(key='B14', title='高难冲顶 22 题 · 题面（对标 2008–2025 真题）',
                 items=[(D('讲义_高难冲顶_11-07.pdf'), [1, 2, 3, 4])],
                 action='P2+ 冲高块：整册 110 min；余力不足只做 H2.9 / H2.10 / H3.11 / H3.13 / H4.16 / H5.20 六道', minutes=70),
            dict(key='B15', title='高难冲顶 · 答案速查与逐题详解',
                 items=[(D('讲义_高难冲顶_11-07.pdf'), [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])],
                 action='做完一组才翻；卡 5 min 只看「卡点」一行，合上重写', minutes=40),
        ],
    },
}


# ---------------------------------------------------------------------------
# LaTeX 片段
# ---------------------------------------------------------------------------
def blanklines(n, gap='0.52cm'):
    one = r'\noindent\rule{\linewidth}{0.25pt}\par\vspace{' + gap + '}'
    # 第一条前先结束题头段落，避免横线接在文字末尾
    return ('\n'.join([r'\par\smallskip' + one] + [one] * (n - 1)))


# ctex 默认中文字体缺少这些 Unicode 数学/圈号字形（渲染为 U+FFFF 空白），
# 统一在写 .tex 前换成 LaTeX 表达式或 ASCII。
UNICODE_FIX = [
    ('①', '(1)'), ('②', '(2)'), ('③', '(3)'), ('④', '(4)'),
    ('⑤', '(5)'), ('⑥', '(6)'), ('⑦', '(7)'), ('⑧', '(8)'), ('⑨', '(9)'),
    ('★', '*'), ('≥', r'$\geq$'), ('≤', r'$\leq$'),
    ('′', "'"), ('″', "''"), ('λ', r'$\lambda$'), ('δ', r'$\delta$'),
]


def fix_unicode(tex):
    for a, b in UNICODE_FIX:
        tex = tex.replace(a, b)
    # A 部分用无编号 section（书签已手写 A1–A10 标签，避免两套编号打架）
    return tex.replace(r'\section{', r'\section*{')


PREAMBLE = r'''% !TeX program = xelatex
% 每日阅读包 —— A 部分（由 Homework/_review/tools/build_daily_pack.py 生成）
\documentclass[10pt,a4paper]{ctexart}
\usepackage[a4paper,top=1.9cm,bottom=1.9cm,left=1.9cm,right=1.9cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,array,longtable,tabularx,multicol}
\usepackage{enumitem}
\usepackage[most]{tcolorbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{hyperref}
\hypersetup{hidelinks,colorlinks=true,linkcolor=blue!50!black}
\linespread{1.20}
\setlength{\parskip}{0.18em}
\renewcommand{\arraystretch}{1.30}
\setlist[enumerate]{itemsep=1pt,topsep=3pt,parsep=0pt,leftmargin=1.7em}
\setlist[itemize]{itemsep=1pt,topsep=3pt,parsep=0pt,leftmargin=1.5em}
\titleformat{\section}{\large\bfseries\color{blue!35!black}}{\thesection}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{blue!25!black}}{\thesubsection}{0.45em}{}
\titlespacing*{\section}{0pt}{1.1ex plus .2ex}{0.6ex}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 每日阅读包 · @@DATE@@}
\fancyhead[R]{\small 陶理}
\fancyfoot[C]{\small \thepage}
\renewcommand{\headrulewidth}{0.4pt}
\newcommand{\R}{\mathbb{R}}
\newcommand{\eps}{\varepsilon}
\newcommand{\abs}[1]{\left\lvert #1\right\rvert}
\newcommand{\ee}{\mathrm{e}}
\newcommand{\D}{\mathrm{d}}
\newcommand{\blank}{\underline{\hspace{1.5cm}}}
\newtcolorbox{keypoint}[1][]{enhanced,breakable,colback=blue!3,colframe=blue!40!black,
  fonttitle=\bfseries,title={#1},boxrule=0.5pt,arc=1.2mm,
  left=2.5mm,right=2.5mm,top=1.5mm,bottom=1.5mm,before skip=3pt,after skip=4pt}
\newtcolorbox{warnbox}[1][]{enhanced,breakable,colback=red!2,colframe=red!45!black,
  fonttitle=\bfseries,title={#1},boxrule=0.5pt,arc=1.2mm,
  left=2.5mm,right=2.5mm,top=1.5mm,bottom=1.5mm,before skip=3pt,after skip=4pt}
\begin{document}
'''


def make_body(cfg, pageno):
    """pageno: dict key -> 该 B 段在整份 PDF 中的起始页（占位值会被迭代修正）"""
    ph = lambda k: str(pageno.get(k, 0))
    rows = []
    for s in cfg['sections']:
        srcs = '、'.join(os.path.basename(f) + ' p' + ','.join(str(x) for x in ps)
                        for f, ps in s['items'])
        rows.append('%s & %s & %s & %s & %s \\\\' %
                    (s['key'], ph(s['key']), srcs.replace('_', r'\_'),
                     s['action'], s['minutes']))
    route = '\n'.join(rows)

    P = []
    # ---------------- A1 封面 + 总表 ----------------
    P.append(r'''
\begin{center}
{\Huge\bfseries 每日阅读包}\\[0.25em]
{\LARGE @@DATE@@（@@WEEKDAY@@）}\\[0.45em]
{\Large\bfseries @@TITLE@@}\\[0.3em]
{\normalsize @@SUBTITLE@@}
\end{center}
\vspace{0.4em}
\begin{keypoint}[今天的一句话]
\textbf{不做整卷}（2024-25 真题卷之前已做过）。今天只有一组「新内容」——补齐 11/6 的
中值定理；其余全部是\textbf{闭卷默写 + 动笔做题 + 订正}。
\textbf{15:23 改订}：数列极限 / 新压轴已按你的要求压缩为最小保温（B6 只 2 页、S 组只 3 题），
省下的时间给 \textbf{B7--B8《今日讲义配套加练》}（50 题题面 + 逐题详解）。
\textbf{16:30 追加}：新编 \textbf{B14--B15《高难冲顶 22 题》}——对标 2008--2025 真题的中高难度与压轴，
题面与详解分开；\textbf{这是冲高块，不替代 P0 的 11/6 补齐}。
只选一样时永远选「动手做题」，\textbf{不通读任何一本笔记}。
\end{keypoint}
\section{五级优先级（时间不够从下往上砍）}
\begin{tabularx}{\linewidth}{c|c|p{8.1cm}|c}
\toprule
级 & 时长 & 内容 & 必做 \\
\midrule
P0 & 150 min & 补齐 11/6：中值定理保底 + 加练讲义 M 组 8 题 + 1 页证明模板卡 & \textbf{必做} \\
P1 & 115 min & 总体复习：九大专题快扫，每格「闭卷默写 + 1 道代表题」直接取 B7 对应组 & \textbf{必做} \\
P2 & 90 min & 散项收口：Q 卷 30 min 掐表 + 订正 / §7 得分补记 / ★加练 / L 组 / 两道 Q17 & 按余力 \\
\textbf{P2+} & 100 min & \textbf{冲高块：高难 22 题（B14--B15）}。整册 110 min；余力不足只做六道优先题（约 50 min） & 按余力 \\
P3 & 15 min & 睡前只回忆 5 张手抄卡 & 固定 \\
\bottomrule
\end{tabularx}
\section{这份 PDF 怎么用}
\begin{enumerate}
\item \textbf{第 2 页}是阅读路线图：B1--B15 对应后面合订的材料原件，按段号顺序翻，不要跳。
\item \textbf{第 3--6 页}是\textbf{可书写页}：闭卷默写区、记录表、验收清单。建议打印，或平板手写。
\item \textbf{核对答案}放在默写页之后：\textbf{做完再看}，否则闭卷无效。
\item \textbf{B7--B8 是新编配套加练讲义}：B7 是 50 道题面（M/L/J/T/G/E/S 七组 + Q 卷），
      B8 是答案速查、逐题详解、错因回炉索引与验收清单。\textbf{按题组与 B2--B5、B9--B13 交叉使用}，
      不必一次做完。
\item \textbf{B14--B15 是 16:30 追加的高难册}：B14 是 22 道题面 + 难度对标表，B15 是答案与逐题详解。\textbf{每题先掐 8--12 min；卡 5 min 只看「卡点」一行，合上重写}。时间不足按 B14 里列的六道优先题做（约 50 min），\textbf{不要为了做完而跳过 P0}。
\item \textbf{第 @@B1@@ 页起}是今天要看的材料原件，每页左下角标了来源与原页码。
      PDF 书签（大纲）可按 B1--B15 直接跳转。
\item 今天\textbf{不动 11/8 起的排期}；Level 4--6（积分 / 多元 / 级数与 ODE）不在本次期中范围。
\end{enumerate}
\begin{warnbox}[产出（明天 11/8 的输入）]
① 2024-25 旧卷得分与错题号补记；② 1 页证明模板卡；③ 九宫格快扫记录；④ 3 条最大风险；
⑤ 加练讲义 Q 卷得分（$\geq 14/20$ 达标）；⑥ \textbf{高难 22 题完成题数}（能独立写出第一行的题数，目标 $\geq 16$）。
\textbf{旧卷得分全库无记录，记不清就写「未记录」，不要估算分数。}
\end{warnbox}
''')

    # ---------------- A2 阅读路线图 ----------------
    P.append(r'''
\section{阅读路线图（按段号顺序翻）}
{\small 表中「页」是\textbf{本 PDF 的绝对页码}；原件页码写在材料列的 p 后面。}
\small
\setlength{\tabcolsep}{3pt}
\begin{longtable}{c|c|p{4.5cm}|p{6.6cm}|c}
\toprule
段 & 页 & 材料（原件页码） & 翻到这里做什么 & min \\
\midrule
\endhead
''' + route + r'''
\bottomrule
\end{longtable}
\normalsize
\begin{keypoint}[路线图的两条纪律]
① 每一段都是「\textbf{先遮住解答自己做}，再对答案」；看不懂才回笔记查一步，\textbf{不从头读笔记}。
② B7 的题面与 B8 的详解\textbf{必须分开用}：做完一组才翻 B8 对答案。
③ 数列极限（B6、九宫格第 7 格、S 组）已压缩，\textbf{只保温不展开}；中值定理两格在 P1 中与 P0 合并计时，\textbf{不重复做}；整卷一格今天只口述 120 分钟分配表。
④ \textbf{B14 与 B15 必须分开用}：B14 是 22 道题面，B15 是答案与详解。做完一组才翻 B15；卡住只看详解里的「卡点」一行，\textbf{不要顺着读下去}。
\end{keypoint}
''')

    # ---------------- A3 P0 闭卷默写 ----------------
    P.append(r'''
\section{P0 闭卷默写（15 min，写不出的标「?」，不翻书）}
\textbf{① 三定理的条件与结论：} Rolle / Lagrange / Cauchy，各写「闭区间 / 开区间 / 端点」三行。
''' + blanklines(4) + r'''
\textbf{② 六个辅助函数（写目标结论 $\to$ 构造 $F(x)$）：}
''' + blanklines(6) + r'''
\textbf{③ 两个中值点模型、Rolle 答题五步、条件三查、证不等式三条路：}
''' + blanklines(7) + r'''
\begin{keypoint}[验收]
6 个辅助函数闭卷全对；答题模板五步不看资料写出；能说清「Peano 余项不能证严格不等式」。
\end{keypoint}
''')

    # ---------------- A4 九宫格默写 ----------------
    P.append(r'''
\section{P1 九宫格快扫：闭卷默写（每格 2 min）}
{\small 统一动作：2 min 闭卷默写决策卡 $\to$ 8--12 min 只做 1 道代表题 $\to$ 2 min 对答案写一句错因。}\par
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{2.45}
\par
\begin{center}
\begin{tabular}{p{3.0cm}|p{13.4cm}}
\toprule
专题 & 闭卷默写（写关键词，不抄书） \\
\midrule
(1) 第 11 题定义证明 & \hrulefill \\
(2) 函数极限计算 / 三型 & \hrulefill \\
(3) 间断点 & \hrulefill \\
(4) 泰勒 & \hrulefill \\
(5) 导数计算 & \hrulefill \\
(6) 函数性态作图 & \hrulefill \\
(7) 数列极限（\textbf{已压缩}，只默写四法名） & \hrulefill \\
(8) 中值定理（与 P0 合并） & \hrulefill \\
(9) 高频四件 + 时间分配 & \hrulefill \\
\bottomrule
\end{tabular}
\end{center}
\renewcommand{\arraystretch}{1.30}
\normalsize
\vspace{0.4em}
\textbf{动笔题直接取 B7 的 M / L / J / T / G / E 组}：每格做完「闭卷默写」就在对应组挑 1 题，
不再另外找题；做完才翻 B8 对答案。S 组 3 题做完即止，\textbf{不追加}。
\vspace{0.4em}

\textbf{高频必考四件（各 2 min，穿插在快扫里）：}
''' + blanklines(3, '0.52cm') + r'''
\textbf{每格错的「第一步」汇总成 3 条最大风险：}
''' + blanklines(3, '0.52cm'))

    # ---------------- A5 记录表 A/B ----------------
    P.append(r'''
\section{记录表 A：2024-25 旧卷补记（11/8 对比表唯一输入）}
\begin{warnbox}[先补记，再开始 P0]
全库查不到那次整卷的得分与错题号。\textbf{记不清写「未记录」，不要估算分数}。
\end{warnbox}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{2.35}
\begin{tabular}{p{5.2cm}|p{11.1cm}}
\toprule
项 & 填写 \\
\midrule
那次得分 & \hrulefill \\
失分题号 & \hrulefill \\
三个最大失分点 & \hrulefill \\
最耗时三题 & \hrulefill \\
\bottomrule
\end{tabular}
\renewcommand{\arraystretch}{1.30}
\vspace{0.8em}
\section{记录表 B：九宫格快扫（每格一行）}
\small\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{2.10}
\begin{tabularx}{\linewidth}{c|p{3.0cm}|p{3.5cm}|X}
\toprule
格 & 默写了什么 & 做的哪道题 & 错的第一步 / 触发条件 \\
\midrule
''' + '\n'.join('%d & & & \\\\' % i for i in range(1, 10)) + r'''
\bottomrule
\end{tabularx}
\renewcommand{\arraystretch}{1.30}
\normalsize
''')

    # ---------------- A6 记录表 C/D + 验收 ----------------
    P.append(r'''
\section{记录表 C：错因 $\to$ 触发条件（每题一句）}
{\small 格式：「下次看到 \blank 先做 \blank」。只写第一步，不抄整题。}
''' + blanklines(8, '0.58cm') + r'''
\section{记录表 D：3 条最大风险 + 明天要防的复发点}
''' + blanklines(5, '0.58cm') + r'''
\section{今日验收清单}
\begin{multicols}{2}
\begin{itemize}
\item[$\square$] 旧卷得分 / 错题已补记（或明确写「未记录」）
\item[$\square$] 6 个辅助函数闭卷全对；Rolle 五步不看资料写出
\item[$\square$] 2009-10 Q18 + 一道 2010/2017 级七(18) 按模板写全
\item[$\square$] 加练讲义 M / L / J / T 四组各至少完成一半，S 组 3 题做完即止
\item[$\square$] Q 卷 30 分钟掐表，$\geq 14/20$ 达标
\item[$\square$] 错的「第一步」写进记录表 C
\item[$\square$] 九大专题每格：闭卷默写 + 1 道代表题（B7 组）
\item[$\square$] \textbf{高难 22 题}：六道优先题（H2.9 / H2.10 / H3.11 / H3.13 / H4.16 / H5.20）至少做完 4 道
\item[$\square$] 压轴三道（H3.11 / H3.13 / H5.22）能不看提示写出第一行
\item[$\square$] §7 得分补记、★加练、L 组复测至少各完成一半
\item[$\square$] 手抄 5 张卡齐：定义三模板 / 泰勒决策 / 求导检查 / 作图七步 / 证明模板
\item[$\square$] 23:30 前睡
\end{itemize}
\end{multicols}
''')

    # ---------------- A7 答案① 中值定理 + 数列 ----------------
    P.append(r'''
\section{核对答案 ①：中值定理（做完 P0 再看）}
\textbf{三定理}：Fermat（内点取极值 + 可导 $\Rightarrow f'(x_0)=0$）；
Rolle（$[a,b]$ 连续、$(a,b)$ 可导、$f(a)=f(b)\Rightarrow\exists\xi:f'(\xi)=0$）；
Lagrange（$f(b)-f(a)=f'(\xi)(b-a)$，用于证不等式与估计函数差）；
Cauchy（$\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(\xi)}{g'(\xi)}$，L'Hôpital 由它推出）。
\small\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{p{4.6cm}|p{4.4cm}|X}
\toprule
目标结论 & 构造 $F(x)$ & 对谁用 Rolle \\
\midrule
$f'(\xi)+\lambda f(\xi)=0$ & $\ee^{\lambda x}f(x)$ & 两个零点 /
$F(a)=F(b)$ \\
$\xi f'(\xi)+n f(\xi)=0$ & $x^n f(x)$（$0\notin[a,b]$） & 同上 \\
$\xi f'(\xi)-n f(\xi)=0$ & $f(x)/x^n$ & 同上 \\
$f'(\xi)+g'(\xi)f(\xi)=0$ & $\ee^{g(x)}f(x)$ & 指数因子一般化 \\
$f'(\xi)=k$ & $f(x)-kx$ & 退化为 Rolle \\
$f'(\xi)=2\xi$ 一类 & $f(x)-x^2$ & 右端积分回去 \\
\bottomrule
\end{tabularx}
\normalsize\smallskip
\textbf{倒推口令}：把要证的式子看成「某个乘积的导数为零」；$u'+\frac{u'}u u$ 型想 $\ln\abs u$
或 $\ee^{\lambda x}$；$\xi f'+nf$ 就是 $(x^n f)'$ 除以 $x^{n-1}$。
\textbf{Rolle 五步}：造 $F$ $\to$ 验连续 / 可导 $\to$ 验 $F(a)=F(b)$ $\to$ 写定理得 $\xi$ $\to$
回代化简。
\textbf{条件三查}：闭区间连续？开区间可导？端点值相等 / $g'\ne0$？缺一条结论就可能假。
\textbf{两个中值点}：分区间两段各用一次，或对不同函数各用一次。
\textbf{证不等式三条路}：单调性（移项造 $g$，$g(a)=0$ 且 $g'>0$）、Lagrange 放缩、
Taylor 带 \textbf{Lagrange} 余项（Peano 余项不能给严格不等号）。

\section{核对答案 ①（压缩版）：数列极限只保温到这一行}
四法只记名字与触发条件：\textbf{夹逼}（两端同极限、放缩分母取最值）、
\textbf{单调有界}（先证单调 + 有界再对递推取极限）、
\textbf{定积分定义}（认准 $\frac1n\sum f(\frac kn)$）、\textbf{Stolz}（$b_n\nearrow+\infty$）。
压轴三型只记构造：不动点看 $g=f(x)-x$ 两端符号；等距平移把 $n$ 段函数值相加反证；
迭代收敛用 $\abs{a_{n+1}-L}\le q\abs{a_n-L},\ 0<q<1$。

\begin{keypoint}[15:23 改订：这一段今天不展开]
用户明确「最后数列极限那部分可以减少」：B6 由 7 页压到 2 页（只留讲义 p1 母题与 p4 速查卡），
\textbf{删去} §3 十道例题、V1--V5 限时卷、模拟卷 C 第 18 题换题。
动笔只做 B7 加练讲义 \textbf{S 组 3 题}（定积分定义 / 递推不动点 / 导数定义型），
2024-25 Q18 已随旧卷做过，今天\textbf{连口述都省了}。省下的时间全部给 M / L / J / T / G / E 六组与 Q 卷。
完整四法与 P1--P3 母题仍在 \texttt{讲义\_数列极限与压轴新题型.pdf}，考后复盘再翻。
\end{keypoint}
''')

    # ---------------- A8 答案② 导数 + 作图 ----------------
    P.append(r'''
\section{核对答案 ②：导数计算六型}
\textbf{动笔前三问}：定义域？前提（$x'_t\ne0$ / $F_y\ne0$ / 分段点先验连续）？用哪条公式？
\small\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{c|p{3.4cm}|p{5.2cm}|X}
\toprule
型 & 触发 & 第一动作 & 必写前提 \\
\midrule
D1 & 分段可导求参数 & 先列连续方程 $f(a^-)=f(a^+)$，再列导数方程 $f'_-(a)=f'_+(a)$ & 分界点两侧用定义 \\
D2 & 参数方程二阶导 & 先算 $y'/x'$，再对 $t$ 求导、除以 $x'$（\textbf{不是} $y''/x''$） & $x'(t)\ne0$ \\
D3 & 隐函数二阶导 & $y'=-F_x/F_y$，再对 $x$ 求导并把 $y'$ 代回 & $F_y\ne0$ \\
D4 & Leibniz / 高阶导 & 找「低阶导归零」的因子，只留 $\binom nk$ 中那几项 & 展开式写到 $n$ \\
D5 & 极坐标切线 & 先化 $(x,y)=(r\cos\theta,r\sin\theta)$，斜率 $\frac{r'\sin\theta+r\cos\theta}{r'\cos\theta-r\sin\theta}$ & $\theta$ 范围、$r<0$ 反向 \\
D6 & 微分 $\D y$ & 求导后乘 $\D x$ & 一阶微分形式不变性 \\
\bottomrule
\end{tabularx}
\normalsize\smallskip
\textbf{达布定理}：区间上处处可导 $\Rightarrow f'$ 无第一类间断点（\textbf{只在处处可导时}；「左 $\le$ 右」不是定理）。

\section{核对答案 ②：函数性态作图七步}
\begin{enumerate}
\item 定义域（\textbf{含不可导点}）$\to$ ② 奇偶 / 周期 $\to$ ③ 渐近线（\textbf{两端分别求}）$\to$
      ④ $f'$ 判单调与极值 $\to$ ⑤ $f''$ 判凹凸与拐点 $\to$ ⑥ \textbf{列表} $\to$
      ⑦ 作图（标极值点、拐点、交点、渐近线）。
\end{enumerate}
\textbf{渐近线三查}：垂直＝无定义点看单侧 $\lim f=\infty$；水平＝$\lim_{x\to\pm\infty}f=b$；
斜＝$k=\lim f/x,\ b=\lim(f-kx)$。两端各查一次；二次增长（$f/x\to\infty$）判「无斜」。
\textbf{列表是给分点}：区间 / $f'$ 符号 / 升降 / $f''$ 符号 / 凹凸，五行齐全；
卷面附了 $y',y''$ 就从第 ④ 步开始，别回头重算。不可导点在第 ① 步单列（$f'=\infty$ 是竖直切线，不是驻点）。
''')

    # ---------------- A9 答案③ 极限/间断点/泰勒/定义 ----------------
    P.append(r'''
\section{核对答案 ③：计算题三型（限时卷 9 / 10 / 11）}
\small\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{c|c|X}
\toprule
型 & 答案 & 第一动作（背这句） \\
\midrule
A 根式差 / 加减相消 & $\frac14$ & 首项会相消的差式\textbf{绝不逐项等价} $\to$ 先有理化，把差变成积 \\
B $1^\infty$ 幂指 & $\ee^{1/3}$ & 写 $u^v=\ee^{v\ln u}$，只需把 $u-1$ 展到与 $1/v$ 同阶 \\
C $\lim_n$ 定义的分段函数 & $x=\pm1$ 均跳跃 & 先写三段 $f(x)$，再列可疑点算左右极限 \\
\bottomrule
\end{tabularx}
\normalsize\smallskip
\textbf{$1^\infty$ 捷径} $\lim u^v=\ee^{\lim v(u-1)}$ 成立三条件：$u\to1$；$\lim v(u-1)$ 存在；
$v(u-1)^2\to0$。\textbf{必背零件}：$\sqrt{1+u}-1\sim\frac u2$，$\tan x-\sin x\sim\frac{x^3}{2}$，
$\tan x-x\sim\frac{x^3}{3}$，$1-\cos x\sim\frac{x^2}{2}$。收尾代 $x=10^{-3}$ 扫符号与量级。

\section{核对答案 ③：泰勒决策卡 / 间断点 / 定义三模板}
\textbf{泰勒总纲}：先问「是不是 $0/0$ 或 $\infty/\infty$」「分母最高几阶」，$k$ 阶就展到 $k$ 阶；
\textbf{乘积 / 商里的因子}用等价无穷小，\textbf{和差里要消低阶项}必须泰勒（$\sin x-x$ 不能替换 $\sin x\sim x$）；
条件给极限求某阶导数就写 $f(x)=f(0)+f'(0)x+\frac{f''(0)}2x^2+o(x^2)$ 比系数。
注意 $\sqrt{1-x}=1-\frac x2-\frac{x^2}{8}+\cdots$（$x^2$ 项是负号）。
\vspace{0.2em}

\textbf{间断点判型}：先算可疑点左右极限。左右都存在且相等（但无定义或取值不等）$\to$ 可去；
都存在但不等 $\to$ 跳跃（第一类）；一侧 $\pm\infty$ $\to$ 无穷（第二类）；都不存在 $\to$ 振荡（第二类）。
\textbf{八类必分左右}：$\abs x$、$\ee^{1/x}$、$\arctan(1/x)$、$[x]$、$\lim_n$ 定义的分段函数、分母为零点、
分段点、$\sin(1/x)$。2018-19 官方给 1 个第一类间断点是错的，正确 \textbf{2} 个（$x=0$ 处 $\arctan(x+\frac1x)$ 跳跃）。
\vspace{0.2em}

\textbf{定义三模板}：
A（$\eps$-$\delta$）$\forall\eps>0\,\exists\delta>0:0<\abs{x-x_0}<\delta\Rightarrow\abs{f(x)-A}<\eps$；
B（$\eps$-$X$）$\forall\eps>0\,\exists X>0:x>X\Rightarrow\abs{f(x)-A}<\eps$；
C（无穷大 $M$ 型）$\forall M>0\,\exists\delta>0:0<\abs{x-x_0}<\delta\Rightarrow\abs{f(x)}>M$（\textbf{方向相反}）。
收尾：A 取 $\delta=\min\{\delta_1,\eps/C\}$，B 取 $X=\max\{X_1,C/\eps\}$，
C 取 $\delta=\min\{\delta_1,k/M\}$。
\begin{warnbox}[2024-25 第 11 题答案册放缩方向错]
$\frac1{|x|(1+|x|)}\ge\frac1{|x|}$ 不成立；官方 $\delta=\min\{\frac12,\frac1G\}$ 有反例
（$G=10,\ x=0.09631$ 时 $\abs f\approx9.497<10$）。
订正 \textbf{$\delta=\min\{\frac12,\frac1{2G}\}$}，见 \texttt{极限定义证明\_题型专题.md} §3.7。
\end{warnbox}
''')

    # ---------------- A10 边界与已知陷阱 ----------------
    P.append(r'''
\section{高频四件与考场 120 分钟分配}
\textbf{高频四件}：① 微分 $\D y=f'(x)\D x$（别漏 $\D x$）；② Leibniz 高阶导「代点消项」；
③ 极坐标 $\theta$ 范围（$r<0$ 时点在反向射线）；④ 间断点\textbf{分左右}。

\textbf{考场时间分配}（只口述，不掐表）：选填 30 分 / 25 min；第 11 题 8 分 / 10 min；
计算 12--14 / 25 min；证明 15--16 / 20 min；作图 17 / 20 min；压轴 18 / 15 min；检查 5 min。

\section{四处官方答案错误（对答案时以本库复核结论为准）}
\begin{enumerate}
\item 2015-16 上界；\quad ② 2024-25 第 11 题不等式 / 放缩方向；
\item 2018-19 间断点个数：官方 1，正确 \textbf{2}；
\item 2023-24 Q15(2)：官方 $+\frac{13!}{3240}$，正确 \textbf{$-\frac{13!}{3240}$}（sympy + mpmath 140 位复核）。
\end{enumerate}
2023-24 选 3 = A、选 4 = D 与复核一致。根目录 \texttt{HANDOFF\_PROMPT.md} 是上午快照，含两处已证伪记录，勿当权威。

\section{边界与明天}
\begin{itemize}
\item 今天\textbf{不做任何整卷}（2024-25 / 2023-24 / 模拟卷 A/B/C 都不掐表）；模拟卷 C 第 18 题今天也不做，留作 11/9 缓冲日备用。
\item \textbf{数列极限已按 15:23 指示压缩}：B6 只 2 页、S 组只 3 题、九宫格第 7 格只默写四法名；\textbf{不追加}夹逼/Stolz 展开与 P1--P3 母题。
\item \textbf{高难 22 题（B14--B15）是冲高块，不是保底块}：\textbf{不替代} P0 的 11/6 补齐，也不替代 B7--B8 的保底 50 题。P0 没做完就先回 P0；做完 P0 且有余力才开 B14。单题 15 min 写不出第一行就标记跳过。
\item \textbf{不动 11/8 起日期}。若 2023-24 卷也已做过，明天开新对话时说明，再改 11/8 为专项回炉日。
\item Level 4--6（积分 / 多元 / 级数与 ODE）是长期概念底座，\textbf{不在本次期中范围}。
\item 2024-25 原卷是扫描件、文本层全空；若需重看某题请用 PDF 阅读器看图。
\item 数学公式只认 \texttt{.tex} 源或渲染 PNG，\textbf{不要}用 \texttt{pdftotext} / \texttt{get\_text()} 核对公式。
\end{itemize}

\vfill
\begin{center}
{\small\color{gray} 生成：\texttt{Homework/\_review/tools/build\_daily\_pack.py} ·
依据 \texttt{11-7\_补齐D-4与总体复习.md} ·
原件合订见 B1--B15（第 @@B1@@ 页起）}
\end{center}
''')

    body = '\n\\clearpage\n'.join(P) + '\n\\end{document}\n'
    for k, v in [('@@DATE@@', cfg['date']), ('@@WEEKDAY@@', cfg['weekday']),
                 ('@@TITLE@@', cfg['title']), ('@@SUBTITLE@@', cfg['subtitle'])]:
        body = body.replace(k, v)
    for k, v in pageno.items():
        body = body.replace('@@' + k + '@@', str(v))
    return body


# ---------------------------------------------------------------------------
# 编译与合并
# ---------------------------------------------------------------------------
def compile_a(tex_path, tag):
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                            os.path.basename(tex_path)],
                           cwd=os.path.dirname(tex_path),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            log = os.path.join(os.path.dirname(tex_path), 'daily_pack.log')
            tail = ''
            if os.path.isfile(log):
                tail = io.open(log, encoding='utf-8', errors='replace').read()[-2500:]
            raise RuntimeError('xelatex 编译失败 (%s):\n%s' % (tag, tail))
    pdf = tex_path[:-4] + '.pdf'
    return pymupdf.open(pdf).page_count


def page_counts(cfg):
    out = []
    for s in cfg['sections']:
        n = 0
        for path, pages in s['items']:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            n += len(pages)
        out.append(n)
    return out


def build(day):
    cfg = DAY_CONFIG[day]
    os.makedirs(WORK, exist_ok=True)
    tex_name = '每日阅读包_%s' % day
    tex_path = os.path.join(WORK, tex_name + '.tex')
    counts = page_counts(cfg)
    keys = [s['key'] for s in cfg['sections']]

    # 迭代求 A 部分页数与 B 段起始页（导航表写绝对页码）
    pageno = {k: 0 for k in keys}
    nA = None
    for _ in range(4):
        tex = fix_unicode((PREAMBLE + make_body(cfg, pageno)).replace('@@DATE@@', cfg['date']))
        assert '@@' not in tex, '有未替换的占位符'
        io.open(tex_path, 'w', encoding='utf-8').write(tex)
        pages = compile_a(tex_path, day)
        start = pages + 1
        new = {}
        for k, c in zip(keys, counts):
            new[k] = start
            start += c
        if nA == pages and new == pageno:
            break
        nA, pageno = pages, new
    else:
        raise RuntimeError('页码迭代未收敛')

    # 同步 tex 源到 Resources/Source/（Note 仓库自包含）
    shutil.copy2(tex_path, os.path.join(SOURCE, tex_name + '.tex'))

    a_pdf = os.path.join(WORK, tex_name + '.pdf')
    doc = pymupdf.open(a_pdf)
    toc = [[1, 'A · 今日执行卡与可书写页', 1]]

    # A 部分各页书签（按 section 标题粗分）
    a_titles = ['A1 封面 · 五级优先级 · 使用说明', 'A2 阅读路线图 B1–B15',
                'A3 P0 闭卷默写', 'A4 P1 九宫格闭卷默写',
                'A5 记录表 A / B', 'A6 记录表 C / D · 验收清单',
                'A7 核对答案① 中值定理 · 数列（压缩版）', 'A8 核对答案② 导数 · 作图',
                'A9 核对答案③ 三型 · 泰勒 · 间断点 · 定义', 'A10 高频四件 · 陷阱 · 边界']
    while len(a_titles) < nA:
        a_titles.append('A%d 延续页' % (len(a_titles) + 1))
    for i, t in enumerate(a_titles[:nA]):
        toc.append([1, t, i + 1])

    # B 部分合订
    cur = nA
    for s, c in zip(cfg['sections'], counts):
        toc.append([1, '%s  %s（本 PDF p%d-%d）' % (s['key'], s['title'],
                                                   pageno[s['key']], pageno[s['key']] + c - 1),
                    pageno[s['key']]])
        for path, pages in s['items']:
            src = pymupdf.open(path)
            for pn in pages:
                doc.insert_pdf(src, from_page=pn - 1, to_page=pn - 1)
                pg = doc[cur]
                label = '%s · %s 原 p%d' % (s['key'], os.path.basename(path).replace('.pdf', ''), pn)
                try:
                    pg.insert_text((26, pg.rect.height - 13), label, fontsize=7,
                                   fontname='china-s', color=(0.42, 0.42, 0.42))
                except Exception:
                    pg.insert_text((26, pg.rect.height - 13),
                                   '%s p%d' % (s['key'], pn), fontsize=7,
                                   color=(0.42, 0.42, 0.42))
                pg.insert_text((pg.rect.width - 42, pg.rect.height - 13),
                               str(cur + 1), fontsize=8, color=(0.42, 0.42, 0.42))
                cur += 1
            src.close()

    assert doc.page_count == nA + sum(counts), (doc.page_count, nA, sum(counts))
    doc.set_toc(toc)
    doc.set_metadata({'title': '每日阅读包 %s · %s' % (cfg['date'], cfg['title']),
                      'author': '陶理',
                      'subject': '高数 I（A 类）期中冲刺 · 当日全部阅读材料合订',
                      'keywords': '高数, 期中, 每日阅读包, %s' % day})
    out = os.path.join(DOWNLOADS, '每日阅读包_%s.pdf' % day)
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    # 清理中间文件
    for f in os.listdir(WORK):
        if f.endswith(('.aux', '.log', '.out', '.toc')):
            os.remove(os.path.join(WORK, f))
    return dict(out=out, nA=nA, nB=sum(counts), total=nA + sum(counts),
                pageno=pageno, counts=counts, sections=cfg['sections'])


if __name__ == '__main__':
    if '--list' in sys.argv:
        for k, v in DAY_CONFIG.items():
            print(k, v['date'], v['title'], '| B 段', len(v['sections']))
        sys.exit(0)
    day = sys.argv[1] if len(sys.argv) > 1 else '11-07'
    if day not in DAY_CONFIG:
        print('未知日期：%s；可用：%s' % (day, ', '.join(DAY_CONFIG)))
        sys.exit(1)
    r = build(day)
    print('OK ->', r['out'])
    print('A 部分 %d 页 / B 部分 %d 页 / 合计 %d 页' % (r['nA'], r['nB'], r['total']))
    for s in r['sections']:
        print('  %-4s p%-3d  %s' % (s['key'], r['pageno'][s['key']], s['title']))
