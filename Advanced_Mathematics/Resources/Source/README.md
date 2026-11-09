# Source —— 可编译源文件与复核脚本

本目录让 `Note` 仓库**自包含**：云端 agent 只需 clone 本仓库，即可拿到本批材料的 LaTeX 源、复核脚本与全部成品，无需访问本机 `D://Run_Tao//Homework//`（该目录不在任何 git 仓库内）。

权威原稿仍在 `D://Run_Tao//Homework//Advanced Mathematics\期中复习\`，本目录是 2026-11-04 的**同步快照**（2026-11-07 已追加当日的加练册、高难册与阅读包源）；如二者不一致，以原稿为准并重新同步。

## 文件对照

| 源文件 | 成品（`../Downloads/`） | 页数 |
| --- | --- | --- |
| `讲义_导数计算专题.tex` | `讲义_导数计算专题.pdf` | 5 |
| `讲义_函数性态与作图.tex` | `讲义_函数性态与作图.pdf` | 4 |
| `讲义_数列极限与压轴新题型.tex` | `讲义_数列极限与压轴新题型.pdf` | 4 |
| `讲义_该翻哪一本.tex` | `讲义_该翻哪一本.pdf` | 4 |
| `模拟卷C.tex` | `模拟卷C.pdf` | 2 |
| `模拟卷C_答案.tex` | `模拟卷C_答案.pdf` | 4 |
| `每日阅读包_11-07.tex` | `每日阅读包_11-07.pdf`（A 部分 10 页；B 部分 64 页由脚本合订） | 74 |
| `讲义_今日配套加练_11-07.tex` | `讲义_今日配套加练_11-07.pdf` | 9 |
| `讲义_高难冲顶_11-07.tex` | `讲义_高难冲顶_11-07.pdf`（题面 p1–4、答案 p5、详解 p6–14、验收 p15） | 15 |
| `讲义_练习_函数极限计算与泰勒.tex` | `讲义_练习_函数极限计算与泰勒.pdf`（2026-11-07 修复两处圈号缺字并重编译，页数不变） | 9 |
| `讲义_泰勒与压轴强化_11-08.tex` | `讲义_泰勒与压轴强化_11-08.pdf` | 13 |
| `讲义_全专题冲顶_11-08.tex` + `part2_body.tex` | `讲义_全专题冲顶_11-08.pdf`（主文件 + `\include` 的后半部分，须放在一起编译） | 24 |
| `讲义_加练_定义与压轴深化_11-08.tex` | `讲义_加练_定义与压轴深化_11-08.pdf` | 12 |
| `每日阅读包_11-08.tex` + `_a1108_body.tex` | `每日阅读包_11-08.pdf`（A 部分正文独立成文件；B 部分 51 页由脚本合订） | 61 |

`verify_daily_drill.py` 是 11/7 配套加练讲义 50 道题的复核脚本（**PASS 106 / FAIL 0**，sympy 符号极限 + 隐函数二阶导 + 参数/极坐标导数 + mpmath 高精度采样）。

`verify_taylor_drill.py`（**PASS 58 / FAIL 0**）、`verify_full_topics_drill.py`（**PASS 154 / FAIL 0**）与 `verify_extra_drill.py`（**PASS 106 / FAIL 0**）是 11/8 三册讲义的复核脚本，**先于讲义写成**，共抓出 21 处自身答案错误；`_a1108_body.tex` 是阅读包 A 部分的独立 LaTeX 正文（由 `build_daily_pack.py` 读入并替换占位符）。

`verify_hard_drill.py` 是 11/7 高难冲顶 22 题的复核脚本（**PASS 91 / FAIL 0**）：H1 极限（含四阶展开与首项抵消）、H2 隐函数与摆线二阶导、$1/(x^2-3x+2)$ 的 $n$ 阶导通项、分段连续可导、H3 构造恒等式与反例抽检（含三明治逐层导数）、H4 斜/竖直渐近线与「驻点非极值」、心形线 $\theta=\pi$ 处斜率极限、H5 牛顿迭代 / 夹逼展开 / 递推单调有界 / 压缩映射误差界。

`verify_new_materials.py` 是本批新增材料的数学复核脚本（**PASS 93 / FAIL 0**），覆盖 Level2 导数、Taylor、作图渐近线与驻点、数列极限、模拟卷 C、三本讲义限时卷与 $n=500/5000$ 数值采样。

`build_daily_pack.py` 是**按日阅读包生成器**：A 部分由本目录 `.tex` 编译，B 部分按 `DAY_CONFIG` 指定的讲义/练习/真题页码用 pymupdf `insert_pdf` 合订，自动加 PDF 书签、来源页脚与绝对页码，输出 `../Downloads/每日阅读包_<MM-DD>.pdf`。新增一天只需在 `DAY_CONFIG` 加一个日期键（标题、时间表、`sections` 材料页段），A 部分正文在 `make_body()` 按日期分支。运行需 `xelatex` + `pymupdf`（复核脚本另需 sympy/mpmath）。

## 编译与运行

```bash
# 六份 .tex 均为自包含文档（各自带 \documentclass，无 \input / \includegraphics 外部依赖）
xelatex -interaction=nonstopmode -halt-on-error 讲义_导数计算专题.tex
xelatex -interaction=nonstopmode -halt-on-error 讲义_导数计算专题.tex   # 两遍以稳定交叉引用

# 复核脚本依赖 sympy / mpmath，控制台中文需 UTF-8
PYTHONIOENCODING=utf-8 python verify_new_materials.py
PYTHONIOENCODING=utf-8 python verify_daily_drill.py   # 11/7 加练 106 项
PYTHONIOENCODING=utf-8 python verify_hard_drill.py    # 11/7 高难 91 项
PYTHONIOENCODING=utf-8 python verify_taylor_drill.py       # 11/8 泰勒册 58 项
PYTHONIOENCODING=utf-8 python verify_full_topics_drill.py  # 11/8 全专题册 154 项
PYTHONIOENCODING=utf-8 python verify_extra_drill.py        # 11/8 加练册 106 项
```

## 注意

- 中文文件名在部分 CI / 云环境需 UTF-8 locale；若编译报 `Missing character`，检查是否使用了 ctex + XeLaTeX（不是 pdflatex）。
- PDF 文本层会打乱数学公式因子顺序，**核对公式只认 `.tex` 源码或渲染 PNG**，不要用 `pdftotext` / `pymupdf.get_text()` 下结论。
- 三本讲义与答案册中的「93 项 PASS」即由本目录脚本产出；改动任一数值结论后必须重跑。

## 新讲义模板约定（2026-11-09 起，用户选定 A 方案）

**从下一份新讲义开始，一律复制 `Source/_新讲义起手式_tlnote.tex`，不要再复用本目录旧 `.tex` 的 preamble。**

- 起手式：`\documentclass[11pt,a4paper]{ctexart}` + `\usepackage[color]{tlnote}`（灰度打印换 `[gray]`）+ `\title` / `\author` / `\noteheader` / `\docdate{YYYY-MM-DD}`；**XeLaTeX 两遍**。`tlnote.sty` 由 `D:\Run_Tao\Homework\install.ps1` 装到 TEXMFHOME，`kpsewhich tlnote.sty` 可验证。
- `tlnote.sty` 已封装 geometry / titlesec / tcolorbox / fancyhdr / hyperref / tikz / pgfplots / enumitem / booktabs / longtable / tabularx / caption / pifont、`keypoint` / `warnbox` / `methodbox` / `notebox` 四个语义盒，以及全部数学简写与页眉页脚。**不要重复 `\usepackage`、不要 `\newtcolorbox`、不要重写页眉页脚。**
- 禁止重复定义（会报 `already defined` 或静默改正文）：`\abs \ee \D \dd \blank \ans \R \eps \N \Z \ue \dif \ds \floor \norm \key \tagline \kw \yes \no \src \ver \card \rowh \q \Q \qi`。`\blank` 用可选参数 `\blank[1.2cm]`；正文提到命令名写 `\texttt{xxx}`，别裸写带参数宏。
- 样板成品 `../Downloads/新讲义起手式_tlnote_11-09.pdf`（**1 页 / 3 书签 / 0.065 MB**），仅作版式示范、不含练习题、不计入题池。xelatex 两遍实测零错误、零警告、缺字 0、overfull 0；默认注释掉 `\tableofcontents` 以收在 1 页（书签由 `\section` 生成，不依赖目录），长讲义再取消注释。
- **存量不迁移**：本目录 11/04–11/09 的 30 份讲义保持 `ctexart` + 手写 preamble；合订本 370 页、阅读包 61 / 74 页、PDF 书签与绝对页码索引全部绑定原排版，重排会连锁破坏。

## 分享版讲义生成链（2026-11-09 新增）

面向「分享给学弟学妹」的完整讲义：`_share_notes.tex`（8 章讲解 + 卷首 + 5 附录）、`verify_share_handout.py`（例题答案复核）、`build_share_handout.py`（装配 + 闸门）、`align_share_bookmarks.py`（书签对齐 + 归档）、`讲义_高数I期中_分享版.tex`（成品源），PDF 在 `../Downloads/讲义_高数I期中_分享版_2026-11-09.pdf`。

**复现顺序**（`PYTHONIOENCODING=utf-8`，解释器用 `/d/anaconda3/python`，它有 sympy / mpmath / pymupdf）：

```bash
python verify_share_handout.py       # 必须 PASS 82 / FAIL 0
python build_share_handout.py        # 生成 .tex；内含两道硬闸门，不全绿直接拒绝
xelatex -interaction=nonstopmode -halt-on-error 讲义_高数I期中_分享版.tex   # 迭代 3 遍
python align_share_bookmarks.py      # 以 .toc 页码重写书签 → 归档到 Downloads/
```

- **迭代收敛判据**：同时比较 `.toc` 与 `.out` 的哈希，两者都稳定才停（只看 `.toc` 会让 PDF 书签差 1 页）。
- **为什么需要对齐书签**：标题被推到页首时 hyperref 把锚点写在前一页，目录页码对、书签差 1。试过 `\usepackage{needspace}` + `\needspace{6\baselineskip}`，书签能修但章末留白明显变差，已撤掉；改用后处理，更省纸也更可控。
- **longtable 必须写全五段**：`\endfirsthead` / `\endhead` / `\endfoot` / `\endlastfoot` 缺 `\endfoot` 会在表尾多吐一页只有「答案（续）」表头的空白页；另外最后一行不要带 `\\`。
- **题库来源**：151 题题面与答案由 `_build_compact_handout.py` 的解析器从本目录四册 .tex 提取（`b.SRC = HERE`，不依赖 Homework 原稿），按 `chapter_of()` 归入 8 章；题号保留原册编号。题面含 `&` 的 3 条不转义（在 `\q` 正文里合法），答案列含 `&` / `\\` 时由生成器自动转义。
- **正文写作约束**：只用 `tlnote.sty` 的命令；正文提到命令名写 `\texttt{xxx}`；表格里 `\rowh` 必须放在 `\\` **之前**；`\cdot` 写进 `\subsection` 时包一层 `\texorpdfstring{$\cdot$}{·}`，否则书签里会消失。
