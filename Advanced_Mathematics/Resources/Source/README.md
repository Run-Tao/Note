# Source —— 可编译源文件与复核脚本

本目录让 `Note` 仓库**自包含**：云端 agent 只需 clone 本仓库，即可拿到本批材料的 LaTeX 源、复核脚本与全部成品，无需访问本机 `D://Run_Tao//Homework//`（该目录不在任何 git 仓库内）。

权威原稿仍在 `D://Run_Tao//Homework//Advanced Mathematics\期中复习\`，本目录是 2026-11-04 的**同步快照**；如二者不一致，以原稿为准并重新同步。

## 文件对照

| 源文件 | 成品（`../Downloads/`） | 页数 |
| --- | --- | --- |
| `讲义_导数计算专题.tex` | `讲义_导数计算专题.pdf` | 5 |
| `讲义_函数性态与作图.tex` | `讲义_函数性态与作图.pdf` | 4 |
| `讲义_数列极限与压轴新题型.tex` | `讲义_数列极限与压轴新题型.pdf` | 4 |
| `讲义_该翻哪一本.tex` | `讲义_该翻哪一本.pdf` | 4 |
| `模拟卷C.tex` | `模拟卷C.pdf` | 2 |
| `模拟卷C_答案.tex` | `模拟卷C_答案.pdf` | 4 |

`verify_new_materials.py` 是本批新增材料的数学复核脚本（**PASS 93 / FAIL 0**），覆盖 Level2 导数、Taylor、作图渐近线与驻点、数列极限、模拟卷 C、三本讲义限时卷与 $n=500/5000$ 数值采样。

## 编译与运行

```bash
# 六份 .tex 均为自包含文档（各自带 \documentclass，无 \input / \includegraphics 外部依赖）
xelatex -interaction=nonstopmode -halt-on-error 讲义_导数计算专题.tex
xelatex -interaction=nonstopmode -halt-on-error 讲义_导数计算专题.tex   # 两遍以稳定交叉引用

# 复核脚本依赖 sympy / mpmath，控制台中文需 UTF-8
PYTHONIOENCODING=utf-8 python verify_new_materials.py
```

## 注意

- 中文文件名在部分 CI / 云环境需 UTF-8 locale；若编译报 `Missing character`，检查是否使用了 ctex + XeLaTeX（不是 pdflatex）。
- PDF 文本层会打乱数学公式因子顺序，**核对公式只认 `.tex` 源码或渲染 PNG**，不要用 `pdftotext` / `pymupdf.get_text()` 下结论。
- 三本讲义与答案册中的「93 项 PASS」即由本目录脚本产出；改动任一数值结论后必须重跑。
