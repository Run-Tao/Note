# -*- coding: utf-8 -*-
r"""分享版讲义：对齐 PDF 书签页码并归档成品。

为什么需要这一步：xelatex 多遍收敛后，目录（.toc）页码是正确的，但个别
\subsection 若恰好被推到页首，hyperref 会把锚点写在前一页，导致 PDF 书签
差 1 页。这里以 .toc 的页码为真值，用 pymupdf 重写书签，同时把成品归档到
Downloads/（Source/ 只留 .tex 与复核脚本）。

运行：PYTHONIOENCODING=utf-8 python align_share_bookmarks.py
前置：xelatex 已迭代编译到 .toc 与 .out 同时稳定（build_share_handout.py 之后）。
"""
import io
import os
import re
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = '讲义_高数I期中_分享版'
SRC_PDF = os.path.join(HERE, STEM + '.pdf')
TOC_FILE = os.path.join(HERE, STEM + '.toc')
OUT_PDF = os.path.join(HERE, '..', 'Downloads',
                       '讲义_高数I期中_分享版_2026-11-09.pdf')


def toc_pages(path):
    out = []
    for ln in io.open(path, encoding='utf-8').read().splitlines():
        if '\\contentsline' not in ln:
            continue
        nums = re.findall(r'\{(\d+)\}', ln)
        if not nums:
            continue
        out.append(int(nums[-1]))
    return out


def main():
    pages = toc_pages(TOC_FILE)
    doc = pymupdf.open(SRC_PDF)
    bm = doc.get_toc()
    print('目录条目 %d / PDF 书签 %d / 页数 %d' % (len(pages), len(bm), doc.page_count))
    if len(pages) != len(bm):
        sys.exit('目录与书签条目数不一致，拒绝改写')

    before = [(i + 1, bm[i][1][:30], bm[i][2], pages[i])
              for i in range(len(bm)) if bm[i][2] != pages[i]]
    print('对齐前不一致 %d 条：' % len(before))
    for row in before:
        print('   ', row)

    for i, p in enumerate(pages):
        bm[i][2] = p
    doc.set_toc(bm)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    doc.save(OUT_PDF, garbage=4, deflate=True)
    doc.close()

    chk = pymupdf.open(OUT_PDF)
    bm2 = chk.get_toc()
    diffs = [(i + 1, bm2[i][2], pages[i])
             for i in range(len(bm2)) if bm2[i][2] != pages[i]]
    text = ''.join(p.get_text() for p in chk)
    print('\n归档：%s' % os.path.abspath(OUT_PDF))
    print('页数 %d / 书签 %d / 对齐后不一致 %d / 缺字 %d'
          % (chk.page_count, len(bm2), len(diffs),
             text.count(chr(0xfffd)) + text.count(chr(0xffff))))
    if diffs:
        sys.exit('书签对齐失败')
    print('全部书签页码与目录一致。')
    chk.close()


if __name__ == '__main__':
    main()
