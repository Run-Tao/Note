# -*- coding: utf-8 -*-
"""生成《高数I期中完整讲义_合订本_2026-11-08.pdf》。

流程：扫描各册页数 -> 生成卷首 LaTeX（目录页码迭代收敛）-> xelatex 两遍
-> pymupdf 合订 -> 重建「编/册/章」层级书签 -> 统一页脚 -> 校验 -> 同步。
只做合订与卷首编纂，绝不改动任何一册正文。
"""
import io
import os
import shutil
import subprocess
import sys

import pymupdf

BASE = r'D:/Run_Tao/Homework/Advanced Mathematics/期中复习'
DOWN = r'D:/Run_Tao/Note/Advanced_Mathematics/Resources/Downloads'
SOURCE = r'D:/Run_Tao/Note/Advanced_Mathematics/Resources/Source'
TOOLS = r'D:/Run_Tao/Homework/_review/tools'

TEMPLATE = os.path.join(BASE, '_complete_front_template.tex')
GEN_TEX = os.path.join(BASE, '讲义_完整版_卷首_11-08.tex')
GEN_PDF = os.path.join(BASE, '讲义_完整版_卷首_11-08.pdf')
OUT_PDF = os.path.join(BASE, '高数I期中完整讲义_合订本_2026-11-08.pdf')
OUT_NAME = os.path.basename(OUT_PDF)

# (编名, [(文件名, 目录, 目录描述)])
PARTS = [
    ('第一编 · 系统与真题精讲', [
        ('高数I期中复习讲义.pdf', 'base',
         '卷面结构与得分策略、核心考点方法、十年真题精讲、作业诊断、附录速查表；37 页 / 30 书签'),
        ('讲义_该翻哪一本.pdf', 'base',
         '四类资料分工、每天标准五步、九大专题页码级对照、卡壳三分法；4 页'),
    ]),
    ('第二编 · 九大专题方法', [
        ('微分中值定理专题讲义.pdf', 'down',
         '万能构造表、母题 15 类、Rolle / Lagrange / Cauchy 压轴证明；25 页 / 43 书签'),
        ('讲义_函数极限与间断点.pdf', 'base',
         '函数极限计算、左右极限、间断点八类判型、八道真题、限时卷与全表；14 页'),
        ('讲义_极限定义证明.pdf', 'base',
         'eps-delta / eps-X / M 型三套模板、例题与证明书写规范；14 页'),
        ('讲义_导数计算专题.pdf', 'base',
         '隐函数、参数方程、幂指对数求导、Leibniz 高阶导、反函数与分段可导；5 页'),
        ('讲义_函数性态与作图.pdf', 'base',
         '单调极值、凹凸拐点、三类渐近线与作图七步；4 页'),
        ('讲义_数列极限与压轴新题型.pdf', 'base',
         '夹逼、单调有界、压缩映射、递推与迭代；4 页'),
    ]),
    ('第三编 · 练习与加练题库（主战场）', [
        ('讲义_练习_函数极限计算与泰勒.pdf', 'base',
         '35 题：函数极限计算与泰勒练习册'),
        ('讲义_练习_补充卷_真题难度.pdf', 'base',
         '18 题：真题难度补充卷'),
        ('讲义_今日配套加练_11-07.pdf', 'base',
         '50 题：M / L / J / T / G / E / S 八组加 Q 卷，附答案'),
        ('讲义_高难冲顶_11-07.pdf', 'base',
         '22 题：H1 极限 / H2 导数高阶导 / H3 中值构造 / H4 作图 / H5 数列压轴，逐题标对标真题'),
        ('讲义_泰勒与压轴强化_11-08.pdf', 'base',
         '册一：系统篇（决策卡 / 母函数表 / 三种余项 / 12 条错误）+ T 组 19 题 + 限时卷 12 题'),
        ('讲义_全专题冲顶_11-08.pdf', 'base',
         '册二：S1--S6 共 62 题 + 模拟卷 A / B 各 18 题 + 答案速查与逐题详解'),
        ('讲义_加练_定义与压轴深化_11-08.pdf', 'base',
         '册三：P1 定义 12 + P2 泰勒高阶导 12 + P3 函数方程迭代 10 + P4 作图 6 + 模拟卷 C 18 题'),
        ('讲义_网络对标加练_11-08.pdf', 'base',
         'W1--W5 共 30 题：对标 MIT OCW / kmath 交大卷 / LibreTexts / APEX / Paul 的卡点结构，题面全新'),
    ]),
    ('第四编 · 模拟卷（掐表用）', [
        ('期中模拟卷.pdf', 'base', '主讲义配套限时模拟卷题面'),
        ('模拟卷A.pdf', 'down', '11/9 整卷掐表题面（120 分钟）'),
        ('模拟卷A_答案.pdf', 'down', '模拟卷 A 逐题答案与详解'),
        ('模拟卷B.pdf', 'down', '11/10 整卷掐表题面（120 分钟）'),
        ('模拟卷B_答案.pdf', 'down', '模拟卷 B 逐题答案与详解'),
        ('模拟卷C.pdf', 'down', '模拟卷 C 题面（单题回炉用，不掐整卷）'),
        ('模拟卷C_答案.pdf', 'down', '模拟卷 C 逐题答案与详解'),
    ]),
    ('第五编 · 每日阅读包原件（按日执行 / 掐表打印）', [
        ('每日阅读包_11-07.pdf', 'down',
         '74 页：A 执行卡 10 页 + B1--B15 原件 64 页，26 书签，绝对页码未改'),
        ('每日阅读包_11-08.pdf', 'down',
         '61 页：A 执行卡 10 页 + B16--B32 原件 51 页，28 书签，B 页脚带来源与原页码'),
    ]),
    ('附录 · 今晚执行卡', [
        ('今晚执行卡_11-08.pdf', 'base',
         '3 页可书写：21:30--23:30 四项任务、12 组进度表、旧卷补记、G3 分数、三型正确率、回炉清单'),
    ]),
]


def book_path(name, where):
    return os.path.join(BASE if where == 'base' else DOWN, name)


def esc_tex(s):
    rep = {
        '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
        '_': r'\_', '%': r'\%', '#': r'\#', '&': r'\&',
        '$': r'\$', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return ''.join(rep.get(c, c) for c in s)


def scan():
    books = []          # dict: part, name, path, pages, toc, start
    for part, items in PARTS:
        for name, where, desc in items:
            p = book_path(name, where)
            if not os.path.exists(p):
                raise SystemExit('MISSING: ' + p)
            d = pymupdf.open(p)
            books.append(dict(part=part, name=name, path=p, pages=d.page_count,
                              toc=d.get_toc(), desc=desc))
            d.close()
    return books


def make_rows(books, front_pages):
    lines = []
    idx = 0
    cursor = front_pages
    for part, items in PARTS:
        lines.append(r'\multicolumn{4}{@{}l}{\textbf{'
                     + esc_tex(part) + '}} \\\\')
        for name, where, desc in items:
            b = next(x for x in books if x['name'] == name and x['part'] == part)
            b['start'] = cursor + 1
            cursor += b['pages']
            idx += 1
            label = esc_tex(os.path.splitext(name)[0])
            lines.append('%d & %s & %d--%d & %s \\\\' %
                         (idx, label, b['start'], cursor, esc_tex(desc)))
        lines.append(r'\midrule')
    return '\n'.join(lines), idx, cursor


def compile_front(front_guess, books, body_pages):
    rows, nbooks, total = make_rows(books, front_guess)
    assert total == front_guess + body_pages
    s = io.open(TEMPLATE, encoding='utf-8').read()
    s = (s.replace('@@TOCROWS@@', rows)
          .replace('@@TOTAL@@', str(total))
          .replace('@@NBOOKS@@', str(nbooks))
          .replace('@@COREBEGIN@@', str(front_guess + 1)))
    io.open(GEN_TEX, 'w', encoding='utf-8').write(s)
    for _ in range(2):
        p = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                            os.path.basename(GEN_TEX)],
                           cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode:
            log = p.stdout.decode('utf-8', 'replace')
            print(log[-4000:])
            raise SystemExit('XELATEX FAILED rc=%d' % p.returncode)
    d = pymupdf.open(GEN_PDF)
    n, toc = d.page_count, d.get_toc()
    d.close()
    return n, toc, nbooks, total


def font_choice():
    for p in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc',
              'C:/Windows/Fonts/msyh.ttc']:
        if os.path.exists(p):
            return dict(fontfile=p, fontname='cjkfont')
    return dict(fontname='china-s')


def main():
    books = scan()
    body_pages = sum(b['pages'] for b in books)
    print('SCAN: %d books, %d body pages' % (len(books), body_pages))

    guess, seen = 8, set()
    for attempt in range(5):
        actual, front_toc, nbooks, total = compile_front(guess, books, body_pages)
        print('  front guess=%d -> actual=%d (total %d)' % (guess, actual, total))
        if actual == guess or actual in seen:
            guess = actual
            break
        seen.add(actual)
        guess = actual
    else:
        raise SystemExit('front page count did not converge')

    # 用最终页数再生成一次，确保目录里的页码与实际卷首页数一致
    front_pages = guess
    compile_front(front_pages, books, body_pages)
    front = pymupdf.open(GEN_PDF)
    assert front.page_count == front_pages, (front.page_count, front_pages)

    out = pymupdf.open()
    out.insert_pdf(front)
    toc = [[lvl, title, pg] for lvl, title, pg in front_toc
           if 1 <= pg <= front_pages]
    catalog_page = next((pg for lvl, title, pg in front_toc
                         if '全书目录' in title), 6)
    toc.append([1, '【全书目录 · 见卷首第 %d 页】' % catalog_page,
                catalog_page])

    for part, items in PARTS:
        part_books = [b for b in books if b['part'] == part]
        toc.append([1, part, part_books[0]['start']])
        for b in part_books:
            toc.append([2, '%d. %s （%d 页）' %
                        (books.index(b) + 1, os.path.splitext(b['name'])[0], b['pages']),
                        b['start']])
            for lvl, title, pg in b['toc']:
                if 1 <= pg <= b['pages']:
                    toc.append([min(lvl + 2, 4), title, b['start'] + pg - 1])
            d = pymupdf.open(b['path'])
            out.insert_pdf(d)
            d.close()
    front.close()

    n_total = out.page_count
    assert n_total == front_pages + body_pages, (n_total, front_pages + body_pages)
    out.set_toc(toc)

    # 统一页脚：合订本连续页码 + 编 + 原册页码
    fk = font_choice()
    part_short = {p: p.split(' · ')[0] for p, _ in PARTS}
    spans = []
    cursor = front_pages
    spans.append((1, front_pages, '卷首', '卷首 p.', 0))
    for b in books:
        spans.append((b['start'], b['start'] + b['pages'] - 1,
                      part_short[b['part']], os.path.splitext(b['name'])[0] + ' p.',
                      b['start']))
    for i, page in enumerate(out):
        lo, hi, pname, blabel, start = next(
            (s for s in spans if s[0] <= i + 1 <= s[1]),
            (1, n_total, '卷首', '卷首 p.', 0))
        if start:
            txt = ('合订本 p.%d / %d   |   %s   |   %s%d'
                   % (i + 1, n_total, pname, blabel, i + 2 - start))
        else:
            txt = '合订本 p.%d / %d   |   %s' % (i + 1, n_total, pname)
        r = page.rect
        try:
            page.insert_text((38, r.height - 9), txt, fontsize=6.6,
                             color=(0.45, 0.45, 0.45), **fk)
        except Exception as e:
            print('FONT FALLBACK:', e)
            page.insert_text((38, r.height - 9), txt, fontsize=6.6,
                             color=(0.45, 0.45, 0.45), fontname='china-s')

    out.set_metadata({
        'title': '高等数学 I（A 类）期中完整讲义 · 合订本 2026-11-08',
        'author': '陶理',
        'subject': '上海交大高数 A 类期中复习；2008-2025 十七套真题对标',
        'keywords': '高数 期中 讲义 合订 泰勒 极限定义 函数作图 中值定理',
    })
    out.save(OUT_PDF, deflate=True, garbage=4)
    out.close()

    # 校验
    chk = pymupdf.open(OUT_PDF)
    bad = 0
    empty = []
    for i, page in enumerate(chk):
        t = page.get_text()
        bad += sum(1 for c in t if ord(c) in (0xFFFD, 0xFFFF))
        if len(t.strip()) < 12:
            empty.append(i + 1)
    real_toc = chk.get_toc()
    print('\n=== RESULT ===')
    print('file      :', OUT_PDF)
    print('pages     : %d (front %d + body %d)' % (chk.page_count, front_pages, body_pages))
    print('toc       : %d entries, levels %s' %
          (len(real_toc), sorted(set(x[0] for x in real_toc))))
    print('size      : %.2f MB' % (os.path.getsize(OUT_PDF) / 1048576.0))
    print('bad chars :', bad)
    print('near-empty pages:', empty[:20])
    print('\n--- 每册起始页抽查 ---')
    for b in books:
        t = chk[b['start'] - 1].get_text().replace('\n', ' ')[:52]
        print(' p%-4d %-42s %s' % (b['start'], os.path.splitext(b['name'])[0], t))
    print('\n--- 目录页码 vs 实际书签 ---')
    for part, items in PARTS:
        b = next(x for x in books if x['part'] == part)
        hit = [x for x in real_toc if x[1] == part]
        print(' %-40s 目录 p%-4d 书签 %s' %
              (part[:38], b['start'], hit[0][2] if hit else 'MISS'))
    chk.close()

    # 同步到资料库
    for dst, files in [(DOWN, [OUT_PDF]),
                       (SOURCE, [GEN_TEX, TEMPLATE, OUT_PDF if False else GEN_PDF,
                                 os.path.join(TOOLS, 'build_complete_handout.py')])]:
        for f in files:
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    shutil.copy2(OUT_PDF, os.path.join(DOWN, OUT_NAME))
    print('\nSYNCED -> Downloads/ and Source/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
