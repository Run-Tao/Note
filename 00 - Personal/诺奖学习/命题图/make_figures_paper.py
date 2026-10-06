# -*- coding: utf-8 -*-
"""生成"光遗传学视觉恢复"论文取材版配图（重绘）。

输出（本目录）：
  P1_视网膜三级通路与病变.png
  P2_AAV载体与启动子特异性.png
  P3_功能评价定性对照.png

用法：python make_figures_paper.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REG = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BLUE = "#1f4e79"
RED = "#c00000"
GREEN = "#2e7d32"
GREY = "#808080"
FILL = "#eef5fb"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, size, index=0)


def ctext(d, xy, text, fnt, fill="black", spacing=8):
    d.multiline_text(xy, text, font=fnt, fill=fill, anchor="mm",
                     align="center", spacing=spacing)


def arrow(d, x1, y, x2, color=BLUE, w=4):
    d.line([x1, y, x2 - 8, y], fill=color, width=w)
    d.polygon([(x2 - 8, y - 8), (x2 + 4, y), (x2 - 8, y + 8)], fill=color)


# ======================= P1：视网膜三级通路 + 病变 + 递送 =======================
def p1(path):
    W, H = 1220, 560
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_c, f_n, f_s = font(26, True), font(19), font(17), font(16)

    cells = [
        ("感光细胞\n（视杆 / 视锥）", "RP：大量死亡", RED),
        ("双极细胞", "相对存活", GREEN),
        ("神经节细胞", "相对存活", GREEN),
        ("视神经 → 大脑", "将信号传向中枢", BLUE),
    ]
    bw, bh, gap, top = 250, 130, 46, 130
    x = 40
    centers = []
    for name, note, col in cells:
        d.rounded_rectangle([x, top, x + bw, top + bh], radius=12,
                            outline=col, width=3, fill=FILL)
        ctext(d, (x + bw / 2, top + 34), name, f_c)
        d.text((x + bw / 2, top + bh - 26), note, font=f_s, fill=col, anchor="mm")
        d.text((x + bw / 2, top + bh + 28), "光", font=f_n, fill=GREY, anchor="mm")
        centers.append((x, x + bw))
        x += bw + gap
    for i in range(len(centers) - 1):
        arrow(d, centers[i][1] + 6, top + bh / 2, centers[i + 1][0] - 6)

    # 病变强调：在感光细胞上打叉
    d.line([centers[0][0] + 18, top + 18, centers[0][1] - 18, top + bh - 18],
           fill=RED, width=5)
    d.line([centers[0][0] + 18, top + bh - 18, centers[0][1] - 18, top + 18],
           fill=RED, width=5)

    # AAV 递送箭头
    ax = (centers[1][0] + centers[1][1]) / 2
    d.line([ax, top + bh + 78, ax, top + bh + 22], fill=BLUE, width=4)
    d.polygon([(ax - 8, top + bh + 22), (ax + 8, top + bh + 22), (ax, top + bh + 8)], fill=BLUE)
    d.text((ax, top + bh + 104), "AAV 载体递送\nChR2 基因", font=f_n, fill=BLUE, anchor="mm")

    d.text((W / 2, 44), "图P1  视网膜三级通路、RP 病变位置与基因递送靶点", font=f_t,
           fill="black", anchor="mm")
    d.text((W / 2, H - 40), "说明：视网膜色素变性（RP）主要损伤感光细胞，内层神经元相对存活；"
           "将光敏蛋白基因送入存活的神经元，可使其重新对光产生反应。", font=f_n,
           fill="#333333", anchor="mm")
    img.save(path)
    return path


# ==================== P2：AAV 载体 + 调控元件 → 细胞特异性表达 ====================
def p2(path):
    W, H = 1220, 580
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_c, f_n, f_s = font(26, True), font(19), font(17), font(16)

    # 左：AAV 载体与基因盒
    d.rounded_rectangle([40, 110, 560, 330], radius=12, outline=BLUE, width=3, fill=FILL)
    d.text((300, 136), "AAV 载体（携带目的基因）", font=f_c, fill=BLUE, anchor="mm")
    # 衣壳
    d.ellipse([70, 170, 140, 240], outline=BLUE, width=3)
    ctext(d, (105, 205), "衣壳", f_s)
    # 基因盒
    segs = [("增强子", "#fff2cc"), ("启动子", "#d9ead3"), ("ChR2 基因", "#f4cccc")]
    bx, by, bw, bh = 170, 172, 118, 66
    for i, (label, col) in enumerate(segs):
        x = bx + i * bw
        d.rectangle([x, by, x + bw, by + bh], outline=BLUE, width=2, fill=col)
        ctext(d, (x + bw / 2, by + bh / 2), label, f_s)
    d.text((300, 268), "调控元件决定「在哪里、表达多少」", font=f_s, fill="#333333", anchor="mm")
    d.text((300, 296), "基因本身只决定「表达什么」", font=f_s, fill="#333333", anchor="mm")

    # 箭头
    arrow(d, 570, 220, 660)

    # 右：视网膜细胞群，只有一个表达
    d.rounded_rectangle([670, 110, 1180, 330], radius=12, outline=GREY, width=3)
    d.text((925, 136), "视网膜内层（多种细胞混居）", font=f_c, fill="#333333", anchor="mm")
    xs = [715, 810, 905, 1000, 1095]
    for i, x in enumerate(xs):
        if i == 2:
            d.ellipse([x - 26, 180, x + 26, 250], fill="#f4cccc", outline=RED, width=3)
            d.text((x, 272), "表达 ChR2", font=f_s, fill=RED, anchor="mm")
        else:
            d.ellipse([x - 26, 180, x + 26, 250], fill="#f2f2f2", outline=GREY, width=2)
            d.text((x, 272), "不表达", font=f_s, fill=GREY, anchor="mm")

    # 底部：光 → 电信号
    d.text((W / 2, 400), "蓝光照射 → 只有表达 ChR2 的细胞通道开放、离子内流 → 产生电信号",
           font=f_c, fill=BLUE, anchor="mm")
    d.text((W / 2, 452), "意义：刺激只作用于「指定的一类细胞」，而不影响周围其他细胞——这就是细胞特异性。",
           font=f_n, fill="#333333", anchor="mm")
    d.text((W / 2, 44), "图P2  载体的调控元件决定目的基因的细胞特异性表达", font=f_t,
           fill="black", anchor="mm")
    img.save(path)
    return path


# ==================== P3：功能评价的定性对照（非原始数据） ====================
def p3(path):
    W, H = 1160, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_h, f_c, f_n = font(26, True), font(18, True), font(18), font(16)

    cols = ["组别（失明小鼠）", "视动反应 OMR", "瞳孔对光收缩"]
    rows = [
        ("未注射病毒（对照）", "未恢复", "不明显"),
        ("wt-ChR2", "恢复", "不明显"),
        ("ChR2-L132C", "恢复", "不明显"),
        ("ChR2-C/C（L132C/T159C）", "恢复", "不明显"),
        ("ChR2-C/S（L132C/T159S）", "恢复", "显著"),
    ]
    x0, y0, cw, rh = 70, 130, 350, 58
    widths = [cw + 40, 250, 250]
    # 表头
    x = x0
    for i, c in enumerate(cols):
        d.rectangle([x, y0, x + widths[i], y0 + rh], fill="#dbe5f1", outline=BLUE, width=2)
        d.text((x + widths[i] / 2, y0 + rh / 2), c, font=f_h, fill=BLUE, anchor="mm")
        x += widths[i]
    # 表体
    for r, row in enumerate(rows):
        x = x0
        yy = y0 + rh * (r + 1)
        for i, cell in enumerate(row):
            fill = "#ffffff" if r % 2 == 0 else "#f7f9fc"
            if i > 0:
                fill = "#e8f5e9" if cell in ("恢复", "显著") else "#fdecea"
            d.rectangle([x, yy, x + widths[i], yy + rh], fill=fill, outline="#9aa7b5", width=1)
            col = GREEN if cell in ("恢复", "显著") else (RED if cell == "未恢复" else "#555555")
            d.text((x + widths[i] / 2, yy + rh / 2), cell, font=f_c,
                   fill=col if i > 0 else "black", anchor="mm")
            x += widths[i]

    d.text((W / 2, 44), "图P3  不同 ChR2 变体恢复视觉功能的定性对照", font=f_t,
           fill="black", anchor="mm")
    d.text((W / 2, 462), "说明：本表按论文结果「定性」整理（非原始数据）：各变体组均能恢复 OMR；",
           font=f_n, fill="#333333", anchor="mm")
    d.text((W / 2, 488), "而瞳孔对光收缩只在光敏性最高的 ChR2-L132C/T159S 组达到显著水平。",
           font=f_n, fill="#333333", anchor="mm")
    img.save(path)
    return path


if __name__ == "__main__":
    for fn, name in ((p1, "P1_视网膜三级通路与病变.png"),
                     (p2, "P2_AAV载体与启动子特异性.png"),
                     (p3, "P3_功能评价定性对照.png")):
        p = fn(os.path.join(HERE, name))
        print(f"{os.path.getsize(p):>8} B  {name}")
