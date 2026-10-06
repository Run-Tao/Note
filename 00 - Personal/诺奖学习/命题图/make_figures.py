# -*- coding: utf-8 -*-
"""生成 2026 诺奖情境题（上海生物等级考风格）配图。

输出（本目录）：
  图1_光遗传学技术路线.png
  图2_光照前后膜电位变化.png

用法：python make_figures.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REG = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, size, index=0)


def ctext(d, xy, text, fnt, fill="black", spacing=8):
    d.multiline_text(xy, text, font=fnt, fill=fill, anchor="mm",
                     align="center", spacing=spacing)


# ============================ 图1：技术路线 ============================
def figure1(path):
    W, H = 1180, 400
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_box, f_title, f_note = font(20), font(25, True), font(17)

    labels = [
        "单细胞藻类\n（含 ChR 基因）",
        "重组质粒\n（载体 + ChR 基因）",
        "神经元\n（体外培养 / 活体）",
        "470 nm 蓝光照射",
        "记录电信号\n/ 观察行为",
    ]
    bw, bh, gap, top = 198, 118, 28, 96
    boxes = []
    x = 42
    for t in labels:
        d.rounded_rectangle([x, top, x + bw, top + bh], radius=12,
                            outline="#1f4e79", width=3, fill="#eef5fb")
        ctext(d, (x + bw / 2, top + bh / 2), t, f_box)
        boxes.append((x, top, x + bw, top + bh))
        x += bw + gap

    for i in range(len(boxes) - 1):
        sx = boxes[i][2] + 4
        ex = boxes[i + 1][0] - 6
        my = top + bh / 2
        d.line([sx, my, ex - 8, my], fill="#1f4e79", width=4)
        d.polygon([(ex - 8, my - 8), (ex + 4, my), (ex - 8, my + 8)], fill="#1f4e79")

    d.text((W / 2, 42), "图1  光遗传学技术路线示意图", font=f_title,
           fill="black", anchor="mm")
    d.text((W / 2, 268), "实验材料：野生型神经元、含 ChR 基因的重组质粒、蓝光光源（470 nm）、微电极记录装置、",
           font=f_note, fill="#333333", anchor="mm")
    d.text((W / 2, 296), "钙离子荧光指示剂；实验在 37 ℃ 生理溶液中完成，神经元静息电位约 −70 mV。",
           font=f_note, fill="#333333", anchor="mm")
    img.save(path)
    return path


# ============================ 图2：膜电位曲线 ============================
def figure2(path):
    W, H = 1080, 560
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_axis, f_title, f_leg = font(18), font(25, True), font(17)

    x0, x1 = 130, 980          # t: 0 → 1
    y0, y1 = 470, 80           # mV: -80 → +40
    vmin, vmax = -80.0, 40.0

    def X(t):
        return x0 + t * (x1 - x0)

    def Y(v):
        return y0 - (v - vmin) / (vmax - vmin) * (y0 - y1)

    t1, t2 = 0.30, 0.70        # 光照时段

    # 光照区间底色
    d.rectangle([X(t1), y1, X(t2), y0], fill="#fff7d6")
    for t in (t1, t2):
        d.line([X(t), y1, X(t), y0], fill="#d9b400", width=2)
    d.text(((X(t1) + X(t2)) / 2, y1 + 22), "蓝光照射", font=f_axis,
           fill="#8a6d00", anchor="mm")

    # 坐标轴
    d.line([x0, y0, x1 + 20, y0], fill="black", width=3)
    d.line([x0, y0, x0, y1 - 20], fill="black", width=3)
    d.text((x1 + 20, y0 + 26), "时间", font=f_axis, fill="black", anchor="mm")
    d.text((x0 - 34, y1 - 34), "膜电位/mV", font=f_axis, fill="black", anchor="mm")

    # 刻度与参考线
    for v in (-80, -70, -55, 0, 40):
        y = Y(v)
        d.line([x0 - 7, y, x0, y], fill="black", width=2)
        d.text((x0 - 14, y), str(v), font=f_axis, fill="black", anchor="rm")
    for v, col in ((-70, "#7f7f7f"), (-55, "#c00000")):
        y = Y(v)
        for xx in range(int(x0) + 6, int(x1) + 1, 16):
            d.line([xx, y, xx + 8, y], fill=col, width=2)

    # 对照组：静息 -70 mV
    d.line([X(0.02), Y(-70), X(0.98), Y(-70)], fill="#2e75b6", width=3)
    d.text((X(0.86), Y(-70) + 26), "曲线Ⅰ（对照组）", font=f_leg,
           fill="#2e75b6", anchor="mm")

    # 实验组：静息 → 去极化 → 连续动作电位 → 恢复
    pts = []
    def add(t, v):
        pts.append((X(t), Y(v)))

    add(0.02, -70); add(t1 - 0.01, -70)
    add(t1 + 0.015, -55)                       # 达阈值
    base, peak = -55.0, 30.0
    t = t1 + 0.02
    step = 0.022
    while t < t2 - 0.03:
        add(t, base)
        add(t + step * 0.4, peak)
        add(t + step * 0.8, -72)               # 超射后的后电位
        add(t + step, base)
        t += step
    add(t2 - 0.01, base)
    add(t2 + 0.03, -70)                        # 恢复静息
    add(0.98, -70)

    d.line(pts, fill="#c00000", width=3, joint="curve")
    d.text((X(0.50), Y(30) - 4), "曲线Ⅱ（实验组）", font=f_leg,
           fill="#c00000", anchor="mm")

    # 图注
    d.text((W / 2, 34), "图2  光照前后神经元膜电位变化曲线", font=f_title,
           fill="black", anchor="mm")
    d.text((x0, y0 + 44), "虚线：静息电位 −70 mV（灰）与阈值 −55 mV（红）；曲线Ⅱ 记录于表达 ChR 的神经元。",
           font=f_leg, fill="#333333", anchor="lm")
    img.save(path)
    return path


if __name__ == "__main__":
    p1 = figure1(os.path.join(HERE, "图1_光遗传学技术路线.png"))
    p2 = figure2(os.path.join(HERE, "图2_光照前后膜电位变化.png"))
    for p in (p1, p2):
        print(f"{os.path.getsize(p):>8} B  {p}")
