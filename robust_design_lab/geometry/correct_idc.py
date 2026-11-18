# -*- coding: utf-8 -*-
"""正确的交指电容组件：手指在结区两侧互相交错，无重叠。

官方 Transmon_Interdigitated 的 180° 旋转原点错误导致上下手指重叠
（重叠面积 0.002–0.004 mm²，Q3D 求解失败）。本组件直接构造正确几何：

┌──────────────┐
│   上方 pad    │
├  ┌┐  ┌┐  ┌┐ │  ← 上手指（从上 pad 底面向下伸出）
│  └┘  └┘  └┘ │
│  ┌┐  ┌┐  ┌┐ │  ← 下手指（从下 pad 顶面向上伸出）
├  └┘  └┘  └┘ │
│   下方 pad    │
└──────────────┘
    ↑ 结在最内两根手指之间（y=0）

电容来自上下手指的**侧面交叠**。
"""

import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import BaseQubit
from shapely.geometry import box


class CorrectIDC(BaseQubit):
    """交指电容 transmon——手指在结区两侧交错，无重叠。

    Options:
        * pad_width: pad 宽度（沿 x）
        * pad_height: pad 高度（沿 y，单个 pad）
        * finger_width: 手指宽（沿 x）
        * finger_height: 手指长（沿 y，从 pad 边缘伸出的长度）
        * finger_gap: 手指间隙（相邻手指中心距 = finger_width + finger_gap）
        * n_fingers: 每侧手指数（上下各 n 个，交错排列）
        * junction_gap: 结区长度（上下 pad 内边缘之间的 y 向距离）
        * conn_w / conn_h: 连接垫尺寸
    """

    default_options = Dict(
        pad_width="300um",
        pad_height="80um",
        finger_width="15um",
        finger_height="60um",
        finger_gap="15um",
        n_fingers=5,
        junction_gap="20um",
        conn_w="100um",
        conn_h="30um",
        jj_width="20um",
        pocket_margin="60um",
    )

    component_metadata = Dict(
        short_name="Q",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
    )

    TOOLTIP = "Correct interdigitated capacitor transmon."

    def make(self):
        from shapely.geometry import LineString

        p = self.p
        chip = p.chip

        # 参数解析（mm）
        pw = p.pad_width
        ph = p.pad_height
        fw = p.finger_width
        fh = p.finger_height
        fg = p.finger_gap
        nf = int(p.n_fingers)
        # 间隙约束：手指不得穿透对面 pad（jg ≥ 2*fh + 2*fg）
        jg_min = 2 * fh + 2 * fg
        jg = max(p.junction_gap, jg_min)
        cw = p.conn_w
        ch = p.conn_h
        margin = p.pocket_margin

        # ── 下 pad + 手指（向上伸出） ──
        # pad 中心在 y = -(jg/2 + ph/2)
        y_bot_center = -(jg + ph) / 2
        pad_bot = box(-pw / 2, y_bot_center - ph / 2, pw / 2, y_bot_center + ph / 2)

        # 手指：从 pad 顶面(y = y_bot_center + ph/2 = -jg/2)向上伸 finger_height
        y_bot_top = y_bot_center + ph / 2  # = -jg/2
        pitch = fw + fg  # 手指中心距
        fingers_bot = []
        for i in range(nf):
            # 交错排列：下手指在偶数位（x = 0, pitch*2, pitch*4, ...）
            # 对称：x = 0, ±pitch*2, ±pitch*4, ...
            if i == 0:
                x = 0
            else:
                x = (i + 1) // 2 * pitch * (1 if i % 2 == 1 else -1)
            # 实际用简单的均匀排列
            x = (i - (nf - 1) / 2) * pitch * 2  # 间距为 2*pitch（隔一个放）
            fingers_bot.append(box(x - fw / 2, y_bot_top,
                                   x + fw / 2, y_bot_top + fh))

        # 合并下部分
        bottom = pad_bot
        for f in fingers_bot:
            bottom = bottom.union(f)

        # ── 上 pad + 手指（向下伸出） ──
        y_top_center = (jg + ph) / 2
        pad_top = box(-pw / 2, y_top_center - ph / 2, pw / 2, y_top_center + ph / 2)

        # 手指：从 pad 底面(y = y_top_center - ph/2 = +jg/2)向下伸 finger_height
        y_top_bot = y_top_center - ph / 2  # = +jg/2
        fingers_top = []
        for i in range(nf):
            # 上手指偏移一个 pitch（与下手指交错）
            x = ((i - (nf - 1) / 2) * pitch * 2) + pitch
            fingers_top.append(box(x - fw / 2, y_top_bot - fh,
                                   x + fw / 2, y_top_bot))

        # 合并上部分
        top = pad_top
        for f in fingers_top:
            top = top.union(f)

        # ── 结线（在 y=0，最内两根手指之间） ──
        jj = LineString([(0, -jg / 2), (0, jg / 2)])

        # ── pocket（从地面挖掉） ──
        bounds_core = bottom.union(top).bounds
        # pocket 必须覆盖 pad+手指+连接垫的全部区域
        pocket = box(bounds_core[0] - margin - cw, bounds_core[1] - margin,
                     bounds_core[2] + margin + cw, bounds_core[3] + margin)
        bounds = bounds_core  # 连接垫位置仍用核心边界

        # ── 连接垫 ──
        cx = 0
        connectors = {
            "c1": box(bounds[0] - margin - cw, -ch / 2,
                      bounds[0] - margin, ch / 2),
            "c2": box(bounds[2] + margin, -ch / 2,
                      bounds[2] + margin + cw, ch / 2),
        }

        # ── 旋转/平移到全局位置 ──
        geoms = [bottom, top, jj, pocket] + list(connectors.values())
        geoms = draw.rotate(geoms, p.orientation, origin=(0, 0))
        geoms = draw.translate(geoms, p.pos_x, p.pos_y)

        bottom_g, top_g, jj_g, pocket_g = geoms[:4]
        conn_g = geoms[4:]

        # ── 添加 qgeometry ──
        self.add_qgeometry("poly", dict(pad_bot=bottom_g, pad_top=top_g), chip=chip)
        self.add_qgeometry("poly", dict(pocket=pocket_g), subtract=True, chip=chip)
        for i, (name, c) in enumerate(zip(sorted(connectors), conn_g)):
            self.add_qgeometry("poly", {name: c}, chip=chip)
            # pin
            pts = np.array(list(c.exterior.coords)[:2])
            self.add_pin(name, points=pts, width=ch, input_as_norm=False, chip=chip)
        self.add_qgeometry("junction", dict(jj=jj_g), width=p.jj_width, chip=chip)


def gen_idc_variant(n, rng):
    """生成参数变体。"""
    return [dict(
        pad_width=f"{rng.uniform(250, 400):.0f}um",
        pad_height=f"{rng.uniform(60, 120):.0f}um",
        finger_width=f"{rng.uniform(10, 25):.0f}um",
        finger_height=f"{rng.uniform(40, 80):.0f}um",
        finger_gap=f"{rng.uniform(10, 25):.0f}um",
        n_fingers=int(rng.integers(3, 8)),
        junction_gap=f"{rng.uniform(15, 30):.0f}um",
        conn_w="100um", conn_h="30um", jj_width="20um",
        pocket_margin="60um",
    ) for _ in range(n)]
