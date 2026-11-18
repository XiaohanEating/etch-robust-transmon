# -*- coding: utf-8 -*-
"""任意形状双岛 transmon（带地面 pocket 与连接垫）——Q3D 官方路径用。

与 FreeTwoPadTransmon（裸双岛、无地面）不同，本组件：
- 生成 pocket（subtract=True）→ Q3D 渲染时从地面平面挖掉
- pad 为任意多边形（圆形/椭圆/十字/交指/不规则……）→ 独立导体（网络）
- 生成连接垫 → 从 pocket 伸出到地面，配合 open_terminations 使用
- 结线照常（junction 表）

用法：
    design = designs.design_planar.DesignPlanar()
    rec = TwoPadPocket(design, "Q1", options=dict(
        pad_left=[...],  # 任意多边形顶点（mm）
        pad_right=[...],
        junction=[[-0.015, 0], [0.015, 0]],
    ))
    c1 = LOManalysis(design, "q3d")
    c1.sim.run(components=["Q1"],
               open_terminations=[("Q1", "c1"), ("Q1", "c2"), ...])
"""

from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import BaseQubit


class TwoPadPocket(BaseQubit):
    """任意双岛形状 + ground pocket + 四方向连接垫。

    Options:
        * pad_left / pad_right: 岛多边形顶点 [[x, y], ...]（毫米）
        * junction: 结线两端点 [[x0, y0], [x1, y1]]（毫米）
        * pocket_margin: pocket 相对 pad 外扩（字符串带单位）
        * conn_pad_w / conn_pad_h: 连接垫尺寸（字符串带单位）
        * n_connectors: 连接垫数量（2 或 4，默认 4）
    """

    default_options = Dict(
        pad_left=[],
        pad_right=[],
        junction=[],
        pocket_margin="50um",
        conn_pad_w="100um",
        conn_pad_h="30um",
        n_connectors=4,
        jj_width="20um",
    )

    component_metadata = Dict(
        short_name="Q",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
    )

    TOOLTIP = "Arbitrary two-island transmon with ground pocket and connectors."

    def make(self):
        from shapely.geometry import Polygon, LineString, box

        p = self.p
        chip = p.chip

        pl = Polygon([tuple(pt) for pt in p.pad_left])
        pr = Polygon([tuple(pt) for pt in p.pad_right])
        jj = LineString([tuple(pt) for pt in p.junction])

        # pocket = pad 外接框 + margin（从地面挖掉）
        margin = p.pocket_margin  # parse_value 处理
        bounds = pl.union(pr).bounds
        pocket = box(bounds[0] - margin, bounds[1] - margin,
                     bounds[2] + margin, bounds[3] + margin)

        # 连接垫：path 型（与官方 TransmonPocket 同款），末端加 pin
        cw, ch = p.conn_pad_w, p.conn_pad_h
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        import numpy as np
        # 每个连接垫：从 pocket 边缘伸出到地面（路径 + 周围 gap 挖除）
        connector_defs = {
            "c1": ([(bounds[0] - margin, cy), (bounds[0] - margin - cw, cy)],
                    (-1, 0)),                                        # 左
            "c2": ([(bounds[2] + margin, cy), (bounds[2] + margin + cw, cy)],
                    (1, 0)),                                         # 右
            "c3": ([(cx, bounds[1] - margin), (cx, bounds[1] - margin - cw)],
                    (0, -1)),                                        # 下
            "c4": ([(cx, bounds[3] + margin), (cx, bounds[3] + margin + cw)],
                    (0, 1)),                                         # 上
        }
        # 按需保留连接垫（默认 4；3=保留 c2/c3/c4 即 2bus+1readout；2=仅 c3/c4）
        if p.n_connectors == 3:
            connector_defs = {k: v for k, v in connector_defs.items()
                              if k in ("c2", "c3", "c4")}
        elif p.n_connectors == 2:
            connector_defs = {k: v for k, v in connector_defs.items()
                              if k in ("c3", "c4")}

        # pad + 结（先加几何）
        geoms_core = [pl, pr, jj]
        geoms_core = draw.rotate(geoms_core, p.orientation, origin=(0, 0))
        geoms_core = draw.translate(geoms_core, p.pos_x, p.pos_y)
        self.add_qgeometry("poly", dict(pad_left=geoms_core[0],
                                        pad_right=geoms_core[1]), chip=chip)
        self.add_qgeometry("junction", dict(jj=geoms_core[2]),
                           width=p.jj_width, chip=chip)
        # pocket：从地面挖掉（pocket 不参与 rotate/translate——已在局部系）
        self.add_qgeometry("poly", dict(pocket=pocket), subtract=True, chip=chip)

        # 连接垫：path + pin（在组件局部系直接加，不经 rotate——简化：orientation=0）
        for name, (path_pts, normal_dir) in connector_defs.items():
            pts = np.array(path_pts, dtype=float)
            # 加到全局位置
            pts[:, 0] += float(p.pos_x)
            pts[:, 1] += float(p.pos_y)
            wire = draw.LineString(pts)
            self.add_qgeometry("path", {name: wire}, width=ch, chip=chip)
            # gap 挖除区（连接垫周围的地面挖除）
            from shapely.geometry import box as _box
            gap = 0.6 * ch  # 与 add_pin 默认一致
            lo = pts.min(axis=0) - gap
            hi = pts.max(axis=0) + gap + abs(ch) * abs(np.array(normal_dir))
            self.add_qgeometry("poly", {f"{name}_gap": _box(lo[0], lo[1], hi[0], hi[1])},
                               subtract=True, chip=chip)
            # pin（open_terminations 依赖）
            self.add_pin(name, points=pts, width=ch, input_as_norm=False, chip=chip)


def make_pocket_record(category, index, seed_master=20261026):
    """从形状库生成 TwoPadPocket 的 options（pad_left/pad_right/junction）。

    复用 geometry/library.py 的八类生成器（矩形/圆角/十字/交指/椭圆/
    梯形/缺口/不规则），输出毫米制顶点。
    """
    import numpy as np
    from .library import _GENERATORS, _mirror, _finish
    from .clean import clean_polygon
    from ..configs import frozen

    W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2
    G2 = frozen.JUNCTION_GAP_UM / 2

    cat_hash = sum(ord(c) * (i + 1) for i, c in enumerate(category)) % (2**20)
    seed = (seed_master * 1_000_003 + cat_hash * 9_77 + index * 31_337) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    body_left, body_right = _GENERATORS[category](rng)

    left = clean_polygon(_mirror(_finish(body_left)))
    right = clean_polygon(_finish(body_right))

    def to_mm(poly):
        return [[x * 1e-3, y * 1e-3] for x, y in poly.exterior.coords]

    return dict(
        pad_left=to_mm(left),
        pad_right=to_mm(right),
        junction=[[-G2 * 1e-3, 0], [G2 * 1e-3, 0]],
        pocket_margin="50um",
        conn_pad_w="100um",
        conn_pad_h="30um",
    )
