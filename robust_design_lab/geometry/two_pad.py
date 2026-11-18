# -*- coding: utf-8 -*-
"""两岛单结组件的 Metal 适配层。

FreeTwoPadTransmon：接受任意多边形（μm 顶点表），落到 QDesign 的
poly/junction 表；与 tutorial 4.4x 验证载体 TwoPadTransmon 同一拓扑
（无地面、无 subtract、零厚度金属片），保证快速层与精确层边界条件一致。
"""

from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import BaseQubit


class FreeTwoPadTransmon(BaseQubit):
    """两岛任意多边形 + 结线的 transmon（无地面口袋）。

    Options:
        * pad_left / pad_right: 岛多边形顶点表 [[x, y], ...]，**毫米**，
          组件局部坐标（结沿 x 轴，两岛分列左右）；
        * junction: 结线两端点 [[x0, y0], [x1, y1]]，毫米；
        * jj_width: 结线绘制宽度（字符串带单位）。
    """

    default_options = Dict(
        pad_left=[],
        pad_right=[],
        junction=[],
        jj_width="20um",
    )

    component_metadata = Dict(
        short_name="Q",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
    )

    TOOLTIP = "Two-island free-polygon transmon, no ground pocket."

    def make(self):
        from shapely.geometry import LineString, Polygon

        p = self.p
        pad_left = Polygon([tuple(pt) for pt in p.pad_left])
        pad_right = Polygon([tuple(pt) for pt in p.pad_right])
        jj = LineString([tuple(pt) for pt in p.junction])
        geoms = draw.rotate([pad_left, pad_right, jj], p.orientation, origin=(0, 0))
        pad_left, pad_right, jj = draw.translate(geoms, p.pos_x, p.pos_y)
        self.add_qgeometry(
            "poly", dict(pad_left=pad_left, pad_right=pad_right), chip=p.chip
        )
        self.add_qgeometry("junction", dict(jj=jj), width=p.jj_width, chip=p.chip)


def record_to_options(rec, um_to_mm=1e-3):
    """GeometryRecord（μm）→ FreeTwoPadTransmon options（毫米）。"""
    def poly_pts(poly):
        return [[x * um_to_mm, y * um_to_mm] for x, y in poly.exterior.coords]

    return dict(
        pad_left=poly_pts(rec.pad_left),
        pad_right=poly_pts(rec.pad_right),
        junction=[[x * um_to_mm, y * um_to_mm] for x, y in rec.junction.coords],
        jj_width=f"{20.0}um",
    )


def build_design(rec, box_mm=(6.0, 6.0, 3.0), t_mm=0.5, eps_note=True,
                 name="Q1", overwrite=True):
    """GeometryRecord → MultiPlanar QDesign（零厚度金属/衬底层栈/封装变量）。

    返回 (design, component_name)。与 tutorial build_design 同一层栈约定：
    layer 1 厚度 0（金属片），layer 3 厚度 −t（衬底），sample_holder 定 lid。
    """
    import qiskit_metal
    from qiskit_metal import designs

    level = qiskit_metal.logger.level
    qiskit_metal.logger.setLevel("WARNING")
    try:
        design = designs.MultiPlanar({}, overwrite_enabled=overwrite)
    finally:
        qiskit_metal.logger.setLevel(level)

    Lx, Ly, Lz = box_mm
    size = design.chips.main.size
    size.center_x, size.center_y = f"{Lx / 2}mm", f"{Ly / 2}mm"
    size.size_x, size.size_y = f"{Lx}mm", f"{Ly}mm"
    ls = design.ls.ls_df
    ls.loc[ls.layer == 1, "thickness"] = "0um"
    ls.loc[ls.layer == 3, "thickness"] = f"-{t_mm}mm"
    design.variables["sample_holder_top"] = f"{Lz - t_mm}mm"
    design.variables["sample_holder_bottom"] = f"{t_mm}mm"

    FreeTwoPadTransmon(design, name, options=dict(
        record_to_options(rec), pos_x=f"{Lx / 2}mm", pos_y=f"{Ly / 2}mm"))
    design.rebuild()
    return design, name
