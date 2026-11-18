# -*- coding: utf-8 -*-
"""几何契约检查（方案 v2.0 §6.2 硬约束 + G0-geometry gate）。

检查项：简单多边形、结区保护带、岛间距、最小线宽（负缓冲侵蚀测试）、
最小顶点间距、结线端点落在对面边、外接框、面积下限。
返回 (ok, reasons)；reasons 为空列表则通过。
"""

import math

from shapely.geometry import Point

from ..configs import frozen


def check_record(rec) -> tuple:
    """对 GeometryRecord 执行全部硬约束检查。"""
    reasons = []

    W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2.0
    area_full = W2 * frozen.FOOTPRINT_UM[1]

    for name, poly in (("left", rec.pad_left), ("right", rec.pad_right)):
        if not (poly.is_valid and poly.is_simple):
            reasons.append(f"{name}: 非简单/非法多边形")
        if poly.area < frozen.MIN_AREA_FRACTION * area_full:
            reasons.append(f"{name}: 面积 {poly.area:.0f} μm² 低于下限")
        if poly.interiors:
            reasons.append(f"{name}: 含孔洞（主实验不允许）")

        b = poly.bounds
        if (b[0] < -frozen.FOOTPRINT_UM[0] / 2 - frozen.ISLAND_MARGIN_UM
                or b[1] < -frozen.FOOTPRINT_UM[1] / 2 - frozen.ISLAND_MARGIN_UM
                or b[2] > frozen.FOOTPRINT_UM[0] / 2 + frozen.ISLAND_MARGIN_UM
                or b[3] > frozen.FOOTPRINT_UM[1] / 2 + frozen.ISLAND_MARGIN_UM):
            reasons.append(f"{name}: 超出外接框 {tuple(round(v, 1) for v in b)}")

        # 最小线宽：内缩 MIN_LINEWIDTH/2 后仍非空且不碎裂
        eroded = poly.buffer(-frozen.MIN_LINEWIDTH_UM / 2)
        if eroded.is_empty or not eroded.is_valid or (
                eroded.geom_type == "MultiPolygon" and len(eroded.geoms) > 1):
            reasons.append(f"{name}: 存在窄于最小线宽的特征")

        ext = list(poly.exterior.coords)[:-1]
        if len(ext) >= 3:
            dmin = min(
                math.dist(ext[i], ext[(i + 1) % len(ext)]) for i in range(len(ext))
            )
            if dmin < frozen.MIN_VERTEX_SPACING_UM:
                reasons.append(f"{name}: 顶点间距 {dmin:.2f} μm 过小")

    # 岛间距：除结区外的任何位置不得低于最小间隙
    d_islands = rec.pad_left.distance(rec.pad_right)
    if d_islands < frozen.MIN_GAP_UM:
        reasons.append(f"岛间距 {d_islands:.2f} μm 低于最小间隙")

    # 结线端点落在两岛边界上（0.5μm 容差）
    for pt, name in ((rec.junction.coords[0], "left"),
                     (rec.junction.coords[-1], "right")):
        poly = rec.pad_left if name == "left" else rec.pad_right
        if poly.exterior.distance(Point(pt)) > 0.5:
            reasons.append(f"结线 {name} 端点未落在岛边界上")

    # 结区保护带：结线端点所在边须为直边且覆盖 |y−端点y| ≤ 保护带
    protect = frozen.JUNCTION_PROTECT_HALF_UM
    for pt, name, poly in ((rec.junction.coords[0], "left", rec.pad_left),
                           (rec.junction.coords[-1], "right", rec.pad_right)):
        ys = _protect_band_edge_coverage(poly, pt[0], pt[1])
        if ys is None or ys[0] > pt[1] - protect + 0.5 or ys[1] < pt[1] + protect - 0.5:
            reasons.append(f"{name}: 结区保护带内对面边不是直边")

    if abs(rec.junction.length - frozen.JUNCTION_GAP_UM) > max(0.5, 2 * frozen.ETCH_UM):
        reasons.append(f"结线长度 {rec.junction.length:.2f} 偏离名义间隙过远")

    return (len(reasons) == 0, reasons)


def _protect_band_edge_coverage(poly, x_edge: float, y_ref: float):
    """端点 (x_edge, y_ref) 处直边覆盖的 y 区间 (ymin, ymax)；无则 None。"""
    ymin, ymax = math.inf, -math.inf
    for a, b in zip(poly.exterior.coords[:-1], poly.exterior.coords[1:]):
        if abs(a[0] - x_edge) < 1e-6 and abs(b[0] - x_edge) < 1e-6:
            ymin = min(ymin, a[1], b[1])
            ymax = max(ymax, a[1], b[1])
    return (ymin, ymax) if ymin < ymax else None
