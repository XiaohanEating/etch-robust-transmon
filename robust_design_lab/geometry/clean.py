# -*- coding: utf-8 -*-
"""多边形清洗：保角抽稀 + 共线剔除 + 环向归一。

buffer/union 会产生重合顶点（0.00 μm 段）与过密顶点，既违反
顶点间距规则，又使 gmsh OCC 出现零长度曲线，
进而触发"triangle not found among the mesh facets"。

抽稀规则（保角）：仅当顶点与上一保留点距离 < min_len **且** 与前后方向
近共线（cos > 0.995）时才删除——union 产生的真角点（如颈角、弧-边交点）
永不删除，它们是保护带直边的端点。
"""

import math

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..configs import frozen


def _cos_angle(a, b, c) -> float:
    """在 b 点处 (a→b, b→c) 夹角余弦；退化返回 1（视为共线可删）。"""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 1.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)


def _decimate_ring(coords, min_len: float):
    """保角抽稀：过近且近共线的顶点删除，角点保留。"""
    n = len(coords) - 1                     # 环点数（去闭合重复）
    if n < 4:
        return list(coords)
    keep = [coords[0]]
    for i in range(1, n):
        pt = coords[i]
        prev = keep[-1]
        nxt = coords[(i + 1) % n]
        if math.dist(pt, prev) < min_len and _cos_angle(prev, pt, nxt) > 0.995:
            continue
        keep.append(pt)
    while len(keep) > 3 and math.dist(keep[-1], keep[0]) < min_len \
            and _cos_angle(keep[-2], keep[-1], keep[1]) > 0.995:
        keep.pop()
    return keep + [keep[0]]


def _drop_collinear(coords, tol_cos: float = 0.99995):
    """剔除严格共线顶点（相邻边夹角近 π）。coords 含闭合重复点。"""
    n = len(coords) - 1
    keep = []
    for i in range(n):
        a, b, c = coords[i - 1] if i > 0 else coords[n - 1], coords[i], coords[(i + 1) % n]
        if _cos_angle(a, b, c) < tol_cos:
            keep.append(b)
    if len(keep) < 3:
        return list(coords)
    return keep + [keep[0]]


def _merge_close_corners(coords, snap: float = 0.8):
    """合并近重合角点：距离 < snap 的相邻顶点合而为一。

    若其中一点位于结区保护线 x=±G2 上（1e-6 容差），吸附到该点以保住
    保护带直边端点；否则取中点。迭代两轮处理连锁近邻。
    """
    from ..configs import frozen
    g2 = frozen.JUNCTION_GAP_UM / 2.0

    def on_protect(pt):
        return abs(abs(pt[0]) - g2) < 1e-6

    pts = [tuple(c) for c in coords[:-1]]
    for _ in range(2):
        merged = []
        i = 0
        while i < len(pts):
            j = (i + 1) % len(pts)
            if math.dist(pts[i], pts[j]) < snap:
                a, b = pts[i], pts[j]
                keep = a if on_protect(a) else (b if on_protect(b) else
                                                ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
                merged.append(keep)
                i += 2
            else:
                merged.append(pts[i])
                i += 1
        if len(merged) < 3:
            break
        pts = merged if len(merged) < len(pts) else pts
    return pts + [pts[0]]


def clean_polygon(poly: Polygon, min_len: float = None) -> Polygon:
    """清洗：保角抽稀 → 近重合角点合并 → 去共线 → CCW 归一 → 简单性兜底。"""
    min_len = frozen.MIN_VERTEX_SPACING_UM if min_len is None else min_len
    coords = list(poly.exterior.coords)
    coords = _decimate_ring(coords, min_len)
    coords = _merge_close_corners(coords)
    coords = _drop_collinear(coords)
    out = orient(Polygon(coords))
    if not (out.is_valid and out.is_simple) or out.area <= 0:
        out = poly.buffer(0)
    return out
