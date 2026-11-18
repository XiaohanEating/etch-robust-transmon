# -*- coding: utf-8 -*-
"""形状库：8 类 × N 个候选的确定性生成器（方案 v2.0 §4.1）。

坐标约定（重要）：
- 全部 μm，组件局部系，结沿 x 轴，两岛分列 x=±g/2；
- **所有生成器的输出（body_left, body_right）一律在右槽坐标**（x ≥ g/2），
  左岛由 make_record 统一镜像到 x ≤ −g/2；
- 统一后处理 _finish：裁剪到槽位 → 并结区颈（x∈[g/2, g/2+40]，|y|≤保护带+5）
  → 归一为最大面积的简单多边形；结区保护带直边与结线端点约束由此自动满足。

八类（方案 §4.1）：
1 rect_rounded        Pocket 矩形及圆角矩形
2 cross               Cross/Xmon 臂变体
3 teeth_interdig      Teeth/梳齿和交指
4 ellipse_arc         椭圆、圆弧和多弧边界
5 trapezoid_asym      梯形、偏置和非对称双岛
6 notch_taper         缺口、凸台或 taper 边界
7 irregular           多尺度不规则多边形
8 combo               组合自由形状（跨类组合左右岛）
"""

import numpy as np
from shapely.affinity import scale as _scale
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from ..configs import frozen
from .record import GeometryRecord

W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2.0   # 单岛槽宽
H2 = frozen.FOOTPRINT_UM[1] / 2.0                               # 半高
G2 = frozen.JUNCTION_GAP_UM / 2.0                               # 半间隙
PROTECT = frozen.JUNCTION_PROTECT_HALF_UM
NECK_HALF_Y = PROTECT + 5.0
NECK_LEN = 40.0
GENERATOR_VERSION = 'library-v4-bounded-semiaxes-junction-neck'

CATEGORY_NAMES = (
    "rect_rounded", "cross", "teeth_interdig", "ellipse_arc",
    "trapezoid_asym", "notch_taper", "irregular", "combo",
)


def _mirror(poly: Polygon) -> Polygon:
    """右槽坐标 → 左槽坐标（x 取反）。"""
    return _scale(poly, xfact=-1.0, origin=(0, 0))


def _largest(geom) -> Polygon:
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area).buffer(0)
    return geom.buffer(0)


def _snap_protected_ellipse_edge(poly: Polygon) -> Polygon:
    """Remove sub-0.5 μm nicks where an ellipse meets the junction neck."""
    coords = []
    for x, y in poly.exterior.coords:
        if abs(x - G2) < 0.6 and abs(y) <= NECK_HALF_Y:
            x = G2
        point = (x, y)
        if not coords or point != coords[-1]:
            coords.append(point)
    if coords[-1] != coords[0]:
        coords.append(coords[0])
    return Polygon(coords)


def _finish(body: Polygon) -> Polygon:
    """右槽坐标：裁剪到槽位 → 并结区颈 → 清洗为简单多边形。"""
    from .clean import clean_polygon
    # Outside the fixed junction region, fingers may enter the centre gap.
    # Previously clipping everything at x=G2 silently removed all such fingers.
    slot = unary_union([box(G2, -H2, G2 + W2, H2),
                        box(frozen.MIN_GAP_UM / 2, -H2, G2, -NECK_HALF_Y),
                        box(frozen.MIN_GAP_UM / 2, NECK_HALF_Y, G2, H2)])
    body = _largest(body).intersection(slot)
    neck = box(G2, -NECK_HALF_Y, G2 + NECK_LEN, NECK_HALF_Y)
    return clean_polygon(_largest(body.union(neck)))


# ── 1. Pocket 矩形及圆角矩形 ──────────────────────────────────
def gen_rect_rounded(rng) -> tuple:
    w = W2 * rng.uniform(0.7, 1.0)
    h = H2 * rng.uniform(0.6, 1.0)
    r = rng.uniform(0.0, min(60.0, max(h - 40.0, 0.0), w * 0.4))
    body = box(G2, -h, G2 + w, h)
    if r > 1.0:
        body = body.buffer(-r).buffer(r)
    return body, body


# ── 2. Cross/Xmon 臂 ─────────────────────────────────────────
def gen_cross(rng) -> tuple:
    aw = rng.uniform(40.0, 120.0)            # 臂宽（≥2×保护带）
    vh = H2 * rng.uniform(0.6, 1.0)          # 竖臂半高
    hl = rng.uniform(60.0, W2 - 10.0)        # 横臂长（不含 taper 段）
    taper = rng.uniform(0.0, min(30.0, W2 - hl))
    tw = aw * rng.uniform(0.5, 1.0)          # 收窄段端宽
    centre = G2 + hl / 2
    poly = unary_union([box(G2, -aw / 2, G2 + hl, aw / 2),
                        box(max(G2, centre - aw / 2), -vh,
                            min(G2 + W2, centre + aw / 2), vh)])
    if taper >= 1.0:
        x0 = G2 + hl
        tip = Polygon([
            (x0, -aw / 2), (x0 + taper, -tw / 2),
            (x0 + taper, tw / 2), (x0, aw / 2),
        ])
        poly = unary_union([poly, tip])
    return poly, poly


# ── 3. Teeth/梳齿和交指 ──────────────────────────────────────
def gen_teeth_interdig(rng) -> tuple:
    n = int(rng.integers(2, 6))
    bh = H2 * rng.uniform(0.8, 1.0)
    bw = W2 * rng.uniform(0.6, 1.0)
    y_max = bh - 50.0
    spacing = 2 * y_max / max(n - 1, 1) if n > 1 else 2 * y_max
    tw = min(rng.uniform(15.0, 40.0), spacing - 5.0) if n > 1 else rng.uniform(15.0, 40.0)
    tw = max(tw, 6.0)
    p = rng.uniform(6.0, (frozen.JUNCTION_GAP_UM - frozen.MIN_GAP_UM) / 2 - 0.5)
    base = box(G2, -bh, G2 + bw, bh)
    ys = np.linspace(-y_max, y_max, n)
    teeth = [box(G2 - p, y - tw / 2, G2 + p, y + tw / 2) for y in ys
             if abs(y) - tw / 2 >= NECK_HALF_Y]
    left = unary_union([base] + teeth)          # 左岛：齿向间隙伸出
    dy = spacing / 2
    ys_r = np.array([y + dy for y in ys if abs(y + dy) < bh - 35.0])
    teeth_r = [box(G2 - p, y - tw / 2, G2 + p, y + tw / 2) for y in ys_r
               if abs(y) - tw / 2 >= NECK_HALF_Y]
    right = unary_union([base] + teeth_r)       # 右岛：齿反向、垂直交错
    return left, right


# ── 4. 椭圆、圆弧和多弧边界 ─────────────────────────────────
def gen_ellipse_arc(rng) -> tuple:
    n = rng.uniform(2.0, 4.0)                # 超椭圆指数
    # ``a`` is a semiaxis; reserve room for both sides inside the single-island slot.
    a = W2 * 0.5 * rng.uniform(0.55, 0.95)
    b = H2 * rng.uniform(0.55, 1.0)
    cx = G2 + a
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    rad = ((np.abs(np.cos(th)) / a) ** n + (np.abs(np.sin(th)) / b) ** n) ** (-1.0 / n)
    pts = [(cx + rad[i] * np.cos(th[i]), rad[i] * np.sin(th[i])) for i in range(96)]
    poly = Polygon(pts)
    if rng.random() < 0.5:
        return poly, poly
    a2 = W2 * 0.5 * rng.uniform(0.55, 0.95)
    b2 = H2 * rng.uniform(0.55, 1.0)
    n2 = rng.uniform(2.0, 4.0)
    rad2 = ((np.abs(np.cos(th)) / a2) ** n2 + (np.abs(np.sin(th)) / b2) ** n2) ** (-1.0 / n2)
    pts2 = [(G2 + a2 + rad2[i] * np.cos(th[i]), rad2[i] * np.sin(th[i]))
            for i in range(96)]
    return poly, Polygon(pts2)


# ── 5. 梯形、偏置和非对称双岛 ─────────────────────────────────
def _trap(h_in, h_out, w, dy):
    return Polygon([
        (G2, -h_in / 2 + dy), (G2 + w, -h_out / 2 + dy),
        (G2 + w, h_out / 2 + dy), (G2, h_in / 2 + dy),
    ])


def gen_trapezoid_asym(rng) -> tuple:
    left = _trap(rng.uniform(80.0, 2 * H2), rng.uniform(80.0, 2 * H2),
                 W2 * rng.uniform(0.6, 1.0), rng.uniform(-20.0, 20.0))
    right = _trap(rng.uniform(80.0, 2 * H2), rng.uniform(80.0, 2 * H2),
                  W2 * rng.uniform(0.6, 1.0), rng.uniform(-20.0, 20.0))
    return left, right


# ── 6. 缺口、凸台或 taper 边界 ───────────────────────────────
def gen_notch_taper(rng) -> tuple:
    w = W2 * rng.uniform(0.8, 1.0)
    h = H2 * rng.uniform(0.7, 1.0)
    r = rng.uniform(15.0, 45.0)
    body = box(G2, -h, G2 + w, h).buffer(-r).buffer(r)
    # 外缘缺口（1–2 个）
    for _ in range(int(rng.integers(1, 3))):
        ny = rng.uniform(-h + 35.0, h - 35.0)
        nw, nh = rng.uniform(25.0, 70.0), rng.uniform(15.0, 40.0)
        body = body.difference(
            box(G2 + w - nw, ny - nh / 2, G2 + w + 5.0, ny + nh / 2))
    # 顶/底凸台（向外）
    if rng.random() < 0.6:
        top = rng.random() < 0.5
        bx = rng.uniform(G2 + 25.0, G2 + w - 70.0)
        if top:
            body = body.union(box(bx, h - 10.0, bx + 60.0, h + rng.uniform(20.0, 45.0)))
        else:
            body = body.union(box(bx, -h - rng.uniform(20.0, 45.0), bx + 60.0, -h + 10.0))
    # 结区 taper 展宽
    fy = NECK_HALF_Y + rng.uniform(10.0, 40.0)
    body = body.union(box(G2, -fy, G2 + 25.0, fy))
    right = _largest(body)
    return right, right


# ── 7. 多尺度不规则多边形 ────────────────────────────────────
def gen_irregular(rng) -> tuple:
    kharm = int(rng.integers(2, 6))
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)

    def one():
        # Avoid producing a large ellipse and silently clipping half of it.
        a = W2 * 0.5 * rng.uniform(0.5, 0.9)
        b = H2 * rng.uniform(0.5, 0.95)
        cx = G2 + a
        amps = rng.uniform(0.0, 0.07, kharm)
        phis = rng.uniform(0, 2 * np.pi, kharm)
        mod = 1.0 + sum(amps[k] * np.sin((k + 2) * th + phis[k]) for k in range(kharm))
        return Polygon([(cx + mod[i] * np.cos(th[i]) * a, mod[i] * np.sin(th[i]) * b)
                        for i in range(128)])

    left = one()
    right = one() if rng.random() >= 0.5 else left
    return left, right


# ── 8. 组合自由形状 ─────────────────────────────────────────
_BASE_GENS = {
    "rect": gen_rect_rounded, "cross": gen_cross,
    "ellipse": gen_ellipse_arc, "trap": gen_trapezoid_asym,
}


def gen_combo(rng) -> tuple:
    names = list(_BASE_GENS)
    rng.shuffle(names)
    _, br = _BASE_GENS[names[0]](rng)
    cl, _ = _BASE_GENS[names[1]](rng)
    return cl, br


# ── 汇总入口 ────────────────────────────────────────────────
_GENERATORS = {
    "rect_rounded": gen_rect_rounded,
    "cross": gen_cross,
    "teeth_interdig": gen_teeth_interdig,
    "ellipse_arc": gen_ellipse_arc,
    "trapezoid_asym": gen_trapezoid_asym,
    "notch_taper": gen_notch_taper,
    "irregular": gen_irregular,
    "combo": gen_combo,
}


def make_record(category: str, index: int, master_seed: int = 20261021) -> GeometryRecord:
    """确定性生成一个候选：seed = f(master, category, index)。"""
    cat_hash = sum(ord(c) * (i + 1) for i, c in enumerate(category)) % (2**20)
    seed = (master_seed * 1_000_003 + cat_hash * 9_77 + index * 31_337) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    body_left, body_right = _GENERATORS[category](rng)
    pad_left, pad_right = _finish(body_left), _finish(body_right)
    if category == "ellipse_arc":
        # clean_polygon can coalesce the short ellipse-to-neck seam and erase
        # the exact junction edge. Reattach the protected landing after
        # cleanup so the 30 μm junction contract remains exact.
        neck = box(G2, -NECK_HALF_Y, G2 + 20.0, NECK_HALF_Y)
        pad_right = _snap_protected_ellipse_edge(_largest(pad_right.union(neck)))
        pad_left = pad_right
    return GeometryRecord(
        family="two_pad_free",
        category=category,
        category_index=index,
        seed=int(seed),
        pad_left=_mirror(pad_left),
        pad_right=pad_right,
        junction=LineString([(-G2, 0.0), (G2, 0.0)]),
    )


def rect_record(w_um: float, h_um: float, r_um: float, note: str = "") -> GeometryRecord:
    """标定用矩形岛记录（G1 方向性检验：同尺寸仅圆角不同）。"""
    body = _rounded_box(G2, -h_um / 2, G2 + w_um, h_um / 2, r_um)
    return GeometryRecord(
        family="two_pad_free",
        category="rect_calib",
        category_index=0,
        seed=0,
        pad_left=_mirror(_finish(body)),
        pad_right=_finish(body),
        junction=LineString([(-G2, 0.0), (G2, 0.0)]),
        note=note,
    )


def _rounded_box(x0, y0, x1, y1, r):
    if r <= 1.0:
        return box(x0, y0, x1, y1)
    r = min(r, (y1 - y0) / 2 - 10.0, (x1 - x0) / 2 - 2.0)
    return box(x0, y0, x1, y1).buffer(-r).buffer(r)


def make_library(n_per_class: int, master_seed: int = 20261021) -> list:
    """全形状库：8 类 × n_per_class（主实验 128/类 → 1024）。"""
    return [make_record(cat, i, master_seed)
            for cat in CATEGORY_NAMES for i in range(n_per_class)]
