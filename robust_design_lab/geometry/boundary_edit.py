# -*- coding: utf-8 -*-
"""区域边界编辑：把固定结构函数推广为有明确约束的局部编辑算子。

设计（docs/plans/new_structures_next_steps_2026-11-02.md 第三步）：
- 动作 = (op, seg, mag, width)，作用在**当前结构**的单个岛上；
  seg 编码 (岛, 环, 区域)：每岛外环 8 段 + 内环 8 段（无孔岛的内环
  段无效）——L外0-7 / L内8-15 / R外16-23 / R内24-31；
- 三类机制不同的算子：bulge（局部法向外凸/内凹，改曲率）、
  notch（边界开槽，增加工作边）、tab（外伸舌，增加对置工作边）；
  notch=金属减少（difference）、tab=金属增加（union），内外环同规则；
- 结区保护：段中点距结线任一端点 < PROTECT_R_UM 的编辑一律无效；
- 编辑后自动检查（check_structure + 面积变化限幅），失败抛
  InvalidEditError——检查规则对所有方法相同，失败由协议层记账；
- 所有方法（VLM 有图/无图、坐标、BO、CMA）共享同一动作空间与
  检查规则；数值法经 [0,1] 编码访问（op/seg 为整数维吸附）。

全部几何 μm（shapely），与 novel_structures 同单位。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from shapely.geometry import Polygon

from robust_design_lab.geometry.novel_structures import check_structure

# ── 动作空间常量 ──
EDIT_OPS = ("bulge", "notch", "tab")
N_REGION_PER_RING = 8          # 每环分段数
N_SEG = 32                     # L外0-7 L内8-15 R外16-23 R内24-31
SEG_STEPS = 200                # 边界细分点数（编辑分辨率）
PROTECT_R_UM = 50.0            # 结线端点保护半径
MAX_AREA_CHANGE = 0.30         # 单次编辑面积相对变化上限
BULGE_AMP_UM = 40.0            # bulge 最大法向位移
NOTCH_R_UM = 30.0              # notch 最大半径
TAB_L_UM = 80.0                # tab 最大长度
TAB_W_UM = 30.0                # tab 最大宽度
MAG_RANGE = (0.2, 1.0)
WIDTH_RANGE = (0.3, 1.0)


class InvalidEditError(ValueError):
    """无效编辑动作（结区保护/结构检查失败/段不存在/面积突变）。

    no_solve 标记：该失败不产生物理求解，协议层据此类失败单独计数
    （预算照耗——无效提案是方法自身的探索损失，与求解失败同价）。"""

    no_solve = True


# ── 边界工具 ──

def _signed_area(coords) -> float:
    n = len(coords)
    return sum(coords[i][0] * coords[(i + 1) % n][1]
               - coords[(i + 1) % n][0] * coords[i][1]
               for i in range(n)) / 2.0


def _densified_ccw(ring) -> List[Tuple]:
    """环按容限细分并统一为逆时针方向（外法线恒为 (dy,-dx)）。"""
    dens = ring.segmentize(ring.length / SEG_STEPS)
    pts = [(float(x), float(y)) for x, y in dens.coords][:-1]
    return pts if _signed_area(pts) > 0 else list(reversed(pts))


def _ring_segments(pts: List[Tuple],
                   n: int = N_REGION_PER_RING) -> List[dict]:
    """细分后的环坐标按弧长等分为 n 段（点数等分近似）。"""
    m = len(pts)
    per = m // n
    if per < 3:
        raise InvalidEditError("边界细分点不足")
    segs = []
    for i in range(n):
        lo = i * per
        hi = (i + 1) * per if i < n - 1 else m
        idx = list(range(lo, hi))
        segs.append(dict(idx=idx, mid=pts[(lo + hi - 1) // 2]))
    return segs


def _outer_normal(pts: List[Tuple], i: int) -> Tuple[float, float]:
    """i 点处逆时针环的外法线。"""
    a, b = pts[(i - 1) % len(pts)], pts[(i + 1) % len(pts)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dy / n, -dx / n


def _mid_normal(pts: List[Tuple], seg: dict) -> Tuple[float, float]:
    return _outer_normal(pts, seg["idx"][len(seg["idx"]) // 2])


def _tab_rect(seg_mid: Tuple, normal: Tuple, mag: float,
              width: float) -> Polygon:
    """以段中点为舌根、沿外法线外伸的矩形（舌根略嵌边界防裂缝）。"""
    L, W = mag * TAB_L_UM, width * TAB_W_UM
    mx, my = seg_mid
    nx, ny = normal
    tx, ty = -ny, nx
    root = 6.0
    return Polygon([
        (mx - nx * root + tx * W / 2, my - ny * root + ty * W / 2),
        (mx - nx * root - tx * W / 2, my - ny * root - ty * W / 2),
        (mx + nx * (L - root) - tx * W / 2, my + ny * (L - root) - ty * W / 2),
        (mx + nx * (L - root) + tx * W / 2, my + ny * (L - root) + ty * W / 2)])


def _notch_circle(seg_mid: Tuple, normal: Tuple, mag: float) -> Polygon:
    r = mag * NOTCH_R_UM
    cx = seg_mid[0] + normal[0] * r * 0.2
    cy = seg_mid[1] + normal[1] * r * 0.2
    return Polygon([(cx + r * math.cos(2 * math.pi * i / 24),
                     cy + r * math.sin(2 * math.pi * i / 24))
                    for i in range(24)])


# ── seg 编解码 ──

def seg_decode(seg: int) -> Tuple[str, str, int]:
    if not (0 <= int(seg) < N_SEG):
        raise InvalidEditError(f"seg={seg} 越界 [0,{N_SEG})")
    island = "L" if seg < 16 else "R"
    ring = "outer" if seg % 16 < 8 else "inner"
    return island, ring, seg % 8


def seg_encode(island: str, ring: str, region: int) -> int:
    return (0 if island == "L" else 16) + (0 if ring == "outer" else 8) \
        + int(region)


def _target_poly(struct: Dict, island: str) -> Polygon:
    return struct["pad_left"] if island == "L" else struct["pad_right"]


# ── 编辑主入口 ──

def junction_protection_zone(struct: Dict):
    """结区保护范围：两结端点各 PROTECT_R_UM 的圆之并。"""
    from shapely.ops import unary_union
    j = struct["junction"]
    return unary_union([Polygon([
        (p[0] + PROTECT_R_UM * math.cos(2 * math.pi * i / 24),
         p[1] + PROTECT_R_UM * math.sin(2 * math.pi * i / 24))
        for i in range(24)]) for p in (j.coords[0], j.coords[-1])])


def apply_edit(struct: Dict, op: str, seg: int, mag: float, width: float,
               label: str = "", check_etched: bool = True) -> Dict:
    """对当前结构施加一个编辑动作，返回新结构（不修改原结构）。

    无效动作（结区保护、无孔岛内环段、结构检查失败、面积突变）抛
    InvalidEditError——检查规则对所有方法相同。

    结区保护（2026-11-04 审阅修正）：作用于**整个几何变化范围**——
    编辑前后岛形的对称差与结端点保护圆相交即拒，不再只查段中点
    （旧检查允许高斯窗/舌/槽的延伸侵入保护圆，反例 8 条）。
    check_etched=True 时对编辑后结构再施加 ±δ 刻蚀变体分别检查
    （名义与扰动后都须通过结构检查）。"""
    if op not in EDIT_OPS:
        raise InvalidEditError(f"未知算子 {op}")
    island, ring, region = seg_decode(seg)
    poly = _target_poly(struct, island)
    if poly.geom_type != "Polygon":
        raise InvalidEditError("目标岛不是单一多边形")
    if ring == "inner" and not poly.interiors:
        raise InvalidEditError(f"{island} 岛无内环，inner 段无效")

    ring_line = poly.exterior if ring == "outer" else poly.interiors[0]
    pts = _densified_ccw(ring_line)
    s = _ring_segments(pts)[region]

    if op == "bulge":
        amp = mag * BULGE_AMP_UM
        sigma = max(width * len(s["idx"]) / 4.0, 1.0)
        mid_i = s["idx"][len(s["idx"]) // 2]
        new_pts = list(pts)
        for i in s["idx"]:
            d = min(abs(i - mid_i), len(pts) - abs(i - mid_i))
            w = math.exp(-((d / sigma) ** 2))
            nx, ny = _outer_normal(pts, i)
            new_pts[i] = (pts[i][0] + nx * amp * w, pts[i][1] + ny * amp * w)
        if ring == "outer":
            new_poly = Polygon(new_pts,
                               [list(r.coords) for r in poly.interiors])
        else:
            # 内环编辑：按段中点距离匹配目标内环并整环替换
            best, best_d = 0, math.inf
            for k, intr in enumerate(poly.interiors):
                im = intr.interpolate(0.5, normalized=True)
                d = math.dist(s["mid"], (im.x, im.y))
                if d < best_d:
                    best, best_d = k, d
            holes = [list(r.coords) for r in poly.interiors]
            holes[best] = Polygon(new_pts).exterior.coords
            new_poly = Polygon(list(poly.exterior.coords), holes)
    else:
        nrm = _mid_normal(pts, s)
        if op == "notch":     # 开槽：金属减少（外环=外缘缺口，内环=孔扩大）
            new_poly = poly.difference(_notch_circle(s["mid"], nrm, mag))
        else:                 # tab：金属增加（外环=外伸舌，内环=伸入孔）
            new_poly = poly.union(_tab_rect(s["mid"], nrm, mag, width))

    new_poly = new_poly.buffer(0)
    if new_poly.geom_type != "Polygon" or new_poly.is_empty:
        raise InvalidEditError("编辑后岛碎裂/为空")
    if abs(new_poly.area - poly.area) / poly.area > MAX_AREA_CHANGE:
        raise InvalidEditError(
            f"面积变化 {abs(new_poly.area - poly.area) / poly.area:.0%}"
            f" 超过 {MAX_AREA_CHANGE:.0%} 上限")

    # 结区保护：几何变化范围（对称差）与保护圆不相交
    changed = new_poly.symmetric_difference(poly)
    if not changed.is_empty and changed.intersects(junction_protection_zone(
            struct)):
        raise InvalidEditError(
            "编辑的几何变化范围侵入结端点保护圆（全支撑检查）")

    out = dict(struct)
    out["pad_left" if island == "L" else "pad_right"] = new_poly
    out["label"] = (struct.get("label", "")
                    + (label or f"+{op}{seg}m{mag:.1f}w{width:.1f}"))
    errs = check_structure(out)
    if errs:
        raise InvalidEditError("结构检查失败: " + "; ".join(errs[:3]))
    if check_etched:  # 工艺变体（±δ）也须通过结构检查
        from robust_design_lab.geometry.novel_structures import etch_variant
        for d in (0.5, -0.5):
            ev = etch_variant(out, d)
            e_errs = check_structure(ev)
            if e_errs:
                raise InvalidEditError(
                    f"刻蚀{+d:+.1f}μm 变体检查失败: "
                    + "; ".join(e_errs[:2]))
    return out


# ── 协议接口 ──

def action_kw(op: str, island: str, ring: str, region: int, mag: float,
              width: float) -> Dict:
    """模型/算法动作 → 协议 kw（离散维编码为整数，连续维截断到范围）。"""
    return dict(op=EDIT_OPS.index(op),
                seg=seg_encode(island, ring, region),
                mag=float(min(max(float(mag), MAG_RANGE[0]), MAG_RANGE[1])),
                width=float(min(max(float(width), WIDTH_RANGE[0]),
                                WIDTH_RANGE[1])))


def action_from_kw(kw: Dict) -> Tuple[str, int, float, float]:
    return (EDIT_OPS[int(kw["op"])], int(kw["seg"]),
            float(kw["mag"]), float(kw["width"]))


def action_describe(kw: Dict) -> str:
    op, seg, mag, width = action_from_kw(kw)
    island, ring, region = seg_decode(seg)
    return (f"{op} {island}岛{ring}环段{region} "
            f"(mag={mag:.2f} width={width:.2f})")


def segment_table(struct: Dict) -> List[str]:
    """各段位置描述（文字臂与提示共用；坐标为段中点 μm）。"""
    rows = []
    for seg in range(N_SEG):
        island, ring, region = seg_decode(seg)
        poly = _target_poly(struct, island)
        if ring == "inner" and not poly.interiors:
            rows.append(f"seg{seg}: {island}岛内环段{region}（该岛无内环，无效）")
            continue
        ring_line = poly.exterior if ring == "outer" else poly.interiors[0]
        s = _ring_segments(_densified_ccw(ring_line))[region]
        rows.append(f"seg{seg}: {island}岛{ring}环段{region} "
                    f"中点({s['mid'][0]:.0f},{s['mid'][1]:.0f})μm")
    return rows
