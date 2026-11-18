# -*- coding: utf-8 -*-
"""跨族新型电容结构：与原五族有本质拓扑区别的结构族。

原五族回顾：
- A 双岛：两矩形 pad 对置（平板式侧向电容）
- B 十字/星形：单岛多臂（臂-地电容）
- C 同心：环形内外圈（环形间隙）
- D 交指：pad 间平行手指梳（并联手指对）

新结构族（拓扑上的本质区别，不是参数变体）：
1. hoya_perforated 镂空岛：pad 内部开圆孔阵列，**复连通区域**——刻蚀
   漂移作用于低场内区孔缘，被大面积稀释；文献锚点 Hoya 表面损耗抑制
   （Mine et al., PRX Applied 2023）。
2. floating_bridge 漂浮岛桥：结上两岛照常由结线相连，两岛**上方**架一
   排漂浮金属岛，形成 pad1—岛—岛—pad2 的**串联电容桥**（导体数 >2）。
   单个小间隙的刻蚀误差被串联组合稀释；C_Σ 公式的 Σ_k 串联项天然支持
   （岛网名 island*，不会被误认成 pad）。
3. koch 科赫雪花岛：**多尺度分形边界**，边缘长度跨尺度分布，δ=0.5μm
   在不同尺度上的相对扰动不同。

（圆弧螺旋/蜿蜒结构留待下轮：直螺旋对置易跨中轴相撞，需专门构造。）

刻蚀建模：与原五族的"参数平移"不同，这里用 shapely 各向同性偏移
（buffer ±δ）直接模拟金属收缩/膨胀——孔自动扩大/缩小、间隙自动变宽/
变窄，物理上更接近真实工艺。

全部几何以 μm 为单位（shapely 坐标），落 Metal 时 ×1e-3 转 mm。
"""

from __future__ import annotations

import math
from typing import Dict, List

from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box

MIN_FEATURE_UM = 5.0  # 最小特征尺寸（孔径/间隙/线宽）

# 名义地口袋边距：电极/岛外接框到地边界的名义余量。
# 同时是结区的工艺保护范围下界——口袋内全部金属与结线距地边界
# 至少该余量（刻蚀扰动下按 δ 递减，见 etch_variant）。
POCKET_MARGIN_UM = 150.0


def nominal_pocket(struct: Dict, margin_um: float = POCKET_MARGIN_UM):
    """名义地口袋矩形：全部金属（电极+漂浮岛）**名义**外接框各边外扩
    margin，保留实际中心坐标。

    口袋是独立几何对象：工艺扰动时地边界从该名义矩形独立偏移
    （etch_variant 对它 buffer），不从扰动后的电极外接框反推——
    否则电极收缩与边距扩大会互相抵消，地边界实际不动
    （2026-11-02 审阅反例，见 docs/reviews/ 同名报告 §1）。
    """
    allg = [struct["pad_left"], struct["pad_right"]] + list(struct["islands"])
    b = [g.bounds for g in allg]
    x0, y0 = min(v[0] for v in b), min(v[1] for v in b)
    x1, y1 = max(v[2] for v in b), max(v[3] for v in b)
    return box(x0 - margin_um, y0 - margin_um, x1 + margin_um, y1 + margin_um)


# ── 结构生成器：返回 dict(pad_left, pad_right, islands, junction, label) ──

def gen_hoya(pad_w=350.0, pad_h=300.0, gap=60.0, n_holes_x=3, n_holes_y=2,
             hole_r=35.0) -> Dict:
    """镂空双岛：矩形 pad + 内部圆孔阵列（复连通）。"""
    def pad(cx):
        outer = Polygon([
            (cx - pad_w / 2, -pad_h / 2), (cx + pad_w / 2, -pad_h / 2),
            (cx + pad_w / 2, pad_h / 2), (cx - pad_w / 2, pad_h / 2)])
        holes = []
        for i in range(n_holes_x):
            for j in range(n_holes_y):
                hx = cx - pad_w / 2 + (i + 1) * pad_w / (n_holes_x + 1)
                hy = -pad_h / 2 + (j + 1) * pad_h / (n_holes_y + 1)
                holes.append(_circle(hx, hy, hole_r))
        return Polygon(outer.exterior, [h.exterior for h in holes])

    pl = pad(-gap / 2 - pad_w / 2)
    pr = pad(+gap / 2 + pad_w / 2)
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-gap / 2, 0), (gap / 2, 0)]),
                label=f"hoya({pad_w:.0f}x{pad_h:.0f},gap{gap:.0f},"
                      f"{n_holes_x}x{n_holes_y}孔r{hole_r:.0f})")


def gen_floating_bridge(pad_w=350.0, pad_h=250.0, gap=60.0, n_isl=3,
                        isl_w=80.0, isl_h=60.0, bridge_gap=25.0) -> Dict:
    """漂浮岛桥：两岛由结线相连（结跨 gap），岛上方一排漂浮岛桥接。

    漂浮岛行位于 y = pad_h/2 + bridge_gap（岛中心），沿 x 从左 pad 顶面
    跨到右 pad 顶面；岛间等距。pad1—岛—…—岛—pad2 构成串联电容桥。
    """
    pl = Polygon([(-gap / 2 - pad_w, -pad_h / 2), (-gap / 2, -pad_h / 2),
                  (-gap / 2, pad_h / 2), (-gap / 2 - pad_w, pad_h / 2)])
    pr = Polygon([(gap / 2, -pad_h / 2), (gap / 2 + pad_w, -pad_h / 2),
                  (gap / 2 + pad_w, pad_h / 2), (gap / 2, pad_h / 2)])
    y0 = pad_h / 2 + bridge_gap + isl_h / 2
    # 岛行横跨两 pad 的 x 投影，含中间 gap 区
    x_start, x_end = -gap / 2 - pad_w, gap / 2 + pad_w
    span = x_end - x_start
    step = isl_w + (span - n_isl * isl_w) / (n_isl - 1) if n_isl > 1 else span
    islands = []
    for i in range(n_isl):
        cx = x_start + isl_w / 2 + i * step
        islands.append(Polygon([
            (cx - isl_w / 2, y0 - isl_h / 2), (cx + isl_w / 2, y0 - isl_h / 2),
            (cx + isl_w / 2, y0 + isl_h / 2), (cx - isl_w / 2, y0 + isl_h / 2)]))
    return dict(pad_left=pl, pad_right=pr, islands=islands,
                junction=LineString([(-gap / 2, 0), (gap / 2, 0)]),
                label=f"fbridge({n_isl}岛,isl{isl_w:.0f}x{isl_h:.0f},"
                      f"bg{bridge_gap:.0f})")


def gen_koch(order=2, r=200.0, gap=60.0) -> Dict:
    """科赫雪花双岛：多尺度分形边界。

    雪花由正三角形迭代而来（尖朝外）；三角形的内缘跨 y=0 处在
    cx ∓ r/√3，故中心取 ±(gap/2 + r/√3) 使内缘贴住结区，结线
    端点按 y=0 与岛边界的实际交点落位（分形边界在 y=0 可能多点
    穿越时取最靠中轴的一段）。
    """
    def snowflake(center_x, o, radius):
        pts = [(center_x + radius * math.cos(a), radius * math.sin(a))
               for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                         math.pi / 2 + 4 * math.pi / 3)]
        poly = Polygon(pts)
        for _ in range(o):
            new_pts = []
            old = list(poly.exterior.coords)
            n = len(old) - 1
            for i in range(n):
                a, b = old[i], old[(i + 1) % n]
                p1 = (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3)
                p2 = (a[0] + 2 * (b[0] - a[0]) / 3, a[1] + 2 * (b[1] - a[1]) / 3)
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                # 逆时针环的外法线是右法线 (dy,-dx)：凸起朝外才是真雪花
                apex = (mx + math.sqrt(3) / 2 * dy, my - math.sqrt(3) / 2 * dx)
                new_pts += [a, p1, apex, p2]
            poly = Polygon(new_pts)
        return poly

    off = gap / 2 + r / math.sqrt(3)
    pl = snowflake(-off, order, r)
    pr = snowflake(+off, order, r)
    # 分形凸起可能越过基准三角形的内缘：迭代外推直到两岛分离出名义间隙
    while pl.intersects(pr) or pl.distance(pr) < gap * 0.5:
        pl = translate(pl, -1.0, 0)
        pr = translate(pr, +1.0, 0)
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=_junction_on_axis(pl, pr),
                label=f"koch(阶{order},r{r:.0f},gap{gap:.0f})")


def _junction_on_axis(pl: Polygon, pr: Polygon) -> LineString:
    """取 y=0 与两岛交线中最靠近中轴的两个端点作为结线。"""
    axis = LineString([(-1e6, 0.0), (1e6, 0.0)])
    xl = max(pl.intersection(axis).bounds[0::2])   # 左岛右缘
    xr = min(pr.intersection(axis).bounds[0::2])   # 右岛左缘
    return LineString([(xl, 0.0), (xr, 0.0)])


def _circle(cx, cy, r, n=48):
    return Polygon([(cx + r * math.cos(2 * math.pi * i / n),
                     cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)])


NOVEL_FAMILIES = {
    "hoya_perforated": gen_hoya,
    "floating_bridge": gen_floating_bridge,
    "koch": gen_koch,
}


# ── 刻蚀：各向同性偏移（+δ=金属收缩，孔扩大、间隙变宽） ──

def etch_variant(struct: Dict, delta_um: float) -> Dict:
    """各向同性刻蚀：金属收缩 δ（电极/岛 buffer −δ，孔扩大、间隙变宽）。

    地平面口袋从**名义口袋独立偏移**：buffer(+δ, mitre) 使口袋四边各
    退 δ（地边界同步蚀刻、口袋变宽）。旧版只把 pocket_margin +δ，但
    struct_to_options 又用扰动后电极外接框重算口袋，电极收缩把边距
    扩大完全抵消，地边界实际不动——按 2026-11-02 审阅反例修正。"""
    out = dict(struct)
    for k in ("pad_left", "pad_right"):
        out[k] = struct[k].buffer(-delta_um, quad_segs=8)
    out["islands"] = [i.buffer(-delta_um, quad_segs=8)
                      for i in struct["islands"]]
    pocket = struct.get("pocket") or nominal_pocket(
        struct, struct.get("pocket_margin", POCKET_MARGIN_UM))
    out["pocket"] = pocket.buffer(delta_um, join_style=2)  # mitre 保矩形直角
    return out


# ── 几何检查 ──

def check_structure(struct: Dict) -> List[str]:
    errs = []
    pl, pr = struct["pad_left"], struct["pad_right"]
    for name, p in (("pad_left", pl), ("pad_right", pr)):
        if p.geom_type == "MultiPolygon":
            errs.append(f"{name} 碎裂为多块（刻蚀后金属断开）")
        elif p.is_empty or p.area <= 0:
            errs.append(f"{name} 为空")
    if pl.intersects(pr):
        errs.append("两岛相交/接触")
    elif pl.distance(pr) < MIN_FEATURE_UM:
        errs.append(f"岛间距 {pl.distance(pr):.1f}μm 过小")
    for i, isl in enumerate(struct["islands"]):
        if isl.is_empty or isl.area <= 0:
            errs.append(f"island{i+1} 为空")
            continue
        for other, nme in ((pl, "pad_left"), (pr, "pad_right")):
            if isl.intersects(other):
                errs.append(f"island{i+1} 与 {nme} 相交")
            elif isl.distance(other) < MIN_FEATURE_UM:
                errs.append(f"island{i+1} 与 {nme} 间距过小")
        for j in range(i + 1, len(struct["islands"])):
            if isl.intersects(struct["islands"][j]):
                errs.append(f"island{i+1} 与 island{j+1} 相交")
    j = struct["junction"]
    jbuf = LineString(j).buffer(2.0)
    if not (pl.intersects(jbuf) and pr.intersects(jbuf)):
        errs.append("结线未连接两岛")
    if not struct["islands"]:
        pass
    else:
        if any(isl.intersects(jbuf) for isl in struct["islands"]):
            errs.append("漂浮岛与结线过近")
    for name in ("pad_left", "pad_right"):
        p = struct[name]
        if p.geom_type != "Polygon":
            continue  # MultiPolygon 已在上面报错
        for hole in p.interiors:
            if Polygon(hole).area < math.pi * (MIN_FEATURE_UM / 2) ** 2:
                errs.append(f"{name} 孔过小")
                break
    return errs


def all_structures() -> Dict[str, Dict]:
    return {k: g() for k, g in NOVEL_FAMILIES.items()}


# ══════════════════════════════════════════════════════════════
# 优质电容候选形状（第二轮，理论先行）：十个候选
#
# 设计公理（本轮强基线+跨族实测归纳 + 经典场论）：
# P1 匀曲率：曲率处处有界且尽量均匀，杜绝尖角电荷聚集（κ→∞ 处
#    表面场发散、损耗与刻蚀敏感性最高）；
# P2 结区实体质量：结两侧必须有足够 pad 实体提供结电容与分流电容，
#    稀疏结构（如漂浮方岛、分形尖角）已被实测证伪；
# P3 工作边原理（本项目实测发现）：让电容由"被刻蚀的边界本身"承载
#    （交指侧壁模式，双板对称刻蚀部分抵消），而非平行板面积——后者
#    的刻蚀漂移全部暴露；
# P4 面积稀释：C_Σ 越大 δf/f = δC/(2C) 越小，pad 目标面积 ~1e5 μm²
#    （实测对应 C_Σ≈80 fF 量级）。
# ══════════════════════════════════════════════════════════════

def _round_convex(poly: Polygon, r: float) -> Polygon:
    """形态学闭运算圆化凸角（buffer(+r) 后 buffer(−r)）。"""
    return poly.buffer(r, quad_segs=16).buffer(-r, quad_segs=16)


def _capsule(p0, p1, w) -> Polygon:
    return LineString([p0, p1]).buffer(w / 2, quad_segs=12)


def _superellipse(w, h, n=4.0, n_pts=120) -> Polygon:
    """超椭圆 |2x/w|^n + |2y/h|^n = 1（n=4 即 squircle）。"""
    import numpy as np
    t = np.linspace(0, 2 * math.pi, n_pts)
    x = w / 2 * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / n)
    y = h / 2 * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / n)
    return Polygon(zip(x, y))


def _annular_sector(r_in, r_out, a0_deg, a1_deg) -> Polygon:
    """环形扇段（角度制，含径向端面）。"""
    n_arc = 48
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    pts = []
    for i in range(n_arc + 1):
        a = a0 + (a1 - a0) * i / n_arc
        pts.append((r_out * math.cos(a), r_out * math.sin(a)))
    for i in range(n_arc + 1):
        a = a1 - (a1 - a0) * i / n_arc
        pts.append((r_in * math.cos(a), r_in * math.sin(a)))
    return Polygon(pts)


def _bend_arc(poly: Polygon, R: float, seg=10.0) -> Polygon:
    """把 x 轴卷成半径 R 的圆弧：θ=x/R，ρ=R+y → (ρ·sinθ, R−ρ·cosθ)。

    卷弧前先按 seg 细分边界——只映射顶点会把长直边变成毁形的弦
    （Δθ 大时偏差可达几十μm）。
    """
    dens = poly.segmentize(seg)
    pts = [(p[0], p[1]) for p in dens.exterior.coords]

    def bend(p):
        th, rho = p[0] / R, R + p[1]
        return (rho * math.sin(th), R - rho * math.cos(th))
    return Polygon([bend(p) for p in pts])


# ── A. 匀曲率类（P1 主导）──────────────────────────────────

def q_circle(r=265.0, gap=28.0) -> Dict:
    """双圆盘：曲率处处 =1/r 完全均匀，无任何电荷聚集点。"""
    pl = _circle(-gap / 2 - r, 0, r)
    pr = _circle(+gap / 2 + r, 0, r)
    return _mk(pl, pr, gap, f"q_circle(r{r:.0f},gap{gap:.0f})")


def q_ellipse(a=340.0, b=200.0, gap=28.0) -> Dict:
    """椭圆对（长轴对置）：正曲率均匀分段，面对长度大于同面积圆盘。"""
    from shapely.affinity import scale, translate
    base = _circle(0, 0, 1.0, 96)
    e = scale(base, a, b)
    pl = translate(e, -gap / 2 - a, 0)
    pr = translate(e, +gap / 2 + a, 0)
    return _mk(pl, pr, gap, f"q_ellipse({a:.0f}x{2*b:.0f},gap{gap:.0f})")


def q_stadium(L=620.0, R=140.0, gap=28.0) -> Dict:
    """胶囊对（长轴垂直结向）：直线面对段提供强结区电容，上下端半圆
    消角；边界切向连续（曲率 0 与 1/R 分段拼接，无跳跃角点）。"""
    from shapely.geometry import box
    body = _round_convex(box(-L / 2, -R, L / 2, R), R)  # 圆角矩形≈胶囊
    pl = translate(body, -gap / 2 - L / 2, 0)
    pr = translate(body, +gap / 2 + L / 2, 0)
    return _mk(pl, pr, gap, f"q_stadium({L:.0f}x{2*R:.0f},gap{gap:.0f})")


def q_squircle(w=580.0, h=400.0, n=4.0, gap=28.0) -> Dict:
    """超椭圆对（n=4）：矩形与圆之间的最优折中——面积效率接近矩形，
    曲率有界且无角点。"""
    base = _superellipse(w, h, n)
    pl = translate(base, -gap / 2 - w / 2, 0)
    pr = translate(base, +gap / 2 + w / 2, 0)
    return _mk(pl, pr, gap, f"q_squircle({w:.0f}x{h:.0f},n{n:.0f})")


def q_rounded_hex(r=275.0, fillet=95.0, gap=30.0) -> Dict:
    """圆角六边形：面积/外接跨度比优于圆盘，全凸边界，角由闭运算圆化。

    顶点取 30°+60°i：面对方向（±x）为平边，位于 x=±r·√3/2，故
    中心偏移取 gap/2 + r·√3/2 使平边贴住结区。
    """
    pts = [(r * math.cos(a), r * math.sin(a))
           for a in (math.pi / 6 + i * math.pi / 3 for i in range(6))]
    hexa = _round_convex(Polygon(pts), fillet)
    off = gap / 2 + r * math.sqrt(3) / 2
    pl = translate(hexa, -off, 0)
    pr = translate(hexa, +off, 0)
    return _mk(pl, pr, gap, f"q_rounded_hex(r{r:.0f},f{fillet:.0f})")


# ── B. 工作边类（P3 主导：电容由被刻蚀边界承载）───────────

def q_ellipse_idc(a=200.0, b=140.0, D=140.0, n_f=4, f_w=16.0,
                  f_gap=32.0, f_len=88.0, jg=40.0) -> Dict:
    """椭圆垫交指：椭圆垫身（P1 去直角）+ 交指工作边（P3，本项目实测
    最强机制——电容来自被刻蚀的指侧壁，双板对称刻蚀部分抵消）。

    布局（CorrectIDC 式）：左右椭圆垫内缘相距 D，手指沿 x 从各垫伸出、
    沿 y 交错（Δy=(f_w+f_gap)/2 > f_w 保证不接触）；手指尖互越中线但
    不触及对垫本体（f_len < D/2+对垫内缘）；结区在手指区下方由两垫
    的短唇收口到 jg。
    """
    from shapely.affinity import scale
    body = scale(_circle(0, 0, 1.0, 96), a, b)
    span = n_f * f_w + (n_f - 1) * f_gap
    fis_l, fis_r = [], []
    for i in range(n_f):
        y = -span / 2 + f_w / 2 + i * (f_w + f_gap)
        rd = _idc_root_depth(a, b, abs(y))
        fis_l.append(_capsule((-D / 2 - rd, y), (-D / 2 + f_len, y), f_w))
        yr = y + (f_w + f_gap) / 2
        rdr = _idc_root_depth(a, b, abs(yr))
        fis_r.append(_capsule((+D / 2 + rdr, yr), (+D / 2 - f_len, yr), f_w))
    # 结区唇（手指区下方；根深自适应）
    y_lip = -(span / 2 + 45)
    rdl = _idc_root_depth(a, b, abs(y_lip), margin=20)
    lip_l = _capsule((-D / 2 - rdl, y_lip), (-jg / 2, y_lip), 30)
    lip_r = _capsule((+D / 2 + rdl, y_lip), (+jg / 2, y_lip), 30)
    pl = shapely_union([translate(body, -D / 2 - a, 0)] + fis_l + [lip_l])
    pr = shapely_union([translate(body, +D / 2 + a, 0)] + fis_r + [lip_r])
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-jg / 2, y_lip), (jg / 2, y_lip)]),
                label=f"q_ellipse_idc({a:.0f}x{2*b:.0f},{n_f}指,D{D:.0f})")


def shapely_union(polys):
    from shapely.ops import unary_union
    return unary_union(polys)


def q_crescent(R=230.0, gap=40.0, t=105.0, span_deg=220.0) -> Dict:
    """等间隙曲面对：一条等宽弯曲缝隙的两个光滑侧壁就是全部工作边——
    内外壁同步刻蚀、间隙变化被两侧对称抵消；无平行板段、无角点。"""
    a_half = span_deg / 2
    inner = _annular_sector(R - gap / 2 - t, R - gap / 2,
                            90 - a_half, 90 + a_half)
    outer = _annular_sector(R + gap / 2, R + gap / 2 + t,
                            90 - a_half, 90 + a_half)
    # 端部圆化（径向端面是平边）
    inner, outer = _round_convex(inner, 20), _round_convex(outer, 20)
    from shapely.ops import nearest_points
    p1, p2 = nearest_points(inner, outer)
    return dict(pad_left=inner, pad_right=outer, islands=[],
                junction=LineString([p1, p2]),
                label=f"q_crescent(R{R:.0f},g{gap:.0f},{span_deg:.0f}°)")


def q_arc_idc(R=280.0, D=170.0, t=45.0, n_f=5, f_w=16.0,
              f_gap=48.0, f_len=140.0, jg=28.0) -> Dict:
    """同心弧交指：交指工作边沿圆弧排布——继承交指的对称刻蚀抵消，
    并消除直线交指的端部效应；紧凑（环形闭合场）。

    平直构型（再卷弧）：内外两条弧带梁（径向相距 D）+ 径向交错指
    （手指尖停在间隙带内，不触及对梁）+ 角向端部收口结区；卷弧后
    梁成为同心弧、手指成为径向指。指间隙按内半径处的弧长压缩
    （因子 ρ_min/R）放大取值。
    """
    from shapely.geometry import box
    span = n_f * f_w + (n_f - 1) * f_gap
    x0, x1 = -span / 2 - 15, span / 2 + 40
    bar_o = box(x0, +D / 2, x1 + 20, +D / 2 + t)      # 外梁
    bar_i = box(x0, -D / 2 - t, x1 + 20, -D / 2)      # 内梁
    fis_o, fis_i = [], []
    for i in range(n_f):
        x = x0 + f_w / 2 + i * (f_w + f_gap)
        fis_o.append(_capsule((x, D / 2 + 8), (x, D / 2 + 8 - f_len), f_w))
        xb = x + (f_w + f_gap) / 2
        fis_i.append(_capsule((xb, -D / 2 - 8), (xb, -D / 2 - 8 + f_len), f_w))
    lip_o = box(x1 - 10, +jg / 2, x1 + 20, +D / 2 + t)   # 端部收口
    lip_i = box(x1 - 10, -D / 2 - t, x1 + 20, -jg / 2)
    pad_o = shapely_union([bar_o, lip_o] + fis_o)
    pad_i = shapely_union([bar_i, lip_i] + fis_i)
    # 卷弧后 buffer(0) 清理数值自交
    pa = _bend_arc(pad_o, R).buffer(0)
    pb = _bend_arc(pad_i, R).buffer(0)

    def bend_pt(x, y, RR):
        th, rho = x / RR, RR + y
        return (rho * math.sin(th), RR - rho * math.cos(th))
    xj = x1 + 5
    j1, j2 = bend_pt(xj, -jg / 2, R), bend_pt(xj, +jg / 2, R)
    return dict(pad_left=pa, pad_right=pb, islands=[],
                junction=LineString([j1, j2]),
                label=f"q_arc_idc(R{R:.0f},{n_f}指,D{D:.0f})")


# ── C. 混合类（多机制叠加）─────────────────────────────────

def q_disk_arcs(r=245.0, gap=30.0, Rf=460.0, t_arc=55.0,
                bridge_gap=20.0, n_isl=3) -> Dict:
    """圆盘浮弧桥：漂浮岛桥的圆化版——方岛换成环形弧段（无角点），
    串联稀释保留、电荷聚集消除；弧段同时增加结区上方电容。

    弧段角域 [145°, 35°]：端点已核算不与圆盘相交（内缘角点距盘心
    ≥ r+10μm）。
    """
    pl = _circle(-gap / 2 - r, 0, r)
    pr = _circle(+gap / 2 + r, 0, r)
    a0, a1 = 145.0, 35.0
    seg_deg = ((a0 - a1) - (n_isl - 1) * math.degrees(bridge_gap / Rf)) / n_isl
    islands = []
    a = a0
    for i in range(n_isl):
        a2 = a - seg_deg
        s = _annular_sector(Rf - t_arc / 2, Rf + t_arc / 2, a2, a)
        islands.append(_round_convex(s, 12))
        a = a2 - math.degrees(bridge_gap / Rf)
    return dict(pad_left=pl, pad_right=pr, islands=islands,
                junction=LineString([(-gap / 2, 0), (gap / 2, 0)]),
                label=f"q_disk_arcs(r{r:.0f},Rf{Rf:.0f},{n_isl}弧)")


def q_annulus(r_out=250.0, r_in=115.0, gap=28.0) -> Dict:
    """环形垫对：两个完整圆环并置——外缘匀曲率，内缘是额外工作边
    （环内缘电容），单位跨度面积效率高。"""
    from shapely.geometry import Point
    def ring(cx):
        return Point(cx, 0).buffer(r_out, 64).difference(
            Point(cx, 0).buffer(r_in, 48))
    pl, pr = ring(-gap / 2 - r_out), ring(+gap / 2 + r_out)
    return _mk(pl, pr, gap, f"q_annulus({r_out:.0f}/{r_in:.0f},gap{gap:.0f})")


def _mk(pl, pr, gap, label) -> Dict:
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-gap / 2, 0), (gap / 2, 0)]),
                label=label)


QUALITY_FAMILIES = {
    "q_circle": q_circle,
    "q_ellipse": q_ellipse,
    "q_stadium": q_stadium,
    "q_squircle": q_squircle,
    "q_rounded_hex": q_rounded_hex,
    "q_ellipse_idc": q_ellipse_idc,
    "q_crescent": q_crescent,
    "q_arc_idc": q_arc_idc,
    "q_disk_arcs": q_disk_arcs,
    "q_annulus": q_annulus,
}

# 各候选的物理依据（审阅用）
RATIONALES = {
    "q_circle": "双圆盘：曲率处处均匀，无任何电荷聚集点（匀曲率基准形）",
    "q_ellipse": "椭圆对：正曲率均匀，长轴对置增大结区面对长度",
    "q_stadium": "胶囊对：直线面对段供结区电容，两端半圆消角，切向连续",
    "q_squircle": "超椭圆：面积效率近矩形且曲率有界无角点",
    "q_rounded_hex": "圆角六边形：面积/跨度比优于圆，全凸边界",
    "q_ellipse_idc": "椭圆垫交指：圆化垫身 + 交指工作边（实测最强机制）",
    "q_crescent": "等间隙曲面对：全部边界都是工作边，双侧同步刻蚀抵消",
    "q_arc_idc": "同心弧交指：交指工作边环形化，消除直线端部效应",
    "q_disk_arcs": "圆盘浮弧桥：漂浮桥圆化版——串联稀释且无角点",
    "q_annulus": "环形垫对：外缘匀曲率，内缘为额外工作边",
}


# ── Metal 组件与选项转换 ──────────────────────────────────────

def struct_to_options(struct: Dict, um_to_mm=1e-3,
                      pocket_margin_um=None) -> Dict:
    """结构（μm shapely）→ NovelTwoPadTransmon options（毫米顶点表）。

    含内孔：pad 顶点表为 [外环顶点, 孔1顶点, ...]；岛为单环顶点表。
    地口袋：优先取 struct['pocket']（刻蚀变体携带的独立偏移口袋）；
    名义结构则按 nominal_pocket（名义外接框 + margin）生成，并输出
    口袋中心 pocket_cx/pocket_cy——口袋锚定名义几何，与扰动后电极
    外接框解耦（2026-11-02 修复同步刻蚀被抵消的问题）。
    （2026-11-01 修复内孔丢失；2026-11-02 补地平面环境。）
    """
    def pts(poly):
        rings = [[[x * um_to_mm, y * um_to_mm]
                  for x, y in poly.exterior.coords]]
        for hole in poly.interiors:
            rings.append([[x * um_to_mm, y * um_to_mm]
                          for x, y in hole.coords])
        return rings
    pocket = struct.get("pocket")
    if pocket is None:
        if pocket_margin_um is None:
            pocket_margin_um = struct.get('pocket_margin', POCKET_MARGIN_UM)
        pocket = nominal_pocket(struct, pocket_margin_um)
    minx, miny, maxx, maxy = pocket.bounds
    options = dict(
        pad_left=pts(struct["pad_left"]),
        pad_right=pts(struct["pad_right"]),
        islands=[pts(i) for i in struct["islands"]],
        junction=[[x * um_to_mm, y * um_to_mm]
                  for x, y in struct["junction"].coords],
        jj_width="20um",
        pocket_w=maxx - minx, pocket_h=maxy - miny,
        pocket_cx=(minx + maxx) / 2, pocket_cy=(miny + maxy) / 2,
    )
    if "readout" in struct:
        options["readout"] = pts(struct["readout"])
        options["readout_cutout"] = pts(struct["readout_cutout"])
    return options


def _novel_component():
    """延迟导入并返回 NovelTwoPadTransmon 组件类（定义在此避免顶层
    依赖 qiskit_metal——本模块其余部分纯 shapely，可独立测试）。"""
    from qiskit_metal import Dict as _Dict
    from qiskit_metal.qlibrary.core import BaseQubit

    class NovelTwoPadTransmon(BaseQubit):
        """两岛任意多边形（含内孔）+ 任意漂浮岛 + 结线 + 可选地口袋。

        pad 顶点表 = [外环, 孔1, ...]；岛命名 island1..N → Q3D 独立网。
        pocket_w/pocket_h（μm）非零时添加 subtract 矩形 rect_pk——渲染器
        据此渲染整片 ground_main_plane 并挖出口袋，与官方 TransmonPocket
        的 Q3D 环境同构（2026-11-02 环境统一修复）。
        """

        default_options = _Dict(
            pad_left=[], pad_right=[], islands=[], junction=[],
            jj_width="20um", pocket_w=0.0, pocket_h=0.0,
            pocket_cx=0.0, pocket_cy=0.0,
            readout=[], readout_cutout=[],
        )
        component_metadata = _Dict(
            short_name="Q",
            _qgeometry_table_poly="True",
            _qgeometry_table_junction="True",
        )
        TOOLTIP = "Two-island arbitrary-polygon transmon with holes, floating islands and ground pocket."

        @staticmethod
        def _to_poly(rings):
            rings = [rings] if rings and isinstance(rings[0][0], (int, float)) \
                else rings
            shell = [tuple(p) for p in rings[0]]
            holes = [[tuple(p) for p in r] for r in rings[1:]]
            return Polygon(shell, holes)

        def make(self):
            polys = dict(
                pad_left=self._to_poly(self.p.pad_left),
                pad_right=self._to_poly(self.p.pad_right),
            )
            for i, isl in enumerate(self.p.islands):
                polys[f"island{i+1}"] = self._to_poly(isl)
            if self.p.readout:
                polys["readout"] = self._to_poly(self.p.readout)
                self.add_qgeometry(
                    "poly", dict(readout_cutout=self._to_poly(self.p.readout_cutout)),
                    subtract=True, chip=self.p.chip)
            pw, ph = float(self.p.pocket_w), float(self.p.pocket_h)
            try:  # metal options Dict 不支持 .get()；旧 options 无该键时回退 0
                pcx = float(self.p.pocket_cx)
            except (TypeError, ValueError):
                pcx = 0.0
            try:
                pcy = float(self.p.pocket_cy)
            except (TypeError, ValueError):
                pcy = 0.0
            if pw > 0 and ph > 0:
                from shapely.geometry import box as _box
                pk = _box((pcx - pw / 2) * 1e-3, (pcy - ph / 2) * 1e-3,
                          (pcx + pw / 2) * 1e-3, (pcy + ph / 2) * 1e-3)
                self.add_qgeometry("poly", dict(rect_pk=pk), subtract=True,
                                   chip=self.p.chip)
            jj = LineString([tuple(p) for p in self.p.junction])
            self.add_qgeometry("poly", polys, chip=self.p.chip)
            self.add_qgeometry("junction", dict(jj=jj),
                               width=self.p.jj_width, chip=self.p.chip)

    return NovelTwoPadTransmon


# 供 build_design_for 的模块级查找使用
def __getattr__(name):
    if name == "NovelTwoPadTransmon":
        return _novel_component()
    raise AttributeError(name)



def _idc_root_depth(a: float, b: float, y_abs: float, margin: float = 14.0) -> float:
    """交指根深：椭圆垫在高度 |y| 处的内缘回缩量 + 余量。

    回缩 = a·(1−√(1−(y/b)²))；根深不足会让指/唇脱锚（多轮实测教训），
    一律自适应计算而不是拍固定值。
    """
    y = min(abs(y_abs), b * 0.95)
    return a * (1 - math.sqrt(max(0.0, 1 - (y / b) ** 2))) + margin


# ══════════════════════════════════════════════════════════════
# 主线新结构（2026-11-01 用户裁决）：两电极、单结、无浮置导体。
# 机制先行的三类代表设计，与已验证基准 q_ellipse_idc 同口径。
# ══════════════════════════════════════════════════════════════

def m_graded_idc(a=190.0, b=138.0, D=150.0, n_f=5, f_w=16.0, f_gap=32.0,
                 f_len_center=100.0, f_len_edge=62.0, jg=40.0) -> Dict:
    """渐变交指：指长沿梳齿方向二次渐变（中心长、两端短），指间隙
    同步渐宽——电容与场能量沿梳齿平滑分配，消除等长交指的端部
    突变（端指对是刻蚀敏感度最不均匀的位置）。"""
    from shapely.affinity import scale
    body = scale(_circle(0, 0, 1.0, 96), a, b)
    step = f_w + f_gap
    span = n_f * f_w + (n_f - 1) * f_gap
    fis_l, fis_r = [], []
    for i in range(n_f):
        frac = (i - (n_f - 1) / 2) / max((n_f - 1) / 2, 1)  # -1..1
        f_len = f_len_center + (f_len_edge - f_len_center) * frac ** 2
        y = -span / 2 + f_w / 2 + i * step
        rd = _idc_root_depth(a, b, abs(y))
        fis_l.append(_capsule((-D / 2 - rd, y), (-D / 2 + f_len, y), f_w))
        yr = y + step / 2
        rdr = _idc_root_depth(a, b, abs(yr))
        fis_r.append(_capsule((+D / 2 + rdr, yr), (+D / 2 - f_len, yr), f_w))
    y_lip = -(span / 2 + 45)
    rdl = _idc_root_depth(a, b, abs(y_lip), margin=20)
    lip_l = _capsule((-D / 2 - rdl, y_lip), (-jg / 2, y_lip), 30)
    lip_r = _capsule((+D / 2 + rdl, y_lip), (+jg / 2, y_lip), 30)
    pl = shapely_union([translate(body, -D / 2 - a, 0)] + fis_l + [lip_l])
    pr = shapely_union([translate(body, +D / 2 + a, 0)] + fis_r + [lip_r])
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-jg / 2, y_lip), (jg / 2, y_lip)]),
                label=f"m_graded_idc({n_f}指,L{f_len_center:.0f}到{f_len_edge:.0f})")


def m_branch_comb(a=195.0, b=140.0, D=150.0, n_f=4, f_w=16.0, f_gap=48.0,
                  f_len=85.0, paddle_len=18.0, paddle_w=18.0, jg=40.0) -> Dict:
    """分支梳：每根指端带桨状二级分支（沿指轴加宽段），全部金属连回
    各自电极——桨肩（横向缘）与相邻对指侧壁构成额外工作边对，单位
    占位的交错边缘密度高于直指交指；桨形避开相邻指车道（y 向横臂
    会与对指相交，已实测否定）。"""
    from shapely.affinity import scale
    body = scale(_circle(0, 0, 1.0, 96), a, b)
    step = f_w + f_gap
    span = n_f * f_w + (n_f - 1) * f_gap
    fis_l, fis_r = [], []
    for i in range(n_f):
        y = -span / 2 + f_w / 2 + i * step
        rd = _idc_root_depth(a, b, abs(y))
        stem = _capsule((-D / 2 - rd, y), (-D / 2 + f_len, y), f_w)
        paddle = _capsule((-D / 2 + f_len - paddle_len / 2, y),
                          (-D / 2 + f_len + paddle_len / 2, y), paddle_w)
        fis_l.append(shapely_union([stem, paddle]))
        yr = y + step / 2
        rdr = _idc_root_depth(a, b, abs(yr))
        stem_r = _capsule((+D / 2 + rdr, yr), (+D / 2 - f_len, yr), f_w)
        paddle_r = _capsule((+D / 2 - f_len - paddle_len / 2, yr),
                            (+D / 2 - f_len + paddle_len / 2, yr), paddle_w)
        fis_r.append(shapely_union([stem_r, paddle_r]))
    y_lip = -(span / 2 + 45)
    rdl = _idc_root_depth(a, b, abs(y_lip), margin=20)
    lip_l = _capsule((-D / 2 - rdl, y_lip), (-jg / 2, y_lip), 30)
    lip_r = _capsule((+D / 2 + rdl, y_lip), (+jg / 2, y_lip), 30)
    pl = shapely_union([translate(body, -D / 2 - a, 0)] + fis_l + [lip_l])
    pr = shapely_union([translate(body, +D / 2 + a, 0)] + fis_r + [lip_r])
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-jg / 2, y_lip), (jg / 2, y_lip)]),
                label=f"m_branch_comb({n_f}指桨{paddle_w:.0f})")


def m_variable_gap(H=480.0, depth=290.0, g0=26.0, g_edge=100.0, n=80) -> Dict:
    """平滑变间隙双岛：对置间隙 g(y) = g0 + (g_edge−g0)·(y/(H/2))²，
    结区窄、两端平滑放宽——电容沿边界按间隙分配，等间隙结构中
    所有长度同处同一敏感度，变间隙使敏感度加权分散。"""
    def facing_pts(sign):
        pts = []
        for i in range(n + 1):
            t = i / n
            y = -H / 2 + H * t
            g = g0 + (g_edge - g0) * (2 * y / H) ** 2
            pts.append((sign * (depth + g / 2), y))
            # 依次内边界（x = sign*(depth+g/2)）
        return pts
    # 左岛：内边界 x = -(depth+g/2)，外边界 x = -g/2 同曲线平移
    inner_l = [(-(depth + _g(y) / 2 if False else 0), y) for y in []]  # placeholder
    ys = [-H / 2 + H * i / n for i in range(n + 1)]

    def g_at(y):
        return g0 + (g_edge - g0) * (2 * y / H) ** 2
    left_pts = [(-(g_at(y) / 2), y) for y in ys]
    left_outer = [(-(depth + g_at(y) / 2), y) for y in reversed(ys)]
    pl = _round_convex(Polygon(left_pts + left_outer), 20)
    right_pts = [((g_at(y) / 2), y) for y in reversed(ys)]
    right_outer = [((depth + g_at(y) / 2), y) for y in ys]
    pr = _round_convex(Polygon(right_pts + right_outer), 20)
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-g0 / 2, 0), (g0 / 2, 0)]),
                label=f"m_variable_gap(g{g0:.0f}→{g_edge:.0f},H{H:.0f})")


MAINLINE_FAMILIES = {
    "m_graded_idc": m_graded_idc,
    "m_branch_comb": m_branch_comb,
    "m_variable_gap": m_variable_gap,
}
