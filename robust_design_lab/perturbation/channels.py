# -*- coding: utf-8 -*-
"""扰动通道（方案 v2.0 §5.1）。

场景 = {etch_um, lj_rel, ti_rel}：
- etch_um ≠ 0：两岛整体 buffer(±δ)（μm，正=外扩），需重解场；
- lj_rel ≠ 1：结电感 ×(1+lj_rel)，只改 LOM 后处理；
- ti_rel ≠ 1：界面层厚度 ×(1+ti_rel)，只改参与比后处理（p_i 线性于 t_i）。
刻蚀后必须重过几何检查；失败场景如实记录（方案 §5.2）。
"""

from shapely.geometry import LineString

from ..configs import frozen
from ..geometry import checks
from ..geometry.clean import clean_polygon


def _facing_x(poly, side: int):
    """y=0 射线与岛边界的交点中最靠近间隙的 x（side=+1 右岛取最小 x，−1 左岛取最大 x）。"""
    line = LineString([(side * -1000.0, 0.0), (side * 1000.0, 0.0)])
    inter = poly.exterior.intersection(line)
    xs = [p.x for p in (inter.geoms if inter.geom_type == "MultiPoint" else [inter])]
    return min(xs) if side > 0 else max(xs)


def apply_etch(rec, delta_um: float):
    """全局刻蚀偏移：两岛 buffer(δ, mitre)，结线端点跟随新的面向边。

    mitre 拐角：微小偏移下不产生圆弧顶点（round join 的弧顶点间距
    ~0.05–0.2 μm，违反顶点间距规则且破坏保护带直边）。
    """
    if abs(delta_um) < 1e-9:
        return rec, None
    bl = clean_polygon(rec.pad_left.buffer(delta_um, join_style=2, mitre_limit=2.0))
    br = clean_polygon(rec.pad_right.buffer(delta_um, join_style=2, mitre_limit=2.0))
    if bl.geom_type != "Polygon" or br.geom_type != "Polygon":
        return None, "刻蚀后岛碎裂"
    # 结电极物理上随金属边移动：端点取新面向边与 y=0 的交点
    x_l = _facing_x(bl, -1)
    x_r = _facing_x(br, +1)
    jj = LineString([(x_l, 0.0), (x_r, 0.0)])
    new = rec.with_operation(f"etch{delta_um:+.3f}um", bl, br, jj)
    ok, reasons = checks.check_record(new)
    if not ok:
        return None, "; ".join(reasons)
    return new, None


def apply_scene(rec, scene: dict):
    """场景变换 → (几何记录, lj_rel, ti_rel, 失败原因)。

    返回的是无量纲缩放因子；绝对 Lj 由调用方结合该候选的名义 Lj 得到
    （按候选配定协议）。几何不变时返回原记录。
    """
    new_rec, err = apply_etch(rec, scene.get("etch_um", 0.0))
    if err:
        return None, None, None, err
    return new_rec, scene.get("lj_rel", 1.0), scene.get("ti_rel", 1.0), None
