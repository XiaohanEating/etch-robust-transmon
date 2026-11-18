# -*- coding: utf-8 -*-
"""空间反馈：从几何直接可核查的空间信息（计划第二步·几何层）。

三个输出共用同一计算，保证 VLM 有图/无图与数值方法获得**同信息**：
- segment_metrics(struct)：逐段（复用 boundary_edit 分段）局部间隙、
  到地边界距离、平均离散曲率、对置段索引；
- spatial_table(struct)：人可读表（VLM 文字臂与有图臂共用）；
- render_spatial(struct, path)：空间叠加图（岛形+对置间隙热力+
  名义与刻蚀边界叠加+段号，标注用白底半透明框）。

物理层（表面电荷/电场）不在本模块——需先核查 Q3D 工程场量可导出性
（见 experiments/spatial_probe.py 的检查项），只有电容矩阵时不得
声称已有空间敏感度。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from robust_design_lab.geometry import boundary_edit as be
from robust_design_lab.geometry.novel_structures import (
    nominal_pocket, etch_variant)

N_SAMPLES = 160  # 每环采样点数（局部间隙计算分辨率）


def _sample_ring(ring, n: int = N_SAMPLES) -> np.ndarray:
    dens = ring.segmentize(ring.length / n)
    return np.array([(p[0], p[1]) for p in dens.coords][:-1])


def _pair_min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 每点到 b 的最近距离（a×b 距离矩阵，n≤200 时开销可忽略）。"""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1), d.argmin(axis=1)


def local_gap_profile(struct: Dict):
    """两电极间的局部间隙：左右岛外环逐采样点到对岛（外环+内环）的
    最近距离，及最近点所在位置。返回 (left, right) 各 (pts, gap,
    nearest_idx)。"""
    rings = {}
    for isl, poly in (("L", struct["pad_left"]), ("R", struct["pad_right"])):
        rings[isl] = [_sample_ring(poly.exterior)] + [
            _sample_ring(h) for h in poly.interiors]
    out = {}
    for a, b in (("L", "R"), ("R", "L")):
        pts = rings[a][0]
        other = np.vstack(rings[b])
        gap, near = _pair_min_dist(pts, other)
        out[a] = dict(pts=pts, gap=gap, nearest=near)
    return out["L"], out["R"]


def _discrete_curvature(pts: np.ndarray) -> np.ndarray:
    """闭合折线逐点离散曲率 κ≈2sin(θ/2)/弦长（三点外接圆近似）。"""
    n = len(pts)
    k = np.zeros(n)
    for i in range(n):
        p0, p1, p2 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        a = np.hypot(*(p1 - p0))
        b = np.hypot(*(p2 - p1))
        c = np.hypot(*(p2 - p0))
        if min(a, b) < 1e-9:
            continue
        # 外接圆半径 R = abc/(4A)；κ = 1/R
        area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1])
                    - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        if area2 < 1e-12:
            continue
        k[i] = 4 * area2 / (a * b * c)
    return k


def segment_metrics(struct: Dict) -> List[Dict]:
    """逐段空间摘要（32 段，与 boundary_edit 编码一致）。

    每段：gap_min（段内采样点对对岛最近距离的最小值）、
    gap_to_ground（段中点到名义口袋边界的距离）、curvature_mean
    （段内离散曲率均值）、facing_seg（间隙最近的对置段索引）。"""
    lg, rg = local_gap_profile(struct)
    pocket = nominal_pocket(struct)
    pocket_line = LineString(list(pocket.exterior.coords))
    rows = []
    for seg in range(be.N_SEG):
        island, ring, region = be.seg_decode(seg)
        poly = struct["pad_left"] if island == "L" else struct["pad_right"]
        if ring == "inner" and not poly.interiors:
            rows.append(dict(seg=seg, valid=False, island=island,
                             ring=ring, region=region))
            continue
        ring_line = poly.exterior if ring == "outer" else poly.interiors[0]
        pts = be._densified_ccw(ring_line)
        s = be._ring_segments(pts)[region]
        seg_pts = np.array([pts[i] for i in s["idx"]])
        prof = lg if island == "L" else rg
        # 段内采样点（按外环均匀采样近似对齐：取段中点最近邻采样窗）
        d = np.hypot(prof["pts"][:, 0] - s["mid"][0],
                     prof["pts"][:, 1] - s["mid"][1])
        win = d <= (be._ring_segments(prof["pts"])[region]["idx"]
                    .__len__() * prof["pts"].shape[0] / be.N_SEG / 2.5)
        if win.sum() < 2:
            i0 = int(d.argmin())
            win = np.zeros(len(d), bool)
            win[max(0, i0 - 3):i0 + 4] = True
        gap_min = float(prof["gap"][win].min())
        # 对置段：最近采样点在对岛外环的弧长位置 → 段索引
        j = int(prof["nearest"][win][int(np.argmin(prof["gap"][win]))])
        other = rg if island == "L" else lg
        m = be._ring_segments(other["pts"])
        arc = j / len(other["pts"]) * len(m)
        facing = m[int(min(arc, len(m) - 1))]["idx"][0]
        facing_seg = (0 if island == "L" else 16) + min(facing // 25, 7)
        curv = float(_discrete_curvature(seg_pts).mean())
        g2g = float(Point(s["mid"]).distance(pocket_line))
        rows.append(dict(seg=seg, valid=True, island=island, ring=ring,
                         region=region, gap_min=round(gap_min, 1),
                         gap_to_ground=round(g2g, 1),
                         curvature_mean=round(curv, 5),
                         facing_seg=facing_seg))
    return rows


def spatial_table(struct: Dict) -> List[str]:
    """人可读逐段空间表（有图/无图 VLM 与数值法共用同格式——
    计划要求所有方法共享同一空间信息）。"""
    rows = segment_metrics(struct)
    out = []
    for r in rows:
        if not r.get("valid"):
            out.append(f"seg{r['seg']}: {r['island']}岛内环（无内环，无效）")
            continue
        out.append(
            "seg{seg}: {i}岛{r}环段{reg} 局部间隙={g}μm 到地边界={gg}μm "
            "曲率={c:.3f}/μm 对置≈seg{f}".format(
                seg=r["seg"], i=r["island"], r=r["ring"], reg=r["region"],
                g=r["gap_min"], gg=r["gap_to_ground"], c=r["curvature_mean"],
                f=r["facing_seg"]))
    return out


def etch_overlay(struct: Dict, delta: float = 0.5):
    """名义与刻蚀边界的叠加（返回每岛 (名义线, 刻蚀线) 列表，渲染用）。"""
    lines = []
    ev = etch_variant(struct, delta)
    for k in ("pad_left", "pad_right"):
        lines.append((LineString(list(struct[k].exterior.coords)
                                 + [struct[k].exterior.coords[0]]),
                      LineString(list(ev[k].exterior.coords)
                                 + [ev[k].exterior.coords[0]])))
    return lines


def render_spatial(struct: Dict, path, title: str = ""):
    """空间叠加图：岛形+局部间隙色带+刻蚀边界叠加+段号
    （标注一律白底半透明框，防遮盖——项目绘图原则）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lg, rg = local_gap_profile(struct)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for isl, poly, prof in (("L", struct["pad_left"], lg),
                            ("R", struct["pad_right"], rg)):
        ax.fill(*poly.exterior.xy, color="#2196F3", alpha=0.55,
                edgecolor="#0D47A1", linewidth=0.8, zorder=3)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="white", zorder=4)
        # 局部间隙色带（外环采样点，颜色=间隙值）
        sc = ax.scatter(prof["pts"][:, 0], prof["pts"][:, 1],
                        c=prof["gap"], cmap="RdYlGn", s=4, zorder=6,
                        vmin=20, vmax=120)
        # 段号（白底半透明框）
        pts = be._densified_ccw(poly.exterior)
        for region, seg in enumerate(be._ring_segments(pts)):
            mx, my = seg["mid"]
            ax.annotate(f"{isl}o{region}", (mx, my), fontsize=6, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))
        for k, hole in enumerate(poly.interiors):
            hpts = be._densified_ccw(hole)
            for region, seg in enumerate(be._ring_segments(hpts)):
                mx, my = seg["mid"]
                ax.annotate(f"{isl}i{region}", (mx, my), fontsize=6,
                            zorder=7, bbox=dict(boxstyle="round,pad=0.15",
                                                fc="white", ec="none",
                                                alpha=0.75))
    # 名义与刻蚀边界叠加（红=名义，橙虚线=+0.5μm 刻蚀）
    for nom, et in etch_overlay(struct, 0.5):
        xs, ys = nom.xy
        ax.plot(xs, ys, color="#B0BEC5", linewidth=0.5, zorder=5)
        xe, ye = et.xy
        ax.plot(xe, ye, color="#FF7043", linewidth=0.5, ls="--", zorder=5)
    x, y = struct["junction"].xy
    ax.plot(x, y, color="#F44336", linewidth=2.5, zorder=8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("local gap (um)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def simple_rule_action(struct: Dict) -> Dict:
    """最简物理规则基线（计划要求保留）：选对置间隙最窄的有效外环段，
    给一个中等 bulge（扩宽最窄间隙）。返回动作 kw。

    用于检验"简单规则是否已经足够"——与 VLM/数值提案同预算比较。"""
    rows = [r for r in segment_metrics(struct)
            if r.get("valid") and r["ring"] == "outer"]
    best = min(rows, key=lambda r: r["gap_min"])
    return be.action_kw("bulge", best["island"], "outer", best["region"],
                        0.6, 0.5)
