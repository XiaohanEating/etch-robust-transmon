# -*- coding: utf-8 -*-
"""形状库可视化：8 类 × 每类 2 个代表（带地面 + pocket + 连接垫）。

按 figures4papers 规范：图注不标实验编号只列含义；标注文字用半透明白底框。
每行一类，左列 index=0，右列 index=1，便于对比类内变异。
颜色：地面=深灰、pocket=浅灰背景、pad=蓝、连接垫=橙、结=红点。
输出 results/shape_library_visual_2026-10-26/shape_library.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.geometry.free_pocket import make_pocket_record
from robust_design_lab.geometry.library import CATEGORY_NAMES
from robust_design_lab.configs import frozen

OUT = ROOT / "results" / "shape_library_visual_2026-10-26"
OUT.mkdir(parents=True, exist_ok=True)

CAT_LABELS = {
    "rect_rounded": "矩形/圆角",
    "cross": "十字/Xmon",
    "teeth_interdig": "梳齿/交指",
    "ellipse_arc": "椭圆/多弧",
    "trapezoid_asym": "梯形/非对称",
    "notch_taper": "缺口/凸台",
    "irregular": "不规则",
    "combo": "组合",
}


def draw_one(ax, opts, title):
    from shapely.geometry import Polygon as Poly
    import shapely.affinity as aff

    pl = Poly(opts["pad_left"])
    pr = Poly(opts["pad_right"])

    # pocket（外接框 + 50μm margin）
    bounds = pl.union(pr).bounds
    m = 0.05  # 50 μm
    pocket = Poly([
        (bounds[0]-m, bounds[1]-m), (bounds[2]+m, bounds[1]-m),
        (bounds[2]+m, bounds[3]+m), (bounds[0]+m, bounds[3]+m),
    ])

    # 地面 = 整片背景（示意：pocket 外的区域）
    ground = Poly([(-0.55, -0.35), (0.55, -0.35), (0.55, 0.35), (-0.55, 0.35)])
    ground = ground.difference(pocket)

    # 连接垫（3 个：c2 左 c3 下 c4 上）
    cx, cy = 0, 0
    cw, ch = 0.1, 0.03
    connectors = {
        "c2": Poly([(bounds[2]+m, cy-ch/2), (bounds[2]+m+cw, cy-ch/2),
                     (bounds[2]+m+cw, cy+ch/2), (bounds[2]+m, cy+ch/2)]),
        "c3": Poly([(cx-cw/2, bounds[1]-m-ch), (cx+cw/2, bounds[1]-m-ch),
                     (cx+cw/2, bounds[1]-m), (cx-cw/2, bounds[1]-m)]),
        "c4": Poly([(cx-cw/2, bounds[3]+m), (cx+cw/2, bounds[3]+m),
                     (cx+cw/2, bounds[3]+m+ch), (cx-cw/2, bounds[3]+m+ch)]),
    }

    # 绘制
    # 1) 地面（灰色）
    if ground.geom_type == "Polygon":
        for poly in [ground]:
            ax.fill(*poly.exterior.xy, color="#8a8a8a", alpha=0.4, zorder=1)
    elif ground.geom_type == "MultiPolygon":
        for poly in ground.geoms:
            ax.fill(*poly.exterior.xy, color="#8a8a8a", alpha=0.4, zorder=1)

    # 2) pocket 边框（虚线）
    ax.plot(*pocket.exterior.xy, color="#555", linewidth=0.8,
            linestyle="--", zorder=2, alpha=0.7)

    # 3) pad（蓝色，带边框）
    for poly, lbl in ((pl, ""), (pr, "")):
        ax.fill(*poly.exterior.xy, color="#2196F3", alpha=0.85, zorder=3,
                edgecolor="#1565C0", linewidth=0.5)

    # 4) 连接垫（橙色）
    for c in connectors.values():
        ax.fill(*c.exterior.xy, color="#FF9800", alpha=0.85, zorder=3,
                edgecolor="#E65100", linewidth=0.5)

    # 5) 结（红色）
    jj = opts["junction"]
    ax.plot([jj[0][0], jj[1][0]], [jj[0][1], jj[1][1]],
            color="#F44336", linewidth=2.5, zorder=4)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.35, 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    # 标题（半透明白底框防遮盖）
    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=7,
            ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.75, edgecolor="gray", linewidth=0.3))


def main():
    n_cats = len(CATEGORY_NAMES)
    fig, axes = plt.subplots(n_cats, 2, figsize=(7, 3.2 * n_cats / 2))

    for i, cat in enumerate(CATEGORY_NAMES):
        for j, idx in enumerate((0, 1)):
            opts = make_pocket_record(cat, idx)
            title = f"{CAT_LABELS[cat]}  #{idx}"
            draw_one(axes[i, j], opts, title)

    # 总标题
    fig.suptitle("Transmon 双岛形状库（带地面平面 + pocket + 连接垫 + 结区）",
                 fontsize=11, fontweight="bold", y=0.98)
    # 图例（半透明白底框）
    legend_elems = [
        mpatches.Patch(facecolor="#2196F3", alpha=0.85, label="电容岛 (pad)"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.85, label="连接垫 (connector)"),
        mpatches.Patch(facecolor="#8a8a8a", alpha=0.4, label="地面平面 (ground)"),
        mpatches.Patch(facecolor="none", edgecolor="#555", linestyle="--",
                      label="pocket 边界"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.8, edgecolor="gray",
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = OUT / "shape_library.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"→ {out}")


if __name__ == "__main__":
    main()
