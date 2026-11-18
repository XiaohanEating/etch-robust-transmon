# -*- coding: utf-8 -*-
"""优质电容候选形状审阅图：十候选各自大图 + 2×5 总览拼图。

按 figures4papers 原则：图注只列含义不标实验编号；图中文字用半透明
白底框防遮盖。输出 results/quality_shapes_2026-10-31/。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

OUT = ROOT / "results" / "quality_shapes_2026-10-31"
OUT.mkdir(parents=True, exist_ok=True)


def draw_struct(ax, s, title=None, note=None):
    for g in (s["pad_left"], s["pad_right"]):
        ax.fill(*g.exterior.xy, color="#2196F3", alpha=0.85, zorder=3,
                edgecolor="#0D47A1", linewidth=0.8)
        for hole in g.interiors:
            ax.fill(*hole.xy, color="white", zorder=4)
    for isl in s["islands"]:
        ax.fill(*isl.exterior.xy, color="#FF9800", alpha=0.9, zorder=3,
                edgecolor="#E65100", linewidth=0.8)
    x, y = s["junction"].xy
    ax.plot(x, y, color="#F44336", linewidth=2.5, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    if note:
        allg = [s["pad_left"], s["pad_right"]] + list(s["islands"])
        x0 = min(g.bounds[0] for g in allg)
        y0 = min(g.bounds[1] for g in allg)
        x1 = max(g.bounds[2] for g in allg)
        y1 = max(g.bounds[3] for g in allg)
        ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999",
                          alpha=0.85))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    from robust_design_lab.geometry.novel_structures import (
        QUALITY_FAMILIES, RATIONALES)

    meta = {}
    # 单独大图
    for name, g in QUALITY_FAMILIES.items():
        s = g()
        fig, ax = plt.subplots(figsize=(5, 5.6))
        draw_struct(ax, s, title=name, note=RATIONALES[name])
        fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        junc = s["junction"]
        meta[name] = dict(
            label=s["label"],
            area_left_um2=round(s["pad_left"].area),
            area_right_um2=round(s["pad_right"].area),
            junction_len_um=round(junc.length, 1),
            n_islands=len(s["islands"]),
            rationale=RATIONALES[name])
    # 总览 2×5
    names = list(QUALITY_FAMILIES)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10.5))
    for ax, name in zip(axes.flat, names):
        s = QUALITY_FAMILIES[name]()
        draw_struct(ax, s, title=name, note=RATIONALES[name])
    fig.suptitle("十个优质电容候选形状（蓝=电容岛，橙=漂浮岛，红=约瑟夫森结；"
                 "坐标为 μm，各子图等比例）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "gallery_10.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    (OUT / "shapes_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"渲染完成 → {OUT}")


if __name__ == "__main__":
    main()
