# -*- coding: utf-8 -*-
"""跨族新结构首次 Q3D 评估：名义 + 刻蚀± 三场景 → C_Σ/f01/α/D。

三个新结构（镂空岛 / 漂浮岛桥 / 科赫雪花）与原五族的评价口径一致：
同 Q3D 求解路径、同 C_Σ 公式（dual-pad 分支，漂浮岛走串联项）、
同 f01/α 换算；差异仅在刻蚀建模——新结构用各向同性 buffer ±δ（原五族
为参数平移）。输出渲染图 + 指标 JSON。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "novel_structures_2026-10-30"
OUT.mkdir(parents=True, exist_ok=True)

ETCH_DELTA = 0.5

# 各结构设计意图（VLM 提示用）
FAMILY_NOTE = """1. hoya_perforated 镂空岛：pad 内圆孔阵列（复连通）。孔位于低场内区，
   刻蚀使孔缘变化但被大面积稀释；文献锚点 Hoya（表面损耗抑制）。
2. floating_bridge 漂浮岛桥：两岛上方一排漂浮金属岛形成串联电容桥
   （pad1—岛—…—岛—pad2）。单个间隙的刻蚀误差被串联组合稀释；
   岛与 pad、岛与岛间距均为可调电容。
3. koch 科赫雪花：多尺度分形边界，边缘长度跨尺度分布，δ=0.5μm 的
   相对扰动随尺度变化。
通用规则：C_Σ 大稀释漂移；gap 大降 D 但降 C_Σ；圆滑边界优于尖角；
f01=√(8EJ·EC)/h−EC/h，Lj=12.31nH、Cj=2fF 固定；目标 C_Σ≥70fF、
f01∈[4.5,5.5]GHz、α∈[−300,−150]MHz。"""


def render(struct, filepath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for name, g in [("pad_left", struct["pad_left"]),
                    ("pad_right", struct["pad_right"])]:
        ax.fill(*g.exterior.xy, color="#2196F3", alpha=0.85, zorder=3,
                edgecolor="#0D47A1", linewidth=0.8)
        for hole in g.interiors:
            ax.fill(*hole.xy, color="white", zorder=4)
    for isl in struct["islands"]:
        ax.fill(*isl.exterior.xy, color="#FF9800", alpha=0.9, zorder=3,
                edgecolor="#E65100", linewidth=0.8)
    x, y = struct["junction"].xy
    ax.plot(x, y, color="#F44336", linewidth=3, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def eval_struct(struct, label):
    """三场景 Q3D 评估 → dict(C_sigma, f01, alpha, D, f_ep, f_em)。"""
    from robust_design_lab.experiments.q3d_shape_sweep import (
        run_q3d, freq_from_csigma)
    from robust_design_lab.geometry.novel_structures import (
        struct_to_options, etch_variant, check_structure)

    errs = check_structure(struct)
    if errs:
        return dict(label=label, failed=f"几何检查: {errs}")

    def solve(s):
        opts = struct_to_options(s)
        r = run_q3d(opts, "NovelTwoPadTransmon", "A_dual_pad")
        return r["C_sigma_fF"], freq_from_csigma(r["C_sigma_fF"])

    cs, (f01, alpha) = solve(struct)
    cs_ep, (f_ep, _) = solve(etch_variant(struct, +ETCH_DELTA))
    cs_em, (f_em, _) = solve(etch_variant(struct, -ETCH_DELTA))
    D = max(abs(math.log(f_ep / f01)), abs(math.log(f_em / f01)))
    return dict(label=label, C_sigma_fF=cs, f01_GHz=f01, alpha_MHz=alpha,
                D=D, f01_ep=f_ep, f01_em=f_em,
                C_ep=cs_ep, C_em=cs_em)


def main():
    from robust_design_lab.geometry.novel_structures import all_structures
    t0 = time.perf_counter()
    structs = all_structures()
    results = []
    for name, s in structs.items():
        print(f"== {name} {s['label']} ==", flush=True)
        render(s, OUT / f"{name}.png")
        t1 = time.perf_counter()
        rec = eval_struct(s, s["label"])
        rec["family"] = name
        rec["eval_s"] = round(time.perf_counter() - t1, 1)
        results.append(rec)
        if "failed" in rec:
            print(f"  失败: {rec['failed']}", flush=True)
        else:
            print(f"  CΣ={rec['C_sigma_fF']:.1f}fF f01={rec['f01_GHz']:.3f}GHz "
                  f"α={rec['alpha_MHz']:.0f}MHz D={rec['D']:.5f}", flush=True)
        (OUT / "novel_first_eval.json").write_text(
            json.dumps(dict(provenance=provenance(), results=results),
                       indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"总耗时 {time.perf_counter()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
