"""统一口径的 T1 复核（用户执行顺序第 5 步前半）。

口径统一（针对用户审计第 3 条）：
- 频率/电容：以 Q3D 名义结果为唯一权威（f01、C_Σ 来自 buffer
  刻蚀三场景的名义解）；
- 参与比：快速场层（corrected 评价器）只提供 p_SM/p_SA/p_MA（场
  分布量，其精度受匹配带 ±20% 限制，如实标注）；
- T1 = 1/(2π·f01_Q3D·Σ p_i·tanδ_i)，tanδ=2e-3（保守，与旧轮一致）；
- 漂移：只用 Q3D buffer 刻蚀通道的 D（评价器自身刻蚀通道本轮
  为空，不再引用）。
对象：资格过门者（m_variable_gap / q_annulus / q_ellipse_idc 基准 /
减容后 m_branch_comb 若过门）+ A 族冠军裸岛参照。
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "mainline_qualification_2026-11-01"
EJ_GHZ, EC_OF_C, TAN_D = 13.28, 19.36, 2e-3


def q3d_nominal(name, struct):
    from robust_design_lab.geometry.novel_structures import struct_to_options
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    r = run_q3d(struct_to_options(struct), "NovelTwoPadTransmon", "A_dual_pad")
    cs = r["C_sigma_fF"]
    ec = EC_OF_C / cs
    return cs, math.sqrt(8 * EJ_GHZ * ec) - ec


def participation(rec):
    from robust_design_lab.evaluator.corrected import evaluate_record
    out = evaluate_record(rec, lj0_H=12.31e-9)
    return out


def main():
    from robust_design_lab.geometry.novel_structures import (
        MAINLINE_FAMILIES, q_ellipse_idc, q_annulus)
    from robust_design_lab.geometry.record import GeometryRecord

    jobs = {
        "m_variable_gap": MAINLINE_FAMILIES["m_variable_gap"],
        "q_annulus": q_annulus,
        "q_ellipse_idc": q_ellipse_idc,
        "m_branch_comb": MAINLINE_FAMILIES["m_branch_comb"],  # 减容复测
        "m_graded_idc": MAINLINE_FAMILIES["m_graded_idc"],
    }
    from shapely.geometry import Polygon, LineString, box
    w, h, g = 560.0, 178.0, 50.0
    ref = dict(pad_left=box(-g/2-w, -h/2, -g/2, h/2),
               pad_right=box(g/2, -h/2, g/2+w, h/2), islands=[],
               junction=LineString([(-g/2, 0), (g/2, 0)]),
               label="A_champion_bare")
    jobs["A_champion_bare"] = (lambda ref=ref: ref)

    results = []
    for name, g in jobs.items():
        s = g()
        row = dict(family=name, label=s["label"])
        t0 = time.perf_counter()
        try:
            cs, f01 = q3d_nominal(name, s)
            rec = GeometryRecord(
                family="two_pad_free", category=name, category_index=0,
                seed=0, pad_left=s["pad_left"].simplify(1.0),
                pad_right=s["pad_right"].simplify(1.0),
                junction=s["junction"], note=s["label"])
            out = participation(rec)
            p = {k: out.get(f"p_{k}") for k in ("SM", "SA", "MA")}
            rate = 2 * math.pi * f01 * 1e9 * sum(
                p[k] * TAN_D for k in p if p[k] is not None)
            t1_us = 1 / rate * 1e6 if rate > 0 else None
            row.update(C_sigma_fF=cs, f01_Q3D=f01,
                       p_SM=p["SM"], p_SA=p["SA"], p_MA=p["MA"],
                       T1_us_unified=t1_us,
                       gate=("通过" if 4.5 <= f01 <= 5.5 and 30
                             <= EJ_GHZ / (EC_OF_C / cs) <= 80 else "不通过"))
            print(f"{name:18s} CΣ={cs:5.1f} f01={f01:5.2f} "
                  f"T1={t1_us:6.1f}us p_SM={p['SM']:.2e} [{row['gate']}]",
                  flush=True)
        except Exception as e:
            row["failed"] = f"{type(e).__name__}: {e}"[:200]
            print(f"{name:18s} 异常: {row['failed'][:80]}", flush=True)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(row)
        (OUT / "unified_t1.json").write_text(
            json.dumps(dict(provenance=provenance(), tan_delta=TAN_D,
                            protocol="T1=1/(2π·f01_Q3D·Σp·tanδ)；f01权威=Q3D；p=快层(±20%带)",
                            results=results),
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
    print("saved:", OUT / "unified_t1.json")


if __name__ == "__main__":
    main()
