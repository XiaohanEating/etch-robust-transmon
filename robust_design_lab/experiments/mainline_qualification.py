"""主线资格试验（统一地平面环境版，2026-11-02）。

结构：基准 q_ellipse_idc（已验证）+ 三类机制明确新结构
（渐变交指/分支梳/平滑变间隙）+ q_annulus（内孔导出修复后复测）。
协议：两电极、单结、无浮置导体、无参数调整；每结构名义+刻蚀±0.5μm
（各向同性 buffer）三场景 Q3D。

逐项检查（用户要求的资格条件）：
  G1 网数==2（pad_left/pad_right，多余网即失败）；
  G2 C_Σ 约化：双节点解析式 C_Σ=C12+Cj+Σ_k C1k·C2k/(C1k+C2k)，
     Cj 只计一次（compute_c_sigma 实现即此式）；
  G3 电路指标：EJ/EC∈[30,80]（甜点[44,62]）、f01∈[4.5,5.5]GHz；
  G4 面积守恒：源 shapely ↔ Metal qgeometry 表面积一致（内孔计入）。
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
OUT.mkdir(parents=True, exist_ok=True)
EJ_GHZ, EC_OF_C = 13.28, 19.36


def area_consistency(s):
    """源 shapely 与 Metal qgeometry 表的面积一致性（内孔计入）。"""
    from robust_design_lab.geometry.novel_structures import struct_to_options
    from robust_design_lab.geometry.four_shape_families import build_design_for
    src_area = (s["pad_left"].area + s["pad_right"].area) * 1e-6  # mm²
    d = build_design_for(struct_to_options(s), "NovelTwoPadTransmon")
    tbl = d.qgeometry.tables["poly"]
    keep = ~tbl["subtract"] if "subtract" in tbl else slice(None)
    metal_area = sum(g.area for g in tbl[keep].geometry)  # mm²（不含 subtract 口袋）
    ok = abs(metal_area - src_area) / src_area < 1e-3
    return ok, src_area * 1e6, metal_area * 1e6


def main():
    from robust_design_lab.geometry.novel_structures import (
        MAINLINE_FAMILIES, q_ellipse_idc, q_annulus, struct_to_options,
        etch_variant)
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d

    jobs = {"m_graded_idc": MAINLINE_FAMILIES["m_graded_idc"],
            "m_branch_comb": MAINLINE_FAMILIES["m_branch_comb"],
            "m_variable_gap": MAINLINE_FAMILIES["m_variable_gap"],
            "q_ellipse_idc": q_ellipse_idc,       # 基准
            "q_annulus": q_annulus}               # 内孔修复复测

    results = []
    t_all = time.perf_counter()
    for name, g in jobs.items():
        s = g()
        g4, a_src, a_metal = area_consistency(s)
        row = dict(family=name, label=s["label"],
                   area_src_um2=round(a_src), area_metal_um2=round(a_metal),
                   area_consistent=g4)
        try:
            cs_all, nets_ok = {}, None
            for tag, struct in (("nom", s),
                                ("ep", etch_variant(s, 0.5)),
                                ("em", etch_variant(s, -0.5))):
                r = run_q3d(struct_to_options(struct), "NovelTwoPadTransmon",
                            "A_dual_pad")
                cs_all[tag] = r["C_sigma_fF"]
                nets = [n for n in r["nets"] if "pad" in n.lower()]
                if nets_ok is None:
                    # 统一环境：pad_left/pad_right + ground_main_plane
                    nets_ok = (sorted(r["nets"]) ==
                               ["ground_main_plane", "pad_left_Q1",
                                "pad_right_Q1"])
                row[f"nets_{tag}"] = r["nets"]
            cs = cs_all["nom"]
            ec = EC_OF_C / cs
            f01 = math.sqrt(8 * EJ_GHZ * ec) - ec
            f_ep = math.sqrt(8 * EJ_GHZ * EC_OF_C / cs_all["ep"]) - EC_OF_C / cs_all["ep"]
            f_em = math.sqrt(8 * EJ_GHZ * EC_OF_C / cs_all["em"]) - EC_OF_C / cs_all["em"]
            D = max(abs(math.log(f_ep / f01)), abs(math.log(f_em / f01)))
            row.update(C_sigma_fF=cs, C_ep=cs_all["ep"], C_em=cs_all["em"],
                       f01_GHz=f01, alpha_MHz=-ec * 1e3,
                       EJ_over_EC=EJ_GHZ / ec, D=D, nets_ok=nets_ok)
            row["verdict"] = "通过" if (
                nets_ok and g4 and 30 <= row["EJ_over_EC"] <= 80
                and 4.5 <= f01 <= 5.5) else "不通过"
            print(f"{name:16s} CΣ={cs:5.1f}fF EJ/EC={row['EJ_over_EC']:5.1f} "
                  f"f01={f01:5.2f} α={row['alpha_MHz']:6.0f}MHz D={D:.5f} "
                  f"网2={nets_ok} 面积守恒={g4} → {row['verdict']}", flush=True)
        except Exception as e:
            row.update(failed=f"{type(e).__name__}: {e}"[:200], verdict="异常")
            print(f"{name:16s} 异常: {row['failed'][:80]}", flush=True)
        results.append(row)
        (OUT / "qualification.json").write_text(
            json.dumps(dict(provenance=provenance(), results=results),
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
    print(f"总耗时 {time.perf_counter()-t_all:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
