"""十个优质电容候选的 Transmon 资格审查：Q3D 实测 C_Σ → EJ/EC / f01 / α。

判据（Koch 2007 / Krantz）：EJ/EC ∈ [30,80] 为 Transmon 区、≈50 甜点；
固定 Lj=12.31nH → EJ=13.28GHz，EC[GHz]=19.36/C_Σ[fF]，故
EJ/EC = 0.686·C_Σ[fF]，甜点要求 C_Σ≈73fF，窗口 [44fF, 117fF]。
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "quality_shapes_2026-10-31"

EJ_GHZ = 13.28          # Lj=12.31 nH 固定
EC_OF_C = 19.36         # EC[GHz] = 19.36 / C_Σ[fF]


def transmon_metrics(c_sigma_fF):
    ec = EC_OF_C / c_sigma_fF
    f01 = math.sqrt(8 * EJ_GHZ * ec) - ec
    return dict(EC_GHz=ec, EJ_over_EC=EJ_GHZ / ec, f01_GHz=f01,
                alpha_MHz=-ec * 1000)


def main():
    from robust_design_lab.geometry.novel_structures import QUALITY_FAMILIES
    from robust_design_lab.geometry.novel_structures import struct_to_options
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d

    results = []
    for name, g in QUALITY_FAMILIES.items():
        s = g()
        t0 = time.perf_counter()
        try:
            opts = struct_to_options(s)
            r = run_q3d(opts, "NovelTwoPadTransmon", "A_dual_pad")
            cs = r["C_sigma_fF"]
            m = transmon_metrics(cs)
            m.update(family=name, C_sigma_fF=cs, label=s["label"],
                     solve_s=r["solve_s"],
                     verdict="甜点" if 44 <= m["EJ_over_EC"] <= 62 else
                             ("Transmon区" if 30 <= m["EJ_over_EC"] <= 80 else "不合格"))
            print(f"{name:16s} CΣ={cs:6.1f}fF  EJ/EC={m['EJ_over_EC']:5.1f}  "
                  f"f01={m['f01_GHz']:5.2f}GHz  α={m['alpha_MHz']:6.0f}MHz  "
                  f"[{m['verdict']}]", flush=True)
        except Exception as e:
            m = dict(family=name, failed=f"{type(e).__name__}: {e}"[:200])
            print(f"{name:16s} 失败: {m['failed'][:80]}", flush=True)
        m["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(m)
        (OUT / "transmon_check.json").write_text(
            json.dumps(dict(provenance=provenance(), EJ_GHZ=EJ_GHZ,
                            results=results), indent=2, ensure_ascii=False),
            encoding="utf-8")
    print("saved:", OUT / "transmon_check.json")


if __name__ == "__main__":
    main()
