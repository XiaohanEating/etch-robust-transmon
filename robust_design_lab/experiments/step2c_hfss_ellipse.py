# -*- coding: utf-8 -*-
"""第二步(c)：ellipse 代表的独立 HFSS 对照（名义 + 刻蚀±，共享 Lj）。

复用 hfss_interfaces 的 build/postprocess（有限 200nm PEC + 命名界面薄片 +
薄层后处理）。Lj 由 ellipse 的 h0p01 名义电容配定（f01→5.10 GHz），三场景共用。
输出 results/step2c_hfss_ellipse_2026-10-25/。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step2c_hfss_ellipse_2026-10-25"
OUT.mkdir(parents=True, exist_ok=True)
MIN_PASSES = 9


def main():
    from robust_design_lab.evaluator.ansys_calibration import connect
    from robust_design_lab.evaluator.lom import match_lj, transmon_metrics
    from robust_design_lab.experiments import hfss_interfaces as hi
    from robust_design_lab.geometry import library
    from robust_design_lab.perturbation.channels import apply_etch
    from robust_design_lab.configs import frozen

    t0 = time.perf_counter()
    # Lj：ellipse h0p01 名义 C_q（g1_step4 存档解）
    sol = json.loads((ROOT / "results" / "g1_step4_matrix_2026-10-24" / "solutions"
                      / "ellipse_arc_0.01_nominal" / "solution.json")
                     .read_text(encoding="utf-8"))
    lj0, info = match_lj(sol["C_q_fF"] * 1e-15, 5.10)
    if lj0 is None:
        raise RuntimeError(f"match_lj: {info}")
    print(f"Lj0 = {lj0*1e9:.4f} nH (ellipse h0p01 C_q={sol['C_q_fF']:.2f} fF)",
          flush=True)

    desktop = connect()
    rec0 = library.make_record("ellipse_arc", 0)
    scenes = [("nominal", rec0, 0.0),
              ("etch_plus", None, frozen.ETCH_UM),
              ("etch_minus", None, -frozen.ETCH_UM)]
    results = {}
    for name, rec, delta in scenes:
        if rec is None:
            rec, err = apply_etch(rec0, delta)
            if err:
                raise RuntimeError(err)
        project = desktop.new_project()
        t1 = time.perf_counter()
        design, setup, surfaces, face_data = hi.build(
            project, rec, 200.0, lj0, MIN_PASSES)
        setup.analyze()          # build 只建模；求解必须显式触发
        rep = hi.postprocess(setup, surfaces)
        tm = transmon_metrics(rep["modal_capacitance_fF"] * 1e-15 or sol["C_q_fF"] * 1e-15,
                              lj0)
        results[name] = dict(
            geometry_hash=rec.geometry_hash, etch_um=delta,
            solve_s=round(time.perf_counter() - t1, 1),
            f_eigen_GHz=rep["f_eigen_Hz"] / 1e9,
            participation=rep["participation"],
            modal_capacitance_fF=rep.get("modal_capacitance_fF"),
            surface_integrals=rep["surface_integrals"],
            lom_f01_GHz=tm["f01_GHz"], lom_alpha_MHz=tm["alpha_MHz"])
        print(f"[{name}] f_eigen={results[name]['f_eigen_GHz']:.4f} GHz "
              f"p_SM={rep['participation']['p_SM']:.4e} "
              f"p_SA={rep['participation']['p_SA']:.4e} "
              f"[{results[name]['solve_s']}s]", flush=True)
        write_json(OUT / "ellipse_hfss.json",
                   dict(provenance=provenance(), lj0_nH=lj0 * 1e9,
                        min_passes=MIN_PASSES, results=results))

    # 漂移（HFSS 参与比 + LOM f01；界面损耗口径）
    import numpy as np
    def rate(r):
        return 2 * math.pi * r["lom_f01_GHz"] * 1e9 * sum(
            r["participation"]["p_" + k] * frozen.TAN_DELTA[k]
            for k in ("SM", "MA", "SA"))
    r0 = rate(results["nominal"])
    drifts = {s: math.log(r0 / rate(results[s])) for s in ("etch_plus", "etch_minus")}
    # 与快速层对比（step2b 未舍入）
    fast = json.loads((ROOT / "results" / "step2b_drift_rank_2026-10-25"
                       / "g1_repost.json").read_text(encoding="utf-8"))
    fd = fast["reps"]["ellipse_arc"]["drifts"]
    report = dict(provenance=provenance(), lj0_nH=lj0 * 1e9, results=results,
                  hfss_signed_drifts=drifts,
                  fast_signed_drifts={k: fd[f"0.005_{k}"] for k in
                                      ("etch_plus", "etch_minus")},
                  fast_level=0.005,
                  wall_s=round(time.perf_counter() - t0, 1))
    write_json(OUT / "ellipse_hfss.json", report)
    print("HFSS drifts:", {k: round(v, 5) for k, v in drifts.items()})
    print("fast(0.005):", {k: round(fd[f'0.005_{k}'], 5) for k in
                           ('etch_plus', 'etch_minus')})


if __name__ == "__main__":
    main()
