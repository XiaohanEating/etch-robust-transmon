# -*- coding: utf-8 -*-
"""G1 第四步：既定 10 代表 × 3 场景 × 2 档快速层矩阵（带修正参与比）。

代表 = 8 类各一（library.make_record(cat, 0)）+ 矩形对照 2 个（r=0 满幅、r=30）。
每代表：名义 + 刻蚀 ±0.25μm；每场景在 h=0.01/0.005 两档求解，名义 Lj 按
候选配定（f01→5.10 GHz），刻蚀沿用名义设计 Lj。
参与比 = step-3 的全局+局部耦合修正口径；判据含登记项（decisions.md 2026-10-24）。
输出 results/g1_step4_matrix_2026-10-24/matrix.json（断点续跑）。
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "g1_step4_matrix_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)
CHECK = OUT / "matrix.jsonl"
WORKERS = 10
LEVELS = {"h0p01": 0.01, "h0p005": 0.005}
F_TARGET = 5.10


def representatives():
    from robust_design_lab.geometry import library
    from robust_design_lab.configs import frozen as F
    reps = []
    for cat in library.CATEGORY_NAMES:
        reps.append((cat, library.make_record(cat, 0)))
    W2 = (F.FOOTPRINT_UM[0] - F.JUNCTION_GAP_UM) / 2
    reps.append(("rect_r0", library.rect_record(W2, F.FOOTPRINT_UM[1], 0.0)))
    reps.append(("rect_r30", library.rect_record(W2, F.FOOTPRINT_UM[1], 30.0)))
    return reps


def _refs_cached():
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    path = ROOT / "results/g1_step3_rectangle_2026-10-23/step3_fast_rectangle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["two_d_reference"]


def eval_one(job):
    tag, rec_rebuild, h_edge = job
    from robust_design_lab.evaluator import finite_film
    from robust_design_lab.evaluator.lom import transmon_metrics, match_lj
    from robust_design_lab.geometry import library, checks
    from robust_design_lab.perturbation.channels import apply_etch
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    from robust_design_lab.experiments.field_audit import save_solution

    cat, idx = rec_rebuild
    if cat == "rect_r0":
        from robust_design_lab.configs import frozen as F
        W2 = (F.FOOTPRINT_UM[0] - F.JUNCTION_GAP_UM) / 2
        rec = library.rect_record(W2, F.FOOTPRINT_UM[1], 0.0)
    elif cat == "rect_r30":
        from robust_design_lab.configs import frozen as F
        W2 = (F.FOOTPRINT_UM[0] - F.JUNCTION_GAP_UM) / 2
        rec = library.rect_record(W2, F.FOOTPRINT_UM[1], 30.0)
    else:
        rec = library.make_record(cat, idx)

    settings = finite_film.MeshSettings(thickness_nm=200.0, h_edge_mm=h_edge,
                                        h_max_mm=0.6, grow_mm=3.0)
    out = dict(rep=tag, level=f"h{h_edge}", category=rec.category,
               geometry_hash=rec.geometry_hash)
    ok, reasons = checks.check_record(rec)
    if not ok:
        return dict(out, failed="checks:" + ";".join(reasons)[:150])
    try:
        refs = _refs_cached()
        results = {}
        lj0 = None
        # 名义
        fs, surfaces = finite_film.solve_fields(rec, settings)
        lj0, info = match_lj(fs.C_q_fF * 1e-15, F_TARGET)
        out["lj0_nH"] = round(lj0 * 1e9, 4) if lj0 else None
        if lj0 is None:
            return dict(out, failed=f"match_lj: {info}")
        folder = OUT / "solutions" / f"{tag}_{h_edge}_nominal"
        folder.mkdir(parents=True, exist_ok=True)
        save_solution(folder, fs, rec, dict(h_edge_mm=h_edge))
        np.savez_compressed(folder / "surfaces.npz", **surfaces)
        p = s3.corrected_participation(fs, surfaces, rec, lj0, refs)
        tm = transmon_metrics(fs.C_q_fF * 1e-15, lj0)
        results["nominal"] = dict(C_q_fF=fs.C_q_fF, p_SM=p["p_SM"], p_SA=p["p_SA"],
                                  p_MA=None, f01_GHz=tm["f01_GHz"],
                                  alpha_MHz=tm["alpha_MHz"])
        # 刻蚀
        for sname, delta in (("etch_plus", frozen.ETCH_UM), ("etch_minus", -frozen.ETCH_UM)):
            erec, err = apply_etch(rec, delta)
            if err:
                results[sname] = dict(failed=err[:150])
                continue
            efs, esurf = finite_film.solve_fields(erec, settings)
            efolder = OUT / "solutions" / f"{tag}_{h_edge}_{sname}"
            efolder.mkdir(parents=True, exist_ok=True)
            save_solution(efolder, efs, erec, dict(h_edge_mm=h_edge))
            np.savez_compressed(efolder / "surfaces.npz", **esurf)
            ep = s3.corrected_participation(efs, esurf, erec, lj0, refs)
            etm = transmon_metrics(efs.C_q_fF * 1e-15, lj0)
            results[sname] = dict(C_q_fF=efs.C_q_fF, p_SM=ep["p_SM"], p_SA=ep["p_SA"],
                                  p_MA=None, f01_GHz=etm["f01_GHz"],
                                  alpha_MHz=etm["alpha_MHz"])
        # 判据（对偶档在汇总时算，单档先给漂移）
        import math
        nom = results["nominal"]
        for sname in ("etch_plus", "etch_minus"):
            r = results[sname]
            if "failed" in r or not r.get("p_MA") and r.get("p_MA") is None:
                pass
            if "failed" not in r:
                rate_n = nom["p_SM"] + nom["p_SA"]
                rate_s = r["p_SM"] + r["p_SA"]
                r["signed_log_drift_partial"] = (math.log(rate_s / rate_n)
                                                 + math.log(r["f01_GHz"] / nom["f01_GHz"]))
                r["log_C_drift"] = math.log(r["C_q_fF"] / nom["C_q_fF"])
        return dict(out, scenarios=results)
    except Exception as e:                                   # noqa: BLE001
        return dict(out, failed=f"{type(e).__name__}: {e}"[:200])


def main():
    t0 = time.perf_counter()
    reps = representatives()
    done = set()
    if CHECK.exists():
        for line in CHECK.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["rep"], d["level"]))
    jobs = [(tag, (tag if tag.startswith("rect") else rec.category, rec.category_index), h)
            for tag, rec in reps for h in LEVELS.values()]
    jobs = [j for j in jobs
            if (j[0], "h" + str(j[2])) not in done]
    print(f"matrix: {len(done)} done, {len(jobs)} to run", flush=True)

    with open(CHECK, "a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(eval_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] elapsed={time.perf_counter()-t0:.0f}s", flush=True)

    # 汇总判据
    rows = [json.loads(l) for l in CHECK.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = {}
    for r in rows:
        if "failed" in r:
            summary.setdefault(r["rep"], {})[r["level"]] = dict(failed=r["failed"])
            continue
        summary.setdefault(r["rep"], {})[r["level"]] = r
    report = dict(provenance=provenance(), frozen_config=frozen.frozen_hash(),
                  reps={}, thresholds=dict(C=0.02, p_each=0.05, drift=0.007))
    for rep, by_level in summary.items():
        a, b = by_level.get("h0.01"), by_level.get("h0.005")
        if not a or not b or "failed" in a or "failed" in b:
            report["reps"][rep] = dict(status="incomplete", detail={k: str(v)[:120] for k, v in by_level.items()})
            continue
        crit = {}
        ok = True
        for s in ("nominal", "etch_plus", "etch_minus"):
            ra, rb = a["scenarios"][s], b["scenarios"][s]
            if "failed" in ra or "failed" in rb:
                ok = False
                crit[s] = dict(status="scene_failed")
                continue
            c = dict(C=abs(rb["C_q_fF"] / ra["C_q_fF"] - 1),
                     p_SM=abs(rb["p_SM"] / ra["p_SM"] - 1),
                     p_SA=abs(rb["p_SA"] / ra["p_SA"] - 1))
            c["ok"] = c["C"] <= 0.02 and c["p_SM"] <= 0.05 and c["p_SA"] <= 0.05
            ok = ok and c["ok"]
            crit[s] = c
        for s in ("etch_plus", "etch_minus"):
            if "signed_log_drift_partial" in a["scenarios"][s] and \
                    "signed_log_drift_partial" in b["scenarios"][s]:
                d = abs(a["scenarios"][s]["signed_log_drift_partial"]
                        - b["scenarios"][s]["signed_log_drift_partial"])
                crit[s]["drift_level_diff"] = d
                ok = ok and d <= 0.007
        report["reps"][rep] = dict(status="passed" if ok else "failed", criteria=crit)
        print(f"[{rep}] {'PASS' if ok else 'FAIL'} "
              + " ".join(f"{s}:C={c.get('C', float('nan'))*100:.1f}%,SM={c.get('p_SM', float('nan'))*100:.1f}%,SA={c.get('p_SA', float('nan'))*100:.1f}%"
                         + (f",drift={c.get('drift_level_diff', float('nan')):.4f}" if 'drift_level_diff' in c else "")
                         for s, c in crit.items() if isinstance(c, dict) and 'ok' in c), flush=True)
    report["matrix_passed"] = all(v.get("status") == "passed" for v in report["reps"].values()) \
        and len(report["reps"]) == 10
    write_json(OUT / "matrix_summary.json", report)
    print(f"matrix passed = {report['matrix_passed']} → {OUT/'matrix_summary.json'} "
          f"[{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
