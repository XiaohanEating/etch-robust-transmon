# -*- coding: utf-8 -*-
"""第三步(a)：固定 Lj 重后处理全部 10 代表存档解 + 原容差 G1 判定。

数据：results/g1_step4_matrix_2026-10-24/solutions/{tag}_{h}_{scene}/（60 态，
修正公式求解，身份在 solution.json）；Lj 每代表固定为其 h0p01 名义配定值。
判据（原容差，docs/decisions.md 2026-10-23）：电容 ≤2%、三个参与比各 ≤5%、
逐方向有符号寿命漂移档位差 ≤0.005。MA 不豁免——另附 T1 影响界限量化
（供用户裁决，不自动替代分项验收）。
输出 results/step3_repost_g1_2026-10-24/。
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

SOL = ROOT / "results" / "g1_step4_matrix_2026-10-24" / "solutions"
OUT = ROOT / "results" / "step2b_drift_rank_2026-10-25"
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ("rect_rounded", "cross", "teeth_interdig", "ellipse_arc", "trapezoid_asym",
        "notch_taper", "irregular", "combo", "rect_r0", "rect_r30")
LEVELS = ("0.01", "0.005")
SCENES = ("nominal", "etch_plus", "etch_minus")


def load_refs():
    p = ROOT / "results" / "g1_step3_rectangle_2026-10-23" / "step3_fast_rectangle.json"
    return json.loads(p.read_text(encoding="utf-8"))["two_d_reference"]


def post_one(job):
    tag, level, scene, lj_pinned = job
    import numpy as np
    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    from robust_design_lab.evaluator.lom import transmon_metrics

    folder = SOL / f"{tag}_{level}_{scene}"
    refs = load_refs()
    fs, rec, _ = load_solution(folder)
    with np.load(folder / "surfaces.npz", allow_pickle=False) as s:
        surfaces = {k: s[k] for k in s.files}
    pr = s3.corrected_participation(fs, surfaces, rec, lj_pinned, refs)
    tm = transmon_metrics(fs.C_q_fF * 1e-15, lj_pinned)
    import math
    rate = (pr["p_SM"] + pr["p_SA"] + pr["p_MA"])
    return dict(tag=tag, level=level, scene=scene, C_q_fF=fs.C_q_fF,
                p_SM=pr["p_SM"], p_SA=pr["p_SA"], p_MA=pr["p_MA"],
                f01_GHz=tm["f01_GHz"], alpha_MHz=tm["alpha_MHz"],
                rate_partial=rate)


PINNED_LJ = {}


def main():
    t0 = time.perf_counter()
    # 每代表固定 Lj：优先取 matrix.jsonl 的 h0p01 名义 lj0（求解时配定）
    rows = [json.loads(l) for l in
            (ROOT / "results" / "g1_step4_matrix_2026-10-24" / "matrix.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if r.get("level") == "h0.01" and "lj0_nH" in r:
            PINNED_LJ[r["rep"]] = r["lj0_nH"] * 1e-9
    missing = [t for t in TAGS if t not in PINNED_LJ]
    if missing:
        raise RuntimeError(f"缺少固定 Lj: {missing}")

    jobs = [(t, lv, sc, PINNED_LJ[t]) for t in TAGS for lv in LEVELS for sc in SCENES]
    results = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(post_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if k % 20 == 0:
                print(f"  [{k}/60] {time.perf_counter()-t0:.0f}s", flush=True)

    table = {(r["tag"], r["level"], r["scene"]): r for r in results}
    report = dict(provenance=provenance(), pinned_Lj_nH={t: v * 1e9 for t, v in PINNED_LJ.items()},
                  thresholds=dict(C=0.02, p_each=0.05, drift=0.005), reps={})

    import math
    for tag in TAGS:
        crit = {}
        ok = True
        for sc in SCENES:
            a, b = table[(tag, "0.01", sc)], table[(tag, "0.005", sc)]
            c = dict(C=abs(b["C_q_fF"] / a["C_q_fF"] - 1),
                     p_SM=abs(b["p_SM"] / a["p_SM"] - 1),
                     p_SA=abs(b["p_SA"] / a["p_SA"] - 1),
                     p_MA=abs(b["p_MA"] / a["p_MA"] - 1))
            c["ok"] = (c["C"] <= 0.02 and c["p_SM"] <= 0.05
                       and c["p_SA"] <= 0.05 and c["p_MA"] <= 0.05)
            ok &= c["ok"]
            crit[sc] = c
        drift = {}
        for lv in LEVELS:
            n = table[(tag, lv, "nominal")]
            rate0 = 2 * math.pi * n["f01_GHz"] * 1e9 * n["rate_partial"]
            for sc in ("etch_plus", "etch_minus"):
                r = table[(tag, lv, sc)]
                rate = 2 * math.pi * r["f01_GHz"] * 1e9 * r["rate_partial"]
                drift[(lv, sc)] = math.log(rate0 / rate)  # log(T1δ/T10)
        for sc in ("etch_plus", "etch_minus"):
            d = abs(drift[("0.01", sc)] - drift[("0.005", sc)])
            crit[sc]["drift_level_diff"] = d
            ok &= d <= 0.005
        # MA 影响界限（量化，供裁决；不改判据）
        nom01, nom005 = table[(tag, "0.01", "nominal")], table[(tag, "0.005", "nominal")]
        w_ma = min(nom01["p_MA"] / nom01["rate_partial"],
                   nom005["p_MA"] / nom005["rate_partial"])
        ma_err = max(crit[s]["p_MA"] for s in SCENES)
        report["reps"][tag] = dict(
            passed=bool(ok), criteria=crit,
            drifts={f"{lv}_{sc}": v for (lv, sc), v in drift.items()},
            D={lv: max(abs(drift[(lv, 'etch_plus')]), abs(drift[(lv, 'etch_minus')]))
               for lv in LEVELS},
            ma_weight=w_ma, ma_level_err=ma_err,
            ma_T1_impact_bound=w_ma * ma_err)

    passed = [t for t, v in report["reps"].items() if v["passed"]]
    report["n_passed"] = len(passed)
    report["G1_passed_original_tolerances"] = bool(len(passed) == len(TAGS))
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "g1_repost.json", report)
    print(f"G1(原容差) passed = {report['G1_passed_original_tolerances']} "
          f"({len(passed)}/{len(TAGS)}: {passed}) → {OUT/'g1_repost.json'} "
          f"[{report['wall_s']}s]")


if __name__ == "__main__":
    main()
