# -*- coding: utf-8 -*-
"""第二步(a)：真窗口敏感性（同步 2D/3D、同场解来源、同单位、固定 Lj）。

修正 2026-10-25 审阅 §3 的三个混淆：
- 同步窗口：W_IN 与匹配带/背景带一起移动（band=[W, W+20]，bg=[W+20, W+70]），
  2D 参考与 3D 后处理用同一组参数；
- 同一场解来源：三个窗口都用 finite_alignment_v1/rect_reference 的独立重网格
  存档（h0p01×3 + h0p005×3），不再混用输运解；
- 单位与 Lj：Lj 以亨为单位（14.19769287109375 nH → e-9），两档共用；
  有符号漂移 = log(T1_δ/T1_0) = log(rate_0/rate_δ)。

输出 results/step2a_window_v2_2026-10-25/。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step2a_window_v2_2026-10-25"
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVES = ROOT / "results" / "finite_alignment_v1" / "rect_reference"
LJ_H = 14.19769287109375e-9      # 矩形 h0p01 名义配定值（亨）
WINDOWS = (15.0, 20.0, 30.0)


def main():
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.evaluator.lom import transmon_metrics
    import numpy as np

    t0 = time.perf_counter()
    report = dict(provenance=provenance(), lj_H=LJ_H, windows={})
    base = json.loads((ROOT / "results" / "g1_step3_rectangle_2026-10-23"
                       / "step3_fast_rectangle.json").read_text(encoding="utf-8"))

    orig = (s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2)
    try:
        for w in WINDOWS:
            # 同步窗口：窗口/带/背景一起移动
            s3.W_IN = w
            s3.B1, s3.B2 = w, w + 20.0
            s3.BG1, s3.BG2 = w + 20.0, w + 70.0
            refs = dict(outer=s3.two_d_reference(), gap=s3.gap_reference())

            rows = {}
            for level in ("h0p01", "h0p005"):
                for scene in ("nominal", "etch_plus", "etch_minus"):
                    folder = ARCHIVES / level / scene
                    fs, rec, _ = load_solution(folder)
                    with np.load(folder / "surfaces.npz", allow_pickle=False) as s:
                        surfaces = {k: s[k] for k in s.files}
                    pr = s3.corrected_participation(fs, surfaces, rec, LJ_H, refs)
                    tm = transmon_metrics(fs.C_q_fF * 1e-15, LJ_H)
                    rate = 2 * math.pi * tm["f01_GHz"] * 1e9 * (
                        pr["p_SM"] + pr["p_SA"] + pr["p_MA"])
                    rows[(level, scene)] = dict(
                        p_SM=pr["p_SM"], p_SA=pr["p_SA"], p_MA=pr["p_MA"],
                        f01_GHz=tm["f01_GHz"], rate=rate)

            drifts = {}
            for level in ("h0p01", "h0p005"):
                n = rows[(level, "nominal")]
                for sc in ("etch_plus", "etch_minus"):
                    r = rows[(level, sc)]
                    drifts[(level, sc)] = math.log(n["rate"] / r["rate"])  # log(T1δ/T10)
            diffs = {sc: abs(drifts[("h0p01", sc)] - drifts[("h0p005", sc)])
                     for sc in ("etch_plus", "etch_minus")}
            n01, n005 = rows[("h0p01", "nominal")], rows[("h0p005", "nominal")]
            report["windows"][str(w)] = dict(
                p_SM=dict(h0p01=n01["p_SM"], h0p005=n005["p_SM"],
                          rel=abs(n005["p_SM"] / n01["p_SM"] - 1)),
                p_SA=dict(h0p01=n01["p_SA"], h0p005=n005["p_SA"],
                          rel=abs(n005["p_SA"] / n01["p_SA"] - 1)),
                drifts={f"{a}_{b}": v for (a, b), v in drifts.items()},
                drift_level_diff=diffs)
            print(f"[W={w}] SM rel={report['windows'][str(w)]['p_SM']['rel']*100:.2f}% "
                  f"SA rel={report['windows'][str(w)]['p_SA']['rel']*100:.2f}% "
                  f"drift_diff={diffs} [{time.perf_counter()-t0:.0f}s]", flush=True)
            write_json(OUT / "window_v2.json", report)
    finally:
        s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2 = orig

    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "window_v2.json", report)
    print(f"→ {OUT/'window_v2.json'} [{report['wall_s']}s]")


if __name__ == "__main__":
    main()
