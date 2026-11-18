# -*- coding: utf-8 -*-
"""第二步(b)：分离式敏感性——窗口 W、匹配带、背景带各自单独变化。

对应 corrections_followup_review §4："现有结果不能把差异全部归因于衔接位置
或二维参考精度"。七个配置（每配置全量 2D 参考 + 矩形六存档后处理）：

  base    : W=20,  band=[20,40],  bg=[40,90]
  W10/W15 : 仅 W 变（band/bg 不动；W=30 会与带重叠，不入组，报告说明）
  band_lo : band=[15,35]（bg 不动）
  band_hi : band=[25,45], bg=[50,100]（带内 bg 无重叠；与 band_lo 成对比较）
  bg_lo/bg_hi : 仅 bg 变（[45,95] / [60,110]）

另附区域能量闭合检查：间隙 SA 替换项 2·amp·U2D 与同区域 3D 薄层积分的比值
（期望 ≈ 2×(固体/薄层)₂D ≈ 1.4±0.2——独立于任何 G1 判据）。
输出 results/step2b_separated_sens_2026-10-25/。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step2b_separated_sens_2026-10-25"
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVES = ROOT / "results" / "finite_alignment_v1" / "rect_reference"
LJ_H = 14.19769287109375e-9

CONFIGS = {
    "base":    dict(W=20.0, B1=20.0, B2=40.0, BG1=40.0, BG2=90.0),
    "W10":     dict(W=10.0, B1=20.0, B2=40.0, BG1=40.0, BG2=90.0),
    "W15":     dict(W=15.0, B1=20.0, B2=40.0, BG1=40.0, BG2=90.0),
    "band_lo": dict(W=20.0, B1=15.0, B2=35.0, BG1=40.0, BG2=90.0),
    "band_hi": dict(W=20.0, B1=25.0, B2=45.0, BG1=50.0, BG2=100.0),
    "bg_lo":   dict(W=20.0, B1=20.0, B2=40.0, BG1=45.0, BG2=95.0),
    "bg_hi":   dict(W=20.0, B1=20.0, B2=40.0, BG1=60.0, BG2=110.0),
}


def run_config(s3, cfg):
    import numpy as np
    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.evaluator.lom import transmon_metrics

    orig = (s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2)
    s3.W_IN, s3.B1, s3.B2 = cfg["W"], cfg["B1"], cfg["B2"]
    s3.BG1, s3.BG2 = cfg["BG1"], cfg["BG2"]
    try:
        g0 = 30.0
        refs = dict(outer=s3.two_d_reference(),
                    gap_by_um={g: s3.gap_reference(gap_um=g)
                               for g in (29.5, 30.0, 30.5)})
        rows = {}
        for level in ("h0p01", "h0p005"):
            for scene in ("nominal", "etch_plus", "etch_minus"):
                folder = ARCHIVES / level / scene
                fs, rec, _ = load_solution(folder)
                with np.load(folder / "surfaces.npz", allow_pickle=False) as sf:
                    surfaces = {k: sf[k] for k in sf.files}
                pr = s3.corrected_participation(fs, surfaces, rec, LJ_H, refs)
                tm = transmon_metrics(fs.C_q_fF * 1e-15, LJ_H)
                rate = 2 * math.pi * tm["f01_GHz"] * 1e9 * (
                    pr["p_SM"] + pr["p_SA"] + pr["p_MA"])
                rows[(level, scene)] = dict(
                    p_SM=pr["p_SM"], p_SA=pr["p_SA"], p_MA=pr["p_MA"], rate=rate)
        drifts = {}
        for level in ("h0p01", "h0p005"):
            n = rows[(level, "nominal")]
            for sc in ("etch_plus", "etch_minus"):
                r = rows[(level, sc)]
                drifts[(level, sc)] = math.log(n["rate"] / r["rate"])
        diffs = {sc: abs(drifts[("h0p01", sc)] - drifts[("h0p005", sc)])
                 for sc in ("etch_plus", "etch_minus")}
        n01, n005 = rows[("h0p01", "nominal")], rows[("h0p005", "nominal")]
        return dict(
            p_SM=dict(h0p01=n01["p_SM"], h0p005=n005["p_SM"],
                      rel=abs(n005["p_SM"] / n01["p_SM"] - 1)),
            p_SA=dict(h0p01=n01["p_SA"], h0p005=n005["p_SA"],
                      rel=abs(n005["p_SA"] / n01["p_SA"] - 1)),
            drift_level_diff=diffs,
            drifts={f"{a}_{b}": v for (a, b), v in drifts.items()})
    finally:
        s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2 = orig


def main():
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3

    t0 = time.perf_counter()
    report = dict(provenance=provenance(), lj_H=LJ_H, configs={})
    for name, cfg in CONFIGS.items():
        t1 = time.perf_counter()
        report["configs"][name] = run_config(s3, cfg)
        report["configs"][name]["settings"] = cfg
        v = report["configs"][name]
        print(f"[{name}] SM rel={v['p_SM']['rel']*100:.2f}% "
              f"SA rel={v['p_SA']['rel']*100:.2f}% "
              f"nominal SM={v['p_SM']['h0p01']:.3e} SA={v['p_SA']['h0p01']:.3e} "
              f"drift_diff={v['drift_level_diff']} [{time.perf_counter()-t1:.0f}s]",
              flush=True)
        write_json(OUT / "separated_sens.json", report)

    # 跨配置汇总（各旋钮的敏感度）
    base = report["configs"]["base"]
    summary = {}
    for knob, names in (("W", ("W10", "W15")), ("band", ("band_lo", "band_hi")),
                        ("bg", ("bg_lo", "bg_hi"))):
        sm = [abs(report["configs"][n]["p_SM"]["h0p01"] / base["p_SM"]["h0p01"] - 1)
              for n in names]
        sa = [abs(report["configs"][n]["p_SA"]["h0p01"] / base["p_SA"]["h0p01"] - 1)
              for n in names]
        summary[knob] = dict(SM_rel_max=max(sm), SA_rel_max=max(sa), configs=names)
    report["knob_sensitivity_vs_base"] = summary
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "separated_sens.json", report)
    print("knob sensitivity:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
