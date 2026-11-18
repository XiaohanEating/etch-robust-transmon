# -*- coding: utf-8 -*-
"""G1 第三步(b)：HFSS 逐面 11/13 轮差异定位 + 跨求解器对照表（纯存档后处理）。

1. 对名义/刻蚀± 的 11/13 轮 HFSS 报告做逐 sheet 参与比差分，
   定位 p_MA 8.21% 档位变化集中在哪些面（顶面/侧面/SM/SA）；
2. 汇总跨求解器对照表：HFSS(13轮) vs 快速层(h=0.005, 存档与 step-3a 修正值)
   的绝对参与比、有符号漂移（并列呈现，不预设何者为真值）；
3. 全部数值带来源哈希。输出 results/g1_step3b_hfss_2026-10-24/。
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.experiments.field_audit import sha256     # noqa: E402
from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "g1_step3b_hfss_2026-10-24"

HFSS = {
    "nominal": {11: ROOT / "results/hfss_finite_film_v1/200nm_refine11/report.json",
                13: ROOT / "results/hfss_finite_film_v1/200nm_refine13/report.json"},
    "etch_plus": {11: ROOT / "results/finite_alignment_v1/hfss/etch_plus_pass11/report.json",
                  13: ROOT / "results/finite_alignment_v1/hfss/etch_plus_pass13/report.json"},
    "etch_minus": {11: ROOT / "results/finite_alignment_v1/hfss/etch_minus_pass11/report.json",
                   13: ROOT / "results/finite_alignment_v1/hfss/etch_minus_pass13/report.json"},
}
FAST = {
    "nominal": ROOT / "results/finite_alignment_v1/rect_reference/h0p005/nominal",
    "etch_plus": ROOT / "results/finite_alignment_transport_v1/etch_plus",
    "etch_minus": ROOT / "results/finite_alignment_transport_v1/etch_minus",
}
STEP3A = ROOT / "results/g1_step3_rectangle_2026-10-23/step3_fast_rectangle.json"


def classify(sheet_name):
    if sheet_name.startswith("SM"):
        return "SM_bottom"
    if sheet_name.startswith("MA"):
        return "MA_top" if sheet_name.endswith("_4") else "MA_side"
    if sheet_name.startswith("SA"):
        return "SA_plane"
    return "other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=provenance(), scope="HFSS per-sheet 11/13 diff + cross-solver table")

    # 1) 逐 sheet 差异
    per_sheet = {}
    for scene, by_pass in HFSS.items():
        rows11 = json.loads(by_pass[11].read_text(encoding="utf-8"))["fields"]["surface_integrals"]
        rows13 = json.loads(by_pass[13].read_text(encoding="utf-8"))["fields"]["surface_integrals"]
        s11 = {r["object"]: r["participation"] for r in rows11}
        s13 = {r["object"]: r["participation"] for r in rows13}
        diffs = {}
        for name in s11:
            if name in s13 and s11[name]:
                diffs[name] = dict(cls=classify(name), p11=s11[name], p13=s13[name],
                                   rel=abs(s13[name] / s11[name] - 1))
        by_class = {}
        for name, d in diffs.items():
            c = by_class.setdefault(d["cls"], dict(n=0, max_rel=0.0, sum_p11=0.0, sum_p13=0.0))
            c["n"] += 1
            c["max_rel"] = max(c["max_rel"], d["rel"])
            c["sum_p11"] += d["p11"]
            c["sum_p13"] += d["p13"]
        for c in by_class.values():
            c["class_rel"] = (abs(c["sum_p13"] / c["sum_p11"] - 1)
                              if c["sum_p11"] else float("inf"))
        per_sheet[scene] = dict(sheets=diffs, by_class=by_class,
                                sources={int(k): sha256(v) for k, v in by_pass.items()})
        tot11 = sum(d["p11"] for d in diffs.values())
        print(f"[{scene}] 逐类档位差: "
              + ", ".join(f"{k}: {v['class_rel']*100:.2f}% (max sheet {v['max_rel']*100:.1f}%)"
                          for k, v in by_class.items()), flush=True)

    report["hfss_per_sheet"] = per_sheet

    # 2) 跨求解器对照表
    def hfss_pass(scene, p):
        r = json.loads(HFSS[scene][p].read_text(encoding="utf-8"))["fields"]
        return r["participation"], r["frequencies_GHz"][0]

    table = {}
    for scene in HFSS:
        p13, f13 = hfss_pass(scene, 13)
        fast = json.loads((FAST[scene] / "report.json").read_text(encoding="utf-8"))["metrics"]
        row = dict(
            hfss13=dict(p13, f_GHz=f13),
            fast_archive=dict(p_SM=fast["p_SM"], p_MA=fast["p_MA"], p_SA=fast["p_SA"],
                              T1_us=fast["T1_us"], C_q_fF=fast["C_q_fF"],
                              f01_GHz=fast["f01_GHz"]),
            sources=dict(fast=sha256(FAST[scene] / "report.json")))
        table[scene] = row

    # 有符号漂移并列
    def drift(d, key):
        import math
        return math.log(d[key] / d["nominal"][key]) if False else None

    for side, getter in (
            ("hfss13_T1", lambda r: 1.0 / (2 * math.pi * r["hfss13"]["f_GHz"] * 1e9
                                           * sum(r["hfss13"]["p_" + k] * 2e-3
                                                 for k in ("SM", "MA", "SA")))),
            ("fast_T1", lambda r: r["fast_archive"]["T1_us"])):
        base = getter(table["nominal"])
        for scene in ("etch_plus", "etch_minus"):
            import math
            table[scene][side + "_signed_log_drift"] = math.log(getter(table[scene]) / base)
    report["cross_solver"] = table
    if STEP3A.exists():
        report["step3a_corrected"] = {
            level: {s: dict(p_SM=v["p_SM_corrected"], p_SA=v["p_SA_corrected"])
                    for s, v in scenes.items()}
            for level, scenes in json.loads(
                STEP3A.read_text(encoding="utf-8"))["scenarios"].items()}
        report["step3a_sha256"] = sha256(STEP3A)

    write_json(OUT / "hfss_sheets.json", report)
    print(f"step-3b → {OUT/'hfss_sheets.json'}")


if __name__ == "__main__":
    main()
