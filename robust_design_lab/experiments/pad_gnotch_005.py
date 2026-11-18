# -*- coding: utf-8 -*-
"""顺序 3（复核 R5）：双岛起点与 G_notch 候选的 0.05% 加严对照。

同几何、同地、同结参数，六场景（两对象 × 三场景，0.05%/30 轮起步，
实际收敛状态决定接收）。比较：改善量在两档（0.1% vs 0.05%）间的变化、
两个刻蚀方向分量、排序是否保持。0.05% 是一次加严检查，不预设为真值。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.experiments import verify_candidates as vc
from robust_design_lab.geometry import boundary_edit as be

OUT = ROOT / "results" / "refs_b_2026-11-06"
OUT.mkdir(parents=True, exist_ok=True)

PCT = 0.05
MP = 30

pad = vc.simple_dual_pad(w=734.0, h=218.0)
objs = {
    "pad_start": pad,
    "pad_G_notch": be.apply_edit(pad, "notch",
                                 be.seg_encode("R", "outer", 1), 0.5, 0.5),
}
ref01 = {
    "pad_start": json.loads((OUT / "refs_b_pad_ref.json").read_text(
        encoding="utf-8"))["row"],
    "pad_G_notch": json.loads((OUT.parent / "mech_c_2026-11-06" /
                               "mech_s2_pad_ref_G_notch.json").read_text(
                                  encoding="utf-8"))["row"],
}
res = {}
for name, struct in objs.items():
    p = OUT / f"pct005_{name}.json"
    if p.exists():
        res[name] = json.loads(p.read_text(encoding="utf-8"))["row"]
        continue
    t0 = time.perf_counter()
    try:
        row = vc.evaluate_object(
            name, dict(kind="edited", struct=struct, kw={},
                       source="R5 加严对照 0.05%%"),
            max_passes=MP, refine=True, percent_error=PCT)
    except RuntimeError as e:
        print(f"{name}: 拒收——{e}", flush=True)
        continue
    p.write_text(json.dumps(dict(row=row), ensure_ascii=False, default=str,
                            indent=1), encoding="utf-8")
    res[name] = row
    print(f"{name}: D={row['D']:.6f} C={row['C_sigma_fF']:.3f} "
          f"dep={row['d_ep']:.6f} dem={row['d_em']:.6f} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

if len(res) == 2:
    d0 = res["pad_start"]["D"]
    d1 = res["pad_G_notch"]["D"]
    r01 = ref01["pad_start"]["D"], ref01["pad_G_notch"]["D"]
    print("\n== 两档对照 ==")
    print("0.1%% : 起点 %.6f → G_notch %.6f（改善 %.2f%%）" % (
        r01[0], r01[1], (r01[1] - r01[0]) / r01[0] * 100))
    print("0.05%%: 起点 %.6f → G_notch %.6f（改善 %.2f%%）" % (
        d0, d1, (d1 - d0) / d0 * 100))
    (OUT / "pct005_compare.json").write_text(json.dumps(dict(
        pct=PCT, max_passes=MP,
        fine={k: dict(D=v["D"], C=v["C_sigma_fF"], d_ep=v["d_ep"],
                      d_em=v["d_em"],
                      conv={t: s["converged"] for t, s in
                            v["scenarios"].items()}) for k, v in res.items()},
        coarse={k: dict(D=v["D"], C=v["C_sigma_fF"], d_ep=v["d_ep"],
                        d_em=v["d_em"]) for k, v in ref01.items()}),
        ensure_ascii=False, indent=1), encoding="utf-8")
