# -*- coding: utf-8 -*-
"""Phase A / G0-geometry：几何契约 gate（方案 v2.0 §9 G0）。

对 8 类 × 3 = 24 个候选执行：几何检查 → Metal 构建 → 网格 → 场解 → 导出哈希。
gate 通过条件：≥20 个候选全链路通过，且每类至少 1 个通过（类别不被整体筛掉时
需记录原因）。输出 results/phaseA_g0/report.json。
"""

import json
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.geometry import checks, library, two_pad  # noqa: E402

N_PER_CLASS = 3


def run_candidate(rec):
    """单候选全链路。返回 dict（含各环节状态与耗时）。"""
    out = dict(category=rec.category, index=rec.category_index,
               geometry_hash=rec.geometry_hash, stages={}, ok=False)

    t0 = time.time()
    ok, reasons = checks.check_record(rec)
    out["stages"]["checks"] = dict(ok=ok, reasons=reasons)
    if not ok:
        out["fail_stage"] = "checks"
        return out

    try:
        t0 = time.time()
        design, name = two_pad.build_design(rec)
        out["stages"]["metal_build"] = dict(ok=True, seconds=round(time.time() - t0, 2))

        from qiskit_metal.analyses.fem.solver import (
            Electrostatics, mesh_package, package_from_design,
        )
        t0 = time.time()
        pkg = package_from_design(design, eps_r=frozen.EPS_R_SUB)
        n_metal = len(pkg.metals)
        # 网格档与 G1 快速档一致但 h 放粗（G0 只验链路）
        mesh = mesh_package(pkg, "full", h_metal=0.12, h_max=0.8)
        out["stages"]["mesh"] = dict(ok=True, seconds=round(time.time() - t0, 2),
                                     n_metals=n_metal)

        t0 = time.time()
        est = Electrostatics(mesh, order=1)
        pa, pb = f"{name}.pad_left", f"{name}.pad_right"
        if pa not in est.names:
            pa, pb = est.names[0], est.names[1]
        C_q = float(est.port_capacitance(pa, pb) * 1e15)
        finite = (C_q > 1.0) and (C_q < 1000.0)
        out["stages"]["solve"] = dict(ok=bool(finite), seconds=round(time.time() - t0, 2),
                                      C_q_fF=round(C_q, 2))
        out["ok"] = bool(finite)
    except Exception as e:                                   # noqa: BLE001
        out["fail_stage"] = out["stages"].get("last", "pipeline")
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'phaseA_g0'/'report.json')
    args=parser.parse_args()
    t0 = time.time()
    recs = [library.make_record(cat, i)
            for cat in library.CATEGORY_NAMES for i in range(N_PER_CLASS)]
    results = [run_candidate(r) for r in recs]

    per_class = {}
    for r in results:
        per_class.setdefault(r["category"], []).append(r["ok"])
    n_pass = sum(r["ok"] for r in results)
    classes_ok = {c: sum(v) for c, v in per_class.items()}
    gate = n_pass >= 20 and all(v >= 1 for v in classes_ok.values())

    report = dict(
        gate="G0-geometry",
        passed=bool(gate),
        n_total=len(results), n_passed=int(n_pass),
        per_class=classes_ok,
        frozen_config=frozen.frozen_hash(),
        results=results,
    )
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"G0 {'PASSED' if gate else 'FAILED'}: {n_pass}/{len(results)} "
          f"({ {c: f'{v}/{len(per_class[c])}' for c, v in classes_ok.items()} })")
    print(f"report → {out}  [{time.time() - t0:.1f}s]")
    for r in results:
        if not r["ok"]:
            why = r.get("error") or r["stages"].get("checks", {}).get("reasons") \
                or r.get("fail_stage")
            print(f"  FAIL {r['category']}#{r['index']}: {why}")


if __name__ == "__main__":
    main()
