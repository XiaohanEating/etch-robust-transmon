# -*- coding: utf-8 -*-
"""G1 第一步：HFSS 取场侧探针（复制已存命名薄片并沿 z 平移，不新建几何）。

在已存 13 轮名义矩形项目上（不求解、不保存，保持 .aedt 哈希不变）：
- SA 整面薄片复制两份，z 平移 ∓/±0.5μm（基片侧/空气侧），积分 Et²、Dn²；
  与在面（z=z0）原薄片同参积分对照 → 判定 AEDT 面积分的取场侧。
- SM 两底面薄片复制到 z0−0.5μm，积分 |E|²，与在面值对照 → 边界取值约定。
快速层侧：同一存档场解，在 z0±0.5μm 全平面（6μm 网格）与在面单元迹
计算同口径积分。输出 results/g1_step1_side_2026-10-23/hfss_side_probe.json。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

EPS0 = 8.8541878128e-12
OUT = ROOT / "results" / "g1_step1_side_2026-10-23"
HFSS_DIR = ROOT / "results" / "hfss_finite_film_v1" / "200nm_refine13"
FAST_DIR = ROOT / "results" / "finite_alignment_v1" / "rect_reference" / "h0p005" / "nominal"
Z0 = float(frozen.SUBSTRATE_MM)
DZ = 0.0005      # 0.5 μm


def _integrate(setup, calc):
    return float(calc.evaluate())


def hfss_phase():
    from pyEPR.ansys import CalcObject
    from robust_design_lab.evaluator.ansys_calibration import connect
    from robust_design_lab.experiments.hfss_interfaces import find_project

    report13 = json.loads((HFSS_DIR / "report.json").read_text(encoding="utf-8"))
    denominator = (report13["fields"]["bulk_electric_energy_integral"]
                   + report13["fields"]["junction_electric_energy_integral"])

    desktop = connect()
    project = find_project(desktop, HFSS_DIR / "finite_film_200p0nm_refine13.aedt")
    design = project.get_design("FiniteFilm")
    setup = design.get_setup()
    setup.get_solutions().set_mode(1)
    editor = design._modeler

    def qty(n):
        return CalcObject([], setup).getQty(n)

    def scalar_abs2(expr):
        return (expr.conj() * expr).real()

    def sa_integrals(name):
        et2 = _integrate(setup, (scalar_abs2(qty("E").scalar_x())
                                 + scalar_abs2(qty("E").scalar_y())).integrate_surf(name))
        dn2 = _integrate(setup, scalar_abs2(qty("E").times_eps().scalar_z())
                         .integrate_surf(name))
        return et2, dn2

    def sm_integrals(name):
        return _integrate(setup, qty("E").norm_2().integrate_surf(name))

    def copy_move(src, dz, tag):
        """复制薄片 → 先 z 平移 → 后改名（顺序颠倒会因旧名失效而报错）。"""
        before = set(editor.GetObjectsInGroup("Sheets"))
        editor.Copy(["NAME:Selections", "Selections:=", src])
        editor.Paste()
        created = sorted(set(editor.GetObjectsInGroup("Sheets")) - before)
        if len(created) != 1:
            raise RuntimeError(f"paste produced {created}")
        new = created[0]
        editor.Move(["NAME:Selections", "Selections:=", new,
                     "NewPartsModelFlag:=", "NonModel"],
                    ["NAME:TranslateParameters", "TranslateVectorX:=", "0mm",
                     "TranslateVectorY:=", "0mm",
                     "TranslateVectorZ:=", f"{dz}mm"])
        editor.ChangeProperty(["NAME:AllTabs",
                               ["NAME:Geometry3DAttributeTab", ["NAME:PropServers", new],
                                ["NAME:ChangedProps", ["NAME:Name", "Value:=", tag]]]])
        return tag

    results = dict(SA={}, SM={})
    leftovers = []
    try:
        # 清理上次中断可能残留的探针副本
        stale = [s for s in editor.GetObjectsInGroup("Sheets")
                 if s.startswith("SA_probe") or s.startswith("SM_probe") or s == "SA1"]
        if stale:
            editor.Delete(["NAME:Selections", "Selections:=", ",".join(stale)])

        # 在面基线（与已存报告同参，重新量一遍以排除会话差异）
        et2, dn2 = sa_integrals("SA")
        results["SA"]["on_plane"] = dict(Et2=et2, Dn2=dn2)
        results["SM"]["on_plane"] = dict(
            E2_left=sm_integrals("SM_PadLeft_0"), E2_right=sm_integrals("SM_PadRight_0"))

        for side, dz in (("substrate", -DZ), ("air", +DZ)):
            name = copy_move("SA", dz, f"SA_probe_{side}")
            et2, dn2 = sa_integrals(name)
            results["SA"][side + "_offset0p5um"] = dict(Et2=et2, Dn2=dn2, sheet=name)
            print(f"[HFSS] SA {side}: Et2={et2:.4e} Dn2={dn2:.4e}", flush=True)

        name = copy_move("SM_PadLeft_0", -DZ, "SM_probe_left")
        results["SM"]["substrate_offset0p5um_left"] = dict(
            E2=sm_integrals(name), sheet=name)
        name = copy_move("SM_PadRight_0", -DZ, "SM_probe_right")
        results["SM"]["substrate_offset0p5um_right"] = dict(
            E2=sm_integrals(name), sheet=name)
        print(f"[HFSS] SM on: L={results['SM']['on_plane']['E2_left']:.4e} "
              f"R={results['SM']['on_plane']['E2_right']:.4e}; "
              f"sub-off: L={results['SM']['substrate_offset0p5um_left']['E2']:.4e} "
              f"R={results['SM']['substrate_offset0p5um_right']['E2']:.4e}", flush=True)
    finally:
        probes = [v.get("sheet") for grp in results.values() for v in grp.values()
                  if isinstance(v, dict) and "sheet" in v]
        if probes:
            try:
                editor.Delete(["NAME:Selections", "Selections:=", ",".join(probes)])
            except Exception:                                 # noqa: BLE001
                pass
    return results, denominator


def fast_phase():
    """快速层：在面单元迹（已有）与 z0±0.5μm 全平面偏移积分（6μm 网格）。"""
    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.evaluator.fields import ElectrostaticField
    from robust_design_lab.evaluator.lom import transmon_metrics
    import shapely

    fs, rec, _ = load_solution(FAST_DIR)
    field = ElectrostaticField(fs)
    lj = json.loads((FAST_DIR / "report.json").read_text(encoding="utf-8"))["lj_H"]
    tm = transmon_metrics(fs.C_q_fF * 1e-15, lj)
    denominator = 2 * field.energy * tm["C_sigma_fF"] / fs.C_q_fF

    lx, ly = (float(v) for v in frozen.SEARCH_BOX_MM[:2])
    step = 0.006
    xs = np.arange(0.0, lx + step / 2, step)
    ys = np.arange(0.0, ly + step / 2, step)
    gx, gy = np.meshgrid(xs, ys)
    out = dict(grid_step_mm=step, n_points=int(gx.size))

    for label, z in (("substrate", Z0 - DZ), ("air", Z0 + DZ)):
        pts = np.vstack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
        E = field.electric(pts) * 1e3                     # V/m（该侧单元）
        eps_local = frozen.EPS_R_SUB if z < Z0 else 1.0
        et2 = float(np.sum(E[:2] ** 2, axis=0).mean() * lx * ly * 1e-6)
        dn2 = float(((eps_local * E[2]) ** 2).mean() * lx * ly * 1e-6)
        out[label + "_offset0p5um"] = dict(Et2=et2, Dn2=dn2)
        print(f"[FAST] SA {label}: Et2={et2:.4e} Dn2={dn2:.4e}", flush=True)

    # SM 足迹（两岛）在 z0−0.5μm 的 |E|² 积分（金属掩膜 + 左右分解）
    z = Z0 - DZ
    pts = np.vstack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    E = field.electric(pts) * 1e3
    x_um = (gx.ravel() - fs.origin_xy_mm[0]) * 1e3
    y_um = (gy.ravel() - fs.origin_xy_mm[1]) * 1e3
    cell = step * step * 1e-6
    for name, poly in (("left", rec.pad_left), ("right", rec.pad_right)):
        m = shapely.contains_xy(poly, x_um, y_um)
        out[f"SM_{name}_offset0p5um"] = dict(
            E2=float((E ** 2).sum(axis=0)[m].sum() * cell),
            n_points=int(m.sum()))
    print(f"[FAST] SM sub-off: L={out['SM_left_offset0p5um']['E2']:.4e} "
          f"R={out['SM_right_offset0p5um']['E2']:.4e}", flush=True)
    return out, denominator


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = dict(provenance=provenance(), scope="HFSS field-side probe (copy+move, no save)",
                  frozen_config=frozen.frozen_hash())
    hfss_rows, hfss_denom = hfss_phase()
    report["hfss"] = dict(results=hfss_rows, denominator=hfss_denom)
    fast_rows, fast_denom = fast_phase()
    report["fast"] = dict(results=fast_rows, denominator=fast_denom)
    write_json(OUT / "hfss_side_probe.json", report)
    print(f"side probe → {OUT/'hfss_side_probe.json'}  [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
