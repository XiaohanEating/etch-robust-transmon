# -*- coding: utf-8 -*-
"""Phase A / G1-physics：物理标定 gate（方案 v2.0 §9 G1）。

(a) 论文对标：tutorial 4.4x 验证器件（30 mm 封装、quarter、P2、h_metal 0.05）
    复算结端口电容，对照文档记录 173.4–174.3 fF（±2% 判据）。
(b) 名义窗口：我们的冻结契约器件（rect 类 0 号）标称 C_Σ/f01/αf vs 电路窗口。
(c) 参与比方向性 + 网格敏感性：rect（r=0）vs rect_rounded（大圆角），
    检查 p_SM 是否随圆角下降（Eun 2022 同族机制），以及 h_metal∈{0.10, 0.08, 0.05}
    是否改变两候选的 p 排序（gate：排序翻转 → 不得放量）。
(d) 场景冻结 + 迷你漂移：S_search 全 7 场景在 rect vs rounded 上的 D_T1。

输出 results/phaseA_g1/report.json。
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "docs" / "tut" / "resources" / "package_modes"))

from robust_design_lab.configs import frozen                       # noqa: E402
from robust_design_lab.geometry import library                    # noqa: E402
from robust_design_lab.evaluator import fast, participation       # noqa: E402
from robust_design_lab.objectives import drift                    # noqa: E402
from robust_design_lab.perturbation import scenarios as scen      # noqa: E402


def paper_benchmark():
    """(a) tutorial 验证器件对标（paper 默认 30mm 封装单比特）。"""
    import package_modes as pm
    from qiskit_metal.analyses.fem.solver import Electrostatics, mesh_package

    design = pm.build_design(n=1)                     # 论文默认 10×10；单比特 30mm 封装
    pkg = pm.package_from_design(design, eps_r=11.9)
    mesh = mesh_package(pkg, "quarter", h_metal=0.05, h_max=0.8)
    est = Electrostatics(mesh, order=2)
    C_q = float(est.port_capacitance("Q_0_0.pad_left", "Q_0_0.pad_right") * 1e15)
    target, tol = 173.8, 0.02 * 173.8
    return dict(C_q_fF=round(C_q, 2), target_fF=target, tol_fF=round(tol, 2),
                passed=bool(abs(C_q - target) <= tol))


def _in_window_f01(rec, cache):
    fs = cache.get_or_solve(rec)
    from robust_design_lab.evaluator.lom import transmon_metrics
    tm = transmon_metrics(fs.C_q_fF * 1e-15, frozen.LJ0_NH * 1e-9)
    return tm["f01_GHz"], tm["alpha_MHz"]


def nominal_and_participation():
    """(b)(c) 冻结契约器件：名义指标 + 参与比方向性 + 网格敏感性。"""
    # 方向性对照：同尺寸满幅岛 r=0；圆角孪生取"窗口内最大圆角"保证频率窗公平
    W2_full = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2
    rect0 = library.rect_record(W2_full, frozen.FOOTPRINT_UM[1], 0.0, note="r=0")
    cache = fast.FieldCache()
    rect_r, r_used = None, 0.0
    for r in (50.0, 40.0, 30.0, 20.0, 10.0):
        cand = library.rect_record(W2_full, frozen.FOOTPRINT_UM[1], r, note=f"r={r:.0f}um")
        f01, alpha = _in_window_f01(cand, cache)
        if (frozen.WINDOW_F01_GHZ[0] <= f01 <= frozen.WINDOW_F01_GHZ[1]
                and frozen.WINDOW_ALPHA_MHZ[0] <= alpha <= frozen.WINDOW_ALPHA_MHZ[1]):
            rect_r, r_used = cand, r
            break
    if rect_r is None:
        rect_r = library.rect_record(W2_full, frozen.FOOTPRINT_UM[1], 20.0, note="r=20um(fallback)")

    rows = []
    for rec in (rect0, rect_r):
        fs = cache.get_or_solve(rec)
        p = participation.surface_participation(fs, rec)
        rows.append(dict(category=rec.category, note=rec.note, hash=rec.geometry_hash,
                         C_q_fF=round(fs.C_q_fF, 2),
                         f01_GHz=round(p["f01_GHz"], 4),
                         alpha_MHz=round(p["alpha_MHz"], 1),
                         p_SM=float(f"{p['p_SM']:.3e}"),
                         p_MA=float(f"{p['p_MA']:.3e}"),
                         p_SA=float(f"{p['p_SA']:.3e}"),
                         T1_us=round(p["T1_us"], 1)))
    nominal = rows[0]

    # 网格敏感性：对照对在 3 档网格下的 p_SM 排序是否稳定
    grid_rows = {}
    order_stable = True
    first_order = None
    for h in (0.10, 0.08, 0.05):
        vals = []
        for rec in (rect0, rect_r):
            fs = fast.solve_fields(rec, mesh_kwargs={"h_metal": h, "h_max": 0.8, "grow": 3.0})
            p = participation.surface_participation(fs, rec)
            vals.append(p["p_SM"])
        grid_rows[f"h={h}"] = [float(f"{v:.3e}") for v in vals]
        order = vals[0] < vals[1]
        first_order = order if first_order is None else first_order
        if order != first_order:
            order_stable = False

    window_ok = (frozen.WINDOW_F01_GHZ[0] <= nominal["f01_GHz"] <= frozen.WINDOW_F01_GHZ[1]
                 and frozen.WINDOW_ALPHA_MHZ[0] <= nominal["alpha_MHz"] <= frozen.WINDOW_ALPHA_MHZ[1])
    report = dict(nominal=nominal, r_used_um=r_used, calib_rows=rows,
                  grid_sensitivity=grid_rows,
                  p_order_stable=bool(order_stable), window_feasible=bool(window_ok))
    return report, rect0, rect_r


def mini_drift(rect0, rect_r):
    """(d) 场景冻结 + S_search 上 r=0 vs 圆角孪生的 D_T1（含几何通道分项）。"""
    frozen_scen = scen.freeze()
    S = frozen_scen["S_search"]
    # 几何通道子集：仅刻蚀场景（方案 §2.3 词典序第 2 项）
    idx_geom = [i for i, s in enumerate(S) if abs(s.get("etch_um", 0.0)) > 1e-9]

    out = {}
    for tag, rec in (("rect_r0", rect0), ("rect_rounded", rect_r)):
        cache = fast.FieldCache()
        nom = drift.evaluate_scene(rec, S[0], cache)
        res = {f"s{i}": drift.evaluate_scene(rec, s, cache) for i, s in enumerate(S)}
        d_all = drift.drift_metrics(nom, res)
        d_geom = drift.drift_metrics(nom, {f"s{i}": res[f"s{i}"] for i in idx_geom})
        out[tag] = dict(nominal=nom, drift_all=d_all, drift_geom=d_geom)
    return dict(scenarios_sha256=frozen_scen["sha256"], drift=out)


def main():
    t0 = time.time()
    report = dict(gate="G1-physics", frozen_config=frozen.frozen_hash())

    print("[G1a] paper benchmark (quarter, P2, h=0.05) ...", flush=True)
    report["paper_benchmark"] = paper_benchmark()
    print(f"      C_q = {report['paper_benchmark']['C_q_fF']} fF "
          f"(target 173.8±3.5) → {'PASS' if report['paper_benchmark']['passed'] else 'FAIL'}",
          flush=True)

    print("[G1b/c] nominal + participation + grid sensitivity ...", flush=True)
    physics, rect0, rect_r = nominal_and_participation()
    report["physics"] = physics
    print(f"      nominal f01={physics['nominal']['f01_GHz']} GHz, "
          f"alpha={physics['nominal']['alpha_MHz']} MHz, r_used={physics['r_used_um']}um, "
          f"window_feasible={physics['window_feasible']}, "
          f"p_order_stable={physics['p_order_stable']}", flush=True)
    for row in physics["calib_rows"]:
        print(f"      {row['note']}: C_q={row['C_q_fF']} fF, p_SM={row['p_SM']:.3e}, "
              f"p_MA={row['p_MA']:.3e}, p_SA={row['p_SA']:.3e}", flush=True)

    print("[G1d] scenarios freeze + mini drift (7 scenes × 2 candidates) ...", flush=True)
    report["mini_drift"] = mini_drift(rect0, rect_r)
    for tag, d in report["mini_drift"]["drift"].items():
        print(f"      {tag}: D_T1(S_all)={d['drift_all']['D_T1']['max']:.4f} "
              f"D_T1(S_geom)={d['drift_geom']['D_T1']['max']:.4f} "
              f"worst_geom={d['drift_geom']['D_T1']['worst']}", flush=True)

    pb = report["paper_benchmark"]["passed"]
    gate = pb and report["physics"]["p_order_stable"]
    report["passed"] = bool(gate)
    out = ROOT / "results" / "phaseA_g1" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"G1 {'PASSED' if gate else 'FAILED'} → {out}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
