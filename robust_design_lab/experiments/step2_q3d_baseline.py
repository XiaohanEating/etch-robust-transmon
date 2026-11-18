# -*- coding: utf-8 -*-
"""第二步：官方 Q3D→LOM 完整器件基线（progress_review 四步计划第二步）。

契约（docs/tut 4.01 官方案例，native_q3d_lom_review_2026-10-23 已核）：
- 几何：4×TransmonPocket 官方布局（Q1..Q4），仅渲染 Q1；
- 端接：open_terminations = readout/bus1/bus2（网络身份 6 网固定序）；
- 求解：max_passes=6, solution_order=Medium, auto_increase=False；
- LOM：Lj=12.31 nH、Cj=2 fF、读出 7 GHz、bus [6.0, 6.2] GHz——
  **三场景共用同一结参数**（刻蚀不重配结）；
- 刻蚀实现（参数化组件上的近似，登记）：pad_width/pad_height 各 ∓0.5 μm、
  pad_gap ±0.5 μm（每边 0.25 μm；连接垫与地面不动）。

交付：原始矩阵（CSV+JSON）、网络身份断言、量子参数、Purcell 寿命、
成本（每求解壁钟）、假设清单。与界面 G1 分开报告。
输出 results/step2_q3d_baseline_2026-10-24/。
"""

import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step2_q3d_baseline_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)

NOMINAL = dict(pad_width=425.0, pad_height=90.0, pad_gap=30.0)   # μm
SCENARIOS = {
    "nominal": NOMINAL,
    "etch_plus": dict(pad_width=424.5, pad_height=89.5, pad_gap=30.5),
    "etch_minus": dict(pad_width=425.5, pad_height=90.5, pad_gap=29.5),
}
LJ_NH, CJ_FF = 12.31, 2.0
READOUT_GHZ, BUS_GHZ = 7.0, [6.0, 6.2]


def build_design(dims):
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

    design = designs.design_planar.DesignPlanar()
    design.overwrite_enabled = True
    base = dict(pad_width=f"{dims['pad_width']}um",
                pad_height=f"{dims['pad_height']}um",
                pad_gap=f"{dims['pad_gap']}um",
                pocket_height="650um",
                connection_pads=dict(
                    readout=dict(loc_W=+1, loc_H=-1, pad_width="200um"),
                    bus1=dict(loc_W=-1, loc_H=+1, pad_height="30um"),
                    bus2=dict(loc_W=-1, loc_H=-1, pad_height="50um")))
    TransmonPocket(design, "Q1", options=dict(pos_x="+2.42251mm", **base))
    TransmonPocket(design, "Q2", options=dict(
        pos_x="+0.0mm", pos_y="-0.95mm", orientation="270", **base))
    TransmonPocket(design, "Q3", options=dict(
        pos_x="-2.42251mm", pos_y="+0.0mm", orientation="180", **base))
    TransmonPocket(design, "Q4", options=dict(
        pos_x="+0.0mm", pos_y="+0.95mm", orientation="90", **base))
    design.rebuild()
    return design


def run_scenario(name, dims):
    """断点续跑：matrix CSV 已存则免解复用，只做 LOM。"""
    import numpy as np
    import pandas as pd
    from qiskit_metal.analyses.quantization import LOManalysis

    design = build_design(dims)
    c1 = LOManalysis(design, "q3d")
    c1.sim.setup.max_passes = 6
    c1.sim.setup_update(solution_order="Medium", auto_increase_solution_order="False")


    csv = OUT / f"matrix_{name}.csv"
    if not csv.exists():
        # pyEPR 不创建新工程：先经 COM 新建空工程并另存到 OUT，再连接
        # （绕开长驻会话里陈旧活动工程导致的 SetActiveProject 失败）
        from robust_design_lab.evaluator.ansys_calibration import connect
        desk = connect()._desktop
        desk.NewProject()
        newp = desk.GetActiveProject()
        newp.SaveAs(str(OUT / f"step2_q3d_{name}.aedt"), True)
        c1.sim.renderer.connect_ansys(project_path=str(OUT),
                                      project_name=f"step2_q3d_{name}")
    if csv.exists():
        cm = pd.read_csv(csv, index_col=0)
        c1.sim.capacitance_matrix = cm
        solve_s = None
    else:
        t0 = time.perf_counter()
        c1.sim.run(components=["Q1"],
                   open_terminations=[("Q1", "readout"), ("Q1", "bus1"), ("Q1", "bus2")])
        solve_s = time.perf_counter() - t0
        cm = c1.sim.capacitance_matrix
        cm.to_csv(csv)
    nets = list(cm.columns)

    c1.setup.junctions.Lj = LJ_NH
    c1.setup.junctions.Cj = CJ_FF
    c1.setup.freq_readout = READOUT_GHZ
    c1.setup.freq_bus = BUS_GHZ
    buf = io.StringIO()
    t1 = time.perf_counter()
    with redirect_stdout(buf):
        c1.run_lom()
    lom_s = time.perf_counter() - t1
    stdout = buf.getvalue()

    lo = c1.lumped_oscillator          # 0.9.0 实现为扁平 dict（fQ/EC/EJ/alpha/...）
    lumped = {k: (float(v) if np.isscalar(v) or getattr(v, "ndim", 0) == 0
                  else np.asarray(v).tolist())
              for k, v in lo.items()}
    # Purcell：总值行单独读取；独立分项（带名字后缀者）调和合并作互证。
    # 旧版把无后缀总值行混入分项导致重复合并（数值约减半），已修。
    m_total = re.search(r"^T1\s+([0-9.]+)\s*\[?us\]?", stdout, re.M)
    total = float(m_total.group(1)) if m_total else None
    per_r = {lbl: float(val) for lbl, val in
             re.findall(r"T1([A-Za-z0-9_]+)\s+([0-9.]+)\s*\[?us\]?", stdout)}
    purcell = 1.0 / sum(1.0 / v for v in per_r.values()) if per_r else None
    if total is not None and purcell is not None:
        if abs(total - purcell) > 1e-3 * max(total, purcell):
            raise ValueError(f"Purcell 总值 {total} 与分项合并 {purcell:.6f} 不一致")
    row = dict(
        scenario=name, nets=nets, matrix_fF=cm.values.tolist(),
        solve_s=(round(solve_s, 1) if solve_s is not None else "reused"),
        lom_s=round(lom_s, 1), lumped=lumped,
        per_resonator_T1_us=per_r, purcell_total_us=purcell,
    )
    (OUT / f"stdout_{name}.txt").write_text(stdout, encoding="utf-8")
    write_json(OUT / f"scenario_{name}.json", row)
    print(f"[{name}] solve={row['solve_s']} lom={lom_s:.0f}s nets={len(nets)} "
          f"fQ={lumped.get('fQ')} purcell={purcell}", flush=True)
    return row


def main():
    t0 = time.perf_counter()
    report = dict(provenance=provenance(),
                  contract=dict(
                      geometry="official 4.01 four-transmon layout, render Q1 only",
                      open_terminations=["readout", "bus1", "bus2"],
                      q3d_setup=dict(max_passes=6, solution_order="Medium",
                                     auto_increase=False),
                      Lj_nH=LJ_NH, Cj_fF=CJ_FF, readout_GHz=READOUT_GHZ,
                      bus_GHz=BUS_GHZ,
                      etch_realization=("pad_width/pad_height ∓0.5μm, pad_gap ±0.5μm "
                                        "(每边 0.25μm)；连接垫与地面不动（参数化组件近似，登记）"),
                      loss_model=("Purcell: Q_readout=1e4, Q_bus=1e5, 谐振器阻抗 50Ω "
                                  "（源码假设，非矩阵导出）")),
                  scenarios={})
    rows = {}
    for name, dims in SCENARIOS.items():
        rows[name] = run_scenario(name, dims)
        report["scenarios"][name] = {k: v for k, v in rows[name].items()
                                     if k not in ("matrix_fF", "stdout_tail")}

    # 网络身份断言
    nets0 = rows["nominal"]["nets"]
    identity_stable = all(r["nets"] == nets0 for r in rows.values())
    report["network_identity_stable"] = bool(identity_stable)
    report["nets"] = nets0

    # 器件级漂移（同 Lj）
    import math
    f0 = rows["nominal"]["lumped"]["fQ"]
    drift = {}
    for s in ("etch_plus", "etch_minus"):
        fs = rows[s]["lumped"]["fQ"]
        drift[s] = dict(fQ_GHz=fs, signed_log_drift=math.log(fs / f0))
    report["device_drift"] = drift
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    report["G1_passed"] = False      # 器件基线与界面 G1 分开，不据此改 G1
    write_json(OUT / "baseline.json", report)
    print(f"network identity stable = {identity_stable}; "
          f"drift = {({k: round(v['signed_log_drift'], 5) for k, v in drift.items()})}; "
          f"wall={report['wall_s']}s → {OUT/'baseline.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                    help="单场景独立进程运行（规避跨场景 COM 会话状态）")
    args = ap.parse_args()
    if args.scenario:
        run_scenario(args.scenario, SCENARIOS[args.scenario])
    else:
        main()
