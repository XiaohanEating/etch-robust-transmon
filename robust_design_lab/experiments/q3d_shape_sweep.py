# -*- coding: utf-8 -*-
"""Q3D 参数扫描：四类电容形状 × 参数变异 × 刻蚀扰动。

对每类形状（A双岛/B十字/B星形/C同心/D交指）各 N 个参数变体，
每个变体跑名义+刻蚀±三场景 Q3D，从电容矩阵直接算 C_Σ → f_01 → 漂移。

C_Σ 计算（通用，不依赖 extract 函数）：
- 双岛类（A/C/D）：C_Σ = C_12 + Cj + Σ_k C_1k·C_2k/(C_1k+C_2k)
- 单岛类（B）：C_Σ = C_ii（Maxwell 自电容）+ Cj

f_01 = √(8·EJ·EC)/h − EC/h    EC = e²/(2C_Σ)
α ≈ −EC/h

刻蚀模拟：各组件的关键几何参数 ±0.5μm（同 Q3D 基线方法）。
输出 results/q3d_shape_sweep_2026-10-26/sweep.json（逐条追加）。
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
import io
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "q3d_shape_sweep_2026-10-26"
OUT.mkdir(parents=True, exist_ok=True)
CHECK = OUT / "sweep.jsonl"

# 物理常数
PHI0 = 2.067833848e-15
H_PLANCK = 6.62607015e-34
E_CHARGE = 1.602176634e-19
LJ_DEFAULT = 12.31e-9  # nH → H
CJ_DEFAULT = 2.0e-15   # fF → F

# 刻蚀参数变化量（μm）
ETCH_DELTA = 0.5


def perturb(opts, family, delta_um):
    """对组件参数施加刻蚀偏移。"""
    o = dict(opts)  # shallow copy
    if family == "A_dual_pad":
        o["pad_width"] = _shift_um(o.get("pad_width", "400um"), -delta_um, "wider")
        o["pad_height"] = _shift_um(o.get("pad_height", "90um"), -delta_um, "taller")
        o["pad_gap"] = _shift_um(o.get("pad_gap", "30um"), 2*delta_um, "wider")
    elif family == "B_cross":
        o["cross_width"] = _shift_um(o.get("cross_width", "30um"), -delta_um)
        o["cross_length"] = _shift_um(o.get("cross_length", "200um"), -delta_um)
        o["cross_gap"] = _shift_um(o.get("cross_gap", "30um"), 2*delta_um)
    elif family == "B_star":
        o["arm_width"] = _shift_um(o.get("arm_width", "30um"), -delta_um)
        o["arm_length"] = _shift_um(o.get("arm_length", "150um"), -delta_um)
        o["arm_gap"] = _shift_um(o.get("arm_gap", "30um"), 2*delta_um)
    elif family == "C_concentric":
        # 刻蚀 → 半径增大（金属缩）→ 间隙增大
        o["rad_i"] = _shift_um(o.get("rad_i", "100um"), -delta_um)
        o["rad_o"] = _shift_um(o.get("rad_o", "170um"), delta_um)
        o["gap"] = _shift_um(o.get("gap", "30um"), 2*delta_um)
    elif family == "D_interdigit":
        o["finger_width"] = _shift_um(o.get("finger_width", "20um"), -delta_um)
        o["finger_space"] = _shift_um(o.get("finger_space", "20um"), 2*delta_um)
        o["comb_width"] = _shift_um(o.get("comb_width", "25um"), -delta_um)
    return o


def _shift_um(val_str, delta_um, _dir=""):
    """在字符串参数上偏移 ±μm。"""
    if not isinstance(val_str, str) or "um" not in val_str:
        return val_str
    num = float(val_str.replace("um", ""))
    new = num + delta_um
    return f"{max(new, 5):.1f}um"


_DESKTOP_CACHE = {"app": None}


def get_desktop():
    """进程内缓存 Ansys Desktop 连接。

    实测每次求解墙钟 ~240s 中分析只占 ~59s，大头是 COM 重连；
    缓存连接后求解间隔大幅缩短。失效（Ansys 被关等 COM 异常）时重连；
    NewProject 每次求解照旧，保留会话残留隔离语义。
    """
    from robust_design_lab.evaluator.ansys_calibration import connect
    if _DESKTOP_CACHE["app"] is not None:
        try:
            _DESKTOP_CACHE["app"]._desktop.GetVersion()
            return _DESKTOP_CACHE["app"]
        except Exception:
            _DESKTOP_CACHE["app"] = None
    _DESKTOP_CACHE["app"] = connect()
    return _DESKTOP_CACHE["app"]


def run_q3d(variant, comp_name, family, name="Q1", max_passes=6,
            percent_error=None, output_dir=None, legacy_c_sigma=True):
    """单次 Q3D 求解 → (C_matrix, LOM_dict)。

    max_passes 可调（默认 6 与历史口径一致）。真加密须**降低
    percent_error 误差目标**（默认 0.5%）：只提高 max_passes 时求解器
    达到原目标即停，不产生独立精度证据（2026-11-04 审阅反例：环形
    12 轮上限仍在第 6 轮 0.5% 目标处停止）。逐轮电容与收敛状态随
    记录落盘（逐轮矩阵已换算为 fF，见 passes_units）。"""
    from qiskit_metal import designs
    from qiskit_metal.analyses.quantization import LOManalysis
    from robust_design_lab.geometry.four_shape_families import build_design_for

    destination = Path(output_dir) if output_dir is not None else OUT
    destination.mkdir(parents=True, exist_ok=True)

    design = build_design_for(variant, comp_name, name)
    c1 = LOManalysis(design, "q3d")
    c1.sim.setup.max_passes = max_passes
    setup_kw = dict(solution_order="Medium",
                    auto_increase_solution_order="False")
    if percent_error is not None:
        setup_kw["percent_error"] = float(percent_error)
    c1.sim.setup_update(**setup_kw)

    # 新工程（绕开会话残留）
    desktop = get_desktop()
    desk = desktop._desktop
    desk.NewProject()
    newp = desk.GetActiveProject()
    proj_name = f"sweep_{family}_{int(time.time())}"
    newp.SaveAs(str(destination / f"{proj_name}.aedt"), True)
    c1.sim.renderer.connect_ansys(project_path=str(destination),
                                  project_name=proj_name)

    # open_terminations 按组件类型
    terms = []
    if "connection_pads" in variant:
        for k in variant["connection_pads"]:
            terms.append((name, k))
    if not terms:
        # 无连接垫的组件（如 Concentric）不传 terminations
        pass

    t0 = time.perf_counter()
    try:
        if terms:
            c1.sim.run(components=[name], open_terminations=terms)
        else:
            c1.sim.run(components=[name])
    except Exception:
        # Preserve a failed project and release only the project created here.
        try:
            newp.Save()
        finally:
            desk.CloseProject(proj_name)
        raise
    solve_s = time.perf_counter() - t0

    cm = c1.sim.capacitance_matrix
    nets = list(cm.columns)
    mat = cm.values  # fF

    # 逐轮电容与实际收敛状态（原生接口；必须在 CloseProject 前读取，
    # 取不到时如实置 None，不虚构收敛证据——2026-11-02 审阅要求）。
    # capacitance_all_passes / is_converged 均为**属性**（非方法）。
    # 单位契约（2026-11-04 审阅修正）：原生 get_capacitance_all_passes
    # 返回 **farad**，此处统一换算为 fF 并以 passes_units="fF" 标注；
    # 旧记录无该字段者其逐轮值为 F，读取方须 ×1e15。
    import numpy as _np
    passes, converged = None, None
    passes_units = None
    try:
        cap_obj = c1.sim.capacitance_all_passes
        if isinstance(cap_obj, dict):
            passes = {str(k): (_np.asarray(v) * 1e15).tolist()
                      for k, v in cap_obj.items()}
            passes_units = "fF"
    except Exception:
        pass
    try:
        converged = c1.sim.is_converged
        converged = bool(converged) if converged is not None else None
    except Exception:
        pass

    # 求解后保存工程（结果随工程落盘、场解可恢复），再关闭释放内存。
    # 旧版仅求解前 SaveAs，关闭后工程文件里没有求解结果
    # （2026-11-02 审阅：4824 字节空壳工程不能当可恢复证据）。
    project_saved = False
    save_error = None
    try:
        newp.Save()
        project_saved = True
    except Exception as exc:
        save_error = repr(exc)
    try:
        desk.CloseProject(proj_name)
    except Exception:
        pass

    # 计算 C_Σ
    c_sigma = compute_c_sigma(mat, nets, family) if legacy_c_sigma else None
    return dict(nets=nets, matrix_fF=mat.tolist(), C_sigma_fF=c_sigma,
                solve_s=round(solve_s, 1), project=proj_name,
                project_path=str(destination / f"{proj_name}.aedt"),
                project_saved=project_saved, project_save_error=save_error,
                capacitance_all_passes_fF=passes, passes_units=passes_units,
                percent_error=(float(percent_error)
                               if percent_error is not None else 0.5),
                converged=converged)


def compute_c_sigma(mat_fF, nets, family):
    """从 Maxwell 电容矩阵计算 C_Σ（fF）。"""
    import numpy as np
    mat = np.array(mat_fF)
    n = len(nets)

    # 找 pad 网络
    if family in ("A_dual_pad", "D_interdigit"):
        # 双岛：找主 pad（排除连接垫——连接垫名含 "connector" 或 "readout"/"bus"）
        pad_nets = [i for i, net in enumerate(nets)
                    if "pad" in net.lower()
                    and "connector" not in net.lower()
                    and "readout" not in net.lower()
                    and "bus" not in net.lower()]
        if len(pad_nets) < 2:
            # 备选：非 ground 中互电容最大的一对
            non_gnd = [i for i, net in enumerate(nets) if "ground" not in net.lower()]
            best = (0, -1, -1)
            for a in range(len(non_gnd)):
                for b in range(a+1, len(non_gnd)):
                    c = -mat[non_gnd[a]][non_gnd[b]]
                    if c > best[0]:
                        best = (c, non_gnd[a], non_gnd[b])
            p1, p2 = best[1], best[2]
        else:
            p1, p2 = pad_nets[0], pad_nets[1]
        C12 = -mat[p1][p2]  # 互电容（正值）
        c_sig = C12 + CJ_DEFAULT * 1e15
        # 各外部导体的串联贡献
        for k in range(n):
            if k in (p1, p2):
                continue
            c1k = max(-mat[p1][k], 0)
            c2k = max(-mat[k][p2], 0)
            if c1k + c2k > 0:
                c_sig += c1k * c2k / (c1k + c2k)
        return float(c_sig)

    elif family in ("B_cross", "B_star"):
        # 单岛：自电容即总电容
        # 找最大的对角元素（排除 ground）
        island_idx = None
        for i, net in enumerate(nets):
            if "ground" in net.lower():
                continue
            if island_idx is None or mat[i][i] > mat[island_idx][island_idx]:
                island_idx = i
        if island_idx is None:
            island_idx = 0
        return float(mat[island_idx][island_idx] + CJ_DEFAULT * 1e15)

    elif family == "C_concentric":
        # 同心圆：内盘+外环，类似双岛
        # 内盘通常自电容较小
        diag = np.diag(mat)
        non_ground = [i for i, net in enumerate(nets) if "ground" not in net.lower()]
        if len(non_ground) >= 2:
            sorted_idx = sorted(non_ground, key=lambda i: diag[i])
            p_inner = sorted_idx[0]   # 自电容小 = 内盘
            p_outer = sorted_idx[1]   # 自电容大 = 外环
            C12 = -mat[p_inner][p_outer]
            c_sig = C12 + CJ_DEFAULT * 1e15
            for k in range(n):
                if k in (p_inner, p_outer):
                    continue
                c1k = max(-mat[p_inner][k], 0)
                c2k = max(-mat[p_outer][k], 0)
                if c1k + c2k > 0:
                    c_sig += c1k * c2k / (c1k + c2k)
            return float(c_sig)
        return float(diag[non_ground[0]] + CJ_DEFAULT * 1e15) if non_ground else 0

    return 0.0


def freq_from_csigma(c_sigma_fF):
    """C_Σ → f_01 (GHz), alpha (MHz)。"""
    c_sig = c_sigma_fF * 1e-15
    EJ_J = PHI0**2 / (4 * math.pi**2 * LJ_DEFAULT)   # J
    EC_J = E_CHARGE**2 / (2 * c_sig)                    # J
    EJ = EJ_J / H_PLANCK / 1e9    # GHz
    EC = EC_J / H_PLANCK / 1e9    # GHz
    f01 = math.sqrt(8 * EJ * EC) - EC                    # GHz（一阶修正）
    alpha = -EC * 1000                                     # MHz
    return f01, alpha


def main():
    from robust_design_lab.geometry.four_shape_families import make_library

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    lib = make_library(n_per_family=args.per_family)
    families = args.families or list(lib.keys())

    t0 = time.perf_counter()
    done = set()
    if CHECK.exists():
        for line in CHECK.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["family"], d["variant_idx"], d["scenario"]))

    total_new = 0
    for fam in families:
        info = lib[fam]
        for vi, variant in enumerate(info["variants"]):
            scenarios = [
                ("nominal", variant),
                ("etch_plus", perturb(variant, fam, +ETCH_DELTA)),
                ("etch_minus", perturb(variant, fam, -ETCH_DELTA)),
            ]
            results = {}
            for sc_name, sc_opts in scenarios:
                if (fam, vi, sc_name) in done:
                    continue
                try:
                    t1 = time.perf_counter()
                    r = run_q3d(sc_opts, info["component"], fam)
                    r["family"] = fam
                    r["variant_idx"] = vi
                    r["scenario"] = sc_name
                    r["options"] = {k: v for k, v in sc_opts.items()
                                    if isinstance(v, (str, int, float))}
                    f01, alpha = freq_from_csigma(r["C_sigma_fF"])
                    r["f01_GHz"] = round(f01, 4)
                    r["alpha_MHz"] = round(alpha, 1)
                    r["total_s"] = round(time.perf_counter() - t1, 1)

                    with open(CHECK, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False,
                                           default=str) + "\n")
                    results[sc_name] = r
                    total_new += 1
                    print(f"[{fam}/{vi}/{sc_name}] CΣ={r['C_sigma_fF']:.1f}fF "
                          f"f01={f01:.3f}GHz [{r['total_s']}s]", flush=True)
                except Exception as e:
                    err = dict(family=fam, variant_idx=vi, scenario=sc_name,
                               failed=f"{type(e).__name__}: {e}"[:200])
                    with open(CHECK, "a", encoding="utf-8") as f:
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    print(f"[{fam}/{vi}/{sc_name}] FAILED: {e}", flush=True)

            # 漂移（该变体三场景齐后算）
            if all(s in results for s in ("nominal", "etch_plus", "etch_minus")):
                f0 = results["nominal"]["f01_GHz"]
                drift = {}
                for s in ("etch_plus", "etch_minus"):
                    drift[s] = math.log(results[s]["f01_GHz"] / f0)
                entry = dict(family=fam, variant_idx=vi,
                             f01_nominal=f0, drift=drift,
                             D=max(abs(v) for v in drift.values()))
                print(f"  → D = {entry['D']:.5f}", flush=True)

    print(f"\nSweep done: {total_new} new solves, wall={time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
