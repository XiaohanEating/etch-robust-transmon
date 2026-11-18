"""Derive readout requirements from saved, converged one-resonator matrices.

No field solves, geometry changes or modifications of the official LOM defaults.
Candidate external loads are conditional calculations, not measured devices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from robust_design_lab.evaluator.purcell import evaluate_matrix, lifetime_from_modes
from robust_design_lab.runtime.records import write_json

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "robust_design_lab/results/purcell_qualification_2026-11-08"


def resonator_times(f_GHz, loaded_Q):
    """kappa is the energy decay rate; field amplitude decays at kappa/2."""
    if not all(math.isfinite(x) and x > 0 for x in (f_GHz, loaded_Q)):
        raise ValueError("Frequency and loaded Q must be finite and positive")
    linewidth_Hz = f_GHz * 1e9 / loaded_Q
    kappa = 2 * math.pi * linewidth_Hz
    return dict(loaded_Q=loaded_Q, linewidth_kHz=linewidth_Hz / 1e3,
                energy_decay_us=1e6 / kappa, field_amplitude_decay_us=2e6 / kappa)


def design_requirements(lom, target_us=3000.0, candidate_Q=120000.0):
    circuit = lom["circuit"]
    if circuit["bus_GHz"] or len(lom["g_MHz"]) != 1:
        raise ValueError("Scaling analysis requires exactly one readout mode")
    if not math.isfinite(target_us) or target_us <= 0:
        raise ValueError("Purcell target must be finite and positive")
    f, fr = lom["f01_GHz"], circuit["readout_GHz"]
    q0, g = circuit["readout_Q"], abs(lom["g_MHz"][0])
    original_us, _ = lifetime_from_modes(f, [g], [fr], [q0])
    if not math.isclose(original_us, lom["T1_Purcell_us"], rel_tol=1e-10):
        raise ValueError("Saved lifetime is inconsistent with its circuit")
    q_needed = q0 * target_us / original_us
    candidate_us, _ = lifetime_from_modes(f, [g], [fr], [candidate_Q])
    g_limit = g * math.sqrt(original_us / target_us)
    # chi is quadratic in g in the official perturbative formula at fixed f/alpha.
    weak_chi = abs(lom["chi_MHz"][0]) * (g_limit / g)**2
    suppression = max(1.0, target_us / original_us)
    return dict(
        official_default_Purcell_us=original_us,
        baseline=resonator_times(fr, q0),
        narrow_linewidth=dict(required_loaded_Q=q_needed,
            required_Q_times=resonator_times(fr, q_needed),
            candidate=resonator_times(fr, candidate_Q),
            conditional_Purcell_us=candidate_us,
            lifetime_margin=candidate_us / target_us,
            total_shift_to_linewidth=abs(lom["chi_MHz"][0]) * 1e3 /
                                     resonator_times(fr, candidate_Q)["linewidth_kHz"]),
        weak_coupling=dict(max_abs_g_over_2pi_MHz=g_limit,
            conditional_total_shift_MHz=weak_chi,
            total_shift_to_default_linewidth=weak_chi * 1e3 /
                                            resonator_times(fr, q0)["linewidth_kHz"],
            condition="fixed frequencies and anharmonicity; geometry must be re-solved"),
        filtered_readout=dict(required_decay_rate_suppression=suppression,
            required_power_suppression_dB=10 * math.log10(suppression),
            condition="required rate suppression only; no filter or admittance model evaluated"),
    )


def analyze(comparison_path, *, target_us=3000.0, candidate_Q=120000.0):
    comparison_path = Path(comparison_path).resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = []
    for entry in comparison["objects"]:
        source = (REPO / entry["source"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["source_sha256"]:
            raise ValueError(f"Source record changed: {source}")
        saved = json.loads(source.read_text(encoding="utf-8"))
        if saved["status"] != "accepted" or saved["raw"]["converged"] is not True:
            raise ValueError("Unconverged or rejected nominal record")
        if saved["lom"]["circuit"] != dict(Lj_nH=12.31, Cj_fF=2.0, readout_GHz=7.0,
                bus_GHz=[], readout_Q=10000.0, bus_Q=100000.0, resonator_impedance_ohm=50.0):
            raise ValueError("Pilot circuit differs from the common registered setup")
        recomputed = evaluate_matrix(saved["raw"]["matrix_fF"], saved["raw"]["nets"],
                                    saved["lom"]["node_order"], converged=True)
        for key in ("f01_GHz", "C_sigma_fF", "T1_Purcell_us", "alpha_MHz"):
            if not math.isclose(recomputed[key], saved["lom"][key], rel_tol=1e-10):
                raise ValueError(f"Official replay mismatch: {key}")
        rows.append(dict(object=entry["object"], label=entry["label"],
                         source=str(source.relative_to(REPO)), source_sha256=digest,
                         requirements=design_requirements(recomputed, target_us, candidate_Q)))
    if len(rows) != 3 or {r["object"] for r in rows} != {"pad", "annulus_edited", "idc"}:
        raise ValueError("Expected three distinct pilot objects")
    q_required = max(r["requirements"]["narrow_linewidth"]["required_loaded_Q"] for r in rows)
    paths = [Path(__file__), REPO / "robust_design_lab/evaluator/purcell.py",
             REPO / "src/qiskit_metal/analyses/quantization/lumped_capacitive.py"]
    return dict(
        scope="offline_common_readout_design_requirements",
        target_Purcell_us=target_us, target_status="engineering_reference_not_user_hard_threshold",
        common_candidate=dict(f_readout_GHz=7.0, loaded_Q=candidate_Q,
            impedance_ohm=50.0, buses=0, additional_filter=False,
            required_loaded_Q_max=q_required,
            Q_margin_above_pilot_requirement=candidate_Q / q_required - 1,
            status="prospective_simulation_proposal_not_measured_readout"),
        objects=rows, original_record_replay_passed=True,
        new_field_solves=0, new_model_requests=0,
        comparison_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths},
        limitations=["linewidth response constants do not establish measurement fidelity or duration",
                     "candidate loaded Q requires a compatible physical loss and coupling budget",
                     "all calibration and search candidates must use one fixed external configuration",
                     "new geometries require fresh coupling, lifetime and etch frequency evaluation",
                     "the pilot does not prove a VLM advantage"],
    )


def render_report(result):
    candidate = result["common_candidate"]
    first = result["objects"][0]["requirements"]
    times = first["narrow_linewidth"]["candidate"]
    weak_shifts = [r["requirements"]["weak_coupling"]["conditional_total_shift_MHz"] * 1e3
                   for r in result["objects"]]
    weak_range = (f"约{sum(weak_shifts)/len(weak_shifts):.1f}"
                  if round(min(weak_shifts), 1) == round(max(weak_shifts), 1)
                  else f"{min(weak_shifts):.1f}–{max(weak_shifts):.1f}")
    lines = [
        "# 共同读出设计要求", "",
        "三个已收敛名义矩阵的复算表明，保持现有耦合、以3 ms为Purcell参照时，"
        f"共同加载Q至少需要{candidate['required_loaded_Q_max']:.1f}。"
        f"可供后续仿真采用的候选为7 GHz、50 Ω、Q={candidate['loaded_Q']:.0f}，"
        "单谐振器、无bus、无额外滤波器。该配置是明确提出的外部环境建议，尚无实际读出性能验证。", "",
        "## 三条设计路径", "",
        "| 路径 | 从当前矩阵得到的要求 | 代价与下一项验证 |",
        "|---|---|---|",
        f"| 缩窄线宽，保持耦合布局 | Q≥{candidate['required_loaded_Q_max']:.0f}；候选Q={candidate['loaded_Q']:.0f} | "
        f"线宽{times['linewidth_kHz']:.2f} kHz，场幅时间常数{times['field_amplitude_decay_us']:.2f} μs；核读出带宽和损耗预算 |",
        "| 保持默认Q=10000，减小耦合 | 三对象的耦合上限约28 MHz | "
        f"相应总色散位移{weak_range} kHz，"
        "低于700 kHz线宽；需要重算几何、读出响应及信噪比 |",
        "| 保持耦合与带宽，增加滤波 | 在比特频率处将Purcell损耗率抑制约6.3–10倍 | "
        "需有滤波电路或环境导纳模型，不能把假定倍数当作已实现寿命 |", "",
        "为了继续研究电容形状，建议优先采用第一条作为共同仿真环境，"
        "明确接受较窄带宽。若实际任务要求更快读出，再依据真实滤波电路评估第三条。"
        "两条都不需要把滤波器创新加入本研究。最不推荐逐候选修改Q或指定相同g；"
        "前者改变比较环境，后者抹去几何影响。", "",
        "## 原始值与候选环境的条件计算", "",
        "| 对象 | 官方默认Q下T1 (ms) | 达到3 ms所需Q | 候选Q下T1 (ms) | "
        "默认Q下耦合上限 (MHz) | 所需滤波损耗率抑制倍数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["objects"]:
        r = row["requirements"]
        lines.append(f"| {row['label']} | {r['official_default_Purcell_us']/1000:.4f} | "
            f"{r['narrow_linewidth']['required_loaded_Q']:.1f} | "
            f"{r['narrow_linewidth']['conditional_Purcell_us']/1000:.4f} | "
            f"{r['weak_coupling']['max_abs_g_over_2pi_MHz']:.3f} | "
            f"{r['filtered_readout']['required_decay_rate_suppression']:.3f} |")
    lines += ["", "候选Q下的毫秒级数值由已提取的频率和耦合代入同一公式得到，"
              "没有新增电磁场解。原始官方Q=10000结果完整保留。"
              "增大Q本身不降低几何刻蚀频漂，也不构成视觉方法的收益。", "",
              "## 公式、单位与适用条件", "",
              "官方单模式公式采用角频率与耦合角频率：", "",
              "$$T_{1,P}=\\frac{(\\omega_r^2-\\omega_q^2)^2}"
              "{4(\\omega_r/Q_L)g^2\\omega_q^2}.\\qquad"
              "\\frac{\\kappa}{2\\pi}=\\frac{f_r}{Q_L},\\quad"
              "\\tau_E=\\frac1{\\kappa},\\quad\\tau_a=\\frac2{\\kappa}.$$", "",
              "此处表中的MHz耦合指角频率耦合除以2π。固定几何、谐振器频率与阻抗，"
              "在该模型内T1正比于加载Q；减小耦合的估算还额外假定频率和非谐性不变。"
              "实际修改耦合几何后必须重新提取矩阵。", "",
              f"候选环境的能量时间常数为{times['energy_decay_us']:.3f} μs，"
              f"场幅时间常数为{times['field_amplitude_decay_us']:.3f} μs。"
              "这些不是测量时间或保真度。加载Q满足1/Q_L=1/Q_i+1/Q_c；"
              "需要内损耗与外耦合共同允许，单纯把仿真数字调高不能证明工艺可实现。", "",
              "3 ms来自300 μs其他损耗目标及十倍余量，是工程参照，"
              "并非通用加工标准。候选Q相对三对象所需最大值多约20%，"
              "该余量只针对当前名义态；尺寸校准及新几何仍逐个检查寿命。", "",
              "## 后续实验边界", "",
              "现有名义电容约93.6–93.8 fF，需按完整器件校准到共同目标。"
              "在新方法比较开始前一次确定外部环境，所有视觉与经典方法共用；"
              "校准后重新计算频率、耦合、寿命及刻蚀三场景。"
              "本轮仅完成读出要求计算，不宣称新器件频漂已经改善。", "",
              "原始记录哈希与官方LOM复算均通过。结果JSON保存逐对象要求、"
              "公式依赖源码哈希及实际源文件。没有电磁求解、模型调用或原始数据覆盖。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS / "nominal_comparison.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "readout_requirements")
    args = parser.parse_args()
    result = analyze(args.input)
    write_json(args.output / "requirements.json", result)
    (args.output / "report.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["common_candidate"], ensure_ascii=False, indent=2))
    print(f"Three nominal records replayed; report saved to {args.output}")


if __name__ == "__main__":
    main()
