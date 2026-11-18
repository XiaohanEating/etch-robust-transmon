"""补测 VLM 探索两个起始变体的真实 D（名义+刻蚀± 各一次 Q3D）。

vlm_shape_exploration.py 的基线只跑了名义场景，D 用了占位值 0.02；
且 make_library 当时用 hash(name)（进程加盐）生成变体，跨进程不可再生。
起始变体从 vlm_exploration.json 反推：
  - final_variant 保留从未被采纳改动覆盖的键（即基线原值）；
  - 被采纳过的键（A: pad_gap/pad_width；D: pad_width/finger_width/finger_gap）
    由 VLM 提案理由中引用的提示词数值还原（A: 545um/34um；D: 291um/11um/16um）。
一致性锚点：基线名义 C_Σ，VLM 理由引用为 A=82.7 fF、D=37 fF。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust_design_lab.experiments.vlm_shape_exploration import eval_variant
from robust_design_lab.experiments.q3d_shape_sweep import run_q3d, freq_from_csigma

OUT = Path(__file__).resolve().parents[1] / "results" / "vlm_shape_exploration_2026-10-26"

BASES = {
    "A_dual_pad": None,   # 从 final_variant 填充，再覆盖被采纳键
    "D_interdigit": None,
}
ADOPTED = {
    "A_dual_pad": {"pad_width": "545um", "pad_gap": "34um"},
    "D_interdigit": {"pad_width": "291um", "finger_width": "11um",
                     "finger_gap": "16um"},
}
ANCHOR_F = {"A_dual_pad": 82.7, "D_interdigit": 37.0}


def main():
    with open(OUT / "vlm_exploration.json", encoding="utf-8") as f:
        data = json.load(f)
    finals = {r["family"]: r["final_variant"]
              for r in data["results"] if r["arm"] == "vlm"}

    out = []
    for family in BASES:
        base = dict(finals[family])
        base.update(ADOPTED[family])
        print(f"== {family} 反推基线 ==", flush=True)
        print("  参数:", {k: str(base[k]) for k in base if k != "connection_pads"},
              flush=True)

        # 锚点核对：名义 C_Σ 应接近 VLM 引用值
        comp = _comp(family)
        nom = run_q3d(base, comp, family)
        cs = nom["C_sigma_fF"]
        print(f"  名义 Cs={cs:.1f}fF（锚点 {ANCHOR_F[family]}fF）", flush=True)
        rec = eval_variant(family, {}, base)
        rec["family"] = family
        rec["reconstructed_base"] = {k: str(base[k]) for k in base
                                     if k != "connection_pads"}
        rec["anchor_C_sigma_fF"] = ANCHOR_F[family]
        rec["anchor_match"] = abs(cs - ANCHOR_F[family]) / ANCHOR_F[family] < 0.03
        out.append(rec)
        print(f"  D={rec['D']:.5f} Cs={rec['C_sigma_fF']:.1f}fF "
              f"f01={rec['f01_GHz']:.3f}GHz", flush=True)
    (OUT / "baseline_D_measured.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", OUT / "baseline_D_measured.json")


def _comp(family):
    from robust_design_lab.experiments.vlm_shape_exploration import _comp_name
    return _comp_name(family)


if __name__ == "__main__":
    main()
