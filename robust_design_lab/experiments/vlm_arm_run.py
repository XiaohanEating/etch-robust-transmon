# -*- coding: utf-8 -*-
"""VLM 臂重复运行驱动：一次调用跑一个（族 × 重复序号）。

与首批 vlm_shape_exploration.py 的差别（其余协议完全一致）：
- 基线指标用实测值（baseline_D_measured.json），不再用占位 0.02；
  接受判据因此从第一轮起就是真实基线 D，与数值基线的"在位者起点"对齐。

用法：
  python vlm_arm_run.py --family A_dual_pad --repeat 0
结果追加写入 strong_baselines_2026-10-27/runs.jsonl（arm="vlm"）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "strong_baselines_2026-10-27"
OUT.mkdir(parents=True, exist_ok=True)

# 实测基线指标（锚点验证过的反推起始变体，三场景实测）
MEASURED_BASE_METRICS = {}
_BASE_REF = (ROOT.parent / "robust_design_lab" / "results" /
             "vlm_shape_exploration_2026-10-26" / "baseline_D_measured.json")


def _load_base_metrics():
    if MEASURED_BASE_METRICS:
        return
    for rec in json.loads(_BASE_REF.read_text(encoding="utf-8")):
        MEASURED_BASE_METRICS[rec["family"]] = dict(
            C_sigma_fF=rec["C_sigma_fF"], f01_GHz=rec["f01_GHz"],
            alpha_MHz=rec["alpha_MHz"], D=rec["D"])


def done_keys(path: Path):
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["arm"], r["family"], r.get("seed")))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["A_dual_pad", "D_interdigit"])
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()

    from robust_design_lab.experiments.strong_baseline_run import STARTS
    from robust_design_lab.experiments.vlm_shape_exploration import run_vlm_arm

    _load_base_metrics()
    base_metrics = dict(MEASURED_BASE_METRICS[args.family])

    out_path = OUT / "runs.jsonl"
    key = ("vlm", args.family, args.repeat)
    if key in done_keys(out_path):
        print(f"跳过（已完成）: {key}")
        return

    t0 = time.perf_counter()
    result = run_vlm_arm(args.family, dict(STARTS[args.family]), base_metrics)
    wall = time.perf_counter() - t0

    line = dict(provenance=provenance(), **result,
                seed=args.repeat, budget=9, wall_s=round(wall, 1),
                base_metrics=base_metrics)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    fm = result.get("final_metrics") or {}
    print(f"vlm {args.family} repeat={args.repeat} → "
          f"D={fm.get('D', float('nan')):.4f} "
          f"CΣ={fm.get('C_sigma_fF', float('nan')):.1f}fF "
          f"耗时{wall/60:.0f}min")


if __name__ == "__main__":
    main()
