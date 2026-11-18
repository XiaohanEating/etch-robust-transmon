# -*- coding: utf-8 -*-
"""强基线正式运行驱动：一次调用跑一个（基线 × 族 × 种子），Q3D 实测。

与 VLM 探索严格同协议：同起点、同 9 次评估预算（每次 = 名义+刻蚀±
3 个 Q3D 场景）、同一 eval_variant 评价器、目标为最小化 D。

起点 = VLM 探索的两个起始变体（不可再生，已反推并经名义 C_Σ 锚点验证：
A=82.7 fF、D=37.0 fF 精确命中，见 baseline_D_measured.json）。

用法：
  python strong_baseline_run.py --baseline cma_es --family A_dual_pad --seed 0
结果追加写入 runs.jsonl；已存在的（基线, 族, 种子）自动跳过（可续跑）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.search.strong_baselines import run_strong_baseline
from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "strong_baselines_2026-10-27"
OUT.mkdir(parents=True, exist_ok=True)

# 反推起始变体（final 未改键 + VLM 理由引用值；锚点验证通过）
STARTS = {
    "A_dual_pad": dict(
        pad_width="545um", pad_height="178um", pad_gap="34um",
        pocket_width="800um", pocket_height="800um",
        connection_pads=dict(
            readout=dict(loc_W=+1, loc_H=-1, pad_width="200um"),
            bus1=dict(loc_W=-1, loc_H=+1, pad_height="30um"),
            bus2=dict(loc_W=-1, loc_H=-1, pad_height="50um"),
        ),
    ),
    "D_interdigit": dict(
        pad_width="291um", pad_height="84um", finger_width="11um",
        finger_height="51um", finger_gap="16um", n_fingers=3,
        junction_gap="24um", conn_w="100um", conn_h="30um",
        jj_width="20um", pocket_margin="60um",
    ),
}


def q3d_eval(variant, family):
    """真实评价：3 个 Q3D 场景；异常向上抛（由基线模块按罚分处理）。"""
    from robust_design_lab.experiments.vlm_shape_exploration import eval_variant
    rec = eval_variant(family, {}, variant)
    if not rec.get("ok"):
        raise RuntimeError(rec.get("failed", "评估失败"))
    return rec


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
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--family", required=True, choices=list(STARTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=9)
    args = ap.parse_args()

    out_path = OUT / "runs.jsonl"
    key = (args.baseline, args.family, args.seed)
    if key in done_keys(out_path):
        print(f"跳过（已完成）: {key}")
        return

    t0 = time.perf_counter()
    result = run_strong_baseline(
        args.baseline, args.family, STARTS[args.family], q3d_eval,
        budget=args.budget, seed=args.seed)
    wall = time.perf_counter() - t0

    fm = result.get("final_metrics") or {}
    line = dict(provenance=provenance(), **result, wall_s=round(wall, 1))
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"{args.baseline:18s} {args.family:14s} seed={args.seed} → "
          f"最优D={result.get('best_objective')}  "
          f"CΣ={fm.get('C_sigma_fF', float('nan')):.1f}fF  "
          f"f01={fm.get('f01_GHz', float('nan')):.2f}GHz  "
          f"成功{result.get('n_ok')}/9  耗时{wall/60:.0f}min")
    print(f"追加: {out_path}")


if __name__ == "__main__":
    main()
