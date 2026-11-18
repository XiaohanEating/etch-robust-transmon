# -*- coding: utf-8 -*-
"""G2 成本测试：50 个候选过修正评价器（搜索档），实测单评估成本与失败率。

候选构成：10 代表 + 40 个（代表 × 合法操作）变体。h=0.01，缓存开启。
记录：每候选场解数、mesh/solve/总耗时、失败原因、缓存命中。
输出 results/g2_cost_2026-10-24/cost.json。
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

OUT = ROOT / "results" / "g2_cost_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 10
N_VARIANTS = 40


def main():
    from robust_design_lab.experiments.g1_step4_matrix import representatives
    from robust_design_lab.geometry.operations import apply
    from robust_design_lab.evaluator.corrected import evaluate_record

    reps = representatives()
    ops = ({'kind': 'resize', 'sx': 0.95, 'sy': 1.0}, {'kind': 'resize', 'sx': 1.0, 'sy': 0.95},
           {'kind': 'resize', 'sx': 1.05, 'sy': 1.0}, {'kind': 'round', 'radius_um': 4.0})
    cands = [("rep:" + tag, rec) for tag, rec in reps]
    i = 0
    while len(cands) < 10 + N_VARIANTS:
        tag, rec = reps[i % len(reps)]
        op = ops[(i // len(reps)) % len(ops)]
        try:
            v = apply(rec, op)
            cands.append((f"var:{tag}:{op['kind']}{i}", v))
        except Exception:                                     # noqa: BLE001
            pass
        i += 1
    cands = cands[:10 + N_VARIANTS]
    print(f"candidates: {len(cands)}", flush=True)

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(evaluate_record, rec, 0.01, True): (name, rec)
                for name, rec in cands}
        for k, fut in enumerate(as_completed(futs), 1):
            name, rec = futs[fut]
            try:
                r = fut.result()
                rows.append(dict(name=name, hash=rec.geometry_hash,
                                 seconds=round(r["seconds"], 1),
                                 mesh_s=round(r["mesh_time_s"], 1),
                                 solve_s=round(r["solve_time_s"], 1),
                                 cached=r["seconds"] < 5,
                                 feasible=r["feasible"],
                                 drift_geom=r["drift_geom"]))
            except Exception as e:                             # noqa: BLE001
                rows.append(dict(name=name, hash=rec.geometry_hash,
                                 failed=f"{type(e).__name__}: {e}"[:150]))
            if k % 10 == 0:
                print(f"  [{k}/{len(cands)}] {time.perf_counter()-t0:.0f}s", flush=True)

    ok = [r for r in rows if "failed" not in r]
    fresh = [r for r in ok if not r["cached"]]
    summary = dict(
        n_total=len(rows), n_ok=len(ok), n_failed=len(rows) - len(ok),
        n_fresh_evals=len(fresh),
        fresh_seconds=dict(mean=sum(r["seconds"] for r in fresh) / max(len(fresh), 1),
                           max=max((r["seconds"] for r in fresh), default=0.0)),
        mesh_s_mean=sum(r["mesh_s"] for r in fresh) / max(len(fresh), 1),
        solve_s_mean=sum(r["solve_s"] for r in fresh) / max(len(fresh), 1),
        n_feasible=sum(1 for r in ok if r["feasible"]),
        wall_s=round(time.perf_counter() - t0, 1), workers=WORKERS,
        rows=rows)
    (OUT / "cost.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"G2 cost: {len(ok)}/{len(rows)} ok, fresh={len(fresh)}, "
          f"mean={summary['fresh_seconds']['mean']:.1f}s, "
          f"feasible={summary['n_feasible']} → {OUT/'cost.json'}")


if __name__ == "__main__":
    main()
