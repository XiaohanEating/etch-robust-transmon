# -*- coding: utf-8 -*-
"""第四步(a)：50 个独立几何的成本重测（progress_review 第四步要求）。

与 G2 v1 的差异（对应审阅指控）：
- 候选按几何哈希去重，保证 50 个独立几何（v1 有 11 条重复）；
- 缓存命中以 evaluate_record 返回的 cache_hit 为准（v1 用 seconds<5 猜测）；
- 每变体（名义/刻蚀±）场解计时由 worker 内对 solve_fields 的包装统计
  （真实场解数 = 包装调用次数）；
- 记录 worker 峰值 RSS（psutil）。
输出 results/step4_cost_v2_2026-10-24/cost.json。
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step4_cost_v2_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 10
N_TARGET = 50

_OPS = ({'kind': 'resize', 'sx': 0.95, 'sy': 1.0}, {'kind': 'resize', 'sx': 1.0, 'sy': 0.95},
        {'kind': 'resize', 'sx': 1.05, 'sy': 1.0}, {'kind': 'resize', 'sx': 1.0, 'sy': 1.05},
        {'kind': 'round', 'radius_um': 2.0}, {'kind': 'round', 'radius_um': 4.0},
        {'kind': 'round', 'radius_um': 8.0})


def build_unique(n_target):
    """代表 + 操作变体，按哈希去重到 n_target 个独立几何。"""
    from robust_design_lab.experiments.g1_step4_matrix import representatives
    from robust_design_lab.geometry.operations import apply

    reps = representatives()
    cands, seen = [], set()
    for tag, rec in reps:
        if rec.geometry_hash not in seen:
            cands.append((f"rep:{tag}", rec))
            seen.add(rec.geometry_hash)
    i = 0
    while len(cands) < n_target:
        tag, rec = reps[i % len(reps)]
        op1 = _OPS[(i // len(reps)) % len(_OPS)]
        op2 = _OPS[(i // (len(reps) * len(_OPS))) % len(_OPS)]
        cur = rec
        ok = True
        for op in (op1, op2) if i >= len(reps) * len(_OPS) else (op1,):
            try:
                cur = apply(cur, op)
            except Exception:                             # noqa: BLE001
                ok = False
                break
        if ok and cur.geometry_hash not in seen:
            cands.append((f"var{i}:{tag}", cur))
            seen.add(cur.geometry_hash)
        i += 1
        if i > n_target * 40:
            break
    return cands[:n_target]


def _timed_eval(args):
    name, rec = args
    import psutil

    from robust_design_lab.evaluator import finite_film
    from robust_design_lab.evaluator.corrected import evaluate_record

    calls = []
    orig = finite_film.solve_fields

    def wrapped(r, settings=None, *a, **kw):
        t0 = time.perf_counter()
        result = orig(r, settings, *a, **kw) if settings is not None else orig(r, *a, **kw)
        fs = result[0] if isinstance(result, tuple) else result   # solve_fields 返回 (fs, surfaces)
        calls.append(dict(hash=r.geometry_hash,
                          mesh_s=round(fs.mesh_time_s, 2),
                          solve_s=round(fs.solve_time_s, 2),
                          wall_s=round(time.perf_counter() - t0, 2)))
        return result

    finite_film.solve_fields = wrapped
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    try:
        r = evaluate_record(rec, 0.01, use_cache=True)
        err = None
    except Exception as e:                                 # noqa: BLE001
        r, err = None, f"{type(e).__name__}: {e}"[:150]
    finally:
        finite_film.solve_fields = orig
    rss_peak = proc.memory_info().rss
    row = dict(name=name, hash=rec.geometry_hash,
               wall_s=round(time.perf_counter() - t0, 2),
               n_field_solves=len(calls), solves=calls,
               rss_delta_mb=round((rss_peak - rss0) / 1e6, 1),
               rss_peak_mb=round(rss_peak / 1e6, 1))
    if err:
        row["failed"] = err
    else:
        row.update(feasible=r["feasible"], drift_geom=r["drift_geom"],
                   drift_all=r["drift_all"], cache_hit=r["cache_hit"],
                   reported_seconds=r["seconds"])
    return row


def main():
    t0 = time.perf_counter()
    cands = build_unique(N_TARGET)
    uniq = len({h for _, r in cands for h in [r.geometry_hash]})
    print(f"candidates={len(cands)} unique={uniq}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_timed_eval, c): c for c in cands}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if k % 10 == 0:
                print(f"  [{k}/{len(cands)}] {time.perf_counter()-t0:.0f}s", flush=True)

    ok = [r for r in rows if "failed" not in r]
    fresh = [r for r in ok if not any(r["cache_hit"].values())]
    cached_rows = [r for r in ok if any(r["cache_hit"].values())]
    # 分场景成本：solves 内 mesh/solve 按调用序 = nominal, etch_plus, etch_minus
    per_variant = {"1st": [], "2nd": [], "3rd": []}
    for r in fresh:
        for i, key in enumerate(("1st", "2nd", "3rd")):
            if i < len(r["solves"]):
                per_variant[key].append(r["solves"][i]["wall_s"])
    summary = dict(
        provenance=provenance(), workers=WORKERS,
        n_tasks=len(rows), n_ok=len(ok), n_failed=len(rows) - len(ok),
        n_unique_geometries=uniq,
        n_fresh=len(fresh), n_with_cache_hits=len(cached_rows),
        fresh_mean_wall_s=(sum(r["wall_s"] for r in fresh) / len(fresh)) if fresh else None,
        fresh_max_wall_s=max((r["wall_s"] for r in fresh), default=0),
        cached_mean_wall_s=(sum(r["wall_s"] for r in cached_rows) / len(cached_rows))
        if cached_rows else None,
        per_variant_wall_s={k: dict(mean=sum(v) / len(v), max=max(v) if v else 0)
                            for k, v in per_variant.items() if v},
        rss_peak_mb_max=max((r["rss_peak_mb"] for r in ok), default=0),
        n_feasible=sum(1 for r in ok if r["feasible"]),
        n_feasible_unique=len({r["hash"] for r in ok if r["feasible"]}),
        wall_total_s=round(time.perf_counter() - t0, 1),
        rows=rows)
    write_json(OUT / "cost.json", summary)
    print(f"cost v2: {len(ok)}/{len(rows)} ok, unique={uniq}, fresh={len(fresh)}, "
          f"feasible_unique={summary['n_feasible_unique']}, "
          f"fresh_mean={summary['fresh_mean_wall_s']}s, rss_max={summary['rss_peak_mb_max']}MB "
          f"wall={summary['wall_total_s']}s")


if __name__ == "__main__":
    main()
