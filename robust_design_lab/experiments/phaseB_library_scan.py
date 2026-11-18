# -*- coding: utf-8 -*-
"""Phase B：1024 形状库粗档扫描（G4-large-shape 前半）。

流程：8 类 × 128 候选 → 几何检查 → 名义场解（粗档 h=0.08，P2）→
C_q / f01 / α 窗口筛选 → 通过者计算 S_search 7 场景漂移（D_f01/D_alpha/D_T1 粗档）。
断点续跑：results/phaseB_library/scan.jsonl 逐行追加，重启按 geometry_hash 跳过。
多进程：spawn + 每 worker 独立 gmsh 会话。
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.geometry import library                   # noqa: E402
from robust_design_lab.perturbation import scenarios as scen     # noqa: E402

N_PER_CLASS = 128
OUT = ROOT / "results" / "phaseB_library" / "scan.jsonl"
WORKERS = 12


def eval_one(args):
    category, index = args
    from robust_design_lab.evaluator import fast
    from robust_design_lab.geometry import checks
    from robust_design_lab.objectives import drift

    rec = library.make_record(category, index)
    entry = dict(category=category, index=index, hash=rec.geometry_hash)

    ok, reasons = checks.check_record(rec)
    entry["checks_ok"] = ok
    if not ok:
        entry["fail"] = "; ".join(reasons)[:200]
        return entry
    try:
        cache = fast.FieldCache()
        S = scen.load()["S_search"]
        nom = drift.evaluate_scene(rec, S[0], cache)
        if "failed" in nom:
            entry["fail"] = "nominal:" + nom["failed"][:150]
            return entry
        entry.update(C_q_fF=round(nom["C_q_fF"], 3),
                     f01_GHz=round(nom["f01_GHz"], 5),
                     alpha_MHz=round(nom["alpha_MHz"], 2),
                     p_SM=float(f"{nom['p_SM']:.4e}"),
                     T1_us=round(nom["T1_us"], 2))
        entry["window_ok"] = bool(
            frozen.WINDOW_F01_GHZ[0] <= nom["f01_GHz"] <= frozen.WINDOW_F01_GHZ[1]
            and frozen.WINDOW_ALPHA_MHZ[0] <= nom["alpha_MHz"] <= frozen.WINDOW_ALPHA_MHZ[1])
        if entry["window_ok"]:
            res = {f"s{i}": drift.evaluate_scene(rec, s, cache) for i, s in enumerate(S)}
            d_all = drift.drift_metrics(nom, res)
            idx_geom = [i for i, s in enumerate(S) if abs(s.get("etch_um", 0)) > 1e-9]
            d_geom = drift.drift_metrics(nom, {f"s{i}": res[f"s{i}"] for i in idx_geom})
            entry.update(
                D_T1_all=round(d_all["D_T1"]["max"], 5),
                D_f01_all=round(d_all["D_f01"]["max"], 5),
                D_alpha_all=round(d_all["D_alpha"]["max"], 5),
                D_T1_geom=round(d_geom["D_T1"]["max"], 5),
                D_f01_geom=round(d_geom["D_f01"]["max"], 5),
                n_failed_scenes=d_all["n_failed"],
            )
    except Exception as e:                                   # noqa: BLE001
        entry["fail"] = f"{type(e).__name__}: {e}"[:200]
    return entry


def main():
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["hash"])

    jobs = [(cat, i) for cat in library.CATEGORY_NAMES for i in range(N_PER_CLASS)]
    jobs = [j for j in jobs
            if library.make_record(j[0], j[1]).geometry_hash not in done]
    print(f"scan: {len(done)} done, {len(jobs)} to run, {WORKERS} workers", flush=True)

    n_ok = n_window = n_drift = 0
    with open(OUT, "a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(eval_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            entry = fut.result()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += entry.get("checks_ok", False)
            n_window += entry.get("window_ok", False)
            n_drift += "D_T1_all" in entry
            if k % 50 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] checks_ok={n_ok} window_ok={n_window} "
                      f"drift_done={n_drift} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"SCAN DONE: total={len(done)+len(jobs)} checks_ok={n_ok} window_ok={n_window} "
          f"drift_done={n_drift} wall={time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
