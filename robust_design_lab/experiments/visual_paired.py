# -*- coding: utf-8 -*-
"""视觉对照实验（paired-three-arm）：有图 vs 无图 vs 数值基线，等预算配对。

协议（方案 §8 精简版，预算依 G2 实测）：
- 3 个种子 × 3 臂（image / no_image / numerical）× R 轮 × 每轮 ≤8 提案；
- 所有臂同一初始记录（rect_r0，有限模型 α 窗下可行）；
- 每轮：渲染三联图 → 臂提案 → 操作应用 → 修正评价器（h=0.01，缓存）
  → 留 drift_geom 最小且可行者进入下一轮；
- 数值基线臂 = 固定操作集合（无模型调用）；
- 判定门槛：image 与 no_image 的最终 drift 配对差 > 0.022（矩阵噪声底）才可声称
  视觉增量（decisions.md 2026-10-24）。
- VLM 响应/图像哈希/提示词全部落盘。输出 results/visual_paired_2026-10-24/。
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "visual_paired_2026-10-24"
SEEDS = (20261024, 20261025, 20261026)
ROUNDS = 2
CLAIM_FLOOR = 0.022


def initial_record():
    from robust_design_lab.geometry import library
    W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2
    return library.rect_record(W2, frozen.FOOTPRINT_UM[1], 0.0)


def _eval_one(rec):
    from robust_design_lab.evaluator.corrected import evaluate_record
    try:
        m = evaluate_record(rec, 0.01, use_cache=True)
        return dict(ok=m["feasible"], drift=m["drift_geom"], C_q=m["C_q_fF"],
                    f01=m["f01_GHz"], alpha=m["alpha_MHz"], hash=rec.geometry_hash)
    except Exception as e:                                     # noqa: BLE001
        return dict(ok=False, failed=f"{type(e).__name__}: {e}"[:120])


def evaluate_candidates(recs):
    """轮内并行评估提案（进程池；顺序保持；worker 在模块级以便 pickle）。"""
    from concurrent.futures import ProcessPoolExecutor
    if not recs:
        return []
    with ProcessPoolExecutor(max_workers=min(8, len(recs))) as ex:
        vals = list(ex.map(_eval_one, recs))
    for v, r in zip(vals, recs):
        v["rec"] = r
    return vals


def run_arm(arm, seed, base, out_dir):
    from robust_design_lab.geometry.operations import apply
    from robust_design_lab.vlm.experiment import call_arm, OPERATIONS
    from robust_design_lab.experiments.rendering import render_triptych

    rec = base
    history = []
    for rnd in range(ROUNDS):
        image = out_dir / f"{arm}_{seed}_r{rnd}.png"
        visual = render_triptych(rec, image)
        from robust_design_lab.evaluator.corrected import evaluate_record
        m0 = evaluate_record(rec, 0.01, use_cache=True)
        summary = dict(category=rec.category, geometry_hash=rec.geometry_hash,
                       C_q_fF=m0["C_q_fF"], f01_GHz=m0["f01_GHz"],
                       alpha_MHz=m0["alpha_MHz"],
                       drift_geom=m0["drift_geom"],
                       field_quantiles=visual.get("field_log_quantiles"))
        if arm == "numerical":
            proposals = list(OPERATIONS)
            response = None
        else:
            try:
                response = call_arm(summary, image if arm == "image" else None)
                proposals = response.get("proposals", [])
            except ValueError as err:
                # 集合外/不可解析提案 = 非法提案：该轮预算作废，如实记录
                history.append(dict(round=rnd, arm=arm, illegal_response=str(err)[:200]))
                proposals = []
                response = None
        cands = []
        for op in proposals:
            try:
                cands.append((op, apply(rec, op)))
            except Exception as e:                             # noqa: BLE001
                history.append(dict(round=rnd, op=op, legal=False, reason=str(e)[:100]))
        evals = evaluate_candidates([c[1] for c in cands])
        legal_ok = [dict(round=rnd, op=op,
                         **{k: v for k, v in ev.items() if k != "rec"})
                    for (op, c), ev in zip(cands, evals)]
        history.extend(legal_ok)
        good = [ev for ev in evals if ev.get("ok") and ev.get("drift") is not None]
        if good:
            best = min(good, key=lambda e: e["drift"])
            if best["drift"] < m0["drift_geom"]:
                rec = best["rec"]
        if response is not None:
            (out_dir / f"{arm}_{seed}_r{rnd}_response.json").write_text(
                json.dumps({k: v for k, v in response.items() if k != "image"},
                           ensure_ascii=False, indent=1), encoding="utf-8")
    from robust_design_lab.evaluator.corrected import evaluate_record
    final = evaluate_record(rec, 0.01, use_cache=True)
    return dict(arm=arm, seed=seed, final_hash=rec.geometry_hash,
                final_drift=final["drift_geom"], final_feasible=final["feasible"],
                final_alpha=final["alpha_MHz"], history=history)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    base = initial_record()
    results = []
    for seed in SEEDS:
        for arm in ("numerical", "no_image", "image"):
            r = run_arm(arm, seed, base, OUT)
            results.append(r)
            print(f"[{arm}/{seed}] final drift={r['final_drift']}", flush=True)
            write_json(OUT / "paired_results.json",
                       dict(provenance=provenance(), claim_floor=CLAIM_FLOOR,
                            rounds=ROUNDS, seeds=list(SEEDS), results=results))

    # 配对统计
    by = {(r["arm"], r["seed"]): r["final_drift"] for r in results}
    import math
    pairs_img, pairs_num = [], []
    for seed in SEEDS:
        if ("image", seed) in by and ("no_image", seed) in by:
            pairs_img.append(by[("no_image", seed)] - by[("image", seed)])
        if ("image", seed) in by and ("numerical", seed) in by:
            pairs_num.append(by[("numerical", seed)] - by[("image", seed)])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    verdict = dict(
        image_vs_text=dict(pairs=pairs_img, mean_delta=mean(pairs_img),
                           claim=(abs(mean(pairs_img)) > CLAIM_FLOOR
                                  and all(abs(d) > 0 for d in pairs_img)
                                  and len({d > 0 for d in pairs_img}) == 1)),
        image_vs_numerical=dict(pairs=pairs_num, mean_delta=mean(pairs_num)),
        claim_floor=CLAIM_FLOOR,
        wall_s=round(time.perf_counter() - t0, 1))
    data = json.loads((OUT / "paired_results.json").read_text(encoding="utf-8"))
    data["verdict"] = verdict
    write_json(OUT / "paired_results.json", data)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
