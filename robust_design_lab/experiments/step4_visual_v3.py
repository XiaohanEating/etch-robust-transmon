# -*- coding: utf-8 -*-
"""第四步(b)：可解释视觉对照 v3（机制检查定位；G1 未通过 → 不作贡献声称）。

与 v2 的差异（对应审阅要求）：
- 独立性 = 不同起始几何（3 个可行起点），数值臂操作顺序由种子真随机洗牌；
- 全溯源：提示原文、原始回复、图像 SHA256、吸附映射逐调用落盘；
- 错配图检查（预注册子集：起点 1 的 image 臂第 2 轮喂"起点 2 的三联图"，
  提示中的数值摘要仍为当前几何）；
- 终选评细档 h=0.005；
- 判定门槛：本试跑仅报告机制观察（提案是否随图变化、错配图是否改变提案），
  不声称视觉增量（G1 原容差未通过，drift 噪声底未获用户裁决）。

输出 results/step4_visual_v3_2026-10-24/。
"""

import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step4_visual_v3_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)
ROUNDS = 2
SEEDS = {"start0": 91001, "start1": 91002, "start2": 91003}


def sha256f(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _eval_one(rec, lj0_H=None):
    from robust_design_lab.evaluator.corrected import evaluate_record
    try:
        m = evaluate_record(rec, 0.01, use_cache=True, lj0_H=lj0_H)
        return dict(ok=m["feasible"], drift=m["drift_geom"], C_q=m["C_q_fF"],
                    f01=m["f01_GHz"], alpha=m["alpha_MHz"], hash=rec.geometry_hash)
    except Exception as e:                                     # noqa: BLE001
        return dict(ok=False, failed=f"{type(e).__name__}: {e}"[:120])


def eval_candidates(recs, lj0_H=None):
    from concurrent.futures import ProcessPoolExecutor
    if not recs:
        return []
    with ProcessPoolExecutor(max_workers=min(8, len(recs))) as ex:
        futs = [ex.submit(_eval_one, r, lj0_H) for r in recs]
        vals = [f.result() for f in futs]          # 先提交后等待（真并行）
    for v, r in zip(vals, recs):
        v["rec"] = r
    return vals


def pick_starts():
    """3 个可行独立起点：rect_r0 + 成本测试缓存中前 2 个可行独立几何。"""
    from robust_design_lab.geometry import library
    from robust_design_lab.configs import frozen
    from robust_design_lab.evaluator.corrected import evaluate_record

    W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2
    starts = [("rect_r0", library.rect_record(W2, frozen.FOOTPRINT_UM[1], 0.0))]
    seen = {starts[0][1].geometry_hash}
    cost = ROOT / "results" / "step4_cost_v2_2026-10-24" / "cost.json"
    if cost.exists():
        rows = json.loads(cost.read_text(encoding="utf-8"))["rows"]
        from robust_design_lab.experiments.g1_step4_matrix import representatives
        from robust_design_lab.geometry.operations import apply
        repmap = {t: r for t, r in representatives()}
        for row in rows:
            if len(starts) >= 3:
                break
            if not row.get("feasible") or row["hash"] in seen:
                continue
            name = row["name"]
            if name.startswith("rep:"):
                tag = name[4:]
                if tag in repmap and repmap[tag].geometry_hash == row["hash"]:
                    starts.append((tag, repmap[tag]))
                    seen.add(row["hash"])
    # 不足则直接评估补充（此时多有缓存）
    i = 0
    ops = ({'kind': 'round', 'radius_um': 4.0}, {'kind': 'resize', 'sx': 0.95, 'sy': 1.0})
    while len(starts) < 3 and i < 20:
        base = starts[0][1]
        try:
            from robust_design_lab.geometry.operations import apply as A
            v = A(base, ops[i % 2])
            if v.geometry_hash in seen:
                i += 1
                continue
            m = evaluate_record(v, 0.01, use_cache=True)
            if m["feasible"]:
                starts.append((f"var{i}", v))
                seen.add(v.geometry_hash)
        except Exception:                                     # noqa: BLE001
            pass
        i += 1
    return starts


def run_arm(start_name, start_rec, arm, seed, mismatch_image=None):
    from robust_design_lab.geometry.operations import apply
    from robust_design_lab.vlm.experiment import call_arm, OPERATIONS
    from robust_design_lab.experiments.rendering import render_triptych
    from robust_design_lab.evaluator.corrected import evaluate_record

    # 该候选的器件定义：入口一次配定，全程（含提案评价与终选）不变
    anchor = evaluate_record(start_rec, 0.01, use_cache=True)
    LJ0 = anchor["lj0_nH"] * 1e-9

    def eval_at(r, h=0.01):
        return evaluate_record(r, h, use_cache=True, lj0_H=LJ0)

    rec = start_rec
    history = []
    provenance_calls = []
    for rnd in range(ROUNDS):
        image = OUT / f"{start_name}_{arm}_r{rnd}.png"
        visual = render_triptych(rec, image)
        img_path_for_call = image if arm == "image" else mismatch_image
        m0 = eval_at(rec)
        summary = dict(category=rec.category, geometry_hash=rec.geometry_hash,
                       C_q_fF=m0["C_q_fF"], f01_GHz=m0["f01_GHz"],
                       alpha_MHz=m0["alpha_MHz"], drift_geom=m0["drift_geom"],
                       field_quantiles=visual.get("field_log_quantiles"))
        if arm == "numerical":
            proposals = list(OPERATIONS)
            random.Random(seed + rnd).shuffle(proposals)   # 真随机化（种子进洗牌）
            response = None
        else:
            try:
                response = call_arm(summary, img_path_for_call)
                proposals = response["proposals"]
            except ValueError as err:
                history.append(dict(round=rnd, arm=arm,
                                    illegal_response=str(err)[:200]))
                proposals, response = [], None
        if response is not None:
            provenance_calls.append(dict(
                round=rnd, arm=arm, start=start_name,
                image=str(img_path_for_call) if img_path_for_call else None,
                image_sha256=(sha256f(img_path_for_call)
                              if img_path_for_call and Path(img_path_for_call).exists() else None),
                prompt_summary_keys=sorted(summary.keys()),
                content=response.get("content"), snap_map=response.get("snap_map"),
                request_id=response.get("request_id"), usage=response.get("usage")))
        cands = []
        for op in proposals:
            try:
                cands.append((op, apply(rec, op)))
            except Exception as e:                             # noqa: BLE001
                history.append(dict(round=rnd, op=op, legal=False, reason=str(e)[:100]))
        evals = eval_candidates([c[1] for c in cands], LJ0)
        history.extend(dict(round=rnd, op=op, **{k: v for k, v in ev.items() if k != "rec"})
                       for (op, c), ev in zip(cands, evals))
        good = [ev for ev in evals if ev.get("ok") and ev.get("drift") is not None]
        if good and m0["drift_geom"] is not None:
            best = min(good, key=lambda e: e["drift"])
            if best["drift"] < m0["drift_geom"]:
                rec = best["rec"]
    # 跨档同一器件：先粗档配定 L0，再以同一 L0 复评两档（v3 旧序两档各配）
    coarse = eval_at(rec)
    fine = eval_at(rec, 0.005)
    return dict(start=start_name, arm=arm, seed=seed,
                final_hash=rec.geometry_hash,
                coarse=dict(drift_geom=coarse["drift_geom"], feasible=coarse["feasible"]),
                fine=dict(drift_geom=fine["drift_geom"], feasible=fine["feasible"],
                          alpha_MHz=fine["alpha_MHz"]),
                history=history, provenance_calls=provenance_calls)


def main():
    t0 = time.perf_counter()
    starts = pick_starts()
    print(f"starts: {[n for n, _ in starts]}", flush=True)
    if len(starts) < 3:
        print(f"WARNING: 仅 {len(starts)} 个可行起点（继续，独立性相应减弱并如实记录）")

    mismatch_img = None
    results = []
    for idx, (name, rec) in enumerate(starts):
        seed = SEEDS[f"start{idx}"]
        # 预注册错配子集：起点 1 的 image 臂第 2 轮用起点 2 的图
        for arm in ("numerical", "no_image", "image"):
            mm = None
            if idx == 0 and arm == "image" and len(starts) > 1:
                mm_img = OUT / "_mismatch_pool.png"
                if not mm_img.exists():
                    from robust_design_lab.experiments.rendering import render_triptych
                    render_triptych(starts[1][1], mm_img)
                mm = mm_img
            r = run_arm(name, rec, arm, seed, mismatch_image=mm)
            results.append(r)
            print(f"[{name}/{arm}] coarse={r['coarse']['drift_geom']} "
                  f"fine={r['fine']['drift_geom']}", flush=True)
            write_json(OUT / "visual_v3.json",
                       dict(provenance=provenance(), rounds=ROUNDS,
                            seeds=SEEDS, starts=[n for n, _ in starts],
                            mismatch_policy="start0 image r1 uses start1 triptych",
                            results=results))

    # 机制检查汇总
    mech = dict(
        proposal_variation=[], mismatch_check=None)
    by = {}
    for r in results:
        by[(r["start"], r["arm"])] = r
    for name, _ in starts:
        img = [c for c in by[(name, "image")]["provenance_calls"] if c["content"]]
        txt = [c for c in by[(name, "no_image")]["provenance_calls"] if c["content"]]
        if img and txt:
            i0 = img[0]["content"]
            t0c = txt[0]["content"]
            mech["proposal_variation"].append(dict(
                start=name, differs=(i0 != t0c)))
    # 错配：start0 image 臂 r0(真图) vs r1(错配图) 提案是否不同
    sc = by.get(("start0_x", "image")) if False else by.get((starts[0][0], "image"))
    if sc:
        calls = sc["provenance_calls"]
        if len(calls) >= 2:
            mech["mismatch_check"] = dict(
                r0_content_sha256=hashlib.sha256(
                    (calls[0]["content"] or "").encode()).hexdigest()[:16],
                r1_content_sha256=hashlib.sha256(
                    (calls[1]["content"] or "").encode()).hexdigest()[:16],
                r0_is_mismatch=False, r1_is_mismatch=True,
                differs=(calls[0]["content"] != calls[1]["content"]))

    data = json.loads((OUT / "visual_v3.json").read_text(encoding="utf-8"))
    data["mechanism"] = mech
    data["claim"] = ("无视觉增量声称（G1 原容差未通过）；本试跑仅机制证据")
    data["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "visual_v3.json", data)
    print(json.dumps(mech, ensure_ascii=False, indent=1))
    print(f"wall={data['wall_s']}s → {OUT/'visual_v3.json'}")


if __name__ == "__main__":
    main()
